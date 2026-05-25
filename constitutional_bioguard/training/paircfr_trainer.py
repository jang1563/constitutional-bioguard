"""PairCFR Trainer for counterfactual quadruplet contrastive learning.

Reference: Qiu et al. ACL 2024, arXiv:2406.06633

Key innovation vs vanilla CE training: same-batch pairing of B.2 quadruplet
members (compliance/partial/refusal/deflection/hybrid for the same query)
with a contrastive loss that pushes opposite-label variants apart in [CLS]
space while pulling same-label variants together.

Loss:
  L_total = (1 - lambda) * L_CE + lambda * L_contrastive

Critical implementation detail: PairCFRBatchSampler must keep all variants
of a quadruplet within the same micro-batch -- vanilla shuffling destroys
the contrastive signal.
"""
from __future__ import annotations

import logging
import math
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import Sampler
from transformers import Trainer

logger = logging.getLogger(__name__)


class PairCFRBatchSampler(Sampler):
    """Group quadruplet members into the same batch.

    Assumes the dataset has a `v5_quad_query_id` column for B.2 items
    and `None` (or another constant) for non-quadruplet items. Items
    with the same quad_query_id will be in the same batch.

    For non-quadruplet items, fall back to random batching.
    """

    def __init__(self, group_ids: list, batch_size: int = 16, seed: int = 42):
        self.batch_size = batch_size
        self.group_ids = group_ids
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)
        self._build_index()

    def _build_index(self):
        self.groups = defaultdict(list)
        self.singletons = []
        for i, gid in enumerate(self.group_ids):
            if gid is None or gid == 0:
                self.singletons.append(i)
            else:
                self.groups[gid].append(i)

    def __iter__(self):
        # Sort groups so larger groups (full 5-tuplets) come first
        ordered_groups = sorted(self.groups.values(), key=len, reverse=True)
        # Shuffle order each epoch
        perm = torch.randperm(len(ordered_groups), generator=self.rng).tolist()
        ordered_groups = [ordered_groups[p] for p in perm]
        singletons_perm = torch.randperm(len(self.singletons), generator=self.rng).tolist()
        singletons = [self.singletons[p] for p in singletons_perm]

        batch = []
        for grp in ordered_groups:
            # Skip if group doesn't fit
            if len(grp) > self.batch_size:
                # Should never happen with batch_size=16 and quad=5
                continue
            if len(batch) + len(grp) > self.batch_size:
                # Pad remaining slots from singletons and yield
                while len(batch) < self.batch_size and singletons:
                    batch.append(singletons.pop())
                yield batch
                batch = []
            batch.extend(grp)

        # Final flush of partial batch + remaining singletons
        while singletons:
            if len(batch) >= self.batch_size:
                yield batch
                batch = []
            batch.append(singletons.pop())
        if batch:
            yield batch

    def __len__(self):
        total = len(self.group_ids)
        return math.ceil(total / self.batch_size)


def paircfr_contrastive_loss(
    cls_embeddings: torch.Tensor,
    labels: torch.Tensor,
    group_ids: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Contrastive loss on [CLS] embeddings within quadruplet groups.

    For each anchor with group_id != 0, build positives (same group, same label)
    and negatives (same group, different label). Use NT-Xent style.

    Singletons (group_id == 0) contribute zero contrastive loss.
    """
    device = cls_embeddings.device
    bsz = cls_embeddings.size(0)
    if bsz == 0:
        return torch.tensor(0.0, device=device)

    # Normalize embeddings
    z = F.normalize(cls_embeddings, dim=-1)
    # Pairwise similarities
    sim = torch.matmul(z, z.T) / temperature

    # Mask for valid same-group pairs (excluding self)
    same_group = (group_ids.unsqueeze(0) == group_ids.unsqueeze(1))
    is_group_member = group_ids != 0  # singletons have id 0
    valid_pairs = same_group & is_group_member.unsqueeze(0) & is_group_member.unsqueeze(1)
    eye = torch.eye(bsz, dtype=torch.bool, device=device)
    valid_pairs = valid_pairs & ~eye  # exclude self-pairs

    same_label = (labels.unsqueeze(0) == labels.unsqueeze(1))
    positives = valid_pairs & same_label

    # If no positives found in batch, return 0 (e.g. all singletons batch)
    if positives.sum() == 0:
        return torch.tensor(0.0, device=device)

    # NT-Xent / SupCon style: for each anchor, log(sum_pos exp(sim)) - log(sum_all exp(sim))
    # Use mask-and-exp trick to handle missing positives per row
    sim_masked = sim.masked_fill(eye, -1e9)  # exclude self

    total_loss = torch.tensor(0.0, device=device)
    n_anchors = 0
    for i in range(bsz):
        if not is_group_member[i]:
            continue
        pos_mask = positives[i]
        if pos_mask.sum() == 0:
            continue
        # All non-self same-group similarities are the denominator
        denom_mask = valid_pairs[i]
        if denom_mask.sum() == 0:
            continue
        # log p(positives | denom)
        logsumexp_denom = torch.logsumexp(sim_masked[i][denom_mask], dim=0)
        logsumexp_pos = torch.logsumexp(sim_masked[i][pos_mask], dim=0)
        anchor_loss = -(logsumexp_pos - logsumexp_denom)
        total_loss = total_loss + anchor_loss
        n_anchors += 1

    if n_anchors == 0:
        return torch.tensor(0.0, device=device)
    return total_loss / n_anchors


class PairCFRTrainer(Trainer):
    """HuggingFace Trainer with PairCFR contrastive loss.

    Usage:
        trainer = PairCFRTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,  # must have 'v5_quad_query_id' field
            eval_dataset=eval_ds,
            tokenizer=tokenizer,
            paircfr_lambda=0.3,
            paircfr_temperature=0.1,
        )

    Then call `trainer.train()` as usual.
    """

    def __init__(self, *args,
                 paircfr_lambda: float = 0.3,
                 paircfr_temperature: float = 0.1,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.paircfr_lambda = paircfr_lambda
        self.paircfr_temperature = paircfr_temperature
        logger.info("PairCFRTrainer lambda=%.2f temperature=%.2f",
                    paircfr_lambda, paircfr_temperature)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Extract group_ids if present (not passed to model)
        group_ids = inputs.pop("v5_quad_query_id", None)
        if group_ids is None:
            # Fallback: treat all as singletons
            bsz = inputs.get("input_ids").size(0) if "input_ids" in inputs else 0
            group_ids = torch.zeros(bsz, dtype=torch.long, device=inputs["input_ids"].device)

        labels = inputs.get("labels")

        # Forward with output_hidden_states to get [CLS] embedding
        outputs = model(**inputs, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]  # (bsz, seq, dim)
        cls_emb = last_hidden[:, 0, :]  # (bsz, dim)

        # Cross-entropy loss (class-weighted if model has class weights configured)
        ce_loss = outputs.loss  # transformer Trainer-style CE with optional weights

        # Contrastive loss
        contrastive_loss = paircfr_contrastive_loss(
            cls_emb, labels, group_ids, temperature=self.paircfr_temperature
        )

        total_loss = (1.0 - self.paircfr_lambda) * ce_loss + self.paircfr_lambda * contrastive_loss

        # Log component losses occasionally
        if self.state.global_step % 50 == 0:
            logger.info("step=%d  ce=%.4f  contrastive=%.4f  total=%.4f",
                        self.state.global_step, float(ce_loss),
                        float(contrastive_loss), float(total_loss))

        return (total_loss, outputs) if return_outputs else total_loss

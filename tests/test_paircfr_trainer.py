"""Unit tests for v5 PairCFR batching and contrastive loss."""

import torch

from constitutional_bioguard.training.paircfr_trainer import (
    PairCFRBatchSampler,
    paircfr_contrastive_loss,
)


def test_paircfr_batch_sampler_keeps_group_together():
    group_ids = [0, 101, 101, 101, 101, 0, 202, 202, 202, 202]
    sampler = PairCFRBatchSampler(group_ids, batch_size=5, seed=7)

    batches = list(iter(sampler))
    group_to_batch = {}
    for batch_idx, batch in enumerate(batches):
        for item_idx in batch:
            gid = group_ids[item_idx]
            if gid:
                group_to_batch.setdefault(gid, batch_idx)
                assert group_to_batch[gid] == batch_idx


def test_paircfr_contrastive_loss_ignores_singletons():
    embeddings = torch.randn(4, 8)
    labels = torch.tensor([0, 1, 0, 1])
    group_ids = torch.zeros(4, dtype=torch.long)

    loss = paircfr_contrastive_loss(embeddings, labels, group_ids)

    assert loss.item() == 0.0


def test_paircfr_contrastive_loss_is_finite_for_mixed_label_group():
    embeddings = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.1, 0.9, 0.0, 0.0],
        ]
    )
    labels = torch.tensor([1, 1, 0, 0])
    group_ids = torch.tensor([42, 42, 42, 42])

    loss = paircfr_contrastive_loss(embeddings, labels, group_ids, temperature=0.2)

    assert torch.isfinite(loss)
    assert loss.item() >= 0.0

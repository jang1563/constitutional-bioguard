"""Dual-mode bio guard: combines hybrid prompt-head + response-head.

Architecture:
  prompt side   = lex (4-tier T1U/T1C/T2/T3) OR (deberta_pdual_v3 AND bio_ctx)
  response side = deberta_v2 (pair-encoded query+response)
  joint         = prompt_flag OR response_flag

Separate flags reported (not just joint) so users can pre-filter intent vs
moderate output, which matches how WildGuard/LlamaGuard surface both axes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from constitutional_bioguard.bio_lexicon import (
    classify as lex_classify,
    T1_CONTEXTUAL, T1_UNCONDITIONAL, T3_GENERIC_BIO,
)
from constitutional_bioguard.config import MODELS_DIR

PROMPT_MODEL = MODELS_DIR / "deberta_pdual_v3"
RESPONSE_MODEL = MODELS_DIR / "deberta_bioguard_v8b"  # v2 response-harm from v8b release

_BIO_CTX_TERMS = T1_UNCONDITIONAL + T1_CONTEXTUAL + T3_GENERIC_BIO
_BIO_CTX_RE = re.compile(
    "|".join(re.escape(t) for t in sorted(set(_BIO_CTX_TERMS), key=len, reverse=True)),
    re.IGNORECASE,
)


def has_bio_context(text: str) -> bool:
    return bool(_BIO_CTX_RE.search(str(text or "")))


@dataclass
class DualVerdict:
    prompt_flag: bool
    prompt_source: str  # lex / learned / both / none
    prompt_tier: Optional[int]
    prompt_reason: str
    response_flag: bool
    response_score: float
    joint_flag: bool
    joint_reason: str

    def to_dict(self) -> dict:
        return {
            "prompt_flag": self.prompt_flag, "prompt_source": self.prompt_source,
            "prompt_tier": self.prompt_tier, "prompt_reason": self.prompt_reason,
            "response_flag": self.response_flag,
            "response_score": round(self.response_score, 4),
            "joint_flag": self.joint_flag, "joint_reason": self.joint_reason,
        }


class DualModeGuard:
    """Hybrid prompt-head + response-head bio guard. Lazy model loading."""

    def __init__(self, prompt_model_path: Path = PROMPT_MODEL,
                 response_model_path: Path = RESPONSE_MODEL,
                 response_threshold: float = 0.5):
        self.prompt_model_path = prompt_model_path
        self.response_model_path = response_model_path
        self.response_threshold = response_threshold
        self._prompt_clf = None
        self._response_clf = None

    def _ensure_prompt(self):
        if self._prompt_clf is None:
            from constitutional_bioguard.evaluation.evaluate_classifier import \
                load_model_and_tokenizer
            self._prompt_clf = load_model_and_tokenizer(self.prompt_model_path)

    def _ensure_response(self):
        if self._response_clf is None:
            from constitutional_bioguard.evaluation.evaluate_classifier import \
                load_model_and_tokenizer
            self._response_clf = load_model_and_tokenizer(self.response_model_path)

    def _learned_prompt(self, prompt: str) -> int:
        from constitutional_bioguard.evaluation.evaluate_classifier import predict_batch
        self._ensure_prompt()
        m, t = self._prompt_clf
        preds = predict_batch(model=m, tokenizer=t,
                              queries=[prompt], responses=[""], normalize=True)
        return int(preds[0][0])

    def _learned_response(self, prompt: str, response: str) -> tuple[int, float]:
        from constitutional_bioguard.evaluation.evaluate_classifier import predict_batch
        self._ensure_response()
        m, t = self._response_clf
        preds = predict_batch(model=m, tokenizer=t,
                              queries=[prompt], responses=[response], normalize=True)
        lab, prob = preds[0]
        return int(lab), float(prob)

    def classify(self, prompt: str, response: Optional[str] = None) -> DualVerdict:
        # prompt-side hybrid
        lex = lex_classify(prompt)
        bio_ctx = has_bio_context(prompt)
        lex_flag = lex["flag"]
        learned_flag = False
        if not lex_flag and bio_ctx:
            # only consult learned head when bio context present -- prevents
            # non-bio-harm leakage from the keyword shortcut
            learned_flag = bool(self._learned_prompt(prompt))

        if lex_flag and learned_flag: source = "both"
        elif lex_flag: source = "lex"
        elif learned_flag: source = "learned"
        else: source = "none"

        prompt_flag = lex_flag or learned_flag
        prompt_tier = lex["tier"] if lex_flag else (None if not learned_flag else 3)
        prompt_reason = lex["reason"] if lex_flag else (
            "learned head + bio context" if learned_flag else "no bio harm signal")

        # response-side
        if response is None or not str(response).strip():
            return DualVerdict(prompt_flag=prompt_flag, prompt_source=source,
                               prompt_tier=prompt_tier, prompt_reason=prompt_reason,
                               response_flag=False, response_score=0.0,
                               joint_flag=prompt_flag,
                               joint_reason="prompt-only; " + prompt_reason)
        _, r_score = self._learned_response(prompt, response)
        response_flag = r_score >= self.response_threshold
        joint_flag = prompt_flag or response_flag
        if joint_flag:
            parts = []
            if prompt_flag: parts.append(f"prompt:{prompt_reason}")
            if response_flag:
                parts.append(f"response:p={r_score:.3f}>=tau={self.response_threshold}")
            reason = "; ".join(parts)
        else:
            reason = "both passed"
        return DualVerdict(prompt_flag=prompt_flag, prompt_source=source,
                           prompt_tier=prompt_tier, prompt_reason=prompt_reason,
                           response_flag=response_flag, response_score=r_score,
                           joint_flag=joint_flag, joint_reason=reason)

    def classify_batch(self, prompts: list[str],
                       responses: Optional[list[str]] = None) -> list[DualVerdict]:
        responses = responses or [None] * len(prompts)
        return [self.classify(p, r) for p, r in zip(prompts, responses)]
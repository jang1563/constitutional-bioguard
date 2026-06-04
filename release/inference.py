#!/usr/bin/env python
"""Standalone inference for constitutional-bioguard-response (bio response-harm classifier).

Applies the shipped input-normalization defense (mitigates char-injection
attacks), then classifies a `query [SEP] response` pair. The model judges whether
the RESPONSE delivers harmful biological content (it is not a prompt filter).

Usage:
    python inference.py --query "..." --response "..."
    python inference.py --query "..." --response "..." --threshold 0.5

Or import and call classify(query, response).
"""
from __future__ import annotations

import argparse
import unicodedata

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "jang1563/constitutional-bioguard-response"  # ships v8bh (density-debiased)
DEFAULT_THRESHOLD = 0.5  # default operating point; see model card for v8bh figures


def _fallback_normalize(text: str) -> str:
    """Minimal normalization if the project package is unavailable: strip
    zero-width/variation/tag chars and combining marks, then NFKC."""
    if not text:
        return text
    drop = {0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF, 0x00AD}
    out = []
    for c in text:
        o = ord(c)
        if o in drop or 0xFE00 <= o <= 0xFE0F or 0xE0000 <= o <= 0xE007F:
            continue
        if unicodedata.combining(c):
            continue
        out.append(c)
    return unicodedata.normalize("NFKC", "".join(out))


try:
    from preprocessing import normalize_text          # shipped alongside this file
except Exception:
    try:
        from constitutional_bioguard.preprocessing import normalize_text
    except Exception:  # last resort: minimal built-in (less robust)
        normalize_text = _fallback_normalize


_MODEL = None
_TOK = None


def _load():
    global _MODEL, _TOK
    if _MODEL is None:
        _TOK = AutoTokenizer.from_pretrained(MODEL_ID)
        _MODEL = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).eval()
    return _MODEL, _TOK


def classify(query: str, response: str, threshold: float = DEFAULT_THRESHOLD):
    """Return {'p_harmful': float, 'flag': bool}. Normalizes inputs first."""
    model, tok = _load()
    q, r = normalize_text(query), normalize_text(response)
    # pair encoding tok(query, response) matches how the model was trained/evaluated
    # (do NOT manually concatenate with [SEP]; that yields a different token sequence)
    inp = tok(q, r, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        p = model(**inp).logits.softmax(-1)[0, 1].item()
    return {"p_harmful": round(p, 4), "flag": p >= threshold}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--response", required=True)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    a = ap.parse_args()
    print(classify(a.query, a.response, a.threshold))


if __name__ == "__main__":
    main()

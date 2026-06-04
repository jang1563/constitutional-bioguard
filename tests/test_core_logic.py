"""Unit tests for safety-critical deterministic logic.

Covers the pure functions the rest of the pipeline relies on but that previously
had no coverage: the adversarial input normalizer (`normalize_text`), the JSON
extractor used on every LLM response (`_extract_json`), and the API retry
classifier (`_is_retryable`). All are mock-free and deterministic.
"""

import base64

from constitutional_bioguard.generation.llm_client import _extract_json, _is_retryable
from constitutional_bioguard.preprocessing import normalize_text

# ── normalize_text ───────────────────────────────────────────────────────────


def test_strips_zero_width_chars():
    assert normalize_text("hel​lo") == "hello"


def test_folds_cyrillic_homoglyph():
    # "c" + Cyrillic small a (U+0430) + "t" -> "cat"
    assert normalize_text("cаt") == "cat"


def test_strips_standalone_combining_mark():
    # diacritic-injection: base char + combining acute (U+0301)
    assert normalize_text("cát") == "cat"


def test_nfkc_folds_fullwidth():
    # fullwidth c/a/t -> ascii
    assert normalize_text("ｃａｔ") == "cat"


def test_decodes_rot13_when_plausible_english():
    # rot13("What is the process") -> the obfuscated input below
    assert normalize_text("Jung vf gur cebprff") == "What is the process"


def test_decodes_base64_blob():
    enc = base64.b64encode(b"this is a plaintext sentence").decode()
    assert normalize_text(enc) == "this is a plaintext sentence"


def test_url_decodes_percent_encoding():
    assert normalize_text("hello%20world") == "hello world"


def test_plain_text_passes_through_unchanged():
    # punctuation prevents false base64 match; no markers favor rot13
    text = "How does PCR work?"
    assert normalize_text(text) == text


def test_conservative_short_hex_not_decoded():
    # under the 20-char floor -> left intact (avoids mangling gene IDs etc.)
    assert normalize_text("DEADBEEF") == "DEADBEEF"


def test_empty_string():
    assert normalize_text("") == ""


# ── _extract_json ──────────────────────────────────────────────────────────--


def test_extract_plain_object():
    assert _extract_json('{"a": 1}') == '{"a": 1}'


def test_extract_json_fenced_block():
    assert _extract_json('```json\n{"a": 1}\n```') == '{"a": 1}'


def test_extract_bare_fenced_block():
    assert _extract_json('```\n{"a": 1}\n```') == '{"a": 1}'


def test_extract_object_embedded_in_prose():
    assert _extract_json('Here is the JSON: {"a": 1} hope that helps') == '{"a": 1}'


def test_extract_array():
    assert _extract_json("prefix [1, 2, 3] suffix") == "[1, 2, 3]"


# ── _is_retryable ──────────────────────────────────────────────────────────--


def test_retryable_transient_strings():
    for msg in ("Error: overloaded", "request timeout", "HTTP 529", "rate_limit_error"):
        assert _is_retryable(Exception(msg)) is True, msg


def test_not_retryable_client_errors():
    for msg in ("invalid request: bad schema", "authentication_error", "not found"):
        assert _is_retryable(ValueError(msg)) is False, msg

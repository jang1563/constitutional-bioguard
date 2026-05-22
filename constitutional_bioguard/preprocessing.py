"""Text normalization and decoding for robust adversarial robustness.

Encoding attacks (ROT13: 47.9% ASR, URL-encode: 29.2% ASR, base64, hex) are the
primary adversarial weakness of embedding-based classifiers. This module applies
best-effort decoding before classification to mitigate these attacks.

Usage::

    from constitutional_bioguard.preprocessing import normalize_text

    cleaned = normalize_text(raw_input)  # decode + strip obfuscation
    # then pass cleaned to the classifier

The normalizer is conservative: it only applies a decoding step if the decoded
result looks like plausible natural language (printable ASCII / UTF-8, no binary
noise). If decoding fails or produces garbage, it falls back to the original text.
"""

from __future__ import annotations

import base64
import codecs
import logging
import re
import unicodedata
from collections import Counter
from urllib.parse import unquote as url_unquote

logger = logging.getLogger(__name__)

# ── Invisible / zero-width character stripping ──────────────────────────────

_INVISIBLE_CHARS = re.compile(
    r"[\u200b\u200c\u200d\u200e\u200f\u202a-\u202e\u2060-\u2064\ufeff\u00ad]"
)

# ── Homoglyph normalization (Cyrillic look-alikes -> Latin) ──────────────────

_HOMOGLYPH_MAP: dict[str, str] = {
    "\u0430": "a",  # Cyrillic а
    "\u0435": "e",  # Cyrillic е
    "\u043e": "o",  # Cyrillic о
    "\u0440": "r",  # Cyrillic р
    "\u0441": "c",  # Cyrillic с
    "\u0445": "x",  # Cyrillic х
    "\u0456": "i",  # Ukrainian і
    "\u0455": "s",  # Cyrillic ѕ
    "\u04cf": "l",  # Cyrillic ӏ
}

_HOMOGLYPH_TABLE = str.maketrans(_HOMOGLYPH_MAP)


def _strip_invisible(text: str) -> str:
    return _INVISIBLE_CHARS.sub("", text)


def _normalize_homoglyphs(text: str) -> str:
    return text.translate(_HOMOGLYPH_TABLE)


def _is_plausible_text(text: str, min_printable_ratio: float = 0.85) -> bool:
    """Return True if text looks like natural language (not binary garbage)."""
    if not text or len(text) < 3:
        return False
    printable = sum(1 for c in text if c.isprintable() or c in "\n\r\t")
    return (printable / len(text)) >= min_printable_ratio


def _english_marker_score(text: str) -> int:
    """Count common English markers that are useful for ROT13 detection."""
    words = re.findall(r"[a-zA-Z]{2,}", text.lower())
    if not words:
        return 0
    common = {
        "a", "an", "and", "are", "as", "be", "can", "could", "for", "from",
        "how", "i", "in", "is", "it", "of", "on", "or", "that", "the",
        "this", "to", "what", "when", "where", "which", "with", "would",
        "you", "your",
    }
    counts = Counter(words)
    return sum(counts[word] for word in common)


# ── ROT13 ────────────────────────────────────────────────────────────────────

def _try_decode_rot13(text: str) -> str | None:
    """Decode ROT13 if the result looks like natural language."""
    try:
        decoded = codecs.decode(text, "rot_13")
        if (
            _is_plausible_text(decoded)
            and decoded != text
            and _english_marker_score(decoded) > _english_marker_score(text)
        ):
            return decoded
    except Exception:
        pass
    return None


# ── Base64 ───────────────────────────────────────────────────────────────────

_B64_PATTERN = re.compile(r"^[A-Za-z0-9+/\s]+=*$")


def _try_decode_base64(text: str) -> str | None:
    """Decode base64 if the input looks like a base64 string."""
    stripped = text.strip().replace("\n", "").replace(" ", "")
    # Must be mostly base64 characters and reasonably long
    if len(stripped) < 20 or not _B64_PATTERN.match(stripped):
        return None
    # Pad if needed
    padding = 4 - len(stripped) % 4
    if padding < 4:
        stripped += "=" * padding
    try:
        decoded_bytes = base64.b64decode(stripped)
        decoded = decoded_bytes.decode("utf-8", errors="strict")
        if _is_plausible_text(decoded):
            return decoded
    except Exception:
        pass
    return None


# ── URL encoding ─────────────────────────────────────────────────────────────

_URL_ENCODED = re.compile(r"%[0-9A-Fa-f]{2}")


def _try_decode_url(text: str) -> str | None:
    """URL-decode if the text contains percent-encoded sequences."""
    if not _URL_ENCODED.search(text):
        return None
    try:
        decoded = url_unquote(text)
        if decoded != text and _is_plausible_text(decoded):
            return decoded
    except Exception:
        pass
    return None


# ── Hex encoding ─────────────────────────────────────────────────────────────

_HEX_PATTERN = re.compile(r"^(?:0x)?([0-9A-Fa-f\s]+)$")
_HEX_BYTE_PATTERN = re.compile(r"(?:0x)?[0-9A-Fa-f]{2}")


def _try_decode_hex(text: str) -> str | None:
    """Decode hex string (e.g. '48656c6c6f') if plausible."""
    stripped = text.strip()
    # Require at least 20 hex chars to avoid false positives on gene IDs, etc.
    if len(stripped) < 20:
        return None
    # Remove 0x prefixes and spaces
    clean = re.sub(r"(0x|\s)", "", stripped)
    if not re.fullmatch(r"[0-9A-Fa-f]+", clean) or len(clean) % 2 != 0:
        return None
    try:
        decoded = bytes.fromhex(clean).decode("utf-8", errors="strict")
        if _is_plausible_text(decoded):
            return decoded
    except Exception:
        pass
    return None


# ── Backspace obfuscation ─────────────────────────────────────────────────────

_BACKSPACE_PATTERN = re.compile(r".\x08")


def _strip_backspaces(text: str) -> str:
    """Remove backspace-overwrite obfuscation (e.g. 'a\x08b' -> 'b')."""
    while _BACKSPACE_PATTERN.search(text):
        text = _BACKSPACE_PATTERN.sub("", text)
    return text


# ── Unicode normalization ─────────────────────────────────────────────────────

def _unicode_normalize(text: str) -> str:
    """Apply NFKC normalization to collapse confusable Unicode variants."""
    return unicodedata.normalize("NFKC", text)


# ── Public API ────────────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """Apply all normalization steps to a query/response string.

    Order of operations:
    1. Strip invisible / zero-width characters
    2. Backspace obfuscation removal
    3. URL-decode (percent-encoding)
    4. Base64 decode (if text looks like a base64 blob)
    5. Hex decode (if text looks like a hex string)
    6. ROT13 decode (if decoded result is plausible)
    7. Homoglyph normalization (Cyrillic look-alikes -> Latin)
    8. NFKC Unicode normalization

    Only a single encoding layer is unwrapped per call. For nested encodings
    (e.g. base64-of-rot13), call normalize_text twice.

    Args:
        text: Raw input string.

    Returns:
        Normalized string. Falls back to original if all decodings fail.
    """
    if not text:
        return text

    # Step 1: strip invisible chars
    text = _strip_invisible(text)

    # Step 2: backspace obfuscation
    text = _strip_backspaces(text)

    # Steps 3-6: try encoding decoders in order of specificity
    for decoder_fn in (_try_decode_url, _try_decode_base64, _try_decode_hex, _try_decode_rot13):
        decoded = decoder_fn(text)
        if decoded is not None:
            logger.debug("Decoded with %s: %d -> %d chars", decoder_fn.__name__, len(text), len(decoded))
            text = decoded
            break  # one layer at a time

    # Step 7: homoglyph normalization
    text = _normalize_homoglyphs(text)

    # Step 8: NFKC
    text = _unicode_normalize(text)

    return text


def normalize_query_response(query: str, response: str) -> tuple[str, str]:
    """Normalize both halves of a classifier input pair.

    Args:
        query: The user query string.
        response: The assistant response string.

    Returns:
        Tuple of (normalized_query, normalized_response).
    """
    return normalize_text(query), normalize_text(response)

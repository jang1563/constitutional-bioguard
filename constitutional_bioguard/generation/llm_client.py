"""Claude API wrapper with rate limiting and retry logic.

Adapted from BioThreat-Eval's llm_client.py rate-limiting patterns.
This client is purpose-built for Constitutional BioGuard's synthetic data
generation pipeline, using the Anthropic Claude API exclusively.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional, TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv(Path(__file__).parent.parent.parent / ".env")

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# ── Rate Limiting State ──────────────────────────────────────────────────────
_last_call_time: float = 0.0
_call_count: int = 0


def _rate_limit(rpm: int) -> None:
    """Sleep if needed to respect RPM limit."""
    global _last_call_time, _call_count
    if rpm <= 0:
        return
    min_interval = 60.0 / rpm
    now = time.time()
    elapsed = now - _last_call_time
    if elapsed < min_interval:
        sleep_time = min_interval - elapsed
        logger.debug("Rate limit: sleeping %.2fs", sleep_time)
        time.sleep(sleep_time)
    _last_call_time = time.time()
    _call_count += 1


# ── Client Singleton ─────────────────────────────────────────────────────────
_client = None


def get_client():
    """Get or create the Anthropic client."""
    global _client
    if _client is not None:
        return _client

    import anthropic

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY not set. "
            "Set it in .env or as an environment variable."
        )
    _client = anthropic.Anthropic(api_key=api_key)
    return _client


# ── Core API Call ────────────────────────────────────────────────────────────


def call_claude(
    prompt: str,
    model: str = "claude-sonnet-4-20250514",
    system: Optional[str] = None,
    temperature: float = 0.8,
    max_tokens: int = 2048,
    rpm: int = 40,
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> str:
    """Call Claude API and return the text response.

    Args:
        prompt: The user message to send.
        model: Model identifier.
        system: Optional system prompt.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in response.
        rpm: Requests per minute limit.
        max_retries: Number of retries on transient errors.
        retry_delay: Base delay between retries (exponential backoff).

    Returns:
        The text content of Claude's response.

    Raises:
        RuntimeError: After all retries are exhausted.
    """
    client = get_client()
    messages = [{"role": "user", "content": prompt}]
    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": messages,
    }
    if system:
        kwargs["system"] = system

    for attempt in range(max_retries + 1):
        _rate_limit(rpm)
        try:
            response = client.messages.create(**kwargs)
            text = response.content[0].text
            if not text:
                raise RuntimeError("Claude returned empty response")
            return text
        except Exception as e:
            error_str = str(e)
            # Retry on transient errors
            if attempt < max_retries and any(
                keyword in error_str.lower()
                for keyword in ["overloaded", "rate_limit", "timeout", "529"]
            ):
                wait = retry_delay * (2**attempt)
                logger.warning(
                    "Transient error (attempt %d/%d): %s. Retrying in %.1fs",
                    attempt + 1,
                    max_retries,
                    error_str[:100],
                    wait,
                )
                time.sleep(wait)
                continue
            raise RuntimeError(
                f"Claude API call failed after {attempt + 1} attempt(s): {error_str}"
            ) from e

    raise RuntimeError("Unreachable: max retries exhausted")


def call_claude_json(
    prompt: str,
    output_model: type[T],
    model: str = "claude-sonnet-4-20250514",
    system: Optional[str] = None,
    temperature: float = 0.8,
    max_tokens: int = 2048,
    rpm: int = 40,
    max_retries: int = 3,
) -> T:
    """Call Claude and parse the response as a Pydantic model.

    The prompt should instruct Claude to respond with valid JSON matching
    the expected schema. This function extracts JSON from the response
    (handling markdown code blocks) and validates it against the model.

    Args:
        prompt: The user message (should request JSON output).
        output_model: Pydantic model class to parse into.
        model: Model identifier.
        system: Optional system prompt.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in response.
        rpm: Requests per minute limit.
        max_retries: Number of retries (includes JSON parse retries).

    Returns:
        Validated Pydantic model instance.
    """
    for attempt in range(max_retries + 1):
        text = call_claude(
            prompt=prompt,
            model=model,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            rpm=rpm,
            max_retries=0,  # handle retries at this level
        )

        # Extract JSON from potential markdown code blocks
        json_text = _extract_json(text)

        try:
            return output_model.model_validate_json(json_text)
        except Exception as e:
            if attempt < max_retries:
                logger.warning(
                    "JSON parse failed (attempt %d/%d): %s",
                    attempt + 1,
                    max_retries,
                    str(e)[:100],
                )
                continue
            raise RuntimeError(
                f"Failed to parse Claude response as {output_model.__name__} "
                f"after {attempt + 1} attempts: {e}\n"
                f"Raw response: {text[:500]}"
            ) from e

    raise RuntimeError("Unreachable: max retries exhausted")


def _extract_json(text: str) -> str:
    """Extract JSON from text, handling markdown code blocks."""
    text = text.strip()

    # Handle ```json ... ``` blocks
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json) and last line (```)
        start = 1
        end = len(lines)
        for i in range(len(lines) - 1, 0, -1):
            if lines[i].strip() == "```":
                end = i
                break
        text = "\n".join(lines[start:end]).strip()

    # Handle cases where JSON is embedded in other text
    # Find the first { or [ and last } or ]
    first_brace = -1
    for i, ch in enumerate(text):
        if ch in "{[":
            first_brace = i
            break

    if first_brace >= 0:
        last_brace = -1
        target = "}" if text[first_brace] == "{" else "]"
        for i in range(len(text) - 1, first_brace, -1):
            if text[i] == target:
                last_brace = i
                break
        if last_brace > first_brace:
            text = text[first_brace : last_brace + 1]

    return text


def get_call_count() -> int:
    """Return the total number of API calls made this session."""
    return _call_count

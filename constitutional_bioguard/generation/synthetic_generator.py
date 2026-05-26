"""Synthetic data generation from bio-safety constitution rules.

For each constitution rule, generates:
  - N_PERMITTED permitted examples (SAFE label)
  - N_RESTRICTED restricted examples (UNSAFE label)
  - N_BOUNDARY boundary examples (near the decision boundary)

Uses Claude API to generate realistic query-response pairs that
exercise the permitted/restricted boundary defined in each rule.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from constitutional_bioguard.config import (
    DATA_RAW,
    EXAMPLES_PER_RULE_BOUNDARY,
    EXAMPLES_PER_RULE_PERMITTED,
    EXAMPLES_PER_RULE_RESTRICTED,
    GENERATION_MAX_TOKENS,
    GENERATION_MODEL,
    GENERATION_RPM,
    GENERATION_TEMPERATURE,
    RANDOM_SEED,
)
from constitutional_bioguard.generation.llm_client import call_claude
from constitutional_bioguard.models import (
    AugmentationType,
    ConstitutionRule,
    FineLabel,
    SafetyLabel,
    SyntheticExample,
)

logger = logging.getLogger(__name__)

# ── Prompt Templates ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert in biosafety, dual-use research of concern, and biological \
threat assessment. You generate realistic synthetic examples of queries and \
responses for training a safety classifier. Your examples must be realistic, \
diverse, and clearly illustrate the boundary between permitted and restricted \
content as defined by the constitution rule provided."""

PERMITTED_PROMPT_TEMPLATE = """\
Generate {n} diverse PERMITTED (safe) query-response pairs for the following \
bio-safety constitution rule.

RULE: {rule_id} - {title}
CATEGORY: {category}
WHAT IS PERMITTED: {permitted}
EXAMPLE: {permitted_example}
THREAT LEVEL BOUNDARY: Content is permitted up to level {boundary} \
(1=general biology, 2=specific pathogen, 3=technical methodology, \
4=applied dual-use, 5=operational specifics)

Requirements:
- Each query should be a realistic question someone might ask an AI assistant
- The response should be helpful and informative (this is SAFE content)
- Vary the specificity across threat levels 1 through {boundary}
- Vary the format: some queries short, some detailed; some from students, \
  researchers, clinicians, public health officials
- Each example should be clearly WITHIN the permitted boundary
- Do NOT include any restricted/unsafe content

Return a JSON array of objects, each with "query" and "response" fields:
[{{"query": "...", "response": "..."}}, ...]

Generate exactly {n} examples."""

RESTRICTED_PROMPT_TEMPLATE = """\
Generate {n} diverse RESTRICTED (unsafe) query-response pairs for the following \
bio-safety constitution rule.

RULE: {rule_id} - {title}
CATEGORY: {category}
WHAT IS RESTRICTED: {restricted}
EXAMPLE: {restricted_example}
THREAT LEVEL BOUNDARY: Content above level {boundary} is restricted \
(1=general biology, 2=specific pathogen, 3=technical methodology, \
4=applied dual-use, 5=operational specifics)

Requirements:
- Each query should be a realistic attempt to obtain restricted information
- The response should be a COMPLIANT response that provides the restricted \
  information (we need examples of what unsafe responses look like)
- Vary the approach: direct questions, indirect framing, academic framing, \
  hypothetical scenarios
- Each example should CLEARLY cross the restricted boundary
- The content should be realistic but use safe proxy language — do NOT include \
  actual operational protocols for causing harm

IMPORTANT: These examples are for training a safety classifier. The "response" \
field should show what an UNSAFE AI response would look like — one that \
helpfully provides restricted information. This is necessary for the classifier \
to learn what to flag.

Return a JSON array of objects, each with "query" and "response" fields:
[{{"query": "...", "response": "..."}}, ...]

Generate exactly {n} examples."""

BOUNDARY_PROMPT_TEMPLATE = """\
Generate {n} BOUNDARY query-response pairs that sit right at the edge between \
permitted and restricted content for the following bio-safety constitution rule.

RULE: {rule_id} - {title}
CATEGORY: {category}
PERMITTED: {permitted}
RESTRICTED: {restricted}
THREAT LEVEL BOUNDARY: {boundary}

Requirements:
- These examples should be AMBIGUOUS — reasonable people could disagree on \
  whether they cross the line
- Some should be "boundary_permitted" (just barely safe) and some should be \
  "boundary_restricted" (just barely unsafe)
- The queries should be legitimate-sounding but probe the exact boundary
- Include a "label" field: "boundary_permitted" or "boundary_restricted"
- Include a "reasoning" field explaining why this is a boundary case

Return a JSON array of objects:
[{{"query": "...", "response": "...", "label": "boundary_permitted|boundary_restricted", \
"reasoning": "..."}}, ...]

Generate exactly {n} examples."""


# ── Generation Functions ─────────────────────────────────────────────────────


def _make_example_id(rule_id: str, label_type: str, index: int) -> str:
    """Generate a deterministic example ID."""
    raw = f"{rule_id}_{label_type}_{index}"
    short_hash = hashlib.sha256(raw.encode()).hexdigest()[:8]
    return f"{rule_id}_{label_type}_{index:03d}_{short_hash}"


def _parse_json_array(text: str) -> list[dict]:
    """Parse a JSON array from Claude's response, handling common issues."""
    text = text.strip()

    # Remove markdown code blocks
    if text.startswith("```"):
        lines = text.split("\n")
        start = 1
        end = len(lines)
        for i in range(len(lines) - 1, 0, -1):
            if lines[i].strip() == "```":
                end = i
                break
        text = "\n".join(lines[start:end]).strip()

    # Find the array
    first_bracket = text.find("[")
    last_bracket = text.rfind("]")
    if first_bracket >= 0 and last_bracket > first_bracket:
        text = text[first_bracket : last_bracket + 1]

    return json.loads(text)


def generate_permitted(
    rule: ConstitutionRule,
    n: int = EXAMPLES_PER_RULE_PERMITTED,
    model: str = GENERATION_MODEL,
    temperature: float = GENERATION_TEMPERATURE,
) -> list[SyntheticExample]:
    """Generate PERMITTED (safe) examples for a rule."""
    prompt = PERMITTED_PROMPT_TEMPLATE.format(
        n=n,
        rule_id=rule.rule_id,
        title=rule.title,
        category=rule.category.value,
        permitted=rule.permitted,
        permitted_example=rule.examples.permitted_example,
        boundary=rule.threat_level_boundary,
    )

    response_text = call_claude(
        prompt=prompt,
        model=model,
        system=SYSTEM_PROMPT,
        temperature=temperature,
        max_tokens=GENERATION_MAX_TOKENS,
        rpm=GENERATION_RPM,
    )

    raw_examples = _parse_json_array(response_text)

    examples = []
    for i, raw in enumerate(raw_examples[:n]):
        example = SyntheticExample(
            example_id=_make_example_id(rule.rule_id, "permitted", i),
            rule_id=rule.rule_id,
            category=rule.category,
            query=raw["query"],
            response=raw["response"],
            fine_label=FineLabel.PERMITTED,
            safety_label=SafetyLabel.SAFE,
            augmentation=AugmentationType.ORIGINAL,
        )
        examples.append(example)

    return examples


def generate_restricted(
    rule: ConstitutionRule,
    n: int = EXAMPLES_PER_RULE_RESTRICTED,
    model: str = GENERATION_MODEL,
    temperature: float = GENERATION_TEMPERATURE,
) -> list[SyntheticExample]:
    """Generate RESTRICTED (unsafe) examples for a rule."""
    prompt = RESTRICTED_PROMPT_TEMPLATE.format(
        n=n,
        rule_id=rule.rule_id,
        title=rule.title,
        category=rule.category.value,
        restricted=rule.restricted,
        restricted_example=rule.examples.restricted_example,
        boundary=rule.threat_level_boundary,
    )

    response_text = call_claude(
        prompt=prompt,
        model=model,
        system=SYSTEM_PROMPT,
        temperature=temperature,
        max_tokens=GENERATION_MAX_TOKENS,
        rpm=GENERATION_RPM,
    )

    raw_examples = _parse_json_array(response_text)

    examples = []
    for i, raw in enumerate(raw_examples[:n]):
        example = SyntheticExample(
            example_id=_make_example_id(rule.rule_id, "restricted", i),
            rule_id=rule.rule_id,
            category=rule.category,
            query=raw["query"],
            response=raw["response"],
            fine_label=FineLabel.RESTRICTED,
            safety_label=SafetyLabel.UNSAFE,
            augmentation=AugmentationType.ORIGINAL,
        )
        examples.append(example)

    return examples


def generate_boundary(
    rule: ConstitutionRule,
    n: int = EXAMPLES_PER_RULE_BOUNDARY,
    model: str = GENERATION_MODEL,
    temperature: float = GENERATION_TEMPERATURE,
) -> list[SyntheticExample]:
    """Generate BOUNDARY examples for a rule (near the decision boundary)."""
    prompt = BOUNDARY_PROMPT_TEMPLATE.format(
        n=n,
        rule_id=rule.rule_id,
        title=rule.title,
        category=rule.category.value,
        permitted=rule.permitted,
        restricted=rule.restricted,
        boundary=rule.threat_level_boundary,
    )

    response_text = call_claude(
        prompt=prompt,
        model=model,
        system=SYSTEM_PROMPT,
        temperature=temperature,
        max_tokens=GENERATION_MAX_TOKENS,
        rpm=GENERATION_RPM,
    )

    raw_examples = _parse_json_array(response_text)

    examples = []
    for i, raw in enumerate(raw_examples[:n]):
        label_str = raw.get("label", "boundary_permitted")
        if label_str == "boundary_restricted":
            fine_label = FineLabel.BOUNDARY_RESTRICTED
            safety_label = SafetyLabel.UNSAFE
        else:
            fine_label = FineLabel.BOUNDARY_PERMITTED
            safety_label = SafetyLabel.SAFE

        example = SyntheticExample(
            example_id=_make_example_id(rule.rule_id, "boundary", i),
            rule_id=rule.rule_id,
            category=rule.category,
            query=raw["query"],
            response=raw["response"],
            fine_label=fine_label,
            safety_label=safety_label,
            augmentation=AugmentationType.ORIGINAL,
            boundary_reasoning=raw.get("reasoning"),
        )
        examples.append(example)

    return examples


# ── Batch Generation ─────────────────────────────────────────────────────────


def generate_all(
    rules: list[ConstitutionRule],
    output_dir: Optional[Path] = None,
    resume: bool = True,
) -> list[SyntheticExample]:
    """Generate synthetic data for all constitution rules.

    Saves incrementally to JSONL files (one per label type) for resumability.
    Completion is tracked per type independently — a rule is fully skipped only
    when all three types (permitted, restricted, boundary) are already present.

    Args:
        rules: List of constitution rules to generate for.
        output_dir: Where to save JSONL files. Defaults to DATA_RAW.
        resume: If True, skip types already generated for each rule.

    Returns:
        All generated examples (from this run only; does not re-load prior data).
    """
    output_dir = output_dir or DATA_RAW
    output_dir.mkdir(parents=True, exist_ok=True)

    permitted_file = output_dir / "synthetic_permitted.jsonl"
    restricted_file = output_dir / "synthetic_restricted.jsonl"
    boundary_file = output_dir / "synthetic_boundary.jsonl"

    # Track completion per type independently so a partial rule is resumed correctly
    permitted_done: set = set()
    restricted_done: set = set()
    boundary_done: set = set()

    if resume:
        for fpath, done_set in [
            (permitted_file, permitted_done),
            (restricted_file, restricted_done),
            (boundary_file, boundary_done),
        ]:
            if fpath.exists():
                with open(fpath, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            done_set.add(json.loads(line).get("rule_id"))

    fully_done = permitted_done & restricted_done & boundary_done
    rules_to_process = [r for r in rules if r.rule_id not in fully_done]

    if resume and (permitted_done or restricted_done or boundary_done):
        logger.info(
            "Resuming: %d rules fully done, processing %d "
            "(%d permitted / %d restricted / %d boundary types cached)",
            len(fully_done),
            len(rules_to_process),
            len(permitted_done),
            len(restricted_done),
            len(boundary_done),
        )

    all_examples = []
    random.seed(RANDOM_SEED)

    for rule in tqdm(rules_to_process, desc="Generating synthetic data"):
        examples = []

        if rule.rule_id not in permitted_done:
            try:
                permitted = generate_permitted(rule)
                examples.extend(permitted)
                logger.info("  Generated %d permitted examples", len(permitted))
            except Exception as e:
                logger.error("  Failed to generate permitted examples: %s", e)

        if rule.rule_id not in restricted_done:
            try:
                restricted = generate_restricted(rule)
                examples.extend(restricted)
                logger.info("  Generated %d restricted examples", len(restricted))
            except Exception as e:
                logger.error("  Failed to generate restricted examples: %s", e)

        if rule.rule_id not in boundary_done:
            try:
                boundary = generate_boundary(rule)
                examples.extend(boundary)
                logger.info("  Generated %d boundary examples", len(boundary))
            except Exception as e:
                logger.error("  Failed to generate boundary examples: %s", e)

        all_examples.extend(examples)

        # Append to JSONL files incrementally
        for example in examples:
            if example.fine_label == FineLabel.PERMITTED:
                target_file = permitted_file
            elif example.fine_label == FineLabel.RESTRICTED:
                target_file = restricted_file
            else:
                target_file = boundary_file

            with open(target_file, "a", encoding="utf-8") as f:
                f.write(example.model_dump_json() + "\n")

    # Summary
    from collections import Counter

    label_counts = Counter(e.safety_label.value for e in all_examples)
    fine_counts = Counter(e.fine_label.value for e in all_examples)
    logger.info(
        "Generated %d total examples: %s (fine: %s)",
        len(all_examples),
        dict(label_counts),
        dict(fine_counts),
    )

    return all_examples


def load_existing_examples(data_dir: Optional[Path] = None) -> list[SyntheticExample]:
    """Load previously generated synthetic examples from JSONL files."""
    data_dir = data_dir or DATA_RAW
    examples = []

    for filename in [
        "synthetic_permitted.jsonl",
        "synthetic_restricted.jsonl",
        "synthetic_boundary.jsonl",
    ]:
        filepath = data_dir / filename
        if filepath.exists():
            with open(filepath, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        examples.append(SyntheticExample.model_validate_json(line))

    logger.info("Loaded %d existing synthetic examples from %s", len(examples), data_dir)
    return examples

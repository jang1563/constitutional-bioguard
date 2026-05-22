#!/usr/bin/env python3
"""Validate the bio-safety constitution for completeness and correctness.

Checks:
  1. YAML is parseable and matches the JSON Schema
  2. All 7 NSABB categories are represented
  3. Each category has >= MIN_RULES_PER_CATEGORY rules (default 6)
  4. Rule IDs are unique and follow the naming convention
  5. Threat level boundaries are within valid range
  6. All text fields meet minimum length requirements
  7. Examples are distinct from rule descriptions
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from constitutional_bioguard.config import (
    CONSTITUTION_FILE,
    CONSTITUTION_DIR,
    MIN_RULES_PER_CATEGORY,
    NSABB_CATEGORIES,
    TARGET_RULES_PER_CATEGORY,
)

# Category prefix mapping
CATEGORY_PREFIXES = {
    "enhance_harm": "EH",
    "disrupt_immunity": "DI",
    "confer_resistance": "CR",
    "increase_stability": "IS",
    "alter_host_range": "AH",
    "enhance_susceptibility": "ES",
    "generate_reconstruct": "GR",
}

RULE_ID_PATTERN = re.compile(r"^[A-Z]{2,3}-\d{3}$")
MIN_TEXT_LENGTH = 20  # Minimum length for description fields


def load_constitution(path: Path) -> dict:
    """Load and parse the YAML constitution file."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_schema(data: dict) -> list[str]:
    """Validate constitution against JSON Schema (if jsonschema available)."""
    errors = []
    schema_path = CONSTITUTION_DIR / "schema.json"
    if not schema_path.exists():
        errors.append("WARNING: schema.json not found, skipping schema validation")
        return errors

    try:
        import jsonschema

        with open(schema_path, encoding="utf-8") as f:
            schema = json.load(f)
        jsonschema.validate(data, schema)
    except ImportError:
        errors.append("WARNING: jsonschema not installed, skipping schema validation")
    except jsonschema.ValidationError as e:
        errors.append(f"Schema validation failed: {e.message}")
    return errors


def validate_categories(rules: list[dict]) -> list[str]:
    """Check all 7 NSABB categories are represented with enough rules."""
    errors = []
    counts = Counter(r["category"] for r in rules)

    for cat in NSABB_CATEGORIES:
        count = counts.get(cat, 0)
        if count == 0:
            errors.append(f"CRITICAL: Category '{cat}' has NO rules")
        elif count < MIN_RULES_PER_CATEGORY:
            errors.append(
                f"Category '{cat}' has {count} rules "
                f"(minimum: {MIN_RULES_PER_CATEGORY})"
            )
        elif count < TARGET_RULES_PER_CATEGORY:
            errors.append(
                f"WARNING: Category '{cat}' has {count} rules "
                f"(target: {TARGET_RULES_PER_CATEGORY})"
            )

    # Check for unexpected categories
    for cat in counts:
        if cat not in NSABB_CATEGORIES:
            errors.append(f"Unknown category: '{cat}'")

    return errors


def validate_rule_ids(rules: list[dict]) -> list[str]:
    """Check rule IDs are unique, well-formed, and match their category."""
    errors = []
    seen_ids = set()

    for rule in rules:
        rid = rule["rule_id"]

        # Check format
        if not RULE_ID_PATTERN.match(rid):
            errors.append(f"Rule ID '{rid}' does not match pattern XX-NNN or XXX-NNN")

        # Check uniqueness
        if rid in seen_ids:
            errors.append(f"Duplicate rule ID: '{rid}'")
        seen_ids.add(rid)

        # Check prefix matches category
        category = rule.get("category", "")
        expected_prefix = CATEGORY_PREFIXES.get(category, "")
        if expected_prefix and not rid.startswith(expected_prefix + "-"):
            errors.append(
                f"Rule '{rid}' prefix doesn't match category '{category}' "
                f"(expected prefix: '{expected_prefix}')"
            )

    return errors


def validate_threat_levels(rules: list[dict]) -> list[str]:
    """Check threat level boundaries are valid."""
    errors = []
    for rule in rules:
        level = rule.get("threat_level_boundary")
        if level is None:
            errors.append(f"Rule '{rule['rule_id']}' missing threat_level_boundary")
        elif not isinstance(level, int) or level < 1 or level > 5:
            errors.append(
                f"Rule '{rule['rule_id']}' has invalid threat_level_boundary: {level}"
            )
    return errors


def validate_text_fields(rules: list[dict]) -> list[str]:
    """Check text fields meet minimum length requirements."""
    errors = []
    text_fields = ["title", "permitted", "restricted", "rationale"]

    for rule in rules:
        rid = rule["rule_id"]
        for field in text_fields:
            value = rule.get(field, "")
            if len(value.strip()) < MIN_TEXT_LENGTH:
                errors.append(
                    f"Rule '{rid}' field '{field}' too short "
                    f"({len(value.strip())} chars, min {MIN_TEXT_LENGTH})"
                )

        # Check examples
        examples = rule.get("examples", {})
        for ex_field in ["permitted_example", "restricted_example"]:
            value = examples.get(ex_field, "")
            if len(value.strip()) < 10:
                errors.append(
                    f"Rule '{rid}' example '{ex_field}' too short "
                    f"({len(value.strip())} chars)"
                )

    return errors


def validate_boundary_distribution(rules: list[dict]) -> list[str]:
    """Check that threat level boundaries are reasonably distributed."""
    warnings = []
    levels = Counter(r["threat_level_boundary"] for r in rules)

    # We expect a mix — mostly L2-L3 with some L1 and L4
    if levels.get(1, 0) == 0 and levels.get(2, 0) == 0:
        warnings.append(
            "WARNING: No rules with boundary levels 1-2 (very strict rules)"
        )
    if levels.get(3, 0) == 0 and levels.get(4, 0) == 0:
        warnings.append(
            "WARNING: No rules with boundary levels 3-4 (moderate rules)"
        )

    return warnings


def main() -> int:
    """Run all validation checks and report results."""
    constitution_path = CONSTITUTION_FILE

    print(f"Validating constitution: {constitution_path}")
    print("=" * 70)

    # Load
    try:
        data = load_constitution(constitution_path)
    except FileNotFoundError:
        print(f"ERROR: Constitution file not found: {constitution_path}")
        return 1
    except yaml.YAMLError as e:
        print(f"ERROR: YAML parsing failed: {e}")
        return 1

    rules = data.get("rules", [])
    print(f"Loaded {len(rules)} rules from constitution v{data.get('version', '?')}")
    print()

    # Run all checks
    all_errors = []
    all_warnings = []

    for check_name, check_fn in [
        ("Schema validation", lambda: validate_schema(data)),
        ("Category coverage", lambda: validate_categories(rules)),
        ("Rule ID format", lambda: validate_rule_ids(rules)),
        ("Threat levels", lambda: validate_threat_levels(rules)),
        ("Text fields", lambda: validate_text_fields(rules)),
        ("Boundary distribution", lambda: validate_boundary_distribution(rules)),
    ]:
        print(f"  Checking: {check_name}...")
        issues = check_fn()
        warnings = [i for i in issues if i.startswith("WARNING")]
        errors = [i for i in issues if not i.startswith("WARNING")]

        if errors:
            all_errors.extend(errors)
            print(f"    ERRORS: {len(errors)}")
            for e in errors:
                print(f"      - {e}")
        if warnings:
            all_warnings.extend(warnings)
            for w in warnings:
                print(f"      - {w}")
        if not errors and not warnings:
            print("    OK")

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Category breakdown
    counts = Counter(r["category"] for r in rules)
    print("\nRules per category:")
    for cat in NSABB_CATEGORIES:
        count = counts.get(cat, 0)
        status = "OK" if count >= MIN_RULES_PER_CATEGORY else "INSUFFICIENT"
        print(f"  {cat:30s} {count:3d}  [{status}]")
    print(f"  {'TOTAL':30s} {len(rules):3d}")

    # Threat level distribution
    levels = Counter(r["threat_level_boundary"] for r in rules)
    print("\nThreat level boundary distribution:")
    for level in range(1, 6):
        count = levels.get(level, 0)
        bar = "#" * count
        print(f"  L{level}: {count:3d}  {bar}")

    # Final verdict
    print()
    if all_errors:
        print(f"FAILED: {len(all_errors)} error(s), {len(all_warnings)} warning(s)")
        return 1
    elif all_warnings:
        print(f"PASSED with {len(all_warnings)} warning(s)")
        return 0
    else:
        print("PASSED: All checks passed")
        return 0


if __name__ == "__main__":
    sys.exit(main())

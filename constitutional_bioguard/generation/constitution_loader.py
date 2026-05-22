"""Load and parse the bio-safety constitution YAML file.

Converts the YAML constitution into validated Pydantic models
for consumption by the synthetic data generation pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import yaml

from constitutional_bioguard.config import CONSTITUTION_FILE
from constitutional_bioguard.models import (
    Constitution,
    ConstitutionRule,
    NSABBCategory,
    RuleExamples,
)

logger = logging.getLogger(__name__)


def load_constitution(path: Optional[Path] = None) -> Constitution:
    """Load the bio-safety constitution from YAML.

    Args:
        path: Path to the YAML file. Defaults to CONSTITUTION_FILE from config.

    Returns:
        Validated Constitution model.

    Raises:
        FileNotFoundError: If the constitution file doesn't exist.
        ValueError: If the YAML is invalid or fails validation.
    """
    path = path or CONSTITUTION_FILE

    if not path.exists():
        raise FileNotFoundError(f"Constitution file not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Constitution YAML must be a mapping, got {type(data)}")

    # Parse rules into Pydantic models
    rules = []
    for i, rule_data in enumerate(data.get("rules", [])):
        try:
            rule = _parse_rule(rule_data)
            rules.append(rule)
        except Exception as e:
            raise ValueError(
                f"Error parsing rule {i} (ID: {rule_data.get('rule_id', '?')}): {e}"
            ) from e

    constitution = Constitution(
        name=data.get("name", "Unnamed Constitution"),
        version=data.get("version", "0.0.0"),
        author=data.get("author", "Unknown"),
        rules=rules,
    )

    logger.info(
        "Loaded constitution '%s' v%s with %d rules across %d categories",
        constitution.name,
        constitution.version,
        len(constitution.rules),
        len(constitution.category_counts()),
    )

    return constitution


def _parse_rule(data: dict) -> ConstitutionRule:
    """Parse a single rule from YAML dict to Pydantic model."""
    examples_data = data.get("examples", {})
    examples = RuleExamples(
        permitted_example=examples_data.get("permitted_example", "").strip(),
        restricted_example=examples_data.get("restricted_example", "").strip(),
    )

    return ConstitutionRule(
        rule_id=data["rule_id"],
        category=NSABBCategory(data["category"]),
        title=data["title"],
        permitted=data["permitted"].strip(),
        restricted=data["restricted"].strip(),
        rationale=data["rationale"].strip(),
        threat_level_boundary=data["threat_level_boundary"],
        examples=examples,
    )


def get_rules_by_category(
    constitution: Constitution,
) -> dict[str, list[ConstitutionRule]]:
    """Group rules by NSABB category.

    Returns:
        Dict mapping category value string to list of rules.
    """
    grouped: dict[str, list[ConstitutionRule]] = {}
    for cat in NSABBCategory:
        rules = constitution.rules_by_category(cat)
        if rules:
            grouped[cat.value] = rules
    return grouped

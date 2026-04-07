"""NSABB-based dual-use taxonomy adapter.

Adapted from BioThreat-Eval taxonomy.py. Provides category and level descriptions
used in constitution validation and synthetic data generation.
"""

from constitutional_bioguard.models import NSABBCategory, ThreatLevel

CATEGORY_DESCRIPTIONS: dict[NSABBCategory, str] = {
    NSABBCategory.ENHANCE_HARM: "Enhance harmful consequences of a biological agent",
    NSABBCategory.DISRUPT_IMMUNITY: "Disrupt immunity or effectiveness of immunization",
    NSABBCategory.CONFER_RESISTANCE: "Confer resistance to therapeutics",
    NSABBCategory.INCREASE_STABILITY: (
        "Increase stability, transmissibility, or dissemination"
    ),
    NSABBCategory.ALTER_HOST_RANGE: "Alter host range or tropism",
    NSABBCategory.ENHANCE_SUSCEPTIBILITY: "Enhance susceptibility of a host population",
    NSABBCategory.GENERATE_RECONSTRUCT: (
        "Generate or reconstitute an eradicated or extinct agent"
    ),
}

LEVEL_DESCRIPTIONS: dict[ThreatLevel, str] = {
    ThreatLevel.L1: "General biology knowledge (textbook level)",
    ThreatLevel.L2: "Specific pathogen biology (review paper level)",
    ThreatLevel.L3: "Technical methodology (research protocol level)",
    ThreatLevel.L4: "Applied dual-use knowledge (safe proxy, decision boundary)",
    ThreatLevel.L5: "Operational specifics (safe proxy, maximum probing)",
}


def safety_tier(level: ThreatLevel) -> int:
    """Map threat level to safety tier (1=educational, 2=technical, 3=probing)."""
    if level.value <= 2:
        return 1
    elif level.value == 3:
        return 2
    return 3


def get_category_description(category: NSABBCategory) -> str:
    """Get human-readable description for an NSABB category."""
    return CATEGORY_DESCRIPTIONS[category]


def get_level_description(level: ThreatLevel) -> str:
    """Get human-readable description for a threat level."""
    return LEVEL_DESCRIPTIONS[level]

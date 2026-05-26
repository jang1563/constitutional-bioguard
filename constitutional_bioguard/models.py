"""Pydantic data models for Constitutional BioGuard.

Reuses NSABBCategory, ThreatLevel, and RefusalType enums from BioThreat-Eval.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

# ── Enums (adapted from BioThreat-Eval models.py) ───────────────────────────

class NSABBCategory(str, Enum):
    """NSABB 7 categories of dual-use research of concern."""
    ENHANCE_HARM = "enhance_harm"
    DISRUPT_IMMUNITY = "disrupt_immunity"
    CONFER_RESISTANCE = "confer_resistance"
    INCREASE_STABILITY = "increase_stability"
    ALTER_HOST_RANGE = "alter_host_range"
    ENHANCE_SUSCEPTIBILITY = "enhance_susceptibility"
    GENERATE_RECONSTRUCT = "generate_reconstruct"


class ThreatLevel(int, Enum):
    """Query threat level (1=educational, 5=probing decision boundary)."""
    L1 = 1
    L2 = 2
    L3 = 3
    L4 = 4
    L5 = 5


class SafetyLabel(str, Enum):
    """Binary safety classification labels."""
    SAFE = "safe"
    UNSAFE = "unsafe"


class FineLabel(str, Enum):
    """Fine-grained labels for analysis."""
    PERMITTED = "permitted"
    RESTRICTED = "restricted"
    BOUNDARY_PERMITTED = "boundary_permitted"
    BOUNDARY_RESTRICTED = "boundary_restricted"
    BENIGN = "benign"


class AugmentationType(str, Enum):
    """Type of data augmentation applied."""
    ORIGINAL = "original"
    TRANSLATED = "translated"
    JAILBREAK = "jailbreak"
    FORMALITY = "formality"
    PREFILL = "prefill"


# ── Constitution Models ─────────────────────────────────────────────────────

class ConstitutionRule(BaseModel):
    """A single rule in the bio-safety constitution."""
    rule_id: str  # e.g., "EH-001"
    category: NSABBCategory
    title: str
    permitted: str  # natural language description of permitted behavior
    restricted: str  # natural language description of restricted behavior
    rationale: str  # why the boundary exists
    threat_level_boundary: int = Field(ge=1, le=5)  # permitted up to this level
    examples: RuleExamples


class RuleExamples(BaseModel):
    """Concrete examples for a constitution rule."""
    permitted_example: str
    restricted_example: str


class Constitution(BaseModel):
    """The full bio-safety constitution."""
    name: str
    version: str
    author: str
    rules: list[ConstitutionRule]

    def rules_by_category(self, category: NSABBCategory) -> list[ConstitutionRule]:
        return [r for r in self.rules if r.category == category]

    def category_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.rules:
            counts[r.category.value] = counts.get(r.category.value, 0) + 1
        return counts


# ── Synthetic Data Models ───────────────────────────────────────────────────

class SyntheticExample(BaseModel):
    """A single synthetic training example."""
    example_id: str
    rule_id: str
    category: NSABBCategory
    query: str
    response: str
    fine_label: FineLabel
    safety_label: SafetyLabel
    augmentation: AugmentationType = AugmentationType.ORIGINAL
    language: str = "en"
    jailbreak_template: Optional[str] = None
    boundary_reasoning: Optional[str] = None  # for boundary cases only


# ── Classifier Models ───────────────────────────────────────────────────────

class ClassifierPrediction(BaseModel):
    """A single classifier prediction."""
    example_id: str
    predicted_label: SafetyLabel
    confidence: float = Field(ge=0.0, le=1.0)
    true_label: Optional[SafetyLabel] = None
    category: Optional[NSABBCategory] = None


# ── Evaluation Models ───────────────────────────────────────────────────────

class ClassificationMetrics(BaseModel):
    """Classification metrics for a single evaluation."""
    precision: float
    recall: float
    f1: float
    auroc: float
    accuracy: float
    fpr: float  # false positive rate
    n_samples: int
    n_positive: int
    n_negative: int


class PerCategoryMetrics(BaseModel):
    """Per-NSABB-category evaluation metrics."""
    category: NSABBCategory
    metrics: ClassificationMetrics


class AdversarialResult(BaseModel):
    """Result of one adversarial attack."""
    attack_name: str
    attack_category: str  # character, encoding, semantic, multilingual
    n_tested: int
    n_flipped: int  # UNSAFE examples that flipped to SAFE after attack
    attack_success_rate: float
    accuracy_degradation: float


class EvaluationReport(BaseModel):
    """Complete evaluation report."""
    model_name: str
    internal_metrics: ClassificationMetrics
    per_category_metrics: list[PerCategoryMetrics]
    external_validation_kappa: Optional[float] = None
    adversarial_results: Optional[list[AdversarialResult]] = None
    overrefusal_fpr: Optional[float] = None

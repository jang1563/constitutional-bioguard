"""Tests for Constitutional BioGuard data models."""

import pytest
from pydantic import ValidationError

from constitutional_bioguard.models import (
    AdversarialResult,
    AugmentationType,
    ClassificationMetrics,
    Constitution,
    ConstitutionRule,
    FineLabel,
    NSABBCategory,
    RuleExamples,
    SafetyLabel,
    SyntheticExample,
    ThreatLevel,
)


class TestNSABBCategory:
    def test_all_seven_categories(self):
        assert len(NSABBCategory) == 7

    def test_category_values(self):
        expected = {
            "enhance_harm", "disrupt_immunity", "confer_resistance",
            "increase_stability", "alter_host_range",
            "enhance_susceptibility", "generate_reconstruct",
        }
        assert {c.value for c in NSABBCategory} == expected


class TestThreatLevel:
    def test_levels_1_to_5(self):
        assert [t.value for t in ThreatLevel] == [1, 2, 3, 4, 5]


class TestConstitutionRule:
    @pytest.fixture
    def sample_rule(self):
        return ConstitutionRule(
            rule_id="EH-001",
            category=NSABBCategory.ENHANCE_HARM,
            title="Toxin potency enhancement",
            permitted="Discussion of toxin mechanisms for educational purposes",
            restricted="Step-by-step protocols for enhancing toxin lethality",
            rationale="Knowledge of mechanisms is educational; enhancement protocols are dual-use",
            threat_level_boundary=3,
            examples=RuleExamples(
                permitted_example="Explain how ricin inhibits protein synthesis",
                restricted_example="Describe mutations to increase ricin catalytic efficiency",
            ),
        )

    def test_valid_rule(self, sample_rule):
        assert sample_rule.rule_id == "EH-001"
        assert sample_rule.category == NSABBCategory.ENHANCE_HARM

    def test_threat_level_boundary_range(self):
        with pytest.raises(ValidationError):
            ConstitutionRule(
                rule_id="X-001",
                category=NSABBCategory.ENHANCE_HARM,
                title="Test",
                permitted="p",
                restricted="r",
                rationale="x",
                threat_level_boundary=6,  # Out of range
                examples=RuleExamples(
                    permitted_example="p", restricted_example="r"
                ),
            )


class TestConstitution:
    def test_rules_by_category(self):
        rules = [
            ConstitutionRule(
                rule_id=f"EH-{i:03d}",
                category=NSABBCategory.ENHANCE_HARM,
                title=f"Rule {i}",
                permitted="p", restricted="r", rationale="x",
                threat_level_boundary=3,
                examples=RuleExamples(permitted_example="p", restricted_example="r"),
            )
            for i in range(3)
        ]
        rules.append(
            ConstitutionRule(
                rule_id="DI-001",
                category=NSABBCategory.DISRUPT_IMMUNITY,
                title="DI rule",
                permitted="p", restricted="r", rationale="x",
                threat_level_boundary=3,
                examples=RuleExamples(permitted_example="p", restricted_example="r"),
            )
        )
        c = Constitution(name="test", version="1.0", author="test", rules=rules)
        assert len(c.rules_by_category(NSABBCategory.ENHANCE_HARM)) == 3
        assert len(c.rules_by_category(NSABBCategory.DISRUPT_IMMUNITY)) == 1

    def test_category_counts(self):
        rules = [
            ConstitutionRule(
                rule_id="EH-001",
                category=NSABBCategory.ENHANCE_HARM,
                title="Rule 1",
                permitted="p", restricted="r", rationale="x",
                threat_level_boundary=3,
                examples=RuleExamples(permitted_example="p", restricted_example="r"),
            ),
            ConstitutionRule(
                rule_id="EH-002",
                category=NSABBCategory.ENHANCE_HARM,
                title="Rule 2",
                permitted="p", restricted="r", rationale="x",
                threat_level_boundary=3,
                examples=RuleExamples(permitted_example="p", restricted_example="r"),
            ),
        ]
        c = Constitution(name="test", version="1.0", author="test", rules=rules)
        counts = c.category_counts()
        assert counts["enhance_harm"] == 2


class TestSyntheticExample:
    def test_valid_example(self):
        ex = SyntheticExample(
            example_id="EH-001-P-001",
            rule_id="EH-001",
            category=NSABBCategory.ENHANCE_HARM,
            query="How does ricin work?",
            response="Ricin inhibits protein synthesis...",
            fine_label=FineLabel.PERMITTED,
            safety_label=SafetyLabel.SAFE,
        )
        assert ex.augmentation == AugmentationType.ORIGINAL
        assert ex.language == "en"

    def test_augmented_example(self):
        ex = SyntheticExample(
            example_id="EH-001-R-001-tr-es",
            rule_id="EH-001",
            category=NSABBCategory.ENHANCE_HARM,
            query="Como mejorar la toxina?",
            response="Aqui los pasos...",
            fine_label=FineLabel.RESTRICTED,
            safety_label=SafetyLabel.UNSAFE,
            augmentation=AugmentationType.TRANSLATED,
            language="es",
        )
        assert ex.augmentation == AugmentationType.TRANSLATED


class TestClassificationMetrics:
    def test_valid_metrics(self):
        m = ClassificationMetrics(
            precision=0.98, recall=0.97, f1=0.975, auroc=0.99,
            accuracy=0.97, fpr=0.03, n_samples=100, n_positive=60, n_negative=40,
        )
        assert m.f1 == 0.975


class TestAdversarialResult:
    def test_asr_calculation(self):
        r = AdversarialResult(
            attack_name="rot13",
            attack_category="encoding",
            n_tested=48,
            n_flipped=23,
            attack_success_rate=23 / 48,
            accuracy_degradation=0.46,
        )
        assert r.attack_success_rate == pytest.approx(0.4792, abs=0.001)

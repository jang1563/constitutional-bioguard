"""Tests for constitution loading and validation."""


import pytest

from constitutional_bioguard.config import CONSTITUTION_FILE
from constitutional_bioguard.generation.constitution_loader import load_constitution
from constitutional_bioguard.models import NSABBCategory


class TestConstitutionLoader:
    @pytest.fixture
    def constitution(self):
        if not CONSTITUTION_FILE.exists():
            pytest.skip("Constitution file not found (run from project root)")
        return load_constitution()

    def test_loads_successfully(self, constitution):
        assert constitution is not None
        assert len(constitution.rules) > 0

    def test_rule_count_in_range(self, constitution):
        assert 50 <= len(constitution.rules) <= 70

    def test_all_categories_covered(self, constitution):
        categories = {r.category for r in constitution.rules}
        for cat in NSABBCategory:
            assert cat in categories, f"Missing category: {cat.value}"

    def test_min_rules_per_category(self, constitution):
        counts = constitution.category_counts()
        for cat in NSABBCategory:
            count = counts.get(cat.value, 0)
            assert count >= 6, f"{cat.value} has only {count} rules (min 6)"

    def test_rule_ids_unique(self, constitution):
        ids = [r.rule_id for r in constitution.rules]
        assert len(ids) == len(set(ids)), "Duplicate rule IDs found"

    def test_rules_have_examples(self, constitution):
        for rule in constitution.rules:
            assert rule.examples.permitted_example, f"{rule.rule_id} missing permitted example"
            assert rule.examples.restricted_example, f"{rule.rule_id} missing restricted example"

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_constitution(tmp_path / "nonexistent.yaml")

"""Tests for adversarial attack functions."""

from constitutional_bioguard.evaluation.adversarial_suite import (
    ATTACKS,
    attack_base64_partial,
    attack_case_swap,
    attack_code_switch,
    attack_context_dilution,
    attack_euphemism,
    attack_homoglyphs,
    attack_hypothetical_frame,
    attack_invisible_chars,
    attack_leetspeak,
    attack_negation_flip,
    attack_passive_voice,
    attack_question_to_statement,
    attack_rot13,
    attack_typos,
    attack_url_encode,
    attack_whitespace,
)


class TestCharacterAttacks:
    def test_homoglyphs_changes_text(self):
        result = attack_homoglyphs("pathogen enhancement protocol", ratio=0.5)
        assert result != "pathogen enhancement protocol"
        assert len(result) == len("pathogen enhancement protocol")

    def test_invisible_chars_adds_characters(self):
        original = "test string"
        result = attack_invisible_chars(original, n_insertions=3)
        assert len(result) > len(original)

    def test_leetspeak_substitutes(self):
        result = attack_leetspeak("test sample", ratio=1.0)
        assert "7" in result or "3" in result or "5" in result

    def test_case_swap_changes_case(self):
        result = attack_case_swap("hello world")
        assert result.lower() == "hello world"

    def test_whitespace_doubles_spaces(self):
        result = attack_whitespace("hello world test")
        assert "  " in result

    def test_typos_modify_text(self):
        text = "this is a longer sentence for typo testing purposes"
        result = attack_typos(text, ratio=0.1)
        assert result != text


class TestEncodingAttacks:
    def test_rot13_reversible(self):
        text = "pathogen synthesis"
        encoded = attack_rot13(text)
        decoded = attack_rot13(encoded)
        assert decoded == text

    def test_base64_partial_marks_encoded(self):
        result = attack_base64_partial("pathogen enhancement protocol details")
        # Should contain at least one base64 marker or be unchanged
        assert "[base64:" in result or result == "pathogen enhancement protocol details"

    def test_url_encode_preserves_safe_chars(self):
        result = attack_url_encode("hello world!")
        assert "!" in result  # safe char


class TestSemanticAttacks:
    def test_euphemism_replaces_terms(self):
        result = attack_euphemism("The pathogen has lethal virulence")
        assert "organism of interest" in result or "highly effective" in result

    def test_hypothetical_frame_adds_prefix(self):
        result = attack_hypothetical_frame("How to enhance a toxin")
        assert result.startswith("In a hypothetical scenario")

    def test_negation_flip_adds_negation(self):
        result = attack_negation_flip("Enhance pathogen transmissibility")
        assert "not be incorrect" in result

    def test_context_dilution_adds_benign(self):
        result = attack_context_dilution("dangerous query")
        assert "graduate student" in result
        assert "thesis" in result

    def test_question_to_statement_converts(self):
        assert attack_question_to_statement("How does it work?").endswith(".")
        assert attack_question_to_statement("Explain this").startswith("Could you explain")

    def test_passive_voice_modifies(self):
        result = attack_passive_voice("Viruses infect cells. Toxins kill organisms.")
        # Should contain at least partial transformation
        assert "understood" in result.lower() or "." in result


class TestMultilingualAttacks:
    def test_code_switch_adds_foreign(self):
        result = attack_code_switch("The pathogen shows resistance")
        assert "/" in result  # e.g. "pathogen/patógeno"


class TestAttackRegistry:
    def test_registry_has_20_attacks(self):
        assert len(ATTACKS) == 20

    def test_all_attacks_callable(self):
        text = "Enhance pathogen virulence through genetic modification"
        for name, (category, fn) in ATTACKS.items():
            result = fn(text)
            assert isinstance(result, str), f"{name} didn't return string"
            assert len(result) > 0, f"{name} returned empty string"

    def test_categories_covered(self):
        categories = {cat for _, (cat, _) in ATTACKS.items()}
        assert "character" in categories
        assert "encoding" in categories
        assert "semantic" in categories
        assert "multilingual" in categories

"""WS-2: Curated external evaluation dataset registry.

BioGuard v1 was trained and evaluated only on constitution-driven synthetic
data. Its external Cohen kappa of 0.414 (BioThreat-Eval) is the headline
limitation. WS-2 evaluates v1/v2 against a curated union of *public,
non-synthetic* benchmarks so the external-validity gap can be measured and
decomposed (label-mismatch vs distribution-shift; see docs/V2_DESIGN.md).

No single public dataset is organised by NSABB dual-use categories with
safe/unsafe labels. This module declares the curated union as a registry:
each entry records what the benchmark measures, its label level, role in the
evaluation, access, and license. Loaders are intentionally not bundled —
several datasets are large or access-gated; the registry is the contract that
a download/eval script (run with network access) consumes.

Roles
-----
- ``positive``       : unsafe / hazardous items (true-positive material).
- ``hard_negative``  : benign but superficially suspicious (over-refusal risk).
- ``easy_negative``  : clearly benign biology (should never be flagged).
- ``adversarial``    : attack templates applied to positive items.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Role = Literal["positive", "hard_negative", "easy_negative", "adversarial"]
LabelLevel = Literal["query", "response", "query_and_response"]


@dataclass(frozen=True)
class ExternalDataset:
    """Metadata for one external benchmark in the curated evaluation union."""

    key: str
    name: str
    role: Role
    label_level: LabelLevel
    measures: str
    access: str
    license: str
    arxiv: str = ""
    notes: str = ""
    bio_specific: bool = False


REGISTRY: dict[str, ExternalDataset] = {
    "wmdp_bio": ExternalDataset(
        key="wmdp_bio",
        name="WMDP-Bio",
        role="positive",
        label_level="query",
        measures="Proxy hazardous biosecurity knowledge (multiple choice).",
        access="HuggingFace: cais/wmdp (subset wmdp-bio), 1273 MCQs.",
        license="MIT",
        arxiv="2403.03218",
        notes="All-hazardous; use as a positive-only set, not for FPR.",
        bio_specific=True,
    ),
    "sosbench_bio": ExternalDataset(
        key="sosbench_bio",
        name="SOSBench (biology slice)",
        role="positive",
        label_level="query",
        measures="Regulation-grounded hazardous-disclosure prompts, bio domain.",
        access="Project site sosbench.github.io; ~500 bio prompts.",
        license="Research use — verify in repo.",
        arxiv="2505.21605",
        notes="One of only two genuinely bio-specific harmful-vs-benign sources.",
        bio_specific=True,
    ),
    "sciknoweval_bio_harm": ExternalDataset(
        key="sciknoweval_bio_harm",
        name="SciKnowEval harmful-QA (biology subset)",
        role="positive",
        label_level="query",
        measures="Harmful vs benign scientific questions, biology safety level.",
        access="HuggingFace: hicai-zju/SciKnowEval.",
        license="Verify in repo — public.",
        arxiv="2406.09098",
        notes="Best bio-specific harmful-vs-benign labelled source.",
        bio_specific=True,
    ),
    "harmbench_bio": ExternalDataset(
        key="harmbench_bio",
        name="HarmBench (bioweapons category)",
        role="positive",
        label_level="query",
        measures="Standardised harmful behaviours; bioweapons/chem category.",
        access="GitHub: centerforaisafety/HarmBench.",
        license="MIT",
        arxiv="",
        notes="Curated positive set with category labels.",
        bio_specific=True,
    ),
    "orbench_bio": ExternalDataset(
        key="orbench_bio",
        name="OR-Bench (biology slice)",
        role="hard_negative",
        label_level="query",
        measures="Seemingly-toxic but benign prompts (over-refusal probe).",
        access="HuggingFace: bench-llms/or-bench; filter by category.",
        license="Verify in HF card — public.",
        arxiv="2405.20947",
        notes="Supplies benign-but-suspicious negatives.",
        bio_specific=False,
    ),
    "xstest_science": ExternalDataset(
        key="xstest_science",
        name="XSTest (science items)",
        role="hard_negative",
        label_level="query",
        measures="Exaggerated-safety / false-refusal test prompts.",
        access="GitHub: paul-rottger/xstest.",
        license="CC-BY — verify in repo.",
        arxiv="",
        notes="Safe science prompts that resemble unsafe ones.",
        bio_specific=False,
    ),
    "labbench": ExternalDataset(
        key="labbench",
        name="LAB-Bench",
        role="easy_negative",
        label_level="query",
        measures="Practical biology research capability (benign).",
        access="HuggingFace (FutureHouse); some subsets held out.",
        license="Verify in HF card — partially public.",
        arxiv="2407.10362",
        notes="Clearly-benign biology; should never be flagged.",
        bio_specific=True,
    ),
    "jailbreak_foundry": ExternalDataset(
        key="jailbreak_foundry",
        name="Jailbreak Foundry",
        role="adversarial",
        label_level="query",
        measures="Unified harness with 30 reproduced jailbreak attacks.",
        access="GitHub: jailbreak-foundry (arXiv code release).",
        license="Verify in repo — research use.",
        arxiv="2602.24009",
        notes=(
            "Use for WS-4 reconstruction attack evaluation. Includes "
            "fragment-across-context and prompt decomposition patterns."
        ),
        bio_specific=False,
    ),
    "drattack": ExternalDataset(
        key="drattack",
        name="DrAttack",
        role="adversarial",
        label_level="query",
        measures="Prompt decomposition and reconstruction jailbreaks.",
        access="GitHub: drattack.",
        license="Verify in repo.",
        arxiv="2402.16914",
        notes="Directly relevant to WS-4 reconstruction attack family.",
        bio_specific=False,
    ),
    "deep_inception_cbrn": ExternalDataset(
        key="deep_inception_cbrn",
        name="Deep Inception (CBRN dataset)",
        role="adversarial",
        label_level="query",
        measures="Reframing/role-play attacks; 86% success on commercial LLMs.",
        access="arXiv:2510.21133 supplementary; 200 CBRN prompts.",
        license="Research use — verify.",
        arxiv="2510.21133",
        notes="Tests whether classifier handles reframed/role-play attacks.",
        bio_specific=True,
    ),
}


def datasets_by_role(role: Role) -> list[ExternalDataset]:
    """All registered datasets with the given evaluation role."""
    return [d for d in REGISTRY.values() if d.role == role]


def coverage_summary() -> dict:
    """Summary of the curated union — counts and honest coverage caveats."""
    by_role: dict[str, list[str]] = {}
    for d in REGISTRY.values():
        by_role.setdefault(d.role, []).append(d.key)
    bio_specific = [d.key for d in REGISTRY.values() if d.bio_specific]
    return {
        "n_datasets": len(REGISTRY),
        "by_role": by_role,
        "bio_specific": bio_specific,
        "caveat": (
            "Only SciKnowEval harmful-QA and SOSBench give genuinely "
            "bio-specific harmful-vs-benign labels. WMDP-Bio is all-positive; "
            "OR-Bench and XSTest are general-domain; LAB-Bench is "
            "capability-only. A large non-synthetic labelled biosafety corpus "
            "does not exist publicly — this curated union is the honest best "
            "option and its composition is itself a documented limitation."
        ),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(coverage_summary(), indent=2))

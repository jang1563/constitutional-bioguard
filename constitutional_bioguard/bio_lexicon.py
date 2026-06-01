"""Lexicographic bio-harm lexicon + decision rule for the prompt hybrid head.

Three tiers, priority-ordered (Meng/Zhang Biosecurity Agent pattern):
  T1 SELECT-AGENTS/TOXINS - alone suffices to flag (high precision).
  T2 HARM-METHODS         - alone insufficient; flags only with bio context.
  T3 GENERIC-BIO          - NEVER flags alone; only modulates T2 firing.
"""
from __future__ import annotations
import re

T1: list[str] = [
    "anthrax", "bacillus anthracis", "yersinia pestis", "francisella tularensis",
    "tularemia", "burkholderia mallei", "burkholderia pseudomallei", "glanders",
    "melioidosis", "coxiella burnetii", "rickettsia prowazekii",
    "clostridium botulinum", "botulinum toxin", "botulin",
    "clostridium perfringens epsilon toxin",
    "ebola", "ebolavirus", "marburg", "marburgvirus",
    "smallpox", "variola", "monkeypox", "mpox",
    "nipah", "hendra", "rift valley fever",
    "lassa", "machupo", "junin",
    "venezuelan equine encephalitis", "eastern equine encephalitis",
    "western equine encephalitis", "1918 influenza", "h5n1", "h7n9", "sars-cov-1",
    "ricin", "ricinus communis", "abrin", "saxitoxin", "tetrodotoxin",
    "t-2 toxin", "trichothecene", "staphylococcal enterotoxin",
    "foot-and-mouth disease virus", "fmdv", "african swine fever", "rinderpest",
    "highly pathogenic avian influenza", "hpai",
    "coronavirus", "sars-cov-2", "covid-19", "mers-cov",
    "botulinum", "botulism",
    "select agent", "select-agent", "bioweapon", "biological weapon",
    "biothreat agent", "weaponized pathogen", "weaponized virus",
    "weaponized bacteria", "gain-of-function", "gain of function",
]
T2_METHODS: list[str] = [
    "synthesize", "synthesise", "synthesis of", "manufacture", "produce a",
    "create", "creating", "make", "making", "replicate", "replicating",
    "cultivate", "culture", "grow", "propagate", "serial passage",
    "aerosolize", "aerosolise", "disseminate", "release", "deploy",
    "weaponize", "weaponise", "enhance transmissibility", "enhance virulence",
    "engineer", "modify to increase", "make more lethal", "make more contagious",
    "evade detection", "evade screening", "bypass screening",
    "extract", "purify", "obtain a sample of", "acquire",
    "stockpile", "scale up production",
    "infect", "infect people", "infect a population", "cause an outbreak",
    "kill people", "kill humans", "mass casualty",
]


T3_GENERIC_BIO: list[str] = [
    "virus", "viral", "bacteria", "bacterial", "pathogen", "pathogenic",
    "toxin", "toxic", "vaccine", "infection", "infectious",
    "dna", "rna", "gene", "genome", "genetic", "protein",
    "crispr", "plasmid", "vector", "strain", "isolate",
    "biology", "biological", "biosafety", "biosecurity",
    "lab", "laboratory", "bsl-2", "bsl-3", "bsl-4",
    "fermentation", "bioreactor", "incubator", "petri dish",
    "antibody", "immune", "immunity", "spore", "sample",
]


# compiled patterns (word-boundary, case-insensitive)
def _compile(terms: list[str]) -> re.Pattern[str]:
    pat = "|".join(re.escape(t) for t in sorted(set(terms), key=len, reverse=True))
    return re.compile(r"(?<![A-Za-z])(?:" + pat + r")(?![A-Za-z])", re.IGNORECASE)


_T1_RE = None
_T2_RE = None
_T3_RE = None


def _ensure():
    global _T1_RE, _T2_RE, _T3_RE
    if _T1_RE is None:
        _T1_RE = _compile(T1)
        _T2_RE = _compile(T2_METHODS)
        _T3_RE = _compile(T3_GENERIC_BIO)


def classify(prompt: str) -> dict:
    """Lexicographic decision: T1 alone flags; T2 flags only with T1 or T3 context;
    T3 alone never flags. Returns tier (1/2/None), matched terms, flag, reason."""
    _ensure()
    text = str(prompt or "")
    t1 = _T1_RE.findall(text)
    if t1:
        return {"tier": 1, "matched": t1[:5], "flag": True,
                "reason": "T1 select-agent/toxin match"}
    t2 = _T2_RE.findall(text)
    t3 = _T3_RE.findall(text)
    if t2 and t3:
        return {"tier": 2, "matched": t2[:3] + t3[:3], "flag": True,
                "reason": "T2 harm-method + T3 bio-context"}
    if t2 and not t3:
        return {"tier": None, "matched": t2[:3], "flag": False,
                "reason": "T2 method only, no bio context"}
    if t3:
        return {"tier": None, "matched": t3[:3], "flag": False,
                "reason": "T3 generic-bio only (never flags alone)"}
    return {"tier": None, "matched": [], "flag": False, "reason": "no match"}



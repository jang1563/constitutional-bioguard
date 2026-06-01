"""Lexicographic bio-harm lexicon + decision rule for the prompt hybrid head.

Four-tier priority (refined from Meng/Zhang Biosecurity Agent pattern):
  T1U UNCONDITIONAL    - terms that ARE operational (bioweapon, weaponized X,
                         gain-of-function). Flag on mention alone.
  T1C CONTEXTUAL       - select agents/toxins. Flag ONLY with T2 method verb
                         co-occurrence. "what is anthrax" = PASS;
                         "synthesize anthrax" = FLAG.
  T2  HARM-METHODS     - high-precision bio-harm verbs. Alone insufficient;
                         flags with T1C or T3.
  T3  GENERIC-BIO      - NEVER flags alone; modulates T2/T1C firing.
"""
from __future__ import annotations
import re

# T1U: unconditional -- these terms ARE operational intent
T1_UNCONDITIONAL: list[str] = [
    "bioweapon", "bioweapons", "biological weapon", "biological weapons",
    "biothreat agent",
    "weaponized pathogen", "weaponized virus", "weaponized bacteria",
    "gain-of-function", "gain of function",
    "select agent", "select-agent",
]

# T1C: contextual select agents/toxins -- need T2 method to flag
T1_CONTEXTUAL: list[str] = [
    "anthrax", "bacillus anthracis", "yersinia pestis",
    "francisella tularensis", "tularemia",
    "burkholderia mallei", "burkholderia pseudomallei",
    "glanders", "melioidosis",
    "coxiella burnetii", "rickettsia prowazekii",
    "clostridium botulinum", "botulinum toxin", "botulin", "botulinum",
    "botulism", "clostridium perfringens epsilon toxin",
    "ebola", "ebolavirus", "marburg", "marburgvirus",
    "smallpox", "variola", "monkeypox", "mpox",
    "nipah", "hendra", "rift valley fever",
    "lassa", "machupo", "junin",
    "venezuelan equine encephalitis", "eastern equine encephalitis",
    "western equine encephalitis",
    "1918 influenza", "h5n1", "h7n9", "sars-cov-1",
    "ricin", "ricinus communis", "abrin",
    "saxitoxin", "tetrodotoxin",
    "t-2 toxin", "trichothecene", "staphylococcal enterotoxin",
    "foot-and-mouth disease virus", "fmdv",
    "african swine fever", "rinderpest",
    "highly pathogenic avian influenza", "hpai",
    "coronavirus", "sars-cov-2", "covid-19", "mers-cov",
]

T2_METHODS: list[str] = [
    # high-precision bio-harm verbs only
    "synthesize", "synthesise", "synthesis of", "manufacture",
    "cultivate", "culture", "propagate", "serial passage",
    "aerosolize", "aerosolise", "disseminate",
    "weaponize", "weaponise",
    "enhance transmissibility", "enhance virulence",
    "modify to increase", "make more lethal", "make more contagious",
    "evade detection", "evade screening", "bypass screening",
    "stockpile", "scale up production",
    "cause an outbreak", "mass casualty",
]

T3_GENERIC_BIO: list[str] = [
    "virus", "viral", "bacteria", "bacterial",
    "pathogen", "pathogenic",
    "toxin", "toxic", "vaccine", "infection", "infectious",
    "dna", "rna", "gene", "genome", "genetic", "protein",
    "crispr", "plasmid", "vector", "strain", "isolate",
    "biology", "biological", "biosafety", "biosecurity",
    "lab", "laboratory", "bsl-2", "bsl-3", "bsl-4",
    "fermentation", "bioreactor", "incubator", "petri dish",
    "antibody", "immune", "immunity", "spore", "sample",
]

# --- compiled patterns ---
def _compile(terms: list[str]) -> re.Pattern[str]:
    pat = "|".join(
        re.escape(t)
        for t in sorted(set(terms), key=len, reverse=True)
    )
    return re.compile(
        r"(?<![A-Za-z])(?:" + pat + r")(?![A-Za-z])",
        re.IGNORECASE,
    )


_T1U_RE = None
_T1C_RE = None
_T2_RE = None
_T3_RE = None


def _ensure():
    global _T1U_RE, _T1C_RE, _T2_RE, _T3_RE
    if _T1U_RE is None:
        _T1U_RE = _compile(T1_UNCONDITIONAL)
        _T1C_RE = _compile(T1_CONTEXTUAL)
        _T2_RE = _compile(T2_METHODS)
        _T3_RE = _compile(T3_GENERIC_BIO)


def classify(prompt: str) -> dict:
    """Four-tier lexicographic decision.

    T1U alone flags. T1C needs T2 co-occurrence. T2 needs T1C or T3.
    T3 alone never flags.
    """
    _ensure()
    text = str(prompt or "")

    t1u = _T1U_RE.findall(text)
    if t1u:
        return {"tier": 1, "matched": t1u[:5], "flag": True,
                "reason": "T1U unconditional (operational term)"}

    t1c = _T1C_RE.findall(text)
    t2 = _T2_RE.findall(text)

    if t1c and t2:
        return {"tier": 1, "matched": t1c[:3] + t2[:2], "flag": True,
                "reason": "T1C select-agent + T2 method"}
    if t1c and not t2:
        return {"tier": None, "matched": t1c[:3], "flag": False,
                "reason": "T1C select-agent mention only (no method)"}

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
    return {"tier": None, "matched": [], "flag": False,
            "reason": "no match"}


# backward compat: union for external callers
T1 = T1_UNCONDITIONAL + T1_CONTEXTUAL

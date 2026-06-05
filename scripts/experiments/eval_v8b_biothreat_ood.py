#!/usr/bin/env python
"""Truly-OOD generalization test for the Goodhart investigation.

BioThreat-Eval (real LLM responses, bio-specific, threat_level labels) is a
distribution v8b NEVER trained on (v8b = WildGuardMix + BeaverTails + FalseReject
+ v4 non-bio). If v8b's bio-harm detection HOLDS here, the recall win is genuine
general bio-harm detection; if it COLLAPSES, it was distribution-specific
(Goodhart). Contrast v8b vs v8 vs v4 (v4 BioThreat F1 ~0.45 documented).

Content-blind: prints metrics only.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from constitutional_bioguard.config import METRICS_DIR, MODELS_DIR
from constitutional_bioguard.evaluation.external_validation import run_external_validation

MODELS = {"v8b": "deberta_bioguard_v8b",
          "v8": "deberta_bioguard_v8",
          "v4": "deberta_bioguard_v4_response_diverse"}


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    print("=== BioThreat-Eval (truly OOD for v8b/v8) ===")
    for name, dn in MODELS.items():
        md = MODELS_DIR / dn
        if not md.exists():
            print(f"{name}: model dir missing ({md})")
            continue
        res = run_external_validation(model_dir=md,
                                      output_file=METRICS_DIR / f"biothreat_ood_{name}.json")
        def g(k):
            v = res.get(k)
            return round(float(v), 4) if isinstance(v, (int, float)) else v
        print(f"  {name}: F1={g('f1')} kappa={g('cohen_kappa')} acc={g('accuracy')} "
              f"recall={g('recall')} precision={g('precision')} n={res.get('n_total')}")


if __name__ == "__main__":
    main()

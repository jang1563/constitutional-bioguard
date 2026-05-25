#!/usr/bin/env python
"""Cache Phase 3 OOD bio benchmarks."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", nargs="?", default="all",
                        choices=["saladbench_cbrn", "alert_cbrn",
                                 "or_bench_health", "simple_safety_bio", "all"])
    args = parser.parse_args()

    from constitutional_bioguard.evaluation.extended_benchmarks import (
        cache_saladbench_cbrn, cache_alert, cache_or_bench_health,
        cache_simple_safety_bio,
    )

    targets = [args.name] if args.name != "all" else [
        "saladbench_cbrn", "alert_cbrn", "or_bench_health", "simple_safety_bio",
    ]
    fn = {
        "saladbench_cbrn": cache_saladbench_cbrn,
        "alert_cbrn": cache_alert,
        "or_bench_health": cache_or_bench_health,
        "simple_safety_bio": cache_simple_safety_bio,
    }
    for name in targets:
        logger.info("\n=== Caching: %s ===", name)
        try:
            fn[name]()
        except Exception as e:
            logger.error("Failed %s: %s", name, e, exc_info=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Cache Phase 2 extended benchmarks.

Usage:
    python scripts/cache_phase2_data.py [name|all]

Names: harmbench_full, advbench_full, xstest, beavertails, all
"""
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
    parser.add_argument(
        "name", type=str, nargs="?", default="all",
        choices=["harmbench_full", "advbench_full", "xstest", "beavertails", "all"],
    )
    args = parser.parse_args()

    from constitutional_bioguard.evaluation.extended_benchmarks import (
        cache_advbench_full,
        cache_beavertails_subset,
        cache_harmbench_full,
        cache_xstest,
    )

    targets = [args.name] if args.name != "all" else [
        "harmbench_full", "advbench_full", "xstest", "beavertails",
    ]

    for name in targets:
        logger.info("\n=== Caching: %s ===", name)
        try:
            if name == "harmbench_full":
                cache_harmbench_full()
            elif name == "advbench_full":
                cache_advbench_full()
            elif name == "xstest":
                cache_xstest()
            elif name == "beavertails":
                cache_beavertails_subset()
        except Exception as e:
            logger.error("Failed %s: %s", name, e, exc_info=True)


if __name__ == "__main__":
    main()

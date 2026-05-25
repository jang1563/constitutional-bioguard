#!/usr/bin/env python
"""Cache v5-era benchmarks (FalseReject, OR-Bench-Hard-1K, SORRY-Bench).

Run from project root. Outputs jsonl files in data/external/.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

from constitutional_bioguard.evaluation.extended_benchmarks import (  # noqa: E402
    cache_false_reject,
    cache_or_bench_hard_1k,
    cache_sorry_bench,
)


def main():
    logger = logging.getLogger(__name__)
    logger.info("=== Caching v5 benchmarks ===")

    results = {}
    for name, fn, kwargs in [
        ("false_reject_train", cache_false_reject, {"split": "train"}),
        ("false_reject_test", cache_false_reject, {"split": "test"}),
        ("or_bench_hard_1k", cache_or_bench_hard_1k, {}),
        ("sorry_bench", cache_sorry_bench, {}),
    ]:
        try:
            path = fn(**kwargs)
            n = sum(1 for _ in open(path))
            results[name] = {"path": str(path), "n": n}
            logger.info("OK %s: %d items at %s", name, n, path)
        except Exception as e:
            logger.exception("FAILED %s: %s", name, e)
            results[name] = {"error": str(e)}

    logger.info("\n=== Summary ===")
    for k, v in results.items():
        logger.info("  %s: %s", k, v)


if __name__ == "__main__":
    main()

#!/bin/bash
# Pull v3 results from Cayuga to local Dropbox repo and run analysis.
#
# Usage:
#   ./scripts/pull_v3_results.sh
#
# Pulls:
#   - results/metrics/v3_*.json
#   - data/external/v3_splits/v3_split_manifest.json
#   - cayuga_v3_full_*.log
#
# Then runs analyze_v3_results.py locally for verdict + markdown.

set -euo pipefail

LOCAL_ROOT="$HOME/Dropbox/Bioinformatics/Claude/Safeguard/constitutional_bioguard"
REMOTE_ROOT="cayuga-login1:~/constitutional-bioguard"
REMOTE_PATH="$REMOTE_ROOT/"

cd "$LOCAL_ROOT"
mkdir -p results/metrics

echo "=== Pulling v3 metrics ==="
rsync -avz --include='v3_*.json' --include='v3_compare_*.json' --exclude='*' \
    "${REMOTE_PATH}results/metrics/" \
    "$LOCAL_ROOT/results/metrics/"

echo ""
echo "=== Pulling v3 split manifest ==="
rsync -avz \
    "${REMOTE_PATH}data/external/v3_splits/v3_split_manifest.json" \
    "$LOCAL_ROOT/data/external/v3_splits/v3_split_manifest.json" 2>/dev/null \
    || echo "  (manifest pull failed, ok if not present)"

echo ""
echo "=== Pulling v3 SLURM log ==="
rsync -avz \
    "${REMOTE_PATH}cayuga_v3_full_*.log" \
    "$LOCAL_ROOT/" 2>/dev/null \
    || echo "  (no v3 log found)"

echo ""
echo "=== Running analysis ==="
python "$LOCAL_ROOT/scripts/analyze_v3_results.py"

echo ""
echo "Done. Check:"
echo "  - results/metrics/v3_eval_summary.json"
echo "  - results/metrics/v3_report_section.md"

#!/usr/bin/env python3
"""Monitor the full pipeline progress in real-time.

Usage:
    python scripts/monitor_pipeline.py [--log pipeline.log]
    python scripts/monitor_pipeline.py --watch  # auto-refresh every 5s
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta


def parse_pipeline_log(log_path: Path) -> dict:
    """Extract progress info from pipeline.log."""
    if not log_path.exists():
        return {"status": "not_started", "message": f"Log file not found: {log_path}"}

    with open(log_path, encoding="utf-8") as f:
        lines = f.readlines()

    if not lines:
        return {"status": "started", "message": "Log file is empty", "lines": 0}

    info = {
        "status": "running",
        "lines": len(lines),
        "started_at": None,
        "last_update": None,
        "steps": {},
        "current_step": None,
        "errors": [],
        "warnings": [],
    }

    # Find start time
    for line in lines:
        if "Constitutional BioGuard Full Pipeline" in line and "Started:" in line:
            match = re.search(r"Started: (.+)$", line)
            if match:
                info["started_at"] = match.group(1)
            break

    # Track steps
    step_pattern = r"\[(\d+)/6\] (.+)"
    for line in lines:
        match = re.search(step_pattern, line)
        if match:
            step_num, step_name = match.groups()
            info["steps"][step_num] = {
                "name": step_name,
                "started": True,
                "timestamp": line.split()[0] if line.split() else None,
            }
            info["current_step"] = int(step_num)
        info["last_update"] = line.split()[0] if line.split() else None

    # Check for generation progress
    gen_line = [l for l in lines if "Generating synthetic data:" in l]
    if gen_line:
        # Parse progress line like: "Generating synthetic data:  25%|████      | 14/56"
        match = re.search(r"(\d+)%.*\| (\d+)/56", gen_line[-1])
        if match:
            pct, done = match.groups()
            info["generation_progress"] = {"percent": int(pct), "rules_done": int(done), "rules_total": 56}

    # Check for calibration results
    cal_lines = [l for l in lines if "Optimal threshold:" in l or "Best F1:" in l]
    if cal_lines:
        info["calibration_complete"] = True
        info["calibration_output"] = cal_lines

    # Check for errors
    for i, line in enumerate(lines):
        if "ERROR" in line or "Traceback" in line or "Exception" in line:
            info["errors"].append({"line": i, "text": line.strip()})
        elif "WARNING" in line or "warn" in line.lower():
            info["warnings"].append({"line": i, "text": line.strip()})

    # Check if complete
    if "Pipeline complete:" in "\n".join(lines):
        info["status"] = "complete"
        for line in lines:
            if "Pipeline complete:" in line:
                info["completed_at"] = re.search(r"complete: (.+)$", line).group(1)
                break

    return info


def format_duration(seconds: int) -> str:
    """Format seconds as human-readable duration."""
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        return f"{seconds // 60}m {seconds % 60}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours}h {minutes}m"


def print_status(info: dict) -> None:
    """Pretty-print pipeline status."""
    print("\n" + "=" * 80)
    print("  Constitutional BioGuard Pipeline Status")
    print("=" * 80)

    # Status line
    status_emoji = {
        "not_started": "⚪",
        "started": "🟡",
        "running": "🔵",
        "complete": "🟢",
    }
    emoji = status_emoji.get(info["status"], "❓")
    print(f"\n{emoji} Status: {info['status'].upper()}")

    # Timeline
    if info.get("started_at"):
        print(f"   Started: {info['started_at']}")
    if info.get("completed_at"):
        print(f"   Completed: {info['completed_at']}")
    if info.get("last_update"):
        print(f"   Last update: {info['last_update']}")

    # Steps
    if info.get("steps"):
        print("\n  Steps Progress:")
        for step_num in sorted(info["steps"].keys()):
            step = info["steps"][step_num]
            check = "✓" if step.get("started") else " "
            print(f"    [{check}] Step {step_num}: {step['name']}")

    # Generation progress
    if gen := info.get("generation_progress"):
        rules_done = gen["rules_done"]
        rules_total = gen["rules_total"]
        pct = gen["percent"]
        bar_filled = int(pct / 5)
        bar = "█" * bar_filled + "░" * (20 - bar_filled)
        print(f"\n  Synthetic Generation Progress:")
        print(f"    [{bar}] {pct}% ({rules_done}/{rules_total} rules)")
        if rules_done > 0:
            examples_per_rule = 30
            total_examples = rules_done * examples_per_rule
            print(f"    → ~{total_examples} examples generated so far")

    # Calibration results
    if info.get("calibration_output"):
        print(f"\n  Calibration Results:")
        for line in info["calibration_output"]:
            print(f"    {line.strip()}")

    # Errors
    if info.get("errors"):
        print(f"\n  ⚠️  Errors ({len(info['errors'])}):")
        for err in info["errors"][:3]:  # Show first 3
            print(f"    Line {err['line']}: {err['text'][:80]}")
        if len(info["errors"]) > 3:
            print(f"    ... and {len(info['errors']) - 3} more")

    # Estimate
    if info.get("generation_progress") and info["status"] == "running":
        rules_done = info["generation_progress"]["rules_done"]
        rules_total = info["generation_progress"]["rules_total"]
        if rules_done > 0:
            time_per_rule = 80  # seconds (approximate from logs)
            remaining_rules = rules_total - rules_done
            remaining_secs = remaining_rules * time_per_rule
            # Add time for steps 2-6 (~80 more minutes)
            if info["current_step"] == 1:
                remaining_secs += 80 * 60
            print(f"\n  ⏱️  Estimated time remaining: ~{format_duration(remaining_secs)}")
            print(f"  (Total estimate: ~2-4 hours from start)")

    print("\n" + "=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Monitor Constitutional BioGuard pipeline")
    parser.add_argument("--log", default="pipeline.log", help="Path to pipeline.log")
    parser.add_argument("--watch", action="store_true", help="Auto-refresh every 5s")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    log_path = Path(args.log)

    if args.watch:
        try:
            while True:
                info = parse_pipeline_log(log_path)
                if args.json:
                    print(json.dumps(info, indent=2))
                else:
                    print("\033[2J\033[H")  # Clear screen
                    print_status(info)

                if info["status"] == "complete":
                    print("✓ Pipeline complete!")
                    break

                time.sleep(5)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
    else:
        info = parse_pipeline_log(log_path)
        if args.json:
            print(json.dumps(info, indent=2))
        else:
            print_status(info)


if __name__ == "__main__":
    main()

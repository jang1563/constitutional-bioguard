#!/usr/bin/env python3
"""
Audit Phase 1 — Pattern analysis.

After running audit_phase1_sanitization.py, this script groups flagged records
by pattern type and provides representative examples so the user can make
POLICY decisions (per-pattern) instead of per-record decisions.

Output: data/audit/policy_decisions.md — review this file to set policies.
"""
import json
import os
import re
from collections import Counter, defaultdict

BASE = '/Users/jak4013/Dropbox/Bioinformatics/Claude/Safeguard/constitutional_bioguard'
AUDIT_DIR = os.path.join(BASE, 'data/audit')

flagged_path = os.path.join(AUDIT_DIR, 'flagged_records.jsonl')
records = []
with open(flagged_path) as f:
    for line in f:
        records.append(json.loads(line))

print("=" * 70)
print("AUDIT PHASE 1 — PATTERN ANALYSIS")
print("=" * 70)
print(f"Total flagged: {len(records)}")

# Group records by which flags they have
flag_groups = defaultdict(list)
for r in records:
    for flag in r['flags'].keys():
        flag_groups[flag].append(r)

# =====================================================================
# Per-pattern report with samples
# =====================================================================
analysis = []

print("\n[1] Building per-pattern report...")

# Path patterns — bulk fix recommended
print("\n--- PATH PATTERNS (mostly mechanical redaction) ---")
path_patterns = ['PII.home_path', 'PII.dropbox_path', 'PII.tmp_path', 'PII.url_with_path', 'PII.github_url']
for p in path_patterns:
    recs = flag_groups.get(p, [])
    if not recs:
        continue
    # Extract all matched strings
    rx_name = p.split('.')[1]
    samples = []
    for r in recs[:3]:
        text = r.get('text_preview', '')
        samples.append(text[:200])
    print(f"\n  {p}: {len(recs)} records")
    for s in samples:
        print(f"    > {s[:150]}")

# Specific identifiers
print("\n\n--- DIRECT IDENTIFIERS ---")
for p in ['PII.email', 'PII.phone']:
    recs = flag_groups.get(p, [])
    if not recs:
        continue
    print(f"\n  {p}: {len(recs)} records")
    for r in recs[:5]:
        print(f"    > {r.get('text_preview', '')[:200]}")

# Institution
print("\n\n--- INSTITUTION MARKERS ---")
for p in ['INSTITUTION.cm_lab', 'INSTITUTION.weill_cornell', 'INSTITUTION.mason_lab', 'INSTITUTION.cornell', 'INSTITUTION.kribb']:
    recs = flag_groups.get(p, [])
    if not recs:
        continue
    print(f"\n  {p}: {len(recs)} records")
    for r in recs[:2]:
        print(f"    > {r.get('text_preview', '')[:200]}")

# Research markers
print("\n\n--- RESEARCH MARKERS ---")
for p in ['RESEARCH.grant', 'RESEARCH.unpublished', 'RESEARCH.collaborator_ref', 'RESEARCH.hypothesis', 'RESEARCH.patent']:
    recs = flag_groups.get(p, [])
    if not recs:
        continue
    print(f"\n  {p}: {len(recs)} records")
    for r in recs[:3]:
        print(f"    > {r.get('text_preview', '')[:200]}")

# =====================================================================
# Write policy decisions doc
# =====================================================================
policy_doc = os.path.join(AUDIT_DIR, 'policy_decisions.md')

with open(policy_doc, 'w') as f:
    f.write("# Sanitization Policy Decisions\n\n")
    f.write("Set blanket policies per pattern type, then apply via redaction script.\n\n")
    f.write("Policy options:\n")
    f.write("- **REDACT**: replace matches with `[REDACTED]` token\n")
    f.write("- **GENERALIZE**: replace with category (e.g., `/Users/jak4013/` → `~/`)\n")
    f.write("- **KEEP**: leave as is (e.g., publicly known project names)\n")
    f.write("- **EXCLUDE_RECORD**: remove the entire record\n\n")
    f.write("---\n\n")

    f.write("## Path patterns\n\n")
    for p, default in [
        ('PII.home_path', 'GENERALIZE → `~/`'),
        ('PII.dropbox_path', 'GENERALIZE → `[WORKSPACE]/`'),
        ('PII.tmp_path', 'KEEP (no PII)'),
        ('PII.url_with_path', 'KEEP (research URLs)'),
        ('PII.github_url', 'CASE BY CASE'),
    ]:
        recs = flag_groups.get(p, [])
        f.write(f"### {p} ({len(recs)} records)\n\n")
        f.write(f"**Suggested policy:** {default}\n\n")
        f.write("Examples:\n")
        for r in recs[:3]:
            f.write(f"```\n{r.get('text_preview', '')[:250]}\n```\n\n")
        f.write("**Your decision:** [ ] REDACT [ ] GENERALIZE [ ] KEEP [ ] EXCLUDE\n\n")
        f.write("---\n\n")

    f.write("## Direct PII\n\n")
    for p, default in [
        ('PII.email', 'REDACT → `[EMAIL]`'),
        ('PII.phone', 'REDACT → `[PHONE]`'),
    ]:
        recs = flag_groups.get(p, [])
        f.write(f"### {p} ({len(recs)} records)\n\n")
        f.write(f"**Suggested policy:** {default}\n\n")
        for r in recs[:5]:
            f.write(f"```\n{r.get('text_preview', '')[:250]}\n```\n\n")
        f.write("**Your decision:** [ ] REDACT [ ] KEEP [ ] EXCLUDE\n\n")
        f.write("---\n\n")

    f.write("## Institution markers\n\n")
    f.write("These appear in CLAUDE.md already as public context. Likely KEEP.\n\n")
    for p in ['INSTITUTION.cm_lab', 'INSTITUTION.weill_cornell', 'INSTITUTION.mason_lab', 'INSTITUTION.cornell', 'INSTITUTION.kribb']:
        recs = flag_groups.get(p, [])
        if not recs:
            continue
        f.write(f"### {p} ({len(recs)} records)\n\n")
        for r in recs[:3]:
            f.write(f"```\n{r.get('text_preview', '')[:250]}\n```\n\n")
        f.write("**Your decision:** [ ] REDACT [ ] KEEP [ ] EXCLUDE\n\n")
        f.write("---\n\n")

    f.write("## Research-sensitive markers\n\n")
    f.write("These need careful review — may reveal unpublished work or collaborators.\n\n")
    for p in ['RESEARCH.grant', 'RESEARCH.unpublished', 'RESEARCH.collaborator_ref', 'RESEARCH.hypothesis', 'RESEARCH.patent']:
        recs = flag_groups.get(p, [])
        if not recs:
            continue
        f.write(f"### {p} ({len(recs)} records)\n\n")
        for r in recs[:5]:
            f.write(f"**[{r.get('id')}]**\n")
            f.write(f"```\n{r.get('text_preview', '')[:300]}\n```\n\n")
        f.write("**Your decision:** [ ] REDACT [ ] KEEP [ ] EXCLUDE per record\n\n")
        f.write("---\n\n")

print(f"\n✓ Policy doc: {policy_doc}")

# =====================================================================
# Summarize sensitive content concentration
# =====================================================================
print("\n\n[2] High-density flagged records (4+ flag types per record)")
print("    These are the most sensitive — likely EXCLUDE candidates")

dense_records = [r for r in records if len(r['flags']) >= 4]
print(f"    Found {len(dense_records)} dense records")

dense_path = os.path.join(AUDIT_DIR, 'dense_flagged_records.md')
with open(dense_path, 'w') as f:
    f.write(f"# Dense flagged records (4+ flag types)\n\n")
    f.write(f"These records have many sensitive patterns and should be reviewed individually.\n\n")
    f.write(f"Total: {len(dense_records)}\n\n---\n\n")

    for i, r in enumerate(dense_records, 1):
        f.write(f"## {i}. [{r['severity']}] `{r['id']}`\n\n")
        f.write(f"**Source:** {r['source']}\n")
        f.write(f"**Project:** {r.get('project_type', 'N/A')}\n")
        f.write(f"**Flag types:** {len(r['flags'])}\n\n")
        f.write(f"**Flags:**\n")
        for flag, count in r['flags'].items():
            f.write(f"- {flag}: {count}\n")
        f.write(f"\n**Text preview:**\n```\n{r.get('text_preview', '')[:500]}\n```\n\n")
        f.write(f"**Decision:** [ ] KEEP [ ] REDACT [ ] EXCLUDE\n\n---\n\n")

print(f"    ✓ {dense_path}")

# =====================================================================
# Source × severity matrix
# =====================================================================
print("\n\n[3] Source × severity breakdown:")
src_sev = defaultdict(lambda: defaultdict(int))
for r in records:
    src_sev[r['source']][r['severity']] += 1

print(f"\n{'Source':<40} {'CRIT':>5} {'HIGH':>5} {'MED':>5} {'LOW':>5}")
print("-" * 65)
for src in sorted(src_sev.keys()):
    sevs = src_sev[src]
    print(f"{src:<40} {sevs.get('CRITICAL',0):>5} {sevs.get('HIGH',0):>5} {sevs.get('MEDIUM',0):>5} {sevs.get('LOW',0):>5}")

print(f"\n{'=' * 70}")
print("ANALYSIS DONE")
print(f"{'=' * 70}")
print(f"""
Next steps:
1. Open policy_decisions.md — set blanket policies (faster than per-record)
2. Open dense_flagged_records.md — review the high-density records
3. Once decisions made, build redaction script (audit_phase1_redact.py)
""")

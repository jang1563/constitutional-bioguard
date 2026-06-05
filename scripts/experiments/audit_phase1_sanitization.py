#!/usr/bin/env python3
"""
Audit Phase 1: Sanitization scan.

Scans session_claude_code and session_codex records in the v2 dataset for:
  1. PII patterns (emails, file paths, phone numbers)
  2. Project codenames (internal project references)
  3. People/lab/institution names
  4. Research-sensitive markers (unpublished work, draft mentions)
  5. Direct identifiers (user names, system info)

Output:
  - data/audit/sanitization_report.json — aggregate counts and pattern stats
  - data/audit/flagged_records.jsonl — records flagged for manual review
  - data/audit/redaction_proposals.jsonl — automated redaction suggestions

The user MUST manually review flagged_records.jsonl before any public release.
"""
import json
import re
import os
from collections import Counter, defaultdict

BASE = '~/Dropbox/Bioinformatics/Claude/Safeguard/constitutional_bioguard'
INPUT = os.path.join(BASE, 'data/raw/unified_overrefusal_taxonomy_v2.jsonl')
OUT_DIR = os.path.join(BASE, 'data/audit')
os.makedirs(OUT_DIR, exist_ok=True)

# =====================================================================
# PATTERN DEFINITIONS
# =====================================================================

# Direct PII
PII_PATTERNS = {
    'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
    'home_path': re.compile(r'/Users/[a-zA-Z0-9_-]+'),
    'tmp_path': re.compile(r'/tmp/[a-zA-Z0-9_/-]+'),
    'dropbox_path': re.compile(r'Dropbox/[A-Za-z0-9_/-]+'),
    'github_url': re.compile(r'github\.com/[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+'),
    'phone': re.compile(r'\b\d{3}[-.]?\d{3,4}[-.]?\d{4}\b'),
    'url_with_path': re.compile(r'https?://[^\s)>\]]+/[^\s)>\]]+'),
}

# JK-specific identifiers
PERSONAL_IDENTIFIERS = {
    'jk4013': re.compile(r'\bjk4013\b', re.IGNORECASE),
    'silveray': re.compile(r'\bsilveray\b', re.IGNORECASE),
    'jihoon': re.compile(r'\bjihoon\b', re.IGNORECASE),
    'jangkim': re.compile(r'\bjang[\s-]?kim\b', re.IGNORECASE),
}

# Institutional / lab markers
INSTITUTION_MARKERS = {
    'weill_cornell': re.compile(r'weill\s*cornell', re.IGNORECASE),
    'mason_lab': re.compile(r'mason\s*lab', re.IGNORECASE),
    'cm_lab': re.compile(r'cmlab|cm[\s_-]+lab', re.IGNORECASE),
    'kribb': re.compile(r'\bkribb\b', re.IGNORECASE),
    'cornell': re.compile(r'\bcornell\b', re.IGNORECASE),
}

# Project codenames (from data/raw/unified_overrefusal_taxonomy_v2)
PROJECT_NAMES = {
    'constitutional_bioguard',
    'bioguard',
    'OverRefusal',
    'FRT_Pilot',
    'FRT',
    'ConstitutionRules',
    'BioSafetyProjestSuite',
    'AmbiguityCasebook',
    'Calibrated_Permissioning',
    'Narrow_Model_Safety_Eval',
    'BioProtocolBench',
    'GeneLab_benchmark',
    'peer-review-benchmark',
    'Negative_result_DB',
    'AI_Scientist_team',
    'BioMed-LLM-Publication',
    'Agentic_system',
    'Anthropic_BioSafety_Portfolio',
    'BioRLHF',
    'BioReview_Training',
    'Manuscript_writing_harness',
    'Evo2',
    'Fram2',
    'i4_mission',
    'Perturb-seq',
    'sceptre',
    'cleanser',
    'jobforge',
}
PROJECT_NAME_RX = re.compile(
    r'\b(' + '|'.join(re.escape(p) for p in PROJECT_NAMES) + r')\b',
    re.IGNORECASE
)

# Research-sensitive markers
RESEARCH_SENSITIVE = {
    'unpublished': re.compile(r'\b(unpublished|in\s+prep(aration)?|under\s+review|submitted|draft|preprint)\b', re.IGNORECASE),
    'hypothesis': re.compile(r'\b(our\s+hypothesis|we\s+hypothesize|we\s+propose|new\s+method|novel\s+approach)\b', re.IGNORECASE),
    'collaborator_ref': re.compile(r'\b(prof(essor)?|dr\.?|pi\b|advisor|collaborator)\s+[A-Z][a-z]+', re.IGNORECASE),
    'grant': re.compile(r'\b(NIH|NSF|R01|R21|K99|U01|grant\s+number|award\s+number)\b'),
    'patent': re.compile(r'\b(patent|provisional|invention\s+disclosure)\b', re.IGNORECASE),
}

# Code/secret patterns
SECRETS = {
    'api_key': re.compile(r'\b(sk-[a-zA-Z0-9-_]{20,}|api[_-]?key[=:]\s*\S+)', re.IGNORECASE),
    'token': re.compile(r'\b(token[=:]\s*\S+|bearer\s+\S+)', re.IGNORECASE),
    'aws_key': re.compile(r'\bAKIA[0-9A-Z]{16}\b'),
    'private_key': re.compile(r'-----BEGIN [A-Z ]*PRIVATE KEY-----'),
}

# =====================================================================
# SCANNING
# =====================================================================

print("=" * 70)
print("AUDIT PHASE 1: SANITIZATION SCAN")
print("=" * 70)

# Load only session records (most sensitive)
print("\n[1] Loading session records...")
session_records = []
with open(INPUT) as f:
    for line in f:
        d = json.loads(line)
        if d.get('source') in ('session_claude_code', 'session_codex'):
            session_records.append(d)
print(f"  Session records: {len(session_records)}")

# Also load OverRefusal/ConstitutionRules (own data, potential drafts/notes)
own_records = []
with open(INPUT) as f:
    for line in f:
        d = json.loads(line)
        if d.get('source') in (
            'overrefusal_api', 'overrefusal_api_context_cond',
            'constitution_rules_phase8', 'constitution_rules_fnr',
            'ambiguity_casebook',
        ):
            own_records.append(d)
print(f"  Own-data records: {len(own_records)}")

# Combined scan target
all_targets = session_records + own_records
print(f"  Total scan targets: {len(all_targets)}")

# =====================================================================
# SCAN
# =====================================================================
print("\n[2] Scanning for patterns...")

def extract_text(record):
    """Concatenate all text content of a record."""
    parts = []
    for k in ('query', 'response', 'assistant_text', 'user_prompt'):
        v = record.get(k, '')
        if isinstance(v, str) and v:
            parts.append(v)
    # Also scan prior_turns
    for turn in record.get('prior_turns', []):
        if isinstance(turn, dict):
            parts.append(turn.get('text', ''))
    return '\n'.join(parts)

flag_counts = defaultdict(Counter)
flagged_records = []
example_per_pattern = defaultdict(list)

for rec in all_targets:
    text = extract_text(rec)
    if not text:
        continue

    rec_flags = {}

    # Run all pattern groups
    for group_name, patterns in [
        ('PII', PII_PATTERNS),
        ('PERSONAL', PERSONAL_IDENTIFIERS),
        ('INSTITUTION', INSTITUTION_MARKERS),
        ('RESEARCH', RESEARCH_SENSITIVE),
        ('SECRETS', SECRETS),
    ]:
        for name, rx in patterns.items():
            matches = rx.findall(text)
            if matches:
                key = f"{group_name}.{name}"
                flag_counts[group_name][name] += len(matches)
                rec_flags[key] = len(matches)
                if len(example_per_pattern[key]) < 3:
                    sample = matches[0] if isinstance(matches[0], str) else str(matches[0])
                    example_per_pattern[key].append({
                        'record_id': rec.get('id'),
                        'sample': sample[:100],
                    })

    # Project names — special handling (very common in own data)
    proj_matches = PROJECT_NAME_RX.findall(text)
    if proj_matches:
        unique_projs = set(p.lower() if isinstance(p, str) else str(p).lower() for p in proj_matches)
        flag_counts['PROJECT'][len(unique_projs) if len(unique_projs) <= 5 else '5+'] += 1
        rec_flags['PROJECT.names_count'] = len(unique_projs)

    if rec_flags:
        flagged_records.append({
            'id': rec.get('id'),
            'source': rec.get('source'),
            'project_type': rec.get('source_detail') or rec.get('project_type'),
            'flags': rec_flags,
            'text_preview': text[:300],
        })

# =====================================================================
# REPORT
# =====================================================================
print("\n" + "=" * 70)
print("PATTERN COUNTS")
print("=" * 70)

for group in ['PII', 'PERSONAL', 'INSTITUTION', 'RESEARCH', 'SECRETS', 'PROJECT']:
    counts = flag_counts.get(group, {})
    if not counts:
        continue
    print(f"\n[{group}]")
    for name, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {count:6d}  {name}")

print(f"\n\n=== TOTALS ===")
print(f"  Records scanned:    {len(all_targets)}")
print(f"  Records flagged:    {len(flagged_records)} ({100*len(flagged_records)/len(all_targets):.1f}%)")
print(f"  Total PII hits:     {sum(flag_counts['PII'].values())}")
print(f"  Total PERSONAL:     {sum(flag_counts['PERSONAL'].values())}")
print(f"  Total INSTITUTION:  {sum(flag_counts['INSTITUTION'].values())}")
print(f"  Total RESEARCH:     {sum(flag_counts['RESEARCH'].values())}")
print(f"  Total SECRETS:      {sum(flag_counts['SECRETS'].values())}")

# Severity classification
def severity(rec_flags):
    if any(k.startswith('SECRETS.') for k in rec_flags):
        return 'CRITICAL'
    if any(k.startswith('PII.email') or k.startswith('PII.home_path') for k in rec_flags):
        return 'HIGH'
    if any(k.startswith('PERSONAL.') or k.startswith('INSTITUTION.') for k in rec_flags):
        return 'MEDIUM'
    if any(k.startswith('RESEARCH.') for k in rec_flags):
        return 'MEDIUM'
    return 'LOW'

sev_counts = Counter()
for r in flagged_records:
    sev = severity(r['flags'])
    r['severity'] = sev
    sev_counts[sev] += 1

print(f"\n=== SEVERITY ===")
for s in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
    print(f"  {sev_counts.get(s, 0):6d}  {s}")

# =====================================================================
# WRITE OUTPUTS
# =====================================================================
print(f"\n[3] Writing audit outputs...")

# Sort flagged records by severity
sev_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
flagged_records.sort(key=lambda r: (sev_order[r['severity']], r['source'], r['id']))

flagged_path = os.path.join(OUT_DIR, 'flagged_records.jsonl')
with open(flagged_path, 'w') as f:
    for r in flagged_records:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
print(f"  ✓ {flagged_path} ({len(flagged_records)} records, sorted by severity)")

# Report
report = {
    'timestamp': '2026-05-28',
    'total_scanned': len(all_targets),
    'session_records': len(session_records),
    'own_records': len(own_records),
    'total_flagged': len(flagged_records),
    'flag_rate_pct': 100 * len(flagged_records) / len(all_targets),
    'severity_breakdown': dict(sev_counts),
    'pattern_counts_by_group': {
        g: dict(c) for g, c in flag_counts.items()
    },
    'example_per_pattern': dict(example_per_pattern),
    'next_actions': [
        f'1. Review CRITICAL records first ({sev_counts.get("CRITICAL", 0)} records) — secrets must be removed',
        f'2. Review HIGH severity ({sev_counts.get("HIGH", 0)} records) — PII like emails, home paths',
        f'3. Review MEDIUM severity ({sev_counts.get("MEDIUM", 0)} records) — institution, research markers',
        f'4. Spot-check LOW severity ({sev_counts.get("LOW", 0)} records) — project names typically OK',
        f'5. Build redaction script after manual review',
    ],
}
report_path = os.path.join(OUT_DIR, 'sanitization_report.json')
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print(f"  ✓ {report_path}")

# Generate human-readable review queue
queue_path = os.path.join(OUT_DIR, 'review_queue_critical_and_high.md')
with open(queue_path, 'w') as f:
    f.write("# Sanitization Review Queue: CRITICAL + HIGH severity\n\n")
    f.write(f"**Total to review:** {sev_counts.get('CRITICAL', 0) + sev_counts.get('HIGH', 0)} records\n\n")
    f.write("Review each entry below. Decision: KEEP / REDACT (partial) / EXCLUDE (full).\n\n")
    f.write("---\n\n")

    n = 0
    for r in flagged_records:
        if r['severity'] not in ('CRITICAL', 'HIGH'):
            continue
        n += 1
        f.write(f"## {n}. [{r['severity']}] {r['id']}\n\n")
        f.write(f"**Source:** {r['source']}\n")
        f.write(f"**Project type:** {r.get('project_type', '')}\n")
        f.write(f"**Flags:** {r['flags']}\n\n")
        f.write(f"**Text preview:**\n")
        f.write(f"```\n{r['text_preview']}\n```\n\n")
        f.write(f"**Decision:** [ ] KEEP  [ ] REDACT  [ ] EXCLUDE\n\n")
        f.write("---\n\n")

print(f"  ✓ {queue_path} ({n} records to manually review)")

print(f"\n{'=' * 70}")
print("AUDIT PHASE 1 SCAN DONE")
print("=" * 70)
print(f"""
Next steps for you (manual review):

1. Open {queue_path}
2. For each entry, decide: KEEP / REDACT / EXCLUDE
   - KEEP: pattern is benign (e.g., common project name in publishable context)
   - REDACT: specific tokens need replacement (e.g., email → [REDACTED])
   - EXCLUDE: whole record removed from public release

3. Estimated time: ~1-2 minutes per CRITICAL/HIGH entry
   Total: {(sev_counts.get('CRITICAL', 0) + sev_counts.get('HIGH', 0)) * 1.5:.0f} minutes ({(sev_counts.get('CRITICAL', 0) + sev_counts.get('HIGH', 0)) * 1.5 / 60:.1f} hours)

4. Once decisions are made, re-run with --apply-redactions to produce
   data/audit/sanitized_for_release.jsonl
""")

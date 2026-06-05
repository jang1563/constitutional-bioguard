#!/usr/bin/env python3
"""
Audit Phase 1 — Apply redaction policies.

Policies (decided 2026-05-28):
  - /Users/jak4013/  →  ~/           (GENERALIZE)
  - jak4013@med.cornell.edu  →  [EMAIL]  (REDACT)
  - All other patterns: KEEP

Input:  data/raw/unified_overrefusal_taxonomy_v2.jsonl
Output: data/audit/sanitized_v2.jsonl  (redacted copy)
        data/audit/redaction_log.json  (what changed)
"""
import json
import os
import re

BASE = '/Users/jak4013/Dropbox/Bioinformatics/Claude/Safeguard/constitutional_bioguard'
INPUT = os.path.join(BASE, 'data/raw/unified_overrefusal_taxonomy_v2.jsonl')
OUT = os.path.join(BASE, 'data/audit/sanitized_v2.jsonl')
LOG = os.path.join(BASE, 'data/audit/redaction_log.json')

REDACTIONS = [
    (re.compile(r'/Users/jak4013/'), '~/'),
    (re.compile(r'/Users/jak4013\b'), '~'),
    (re.compile(r'jak4013@med\.cornell\.edu'), '[EMAIL]'),
]

TEXT_FIELDS = ['query', 'response', 'assistant_text', 'user_prompt', 'block_message',
               'assistant_response', 'text_preview']

def redact_value(val):
    if not isinstance(val, str):
        return val, 0
    changes = 0
    for rx, repl in REDACTIONS:
        new_val = rx.sub(repl, val)
        if new_val != val:
            changes += len(rx.findall(val))
            val = new_val
    return val, changes

def redact_record(rec):
    total = 0
    for field in TEXT_FIELDS:
        if field in rec and isinstance(rec[field], str):
            rec[field], n = redact_value(rec[field])
            total += n
    if 'prior_turns' in rec and isinstance(rec['prior_turns'], list):
        for turn in rec['prior_turns']:
            if isinstance(turn, dict):
                for k in ('text', 'content'):
                    if k in turn and isinstance(turn[k], str):
                        turn[k], n = redact_value(turn[k])
                        total += n
    return total

records_changed = 0
total_substitutions = 0
per_pattern = {'home_path': 0, 'email': 0}

with open(INPUT) as fin, open(OUT, 'w') as fout:
    for line in fin:
        rec = json.loads(line)
        raw = line
        n = redact_record(rec)
        if n > 0:
            records_changed += 1
            total_substitutions += n
        fout.write(json.dumps(rec, ensure_ascii=False) + '\n')

with open(INPUT) as f:
    raw_text = f.read()
    per_pattern['home_path'] = len(re.findall(r'/Users/jak4013', raw_text))
    per_pattern['email'] = len(re.findall(r'jak4013@med\.cornell\.edu', raw_text))

log = {
    'timestamp': '2026-05-28',
    'input': INPUT,
    'output': OUT,
    'policies_applied': [
        {'pattern': '/Users/jak4013/', 'action': 'GENERALIZE', 'replacement': '~/'},
        {'pattern': 'jak4013@med.cornell.edu', 'action': 'REDACT', 'replacement': '[EMAIL]'},
    ],
    'policies_kept': [
        'INSTITUTION (Weill Cornell, Mason Lab, cmlab) — KEEP',
        'RESEARCH (grant, unpublished, collaborator_ref) — KEEP all',
        'PII.github_url — KEEP (public repo)',
        'PII.phone — KEEP (false positive: mp4 filename)',
        'PII.dropbox_path — resolved by home_path generalization',
        'PII.tmp_path — resolved by home_path generalization',
        'PROJECT names — KEEP',
    ],
    'records_changed': records_changed,
    'total_substitutions': total_substitutions,
    'per_pattern_counts': per_pattern,
}

with open(LOG, 'w') as f:
    json.dump(log, f, indent=2)

print(f'Redacted {records_changed} records ({total_substitutions} substitutions)')
print(f'  home_path matches: {per_pattern["home_path"]}')
print(f'  email matches: {per_pattern["email"]}')
print(f'Output: {OUT}')
print(f'Log: {LOG}')

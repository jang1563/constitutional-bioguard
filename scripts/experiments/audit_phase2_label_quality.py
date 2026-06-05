#!/usr/bin/env python3
"""
Audit Phase 2 — Label quality.

Produces a stratified sample of ~100 records from sanitized_v2.jsonl
balanced across (source × binary_label) cells, with the assigned tier
and binary label, for user review.

Output:
  - data/audit/label_quality_sample.jsonl  — 100 sampled records (machine-readable)
  - data/audit/label_quality_review.md     — review form (human-readable)
"""
import json
import os
import random
from collections import defaultdict, Counter

BASE = '/Users/jak4013/Dropbox/Bioinformatics/Claude/Safeguard/constitutional_bioguard'
INPUT = os.path.join(BASE, 'data/audit/sanitized_v2.jsonl')
OUT_JSONL = os.path.join(BASE, 'data/audit/label_quality_sample.jsonl')
OUT_MD = os.path.join(BASE, 'data/audit/label_quality_review.md')

LEGIT_TIERS = {
    'select_agent_research_legitimate', 'control_protein_clearly_benign',
    'dual_use_benign_context', 'textbook_knowledge', 'published_literature',
    'research_workflow_legitimate', 'constitution_assist_freely',
    'constitution_assist_with_caveats', 'benchmark_should_not_refuse',
    'expert_decision_allow', 'expert_over_refusal', 'expert_aligned',
    'anonymous_protein_unclear',
}
NEG_TIERS = {
    'expected_refusal', 'benchmark_correctly_refused',
    'constitution_hard_refuse', 'constitution_decline_with_alternative',
    'expert_decision_refuse',
}
SESSION_TIERS = {
    'meta_refusal_continuation', 'meta_refusal_tool_output',
    'meta_refusal_session_context',
}
def to_bin(t):
    if t in LEGIT_TIERS: return 'legitimate'
    if t in NEG_TIERS: return 'negative'
    if t in SESSION_TIERS: return 'session_block_legit'
    return 'ambiguous'

random.seed(42)

records = [json.loads(l) for l in open(INPUT)]
print(f'Loaded {len(records)} records')

# Bucket by (source, binary_label)
buckets = defaultdict(list)
for r in records:
    src = r.get('source', 'UNK')
    tier = r.get('legitimacy_tier', '')
    binlab = to_bin(tier)

    if src == 'external_benchmark' and r.get('source_detail') == 'wildguard':
        key = ('wildguard_test', binlab)
    else:
        key = (src, binlab)
    buckets[key].append(r)

print(f'\nBuckets ({len(buckets)} cells):')
for k in sorted(buckets, key=lambda x: -len(buckets[x])):
    print(f'  {len(buckets[k]):5d}  {k[0]:35s} {k[1]}')

# Sample: 5 per cell, capped (no top-up). Even coverage > target count.
# Rationale: frt_p2_kill_chain homogeneity is already documented (C2 finding);
# oversampling it adds little information about label quality.
SAMPLES_PER_CELL = 5
sampled = []
for key, recs in buckets.items():
    n = min(SAMPLES_PER_CELL, len(recs))
    sampled.extend(random.sample(recs, n))

print(f'\nSampled {len(sampled)} records for review')

# Write JSONL
with open(OUT_JSONL, 'w') as f:
    for r in sampled:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')

# Write review markdown
def truncate(s, n=400):
    if not isinstance(s, str): return ''
    s = s.strip()
    if len(s) <= n: return s
    return s[:n] + f' ...[truncated {len(s)-n} chars]'

with open(OUT_MD, 'w') as f:
    f.write('# Label Quality Review — Phase 2\n\n')
    f.write(f'**Sampled:** {len(sampled)} records, stratified by (source × binary_label).\n')
    f.write(f'**Source data:** `data/audit/sanitized_v2.jsonl` (post-redaction)\n')
    f.write(f'**Random seed:** 42 (reproducible)\n\n')
    f.write('## How to review\n\n')
    f.write('For each record:\n')
    f.write('- ✅ CORRECT — assigned label matches the content\n')
    f.write('- ❌ WRONG — should be different (specify what)\n')
    f.write('- ❓ AMBIGUOUS — could go either way, document why\n\n')
    f.write('---\n\n')

    # Group by source for easier review
    by_src = defaultdict(list)
    for r in sampled:
        src = r.get('source', 'UNK')
        if src == 'external_benchmark' and r.get('source_detail') == 'wildguard':
            src = 'wildguard_test'
        by_src[src].append(r)

    for src in sorted(by_src):
        recs = by_src[src]
        f.write(f'## Source: `{src}` (n={len(recs)})\n\n')
        for i, r in enumerate(recs, 1):
            tier = r.get('legitimacy_tier', '')
            binlab = to_bin(tier)
            f.write(f'### {src} — {i}. `{r.get("id", "?")}`\n\n')
            f.write(f'- **legitimacy_tier:** `{tier}`\n')
            f.write(f'- **binary_label:** `{binlab}`\n')
            f.write(f'- **model:** `{r.get("model", "?")}`\n')
            if r.get('source_detail'):
                f.write(f'- **source_detail:** `{r["source_detail"]}`\n')
            if r.get('content_domain'):
                f.write(f'- **content_domain:** `{r["content_domain"]}`\n')
            f.write(f'\n**Query:**\n```\n{truncate(r.get("query", "") or r.get("user_prompt", ""))}\n```\n\n')
            resp = r.get('response') or r.get('assistant_response') or r.get('assistant_text', '')
            if resp:
                f.write(f'**Response:**\n```\n{truncate(resp)}\n```\n\n')
            f.write(f'**Verdict:** [ ] ✅ CORRECT  [ ] ❌ WRONG (\\_\\_\\_\\_)  [ ] ❓ AMBIGUOUS\n\n')
            f.write('**Notes:** \n\n---\n\n')

print(f'\nOutput:')
print(f'  {OUT_JSONL}')
print(f'  {OUT_MD}')

# Summary
print(f'\nSample distribution:')
sample_dist = Counter()
for r in sampled:
    src = r.get('source', 'UNK')
    if src == 'external_benchmark' and r.get('source_detail') == 'wildguard':
        src = 'wildguard_test'
    binlab = to_bin(r.get('legitimacy_tier', ''))
    sample_dist[(src, binlab)] += 1
for (src, lab), n in sorted(sample_dist.items()):
    print(f'  {n:3d}  {src:35s} {lab}')

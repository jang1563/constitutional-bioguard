#!/usr/bin/env python3
"""
Build 4-way data splits for bioguard classifier:

  TRAIN:      High-volume sources safe for training
  VAL:        In-distribution hold-out (same sources as train)
  OOD_FPR:    Should-NOT-refuse evaluation (over-refusal / false positive rate)
  OOD_FNR:    Should-REFUSE evaluation (under-refusal / false negative rate)

Design rationale:
  - WildGuardTest & SALAD-Bench are evaluation-only benchmarks (benchmark contamination)
  - LODO-inspired source-level separation (no source spans both train and OOD)
  - XSTest-style dual partition: FPR and FNR reported separately
  - FRT P2 downsampled to reduce protein_engineering dominance
  - Session blocks reserved entirely for OOD (real user distribution)

References:
  - "When Benchmarks Lie" (2026): LODO protocol, 8.4 AUC overestimation from naive splits
  - WildGuard (NeurIPS 2024): Train/test architectural separation
  - XSTest (NAACL 2024): Dual-partition should/should-not refuse
  - RefusalBench-Bio (2026): Youden's J for tier discrimination
"""
import json
import os
import random
import hashlib
from collections import Counter, defaultdict

random.seed(42)

BASE = '~/Dropbox/Bioinformatics/Claude/Safeguard/constitutional_bioguard'
INPUT = os.path.join(BASE, 'data/raw/unified_overrefusal_taxonomy_v2.jsonl')
OUT_DIR = os.path.join(BASE, 'data/splits')
os.makedirs(OUT_DIR, exist_ok=True)

# =====================================================================
# CONFIGURATION
# =====================================================================

FRT_P2_DOWNSAMPLE = 5000

# Sources assigned to each split
TRAIN_SOURCES = {
    'frt_p1_stepping_stone',
    'frt_p2_kill_chain',
    'frt_p3_mcp_keyword',
    'overrefusal_api',
    'constitution_rules_phase8',
}
TRAIN_EXTERNAL = {
    'alert_cbrn',
    'advbench',
}

OOD_FPR_SOURCES = {
    'session_logs_primary',
    'session_logs_secondary',
    'overrefusal_api_context_cond',
    'ambiguity_casebook',
    'biosafety_suite_decision_compiler',
    'biosafety_suite_fixture',
}

OOD_FNR_SOURCES = {
    'wildguard',
    'saladbench',
    'constitution_rules_fnr',
}

# Legitimacy tiers → binary label
LEGIT_TIERS = {
    'select_agent_research_legitimate', 'control_protein_clearly_benign',
    'dual_use_benign_context', 'textbook_knowledge', 'published_literature',
    'research_workflow_legitimate', 'constitution_assist_freely',
    'constitution_assist_with_caveats', 'benchmark_should_not_refuse',
    'expert_decision_allow', 'expert_over_refusal', 'expert_aligned',
    'expert_decision_caveat',
    'anonymous_protein_unclear',
    # Audit Phase 2 (2026-05-28): over-refusal categories mapped to legitimate
    'dual_use_ambiguous',  # mechanistic questions about pathogens in research context
    'meta_refusal_continuation',  # session resumptions ("이어서", "resume")
    'meta_refusal_session_context',  # project/workflow instructions
    'meta_refusal_tool_output',  # tool artifacts (should not trigger refusal)
}
NEG_TIERS = {
    'expected_refusal', 'benchmark_correctly_refused',
    'constitution_hard_refuse', 'constitution_decline_with_alternative',
    'expert_decision_refuse',
}

VAL_RATIO = 0.15

# =====================================================================
# LOAD & ROUTE
# =====================================================================
print("=" * 70)
print("BUILD SPLITS: train / val / ood-fpr / ood-fnr")
print("=" * 70)

with open(INPUT) as f:
    all_records = [json.loads(line) for line in f]
print(f"\nLoaded {len(all_records)} records")

def assign_binary_label(rec):
    lt = rec.get('legitimacy_tier', '')
    if lt in LEGIT_TIERS:
        return 'legitimate'
    elif lt in NEG_TIERS:
        return 'negative'
    return 'ambiguous'

def source_bucket(rec):
    """Map record source to the external benchmark name if applicable."""
    src = rec.get('source', '')
    detail = rec.get('source_detail', '')
    bsrc = rec.get('benchmark_source', '')

    if src in TRAIN_SOURCES:
        return 'train_internal', src
    if src == 'external_benchmark':
        if bsrc in ('alert_cbrn', 'advbench_bio', 'advbench_full'):
            return 'train_external', bsrc
        elif bsrc in ('wildguard_test',):
            return 'ood_fnr', 'wildguard'
        elif bsrc in ('saladbench_cbrn',):
            return 'ood_fnr', 'saladbench'
        else:
            return 'train_external', bsrc
    if src in OOD_FPR_SOURCES:
        return 'ood_fpr', src
    if src in OOD_FNR_SOURCES:
        return 'ood_fnr', src
    return 'unassigned', src

train_pool = []
ood_fpr = []
ood_fnr = []
unassigned = []

for rec in all_records:
    rec['binary_label'] = assign_binary_label(rec)
    bucket, subsrc = source_bucket(rec)

    if bucket == 'train_internal' or bucket == 'train_external':
        train_pool.append(rec)
    elif bucket == 'ood_fpr':
        ood_fpr.append(rec)
    elif bucket == 'ood_fnr':
        ood_fnr.append(rec)
    else:
        unassigned.append(rec)

print(f"\nInitial routing:")
print(f"  Train pool:  {len(train_pool)}")
print(f"  OOD-FPR:     {len(ood_fpr)}")
print(f"  OOD-FNR:     {len(ood_fnr)}")
if unassigned:
    print(f"  Unassigned:  {len(unassigned)}")
    srcs = Counter(r.get('source') for r in unassigned)
    for s, n in srcs.most_common():
        print(f"    {n:5d}  {s}")

# =====================================================================
# DOWNSAMPLE FRT P2
# =====================================================================
print(f"\n[1] Downsampling FRT P2...")
frt_p2 = [r for r in train_pool if r['source'] == 'frt_p2_kill_chain']
non_p2 = [r for r in train_pool if r['source'] != 'frt_p2_kill_chain']

if len(frt_p2) > FRT_P2_DOWNSAMPLE:
    # Stratified downsample: maintain protein_id distribution
    by_protein = defaultdict(list)
    for r in frt_p2:
        by_protein[r.get('protein_id', 'unknown')].append(r)

    sampled_p2 = []
    per_protein = FRT_P2_DOWNSAMPLE // len(by_protein)
    remainder = FRT_P2_DOWNSAMPLE % len(by_protein)

    for i, (pid, recs) in enumerate(sorted(by_protein.items())):
        n = per_protein + (1 if i < remainder else 0)
        random.shuffle(recs)
        sampled_p2.extend(recs[:n])

    random.shuffle(sampled_p2)
    print(f"  FRT P2: {len(frt_p2)} → {len(sampled_p2)} (stratified by protein_id)")
    frt_p2 = sampled_p2

train_pool = non_p2 + frt_p2
random.shuffle(train_pool)

# =====================================================================
# TRAIN / VAL SPLIT (stratified by source + binary_label)
# =====================================================================
print(f"\n[2] Stratified train/val split (val_ratio={VAL_RATIO})...")

strata = defaultdict(list)
for r in train_pool:
    key = (r['source'], r['binary_label'])
    strata[key].append(r)

train_set = []
val_set = []

for key, recs in strata.items():
    random.shuffle(recs)
    n_val = max(1, int(len(recs) * VAL_RATIO))
    val_set.extend(recs[:n_val])
    train_set.extend(recs[n_val:])

random.shuffle(train_set)
random.shuffle(val_set)

# =====================================================================
# SUMMARY STATS
# =====================================================================
def split_stats(name, records):
    labels = Counter(r['binary_label'] for r in records)
    sources = Counter(r['source'] for r in records)
    domains = Counter(r['content_domain'] for r in records)
    total = len(records)

    print(f"\n{'─' * 50}")
    print(f"  {name}: {total} records")
    print(f"{'─' * 50}")

    print(f"  Labels:")
    for l in ['legitimate', 'negative', 'ambiguous']:
        n = labels.get(l, 0)
        pct = 100 * n / total if total > 0 else 0
        print(f"    {n:6d} ({pct:5.1f}%)  {l}")

    print(f"  Sources (top 5):")
    for s, n in sources.most_common(5):
        print(f"    {n:6d}  {s}")

    print(f"  Domains (top 5):")
    for d, n in domains.most_common(5):
        print(f"    {n:6d}  {d}")

    return {
        'total': total,
        'labels': dict(labels),
        'sources': dict(sources),
        'domains': dict(domains),
    }

print("\n" + "=" * 70)
print("SPLIT SUMMARY")
print("=" * 70)

stats = {}
stats['train'] = split_stats('TRAIN', train_set)
stats['val'] = split_stats('VAL (in-distribution)', val_set)
stats['ood_fpr'] = split_stats('OOD-FPR (should NOT refuse)', ood_fpr)
stats['ood_fnr'] = split_stats('OOD-FNR (should REFUSE)', ood_fnr)

# =====================================================================
# WRITE SPLITS
# =====================================================================
print(f"\n\n[3] Writing split files...")

def write_split(name, records):
    path = os.path.join(OUT_DIR, f'{name}.jsonl')
    with open(path, 'w') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f"  ✓ {path}: {len(records)} records")
    return path

write_split('train', train_set)
write_split('val', val_set)
write_split('ood_fpr', ood_fpr)
write_split('ood_fnr', ood_fnr)

# =====================================================================
# WRITE SPLIT METADATA
# =====================================================================
meta = {
    'created': '2026-05-28',
    'input': INPUT,
    'config': {
        'frt_p2_downsample': FRT_P2_DOWNSAMPLE,
        'val_ratio': VAL_RATIO,
        'seed': 42,
    },
    'design': {
        'train': 'FRT P1/P2(downsampled)/P3 + alert_cbrn + advbench + ConstitutionRules P8 + OverRefusal API base',
        'val': f'{VAL_RATIO:.0%} stratified holdout from train sources',
        'ood_fpr': 'Session blocks + OverRefusal ctx_cond + AmbiguityCasebook + BioSafetyProj (should NOT refuse)',
        'ood_fnr': 'WildGuardTest + SALAD-Bench CBRN + ConstitutionRules FNR (should REFUSE)',
    },
    'references': [
        'LODO: "When Benchmarks Lie" (2026) — source-level OOD separation',
        'XSTest (NAACL 2024) — dual-partition FPR/FNR',
        'WildGuard (NeurIPS 2024) — train/test architectural separation',
        'RefusalBench-Bio (2026) — matched-triple, Youden\'s J',
    ],
    'stats': stats,
}

meta_path = os.path.join(OUT_DIR, 'split_metadata.json')
with open(meta_path, 'w') as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)
print(f"  ✓ {meta_path}")

# =====================================================================
# LODO CROSS-VALIDATION FOLDS
# =====================================================================
print(f"\n\n[4] Generating LODO cross-validation folds...")

lodo_sources = {
    'frt_p2': lambda r: r['source'] == 'frt_p2_kill_chain',
    'alert_cbrn': lambda r: r.get('benchmark_source') == 'alert_cbrn',
    'constitution_rules_phase8': lambda r: r['source'] == 'constitution_rules_phase8',
    'advbench': lambda r: r.get('benchmark_source') in ('advbench_bio', 'advbench_full'),
}
for fold_key, match_fn in lodo_sources.items():
    fold_name = f"lodo_holdout_{fold_key.replace('_', '-')}"
    fold_train = [r for r in train_set if not match_fn(r)]
    fold_test = [r for r in train_set if match_fn(r)]
    if fold_test:
        write_split(fold_name + '_train', fold_train)
        write_split(fold_name + '_test', fold_test)

print(f"\n" + "=" * 70)
print("DONE")
print("=" * 70)

total_all = len(train_set) + len(val_set) + len(ood_fpr) + len(ood_fnr)
print(f"\nTotal records across all splits: {total_all}")
print(f"  TRAIN:    {len(train_set):>6} ({100*len(train_set)/total_all:.1f}%)")
print(f"  VAL:      {len(val_set):>6} ({100*len(val_set)/total_all:.1f}%)")
print(f"  OOD-FPR:  {len(ood_fpr):>6} ({100*len(ood_fpr)/total_all:.1f}%)")
print(f"  OOD-FNR:  {len(ood_fnr):>6} ({100*len(ood_fnr)/total_all:.1f}%)")

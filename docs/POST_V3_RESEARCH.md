# Post-v3 Research Roadmap

What to investigate once v3 lands (regardless of outcome).

---

## If v3 succeeds (meets all 3 success criteria)

### Phase 1: Productionization (1-2 weeks)
1. **Threshold calibration on held-outs.** Find the operating point that
   maximizes utility = recall on bio adv held-outs - 0.1 * FAR on cross
   domain held-outs. Save calibrated threshold to v3_provenance.json.
2. **Latency benchmarking.** Measure inference latency on Cayuga CPU
   (target: serving budget < 50ms per query/response pair).
3. **ONNX export + serving script.** Update `scripts/export_onnx.py` and
   `scripts/serve.py` for the v3 model.

### Phase 2: External baseline comparison (2-3 weeks)
Compare v3 against published safety classifiers on identical benchmarks:
- **LLaMA-Guard 3 8B** (Meta) — standard baseline
- **WildGuard 7B** (AI2) — designed for the WildGuardMix distribution
- **Aegis-AI-Content-Safety-LlamaGuard-Defensive-1.0** (NVIDIA)
- **ShieldGemma-9B** (Google)

Run all on BioThreat-Eval + HarmBench bio + AdvBench bio + WildGuardMix.
v3 should be competitive on bio recall (it's domain-specialized) but
likely lose on cross-domain breadth (it's only 184M parameters vs 7-9B).

The story: "Domain-specialized small models can match larger
general-purpose safety LMs on the target domain at 30-50x lower
inference cost."

### Phase 3: CC++ component re-analysis on v3 base (2-3 weeks)
Now that the base classifier actually learns bio content (not framing):
1. **WS-1 cascade**: re-run escalation calibration. Does v3 give a
   meaningful Stage 1 -> Stage 2 ratio at <2% flag rate?
2. **WS-3 probe ensemble**: with non-saturated metrics, does the
   ensemble actually outperform individual components?
3. **WS-4 attacks**: do the 25 reconstruction attacks still get 0% ASR
   on v3, or does meaningful bio understanding open up new attack
   surfaces?

These are the experiments that the synthetic-data-ceiling effect
previously made impossible. v3 unlocks them.

---

## If v3 partially succeeds (v3b/c needed)

Iterate on the contingency plan (V3_CONTINGENCY.md). Most likely
outcomes:
- v3 too aggressive on UNSAFE -> v3b at weight=1.5
- v3 too conservative -> v3c at weight=2.5 + larger UNSAFE pool
- Two trainings can run in parallel on Cayuga (different GPUs)

Once a working variant lands, proceed with Phase 1-3 above.

---

## If v3 fails entirely (data ceiling confirmed)

The data-centric remedy hypothesis is wrong. Switch to:

### Option A: Synthetic data regeneration (1 week)
- Regenerate UNSAFE synthetic data with explicit instructions to
  produce diverse, non-adversarial phrasings of bio harm.
- Use Claude 4.5 Sonnet, 3000-4000 items.
- Persona-diversified generation (per arXiv:2511.01490) for SAFE bio.
- Retrain from scratch (v4 with regenerated data).

### Option B: Larger base model (1-2 weeks)
- Move from DeBERTa-v3-base (184M) to DeBERTa-v3-large (435M) or
  ModernBERT-large (395M).
- Larger models have demonstrably better OOD generalization on safety
  tasks (Inan et al. 2023, "LLaMA Guard").
- Cost: more compute, longer training, more inference latency.

### Option C: Hybrid filter + classifier (2-3 weeks)
- Keep v3 as the "second stage" classifier for bio-content queries.
- Add a "first stage" generic safety filter (LLaMA-Guard or similar)
  to catch obviously harmful non-bio content.
- Two-stage cascade follows CC++ design: cheap general filter + expensive
  specialized classifier.

---

## Cross-cutting: Documentation & write-up

After v3 lands (any outcome), write up:
1. **Technical Report v2.0** — full v2 + v3 results, lessons learned,
   updated CC++ implications. Targeted at AIES 2026 (already submitted
   v1.x) or a follow-up workshop.
2. **Hugging Face model card** for v3 (or chosen variant).
3. **Blog post / lab notebook** on the shortcut-learning diagnosis
   and the data-centric fix.

---

## Calibration with Anthropic Safeguards goals

This is research aligned with what Safeguards Labs would actually care
about:
- **Empirical evidence on CC++ transferability** to domain-specific
  threats (CBRN gap analysis).
- **Shortcut learning diagnosis** — the WS-2/6.8/6.8b chain is a clean
  case study of how internal metrics can mislead.
- **Data-centric fixes vs architectural fixes** — v2/v3 results
  directly inform "what kind of data do we need next" debates.

The technical report becomes a portfolio piece for the Safeguards Labs
RE application: it demonstrates exactly the diagnostic + iterative
fixing loop the role asks for.

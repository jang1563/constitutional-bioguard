# docs/ index

A map of this directory. Most files are the working research record; only a handful are the
curated, authoritative read. **If you are reviewing this project, read the first group and stop.**

> **New here?** Start at the [root README](../README.md) for the 30-second overview and the
> size-peer Pareto figure. Released models live on Hugging Face:
> [response head](https://huggingface.co/jang1563/constitutional-bioguard-response) ·
> [prompt head](https://huggingface.co/jang1563/constitutional-bioguard-prompt) ·
> [deberta-v1 (legacy)](https://huggingface.co/jang1563/constitutional-bioguard-deberta-v1).
>
> **5-minute reviewer path:** [MODEL_CARD.md](MODEL_CARD.md) (what shipped + honest performance) →
> [CASE_STUDY_eval_self_red_team.md](CASE_STUDY_eval_self_red_team.md) (the 8-point lessons) →
> skim [INTEGRITY_REVIEW_2026-06-04.md](INTEGRITY_REVIEW_2026-06-04.md). Everything else is provenance.

## Start here (authoritative)

| Doc | What it is |
|---|---|
| [MODEL_CARD.md](MODEL_CARD.md) | **Authoritative model card** — the released dual-mode guard (response head v8bh + prompt head), honest performance, limitations, license. |
| [CASE_STUDY_eval_self_red_team.md](CASE_STUDY_eval_self_red_team.md) | Self-red-team case study: the seven ways the evaluation misled, + a reusable 8-point checklist. |
| [INTEGRITY_REVIEW_2026-06-04.md](INTEGRITY_REVIEW_2026-06-04.md) | Full audit trail — every headline claim stress-tested (17 confirmed / 8 refuted), with the corrected numbers. |
| [POSTMORTEM_2026-06-04.md](POSTMORTEM_2026-06-04.md) | Plan vs. reality: what was intended, what shipped, what was abandoned and why. |
| [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) | Full primary-evidence report (v1→v6 arc). Long by design; has an Executive summary + Contents at the top. |

Everything below is the **supporting record** — point-in-time notes kept for provenance, not the current word.

## Superseded model cards (historical)

- [MODEL_CARD_V4.md](MODEL_CARD_V4.md) — v4 single-head card. Superseded by `MODEL_CARD.md`.
- [V8B_MODEL_CARD.md](V8B_MODEL_CARD.md) — v8b card. Superseded by the shipped v8bh (`MODEL_CARD.md`).
- [DUAL_MODE_GUARD_ARTIFACT.md](DUAL_MODE_GUARD_ARTIFACT.md), [V8B_SHIP_EVIDENCE.md](V8B_SHIP_EVIDENCE.md) — earlier artifact / ship-evidence notes.

## Release notes & status (point-in-time)

- [RELEASE_NOTES_2026-05-21.md](RELEASE_NOTES_2026-05-21.md), [RELEASE_NOTES_2026-05-25.md](RELEASE_NOTES_2026-05-25.md)
- [RELEASE_PLAN_2026-06-03.md](RELEASE_PLAN_2026-06-03.md), [V8B_RELEASE_PLAN.md](V8B_RELEASE_PLAN.md)
- [RELEASE_STATUS_2026-06-04.md](RELEASE_STATUS_2026-06-04.md) — latest of this group; superseded by the root README "Current status".

## Design docs (the research arc, per version)

- [V2_DESIGN.md](V2_DESIGN.md) · [V3_CONTINGENCY.md](V3_CONTINGENCY.md) · [V3_GAPS_AUDIT.md](V3_GAPS_AUDIT.md) · [V4_DESIGN.md](V4_DESIGN.md) · [V5_DESIGN.md](V5_DESIGN.md) · [V5_NEXT_ANALYSIS_PLAN.md](V5_NEXT_ANALYSIS_PLAN.md) · [V6_DESIGN_v2.md](V6_DESIGN_v2.md) · [V7_DESIGN.md](V7_DESIGN.md) · [V7B_OVER_REFUSAL_ANALYSIS.md](V7B_OVER_REFUSAL_ANALYSIS.md) · [V8_DESIGN.md](V8_DESIGN.md) · [DUAL_MODE_DESIGN.md](DUAL_MODE_DESIGN.md)

## Dual-mode development research (2026-06)

- [STEP1_DISTILL_PILOT_2026-06-03.md](STEP1_DISTILL_PILOT_2026-06-03.md) · [STEP1B_RESEARCH_2026-06-03.md](STEP1B_RESEARCH_2026-06-03.md) · [STEP2_DUALMODE_2026-06-03.md](STEP2_DUALMODE_2026-06-03.md) · [STEP3_CONFORMAL_2026-06-03.md](STEP3_CONFORMAL_2026-06-03.md) · [STEP4_COMPETITIVE_2026-06-03.md](STEP4_COMPETITIVE_2026-06-03.md) · [STEP4B_DEBIAS_2026-06-04.md](STEP4B_DEBIAS_2026-06-04.md)
- [RESEARCH_REFRESH_2026-06-03.md](RESEARCH_REFRESH_2026-06-03.md) · [POST_V3_RESEARCH.md](POST_V3_RESEARCH.md) · [PHASE3_OOD_SHORTCUT_PLAN.md](PHASE3_OOD_SHORTCUT_PLAN.md)

## Process, checklists, upgrade notes

- [POST_PIPELINE_CHECKLIST.md](POST_PIPELINE_CHECKLIST.md) · [REPOSITORY_QUALITY_CHECKLIST.md](REPOSITORY_QUALITY_CHECKLIST.md) · [VARIANT_EXPERIMENTS.md](VARIANT_EXPERIMENTS.md) · [deberta_v3_upgrade_notes.md](deberta_v3_upgrade_notes.md)

## Working notes (handoffs / session summaries)

- [HANDOFF_2026-05-21.md](HANDOFF_2026-05-21.md) · [HANDOFF_2026-05-31.md](HANDOFF_2026-05-31.md) · [HANDOFF_overrefusal_bridge_2026-06-03.md](HANDOFF_overrefusal_bridge_2026-06-03.md) · [SESSION_SUMMARY_2026_05_20.md](SESSION_SUMMARY_2026_05_20.md)

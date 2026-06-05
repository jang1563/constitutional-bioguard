# scripts/experiments/

Research and experiment scripts from the BioGuard development arc (v1 → v8): data
builds, training variants, concept-erasure/refusal probes, leakage/label audits, and
the matching Cayuga SLURM wrappers.

**These are not maintained entry points.** The supported commands live in `scripts/`
(repo root) and the `Makefile` — see the README "Quick Start". The scripts here are
kept for provenance and to back the experiment trail documented in `docs/`
(`TECHNICAL_REPORT.md`, the `STEP*`/`V*_DESIGN` notes, etc.); paths in those docs point
here. They may reference local data or checkpoints not shipped in this repo and are
not guaranteed to run standalone.

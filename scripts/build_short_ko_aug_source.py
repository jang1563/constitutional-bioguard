#!/usr/bin/env python
"""Build the SHORT-form KO augmentation source — targets the residual KO/short over-refusal.

Context: after the first KO aug (955+955 long, data/aug/ko_aug_source.jsonl) the model's
Korean FP dropped 78.5%->50.5%, but the residual is concentrated in SHORT Korean turns
(KO/short 63.4% vs KO/long 16.4%). Diagnosis (data/audit/diag_v8_koaug_ood_fpr.json): the
92 KO/short FPs are 100% generic conversational/workflow turns ("continue", "next step",
"review again", "prepare a handoff") with ZERO bio content — the model flags them purely
because they are short Korean. TRAIN has essentially no short *conversational* benign
(only 36 short natural-bio queries + 3755 "Query about protein X" stubs), so this register
must be added.

DESIGN (Goodhart-responsible — chosen by JK 2026-05-30, "Author universal templates"):
  1. The conversational benign are AUTHORED in ENGLISH from the UNIVERSAL coding-assistant
     register (continue / approve / review / delegate / plan / handoff / research / status /
     write-up / iterate). Authored in English, NOT transliterated from the Korean probe;
     session-specific items seen in the probe (CodeSignal, ICML scheduling, autopilot, job
     applications) are deliberately EXCLUDED. These land in the residual register after
     NLLB En->Ko, which is the point (language-register coverage), without copying eval
     strings.
  2. Benign side is NOT pure-conversational: short natural-bio benign from TRAIN are mixed
     in, so the model cannot learn "conversational => safe" (it must judge on content).
  3. Harmful side = LENGTH-MATCHED short negatives from TRAIN (mostly cbrn_safety, in-domain
     short attempts like "Tell me how to use Lassa Virus."), so the model cannot learn
     "short KO => safe" or "Korean => safe". The Korean=safe shortcut is additionally
     probed by data/audit/ko_harmful_probe.jsonl (build_ko_harmful_probe.py).
  4. Query-only form (response_en=""): the session FPs are query-only, so the deployment
     text is just the query — match that.

Output: data/aug/short_ko_aug_source.jsonl (English; translate step fills Korean).
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
TRAIN = ROOT / "data" / "splits" / "train.jsonl"

TEMPLATE_RE = re.compile(r"^Query about protein \S+\s*$", re.I)

# ── Authored universal coding-assistant conversational register (ENGLISH) ──────────
# Universal interaction acts present in ANY IDE/agent session. NOT derived by
# transliterating the held-out Korean probe; session-idiosyncratic content excluded.
AUTHORED_CONVERSATIONAL = [
    # continuation
    "Continue.", "Continue please.", "Keep going.", "Let's continue.", "Go on.",
    "Proceed.", "Proceed as suggested.", "Let's proceed as planned.",
    "Move on to the next step.", "Next step.", "On to the next part.",
    "Resume where we left off.", "Pick up where we stopped.", "Carry on.",
    "Let's keep moving.", "Continue from here.", "Go ahead.", "Please go ahead.",
    "Let's go.", "Let's pick this back up.", "Continue the work.",
    # approval / confirmation
    "Sounds good.", "Looks good.", "That works.", "Okay, do that.",
    "Yes, go ahead.", "Approved.", "Agreed.", "Perfect, proceed.", "That's fine.",
    "Fine by me.", "Let's do it.", "Yes, that's right.", "Exactly, continue.",
    "Good, move forward.", "That's the plan.", "Confirmed, proceed.", "Yes please.",
    "Great, next.", "Okay, next.", "Alright, let's proceed.",
    # review / quality
    "Review this again.", "Let's review it once more.", "Double-check the output.",
    "Please proofread it again.", "Verify the facts.", "Check the tone and quality.",
    "Re-examine this carefully.", "Go over it one more time.",
    "Sanity-check the results.", "Audit this section.",
    "Review the current progress thoroughly.", "Take another careful pass.",
    "Check it once more for quality.", "Re-read and refine.", "One more review.",
    "Look it over again.", "Inspect this part closely.", "Quality-check the whole thing.",
    "Recheck the priorities.", "Carefully review the new metrics strategy.",
    "Let's deeply review what we did.", "Review it once more, deeply.",
    # delegation
    "Can you try this?", "You give it a try.", "Want to take a shot at it?",
    "You handle this part.", "You decide.", "Do it your way.",
    "Take the lead on this.", "Go ahead and try.", "See what you can do.",
    "Have a go at it.", "You drive this one.", "Pick the approach you think is best.",
    "You proceed without me.", "Run with it.",
    # explanation
    "Explain that further.", "Walk me through it.", "Clarify this point.",
    "Please elaborate.", "Explain it again.", "Break it down for me.",
    "Tell me more about this.", "Continue the explanation.", "Explain step by step.",
    "Help me understand this.", "What do you mean here?", "Go deeper on this.",
    "Expand on that.", "Describe how this works.", "Explain and then continue.",
    # planning
    "Let's plan this out.", "Outline the steps.", "Draft a plan.",
    "Think it through carefully.", "Brainstorm some approaches.", "Map out the work.",
    "Let's make a plan first.", "Plan the next phase.", "Sketch the overall approach.",
    "Let's think deeply about this.", "Consider the options and decide.",
    "Lay out a roadmap.", "Plan it before we start.", "Let's strategize.",
    "Plan calmly and thoroughly.",
    # session / handoff
    "Prepare a handoff note.", "Summarize the progress.", "Save the plan for later.",
    "Write a handoff document.", "Create a session summary.",
    "Prepare to continue this elsewhere.",
    "Make a markdown handoff with the research included.",
    "Hand this off to another session.",
    "Write up where we are so we can resume.",
    "Bundle this so I can pick it up later.",
    "Prepare a summary so a new session can continue.",
    "Document the state for handoff.", "Save our progress notes.",
    "Get this ready to pass along.", "Prepare a handoff and a ping for the other session.",
    # research
    "Do deeper research on this.", "Look up related work.",
    "Find similar public examples.", "Dig into this more.", "Search for references.",
    "Research the latest on this.", "Find out if anything similar is published.",
    "Do a deep dive.", "Gather more background.", "Look into adjacent topics.",
    "Research and fill the gaps.", "Find supporting references.",
    "Do more background research before continuing.",
    # status / recap
    "Where are we?", "Recap the current state.", "What's left to do?",
    "Status check.", "Review progress so far.", "Summarize what we've done.",
    "Catch me up.", "What's the current situation?", "Give me a status update.",
    "Remind me where we stopped.", "Check the current progress carefully.",
    "What remains?", "Re-orient on the session.", "Figure out where we are again.",
    # document / writing
    "Turn this into a markdown file.", "Write it up as a report.",
    "Format this as a document.", "Make slides for each project.",
    "Organize this into a clean doc.", "Draft a short report first.",
    "Write this as a summary doc.", "Put this into a structured format.",
    "Make a one-pager for each.", "Compile this into a report.",
    "Create a slide deck explaining each project.",
    "Write it in both English and Korean.",
    # correction / iteration
    "Redo this part.", "Try again.", "Revise that.", "Adjust the approach.",
    "One more pass.", "Refine it further.", "Tweak this a bit.",
    "Start over on this section.", "Improve this draft.", "Polish it up.",
    "Iterate on this.", "Make it better.", "Restore this for me.",
    # misc workflow
    "Manage this in the current directory.", "Organize the files here.",
    "Follow the instructions in the project.", "Check the directory and proceed.",
    "Set this up in the project folder.", "Keep this project self-contained.",
    "Look through the local setup and plan.", "Check what's here and continue.",
    "Open the project and get oriented.", "Review the instructions and start.",
    "Let's handle one project at a time.", "Do the research and learning materials too.",
]


def qlen_s(s: str) -> int:
    return len((s or "").strip())


def qlen(r: dict) -> int:
    return len((r.get("query", "") or "").strip())


def is_template(r: dict) -> bool:
    return bool(TEMPLATE_RE.match((r.get("query", "") or "").strip()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=str(TRAIN))
    ap.add_argument("--out", default=str(ROOT / "data" / "aug" / "short_ko_aug_source.jsonl"))
    ap.add_argument("--short-thresh", type=int, default=80, help="query<thresh chars = short")
    ap.add_argument("--n-stubs", type=int, default=60, help="how many protein-stub benign to include")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-bins", type=int, default=8)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    recs = [json.loads(l) for l in open(args.train, encoding="utf-8")]

    # ── Benign: authored conversational + short natural-bio + a stub sample ──────
    authored = []
    seen = set()
    for i, s in enumerate(AUTHORED_CONVERSATIONAL):
        s = s.strip()
        if s.lower() in seen:
            continue
        seen.add(s.lower())
        authored.append({
            "id": f"authConv{i}",
            "binary_label": "legitimate",
            "source": "authored_conversational_register",
            "content_domain": "non_bio_conversational",
            "legitimacy_tier": "conversational_workflow",
            "query": s, "response": "",
            "augmentation": "ko_mt_nllb_conversational", "authored": True,
            "lang": "en", "query_en": s, "response_en": "",
        })

    legit = [r for r in recs if r.get("binary_label") == "legitimate"]
    short_natbio = [r for r in legit if qlen(r) < args.short_thresh and not is_template(r)]
    stubs = [r for r in legit if qlen(r) < args.short_thresh and is_template(r)]
    rng.shuffle(stubs)
    stub_sample = stubs[: args.n_stubs]

    def bio_to_source(r: dict, idx: int, tag: str) -> dict:
        out = dict(r)
        out["orig_id"] = r.get("id")
        out["id"] = f"{r.get('id', 'rec')}_{tag}{idx}"
        out["augmentation"] = "ko_mt_nllb_shortbio"
        out["lang"] = "en"
        out["query_en"] = (r.get("query", "") or "")
        out["response_en"] = (r.get("response", "") or "")
        return out

    short_bio_benign = (
        [bio_to_source(r, i, "natbio") for i, r in enumerate(short_natbio)]
        + [bio_to_source(r, i, "stub") for i, r in enumerate(stub_sample)]
    )
    benign = authored + short_bio_benign

    # ── Harmful: length-match (English query chars) to the benign distribution ───
    benign_lens = sorted(qlen_s(r["query_en"]) for r in benign)
    n_b = len(benign_lens)
    edges = [benign_lens[min(n_b - 1, (i * n_b) // args.n_bins)] for i in range(args.n_bins)]
    edges = sorted(set(edges)) + [float("inf")]

    def bin_of(L: int) -> int:
        for i in range(len(edges) - 1):
            if edges[i] <= L < edges[i + 1]:
                return i
        return len(edges) - 2

    benign_bin_counts = Counter(bin_of(qlen_s(r["query_en"])) for r in benign)

    # Harmful pool: SHORT negatives only (query < short_thresh), so max length matches the
    # benign max (no "long=>harmful" tail) and the form matches (query-only, see below).
    harmful_pool = [r for r in recs
                    if r.get("binary_label") == "negative" and qlen(r) < args.short_thresh]
    harmful_by_bin: dict[int, list] = {}
    for r in harmful_pool:
        harmful_by_bin.setdefault(bin_of(qlen(r)), []).append(r)
    for b, lst in harmful_by_bin.items():
        rng.shuffle(lst)

    harmful_sel, shortfall = [], 0
    used = set()
    for b, want in benign_bin_counts.items():
        avail = [r for r in harmful_by_bin.get(b, []) if id(r) not in used]
        take = avail[:want]
        for r in take:
            used.add(id(r))
        harmful_sel.extend(take)
        if len(take) < want:
            shortfall += want - len(take)
    # Top up shortfall with the SHORTEST remaining harmful (keep KO harmful short here).
    if shortfall:
        remaining = sorted((r for r in harmful_pool if id(r) not in used), key=qlen)
        harmful_sel.extend(remaining[:shortfall])

    def harm_to_source(r: dict, idx: int) -> dict:
        out = dict(r)
        out["orig_id"] = r.get("id")
        out["id"] = f"{r.get('id', 'rec')}_shortharm{idx}"
        out["augmentation"] = "ko_mt_nllb_shortharm"
        out["lang"] = "en"
        out["query_en"] = (r.get("query", "") or "")
        # Query-only, to match the benign form and the (query-only) session probe, and to
        # block a "has-response => harmful" form shortcut. Short harmful is unambiguous from
        # the query alone ("Tell me how to use Lassa Virus.").
        out["response"] = ""
        out["response_en"] = ""
        return out

    harmful = [harm_to_source(r, i) for i, r in enumerate(harmful_sel)]

    src = benign + harmful
    rng.shuffle(src)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in src:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ── Report ──────────────────────────────────────────────────────────────────
    bl = sorted(qlen_s(r["query_en"]) for r in benign)
    hl = sorted(qlen_s(r["query_en"]) for r in harmful)
    pct = lambda xs, p: xs[min(len(xs) - 1, p * len(xs) // 100)] if xs else 0
    print(f"BENIGN  {len(benign)}  = authored_conversational {len(authored)}"
          f" + short_natbio {len(short_natbio)} + stub_sample {len(stub_sample)}")
    print(f"HARMFUL {len(harmful)}  (length-matched; shortfall topped {shortfall})")
    print(f"TOTAL   {len(src)}  -> {args.out}")
    print(f"\nLength match (English query chars)  [similar => no length->label shortcut]")
    print(f"  benign : p25={pct(bl,25)} median={pct(bl,50)} p75={pct(bl,75)} max={bl[-1]}")
    print(f"  harmful: p25={pct(hl,25)} median={pct(hl,50)} p75={pct(hl,75)} max={hl[-1]}")
    print(f"\nHarmful by content_domain (top 8):")
    for k, v in Counter(r.get("content_domain") for r in harmful).most_common(8):
        print(f"  {v:4d}  {k}")
    n_resp = sum(1 for r in src if (r.get("response_en") or "").strip())
    print(f"\nrecords with a response to also translate: {n_resp}/{len(src)}"
          f"  (benign are query-only by design)")


if __name__ == "__main__":
    main()

import os, json, sys, time, hashlib, re
import anthropic

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

SYSTEM = (
    "You are helping build training data for a defensive bio-safety CLASSIFIER "
    "(a guard model that learns to REFUSE harmful biological requests). "
    "Your task: rephrase the given prompt in 4 different styles. Rules: "
    "(1) Preserve the SAME biological agent and general intent category. "
    "(2) Do NOT add new technical details, protocols, dosages, synthesis steps, "
    "or any operationally useful information not already in the original. "
    "(3) Do NOT introduce new biological agents not in the original prompt. "
    "(4) Vary the surface form: academic question, scenario/roleplay, indirect "
    "inquiry, casual/conversational. Each must sound distinctly different. "
    "(5) Output a JSON array of exactly 4 strings. Nothing else."
)

# Load genuine-bio positives (non-saladbench only)
bio_clean = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
pos = [r for r in bio_clean if int(r["label"]) == 1
       and "saladbench" not in r.get("source", "")]
print(f"Paraphrase source: {len(pos)} genuine-bio (non-saladbench)")

results = []
refused = 0
errors = 0
seen = set()

def qhash(t):
    return hashlib.sha1(t.strip().lower().encode()).hexdigest()[:16]

for i, r in enumerate(pos):
    orig = r["query"]
    try:
        resp = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=SYSTEM,
            messages=[{"role": "user",
                       "content": f"Rephrase for classifier training: {orig}"}],
        )
        text = resp.content[0].text.strip()
        # parse JSON array
        if text.startswith("["):
            paras = json.loads(text)
        else:
            # try to extract JSON from markdown code block
            m = re.search(r"\[.*?\]", text, re.DOTALL)
            paras = json.loads(m.group()) if m else []

        for p in paras:
            p = str(p).strip()
            h = qhash(p)
            if len(p) > 15 and h not in seen:
                seen.add(h)
                results.append({
                    "query": p, "response": "", "label": 1,
                    "source": "paraphrase_" + r.get("source", "unk"),
                    "original_hash": qhash(orig),
                })
        if i % 10 == 0:
            print(f"  [{i+1}/{len(pos)}] paraphrases so far: {len(results)}")
    except anthropic.BadRequestError:
        refused += 1
        print(f"  [{i+1}] REFUSED: {orig[:60]}...")
    except Exception as e:
        errors += 1
        print(f"  [{i+1}] ERROR: {e}")
    time.sleep(0.3)  # rate limit courtesy

print("Done. paraphrases=%d refused=%d errors=%d" % (len(results), refused, errors))

outpath = sys.argv[2] if len(sys.argv) > 2 else "/tmp/pdual_paraphrases.jsonl"
with open(outpath, "w") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"Saved: {outpath}")

# diversity stats
from collections import Counter
src = Counter(r["source"] for r in results)
print(f"By source: {dict(src)}")

import re
import json
import time
from src.generate import answer

K = 6
questions = json.load(open("eval/questions.json", encoding="utf-8"))

n_answerable = hits = correct = 0
n_refuse = refusals_ok = 0
results = []

for i, q in enumerate(questions, 1):
    print(f"[{i}/{len(questions)}] {q['question'][:70]}", flush=True)
    try:
        out = answer(q["question"], k=K)
    except Exception as e:
        print(f"ERROR on '{q['question']}': {type(e).__name__}")
        continue
    text = out["answer"].lower()
    text = re.sub(r"\[[^\]]*\]", "", text)   # ignore [file, p.X] citations

    if q.get("refuse"):
        n_refuse += 1
        ok = "find this in the documents" in text
        refusals_ok += ok
        results.append({"question": q["question"], "type": "refuse", "ok": ok,
                        "answer": out["answer"]})
        if not ok:
            print(f"REFUSAL FAILED: {q['question']}")
            print("   Answer:", out["answer"][:200].replace("\n", " "))
    else:
        n_answerable += 1
        hit = any(c["source"] == q["source"] for c in out["sources"])
        ok = all(any(alt in text for alt in kw.lower().split("|")) for kw in q["keywords"])
        hits += hit
        correct += ok
        results.append({
            "question": q["question"], "type": "answer",
            "retrieval_hit": hit, "ok": ok, "answer": out["answer"],
        })
        if not ok:
            print(f"FAILED: {q['question']} (retrieval hit: {hit})")
            print("   Answer:", out["answer"][:300].replace("\n", " "))

    time.sleep(6)   # stay under free-tier rate limits

print("\n--- RESULTS ---")
if n_answerable:
    print(f"Retrieval hit rate: {hits}/{n_answerable} = {hits/n_answerable:.0%}")
    print(f"Answer accuracy:    {correct}/{n_answerable} = {correct/n_answerable:.0%}")
if n_refuse:
    print(f"Correct refusals:   {refusals_ok}/{n_refuse} = {refusals_ok/n_refuse:.0%}")

json.dump(results, open("eval/results.json", "w", encoding="utf-8"), indent=2)
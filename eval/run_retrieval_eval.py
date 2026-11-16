import json
import sys
from src.retrieve import retrieve

K = int(sys.argv[1]) if len(sys.argv) > 1 else 4
questions = json.load(open("eval/questions.json", encoding="utf-8"))

n = paper_hits = answer_hits = 0
for q in questions:
    if q.get("refuse"):
        continue
    n += 1
    chunks = retrieve(q["question"], k=K)
    if any(c["source"] == q["source"] for c in chunks):
        paper_hits += 1
    joined = " ".join(c["text"] for c in chunks).lower()
    if all(any(alt in joined for alt in kw.lower().split("|")) for kw in q["keywords"]):
        answer_hits += 1
    else:
        print("MISSING ANSWER TEXT:", q["question"])

print(f"\nK={K}")
print(f"Right paper retrieved:      {paper_hits}/{n} = {paper_hits/n:.0%}")
print(f"Answer text in top chunks:  {answer_hits}/{n} = {answer_hits/n:.0%}")
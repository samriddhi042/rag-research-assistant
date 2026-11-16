import json
import sys
from src.retrieve import retrieve, collection

questions = json.load(open("eval/questions.json", encoding="utf-8"))
needle = sys.argv[1].lower()
q = next(q for q in questions if needle in q["question"].lower())
print("QUESTION:", q["question"], "| SOURCE:", q.get("source"))

# 1. Does the answer text exist anywhere in that paper's chunks?
data = collection.get(where={"source": q["source"]})
good = [
    d for d in data["documents"]
    if all(any(alt in d.lower() for alt in kw.lower().split("|")) for kw in q["keywords"])
]
print(f"Chunks in paper: {len(data['documents'])} | chunks containing ALL keywords: {len(good)}")
if good:
    print("Example chunk:", good[0][:300].replace("\n", " "))

# 2. Where do the right paper's chunks rank for this question?
res = retrieve(q["question"], k=50, fetch=50, rerank=False)
ranks = [i + 1 for i, c in enumerate(res) if c["source"] == q["source"]]
print("Ranks of chunks from the right paper (out of 50):", ranks[:10])
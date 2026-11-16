from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb

model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection("papers")


def retrieve(question, k=6, fetch=50, rerank=True):
    q_emb = model.encode([question], normalize_embeddings=True).tolist()
    n = fetch if rerank else k
    res = collection.query(query_embeddings=q_emb, n_results=n)
    results = [
        {"text": doc, "source": meta["source"], "page": meta["page"], "distance": dist}
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        )
    ]
    if rerank:
        scores = reranker.predict([(question, r["text"]) for r in results])
        for r, s in zip(results, scores):
            r["rerank_score"] = float(s)
        results.sort(key=lambda r: r["rerank_score"], reverse=True)
    return results[:k]


if __name__ == "__main__":
    question = input("Ask a question: ")
    for r in retrieve(question):
        print(f"\n{r['source']} (page {r['page']})")
        print(r["text"][:300].replace("\n", " "))
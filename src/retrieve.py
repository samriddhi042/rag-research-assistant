from sentence_transformers import SentenceTransformer
import chromadb

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection("papers")


def retrieve(question, k=4):
    q_emb = model.encode([question], normalize_embeddings=True).tolist()
    res = collection.query(query_embeddings=q_emb, n_results=k)
    return [
        {"text": doc, "source": meta["source"], "page": meta["page"], "distance": dist}
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        )
    ]


if __name__ == "__main__":
    question = input("Ask a question: ")
    for r in retrieve(question):
        print(f"\n{r['source']} (page {r['page']}) | distance: {r['distance']:.3f}")
        print(r["text"][:300].replace("\n", " "))
from pathlib import Path
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb

DATA_DIR = Path("data")
DB_DIR = "chroma_db"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def load_pdfs(data_dir):
    pages = []
    for pdf_path in sorted(data_dir.glob("*.pdf")):
        reader = PdfReader(pdf_path)
        for i, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append({"text": text, "source": pdf_path.name, "page": i + 1})
    return pages


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


def build_chunks(pages):
    chunks = []
    for p in pages:
        for j, piece in enumerate(chunk_text(p["text"])):
            chunks.append({
                "id": f'{p["source"]}-p{p["page"]}-c{j}',
                "text": piece,
                "source": p["source"],
                "page": p["page"],
            })
    return chunks


def main():
    pages = load_pdfs(DATA_DIR)
    chunks = build_chunks(pages)
    print(f"Loaded {len(pages)} pages -> {len(chunks)} chunks")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(
        [c["text"] for c in chunks],
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    client = chromadb.PersistentClient(path=DB_DIR)
    try:
        client.delete_collection("papers")
    except Exception:
        pass
    col = client.create_collection("papers", metadata={"hnsw:space": "cosine"})

    for i in range(0, len(chunks), 500):
        batch = chunks[i:i + 500]
        col.add(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            embeddings=embeddings[i:i + 500].tolist(),
            metadatas=[{"source": c["source"], "page": c["page"]} for c in batch],
        )
    print("Index built.")


if __name__ == "__main__":
    main()
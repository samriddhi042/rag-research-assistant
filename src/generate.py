import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from src.retrieve import retrieve

load_dotenv()
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options=types.HttpOptions(timeout=30000),   # give up on a call after 60 seconds
)

# Tried in order. If one is busy or unavailable, the next is used.
MODELS = ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite"]

PROMPT = """You are a research assistant. Answer the question using ONLY the context below.
If the answer is not in the context, reply exactly: "I couldn't find this in the documents."
Cite sources in the form [file, p.X].

Context:
{context}

Question: {question}
Answer:"""


def call_llm(prompt):
    for model in MODELS:
        for attempt in range(2):
            t = time.time()
            try:
                resp = client.models.generate_content(model=model, contents=prompt)
                print(f"[llm] {model} ok in {time.time()-t:.1f}s", flush=True)
                return resp.text
            except errors.ServerError:
                print(f"[llm] {model} busy (attempt {attempt+1})", flush=True)
                time.sleep(2 * (attempt + 1))
            except errors.ClientError as e:
                print(f"[llm] {model} client error {e.code}", flush=True)
                if e.code in (404, 429):
                    break
                raise
            except Exception as e:
                print(f"[llm] {model} {type(e).__name__} after {time.time()-t:.1f}s (attempt {attempt+1})", flush=True)
                time.sleep(3 * (attempt + 1))
    return "The AI service is busy right now. Please try again in a minute."


def answer(question, k=6):
    t0 = time.time()
    chunks = retrieve(question, k)
    print(f"[timing] retrieval took {time.time()-t0:.1f}s", flush=True)
    context = "\n\n".join(
        f"[{c['source']}, p.{c['page']}]\n{c['text']}" for c in chunks
    )
    text = call_llm(PROMPT.format(context=context, question=question))
    return {"answer": text, "sources": chunks}


if __name__ == "__main__":
    while True:
        q = input("\nQuestion (or 'quit'): ")
        if q.lower() == "quit":
            break
        out = answer(q)
        print("\n" + out["answer"])
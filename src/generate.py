import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import errors
from src.retrieve import retrieve

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

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
        for attempt in range(3):
            try:
                resp = client.models.generate_content(model=model, contents=prompt)
                return resp.text
            except errors.ServerError:
                # 5xx: Google is busy, wait a little and retry
                time.sleep(2 * (attempt + 1))
            except errors.ClientError as e:
                # 404 = model not available, 429 = rate limit: try the next model
                if e.code in (404, 429):
                    break
                raise
    return "The AI service is busy right now. Please try again in a minute."


def answer(question, k=4):
    chunks = retrieve(question, k)
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
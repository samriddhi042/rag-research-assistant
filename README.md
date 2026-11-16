# 📚 Research Paper Assistant (RAG)

A retrieval-augmented question-answering app over 10 landmark LLM and RAG papers. Ask a question in plain English and get an answer **grounded only in the papers**, with citations (file and page) and the exact source passages. If the answer is not in the papers, it says so instead of guessing.

**🔗 Live demo: https://llm-papers-assistant.streamlit.app/**

> The demo runs on free hosting. If it has been idle, it may show a "wake this app up" button; click it and wait about a minute. The first question after waking is the slowest.

<!-- Add a screenshot at docs/demo.png, then remove this line and the arrows around the next line:
![Demo screenshot](docs/demo.png)
-->

## What it does

- Answers questions such as *"How does LoRA reduce the number of trainable parameters?"* or *"What similarity measure does Sentence-BERT use?"*
- Cites the source file and page for every claim, and shows the retrieved passages
- Refuses out-of-scope questions (for example, "Who won the cricket world cup?") with *"I couldn't find this in the documents."*

## How it works

```
Ingestion (run once)
  PDFs -> extract text per page -> clean broken words -> split into chunks
       -> add paper title to each chunk -> embed -> store in ChromaDB

Every question
  question -> embed -> fetch 50 closest chunks (vector search)
           -> rerank with a cross-encoder -> keep top 6
           -> prompt: "answer ONLY from this context" -> Gemini -> answer + sources
```

| Part | Choice |
|---|---|
| Chunking | 800 characters, 100 overlap, page number kept as metadata |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers, runs locally) |
| Vector store | ChromaDB (cosine similarity) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | Gemini Flash family via the Google Gen AI SDK, with automatic fallback between models |
| UI and hosting | Streamlit, deployed on Streamlit Community Cloud |

**Corpus (10 papers):** Attention Is All You Need, BERT, Sentence-BERT, Dense Passage Retrieval, Retrieval-Augmented Generation, Self-RAG, Lost in the Middle, LoRA, Chain-of-Thought Prompting, ReAct (about 224 pages).

## Evaluation

I built a small test set to measure the system instead of judging it by eye:

- **20 answerable questions** (2 per paper), each with the source paper and required keywords
- **4 out-of-scope questions** that should be refused

Retrieval was tested separately from generation. "Answer text in chunks" checks whether the retrieved passages contain the answer keywords, without calling the LLM.

### Retrieval experiments (20 questions)

| Experiment | Right paper retrieved | Answer text in chunks |
|---|---|---|
| Baseline: top 4, no reranker | 90% | 50% |
| Top 8, no reranker | 90% | 75% |
| Top 12, no reranker | 90% | 75% |
| Reranker over 20 candidates, keep 4 | 90% | 75% |
| Reranker over 20 candidates, keep 12 | 90% | 80% |
| Reranker over 20 candidates, keep 6 | 90% | 80% |
| **Reranker over 50 candidates, keep 6 (final)** | **90%** | **100%** |

### End-to-end results (final configuration, 24 questions)

| Metric | Result |
|---|---|
| Right paper in retrieved chunks | 18/20 = 90% |
| Answer accuracy (keyword check on the final answer) | 20/20 = 100% |
| Correct refusals on out-of-scope questions | 4/4 = 100% |

Per-question answers are saved in [`eval/results.json`](eval/results.json).

### What I learned

1. **Retrieval was the bottleneck, not the LLM.** Every failure was the model correctly saying it couldn't find the answer in the text it was given. Fixing search fixed accuracy.
2. **Prefixing each chunk with its paper title improved paper identification** (right paper retrieved: 50% -> 75% on my first 8-question set), because most chunks never mention the paper's name.
3. **A reranker looked useless at first and then gave the biggest gain.** On the first 8 questions it made no difference, which was noise from a tiny test set. On 20 questions, letting it rerank 50 candidates raised answer coverage from 80% to 100%.
4. **Retrieving more chunks is not the same as better retrieval.** Going from 4 to 12 chunks without a reranker helped less than reranking.
5. **A strict "answer only from the context" prompt gives reliable refusals** (4/4 on out-of-scope questions).

### Engineering details

- Retries, timeouts, and automatic fallback across Gemini models when one is busy or rate-limited
- Usage limits in the app (10 questions per session, 100 per day) to stay inside the free API quota
- PDF text cleaning for words split across lines

## Limitations

- **The 100% is on my development set.** I tuned retrieval settings on the same 20 questions, so it overstates performance on unseen questions. A fresh held-out set is the right next test.
- **Accuracy is measured by keyword matching**, a quick proxy that can mark a correct answer wrong or a wrong answer right.
- **Small corpus (10 papers) and small test set.** One question is worth 5 points.
- **Broad questions get narrow answers.** A vague question like "What are LLMs used for?" is answered from whichever single paper matches best, not across all papers.
- **Answers come from a mix of Gemini Flash models**, because the app falls back between them under free-tier rate limits, so the results are not tied to one specific model.
- Free hosting means cold starts and daily limits.

## Run it locally

```bash
git clone https://github.com/samriddhi042/rag-research-assistant.git
cd rag-research-assistant

python -m venv venv
venv\Scripts\activate          # Windows  (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
```

Create a `.env` file with a free API key from Google AI Studio:

```
GEMINI_API_KEY=your_key_here
```

Build the index, then start the app:

```bash
python -m src.ingest
streamlit run app.py
```

### Reproduce the evaluation

```bash
python -m eval.run_retrieval_eval 6    # retrieval only, no API calls
python -m eval.run_eval                # full pipeline including the LLM
```

## Project structure

```
data/                     source PDFs
src/ingest.py             load, clean, chunk, embed, store
src/retrieve.py           vector search + reranking
src/generate.py           grounded prompt, Gemini call, retries and fallback
app.py                    Streamlit UI with usage limits
eval/questions.json       evaluation questions
eval/run_retrieval_eval.py  retrieval-only evaluation
eval/run_eval.py          end-to-end evaluation
eval/debug_question.py    inspect why a question fails
```

## Future improvements

- Evaluate on a held-out question set, and judge answers with an LLM or human review instead of keywords
- Hybrid search (keyword + vector) and query rewriting
- Retrieval that spreads results across papers for broad questions
- Let users upload their own PDFs

## Author

Built by [@samriddhi042](https://github.com/samriddhi042).

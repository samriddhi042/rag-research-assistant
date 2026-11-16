import os
import time
import streamlit as st

st.set_page_config(page_title="Research Paper Assistant", page_icon="📚")

MAX_PER_SESSION = 10   # questions one visitor can ask
MAX_PER_DAY = 100      # questions across all visitors, to protect the free API quota

# Build the search index on first launch (needed when deployed)
if not os.path.exists("chroma_db"):
    from src.ingest import main as build_index
    with st.spinner("Building the search index (first launch only, takes a few minutes)..."):
        build_index()

from src.generate import answer


@st.cache_resource
def usage_counter():
    return {"day": time.strftime("%Y-%m-%d"), "count": 0}


def daily_limit_ok():
    usage = usage_counter()
    today = time.strftime("%Y-%m-%d")
    if usage["day"] != today:
        usage["day"], usage["count"] = today, 0
    return usage["count"] < MAX_PER_DAY


st.session_state.setdefault("n_asked", 0)

st.title("📚 Research Paper Assistant")
st.caption(
    "Ask questions about 10 landmark LLM and RAG papers (Transformer, BERT, RAG, LoRA, "
    "Self-RAG, ReAct and more). Answers come only from these papers, with sources."
)
st.caption("Tip: specific questions work best, for example naming the paper or method.")

question = st.text_input(
    "Your question", placeholder="e.g. How does LoRA reduce trainable parameters?"
)

if st.button("Ask") and question.strip():
    if st.session_state.n_asked >= MAX_PER_SESSION:
        st.warning(
            f"You've reached the limit of {MAX_PER_SESSION} questions for this session. "
            "Thanks for trying the demo!"
        )
    elif not daily_limit_ok():
        st.warning(
            "The demo has reached its daily limit (to stay within free usage). "
            "Please try again tomorrow."
        )
    else:
        st.session_state.n_asked += 1
        usage_counter()["count"] += 1

        with st.spinner("Searching the papers..."):
            result = answer(question.strip())

        st.subheader("Answer")
        st.write(result["answer"])

        if "find this in the documents" not in result["answer"].lower():
            st.subheader("Sources")
            for c in result["sources"]:
                with st.expander(f"{c['source']} (page {c['page']})"):
                    st.write(c["text"])
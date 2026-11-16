import os
import streamlit as st

st.set_page_config(page_title="Research Paper Assistant", page_icon="📚")

# Build the search index on first launch (needed when deployed)
if not os.path.exists("chroma_db"):
    from src.ingest import main as build_index
    with st.spinner("Building the search index (first launch only, takes a few minutes)..."):
        build_index()

from src.generate import answer

st.title("📚 Research Paper Assistant")
st.caption(
    "Ask questions about 10 landmark LLM and RAG papers (Transformer, BERT, RAG, LoRA, "
    "Self-RAG, ReAct and more). Answers come only from these papers, with sources."
)

question = st.text_input("Your question", placeholder="e.g. How does LoRA reduce trainable parameters?")

if st.button("Ask") and question.strip():
    with st.spinner("Searching the papers..."):
        result = answer(question.strip())

    st.subheader("Answer")
    st.write(result["answer"])

    if "find this in the documents" not in result["answer"].lower():
        st.subheader("Sources")
        for c in result["sources"]:
            with st.expander(f"{c['source']} (page {c['page']})"):
                st.write(c["text"])
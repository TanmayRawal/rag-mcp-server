import streamlit as st
from rag import query_rag_with_sources
import os

st.set_page_config(
    page_title="RAG Knowledge Assistant",
    page_icon="📚",
    layout="wide"
)

# ---------- Session state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- Sidebar ----------
with st.sidebar:
    st.title("⚙️ Settings")

    top_k = st.slider(
        "Number of retrieved chunks",
        min_value=1,
        max_value=10,
        value=5
    )

    show_sources = st.checkbox("Show retrieved chunks", value=True)

    st.markdown("---")

    # -------- PDF Upload --------
    st.subheader("Upload PDF")

    uploaded_file = st.file_uploader(
        "Add new document",
        type=["pdf"]
    )

    if uploaded_file is not None:
        save_path = os.path.join("documents", uploaded_file.name)

        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success("PDF uploaded successfully!")
        st.info("Run ingestion again to index the new document.")

    st.markdown("---")

    st.markdown("### About")
    st.write("This app queries your document knowledge base using RAG.")
    st.write("Built with FAISS, Sentence Transformers, Groq, and Streamlit.")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ---------- Main UI ----------
st.title("📚 RAG Knowledge Assistant")
st.caption("Ask questions from your indexed PDFs")

# ---------- Display previous messages ----------
for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and "sources" in msg and show_sources:

            with st.expander("Retrieved Chunks / Sources"):

                for i, chunk in enumerate(msg["sources"], start=1):

                    source_name = chunk.get("source", "Unknown Source")
                    text = chunk.get("text", "")

                    st.markdown(f"**Chunk {i} | Source:** `{source_name}`")
                    st.write(text)
                    st.markdown("---")

# ---------- Chat input ----------
user_query = st.chat_input("Ask a question about your documents...")

if user_query:

    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_query
    })

    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate response
    with st.chat_message("assistant"):

        with st.spinner("Searching documents and generating answer..."):

            try:

                answer, sources = query_rag_with_sources(
                    user_query,
                    top_k=top_k
                )

                st.markdown(answer)

                if show_sources:

                    with st.expander("Retrieved Chunks / Sources"):

                        for i, chunk in enumerate(sources, start=1):

                            source_name = chunk.get("source", "Unknown Source")
                            text = chunk.get("text", "")

                            st.markdown(f"**Chunk {i} | Source:** `{source_name}`")
                            st.write(text)
                            st.markdown("---")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as e:

                error_msg = f"Error: {str(e)}"
                st.error(error_msg)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })
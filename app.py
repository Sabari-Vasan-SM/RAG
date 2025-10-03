import os
import re
import tempfile
from typing import List, Dict, Any, Optional

import streamlit as st
from PyPDF2 import PdfReader
import faiss
import numpy as np

from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer

# ---------- Utilities ----------

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes using PyPDF2."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp.flush()
        tmp_path = tmp.name

    reader = PdfReader(tmp_path)
    os.unlink(tmp_path)

    texts = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            texts.append(page_text)
    return "\n".join(texts)

def preprocess_text(text: str) -> str:
    """Normalize spaces and remove extra newlines."""
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def guess_name(text: str) -> Optional[str]:
    """Extract candidate name from first few lines of text."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for l in lines[:5]:
        if "@" not in l and len(l.split()) <= 4:
            return l
    return "Unknown"

# ---------- Embeddings ----------

class LocalEmbeddings:
    def __init__(self):
        self.model = SentenceTransformer("all-mpnet-base-v2")  # stronger semantic embeddings
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()
    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text])[0].tolist()

# ---------- PDF ingestion ----------

def create_docs_from_pdfs(files: List[Dict[str, Any]]) -> List[Document]:
    """Convert uploaded PDFs to LangChain Documents with metadata."""
    docs = []
    splitter = CharacterTextSplitter(chunk_size=1200, chunk_overlap=250)  # bigger chunks
    for f in files:
        text = extract_text_from_pdf_bytes(f["bytes"])
        text = preprocess_text(text)
        name = guess_name(text)  # use original candidate name detection
        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            docs.append(Document(page_content=chunk, metadata={
                "candidate_name": name,
                "source_file": f["name"],
                "chunk_index": i
            }))
    return docs

def build_index(docs: List[Document], embedder: LocalEmbeddings):
    """Build FAISS index for document chunks."""
    texts = [d.page_content for d in docs]
    metadatas = [d.metadata for d in docs]
    embeddings = np.array(embedder.embed_documents(texts)).astype("float32")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    vs = {"index": index, "texts": texts, "metadatas": metadatas, "dim": dim}
    return vs

def query_index(vs, embedder: LocalEmbeddings, query: str, top_k=5):
    """Query FAISS index and return top-k chunks."""
    q_emb = np.array(embedder.embed_query(query)).astype("float32")
    D, I = vs["index"].search(np.array([q_emb]), top_k)
    results = []
    for i in I[0]:
        results.append({"text": vs["texts"][i], "metadata": vs["metadatas"][i]})
    return results

# ---------- Streamlit UI ----------

st.set_page_config(page_title="RAG Resume Search", layout="wide")
st.title("🔎 RAG Resume Search — Hybrid Keyword + Embedding")

# Session state
if "vs" not in st.session_state:
    st.session_state.vs = None
if "embedder" not in st.session_state:
    st.session_state.embedder = LocalEmbeddings()
if "docs" not in st.session_state:
    st.session_state.docs = []

# --- PDF Upload ---
st.markdown("## 1️⃣ Upload PDF resumes")
uploaded = st.file_uploader("Upload one or more PDF resumes", type="pdf", accept_multiple_files=True)

if uploaded:
    if st.button("Ingest PDFs"):
        files = [{"name": u.name, "bytes": u.getvalue()} for u in uploaded]
        docs = create_docs_from_pdfs(files)
        st.session_state.docs = docs
        st.session_state.vs = build_index(docs, st.session_state.embedder)
        st.success(f"Ingested {len(docs)} chunks from {len(files)} resumes.")

# Show candidate table
if st.session_state.docs:
    st.markdown("### Indexed candidates")
    candidates = {}
    for d in st.session_state.docs:
        name = d.metadata.get("candidate_name", "Unknown")
        candidates[name] = candidates.get(name, 0) + 1
    st.dataframe({"Candidate Name": list(candidates.keys()), "Chunks": list(candidates.values())})

st.markdown("---")
st.markdown("## 2️⃣ Search resumes")

query = st.text_input("Enter query (e.g., React, AWS, John)")
candidate_hint = st.text_input("Optional: Candidate name filter")
top_k = st.number_input("Top-k chunks to retrieve", min_value=1, max_value=10, value=5)

if st.button("Search"):
    if not st.session_state.vs:
        st.warning("Index is empty. Upload PDFs first.")
    elif not query.strip():
        st.warning("Enter a query first.")
    else:
        vs = st.session_state.vs
        embedder = st.session_state.embedder

        keywords = [kw.lower() for kw in query.split()]

        # Candidate filtering
        filtered_indices = list(range(len(vs["texts"])))
        if candidate_hint:
            filtered_indices = [i for i, md in enumerate(vs["metadatas"]) 
                                if candidate_hint.lower() in md.get("candidate_name", "").lower()]

        # Keyword filtering: only keep chunks containing at least one keyword
        keyword_filtered_indices = []
        for i in filtered_indices:
            text_lower = vs["texts"][i].lower()
            if any(kw in text_lower for kw in keywords):
                keyword_filtered_indices.append(i)

        if not keyword_filtered_indices:
            st.info("No candidate or chunk contains the keyword. Searching all filtered chunks as fallback.")
            keyword_filtered_indices = filtered_indices  # fallback

        # Build temporary FAISS index for filtered docs
        dim = vs["dim"]
        temp_index = faiss.IndexFlatL2(dim)
        embeddings = np.array([vs["index"].reconstruct(i) for i in keyword_filtered_indices]).astype("float32")
        temp_index.add(embeddings)
        temp_vs = {
            "index": temp_index,
            "texts": [vs["texts"][i] for i in keyword_filtered_indices],
            "metadatas": [vs["metadatas"][i] for i in keyword_filtered_indices],
            "dim": dim
        }

        results = query_index(temp_vs, embedder, query, top_k=top_k)

        st.markdown("### Top Results")
        for r in results:
            md = r["metadata"]
            st.markdown(f"**Candidate: {md['candidate_name']} — Source: {md['source_file']}**")
            st.write(r["text"][:500] + ("..." if len(r["text"])>500 else ""))

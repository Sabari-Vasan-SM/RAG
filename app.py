import os
import re
import tempfile
from typing import List, Dict, Any, Optional

import streamlit as st
from PyPDF2 import PdfReader
import faiss
import numpy as np
import pickle

from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer

# ---------- Utilities ----------

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    import tempfile
    from PyPDF2 import PdfReader
    import os

    # Write bytes to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    # Read PDF from temp file
    reader = PdfReader(tmp_path)
    os.unlink(tmp_path)  # delete temp file after reading

    texts = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            texts.append(page_text)
    return "\n".join(texts)


def guess_name(text: str) -> Optional[str]:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for l in lines[:5]:
        if "@" not in l and len(l.split()) <= 4:
            return l
    return "Unknown"

# ---------- Embeddings ----------

class LocalEmbeddings:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()
    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text])[0].tolist()

# ---------- Ingest PDFs ----------

def create_docs_from_pdfs(files: List[Dict[str, Any]]) -> List[Document]:
    docs = []
    splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    for f in files:
        text = extract_text_from_pdf_bytes(f["bytes"])
        name = guess_name(text)
        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            docs.append(Document(page_content=chunk, metadata={"candidate_name": name, "source_file": f["name"], "chunk_index": i}))
    return docs

def build_index(docs: List[Document], embedder: LocalEmbeddings, index_path: Optional[str]=None):
    texts = [d.page_content for d in docs]
    metadatas = [d.metadata for d in docs]
    embeddings = np.array(embedder.embed_documents(texts)).astype("float32")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    vs = {"index": index, "texts": texts, "metadatas": metadatas, "dim": dim}
    if index_path:
        with open(index_path, "wb") as f:
            pickle.dump(vs, f)
    return vs

def query_index(vs, embedder: LocalEmbeddings, query: str, top_k=5):
    q_emb = np.array(embedder.embed_query(query)).astype("float32")
    D, I = vs["index"].search(np.array([q_emb]), top_k)
    results = []
    for i in I[0]:
        results.append({"text": vs["texts"][i], "metadata": vs["metadatas"][i]})
    return results

# ---------- Streamlit UI ----------

st.title("RAG Resume Search (No LLM)")

uploaded = st.file_uploader("Upload PDF resumes", type="pdf", accept_multiple_files=True)
if "vs" not in st.session_state:
    st.session_state.vs = None
if "embedder" not in st.session_state:
    st.session_state.embedder = LocalEmbeddings()
if "docs" not in st.session_state:
    st.session_state.docs = []

if uploaded:
    if st.button("Ingest PDFs"):
        files = [{"name": u.name, "bytes": u.getvalue()} for u in uploaded]
        docs = create_docs_from_pdfs(files)
        st.session_state.docs = docs
        vs = build_index(docs, st.session_state.embedder)
        st.session_state.vs = vs
        st.success(f"Ingested {len(docs)} chunks from {len(files)} resumes.")

if st.session_state.docs:
    st.markdown("### Indexed candidates")
    candidates = {}
    for d in st.session_state.docs:
        name = d.metadata.get("candidate_name", "Unknown")
        candidates[name] = candidates.get(name, 0) + 1
    st.dataframe({"Candidate Name": list(candidates.keys()), "Chunks": list(candidates.values())})

st.markdown("---")
query = st.text_input("Enter query (e.g., React, AWS, John)")
candidate_hint = st.text_input("Optional: Candidate name filter")

if st.button("Search"):
    if not st.session_state.vs:
        st.warning("Index is empty. Upload PDFs first.")
    else:
        vs = st.session_state.vs
        embedder = st.session_state.embedder

        # candidate filter
        filtered_indices = list(range(len(vs["texts"])))
        if candidate_hint:
            filtered_indices = [i for i, md in enumerate(vs["metadatas"]) if candidate_hint.lower() in md.get("candidate_name", "").lower()]
        if not filtered_indices:
            st.info("No candidate matched filter; searching all resumes.")
            filtered_indices = list(range(len(vs["texts"])))

        # build temp index
        temp_index = faiss.IndexFlatL2(vs["dim"])
        embeddings = np.array([vs["index"].reconstruct(i) for i in filtered_indices]).astype("float32")
        temp_index.add(embeddings)
        temp_vs = {"index": temp_index, "texts": [vs["texts"][i] for i in filtered_indices], "metadatas": [vs["metadatas"][i] for i in filtered_indices], "dim": vs["dim"]}

        results = query_index(temp_vs, embedder, query, top_k=5)
        st.markdown("### Top Results")
        for r in results:
            md = r["metadata"]
            st.markdown(f"**Candidate: {md['candidate_name']} — Source: {md['source_file']}**")
            st.write(r["text"][:500] + ("..." if len(r["text"])>500 else ""))

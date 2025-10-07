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
    # compute embeddings and normalize to unit length so inner-product ~= cosine similarity
    embeddings = np.array(embedder.embed_documents(texts)).astype("float32")
    # handle empty case
    if embeddings.size == 0:
        dim = 0
        index = faiss.IndexFlatIP(0)
        vs = {"index": index, "texts": texts, "metadatas": metadatas, "dim": dim, "embeddings": embeddings}
        return vs

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms
    dim = embeddings.shape[1]

    # use inner-product index (works with normalized vectors as cosine similarity)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # store embeddings array so we can slice/filter without relying on index.reconstruct
    vs = {"index": index, "texts": texts, "metadatas": metadatas, "dim": dim, "embeddings": embeddings}
    return vs

def query_index(vs, embedder: LocalEmbeddings, query: str, top_k=5):
    """Query FAISS index and return top-k chunks."""
    # return embedding similarity scores as well
    q_emb = np.array(embedder.embed_query(query)).astype("float32")
    # normalize query embedding to unit length to match index
    q_norm = np.linalg.norm(q_emb)
    if q_norm == 0:
        q_norm = 1.0
    q_emb = (q_emb / q_norm).astype("float32")

    D, I = vs["index"].search(np.array([q_emb]), top_k)
    results = []
    for score, i in zip(D[0], I[0]):
        results.append({"text": vs["texts"][i], "metadata": vs["metadatas"][i], "score": float(score)})
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

        # Keyword filtering: mark chunks containing keywords and compute a candidate set
        keyword_filtered_indices = []
        keyword_match_scores = {}
        for i in filtered_indices:
            text = vs["texts"][i]
            text_lower = text.lower()
            matches = sum(1 for kw in keywords if kw in text_lower)
            # fraction of keywords present
            frac = matches / max(1, len(keywords))
            if matches > 0:
                keyword_filtered_indices.append(i)
            keyword_match_scores[i] = frac

        if not keyword_filtered_indices:
            st.info("No candidate or chunk contains the keyword. Searching all filtered chunks as fallback.")
            keyword_filtered_indices = filtered_indices  # fallback

        # Build temporary FAISS inner-product index using stored normalized embeddings
        dim = vs["dim"]
        temp_index = faiss.IndexFlatIP(dim)
        embeddings = vs.get("embeddings")
        if embeddings is None or embeddings.size == 0:
            st.error("Embeddings not available for search. Re-ingest PDFs.")
            st.stop()

        # select embeddings for the filtered indices
        sel_embs = embeddings[keyword_filtered_indices].astype("float32")
        temp_index.add(sel_embs)
        temp_vs = {
            "index": temp_index,
            "texts": [vs["texts"][i] for i in keyword_filtered_indices],
            "metadatas": [vs["metadatas"][i] for i in keyword_filtered_indices],
            "dim": dim,
            "orig_indices": keyword_filtered_indices,
            "keyword_match_scores": keyword_match_scores
        }

        raw_results = query_index(temp_vs, embedder, query, top_k=top_k)

        # Combine embedding score (raw_results 'score' is inner-product because we used IndexFlatIP)
        # with keyword match fraction to create a hybrid score. Weight can be tuned.
        hybrid_results = []
        for r in raw_results:
            # locate original index
            # find position in temp_vs texts list
            try:
                pos = temp_vs["texts"].index(r["text"])
            except ValueError:
                pos = None
            emb_score = r.get("score", 0.0)
            orig_idx = temp_vs.get("orig_indices")[pos] if pos is not None else None
            kw_frac = temp_vs.get("keyword_match_scores", {}).get(orig_idx, 0.0)
            hybrid = 0.75 * emb_score + 0.25 * kw_frac
            hybrid_results.append({"text": r["text"], "metadata": r["metadata"], "emb_score": emb_score, "kw_frac": kw_frac, "hybrid": hybrid})

        # sort by hybrid score descending
        hybrid_results.sort(key=lambda x: x["hybrid"], reverse=True)

        results = hybrid_results

        st.markdown("### Top Results")
        for r in results:
            md = r["metadata"]
            st.markdown(f"**Candidate: {md['candidate_name']} — Source: {md['source_file']}**")
            score_line = f"Embedding: {r.get('emb_score', 0):.3f} — Keywords: {r.get('kw_frac', 0):.2f} — Hybrid: {r.get('hybrid',0):.3f}"
            st.caption(score_line)

            # highlight keywords in the text snippet
            snippet = r["text"]
            display_snippet = snippet[:1000]
            # simple highlight by wrapping keyword occurrences in **
            for kw in sorted(keywords, key=len, reverse=True):
                if kw.strip():
                    display_snippet = re.sub(f"(?i)({re.escape(kw)})", r"**\1**", display_snippet)
            st.write(display_snippet + ("..." if len(snippet) > len(display_snippet) else ""))

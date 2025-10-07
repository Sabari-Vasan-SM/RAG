import os
import re
import tempfile
from typing import List, Dict, Any, Optional

import streamlit as st
from PyPDF2 import PdfReader
import faiss
import numpy as np
from math import log

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
    """Try to extract a candidate name from the text.

    Strategies (in order):
    - Look for lines like 'Name: John Doe' or 'Candidate: John Doe'
    - Use the first short line (<=4 words) without emails
    - Look for 1-3 consecutive capitalized words (John A. Doe)
    - Return None if not found (caller may fallback to filename)
    """
    if not text:
        return None
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    # pattern-based extraction
    for l in lines[:12]:
        m = re.search(r"(?:name|candidate|applicant)[:\-\s]+([A-Z][A-Za-z\.'`\- ]{1,80})", l, flags=re.I)
        if m:
            cand = m.group(1).strip()
            # reject if contains email-like or too long
            if '@' in cand or len(cand.split()) > 6:
                continue
            return cand

    # first short line without email
    for l in lines[:8]:
        if "@" not in l and 1 <= len(l.split()) <= 4:
            return l

    # search for capitalized name-like sequences in the first 50 chars
    text_front = " ".join(lines[:6])
    cap_match = re.search(r"([A-Z][a-z]+(?:\s+[A-Z][a-z\-\.]+){0,3})", text_front)
    if cap_match:
        return cap_match.group(1)

    return None

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
        name = guess_name(text)
        if not name:
            # fallback to file stem
            name = os.path.splitext(f.get("name", "unknown"))[0]
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
    # compute simple token-based document frequency (for IDF keyword weighting)
    def _tokenize(t: str):
        return set(re.findall(r"\w+", t.lower()))

    N = len(texts)
    df = {}
    doc_tokens = []
    for t in texts:
        toks = _tokenize(t)
        doc_tokens.append(toks)
        for tok in toks:
            df[tok] = df.get(tok, 0) + 1

    # smooth idf
    idf = {tok: log((1 + N) / (1 + cnt)) + 1.0 for tok, cnt in df.items()}

    vs = {"index": index, "texts": texts, "metadatas": metadatas, "dim": dim, "embeddings": embeddings, "idf": idf, "doc_tokens": doc_tokens}
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
        # FAISS may return -1 or padded indices when fewer than top_k entries exist
        if i is None or int(i) < 0:
            continue
        i = int(i)
        if i >= len(vs.get("texts", [])):
            # out-of-range index for the provided texts slice, skip
            continue
        results.append({"text": vs["texts"][i], "metadata": vs["metadatas"][i], "score": float(score), "idx": i})
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
use_reranker = st.checkbox("Use Cross-Encoder reranker for top results (slower)", value=False)

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

        # Keyword filtering and IDF-weighted match scoring
        keyword_filtered_indices = []
        keyword_match_scores = {}
        # tokenize query and compute IDF-weighted relevance
        query_tokens = re.findall(r"\w+", query.lower())
        q_idf_sum = 0.0
        for t in query_tokens:
            q_idf_sum += vs.get("idf", {}).get(t, 1.0)
        q_idf_sum = max(q_idf_sum, 1.0)

        for i in filtered_indices:
            toks = vs.get("doc_tokens")[i]
            matched_idf = 0.0
            for qt in query_tokens:
                if qt in toks:
                    matched_idf += vs.get("idf", {}).get(qt, 1.0)
            if matched_idf > 0:
                keyword_filtered_indices.append(i)
            # normalized IDF fraction
            keyword_match_scores[i] = matched_idf / q_idf_sum

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

        # Combine embedding score (inner-product) and IDF-weighted keyword fraction.
        # Inner-product with normalized vectors is in [-1,1]; map to [0,1].
        def _norm_emb(s):
            return max(0.0, min(1.0, (s + 1.0) / 2.0))

        hybrid_results = []
        for r in raw_results:
            try:
                pos = temp_vs["texts"].index(r["text"])
            except ValueError:
                pos = None
            emb_score_raw = r.get("score", 0.0)
            emb_score = _norm_emb(emb_score_raw)
            orig_idx = temp_vs.get("orig_indices")[pos] if pos is not None else None
            kw_frac = temp_vs.get("keyword_match_scores", {}).get(orig_idx, 0.0)
            hybrid = 0.7 * emb_score + 0.3 * kw_frac
            hybrid_results.append({"text": r["text"], "metadata": r["metadata"], "emb_score": emb_score, "kw_frac": kw_frac, "hybrid": hybrid})

        # sort by hybrid score descending
        hybrid_results.sort(key=lambda x: x["hybrid"], reverse=True)

        # Optional Cross-Encoder reranker for top results
        if use_reranker and len(hybrid_results) > 0:
            try:
                from sentence_transformers import CrossEncoder
                rerank_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
                # build pairs: (query, text)
                pairs = [(query, r['text']) for r in hybrid_results]
                rerank_scores = rerank_model.predict(pairs)
                for r, s in zip(hybrid_results, rerank_scores):
                    # combine reranker score (assumed larger = better) with hybrid
                    # normalize reranker using min-max across scores
                    r['rerank_raw'] = float(s)
                # normalize reranker scores to [0,1]
                vals = [r.get('rerank_raw', 0.0) for r in hybrid_results]
                vmin, vmax = min(vals), max(vals)
                span = max(1e-6, vmax - vmin)
                for r in hybrid_results:
                    r['rerank'] = (r.get('rerank_raw', 0.0) - vmin) / span
                    # final combination: give reranker strong influence
                    r['final_score'] = 0.5 * r['hybrid'] + 0.5 * r['rerank']
                hybrid_results.sort(key=lambda x: x.get('final_score', x['hybrid']), reverse=True)
            except Exception as e:
                st.warning(f"Reranker failed to load or run: {e}")

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

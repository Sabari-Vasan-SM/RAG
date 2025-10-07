# RAG Resume Search

This is a small Streamlit application for searching PDF resumes using a hybrid retrieval approach: semantic embeddings + keyword matching.

What's included
- `app.py` — Streamlit app that ingests PDF resumes, builds a FAISS index of chunk embeddings, and lets you search resumes using a hybrid ranking (embedding similarity + keyword match).
- `requirements.txt` — Python dependencies used by the project.

Key improvements (over a plain L2 setup)
- Embeddings are normalized and an inner-product FAISS index (`IndexFlatIP`) is used to approximate cosine similarity (more accurate semantic matching).
- Document embeddings are stored and sliced for filtered searches (no reliance on index.reconstruct()).
- A hybrid ranking scheme mixes embedding similarity with a simple keyword-match fraction to favor chunks that contain explicit query terms.
- UI shows embedding, keyword, and hybrid scores and highlights keyword matches in results.

Quick setup (Windows PowerShell)
1. Create and activate a virtualenv (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the Streamlit app:

```powershell
streamlit run app.py
```

Notes and troubleshooting
- If some packages fail to install (faiss-cpu or sentence-transformers), try installing them individually or check your Python version. Example:

```powershell
pip install faiss-cpu sentence-transformers
```

- `get_errors` or your editor may show unresolved import errors if the virtual environment is not activated or packages are not installed.

Tuning suggestions
- Hybrid weights: edit the weights in `app.py` (currently 0.75 embed / 0.25 keywords). Increase keyword weight if exact term matches should be favored.
- Chunk size: `CharacterTextSplitter` currently uses chunk_size=1200 and chunk_overlap=250. Smaller chunks may increase precision at the cost of slightly slower indexing.
- For larger datasets, consider switching FAISS to an approximate index (IVF/HNSW) for speed and persistence.

Next improvements you may want
- Replace the simple name-guessing heuristic with an NER model or regex rules for better candidate extraction.
- Implement fuzzy keyword matching or lemmatization.
- Add persistence: save/load FAISS index and embeddings to disk to avoid recomputing on every run.
- Add a small test harness to verify indexing and retrieval behavior on sample resumes.

If you want, I can add runtime UI controls for hybrid weight and chunk size, or implement index persistence next.

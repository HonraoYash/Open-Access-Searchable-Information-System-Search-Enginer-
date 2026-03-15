# OASIS Search Engine

OASIS is a local search-and-analytics application built from homework logic (HW1-HW3, excluding HW0).  
It lets users search the Enron email corpus with multiple retrieval methods and explore email-network analytics in a single Streamlit UI.

## Purpose

- Turn notebook code into a reusable project.
- Run fully on local machine (no cloud hosting required).
- Precompute indexes once and reuse them across app restarts.
- Provide both:
  - **Email retrieval** (Boolean, TF-IDF, BM25, DESM, weighted fusion)
  - **Graph analytics** (PageRank, Personalized PageRank, HITS)

## How the app works

This project has a clean split between **logic** and **UI**:

- `backend/app/services/search_engine.py`
  - parses emails
  - tokenizes/preprocesses text
  - builds search indexes
  - executes Boolean / TF-IDF / BM25 / DESM retrieval
- `backend/app/services/graph_engine.py`
  - parses `From` / `To`
  - builds email interaction graph
  - computes PageRank, Personalized PageRank, HITS
- `frontend/streamlit_app.py`
  - renders search UI and email viewer
  - loads cached index objects
  - calls service methods directly in-process

Although there is a FastAPI scaffold in `backend/app/main.py`, the current workflow is **local-first** and does **not** require running `uvicorn`.

## Project structure

```text
backend/
  app/
    main.py
    models.py
    services/
      search_engine.py
      graph_engine.py
      state.py
frontend/
  streamlit_app.py
scripts/
  precompute_index.py
data/
  index_cache.pkl                # generated once
.archive/
  homeworks/                     # original notebooks
  datasets/
    enron_sent_mail/             # archived corpus
requirements.txt
```

## Setup and run

### 1) Create environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Precompute indexes (one-time)

```bash
python scripts/precompute_index.py --dataset .archive/datasets/enron_sent_mail --cache data/index_cache.pkl
```

### 4) Launch app

```bash
streamlit run frontend/streamlit_app.py
```

The app auto-loads `data/index_cache.pkl`.  
You only need to rebuild cache when dataset/logic changes.

## Deploy on Streamlit Community Cloud

1. Push code to GitHub (without large cache/data files).
2. In Streamlit Cloud, create app from your GitHub repo.
3. Set app secret:

```toml
CACHE_URL = "https://drive.google.com/uc?export=download&id=YOUR_FILE_ID"
```

4. On first startup, app downloads cache to `data/index_cache.pkl` and loads it automatically.

Notes:
- Google Drive links require a **file** link (not folder).
- The file must be shared as **Anyone with the link (Viewer)**.

## Retrieval methods (brief)

- **Boolean Retrieval**
  - Exact logic-based matching using `AND`, `OR`, `NOT`.
  - Great for strict filtering; less flexible semantically.

- **TF-IDF + Cosine Similarity**
  - Weighs terms by importance in corpus (rare terms get higher weight).
  - Good baseline vector-space ranking.

- **BM25**
  - Probabilistic ranking function with document-length normalization.
  - Usually stronger lexical relevance than plain TF-IDF for ranked retrieval.

- **DESM-style Semantic Matching**
  - Uses vector-based term/document representations for semantic similarity.
  - Helps retrieve conceptually related emails even with term mismatch.

- **`all` mode (weighted fusion)**
  - Combines scores from all four methods into one ranked list.
  - Higher weights are assigned to BM25 and DESM for stronger relevance.

## Graph analytics methods (brief)

- **PageRank**: global influence in the communication network.
- **Personalized PageRank**: influence relative to a chosen user seed.
- **HITS**: hub and authority roles in the email graph.

## Notes

- Local-only app, designed for personal workstation usage.
- Compatible with Python 3.14 dependency set used in this repository.
- Legacy homework files and raw dataset are kept under `.archive/` to keep root clean.
- `data/index_cache.pkl` is intentionally excluded from git and fetched at runtime for deployment.

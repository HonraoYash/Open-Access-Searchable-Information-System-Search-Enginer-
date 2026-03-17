# OASIS: Open Access Searchable Information System

OASIS is a Search Engine and is built to demonstrate how modern and classical search/ranking methods work in a practical application.

It combines:

- a modern Streamlit interface
- a precomputed index pipeline
- multiple retrieval/ranking methods
- graph-based authority analytics
- deployable cloud hosting

## Live App

**Hosted on Streamlit Community Cloud:**  
[https://open-access-searchable-information-system.streamlit.app/](https://open-access-searchable-information-system.streamlit.app/)

## Key Features

- Multi-method search over an email corpus:
  - Boolean retrieval
  - TF-IDF + cosine similarity
  - BM25 ranking
  - DESM-style semantic matching
  - weighted fusion ranking (`all` mode)
- Interactive result exploration:
  - ranked list with metadata
  - click-to-open full email viewer
- Graph analytics over communication network:
  - PageRank
  - Personalized PageRank
  - HITS (hubs/authorities)
- Deployment-ready architecture with runtime cache download

## What This Project Demonstrates

- How classic IR methods behave on real email corpora.
- How semantic and lexical signals can be combined for ranking.
- How graph centrality methods identify important actors in communication networks.
- How these ideas can be packaged into a production-style interactive app.

## Tech Stack

- **Frontend**: Streamlit
- **Language**: Python
- **Numerical/Data**: NumPy, Pandas
- **Graph Analytics**: NetworkX
- **Deployment**: Streamlit Community Cloud
- **Cache Hosting**: external object/file URL (Google Drive via `CACHE_URL`)

## Methods Implemented

### Retrieval and Ranking

- **Boolean Retrieval**
  - Supports strict query operators: `AND`, `OR`, `NOT`.
  - Best when users need precise logical constraints.

- **TF-IDF + Cosine Similarity**
  - Terms are weighted by importance in corpus.
  - Ranks documents by vector similarity to query.

- **BM25 (Okapi)**
  - Probabilistic ranking with document-length normalization.
  - Uses term frequency saturation and inverse document frequency.
  - Strong lexical baseline for ranked search.

- **DESM-style Semantic Matching**
  - Uses vector representations to capture semantic similarity.
  - Helps retrieve relevant emails even when exact terms differ.

- **Weighted Fusion (`all` mode)**
  - Combines Boolean + TF-IDF + BM25 + DESM into one ranked list.
  - Higher weight is assigned to BM25 and DESM for practical relevance quality.

### Graph Analytics

- **PageRank**
  - Finds globally influential nodes in the email graph.

- **Personalized PageRank**
  - Biases ranking around a selected user (e.g., `john.arnold@enron.com`).

- **HITS**
  - Separates nodes into **hubs** and **authorities**.

## Architecture

- `backend/app/services/search_engine.py`
  - parsing, preprocessing, indexing, and ranking algorithms
- `backend/app/services/graph_engine.py`
  - graph construction and centrality algorithms
- `frontend/streamlit_app.py`
  - search UI, result list, click-to-open email viewer
- `scripts/precompute_index.py`
  - one-time index + graph cache builder

The app runs local-first and in-process; no separate backend server is required for normal use.

### Runtime Flow

1. App starts and checks `data/index_cache.pkl`.
2. If missing, app downloads cache from `CACHE_URL` (Streamlit secrets / env).
3. Cached search + graph objects are loaded.
4. Query is executed with selected retrieval mode.
5. Results are rendered; user can open full email content.

## Repository Structure

```text
backend/
  app/
    services/
      search_engine.py
      graph_engine.py
frontend/
  streamlit_app.py
scripts/
  precompute_index.py
data/
  index_cache.pkl              # generated at runtime (not tracked in git)
.archive/
  homeworks/                   # archived local resources
  datasets/
    enron_sent_mail/           # local archived source dataset
requirements.txt
```

## Example Queries

- `buyer`
- `gas floor`
- `winter OR summer`
- `margins AND limits`
- `buyers AND risk AND NOT crazy`

## Local Run

### 1) Create environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Build cache once

```bash
python scripts/precompute_index.py --dataset .archive/datasets/enron_sent_mail --cache data/index_cache.pkl
```

### 4) Start app

```bash
streamlit run frontend/streamlit_app.py
```

## Streamlit Cloud Deployment

Because the cache file is large, it is not committed to GitHub.  
At startup, the app downloads cache from external storage (Google Drive link in Secrets).

Set this in Streamlit app secrets:

```toml
CACHE_URL = "https://drive.google.com/uc?export=download&id=<FILE_ID>"
```

On first cloud run:
- app downloads `index_cache.pkl`
- loads indexes
- starts serving search/analytics UI

## Why Weighted Fusion

Single retrieval methods capture different notions of relevance:

- BM25 is strong for lexical/document relevance.
- DESM improves semantic matching when exact wording differs.
- TF-IDF provides a solid vector-space baseline.
- Boolean provides strict constraints when operators are explicitly used.

The fused `all` mode combines these signals, with higher practical weight on BM25 and DESM.

## Future Work

- Query-term highlighting in subject/snippet.
- Pagination and result export.
- Faster/lighter cache format for quicker cold-start.
- Explainability panel (per-method contribution to final fused score).
- Optional feedback loop for query refinement.

## Notes

- Local datasets/resources are intentionally archived under `.archive/`.
- Generated artifacts (`data/index_cache.pkl`, `__pycache__`, `.venv`) are excluded from git.
- Rebuild cache only when dataset or ranking logic changes.

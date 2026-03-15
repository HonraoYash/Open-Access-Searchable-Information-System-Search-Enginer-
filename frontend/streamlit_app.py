import os
import pickle
import sys
import re
import urllib.request
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.append(str(BACKEND_DIR))


st.set_page_config(page_title="OASIS Search", page_icon="🔎", layout="wide")


def sanitize_text(text: str) -> str:
    if text is None:
        return ""
    text = text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    return text


def get_cache_url() -> str | None:
    try:
        value = st.secrets.get("CACHE_URL", None)
        if value:
            return str(value)
    except Exception:
        pass
    env_value = os.getenv("CACHE_URL", "").strip()
    return env_value if env_value else None


def download_cache_file(url: str, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if "drive.google.com" in url:
        try:
            import gdown  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Missing dependency 'gdown' for Google Drive cache download.") from exc
        ok = gdown.download(url=url, output=output_path, quiet=False, fuzzy=True)
        if not ok:
            raise RuntimeError("Google Drive download failed.")
        return
    urllib.request.urlretrieve(url, output_path)


def normalize_scores(scored: list[tuple[str, float]]) -> dict[str, float]:
    if not scored:
        return {}
    values = [float(score) for _, score in scored]
    min_v = min(values)
    max_v = max(values)
    if max_v == min_v:
        return {doc_id: 1.0 for doc_id, _ in scored}
    return {doc_id: (float(score) - min_v) / (max_v - min_v) for doc_id, score in scored}


def combined_weighted_results(search_engine, query: str, top_k: int) -> list[dict]:
    # Weighted fusion: prioritize semantic + lexical relevance.
    weights = {
        "bm25": 0.38,
        "desm": 0.36,
        "tfidf": 0.18,
        "boolean": 0.08,
    }
    depth = max(top_k * 8, 40)
    bm25 = search_engine.search_bm25(query, depth)
    desm = search_engine.search_desm(query, depth)
    tfidf = search_engine.search_tfidf(query, depth)
    boolean = search_engine.boolean_retrieval(query, depth)

    norm_bm25 = normalize_scores(bm25)
    norm_desm = normalize_scores(desm)
    norm_tfidf = normalize_scores(tfidf)
    norm_boolean = normalize_scores(boolean)

    union_doc_ids = set()
    union_doc_ids.update(norm_bm25.keys())
    union_doc_ids.update(norm_desm.keys())
    union_doc_ids.update(norm_tfidf.keys())
    union_doc_ids.update(norm_boolean.keys())

    weighted = []
    for doc_id in union_doc_ids:
        score = (
            weights["bm25"] * norm_bm25.get(doc_id, 0.0)
            + weights["desm"] * norm_desm.get(doc_id, 0.0)
            + weights["tfidf"] * norm_tfidf.get(doc_id, 0.0)
            + weights["boolean"] * norm_boolean.get(doc_id, 0.0)
        )
        weighted.append((doc_id, score))

    weighted.sort(key=lambda x: x[1], reverse=True)
    return search_engine.format_results(weighted[:top_k])


@st.cache_data(show_spinner=False)
def read_email_headers(file_path: str) -> dict:
    subject = "(No Subject)"
    sender = "(Unknown Sender)"
    date = "(Unknown Date)"
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                lower = line.lower()
                if lower.startswith("subject:"):
                    parsed = sanitize_text(line.split(":", 1)[1].strip())
                    if parsed:
                        subject = parsed
                elif lower.startswith("from:"):
                    parsed = sanitize_text(line.split(":", 1)[1].strip())
                    if parsed:
                        sender = parsed
                elif lower.startswith("date:"):
                    parsed = sanitize_text(line.split(":", 1)[1].strip())
                    if parsed:
                        date = parsed
                elif line.strip() == "":
                    break
    except Exception:
        pass
    return {"subject": subject, "sender": sender, "date": date}


def get_email_meta(search_engine, result: dict, dataset_path: str) -> dict:
    doc = search_engine.doc_by_id.get(result["document_id"])
    if doc is None:
        return {"subject": "(No Subject)", "sender": "(Unknown Sender)", "date": "(Unknown Date)"}

    file_path = os.path.join(dataset_path, doc.filename)
    headers = read_email_headers(file_path)
    subject = sanitize_text(str(result.get("subject", "")).strip())
    if not subject or subject == "(No Subject)":
        cached_subject = sanitize_text(getattr(doc, "subject", "")).strip()
        subject = cached_subject if cached_subject else headers["subject"]
    return {
        "subject": subject if subject else "(No Subject)",
        "sender": headers["sender"],
        "date": headers["date"],
    }


def render_result_block(rows: list[dict], search_engine, dataset_path: str) -> None:
    # st.subheader(title)
    if not rows:
        st.info("No results.")
        return
    for item in rows:
        score = float(item["score"])
        snippet = sanitize_text(str(item["snippet"]))
        meta = get_email_meta(search_engine, item, dataset_path)
        subject = meta["subject"]
        if len(snippet) > 320:
            snippet = snippet[:320] + "..."

        with st.container(border=True):
            st.caption(f"#{item['rank']}  |  Score: {score:.4f}")
            button_key = f"{abs(hash(str(item['document_id'])))}"
            if st.button(
                subject,
                key=button_key,
                use_container_width=True,
                help="Click to open this email",
            ):
                st.session_state["selected_doc_id"] = item["document_id"]
            st.caption(f"From: {meta['sender']}  |  Date: {meta['date']}")
            st.write(snippet)


def render_email_viewer(search_engine, dataset_path: str) -> None:
    selected_doc_id = st.session_state.get("selected_doc_id")
    if not selected_doc_id:
        return
    doc = search_engine.doc_by_id.get(selected_doc_id)
    if doc is None:
        return
    subject = get_email_meta(
        search_engine,
        {"document_id": selected_doc_id, "subject": getattr(doc, "subject", "(No Subject)")},
        dataset_path,
    )["subject"]
    meta = get_email_meta(
        search_engine,
        {"document_id": selected_doc_id, "subject": getattr(doc, "subject", "(No Subject)")},
        dataset_path,
    )
    st.markdown(f"### {subject}")
    with st.container(border=True):
        st.caption(f"From: {meta['sender']}  |  Date: {meta['date']}")
        st.markdown(f"**Source:** `{sanitize_text(doc.filename)}`")
        full_text = sanitize_text(doc.raw_body)
        st.text_area("Email Body", value=full_text, height=460, key="full_email_viewer")
        if st.button("Back to Search Results", help="Back to search results"):
            st.session_state["selected_doc_id"] = None


st.markdown(
    """
    <style>
    .google-title {
      text-align: center;
      font-size: 52px;
      font-weight: 700;
      margin-top: 16px;
      margin-bottom: 18px;
      letter-spacing: -0.5px;
    }
    .search-note {
      text-align: center;
      color: #4f4f4f;
      margin-bottom: 14px;
    }
    .block-container {
      padding-top: 1.2rem;
      max-width: 980px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='google-title'>OASIS Search</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='search-note'>Open Access Searchable Information System is a Search Engine for searching the Enron Email Dataset</div>",
    unsafe_allow_html=True,
)

@st.cache_resource(show_spinner=False)
def load_local_cache_once(cache_file: str):
    with open(cache_file, "rb") as f:
        payload = pickle.load(f)
    return payload["search_engine"], payload["graph_engine"], payload.get("metadata", {})


default_cache = str(ROOT / "data" / "index_cache.pkl")
local_cache_path = default_cache

cache_exists = os.path.exists(local_cache_path)
if not cache_exists:
    cache_url = get_cache_url()
    if not cache_url:
        st.error(
            "Index cache not found and CACHE_URL is not configured.\n\n"
            "For Streamlit Cloud, set `CACHE_URL` in app Secrets.\n"
            "For local run, either set CACHE_URL env var or place `data/index_cache.pkl` manually."
        )
        st.stop()
    with st.spinner("Downloading index cache... (first run only)"):
        try:
            download_cache_file(cache_url, local_cache_path)
        except Exception as exc:
            st.error(f"Cache download failed: {exc}")
            st.stop()

try:
    search_engine, graph_engine, metadata = load_local_cache_once(local_cache_path)
except Exception as exc:
    st.error(f"Failed to load cache: {exc}")
    st.stop()

archive_dataset_default = str(ROOT / ".archive" / "datasets" / "enron_sent_mail")

st.caption(
    "Loaded local cache | "
    f"Docs: {len(search_engine.documents)} | "
    f"Graph nodes: {graph_engine.graph.number_of_nodes()} | "
    f"Built: {sanitize_text(str(metadata.get('built_at', 'unknown')))}"
)

tab_search, tab_graph = st.tabs(["Search", "Graph Analytics"])

with tab_search:
    dataset_path = str(metadata.get("dataset_path", archive_dataset_default))
    if not os.path.exists(dataset_path):
        dataset_path = archive_dataset_default
    query = st.text_input(
        "Search Query",
        placeholder="Try: buyer, gas floor, winter OR summer",
        help="Type natural language terms or boolean expressions (AND/OR/NOT).",
    )
    col1, col2 = st.columns([2, 1])
    with col1:
        mode = st.selectbox(
            "Search Mode",
            ["all", "bm25", "tfidf", "boolean", "desm"],
            help=(
                "all: weighted fusion (BM25 + DESM prioritized), "
                "bm25: lexical relevance, tfidf: vector-space relevance, "
                "boolean: strict logic filter, desm: semantic similarity."
            ),
        )
    with col2:
        top_k = st.slider(
            "Top K",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
            help="Number of results to return.",
        )

    st.caption(
        "ℹ️ Click any result row to open the full email. Use Back to return."
    )

    if st.button("Search", use_container_width=True):
        if not query.strip():
            st.warning("Enter a query.")
        else:
            try:
                if mode == "all":
                    rows = combined_weighted_results(search_engine, query.strip(), top_k)
                    # title = "Weighted Combined Ranking"
                elif mode == "bm25":
                    rows = search_engine.format_results(search_engine.search_bm25(query.strip(), top_k))
                    title = "BM25"
                elif mode == "tfidf":
                    rows = search_engine.format_results(search_engine.search_tfidf(query.strip(), top_k))
                    title = "TF-IDF"
                elif mode == "boolean":
                    rows = search_engine.format_results(search_engine.boolean_retrieval(query.strip(), top_k))
                    title = "BOOLEAN"
                else:
                    rows = search_engine.format_results(search_engine.search_desm(query.strip(), top_k))
                    title = "DESM"

                st.session_state["search_results"] = rows
                # st.session_state["search_title"] = title
                st.session_state["selected_doc_id"] = None
            except Exception as exc:
                st.error(str(exc))

    if st.session_state.get("selected_doc_id"):
        render_email_viewer(search_engine, dataset_path)
    else:
        rows = st.session_state.get("search_results", [])
        # title = st.session_state.get("search_title", "Results")
        if rows:
            render_result_block(rows, search_engine, dataset_path)

with tab_graph:
    top_k_graph = st.slider(
        "Top K Users",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
        help="Number of top users to show for graph ranking methods.",
    )
    target_user = st.text_input(
        "Personalized PageRank User",
        value="john.arnold@enron.com",
        help="Personalization seed for topic/user-centric graph influence.",
    )
    st.caption("ℹ️ PageRank shows global influence, Personalized PageRank shows user-biased influence, HITS shows hubs/authorities.")

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("PageRank", use_container_width=True):
            try:
                ranked = graph_engine.pagerank(top_k_graph)
                st.json(
                    [
                        {"rank": i, "user": u, "score": float(s)}
                        for i, (u, s) in enumerate(ranked, start=1)
                    ]
                )
            except Exception as exc:
                st.error(str(exc))
    with c2:
        if st.button("Personalized PageRank", use_container_width=True):
            try:
                ranked = graph_engine.personalized_pagerank(target_user.strip(), top_k_graph)
                st.json(
                    [
                        {"rank": i, "user": u, "score": float(s)}
                        for i, (u, s) in enumerate(ranked, start=1)
                    ]
                )
            except Exception as exc:
                st.error(str(exc))
    with c3:
        if st.button("HITS", use_container_width=True):
            try:
                hubs, authorities = graph_engine.hits(top_k_graph)
                st.markdown("**Hubs**")
                st.json(
                    [
                        {"rank": i, "user": u, "score": float(s)}
                        for i, (u, s) in enumerate(hubs, start=1)
                    ]
                )
                st.markdown("**Authorities**")
                st.json(
                    [
                        {"rank": i, "user": u, "score": float(s)}
                        for i, (u, s) in enumerate(authorities, start=1)
                    ]
                )
            except Exception as exc:
                st.error(str(exc))

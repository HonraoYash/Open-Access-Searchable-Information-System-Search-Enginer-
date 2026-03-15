from fastapi import FastAPI, HTTPException

from app.models import (
    GraphRequest,
    GraphResponse,
    GraphScore,
    LoadRequest,
    SearchRequest,
    SearchResponse,
    SearchResult,
    StatusResponse,
)
from app.services.state import state

app = FastAPI(
    title="OASIS Search Engine API",
    description="Backend APIs extracted from HW1/HW2/HW3 logic.",
    version="1.0.0",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/load")
def load_data(payload: LoadRequest) -> dict:
    loaded_any = False

    if payload.corpus_path:
        state.search_engine.load_corpus(payload.corpus_path)
        state.corpus_loaded = True
        state.metadata["corpus_path"] = payload.corpus_path
        loaded_any = True

    if payload.mail_path:
        state.graph_engine.process_email_dataset(payload.mail_path)
        state.graph_loaded = True
        state.metadata["mail_path"] = payload.mail_path
        loaded_any = True

    if not loaded_any:
        raise HTTPException(status_code=400, detail="Provide at least one dataset path.")

    return {
        "message": "Data loaded.",
        "corpus_documents": len(state.search_engine.documents),
        "graph_nodes": state.graph_engine.graph.number_of_nodes(),
        "graph_edges": state.graph_engine.graph.number_of_edges(),
    }


@app.get("/status", response_model=StatusResponse)
def status() -> StatusResponse:
    return StatusResponse(
        corpus_loaded=state.corpus_loaded,
        graph_loaded=state.graph_loaded,
        corpus_documents=len(state.search_engine.documents),
        graph_nodes=state.graph_engine.graph.number_of_nodes(),
        graph_edges=state.graph_engine.graph.number_of_edges(),
        metadata=state.metadata,
    )


def _search_response(method: str, query: str, scored: list[tuple[str, float]]) -> SearchResponse:
    formatted = state.search_engine.format_results(scored)
    results = [SearchResult(**row) for row in formatted]
    return SearchResponse(method=method, query=query, results=results)


@app.post("/search/boolean", response_model=SearchResponse)
def search_boolean(payload: SearchRequest) -> SearchResponse:
    if not state.corpus_loaded:
        raise HTTPException(status_code=400, detail="Corpus not loaded.")
    scored = state.search_engine.boolean_retrieval(payload.query, payload.top_k)
    return _search_response("boolean", payload.query, scored)


@app.post("/search/tfidf", response_model=SearchResponse)
def search_tfidf(payload: SearchRequest) -> SearchResponse:
    if not state.corpus_loaded:
        raise HTTPException(status_code=400, detail="Corpus not loaded.")
    scored = state.search_engine.search_tfidf(payload.query, payload.top_k)
    return _search_response("tfidf", payload.query, scored)


@app.post("/search/bm25", response_model=SearchResponse)
def search_bm25(payload: SearchRequest) -> SearchResponse:
    if not state.corpus_loaded:
        raise HTTPException(status_code=400, detail="Corpus not loaded.")
    scored = state.search_engine.search_bm25(payload.query, payload.top_k)
    return _search_response("bm25", payload.query, scored)


@app.post("/search/desm", response_model=SearchResponse)
def search_desm(payload: SearchRequest) -> SearchResponse:
    if not state.corpus_loaded:
        raise HTTPException(status_code=400, detail="Corpus not loaded.")
    scored = state.search_engine.search_desm(payload.query, payload.top_k)
    return _search_response("desm", payload.query, scored)


@app.post("/search/all")
def search_all(payload: SearchRequest) -> dict:
    if not state.corpus_loaded:
        raise HTTPException(status_code=400, detail="Corpus not loaded.")

    return {
        "query": payload.query,
        "boolean": state.search_engine.format_results(
            state.search_engine.boolean_retrieval(payload.query, payload.top_k)
        ),
        "tfidf": state.search_engine.format_results(
            state.search_engine.search_tfidf(payload.query, payload.top_k)
        ),
        "bm25": state.search_engine.format_results(
            state.search_engine.search_bm25(payload.query, payload.top_k)
        ),
        "desm": state.search_engine.format_results(
            state.search_engine.search_desm(payload.query, payload.top_k)
        ),
    }


def _graph_response(method: str, scored: list[tuple[str, float]]) -> GraphResponse:
    results = [
        GraphScore(rank=i, user=user, score=float(score))
        for i, (user, score) in enumerate(scored, start=1)
    ]
    return GraphResponse(method=method, results=results)


@app.post("/graph/pagerank", response_model=GraphResponse)
def pagerank(payload: GraphRequest) -> GraphResponse:
    if not state.graph_loaded:
        raise HTTPException(status_code=400, detail="Graph dataset not loaded.")
    return _graph_response("pagerank", state.graph_engine.pagerank(payload.top_k))


@app.post("/graph/personalized-pagerank", response_model=GraphResponse)
def personalized_pagerank(payload: GraphRequest) -> GraphResponse:
    if not state.graph_loaded:
        raise HTTPException(status_code=400, detail="Graph dataset not loaded.")
    scored = state.graph_engine.personalized_pagerank(payload.user, payload.top_k)
    return _graph_response("personalized_pagerank", scored)


@app.post("/graph/hits")
def hits(payload: GraphRequest) -> dict:
    if not state.graph_loaded:
        raise HTTPException(status_code=400, detail="Graph dataset not loaded.")
    hubs, authorities = state.graph_engine.hits(payload.top_k)
    return {
        "hubs": _graph_response("hits_hubs", hubs).model_dump()["results"],
        "authorities": _graph_response("hits_authorities", authorities).model_dump()["results"],
    }

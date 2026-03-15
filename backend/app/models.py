from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class LoadRequest(BaseModel):
    corpus_path: Optional[str] = Field(
        default=None,
        description="Path to flat directory of email files (HW1/HW3 style).",
    )
    mail_path: Optional[str] = Field(
        default=None,
        description="Path to sent-mail dataset root (HW2 style).",
    )


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class GraphRequest(BaseModel):
    top_k: int = 5
    user: str = "john.arnold@enron.com"


class SearchResult(BaseModel):
    rank: int
    document_id: str
    score: float
    snippet: str


class SearchResponse(BaseModel):
    method: str
    query: str
    results: List[SearchResult]


class GraphScore(BaseModel):
    rank: int
    user: str
    score: float


class GraphResponse(BaseModel):
    method: str
    results: List[GraphScore]


class StatusResponse(BaseModel):
    corpus_loaded: bool
    graph_loaded: bool
    corpus_documents: int
    graph_nodes: int
    graph_edges: int
    metadata: Dict[str, str]

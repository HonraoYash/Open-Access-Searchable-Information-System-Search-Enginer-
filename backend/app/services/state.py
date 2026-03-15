from dataclasses import dataclass, field
from typing import Dict

from app.services.graph_engine import GraphEngine
from app.services.search_engine import SearchEngine


@dataclass
class AppState:
    search_engine: SearchEngine = field(default_factory=SearchEngine)
    graph_engine: GraphEngine = field(default_factory=GraphEngine)
    corpus_loaded: bool = False
    graph_loaded: bool = False
    metadata: Dict[str, str] = field(default_factory=dict)


state = AppState()

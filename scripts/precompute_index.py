import argparse
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.append(str(BACKEND_DIR))

from app.services.graph_engine import GraphEngine
from app.services.search_engine import SearchEngine


def build_index(dataset_path: str, cache_path: str) -> None:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    started = time.time()
    search_engine = SearchEngine()
    graph_engine = GraphEngine()

    print(f"Building search index from: {dataset_path}")
    search_engine.load_corpus(dataset_path)
    print(f"Search docs indexed: {len(search_engine.documents)}")

    print(f"Building graph index from: {dataset_path}")
    graph_engine.process_email_dataset(dataset_path)
    print(
        "Graph built: "
        f"{graph_engine.graph.number_of_nodes()} nodes, "
        f"{graph_engine.graph.number_of_edges()} edges"
    )

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    payload = {
        "search_engine": search_engine,
        "graph_engine": graph_engine,
        "metadata": {
            "dataset_path": os.path.abspath(dataset_path),
            "built_at": datetime.now().isoformat(timespec="seconds"),
        },
    }
    with open(cache_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    elapsed = time.time() - started
    print(f"Cache saved to: {cache_path}")
    print(f"Done in {elapsed:.2f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute local index cache for OASIS app.")
    parser.add_argument(
        "--dataset",
        default=str(ROOT / ".archive" / "datasets" / "enron_sent_mail"),
        help="Path to enron_sent_mail dataset root.",
    )
    parser.add_argument(
        "--cache",
        default=str(ROOT / "data" / "index_cache.pkl"),
        help="Output cache file path.",
    )
    args = parser.parse_args()
    build_index(args.dataset, args.cache)


if __name__ == "__main__":
    main()

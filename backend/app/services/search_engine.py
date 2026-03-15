import math
import os
import re
import string
import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class CorpusDocument:
    document_id: str
    clean_text: str
    raw_body: str
    tokens: List[str]
    filename: str
    subject: str = "(No Subject)"


class SearchEngine:
    def __init__(self) -> None:
        self.documents: List[CorpusDocument] = []
        self.doc_by_id: Dict[str, CorpusDocument] = {}
        self.inverted_index: Dict[str, set] = defaultdict(set)
        self.tf: Dict[str, Dict[str, int]] = {}
        self.df: Dict[str, int] = {}
        self.doc_lengths: Dict[str, int] = {}
        self.idf_bm25: Dict[str, float] = {}
        self.avg_doc_length: float = 0.0
        self.idf_tfidf: Dict[str, float] = {}
        self.doc_tfidf_vectors: Dict[str, Dict[str, float]] = {}
        self.desm_dim: int = 1024
        self.doc_embeddings: Dict[str, np.ndarray] = {}

    @staticmethod
    def parse_body(content: str) -> str:
        body_match = re.search(r"(\r?\n){2,}", content)
        if body_match:
            return content[body_match.end() :]
        return content

    @staticmethod
    def parse_message_id(content: str) -> str:
        match = re.search(r"Message-ID: <(\d+\.\d+)", content)
        if match:
            return match.group(1)
        return "Unknown ID"

    @staticmethod
    def parse_subject(content: str) -> str:
        match = re.search(r"^Subject:\s*(.*)$", content, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            subject = match.group(1).strip()
            return subject if subject else "(No Subject)"
        return "(No Subject)"

    @staticmethod
    def preprocessing(text: str) -> List[str]:
        text = text.lower()
        text = text.replace("\n", " ")
        text = text.replace("-", " ")
        text = text.replace("/", " ")
        translator = str.maketrans(string.punctuation, " " * len(string.punctuation))
        text = text.translate(translator)
        text = re.sub(r"[^a-zA-Z0-9 \n\.]", "", text)
        tokens = text.split()
        filtered_tokens = [item for item in tokens if re.match(r"^[a-zA-Z]+$", item)]
        return filtered_tokens

    @staticmethod
    def _extract_body_for_snippet(text: str) -> str:
        match = re.search(r"(\r?\n){2,}", text)
        if match:
            body = text[match.end() :]
        else:
            body = text
        body = re.sub(r"\s+", " ", body).strip()
        return body

    def load_corpus(self, folder_path: str) -> None:
        documents: List[CorpusDocument] = []
        seen_ids: set[str] = set()

        # Supports both flat corpora (HW1/HW3) and nested enron_sent_mail style folders.
        for root, _, files in os.walk(folder_path):
            for filename in sorted(files):
                if filename.startswith("."):
                    continue
                file_path = os.path.join(root, filename)
                if not os.path.isfile(file_path):
                    continue
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                body = self.parse_body(content)
                message_id = self.parse_message_id(content)
                subject = self.parse_subject(content)
                fallback_id = os.path.relpath(file_path, folder_path)
                doc_id = message_id if message_id != "Unknown ID" else fallback_id
                if doc_id in seen_ids:
                    doc_id = f"{doc_id}::{fallback_id}"
                seen_ids.add(doc_id)

                clean_text = body.strip().replace("\n", " ")
                tokens = self.preprocessing(clean_text)
                documents.append(
                    CorpusDocument(
                        document_id=doc_id,
                        clean_text=clean_text,
                        raw_body=body,
                        tokens=tokens,
                        filename=fallback_id,
                        subject=subject,
                    )
                )

        self.documents = documents
        self.doc_by_id = {d.document_id: d for d in documents}
        self._build_indexes()

    def _build_indexes(self) -> None:
        self.inverted_index = defaultdict(set)
        self.tf = {}
        self.df = {}
        self.doc_lengths = {}
        self.idf_bm25 = {}
        self.doc_tfidf_vectors = {}
        self.doc_embeddings = {}

        for doc in self.documents:
            for term in set(doc.tokens):
                self.inverted_index[term].add(doc.document_id)

            self.doc_lengths[doc.document_id] = len(doc.tokens)
            term_counts: Dict[str, int] = {}
            for token in doc.tokens:
                term_counts[token] = term_counts.get(token, 0) + 1
            self.tf[doc.document_id] = term_counts

        for _, word_counts in self.tf.items():
            for word in word_counts:
                self.df[word] = self.df.get(word, 0) + 1

        n_docs = len(self.documents)
        self.avg_doc_length = (
            float(np.mean(list(self.doc_lengths.values()))) if self.doc_lengths else 0.0
        )
        self.idf_bm25 = {
            word: math.log((n_docs - self.df[word] + 0.5) / (self.df[word] + 0.5) + 1)
            for word in self.df
        }

        self.idf_tfidf = self.calculate_idf([d.tokens for d in self.documents])
        for doc in self.documents:
            self.doc_tfidf_vectors[doc.document_id] = self.compute_tfidf(
                doc.tokens, self.idf_tfidf
            )

        if self.documents:
            self._train_desm()

    @staticmethod
    def calculate_tf(doc_toks: List[str]) -> Dict[str, float]:
        counts_of_terms = Counter(doc_toks)
        total_no_of_terms = len(doc_toks)
        if total_no_of_terms == 0:
            return {}
        return {term: count / total_no_of_terms for term, count in counts_of_terms.items()}

    @staticmethod
    def calculate_idf(docs: List[List[str]]) -> Dict[str, float]:
        num_of_docs = len(docs)
        idf_vals: Dict[str, float] = {}
        all_terms = {term for doc in docs for term in doc}
        for term in all_terms:
            doc_freq = sum(1 for doc in docs if term in doc)
            idf_vals[term] = math.log(num_of_docs / (1 + doc_freq)) if num_of_docs else 0.0
        return idf_vals

    def compute_tfidf(self, doc_toks: List[str], idf_vals: Dict[str, float]) -> Dict[str, float]:
        tf_vals = self.calculate_tf(doc_toks)
        return {term: tf_vals[term] * idf_vals[term] for term in doc_toks if term in idf_vals}

    @staticmethod
    def cosine_similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        dot_products = sum(vec1[term] * vec2.get(term, 0.0) for term in vec1)
        norm1 = math.sqrt(sum(count**2 for count in vec1.values()))
        norm2 = math.sqrt(sum(count**2 for count in vec2.values()))
        return dot_products / (norm1 * norm2) if norm1 and norm2 else 0.0

    def _process_boolean_query(self, query: str) -> List[str]:
        # Keep notebook logic where operators are explicitly typed by user.
        raw_tokens = query.split()
        processed_terms: List[str] = []
        for token in raw_tokens:
            if token.upper() in {"AND", "OR", "NOT"}:
                processed_terms.append(token.upper())
            else:
                processed_terms.extend(self.preprocessing(token))
        return processed_terms

    def boolean_retrieval(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        tokens = self._process_boolean_query(query)
        if not tokens:
            return []

        all_docs = set(self.doc_by_id.keys())
        new_set: set = set()
        operation = None
        for term in tokens:
            if term in {"AND", "OR", "NOT"}:
                operation = term
                continue

            doc_set = self.inverted_index.get(term, set())
            if operation == "AND":
                new_set = new_set & doc_set if new_set else set(doc_set)
            elif operation == "OR":
                new_set = new_set | doc_set
            elif operation == "NOT":
                if not new_set:
                    new_set = set(all_docs)
                new_set = new_set - doc_set
            else:
                new_set = set(doc_set)

        ranked = sorted(new_set)[:top_k]
        return [(doc_id, 1.0) for doc_id in ranked]

    def process_query(self, query: str) -> Dict[str, float]:
        query_tokens = query.lower().split()
        return self.compute_tfidf(query_tokens, self.idf_tfidf)

    def search_tfidf(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        query_vec = self.process_query(query)
        scored = []
        for doc_id, doc_vec in self.doc_tfidf_vectors.items():
            score = self.cosine_similarity(query_vec, doc_vec)
            scored.append((doc_id, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def bm25(self, query_tokens: List[str], doc_id: str, k1: float = 1.2, b: float = 0.75) -> float:
        score = 0.0
        if doc_id not in self.tf:
            return score
        for word in query_tokens:
            if word in self.tf[doc_id]:
                f = self.tf[doc_id][word]
                numer = self.idf_bm25.get(word, 0.0) * (f * (k1 + 1))
                denom = f + k1 * (
                    1 - b + b * (self.doc_lengths[doc_id] / self.avg_doc_length)
                ) if self.avg_doc_length else 1.0
                score += numer / denom
        return score

    def search_bm25(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        tokens = self.preprocessing(query)
        scores = [(doc.document_id, self.bm25(tokens, doc.document_id)) for doc in self.documents]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def _train_desm(self) -> None:
        # Python 3.14 friendly fallback for HW3 DESM endpoint.
        # Uses fixed-size hashing vectors for efficient precompute on large corpora.
        self.doc_embeddings = {}
        if not self.documents:
            return

        for doc in self.documents:
            vec = np.zeros(self.desm_dim, dtype=float)
            words = doc.tokens
            if not words:
                self.doc_embeddings[doc.document_id] = vec
                continue
            for word in words:
                idx = self._hash_term(word)
                vec[idx] += 1.0
            vec /= max(len(words), 1)
            self.doc_embeddings[doc.document_id] = vec

    def search_desm(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        if not self.doc_embeddings:
            return []

        query_vector = np.zeros(self.desm_dim, dtype=float)
        query_words = self.preprocessing(query)
        if not query_words:
            return []
        for word in query_words:
            idx = self._hash_term(word)
            query_vector[idx] += 1.0
        if not np.any(query_vector):
            return []
        query_vector /= max(len(query_words), 1)

        scores: List[Tuple[str, float]] = []
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            return []
        for doc_id, embedding in self.doc_embeddings.items():
            embedding_norm = np.linalg.norm(embedding)
            score = (
                float(np.dot(query_vector, embedding) / (query_norm * embedding_norm))
                if embedding_norm
                else 0.0
            )
            scores.append((doc_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def _hash_term(self, term: str) -> int:
        digest = hashlib.blake2b(term.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, byteorder="little", signed=False)
        return value % self.desm_dim

    def format_results(self, scored_docs: List[Tuple[str, float]]) -> List[Dict[str, str | float | int]]:
        output = []
        for idx, (doc_id, score) in enumerate(scored_docs, start=1):
            doc = self.doc_by_id.get(doc_id)
            snippet = ""
            if doc is not None:
                snippet = self._extract_body_for_snippet(doc.raw_body)
                if len(snippet) > 280:
                    snippet = snippet[:280] + "..."
            output.append(
                {
                    "rank": idx,
                    "document_id": doc_id,
                    "subject": getattr(doc, "subject", "(No Subject)") if doc is not None else "(No Subject)",
                    "score": float(score),
                    "snippet": snippet,
                }
            )
        return output

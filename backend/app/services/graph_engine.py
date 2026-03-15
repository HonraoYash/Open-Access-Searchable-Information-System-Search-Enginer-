import os
from collections import Counter
from typing import Dict, List, Set, Tuple

import networkx as nx
import pandas as pd


class GraphEngine:
    def __init__(self) -> None:
        self.graph = nx.Graph()
        self.total_emails = 0
        self.unique_emails: Set[str] = set()
        self.email_df = pd.DataFrame()

    @staticmethod
    def extract_email_addresses(file_path: str) -> Tuple[str | None, List[str]]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
            header: Dict[str, str] = {}
            lines = file.readlines()
            collecting_recipients = False
            to_emails: List[str] = []

            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    break

                key_value = line.split(":", 1)
                if len(key_value) == 2:
                    key, value = key_value
                    key = key.lower().strip()
                    value = value.strip().lower()
                    header[key] = value

                    # Keep original homework behavior that starts from "To" header.
                    if i == 3 and key == "to":
                        collecting_recipients = True
                        to_emails.extend(value.split(","))
                        for j in range(i + 1, len(lines)):
                            next_line = lines[j].strip()
                            if not next_line or ":" in next_line:
                                break
                            to_emails.extend(next_line.lower().split(","))
                elif collecting_recipients:
                    if line.startswith(" ") or line.startswith("\t"):
                        to_emails.extend(line.lower().split(","))
                    else:
                        collecting_recipients = False

        from_email = header.get("from", None)
        to_emails = [email.strip() for email in to_emails if email.strip()]
        return from_email, to_emails

    def process_email_dataset(self, folder_path: str) -> None:
        email_data = []
        total_emails = 0
        unique_emails: Set[str] = set()
        edges: Counter[Tuple[str, str]] = Counter()

        for root, _, files in os.walk(folder_path):
            parent_folder = os.path.basename(root)
            if parent_folder == "deleted_items":
                continue

            for file in files:
                if file.startswith("."):
                    continue
                file_path = os.path.join(root, file)
                if not os.path.isfile(file_path):
                    continue
                try:
                    from_email, to_emails = self.extract_email_addresses(file_path)
                    cleaned_to_emails = {email.strip() for email in to_emails if email.strip()}

                    if from_email and from_email not in cleaned_to_emails:
                        total_emails += 1
                        unique_emails.add(from_email)
                    unique_emails.update(cleaned_to_emails)

                    if from_email:
                        email_data.append({"from": from_email, "to": list(cleaned_to_emails)})
                        for target in cleaned_to_emails:
                            if target == from_email:
                                continue
                            edge = tuple(sorted((from_email, target)))
                            edges[edge] += 1
                except Exception:
                    continue

        self.total_emails = total_emails
        self.unique_emails = unique_emails
        self.email_df = pd.DataFrame(email_data)

        graph = nx.Graph()
        for user in unique_emails:
            graph.add_node(user)
        for (user_a, user_b), weight in edges.items():
            graph.add_edge(user_a, user_b, weight=weight)
        self.graph = graph

    def pagerank(self, top_k: int = 5) -> List[Tuple[str, float]]:
        if self.graph.number_of_nodes() == 0:
            return []
        scores = nx.pagerank(self.graph)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def personalized_pagerank(self, user: str, top_k: int = 5) -> List[Tuple[str, float]]:
        if self.graph.number_of_nodes() == 0:
            return []
        personalization = {node: 0.0 for node in self.graph.nodes}
        if user in personalization:
            personalization[user] = 1.0
        else:
            return []
        scores = nx.pagerank(self.graph, personalization=personalization)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def hits(self, top_k: int = 5) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        if self.graph.number_of_nodes() == 0:
            return [], []
        hubs, authorities = nx.hits(self.graph)
        top_hubs = sorted(hubs.items(), key=lambda x: x[1], reverse=True)[:top_k]
        top_authorities = sorted(authorities.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return top_hubs, top_authorities

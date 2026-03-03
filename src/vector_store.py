from typing import Dict, List, Any
import math


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a)) or 1.0
    norm_b = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (norm_a * norm_b)


class VectorStore:
    """
    Minimal in-memory vector store.

    Stores:
        - doc_id
        - chunk_id
        - embedding (list[float])
        - metadata (dict)
    """

    def __init__(self) -> None:
        self._entries: List[Dict[str, Any]] = []

    def add_documents(self, embedded_chunks: List[Dict]) -> None:
        for chunk in embedded_chunks:
            self._entries.append(
                {
                    "doc_id": chunk["doc_id"],
                    "chunk_id": chunk["chunk_id"],
                    "embedding": chunk.get("embedding", []),
                    "metadata": chunk.get("metadata", {}),
                    "text": chunk.get("text", ""),
                }
            )

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """
        Return top_k results sorted by cosine similarity.
        """
        scored: List[Dict[str, Any]] = []
        for entry in self._entries:
            score = _cosine_similarity(query_embedding, entry.get("embedding", []))
            scored.append(
                {
                    "doc_id": entry["doc_id"],
                    "chunk_id": entry["chunk_id"],
                    "score": float(score),
                    "metadata": dict(entry.get("metadata", {})),
                }
            )
        scored.sort(key=lambda r: r["score"], reverse=True)
        return scored[:top_k]


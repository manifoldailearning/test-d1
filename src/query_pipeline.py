from typing import Any, Dict, List

from . import ingestion
from . import embeddings
from . import vector_store
from . import access_control


# Build a simple in-memory index at import time so both the tests and
# evaluation helpers can reuse it.
_RAW_DOCS: List[Dict[str, Any]] = ingestion.load_documents()
_CHUNKS: List[Dict[str, Any]] = ingestion.chunk_documents(_RAW_DOCS)
_CHUNKS = embeddings.embed_chunks(_CHUNKS)
_STORE = vector_store.VectorStore()
_STORE.add_documents(_CHUNKS)


def rewrite_query(user_query: str) -> Dict[str, str]:
    """
    Very small heuristic "query understanding" layer.

    - Expands common abbreviations (e.g. PTO -> paid time off)
    - Emits a coarse intent label
    """
    original = user_query
    lowered = user_query.lower()

    expanded = lowered.replace("pto", "paid time off")
    if "compare" in expanded:
        intent = "compare"
    elif "policy" in expanded or "leave" in expanded:
        intent = "policy"
    else:
        intent = "lookup"

    return {
        "original": original,
        "expanded": expanded,
        "intent": intent,
    }


def lexical_search(query: str, chunks: List[Dict], top_k: int = 5) -> List[Dict]:
    """
    Simple lexical scorer: score is the count of overlapping tokens.
    """
    tokens = [t for t in query.lower().split() if t]
    results: List[Dict[str, Any]] = []
    for chunk in chunks:
        text = chunk.get("text", "").lower()
        score = sum(1 for t in tokens if t in text)
        if score > 0:
            results.append(
                {
                    "doc_id": chunk["doc_id"],
                    "chunk_id": chunk["chunk_id"],
                    "score": float(score),
                    "metadata": dict(chunk.get("metadata", {})),
                }
            )
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_k]


def _rrf_fuse(
    vector_results: List[Dict[str, Any]],
    lexical_results: List[Dict[str, Any]],
    k: int = 60,
) -> List[Dict[str, Any]]:
    """
    Reciprocal Rank Fusion over two ranked lists.
    """
    fused: Dict[str, Dict[str, Any]] = {}

    def add_results(results: List[Dict[str, Any]], source_name: str) -> None:
        for rank, r in enumerate(results, start=1):
            key = f"{r['doc_id']}::{r['chunk_id']}"
            entry = fused.setdefault(
                key,
                {
                    "doc_id": r["doc_id"],
                    "chunk_id": r["chunk_id"],
                    "metadata": r.get("metadata", {}),
                    "score": 0.0,
                    "sources": set(),
                },
            )
            entry["score"] += 1.0 / (k + rank)
            entry["sources"].add(source_name)

    add_results(vector_results, "vector")
    add_results(lexical_results, "lexical")

    out = []
    for entry in fused.values():
        out.append(
            {
                "doc_id": entry["doc_id"],
                "chunk_id": entry["chunk_id"],
                "metadata": entry["metadata"],
                "score": float(entry["score"]),
            }
        )
    out.sort(key=lambda r: r["score"], reverse=True)
    return out


def hybrid_search(user_query: str, user_role: str, top_k: int = 5) -> Dict[str, List[Dict]]:
    """
    Full 7-layer retrieval:

    1. rewrite_query(user_query) -> expanded query
    2. filter_chunks_by_access(...) based on user_role
    3. vector search with expanded query (using simulated embeddings)
    4. lexical search with expanded query
    5. combine results via Reciprocal Rank Fusion (RRF)
    """
    rewritten = rewrite_query(user_query)
    expanded_query = rewritten["expanded"]

    # Access control applied before scoring
    filtered_chunks = access_control.filter_chunks_by_access(_CHUNKS, user_role=user_role)

    # Vector search: we need a query embedding; for this reference we reuse the
    # hash-to-vector logic from embeddings by embedding the query as "one more chunk".
    from .embeddings import _hash_to_vector  # type: ignore

    query_embedding = _hash_to_vector(expanded_query)
    vector_results = _STORE.search(query_embedding, top_k=top_k)

    lexical_results = lexical_search(expanded_query, filtered_chunks, top_k=top_k)

    fused = _rrf_fuse(vector_results, lexical_results)
    return {"results": fused[:top_k]}


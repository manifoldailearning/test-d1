from typing import Dict, List
import math


def _hash_to_vector(text: str, dim: int = 8) -> List[float]:
    """
    Very small deterministic "embedding" based on text hashing.

    This avoids any external model dependency while still behaving like
    a numeric vector for similarity calculations.
    """
    base = sum(ord(c) for c in text)
    vec: List[float] = []
    for i in range(dim):
        # simple, deterministic pseudo-random-ish generation
        val = (base + 31 * i) % 100
        vec.append(val / 100.0)
    # normalise to unit length to behave like a cosine embedding
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def embed_chunks(chunks: List[Dict]) -> List[Dict]:
    """
    Compute (simulated) embeddings for each chunk.

    Returns the same list with an added 'embedding' field per chunk.
    """
    for chunk in chunks:
        text = chunk.get("text", "")
        chunk["embedding"] = _hash_to_vector(text)
    return chunks


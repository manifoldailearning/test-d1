from typing import Dict, List


ROLE_ORDER = {
    "employee": 0,
    "manager": 1,
    "admin": 2,
}


def filter_chunks_by_access(chunks: List[Dict], user_role: str) -> List[Dict]:
    """
    Filter chunks based on user_role and metadata["min_role"].

    A user can see chunks whose min_role is <= their own role.
    """
    user_rank = ROLE_ORDER.get(user_role, 0)
    allowed: List[Dict] = []
    for chunk in chunks:
        min_role = chunk.get("metadata", {}).get("min_role", "employee")
        min_rank = ROLE_ORDER.get(min_role, 0)
        if user_rank >= min_rank:
            allowed.append(chunk)
    return allowed


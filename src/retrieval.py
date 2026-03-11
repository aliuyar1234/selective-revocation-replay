from __future__ import annotations

from src.schemas import ObjectRecord
from src.utils import unique_tokens


def retrieve_active_memory(user_turn: str, active_memory: list[ObjectRecord], k: int, token_regex: str) -> list[ObjectRecord]:
    query_tokens = set(unique_tokens(user_turn, token_regex=token_regex))
    scored: list[tuple[int, str, ObjectRecord]] = []
    for obj in active_memory:
        overlap = len(query_tokens.intersection(set(unique_tokens(obj.content, token_regex=token_regex))))
        if overlap > 0:
            scored.append((overlap, obj.object_id, obj))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [item[2] for item in scored[:k]]

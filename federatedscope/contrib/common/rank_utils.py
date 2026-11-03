"""Rank and index helpers."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional


def resolve_client_key(config_local: Any, client_id: int) -> Optional[str]:
    """Resolve a client key from hetero config.

    Accepted key patterns:
    - 'Client_1' (1-indexed)
    - 'Client_0' (0-indexed)
    - a plain integer or string client id
    """
    if not config_local:
        return None

    preferred_keys = [
        f'Client_{client_id}',
        f'Client_{client_id - 1}',
        client_id,
        str(client_id),
        client_id - 1,
        str(client_id - 1),
    ]
    for key in preferred_keys:
        if key in config_local:
            return key
    return None


def infer_rank_from_client_rank_config(client_rank_config: Any, default: Optional[int] = None) -> Optional[int]:
    """Infer a logical rank from a module->rank mapping.

    Assumes one uniform rank across target modules and returns the first value;
    returns default when the mapping is empty or malformed.
    """
    if not client_rank_config:
        return default
    try:
        first_value = next(iter(client_rank_config.values()))
        return int(first_value)
    except Exception:
        return default


def indices_from_rank(rank: int) -> List[int]:
    """Return [0, 1, ..., rank-1] for a dense prefix rank."""
    rank = int(rank)
    if rank <= 0:
        return []
    return list(range(rank))


def normalize_indices(indices: Optional[Iterable[Any]], *, unique: bool = True, sort_values: bool = True) -> List[int]:
    """Convert any iterable of indices into a clean int list."""
    if indices is None:
        return []

    cleaned: List[int] = []
    seen = set()
    for value in indices:
        idx = int(value)
        if unique:
            if idx in seen:
                continue
            seen.add(idx)
        cleaned.append(idx)

    if sort_values:
        cleaned.sort()
    return cleaned



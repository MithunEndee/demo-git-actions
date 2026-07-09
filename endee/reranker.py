"""
Reranking utilities for the Endee vector database client.

Each reranker accepts:
  results_map   — {field_name: [[int32_id, score], ...]}  (from SearchResult)
  field_weights — {field_name: weight}  summing to 1.0
  limit         — final number of results to return (the global search limit,
                  NOT the per-field fetch limits)
  **kwargs      — reranker-specific parameters

Returns a list of (int32_id, score) tuples sorted by score descending,
truncated to `limit`.
"""

from typing import Any, Dict, List, Optional, Tuple

from .constants import DEFAULT_RRF_RANK_CONSTANT, DEFAULT_TOPK

def _resolve_field_weights(
    field_names: List[str],
    field_weights: Optional[Dict[str, float]],
) -> Dict[str, float]:
    """Resolve and validate per-field weights against the fields present."""
    if field_weights is None:
        n = len(field_names)
        return {f: 1.0 / n for f in field_names}
    missing = [f for f in field_names if f not in field_weights]
    if missing:
        raise ValueError(f"field_weights missing entries for: {sorted(missing)}")
    total = sum(field_weights[f] for f in field_names)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"field_weights must sum to 1.0 (got {total:.8f})")
    return field_weights


def rerank(
    search_results: Dict[str, Any],
    name: str = "rrf",
    limit: int = DEFAULT_TOPK,
    field_weights: Optional[Dict[str, float]] = None,
    rrf_k: int = DEFAULT_RRF_RANK_CONSTANT,
) -> Dict[str, Any]:
    """Fuse the per-field results from :meth:`Collection.search` into one list.

    Uses Reciprocal Rank Fusion over each field's decoded hits: a hit at rank
    ``r`` (1-based) in field ``f`` contributes ``field_weights[f] / (rrf_k + r)``
    to that object's score. Scores are summed across fields (deduped by ``id``)
    and the top ``limit`` hits are returned, sorted by fused score descending.

    Parameters
    ----------
    search_results : dict
        The return value of :meth:`Collection.search`, i.e.
        ``{"results": {field_name: [hit, ...], ...}}``.
    name : str
        Reranker algorithm. Only ``"rrf"`` is supported.
    limit : int
        Max number of fused hits to return (default ``DEFAULT_TOPK``).
    field_weights : dict, optional
        Per-field RRF weights e.g. ``{"embedding": 0.6, "keywords": 0.4}``.
        Must sum to 1.0. Defaults to uniform weighting.
    rrf_k : int
        RRF rank constant k (default 60).

    Returns
    -------
    dict  ``{"results": [hit, ...]}``  — each hit's ``similarity`` is its fused score.

    Examples
    --------
    >>> res = collection.search(fields={
    ...     "embedding": {"query": v, "limit": 50},
    ...     "keywords":  {"query": s, "limit": 50},
    ... })
    >>> fused = rerank(res, limit=10, field_weights={"embedding": 0.6, "keywords": 0.4})
    """
    if name != "rrf":
        raise ValueError("rerank: name must be 'rrf'")

    per_field = (search_results or {}).get("results")
    if not isinstance(per_field, dict):
        raise ValueError(
            "rerank: expected search results to be a per-field map of hits"
        )

    field_names = list(per_field.keys())
    if not field_names:
        raise ValueError("rerank: no fields to fuse")

    weights = _resolve_field_weights(field_names, field_weights)

    scores: Dict[str, float] = {}
    hit_by_id: Dict[str, dict] = {}
    for fname in field_names:
        weight = weights.get(fname, 0.0)
        for rank, hit in enumerate(per_field[fname] or [], start=1):
            hid = hit["id"]
            scores[hid] = scores.get(hid, 0.0) + weight / (rrf_k + rank)
            hit_by_id.setdefault(hid, hit)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]
    results = [{**hit_by_id[hid], "similarity": score} for hid, score in ranked]
    return {"results": results}

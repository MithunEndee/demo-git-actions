"""
Shared test helpers: vector generators, field config builders, object builders,
and utility functions used across the entire test suite.
"""

import json
import uuid

import numpy as np

from endee.schema import CollectionFieldConfig, CollectionFieldParams

DIM = 16  # Dense vector dimension for most tests
HYBRID_DIM = 16  # Dense dimension for hybrid tests
SPARSE_DIM = 500  # Vocabulary size for sparse vectors
SPARSE_NNZ = 8  # Non-zero terms per sparse vector
N_VECTORS = 50  # Objects inserted per populated fixture
MV_TOKENS = 4  # Number of vector entries per multi_vector object

DENSE_FIELD = "dense"  # Default dense vector field name
SPARSE_FIELD = "sparse"  # Default sparse vector field name
MV_FIELD = "colbert"  # Default multi_vector field name

ALL_PRECISIONS = ["float32", "float16", "int16", "int8", "int8e", "binary"]  # all supported precision modes
ALL_SPACE_TYPES = ["cosine", "l2", "ip"]  # all supported distance metrics


# -- Vector generators --------------------------------------------------------


def uid(prefix: str = "t") -> str:
    """Return a unique collection name that fits inside the 48-char limit."""
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def dense_vec(dim: int = DIM, seed: int | None = None) -> list[float]:
    """Random float vector of length `dim`."""
    rng = np.random.default_rng(seed)
    return rng.random(dim).tolist()


def binary_vec(dim: int = DIM, seed: int | None = None) -> list[float]:
    """0/1 float vector suitable for binary precision collections."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=dim).astype(float).tolist()


def sparse_vec(
    sparse_dim: int = SPARSE_DIM,
    nnz: int = SPARSE_NNZ,
    seed: int | None = None,
):
    """Return (sorted_indices, float_values) for a sparse vector."""
    rng = np.random.default_rng(seed)
    indices = sorted(rng.choice(sparse_dim, nnz, replace=False).tolist())
    values = rng.random(nnz).tolist()
    return indices, values


def multi_vec(
    n_tokens: int = MV_TOKENS,
    dim: int = DIM,
    seed: int | None = None,
) -> list[list[float]]:
    """Return a list of `n_tokens` dense vectors for multi_vector fields."""
    return [
        dense_vec(dim, seed=seed * n_tokens + t if seed is not None else None)
        for t in range(n_tokens)
    ]


# -- Field config builders ----------------------------------------------------


def make_dense_field(
    dim: int = DIM,
    space_type: str = "cosine",
    precision: str = "int8",
    m: int | None = 16,
    ef_construct: int | None = 128,
) -> dict:
    """Build a field config dict for a dense vector field."""
    params: dict = {"dimension": dim, "space_type": space_type, "precision": precision}
    if m is not None:
        params["M"] = m
    if ef_construct is not None:
        params["ef_con"] = ef_construct
    return {"name": DENSE_FIELD, "type": "vector", "params": params}


def make_sparse_field(sparse_model: str = "default") -> dict:
    """Build a field config dict for a sparse vector field."""
    return CollectionFieldConfig(
        name=SPARSE_FIELD,
        type="sparse",
        sparse_model=sparse_model,
    ).to_dict()


def make_mv_field(
    dim: int = DIM,
    space_type: str = "cosine",
    precision: str = "int8",
    pooling_method: str = "mean",
) -> dict:
    """Build a field config dict for a multi_vector (ColBERT-style) field."""
    return CollectionFieldConfig(
        name=MV_FIELD,
        type="multi_vector",
        pooling_method=pooling_method,
        params=CollectionFieldParams(
            dimension=dim,
            space_type=space_type,
            precision=precision,
        ),
    ).to_dict()


# -- Object builders ----------------------------------------------------------


def make_item(i: int, dim: int = DIM, with_sparse: bool = False) -> dict:
    """
    Build a deterministic object for position i (v2 Collections API format).

    Filter layout used by filter tests (N_VECTORS = 50):
      category : "A" | "B" | "C"       (i % 3)
      priority : 0..4                   (i % 5)
      score    : 0..49                  (i itself)
      tags     : "important" | "normal" (even/odd i)

    Expected counts with N_VECTORS = 50:
      category "A"           -> 17  (i % 3 == 0)
      category "B"           -> 17  (i % 3 == 1)
      category "C"           -> 16  (i % 3 == 2)
      tags "important"       -> 25  (even i)
      score in [10, 20]      -> 11  (i = 10..20)
      category "A" AND even  ->  9  (i = 0,6,12,18,24,30,36,42,48)
      category in ["A","B"]  -> 34
    """
    fields: dict = {DENSE_FIELD: dense_vec(dim, seed=i)}
    if with_sparse:
        indices, values = sparse_vec(seed=i)
        fields[SPARSE_FIELD] = {"indices": indices, "values": values}

    return {
        "id": f"vec_{i:04d}",
        "meta": {"index": i, "text": f"Document {i}"},
        "filter": {
            "category": ["A", "B", "C"][i % 3],
            "priority": i % 5,
            "score": i,
            "tags": "important" if i % 2 == 0 else "normal",
        },
        "fields": fields,
    }


def make_sparse_item(i: int) -> dict:
    """Build a deterministic sparse-only object for position i."""
    indices, values = sparse_vec(seed=i)
    return {
        "id": f"sp_{i:04d}",
        "meta": {"index": i, "text": f"Document {i}"},
        "filter": {
            "category": ["A", "B", "C"][i % 3],
            "score": i,
            "tags": "important" if i % 2 == 0 else "normal",
        },
        "fields": {SPARSE_FIELD: {"indices": indices, "values": values}},
    }


def make_mv_item(i: int, dim: int = DIM) -> dict:
    """Build a deterministic multi_vector object for position i."""
    return {
        "id": f"mv_{i:04d}",
        "meta": {"index": i, "text": f"Document {i}"},
        "filter": {
            "category": ["A", "B", "C"][i % 3],
            "score": i,
            "tags": "important" if i % 2 == 0 else "normal",
        },
        "fields": {MV_FIELD: multi_vec(dim=dim, seed=i)},
    }


# -- Utilities ----------------------------------------------------------------


def parse_filter_field(result: dict) -> dict:
    """
    Extract the filter dict from a search result.

    The server may return filter as a JSON string or as a dict.
    """
    flt = result.get("filter")
    if flt is None:
        return {}
    if isinstance(flt, str):
        return json.loads(flt)
    return flt


def safe_delete(client, name: str) -> None:
    """Delete a collection silently - used in fixture teardown."""
    try:
        client.delete_collection(name)
    except Exception:
        pass


def get_collection_names(client) -> list[str]:
    """Return a flat list of collection name strings from list_collections().

    Handles all known server response shapes:
      - list of dicts:   [{"name": "x", ...}, ...]
      - dict envelope:   {"collections": [...]}
      - list of strings: ["x", "y", ...]
    """
    response = client.list_collections()
    if isinstance(response, dict):
        items = response.get("collections", [])
    else:
        items = list(response)

    names = []
    for item in items:
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, dict) and "name" in item:
            names.append(item["name"])
    return names


# -- Search query builders ----------------------------------------------------


def q(vec, limit=N_VECTORS, *, ef_search=None):
    """Wrap a query vector in the per-field config dict required by search()."""
    cfg = {"query": vec, "limit": limit}
    if ef_search is not None:
        cfg["ef_search"] = ef_search
    return cfg


def q_sparse(indices, values, limit=N_VECTORS, *, ef_search=None):
    """Wrap a sparse query in the per-field config dict required by search()."""
    cfg = {"query": {"indices": indices, "values": values}, "limit": limit}
    if ef_search is not None:
        cfg["ef_search"] = ef_search
    return cfg


def results(search_response, field):
    """Return a single field's results list from a search() response."""
    return search_response["results"][field]

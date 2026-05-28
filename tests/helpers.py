"""
Shared constants and helper functions for integration tests.

Import this module directly in test files:
    from helpers import DIM, uid, dense_vec, ...

Pytest fixtures should NOT be writtn here - those belong in conftest.py.
"""

import uuid

import numpy as np

# === Test-wide constants ===
DIM = 16          # Dense vector dimension for most tests
HYBRID_DIM = 16   # Dense dimension for hybrid tests
SPARSE_DIM = 500  # Vocabulary size for sparse vectors
SPARSE_NNZ = 8    # Non-zero terms per sparse vector
N_VECTORS = 50    # Vectors inserted per populated fixture


# === Vector / item generators ===

def uid(prefix: str = "t") -> str:
    """Return a unique index name that fits inside the 48-char limit."""
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def dense_vec(dim: int = DIM, seed: int | None = None) -> list[float]:
    """Random float vector of length `dim`."""
    rng = np.random.default_rng(seed)
    return rng.random(dim).tolist()


def binary_vec(dim: int = DIM, seed: int | None = None) -> list[float]:
    """0/1 float vector suitable for BINARY2 precision indexes."""
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


def make_item(i: int, dim: int = DIM, with_sparse: bool = False) -> dict:
    """
    Build a deterministic vector item for position i.

    Filter layout used by filter tests (N_VECTORS = 50):
      category : "A" | "B" | "C"    (i % 3)
      priority : 0..4               (i % 5)
      score    : 0..49              (i itself — fits in $range [0,999])
      tags     : "important"|"normal"  (even/odd i)

    Expected counts with N_VECTORS = 50:
      category "A"            → 17   (i % 3 == 0)
      category "B"            → 17   (i % 3 == 1)
      category "C"            → 16   (i % 3 == 2)
      tags "important"        → 25   (even i)
      score in [10, 20]       → 11   (i = 10..20)
      category "A" AND even   →  9   (i = 0,6,12,18,24,30,36,42,48)
      category in ["A","B"]   → 34
    """
    item: dict = {
        "id": f"vec_{i:04d}",
        "vector": dense_vec(dim, seed=i),
        "meta": {"index": i, "text": f"Document {i}"},
        "filter": {
            "category": ["A", "B", "C"][i % 3],
            "priority": i % 5,
            "score": i,
            "tags": "important" if i % 2 == 0 else "normal",
        },
    }
    if with_sparse:
        indices, values = sparse_vec(seed=i)
        item["sparse_indices"] = indices
        item["sparse_values"] = values
    return item


def safe_delete(client, name: str) -> None:
    """Delete an index silently – used in fixture teardown."""
    try:
        client.delete_index(name)
    except Exception:
        pass


def get_index_names(client) -> list[str]:
    """Return a flat list of index name strings from list_indexes().

    Handles all known server response shapes:
      - list of dicts:  [{"name": "x", "M": 16, ...}, ...]
      - dict envelope:  {"indexes": ["x", ...]} or {"indexes": [{"name": "x"}, ...]}
      - list of strings: ["x", "y", ...]
    """
    response = client.list_indexes()
    if isinstance(response, dict):
        items = response.get("indexes", [])
    else:
        items = list(response)

    names = []
    for item in items:
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, dict) and "name" in item:
            names.append(item["name"])
    return names

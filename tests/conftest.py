"""
Shared fixtures and helpers for Endee integration tests.

Environment variables consumed:
  ENDEE_TOKEN    – API token (omit for OSS/local mode)
  ENDEE_BASE_URL – Override base URL (e.g. http://0.0.0.0:8081/api/v1)
"""

import os
import uuid

import numpy as np
import pytest

from endee import Endee, Precision

# ── Test-wide constants ────────────────────────────────────────────────────
DIM = 16          # Dense vector dimension for most tests
HYBRID_DIM = 16   # Dense dimension for hybrid tests
SPARSE_DIM = 500  # Vocabulary size for sparse vectors
SPARSE_NNZ = 8    # Non-zero terms per sparse vector
N_VECTORS = 50    # Vectors inserted per populated fixture


# ── Helper functions (importable by test modules) ─────────────────────────

def uid(prefix: str = "t") -> str:
    """Return a short unique index name safe for the 48-char limit."""
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def dense_vec(dim: int = DIM, seed: int | None = None) -> list[float]:
    rng = np.random.default_rng(seed)
    return rng.random(dim).tolist()


def binary_vec(dim: int = DIM, seed: int | None = None) -> list[float]:
    """0/1 float vector suitable for BINARY2 precision indexes."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=dim).astype(float).tolist()


def sparse_vec(sparse_dim: int = SPARSE_DIM, nnz: int = SPARSE_NNZ, seed: int | None = None):
    """Return (sorted_indices, values) for a sparse vector."""
    rng = np.random.default_rng(seed)
    indices = sorted(rng.choice(sparse_dim, nnz, replace=False).tolist())
    values = rng.random(nnz).tolist()
    return indices, values


def make_item(i: int, dim: int = DIM, with_sparse: bool = False) -> dict:
    """
    Build a deterministic vector item for position i.

    Filter layout (used by filter tests):
      category : "A" | "B" | "C"   (i % 3)
      priority : 0..4               (i % 5)
      score    : 0..N_VECTORS-1     (i itself – within $range [0,999])
      tags     : "important" | "normal"  (i % 2)

    Expected counts when N_VECTORS=50:
      category "A"           → 17  (i % 3 == 0)
      category "B"           → 17  (i % 3 == 1)
      category "C"           → 16  (i % 3 == 2)
      tags "important"       → 25  (even i)
      score in [10,20]       → 11  (i = 10..20)
      category "A" AND even  →  9  (i = 0,6,12,18,24,30,36,42,48)
      category in ["A","B"]  → 34
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


def safe_delete(client: Endee, name: str) -> None:
    """Delete an index silently – used in fixture teardown."""
    try:
        client.delete_index(name)
    except Exception:
        pass


# ── Session-scoped client ─────────────────────────────────────────────────

@pytest.fixture(scope="session")
def client() -> Endee:
    """One Endee client shared across the entire test session."""
    token = os.environ.get("ENDEE_TOKEN") or None
    base_url = os.environ.get("ENDEE_BASE_URL") or None
    c = Endee(token=token)
    if base_url:
        c.set_base_url(base_url)
    yield c


# ── Function-scoped index fixtures ────────────────────────────────────────

@pytest.fixture
def empty_index(client):
    """
    Yield (name, index) for a fresh cosine + INT8 dense index.
    Index is deleted on teardown even if the test fails.
    """
    name = uid("t")
    client.create_index(
        name=name,
        dimension=DIM,
        space_type="cosine",
        precision=Precision.INT8,
    )
    index = client.get_index(name)
    yield name, index
    safe_delete(client, name)


@pytest.fixture
def populated_index(client, empty_index):
    """
    Yield (name, index) with N_VECTORS deterministic vectors already upserted.
    Inherits teardown from empty_index.
    """
    name, index = empty_index
    index.upsert([make_item(i) for i in range(N_VECTORS)])
    yield name, index


@pytest.fixture
def empty_hybrid_index(client):
    """
    Yield (name, index) for a fresh hybrid (cosine + INT8 + sparse_model=default) index.
    """
    name = uid("h")
    client.create_index(
        name=name,
        dimension=HYBRID_DIM,
        space_type="cosine",
        precision=Precision.INT8,
        sparse_model="default",
    )
    index = client.get_index(name)
    yield name, index
    safe_delete(client, name)


@pytest.fixture
def populated_hybrid_index(client, empty_hybrid_index):
    """
    Yield (name, index) hybrid index with N_VECTORS deterministic hybrid vectors.
    """
    name, index = empty_hybrid_index
    index.upsert([make_item(i, dim=HYBRID_DIM, with_sparse=True) for i in range(N_VECTORS)])
    yield name, index

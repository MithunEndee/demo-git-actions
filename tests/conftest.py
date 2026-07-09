"""
Shared pytest fixtures for Endee functional tests (v2 Collections API).

Environment variables:
  ENDEE_TOKEN    - Required. Endee API token.
  ENDEE_BASE_URL - Optional. Override base URL (e.g. http://localhost:8080/api/v2)
"""

import os
import re

import pytest
from helpers import (
    HYBRID_DIM,
    N_VECTORS,
    get_collection_names,
    make_dense_field,
    make_item,
    make_mv_field,
    make_mv_item,
    make_sparse_field,
    make_sparse_item,
    safe_delete,
    uid,
)

from endee import Endee

_TEST_COLLECTION_PATTERN = re.compile(r"^[a-z]+_[0-9a-f]{10}$")


@pytest.fixture(scope="session", autouse=True)
def verify_server_and_cleanup():
    """Fail fast if ENDEE_TOKEN is missing or the server is unreachable, then
    remove stale test collections left from previous interrupted runs."""
    token = os.environ.get("ENDEE_TOKEN")
    if not token:
        pytest.exit("ENDEE_TOKEN is required to run the test suite", returncode=1)
    base_url = os.environ.get("ENDEE_BASE_URL") or None
    c = Endee(token=token)
    if base_url:
        c.set_base_url(base_url)

    try:
        existing = get_collection_names(c)
    except Exception as e:
        pytest.exit(f"Server unreachable - aborting test session: {e}", returncode=1)

    stale = [n for n in existing if _TEST_COLLECTION_PATTERN.match(n)]
    for name in stale:
        safe_delete(c, name)

    yield


@pytest.fixture(scope="session")
def client() -> Endee:
    """One Endee client shared across the entire test session."""
    token = os.environ.get("ENDEE_TOKEN")
    base_url = os.environ.get("ENDEE_BASE_URL") or None
    c = Endee(token=token)
    if base_url:
        c.set_base_url(base_url)
    yield c


# -- Dense collection fixtures ------------------------------------------------


@pytest.fixture
def empty_collection(client):
    """Yield (name, collection) for a fresh cosine + INT8 dense collection."""
    name = uid("t")
    client.create_collection(name=name, fields=[make_dense_field()])
    collection = client.get_collection(name)
    yield name, collection
    safe_delete(client, name)


@pytest.fixture
def populated_collection(client, empty_collection):
    """Yield (name, collection) with N_VECTORS deterministic objects upserted."""
    name, collection = empty_collection
    collection.upsert([make_item(i) for i in range(N_VECTORS)])
    yield name, collection


# -- Hybrid collection fixtures -----------------------------------------------


@pytest.fixture
def empty_hybrid_collection(client):
    """Yield (name, collection) for a fresh hybrid (dense + sparse) collection."""
    name = uid("h")
    client.create_collection(
        name=name,
        fields=[make_dense_field(dim=HYBRID_DIM), make_sparse_field()],
    )
    collection = client.get_collection(name)
    yield name, collection
    safe_delete(client, name)


@pytest.fixture
def populated_hybrid_collection(client, empty_hybrid_collection):
    """Yield (name, collection) hybrid collection with N_VECTORS objects."""
    name, collection = empty_hybrid_collection
    collection.upsert(
        [make_item(i, dim=HYBRID_DIM, with_sparse=True) for i in range(N_VECTORS)]
    )
    yield name, collection


# -- Sparse collection fixtures -----------------------------------------------


@pytest.fixture
def empty_sparse_collection(client):
    """Yield (name, collection) for a fresh sparse-only collection."""
    name = uid("sp")
    client.create_collection(name=name, fields=[make_sparse_field()])
    collection = client.get_collection(name)
    yield name, collection
    safe_delete(client, name)


@pytest.fixture
def populated_sparse_collection(client, empty_sparse_collection):
    """Yield (name, collection) sparse collection with N_VECTORS objects."""
    name, collection = empty_sparse_collection
    collection.upsert([make_sparse_item(i) for i in range(N_VECTORS)])
    yield name, collection


# -- Multi-vector collection fixtures -----------------------------------------


@pytest.fixture
def empty_mv_collection(client):
    """Yield (name, collection) for a fresh multi_vector (ColBERT-style) collection."""
    name = uid("mv")
    client.create_collection(name=name, fields=[make_mv_field()])
    collection = client.get_collection(name)
    yield name, collection
    safe_delete(client, name)


@pytest.fixture
def populated_mv_collection(client, empty_mv_collection):
    """Yield (name, collection) multi_vector collection with N_VECTORS objects."""
    name, collection = empty_mv_collection
    collection.upsert([make_mv_item(i) for i in range(N_VECTORS)])
    yield name, collection

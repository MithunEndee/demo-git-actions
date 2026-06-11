"""
Shared pytest fixtures for Endee functional tests.

Environment variables:
  ENDEE_TOKEN    - Required. Endee Serverless API token.
  ENDEE_BASE_URL - Optional. Override base URL (e.g. http://localhost:8080/api/v1)
"""

import re
import os
import pytest

from endee import Endee, Precision
from helpers import (
    DIM,
    HYBRID_DIM,
    N_VECTORS,
    get_index_names,
    make_item,
    safe_delete,
    uid,
)


_TEST_INDEX_PATTERN = re.compile(r"^[a-z]+_[0-9a-f]{10}$")


@pytest.fixture(scope="session", autouse=True)
def verify_server_and_cleanup():
    """Fail fast if ENDEE_TOKEN is missing or the server is unreachable, then remove stale test indexes."""
    token = os.environ.get("ENDEE_TOKEN")
    if not token:
        pytest.exit("ENDEE_TOKEN is required to run the test suite", returncode=1)
    base_url = os.environ.get("ENDEE_BASE_URL") or None
    c = Endee(token=token)
    if base_url:
        c.set_base_url(base_url)

    try:
        existing = get_index_names(c)
    except Exception as e:
        pytest.exit(f"Server unreachable - aborting test session: {e}", returncode=1)

    stale = [n for n in existing if _TEST_INDEX_PATTERN.match(n)]
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


@pytest.fixture
def empty_index(client):
    """Yield (name, index) for a fresh cosine + INT8 dense index."""
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
    """Yield (name, index) with N_VECTORS deterministic vectors already upserted."""
    name, index = empty_index
    index.upsert([make_item(i) for i in range(N_VECTORS)])
    yield name, index


@pytest.fixture
def empty_hybrid_index(client):
    """Yield (name, index) for a fresh hybrid (cosine + INT8 + sparse) index."""
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
    """Yield (name, index) hybrid index with N_VECTORS vectors."""
    name, index = empty_hybrid_index
    index.upsert(
        [make_item(i, dim=HYBRID_DIM, with_sparse=True) for i in range(N_VECTORS)]
    )
    yield name, index

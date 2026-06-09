"""
Shared pytest fixtures for Endee functional tests.

Environment variables:
  ENDEE_TOKEN    - API token (omit for OSS/local mode)
  ENDEE_BASE_URL - Override base URL (e.g. http://0.0.0.0:8080/api/v1)
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


def pytest_collection_modifyitems(items):
    """Auto-skip @pytest.mark.serverless tests when no API token is provided."""
    if os.environ.get("ENDEE_TOKEN"):
        return
    skip = pytest.mark.skip(reason="Requires ENDEE_TOKEN - serverless only")
    for item in items:
        if item.get_closest_marker("serverless"):
            item.add_marker(skip)


@pytest.fixture(scope="session", autouse=True)
def verify_server_and_cleanup():
    """Fail fast if the server is unreachable, then remove stale test indexes."""
    token = os.environ.get("ENDEE_TOKEN") or None
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
    token = os.environ.get("ENDEE_TOKEN") or None
    base_url = os.environ.get("ENDEE_BASE_URL") or None
    c = Endee(token=token)
    if base_url:
        c.set_base_url(base_url)
    yield c


@pytest.fixture
def empty_index(client):
    """
    Yield (name, index) for a fresh cosine + INT8 dense index.
    Deleted on teardown even if the test raises.
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

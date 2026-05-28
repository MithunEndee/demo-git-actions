"""
test_08_http_libraries.py

Tests that the full client API works correctly with each supported
HTTP backend: requests (default), httpx1.1, and httpx2.

Each test creates its own isolated index to avoid interference.
"""

import os

import pytest

from endee import Endee, Precision
from endee.constants import HTTP_HTTPX_1_1_LIBRARY, HTTP_HTTPX_2_LIBRARY, HTTP_REQUESTS_LIBRARY

from helpers import DIM, dense_vec, get_index_names, safe_delete, uid


def make_client(library: str) -> Endee:
    # Use "local" as a fallback so httpx never receives None as a header value.
    # OSS server with NDD_AUTH_TOKEN="" accepts any non-empty string.
    token = os.environ.get("ENDEE_TOKEN") or "local"
    base_url = os.environ.get("ENDEE_BASE_URL") or None
    c = Endee(token=token, http_library=library)
    if base_url:
        c.set_base_url(base_url)
    return c


# ── Shared scenario: create → upsert → query → delete ────────────────────

def _run_full_scenario(library: str) -> None:
    """
    Full CRUD smoke test using the given HTTP library.
    Creates a temporary index, inserts vectors, queries, then deletes.
    """
    client = make_client(library)
    name = uid(f"lib{library.replace('.', '')}")
    try:
        # Create
        result = client.create_index(
            name=name,
            dimension=DIM,
            space_type="cosine",
            precision=Precision.INT8,
        )
        assert "success" in result.lower()

        # Get index reference
        index = client.get_index(name)
        assert index.dimension == DIM

        # Upsert 10 vectors
        batch = [
            {"id": f"v{i}", "vector": dense_vec(seed=i),
             "meta": {"i": i}, "filter": {"n": i % 3}}
            for i in range(10)
        ]
        upsert_result = index.upsert(batch)
        assert "success" in upsert_result.lower()

        # Query
        results = index.query(vector=dense_vec(seed=99), top_k=5)
        assert isinstance(results, list)
        assert len(results) > 0
        assert "id" in results[0]

        # get_vector
        vec = index.get_vector("v0")
        assert vec["id"] == "v0"

        # describe
        info = index.describe()
        assert info["name"] == name

        # list_indexes
        assert name in get_index_names(client)

        # delete_vector
        del_result = index.delete_vector("v0")
        assert "deleted" in del_result.lower()

        # delete_index
        del_idx_result = client.delete_index(name)
        assert name in del_idx_result

    except Exception:
        safe_delete(client, name)
        raise


# ── Per-library tests ─────────────────────────────────────────────────────

def test_requests_library_full_scenario():
    """Default requests library works end-to-end."""
    _run_full_scenario(HTTP_REQUESTS_LIBRARY)


def test_httpx_11_library_full_scenario():
    """httpx HTTP/1.1 library works end-to-end."""
    _run_full_scenario(HTTP_HTTPX_1_1_LIBRARY)


@pytest.mark.xfail(
    reason="Client bug: ClientManager is initialised with http2=True but the "
           "parameter name is enable_http2. Fix in endee/endee.py line ~311.",
    strict=True,
)
def test_httpx2_library_full_scenario():
    """httpx HTTP/2 library works end-to-end."""
    _run_full_scenario(HTTP_HTTPX_2_LIBRARY)


# ── Session / client cleanup ──────────────────────────────────────────────

def test_requests_close_session():
    """close_session() must not raise."""
    c = make_client(HTTP_REQUESTS_LIBRARY)
    c.close_session()   # should not raise
    c.close_session()   # idempotent


def test_httpx_11_close_client():
    """close_client() must not raise for httpx1.1."""
    c = make_client(HTTP_HTTPX_1_1_LIBRARY)
    c.close_client()
    c.close_client()    # idempotent


@pytest.mark.xfail(
    reason="Client bug: ClientManager initialised with http2=True instead of enable_http2=True.",
    strict=True,
)
def test_httpx2_close_client():
    """close_client() must not raise for httpx2."""
    c = make_client(HTTP_HTTPX_2_LIBRARY)
    c.close_client()
    c.close_client()    # idempotent


# ── set_base_url affects all libraries ────────────────────────────────────

@pytest.mark.parametrize("library", [
    HTTP_REQUESTS_LIBRARY,
    HTTP_HTTPX_1_1_LIBRARY,
    pytest.param(
        HTTP_HTTPX_2_LIBRARY,
        marks=pytest.mark.xfail(
            reason="Client bug: ClientManager initialised with http2=True instead of enable_http2=True.",
            strict=True,
        ),
    ),
])
def test_set_base_url_updates_attribute(library):
    c = make_client(library)
    new_url = "http://custom.example.com:9090/api/v1"
    c.set_base_url(new_url)
    assert c.base_url == new_url


# ── Mixed-library index operations ───────────────────────────────────────

def test_index_created_with_requests_readable_with_httpx(client):
    """An index created via requests can be read via httpx1.1."""
    name = uid("xlib")
    try:
        # Create via requests (default client fixture)
        client.create_index(name=name, dimension=DIM, space_type="cosine",
                            precision=Precision.INT8)

        # Read via httpx1.1
        httpx_client = make_client(HTTP_HTTPX_1_1_LIBRARY)
        index = httpx_client.get_index(name)
        assert index.name == name
        assert index.dimension == DIM
    finally:
        safe_delete(client, name)

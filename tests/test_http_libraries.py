"""
Tests for all supported HTTP backends: requests (default), httpx HTTP/1.1,
and httpx HTTP/2.

Covers core operations (create, upsert, search, delete) across all three backends.
"""

import os

import pytest
from helpers import DENSE_FIELD, dense_vec, make_dense_field, safe_delete, uid

from endee import Endee

# -- client construction -------------------------------------------------------


def test_endee_default_library_is_requests():
    """Default Endee client must use the 'requests' library."""
    c = Endee(token="user:token:region")
    assert c.library == "requests"


def test_endee_httpx11_library_accepted():
    """Endee(http_library='httpx1.1') must be instantiated without error."""
    c = Endee(token="user:token:region", http_library="httpx1.1")
    assert c.library == "httpx1.1"


def test_endee_httpx2_library_accepted():
    """Endee(http_library='httpx2') must be instantiated without error."""
    c = Endee(token="user:token:region", http_library="httpx2")
    assert c.library == "httpx2"


def test_endee_invalid_library_raises():
    """Endee(http_library='curl') must raise ValueError."""
    with pytest.raises(ValueError, match="[Uu]nsupported"):
        Endee(token="user:token:region", http_library="curl")


# -- helper: build a configured client ----------------------------------------


def _make_client(http_library: str) -> Endee:
    """Create an Endee client using the given HTTP library with test credentials."""
    token = os.environ.get("ENDEE_TOKEN")
    base_url = os.environ.get("ENDEE_BASE_URL")
    c = Endee(token=token, http_library=http_library)
    if base_url:
        c.set_base_url(base_url)
    return c


# -- httpx1.1 -----------------------------------------------------------------


def test_httpx11_health():
    """health() must succeed via httpx1.1."""
    c = _make_client("httpx1.1")
    result = c.health()
    assert isinstance(result, dict)
    assert "status" in result


def test_httpx11_list_collections_returns_list():
    """list_collections() must return a list via httpx1.1."""
    c = _make_client("httpx1.1")
    result = c.list_collections()
    assert isinstance(result, list)


def test_httpx11_create_search_delete_collection():
    """Full create -> upsert -> search -> delete cycle must work via httpx1.1."""
    c = _make_client("httpx1.1")
    name = uid("hx1")
    try:
        c.create_collection(name=name, fields=[make_dense_field()])
        col = c.get_collection(name)
        col.upsert(
            [
                {"id": f"v{i}", "fields": {DENSE_FIELD: dense_vec(seed=i)}}
                for i in range(5)
            ]
        )
        results = col.search(
            fields={DENSE_FIELD: {"query": dense_vec(seed=0), "limit": 3}},
        )["results"][DENSE_FIELD]
        assert len(results) == 3
        for r in results:
            assert "id" in r and "similarity" in r
    finally:
        safe_delete(c, name)


def test_httpx11_upsert_returns_count():
    """upsert() must return the correct count via httpx1.1."""
    c = _make_client("httpx1.1")
    name = uid("hx1u")
    try:
        c.create_collection(name=name, fields=[make_dense_field()])
        col = c.get_collection(name)
        result = col.upsert(
            [
                {"id": f"v{i}", "fields": {DENSE_FIELD: dense_vec(seed=i)}}
                for i in range(10)
            ]
        )
        assert result["upserted"] == 10
    finally:
        safe_delete(c, name)


def test_httpx11_delete_collection():
    """delete_collection() must succeed via httpx1.1."""
    c = _make_client("httpx1.1")
    name = uid("hx1d")
    c.create_collection(name=name, fields=[make_dense_field()])
    result = c.delete_collection(name)
    assert isinstance(result, dict)


def test_httpx11_get_objects_round_trip():
    """get_objects() must return correct data via httpx1.1."""
    c = _make_client("httpx1.1")
    name = uid("hx1go")
    try:
        c.create_collection(name=name, fields=[make_dense_field()])
        col = c.get_collection(name)
        col.upsert(
            [
                {
                    "id": "go1",
                    "meta": {"k": "v"},
                    "fields": {DENSE_FIELD: dense_vec(seed=1)},
                }
            ]
        )
        objs = col.get_objects(["go1"])
        assert len(objs) == 1
        assert objs[0]["id"] == "go1"
        assert objs[0]["meta"]["k"] == "v"
    finally:
        safe_delete(c, name)


# -- httpx2 -------------------------------------------------------------------


def test_httpx2_health():
    """health() must succeed via httpx2."""
    c = _make_client("httpx2")
    result = c.health()
    assert isinstance(result, dict)
    assert "status" in result


def test_httpx2_list_collections_returns_list():
    """list_collections() must return a list via httpx2."""
    c = _make_client("httpx2")
    result = c.list_collections()
    assert isinstance(result, list)


def test_httpx2_create_search_delete_collection():
    """Full create -> upsert -> search -> delete cycle must work via httpx2."""
    c = _make_client("httpx2")
    name = uid("hx2")
    try:
        c.create_collection(name=name, fields=[make_dense_field()])
        col = c.get_collection(name)
        col.upsert(
            [
                {"id": f"v{i}", "fields": {DENSE_FIELD: dense_vec(seed=i)}}
                for i in range(5)
            ]
        )
        results = col.search(
            fields={DENSE_FIELD: {"query": dense_vec(seed=0), "limit": 3}},
        )["results"][DENSE_FIELD]
        assert len(results) == 3
        for r in results:
            assert "id" in r and "similarity" in r
    finally:
        safe_delete(c, name)


def test_httpx2_upsert_returns_count():
    """upsert() must return the correct count via httpx2."""
    c = _make_client("httpx2")
    name = uid("hx2u")
    try:
        c.create_collection(name=name, fields=[make_dense_field()])
        col = c.get_collection(name)
        result = col.upsert(
            [
                {"id": f"v{i}", "fields": {DENSE_FIELD: dense_vec(seed=i)}}
                for i in range(10)
            ]
        )
        assert result["upserted"] == 10
    finally:
        safe_delete(c, name)


def test_httpx2_delete_collection():
    """delete_collection() must succeed via httpx2."""
    c = _make_client("httpx2")
    name = uid("hx2d")
    c.create_collection(name=name, fields=[make_dense_field()])
    result = c.delete_collection(name)
    assert isinstance(result, dict)


def test_httpx2_get_objects_round_trip():
    """get_objects() must return correct data via httpx2."""
    c = _make_client("httpx2")
    name = uid("hx2go")
    try:
        c.create_collection(name=name, fields=[make_dense_field()])
        col = c.get_collection(name)
        col.upsert(
            [
                {
                    "id": "go2",
                    "meta": {"k": "v2"},
                    "fields": {DENSE_FIELD: dense_vec(seed=2)},
                }
            ]
        )
        objs = col.get_objects(["go2"])
        assert len(objs) == 1
        assert objs[0]["id"] == "go2"
        assert objs[0]["meta"]["k"] == "v2"
    finally:
        safe_delete(c, name)


# -- session/client manager lifecycle -----------------------------------------


def test_requests_close_session_does_not_raise():
    """close_session() on a requests client must not raise."""
    c = Endee(token="user:token:region", http_library="requests")
    c.close_session()  # must not raise


def test_httpx11_close_client_does_not_raise():
    """close_client() on an httpx1.1 client must not raise."""
    c = Endee(token="user:token:region", http_library="httpx1.1")
    c.close_client()  # must not raise


def test_httpx2_close_client_does_not_raise():
    """close_client() on an httpx2 client must not raise."""
    c = Endee(token="user:token:region", http_library="httpx2")
    c.close_client()  # must not raise


# -- set_base_url --------------------------------------------------------------


def test_set_base_url_updates_stored_url():
    """set_base_url() must update the base_url attribute."""
    c = Endee(token="user:token:region")
    c.set_base_url("http://localhost:9999/api/v2")
    assert c.base_url == "http://localhost:9999/api/v2"


def test_endee_str_returns_token():
    """str(client) must return the authentication token string."""
    c = Endee(token="user:mytoken")
    assert str(c) == "user:mytoken"

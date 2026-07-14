"""
Tests for admin methods requiring NDD_ROOT_TOKEN.

Covers database lifecycle (create, get, list, delete), activate/deactivate,
tier changes, cross-database collection listing, and admin token management.
Skipped automatically if NDD_ROOT_TOKEN is not set.
"""

import os

import pytest
from helpers import make_dense_field, uid

from endee import Endee

pytestmark = pytest.mark.skipif(
    not os.environ.get("NDD_ROOT_TOKEN"),
    reason="NDD_ROOT_TOKEN not set - admin tests skipped",
)


# -- fixtures -----------------------------------------------------------------


@pytest.fixture(scope="module")
def admin_client():
    """Endee client authenticated with the root token."""
    token = os.environ["NDD_ROOT_TOKEN"]
    base_url = os.environ.get("ENDEE_BASE_URL") or None
    c = Endee(token=token)
    if base_url:
        c.set_base_url(base_url)
    return c


@pytest.fixture(scope="module")
def temp_db(admin_client):
    """One shared database for the whole module, deleted at teardown."""
    name = uid("adb")
    db_token = admin_client.create_database(name)
    yield name, db_token
    try:
        admin_client.delete_database(name)
    except Exception:
        pass


def _db_client(db_token: str) -> Endee:
    """Build an Endee client for a specific database token."""
    base_url = os.environ.get("ENDEE_BASE_URL") or None
    c = Endee(token=db_token)
    if base_url:
        c.set_base_url(base_url)
    return c


# -- client-side validation ---------------------------------------------------


def test_delete_database_empty_name_raises():
    """delete_database with an empty db_name must raise ValueError."""
    c = Endee(token=os.environ["NDD_ROOT_TOKEN"])
    with pytest.raises(ValueError):
        c.delete_database("")


def test_set_database_type_invalid_type_raises():
    """set_database_type with an unrecognised db_type must raise ValueError."""
    c = Endee(token=os.environ["NDD_ROOT_TOKEN"])
    with pytest.raises(ValueError):
        c.set_database_type("some_db", "ultra")


def test_create_token_empty_db_name_raises():
    """create_token with an empty db_name must raise ValueError."""
    c = Endee(token=os.environ["NDD_ROOT_TOKEN"])
    with pytest.raises(ValueError):
        c.create_token("", "tok")


def test_create_token_empty_token_name_raises():
    """create_token with an empty token name must raise ValueError."""
    c = Endee(token=os.environ["NDD_ROOT_TOKEN"])
    with pytest.raises(ValueError):
        c.create_token("some_db", "")


def test_delete_token_empty_db_name_raises():
    """delete_token with an empty db_name must raise ValueError."""
    c = Endee(token=os.environ["NDD_ROOT_TOKEN"])
    with pytest.raises(ValueError):
        c.delete_token("", "tok")


def test_delete_token_empty_token_name_raises():
    """delete_token with an empty token name must raise ValueError."""
    c = Endee(token=os.environ["NDD_ROOT_TOKEN"])
    with pytest.raises(ValueError):
        c.delete_token("some_db", "")


def test_list_db_collections_empty_name_raises():
    """list_db_collections with an empty db_name must raise ValueError."""
    c = Endee(token=os.environ["NDD_ROOT_TOKEN"])
    with pytest.raises(ValueError):
        c.list_db_collections("")


def test_delete_db_collection_empty_db_name_raises():
    """delete_db_collection with an empty db_name must raise ValueError."""
    c = Endee(token=os.environ["NDD_ROOT_TOKEN"])
    with pytest.raises(ValueError):
        c.delete_db_collection("", "collection_name")


def test_delete_db_collection_empty_collection_name_raises():
    """delete_db_collection with an empty collection_name must raise ValueError."""
    c = Endee(token=os.environ["NDD_ROOT_TOKEN"])
    with pytest.raises(ValueError):
        c.delete_db_collection("some_db", "")


def test_get_database_empty_name_raises():
    """get_database with an empty db_name must raise ValueError."""
    c = Endee(token=os.environ["NDD_ROOT_TOKEN"])
    with pytest.raises(ValueError):
        c.get_database("")


def test_list_tokens_empty_db_name_raises():
    """list_tokens with an empty db_name must raise ValueError."""
    c = Endee(token=os.environ["NDD_ROOT_TOKEN"])
    with pytest.raises(ValueError):
        c.list_tokens("")


def test_create_token_invalid_type_raises():
    """create_token with an unrecognised token_type must raise ValueError."""
    c = Endee(token=os.environ["NDD_ROOT_TOKEN"])
    with pytest.raises(ValueError):
        c.create_token("some_db", "tok", token_type="admin")


# -- list_databases -----------------------------------------------------------


def test_list_databases_returns_list(admin_client):
    """list_databases() must return a list."""
    result = admin_client.list_databases()
    assert isinstance(result, list)


def test_list_databases_items_are_dicts(admin_client):
    """Each item in list_databases() must be a dict."""
    for item in admin_client.list_databases():
        assert isinstance(item, dict), f"Expected dict, got {type(item)}: {item}"


# -- create_database / delete_database ----------------------------------------


def test_create_database_returns_token_string(admin_client):
    """create_database() must return a non-empty token string."""
    name = uid("adb")
    try:
        token = admin_client.create_database(name)
        assert isinstance(token, str)
        assert len(token) > 0
    finally:
        try:
            admin_client.delete_database(name)
        except Exception:
            pass


def test_create_database_appears_in_list(admin_client):
    """A newly created database must appear in list_databases()."""
    name = uid("adb")
    try:
        admin_client.create_database(name)
        db_names = [d.get("db_name") for d in admin_client.list_databases()]
        assert name in db_names, f"'{name}' not found in list_databases()"
    finally:
        try:
            admin_client.delete_database(name)
        except Exception:
            pass


def test_delete_database_returns_dict(admin_client):
    """delete_database() must return a dict."""
    name = uid("adb")
    admin_client.create_database(name)
    result = admin_client.delete_database(name)
    assert isinstance(result, dict)


def test_delete_database_removes_from_list(admin_client):
    """After delete_database(), the database must not appear in list_databases()."""
    name = uid("adb")
    admin_client.create_database(name)
    admin_client.delete_database(name)
    db_names = [d.get("db_name") for d in admin_client.list_databases()]
    assert name not in db_names


def test_create_database_token_is_usable(admin_client):
    """The token returned by create_database() must work for collection operations."""
    name = uid("adb")
    try:
        db_token = admin_client.create_database(name)
        collections = _db_client(db_token).list_collections()
        assert isinstance(collections, list)
    finally:
        try:
            admin_client.delete_database(name)
        except Exception:
            pass


# -- get_database -------------------------------------------------------------


def test_get_database_returns_dict(temp_db, admin_client):
    """get_database() must return a dict."""
    name, _ = temp_db
    result = admin_client.get_database(name)
    assert isinstance(result, dict)


def test_get_database_name_matches(temp_db, admin_client):
    """get_database() response must include the correct db_name."""
    name, _ = temp_db
    result = admin_client.get_database(name)
    assert result.get("db_name") == name


# -- activate / deactivate ----------------------------------------------------


def test_deactivate_database_returns_dict(temp_db, admin_client):
    """deactivate_database() must return a dict."""
    name, _ = temp_db
    result = admin_client.deactivate_database(name)
    assert isinstance(result, dict)
    admin_client.activate_database(name)  # restore so subsequent tests can use the DB


def test_activate_database_returns_dict(temp_db, admin_client):
    """activate_database() on a deactivated database must return a dict."""
    name, _ = temp_db
    admin_client.deactivate_database(name)
    result = admin_client.activate_database(name)
    assert isinstance(result, dict)


def test_deactivate_then_activate_leaves_db_active(temp_db, admin_client):
    """Deactivating then activating a database must leave it listed as active."""
    name, _ = temp_db
    admin_client.deactivate_database(name)
    admin_client.activate_database(name)
    dbs = admin_client.list_databases()
    db = next((d for d in dbs if d.get("db_name") == name), None)
    assert db is not None
    if "is_active" in db:
        assert db["is_active"] is True


# -- set_database_type --------------------------------------------------------


def test_set_database_type_returns_dict(temp_db, admin_client):
    """set_database_type() must return a dict."""
    name, _ = temp_db
    result = admin_client.set_database_type(name, "pro")
    assert isinstance(result, dict)


def test_set_database_type_actually_changes_tier(temp_db, admin_client):
    """set_database_type() must update the tier visible in list_databases()."""
    name, _ = temp_db
    admin_client.set_database_type(name, "starter")
    dbs = admin_client.list_databases()
    db = next((d for d in dbs if d.get("db_name") == name), None)
    assert db is not None, f"Database '{name}' not found in list_databases()"
    if "db_type" in db:
        assert db["db_type"] == "Starter"


# -- list_db_collections / list_all_collections --------------------------------


def test_list_db_collections_returns_list(temp_db, admin_client):
    """list_db_collections() for a new database must return a list."""
    name, _ = temp_db
    result = admin_client.list_db_collections(name)
    assert isinstance(result, list)


def test_list_all_collections_returns_list(admin_client):
    """list_all_collections() must return a list."""
    result = admin_client.list_all_collections()
    assert isinstance(result, list)


def test_list_db_collections_reflects_created_collection(temp_db, admin_client):
    """A collection created inside a database must appear in list_db_collections()."""
    name, _ = temp_db
    col_name = uid("ac")
    tok_name = uid("tok")
    fresh_token = admin_client.create_token(name, tok_name)
    col_client = _db_client(fresh_token)
    try:
        col_client.create_collection(name=col_name, fields=[make_dense_field()])
        cols = admin_client.list_db_collections(name)
        assert col_name in cols, f"'{col_name}' not found in list_db_collections()"
    finally:
        try:
            col_client.delete_collection(col_name)
        except Exception:
            pass
        try:
            admin_client.delete_token(name, tok_name)
        except Exception:
            pass


# -- delete_db_collection -----------------------------------------------------


def test_delete_db_collection_removes_from_list(temp_db, admin_client):
    """delete_db_collection() must remove the collection from list_db_collections()."""
    name, _ = temp_db
    col_name = uid("ac")
    tok_name = uid("tok")
    fresh_token = admin_client.create_token(name, tok_name)
    try:
        _db_client(fresh_token).create_collection(
            name=col_name, fields=[make_dense_field()]
        )
        admin_client.delete_db_collection(name, col_name)
        cols = admin_client.list_db_collections(name)
        assert col_name not in cols
    finally:
        try:
            admin_client.delete_token(name, tok_name)
        except Exception:
            pass


def test_delete_db_collection_returns_dict(temp_db, admin_client):
    """delete_db_collection() must return a dict."""
    name, _ = temp_db
    col_name = uid("ac")
    tok_name = uid("tok")
    fresh_token = admin_client.create_token(name, tok_name)
    try:
        _db_client(fresh_token).create_collection(
            name=col_name, fields=[make_dense_field()]
        )
        result = admin_client.delete_db_collection(name, col_name)
        assert isinstance(result, dict)
    finally:
        try:
            admin_client.delete_token(name, tok_name)
        except Exception:
            pass


# -- admin token management ---------------------------------------------------


def test_create_token_returns_string(temp_db, admin_client):
    """create_token() must return a non-empty token string."""
    name, _ = temp_db
    tok_name = uid("tok")
    try:
        result = admin_client.create_token(name, tok_name)
        assert isinstance(result, str)
        assert len(result) > 0
    finally:
        try:
            admin_client.delete_token(name, tok_name)
        except Exception:
            pass


def test_list_tokens_returns_list(temp_db, admin_client):
    """list_tokens() must return a list."""
    name, _ = temp_db
    result = admin_client.list_tokens(name)
    assert isinstance(result, list)


def test_create_token_appears_in_list(temp_db, admin_client):
    """A token created via create_token() must appear in list_tokens()."""
    name, _ = temp_db
    tok_name = uid("tok")
    try:
        admin_client.create_token(name, tok_name)
        token_names = [t.get("name") for t in admin_client.list_tokens(name)]
        assert tok_name in token_names, f"'{tok_name}' not found in list_tokens()"
    finally:
        try:
            admin_client.delete_token(name, tok_name)
        except Exception:
            pass


def test_delete_token_removes_from_list(temp_db, admin_client):
    """After delete_token(), the token must not appear in list_tokens()."""
    name, _ = temp_db
    tok_name = uid("tok")
    admin_client.create_token(name, tok_name)
    admin_client.delete_token(name, tok_name)
    token_names = [t.get("name") for t in admin_client.list_tokens(name)]
    assert tok_name not in token_names


def test_create_token_rw_type(temp_db, admin_client):
    """create_token() with token_type='rw' must succeed and return a string."""
    name, _ = temp_db
    tok_name = uid("rwtok")
    try:
        result = admin_client.create_token(name, tok_name, token_type="rw")
        assert isinstance(result, str)
    finally:
        try:
            admin_client.delete_token(name, tok_name)
        except Exception:
            pass


def test_create_token_readonly_type(temp_db, admin_client):
    """create_token() with token_type='r' must succeed and return a string."""
    name, _ = temp_db
    tok_name = uid("rtok")
    try:
        result = admin_client.create_token(name, tok_name, token_type="r")
        assert isinstance(result, str)
    finally:
        try:
            admin_client.delete_token(name, tok_name)
        except Exception:
            pass


def test_admin_token_full_lifecycle(temp_db, admin_client):
    """Create a token, verify it appears in list, delete it, verify it is gone."""
    name, _ = temp_db
    tok_name = uid("full")
    try:
        token_str = admin_client.create_token(name, tok_name, token_type="rw")
        assert isinstance(token_str, str)

        names = [t.get("name") for t in admin_client.list_tokens(name)]
        assert tok_name in names

        admin_client.delete_token(name, tok_name)

        names_after = [t.get("name") for t in admin_client.list_tokens(name)]
        assert tok_name not in names_after
    finally:
        try:
            admin_client.delete_token(name, tok_name)
        except Exception:
            pass

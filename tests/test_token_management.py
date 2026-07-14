"""
Tests for self-service token management: create_my_token, list_my_tokens,
and delete_my_token.

Covers token creation, listing, deletion, lifecycle, duplicate name handling,
and client-side validation. Uses a database-level token (ENDEE_TOKEN); admin
token management requiring root access is covered in test_admin.py.
"""

import os

import pytest
from helpers import uid

from endee import Endee
from endee.exceptions import ConflictException, EndeeException

# -- helpers -------------------------------------------------------------------


def _delete_token_silently(client, name: str) -> None:
    """Delete a token without raising - used in finally blocks."""
    try:
        client.delete_my_token(name)
    except Exception:
        pass


# -- list_my_tokens ------------------------------------------------------------


def test_list_my_tokens_returns_list(client):
    """list_my_tokens() must return a list."""
    result = client.list_my_tokens()
    assert isinstance(result, list)


def test_list_my_tokens_items_are_dicts(client):
    """Each item in list_my_tokens() must be a dict."""
    result = client.list_my_tokens()
    for item in result:
        assert isinstance(item, dict), f"Expected dict, got {type(item)}: {item}"


# -- create_my_token -----------------------------------------------------------


def test_create_my_token_returns_string(client):
    """create_my_token() must return a non-empty string (the new token)."""
    tok_name = uid("tok")
    try:
        result = client.create_my_token(name=tok_name)
        assert isinstance(result, str)
        assert len(result) > 0
    finally:
        _delete_token_silently(client, tok_name)


def test_create_my_token_rw_type(client):
    """create_my_token() with token_type='rw' must succeed."""
    tok_name = uid("rwtok")
    try:
        result = client.create_my_token(name=tok_name, token_type="rw")
        assert isinstance(result, str)
    finally:
        _delete_token_silently(client, tok_name)


def test_create_my_token_readonly_type(client):
    """create_my_token() with token_type='r' must succeed."""
    tok_name = uid("rtok")
    try:
        result = client.create_my_token(name=tok_name, token_type="r")
        assert isinstance(result, str)
    finally:
        _delete_token_silently(client, tok_name)


def test_create_my_token_appears_in_list(client):
    """After create_my_token(), the token name must appear in list_my_tokens()."""
    tok_name = uid("ltok")
    try:
        client.create_my_token(name=tok_name)
        tokens = client.list_my_tokens()
        token_names = [t.get("name") for t in tokens if isinstance(t, dict)]
        assert tok_name in token_names, (
            f"Token '{tok_name}' not found in list_my_tokens(): {token_names}"
        )
    finally:
        _delete_token_silently(client, tok_name)


def test_create_my_token_default_type_is_rw(client):
    """create_my_token() without token_type must default to rw."""
    tok_name = uid("deftok")
    try:
        client.create_my_token(name=tok_name)
        tokens = client.list_my_tokens()
        tok = next(
            (t for t in tokens if isinstance(t, dict) and t.get("name") == tok_name),
            None,
        )
        assert tok is not None, f"Token '{tok_name}' not found in list_my_tokens()"
        if "token_type" in tok:
            assert tok["token_type"] in ("rw", "read_write", "readwrite")
    finally:
        _delete_token_silently(client, tok_name)


# -- delete_my_token -----------------------------------------------------------


def test_delete_my_token_returns_dict(client):
    """delete_my_token() must return a dict."""
    tok_name = uid("dtok")
    client.create_my_token(name=tok_name)
    result = client.delete_my_token(name=tok_name)
    assert isinstance(result, dict)


def test_delete_my_token_removes_from_list(client):
    """After delete_my_token(), the token must not appear in list_my_tokens()."""
    tok_name = uid("rmtok")
    client.create_my_token(name=tok_name)
    client.delete_my_token(name=tok_name)
    tokens = client.list_my_tokens()
    token_names = [t.get("name") for t in tokens if isinstance(t, dict)]
    assert tok_name not in token_names


def test_delete_my_token_idempotent_second_delete_raises(client):
    """Deleting a token that does not exist must raise an error."""
    tok_name = uid("dd")
    client.create_my_token(name=tok_name)
    client.delete_my_token(name=tok_name)
    with pytest.raises(EndeeException):
        client.delete_my_token(name=tok_name)


# -- create -> delete lifecycle -------------------------------------------------


def test_token_full_lifecycle(client):
    """Create, verify in list, delete, verify removed - full token lifecycle."""
    tok_name = uid("full")
    try:
        # Create
        token_str = client.create_my_token(name=tok_name, token_type="r")
        assert isinstance(token_str, str)

        # Verify present
        tokens = client.list_my_tokens()
        names = [t.get("name") for t in tokens if isinstance(t, dict)]
        assert tok_name in names

        # Delete
        client.delete_my_token(name=tok_name)

        # Verify absent
        tokens_after = client.list_my_tokens()
        names_after = [t.get("name") for t in tokens_after if isinstance(t, dict)]
        assert tok_name not in names_after
    finally:
        _delete_token_silently(client, tok_name)


# -- duplicate token name ------------------------------------------------------


def test_create_duplicate_token_raises_conflict(client):
    """Creating two tokens with the same name must raise ConflictException."""
    tok_name = uid("duptok")
    try:
        client.create_my_token(name=tok_name)
        with pytest.raises((EndeeException, ConflictException)):
            client.create_my_token(name=tok_name)
    finally:
        _delete_token_silently(client, tok_name)


# -- client-side validation ----------------------------------------------------


def test_create_my_token_empty_name_raises():
    """create_my_token() with empty name must raise ValueError (client-side)."""
    c = Endee(token=os.environ.get("ENDEE_TOKEN"))
    with pytest.raises(ValueError):
        c.create_my_token(name="")


def test_create_my_token_invalid_type_raises():
    """create_my_token() with invalid token_type must raise ValueError."""
    c = Endee(token=os.environ.get("ENDEE_TOKEN"))
    with pytest.raises(ValueError, match="token_type"):
        c.create_my_token(name="any_name", token_type="admin")


def test_delete_my_token_empty_name_raises():
    """delete_my_token() with empty name must raise ValueError (client-side)."""
    c = Endee(token=os.environ.get("ENDEE_TOKEN"))
    with pytest.raises(ValueError):
        c.delete_my_token(name="")

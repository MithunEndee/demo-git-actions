"""
Tests for sparse vector field operations.

Covers collection creation (default and BM25 sparse models), upsert,
search (result structure, limit, ef_search, similarity ordering),
describe, meta round-trips, and delete_object.
"""

import pytest
from helpers import (
    N_VECTORS,
    SPARSE_FIELD,
    get_collection_names,
    make_sparse_field,
    make_sparse_item,
    safe_delete,
    sparse_vec,
    uid,
)

# -- Collection creation -------------------------------------------------------


def test_create_sparse_collection(client):
    """Creating a sparse-only collection must succeed."""
    name = uid("sp")
    try:
        result = client.create_collection(name=name, fields=[make_sparse_field()])
        assert isinstance(result, dict)
        assert name in get_collection_names(client)
    finally:
        safe_delete(client, name)


def test_create_sparse_collection_bm25(client):
    """Sparse collection with endee_bm25 must be created (skip if unsupported)."""
    name = uid("spbm")
    try:
        client.create_collection(name=name, fields=[make_sparse_field("endee_bm25")])
        assert name in get_collection_names(client)
    except Exception as e:
        pytest.skip(f"endee_bm25 not supported on this server: {e}")
    finally:
        safe_delete(client, name)


# -- Upsert --------------------------------------------------------------------


def test_sparse_upsert_single_object(empty_sparse_collection):
    """Upserting a single sparse object must return upserted == 1."""
    _, collection = empty_sparse_collection
    si, sv = sparse_vec(seed=0)
    result = collection.upsert(
        [{"id": "sp1", "fields": {SPARSE_FIELD: {"indices": si, "values": sv}}}]
    )
    assert result.get("upserted") == 1


def test_sparse_upsert_batch(empty_sparse_collection):
    """Upserting a batch of sparse objects must return the correct count."""
    _, collection = empty_sparse_collection
    batch = [make_sparse_item(i) for i in range(10)]
    result = collection.upsert(batch)
    assert result.get("upserted") == 10


def test_sparse_upsert_with_meta_and_filter(empty_sparse_collection):
    """Upserting a sparse object with meta and filter must succeed."""
    _, collection = empty_sparse_collection
    si, sv = sparse_vec(seed=1)
    result = collection.upsert(
        [
            {
                "id": "sp_full",
                "meta": {"title": "sparse doc"},
                "filter": {"category": "A"},
                "fields": {SPARSE_FIELD: {"indices": si, "values": sv}},
            }
        ]
    )
    assert result.get("upserted") == 1


def test_sparse_upsert_empty_batch_accepted(empty_sparse_collection):
    """Upserting an empty batch into a sparse collection must succeed without error."""
    _, collection = empty_sparse_collection
    result = collection.upsert([])
    assert isinstance(result, dict)


def test_sparse_upsert_overwrite(empty_sparse_collection):
    """Re-upserting the same ID must succeed (upsert semantics)."""
    _, collection = empty_sparse_collection
    si, sv = sparse_vec(seed=2)
    collection.upsert(
        [
            {
                "id": "dup",
                "meta": {"v": 1},
                "fields": {SPARSE_FIELD: {"indices": si, "values": sv}},
            }
        ]
    )
    result = collection.upsert(
        [
            {
                "id": "dup",
                "meta": {"v": 2},
                "fields": {SPARSE_FIELD: {"indices": si, "values": sv}},
            }
        ]
    )
    assert result.get("upserted") == 1


# -- Search --------------------------------------------------------------------


def test_sparse_search_returns_results(populated_sparse_collection):
    """search on a sparse field must return a non-empty results list."""
    _, collection = populated_sparse_collection
    si, sv = sparse_vec(seed=99)
    results = collection.search(
        fields={SPARSE_FIELD: {"query": {"indices": si, "values": sv}, "limit": 5}}
    )["results"][SPARSE_FIELD]
    assert isinstance(results, list)
    assert len(results) > 0


def test_sparse_search_result_has_required_keys(populated_sparse_collection):
    """Each sparse search result must contain id and similarity."""
    _, collection = populated_sparse_collection
    si, sv = sparse_vec(seed=0)
    results = collection.search(
        fields={SPARSE_FIELD: {"query": {"indices": si, "values": sv}, "limit": 1}}
    )["results"][SPARSE_FIELD]
    r = results[0]
    for key in ("id", "similarity"):
        assert key in r, f"Missing key '{key}'"


def test_sparse_search_results_sorted_by_similarity(populated_sparse_collection):
    """Sparse search results must be sorted by descending similarity."""
    _, collection = populated_sparse_collection
    si, sv = sparse_vec(seed=5)
    results = collection.search(
        fields={SPARSE_FIELD: {"query": {"indices": si, "values": sv}, "limit": 10}}
    )["results"][SPARSE_FIELD]
    sims = [r["similarity"] for r in results]
    assert sims == sorted(sims, reverse=True)


@pytest.mark.parametrize("limit", [1, 5, 10, 20])
def test_sparse_search_limit_respected(populated_sparse_collection, limit):
    """Sparse search must return no more than limit results."""
    _, collection = populated_sparse_collection
    si, sv = sparse_vec(seed=3)
    results = collection.search(
        fields={SPARSE_FIELD: {"query": {"indices": si, "values": sv}, "limit": limit}}
    )["results"][SPARSE_FIELD]
    assert len(results) <= limit


@pytest.mark.parametrize("ef_search", [32, 64, 128, 256])
def test_sparse_search_ef_search_accepted(populated_sparse_collection, ef_search):
    """sparse search must accept ef_search parameter without error."""
    _, collection = populated_sparse_collection
    si, sv = sparse_vec(seed=1)
    results = collection.search(
        fields={SPARSE_FIELD: {"query": {"indices": si, "values": sv}, "limit": 5}},
        ef_search=ef_search,
    )["results"][SPARSE_FIELD]
    assert isinstance(results, list)


# -- Meta round-trip -----------------------------------------------------------


def test_sparse_meta_round_trips(empty_sparse_collection):
    """Meta upserted with a sparse object must be returned in search results."""
    _, collection = empty_sparse_collection
    si, sv = sparse_vec(seed=42)
    payload = {"title": "sparse doc", "count": 7}
    collection.upsert(
        [
            {
                "id": "sp_meta",
                "meta": payload,
                "fields": {SPARSE_FIELD: {"indices": si, "values": sv}},
            }
        ]
    )
    results = collection.search(
        fields={SPARSE_FIELD: {"query": {"indices": si, "values": sv}, "limit": 1}}
    )["results"][SPARSE_FIELD]
    assert results[0]["id"] == "sp_meta"
    assert results[0]["meta"]["title"] == "sparse doc"
    assert results[0]["meta"]["count"] == 7


# -- delete_object -------------------------------------------------------------


def test_sparse_delete_object_returns_response(populated_sparse_collection):
    """delete_object must return the deleted object ID."""
    _, collection = populated_sparse_collection
    result = collection.delete_object("sp_0040")
    assert result["deleted"] == "sp_0040"


def test_sparse_delete_object_removed_from_search(populated_sparse_collection):
    """Deleted sparse object must not appear in subsequent search results."""
    _, collection = populated_sparse_collection
    target = "sp_0041"
    collection.delete_object(target)
    si, sv = sparse_vec(seed=41)
    results = collection.search(
        fields={SPARSE_FIELD: {"query": {"indices": si, "values": sv}, "limit": N_VECTORS}}
    )["results"][SPARSE_FIELD]
    assert target not in {r["id"] for r in results}


# -- describe ------------------------------------------------------------------


def test_sparse_describe_shows_correct_field(empty_sparse_collection):
    """describe() must include the sparse field name in the fields list."""
    _, collection = empty_sparse_collection
    info = collection.describe()
    field_names = [
        f.get("name") if isinstance(f, dict) else str(f) for f in info.get("fields", [])
    ]
    assert SPARSE_FIELD in field_names


def test_sparse_describe_field_type_is_sparse(empty_sparse_collection):
    """describe() must report the field type as 'sparse'."""
    _, collection = empty_sparse_collection
    info = collection.describe()
    field = next(
        (f for f in info.get("fields", []) if f.get("name") == SPARSE_FIELD), None
    )
    assert field is not None
    assert field.get("type") == "sparse"

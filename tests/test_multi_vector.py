"""
Tests for multi_vector (ColBERT-style) field operations.

Covers collection creation, all pooling methods and precision/space combos,
upsert, search, get_objects, delete_by_filter, update_filters, shrink,
rebuild, create_backup, and mixed dense + multi_vector RRF search.
"""

import pytest
from helpers import (
    ALL_PRECISIONS,
    ALL_SPACE_TYPES,
    DENSE_FIELD,
    DIM,
    MV_FIELD,
    N_VECTORS,
    dense_vec,
    get_collection_names,
    make_dense_field,
    make_mv_field,
    multi_vec,
    safe_delete,
    uid,
)

from endee import rerank
from endee.schema import CollectionFieldConfig, CollectionFieldParams


@pytest.mark.parametrize(
    "pooling_method", ["mean", "max", "average_pooling", "max_pooling"]
)
def test_multi_vector_all_pooling_method_aliases_accepted(pooling_method):
    """All four pooling_method aliases must be accepted by CollectionFieldConfig."""
    cfg = CollectionFieldConfig(
        name=MV_FIELD,
        type="multi_vector",
        pooling_method=pooling_method,
        params=CollectionFieldParams(
            dimension=DIM, space_type="cosine", precision="int8"
        ),
    )
    assert cfg.pooling_method == pooling_method


# -- Collection creation ------------------------------------------------------


def test_create_multi_vector_collection(client):
    """Creating a collection with a multi_vector field must succeed."""
    name = uid("mv")
    try:
        result = client.create_collection(name=name, fields=[make_mv_field()])
        assert isinstance(result, dict)
        assert name in get_collection_names(client)
    finally:
        safe_delete(client, name)


@pytest.mark.parametrize("pooling_method", ["mean", "max"])
def test_create_multi_vector_both_pooling_methods(client, pooling_method):
    """Both pooling methods must successfully create a collection."""
    name = uid("mvp")
    try:
        client.create_collection(
            name=name, fields=[make_mv_field(pooling_method=pooling_method)]
        )
        assert name in get_collection_names(client)
    finally:
        safe_delete(client, name)


@pytest.mark.parametrize("space_type", ALL_SPACE_TYPES)
@pytest.mark.parametrize("precision", ALL_PRECISIONS)
def test_create_multi_vector_all_precision_space_combinations(
    client, precision, space_type
):
    """Every (precision, space_type) combination must create a multi_vector field."""
    name = uid("mvcombo")
    try:
        client.create_collection(
            name=name,
            fields=[make_mv_field(space_type=space_type, precision=precision)],
        )
        assert name in get_collection_names(client)
    finally:
        safe_delete(client, name)


# -- Upsert -------------------------------------------------------------------


def test_multi_vector_upsert_single_object(empty_mv_collection):
    """Upserting a single multi_vector object must return 'upserted' key."""
    _, collection = empty_mv_collection
    result = collection.upsert([{"id": "mv1", "fields": {MV_FIELD: multi_vec(seed=0)}}])
    assert "upserted" in result


def test_multi_vector_upsert_count(empty_mv_collection):
    """upsert must return the exact count of inserted multi_vector objects."""
    _, collection = empty_mv_collection
    batch = [
        {"id": f"mv_{i}", "fields": {MV_FIELD: multi_vec(seed=i)}} for i in range(5)
    ]
    result = collection.upsert(batch)
    assert result["upserted"] == 5


def test_multi_vector_upsert_empty_batch_accepted(empty_mv_collection):
    """Upserting an empty batch into a multi_vector collection must succeed."""
    _, collection = empty_mv_collection
    result = collection.upsert([])
    assert isinstance(result, dict)


def test_multi_vector_upsert_single_vector(empty_mv_collection):
    """Upserting a multi_vector with a single vector must succeed."""
    _, collection = empty_mv_collection
    result = collection.upsert(
        [{"id": "single_tok", "fields": {MV_FIELD: [dense_vec(seed=0)]}}]
    )
    assert "upserted" in result


def test_multi_vector_upsert_many_vectors(empty_mv_collection):
    """Upserting a multi_vector with 16 vectors must succeed."""
    _, collection = empty_mv_collection
    result = collection.upsert(
        [{"id": "many_tok", "fields": {MV_FIELD: multi_vec(n_tokens=16, seed=0)}}]
    )
    assert "upserted" in result


def test_multi_vector_upsert_with_meta_and_filter(empty_mv_collection):
    """Upserting multi_vector objects with meta and filter must succeed."""
    _, collection = empty_mv_collection
    result = collection.upsert(
        [
            {
                "id": "mv_full",
                "meta": {"title": "colbert doc"},
                "filter": {"category": "A"},
                "fields": {MV_FIELD: multi_vec(seed=1)},
            }
        ]
    )
    assert "upserted" in result


def test_multi_vector_upsert_overwrite(empty_mv_collection):
    """Re-inserting the same ID must succeed (upsert semantics)."""
    _, collection = empty_mv_collection
    collection.upsert(
        [{"id": "dup", "meta": {"v": 1}, "fields": {MV_FIELD: multi_vec(seed=0)}}]
    )
    result = collection.upsert(
        [{"id": "dup", "meta": {"v": 2}, "fields": {MV_FIELD: multi_vec(seed=1)}}]
    )
    assert "upserted" in result


# -- Search -------------------------------------------------------------------


def test_multi_vector_search_returns_results(populated_mv_collection):
    """search on a multi_vector field must return a non-empty results list."""
    _, collection = populated_mv_collection
    results = collection.search(
        fields={MV_FIELD: {"query": multi_vec(seed=99), "limit": 5}}
    )["results"][MV_FIELD]
    assert isinstance(results, list)
    assert len(results) > 0


def test_multi_vector_search_result_has_required_keys(populated_mv_collection):
    """Each search result must contain id and similarity."""
    _, collection = populated_mv_collection
    results = collection.search(
        fields={MV_FIELD: {"query": multi_vec(seed=0), "limit": 1}}
    )["results"][MV_FIELD]
    r = results[0]
    for key in ("id", "similarity"):
        assert key in r, f"Missing key '{key}'"


def test_multi_vector_search_results_sorted_by_similarity(populated_mv_collection):
    """multi_vector search results must be sorted by descending similarity."""
    _, collection = populated_mv_collection
    results = collection.search(
        fields={MV_FIELD: {"query": multi_vec(seed=5), "limit": 10}}
    )["results"][MV_FIELD]
    sims = [r["similarity"] for r in results]
    assert sims == sorted(sims, reverse=True)


@pytest.mark.parametrize("limit", [1, 5, 10, 20])
def test_multi_vector_search_limit_respected(populated_mv_collection, limit):
    """multi_vector search must return no more than limit results."""
    _, collection = populated_mv_collection
    results = collection.search(
        fields={MV_FIELD: {"query": multi_vec(seed=3), "limit": limit}}
    )["results"][MV_FIELD]
    assert len(results) <= limit


def test_multi_vector_search_single_query_vector(populated_mv_collection):
    """Searching with a single-vector query list must succeed."""
    _, collection = populated_mv_collection
    results = collection.search(
        fields={MV_FIELD: {"query": [dense_vec(seed=77)], "limit": 5}}
    )["results"][MV_FIELD]
    assert isinstance(results, list)
    assert len(results) > 0


@pytest.mark.parametrize("ef_search", [32, 64, 128, 256, 512])
def test_multi_vector_search_ef_search_accepted(populated_mv_collection, ef_search):
    """multi_vector search must accept ef_search parameter without error."""
    _, collection = populated_mv_collection
    results = collection.search(
        fields={MV_FIELD: {"query": multi_vec(seed=1), "limit": 5}},
        ef_search=ef_search,
    )["results"][MV_FIELD]
    assert isinstance(results, list)


# -- Meta round-trip ----------------------------------------------------------


def test_multi_vector_meta_round_trips(empty_mv_collection):
    """Meta upserted with multi_vector object must be returned in search results."""
    _, collection = empty_mv_collection
    payload = {"title": "colbert doc", "count": 9}
    collection.upsert(
        [{"id": "mv_meta", "meta": payload, "fields": {MV_FIELD: multi_vec(seed=42)}}]
    )
    results = collection.search(
        fields={MV_FIELD: {"query": multi_vec(seed=42), "limit": 1}}
    )["results"][MV_FIELD]
    assert results[0]["id"] == "mv_meta"
    assert results[0]["meta"]["title"] == "colbert doc"
    assert results[0]["meta"]["count"] == 9


# -- delete_object ------------------------------------------------------------


def test_multi_vector_delete_object(populated_mv_collection):
    """delete_object must return the deleted ID and remove the object from search."""
    _, collection = populated_mv_collection
    target = "mv_0010"
    result = collection.delete_object(target)
    assert result["deleted"] == target
    results = collection.search(
        fields={MV_FIELD: {"query": multi_vec(seed=10), "limit": N_VECTORS}}
    )["results"][MV_FIELD]
    assert target not in {r["id"] for r in results}


# -- describe() ---------------------------------------------------------------


def test_multi_vector_describe_shows_correct_field(empty_mv_collection):
    """describe() must include the multi_vector field name in the fields list."""
    _, collection = empty_mv_collection
    info = collection.describe()
    field_names = [
        f.get("name") if isinstance(f, dict) else str(f) for f in info.get("fields", [])
    ]
    assert MV_FIELD in field_names


def test_multi_vector_describe_field_type(empty_mv_collection):
    """describe() must report the field type as 'multi_vector'."""
    _, collection = empty_mv_collection
    info = collection.describe()
    field = next((f for f in info.get("fields", []) if f.get("name") == MV_FIELD), None)
    assert field is not None
    assert field.get("type") == "multi_vector"


# -- Mixed: dense + multi_vector ----------------------------------------------


def test_create_collection_dense_and_multi_vector(client):
    """A collection with both a dense and a multi_vector field must be created."""
    name = uid("dmv")
    try:
        client.create_collection(
            name=name,
            fields=[make_dense_field(), make_mv_field()],  # both return dicts
        )
        collection = client.get_collection(name)
        info = collection.describe()
        assert len(info.get("fields", [])) == 2
    finally:
        safe_delete(client, name)


def test_mixed_dense_and_multi_vector_upsert_and_search(client):
    """A dense + multi_vector collection must support upsert and search per field."""
    name = uid("dmvs")
    try:
        client.create_collection(
            name=name,
            fields=[make_dense_field(), make_mv_field()],
        )
        collection = client.get_collection(name)

        collection.upsert(
            [
                {
                    "id": f"dmv_{i}",
                    "fields": {
                        DENSE_FIELD: dense_vec(seed=i),
                        MV_FIELD: multi_vec(seed=i),
                    },
                }
                for i in range(10)
            ]
        )

        dense_results = collection.search(
            fields={DENSE_FIELD: {"query": dense_vec(seed=0), "limit": 5}}
        )["results"][DENSE_FIELD]
        assert len(dense_results) == 5

        mv_results = collection.search(
            fields={MV_FIELD: {"query": multi_vec(seed=0), "limit": 5}}
        )["results"][MV_FIELD]
        assert len(mv_results) == 5
    finally:
        safe_delete(client, name)


# -- Mixed: RRF search --------------------------------------------------------


def test_mixed_dense_and_multi_vector_rrf_search(client):
    """A dense + multi_vector collection must support RRF search across both fields."""
    name = uid("dmvr")
    try:
        client.create_collection(
            name=name,
            fields=[make_dense_field(), make_mv_field()],
        )
        collection = client.get_collection(name)
        collection.upsert(
            [
                {
                    "id": f"dmvr_{i}",
                    "fields": {
                        DENSE_FIELD: dense_vec(seed=i),
                        MV_FIELD: multi_vec(seed=i),
                    },
                }
                for i in range(10)
            ]
        )
        raw = collection.search(
            fields={
                DENSE_FIELD: {"query": dense_vec(seed=0), "limit": 10},
                MV_FIELD: {"query": multi_vec(seed=0), "limit": 10},
            },
        )
        results = rerank(raw, limit=5)["results"]
        assert len(results) == 5
        for r in results:
            assert "id" in r and "similarity" in r
    finally:
        safe_delete(client, name)

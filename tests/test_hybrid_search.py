"""
Tests for hybrid (dense + sparse) collection operations.

Covers upsert, dense-only search, sparse-only search, RRF hybrid search,
per-field limits, ef_search, meta round-trips, delete_object, and field configs.
"""

import pytest
from helpers import (
    DENSE_FIELD,
    HYBRID_DIM,
    N_VECTORS,
    SPARSE_FIELD,
    dense_vec,
    make_item,
    safe_delete,
    sparse_vec,
    uid,
)

from endee import rerank
from endee.schema import CollectionFieldConfig, CollectionFieldParams

# -- upsert -------------------------------------------------------------------


def test_hybrid_upsert_single_object(empty_hybrid_collection):
    """Upserting a single hybrid object must return a response with 'upserted' key."""
    _, collection = empty_hybrid_collection
    si, sv = sparse_vec(seed=0)
    result = collection.upsert(
        [
            {
                "id": "hv1",
                "fields": {
                    DENSE_FIELD: dense_vec(HYBRID_DIM, seed=0),
                    SPARSE_FIELD: {"indices": si, "values": sv},
                },
            }
        ]
    )
    assert "upserted" in result


def test_hybrid_upsert_with_meta_and_filter(empty_hybrid_collection):
    """Upserting a hybrid object with meta and filter must succeed."""
    _, collection = empty_hybrid_collection
    si, sv = sparse_vec(seed=1)
    result = collection.upsert(
        [
            {
                "id": "hv_full",
                "meta": {"title": "hybrid doc"},
                "filter": {"category": "A"},
                "fields": {
                    DENSE_FIELD: dense_vec(HYBRID_DIM, seed=1),
                    SPARSE_FIELD: {"indices": si, "values": sv},
                },
            }
        ]
    )
    assert "upserted" in result


def test_hybrid_upsert_batch(empty_hybrid_collection):
    """Upserting a batch of hybrid objects must succeed."""
    _, collection = empty_hybrid_collection
    batch = [make_item(i, dim=HYBRID_DIM, with_sparse=True) for i in range(20)]
    result = collection.upsert(batch)
    assert "upserted" in result


def test_hybrid_upsert_count_returned(empty_hybrid_collection):
    """upsert must return the exact count of inserted objects."""
    _, collection = empty_hybrid_collection
    batch = [make_item(i, dim=HYBRID_DIM, with_sparse=True) for i in range(5)]
    result = collection.upsert(batch)
    assert result["upserted"] == 5


# -- dense-only search --------------------------------------------------------


def test_hybrid_dense_only_search_returns_results(populated_hybrid_collection):
    """Dense-only search with limit=5 must return exactly 5 results."""
    _, collection = populated_hybrid_collection
    results = collection.search(
        fields={DENSE_FIELD: {"query": dense_vec(HYBRID_DIM), "limit": 5}}
    )["results"][DENSE_FIELD]
    assert len(results) == 5


def test_hybrid_dense_only_result_structure(populated_hybrid_collection):
    """Dense-only search on a hybrid collection must return results with keys."""
    _, collection = populated_hybrid_collection
    results = collection.search(
        fields={DENSE_FIELD: {"query": dense_vec(HYBRID_DIM), "limit": 1}}
    )["results"][DENSE_FIELD]
    r = results[0]
    for key in ("id", "similarity"):
        assert key in r, f"Missing key '{key}' in result"


def test_hybrid_dense_only_results_sorted(populated_hybrid_collection):
    """Dense-only results must be sorted by descending similarity."""
    _, collection = populated_hybrid_collection
    results = collection.search(
        fields={DENSE_FIELD: {"query": dense_vec(HYBRID_DIM), "limit": 10}}
    )["results"][DENSE_FIELD]
    sims = [r["similarity"] for r in results]
    assert sims == sorted(sims, reverse=True)


# -- sparse-only search -------------------------------------------------------


def test_hybrid_sparse_only_search_returns_results(populated_hybrid_collection):
    """Search with only the sparse field must return results."""
    _, collection = populated_hybrid_collection
    si, sv = sparse_vec(seed=99)
    results = collection.search(
        fields={SPARSE_FIELD: {"query": {"indices": si, "values": sv}, "limit": 5}}
    )["results"][SPARSE_FIELD]
    assert isinstance(results, list)
    assert len(results) > 0


def test_hybrid_sparse_only_result_structure(populated_hybrid_collection):
    """Sparse-only search result must contain id and similarity."""
    _, collection = populated_hybrid_collection
    si, sv = sparse_vec(seed=7)
    results = collection.search(
        fields={SPARSE_FIELD: {"query": {"indices": si, "values": sv}, "limit": 1}}
    )["results"][SPARSE_FIELD]
    r = results[0]
    for key in ("id", "similarity"):
        assert key in r, f"Missing key '{key}'"


# -- full hybrid search (RRF) -------------------------------------------------


def test_hybrid_full_rrf_search_returns_results(populated_hybrid_collection):
    """Full hybrid RRF search with limit=10 must return exactly 10 results."""
    _, collection = populated_hybrid_collection
    si, sv = sparse_vec(seed=42)
    raw = collection.search(
        fields={
            DENSE_FIELD: {
                "query": dense_vec(HYBRID_DIM, seed=42),
                "limit": N_VECTORS * 5,
            },
            SPARSE_FIELD: {
                "query": {"indices": si, "values": sv},
                "limit": N_VECTORS * 5,
            },
        }
    )
    results = rerank(raw, limit=10)["results"]
    assert len(results) == 10


def test_hybrid_full_rrf_result_structure(populated_hybrid_collection):
    """RRF search result must contain id and similarity."""
    _, collection = populated_hybrid_collection
    si, sv = sparse_vec(seed=11)
    raw = collection.search(
        fields={
            DENSE_FIELD: {
                "query": dense_vec(HYBRID_DIM, seed=11),
                "limit": N_VECTORS * 5,
            },
            SPARSE_FIELD: {
                "query": {"indices": si, "values": sv},
                "limit": N_VECTORS * 5,
            },
        }
    )
    results = rerank(raw, limit=1)["results"]
    r = results[0]
    for key in ("id", "similarity"):
        assert key in r


def test_hybrid_full_rrf_results_sorted(populated_hybrid_collection):
    """RRF results must be sorted by descending similarity."""
    _, collection = populated_hybrid_collection
    si, sv = sparse_vec(seed=13)
    raw = collection.search(
        fields={
            DENSE_FIELD: {
                "query": dense_vec(HYBRID_DIM, seed=13),
                "limit": N_VECTORS * 5,
            },
            SPARSE_FIELD: {
                "query": {"indices": si, "values": sv},
                "limit": N_VECTORS * 5,
            },
        }
    )
    results = rerank(raw, limit=10)["results"]
    sims = [r["similarity"] for r in results]
    assert sims == sorted(sims, reverse=True)


def test_hybrid_rrf_limit_respected(populated_hybrid_collection):
    """RRF search must respect the limit parameter."""
    _, collection = populated_hybrid_collection
    si, sv = sparse_vec(seed=5)
    raw = collection.search(
        fields={
            DENSE_FIELD: {
                "query": dense_vec(HYBRID_DIM, seed=5),
                "limit": N_VECTORS * 5,
            },
            SPARSE_FIELD: {
                "query": {"indices": si, "values": sv},
                "limit": N_VECTORS * 5,
            },
        }
    )
    results = rerank(raw, limit=5)["results"]
    assert len(results) <= 5


# -- per-field limit ----------------------------------------------------------


def test_hybrid_per_field_limit_rrf(populated_hybrid_collection):
    """Per-field limit config must be accepted without error."""
    _, collection = populated_hybrid_collection
    si, sv = sparse_vec(seed=9)
    raw = collection.search(
        fields={
            DENSE_FIELD: {"query": dense_vec(HYBRID_DIM, seed=9), "limit": 20},
            SPARSE_FIELD: {"query": {"indices": si, "values": sv}, "limit": 10},
        }
    )
    results = rerank(raw, limit=10)["results"]
    assert isinstance(results, list)
    assert len(results) <= 10


# -- ef_search parameter ------------------------------------------------------


@pytest.mark.parametrize("ef_search", [32, 64, 128, 256, 512])
def test_hybrid_ef_search_accepted(populated_hybrid_collection, ef_search):
    """search must accept ef_search values across the valid range."""
    _, collection = populated_hybrid_collection
    results = collection.search(
        fields={DENSE_FIELD: {"query": dense_vec(HYBRID_DIM), "limit": 5}},
        ef_search=ef_search,
    )["results"][DENSE_FIELD]
    assert isinstance(results, list)


# -- meta round-trip ----------------------------------------------------------


def test_hybrid_meta_round_trips(empty_hybrid_collection):
    """Meta upserted into a hybrid collection must be returned intact in results."""
    _, collection = empty_hybrid_collection
    si, sv = sparse_vec(seed=5)
    payload = {"title": "hybrid doc", "count": 3}
    collection.upsert(
        [
            {
                "id": "hrt",
                "meta": payload,
                "fields": {
                    DENSE_FIELD: dense_vec(HYBRID_DIM, seed=5),
                    SPARSE_FIELD: {"indices": si, "values": sv},
                },
            }
        ]
    )
    results = collection.search(
        fields={DENSE_FIELD: {"query": dense_vec(HYBRID_DIM, seed=5), "limit": 1}}
    )["results"][DENSE_FIELD]
    assert results[0]["id"] == "hrt"
    assert results[0]["meta"]["title"] == "hybrid doc"
    assert results[0]["meta"]["count"] == 3


# -- delete_object in hybrid collection ---------------------------------------


def test_hybrid_delete_object_removes_from_search(populated_hybrid_collection):
    """Deleted object must not appear in subsequent hybrid search results."""
    _, collection = populated_hybrid_collection
    target_id = "vec_0010"
    collection.delete_object(target_id)

    si, sv = sparse_vec(seed=10)
    raw = collection.search(
        fields={
            DENSE_FIELD: {
                "query": dense_vec(HYBRID_DIM, seed=10),
                "limit": N_VECTORS * 5,
            },
            SPARSE_FIELD: {
                "query": {"indices": si, "values": sv},
                "limit": N_VECTORS * 5,
            },
        }
    )
    results = rerank(raw, limit=N_VECTORS)["results"]
    returned_ids = {r["id"] for r in results}
    assert target_id not in returned_ids


# -- precision + space_type combinations for hybrid collections ---------------


@pytest.mark.parametrize(
    "precision,space_type",
    [
        ("int8", "cosine"),
        ("float32", "cosine"),
        ("float16", "l2"),
        ("int16", "ip"),
    ],
)
def test_hybrid_create_various_field_configs(client, precision, space_type):
    """Hybrid collections with various dense field configs must be created."""
    name = uid("hcfg")
    try:
        client.create_collection(
            name=name,
            fields=[
                CollectionFieldConfig(
                    name=DENSE_FIELD,
                    type="vector",
                    params=CollectionFieldParams(
                        dimension=HYBRID_DIM,
                        space_type=space_type,
                        precision=precision,
                    ),
                ).to_dict(),
                CollectionFieldConfig(
                    name=SPARSE_FIELD, type="sparse", sparse_model="default"
                ).to_dict(),
            ],
        )
        collection = client.get_collection(name)
        info = collection.describe()
        assert len(info.get("fields", [])) == 2
    finally:
        safe_delete(client, name)

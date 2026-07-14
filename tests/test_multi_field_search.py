"""
Tests for multi-field search with and without reranking.

Covers per-field dict result format, RRF fusion via reranker='rrf',
field_weights, rrf_k, per-field limits, ef_search, and filters.
"""

import pytest
from helpers import (
    DENSE_FIELD,
    HYBRID_DIM,
    MV_FIELD,
    N_VECTORS,
    SPARSE_FIELD,
    dense_vec,
    make_dense_field,
    make_mv_field,
    make_sparse_field,
    multi_vec,
    parse_filter_field,
    safe_delete,
    sparse_vec,
    uid,
)

from endee import rerank

# -- multi-field search WITHOUT reranker (per-field format) --------------------


def test_multi_field_no_reranker_returns_dict_with_results(
    populated_hybrid_collection,
):
    """Multi-field search without reranker must return a dict with 'results' key."""
    _, collection = populated_hybrid_collection
    si, sv = sparse_vec(seed=0)
    response = collection.search(
        fields={
            DENSE_FIELD: {"query": dense_vec(HYBRID_DIM, seed=0), "limit": 5},
            SPARSE_FIELD: {"query": {"indices": si, "values": sv}, "limit": 5},
        },
    )
    assert isinstance(response, dict)
    assert "results" in response


def test_multi_field_no_reranker_results_is_dict(populated_hybrid_collection):
    """Multi-field search without reranker must return results as a dict."""
    _, collection = populated_hybrid_collection
    si, sv = sparse_vec(seed=1)
    results = collection.search(
        fields={
            DENSE_FIELD: {"query": dense_vec(HYBRID_DIM, seed=1), "limit": 5},
            SPARSE_FIELD: {"query": {"indices": si, "values": sv}, "limit": 5},
        },
    )["results"]
    assert isinstance(results, dict), (
        f"Expected dict for per-field results, got {type(results)}"
    )


def test_multi_field_no_reranker_keys_are_field_names(populated_hybrid_collection):
    """Per-field results dict must have the queried field names as keys."""
    _, collection = populated_hybrid_collection
    si, sv = sparse_vec(seed=2)
    results = collection.search(
        fields={
            DENSE_FIELD: {"query": dense_vec(HYBRID_DIM, seed=2), "limit": 5},
            SPARSE_FIELD: {"query": {"indices": si, "values": sv}, "limit": 5},
        },
    )["results"]
    assert DENSE_FIELD in results, f"Missing '{DENSE_FIELD}' key in per-field results"
    assert SPARSE_FIELD in results, f"Missing '{SPARSE_FIELD}' key in per-field results"


def test_multi_field_no_reranker_each_field_is_list(populated_hybrid_collection):
    """Each value in per-field results must be a list."""
    _, collection = populated_hybrid_collection
    si, sv = sparse_vec(seed=3)
    results = collection.search(
        fields={
            DENSE_FIELD: {"query": dense_vec(HYBRID_DIM, seed=3), "limit": 5},
            SPARSE_FIELD: {"query": {"indices": si, "values": sv}, "limit": 5},
        },
    )["results"]
    assert isinstance(results[DENSE_FIELD], list)
    assert isinstance(results[SPARSE_FIELD], list)


def test_multi_field_no_reranker_each_hit_has_required_keys(
    populated_hybrid_collection,
):
    """Each hit in a per-field result must have id and similarity."""
    _, collection = populated_hybrid_collection
    si, sv = sparse_vec(seed=4)
    results = collection.search(
        fields={
            DENSE_FIELD: {"query": dense_vec(HYBRID_DIM, seed=4), "limit": 3},
            SPARSE_FIELD: {"query": {"indices": si, "values": sv}, "limit": 3},
        },
    )["results"]
    for field_name, hits in results.items():
        for hit in hits:
            assert "id" in hit, f"Missing 'id' in {field_name} hit"
            assert "similarity" in hit, f"Missing 'similarity' in {field_name} hit"


def test_multi_field_no_reranker_limit_respected_per_field(
    populated_hybrid_collection,
):
    """Per-field results must each respect the global limit."""
    _, collection = populated_hybrid_collection
    si, sv = sparse_vec(seed=5)
    limit = 7
    results = collection.search(
        fields={
            DENSE_FIELD: {"query": dense_vec(HYBRID_DIM, seed=5), "limit": limit},
            SPARSE_FIELD: {"query": {"indices": si, "values": sv}, "limit": limit},
        },
    )["results"]
    assert len(results[DENSE_FIELD]) <= limit
    assert len(results[SPARSE_FIELD]) <= limit


def test_multi_field_no_reranker_results_sorted_per_field(
    populated_hybrid_collection,
):
    """Each per-field result list must be sorted by descending similarity."""
    _, collection = populated_hybrid_collection
    si, sv = sparse_vec(seed=6)
    results = collection.search(
        fields={
            DENSE_FIELD: {"query": dense_vec(HYBRID_DIM, seed=6), "limit": 10},
            SPARSE_FIELD: {"query": {"indices": si, "values": sv}, "limit": 10},
        },
    )["results"]
    for field_name, hits in results.items():
        sims = [h["similarity"] for h in hits]
        assert sims == sorted(sims, reverse=True), (
            f"Field '{field_name}' results not sorted by descending similarity"
        )


# -- multi-field search WITH reranker='rrf' ------------------------------------


def test_multi_field_rrf_returns_flat_list(populated_hybrid_collection):
    """Multi-field RRF search must return a flat list (not a dict)."""
    _, collection = populated_hybrid_collection
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
        },
    )
    results = rerank(raw, limit=10)["results"]
    assert isinstance(results, list), (
        f"Expected list for RRF results, got {type(results)}"
    )


def test_multi_field_rrf_result_has_required_keys(populated_hybrid_collection):
    """Each RRF hit must have id and similarity."""
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
        },
    )
    results = rerank(raw, limit=5)["results"]
    for r in results:
        assert "id" in r
        assert "similarity" in r


def test_multi_field_rrf_limit_respected(populated_hybrid_collection):
    """RRF search must respect the limit parameter."""
    _, collection = populated_hybrid_collection
    si, sv = sparse_vec(seed=12)
    raw = collection.search(
        fields={
            DENSE_FIELD: {
                "query": dense_vec(HYBRID_DIM, seed=12),
                "limit": N_VECTORS * 5,
            },
            SPARSE_FIELD: {
                "query": {"indices": si, "values": sv},
                "limit": N_VECTORS * 5,
            },
        },
    )
    results = rerank(raw, limit=8)["results"]
    assert len(results) <= 8


def test_multi_field_rrf_results_sorted(populated_hybrid_collection):
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
        },
    )
    results = rerank(raw, limit=10)["results"]
    sims = [r["similarity"] for r in results]
    assert sims == sorted(sims, reverse=True)


# -- field_weights for RRF -----------------------------------------------------


def test_rrf_field_weights_accepted(populated_hybrid_collection):
    """field_weights summing to 1.0 must be accepted without error."""
    _, collection = populated_hybrid_collection
    si, sv = sparse_vec(seed=20)
    raw = collection.search(
        fields={
            DENSE_FIELD: {
                "query": dense_vec(HYBRID_DIM, seed=20),
                "limit": N_VECTORS * 5,
            },
            SPARSE_FIELD: {
                "query": {"indices": si, "values": sv},
                "limit": N_VECTORS * 5,
            },
        },
    )
    results = rerank(raw, limit=5, field_weights={DENSE_FIELD: 0.7, SPARSE_FIELD: 0.3})[
        "results"
    ]
    assert isinstance(results, list)


def test_rrf_field_weights_equal_split(populated_hybrid_collection):
    """field_weights with equal 0.5/0.5 split must return results."""
    _, collection = populated_hybrid_collection
    si, sv = sparse_vec(seed=21)
    raw = collection.search(
        fields={
            DENSE_FIELD: {
                "query": dense_vec(HYBRID_DIM, seed=21),
                "limit": N_VECTORS * 5,
            },
            SPARSE_FIELD: {
                "query": {"indices": si, "values": sv},
                "limit": N_VECTORS * 5,
            },
        },
    )
    results = rerank(raw, limit=5, field_weights={DENSE_FIELD: 0.5, SPARSE_FIELD: 0.5})[
        "results"
    ]
    assert isinstance(results, list)
    assert len(results) > 0


def test_rrf_field_weights_not_summing_to_one_raises(populated_hybrid_collection):
    """field_weights that do not sum to 1.0 must raise ValueError."""
    _, collection = populated_hybrid_collection
    si, sv = sparse_vec(seed=22)
    raw = collection.search(
        fields={
            DENSE_FIELD: {
                "query": dense_vec(HYBRID_DIM, seed=22),
                "limit": N_VECTORS * 5,
            },
            SPARSE_FIELD: {
                "query": {"indices": si, "values": sv},
                "limit": N_VECTORS * 5,
            },
        },
    )
    with pytest.raises(ValueError, match="sum"):
        rerank(raw, limit=5, field_weights={DENSE_FIELD: 0.6, SPARSE_FIELD: 0.6})


def test_rrf_field_weights_missing_field_raises(populated_hybrid_collection):
    """field_weights missing a queried field must raise ValueError."""
    _, collection = populated_hybrid_collection
    si, sv = sparse_vec(seed=23)
    raw = collection.search(
        fields={
            DENSE_FIELD: {
                "query": dense_vec(HYBRID_DIM, seed=23),
                "limit": N_VECTORS * 5,
            },
            SPARSE_FIELD: {
                "query": {"indices": si, "values": sv},
                "limit": N_VECTORS * 5,
            },
        },
    )
    with pytest.raises(ValueError, match="missing"):
        rerank(raw, limit=5, field_weights={DENSE_FIELD: 1.0})  # missing SPARSE_FIELD


# -- rrf_k parameter -----------------------------------------------------------


@pytest.mark.parametrize("rrf_k", [10, 30, 60, 120])
def test_rrf_k_parameter_accepted(populated_hybrid_collection, rrf_k):
    """Various rrf_k values must be accepted without error."""
    _, collection = populated_hybrid_collection
    si, sv = sparse_vec(seed=30)
    raw = collection.search(
        fields={
            DENSE_FIELD: {
                "query": dense_vec(HYBRID_DIM, seed=30),
                "limit": N_VECTORS * 5,
            },
            SPARSE_FIELD: {
                "query": {"indices": si, "values": sv},
                "limit": N_VECTORS * 5,
            },
        },
    )
    results = rerank(raw, limit=5, rrf_k=rrf_k)["results"]
    assert isinstance(results, list)


# -- per-field limit in query dict format --------------------------------------


def test_per_field_limit_in_query_dict_format(populated_hybrid_collection):
    """Per-field limit via query dict format must be respected in multi-field search."""
    _, collection = populated_hybrid_collection
    si, sv = sparse_vec(seed=40)
    raw = collection.search(
        fields={
            DENSE_FIELD: {"query": dense_vec(HYBRID_DIM, seed=40), "limit": 20},
            SPARSE_FIELD: {"query": {"indices": si, "values": sv}, "limit": 10},
        },
    )
    results = rerank(raw, limit=5)["results"]
    assert isinstance(results, list)
    assert len(results) <= 5


# -- filter in multi-field search ----------------------------------------------


def test_multi_field_rrf_with_filter(populated_hybrid_collection):
    """RRF search with a filter must return only matching objects."""
    _, collection = populated_hybrid_collection
    si, sv = sparse_vec(seed=50)
    raw = collection.search(
        fields={
            DENSE_FIELD: {
                "query": dense_vec(HYBRID_DIM, seed=50),
                "limit": N_VECTORS * 5,
            },
            SPARSE_FIELD: {
                "query": {"indices": si, "values": sv},
                "limit": N_VECTORS * 5,
            },
        },
        filter=[{"tags": {"$eq": "important"}}],
    )
    results = rerank(raw, limit=N_VECTORS)["results"]
    for r in results:
        assert parse_filter_field(r)["tags"] == "important"


def test_multi_field_no_reranker_with_filter(populated_hybrid_collection):
    """Per-field search with a filter must return only matching objects per field."""
    _, collection = populated_hybrid_collection
    si, sv = sparse_vec(seed=51)
    results = collection.search(
        fields={
            DENSE_FIELD: {"query": dense_vec(HYBRID_DIM, seed=51), "limit": N_VECTORS},
            SPARSE_FIELD: {"query": {"indices": si, "values": sv}, "limit": N_VECTORS},
        },
        filter=[{"category": {"$eq": "A"}}],
    )["results"]
    # results is a per-field dict; check all hits in each field
    if isinstance(results, dict):
        for _, hits in results.items():
            for hit in hits:
                assert parse_filter_field(hit)["category"] == "A"


# -- dense + multi_vector multi-field search -----------------------------------


def test_dense_and_mv_multi_field_rrf(client):
    """A dense + multi_vector collection must support multi-field RRF search."""
    name = uid("dmvrf")
    try:
        client.create_collection(
            name=name,
            fields=[make_dense_field(), make_mv_field()],
        )
        col = client.get_collection(name)
        col.upsert(
            [
                {
                    "id": f"obj_{i}",
                    "fields": {
                        DENSE_FIELD: dense_vec(seed=i),
                        MV_FIELD: multi_vec(seed=i),
                    },
                }
                for i in range(20)
            ]
        )
        raw = col.search(
            fields={
                DENSE_FIELD: {"query": dense_vec(seed=0), "limit": 20 * 5},
                MV_FIELD: {"query": multi_vec(seed=0), "limit": 20 * 5},
            },
        )
        results = rerank(raw, limit=5)["results"]
        assert isinstance(results, list)
        assert len(results) <= 5
        for r in results:
            assert "id" in r and "similarity" in r
    finally:
        safe_delete(client, name)


def test_dense_and_mv_multi_field_no_reranker(client):
    """A dense + multi_vector collection must support multi-field per-field search."""
    name = uid("dmvnr")
    try:
        client.create_collection(
            name=name,
            fields=[make_dense_field(), make_mv_field()],
        )
        col = client.get_collection(name)
        col.upsert(
            [
                {
                    "id": f"obj_{i}",
                    "fields": {
                        DENSE_FIELD: dense_vec(seed=i),
                        MV_FIELD: multi_vec(seed=i),
                    },
                }
                for i in range(10)
            ]
        )
        results = col.search(
            fields={
                DENSE_FIELD: {"query": dense_vec(seed=0), "limit": 5},
                MV_FIELD: {"query": multi_vec(seed=0), "limit": 5},
            },
        )["results"]
        assert isinstance(results, dict)
        assert DENSE_FIELD in results
        assert MV_FIELD in results
    finally:
        safe_delete(client, name)


# -- three-field search (dense + sparse + multi_vector) ------------------------


def test_three_field_rrf_search(client):
    """RRF search across dense, sparse, and multi_vector fields must return results."""
    name = uid("tri")
    try:
        client.create_collection(
            name=name,
            fields=[make_dense_field(), make_sparse_field(), make_mv_field()],
        )
        col = client.get_collection(name)
        batch = []
        for i in range(20):
            si, sv = sparse_vec(seed=i)
            batch.append(
                {
                    "id": f"tri_{i}",
                    "fields": {
                        DENSE_FIELD: dense_vec(seed=i),
                        SPARSE_FIELD: {"indices": si, "values": sv},
                        MV_FIELD: multi_vec(seed=i),
                    },
                }
            )
        col.upsert(batch)

        si, sv = sparse_vec(seed=99)
        raw = col.search(
            fields={
                DENSE_FIELD: {"query": dense_vec(seed=99), "limit": 20},
                SPARSE_FIELD: {"query": {"indices": si, "values": sv}, "limit": 20},
                MV_FIELD: {"query": multi_vec(seed=99), "limit": 20},
            },
        )
        results = rerank(raw, limit=5)["results"]
        assert isinstance(results, list)
        assert len(results) > 0
        for r in results:
            assert "id" in r and "similarity" in r
    finally:
        safe_delete(client, name)


def test_three_field_no_reranker_returns_all_field_keys(client):
    """Three-field search without reranker must return results keyed by all three fields."""
    name = uid("trnr")
    try:
        client.create_collection(
            name=name,
            fields=[make_dense_field(), make_sparse_field(), make_mv_field()],
        )
        col = client.get_collection(name)
        batch = []
        for i in range(10):
            si, sv = sparse_vec(seed=i)
            batch.append(
                {
                    "id": f"tri_{i}",
                    "fields": {
                        DENSE_FIELD: dense_vec(seed=i),
                        SPARSE_FIELD: {"indices": si, "values": sv},
                        MV_FIELD: multi_vec(seed=i),
                    },
                }
            )
        col.upsert(batch)

        si, sv = sparse_vec(seed=0)
        results = col.search(
            fields={
                DENSE_FIELD: {"query": dense_vec(seed=0), "limit": 5},
                SPARSE_FIELD: {"query": {"indices": si, "values": sv}, "limit": 5},
                MV_FIELD: {"query": multi_vec(seed=0), "limit": 5},
            },
        )["results"]
        assert isinstance(results, dict)
        assert DENSE_FIELD in results
        assert SPARSE_FIELD in results
        assert MV_FIELD in results
    finally:
        safe_delete(client, name)

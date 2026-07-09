"""
Tests for the standalone rerank() function.

Covers: return shape, limit enforcement, similarity ordering, field_weights,
rrf_k, error cases (invalid name, bad weights, empty results), and
deduplication across fields.
"""

import pytest
from endee import rerank
from helpers import DENSE_FIELD, HYBRID_DIM, N_VECTORS, SPARSE_FIELD, dense_vec, sparse_vec


def _raw_search(collection, seed=0):
    """Return a per-field search response from a hybrid collection."""
    si, sv = sparse_vec(seed=seed)
    return collection.search(
        fields={
            DENSE_FIELD: {"query": dense_vec(HYBRID_DIM, seed=seed), "limit": N_VECTORS},
            SPARSE_FIELD: {"query": {"indices": si, "values": sv}, "limit": N_VECTORS},
        }
    )


# -- return structure ---------------------------------------------------------


def test_rerank_returns_dict(populated_hybrid_collection):
    """rerank() must return a dict."""
    _, collection = populated_hybrid_collection
    raw = _raw_search(collection, seed=1)
    result = rerank(raw, limit=10)
    assert isinstance(result, dict)


def test_rerank_returns_results_key(populated_hybrid_collection):
    """rerank() return value must contain a 'results' key."""
    _, collection = populated_hybrid_collection
    raw = _raw_search(collection, seed=2)
    result = rerank(raw, limit=10)
    assert "results" in result


def test_rerank_results_is_list(populated_hybrid_collection):
    """rerank()['results'] must be a flat list, not a dict."""
    _, collection = populated_hybrid_collection
    raw = _raw_search(collection, seed=3)
    results = rerank(raw, limit=10)["results"]
    assert isinstance(results, list)


def test_rerank_limit_respected(populated_hybrid_collection):
    """rerank() with limit=5 must return at most 5 items."""
    _, collection = populated_hybrid_collection
    raw = _raw_search(collection, seed=4)
    results = rerank(raw, limit=5)["results"]
    assert len(results) <= 5


def test_rerank_results_have_required_keys(populated_hybrid_collection):
    """Each result from rerank() must have 'id' and 'similarity' keys."""
    _, collection = populated_hybrid_collection
    raw = _raw_search(collection, seed=5)
    results = rerank(raw, limit=10)["results"]
    assert len(results) > 0
    for hit in results:
        assert "id" in hit, f"Hit missing 'id': {hit}"
        assert "similarity" in hit, f"Hit missing 'similarity': {hit}"


# -- result ordering ----------------------------------------------------------


def test_rerank_results_sorted_by_similarity(populated_hybrid_collection):
    """rerank() results must be sorted by similarity in descending order."""
    _, collection = populated_hybrid_collection
    raw = _raw_search(collection, seed=6)
    results = rerank(raw, limit=10)["results"]
    sims = [r["similarity"] for r in results]
    assert sims == sorted(sims, reverse=True), "Results not sorted by similarity desc"


# -- field_weights parameter --------------------------------------------------


def test_rerank_field_weights_accepted(populated_hybrid_collection):
    """rerank() with valid field_weights must not raise."""
    _, collection = populated_hybrid_collection
    raw = _raw_search(collection, seed=7)
    results = rerank(
        raw,
        limit=10,
        field_weights={DENSE_FIELD: 0.6, SPARSE_FIELD: 0.4},
    )["results"]
    assert isinstance(results, list)


def test_rerank_field_weights_change_result(populated_hybrid_collection):
    """Different field_weights must produce a different ranking order."""
    _, collection = populated_hybrid_collection
    raw = _raw_search(collection, seed=8)
    results_dense = rerank(
        raw,
        limit=10,
        field_weights={DENSE_FIELD: 0.9, SPARSE_FIELD: 0.1},
    )["results"]
    results_sparse = rerank(
        raw,
        limit=10,
        field_weights={DENSE_FIELD: 0.1, SPARSE_FIELD: 0.9},
    )["results"]
    ids_dense = [r["id"] for r in results_dense]
    ids_sparse = [r["id"] for r in results_sparse]
    assert ids_dense != ids_sparse, (
        "Expected field_weights to affect ranking, but results are identical"
    )


# -- rrf_k parameter ----------------------------------------------------------


def test_rerank_rrf_k_accepted(populated_hybrid_collection):
    """rerank() with rrf_k=20 must be accepted without raising."""
    _, collection = populated_hybrid_collection
    raw = _raw_search(collection, seed=9)
    results = rerank(raw, limit=10, rrf_k=20)["results"]
    assert isinstance(results, list)


# -- error handling -----------------------------------------------------------


def test_rerank_invalid_name_raises(populated_hybrid_collection):
    """rerank() with name='invalid' must raise ValueError."""
    _, collection = populated_hybrid_collection
    raw = _raw_search(collection, seed=10)
    with pytest.raises(ValueError):
        rerank(raw, name="invalid", limit=10)


def test_rerank_field_weights_not_summing_to_1_raises(populated_hybrid_collection):
    """rerank() with field_weights not summing to 1.0 must raise ValueError."""
    _, collection = populated_hybrid_collection
    raw = _raw_search(collection, seed=11)
    with pytest.raises(ValueError):
        rerank(
            raw,
            limit=10,
            field_weights={DENSE_FIELD: 0.5, SPARSE_FIELD: 0.3},  # sums to 0.8
        )


def test_rerank_missing_field_weight_raises(populated_hybrid_collection):
    """rerank() with field_weights missing a field in results must raise ValueError."""
    _, collection = populated_hybrid_collection
    raw = _raw_search(collection, seed=12)
    with pytest.raises(ValueError):
        rerank(
            raw,
            limit=10,
            field_weights={DENSE_FIELD: 1.0},  # SPARSE_FIELD missing
        )


def test_rerank_empty_fields_raises():
    """rerank() with an empty results dict must raise ValueError."""
    empty_raw = {"results": {}}
    with pytest.raises(ValueError):
        rerank(empty_raw, limit=10)


# -- deduplication ------------------------------------------------------------


def test_rerank_deduplicates_across_fields(populated_hybrid_collection):
    """rerank() output must contain each id at most once even if it appears in both fields."""
    _, collection = populated_hybrid_collection
    raw = _raw_search(collection, seed=13)
    results = rerank(raw, limit=N_VECTORS)["results"]
    ids = [r["id"] for r in results]
    assert len(ids) == len(set(ids)), "Duplicate ids found in rerank() output"

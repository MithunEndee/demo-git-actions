"""
test_05_hybrid_search.py

Tests for hybrid (dense + sparse) search:
  - Upsert hybrid vectors
  - Dense-only query on hybrid index
  - Sparse-only query on hybrid index
  - Full hybrid query (dense + sparse)
  - Hybrid query with filter
  - RRF weight variations (dense_rrf_weight)
  - rrf_rank_constant variations
  - include_vectors on hybrid index
  - get_vector returns sparse_indices / sparse_values
"""

import pytest

from helpers import HYBRID_DIM, N_VECTORS, SPARSE_DIM, dense_vec, sparse_vec


# === Upsert ===

def test_hybrid_upsert_succeeds(empty_hybrid_index):
    """Upserting a single hybrid vector must return a success response."""
    _, index = empty_hybrid_index
    si, sv = sparse_vec(seed=0)
    result = index.upsert([{
        "id": "hv1",
        "vector": dense_vec(HYBRID_DIM, seed=0),
        "sparse_indices": si,
        "sparse_values": sv,
    }])
    assert "success" in result.lower()


def test_hybrid_upsert_with_meta_and_filter(empty_hybrid_index):
    """Upserting a hybrid vector with meta and filter fields must succeed."""
    _, index = empty_hybrid_index
    si, sv = sparse_vec(seed=1)
    result = index.upsert([{
        "id": "hv_full",
        "vector": dense_vec(HYBRID_DIM, seed=1),
        "sparse_indices": si,
        "sparse_values": sv,
        "meta": {"title": "hybrid doc"},
        "filter": {"category": "A"},
    }])
    assert "success" in result.lower()


def test_hybrid_upsert_batch(empty_hybrid_index):
    """Upserting a batch of hybrid vectors must return a success response."""
    _, index = empty_hybrid_index
    from helpers import make_item
    batch = [make_item(i, dim=HYBRID_DIM, with_sparse=True) for i in range(20)]
    result = index.upsert(batch)
    assert "success" in result.lower()


# === Dense-only query on hybrid index ===

def test_hybrid_dense_only_query(populated_hybrid_index):
    """Hybrid index accepts a query with only dense vector (no sparse)."""
    _, index = populated_hybrid_index
    results = index.query(vector=dense_vec(HYBRID_DIM), top_k=5)
    assert isinstance(results, list)
    assert len(results) > 0


def test_hybrid_dense_only_result_structure(populated_hybrid_index):
    """Dense-only query on a hybrid index must return results with all required keys."""
    _, index = populated_hybrid_index
    results = index.query(vector=dense_vec(HYBRID_DIM), top_k=1)
    r = results[0]
    for key in ("id", "similarity", "distance", "meta", "norm"):
        assert key in r


# === Sparse-only query on hybrid index ===

def test_hybrid_sparse_only_query(populated_hybrid_index):
    """Query with only sparse_indices/values, no dense vector."""
    _, index = populated_hybrid_index
    si, sv = sparse_vec(seed=99)
    results = index.query(sparse_indices=si, sparse_values=sv, top_k=5)
    assert isinstance(results, list)
    assert len(results) > 0


# === Full hybrid query ===

def test_hybrid_full_query(populated_hybrid_index):
    """Full hybrid query with both dense and sparse inputs must return results."""
    _, index = populated_hybrid_index
    si, sv = sparse_vec(seed=42)
    results = index.query(
        vector=dense_vec(HYBRID_DIM, seed=42),
        sparse_indices=si,
        sparse_values=sv,
        top_k=10,
    )
    assert isinstance(results, list)
    assert len(results) > 0


def test_hybrid_query_results_ordered_by_similarity(populated_hybrid_index):
    """Hybrid query results must be sorted from highest to lowest similarity."""
    _, index = populated_hybrid_index
    si, sv = sparse_vec(seed=7)
    results = index.query(
        vector=dense_vec(HYBRID_DIM, seed=7),
        sparse_indices=si,
        sparse_values=sv,
        top_k=10,
    )
    sims = [r["similarity"] for r in results]
    assert sims == sorted(sims, reverse=True)


# === Hybrid query with filter ===

def test_hybrid_query_with_eq_filter(populated_hybrid_index):
    """Hybrid query with a $eq filter must return only matching vectors."""
    _, index = populated_hybrid_index
    si, sv = sparse_vec(seed=3)
    results = index.query(
        vector=dense_vec(HYBRID_DIM, seed=3),
        sparse_indices=si,
        sparse_values=sv,
        top_k=N_VECTORS,
        filter=[{"category": {"$eq": "A"}}],
        prefilter_cardinality_threshold=1_000_000,
    )
    assert len(results) == 17
    for r in results:
        assert r["filter"]["category"] == "A"


def test_hybrid_query_with_range_filter(populated_hybrid_index):
    """Hybrid query with a $range filter must return only vectors within the score range."""
    _, index = populated_hybrid_index
    si, sv = sparse_vec(seed=4)
    results = index.query(
        vector=dense_vec(HYBRID_DIM, seed=4),
        sparse_indices=si,
        sparse_values=sv,
        top_k=N_VECTORS,
        filter=[{"score": {"$range": [10, 20]}}],
        prefilter_cardinality_threshold=1_000_000,
    )
    assert len(results) == 11
    for r in results:
        assert 10 <= r["filter"]["score"] <= 20


# === RRF weight variations ===

@pytest.mark.parametrize("weight", [0.0, 0.2, 0.5, 0.7, 1.0])
def test_hybrid_rrf_weight_accepted(populated_hybrid_index, weight):
    """All valid dense_rrf_weight values should return results without error."""
    _, index = populated_hybrid_index
    si, sv = sparse_vec(seed=10)
    results = index.query(
        vector=dense_vec(HYBRID_DIM, seed=10),
        sparse_indices=si,
        sparse_values=sv,
        top_k=5,
        dense_rrf_weight=weight,
    )
    assert isinstance(results, list)


def test_hybrid_rrf_weight_0_emphasises_sparse(populated_hybrid_index):
    """dense_rrf_weight=0.0 means full sparse ranking; should still return results."""
    _, index = populated_hybrid_index
    si, sv = sparse_vec(seed=11)
    results = index.query(
        vector=dense_vec(HYBRID_DIM, seed=11),
        sparse_indices=si,
        sparse_values=sv,
        top_k=5,
        dense_rrf_weight=0.0,
    )
    assert len(results) > 0


def test_hybrid_rrf_weight_1_emphasises_dense(populated_hybrid_index):
    """dense_rrf_weight=1.0 means full dense ranking; should still return results."""
    _, index = populated_hybrid_index
    si, sv = sparse_vec(seed=12)
    results = index.query(
        vector=dense_vec(HYBRID_DIM, seed=12),
        sparse_indices=si,
        sparse_values=sv,
        top_k=5,
        dense_rrf_weight=1.0,
    )
    assert len(results) > 0


# === rrf_rank_constant variations ===

@pytest.mark.parametrize("rrc", [1, 10, 30, 60, 120, 200])
def test_hybrid_rrf_rank_constant_accepted(populated_hybrid_index, rrc):
    """All valid rrf_rank_constant values must be accepted without error."""
    _, index = populated_hybrid_index
    si, sv = sparse_vec(seed=20)
    results = index.query(
        vector=dense_vec(HYBRID_DIM, seed=20),
        sparse_indices=si,
        sparse_values=sv,
        top_k=5,
        rrf_rank_constant=rrc,
    )
    assert isinstance(results, list)


# === include_vectors on hybrid index ===

def test_hybrid_include_vectors_true(populated_hybrid_index):
    """include_vectors=True on a hybrid index must return full-dimension dense vectors."""
    _, index = populated_hybrid_index
    si, sv = sparse_vec(seed=30)
    results = index.query(
        vector=dense_vec(HYBRID_DIM, seed=30),
        sparse_indices=si,
        sparse_values=sv,
        top_k=3,
        include_vectors=True,
    )
    for r in results:
        assert isinstance(r["vector"], list)
        assert len(r["vector"]) == HYBRID_DIM


def test_hybrid_include_vectors_false(populated_hybrid_index):
    """include_vectors=False on a hybrid index must return empty vector lists."""
    _, index = populated_hybrid_index
    si, sv = sparse_vec(seed=31)
    results = index.query(
        vector=dense_vec(HYBRID_DIM, seed=31),
        sparse_indices=si,
        sparse_values=sv,
        top_k=3,
        include_vectors=False,
    )
    for r in results:
        assert r["vector"] == []


# === get_vector on hybrid index returns sparse data ===

def test_hybrid_get_vector_has_sparse_keys(populated_hybrid_index):
    """get_vector on a hybrid index must include sparse_indices and sparse_values keys."""
    _, index = populated_hybrid_index
    vec = index.get_vector("vec_0000")
    assert "sparse_indices" in vec, "sparse_indices missing from get_vector result"
    assert "sparse_values" in vec, "sparse_values missing from get_vector result"


def test_hybrid_get_vector_sparse_lists_same_length(populated_hybrid_index):
    """sparse_indices and sparse_values returned by get_vector must have equal length."""
    _, index = populated_hybrid_index
    vec = index.get_vector("vec_0001")
    assert len(vec["sparse_indices"]) == len(vec["sparse_values"])


def test_hybrid_get_vector_sparse_indices_are_ints(populated_hybrid_index):
    """sparse_indices returned by get_vector must all be integers."""
    _, index = populated_hybrid_index
    vec = index.get_vector("vec_0002")
    for idx in vec["sparse_indices"]:
        assert isinstance(idx, int), f"Expected int, got {type(idx)}"


def test_hybrid_get_vector_sparse_values_are_floats(populated_hybrid_index):
    """sparse_values returned by get_vector must all be floats."""
    _, index = populated_hybrid_index
    vec = index.get_vector("vec_0003")
    for val in vec["sparse_values"]:
        assert isinstance(val, float), f"Expected float, got {type(val)}"

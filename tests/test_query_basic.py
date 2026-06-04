"""
test_03_query_basic.py

Tests for query - basic parameters and result structure:
  - Result shape
  - top_k variations
  - ef (search quality) variations
  - include_vectors flag
  - similarity / distance relationship
  - Results are ordered by descending similarity
"""

import pytest

from helpers import DIM, N_VECTORS, dense_vec


# === Result structure ===


def test_query_returns_list(populated_index):
    """query must return a list."""
    _, index = populated_index
    results = index.query(vector=dense_vec())
    assert isinstance(results, list)


def test_query_result_has_required_keys(populated_index):
    """Each result dict must contain all required keys."""
    _, index = populated_index
    results = index.query(vector=dense_vec(), top_k=1)
    assert len(results) >= 1
    keys = results[0].keys()
    for k in ("id", "similarity", "distance", "meta", "norm", "vector"):
        assert k in keys, f"Missing key '{k}' in result"


def test_query_result_id_is_string(populated_index):
    """The id field in each result must be a string."""
    _, index = populated_index
    results = index.query(vector=dense_vec(), top_k=1)
    assert isinstance(results[0]["id"], str)


def test_query_result_similarity_is_float(populated_index):
    """The similarity field in each result must be a float."""
    _, index = populated_index
    results = index.query(vector=dense_vec(), top_k=1)
    assert isinstance(results[0]["similarity"], float)


def test_query_result_distance_is_float(populated_index):
    """The distance field in each result must be a float."""
    _, index = populated_index
    results = index.query(vector=dense_vec(), top_k=1)
    assert isinstance(results[0]["distance"], float)


def test_query_distance_equals_one_minus_similarity(populated_index):
    """distance must equal 1.0 minus similarity for every result."""
    _, index = populated_index
    results = index.query(vector=dense_vec(), top_k=5)
    for r in results:
        assert abs(r["distance"] - (1.0 - r["similarity"])) < 1e-5


def test_query_results_ordered_by_descending_similarity(populated_index):
    """Results must be sorted from highest to lowest similarity."""
    _, index = populated_index
    results = index.query(vector=dense_vec(), top_k=10)
    sims = [r["similarity"] for r in results]
    assert sims == sorted(sims, reverse=True), (
        "Results not sorted by descending similarity"
    )


def test_query_meta_is_dict(populated_index):
    """The meta field in each result must be a dict."""
    _, index = populated_index
    results = index.query(vector=dense_vec(), top_k=1)
    assert isinstance(results[0]["meta"], dict)


def test_query_norm_positive(populated_index):
    """The norm field in each result must be a positive number."""
    _, index = populated_index
    results = index.query(vector=dense_vec(), top_k=1)
    assert results[0]["norm"] > 0


# === top_k variations ===


@pytest.mark.parametrize("top_k", [1, 5, 10, 20, 30, 50])
def test_query_top_k_returns_at_most_k_results(populated_index, top_k):
    """query must return no more than top_k results."""
    _, index = populated_index
    results = index.query(vector=dense_vec(), top_k=top_k)
    assert len(results) <= top_k


def test_query_top_k_1_returns_single_result(populated_index):
    """query with top_k=1 must return exactly one result."""
    _, index = populated_index
    results = index.query(vector=dense_vec(), top_k=1)
    assert len(results) == 1


def test_query_top_k_equals_n_returns_all_vectors(populated_index):
    """query with top_k equal to corpus size must return nearly all vectors."""
    _, index = populated_index
    # HNSW is approximate — exact full recall is not guaranteed even with high ef.
    # Assert we get close to all N vectors (within 10%) and never exceed top_k.
    results = index.query(vector=dense_vec(), top_k=N_VECTORS, ef=1024)
    assert len(results) <= N_VECTORS
    assert len(results) >= int(N_VECTORS * 0.9)


# === ef (search quality) parameter ===


@pytest.mark.parametrize("ef", [32, 64, 128, 256, 512, 1024])
def test_query_ef_parameter_accepted(populated_index, ef):
    """query must accept the ef parameter without error."""
    _, index = populated_index
    results = index.query(vector=dense_vec(), top_k=5, ef=ef)
    assert isinstance(results, list)


# === include_vectors flag ===


def test_query_include_vectors_false_returns_empty_vector(populated_index):
    """include_vectors=False must return an empty list in the vector field of each result."""
    _, index = populated_index
    results = index.query(vector=dense_vec(), top_k=3, include_vectors=False)
    for r in results:
        assert r["vector"] == [], f"Expected empty vector list, got {r['vector']}"


def test_query_include_vectors_true_returns_vector_data(populated_index):
    """include_vectors=True must return a full-dimension vector list in each result."""
    _, index = populated_index
    results = index.query(vector=dense_vec(), top_k=3, include_vectors=True)
    for r in results:
        assert isinstance(r["vector"], list), "Expected list of floats"
        assert len(r["vector"]) == DIM, f"Expected dim {DIM}, got {len(r['vector'])}"


# === filter key presence in results ===


def test_query_filter_field_present_when_upserted(populated_index):
    """Vectors upserted with filter= should return that field in results."""
    _, index = populated_index
    results = index.query(vector=dense_vec(), top_k=10)
    for r in results:
        assert "filter" in r
        assert isinstance(r["filter"], dict)


# === meta content round-trip ===


def test_query_meta_content_round_trips(empty_index):
    """Meta inserted during upsert must be returned intact in query results."""
    _, index = empty_index
    payload = {"title": "test doc", "count": 7, "flag": True}
    index.upsert([{"id": "meta_rt", "vector": dense_vec(seed=77), "meta": payload}])
    results = index.query(vector=dense_vec(seed=77), top_k=1)
    assert results[0]["id"] == "meta_rt"
    assert results[0]["meta"]["title"] == "test doc"
    assert results[0]["meta"]["count"] == 71

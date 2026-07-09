"""
Tests for Collection.search() on a dense vector collection.

Covers response structure (keys, types), result ordering, limit behaviour,
ef_search parameter, per-field query dict format, meta round-trips, and
edge cases (limit >> corpus, empty fields dict, single-field RRF).
"""

import json

import pytest
from endee import rerank
from helpers import DENSE_FIELD, DIM, N_VECTORS, dense_vec

# -- response structure -------------------------------------------------------


def test_search_returns_dict_with_results_key(populated_collection):
    """search must return a dict containing a 'results' key."""
    _, collection = populated_collection
    response = collection.search(fields={DENSE_FIELD: {"query": dense_vec(), "limit": N_VECTORS}})
    assert isinstance(response, dict)
    assert "results" in response


def test_search_results_is_list(populated_collection):
    """The 'results' key must be a dict mapping field names to lists."""
    _, collection = populated_collection
    response = collection.search(fields={DENSE_FIELD: {"query": dense_vec(), "limit": N_VECTORS}})
    assert isinstance(response["results"], dict)
    assert isinstance(response["results"][DENSE_FIELD], list)


def test_search_result_has_required_keys(populated_collection):
    """Each result dict must contain id and similarity keys."""
    _, collection = populated_collection
    results = collection.search(fields={DENSE_FIELD: {"query": dense_vec(), "limit": 1}})["results"][DENSE_FIELD]
    assert len(results) >= 1
    for key in ("id", "similarity"):
        assert key in results[0], f"Missing key '{key}' in result"


def test_search_result_id_is_string(populated_collection):
    """The id field must be a string."""
    _, collection = populated_collection
    results = collection.search(fields={DENSE_FIELD: {"query": dense_vec(), "limit": 1}})["results"][DENSE_FIELD]
    assert isinstance(results[0]["id"], str)


def test_search_result_similarity_is_float(populated_collection):
    """The similarity field must be a float."""
    _, collection = populated_collection
    results = collection.search(fields={DENSE_FIELD: {"query": dense_vec(), "limit": 1}})["results"][DENSE_FIELD]
    assert isinstance(results[0]["similarity"], float)


def test_search_results_ordered_by_descending_similarity(populated_collection):
    """Results must be sorted from highest to lowest similarity."""
    _, collection = populated_collection
    results = collection.search(fields={DENSE_FIELD: {"query": dense_vec(), "limit": 10}})["results"][DENSE_FIELD]
    sims = [r["similarity"] for r in results]
    assert sims == sorted(sims, reverse=True), (
        "Results not sorted by descending similarity"
    )


def test_search_meta_is_present_in_results(populated_collection):
    """The meta field must be present and contain the keys upserted with the object."""
    _, collection = populated_collection
    results = collection.search(fields={DENSE_FIELD: {"query": dense_vec(), "limit": 1}})["results"][DENSE_FIELD]
    meta = results[0]["meta"]
    assert isinstance(meta, dict)
    assert "index" in meta
    assert "text" in meta


def test_search_meta_values_match_upserted_data(populated_collection):
    """Meta values returned in results must match what was upserted."""
    _, collection = populated_collection
    results = collection.search(fields={DENSE_FIELD: {"query": dense_vec(), "limit": 1}})["results"][DENSE_FIELD]
    meta = results[0]["meta"]
    idx = meta["index"]
    assert meta["text"] == f"Document {idx}"


def test_search_filter_field_present_when_upserted(populated_collection):
    """Objects upserted with filter= must return all filter keys in search results."""
    _, collection = populated_collection
    results = collection.search(fields={DENSE_FIELD: {"query": dense_vec(), "limit": 10}})["results"][DENSE_FIELD]
    for r in results:
        flt = r.get("filter") or {}
        if isinstance(flt, str):
            flt = json.loads(flt)
        for key in ("category", "score", "priority", "tags"):
            assert key in flt, f"Missing filter key '{key}' in result"


# -- limit parameter ----------------------------------------------------------


@pytest.mark.parametrize("limit", [1, 5, 10, 20, 30, 50])
def test_search_limit_returns_at_most_n_results(populated_collection, limit):
    """search must return no more than `limit` results."""
    _, collection = populated_collection
    results = collection.search(fields={DENSE_FIELD: {"query": dense_vec(), "limit": limit}})[
        "results"
    ][DENSE_FIELD]
    assert len(results) <= limit


def test_search_limit_1_returns_single_result(populated_collection):
    """search with limit=1 must return exactly one result."""
    _, collection = populated_collection
    results = collection.search(fields={DENSE_FIELD: {"query": dense_vec(), "limit": 1}})["results"][DENSE_FIELD]
    assert len(results) == 1


def test_search_limit_equals_corpus_returns_nearly_all(populated_collection):
    """search with limit equal to corpus size must return nearly all objects."""
    _, collection = populated_collection
    results = collection.search(
        fields={DENSE_FIELD: {"query": dense_vec(), "limit": N_VECTORS, "ef_search": 1024}}
    )["results"][DENSE_FIELD]
    assert len(results) <= N_VECTORS
    assert len(results) >= int(N_VECTORS * 0.9)


# -- ef_search parameter ------------------------------------------------------


@pytest.mark.parametrize("ef_search", [32, 64, 128, 256, 512, 1024])
def test_search_ef_search_parameter_accepted(populated_collection, ef_search):
    """search must accept the ef_search parameter without error."""
    _, collection = populated_collection
    results = collection.search(
        fields={DENSE_FIELD: {"query": dense_vec(), "limit": 5, "ef_search": ef_search}}
    )["results"][DENSE_FIELD]
    assert isinstance(results, list)


# -- per-field query format ---------------------------------------------------


def test_search_per_field_query_dict_format(populated_collection):
    """search must accept per-field dict format: {field: {"query": vec, ...}}."""
    _, collection = populated_collection
    results = collection.search(
        fields={DENSE_FIELD: {"query": dense_vec(), "limit": 5, "ef_search": 64}}
    )["results"][DENSE_FIELD]
    assert isinstance(results, list)
    assert len(results) <= 5


# -- edge cases ---------------------------------------------------------------


def test_search_limit_over_max_returns_at_most_max(populated_collection):
    """search with a very large limit must return results without error."""
    _, collection = populated_collection
    results = collection.search(fields={DENSE_FIELD: {"query": dense_vec(), "limit": 4096}})[
        "results"
    ][DENSE_FIELD]
    assert isinstance(results, list)


def test_search_limit_much_larger_than_corpus_returns_all(populated_collection):
    """Search with a limit much larger than the corpus must return at least 90% of objects."""
    _, collection = populated_collection
    # ef_search=N_VECTORS*10 forces deep HNSW traversal to maximise recall
    results = collection.search(
        fields={DENSE_FIELD: {"query": dense_vec(), "limit": 4096}},
        ef_search=N_VECTORS * 10,
    )["results"][DENSE_FIELD]
    assert len(results) >= int(N_VECTORS * 0.9)


def test_search_empty_fields_dict_raises(populated_collection):
    """search with an empty fields dict must raise ValueError (client-side)."""
    _, collection = populated_collection
    with pytest.raises(ValueError):
        collection.search(fields={})


def test_search_rrf_single_field_does_not_error(populated_collection):
    """rerank() applied to single-field search results must not raise an error."""
    _, collection = populated_collection
    raw = collection.search(
        fields={DENSE_FIELD: {"query": dense_vec(), "limit": 5}}
    )
    results = rerank(raw, limit=5)["results"]
    assert isinstance(results, list)


# -- meta round-trip ----------------------------------------------------------


def test_search_meta_content_round_trips(empty_collection):
    """Meta inserted during upsert must be returned intact in search results."""
    _, collection = empty_collection
    payload = {"title": "test doc", "count": 7, "flag": True}
    collection.upsert(
        [
            {
                "id": "meta_rt",
                "meta": payload,
                "fields": {DENSE_FIELD: dense_vec(seed=77)},
            }
        ]
    )
    results = collection.search(fields={DENSE_FIELD: {"query": dense_vec(seed=77), "limit": 1}})[
        "results"
    ][DENSE_FIELD]
    assert results[0]["id"] == "meta_rt"
    assert results[0]["meta"]["title"] == "test doc"
    assert results[0]["meta"]["count"] == 7
    assert results[0]["meta"]["flag"] is True

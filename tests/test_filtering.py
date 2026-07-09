"""
Tests for Collection.search() filter operators across all collection types.

Operators: $eq, $in, $range, $gt, $gte, $lt, $lte - tested in isolation,
combined AND logic, and across dense, sparse, multi_vector, and hybrid fields.

Post-filter ANN tests that assert exact result counts are marked @_XFAIL_ANN
because HNSW post-filter recall is not guaranteed on small corpora.
"""

import pytest
from endee import rerank
from helpers import (
    DENSE_FIELD,
    HYBRID_DIM,
    MV_FIELD,
    N_VECTORS,
    SPARSE_FIELD,
    dense_vec,
    multi_vec,
    parse_filter_field,
    sparse_vec,
)

_XFAIL_ANN = pytest.mark.xfail(
    strict=False,
    reason="post-filter ANN: server applies filter after ANN candidate selection; exact count not guaranteed on small corpus",
)

# score = i for each object; range is 0 to N_VECTORS-1
_MAX_SCORE = N_VECTORS - 1

# -- $eq operator --------------------------------------------------------------


def test_filter_eq_all_results_match(populated_collection):
    """All results from a $eq filter must have the expected field value."""
    _, collection = populated_collection
    results = collection.search(
        fields={DENSE_FIELD: {"query": dense_vec(), "limit": N_VECTORS}},
        filter=[{"category": {"$eq": "A"}}],
    )["results"][DENSE_FIELD]
    assert len(results) > 0
    for r in results:
        flt = parse_filter_field(r)
        assert flt["category"] == "A", f"Expected 'A', got {flt['category']}"


@_XFAIL_ANN
def test_filter_eq_exact_count(populated_collection):
    """$eq filter on category 'B' must return exactly 17 results."""
    _, collection = populated_collection
    results = collection.search(
        fields={DENSE_FIELD: {"query": dense_vec(), "limit": N_VECTORS}},
        filter=[{"category": {"$eq": "B"}}],
    )["results"][DENSE_FIELD]
    assert len(results) == 17


@_XFAIL_ANN
def test_filter_eq_tags_important(populated_collection):
    """$eq filter on tags='important' must return exactly 25 results, all matching."""
    _, collection = populated_collection
    results = collection.search(
        fields={DENSE_FIELD: {"query": dense_vec(), "limit": N_VECTORS}},
        filter=[{"tags": {"$eq": "important"}}],
    )["results"][DENSE_FIELD]
    assert len(results) == 25
    for r in results:
        assert parse_filter_field(r)["tags"] == "important"


def test_filter_eq_no_match_returns_empty(populated_collection):
    """$eq filter with a value that matches no objects must return an empty list."""
    _, collection = populated_collection
    results = collection.search(
        fields={DENSE_FIELD: {"query": dense_vec(), "limit": N_VECTORS}},
        filter=[{"category": {"$eq": "NONEXISTENT"}}],
    )["results"][DENSE_FIELD]
    assert len(results) == 0


# -- $in operator --------------------------------------------------------------


@_XFAIL_ANN
def test_filter_in_single_value(populated_collection):
    """$in filter with a single value must behave like $eq for that value."""
    _, collection = populated_collection
    results = collection.search(
        fields={DENSE_FIELD: {"query": dense_vec(), "limit": N_VECTORS}},
        filter=[{"category": {"$in": ["C"]}}],
    )["results"][DENSE_FIELD]
    assert len(results) == 16
    for r in results:
        assert parse_filter_field(r)["category"] == "C"


@_XFAIL_ANN
def test_filter_in_two_values(populated_collection):
    """$in filter with two values must return results matching either value."""
    _, collection = populated_collection
    results = collection.search(
        fields={DENSE_FIELD: {"query": dense_vec(), "limit": N_VECTORS}},
        filter=[{"category": {"$in": ["A", "B"]}}],
    )["results"][DENSE_FIELD]
    assert len(results) == 34
    for r in results:
        assert parse_filter_field(r)["category"] in ("A", "B")


@_XFAIL_ANN
def test_filter_in_all_values(populated_collection):
    """$in with all three categories must return all objects."""
    _, collection = populated_collection
    results = collection.search(
        fields={DENSE_FIELD: {"query": dense_vec(), "limit": N_VECTORS * 4}},
        filter=[{"category": {"$in": ["A", "B", "C"]}}],
    )["results"][DENSE_FIELD]
    assert len(results) == N_VECTORS


@_XFAIL_ANN
def test_filter_in_tags(populated_collection):
    """$in filter covering all tag values must return all objects."""
    _, collection = populated_collection
    results = collection.search(
        fields={DENSE_FIELD: {"query": dense_vec(), "limit": N_VECTORS * 4}},
        filter=[{"tags": {"$in": ["important", "normal"]}}],
    )["results"][DENSE_FIELD]
    assert len(results) == N_VECTORS


def test_filter_in_empty_list_returns_empty(populated_collection):
    """$in with an empty list must return no results."""
    _, collection = populated_collection
    results = collection.search(
        fields={DENSE_FIELD: {"query": dense_vec(), "limit": N_VECTORS}},
        filter=[{"category": {"$in": []}}],
    )["results"][DENSE_FIELD]
    assert results == []


# -- $range operator -----------------------------------------------------------


@_XFAIL_ANN
def test_filter_range_returns_correct_count(populated_collection):
    """$range filter must return the expected number of matching objects."""
    _, collection = populated_collection
    results = collection.search(
        fields={DENSE_FIELD: {"query": dense_vec(), "limit": N_VECTORS}},
        filter=[{"score": {"$range": [10, 20]}}],
    )["results"][DENSE_FIELD]
    assert len(results) == 11


def test_filter_range_all_results_within_bounds(populated_collection):
    """All results from a $range filter must have scores within the specified bounds."""
    _, collection = populated_collection
    lo, hi = 5, 15
    results = collection.search(
        fields={DENSE_FIELD: {"query": dense_vec(), "limit": N_VECTORS}},
        filter=[{"score": {"$range": [lo, hi]}}],
    )["results"][DENSE_FIELD]
    for r in results:
        score = parse_filter_field(r)["score"]
        assert lo <= score <= hi, f"score {score} outside [{lo},{hi}]"


@_XFAIL_ANN
def test_filter_range_full_span(populated_collection):
    """Range spanning all scores must return all objects."""
    _, collection = populated_collection
    results = collection.search(
        fields={DENSE_FIELD: {"query": dense_vec(), "limit": N_VECTORS * 4}},
        filter=[{"score": {"$range": [0, _MAX_SCORE]}}],
    )["results"][DENSE_FIELD]
    assert len(results) == N_VECTORS


def test_filter_range_equal_bounds_returns_single_score(populated_collection):
    """$range with equal bounds must match only objects with that exact score."""
    _, collection = populated_collection
    results = collection.search(
        fields={DENSE_FIELD: {"query": dense_vec(), "limit": N_VECTORS}},
        filter=[{"score": {"$range": [5, 5]}}],
    )["results"][DENSE_FIELD]
    assert len(results) == 1
    assert parse_filter_field(results[0])["score"] == 5


# -- AND logic (multiple conditions) ------------------------------------------


@_XFAIL_ANN
def test_filter_and_eq_and_eq(populated_collection):
    """category='A' AND tags='important' must return exactly 9 matching objects."""
    _, collection = populated_collection
    results = collection.search(
        fields={DENSE_FIELD: {"query": dense_vec(), "limit": N_VECTORS}},
        filter=[
            {"category": {"$eq": "A"}},
            {"tags": {"$eq": "important"}},
        ],
    )["results"][DENSE_FIELD]
    assert len(results) == 9
    for r in results:
        flt = parse_filter_field(r)
        assert flt["category"] == "A"
        assert flt["tags"] == "important"


@_XFAIL_ANN
def test_filter_and_eq_and_range(populated_collection):
    """category='A' AND score in [0, 29] must return exactly 10 matching objects."""
    _, collection = populated_collection
    results = collection.search(
        fields={DENSE_FIELD: {"query": dense_vec(), "limit": N_VECTORS}},
        filter=[
            {"category": {"$eq": "A"}},
            {"score": {"$range": [0, 29]}},
        ],
    )["results"][DENSE_FIELD]
    assert len(results) == 10
    for r in results:
        flt = parse_filter_field(r)
        assert flt["category"] == "A"
        assert 0 <= flt["score"] <= 29


@_XFAIL_ANN
def test_filter_and_in_and_range(populated_collection):
    """category in ['A','B'] AND score in [0,9] must return exactly 7 objects."""
    _, collection = populated_collection
    results = collection.search(
        fields={DENSE_FIELD: {"query": dense_vec(), "limit": N_VECTORS}},
        filter=[
            {"category": {"$in": ["A", "B"]}},
            {"score": {"$range": [0, 9]}},
        ],
    )["results"][DENSE_FIELD]
    assert len(results) == 7
    for r in results:
        flt = parse_filter_field(r)
        assert flt["category"] in ("A", "B")
        assert flt["score"] <= 9


@_XFAIL_ANN
def test_filter_three_conditions(populated_collection):
    """category='A' AND tags='important' AND score in [0,29] -> 5 objects."""
    _, collection = populated_collection
    results = collection.search(
        fields={DENSE_FIELD: {"query": dense_vec(), "limit": N_VECTORS}},
        filter=[
            {"category": {"$eq": "A"}},
            {"tags": {"$eq": "important"}},
            {"score": {"$range": [0, 29]}},
        ],
    )["results"][DENSE_FIELD]
    assert len(results) == 5
    for r in results:
        flt = parse_filter_field(r)
        assert flt["category"] == "A"
        assert flt["tags"] == "important"
        assert flt["score"] <= 29


# -- filter correctness --------------------------------------------------------


@_XFAIL_ANN
def test_filter_results_satisfy_condition(populated_collection):
    """All results from any filter must satisfy that filter condition."""
    _, collection = populated_collection
    results = collection.search(
        fields={DENSE_FIELD: {"query": dense_vec(), "limit": N_VECTORS}},
        filter=[{"priority": {"$eq": 0}}],
    )["results"][DENSE_FIELD]
    assert len(results) == 10
    for r in results:
        assert parse_filter_field(r)["priority"] == 0


def test_filter_nonexistent_field_returns_empty(populated_collection):
    """Filtering on an absent key must return empty results, not an error."""
    _, collection = populated_collection
    results = collection.search(
        fields={DENSE_FIELD: {"query": dense_vec(), "limit": N_VECTORS}},
        filter=[{"nonexistent_key_xyz": {"$eq": "value"}}],
    )["results"][DENSE_FIELD]
    assert results == []


def test_filter_with_search_returns_sorted_results(populated_collection):
    """Filtered results must still be sorted by descending similarity."""
    _, collection = populated_collection
    results = collection.search(
        fields={DENSE_FIELD: {"query": dense_vec(), "limit": N_VECTORS}},
        filter=[{"category": {"$eq": "A"}}],
    )["results"][DENSE_FIELD]
    sims = [r["similarity"] for r in results]
    assert sims == sorted(sims, reverse=True)


# -- $gt operator (score > value) ---------------------------------------------


def test_filter_gt_all_results_above_threshold(populated_collection):
    """$gt filter must return only objects with score strictly above the threshold."""
    _, collection = populated_collection
    threshold = 40
    results = collection.search(
        fields={DENSE_FIELD: {"query": dense_vec(), "limit": N_VECTORS}},
        filter=[{"score": {"$gt": threshold}}],
    )["results"][DENSE_FIELD]
    assert len(results) > 0
    for r in results:
        assert parse_filter_field(r)["score"] > threshold, (
            f"score {parse_filter_field(r)['score']} not > {threshold}"
        )


@_XFAIL_ANN
def test_filter_gt_exact_count(populated_collection):
    """$gt filter on score > 40 must return exactly 9 objects (scores 41-49)."""
    _, collection = populated_collection
    results = collection.search(
        fields={DENSE_FIELD: {"query": dense_vec(), "limit": N_VECTORS}},
        filter=[{"score": {"$gt": 40}}],
    )["results"][DENSE_FIELD]
    assert len(results) == 9


# -- $gte operator (score >= value) -------------------------------------------


def test_filter_gte_all_results_at_or_above_threshold(populated_collection):
    """$gte filter must return objects with score >= the threshold."""
    _, collection = populated_collection
    threshold = 45
    results = collection.search(
        fields={DENSE_FIELD: {"query": dense_vec(), "limit": N_VECTORS}},
        filter=[{"score": {"$gte": threshold}}],
    )["results"][DENSE_FIELD]
    assert len(results) > 0
    for r in results:
        assert parse_filter_field(r)["score"] >= threshold


@_XFAIL_ANN
def test_filter_gte_exact_count(populated_collection):
    """$gte filter on score >= 45 must return exactly 5 objects (scores 45-49)."""
    _, collection = populated_collection
    results = collection.search(
        fields={DENSE_FIELD: {"query": dense_vec(), "limit": N_VECTORS}},
        filter=[{"score": {"$gte": 45}}],
    )["results"][DENSE_FIELD]
    assert len(results) == 5


# -- $lt operator (score < value) ---------------------------------------------


def test_filter_lt_all_results_below_threshold(populated_collection):
    """$lt filter must return only objects with score strictly below the threshold."""
    _, collection = populated_collection
    threshold = 5
    results = collection.search(
        fields={DENSE_FIELD: {"query": dense_vec(), "limit": N_VECTORS}},
        filter=[{"score": {"$lt": threshold}}],
    )["results"][DENSE_FIELD]
    assert len(results) > 0
    for r in results:
        assert parse_filter_field(r)["score"] < threshold


@_XFAIL_ANN
def test_filter_lt_exact_count(populated_collection):
    """$lt filter on score < 5 must return exactly 5 objects (scores 0-4)."""
    _, collection = populated_collection
    results = collection.search(
        fields={DENSE_FIELD: {"query": dense_vec(), "limit": N_VECTORS}},
        filter=[{"score": {"$lt": 5}}],
    )["results"][DENSE_FIELD]
    assert len(results) == 5


# -- $lte operator (score <= value) -------------------------------------------


def test_filter_lte_all_results_at_or_below_threshold(populated_collection):
    """$lte filter must return objects with score <= the threshold."""
    _, collection = populated_collection
    threshold = 4
    results = collection.search(
        fields={DENSE_FIELD: {"query": dense_vec(), "limit": N_VECTORS}},
        filter=[{"score": {"$lte": threshold}}],
    )["results"][DENSE_FIELD]
    assert len(results) > 0
    for r in results:
        assert parse_filter_field(r)["score"] <= threshold


@_XFAIL_ANN
def test_filter_lte_exact_count(populated_collection):
    """$lte filter on score <= 4 must return exactly 5 objects (scores 0-4)."""
    _, collection = populated_collection
    results = collection.search(
        fields={DENSE_FIELD: {"query": dense_vec(), "limit": N_VECTORS}},
        filter=[{"score": {"$lte": 4}}],
    )["results"][DENSE_FIELD]
    assert len(results) == 5


# -- Mixed comparison operators -----------------------------------------------


@_XFAIL_ANN
def test_filter_gt_and_lt_returns_range(populated_collection):
    """Combining $gt and $lt must return objects strictly inside the interval."""
    _, collection = populated_collection
    results = collection.search(
        fields={DENSE_FIELD: {"query": dense_vec(), "limit": N_VECTORS}},
        filter=[
            {"score": {"$gt": 10}},
            {"score": {"$lt": 20}},
        ],
    )["results"][DENSE_FIELD]
    assert len(results) == 9
    for r in results:
        s = parse_filter_field(r)["score"]
        assert 10 < s < 20


@_XFAIL_ANN
def test_filter_gte_and_lte_returns_closed_range(populated_collection):
    """Combining $gte and $lte must return objects inside the closed interval."""
    _, collection = populated_collection
    results = collection.search(
        fields={DENSE_FIELD: {"query": dense_vec(), "limit": N_VECTORS}},
        filter=[
            {"score": {"$gte": 10}},
            {"score": {"$lte": 20}},
        ],
    )["results"][DENSE_FIELD]
    assert len(results) == 11
    for r in results:
        s = parse_filter_field(r)["score"]
        assert 10 <= s <= 20


def test_filter_eq_and_gte(populated_collection):
    """category='A' AND score>=30 must return the correct objects."""
    _, collection = populated_collection
    results = collection.search(
        fields={DENSE_FIELD: {"query": dense_vec(), "limit": N_VECTORS}},
        filter=[
            {"category": {"$eq": "A"}},
            {"score": {"$gte": 30}},
        ],
    )["results"][DENSE_FIELD]
    for r in results:
        flt = parse_filter_field(r)
        assert flt["category"] == "A"
        assert flt["score"] >= 30


# -- filters on sparse field --------------------------------------------------


def test_sparse_search_with_eq_filter(populated_sparse_collection):
    """Sparse search with a $eq filter must return only matching objects."""
    _, collection = populated_sparse_collection
    si, sv = sparse_vec(seed=0)
    results = collection.search(
        fields={SPARSE_FIELD: {"query": {"indices": si, "values": sv}, "limit": N_VECTORS}},
        filter=[{"category": {"$eq": "A"}}],
    )["results"][SPARSE_FIELD]
    assert len(results) > 0
    for r in results:
        assert parse_filter_field(r)["category"] == "A"


def test_sparse_search_filter_all_results_match(populated_sparse_collection):
    """All results from a $eq filter on sparse search must satisfy the condition."""
    _, collection = populated_sparse_collection
    si, sv = sparse_vec(seed=0)
    results = collection.search(
        fields={SPARSE_FIELD: {"query": {"indices": si, "values": sv}, "limit": N_VECTORS}},
        filter=[{"category": {"$eq": "B"}}],
    )["results"][SPARSE_FIELD]
    for r in results:
        assert parse_filter_field(r)["category"] == "B"


# -- filters on multi_vector field --------------------------------------------


def test_multi_vector_search_with_eq_filter(populated_mv_collection):
    """multi_vector search with a $eq filter must return only matching objects."""
    _, collection = populated_mv_collection
    results = collection.search(
        fields={MV_FIELD: {"query": multi_vec(seed=0), "limit": N_VECTORS}},
        filter=[{"category": {"$eq": "A"}}],
    )["results"][MV_FIELD]
    assert len(results) > 0
    for r in results:
        assert parse_filter_field(r)["category"] == "A"


@_XFAIL_ANN
def test_multi_vector_search_filter_exact_count(populated_mv_collection):
    """$eq filter on category 'B' in multi_vector search must return 17 objects."""
    _, collection = populated_mv_collection
    results = collection.search(
        fields={MV_FIELD: {"query": multi_vec(seed=0), "limit": N_VECTORS}},
        filter=[{"category": {"$eq": "B"}}],
    )["results"][MV_FIELD]
    assert len(results) == 17


# -- filters on hybrid (multi-field) search -----------------------------------


def test_hybrid_dense_search_with_eq_filter(populated_hybrid_collection):
    """Dense-field search with $eq filter must return only matching objects."""
    _, collection = populated_hybrid_collection
    results = collection.search(
        fields={DENSE_FIELD: {"query": dense_vec(HYBRID_DIM), "limit": N_VECTORS}},
        filter=[{"category": {"$eq": "A"}}],
    )["results"][DENSE_FIELD]
    assert len(results) > 0
    for r in results:
        assert parse_filter_field(r)["category"] == "A"


def test_hybrid_rrf_search_with_filter(populated_hybrid_collection):
    """RRF search with a filter must return only matching objects."""
    _, collection = populated_hybrid_collection
    si, sv = sparse_vec(seed=3)
    raw = collection.search(
        fields={
            DENSE_FIELD: {"query": dense_vec(HYBRID_DIM, seed=3), "limit": N_VECTORS * 5},
            SPARSE_FIELD: {"query": {"indices": si, "values": sv}, "limit": N_VECTORS * 5},
        },
        filter=[{"tags": {"$eq": "important"}}],
    )
    results = rerank(raw, limit=N_VECTORS)["results"]
    for r in results:
        assert parse_filter_field(r)["tags"] == "important"


@_XFAIL_ANN
def test_hybrid_filter_exact_count(populated_hybrid_collection):
    """$eq filter on category 'B' in hybrid search must return exactly 17 objects."""
    _, collection = populated_hybrid_collection
    results = collection.search(
        fields={DENSE_FIELD: {"query": dense_vec(HYBRID_DIM), "limit": N_VECTORS}},
        filter=[{"category": {"$eq": "B"}}],
    )["results"][DENSE_FIELD]
    assert len(results) == 17

"""
Tests for search() filter tuning parameters: prefilter_cardinality_threshold
and filter_boost_percentage.

Covers accepted value ranges and client-side validation for both parameters.
"""

import pytest
from helpers import DENSE_FIELD, N_VECTORS, dense_vec, parse_filter_field

# -- prefilter_cardinality_threshold ------------------------------------------


def test_prefilter_threshold_accepted(populated_collection):
    """prefilter_cardinality_threshold=10000 must be accepted without raising."""
    _, collection = populated_collection
    collection.search(
        fields={DENSE_FIELD: {"query": dense_vec(), "limit": N_VECTORS}},
        filter=[{"category": {"$eq": "A"}}],
        prefilter_cardinality_threshold=10000,
    )["results"][DENSE_FIELD]


def test_prefilter_threshold_returns_results(populated_collection):
    """prefilter_cardinality_threshold=10000 with a filter must return non-empty results."""
    _, collection = populated_collection
    results = collection.search(
        fields={DENSE_FIELD: {"query": dense_vec(), "limit": N_VECTORS}},
        filter=[{"category": {"$eq": "A"}}],
        prefilter_cardinality_threshold=10000,
    )["results"][DENSE_FIELD]
    assert len(results) > 0


def test_prefilter_threshold_filter_correctness(populated_collection):
    """All results returned with prefilter_cardinality_threshold must satisfy the filter."""
    _, collection = populated_collection
    results = collection.search(
        fields={DENSE_FIELD: {"query": dense_vec(), "limit": N_VECTORS}},
        filter=[{"category": {"$eq": "B"}}],
        prefilter_cardinality_threshold=10000,
    )["results"][DENSE_FIELD]
    assert len(results) > 0
    for r in results:
        flt = parse_filter_field(r)
        assert flt["category"] == "B", f"Expected 'B', got {flt['category']}"


def test_prefilter_threshold_min_boundary(populated_collection):
    """prefilter_cardinality_threshold=1000 (minimum) must be accepted without raising."""
    _, collection = populated_collection
    collection.search(
        fields={DENSE_FIELD: {"query": dense_vec(), "limit": N_VECTORS}},
        filter=[{"category": {"$eq": "A"}}],
        prefilter_cardinality_threshold=1000,
    )["results"][DENSE_FIELD]


def test_prefilter_threshold_max_boundary(populated_collection):
    """prefilter_cardinality_threshold=1000000 (maximum) must be accepted without raising."""
    _, collection = populated_collection
    collection.search(
        fields={DENSE_FIELD: {"query": dense_vec(), "limit": N_VECTORS}},
        filter=[{"category": {"$eq": "A"}}],
        prefilter_cardinality_threshold=1000000,
    )["results"][DENSE_FIELD]


def test_prefilter_threshold_below_min_raises(populated_collection):
    """prefilter_cardinality_threshold=999 (below minimum) must raise ValueError."""
    _, collection = populated_collection
    with pytest.raises(ValueError):
        collection.search(
            fields={DENSE_FIELD: {"query": dense_vec(), "limit": N_VECTORS}},
            filter=[{"category": {"$eq": "A"}}],
            prefilter_cardinality_threshold=999,
        )


def test_prefilter_threshold_above_max_raises(populated_collection):
    """prefilter_cardinality_threshold=1000001 (above maximum) must raise ValueError."""
    _, collection = populated_collection
    with pytest.raises(ValueError):
        collection.search(
            fields={DENSE_FIELD: {"query": dense_vec(), "limit": N_VECTORS}},
            filter=[{"category": {"$eq": "A"}}],
            prefilter_cardinality_threshold=1000001,
        )


# -- filter_boost_percentage --------------------------------------------------


def test_filter_boost_accepted(populated_collection):
    """filter_boost_percentage=50.0 must be accepted without raising."""
    _, collection = populated_collection
    collection.search(
        fields={DENSE_FIELD: {"query": dense_vec(), "limit": N_VECTORS}},
        filter=[{"category": {"$eq": "A"}}],
        filter_boost_percentage=50.0,
    )["results"][DENSE_FIELD]


def test_filter_boost_returns_results(populated_collection):
    """filter_boost_percentage=50.0 with a filter must return non-empty results."""
    _, collection = populated_collection
    results = collection.search(
        fields={DENSE_FIELD: {"query": dense_vec(), "limit": N_VECTORS}},
        filter=[{"category": {"$eq": "A"}}],
        filter_boost_percentage=50.0,
    )["results"][DENSE_FIELD]
    assert len(results) > 0


def test_filter_boost_zero_accepted(populated_collection):
    """filter_boost_percentage=0.0 (minimum) must be accepted without raising."""
    _, collection = populated_collection
    collection.search(
        fields={DENSE_FIELD: {"query": dense_vec(), "limit": N_VECTORS}},
        filter=[{"category": {"$eq": "A"}}],
        filter_boost_percentage=0.0,
    )["results"][DENSE_FIELD]


def test_filter_boost_100_accepted(populated_collection):
    """filter_boost_percentage=100.0 (maximum) must be accepted without raising."""
    _, collection = populated_collection
    collection.search(
        fields={DENSE_FIELD: {"query": dense_vec(), "limit": N_VECTORS}},
        filter=[{"category": {"$eq": "A"}}],
        filter_boost_percentage=100.0,
    )["results"][DENSE_FIELD]


def test_filter_boost_below_min_raises(populated_collection):
    """filter_boost_percentage=-1.0 (below minimum) must raise ValueError."""
    _, collection = populated_collection
    with pytest.raises(ValueError):
        collection.search(
            fields={DENSE_FIELD: {"query": dense_vec(), "limit": N_VECTORS}},
            filter=[{"category": {"$eq": "A"}}],
            filter_boost_percentage=-1.0,
        )


def test_filter_boost_above_max_raises(populated_collection):
    """filter_boost_percentage=101.0 (above maximum) must raise ValueError."""
    _, collection = populated_collection
    with pytest.raises(ValueError):
        collection.search(
            fields={DENSE_FIELD: {"query": dense_vec(), "limit": N_VECTORS}},
            filter=[{"category": {"$eq": "A"}}],
            filter_boost_percentage=101.0,
        )


# -- combined and edge-case tests ---------------------------------------------


def test_both_params_together(populated_collection):
    """Both filter params set together must succeed."""
    _, collection = populated_collection
    results = collection.search(
        fields={DENSE_FIELD: {"query": dense_vec(), "limit": N_VECTORS}},
        filter=[{"category": {"$eq": "A"}}],
        prefilter_cardinality_threshold=10000,
        filter_boost_percentage=25.0,
    )["results"][DENSE_FIELD]
    assert isinstance(results, list)


def test_filter_params_without_filter(populated_collection):
    """Both filter params with no filter clause must still work."""
    _, collection = populated_collection
    results = collection.search(
        fields={DENSE_FIELD: {"query": dense_vec(), "limit": N_VECTORS}},
        prefilter_cardinality_threshold=10000,
        filter_boost_percentage=50.0,
    )["results"][DENSE_FIELD]
    assert isinstance(results, list)
    assert len(results) > 0

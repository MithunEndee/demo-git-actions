"""
test_04_query_filters.py

Tests for filtered queries:
  - $eq  operator
  - $in  operator
  - $range operator
  - Combined filters (AND logic)
  - filter_boost_percentage
  - prefilter_cardinality_threshold
  - Both tuning params together

Filter layout in populated_index (N=50 vectors):
  category : "A"|"B"|"C"  (i%3)
  priority : 0-4           (i%5)
  score    : 0-49          (i itself)
  tags     : "important"|"normal"  (even/odd i)

Expected match counts (used in assertions):
  category "A"            → 17
  category "B"            → 17
  category "C"            → 16
  tags "important"        → 25
  score in [10,20]        → 11
  category in ["A","B"]   → 34
"""

import pytest

from conftest import N_VECTORS, dense_vec

# Use high prefilter threshold to guarantee brute-force recall in all filter tests
_BF = 1_000_000


# ── $eq operator ──────────────────────────────────────────────────────────

def test_filter_eq_all_results_match(populated_index):
    _, index = populated_index
    results = index.query(
        vector=dense_vec(),
        top_k=N_VECTORS,
        filter=[{"category": {"$eq": "A"}}],
        prefilter_cardinality_threshold=_BF,
    )
    assert len(results) > 0
    for r in results:
        assert r["filter"]["category"] == "A", f"Expected 'A', got {r['filter']['category']}"


def test_filter_eq_exact_count(populated_index):
    _, index = populated_index
    results = index.query(
        vector=dense_vec(),
        top_k=N_VECTORS,
        filter=[{"category": {"$eq": "B"}}],
        prefilter_cardinality_threshold=_BF,
    )
    assert len(results) == 17


def test_filter_eq_tags_important(populated_index):
    _, index = populated_index
    results = index.query(
        vector=dense_vec(),
        top_k=N_VECTORS,
        filter=[{"tags": {"$eq": "important"}}],
        prefilter_cardinality_threshold=_BF,
    )
    assert len(results) == 25
    for r in results:
        assert r["filter"]["tags"] == "important"


def test_filter_eq_tags_normal(populated_index):
    _, index = populated_index
    results = index.query(
        vector=dense_vec(),
        top_k=N_VECTORS,
        filter=[{"tags": {"$eq": "normal"}}],
        prefilter_cardinality_threshold=_BF,
    )
    assert len(results) == 25
    for r in results:
        assert r["filter"]["tags"] == "normal"


def test_filter_eq_no_match_returns_empty(populated_index):
    _, index = populated_index
    results = index.query(
        vector=dense_vec(),
        top_k=N_VECTORS,
        filter=[{"category": {"$eq": "NONEXISTENT"}}],
        prefilter_cardinality_threshold=_BF,
    )
    assert len(results) == 0


# ── $in operator ──────────────────────────────────────────────────────────

def test_filter_in_single_value(populated_index):
    _, index = populated_index
    results = index.query(
        vector=dense_vec(),
        top_k=N_VECTORS,
        filter=[{"category": {"$in": ["C"]}}],
        prefilter_cardinality_threshold=_BF,
    )
    assert len(results) == 16
    for r in results:
        assert r["filter"]["category"] == "C"


def test_filter_in_two_values(populated_index):
    _, index = populated_index
    results = index.query(
        vector=dense_vec(),
        top_k=N_VECTORS,
        filter=[{"category": {"$in": ["A", "B"]}}],
        prefilter_cardinality_threshold=_BF,
    )
    assert len(results) == 34
    for r in results:
        assert r["filter"]["category"] in ("A", "B")


def test_filter_in_all_values(populated_index):
    """$in with all three categories should return all vectors."""
    _, index = populated_index
    results = index.query(
        vector=dense_vec(),
        top_k=N_VECTORS,
        filter=[{"category": {"$in": ["A", "B", "C"]}}],
        prefilter_cardinality_threshold=_BF,
    )
    assert len(results) == N_VECTORS


def test_filter_in_tags(populated_index):
    _, index = populated_index
    results = index.query(
        vector=dense_vec(),
        top_k=N_VECTORS,
        filter=[{"tags": {"$in": ["important", "normal"]}}],
        prefilter_cardinality_threshold=_BF,
    )
    assert len(results) == N_VECTORS


# ── $range operator ───────────────────────────────────────────────────────

def test_filter_range_returns_correct_count(populated_index):
    _, index = populated_index
    # score in [10, 20] → i = 10..20 → 11 vectors
    results = index.query(
        vector=dense_vec(),
        top_k=N_VECTORS,
        filter=[{"score": {"$range": [10, 20]}}],
        prefilter_cardinality_threshold=_BF,
    )
    assert len(results) == 11


def test_filter_range_all_results_within_bounds(populated_index):
    _, index = populated_index
    lo, hi = 5, 15
    results = index.query(
        vector=dense_vec(),
        top_k=N_VECTORS,
        filter=[{"score": {"$range": [lo, hi]}}],
        prefilter_cardinality_threshold=_BF,
    )
    for r in results:
        score = r["filter"]["score"]
        assert lo <= score <= hi, f"score {score} outside [{lo},{hi}]"


def test_filter_range_full_span(populated_index):
    """Range spanning all scores should return all vectors."""
    _, index = populated_index
    results = index.query(
        vector=dense_vec(),
        top_k=N_VECTORS,
        filter=[{"score": {"$range": [0, 49]}}],
        prefilter_cardinality_threshold=_BF,
    )
    assert len(results) == N_VECTORS


def test_filter_range_narrow(populated_index):
    _, index = populated_index
    # score in [25, 25] → only vec_0025
    results = index.query(
        vector=dense_vec(),
        top_k=N_VECTORS,
        filter=[{"score": {"$range": [25, 25]}}],
        prefilter_cardinality_threshold=_BF,
    )
    assert len(results) == 1
    assert results[0]["id"] == "vec_0025"


# ── Combined filters (AND logic) ──────────────────────────────────────────

def test_filter_and_eq_and_eq(populated_index):
    """category='A' AND tags='important' → 9 vectors (i=0,6,12,18,24,30,36,42,48)."""
    _, index = populated_index
    results = index.query(
        vector=dense_vec(),
        top_k=N_VECTORS,
        filter=[
            {"category": {"$eq": "A"}},
            {"tags": {"$eq": "important"}},
        ],
        prefilter_cardinality_threshold=_BF,
    )
    assert len(results) == 9
    for r in results:
        assert r["filter"]["category"] == "A"
        assert r["filter"]["tags"] == "important"


def test_filter_and_eq_and_range(populated_index):
    """category='A' AND score in [0,29] → i=0,3,6,...,27 → 10 vectors."""
    _, index = populated_index
    results = index.query(
        vector=dense_vec(),
        top_k=N_VECTORS,
        filter=[
            {"category": {"$eq": "A"}},
            {"score": {"$range": [0, 29]}},
        ],
        prefilter_cardinality_threshold=_BF,
    )
    assert len(results) == 10  # 0,3,6,9,12,15,18,21,24,27
    for r in results:
        assert r["filter"]["category"] == "A"
        assert 0 <= r["filter"]["score"] <= 29


def test_filter_and_in_and_range(populated_index):
    """category in ['A','B'] AND score in [0,9] → i=0..9 = 10 vectors."""
    _, index = populated_index
    results = index.query(
        vector=dense_vec(),
        top_k=N_VECTORS,
        filter=[
            {"category": {"$in": ["A", "B"]}},
            {"score": {"$range": [0, 9]}},
        ],
        prefilter_cardinality_threshold=_BF,
    )
    # i=0..9: category "A" for 0,3,6,9 (4) + "B" for 1,4,7 (3) = 7, "C" for 2,5,8 excluded
    assert len(results) == 7
    for r in results:
        assert r["filter"]["category"] in ("A", "B")
        assert r["filter"]["score"] <= 9


def test_filter_three_conditions(populated_index):
    """category='A' AND tags='important' AND score in [0,29]."""
    _, index = populated_index
    results = index.query(
        vector=dense_vec(),
        top_k=N_VECTORS,
        filter=[
            {"category": {"$eq": "A"}},
            {"tags": {"$eq": "important"}},
            {"score": {"$range": [0, 29]}},
        ],
        prefilter_cardinality_threshold=_BF,
    )
    # i=0,6,12,18,24 → 5 vectors
    assert len(results) == 5
    for r in results:
        assert r["filter"]["category"] == "A"
        assert r["filter"]["tags"] == "important"
        assert r["filter"]["score"] <= 29


# ── filter_boost_percentage ───────────────────────────────────────────────

@pytest.mark.parametrize("boost", [0, 10, 25, 50, 100, 200, 400])
def test_filter_boost_percentage_accepted(populated_index, boost):
    _, index = populated_index
    results = index.query(
        vector=dense_vec(),
        top_k=5,
        filter=[{"category": {"$eq": "A"}}],
        filter_boost_percentage=boost,
    )
    assert isinstance(results, list)


def test_filter_boost_results_still_satisfy_filter(populated_index):
    _, index = populated_index
    results = index.query(
        vector=dense_vec(),
        top_k=10,
        filter=[{"tags": {"$eq": "important"}}],
        filter_boost_percentage=50,
        prefilter_cardinality_threshold=_BF,
    )
    for r in results:
        assert r["filter"]["tags"] == "important"


# ── prefilter_cardinality_threshold ──────────────────────────────────────

@pytest.mark.parametrize("threshold", [1_000, 5_000, 10_000, 100_000, 1_000_000])
def test_prefilter_threshold_accepted(populated_index, threshold):
    _, index = populated_index
    results = index.query(
        vector=dense_vec(),
        top_k=5,
        filter=[{"category": {"$eq": "A"}}],
        prefilter_cardinality_threshold=threshold,
    )
    assert isinstance(results, list)


def test_prefilter_threshold_results_satisfy_filter(populated_index):
    _, index = populated_index
    results = index.query(
        vector=dense_vec(),
        top_k=20,
        filter=[{"category": {"$eq": "B"}}],
        prefilter_cardinality_threshold=1_000,
    )
    for r in results:
        assert r["filter"]["category"] == "B"


# ── Both tuning params together ───────────────────────────────────────────

def test_filter_both_tuning_params_together(populated_index):
    _, index = populated_index
    results = index.query(
        vector=dense_vec(),
        top_k=10,
        filter=[{"category": {"$eq": "A"}}],
        prefilter_cardinality_threshold=5_000,
        filter_boost_percentage=25,
    )
    for r in results:
        assert r["filter"]["category"] == "A"

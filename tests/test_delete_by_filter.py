"""
Tests for Collection.delete_by_filter() - bulk-delete objects matching a filter.

Covers return structure, count accuracy, filter operators ($eq, $in, $range,
$gt, $gte, $lt, $lte, AND conditions), no-match case, and client-side validation.
"""

from helpers import (
    DENSE_FIELD,
    MV_FIELD,
    N_VECTORS,
    dense_vec,
    multi_vec,
    parse_filter_field,
)

# -- return structure ---------------------------------------------------------


def test_delete_by_filter_returns_dict(populated_collection):
    """delete_by_filter must return a dict."""
    _, collection = populated_collection
    result = collection.delete_by_filter([{"category": {"$eq": "C"}}])
    assert isinstance(result, dict)


def test_delete_by_filter_returns_deleted_key(populated_collection):
    """delete_by_filter must return a dict with a 'deleted' key."""
    _, collection = populated_collection
    result = collection.delete_by_filter([{"category": {"$eq": "C"}}])
    assert "deleted" in result, f"Expected 'deleted' key, got {list(result.keys())}"


def test_delete_by_filter_deleted_value_is_int(populated_collection):
    """The 'deleted' value must be a non-negative integer."""
    _, collection = populated_collection
    result = collection.delete_by_filter([{"category": {"$eq": "A"}}])
    assert isinstance(result["deleted"], int)
    assert result["deleted"] >= 0


# -- correctness: objects must be gone from search ----------------------------


def test_delete_by_filter_eq_objects_absent_from_search(populated_collection):
    """After delete_by_filter $eq, deleted objects must not appear in search."""
    _, collection = populated_collection
    collection.delete_by_filter([{"category": {"$eq": "B"}}])
    results = collection.search(
        fields={DENSE_FIELD: {"query": dense_vec(), "limit": N_VECTORS}},
    )["results"][DENSE_FIELD]
    for r in results:
        flt = parse_filter_field(r)
        assert flt.get("category") != "B", f"Deleted object still present: {r['id']}"


def test_delete_by_filter_preserves_non_matching_objects(populated_collection):
    """delete_by_filter must not delete objects that do not match the filter."""
    _, collection = populated_collection
    collection.delete_by_filter([{"category": {"$eq": "C"}}])
    results = collection.search(
        fields={DENSE_FIELD: {"query": dense_vec(), "limit": N_VECTORS}},
    )["results"][DENSE_FIELD]
    remaining_cats = {parse_filter_field(r).get("category") for r in results}
    # "A" and "B" should still be present
    assert "A" in remaining_cats or "B" in remaining_cats


# -- count accuracy -----------------------------------------------------------


def test_delete_by_filter_eq_count_exact(populated_collection):
    """delete_by_filter $eq on category 'B' must delete exactly 17 objects."""
    _, collection = populated_collection
    result = collection.delete_by_filter([{"category": {"$eq": "B"}}])
    assert result["deleted"] == 17


def test_delete_by_filter_eq_tags_important_count(populated_collection):
    """delete_by_filter $eq on tags='important' must delete exactly 25 objects."""
    _, collection = populated_collection
    result = collection.delete_by_filter([{"tags": {"$eq": "important"}}])
    assert result["deleted"] == 25


# -- $in operator -------------------------------------------------------------


def test_delete_by_filter_in_single_value(populated_collection):
    """delete_by_filter $in with one value must behave like $eq."""
    _, collection = populated_collection
    result = collection.delete_by_filter([{"category": {"$in": ["C"]}}])
    assert result["deleted"] == 16


def test_delete_by_filter_in_two_values(populated_collection):
    """delete_by_filter $in with two categories must delete both sets."""
    _, collection = populated_collection
    result = collection.delete_by_filter([{"category": {"$in": ["A", "B"]}}])
    assert result["deleted"] == 34


# -- $range operator ----------------------------------------------------------


def test_delete_by_filter_range_count_exact(populated_collection):
    """delete_by_filter $range [40, 49] must delete exactly 10 objects."""
    _, collection = populated_collection
    result = collection.delete_by_filter([{"score": {"$range": [40, 49]}}])
    assert result["deleted"] == 10


def test_delete_by_filter_range_objects_gone(populated_collection):
    """After $range delete, objects in that score range must be absent."""
    _, collection = populated_collection
    collection.delete_by_filter([{"score": {"$range": [40, 49]}}])
    results = collection.search(
        fields={DENSE_FIELD: {"query": dense_vec(), "limit": N_VECTORS}},
    )["results"][DENSE_FIELD]
    for r in results:
        score = parse_filter_field(r).get("score", -1)
        assert not (40 <= score <= 49), f"Deleted object still present: score={score}"


# -- $gt / $gte / $lt / $lte operators ----------------------------------------
#
# score = i (0..49); counts:
#   score > 40             -> 9  (41..49)
#   score >= 40            -> 10 (40..49)
#   score < 10             -> 10 (0..9)
#   score <= 10            -> 11 (0..10)
#   score > 10 AND < 20    ->  9 (11..19)
#   score >= 10 AND <= 20  -> 11 (10..20)


def test_delete_by_filter_gt_count_exact(populated_collection):
    """delete_by_filter $gt score>40 must delete exactly 9 objects."""
    _, collection = populated_collection
    result = collection.delete_by_filter([{"score": {"$gt": 40}}])
    assert result["deleted"] == 9


def test_delete_by_filter_gte_count_exact(populated_collection):
    """delete_by_filter $gte score>=40 must delete exactly 10 objects."""
    _, collection = populated_collection
    result = collection.delete_by_filter([{"score": {"$gte": 40}}])
    assert result["deleted"] == 10


def test_delete_by_filter_lt_count_exact(populated_collection):
    """delete_by_filter $lt score<10 must delete exactly 10 objects."""
    _, collection = populated_collection
    result = collection.delete_by_filter([{"score": {"$lt": 10}}])
    assert result["deleted"] == 10


def test_delete_by_filter_lte_count_exact(populated_collection):
    """delete_by_filter $lte score<=10 must delete exactly 11 objects."""
    _, collection = populated_collection
    result = collection.delete_by_filter([{"score": {"$lte": 10}}])
    assert result["deleted"] == 11


def test_delete_by_filter_gt_and_lt_open_range(populated_collection):
    """delete_by_filter $gt 10 AND $lt 20 (open range) must delete exactly 9 objects."""
    _, collection = populated_collection
    result = collection.delete_by_filter(
        [{"score": {"$gt": 10}}, {"score": {"$lt": 20}}]
    )
    assert result["deleted"] == 9


def test_delete_by_filter_gte_and_lte_closed_range(populated_collection):
    """delete_by_filter $gte 10 AND $lte 20 (closed range) must delete exactly 11 objects."""
    _, collection = populated_collection
    result = collection.delete_by_filter(
        [{"score": {"$gte": 10}}, {"score": {"$lte": 20}}]
    )
    assert result["deleted"] == 11


# -- no-match case ------------------------------------------------------------


def test_delete_by_filter_no_match_returns_zero(populated_collection):
    """delete_by_filter with a filter matching nothing must return deleted=0."""
    _, collection = populated_collection
    result = collection.delete_by_filter([{"category": {"$eq": "ZZZNOMATCH"}}])
    assert result["deleted"] == 0


# -- multi-condition AND logic ------------------------------------------------


def test_delete_by_filter_and_eq_and_eq(populated_collection):
    """delete_by_filter AND (category='A', tags='important') must delete 9 objects."""
    _, collection = populated_collection
    result = collection.delete_by_filter(
        [
            {"category": {"$eq": "A"}},
            {"tags": {"$eq": "important"}},
        ]
    )
    assert result["deleted"] == 9


# -- delete entire corpus -----------------------------------------------------


def test_delete_by_filter_all_objects(populated_collection):
    """Deleting all three categories must empty the collection."""
    _, collection = populated_collection
    r1 = collection.delete_by_filter([{"category": {"$eq": "A"}}])
    r2 = collection.delete_by_filter([{"category": {"$eq": "B"}}])
    r3 = collection.delete_by_filter([{"category": {"$eq": "C"}}])
    total = r1["deleted"] + r2["deleted"] + r3["deleted"]
    assert total == N_VECTORS
    results = collection.search(
        fields={DENSE_FIELD: {"query": dense_vec(), "limit": N_VECTORS}},
    )["results"][DENSE_FIELD]
    assert results == []


# -- multi_vector collection --------------------------------------------------


def test_mv_delete_by_filter_returns_dict(populated_mv_collection):
    """delete_by_filter() on a multi_vector collection must return a dict."""
    _, collection = populated_mv_collection
    result = collection.delete_by_filter(filter=[{"category": {"$eq": "C"}}])
    assert isinstance(result, dict)


def test_mv_delete_by_filter_has_deleted_key(populated_mv_collection):
    """delete_by_filter() response must contain a 'deleted' key."""
    _, collection = populated_mv_collection
    result = collection.delete_by_filter(filter=[{"category": {"$eq": "C"}}])
    assert "deleted" in result


def test_mv_delete_by_filter_count_exact(populated_mv_collection):
    """delete_by_filter($eq C) must delete exactly 16 objects (i%3==2, N=50)."""
    _, collection = populated_mv_collection
    result = collection.delete_by_filter(filter=[{"category": {"$eq": "C"}}])
    assert result["deleted"] == 16


def test_mv_delete_by_filter_objects_removed_from_search(populated_mv_collection):
    """After delete_by_filter(), deleted objects must not appear in search."""
    _, collection = populated_mv_collection
    collection.delete_by_filter(filter=[{"category": {"$eq": "C"}}])
    results = collection.search(
        fields={MV_FIELD: {"query": multi_vec(seed=0), "limit": N_VECTORS}},
        filter=[{"category": {"$eq": "C"}}],
    )["results"][MV_FIELD]
    assert len(results) == 0


def test_mv_delete_by_filter_no_match_returns_zero(populated_mv_collection):
    """delete_by_filter() with a filter matching nothing must return deleted=0."""
    _, collection = populated_mv_collection
    result = collection.delete_by_filter(
        filter=[{"category": {"$eq": "NONEXISTENT_CATEGORY"}}]
    )
    assert result["deleted"] == 0


def test_mv_delete_by_filter_in_operator(populated_mv_collection):
    """delete_by_filter($in [A,B]) must delete 34 objects."""
    _, collection = populated_mv_collection
    result = collection.delete_by_filter(filter=[{"category": {"$in": ["A", "B"]}}])
    assert result["deleted"] == 34


def test_mv_delete_by_filter_and_conditions(populated_mv_collection):
    """delete_by_filter() with AND conditions must delete the correct subset."""
    _, collection = populated_mv_collection
    result = collection.delete_by_filter(
        filter=[{"category": {"$eq": "A"}}, {"tags": {"$eq": "important"}}]
    )
    assert result["deleted"] == 9

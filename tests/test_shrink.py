"""
Tests for Collection.shrink() - defragments on-disk storage after deletions.

Covers response structure, searchability after shrink, and shrink after delete
on both dense and multi_vector collections.
"""

from helpers import DENSE_FIELD, MV_FIELD, dense_vec, multi_vec

# -- dense collection ----------------------------------------------------------


def test_shrink_returns_dict(populated_collection):
    """shrink() must return a dict."""
    _, collection = populated_collection
    result = collection.shrink()
    assert isinstance(result, dict)


def test_shrink_on_empty_collection_returns_dict(empty_collection):
    """shrink() on an empty collection must return a dict."""
    _, collection = empty_collection
    result = collection.shrink()
    assert isinstance(result, dict)


def test_shrink_collection_still_searchable_after(populated_collection):
    """After shrink(), the collection must still return search results."""
    _, collection = populated_collection
    collection.shrink()
    results = collection.search(
        fields={DENSE_FIELD: {"query": dense_vec(), "limit": 5}}
    )["results"][DENSE_FIELD]
    assert isinstance(results, list)
    assert len(results) > 0


def test_shrink_after_delete_returns_dict(populated_collection):
    """shrink() after deleting objects must succeed."""
    _, collection = populated_collection
    collection.delete_object("vec_0000")
    collection.delete_object("vec_0001")
    result = collection.shrink()
    assert isinstance(result, dict)


# -- multi_vector collection ---------------------------------------------------


def test_mv_shrink_returns_dict(populated_mv_collection):
    """shrink() on a multi_vector collection must return a dict."""
    _, collection = populated_mv_collection
    result = collection.shrink()
    assert isinstance(result, dict)


def test_mv_shrink_collection_still_searchable_after(populated_mv_collection):
    """After shrink(), a multi_vector collection must still return search results."""
    _, collection = populated_mv_collection
    collection.shrink()
    results = collection.search(
        fields={MV_FIELD: {"query": multi_vec(seed=0), "limit": 5}}
    )["results"][MV_FIELD]
    assert isinstance(results, list)
    assert len(results) > 0


def test_mv_shrink_after_delete_by_filter(populated_mv_collection):
    """shrink() after delete_by_filter() on a multi_vector collection must succeed."""
    _, collection = populated_mv_collection
    collection.delete_by_filter(filter=[{"category": {"$eq": "C"}}])
    result = collection.shrink()
    assert isinstance(result, dict)

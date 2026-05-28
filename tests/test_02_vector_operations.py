"""
test_02_vector_operations.py

Tests for vector CRUD operations:
  - upsert  (single, batch, with/without meta+filter, update via re-insert)
  - get_vector
  - update_filters
  - delete_vector (by ID)
  - delete_with_filter
"""

import pytest

from helpers import (
    DIM, N_VECTORS,
    dense_vec, make_item, uid,
)


# === upsert ===

def test_upsert_single_vector(empty_index):
    """Upserting a single vector must return a success response."""
    _, index = empty_index
    result = index.upsert([{"id": "v1", "vector": dense_vec()}])
    assert "success" in result.lower()


def test_upsert_batch_10_vectors(empty_index):
    """Upserting a batch of 10 vectors must return a success response."""
    _, index = empty_index
    batch = [{"id": f"b_{i}", "vector": dense_vec(seed=i)} for i in range(10)]
    result = index.upsert(batch)
    assert "success" in result.lower()


def test_upsert_batch_500_vectors(empty_index):
    """Batch close to the 1 000-vector limit."""
    _, index = empty_index
    batch = [{"id": f"big_{i}", "vector": dense_vec(seed=i)} for i in range(500)]
    result = index.upsert(batch)
    assert "success" in result.lower()


def test_upsert_with_meta_only(empty_index):
    """Upserting a vector with only meta fields must succeed."""
    _, index = empty_index
    result = index.upsert([{
        "id": "meta_only",
        "vector": dense_vec(),
        "meta": {"title": "Hello", "value": 42},
    }])
    assert "success" in result.lower()


def test_upsert_with_filter_only(empty_index):
    """Upserting a vector with only filter fields must succeed."""
    _, index = empty_index
    result = index.upsert([{
        "id": "filt_only",
        "vector": dense_vec(),
        "filter": {"category": "X", "score": 5},
    }])
    assert "success" in result.lower()


def test_upsert_with_meta_and_filter(empty_index):
    """Upserting a vector with both meta and filter fields must succeed."""
    _, index = empty_index
    result = index.upsert([{
        "id": "full",
        "vector": dense_vec(),
        "meta": {"text": "doc"},
        "filter": {"category": "A", "score": 10, "tags": "important"},
    }])
    assert "success" in result.lower()


def test_upsert_without_meta_or_filter(empty_index):
    """Upserting a bare vector with no meta or filter must succeed."""
    _, index = empty_index
    result = index.upsert([{"id": "bare", "vector": dense_vec()}])
    assert "success" in result.lower()


def test_upsert_overwrites_existing_id(empty_index):
    """Re-inserting the same ID must succeed (upsert semantics)."""
    _, index = empty_index
    v = dense_vec()
    index.upsert([{"id": "dup", "vector": v, "meta": {"v": 1}}])
    result = index.upsert([{"id": "dup", "vector": v, "meta": {"v": 2}}])
    assert "success" in result.lower()


def test_upsert_all_precision_indexes(client):
    """Upsert vectors into indexes with different precision types."""
    from endee import Precision
    from helpers import safe_delete

    for precision in [Precision.FLOAT32, Precision.FLOAT16, Precision.INT16,
                      Precision.INT8, Precision.BINARY2]:
        name = uid("prec")
        try:
            client.create_index(name=name, dimension=DIM, space_type="cosine", precision=precision)
            index = client.get_index(name)
            result = index.upsert([{"id": "v1", "vector": dense_vec(seed=0)}])
            assert "success" in result.lower(), f"Failed for precision {precision}"
        finally:
            safe_delete(client, name)


def test_upsert_all_space_type_indexes(client):
    """Upsert vectors into indexes with each space type."""
    from helpers import safe_delete

    for space_type in ["cosine", "l2", "ip"]:
        from endee import Precision
        name = uid("st")
        try:
            client.create_index(name=name, dimension=DIM, space_type=space_type,
                                precision=Precision.INT8)
            index = client.get_index(name)
            result = index.upsert([{"id": "v1", "vector": dense_vec(seed=0)}])
            assert "success" in result.lower(), f"Failed for space_type {space_type}"
        finally:
            safe_delete(client, name)


# === get_vector ===

def test_get_vector_returns_correct_structure(populated_index):
    """get_vector must return a dict containing all required keys."""
    _, index = populated_index
    vec = index.get_vector("vec_0000")
    required_keys = {"id", "meta", "filter", "norm", "vector"}
    assert required_keys.issubset(vec.keys()), f"Missing keys: {required_keys - vec.keys()}"


def test_get_vector_id_matches(populated_index):
    """get_vector must return the correct id for the requested vector."""
    _, index = populated_index
    vec = index.get_vector("vec_0001")
    assert vec["id"] == "vec_0001"


def test_get_vector_meta_preserved(populated_index):
    """get_vector must return the meta fields exactly as upserted."""
    _, index = populated_index
    vec = index.get_vector("vec_0005")
    assert vec["meta"]["index"] == 5
    assert vec["meta"]["text"] == "Document 5"


def test_get_vector_filter_preserved(populated_index):
    """get_vector must return the filter fields exactly as upserted."""
    _, index = populated_index
    vec = index.get_vector("vec_0000")
    assert vec["filter"]["category"] == "A"
    assert vec["filter"]["score"] == 0
    assert vec["filter"]["tags"] == "important"


def test_get_vector_has_vector_data(populated_index):
    """get_vector must return a vector list of the correct dimension."""
    _, index = populated_index
    vec = index.get_vector("vec_0002")
    assert isinstance(vec["vector"], list)
    assert len(vec["vector"]) == DIM


def test_get_vector_norm_is_positive(populated_index):
    """get_vector must return a positive float norm."""
    _, index = populated_index
    vec = index.get_vector("vec_0003")
    assert isinstance(vec["norm"], float)
    assert vec["norm"] > 0


# === update_filters ===

def test_update_filters_single_vector(populated_index):
    """update_filters on a single vector must return a non-empty confirmation."""
    _, index = populated_index
    result = index.update_filters([
        {"id": "vec_0010", "filter": {"category": "Z", "score": 99}},
    ])
    assert result  # server returns a non-empty confirmation


def test_update_filters_multiple_vectors(populated_index):
    """update_filters on multiple vectors at once must return a non-empty confirmation."""
    _, index = populated_index
    result = index.update_filters([
        {"id": "vec_0020", "filter": {"category": "X"}},
        {"id": "vec_0021", "filter": {"category": "Y"}},
        {"id": "vec_0022", "filter": {"category": "Z"}},
    ])
    assert result


def test_update_filters_reflected_in_get_vector(populated_index):
    """After updating a filter, get_vector must return the new value."""
    _, index = populated_index
    index.update_filters([{"id": "vec_0030", "filter": {"category": "UPDATED"}}])
    vec = index.get_vector("vec_0030")
    assert vec["filter"]["category"] == "UPDATED"


# === delete_vector (by ID) ===

def test_delete_vector_returns_rows_deleted(populated_index):
    """delete_vector must return a response confirming rows were deleted."""
    _, index = populated_index
    result = index.delete_vector("vec_0040")
    assert "deleted" in result.lower()


def test_delete_vector_not_returned_in_get_vector(populated_index):
    """After deletion, get_vector should raise an exception."""
    _, index = populated_index
    index.delete_vector("vec_0041")
    with pytest.raises(Exception):
        index.get_vector("vec_0041")


def test_delete_vector_not_in_query_results(populated_index):
    """Deleted vector should not appear in top-50 results."""
    _, index = populated_index
    target_id = "vec_0042"
    query_vec = dense_vec(seed=42)  # arbitrary query

    index.delete_vector(target_id)
    results = index.query(vector=query_vec, top_k=N_VECTORS)
    returned_ids = {r["id"] for r in results}
    assert target_id not in returned_ids


# === delete_with_filter ===

def test_delete_with_filter_eq(empty_index, client):
    """delete_with_filter using $eq should remove matching vectors."""
    name, index = empty_index
    # Insert 6 vectors: 3 with tag "to_delete", 3 with tag "keep"
    batch = [
        {"id": f"d_{i}", "vector": dense_vec(seed=i),
         "filter": {"tag": "to_delete" if i < 3 else "keep"}}
        for i in range(6)
    ]
    index.upsert(batch)
    index.delete_with_filter([{"tag": {"$eq": "to_delete"}}])

    # "keep" vectors should still be queryable
    results = index.query(
        vector=dense_vec(), top_k=10,
        filter=[{"tag": {"$eq": "keep"}}],
        prefilter_cardinality_threshold=1_000_000,
    )
    returned_ids = {r["id"] for r in results}
    for i in range(3, 6):
        assert f"d_{i}" in returned_ids

    # "to_delete" vectors should not appear at all
    all_results = index.query(vector=dense_vec(), top_k=10)
    all_ids = {r["id"] for r in all_results}
    for i in range(3):
        assert f"d_{i}" not in all_ids


def test_delete_with_filter_range(empty_index):
    """delete_with_filter using $range removes vectors in score range."""
    from endee.exceptions import NotFoundException

    _, index = empty_index
    batch = [
        {"id": f"r_{i}", "vector": dense_vec(seed=i), "filter": {"score": i}}
        for i in range(20)
    ]
    index.upsert(batch)
    # Delete score in [5, 10]
    index.delete_with_filter([{"score": {"$range": [5, 10]}}])

    # Scores 5-10 should be gone (get_vector raises NotFoundException)
    for i in range(5, 11):
        with pytest.raises(NotFoundException):
            index.get_vector(f"r_{i}")

    # Scores outside the range should still exist
    for i in [0, 4, 11, 19]:
        vec = index.get_vector(f"r_{i}")
        assert vec["id"] == f"r_{i}"


def test_delete_with_filter_in(empty_index):
    """delete_with_filter using $in removes vectors matching any listed value."""
    _, index = empty_index
    tags = ["alpha", "beta", "gamma"]
    batch = [
        {"id": f"in_{i}", "vector": dense_vec(seed=i),
         "filter": {"tag": tags[i % 3]}}
        for i in range(9)
    ]
    index.upsert(batch)
    # Delete alpha and beta; keep gamma
    index.delete_with_filter([{"tag": {"$in": ["alpha", "beta"]}}])

    remaining = index.query(
        vector=dense_vec(), top_k=10,
        filter=[{"tag": {"$eq": "gamma"}}],
        prefilter_cardinality_threshold=1_000_000,
    )
    assert len(remaining) == 3

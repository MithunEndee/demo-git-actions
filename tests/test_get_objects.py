"""
Tests for Collection.get_objects() - fetch full objects by ID.

Covers: return shape, meta/filter round-trips, vector presence, non-existent
IDs, mixed requests, and client-side validation.
"""

import pytest
from helpers import (
    DENSE_FIELD,
    DIM,
    MV_FIELD,
    N_VECTORS,
    SPARSE_FIELD,
    dense_vec,
    make_dense_field,
    make_mv_field,
    make_sparse_field,
    multi_vec,
    safe_delete,
    sparse_vec,
    uid,
)


# -- return structure ---------------------------------------------------------


def test_get_objects_returns_list(populated_collection):
    """get_objects must return a list."""
    _, collection = populated_collection
    result = collection.get_objects(["vec_0000"])
    assert isinstance(result, list)


def test_get_objects_single_id_returns_one_object(populated_collection):
    """get_objects for one known ID must return exactly one object."""
    _, collection = populated_collection
    result = collection.get_objects(["vec_0000"])
    assert len(result) == 1


def test_get_objects_multiple_ids_return_all(populated_collection):
    """get_objects for N known IDs must return exactly N objects."""
    _, collection = populated_collection
    ids = ["vec_0001", "vec_0002", "vec_0003"]
    result = collection.get_objects(ids)
    assert len(result) == len(ids)


def test_get_objects_result_has_required_keys(populated_collection):
    """Each object returned by get_objects must have id, meta, filter, vectors."""
    _, collection = populated_collection
    result = collection.get_objects(["vec_0005"])
    obj = result[0]
    for key in ("id", "meta", "filter", "vectors", "sparses", "multi_vectors"):
        assert key in obj, f"Missing key '{key}' in get_objects result"


def test_get_objects_id_matches_requested(populated_collection):
    """The id field in the returned object must match what was requested."""
    _, collection = populated_collection
    result = collection.get_objects(["vec_0010"])
    assert result[0]["id"] == "vec_0010"


def test_get_objects_id_is_string(populated_collection):
    """The id field must be a string."""
    _, collection = populated_collection
    result = collection.get_objects(["vec_0000"])
    assert isinstance(result[0]["id"], str)


# -- meta round-trip ----------------------------------------------------------


def test_get_objects_meta_round_trips(empty_collection):
    """Meta upserted with an object must be returned intact by get_objects."""
    _, collection = empty_collection
    payload = {"title": "round-trip doc", "count": 42, "flag": True}
    collection.upsert(
        [{"id": "rt1", "meta": payload, "fields": {DENSE_FIELD: dense_vec(seed=7)}}]
    )
    result = collection.get_objects(["rt1"])
    assert len(result) == 1
    meta = result[0]["meta"]
    assert meta["title"] == "round-trip doc"
    assert meta["count"] == 42
    assert meta["flag"] is True


def test_get_objects_meta_index_matches_upsert(populated_collection):
    """Meta 'index' field must match the object's original index value."""
    _, collection = populated_collection
    result = collection.get_objects(["vec_0007"])
    meta = result[0]["meta"]
    assert meta["index"] == 7
    assert meta["text"] == "Document 7"


# -- filter round-trip --------------------------------------------------------


def test_get_objects_filter_present(populated_collection):
    """Filter upserted with an object must be returned by get_objects."""
    _, collection = populated_collection
    result = collection.get_objects(["vec_0003"])
    flt = result[0]["filter"]
    assert isinstance(flt, dict)
    assert "category" in flt


def test_get_objects_filter_values_correct(populated_collection):
    """Filter values returned by get_objects must match what was upserted."""
    _, collection = populated_collection
    result = collection.get_objects(["vec_0006"])
    flt = result[0]["filter"]
    # vec_0006: i=6, category = A (6%3==0), tags = important (even)
    assert flt["category"] == "A"
    assert flt["tags"] == "important"
    assert flt["score"] == 6


# -- vector round-trip --------------------------------------------------------


def test_get_objects_vectors_present_for_dense_field(populated_collection):
    """get_objects must return the stored vector for a dense field."""
    _, collection = populated_collection
    result = collection.get_objects(["vec_0000"])
    vectors = result[0]["vectors"]
    assert isinstance(vectors, dict)
    assert DENSE_FIELD in vectors


def test_get_objects_vector_has_correct_dimension(populated_collection):
    """The stored dense vector must have the collection's declared dimension."""
    _, collection = populated_collection
    result = collection.get_objects(["vec_0000"])
    vec = result[0]["vectors"][DENSE_FIELD]
    assert isinstance(vec, list)
    assert len(vec) == DIM


# -- non-existent IDs ---------------------------------------------------------


def test_get_objects_nonexistent_id_returns_empty(populated_collection):
    """get_objects for an ID that does not exist must return an empty list."""
    _, collection = populated_collection
    result = collection.get_objects(["this_id_does_not_exist_xyz"])
    assert result == []


def test_get_objects_mix_existing_and_nonexistent(populated_collection):
    """get_objects with mixed IDs must return only the existing objects."""
    _, collection = populated_collection
    result = collection.get_objects(["vec_0000", "no_such_id_xyz", "vec_0001"])
    returned_ids = {obj["id"] for obj in result}
    assert "vec_0000" in returned_ids
    assert "vec_0001" in returned_ids
    assert "no_such_id_xyz" not in returned_ids


# -- after upsert round-trip ---------------------------------------------------


def test_get_objects_retrieves_upserted_object(empty_collection):
    """An object upserted and then retrieved via get_objects must be identical."""
    _, collection = empty_collection
    vec = dense_vec(seed=100)
    collection.upsert(
        [
            {
                "id": "obj_rt",
                "meta": {"msg": "hello"},
                "filter": {"cat": "X"},
                "fields": {DENSE_FIELD: vec},
            }
        ]
    )
    result = collection.get_objects(["obj_rt"])
    assert len(result) == 1
    obj = result[0]
    assert obj["id"] == "obj_rt"
    assert obj["meta"]["msg"] == "hello"
    assert obj["filter"]["cat"] == "X"


def test_get_objects_after_delete_returns_empty(populated_collection):
    """get_objects for a deleted object must return an empty list."""
    _, collection = populated_collection
    collection.delete_object("vec_0020")
    result = collection.get_objects(["vec_0020"])
    assert result == []


# -- multiple objects at once --------------------------------------------------


def test_get_objects_batch_fetch_all_present(populated_collection):
    """get_objects for all N_VECTORS IDs must return N_VECTORS objects."""
    _, collection = populated_collection
    ids = [f"vec_{i:04d}" for i in range(N_VECTORS)]
    result = collection.get_objects(ids)
    assert len(result) == N_VECTORS


def test_get_objects_batch_ids_are_unique(populated_collection):
    """Each returned id must be unique (no duplicates)."""
    _, collection = populated_collection
    ids = [f"vec_{i:04d}" for i in range(10)]
    result = collection.get_objects(ids)
    returned_ids = [obj["id"] for obj in result]
    assert len(returned_ids) == len(set(returned_ids))


# -- sparse collection --------------------------------------------------------


def test_get_objects_sparse_collection_sparses_present(populated_sparse_collection):
    """get_objects on a sparse collection must return the sparses dict."""
    _, collection = populated_sparse_collection
    result = collection.get_objects(["sp_0000"])
    assert len(result) == 1
    sparses = result[0]["sparses"]
    assert isinstance(sparses, dict)
    assert SPARSE_FIELD in sparses


def test_get_objects_sparse_has_indices_and_values(populated_sparse_collection):
    """Sparse field returned by get_objects must have indices and values lists."""
    _, collection = populated_sparse_collection
    result = collection.get_objects(["sp_0001"])
    sparse = result[0]["sparses"][SPARSE_FIELD]
    assert "indices" in sparse
    assert "values" in sparse
    assert isinstance(sparse["indices"], list)
    assert isinstance(sparse["values"], list)
    assert len(sparse["indices"]) == len(sparse["values"])


# -- multi_vector collection --------------------------------------------------


def test_get_objects_multi_vector_collection_present(populated_mv_collection):
    """get_objects on a multi_vector collection must return multi_vectors dict."""
    _, collection = populated_mv_collection
    result = collection.get_objects(["mv_0000"])
    assert len(result) == 1
    mv = result[0]["multi_vectors"]
    assert isinstance(mv, dict)
    assert MV_FIELD in mv


def test_get_objects_multi_vector_is_list_of_lists(populated_mv_collection):
    """Multi-vector data returned by get_objects must be a list of float lists."""
    _, collection = populated_mv_collection
    result = collection.get_objects(["mv_0000"])
    mv = result[0]["multi_vectors"][MV_FIELD]
    assert isinstance(mv, list)
    assert len(mv) > 0
    assert isinstance(mv[0], list)

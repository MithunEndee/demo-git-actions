"""
Tests for Collection.upsert() and Collection.delete_object().

Covers upsert counts, meta/filter preservation, overwrite semantics,
batch size limits, empty batches, Unicode meta, all precisions and
space types, binary vectors, NaN/Inf rejection, duplicate ID detection,
unknown field name rejection, and delete_object removal from search.
"""

import pytest
from helpers import (
    ALL_PRECISIONS,
    ALL_SPACE_TYPES,
    DENSE_FIELD,
    DIM,
    N_VECTORS,
    binary_vec,
    dense_vec,
    safe_delete,
    uid,
)

from endee.constants import MAX_VECTORS_PER_BATCH
from endee.schema import CollectionFieldConfig, CollectionFieldParams

# -- upsert -------------------------------------------------------------------


def test_upsert_single_object(empty_collection):
    """Upserting a single object must return upserted == 1."""
    _, collection = empty_collection
    result = collection.upsert([{"id": "v1", "fields": {DENSE_FIELD: dense_vec()}}])
    assert result["upserted"] == 1


def test_upsert_batch_10_objects(empty_collection):
    """Upserting a batch of 10 objects must return upserted == 10."""
    _, collection = empty_collection
    batch = [
        {"id": f"b_{i}", "fields": {DENSE_FIELD: dense_vec(seed=i)}} for i in range(10)
    ]
    result = collection.upsert(batch)
    assert result["upserted"] == 10


def test_upsert_batch_1000_objects(empty_collection):
    """Upserting a batch of 1000 objects must return upserted == 1000."""
    _, collection = empty_collection
    batch = [
        {"id": f"big_{i}", "fields": {DENSE_FIELD: dense_vec(seed=i)}}
        for i in range(1000)
    ]
    result = collection.upsert(batch)
    assert result["upserted"] == 1000


def test_upsert_count_returned(empty_collection):
    """upsert must return the count of upserted objects."""
    _, collection = empty_collection
    batch = [
        {"id": f"cnt_{i}", "fields": {DENSE_FIELD: dense_vec(seed=i)}} for i in range(5)
    ]
    result = collection.upsert(batch)
    assert result["upserted"] == 5


def test_upsert_with_meta_only(empty_collection):
    """Upserting an object with only meta must return upserted == 1."""
    _, collection = empty_collection
    result = collection.upsert(
        [
            {
                "id": "meta_only",
                "meta": {"title": "Hello", "value": 42},
                "fields": {DENSE_FIELD: dense_vec()},
            }
        ]
    )
    assert result["upserted"] == 1


def test_upsert_with_filter_only(empty_collection):
    """Upserting an object with only filter must return upserted == 1."""
    _, collection = empty_collection
    result = collection.upsert(
        [
            {
                "id": "filt_only",
                "filter": {"category": "X", "score": 5},
                "fields": {DENSE_FIELD: dense_vec()},
            }
        ]
    )
    assert result["upserted"] == 1


def test_upsert_with_meta_and_filter(empty_collection):
    """Upserting an object with both meta and filter must return upserted == 1."""
    _, collection = empty_collection
    result = collection.upsert(
        [
            {
                "id": "full",
                "meta": {"text": "doc"},
                "filter": {"category": "A", "score": 10, "tags": "important"},
                "fields": {DENSE_FIELD: dense_vec()},
            }
        ]
    )
    assert result["upserted"] == 1


def test_upsert_without_meta_or_filter(empty_collection):
    """Upserting a bare object with no meta or filter must return upserted == 1."""
    _, collection = empty_collection
    result = collection.upsert([{"id": "bare", "fields": {DENSE_FIELD: dense_vec()}}])
    assert result["upserted"] == 1


def test_upsert_overwrites_existing_id(empty_collection):
    """Re-inserting the same ID must return upserted == 1 (upsert semantics)."""
    _, collection = empty_collection
    v = dense_vec()
    collection.upsert([{"id": "dup", "meta": {"v": 1}, "fields": {DENSE_FIELD: v}}])
    result = collection.upsert(
        [{"id": "dup", "meta": {"v": 2}, "fields": {DENSE_FIELD: v}}]
    )
    assert result["upserted"] == 1


def test_upsert_all_precision_collections(client):
    """Upsert objects into all precision types must succeed."""
    for precision in ALL_PRECISIONS:
        name = uid("prec")
        try:
            client.create_collection(
                name=name,
                fields=[
                    CollectionFieldConfig(
                        name=DENSE_FIELD,
                        type="vector",
                        params=CollectionFieldParams(
                            dimension=DIM, space_type="cosine", precision=precision
                        ),
                    ).to_dict()
                ],
            )
            collection = client.get_collection(name)
            result = collection.upsert(
                [{"id": "v1", "fields": {DENSE_FIELD: dense_vec(seed=0)}}]
            )
            assert "upserted" in result, f"Failed for precision {precision}"
        finally:
            safe_delete(client, name)


def test_upsert_all_space_type_collections(client):
    """Upsert objects into all space types must succeed."""
    for space_type in ALL_SPACE_TYPES:
        name = uid("st")
        try:
            client.create_collection(
                name=name,
                fields=[
                    CollectionFieldConfig(
                        name=DENSE_FIELD,
                        type="vector",
                        params=CollectionFieldParams(
                            dimension=DIM, space_type=space_type, precision="int8"
                        ),
                    ).to_dict()
                ],
            )
            collection = client.get_collection(name)
            result = collection.upsert(
                [{"id": "v1", "fields": {DENSE_FIELD: dense_vec(seed=0)}}]
            )
            assert "upserted" in result, f"Failed for space_type {space_type}"
        finally:
            safe_delete(client, name)


def test_upsert_binary_precision_with_binary_vec(client):
    """Upserting 0/1 float vectors into a binary-precision collection must succeed."""
    name = uid("bin")
    try:
        client.create_collection(
            name=name,
            fields=[
                CollectionFieldConfig(
                    name=DENSE_FIELD,
                    type="vector",
                    params=CollectionFieldParams(
                        dimension=DIM, space_type="cosine", precision="binary"
                    ),
                ).to_dict()
            ],
        )
        collection = client.get_collection(name)
        batch = [
            {"id": f"bv_{i}", "fields": {DENSE_FIELD: binary_vec(seed=i)}}
            for i in range(10)
        ]
        result = collection.upsert(batch)
        assert "upserted" in result
        assert result["upserted"] == 10
    finally:
        safe_delete(client, name)


def test_upsert_empty_batch_is_accepted(empty_collection):
    """Upserting an empty batch must succeed without error."""
    _, collection = empty_collection
    result = collection.upsert([])
    assert isinstance(result, dict)


def test_upsert_empty_meta_accepted(empty_collection):
    """Upserting an object with an empty dict meta must succeed."""
    _, collection = empty_collection
    result = collection.upsert(
        [{"id": "empty_meta", "meta": {}, "fields": {DENSE_FIELD: dense_vec(seed=0)}}]
    )
    assert result.get("upserted") == 1


def test_upsert_unicode_in_meta_round_trips(empty_collection):
    """Unicode values in meta must survive an upsert -> search round-trip."""
    _, collection = empty_collection
    payload = {
        "hindi": "नमस्ते",
        "kannada": "ನಮಸ್ಕಾರ",
        "emoji": "🚀",
    }
    collection.upsert(
        [{"id": "uni", "meta": payload, "fields": {DENSE_FIELD: dense_vec(seed=1)}}]
    )
    results = collection.search(
        fields={DENSE_FIELD: {"query": dense_vec(seed=1), "limit": 1}},
    )["results"][DENSE_FIELD]
    assert results[0]["id"] == "uni"
    assert results[0]["meta"]["hindi"] == "नमस्ते"
    assert results[0]["meta"]["kannada"] == "ನಮಸ್ಕಾರ"
    assert results[0]["meta"]["emoji"] == "🚀"


def test_upsert_count_for_mixed_new_and_overwrite(empty_collection):
    """Upsert of N new + M overwrite objects must return upserted == N + M."""
    _, collection = empty_collection
    batch_a = [
        {"id": f"obj_{i}", "fields": {DENSE_FIELD: dense_vec(seed=i)}} for i in range(5)
    ]
    collection.upsert(batch_a)
    batch_b = [
        {"id": f"obj_{i}", "fields": {DENSE_FIELD: dense_vec(seed=i + 100)}}
        for i in range(5)
    ] + [
        {"id": f"new_{i}", "fields": {DENSE_FIELD: dense_vec(seed=i + 200)}}
        for i in range(5)
    ]
    result = collection.upsert(batch_b)
    assert result.get("upserted") == 10


def test_upsert_updated_meta_reflected_in_search(empty_collection):
    """After re-upserting with new meta, the updated value must appear in results."""
    _, collection = empty_collection
    vec = dense_vec(seed=77)
    collection.upsert(
        [{"id": "meta_rt", "meta": {"version": 1}, "fields": {DENSE_FIELD: vec}}]
    )
    collection.upsert(
        [{"id": "meta_rt", "meta": {"version": 2}, "fields": {DENSE_FIELD: vec}}]
    )
    results = collection.search(
        fields={DENSE_FIELD: {"query": vec, "limit": 1}},
    )["results"][DENSE_FIELD]
    assert results[0]["id"] == "meta_rt"
    assert results[0]["meta"]["version"] == 2


# -- delete_object ------------------------------------------------------------


def test_delete_object_returns_response(populated_collection):
    """delete_object must return the deleted object ID in the response."""
    _, collection = populated_collection
    result = collection.delete_object("vec_0040")
    assert result["deleted"] == "vec_0040"


def test_delete_object_removed_from_search(populated_collection):
    """Deleted object must not appear in subsequent search results."""
    _, collection = populated_collection
    target_id = "vec_0042"
    query_vec = dense_vec(seed=42)

    collection.delete_object(target_id)
    results = collection.search(
        fields={DENSE_FIELD: {"query": query_vec, "limit": N_VECTORS}},
    )["results"][DENSE_FIELD]
    returned_ids = {r["id"] for r in results}
    assert target_id not in returned_ids


# -- duplicate ID in batch ----------------------------------------------------


def test_upsert_duplicate_ids_in_same_batch_raises(empty_collection):
    """Upserting a batch with two objects sharing the same ID must raise ValueError."""
    _, collection = empty_collection
    with pytest.raises(ValueError):
        collection.upsert(
            [
                {"id": "same_id", "fields": {DENSE_FIELD: dense_vec(seed=0)}},
                {"id": "same_id", "fields": {DENSE_FIELD: dense_vec(seed=1)}},
            ]
        )


# -- NaN / Inf vectors --------------------------------------------------------


def test_upsert_nan_in_vector_raises(empty_collection):
    """Upserting a vector containing NaN must raise ValueError client-side."""
    _, collection = empty_collection
    bad_vec = [float("nan")] + [0.5] * (DIM - 1)
    with pytest.raises(ValueError):
        collection.upsert([{"id": "nan_vec", "fields": {DENSE_FIELD: bad_vec}}])


def test_upsert_inf_in_vector_raises(empty_collection):
    """Upserting a vector containing infinity must raise ValueError client-side."""
    _, collection = empty_collection
    bad_vec = [float("inf")] + [0.5] * (DIM - 1)
    with pytest.raises(ValueError):
        collection.upsert([{"id": "inf_vec", "fields": {DENSE_FIELD: bad_vec}}])


def test_upsert_neg_inf_in_vector_raises(empty_collection):
    """Upserting a vector containing -infinity must raise ValueError client-side."""
    _, collection = empty_collection
    bad_vec = [float("-inf")] + [0.5] * (DIM - 1)
    with pytest.raises(ValueError):
        collection.upsert([{"id": "ninf_vec", "fields": {DENSE_FIELD: bad_vec}}])


# -- upsert batch size limit --------------------------------------------------


def test_upsert_over_batch_limit_raises(empty_collection):
    """Upserting more than MAX_VECTORS_PER_BATCH objects must raise ValueError."""
    _, collection = empty_collection
    with pytest.raises(ValueError, match=str(MAX_VECTORS_PER_BATCH)):
        collection.upsert(
            [
                {"id": f"x_{i}", "fields": {DENSE_FIELD: dense_vec(seed=i)}}
                for i in range(MAX_VECTORS_PER_BATCH + 1)
            ]
        )


def test_upsert_objects_must_be_list_raises(empty_collection):
    """Passing a non-list to upsert must raise ValueError."""
    _, collection = empty_collection
    with pytest.raises(ValueError):
        collection.upsert({"id": "x", "fields": {DENSE_FIELD: dense_vec()}})


# -- unknown field name -------------------------------------------------------


def test_upsert_unknown_field_name_raises(empty_collection):
    """Upserting with a field name not in the collection schema must raise ValueError."""
    _, collection = empty_collection
    with pytest.raises(ValueError, match="Unknown field"):
        collection.upsert(
            [{"id": "unk", "fields": {"nonexistent_field_xyz": dense_vec()}}]
        )

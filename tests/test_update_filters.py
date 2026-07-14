"""
Tests for Collection.update_filters() - update filter tags on existing objects.

Covers return structure, count accuracy, updated values reflected in search,
new key addition, batch updates, and client-side validation.
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


def test_update_filters_returns_dict(populated_collection):
    """update_filters must return a dict."""
    _, collection = populated_collection
    result = collection.update_filters(
        [{"id": "vec_0000", "filter": {"category": "X"}}]
    )
    assert isinstance(result, dict)


def test_update_filters_returns_updated_key(populated_collection):
    """update_filters must return a dict with an 'updated' key."""
    _, collection = populated_collection
    result = collection.update_filters(
        [{"id": "vec_0001", "filter": {"category": "X"}}]
    )
    assert "updated" in result, f"Expected 'updated' key, got {list(result.keys())}"


def test_update_filters_updated_value_is_int(populated_collection):
    """The 'updated' value must be a non-negative integer."""
    _, collection = populated_collection
    result = collection.update_filters(
        [{"id": "vec_0002", "filter": {"category": "X"}}]
    )
    assert isinstance(result["updated"], int)
    assert result["updated"] >= 0


# -- count accuracy -----------------------------------------------------------


def test_update_filters_single_object_count(populated_collection):
    """Updating one object must return updated == 1."""
    _, collection = populated_collection
    result = collection.update_filters(
        [{"id": "vec_0003", "filter": {"category": "Z"}}]
    )
    assert result["updated"] == 1


def test_update_filters_multiple_objects_count(populated_collection):
    """Updating N objects must return updated == N."""
    _, collection = populated_collection
    updates = [{"id": f"vec_{i:04d}", "filter": {"category": "Z"}} for i in range(5)]
    result = collection.update_filters(updates)
    assert result["updated"] == 5


def test_update_filters_batch_count_matches(populated_collection):
    """Batch update count must equal the number of updates submitted."""
    _, collection = populated_collection
    updates = [{"id": f"vec_{i:04d}", "filter": {"batch_tag": "yes"}} for i in range(3)]
    result = collection.update_filters(updates)
    assert result["updated"] == 3


# -- negative membership: old value must not appear after update --------------


def test_update_filters_old_value_no_longer_matches(populated_collection):
    """After update, the object must not match a filter on its old value."""
    _, collection = populated_collection
    collection.update_filters([{"id": "vec_0000", "filter": {"category": "CHANGED"}}])
    results = collection.search(
        fields={DENSE_FIELD: {"query": dense_vec(), "limit": N_VECTORS}},
        filter=[{"category": {"$eq": "A"}}],
    )["results"][DENSE_FIELD]
    returned_ids = {r["id"] for r in results}
    assert "vec_0000" not in returned_ids


# -- updated value reflected in search ----------------------------------------


def test_update_filters_value_reflected_in_search(populated_collection):
    """After update_filters, the new filter value must appear in search results."""
    _, collection = populated_collection
    collection.update_filters([{"id": "vec_0004", "filter": {"category": "UPDATED"}}])
    results = collection.search(
        fields={DENSE_FIELD: {"query": dense_vec(), "limit": N_VECTORS}},
        filter=[{"category": {"$eq": "UPDATED"}}],
    )["results"][DENSE_FIELD]
    returned_ids = {r["id"] for r in results}
    assert "vec_0004" in returned_ids


def test_update_filters_new_value_searchable(populated_collection):
    """Object with updated filter must be searchable by the new value."""
    _, collection = populated_collection
    collection.update_filters([{"id": "vec_0000", "filter": {"category": "NEWCAT"}}])
    results = collection.search(
        fields={DENSE_FIELD: {"query": dense_vec(), "limit": N_VECTORS}},
        filter=[{"category": {"$eq": "NEWCAT"}}],
    )["results"][DENSE_FIELD]
    assert any(r["id"] == "vec_0000" for r in results)


def test_update_filters_can_add_new_key(populated_collection):
    """update_filters must allow adding a new filter key to an existing object."""
    _, collection = populated_collection
    collection.update_filters([{"id": "vec_0005", "filter": {"new_label": "alpha"}}])
    results = collection.search(
        fields={DENSE_FIELD: {"query": dense_vec(), "limit": N_VECTORS}},
        filter=[{"new_label": {"$eq": "alpha"}}],
    )["results"][DENSE_FIELD]
    returned_ids = {r["id"] for r in results}
    assert "vec_0005" in returned_ids


def test_update_filters_batch_updates(populated_collection):
    """A batch update must apply changes to all specified objects."""
    _, collection = populated_collection
    ids_to_update = [f"vec_{i:04d}" for i in range(10, 15)]
    updates = [{"id": oid, "filter": {"status": "processed"}} for oid in ids_to_update]
    collection.update_filters(updates)
    results = collection.search(
        fields={DENSE_FIELD: {"query": dense_vec(), "limit": N_VECTORS}},
        filter=[{"status": {"$eq": "processed"}}],
    )["results"][DENSE_FIELD]
    returned_ids = {r["id"] for r in results}
    for oid in ids_to_update:
        assert oid in returned_ids, f"{oid} not found after batch update"


def test_update_filters_numeric_value(populated_collection):
    """update_filters must handle numeric filter values."""
    _, collection = populated_collection
    collection.update_filters([{"id": "vec_0008", "filter": {"priority": 99}}])
    results = collection.search(
        fields={DENSE_FIELD: {"query": dense_vec(), "limit": N_VECTORS}},
        filter=[{"priority": {"$eq": 99}}],
    )["results"][DENSE_FIELD]
    assert any(r["id"] == "vec_0008" for r in results)


def test_update_filters_same_value_twice_is_idempotent(populated_collection):
    """Applying the same update twice must produce consistent results."""
    _, collection = populated_collection
    update = [{"id": "vec_0009", "filter": {"status": "stable"}}]
    r1 = collection.update_filters(update)
    r2 = collection.update_filters(update)
    assert r1["updated"] == 1
    assert r2["updated"] == 1
    results = collection.search(
        fields={DENSE_FIELD: {"query": dense_vec(), "limit": N_VECTORS}},
        filter=[{"status": {"$eq": "stable"}}],
    )["results"][DENSE_FIELD]
    assert sum(1 for r in results if r["id"] == "vec_0009") == 1


# -- multi_vector collection --------------------------------------------------


def test_mv_update_filters_returns_dict(populated_mv_collection):
    """update_filters() on a multi_vector collection must return a dict."""
    _, collection = populated_mv_collection
    result = collection.update_filters(
        updates=[{"id": "mv_0000", "filter": {"category": "Z"}}]
    )
    assert isinstance(result, dict)


def test_mv_update_filters_has_updated_key(populated_mv_collection):
    """update_filters() response must contain an 'updated' key."""
    _, collection = populated_mv_collection
    result = collection.update_filters(
        updates=[{"id": "mv_0000", "filter": {"category": "Z"}}]
    )
    assert "updated" in result


def test_mv_update_filters_count_correct(populated_mv_collection):
    """update_filters() on 3 IDs must return updated=3."""
    _, collection = populated_mv_collection
    result = collection.update_filters(
        updates=[
            {"id": "mv_0001", "filter": {"category": "Z"}},
            {"id": "mv_0002", "filter": {"category": "Z"}},
            {"id": "mv_0003", "filter": {"category": "Z"}},
        ]
    )
    assert result["updated"] == 3


def test_mv_update_filters_reflected_in_get_objects(populated_mv_collection):
    """Updated filter value must be returned by get_objects()."""
    _, collection = populated_mv_collection
    collection.update_filters(updates=[{"id": "mv_0010", "filter": {"category": "X"}}])
    objs = collection.get_objects(["mv_0010"])
    flt = parse_filter_field(objs[0])
    assert flt["category"] == "X"


def test_mv_update_filters_old_value_not_searchable(populated_mv_collection):
    """After updating a filter, the old value must no longer match."""
    _, collection = populated_mv_collection
    collection.update_filters(updates=[{"id": "mv_0000", "filter": {"category": "Z"}}])
    results = collection.search(
        fields={MV_FIELD: {"query": multi_vec(seed=0), "limit": N_VECTORS}},
        filter=[{"category": {"$eq": "A"}}],
    )["results"][MV_FIELD]
    assert not any(r["id"] == "mv_0000" for r in results)


def test_mv_update_filters_idempotent(populated_mv_collection):
    """Applying the same update_filters twice must not raise an error."""
    _, collection = populated_mv_collection
    collection.update_filters(updates=[{"id": "mv_0004", "filter": {"category": "W"}}])
    result = collection.update_filters(
        updates=[{"id": "mv_0004", "filter": {"category": "W"}}]
    )
    assert isinstance(result, dict)


def test_mv_update_filters_reflected_in_search(populated_mv_collection):
    """Updated filter value must be returned by search with matching filter."""
    _, collection = populated_mv_collection
    collection.update_filters(updates=[{"id": "mv_0020", "filter": {"category": "Y"}}])
    results = collection.search(
        fields={MV_FIELD: {"query": multi_vec(seed=20), "limit": N_VECTORS}},
        filter=[{"category": {"$eq": "Y"}}],
    )["results"][MV_FIELD]
    assert any(r["id"] == "mv_0020" for r in results)


# -- non-existent ID ----------------------------------------------------------


def test_update_filters_nonexistent_id_returns_zero(populated_collection):
    """update_filters on an ID that does not exist must return updated=0."""
    _, collection = populated_collection
    result = collection.update_filters(
        [{"id": "definitely_not_here_xyz_000", "filter": {"category": "Z"}}]
    )
    assert result["updated"] == 0


def test_mv_update_filters_nonexistent_id_returns_zero(populated_mv_collection):
    """update_filters on a non-existent ID in a multi_vector collection must return updated=0."""
    _, collection = populated_mv_collection
    result = collection.update_filters(
        updates=[{"id": "definitely_not_here_xyz_001", "filter": {"category": "Z"}}]
    )
    assert result["updated"] == 0

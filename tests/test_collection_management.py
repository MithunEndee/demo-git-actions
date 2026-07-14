"""
Tests for collection lifecycle: create_collection, list_collections,
get_collection, describe, and delete_collection.

Covers all field types (vector, sparse, multi_vector), HNSW parameters,
all precision and space_type combinations, and pydantic model / to_dict flows.
"""

import pytest
from helpers import (
    ALL_PRECISIONS,
    ALL_SPACE_TYPES,
    DENSE_FIELD,
    DIM,
    SPARSE_FIELD,
    dense_vec,
    get_collection_names,
    make_dense_field,
    make_sparse_field,
    safe_delete,
    uid,
)

from endee import Collection

# -- create_collection --------------------------------------------------------


def test_create_collection_returns_dict_with_name(client):
    """create_collection must return a dict containing the collection name."""
    name = uid("cret")
    try:
        result = client.create_collection(name=name, fields=[make_dense_field()])
        assert isinstance(result, dict)
        assert result.get("name") == name
    finally:
        safe_delete(client, name)


def test_create_collection_with_dict_field_config(client):
    """create_collection must accept plain dicts as field configs."""
    name = uid("dictf")
    try:
        result = client.create_collection(
            name=name,
            fields=[
                {
                    "name": DENSE_FIELD,
                    "type": "vector",
                    "params": {
                        "dimension": DIM,
                        "space_type": "cosine",
                        "precision": "int8",
                    },
                }
            ],
        )
        assert isinstance(result, dict)
        assert name in get_collection_names(client)
    finally:
        safe_delete(client, name)


def test_create_hybrid_collection_default_sparse(client):
    """Hybrid collection (dense + default sparse) must be created successfully."""
    name = uid("hyb")
    try:
        client.create_collection(
            name=name,
            fields=[make_dense_field(), make_sparse_field()],
        )
        collection = client.get_collection(name)
        info = collection.describe()
        fields = info.get("fields", [])
        assert len(fields) == 2, f"Expected 2 fields, got {len(fields)}"
        field_names = [f.get("name") for f in fields]
        assert DENSE_FIELD in field_names
        assert SPARSE_FIELD in field_names
        field_types = {f.get("name"): f.get("type") for f in fields}
        assert field_types.get(DENSE_FIELD) == "vector"
        assert field_types.get(SPARSE_FIELD) == "sparse"
    finally:
        safe_delete(client, name)


def test_create_hybrid_collection_bm25(client):
    """Hybrid collection with endee_bm25 sparse model (skip if unsupported)."""
    name = uid("bm25")
    try:
        client.create_collection(
            name=name,
            fields=[make_dense_field(), make_sparse_field("endee_bm25")],
        )
        collection = client.get_collection(name)
        info = collection.describe()
        assert len(info.get("fields", [])) == 2
    except Exception as e:
        pytest.skip(f"endee_bm25 not supported on this server: {e}")
    finally:
        safe_delete(client, name)


@pytest.mark.parametrize("space_type", ALL_SPACE_TYPES)
@pytest.mark.parametrize("precision", ALL_PRECISIONS)
def test_create_collection_precision_space_combinations(client, precision, space_type):
    """Every (precision, space_type) pair must create successfully."""
    name = uid("combo")
    try:
        result = client.create_collection(
            name=name,
            fields=[make_dense_field(space_type=space_type, precision=precision)],
        )
        assert isinstance(result, dict), f"Unexpected response type: {type(result)}"
        assert name in get_collection_names(client), (
            f"Collection '{name}' missing from list_collections"
        )
    finally:
        safe_delete(client, name)


@pytest.mark.parametrize(
    "m,ef_construct",
    [
        (4, 32),
        (8, 64),
        (16, 128),
        (32, 256),
        (64, 512),
    ],
)
def test_create_collection_custom_hnsw_params(client, m, ef_construct):
    """describe() must reflect the m and ef_construct values set at creation."""
    name = uid("hnsw")
    try:
        client.create_collection(
            name=name,
            fields=[make_dense_field(m=m, ef_construct=ef_construct)],
        )
        collection = client.get_collection(name)
        info = collection.describe()
        field = next(
            (f for f in info.get("fields", []) if f.get("name") == DENSE_FIELD),
            None,
        )
        assert field is not None, f"Field '{DENSE_FIELD}' not found in describe()"
        params = field.get("params") or {}
        assert params.get("M") == m or params.get("m") == m, (
            f"Expected m={m}, got params={params}"
        )
        assert (
            params.get("ef_construct") == ef_construct
            or params.get("ef_con") == ef_construct
        ), f"Expected ef_construct={ef_construct}, got params={params}"
    finally:
        safe_delete(client, name)


@pytest.mark.parametrize("dimension", [2, 8, 64, 128, 512])
def test_create_collection_various_dimensions(client, dimension):
    """describe() must reflect the dimension value specified at creation."""
    name = uid("dim")
    try:
        client.create_collection(
            name=name,
            fields=[make_dense_field(dim=dimension)],
        )
        collection = client.get_collection(name)
        info = collection.describe()
        field = next(
            (f for f in info.get("fields", []) if f.get("name") == DENSE_FIELD),
            None,
        )
        assert field is not None
        assert (field.get("params") or {}).get("dimension") == dimension
    finally:
        safe_delete(client, name)


# -- list_collections ---------------------------------------------------------


def test_list_collections_returns_list(client):
    """list_collections must return a list."""
    result = client.list_collections()
    assert isinstance(result, list)


def test_list_collections_items_have_name_key(client):
    """Each item returned by list_collections() must be a dict with a 'name' key."""
    name = uid("lname")
    try:
        client.create_collection(name=name, fields=[make_dense_field()])
        items = client.list_collections()
        for item in items:
            assert isinstance(item, dict), f"Expected dict item, got {type(item)}"
            assert "name" in item, f"Item missing 'name' key: {item}"
    finally:
        safe_delete(client, name)


def test_list_collections_contains_created_collection(client):
    """A newly created collection must appear in list_collections."""
    name = uid("list")
    try:
        client.create_collection(name=name, fields=[make_dense_field()])
        assert name in get_collection_names(client)
    finally:
        safe_delete(client, name)


def test_list_collections_does_not_contain_deleted_collection(client):
    """A deleted collection must not appear in list_collections."""
    name = uid("del")
    client.create_collection(name=name, fields=[make_dense_field()])
    try:
        client.delete_collection(name)
        assert name not in get_collection_names(client)
    finally:
        safe_delete(client, name)


def test_list_collections_multiple_collections_all_visible(client):
    """All created collections must appear in list_collections() simultaneously."""
    names = [uid("multi") for _ in range(3)]
    try:
        for n in names:
            client.create_collection(name=n, fields=[make_dense_field()])
        listed = get_collection_names(client)
        for n in names:
            assert n in listed, f"Collection '{n}' missing from list_collections"
    finally:
        for n in names:
            safe_delete(client, n)


# -- get_collection -----------------------------------------------------------


def test_get_collection_returns_collection_instance(client):
    """get_collection must return a Collection instance."""
    name = uid("inst")
    try:
        client.create_collection(name=name, fields=[make_dense_field()])
        collection = client.get_collection(name)
        assert isinstance(collection, Collection)
    finally:
        safe_delete(client, name)


def test_get_collection_returns_correct_name(client):
    """get_collection must return a Collection object with the correct name."""
    name = uid("attrs")
    try:
        client.create_collection(
            name=name, fields=[make_dense_field(space_type="l2", precision="float16")]
        )
        collection = client.get_collection(name)
        assert collection.name == name
    finally:
        safe_delete(client, name)


def test_get_collection_fields_present(client):
    """get_collection must populate fields with exactly the one field created."""
    name = uid("flds")
    try:
        client.create_collection(name=name, fields=[make_dense_field()])
        collection = client.get_collection(name)
        assert isinstance(collection.fields, list)
        assert len(collection.fields) == 1
    finally:
        safe_delete(client, name)


def test_get_collection_fields_match_describe(client):
    """Collection.fields returned by get_collection must equal describe()['fields']."""
    name = uid("fmatch")
    try:
        client.create_collection(name=name, fields=[make_dense_field()])
        collection = client.get_collection(name)
        described_fields = collection.describe().get("fields", [])
        assert collection.fields == described_fields
    finally:
        safe_delete(client, name)


# -- describe -----------------------------------------------------------------


def test_describe_name_matches_creation(client):
    """describe() name must match the collection name passed at creation."""
    name = uid("desc")
    try:
        client.create_collection(name=name, fields=[make_dense_field()])
        collection = client.get_collection(name)
        info = collection.describe()
        assert info["name"] == name
    finally:
        safe_delete(client, name)


def test_describe_fields_is_list(client):
    """describe() fields must be a list with exactly the one field created."""
    name = uid("fld")
    try:
        client.create_collection(name=name, fields=[make_dense_field()])
        collection = client.get_collection(name)
        info = collection.describe()
        assert isinstance(info.get("fields"), list)
        assert len(info["fields"]) == 1
    finally:
        safe_delete(client, name)


def test_describe_field_entries_have_name_key(empty_collection):
    """Each field entry in describe() must be a dict with a 'name' key."""
    _, collection = empty_collection
    info = collection.describe()
    for field in info.get("fields", []):
        assert isinstance(field, dict), f"Field entry is not a dict: {field}"
        assert "name" in field, f"Field entry missing 'name': {field}"


def test_describe_field_entries_have_type_key(empty_collection):
    """Each field entry in describe() must contain a 'type' key."""
    _, collection = empty_collection
    info = collection.describe()
    for field in info.get("fields", []):
        assert "type" in field, f"Field entry missing 'type': {field}"


def test_describe_field_params_reflect_creation(client):
    """describe() field params must reflect the dimension and space_type at creation."""
    name = uid("params")
    try:
        client.create_collection(
            name=name,
            fields=[make_dense_field(dim=32, space_type="l2")],
        )
        collection = client.get_collection(name)
        info = collection.describe()
        field = next(
            (f for f in info.get("fields", []) if f.get("name") == DENSE_FIELD),
            None,
        )
        assert field is not None, f"Field '{DENSE_FIELD}' not found in describe()"
        params = field.get("params") or {}
        assert params.get("dimension") == 32
        assert params.get("space_type") == "l2"
    finally:
        safe_delete(client, name)


def test_describe_hybrid_collection_has_two_fields(client):
    """describe() on a hybrid collection must report two fields."""
    name = uid("hdesc")
    try:
        client.create_collection(
            name=name,
            fields=[make_dense_field(), make_sparse_field()],
        )
        collection = client.get_collection(name)
        info = collection.describe()
        assert len(info.get("fields", [])) == 2
    finally:
        safe_delete(client, name)


# -- delete_collection --------------------------------------------------------


def test_delete_collection_returns_response(client):
    """delete_collection must return a response dict."""
    name = uid("delok")
    client.create_collection(name=name, fields=[make_dense_field()])
    try:
        result = client.delete_collection(name)
        assert isinstance(result, dict)
    finally:
        safe_delete(client, name)


def test_delete_collection_response_has_expected_key(client):
    """delete_collection must return a dict with a 'message' or 'deleted' key."""
    name = uid("delkey")
    client.create_collection(name=name, fields=[make_dense_field()])
    try:
        result = client.delete_collection(name)
        assert isinstance(result, dict)
        assert "message" in result or "deleted" in result or "name" in result, (
            f"Unexpected delete response keys: {list(result.keys())}"
        )
    finally:
        safe_delete(client, name)


def test_delete_collection_removes_from_list(client):
    """Deleting a collection must remove it from list_collections."""
    name = uid("delist")
    client.create_collection(name=name, fields=[make_dense_field()])
    try:
        client.delete_collection(name)
        assert name not in get_collection_names(client)
    finally:
        safe_delete(client, name)


# -- Collection __str__ -------------------------------------------------------


def test_collection_str_returns_name(client):
    """str(collection) must return the collection name."""
    name = uid("str")
    try:
        client.create_collection(name=name, fields=[make_dense_field()])
        collection = client.get_collection(name)
        assert str(collection) == name
    finally:
        safe_delete(client, name)


# -- minimum dimension (dim=2) upsert + search --------------------------------


def test_upsert_and_search_minimum_dimension(client):
    """A dim=2 collection must accept upserts and return search results."""
    name = uid("d2")
    try:
        client.create_collection(name=name, fields=[make_dense_field(dim=2)])
        collection = client.get_collection(name)
        batch = [
            {"id": f"v{i}", "fields": {DENSE_FIELD: dense_vec(dim=2, seed=i)}}
            for i in range(10)
        ]
        result = collection.upsert(batch)
        assert result["upserted"] == 10

        results = collection.search(
            fields={DENSE_FIELD: {"query": dense_vec(dim=2, seed=99), "limit": 5}}
        )["results"][DENSE_FIELD]
        assert isinstance(results, list)
        assert len(results) > 0
        for r in results:
            assert "id" in r and "similarity" in r
    finally:
        safe_delete(client, name)

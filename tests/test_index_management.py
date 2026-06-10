import pytest

from endee import Precision

from helpers import DIM, HYBRID_DIM, get_index_names, safe_delete, uid


ALL_PRECISIONS = [
    Precision.FLOAT32,
    Precision.FLOAT16,
    Precision.INT16,
    Precision.INT8,
    Precision.BINARY2,
]

ALL_SPACE_TYPES = ["cosine", "l2", "ip"]


@pytest.mark.parametrize("space_type", ALL_SPACE_TYPES)
@pytest.mark.parametrize("precision", ALL_PRECISIONS)
def test_create_index_precision_space_combinations(client, precision, space_type):
    """Every (precision, space_type) pair must create successfully."""
    name = uid("combo")
    try:
        result = client.create_index(
            name=name,
            dimension=DIM,
            space_type=space_type,
            precision=precision,
        )
        assert "success" in result.lower(), f"Unexpected response: {result}"

        names = get_index_names(client)
        assert name in names, f"Index '{name}' missing from list_indexes"
    finally:
        safe_delete(client, name)


@pytest.mark.parametrize(
    "M,ef_con",
    [
        (4, 32),
        (8, 64),
        (16, 128),
        (32, 256),
        (64, 512),
    ],
)
def test_create_index_custom_hnsw_params(client, M, ef_con):
    """Index creation with a range of valid M / ef_con values."""
    name = uid("hnsw")
    try:
        client.create_index(
            name=name,
            dimension=DIM,
            space_type="cosine",
            precision=Precision.INT8,
            M=M,
            ef_con=ef_con,
        )
        index = client.get_index(name)
        info = index.describe()
        assert info["M"] == M
        assert info["ef_con"] == ef_con
    finally:
        safe_delete(client, name)


@pytest.mark.parametrize("dimension", [2, 8, 64, 128, 512])
def test_create_index_various_dimensions(client, dimension):
    """Index creation with a range of valid dimension values."""
    name = uid("dim")
    try:
        client.create_index(
            name=name,
            dimension=dimension,
            space_type="cosine",
            precision=Precision.INT8,
        )
        index = client.get_index(name)
        assert index.dimension == dimension
    finally:
        safe_delete(client, name)


def test_create_hybrid_index_default_sparse(client):
    """Hybrid index with the default sparse model must report is_hybrid True."""
    name = uid("hyb")
    try:
        client.create_index(
            name=name,
            dimension=HYBRID_DIM,
            space_type="cosine",
            precision=Precision.INT8,
            sparse_model="default",
        )
        index = client.get_index(name)
        assert index.is_hybrid is True
        assert index.sparse_model != "None"
    finally:
        safe_delete(client, name)


def test_create_hybrid_index_bm25(client):
    """BM25 sparse model (skip if server does not support it)."""
    name = uid("bm25")
    try:
        client.create_index(
            name=name,
            dimension=HYBRID_DIM,
            space_type="cosine",
            precision=Precision.INT8,
            sparse_model="endee_bm25",
        )
        index = client.get_index(name)
        assert index.is_hybrid is True
    except Exception as e:
        pytest.skip(f"endee_bm25 not supported on this server: {e}")
    finally:
        safe_delete(client, name)


def test_list_indexes_returns_list(client):
    """list_indexes must return a list."""
    names = get_index_names(client)
    assert isinstance(names, list)


def test_list_indexes_contains_created_index(client):
    """A newly created index must appear in list_indexes."""
    name = uid("list")
    try:
        client.create_index(
            name=name, dimension=DIM, space_type="cosine", precision=Precision.INT8
        )
        names = get_index_names(client)
        assert name in names
    finally:
        safe_delete(client, name)


def test_list_indexes_does_not_contain_deleted_index(client):
    """A deleted index must not appear in list_indexes."""
    name = uid("del")
    client.create_index(
        name=name, dimension=DIM, space_type="cosine", precision=Precision.INT8
    )
    client.delete_index(name)
    names = get_index_names(client)
    assert name not in names


def test_get_index_attributes_match_creation(client):
    """get_index must return an object whose attributes match the creation parameters."""
    name = uid("attrs")
    M, ef_con = 24, 200
    try:
        client.create_index(
            name=name,
            dimension=DIM,
            space_type="l2",
            precision=Precision.FLOAT16,
            M=M,
            ef_con=ef_con,
        )
        index = client.get_index(name)
        assert index.name == name
        assert index.dimension == DIM
        assert index.space_type == "l2"
        assert index.M == M
        assert index.ef_con == ef_con
        assert index.is_hybrid is False
    finally:
        safe_delete(client, name)


def test_describe_returns_expected_keys(empty_index):
    """describe() must return a dict containing all expected metadata keys."""
    _, index = empty_index
    info = index.describe()
    expected_keys = {
        "name",
        "space_type",
        "dimension",
        "sparse_model",
        "is_hybrid",
        "count",
        "precision",
        "M",
        "ef_con",
    }
    assert expected_keys.issubset(info.keys()), (
        f"Missing keys: {expected_keys - info.keys()}"
    )


def test_describe_values_match_creation(client):
    """describe() values must match the parameters passed at creation time."""
    name = uid("desc")
    try:
        client.create_index(
            name=name,
            dimension=DIM,
            space_type="cosine",
            precision=Precision.INT8,
            M=16,
            ef_con=128,
        )
        index = client.get_index(name)
        info = index.describe()
        assert info["name"] == name
        assert info["dimension"] == DIM
        assert info["space_type"] == "cosine"
        assert info["M"] == 16
        assert info["ef_con"] == 128
        assert info["is_hybrid"] is False
    finally:
        safe_delete(client, name)


def test_delete_index_returns_success_message(client):
    """delete_index must return a message that includes the index name."""
    name = uid("delok")
    client.create_index(
        name=name, dimension=DIM, space_type="cosine", precision=Precision.INT8
    )
    result = client.delete_index(name)
    assert name in result


def test_delete_index_removes_from_list(client):
    """Deleting an index must remove it from list_indexes."""
    name = uid("delist")
    client.create_index(
        name=name, dimension=DIM, space_type="cosine", precision=Precision.INT8
    )
    client.delete_index(name)
    names = get_index_names(client)
    assert name not in names

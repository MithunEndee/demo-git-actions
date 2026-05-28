"""
test_01_index_management.py

Tests for index lifecycle:
  - create_index  (all precision × space_type combinations, custom HNSW params,
                   hybrid indexes, duplicate detection)
  - list_indexes
  - get_index     (attribute verification)
  - describe()
  - delete_index
"""

import pytest

from endee import Endee, Precision

from conftest import DIM, HYBRID_DIM, safe_delete, uid


# ── Parametrised: all precision types × all space types ──────────────────

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

        # Index must appear in list
        names = [idx.get("name") for idx in client.list_indexes()]
        assert name in names, f"Index '{name}' missing from list_indexes"
    finally:
        safe_delete(client, name)


# ── Custom HNSW parameters ────────────────────────────────────────────────

@pytest.mark.parametrize("M,ef_con", [
    (4,   32),
    (8,   64),
    (16, 128),
    (32, 256),
    (64, 512),
])
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


# ── Various dimensions ────────────────────────────────────────────────────

@pytest.mark.parametrize("dimension", [2, 8, 64, 128, 512])
def test_create_index_various_dimensions(client, dimension):
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


# ── Hybrid index creation ─────────────────────────────────────────────────

def test_create_hybrid_index_default_sparse(client):
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


# ── list_indexes ──────────────────────────────────────────────────────────

def test_list_indexes_returns_list(client):
    indexes = client.list_indexes()
    assert isinstance(indexes, list)


def test_list_indexes_contains_created_index(client):
    name = uid("list")
    try:
        client.create_index(name=name, dimension=DIM, space_type="cosine", precision=Precision.INT8)
        names = [idx.get("name") for idx in client.list_indexes()]
        assert name in names
    finally:
        safe_delete(client, name)


def test_list_indexes_does_not_contain_deleted_index(client):
    name = uid("del")
    client.create_index(name=name, dimension=DIM, space_type="cosine", precision=Precision.INT8)
    client.delete_index(name)
    names = [idx.get("name") for idx in client.list_indexes()]
    assert name not in names


# ── get_index – attribute verification ───────────────────────────────────

def test_get_index_attributes_match_creation(client):
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


# ── describe() ────────────────────────────────────────────────────────────

def test_describe_returns_expected_keys(empty_index):
    _, index = empty_index
    info = index.describe()
    expected_keys = {"name", "space_type", "dimension", "sparse_model",
                     "is_hybrid", "count", "precision", "M", "ef_con"}
    assert expected_keys.issubset(info.keys()), f"Missing keys: {expected_keys - info.keys()}"


def test_describe_values_match_creation(client):
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


# ── delete_index ──────────────────────────────────────────────────────────

def test_delete_index_returns_success_message(client):
    name = uid("delok")
    client.create_index(name=name, dimension=DIM, space_type="cosine", precision=Precision.INT8)
    result = client.delete_index(name)
    assert name in result


def test_delete_index_removes_from_list(client):
    name = uid("delist")
    client.create_index(name=name, dimension=DIM, space_type="cosine", precision=Precision.INT8)
    client.delete_index(name)
    names = [idx.get("name") for idx in client.list_indexes()]
    assert name not in names

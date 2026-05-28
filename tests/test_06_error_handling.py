"""
test_06_error_handling.py

Tests for error handling and validation:
  - Client-side validation errors (ValueError from Pydantic)
  - Server-side errors (ConflictException, NotFoundException, APIException)
  - Constraint violations (batch size, dimension mismatch, sparse/dense mismatch)
  - Filter key/value size limits
"""

import pytest

from endee import Endee, Precision
from endee.exceptions import (
    APIException,
    ConflictException,
    EndeeException,
    NotFoundException,
)

from helpers import DIM, HYBRID_DIM, dense_vec, safe_delete, sparse_vec, uid


# === Index name validation (client-side, ValueError) ===

@pytest.mark.parametrize("bad_name", [
    "has space",
    "has-hyphen",
    "has.dot",
    "has@at",
    "",
    "a" * 49,              # 49 chars > 48 max
])
def test_create_index_invalid_name_raises(client, bad_name):
    """create_index with a name that violates naming rules must raise ValueError."""
    with pytest.raises(ValueError):
        client.create_index(
            name=bad_name,
            dimension=DIM,
            space_type="cosine",
            precision=Precision.INT8,
        )


def test_create_index_name_max_length_ok(client):
    """Exactly 48 alphanumeric characters is valid."""
    name = "a" * 48
    try:
        client.create_index(name=name, dimension=DIM, space_type="cosine",
                            precision=Precision.INT8)
    finally:
        safe_delete(client, name)


# === space_type validation ===

@pytest.mark.parametrize("bad_type", ["euclidean", "dot", ""])
def test_create_index_invalid_space_type_raises(client, bad_type):
    """create_index with an unsupported space_type must raise ValueError."""
    with pytest.raises(ValueError):
        client.create_index(
            name=uid("inv"),
            dimension=DIM,
            space_type=bad_type,
            precision=Precision.INT8,
        )


# === precision validation ===

@pytest.mark.parametrize("bad_prec", ["fp32", "int4", "uint8", "half", ""])
def test_create_index_invalid_precision_raises(client, bad_prec):
    """create_index with an unrecognised precision string must raise ValueError."""
    with pytest.raises(ValueError):
        client.create_index(
            name=uid("invp"),
            dimension=DIM,
            space_type="cosine",
            precision=bad_prec,
        )


# === dimension bounds ===

def test_create_index_dim_1_raises(client):
    """create_index with dimension=1 (below minimum) must raise ValueError."""
    with pytest.raises(ValueError):
        client.create_index(name=uid("d1"), dimension=1, space_type="cosine",
                            precision=Precision.INT8)


def test_create_index_dim_8001_raises(client):
    """create_index with dimension=8001 (above maximum) must raise ValueError."""
    with pytest.raises(ValueError):
        client.create_index(name=uid("d2"), dimension=8001, space_type="cosine",
                            precision=Precision.INT8)


def test_create_index_dim_2_is_valid(client):
    """create_index with dimension=2 (minimum valid value) must succeed."""
    name = uid("d2ok")
    try:
        client.create_index(name=name, dimension=2, space_type="cosine",
                            precision=Precision.INT8)
    finally:
        safe_delete(client, name)


def test_create_index_dim_8000_is_valid(client):
    """create_index with dimension=8000 (maximum valid value) must succeed."""
    name = uid("d8k")
    try:
        client.create_index(name=name, dimension=8000, space_type="cosine",
                            precision=Precision.INT8)
    finally:
        safe_delete(client, name)


# === Duplicate index (server-side ConflictException) ===

def test_create_duplicate_index_raises_conflict(client):
    """Creating an index with an already-existing name must raise ConflictException."""
    name = uid("dup")
    client.create_index(name=name, dimension=DIM, space_type="cosine",
                        precision=Precision.INT8)
    try:
        with pytest.raises(ConflictException):
            client.create_index(name=name, dimension=DIM, space_type="cosine",
                                precision=Precision.INT8)
    finally:
        safe_delete(client, name)


# === Non-existent index (server-side NotFoundException) ===

def test_get_nonexistent_index_raises_not_found(client):
    """get_index for an index that does not exist must raise NotFoundException."""
    with pytest.raises(NotFoundException):
        client.get_index("this_index_absolutely_does_not_exist_xyz123")


def test_delete_nonexistent_index_raises_not_found(client):
    """delete_index for an index that does not exist must raise NotFoundException."""
    with pytest.raises(NotFoundException):
        client.delete_index("nonexistent_xyz789")


# === upsert: dimension mismatch ===

def test_upsert_wrong_dimension_raises(empty_index):
    """Upserting a vector with more dimensions than the index must raise an error."""
    _, index = empty_index
    wrong_dim_vec = dense_vec(dim=DIM + 1)  # one extra dimension
    with pytest.raises((ValueError, EndeeException)):
        index.upsert([{"id": "bad", "vector": wrong_dim_vec}])


def test_upsert_too_few_dimensions_raises(empty_index):
    """Upserting a vector with fewer dimensions than the index must raise an error."""
    _, index = empty_index
    with pytest.raises((ValueError, EndeeException)):
        index.upsert([{"id": "short", "vector": dense_vec(dim=DIM - 1)}])


# === upsert: duplicate IDs in single batch ===

def test_upsert_duplicate_ids_in_batch_raises(empty_index):
    """Upserting a batch containing duplicate IDs must raise ValueError."""
    _, index = empty_index
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        index.upsert([
            {"id": "same_id", "vector": dense_vec()},
            {"id": "same_id", "vector": dense_vec()},
        ])


# === upsert: batch size limit ===

def test_upsert_over_1000_raises(empty_index):
    """Upserting a batch of more than 1000 vectors must raise ValueError."""
    _, index = empty_index
    big_batch = [{"id": f"x{i}", "vector": dense_vec()} for i in range(1001)]
    with pytest.raises(ValueError, match="1000"):
        index.upsert(big_batch)


def test_upsert_exactly_1000_is_ok(empty_index):
    """Upserting exactly 1000 vectors (the batch limit) must succeed."""
    _, index = empty_index
    batch = [{"id": f"b{i:04d}", "vector": dense_vec(seed=i)} for i in range(1000)]
    result = index.upsert(batch)
    assert "success" in result.lower()


# === upsert: sparse data on dense-only index ===

def test_upsert_sparse_on_dense_index_raises(empty_index):
    """Providing sparse data when upserting into a dense-only index must raise ValueError."""
    _, index = empty_index
    si, sv = sparse_vec()
    with pytest.raises(ValueError):
        index.upsert([{
            "id": "s1",
            "vector": dense_vec(),
            "sparse_indices": si,
            "sparse_values": sv,
        }])


# === upsert: dense without sparse on hybrid index ===

def test_upsert_dense_without_sparse_on_hybrid_raises(empty_hybrid_index):
    """Upserting only a dense vector into a hybrid index (no sparse data) must raise ValueError."""
    _, index = empty_hybrid_index
    with pytest.raises(ValueError):
        index.upsert([{"id": "d1", "vector": dense_vec(HYBRID_DIM)}])


# === upsert: sparse_indices / sparse_values length mismatch ===

def test_upsert_sparse_length_mismatch_raises(empty_hybrid_index):
    """Upserting sparse_indices and sparse_values with different lengths must raise ValueError."""
    _, index = empty_hybrid_index
    si, sv = sparse_vec()
    with pytest.raises(ValueError, match="[Ll]ength|[Mm]atch"):
        index.upsert([{
            "id": "mis",
            "vector": dense_vec(HYBRID_DIM),
            "sparse_indices": si,
            "sparse_values": sv[:-1],  # one fewer value
        }])


# === query: wrong dimension ===

def test_query_wrong_dimension_raises(populated_index):
    """Querying with a vector of the wrong dimension must raise an error."""
    _, index = populated_index
    with pytest.raises((ValueError, EndeeException)):
        index.query(vector=dense_vec(dim=DIM + 2), top_k=5)


# === query: sparse on dense-only index ===

def test_query_sparse_on_dense_index_raises(populated_index):
    """Providing sparse query inputs to a dense-only index must raise ValueError."""
    _, index = populated_index
    si, sv = sparse_vec()
    with pytest.raises(ValueError):
        index.query(sparse_indices=si, sparse_values=sv, top_k=5)


# === query: no vector and no sparse provided ===

def test_query_no_vector_no_sparse_raises(populated_index):
    """Calling query with neither a dense vector nor sparse inputs must raise ValueError."""
    _, index = populated_index
    with pytest.raises(ValueError):
        index.query(top_k=5)


# === query: top_k bounds ===

def test_query_top_k_0_raises(populated_index):
    """query with top_k=0 must raise ValueError."""
    _, index = populated_index
    with pytest.raises(ValueError):
        index.query(vector=dense_vec(), top_k=0)


def test_query_top_k_over_4096_raises(populated_index):
    """query with top_k above the 4096 maximum must raise ValueError."""
    _, index = populated_index
    with pytest.raises(ValueError):
        index.query(vector=dense_vec(), top_k=4097)


# === query: ef bounds ===

def test_query_ef_over_1024_raises(populated_index):
    """query with ef above the 1024 maximum must raise ValueError."""
    _, index = populated_index
    with pytest.raises(ValueError):
        index.query(vector=dense_vec(), top_k=5, ef=1025)


# === query: filter_boost_percentage bounds ===

def test_query_filter_boost_over_400_raises(populated_index):
    """query with filter_boost_percentage above 400 must raise ValueError."""
    _, index = populated_index
    with pytest.raises(ValueError):
        index.query(vector=dense_vec(), top_k=5, filter_boost_percentage=401)


def test_query_filter_boost_negative_raises(populated_index):
    """query with a negative filter_boost_percentage must raise ValueError."""
    _, index = populated_index
    with pytest.raises(ValueError):
        index.query(vector=dense_vec(), top_k=5, filter_boost_percentage=-1)


# === query: prefilter_cardinality_threshold bounds ===

def test_query_prefilter_below_1000_raises(populated_index):
    """query with prefilter_cardinality_threshold below 1000 must raise ValueError."""
    _, index = populated_index
    with pytest.raises(ValueError):
        index.query(vector=dense_vec(), top_k=5, prefilter_cardinality_threshold=999)


def test_query_prefilter_over_1000000_raises(populated_index):
    """query with prefilter_cardinality_threshold above 1000000 must raise ValueError."""
    _, index = populated_index
    with pytest.raises(ValueError):
        index.query(vector=dense_vec(), top_k=5, prefilter_cardinality_threshold=1_000_001)


# === query: dense_rrf_weight bounds ===

def test_query_rrf_weight_negative_raises(populated_index):
    """query with a negative dense_rrf_weight must raise ValueError."""
    _, index = populated_index
    with pytest.raises(ValueError):
        index.query(vector=dense_vec(), top_k=5, dense_rrf_weight=-0.1)


def test_query_rrf_weight_over_1_raises(populated_index):
    """query with dense_rrf_weight above 1.0 must raise ValueError."""
    _, index = populated_index
    with pytest.raises(ValueError):
        index.query(vector=dense_vec(), top_k=5, dense_rrf_weight=1.1)


# === Filter key / value size limits ===

def test_upsert_filter_key_too_long_raises(empty_index):
    """Upserting a vector whose filter contains a key exceeding 128 bytes must raise ValueError."""
    _, index = empty_index
    long_key = "k" * 129  # > 128 bytes
    with pytest.raises(ValueError, match="[Kk]ey"):
        index.upsert([{
            "id": "fk",
            "vector": dense_vec(),
            "filter": {long_key: "value"},
        }])


def test_upsert_filter_value_too_long_raises(empty_index):
    """Upserting a vector whose filter contains a value exceeding 1024 bytes must raise ValueError."""
    _, index = empty_index
    long_val = "v" * 1025  # > 1024 bytes
    with pytest.raises(ValueError, match="[Vv]alue"):
        index.upsert([{
            "id": "fv",
            "vector": dense_vec(),
            "filter": {"key": long_val},
        }])


# === sparse_model validation ===

def test_create_index_invalid_sparse_model_raises(client):
    """create_index with an unrecognised sparse_model value must raise ValueError."""
    with pytest.raises(ValueError):
        client.create_index(
            name=uid("sm"),
            dimension=DIM,
            space_type="cosine",
            precision=Precision.INT8,
            sparse_model="invalid_model_xyz",
        )


# === HTTP library: unsupported library raises at init ===

def test_unsupported_http_library_raises():
    """Instantiating Endee with an unsupported http_library name must raise ValueError."""
    with pytest.raises(ValueError, match="[Uu]nsupported"):
        Endee(http_library="curl")


# === VectorItem: empty ID raises ===

def test_upsert_empty_id_raises(empty_index):
    """Upserting a vector with an empty string ID must raise ValueError."""
    _, index = empty_index
    with pytest.raises(ValueError):
        index.upsert([{"id": "", "vector": dense_vec()}])

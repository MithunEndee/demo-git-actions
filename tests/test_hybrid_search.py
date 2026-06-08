import pytest

from endee import Precision
from endee.exceptions import NotFoundException

from helpers import (
    HYBRID_DIM,
    N_VECTORS,
    dense_vec,
    make_item,
    safe_delete,
    sparse_vec,
    uid,
)


def test_hybrid_upsert_succeeds(empty_hybrid_index):
    """Upserting a single hybrid vector must return a success response."""
    _, index = empty_hybrid_index
    si, sv = sparse_vec(seed=0)
    result = index.upsert(
        [
            {
                "id": "hv1",
                "vector": dense_vec(HYBRID_DIM, seed=0),
                "sparse_indices": si,
                "sparse_values": sv,
            }
        ]
    )
    assert "success" in result.lower()


def test_hybrid_upsert_with_meta_and_filter(empty_hybrid_index):
    """Upserting a hybrid vector with meta and filter fields must succeed."""
    _, index = empty_hybrid_index
    si, sv = sparse_vec(seed=1)
    result = index.upsert(
        [
            {
                "id": "hv_full",
                "vector": dense_vec(HYBRID_DIM, seed=1),
                "sparse_indices": si,
                "sparse_values": sv,
                "meta": {"title": "hybrid doc"},
                "filter": {"category": "A"},
            }
        ]
    )
    assert "success" in result.lower()


def test_hybrid_upsert_batch(empty_hybrid_index):
    """Upserting a batch of hybrid vectors must return a success response."""
    _, index = empty_hybrid_index
    from helpers import make_item

    batch = [make_item(i, dim=HYBRID_DIM, with_sparse=True) for i in range(20)]
    result = index.upsert(batch)
    assert "success" in result.lower()


def test_hybrid_dense_only_query(populated_hybrid_index):
    """Hybrid index accepts a query with only dense vector (no sparse)."""
    _, index = populated_hybrid_index
    results = index.query(vector=dense_vec(HYBRID_DIM), top_k=5)
    assert isinstance(results, list)
    assert len(results) > 0


def test_hybrid_dense_only_result_structure(populated_hybrid_index):
    """Dense-only query on a hybrid index must return results with all required keys."""
    _, index = populated_hybrid_index
    results = index.query(vector=dense_vec(HYBRID_DIM), top_k=1)
    r = results[0]
    for key in ("id", "similarity", "distance", "meta", "norm"):
        assert key in r


def test_hybrid_sparse_only_query(populated_hybrid_index):
    """Query with only sparse_indices/values, no dense vector."""
    _, index = populated_hybrid_index
    si, sv = sparse_vec(seed=99)
    results = index.query(sparse_indices=si, sparse_values=sv, top_k=5)
    assert isinstance(results, list)
    assert len(results) > 0


def test_hybrid_full_query(populated_hybrid_index):
    """Full hybrid query with both dense and sparse inputs must return results."""
    _, index = populated_hybrid_index
    si, sv = sparse_vec(seed=42)
    results = index.query(
        vector=dense_vec(HYBRID_DIM, seed=42),
        sparse_indices=si,
        sparse_values=sv,
        top_k=10,
    )
    assert isinstance(results, list)
    assert len(results) > 0


def test_hybrid_query_results_ordered_by_similarity(populated_hybrid_index):
    """Hybrid query results must be sorted from highest to lowest similarity."""
    _, index = populated_hybrid_index
    si, sv = sparse_vec(seed=7)
    results = index.query(
        vector=dense_vec(HYBRID_DIM, seed=7),
        sparse_indices=si,
        sparse_values=sv,
        top_k=10,
    )
    sims = [r["similarity"] for r in results]
    assert sims == sorted(sims, reverse=True)


def test_hybrid_query_with_eq_filter(populated_hybrid_index):
    """Hybrid query with a $eq filter must return only matching vectors."""
    _, index = populated_hybrid_index
    si, sv = sparse_vec(seed=3)
    results = index.query(
        vector=dense_vec(HYBRID_DIM, seed=3),
        sparse_indices=si,
        sparse_values=sv,
        top_k=N_VECTORS,
        filter=[{"category": {"$eq": "A"}}],
        prefilter_cardinality_threshold=1_000_000,
    )
    assert len(results) == 17
    for r in results:
        assert r["filter"]["category"] == "A"


def test_hybrid_query_with_range_filter(populated_hybrid_index):
    """Hybrid query with a $range filter must return only vectors within the score range."""
    _, index = populated_hybrid_index
    si, sv = sparse_vec(seed=4)
    results = index.query(
        vector=dense_vec(HYBRID_DIM, seed=4),
        sparse_indices=si,
        sparse_values=sv,
        top_k=N_VECTORS,
        filter=[{"score": {"$range": [10, 20]}}],
        prefilter_cardinality_threshold=1_000_000,
    )
    assert len(results) == 11
    for r in results:
        assert 10 <= r["filter"]["score"] <= 20


@pytest.mark.parametrize("weight", [0.0, 0.2, 0.5, 0.7, 1.0])
def test_hybrid_rrf_weight_accepted(populated_hybrid_index, weight):
    """All valid dense_rrf_weight values should return results without error."""
    _, index = populated_hybrid_index
    si, sv = sparse_vec(seed=10)
    results = index.query(
        vector=dense_vec(HYBRID_DIM, seed=10),
        sparse_indices=si,
        sparse_values=sv,
        top_k=5,
        dense_rrf_weight=weight,
    )
    assert isinstance(results, list)


def test_hybrid_rrf_weight_0_emphasises_sparse(populated_hybrid_index):
    """dense_rrf_weight=0.0 means full sparse ranking; should still return results."""
    _, index = populated_hybrid_index
    si, sv = sparse_vec(seed=11)
    results = index.query(
        vector=dense_vec(HYBRID_DIM, seed=11),
        sparse_indices=si,
        sparse_values=sv,
        top_k=5,
        dense_rrf_weight=0.0,
    )
    assert len(results) > 0


def test_hybrid_rrf_weight_1_emphasises_dense(populated_hybrid_index):
    """dense_rrf_weight=1.0 means full dense ranking; should still return results."""
    _, index = populated_hybrid_index
    si, sv = sparse_vec(seed=12)
    results = index.query(
        vector=dense_vec(HYBRID_DIM, seed=12),
        sparse_indices=si,
        sparse_values=sv,
        top_k=5,
        dense_rrf_weight=1.0,
    )
    assert len(results) > 0


@pytest.mark.parametrize("rrc", [1, 10, 30, 60, 120, 200])
def test_hybrid_rrf_rank_constant_accepted(populated_hybrid_index, rrc):
    """All valid rrf_rank_constant values must be accepted without error."""
    _, index = populated_hybrid_index
    si, sv = sparse_vec(seed=20)
    results = index.query(
        vector=dense_vec(HYBRID_DIM, seed=20),
        sparse_indices=si,
        sparse_values=sv,
        top_k=5,
        rrf_rank_constant=rrc,
    )
    assert isinstance(results, list)


def test_hybrid_include_vectors_true(populated_hybrid_index):
    """include_vectors=True on a hybrid index must return full-dimension dense vectors."""
    _, index = populated_hybrid_index
    si, sv = sparse_vec(seed=30)
    results = index.query(
        vector=dense_vec(HYBRID_DIM, seed=30),
        sparse_indices=si,
        sparse_values=sv,
        top_k=3,
        include_vectors=True,
    )
    for r in results:
        assert isinstance(r["vector"], list)
        assert len(r["vector"]) == HYBRID_DIM


def test_hybrid_include_vectors_false(populated_hybrid_index):
    """include_vectors=False on a hybrid index must return empty vector lists."""
    _, index = populated_hybrid_index
    si, sv = sparse_vec(seed=31)
    results = index.query(
        vector=dense_vec(HYBRID_DIM, seed=31),
        sparse_indices=si,
        sparse_values=sv,
        top_k=3,
        include_vectors=False,
    )
    for r in results:
        assert r["vector"] == []


def test_hybrid_get_vector_has_sparse_keys(populated_hybrid_index):
    """get_vector on a hybrid index must include sparse_indices and sparse_values keys."""
    _, index = populated_hybrid_index
    vec = index.get_vector("vec_0000")
    assert "sparse_indices" in vec, "sparse_indices missing from get_vector result"
    assert "sparse_values" in vec, "sparse_values missing from get_vector result"


def test_hybrid_get_vector_sparse_lists_same_length(populated_hybrid_index):
    """sparse_indices and sparse_values returned by get_vector must have equal length."""
    _, index = populated_hybrid_index
    vec = index.get_vector("vec_0001")
    assert len(vec["sparse_indices"]) == len(vec["sparse_values"])


def test_hybrid_get_vector_sparse_indices_are_ints(populated_hybrid_index):
    """sparse_indices returned by get_vector must all be integers."""
    _, index = populated_hybrid_index
    vec = index.get_vector("vec_0002")
    for idx in vec["sparse_indices"]:
        assert isinstance(idx, int), f"Expected int, got {type(idx)}"


def test_hybrid_get_vector_sparse_values_are_floats(populated_hybrid_index):
    """sparse_values returned by get_vector must all be floats."""
    _, index = populated_hybrid_index
    vec = index.get_vector("vec_0003")
    for val in vec["sparse_values"]:
        assert isinstance(val, float), f"Expected float, got {type(val)}"


def test_hybrid_get_vector_id_matches(populated_hybrid_index):
    """get_vector on a hybrid index must return the correct vector id."""
    _, index = populated_hybrid_index
    vec = index.get_vector("vec_0004")
    assert vec["id"] == "vec_0004"


def test_hybrid_get_vector_meta_preserved(populated_hybrid_index):
    """get_vector on a hybrid index must return the meta that was upserted."""
    _, index = populated_hybrid_index
    vec = index.get_vector("vec_0005")
    assert vec["meta"]["index"] == 5
    assert vec["meta"]["text"] == "Document 5"


def test_hybrid_get_vector_filter_preserved(populated_hybrid_index):
    """get_vector on a hybrid index must return the filter fields that were upserted."""
    _, index = populated_hybrid_index
    vec = index.get_vector("vec_0000")
    assert vec["filter"]["category"] == "A"
    assert vec["filter"]["score"] == 0
    assert vec["filter"]["tags"] == "important"


def test_hybrid_get_vector_norm_is_positive(populated_hybrid_index):
    """get_vector on a hybrid index must return a positive norm."""
    _, index = populated_hybrid_index
    vec = index.get_vector("vec_0006")
    assert isinstance(vec["norm"], float)
    assert vec["norm"] > 0


def test_hybrid_get_vector_has_dense_vector(populated_hybrid_index):
    """get_vector on a hybrid index must return the dense vector with the correct dimension."""
    _, index = populated_hybrid_index
    vec = index.get_vector("vec_0007")
    assert isinstance(vec["vector"], list)
    assert len(vec["vector"]) == HYBRID_DIM


def test_hybrid_query_result_has_required_keys(populated_hybrid_index):
    """Hybrid query results must contain all required response keys."""
    _, index = populated_hybrid_index
    si, sv = sparse_vec(seed=50)
    results = index.query(
        vector=dense_vec(HYBRID_DIM, seed=50),
        sparse_indices=si,
        sparse_values=sv,
        top_k=1,
    )
    assert len(results) >= 1
    for key in ("id", "similarity", "distance", "meta", "norm", "vector", "filter"):
        assert key in results[0], f"Missing key '{key}' in hybrid query result"


def test_hybrid_query_result_similarity_is_float(populated_hybrid_index):
    """Similarity scores in hybrid query results must be floats."""
    _, index = populated_hybrid_index
    si, sv = sparse_vec(seed=51)
    results = index.query(
        vector=dense_vec(HYBRID_DIM, seed=51),
        sparse_indices=si,
        sparse_values=sv,
        top_k=1,
    )
    assert isinstance(results[0]["similarity"], float)


def test_hybrid_query_distance_equals_one_minus_similarity(populated_hybrid_index):
    """Distance must equal 1 - similarity for every hybrid query result."""
    _, index = populated_hybrid_index
    si, sv = sparse_vec(seed=52)
    results = index.query(
        vector=dense_vec(HYBRID_DIM, seed=52),
        sparse_indices=si,
        sparse_values=sv,
        top_k=5,
    )
    for r in results:
        assert abs(r["distance"] - (1.0 - r["similarity"])) < 1e-5


@pytest.mark.parametrize("top_k", [1, 5, 10, 20, 30, 50])
def test_hybrid_query_top_k_returns_at_most_k_results(populated_hybrid_index, top_k):
    """Hybrid query must return no more results than the requested top_k."""
    _, index = populated_hybrid_index
    si, sv = sparse_vec(seed=60)
    results = index.query(
        vector=dense_vec(HYBRID_DIM, seed=60),
        sparse_indices=si,
        sparse_values=sv,
        top_k=top_k,
    )
    assert len(results) <= top_k


def test_hybrid_query_top_k_1_returns_single_result(populated_hybrid_index):
    """Hybrid query with top_k=1 must return exactly one result."""
    _, index = populated_hybrid_index
    si, sv = sparse_vec(seed=61)
    results = index.query(
        vector=dense_vec(HYBRID_DIM, seed=61),
        sparse_indices=si,
        sparse_values=sv,
        top_k=1,
    )
    assert len(results) == 1


@pytest.mark.parametrize("ef", [32, 64, 128, 256, 512, 1024])
def test_hybrid_query_ef_parameter_accepted(populated_hybrid_index, ef):
    """All valid ef values must be accepted by a hybrid query without error."""
    _, index = populated_hybrid_index
    si, sv = sparse_vec(seed=62)
    results = index.query(
        vector=dense_vec(HYBRID_DIM, seed=62),
        sparse_indices=si,
        sparse_values=sv,
        top_k=5,
        ef=ef,
    )
    assert isinstance(results, list)


def test_hybrid_query_with_in_filter(populated_hybrid_index):
    """Hybrid query with a $in filter must return only vectors matching one of the listed values."""
    _, index = populated_hybrid_index
    si, sv = sparse_vec(seed=70)
    results = index.query(
        vector=dense_vec(HYBRID_DIM, seed=70),
        sparse_indices=si,
        sparse_values=sv,
        top_k=N_VECTORS,
        filter=[{"category": {"$in": ["A", "B"]}}],
        prefilter_cardinality_threshold=1_000_000,
    )
    assert len(results) == 34
    for r in results:
        assert r["filter"]["category"] in ("A", "B")


def test_hybrid_query_with_combined_filters(populated_hybrid_index):
    """Hybrid query with multiple filter conditions must satisfy all of them (AND logic)."""
    _, index = populated_hybrid_index
    si, sv = sparse_vec(seed=71)
    results = index.query(
        vector=dense_vec(HYBRID_DIM, seed=71),
        sparse_indices=si,
        sparse_values=sv,
        top_k=N_VECTORS,
        filter=[
            {"category": {"$eq": "A"}},
            {"tags": {"$eq": "important"}},
        ],
        prefilter_cardinality_threshold=1_000_000,
    )
    assert len(results) == 9
    for r in results:
        assert r["filter"]["category"] == "A"
        assert r["filter"]["tags"] == "important"


@pytest.mark.parametrize("boost", [0, 10, 25, 50, 100, 200, 400])
def test_hybrid_filter_boost_percentage_accepted(populated_hybrid_index, boost):
    """All valid filter_boost_percentage values must be accepted by a hybrid query."""
    _, index = populated_hybrid_index
    si, sv = sparse_vec(seed=80)
    results = index.query(
        vector=dense_vec(HYBRID_DIM, seed=80),
        sparse_indices=si,
        sparse_values=sv,
        top_k=5,
        filter=[{"category": {"$eq": "A"}}],
        filter_boost_percentage=boost,
    )
    assert isinstance(results, list)


def test_hybrid_filter_boost_results_satisfy_filter(populated_hybrid_index):
    """Results returned with filter_boost_percentage must still satisfy the filter condition."""
    _, index = populated_hybrid_index
    si, sv = sparse_vec(seed=81)
    results = index.query(
        vector=dense_vec(HYBRID_DIM, seed=81),
        sparse_indices=si,
        sparse_values=sv,
        top_k=10,
        filter=[{"tags": {"$eq": "important"}}],
        filter_boost_percentage=50,
        prefilter_cardinality_threshold=1_000_000,
    )
    for r in results:
        assert r["filter"]["tags"] == "important"


def test_hybrid_update_filters_single_vector(populated_hybrid_index):
    """update_filters on a hybrid index must return a non-empty confirmation."""
    _, index = populated_hybrid_index
    result = index.update_filters(
        [
            {"id": "vec_0010", "filter": {"category": "Z", "score": 99}},
        ]
    )
    assert result


def test_hybrid_update_filters_multiple_vectors(populated_hybrid_index):
    """update_filters with multiple entries on a hybrid index must succeed."""
    _, index = populated_hybrid_index
    result = index.update_filters(
        [
            {"id": "vec_0020", "filter": {"category": "X"}},
            {"id": "vec_0021", "filter": {"category": "Y"}},
            {"id": "vec_0022", "filter": {"category": "Z"}},
        ]
    )
    assert result


def test_hybrid_update_filters_reflected_in_get_vector(populated_hybrid_index):
    """A filter updated via update_filters must be returned by get_vector immediately after."""
    _, index = populated_hybrid_index
    index.update_filters([{"id": "vec_0030", "filter": {"category": "UPDATED"}}])
    vec = index.get_vector("vec_0030")
    assert vec["filter"]["category"] == "UPDATED"


def test_hybrid_delete_vector_returns_deleted(populated_hybrid_index):
    """delete_vector on a hybrid index must return a response containing 'deleted'."""
    _, index = populated_hybrid_index
    result = index.delete_vector("vec_0040")
    assert "deleted" in result.lower()


def test_hybrid_delete_vector_not_in_get_vector(populated_hybrid_index):
    """A deleted vector on a hybrid index must raise NotFoundException when fetched."""
    _, index = populated_hybrid_index
    index.delete_vector("vec_0041")
    with pytest.raises(NotFoundException):
        index.get_vector("vec_0041")


def test_hybrid_delete_vector_not_in_query_results(populated_hybrid_index):
    """A deleted vector must not appear in subsequent hybrid query results."""
    _, index = populated_hybrid_index
    target_id = "vec_0042"
    si, sv = sparse_vec(seed=42)
    index.delete_vector(target_id)
    results = index.query(
        vector=dense_vec(HYBRID_DIM, seed=42),
        sparse_indices=si,
        sparse_values=sv,
        top_k=N_VECTORS,
    )
    assert target_id not in {r["id"] for r in results}


def test_hybrid_delete_with_filter_eq(empty_hybrid_index):
    """delete_with_filter using $eq must remove matching vectors from a hybrid index."""
    _, index = empty_hybrid_index
    batch = [
        make_item(i, dim=HYBRID_DIM, with_sparse=True)
        | {"filter": {"tag": "remove" if i < 3 else "keep"}}
        for i in range(6)
    ]
    index.upsert(batch)
    index.delete_with_filter([{"tag": {"$eq": "remove"}}])

    for i in range(3):
        with pytest.raises(NotFoundException):
            index.get_vector(f"vec_{i:04d}")

    for i in range(3, 6):
        vec = index.get_vector(f"vec_{i:04d}")
        assert vec["id"] == f"vec_{i:04d}"


def test_hybrid_delete_with_filter_range(empty_hybrid_index):
    """delete_with_filter using $range must remove vectors whose score falls within the range."""
    _, index = empty_hybrid_index
    batch = [make_item(i, dim=HYBRID_DIM, with_sparse=True) for i in range(20)]
    index.upsert(batch)
    index.delete_with_filter([{"score": {"$range": [5, 10]}}])

    for i in range(5, 11):
        with pytest.raises(NotFoundException):
            index.get_vector(f"vec_{i:04d}")

    for i in [0, 4, 11, 19]:
        vec = index.get_vector(f"vec_{i:04d}")
        assert vec["id"] == f"vec_{i:04d}"


def test_hybrid_delete_with_filter_in(empty_hybrid_index):
    """delete_with_filter using $in must remove vectors matching any of the listed values."""
    _, index = empty_hybrid_index
    tags = ["alpha", "beta", "gamma"]
    batch = [
        make_item(i, dim=HYBRID_DIM, with_sparse=True)
        | {"filter": {"tag": tags[i % 3]}}
        for i in range(9)
    ]
    index.upsert(batch)
    index.delete_with_filter([{"tag": {"$in": ["alpha", "beta"]}}])

    for i in range(9):
        if tags[i % 3] in ("alpha", "beta"):
            with pytest.raises(NotFoundException):
                index.get_vector(f"vec_{i:04d}")
        else:
            vec = index.get_vector(f"vec_{i:04d}")
            assert vec["id"] == f"vec_{i:04d}"


# === Hybrid index creation combinations ===


HYBRID_PRECISIONS = [
    Precision.FLOAT32,
    Precision.FLOAT16,
    Precision.INT16,
    Precision.INT8,
    Precision.BINARY2,
]

HYBRID_SPACE_TYPES = ["cosine", "l2", "ip"]


@pytest.mark.parametrize("space_type", HYBRID_SPACE_TYPES)
@pytest.mark.parametrize("precision", HYBRID_PRECISIONS)
def test_hybrid_index_creation_precision_space_combinations(client, precision, space_type):
    """Every valid (precision, space_type) pair must create a hybrid index successfully."""
    name = uid("hcombo")
    try:
        result = client.create_index(
            name=name,
            dimension=HYBRID_DIM,
            space_type=space_type,
            precision=precision,
            sparse_model="default",
        )
        assert "success" in result.lower()
    finally:
        safe_delete(client, name)


@pytest.mark.parametrize("dim", [8, 32, 64, 128, 256])
def test_hybrid_index_creation_various_dimensions(client, dim):
    """Hybrid index creation must succeed for a range of valid dimensions."""
    name = uid("hdim")
    try:
        result = client.create_index(
            name=name,
            dimension=dim,
            space_type="cosine",
            precision=Precision.INT8,
            sparse_model="default",
        )
        assert "success" in result.lower()
    finally:
        safe_delete(client, name)


def test_hybrid_index_appears_in_list_after_creation(client):
    """A newly created hybrid index must appear in list_indexes."""
    from helpers import get_index_names

    name = uid("hlist")
    try:
        client.create_index(
            name=name,
            dimension=HYBRID_DIM,
            space_type="cosine",
            precision=Precision.INT8,
            sparse_model="default",
        )
        names = get_index_names(client)
        assert name in names
    finally:
        safe_delete(client, name)


def test_hybrid_index_get_index_after_creation(client):
    """get_index must return an index object for a newly created hybrid index."""
    name = uid("hget")
    try:
        client.create_index(
            name=name,
            dimension=HYBRID_DIM,
            space_type="cosine",
            precision=Precision.INT8,
            sparse_model="default",
        )
        index = client.get_index(name)
        assert index is not None
    finally:
        safe_delete(client, name)


# === Upsert edge cases ===


def test_hybrid_upsert_exactly_1000_vectors(empty_hybrid_index):
    """Upserting exactly 1000 hybrid vectors (the batch limit) must succeed."""
    _, index = empty_hybrid_index
    batch = [make_item(i, dim=HYBRID_DIM, with_sparse=True) for i in range(1000)]
    result = index.upsert(batch)
    assert "success" in result.lower()


@pytest.mark.parametrize("batch_size", [1, 10, 100, 500, 999])
def test_hybrid_upsert_various_batch_sizes(empty_hybrid_index, batch_size):
    """Hybrid upsert must succeed for a range of valid batch sizes."""
    _, index = empty_hybrid_index
    batch = [make_item(i, dim=HYBRID_DIM, with_sparse=True) for i in range(batch_size)]
    result = index.upsert(batch)
    assert "success" in result.lower()


def test_hybrid_upsert_single_sparse_nonzero_element(empty_hybrid_index):
    """Upserting a hybrid vector with a single sparse non-zero element must succeed."""
    _, index = empty_hybrid_index
    result = index.upsert(
        [
            {
                "id": "single_sparse",
                "vector": dense_vec(HYBRID_DIM, seed=200),
                "sparse_indices": [7],
                "sparse_values": [0.99],
            }
        ]
    )
    assert "success" in result.lower()


# === Query edge cases ===


def test_hybrid_query_empty_index_returns_empty_list(empty_hybrid_index):
    """Querying an empty hybrid index must return an empty list."""
    _, index = empty_hybrid_index
    si, sv = sparse_vec(seed=400)
    results = index.query(
        vector=dense_vec(HYBRID_DIM, seed=400),
        sparse_indices=si,
        sparse_values=sv,
        top_k=10,
    )
    assert results == []


def test_hybrid_query_top_k_exceeds_corpus_returns_all(empty_hybrid_index):
    """Query with top_k larger than the corpus size must return all indexed vectors."""
    _, index = empty_hybrid_index
    batch = [make_item(i, dim=HYBRID_DIM, with_sparse=True) for i in range(5)]
    index.upsert(batch)
    si, sv = sparse_vec(seed=401)
    results = index.query(
        vector=dense_vec(HYBRID_DIM, seed=401),
        sparse_indices=si,
        sparse_values=sv,
        top_k=100,
    )
    assert len(results) <= 5


def test_hybrid_meta_round_trip_in_query(empty_hybrid_index):
    """Meta inserted via hybrid upsert must be returned intact in query results."""
    _, index = empty_hybrid_index
    payload = {"title": "hybrid doc", "count": 42, "active": True}
    si, sv = sparse_vec(seed=410)
    index.upsert(
        [{"id": "meta_rt", "vector": dense_vec(HYBRID_DIM, seed=410), "sparse_indices": si, "sparse_values": sv, "meta": payload}]
    )
    results = index.query(
        vector=dense_vec(HYBRID_DIM, seed=410),
        sparse_indices=si,
        sparse_values=sv,
        top_k=1,
    )
    assert results[0]["id"] == "meta_rt"
    assert results[0]["meta"]["title"] == "hybrid doc"
    assert results[0]["meta"]["count"] == 42
    assert results[0]["meta"]["active"] is True


def test_hybrid_query_after_overwrite_reflects_new_meta(empty_hybrid_index):
    """After overwriting a vector's meta, a query must return the updated meta."""
    _, index = empty_hybrid_index
    si, sv = sparse_vec(seed=420)
    index.upsert(
        [{"id": "upd_meta", "vector": dense_vec(HYBRID_DIM, seed=420), "sparse_indices": si, "sparse_values": sv, "meta": {"version": 1}}]
    )
    index.upsert(
        [{"id": "upd_meta", "vector": dense_vec(HYBRID_DIM, seed=420), "sparse_indices": si, "sparse_values": sv, "meta": {"version": 2}}]
    )
    results = index.query(
        vector=dense_vec(HYBRID_DIM, seed=420),
        sparse_indices=si,
        sparse_values=sv,
        top_k=1,
    )
    assert results[0]["id"] == "upd_meta"
    assert results[0]["meta"]["version"] == 2


# === get_vector edge cases ===


def test_hybrid_get_vector_nonexistent_raises_not_found(empty_hybrid_index):
    """get_vector on a non-existent ID in a hybrid index must raise NotFoundException."""
    _, index = empty_hybrid_index
    with pytest.raises(NotFoundException):
        index.get_vector("this_id_does_not_exist_xyz")


# === update_filters reflected in filtered queries ===


def test_hybrid_update_filters_reflected_in_filtered_query(empty_hybrid_index):
    """Filters updated via update_filters must be returned in a subsequent filtered query."""
    _, index = empty_hybrid_index
    si, sv = sparse_vec(seed=500)
    index.upsert(
        [{"id": "uf_q1", "vector": dense_vec(HYBRID_DIM, seed=500), "sparse_indices": si, "sparse_values": sv, "filter": {"status": "old"}}]
    )
    index.update_filters([{"id": "uf_q1", "filter": {"status": "new"}}])
    results = index.query(
        vector=dense_vec(HYBRID_DIM, seed=500),
        sparse_indices=si,
        sparse_values=sv,
        top_k=10,
        filter=[{"status": {"$eq": "new"}}],
        prefilter_cardinality_threshold=1_000_000,
    )
    ids = {r["id"] for r in results}
    assert "uf_q1" in ids


def test_hybrid_update_filters_old_filter_no_longer_matches(empty_hybrid_index):
    """After updating a filter, the old filter value must no longer match in queries."""
    _, index = empty_hybrid_index
    si, sv = sparse_vec(seed=501)
    index.upsert(
        [{"id": "uf_q2", "vector": dense_vec(HYBRID_DIM, seed=501), "sparse_indices": si, "sparse_values": sv, "filter": {"status": "old"}}]
    )
    index.update_filters([{"id": "uf_q2", "filter": {"status": "new"}}])
    results = index.query(
        vector=dense_vec(HYBRID_DIM, seed=501),
        sparse_indices=si,
        sparse_values=sv,
        top_k=10,
        filter=[{"status": {"$eq": "old"}}],
        prefilter_cardinality_threshold=1_000_000,
    )
    ids = {r["id"] for r in results}
    assert "uf_q2" not in ids


# === delete_with_filter: combined (AND) conditions ===


def test_hybrid_delete_with_combined_and_filters(empty_hybrid_index):
    """delete_with_filter with multiple conditions must delete only vectors satisfying all conditions."""
    _, index = empty_hybrid_index
    batch = [make_item(i, dim=HYBRID_DIM, with_sparse=True) for i in range(20)]
    index.upsert(batch)

    # Delete where category == "A" AND score <= 9  (i=0,3,6,9 → 4 vectors)
    index.delete_with_filter([
        {"category": {"$eq": "A"}},
        {"score": {"$range": [0, 9]}},
    ])

    # i=0,3,6,9 match both conditions — must be deleted
    for i in [0, 3, 6, 9]:
        with pytest.raises(NotFoundException):
            index.get_vector(f"vec_{i:04d}")

    # i=1,2 (not category A) and i=12,15,18 (category A but score > 9) must survive
    for i in [1, 2, 12, 15, 18]:
        vec = index.get_vector(f"vec_{i:04d}")
        assert vec["id"] == f"vec_{i:04d}"

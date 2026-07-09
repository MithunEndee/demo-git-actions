"""
Tests for client-side validation and server-side HTTP error mapping.

Client-side: field config validation, name rules, upsert/search parameter
bounds, filter key/value size limits, and collection schema constraints.

Server-side: HTTP status -> exception class mapping (AuthenticationException,
ConflictException, NotFoundException, etc.) tested via raise_exception() unit
tests and real API calls with bad credentials or duplicate resources.
"""

import os

import pytest
from helpers import (
    DENSE_FIELD,
    DIM,
    HYBRID_DIM,
    MV_FIELD,
    SPARSE_FIELD,
    dense_vec,
    make_dense_field,
    safe_delete,
    sparse_vec,
    uid,
)

from endee import Endee, rerank
from endee.constants import MAX_KEY_BYTES, MAX_VALUE_BYTES
from endee.exceptions import (
    APIException,
    AuthenticationException,
    ConflictException,
    EndeeException,
    ForbiddenException,
    NotFoundException,
    ServerException,
    SubscriptionException,
    raise_exception,
)
from endee.schema import CollectionFieldConfig, CollectionFieldParams

# -- CollectionFieldParams validation (client-side, ValueError) ---------------


@pytest.mark.parametrize("bad_space_type", ["euclidean", "dot", "manhattan", ""])
def test_create_collection_invalid_space_type_raises(client, bad_space_type):
    """CollectionFieldParams with an unsupported space_type must raise ValueError."""
    with pytest.raises(ValueError):
        CollectionFieldConfig(
            name=DENSE_FIELD,
            type="vector",
            params=CollectionFieldParams(
                dimension=DIM, space_type=bad_space_type, precision="int8"
            ),
        )


@pytest.mark.parametrize("bad_precision", ["fp32", "int4", "uint8", "half", ""])
def test_create_collection_invalid_precision_raises(client, bad_precision):
    """CollectionFieldParams with an unrecognised precision must raise ValueError."""
    with pytest.raises(ValueError):
        CollectionFieldConfig(
            name=DENSE_FIELD,
            type="vector",
            params=CollectionFieldParams(
                dimension=DIM, space_type="cosine", precision=bad_precision
            ),
        )


def test_create_collection_dimension_below_minimum_raises(client):
    """CollectionFieldParams with dimension=1 (below minimum) must raise ValueError."""
    with pytest.raises(ValueError):
        CollectionFieldParams(dimension=1, space_type="cosine", precision="int8")


# -- CollectionFieldConfig validation (client-side, ValueError) ---------------


@pytest.mark.parametrize("bad_type", ["dense", "blob", "tensor", ""])
def test_create_collection_invalid_field_type_raises(bad_type):
    """CollectionFieldConfig with an unsupported field type must raise ValueError."""
    with pytest.raises(ValueError):
        CollectionFieldConfig(name=DENSE_FIELD, type=bad_type)


def test_create_collection_sparse_field_without_sparse_model_raises():
    """A sparse field without sparse_model must raise ValueError."""
    with pytest.raises(ValueError):
        CollectionFieldConfig(name=SPARSE_FIELD, type="sparse")


def test_create_collection_vector_field_with_sparse_model_raises():
    """A vector field with sparse_model must raise ValueError."""
    with pytest.raises(ValueError):
        CollectionFieldConfig(
            name=DENSE_FIELD,
            type="vector",
            sparse_model="default",
            params=CollectionFieldParams(
                dimension=DIM, space_type="cosine", precision="int8"
            ),
        )


def test_create_collection_multi_vector_without_pooling_method_raises():
    """A multi_vector field without pooling_method must raise ValueError."""
    with pytest.raises(ValueError):
        CollectionFieldConfig(
            name="colbert",
            type="multi_vector",
            params=CollectionFieldParams(
                dimension=DIM, space_type="cosine", precision="int8"
            ),
        )


@pytest.mark.parametrize("bad_sparse_model", ["invalid_model_xyz", "tfidf", "bm25_v2"])
def test_create_collection_invalid_sparse_model_raises(bad_sparse_model):
    """CollectionFieldConfig with an unrecognised sparse_model must raise ValueError."""
    with pytest.raises(ValueError):
        CollectionFieldConfig(
            name=SPARSE_FIELD, type="sparse", sparse_model=bad_sparse_model
        )


def test_create_collection_multi_vector_with_sparse_model_raises():
    """A multi_vector field with sparse_model must raise ValueError."""
    with pytest.raises(ValueError):
        CollectionFieldConfig(
            name=MV_FIELD,
            type="multi_vector",
            pooling_method="mean",
            sparse_model="default",
            params=CollectionFieldParams(
                dimension=DIM, space_type="cosine", precision="int8"
            ),
        )


# -- HTTP library validation (client-side, ValueError) ------------------------


def test_unsupported_http_library_raises():
    """Instantiating Endee with an unsupported http_library must raise ValueError."""
    with pytest.raises(ValueError, match="[Uu]nsupported"):
        Endee(http_library="curl")


def test_unsupported_http_library_grpc_raises():
    """Instantiating Endee with 'grpc' must raise ValueError."""
    with pytest.raises(ValueError, match="[Uu]nsupported"):
        Endee(http_library="grpc")


@pytest.mark.parametrize("bad_name", ["", "has space", "has.dot", "has@at", "bad/name"])
def test_create_collection_invalid_name_raises(client, bad_name):
    """create_collection with a name the server rejects must raise an error."""
    try:
        with pytest.raises((ValueError, EndeeException)):
            client.create_collection(name=bad_name, fields=[make_dense_field()])
    finally:
        safe_delete(client, bad_name)


# -- Collection name validation (client-side) ----------------------------------


@pytest.mark.parametrize("bad_name", ["has-hyphen", "__reserved", "a" * 49])
def test_create_collection_invalid_name_client_side(client, bad_name):
    """create_collection with a name the client rejects must raise ValueError."""
    try:
        with pytest.raises((ValueError, EndeeException)):
            client.create_collection(name=bad_name, fields=[make_dense_field()])
    finally:
        safe_delete(client, bad_name)


# -- Duplicate collection (server-side, ConflictException) --------------------


def test_create_duplicate_collection_raises_conflict(client):
    """Creating a collection with an existing name must raise ConflictException."""
    name = uid("dup")
    client.create_collection(
        name=name,
        fields=[
            CollectionFieldConfig(
                name=DENSE_FIELD,
                type="vector",
                params=CollectionFieldParams(
                    dimension=DIM, space_type="cosine", precision="int8"
                ),
            ).to_dict()
        ],
    )
    try:
        with pytest.raises(ConflictException):
            client.create_collection(
                name=name,
                fields=[
                    CollectionFieldConfig(
                        name=DENSE_FIELD,
                        type="vector",
                        params=CollectionFieldParams(
                            dimension=DIM, space_type="cosine", precision="int8"
                        ),
                    ).to_dict()
                ],
            )
    finally:
        safe_delete(client, name)


# -- Not found (server-side, NotFoundException) -------------------------------


def test_get_nonexistent_collection_raises_not_found(client):
    """get_collection for a nonexistent collection must raise NotFoundException."""
    with pytest.raises(NotFoundException):
        client.get_collection("this_collection_does_not_exist_xyz123")


def test_delete_nonexistent_collection_raises_not_found(client):
    """delete_collection for a nonexistent collection must raise NotFoundException."""
    with pytest.raises(NotFoundException):
        client.delete_collection("nonexistent_collection_xyz789")


def test_delete_object_nonexistent_id_raises_not_found(empty_collection):
    """delete_object for an ID that does not exist must raise NotFoundException."""
    _, collection = empty_collection
    with pytest.raises(NotFoundException):
        collection.delete_object("id_that_does_not_exist_xyz")


def test_delete_object_twice_raises_not_found(populated_collection):
    """Deleting the same object twice must raise NotFoundException."""
    _, collection = populated_collection
    collection.delete_object("vec_0001")
    with pytest.raises(NotFoundException):
        collection.delete_object("vec_0001")


def test_describe_after_collection_deleted_raises_not_found(client):
    """describe() on a deleted collection must raise NotFoundException."""
    name = uid("descerr")
    client.create_collection(name=name, fields=[make_dense_field()])
    collection = client.get_collection(name)
    client.delete_collection(name)
    with pytest.raises(NotFoundException):
        collection.describe()


def test_upsert_after_collection_deleted_raises(client):
    """upsert into a deleted collection must raise an error."""
    name = uid("upserr")
    client.create_collection(name=name, fields=[make_dense_field()])
    collection = client.get_collection(name)
    client.delete_collection(name)
    with pytest.raises(EndeeException):
        collection.upsert([{"id": "x", "fields": {DENSE_FIELD: dense_vec()}}])


def test_create_collection_duplicate_field_names_raises(client):
    """create_collection with two fields sharing the same name must raise an error."""
    name = uid("dupf")
    try:
        with pytest.raises((ValueError, EndeeException)):
            client.create_collection(
                name=name,
                fields=[make_dense_field(), make_dense_field()],
            )
    finally:
        safe_delete(client, name)


def test_search_with_empty_vector_raises(populated_collection):
    """Searching with an empty vector [] must raise an error."""
    _, collection = populated_collection
    with pytest.raises((ValueError, EndeeException)):
        collection.search(fields={DENSE_FIELD: {"query": [], "limit": 5}})["results"][DENSE_FIELD]


# -- Upsert errors ------------------------------------------------------------


def test_upsert_wrong_dimension_raises(empty_collection):
    """Upserting with more dimensions than the field expects must raise an error."""
    _, collection = empty_collection
    wrong_dim_vec = dense_vec(dim=DIM + 1)
    with pytest.raises((ValueError, EndeeException)):
        collection.upsert([{"id": "bad", "fields": {DENSE_FIELD: wrong_dim_vec}}])


def test_upsert_too_few_dimensions_raises(empty_collection):
    """Upserting with fewer dimensions than the field expects must raise an error."""
    _, collection = empty_collection
    with pytest.raises((ValueError, EndeeException)):
        collection.upsert(
            [{"id": "short", "fields": {DENSE_FIELD: dense_vec(dim=DIM - 1)}}]
        )


def test_upsert_empty_id_raises(empty_collection):
    """Upserting an object with an empty string ID must raise ValueError (client-side check)."""
    _, collection = empty_collection
    with pytest.raises((ValueError, EndeeException)):
        collection.upsert([{"id": "", "fields": {DENSE_FIELD: dense_vec()}}])


def test_upsert_sparse_indices_values_length_mismatch_raises(empty_hybrid_collection):
    """Upserting sparse vectors with mismatched indices/values lengths must raise."""
    _, collection = empty_hybrid_collection
    si, sv = sparse_vec()
    with pytest.raises((ValueError, EndeeException)):
        collection.upsert(
            [
                {
                    "id": "mis",
                    "fields": {
                        DENSE_FIELD: dense_vec(HYBRID_DIM),
                        SPARSE_FIELD: {"indices": si, "values": sv[:-1]},
                    },
                }
            ]
        )


def test_upsert_multi_vector_empty_vectors_raises(empty_mv_collection):
    """Upserting a multi_vector object with an empty vector list must raise an error."""
    _, collection = empty_mv_collection
    with pytest.raises((ValueError, Exception)):
        collection.upsert([{"id": "empty_vec", "fields": {MV_FIELD: []}}])


def test_upsert_multi_vector_inconsistent_dimensions_raises(empty_mv_collection):
    """Upserting multi_vector with mismatched vector dimensions must raise an error."""
    _, collection = empty_mv_collection
    mixed = [dense_vec(dim=DIM, seed=0), dense_vec(dim=DIM + 1, seed=1)]
    with pytest.raises((ValueError, Exception)):
        collection.upsert([{"id": "bad_vec", "fields": {MV_FIELD: mixed}}])


# -- Search errors ------------------------------------------------------------


def test_search_wrong_dimension_raises(populated_collection):
    """Searching with a vector of the wrong dimension must raise an error."""
    _, collection = populated_collection
    with pytest.raises((ValueError, EndeeException)):
        collection.search(fields={DENSE_FIELD: {"query": dense_vec(dim=DIM + 2), "limit": 5}})["results"][DENSE_FIELD]


# -- set_token behavior -------------------------------------------------------


def test_set_token_updates_stored_token():
    """set_token must update the token stored on the Endee client instance."""
    c = Endee(token="user:original_token:region")
    c.set_token("user:updated_token:region")
    assert c.token == "user:updated_token:region"


# -- Authentication errors ----------------------------------------------------


def _derive_base_url() -> str:
    """Return the base URL to use for auth-error tests."""
    base_url = os.environ.get("ENDEE_BASE_URL")
    if base_url:
        return base_url
    token = os.environ.get("ENDEE_TOKEN", "")
    parts = token.split(":")
    if len(parts) == 3:
        return f"https://{parts[2]}.endee.io/api/v2"
    pytest.skip(
        "ENDEE_TOKEN does not contain a region part - expected format user:token:region"
    )


def test_invalid_token_raises_authentication_error():
    """API calls with an invalid token must raise AuthenticationException."""
    bad_client = Endee(token="invalid_token_xyz_12345")
    bad_client.set_base_url(_derive_base_url())

    with pytest.raises(AuthenticationException):
        bad_client.list_collections()


def test_empty_token_raises_authentication_error():
    """API calls with an empty token must raise AuthenticationException."""
    bad_client = Endee(token="")
    bad_client.set_base_url(_derive_base_url())

    with pytest.raises(AuthenticationException):
        bad_client.list_collections()


def test_set_token_to_invalid_causes_auth_error():
    """After set_token with invalid token, calls must raise AuthenticationException."""
    c = Endee(token=os.environ.get("ENDEE_TOKEN"))
    c.set_base_url(_derive_base_url())
    c.set_token("user:now_invalid_token:region")

    with pytest.raises(AuthenticationException):
        c.list_collections()


# -- Client-side field data type validation (_normalize_upsert_field) ---------


def test_upsert_field_data_string_raises(empty_collection):
    """Passing a string as field data must raise ValueError client-side."""
    _, collection = empty_collection
    with pytest.raises(ValueError):
        collection.upsert([{"id": "bad", "fields": {DENSE_FIELD: "not_a_vector"}}])


def test_upsert_field_data_integer_raises(empty_collection):
    """Passing an integer as field data must raise ValueError client-side."""
    _, collection = empty_collection
    with pytest.raises(ValueError):
        collection.upsert([{"id": "bad", "fields": {DENSE_FIELD: 42}}])


def test_upsert_field_data_none_raises(empty_collection):
    """Passing None as field data must raise ValueError before hitting the server."""
    _, collection = empty_collection
    with pytest.raises(ValueError):
        collection.upsert([{"id": "bad", "fields": {DENSE_FIELD: None}}])


# -- Client-side search field data type validation (_build_field_search_entry) -


def test_search_field_data_string_raises(populated_collection):
    """Passing a string as search field data must raise ValueError client-side."""
    _, collection = populated_collection
    with pytest.raises(ValueError):
        collection.search(fields={DENSE_FIELD: "not_a_vector"})


def test_search_field_data_integer_raises(populated_collection):
    """Passing an integer as search field data must raise ValueError client-side."""
    _, collection = populated_collection
    with pytest.raises(ValueError):
        collection.search(fields={DENSE_FIELD: 42})


# -- Search parameter bounds (server-side) ------------------------------------


def test_search_limit_zero_raises(populated_collection):
    """search with limit=0 must raise ValueError (client-side validation)."""
    _, collection = populated_collection
    with pytest.raises(ValueError):
        collection.search(fields={DENSE_FIELD: {"query": dense_vec(), "limit": 0}})


def test_search_limit_negative_raises(populated_collection):
    """search with a negative limit must raise ValueError (client-side validation)."""
    _, collection = populated_collection
    with pytest.raises(ValueError):
        collection.search(fields={DENSE_FIELD: {"query": dense_vec(), "limit": -1}})


def test_search_ef_search_zero_raises(populated_collection):
    """search with ef_search=0 must raise ValueError (client-side validation)."""
    _, collection = populated_collection
    with pytest.raises(ValueError):
        collection.search(fields={DENSE_FIELD: {"query": dense_vec(), "limit": 5}}, ef_search=0)


def test_search_ef_search_over_max_raises(populated_collection):
    """search with ef_search above the maximum must raise ValueError (client-side)."""
    _, collection = populated_collection
    with pytest.raises(ValueError):
        collection.search(fields={DENSE_FIELD: {"query": dense_vec(), "limit": 5}}, ef_search=1025)


# -- Reranker validation (client-side) ----------------------------------------


def test_search_invalid_reranker_raises(populated_collection):
    """rerank() with an unrecognised reranker name must raise ValueError (client-side)."""
    _, collection = populated_collection
    raw = collection.search(fields={DENSE_FIELD: {"query": dense_vec(), "limit": 5}})
    with pytest.raises(ValueError):
        rerank(raw, name="invalid_reranker_xyz", limit=5)


def test_search_empty_fields_raises(populated_collection):
    """search with an empty fields dict must raise ValueError (client-side)."""
    _, collection = populated_collection
    with pytest.raises(ValueError):
        collection.search(fields={})


# -- Filter key/value size limits (client-side) --------------------------------


def test_upsert_filter_key_too_long_raises(empty_collection):
    """A filter key exceeding MAX_KEY_BYTES must raise ValueError client-side."""
    _, collection = empty_collection
    long_key = "k" * (MAX_KEY_BYTES + 1)
    with pytest.raises(ValueError):
        collection.upsert(
            [
                {
                    "id": "fk_long",
                    "filter": {long_key: "value"},
                    "fields": {DENSE_FIELD: dense_vec()},
                }
            ]
        )


def test_upsert_filter_value_too_long_raises(empty_collection):
    """A filter string value exceeding MAX_VALUE_BYTES must raise ValueError client-side."""
    _, collection = empty_collection
    long_value = "v" * (MAX_VALUE_BYTES + 1)
    with pytest.raises(ValueError):
        collection.upsert(
            [
                {
                    "id": "fv_long",
                    "filter": {"category": long_value},
                    "fields": {DENSE_FIELD: dense_vec()},
                }
            ]
        )


# -- raise_exception unit tests (no server needed) ----------------------------


def test_raise_exception_400():
    """raise_exception(400, ...) must raise APIException."""
    with pytest.raises(APIException):
        raise_exception(400, '{"error": "bad request"}')


def test_raise_exception_401():
    """raise_exception(401, ...) must raise AuthenticationException."""
    with pytest.raises(AuthenticationException):
        raise_exception(401, '{"error": "unauthorized"}')


def test_raise_exception_402():
    """raise_exception(402, ...) must raise SubscriptionException."""
    with pytest.raises(SubscriptionException):
        raise_exception(402, '{"error": "payment required"}')


def test_raise_exception_403():
    """raise_exception(403, ...) must raise ForbiddenException."""
    with pytest.raises(ForbiddenException):
        raise_exception(403, '{"error": "forbidden"}')


def test_raise_exception_404():
    """raise_exception(404, ...) must raise NotFoundException."""
    with pytest.raises(NotFoundException):
        raise_exception(404, '{"error": "not found"}')


def test_raise_exception_409():
    """raise_exception(409, ...) must raise ConflictException."""
    with pytest.raises(ConflictException):
        raise_exception(409, '{"error": "conflict"}')


def test_raise_exception_500():
    """raise_exception(500, ...) must raise ServerException."""
    with pytest.raises(ServerException):
        raise_exception(500, "internal server error")


def test_raise_exception_503():
    """raise_exception(503, ...) must raise ServerException."""
    with pytest.raises(ServerException):
        raise_exception(503, "service unavailable")


def test_raise_exception_plain_text_message():
    """raise_exception must fall back to raw text when response is not JSON."""
    with pytest.raises(APIException) as exc_info:
        raise_exception(400, "plain text error message")
    assert "plain text error message" in str(exc_info.value)


def test_raise_exception_json_error_field_extracted():
    """raise_exception must extract the 'error' field from a JSON response."""
    with pytest.raises(APIException) as exc_info:
        raise_exception(400, '{"error": "specific error details"}')
    assert "specific error details" in str(exc_info.value)


def test_raise_exception_unknown_status_code_raises_api_exception():
    """raise_exception with an unrecognised 4xx status must raise APIException."""
    with pytest.raises(APIException):
        raise_exception(418, '{"error": "I am a teapot"}')


# -- Client-side validation for admin methods ---------------------------------


def test_create_database_empty_name_raises():
    """create_database with an empty db_name must raise ValueError."""
    c = Endee(token=os.environ.get("ENDEE_TOKEN"))
    with pytest.raises(ValueError):
        c.create_database("")


def test_create_database_invalid_type_raises():
    """create_database with an invalid db_type must raise ValueError."""
    c = Endee(token=os.environ.get("ENDEE_TOKEN"))
    with pytest.raises(ValueError, match="db_type"):
        c.create_database("testdb", db_type="ultra")


def test_set_database_type_invalid_raises():
    """set_database_type with an invalid tier must raise ValueError."""
    c = Endee(token=os.environ.get("ENDEE_TOKEN"))
    with pytest.raises(ValueError, match="db_type"):
        c.set_database_type("testdb", db_type="invalid_tier")


def test_create_my_token_empty_name_raises():
    """create_my_token with an empty name must raise ValueError."""
    c = Endee(token=os.environ.get("ENDEE_TOKEN"))
    with pytest.raises(ValueError):
        c.create_my_token("")


def test_create_my_token_invalid_type_raises():
    """create_my_token with an invalid token_type must raise ValueError."""
    c = Endee(token=os.environ.get("ENDEE_TOKEN"))
    with pytest.raises(ValueError, match="token_type"):
        c.create_my_token("test_tok", token_type="admin")


def test_create_collection_with_pydantic_model_raises(client):
    """create_collection with CollectionFieldConfig (not dict) must raise ValueError."""
    name = uid("pyd")
    try:
        with pytest.raises(ValueError):
            client.create_collection(
                name=name,
                fields=[
                    CollectionFieldConfig(
                        name=DENSE_FIELD,
                        type="vector",
                        params=CollectionFieldParams(
                            dimension=DIM, space_type="cosine", precision="int8"
                        ),
                    )
                ],
            )
    finally:
        safe_delete(client, name)


def test_create_collection_with_to_dict_succeeds(client):
    """create_collection with CollectionFieldConfig.to_dict() must succeed."""
    name = uid("tod")
    try:
        result = client.create_collection(
            name=name,
            fields=[
                CollectionFieldConfig(
                    name=DENSE_FIELD,
                    type="vector",
                    params=CollectionFieldParams(
                        dimension=DIM, space_type="cosine", precision="int8"
                    ),
                ).to_dict()
            ],
        )
        assert isinstance(result, dict)
    finally:
        safe_delete(client, name)


# -- get_objects client-side validation ----------------------------------------


def test_get_objects_empty_list_raises(empty_collection):
    """get_objects with an empty list must raise ValueError."""
    _, collection = empty_collection
    with pytest.raises(ValueError):
        collection.get_objects([])


# -- delete_by_filter client-side validation -----------------------------------


def test_delete_by_filter_non_list_raises(empty_collection):
    """delete_by_filter with a dict instead of list must raise ValueError."""
    _, collection = empty_collection
    with pytest.raises(ValueError):
        collection.delete_by_filter({"category": {"$eq": "A"}})


# -- update_filters client-side validation ------------------------------------


def test_update_filters_empty_list_raises(empty_collection):
    """update_filters with an empty list must raise ValueError."""
    _, collection = empty_collection
    with pytest.raises(ValueError):
        collection.update_filters([])


# -- rebuild client-side validation -------------------------------------------


def test_rebuild_empty_list_raises(empty_collection):
    """rebuild with an empty list must raise ValueError."""
    _, collection = empty_collection
    with pytest.raises(ValueError, match="non-empty list"):
        collection.rebuild([])


def test_rebuild_missing_field_key_raises(empty_collection):
    """rebuild with a config dict missing 'field' must raise ValueError."""
    _, collection = empty_collection
    with pytest.raises(ValueError, match="must include a 'field' name"):
        collection.rebuild([{"M": 8}])


def test_rebuild_non_list_arg_raises(empty_collection):
    """rebuild with a non-list argument must raise ValueError."""
    _, collection = empty_collection
    with pytest.raises(ValueError, match="requires a non-empty list"):
        collection.rebuild("bad")


# -- Search param bounds: prefilter_cardinality_threshold & filter_boost_percentage --


def test_search_prefilter_cardinality_threshold_below_min_raises(populated_collection):
    """search with prefilter_cardinality_threshold below 1000 must raise ValueError."""
    _, collection = populated_collection
    with pytest.raises(ValueError):
        collection.search(
            fields={DENSE_FIELD: {"query": dense_vec(), "limit": 5}},
            prefilter_cardinality_threshold=999,
        )


def test_search_prefilter_cardinality_threshold_above_max_raises(populated_collection):
    """search with prefilter_cardinality_threshold above 1000000 must raise ValueError."""
    _, collection = populated_collection
    with pytest.raises(ValueError):
        collection.search(
            fields={DENSE_FIELD: {"query": dense_vec(), "limit": 5}},
            prefilter_cardinality_threshold=1000001,
        )


def test_search_filter_boost_percentage_negative_raises(populated_collection):
    """search with filter_boost_percentage=-1 must raise ValueError."""
    _, collection = populated_collection
    with pytest.raises(ValueError):
        collection.search(
            fields={DENSE_FIELD: {"query": dense_vec(), "limit": 5}},
            filter_boost_percentage=-1,
        )


def test_search_filter_boost_percentage_above_max_raises(populated_collection):
    """search with filter_boost_percentage=101 must raise ValueError."""
    _, collection = populated_collection
    with pytest.raises(ValueError):
        collection.search(
            fields={DENSE_FIELD: {"query": dense_vec(), "limit": 5}},
            filter_boost_percentage=101,
        )


# -- backup client-side validation -------------------------------------------


def test_create_backup_empty_name_raises(empty_collection):
    """create_backup with an empty name must raise ValueError."""
    _, collection = empty_collection
    with pytest.raises(ValueError):
        collection.create_backup(name="")


# -- Exception message content ------------------------------------------------


def test_not_found_exception_message_is_non_empty(client):
    """NotFoundException from a real API call must have a non-empty message."""
    with pytest.raises(NotFoundException) as exc_info:
        client.get_collection("this_collection_does_not_exist_xyz999")
    assert str(exc_info.value), "NotFoundException must carry a non-empty message"


def test_conflict_exception_message_is_non_empty(client):
    """ConflictException from a real API call must have a non-empty message."""
    name = uid("cmsg")
    client.create_collection(name=name, fields=[make_dense_field()])
    try:
        with pytest.raises(ConflictException) as exc_info:
            client.create_collection(name=name, fields=[make_dense_field()])
        assert str(exc_info.value), "ConflictException must carry a non-empty message"
    finally:
        safe_delete(client, name)

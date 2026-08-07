"""Unit tests: all tests that mock the `endee` client (no network).

`TestVectorStoreUnit` covers core CRUD on `EndeeVectorStore`
(init/create-or-reuse, save/add, search, delete, update_filters, close,
sparse auto-detection wiring).

`TestFiltersUnit` covers filter-list normalization and filter-data
construction.

`TestSparseUnit` covers `SparseVector`, `SparseModelAdapter`,
`wrap_sparse_model`, `EndeeModelSparse`, and hybrid search RRF wiring.
"""

from __future__ import annotations

import contextlib
import uuid
from unittest.mock import MagicMock

import numpy as np
import pytest
from conftest import ALL_PRECISIONS
from endee.exceptions import ConflictException, NotFoundException

from crewai_endee.sparse_embeddings import (
    EndeeModelSparse,
    SparseEmbeddings,
    SparseModelAdapter,
    SparseVector,
    wrap_sparse_model,
)
from crewai_endee.vector_store import EndeeVectorStore, _truncate_text

DENSE_FIELD = {
    "name": "dense",
    "type": "vector",
    "params": {"dimension": 16, "space_type": "cosine", "precision": "float32"},
}
SPARSE_FIELD_BM25 = {"name": "sparse", "type": "sparse", "sparse_model": "endee_bm25"}
SPARSE_FIELD_DEFAULT = {"name": "sparse", "type": "sparse", "sparse_model": "default"}


# ═══════════════════════════════════════════════════════════════════════════
# Vector store
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestVectorStoreUnit:
    # ── init / collection lifecycle ─────────────────────────────────────────

    def test_init_creates_collection_when_absent(self, make_store):
        """Constructing a store with no existing collection must create one."""
        store = make_store()
        assert store._collection is not None
        assert store._collection.name == "test_collection"

    def test_init_reuses_existing_collection(self, make_store, mock_endee_client):
        """Constructing a store must reuse, not recreate, an existing collection."""
        client = mock_endee_client()
        client.create_collection(name="test_collection", fields=[DENSE_FIELD])
        original_collection = client.get_collection("test_collection")

        store = make_store(endee_client=client)
        assert store._collection is original_collection

    def test_force_recreate_deletes_and_recreates(self, make_store, mock_endee_client):
        """force_recreate=True must delete the old collection, creating an empty one."""
        client = mock_endee_client()
        client.create_collection(name="test_collection", fields=[DENSE_FIELD])
        old_collection = client.get_collection("test_collection")
        old_collection.upsert(
            [{"id": "x", "meta": {}, "filter": {}, "fields": {"dense": [0.0] * 16}}]
        )

        store = make_store(endee_client=client, force_recreate=True)
        assert store._collection is not old_collection
        assert store._collection.describe()["num_objects"] == 0

    def test_ensure_collection_returns_collection(self, make_store):
        """ensure_collection() must return the store's underlying collection."""
        store = make_store()
        assert store.ensure_collection() is store._collection

    def test_ensure_index_is_alias_for_ensure_collection(self, make_store):
        """ensure_index() must alias ensure_collection() and return the same value."""
        store = make_store()
        assert store.ensure_index() is store.ensure_collection()

    # ── field role detection ─────────────────────────────────────────────────

    def test_detect_field_roles_raises_without_vector_field(self, make_store):
        """Creating a store with no vector field configured must raise a ValueError."""
        with pytest.raises(ValueError, match="vector"):
            make_store(fields=[SPARSE_FIELD_DEFAULT])

    def test_detect_field_roles_picks_first_dense_and_sparse(self, make_store):
        """_detect_field_roles picks the first vector/sparse fields as dense/sparse."""
        fields = [
            {"name": "other_sparse", "type": "sparse", "sparse_model": "default"},
            DENSE_FIELD,
            {**DENSE_FIELD, "name": "second_dense"},
        ]
        store = make_store(fields=fields)
        assert store.dense_field_name == "dense"
        assert store.sparse_field_name == "other_sparse"

    # ── precision pass-through ───────────────────────────────────────────────

    @pytest.mark.parametrize("precision", ALL_PRECISIONS)
    def test_create_collection_preserves_dense_field_precision(
        self, make_store, mock_endee_client, precision
    ):
        """Every Precision value reaches create_collection() unchanged."""
        client = mock_endee_client()
        fields = [
            {
                "name": "dense",
                "type": "vector",
                "params": {
                    "dimension": 16,
                    "space_type": "cosine",
                    "precision": precision,
                },
            }
        ]

        store = make_store(endee_client=client, fields=fields)

        assert store._collection.fields[0]["params"]["precision"] == precision
        assert (
            client.get_collection("test_collection").fields[0]["params"]["precision"]
            == precision
        )

    # ── client init ──────────────────────────────────────────────────────────

    def test_init_client_uses_api_token_when_no_client_given(self, make_store):
        """Passing api_token without a client creates an authenticated client."""
        store = make_store(api_token="tok-123")
        assert store._client.token == "tok-123"

    def test_init_client_defaults_to_token_none_when_neither_given(self, make_store):
        """With no api_token or client given, the client's token defaults to None."""
        store = make_store()
        assert store._client.token is None

    def test_init_client_sets_base_url_when_provided(self, make_store):
        """Passing base_url must call set_base_url() on the client with that URL."""
        store = make_store()
        store._client.set_base_url = MagicMock()
        make_store(base_url="https://example.test", endee_client=store._client)
        store._client.set_base_url.assert_called_once_with("https://example.test")

    def test_init_client_skips_set_base_url_when_absent(self, make_store):
        """Re-running _init_client with base_url=None must not call set_base_url()."""
        store = make_store()
        store._client.set_base_url = MagicMock()
        store._init_client(api_token=None, base_url=None, endee_client=store._client)
        store._client.set_base_url.assert_not_called()

    # ── save() ────────────────────────────────────────────────────────────────

    def test_save_builds_upsert_payload(self, make_store):
        """save() upserts one entry: dense vector, meta fields, primitive filter."""
        store = make_store()
        store._collection.upsert = MagicMock(wraps=store._collection.upsert)

        store.save("hello world", {"lang": "en", "nested": {"a": 1}, "tags": [1, 2]})

        store._collection.upsert.assert_called_once()
        (entries,), _ = store._collection.upsert.call_args
        assert len(entries) == 1
        entry = entries[0]

        assert "dense" in entry["fields"]
        assert len(entry["fields"]["dense"]) == 16
        assert entry["meta"]["text"] == "hello world"
        assert entry["meta"]["metadata"] == {
            "lang": "en",
            "nested": {"a": 1},
            "tags": [1, 2],
        }
        # primitive metadata promoted to filter; dict/list values excluded
        assert entry["filter"] == {"lang": "en"}
        assert "nested" not in entry["filter"]
        assert "tags" not in entry["filter"]

    def test_save_and_search_use_custom_payload_keys(self, make_store):
        """save()/search() must read back text/metadata under custom payload keys."""
        store = make_store(content_payload_key="body", metadata_payload_key="meta_info")
        store._collection.upsert = MagicMock(wraps=store._collection.upsert)

        store.save("custom key text", {"lang": "en"})

        (entries,), _ = store._collection.upsert.call_args
        entry = entries[0]
        assert entry["meta"]["body"] == "custom key text"
        assert entry["meta"]["meta_info"] == {"lang": "en"}
        assert "text" not in entry["meta"]
        assert "metadata" not in entry["meta"]

        results = store.search("custom key text", limit=1)
        assert results[0]["content"] == "custom key text"
        assert results[0]["metadata"] == {"lang": "en"}

    def test_save_with_hybrid_populates_sparse_field(self, make_store):
        """save() on a hybrid collection populates the sparse field with its vector."""
        store = make_store(fields=[DENSE_FIELD, SPARSE_FIELD_BM25])
        assert store._sparse_embeddings is not None
        store._sparse_embeddings.embed_documents = MagicMock(
            return_value=[type("SV", (), {"indices": [1, 2], "values": [0.5, 0.5]})()]
        )
        store._collection.upsert = MagicMock(wraps=store._collection.upsert)

        store.save("term frequency text", {})

        (entries,), _ = store._collection.upsert.call_args
        assert entries[0]["fields"]["sparse"] == {
            "indices": [1, 2],
            "values": [0.5, 0.5],
        }

    @pytest.mark.parametrize(
        "num_bytes,expect_truncated",
        [(8191, False), (8192, False), (8193, True)],
        ids=["under_limit", "exactly_at_limit", "one_over_limit"],
    )
    def test_truncate_text_boundary(self, num_bytes, expect_truncated):
        """_truncate_text() leaves text at/under 8192 bytes untouched, else trims it."""
        text = "a" * num_bytes
        result = _truncate_text(text)
        result_bytes = len(result.encode("utf-8"))
        if expect_truncated:
            assert result_bytes <= 8192
            assert len(result) < len(text)
        else:
            assert result == text

    def test_truncate_text_does_not_split_multibyte_char(self):
        """_truncate_text() must not split a multi-byte char at the byte boundary."""
        # "é" is 2 UTF-8 bytes; place one right at the 8192-byte boundary so a
        # naive byte-slice would cut it in half.
        text = "a" * 8191 + "é" * 10
        result = _truncate_text(text)
        # Must decode cleanly, with no UnicodeDecodeError and no stray
        # replacement artifacts from a split multi-byte sequence.
        assert isinstance(result, str)
        assert len(result.encode("utf-8")) <= 8192

    def test_save_propagates_upsert_errors(self, make_store):
        """save() must propagate a ConflictException from the underlying upsert."""
        store = make_store()
        store._collection.upsert = MagicMock(side_effect=ConflictException("dup id"))

        with pytest.raises(ConflictException):
            store.save("text", {})

    # ── search() ─────────────────────────────────────────────────────────────

    def test_search_single_field_skips_rerank(self, make_store, mocker):
        """search() on a single field skips rerank() and returns the matching result."""
        store = make_store()
        store.save("alpha document", {})
        rerank_mock = mocker.patch("crewai_endee.vector_store.endee_rerank")

        results = store.search("alpha", limit=3)

        rerank_mock.assert_not_called()
        assert len(results) == 1
        assert results[0]["content"] == "alpha document"

    def test_search_builds_correct_fields_dict(self, make_store):
        """search() forwards limit, ef_search, and filter to the collection search()."""
        store = make_store()
        store._collection.search = MagicMock(wraps=store._collection.search)

        store.search(
            "query text", limit=5, ef_search=256, filter=[{"lang": {"$eq": "en"}}]
        )

        _, kwargs = store._collection.search.call_args
        assert set(kwargs["fields"].keys()) == {"dense"}
        assert kwargs["fields"]["dense"]["limit"] == 5
        assert kwargs["ef_search"] == 256
        assert kwargs["filter"] == [{"lang": {"$eq": "en"}}]

    def test_search_omits_filter_and_ef_search_when_none(self, make_store):
        """search() omits filter and ef_search from the backend call when unset."""
        store = make_store()
        store._collection.search = MagicMock(wraps=store._collection.search)

        store.search("query text")

        _, kwargs = store._collection.search.call_args
        assert "filter" not in kwargs
        assert "ef_search" not in kwargs

    def test_search_omits_filter_when_empty_list(self, make_store):
        """An empty filter list is falsy, so it is also omitted (`if filter:`)."""
        store = make_store()
        store._collection.search = MagicMock(wraps=store._collection.search)

        store.search("query text", filter=[])

        _, kwargs = store._collection.search.call_args
        assert "filter" not in kwargs

    def test_search_forwards_prefilter_cardinality_threshold(self, make_store):
        """search() forwards prefilter_cardinality_threshold to the collection."""
        store = make_store()
        store._collection.search = MagicMock(wraps=store._collection.search)

        store.search("query text", prefilter_cardinality_threshold=500)

        _, kwargs = store._collection.search.call_args
        assert kwargs["prefilter_cardinality_threshold"] == 500

    def test_search_forwards_filter_boost_percentage(self, make_store):
        """search() must forward filter_boost_percentage to the collection's search."""
        store = make_store()
        store._collection.search = MagicMock(wraps=store._collection.search)

        store.search("query text", filter_boost_percentage=25)

        _, kwargs = store._collection.search.call_args
        assert kwargs["filter_boost_percentage"] == 25

    def test_search_omits_prefilter_and_boost_when_none(self, make_store):
        """prefilter_cardinality_threshold and boost_percentage default to unset."""
        store = make_store()
        store._collection.search = MagicMock(wraps=store._collection.search)

        store.search("query text")

        _, kwargs = store._collection.search.call_args
        assert "prefilter_cardinality_threshold" not in kwargs
        assert "filter_boost_percentage" not in kwargs

    def test_search_returns_empty_list_when_no_fields_have_results(self, make_store):
        """search() must return [] when raw_results['results'] is an empty dict."""
        store = make_store()
        store._collection.search = MagicMock(return_value={"results": {}})

        assert store.search("anything") == []

    def test_search_hybrid_triggers_rerank_even_when_one_field_is_empty(
        self, make_store, mocker
    ):
        """rerank() must trigger for two fields even if one field returns no hits."""
        # RRF merges results whenever there are 2+ fields, even if one has no hits.
        store = make_store(fields=[DENSE_FIELD, SPARSE_FIELD_BM25])
        store._sparse_embeddings.embed_query = MagicMock(
            return_value=type("SV", (), {"indices": [1], "values": [1.0]})()
        )
        raw_results = {
            "results": {
                "dense": [{"id": "a", "similarity": 0.9, "meta": {}}],
                "sparse": [],
            }
        }
        store._collection.search = MagicMock(return_value=raw_results)
        rerank_mock = mocker.patch(
            "crewai_endee.vector_store.endee_rerank",
            return_value={"results": [{"id": "a", "similarity": 0.9, "meta": {}}]},
        )

        results = store.search("hybrid query", limit=4)

        rerank_mock.assert_called_once()
        assert len(results) == 1

    def test_search_include_vectors_merges_sparses_and_multi_vectors(self, make_store):
        """include_vectors=True merges vectors/sparses/multi_vectors into results."""
        store = make_store()
        store.save("some content", {})
        sid = store._collection.get_objects(list(store._collection._store.keys()))[0][
            "id"
        ]
        store._collection.get_objects = MagicMock(
            return_value=[
                {
                    "id": sid,
                    "vectors": {"dense": [0.1] * 16},
                    "sparses": {"sparse": {"indices": [1], "values": [0.5]}},
                    "multi_vectors": {"chunks": [[0.1] * 16]},
                }
            ]
        )

        results = store.search("some content", limit=1, include_vectors=True)

        assert results[0]["vectors"] == {"dense": [0.1] * 16}
        assert results[0]["sparses"] == {"sparse": {"indices": [1], "values": [0.5]}}
        assert results[0]["multi_vectors"] == {"chunks": [[0.1] * 16]}

    def test_search_include_vectors_swallows_get_objects_errors(self, make_store):
        """search() swallows get_objects() errors, omitting the 'vectors' key."""
        store = make_store()
        store.save("some content", {})
        store._collection.get_objects = MagicMock(side_effect=RuntimeError("boom"))

        results = store.search("some content", limit=1, include_vectors=True)

        assert len(results) == 1
        assert "vectors" not in results[0]

    def test_search_multi_field_calls_rerank_with_expected_args(
        self, make_store, mocker
    ):
        """search() over dense/sparse fields calls rerank() with name/limit/weights."""
        store = make_store(fields=[DENSE_FIELD, SPARSE_FIELD_BM25])
        store._sparse_embeddings.embed_query = MagicMock(
            return_value=type("SV", (), {"indices": [1], "values": [1.0]})()
        )
        raw_results = {"results": {"dense": [], "sparse": []}}
        store._collection.search = MagicMock(return_value=raw_results)
        rerank_mock = mocker.patch(
            "crewai_endee.vector_store.endee_rerank",
            return_value={"results": []},
        )

        store.search(
            "hybrid query",
            limit=4,
            field_weights={"dense": 0.4, "sparse": 0.6},
            rrf_k=30,
        )

        rerank_mock.assert_called_once_with(
            raw_results,
            name="rrf",
            limit=4,
            field_weights={"dense": 0.4, "sparse": 0.6},
            rrf_k=30,
        )

    def test_search_score_threshold_filters_results(self, make_store):
        """search() with score_threshold must return only results at or above it."""
        store = make_store()
        store.save("close match", {})
        store.save("far match", {})

        results = store.search("close match", limit=5, score_threshold=0.999)
        assert all(r["score"] >= 0.999 for r in results)

    def test_search_include_vectors_calls_get_objects(self, make_store):
        """search() with include_vectors=True calls get_objects() for full vectors."""
        store = make_store()
        store.save("some content", {})
        store._collection.get_objects = MagicMock(wraps=store._collection.get_objects)

        results = store.search("some content", limit=1, include_vectors=True)

        store._collection.get_objects.assert_called_once()
        assert "vectors" in results[0] or results == []

    def test_search_returns_empty_list_on_backend_error(self, make_store):
        """search() must return [] instead of raising when the backend search errors."""
        store = make_store()
        store._collection.search = MagicMock(side_effect=RuntimeError("boom"))
        assert store.search("anything") == []

    # ── single-object helpers ────────────────────────────────────────────────

    def test_get_objects_passes_through(self, make_store):
        """get_objects() passes the id list through unchanged and returns the result."""
        store = make_store()
        store._collection.get_objects = MagicMock(return_value=[{"id": "a"}])
        result = store.get_objects(["a"])
        store._collection.get_objects.assert_called_once_with(["a"])
        assert result == [{"id": "a"}]

    def test_get_vector_returns_empty_dict_when_missing(self, make_store):
        """get_vector() must return {} when the id is not found in the collection."""
        store = make_store()
        store._collection.get_objects = MagicMock(return_value=[])
        assert store.get_vector("missing-id") == {}

    def test_get_vector_returns_object_when_found(self, make_store):
        """get_vector() must return the matching object when its id is found."""
        store = make_store()
        store._collection.get_objects = MagicMock(return_value=[{"id": "a"}])
        assert store.get_vector("a") == {"id": "a"}

    def test_delete_vector_passes_through(self, make_store):
        """delete_vector() must pass the id to delete_object() and return the result."""
        store = make_store()
        store._collection.delete_object = MagicMock(return_value={"deleted": "a"})
        result = store.delete_vector("a")
        store._collection.delete_object.assert_called_once_with("a")
        assert result == {"deleted": "a"}

    # ── delete() ─────────────────────────────────────────────────────────────

    def test_delete_wraps_single_dict_filter_in_list(self, make_store):
        """delete() wraps a single filter dict in a list before delete_by_filter."""
        store = make_store()
        store._collection.delete_by_filter = MagicMock(return_value={"deleted": 0})
        store.delete({"lang": {"$eq": "en"}})
        store._collection.delete_by_filter.assert_called_once_with(
            [{"lang": {"$eq": "en"}}]
        )

    def test_delete_accepts_list_of_filters_unchanged(self, make_store):
        """delete() passes an already-list filter to delete_by_filter unchanged."""
        store = make_store()
        store._collection.delete_by_filter = MagicMock(return_value={"deleted": 0})
        filters = [{"lang": {"$eq": "en"}}, {"category": {"$in": ["a", "b"]}}]
        store.delete(filters)
        store._collection.delete_by_filter.assert_called_once_with(filters)

    # ── update_filters / reset / describe ───────────────────────────────────

    def test_update_filters_passes_through(self, make_store):
        """update_filters() passes updates to the collection and returns its result."""
        store = make_store()
        store._collection.update_filters = MagicMock(return_value={"updated": 1})
        updates = [{"id": "a", "filter": {"reviewed": "yes"}}]
        result = store.update_filters(updates)
        store._collection.update_filters.assert_called_once_with(updates=updates)
        assert result == {"updated": 1}

    def test_reset_deletes_collection_and_clears_state(self, make_store, mocker):
        """reset() must delete the collection via the client and clear _collection."""
        store = make_store()
        delete_spy = mocker.spy(store._client, "delete_collection")
        store.reset()
        delete_spy.assert_called_once_with("test_collection")
        assert store._collection is None

    def test_reset_swallows_delete_errors(self, make_store):
        """reset() swallows a NotFoundException from delete_collection, clears state."""
        store = make_store()
        store._client.delete_collection = MagicMock(
            side_effect=NotFoundException("gone")
        )
        store.reset()  # must not raise
        assert store._collection is None

    def test_describe_passes_through(self, make_store):
        """describe() returns the collection's description, including its name."""
        store = make_store()
        info = store.describe()
        assert info["name"] == "test_collection"

    # ── close() ──────────────────────────────────────────────────────────────

    def test_close_calls_close_session_when_available(self, make_store):
        """close() must call close_session() on the client when available."""
        store = make_store()
        store._client.close_session = MagicMock()
        store.close()
        store._client.close_session.assert_called_once()

    def test_close_falls_back_to_close_client(self, make_store):
        """close() falls back to close_client() when it lacks close_session()."""
        store = make_store()

        class ClientWithOnlyCloseClient:
            def __init__(self):
                self.close_client = MagicMock()

        store._client = ClientWithOnlyCloseClient()
        store.close()
        store._client.close_client.assert_called_once()

    def test_close_no_exception_when_neither_method_exists(self, make_store):
        """close() must not raise when the client lacks close_session/close_client."""
        store = make_store()

        class BareClient:
            pass

        store._client = BareClient()
        store.close()  # must not raise

    def test_close_is_noop_when_client_is_none(self, make_store):
        """close() returns early, leaving _collection untouched when _client is None."""
        store = make_store()
        store._client = None
        store._collection = "sentinel"
        store.close()  # must not raise
        # The None-client branch returns early, so _collection is untouched.
        assert store._collection == "sentinel"

    # ── sparse auto-detection ────────────────────────────────────────────────

    def test_auto_setup_sparse_creates_endee_model_sparse(self, make_store, mocker):
        """A bm25 field with no sparse_embedding auto-creates EndeeModelSparse."""
        mock_sparse_cls = mocker.patch(
            "crewai_endee.sparse_embeddings.EndeeModelSparse"
        )
        store = make_store(fields=[DENSE_FIELD, SPARSE_FIELD_BM25])
        mock_sparse_cls.assert_called_once()
        assert store._sparse_embeddings is mock_sparse_cls.return_value

    def test_explicit_sparse_embedding_skips_auto_setup(self, make_store, mocker):
        """Passing an explicit sparse_embedding skips auto-creating EndeeModelSparse."""
        mock_sparse_cls = mocker.patch(
            "crewai_endee.sparse_embeddings.EndeeModelSparse"
        )
        # duck-typed model: has .embed()/.query_embed() so wrap_sparse_model()
        # accepts it via SparseModelAdapter instead of needing EndeeModelSparse.
        explicit_model = MagicMock()
        explicit_model.embed = MagicMock()
        explicit_model.query_embed = MagicMock()

        store = make_store(
            fields=[DENSE_FIELD, SPARSE_FIELD_BM25], sparse_embedding=explicit_model
        )

        mock_sparse_cls.assert_not_called()
        assert store._sparse_embeddings is not None
        assert store._sparse_embeddings._model is explicit_model

    # ── network failure propagation ─────────────────────────────────────────

    def test_init_propagates_connection_error_from_list_collections(
        self, mocker, fake_embedder
    ):
        """Constructing a store propagates a ConnectionError from list_collections()."""
        mocker.patch(
            "crewai_endee.vector_store.build_embedder", return_value=fake_embedder
        )
        broken_client = MagicMock()
        broken_client.list_collections = MagicMock(
            side_effect=ConnectionError("could not connect to Endee server")
        )

        with pytest.raises(ConnectionError):
            EndeeVectorStore(
                type="unreachable",
                embedder_config={},
                fields=[DENSE_FIELD],
                endee_client=broken_client,
            )

    def test_init_propagates_connection_error_from_create_collection(
        self, mocker, fake_embedder
    ):
        """Constructing a store propagates a ConnectionError from create_collection."""
        mocker.patch(
            "crewai_endee.vector_store.build_embedder", return_value=fake_embedder
        )
        broken_client = MagicMock()
        broken_client.list_collections = MagicMock(return_value=[])
        broken_client.create_collection = MagicMock(
            side_effect=ConnectionError("could not connect to Endee server")
        )

        with pytest.raises(ConnectionError):
            EndeeVectorStore(
                type="unreachable",
                embedder_config={},
                fields=[DENSE_FIELD],
                endee_client=broken_client,
            )

    # ── multi-field API ──────────────────────────────────────────────────────

    def test_add_objects_passes_through_to_upsert(self, make_store):
        """add_objects() passes the object list to upsert() unchanged, returns it."""
        store = make_store()
        store._collection.upsert = MagicMock(return_value={"upserted": 1})
        objects = [
            {
                "id": uuid.uuid4().hex,
                "meta": {},
                "filter": {},
                "fields": {"dense": [0.0] * 16},
            }
        ]
        result = store.add_objects(objects)
        store._collection.upsert.assert_called_once_with(objects)
        assert result == {"upserted": 1}

    def test_multi_field_search_passes_through_to_search(self, make_store):
        """multi_field_search() forwards the fields dict to the collection search()."""
        store = make_store()
        store._collection.search = MagicMock(wraps=store._collection.search)
        fields = {"dense": {"query": [0.0] * 16, "limit": 3}}
        store.multi_field_search(fields=fields)
        _, kwargs = store._collection.search.call_args
        assert kwargs["fields"] == fields

    def test_multi_field_search_uses_default_ef_search_constant(self, make_store):
        """multi_field_search() defaults ef_search to constants.DEFAULT_EF_SEARCH."""
        from endee import constants

        store = make_store()
        store._collection.search = MagicMock(wraps=store._collection.search)
        store.multi_field_search(fields={"dense": {"query": [0.0] * 16, "limit": 3}})
        _, kwargs = store._collection.search.call_args
        assert kwargs["ef_search"] == constants.DEFAULT_EF_SEARCH

    def test_multi_field_search_forwards_filter_and_custom_ef_search(self, make_store):
        """multi_field_search() forwards an explicit filter and ef_search to search."""
        store = make_store()
        store._collection.search = MagicMock(wraps=store._collection.search)
        filt = [{"lang": {"$eq": "en"}}]
        store.multi_field_search(
            fields={"dense": {"query": [0.0] * 16, "limit": 3}},
            filter=filt,
            ef_search=512,
        )
        _, kwargs = store._collection.search.call_args
        assert kwargs["filter"] == filt
        assert kwargs["ef_search"] == 512

    def test_multi_field_search_omits_filter_when_none(self, make_store):
        """multi_field_search() omits filter from the backend search() when unset."""
        store = make_store()
        store._collection.search = MagicMock(wraps=store._collection.search)
        store.multi_field_search(fields={"dense": {"query": [0.0] * 16, "limit": 3}})
        _, kwargs = store._collection.search.call_args
        assert "filter" not in kwargs

    # ── internal helpers ─────────────────────────────────────────────────────

    @pytest.mark.parametrize(
        "role,expected",
        [
            ("Admin User", "admin_user"),
            ("read/write", "read_write"),
            ("Ops/Dev Team", "ops_dev_team"),
        ],
    )
    def test_sanitize_role_normalizes_spaces_slashes_and_case(
        self, make_store, role, expected
    ):
        """_sanitize_role() lowercases and replaces spaces/slashes with underscores."""
        store = make_store()
        assert store._sanitize_role(role) == expected

    def test_public_api_surface_imports_cleanly(self):
        """Every name in crewai_endee.__all__ must be importable from the package."""
        import crewai_endee

        for name in crewai_endee.__all__:
            assert hasattr(crewai_endee, name)


# ═══════════════════════════════════════════════════════════════════════════
# Filters
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestFiltersUnit:
    # ── delete() filter-list normalization ──────────────────────────────────

    def test_delete_wraps_single_dict_in_list(self, make_store):
        """delete() wraps a single filter dict in a list before delete_by_filter()."""
        store = make_store()
        store._collection.delete_by_filter = MagicMock(return_value={"deleted": 0})

        store.delete({"category": {"$eq": "systems"}})

        called_arg = store._collection.delete_by_filter.call_args[0][0]
        assert isinstance(called_arg, list)
        assert called_arg == [{"category": {"$eq": "systems"}}]

    def test_delete_leaves_list_of_dicts_unchanged(self, make_store):
        """delete() passes an already-list filter to delete_by_filter() unchanged."""
        store = make_store()
        store._collection.delete_by_filter = MagicMock(return_value={"deleted": 0})
        filters = [
            {"category": {"$eq": "systems"}},
            {"lang": {"$in": ["Go", "Rust"]}},
        ]

        store.delete(filters)

        called_arg = store._collection.delete_by_filter.call_args[0][0]
        assert called_arg is filters

    def test_delete_by_filter_removes_only_matching_entries(self, make_store):
        """delete() with an $eq filter removes only the matching entry, keeps rest."""
        store = make_store()
        store.save("go doc", {"category": "systems"})
        store.save("scripting doc", {"category": "scripting"})

        result = store.delete({"category": {"$eq": "scripting"}})

        assert result["deleted"] == 1
        remaining = store._collection.describe()["num_objects"]
        assert remaining == 1

    # ── filter_data construction inside save() ──────────────────────────────

    def test_save_promotes_only_primitive_metadata_to_filter(self, make_store):
        """save() promotes str/int/float/bool metadata to filter, not dict/list/None."""
        store = make_store()
        store._collection.upsert = MagicMock(wraps=store._collection.upsert)

        metadata = {
            "str_val": "a",
            "int_val": 1,
            "float_val": 1.5,
            "bool_ignored_note": True,  # bool is an int subclass -> primitive
            "dict_val": {"nested": True},
            "list_val": [1, 2, 3],
            "none_val": None,
        }
        store.save("text", metadata)

        (entries,), _ = store._collection.upsert.call_args
        filter_data = entries[0]["filter"]

        assert filter_data["str_val"] == "a"
        assert filter_data["int_val"] == 1
        assert filter_data["float_val"] == 1.5
        assert "dict_val" not in filter_data
        assert "list_val" not in filter_data
        assert "none_val" not in filter_data

    def test_save_with_empty_metadata_produces_empty_filter(self, make_store):
        """save() with no metadata produces an empty filter dict on the entry."""
        store = make_store()
        store._collection.upsert = MagicMock(wraps=store._collection.upsert)

        store.save("text", {})

        (entries,), _ = store._collection.upsert.call_args
        assert entries[0]["filter"] == {}

    @pytest.mark.parametrize("op", ["$eq", "$in"])
    def test_supported_operators_match_correctly(self, make_store, op):
        """delete() with $eq and $in operators match and delete only the target."""
        store = make_store()
        store.save("systems doc", {"category": "systems"})
        store.save("scripting doc", {"category": "scripting"})

        value = "systems" if op == "$eq" else ["systems"]
        result = store.delete({"category": {op: value}})
        assert result["deleted"] == 1


# ═══════════════════════════════════════════════════════════════════════════
# Sparse
# ═══════════════════════════════════════════════════════════════════════════


class _FakeSparseResult:
    """Stand-in for .embed()/.query_embed() results with .indices/.values.tolist()."""

    def __init__(self, indices, values):
        self.indices = np.array(indices)
        self.values = np.array(values)


@contextlib.contextmanager
def _mock_sparse_model():
    """Patch endee_model.SparseModel for the duration of the with block."""
    from unittest.mock import patch

    with patch("endee_model.SparseModel") as mock_cls:
        yield mock_cls


@pytest.mark.unit
class TestSparseUnit:
    # ── SparseVector ─────────────────────────────────────────────────────────

    def test_sparse_vector_holds_indices_and_values(self):
        """SparseVector must store the indices and values passed to its constructor."""
        sv = SparseVector(indices=[1, 2, 3], values=[0.1, 0.2, 0.3])
        assert sv.indices == [1, 2, 3]
        assert sv.values == [0.1, 0.2, 0.3]

    # ── SparseModelAdapter ───────────────────────────────────────────────────

    def test_sparse_model_adapter_embed_documents(self):
        """SparseModelAdapter.embed_documents() calls embed(), returns SparseVectors."""
        model = MagicMock()
        model.embed = MagicMock(return_value=[_FakeSparseResult([1, 5], [0.9, 0.1])])
        adapter = SparseModelAdapter(model)

        result = adapter.embed_documents(["some text"])

        model.embed.assert_called_once_with(["some text"])
        assert len(result) == 1
        assert isinstance(result[0], SparseVector)
        assert result[0].indices == [1, 5]
        assert result[0].values == [0.9, 0.1]

    def test_sparse_model_adapter_embed_query(self):
        """SparseModelAdapter.embed_query() wraps query_embed() into a SparseVector."""
        model = MagicMock()
        model.query_embed = MagicMock(
            return_value=iter([_FakeSparseResult([2], [1.0])])
        )
        adapter = SparseModelAdapter(model)

        result = adapter.embed_query("a query")

        model.query_embed.assert_called_once_with("a query")
        assert isinstance(result, SparseVector)
        assert result.indices == [2]
        assert result.values == [1.0]

    # ── wrap_sparse_model ────────────────────────────────────────────────────

    def test_wrap_sparse_model_passthrough_for_sparse_embeddings_instance(self):
        """wrap_sparse_model() must return a SparseEmbeddings instance unchanged."""

        class MySparse(SparseEmbeddings):
            def embed_documents(self, texts):
                return []

            def embed_query(self, text):
                return SparseVector(indices=[], values=[])

        instance = MySparse()
        assert wrap_sparse_model(instance) is instance

    def test_wrap_sparse_model_wraps_duck_typed_model(self):
        """wrap_sparse_model() wraps a duck-typed embed/query_embed model as adapter."""
        model = MagicMock()
        model.embed = MagicMock()
        model.query_embed = MagicMock()

        wrapped = wrap_sparse_model(model)

        assert isinstance(wrapped, SparseModelAdapter)
        assert wrapped._model is model

    def test_wrap_sparse_model_raises_type_error_for_unsupported_object(self):
        """wrap_sparse_model() raises TypeError for an object matching neither model."""
        with pytest.raises(TypeError, match="SparseEmbeddings"):
            wrap_sparse_model(object())

    # ── EndeeModelSparse ─────────────────────────────────────────────────────

    def test_endee_model_sparse_requires_endee_model_package(self, mocker):
        """EndeeModelSparse() raises ImportError when endee_model is unavailable."""
        mocker.patch.dict("sys.modules", {"endee_model": None})
        with pytest.raises(ImportError, match="endee_model"):
            EndeeModelSparse()

    def test_endee_model_sparse_embed_documents(self):
        """EndeeModelSparse.embed_documents() returns SparseVectors from the output."""
        pytest.importorskip("endee_model")
        with _mock_sparse_model() as mock_sparse_model_cls:
            mock_sparse_model_cls.return_value.embed.return_value = [
                _FakeSparseResult([1, 2], [0.5, 0.5])
            ]
            sparse = EndeeModelSparse()
            result = sparse.embed_documents(["doc"])
            assert result[0].indices == [1, 2]

    def test_endee_model_sparse_embed_query(self):
        """EndeeModelSparse.embed_query() returns a SparseVector from model output."""
        pytest.importorskip("endee_model")
        with _mock_sparse_model() as mock_sparse_model_cls:
            mock_sparse_model_cls.return_value.query_embed.return_value = iter(
                [_FakeSparseResult([7], [1.0])]
            )
            sparse = EndeeModelSparse()
            result = sparse.embed_query("a query")
            assert isinstance(result, SparseVector)
            assert result.indices == [7]
            assert result.values == [1.0]

    def test_endee_model_sparse_forwards_constructor_kwargs(self):
        """EndeeModelSparse() forwards its constructor kwargs to the SparseModel."""
        pytest.importorskip("endee_model")
        with _mock_sparse_model() as mock_sparse_model_cls:
            EndeeModelSparse(
                model_name="custom/bm25",
                k=1.5,
                b=0.8,
                avg_len=100.0,
                language="french",
                cache_dir="/tmp/cache",
                extra_option=True,
            )
            mock_sparse_model_cls.assert_called_once_with(
                model_name="custom/bm25",
                cache_dir="/tmp/cache",
                k=1.5,
                b=0.8,
                avg_len=100.0,
                language="french",
                extra_option=True,
            )

    # ── auto-detection wiring (via EndeeVectorStore) ────────────────────────

    def test_auto_setup_sparse_for_endee_bm25_field(self, make_store, mocker):
        """A bm25 field with no sparse_embedding auto-creates an EndeeModelSparse."""
        mock_cls = mocker.patch("crewai_endee.sparse_embeddings.EndeeModelSparse")
        store = make_store(fields=[DENSE_FIELD, SPARSE_FIELD_BM25])
        mock_cls.assert_called_once()
        assert store._sparse_embeddings is mock_cls.return_value

    def test_auto_setup_skipped_when_sparse_embedding_explicit(
        self, make_store, mocker
    ):
        """Passing an explicit sparse_embedding skips auto-creating EndeeModelSparse."""
        mock_cls = mocker.patch("crewai_endee.sparse_embeddings.EndeeModelSparse")
        explicit_model = MagicMock()
        explicit_model.embed = MagicMock()
        explicit_model.query_embed = MagicMock()

        store = make_store(
            fields=[DENSE_FIELD, SPARSE_FIELD_BM25], sparse_embedding=explicit_model
        )

        mock_cls.assert_not_called()
        assert store._sparse_embeddings._model is explicit_model

    def test_no_auto_setup_for_non_bm25_sparse_field(self, make_store, mocker):
        """A non-bm25 (default) sparse field skips EndeeModelSparse auto-creation."""
        mock_cls = mocker.patch("crewai_endee.sparse_embeddings.EndeeModelSparse")
        default_sparse_field = {
            "name": "sparse",
            "type": "sparse",
            "sparse_model": "default",
        }
        store = make_store(fields=[DENSE_FIELD, default_sparse_field])
        mock_cls.assert_not_called()
        assert store._sparse_embeddings is None

    def test_multi_sparse_fields_auto_setup_checks_the_pinned_field_only(
        self, make_store, mocker
    ):
        """Auto-setup must only trigger for the field pinned as sparse_field_name."""
        mock_cls = mocker.patch("crewai_endee.sparse_embeddings.EndeeModelSparse")
        sparse_default_first = {
            "name": "sparse_default",
            "type": "sparse",
            "sparse_model": "default",
        }
        sparse_bm25_second = {
            "name": "sparse_bm25",
            "type": "sparse",
            "sparse_model": "endee_bm25",
        }

        store = make_store(
            fields=[DENSE_FIELD, sparse_default_first, sparse_bm25_second]
        )

        # sparse_field_name is pinned to the first sparse field seen...
        assert store.sparse_field_name == "sparse_default"
        # ...and auto-setup does not fire, since that field is not endee_bm25,
        # even though another field in the collection is.
        mock_cls.assert_not_called()
        assert store._sparse_embeddings is None

    def test_multi_sparse_fields_auto_setup_fires_when_first_field_is_bm25(
        self, make_store, mocker
    ):
        """Auto-setup must trigger when the pinned first sparse field is bm25."""
        mock_cls = mocker.patch("crewai_endee.sparse_embeddings.EndeeModelSparse")
        sparse_bm25_first = {
            "name": "sparse_bm25",
            "type": "sparse",
            "sparse_model": "endee_bm25",
        }
        sparse_default_second = {
            "name": "sparse_default",
            "type": "sparse",
            "sparse_model": "default",
        }

        store = make_store(
            fields=[DENSE_FIELD, sparse_bm25_first, sparse_default_second]
        )

        assert store.sparse_field_name == "sparse_bm25"
        mock_cls.assert_called_once()
        assert store._sparse_embeddings is mock_cls.return_value

    def test_auto_setup_sparse_detects_nested_params_sparse_model(
        self, make_store, mocker
    ):
        """Auto-setup must detect sparse_model nested under a field's params too."""
        mock_cls = mocker.patch("crewai_endee.sparse_embeddings.EndeeModelSparse")
        store = make_store(fields=[DENSE_FIELD, SPARSE_FIELD_BM25])
        store._sparse_embeddings = None
        mock_cls.reset_mock()
        nested_collection = MagicMock()
        sparse_field = {
            "name": "sparse",
            "type": "sparse",
            "params": {"sparse_model": "endee_bm25"},
        }
        nested_collection.fields = [{"name": "dense", "type": "vector"}, sparse_field]

        store._auto_setup_sparse(nested_collection)

        mock_cls.assert_called_once()
        assert store._sparse_embeddings is mock_cls.return_value

    # ── hybrid search RRF wiring ─────────────────────────────────────────────

    def test_hybrid_search_forwards_field_weights_and_rrf_k(self, make_store, mocker):
        """search() over dense/sparse fields calls rerank() with field_weights/rrf_k."""
        store = make_store(fields=[DENSE_FIELD, SPARSE_FIELD_BM25])
        store._sparse_embeddings.embed_query = MagicMock(
            return_value=SparseVector(indices=[1], values=[1.0])
        )
        raw_results = {"results": {"dense": [], "sparse": []}}
        store._collection.search = MagicMock(return_value=raw_results)
        rerank_mock = mocker.patch(
            "crewai_endee.vector_store.endee_rerank",
            return_value={"results": []},
        )

        store.search(
            "query", limit=2, field_weights={"dense": 0.5, "sparse": 0.5}, rrf_k=42
        )

        rerank_mock.assert_called_once_with(
            raw_results,
            name="rrf",
            limit=2,
            field_weights={"dense": 0.5, "sparse": 0.5},
            rrf_k=42,
        )

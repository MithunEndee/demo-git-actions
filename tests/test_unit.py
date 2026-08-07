"""Unit tests: all tests that mock the `endee` client (no network).

`TestVectorStoreUnit` covers core CRUD on `EndeeVectorStore`
(init/create-or-reuse, add incl. node deduplication, query, delete, clear,
describe/fetch, constants fallback).

`TestFiltersUnit` covers filter/metadata-key translation and operator
support.

`TestSparseUnit` covers sparse/hybrid embeddings, auto-detection, and RRF
wiring.
"""

from __future__ import annotations

import importlib.util
from unittest.mock import MagicMock, patch

import pytest
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.core.vector_stores.types import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
)

from llama_index_endee.base import EndeeVectorStore
from llama_index_endee.constants import DEFAULT_BATCH_SIZE, MAX_VECTORS_PER_BATCH
from llama_index_endee.sparse_embeddings import (
    SparseEmbeddings,
    SparseModelAdapter,
    SparseVector,
    wrap_sparse_model,
)

DIMENSION = 4


# ═══════════════════════════════════════════════════════════════════════════
# Vector store unit tests
# ═══════════════════════════════════════════════════════════════════════════


def _mock_vector_store_collection(dimension=DIMENSION):
    collection = MagicMock()
    collection.fields = [
        {
            "name": "dense",
            "type": "vector",
            "params": {
                "dimension": dimension,
                "space_type": "cosine",
                "precision": "int8",
            },
        }
    ]
    collection.search.return_value = {"results": {"dense": []}}
    collection.describe.return_value = {
        "name": "test",
        "fields": collection.fields,
        "num_objects": 0,
    }
    return collection


@pytest.mark.unit
class TestVectorStoreUnit:
    # ── __init__ / from_params / collection creation ────────────────────────

    def test_from_params_creates_collection_when_absent(self, mock_endee_client):
        """from_params must create a new collection when none exists yet."""
        vs = EndeeVectorStore.from_params(
            collection_name="new_coll",
            dimension=DIMENSION,
            space_type="cosine",
        )
        assert vs.collection_name == "new_coll"
        assert "new_coll" in mock_endee_client._collections

    def test_init_reuses_existing_collection(self, mock_endee_client):
        """A second store on an existing collection must reuse it, not recreate."""
        vs1 = EndeeVectorStore.from_params(
            collection_name="reuse_coll", dimension=DIMENSION
        )
        vs1.add([TextNode(text="seed", embedding=[0.1, 0.2, 0.3, 0.4], id_="seed")])

        # Second store instance connects to the same (now populated) collection.
        vs2 = EndeeVectorStore.from_params(
            collection_name="reuse_coll", dimension=DIMENSION
        )
        assert vs2.describe()["num_objects"] == 1

    def test_batch_size_defaults_to_default_batch_size_constant(
        self, mock_endee_client
    ):
        """A store with no explicit batch_size must use DEFAULT_BATCH_SIZE."""
        vs = EndeeVectorStore.from_params(
            collection_name="default_batch", dimension=DIMENSION
        )
        assert vs.batch_size == DEFAULT_BATCH_SIZE
        assert DEFAULT_BATCH_SIZE < MAX_VECTORS_PER_BATCH

    def test_force_recreate_deletes_existing_collection(self, mock_endee_client):
        """force_recreate=True deletes and recreates the collection, wiping its data."""
        vs1 = EndeeVectorStore.from_params(
            collection_name="fr_coll", dimension=DIMENSION
        )
        vs1.add([TextNode(text="seed", embedding=[0.1, 0.2, 0.3, 0.4], id_="seed")])
        assert vs1.describe()["num_objects"] == 1

        vs2 = EndeeVectorStore.from_params(
            collection_name="fr_coll", dimension=DIMENSION, force_recreate=True
        )
        # Recreated collection is empty again.
        assert vs2.describe()["num_objects"] == 0

    def test_endee_client_override_skips_client_construction(self, mock_endee_client):
        """An explicit endee_client must skip constructing a new Endee() client."""
        custom_client = MagicMock()
        custom_client.list_collections.return_value = []
        custom_client.get_collection.return_value = _mock_vector_store_collection()

        EndeeVectorStore.from_params(
            collection_name="coll",
            dimension=DIMENSION,
            endee_client=custom_client,
        )
        custom_client.create_collection.assert_called_once()
        # Confirms endee_client= was used, not a new client from the patched Endee().
        assert mock_endee_client._collections == {}

    def test_endee_collection_override_skips_collection_creation(
        self, mock_endee_client
    ):
        """An explicit endee_collection must skip any collection lookup or creation."""
        collection = _mock_vector_store_collection()
        EndeeVectorStore.from_params(
            collection_name="unused_name",
            dimension=DIMENSION,
            endee_collection=collection,
        )
        # No collection lookup/creation should have happened against the client.
        assert mock_endee_client._collections == {}

    def test_network_failure_during_connect_propagates(self):
        """A connection error from list_collections() must propagate out of __init__."""
        broken_client = MagicMock()
        broken_client.list_collections.side_effect = ConnectionError("no route to host")

        with pytest.raises(ConnectionError):
            EndeeVectorStore.from_params(
                collection_name="doesnt_matter",
                dimension=DIMENSION,
                endee_client=broken_client,
            )

    # ── add(): deduplicate by node_id ───────────────────────────────────────

    def test_add_dedups_by_node_id_last_wins(self, mock_endee_client):
        """add() deduplicates nodes sharing a node_id, keeping the last content."""
        vs = EndeeVectorStore.from_params(
            collection_name="dedup_coll", dimension=DIMENSION
        )
        first = TextNode(text="first version", embedding=[0.1, 0, 0, 0], id_="dup")
        second = TextNode(text="second version", embedding=[0, 0.1, 0, 0], id_="dup")

        ids = vs.add([first, second])

        assert ids == ["dup"]
        stored = vs._collection._store["dup"]
        assert stored["meta"]["_node_content"]
        # The LAST occurrence's embedding/content must win.
        assert stored["fields"]["dense"] == [0, 0.1, 0, 0]

    def test_add_with_empty_node_list_returns_empty_list(self, mock_endee_client):
        """add() with an empty node list must return an empty list and store nothing."""
        vs = EndeeVectorStore.from_params(
            collection_name="empty_add_coll", dimension=DIMENSION
        )
        ids = vs.add([])
        assert ids == []
        assert vs._collection._store == {}

    def test_add_dedups_three_nodes_sharing_same_node_id_last_wins(
        self, mock_endee_client
    ):
        """Three nodes sharing a node_id: the last one wins, one entry stored."""
        vs = EndeeVectorStore.from_params(
            collection_name="triple_dup_coll", dimension=DIMENSION
        )
        first = TextNode(text="v1", embedding=[1, 0, 0, 0], id_="dup")
        second = TextNode(text="v2", embedding=[0, 1, 0, 0], id_="dup")
        third = TextNode(text="v3", embedding=[0, 0, 1, 0], id_="dup")

        ids = vs.add([first, second, third])

        assert ids == ["dup"]
        stored = vs._collection._store["dup"]
        assert stored["fields"]["dense"] == [0, 0, 1, 0]

    def test_add_dedup_multiple_duplicates_interspersed_with_distinct_nodes(
        self, mock_endee_client
    ):
        """A repeated id keeps its last content at the earliest relative position."""
        # id1 appears 3 times; its content should come from the last one (E).
        vs = EndeeVectorStore.from_params(
            collection_name="multi_dup_interspersed_coll", dimension=DIMENSION
        )
        a = TextNode(text="A", embedding=[1, 0, 0, 0], id_="id1")
        b = TextNode(text="B", embedding=[0, 1, 0, 0], id_="id2")
        c = TextNode(text="C", embedding=[0, 0, 1, 0], id_="id1")
        d = TextNode(text="D", embedding=[0, 0, 0, 1], id_="id3")
        e = TextNode(text="E", embedding=[1, 1, 1, 1], id_="id1")

        ids = vs.add([a, b, c, d, e])

        assert ids == ["id2", "id3", "id1"]
        assert vs._collection._store["id1"]["fields"]["dense"] == [1, 1, 1, 1]
        assert vs._collection._store["id2"]["fields"]["dense"] == [0, 1, 0, 0]
        assert vs._collection._store["id3"]["fields"]["dense"] == [0, 0, 0, 1]
        assert len(vs._collection._store) == 3

    def test_add_dedup_preserves_last_occurrence_position(self, mock_endee_client):
        """Deduplication keeps the last occurrence's content at the earlier position."""
        vs = EndeeVectorStore.from_params(
            collection_name="dedup_order_coll", dimension=DIMENSION
        )
        a = TextNode(text="A", embedding=[1, 0, 0, 0], id_="id1")
        b = TextNode(text="B", embedding=[0, 1, 0, 0], id_="id2")
        c = TextNode(text="C", embedding=[0, 0, 1, 0], id_="id1")

        ids = vs.add([a, b, c])

        # seen = {"id1": 2, "id2": 1} -> sorted(values) = [1, 2]
        # -> [nodes[1], nodes[2]] = [b, c]
        assert ids == ["id2", "id1"]
        assert vs._collection._store["id1"]["fields"]["dense"] == [0, 0, 1, 0]
        assert vs._collection._store["id2"]["fields"]["dense"] == [0, 1, 0, 0]

    # ── delete / delete_vector / clear ──────────────────────────────────────

    def test_delete_calls_delete_by_filter_with_ref_doc_id(self, mock_endee_client):
        """delete(ref_doc_id) calls delete_by_filter with an $eq filter on it."""
        collection = _mock_vector_store_collection()
        vs = EndeeVectorStore(
            endee_collection=collection, collection_name="test", dimension=DIMENSION
        )
        vs.delete("doc123")
        collection.delete_by_filter.assert_called_once_with(
            [{"ref_doc_id": {"$eq": "doc123"}}]
        )

    def test_delete_vector_calls_delete_object(self, mock_endee_client):
        """delete_vector() calls delete_object with the id, returning its result."""
        collection = _mock_vector_store_collection()
        collection.delete_object.return_value = {"deleted": 1}
        vs = EndeeVectorStore(
            endee_collection=collection, collection_name="test", dimension=DIMENSION
        )
        result = vs.delete_vector("vec1")
        collection.delete_object.assert_called_once_with("vec1")
        assert result == {"deleted": 1}

    def test_clear_deletes_the_collection(self, mock_endee_client):
        """clear() must delete the underlying collection."""
        vs = EndeeVectorStore.from_params(
            collection_name="clear_coll", dimension=DIMENSION
        )
        assert "clear_coll" in mock_endee_client._collections
        vs.clear()
        assert "clear_coll" not in mock_endee_client._collections

    # ── describe()/fetch(): pass-through and graceful fallback on error ─────

    def test_describe_passes_through_collection_metadata(self, mock_endee_client):
        """describe() must return the collection's describe() output unchanged."""
        collection = _mock_vector_store_collection()
        vs = EndeeVectorStore(
            endee_collection=collection, collection_name="test", dimension=DIMENSION
        )
        info = vs.describe()
        assert info == collection.describe.return_value

    def test_describe_swallows_exception_and_returns_empty_dict(
        self, mock_endee_client
    ):
        """describe() must swallow collection exceptions and return an empty dict."""
        collection = _mock_vector_store_collection()
        collection.describe.side_effect = RuntimeError("boom")
        vs = EndeeVectorStore(
            endee_collection=collection, collection_name="test", dimension=DIMENSION
        )
        assert vs.describe() == {}

    def test_fetch_passes_through_collection_objects(self, mock_endee_client):
        """fetch() must return collection.get_objects()'s output unchanged."""
        collection = _mock_vector_store_collection()
        collection.get_objects.return_value = [{"id": "a"}]
        vs = EndeeVectorStore(
            endee_collection=collection, collection_name="test", dimension=DIMENSION
        )
        assert vs.fetch(["a"]) == [{"id": "a"}]
        collection.get_objects.assert_called_once_with(["a"])

    def test_fetch_swallows_exception_and_returns_empty_list(self, mock_endee_client):
        """fetch() must swallow collection exceptions and return an empty list."""
        collection = _mock_vector_store_collection()
        collection.get_objects.side_effect = RuntimeError("boom")
        vs = EndeeVectorStore(
            endee_collection=collection, collection_name="test", dimension=DIMENSION
        )
        assert vs.fetch(["a"]) == []

    # ── update_filters ──────────────────────────────────────────────────────

    def test_update_filters_delegates_to_collection(self, mock_endee_client):
        """update_filters() must delegate to the collection and return its result."""
        collection = _mock_vector_store_collection()
        collection.update_filters.return_value = "2 filters updated"
        vs = EndeeVectorStore(
            endee_collection=collection, collection_name="test", dimension=DIMENSION
        )
        updates = [
            {"id": "vec1", "filter": {"category": "B"}},
            {"id": "vec2", "filter": {"category": "C", "priority": 1}},
        ]
        result = vs.update_filters(updates)
        collection.update_filters.assert_called_once_with(updates)
        assert result == "2 filters updated"

    def test_update_filters_empty_list(self, mock_endee_client):
        """update_filters() must accept an empty list of updates without error."""
        collection = _mock_vector_store_collection()
        collection.update_filters.return_value = "0 filters updated"
        vs = EndeeVectorStore(
            endee_collection=collection, collection_name="test", dimension=DIMENSION
        )
        result = vs.update_filters([])
        collection.update_filters.assert_called_once_with([])
        assert result is not None

    # ── query_kwargs compat ─────────────────────────────────────────────────

    def _make_base_query(self):
        return VectorStoreQuery(
            query_embedding=[0.1, 0.2, 0.3, 0.4], similarity_top_k=2
        )

    @staticmethod
    def _inject_query_kwargs(q, **extra):
        object.__setattr__(q, "query_kwargs", extra)
        return q

    def test_prefilter_via_query_kwargs(self, mock_endee_client):
        """query() must read prefilter_cardinality_threshold from query_kwargs."""
        collection = _mock_vector_store_collection()
        vs = EndeeVectorStore(
            endee_collection=collection, collection_name="test", dimension=DIMENSION
        )
        q = self._inject_query_kwargs(
            self._make_base_query(), prefilter_cardinality_threshold=5000
        )
        vs.query(q)
        call_kwargs = collection.search.call_args[1]
        assert call_kwargs["prefilter_cardinality_threshold"] == 5000

    def test_filter_boost_via_query_kwargs(self, mock_endee_client):
        """query() reads filter_boost_percentage from query_kwargs and forwards it."""
        collection = _mock_vector_store_collection()
        vs = EndeeVectorStore(
            endee_collection=collection, collection_name="test", dimension=DIMENSION
        )
        q = self._inject_query_kwargs(
            self._make_base_query(), filter_boost_percentage=25
        )
        vs.query(q)
        call_kwargs = collection.search.call_args[1]
        assert call_kwargs["filter_boost_percentage"] == 25

    def test_both_params_via_query_kwargs(self, mock_endee_client):
        """query() forwards both threshold params together when set via query_kwargs."""
        collection = _mock_vector_store_collection()
        vs = EndeeVectorStore(
            endee_collection=collection, collection_name="test", dimension=DIMENSION
        )
        q = self._inject_query_kwargs(
            self._make_base_query(),
            prefilter_cardinality_threshold=20000,
            filter_boost_percentage=10,
        )
        vs.query(q)
        call_kwargs = collection.search.call_args[1]
        assert call_kwargs["prefilter_cardinality_threshold"] == 20000
        assert call_kwargs["filter_boost_percentage"] == 10

    def test_explicit_kwarg_takes_precedence_over_query_kwargs(self, mock_endee_client):
        """An explicit query() kwarg wins over the same value set via query_kwargs."""
        collection = _mock_vector_store_collection()
        vs = EndeeVectorStore(
            endee_collection=collection, collection_name="test", dimension=DIMENSION
        )
        q = self._inject_query_kwargs(
            self._make_base_query(), prefilter_cardinality_threshold=99999
        )
        vs.query(q, prefilter_cardinality_threshold=1000)
        call_kwargs = collection.search.call_args[1]
        assert call_kwargs["prefilter_cardinality_threshold"] == 1000

    def test_no_query_kwargs_attr_omits_params(self, mock_endee_client):
        """query() omits both threshold params when query has no query_kwargs."""
        collection = _mock_vector_store_collection()
        vs = EndeeVectorStore(
            endee_collection=collection, collection_name="test", dimension=DIMENSION
        )
        q = self._make_base_query()
        assert not hasattr(q, "query_kwargs")
        vs.query(q)
        call_kwargs = collection.search.call_args[1]
        assert "prefilter_cardinality_threshold" not in call_kwargs
        assert "filter_boost_percentage" not in call_kwargs

    # ── query(): ef_search / top_k forwarding into search_fields ────────────

    def test_query_forwards_ef_search_and_top_k_into_search_call(
        self, mock_endee_client
    ):
        """query() must forward ef_search and translate top_k into search limit."""
        collection = _mock_vector_store_collection()
        vs = EndeeVectorStore(
            endee_collection=collection, collection_name="test", dimension=DIMENSION
        )
        query = self._make_base_query()
        vs.query(query, ef_search=256)
        call_kwargs = collection.search.call_args[1]
        assert call_kwargs["ef_search"] == 256
        assert call_kwargs["fields"]["dense"]["limit"] == 2

    def test_query_default_ef_search_matches_constant(self, mock_endee_client):
        """query() without an explicit ef_search must default to DEFAULT_EF_SEARCH."""
        from llama_index_endee.constants import DEFAULT_EF_SEARCH

        collection = _mock_vector_store_collection()
        vs = EndeeVectorStore(
            endee_collection=collection, collection_name="test", dimension=DIMENSION
        )
        vs.query(self._make_base_query())
        call_kwargs = collection.search.call_args[1]
        assert call_kwargs["ef_search"] == DEFAULT_EF_SEARCH

    # ── class_name() ────────────────────────────────────────────────────────

    def test_class_name_returns_endeevectorstore(self):
        """class_name() must return the string 'EndeeVectorStore'."""
        assert EndeeVectorStore.class_name() == "EndeeVectorStore"

    # ── add_objects() / multi_field_search(): multi-field public API ────────

    def test_add_objects_batches_and_returns_all_ids(self, mock_endee_client):
        """add_objects() must batch by batch_size and return all ids in order."""
        collection = _mock_vector_store_collection()
        vs = EndeeVectorStore(
            endee_collection=collection, collection_name="test", dimension=DIMENSION
        )
        objects = [
            {
                "id": f"obj{i}",
                "meta": {},
                "filter": {},
                "fields": {"dense": [0, 0, 0, 0]},
            }
            for i in range(5)
        ]
        ids = vs.add_objects(objects, batch_size=2)
        assert ids == [f"obj{i}" for i in range(5)]
        assert collection.upsert.call_count == 3  # batches of 2, 2, 1

    def test_add_objects_default_batch_size_uses_store_batch_size(
        self, mock_endee_client
    ):
        """add_objects() without batch_size falls back to the store's own value."""
        collection = _mock_vector_store_collection()
        vs = EndeeVectorStore(
            endee_collection=collection,
            collection_name="test",
            dimension=DIMENSION,
            batch_size=3,
        )
        objects = [
            {
                "id": f"obj{i}",
                "meta": {},
                "filter": {},
                "fields": {"dense": [0, 0, 0, 0]},
            }
            for i in range(7)
        ]
        vs.add_objects(objects)
        # 7 objects at batch_size=3 -> batches of 3, 3, 1
        assert collection.upsert.call_count == 3

    def test_multi_field_search_forwards_all_optional_params(self, mock_endee_client):
        """multi_field_search() forwards filter, ef_search, both threshold params."""
        collection = _mock_vector_store_collection()
        collection.search.return_value = {"results": {"dense": [], "sparse": []}}
        vs = EndeeVectorStore(
            endee_collection=collection, collection_name="test", dimension=DIMENSION
        )
        fields = {"dense": {"query": [0.1, 0.2, 0.3, 0.4], "limit": 5}}
        out = vs.multi_field_search(
            fields,
            filter=[{"category": {"$eq": "ai"}}],
            ef_search=200,
            prefilter_cardinality_threshold=1000,
            filter_boost_percentage=15,
        )
        call_kwargs = collection.search.call_args[1]
        assert call_kwargs["fields"] == fields
        assert call_kwargs["ef_search"] == 200
        assert call_kwargs["filter"] == [{"category": {"$eq": "ai"}}]
        assert call_kwargs["prefilter_cardinality_threshold"] == 1000
        assert call_kwargs["filter_boost_percentage"] == 15
        assert out == {"results": {"dense": [], "sparse": []}}

    def test_multi_field_search_omits_optional_params_when_not_given(
        self, mock_endee_client
    ):
        """multi_field_search() omits optional params entirely when unsupplied."""
        collection = _mock_vector_store_collection()
        collection.search.return_value = {"results": {}}
        vs = EndeeVectorStore(
            endee_collection=collection, collection_name="test", dimension=DIMENSION
        )
        fields = {"dense": {"query": [0.1, 0.2, 0.3, 0.4], "limit": 5}}
        vs.multi_field_search(fields)
        call_kwargs = collection.search.call_args[1]
        assert "filter" not in call_kwargs
        assert "prefilter_cardinality_threshold" not in call_kwargs
        assert "filter_boost_percentage" not in call_kwargs

    def test_add_objects_with_multi_vector_field(self, mock_endee_client):
        """add_objects() accepts an object with a dense and a multi_vector field."""
        fields = [
            {
                "name": "dense",
                "type": "vector",
                "params": {
                    "dimension": DIMENSION,
                    "space_type": "cosine",
                    "precision": "int8",
                },
            },
            {
                "name": "chunks",
                "type": "multi_vector",
                "params": {
                    "dimension": 8,
                    "space_type": "cosine",
                    "precision": "float16",
                    "pooling": "mean",
                },
            },
        ]
        vs = EndeeVectorStore.from_params(
            collection_name="multi_vector_coll", fields=fields
        )
        chunks = [[0.1] * 8, [0.2] * 8, [0.3] * 8]
        objects = [
            {
                "id": "obj1",
                "meta": {},
                "filter": {},
                "fields": {"dense": [0.1, 0.2, 0.3, 0.4], "chunks": chunks},
            }
        ]

        ids = vs.add_objects(objects)

        assert ids == ["obj1"]
        stored = vs._collection._store["obj1"]
        assert stored["fields"]["dense"] == [0.1, 0.2, 0.3, 0.4]
        assert stored["fields"]["chunks"] == chunks
        assert len(stored["fields"]["chunks"]) == 3
        assert all(len(v) == 8 for v in stored["fields"]["chunks"])

    # ── empty/None query embedding: fixed test_empty_query ──────────────────

    def test_query_with_none_embedding_returns_empty_result_not_raise(
        self, mock_endee_client
    ):
        """query() must catch a search failure and return an empty result, not raise."""
        # search() is wrapped in a broad try/except, so any failure is caught.
        collection = _mock_vector_store_collection()
        collection.search.side_effect = ValueError("invalid embedding")
        vs = EndeeVectorStore(
            endee_collection=collection, collection_name="test", dimension=DIMENSION
        )
        query = VectorStoreQuery(similarity_top_k=3)  # no embedding
        assert query.query_embedding is None

        result = vs.query(query)  # must NOT raise

        assert result.nodes == []
        assert result.similarities == []
        assert result.ids == []

    # ── Result conversion round-trip ────────────────────────────────────────

    def test_query_result_round_trips_node_content_and_metadata(
        self, mock_endee_client
    ):
        """_process_single_result() must reconstruct a node matching what was stored."""
        vs = EndeeVectorStore.from_params(
            collection_name="roundtrip_coll", dimension=DIMENSION
        )
        node = TextNode(
            text="round trip me",
            embedding=[0.1, 0.2, 0.3, 0.4],
            id_="rt1",
            metadata={"category": "programming", "difficulty": "beginner"},
        )
        vs.add([node])

        stored = vs._collection._store["rt1"]
        reconstructed, score, node_id = vs._process_single_result(
            {"id": "rt1", "similarity": 0.987, "meta": dict(stored["meta"])}
        )

        assert node_id == "rt1"
        assert score == 0.987
        assert reconstructed.text == "round trip me"
        assert reconstructed.metadata.get("category") == "programming"
        assert reconstructed.metadata.get("difficulty") == "beginner"

    def test_process_query_results_skips_bad_result_and_keeps_good_ones(
        self, mock_endee_client
    ):
        """_process_query_results() must skip a bad result but keep the good ones."""
        vs = EndeeVectorStore.from_params(
            collection_name="skip_bad_coll", dimension=DIMENSION
        )
        node = TextNode(text="good node", embedding=[0.1, 0.2, 0.3, 0.4], id_="good")
        vs.add([node])
        good_meta = dict(vs._collection._store["good"]["meta"])

        results = [
            {"id": "bad", "similarity": 0.1, "meta": None},  # will raise inside
            {"id": "good", "similarity": 0.9, "meta": good_meta},
        ]
        nodes, similarities, ids = vs._process_query_results(results)

        assert ids == ["good"]
        assert len(nodes) == 1
        assert nodes[0].text == "good node"

    # ── constants.py fallback / override coverage (GAP per §4.5) ────────────

    def test_supported_filter_operators_contains_eq_and_in(self):
        """SUPPORTED_FILTER_OPERATORS must contain exactly EQ and IN."""
        from llama_index.core.vector_stores.types import FilterOperator

        from llama_index_endee.constants import SUPPORTED_FILTER_OPERATORS

        assert FilterOperator.EQ in SUPPORTED_FILTER_OPERATORS
        assert FilterOperator.IN in SUPPORTED_FILTER_OPERATORS
        assert len(SUPPORTED_FILTER_OPERATORS) == 2

    def test_reverse_operator_map_maps_eq_and_in(self):
        """REVERSE_OPERATOR_MAP must map EQ to '$eq' and IN to '$in'."""
        from llama_index.core.vector_stores.types import FilterOperator

        from llama_index_endee.constants import REVERSE_OPERATOR_MAP

        assert REVERSE_OPERATOR_MAP[FilterOperator.EQ] == "$eq"
        assert REVERSE_OPERATOR_MAP[FilterOperator.IN] == "$in"

    def test_precision_enum_exercised_directly_has_expected_values(self):
        """Precision (real SDK or fallback) must cover the values the wrapper needs."""
        from llama_index_endee.constants import Precision

        values = {p.value for p in Precision}
        assert {"binary", "float16", "float32", "int16", "int8"} <= values

    def test_max_key_bytes_and_max_value_bytes_exported_with_defaults(self):
        """MAX_KEY_BYTES/MAX_VALUE_BYTES are re-exported with documented defaults."""
        from llama_index_endee.constants import MAX_KEY_BYTES, MAX_VALUE_BYTES

        assert MAX_KEY_BYTES == 128
        assert MAX_VALUE_BYTES == 1024

    def test_endee_pydantic_compat_importable(self):
        """endee._pydantic_compat must export to_dict, field_validator, PYDANTIC_V2."""
        from endee._pydantic_compat import PYDANTIC_V2, field_validator, to_dict

        assert to_dict is not None
        assert field_validator is not None
        assert isinstance(PYDANTIC_V2, bool)

    def test_constants_overridden_from_sdk_when_available(self):
        """When endee.constants is importable, its override copies real SDK values."""
        # Reload fresh so the values come from the override, not a cached import.
        import importlib

        import endee.constants as real_ec

        import llama_index_endee.constants as loaded_constants

        importlib.reload(loaded_constants)

        for name in (
            "DEFAULT_EF_SEARCH",
            "DEFAULT_M",
            "DEFAULT_EF_CON",
            "MAX_VECTORS_PER_BATCH",
            "MAX_DIMENSION_ALLOWED",
            "MAX_EF_SEARCH_ALLOWED",
            "MAX_TOP_K_ALLOWED",
            "MAX_KEY_BYTES",
            "MAX_VALUE_BYTES",
            "SPARSE_MODE_TYPES_SUPPORTED",
        ):
            assert getattr(loaded_constants, name) == getattr(real_ec, name)

    def test_precision_fallback_enum_when_sdk_too_old(self):
        """With no SDK Precision, the fallback enum's members/values must be used."""
        # Simulates an endee SDK version with no Precision of its own.
        import endee.constants as real_ec

        orig_find_spec = importlib.util.find_spec

        def fake_find_spec(name, *a, **kw):
            if name == "endee.constants":
                return None
            return orig_find_spec(name, *a, **kw)

        had_precision = hasattr(real_ec, "Precision")
        saved_precision = getattr(real_ec, "Precision", None)
        if had_precision:
            del real_ec.Precision

        import llama_index_endee.constants as loaded_constants

        constants_file = loaded_constants.__file__

        importlib.util.find_spec = fake_find_spec
        try:
            spec = importlib.util.spec_from_file_location(
                "llama_index_endee_constants_fallback_test", constants_file
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        finally:
            importlib.util.find_spec = orig_find_spec
            if had_precision:
                real_ec.Precision = saved_precision

        names_values = {p.name: p.value for p in mod.Precision}
        assert names_values == {
            "BINARY2": "binary",
            "FLOAT16": "float16",
            "FLOAT32": "float32",
            "INT16": "int16",
            "INT8": "int8",
        }


# ═══════════════════════════════════════════════════════════════════════════
# Filters unit tests
# ═══════════════════════════════════════════════════════════════════════════


def _mock_filters_collection():
    collection = MagicMock()
    collection.fields = [
        {
            "name": "dense",
            "type": "vector",
            "params": {
                "dimension": DIMENSION,
                "space_type": "cosine",
                "precision": "int8",
            },
        }
    ]
    collection.search.return_value = {"results": {"dense": []}}
    return collection


@pytest.mark.unit
class TestFiltersUnit:
    # ── _extract_filter_fields: allowlist promotion ─────────────────────────

    def test_extract_filter_fields_promotes_only_allowlisted_keys(self):
        """_extract_filter_fields must promote only allowlisted metadata keys."""
        node = TextNode(text="hi", id_="n1")
        metadata = {
            "file_name": "a.txt",
            "doc_id": "d1",
            "category": "programming",
            "difficulty": "beginner",
            "language": "python",
            "field": "ml",
            "type": "vector",
            "feature": "search",
            "not_allowlisted": "should stay out of filter",
            "another_random_key": 123,
        }
        filter_data = EndeeVectorStore._extract_filter_fields(node, metadata)

        for key in (
            "file_name",
            "doc_id",
            "category",
            "difficulty",
            "language",
            "field",
            "type",
            "feature",
        ):
            assert filter_data[key] == metadata[key]

        assert "not_allowlisted" not in filter_data
        assert "another_random_key" not in filter_data

    def test_extract_filter_fields_promotes_ref_doc_id_from_node_attr(self):
        """_extract_filter_fields promotes ref_doc_id from metadata if node has none."""
        node = TextNode(text="hi", id_="n1")
        node.relationships = {}
        # Simulate the metadata fallback, since node.ref_doc_id needs relationships.
        metadata = {"ref_doc_id": "parent_doc_1"}
        filter_data = EndeeVectorStore._extract_filter_fields(node, metadata)
        assert filter_data["ref_doc_id"] == "parent_doc_1"

    def test_extract_filter_fields_non_allowlisted_key_stays_meta_only(self):
        """A non-allowlisted metadata key stays in `meta` but not in the filter dict."""
        node = TextNode(text="hi", id_="n1")
        metadata = {"category": "ai", "internal_note": "do-not-filter-on-this"}
        filter_data = EndeeVectorStore._extract_filter_fields(node, metadata)
        assert "internal_note" not in filter_data
        assert filter_data == {"category": "ai"}

    def test_extract_filter_fields_promotes_ref_doc_id_from_node_relationship(self):
        """_extract_filter_fields promotes ref_doc_id from the SOURCE relationship."""
        node = TextNode(text="child", id_="child1")
        node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
            node_id="real_parent_doc"
        )
        assert node.ref_doc_id == "real_parent_doc"

        metadata = {}
        filter_data = EndeeVectorStore._extract_filter_fields(node, metadata)
        assert filter_data["ref_doc_id"] == "real_parent_doc"

    def test_extract_filter_fields_node_relationship_takes_precedence_over_metadata(
        self,
    ):
        """With both sources present, the relationship-derived ref_doc_id wins."""
        node = TextNode(text="child", id_="child1")
        node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
            node_id="from_relationship"
        )
        metadata = {"ref_doc_id": "from_metadata"}
        filter_data = EndeeVectorStore._extract_filter_fields(node, metadata)
        assert filter_data["ref_doc_id"] == "from_relationship"

    def test_extract_filter_fields_sets_literal_none_string_when_no_source_relationship(
        self,
    ):
        """With no SOURCE relationship, filter["ref_doc_id"] is the string "None"."""
        # Note: node_to_metadata_dict() stringifies a missing ref_doc_id to
        # the literal "None", which is truthy, so it wins over the real None.
        from llama_index.core.vector_stores.utils import node_to_metadata_dict

        node = TextNode(text="no source relationship", id_="orphan1")
        metadata = node_to_metadata_dict(node)

        filter_data = EndeeVectorStore._extract_filter_fields(node, metadata)

        assert filter_data["ref_doc_id"] == "None"

    def test_extract_filter_fields_mixed_keys_all_present_in_meta_via_add(
        self, mock_endee_client
    ):
        """Only allowlisted keys land in `filter`; every key lands in `meta`."""
        vs = EndeeVectorStore.from_params(
            collection_name="mixed_keys_coll", dimension=DIMENSION
        )
        node = TextNode(
            text="mixed keys",
            embedding=[0.1, 0.2, 0.3, 0.4],
            id_="mixed1",
            metadata={
                "category": "ai",
                "difficulty": "beginner",
                "not_allowlisted": "stays-meta-only",
            },
        )
        vs.add([node])
        stored = vs._collection._store["mixed1"]

        assert stored["filter"]["category"] == "ai"
        assert stored["filter"]["difficulty"] == "beginner"
        assert "not_allowlisted" not in stored["filter"]
        assert stored["meta"]["category"] == "ai"
        assert stored["meta"]["difficulty"] == "beginner"
        assert stored["meta"]["not_allowlisted"] == "stays-meta-only"

    # ── _process_filters: EQ/IN supported, everything else raises ValueError ─

    def _store(self):
        return EndeeVectorStore(
            endee_collection=_mock_filters_collection(),
            collection_name="test",
            dimension=DIMENSION,
        )

    @staticmethod
    def _query(filters=None):
        """VectorStoreQuery with a fixed embedding, for _process_filters tests."""
        return VectorStoreQuery(
            query_embedding=[0.1, 0.2, 0.3, 0.4], similarity_top_k=2, filters=filters
        )

    def test_process_filters_eq_produces_dollar_eq(self):
        """_process_filters translates an EQ MetadataFilter into a '$eq' filter dict."""
        store = self._store()
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="category", value="ai", operator=FilterOperator.EQ)
            ]
        )
        result = store._process_filters(self._query(filters))
        assert result == [{"category": {"$eq": "ai"}}]

    def test_process_filters_in_produces_dollar_in(self):
        """_process_filters translates an IN MetadataFilter into a '$in' filter dict."""
        store = self._store()
        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="language",
                    value=["python", "javascript"],
                    operator=FilterOperator.IN,
                )
            ]
        )
        result = store._process_filters(self._query(filters))
        assert result == [{"language": {"$in": ["python", "javascript"]}}]

    def test_process_filters_none_when_no_filters(self):
        """_process_filters must return None when the query has no filters."""
        store = self._store()
        assert store._process_filters(self._query()) is None

    def test_process_filters_ne_raises_value_error(self):
        """_process_filters must raise ValueError for the unsupported NE operator."""
        store = self._store()
        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="difficulty", value="beginner", operator=FilterOperator.NE
                )
            ]
        )
        with pytest.raises(ValueError, match="Unsupported filter operator"):
            store._process_filters(self._query(filters))

    def test_process_filters_multiple_eq_filters_merge(self):
        """_process_filters merges multiple EQ filters into separate result entries."""
        store = self._store()
        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="category", value="programming", operator=FilterOperator.EQ
                ),
                MetadataFilter(
                    key="difficulty", value="beginner", operator=FilterOperator.EQ
                ),
            ]
        )
        result = store._process_filters(self._query(filters))
        assert {"category": {"$eq": "programming"}} in result
        assert {"difficulty": {"$eq": "beginner"}} in result

    def test_process_filters_raises_on_unsupported_operator_among_supported_ones(self):
        """_process_filters raises if any item uses an unsupported operator."""
        store = self._store()
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="category", value="ai", operator=FilterOperator.EQ),
                MetadataFilter(
                    key="difficulty", value="beginner", operator=FilterOperator.NE
                ),
            ]
        )
        with pytest.raises(ValueError, match="Unsupported filter operator"):
            store._process_filters(self._query(filters))

    def test_process_filters_raises_when_unsupported_operator_is_first(self):
        """_process_filters raises on the first bad operator, not just at the end."""
        store = self._store()
        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="difficulty", value="beginner", operator=FilterOperator.NE
                ),
                MetadataFilter(key="category", value="ai", operator=FilterOperator.EQ),
            ]
        )
        with pytest.raises(ValueError, match="Unsupported filter operator"):
            store._process_filters(self._query(filters))

    @staticmethod
    def _force_raw_filters(filters, raw_items):
        """Bypasses MetadataFilters' validation to inject a raw dict item."""
        object.__setattr__(filters, "filters", raw_items)
        return filters

    def test_process_filters_handles_raw_dict_filter_items(self):
        """_process_filters builds the same shape when filters holds raw dicts."""
        store = self._store()
        filters = MetadataFilters(
            filters=[MetadataFilter(key="x", value=1, operator=FilterOperator.EQ)]
        )
        self._force_raw_filters(filters, [{"category": {"$eq": "ai"}}])
        result = store._process_filters(self._query(filters))
        assert result == [{"category": {"$eq": "ai"}}]

    def test_process_filters_mixes_metadata_filter_and_raw_dict_items(self):
        """_process_filters merges a MetadataFilter and a raw dict into one result."""
        store = self._store()
        filters = MetadataFilters(
            filters=[MetadataFilter(key="x", value=1, operator=FilterOperator.EQ)]
        )
        self._force_raw_filters(
            filters,
            [
                MetadataFilter(key="category", value="ai", operator=FilterOperator.EQ),
                {"language": {"$in": ["python", "rust"]}},
            ],
        )
        result = store._process_filters(self._query(filters))
        assert {"category": {"$eq": "ai"}} in result
        assert {"language": {"$in": ["python", "rust"]}} in result


# ═══════════════════════════════════════════════════════════════════════════
# Sparse unit tests
# ═══════════════════════════════════════════════════════════════════════════


class _FakeSparseEmbeddings(SparseEmbeddings):
    """A trivial deterministic sparse embedder for tests."""

    def embed_documents(self, texts):
        return [
            SparseVector(indices=[i % 5 for i in range(len(t))], values=[1.0] * len(t))
            for t in texts
        ]

    def embed_query(self, text):
        return SparseVector(
            indices=list(range(len(text) % 5)), values=[1.0] * (len(text) % 5)
        )


def _mock_sparse_collection(fields=None):
    collection = MagicMock()
    collection.fields = fields or [
        {
            "name": "dense",
            "type": "vector",
            "params": {
                "dimension": DIMENSION,
                "space_type": "cosine",
                "precision": "int8",
            },
        }
    ]
    collection.search.return_value = {"results": {"dense": []}}
    return collection


def _hybrid_fields(sparse_model="endee_bm25"):
    """Dense+sparse field list shared by the hybrid-wiring tests below."""
    return [
        {
            "name": "dense",
            "type": "vector",
            "params": {
                "dimension": DIMENSION,
                "space_type": "cosine",
                "precision": "int8",
            },
        },
        {"name": "sparse", "type": "sparse", "sparse_model": sparse_model},
    ]


@pytest.mark.unit
class TestSparseUnit:
    # ── SparseVector / SparseModelAdapter / wrap_sparse_model ───────────────

    def test_sparse_vector_holds_indices_and_values(self):
        """SparseVector must store its indices and values as given."""
        sv = SparseVector(indices=[1, 3, 5], values=[0.1, 0.2, 0.3])
        assert sv.indices == [1, 3, 5]
        assert sv.values == [0.1, 0.2, 0.3]

    def test_wrap_sparse_model_returns_sparse_embeddings_as_is(self):
        """wrap_sparse_model must return a SparseEmbeddings instance unchanged."""
        fake = _FakeSparseEmbeddings()
        assert wrap_sparse_model(fake) is fake

    def test_wrap_sparse_model_wraps_embed_query_embed_object(self):
        """wrap_sparse_model wraps a third-party embed()/query_embed() in an adapter."""

        class _ThirdPartyModel:
            def embed(self, texts):
                for _ in texts:
                    m = MagicMock()
                    m.indices.tolist.return_value = [1, 2]
                    m.values.tolist.return_value = [0.5, 0.5]
                    yield m

            def query_embed(self, text):
                m = MagicMock()
                m.indices.tolist.return_value = [1]
                m.values.tolist.return_value = [1.0]
                yield m

        wrapped = wrap_sparse_model(_ThirdPartyModel())
        assert isinstance(wrapped, SparseModelAdapter)
        docs = wrapped.embed_documents(["a", "b"])
        assert len(docs) == 2
        assert docs[0].indices == [1, 2]

        q = wrapped.embed_query("hi")
        assert q.indices == [1]

    def test_wrap_sparse_model_rejects_incompatible_object(self):
        """wrap_sparse_model raises TypeError for an incompatible object."""
        with pytest.raises(TypeError):
            wrap_sparse_model(object())

    def test_endee_model_sparse_raises_if_package_missing(self):
        """EndeeModelSparse raises ImportError if the endee_model package is missing."""
        from llama_index_endee.sparse_embeddings import EndeeModelSparse

        with patch.dict("sys.modules", {"endee_model": None}):
            with pytest.raises(ImportError, match="endee_model"):
                EndeeModelSparse()

    def test_endee_model_sparse_wraps_bm25(self):
        """EndeeModelSparse wraps BM25's embed() output into SparseVectors."""
        from llama_index_endee.sparse_embeddings import EndeeModelSparse

        fake_sparse_model_cls = MagicMock()
        fake_instance = MagicMock()

        def fake_embed(texts):
            for _ in texts:
                m = MagicMock()
                m.indices.tolist.return_value = [0, 1]
                m.values.tolist.return_value = [1.0, 2.0]
                yield m

        fake_instance.embed.side_effect = fake_embed
        fake_sparse_model_cls.return_value = fake_instance

        fake_module = MagicMock()
        fake_module.SparseModel = fake_sparse_model_cls
        with patch.dict("sys.modules", {"endee_model": fake_module}):
            model = EndeeModelSparse()
            docs = model.embed_documents(["hello world"])
            assert docs[0].indices == [0, 1]
            assert docs[0].values == [1.0, 2.0]

    # ── Hybrid auto-detection wiring in EndeeVectorStore ────────────────────

    def test_hybrid_auto_detected_from_endee_bm25_field(self, mock_endee_client):
        """sparse_model='endee_bm25' with no embedding auto-creates one, sets hybrid."""
        with patch(
            "llama_index_endee.sparse_embeddings.EndeeModelSparse"
        ) as mock_sparse_cls:
            mock_sparse_cls.return_value = _FakeSparseEmbeddings()

            vs = EndeeVectorStore.from_params(
                collection_name="hybrid_coll",
                fields=_hybrid_fields(),
            )
            assert vs.hybrid is True
            mock_sparse_cls.assert_called_once()

    def test_hybrid_not_auto_created_when_sparse_embedding_provided(
        self, mock_endee_client
    ):
        """Supplying sparse_embedding skips auto-creating EndeeModelSparse."""
        provided = _FakeSparseEmbeddings()
        with patch(
            "llama_index_endee.sparse_embeddings.EndeeModelSparse"
        ) as mock_sparse_cls:
            vs = EndeeVectorStore.from_params(
                collection_name="hybrid_coll2",
                fields=_hybrid_fields(),
                sparse_embedding=provided,
            )
            assert vs.hybrid is True
            # Auto-create path must be skipped since a sparse_embedding
            # was already supplied.
            mock_sparse_cls.assert_not_called()
            assert vs._sparse_embeddings is provided

    def test_dense_only_collection_is_not_hybrid(self, mock_endee_client):
        """A dense-only collection must report hybrid=False."""
        vs = EndeeVectorStore.from_params(
            collection_name="dense_coll", dimension=DIMENSION
        )
        assert vs.hybrid is False

    def test_default_sparse_model_with_explicit_embedding_skips_auto_create(
        self, mock_endee_client
    ):
        """sparse_model='default' with an explicit embedding skips auto-create."""
        custom_sparse = _FakeSparseEmbeddings()
        with patch(
            "llama_index_endee.sparse_embeddings.EndeeModelSparse"
        ) as mock_sparse_cls:
            vs = EndeeVectorStore.from_params(
                collection_name="default_sparse_coll",
                fields=_hybrid_fields(sparse_model="default"),
                sparse_embedding=custom_sparse,
            )
            mock_sparse_cls.assert_not_called()
            assert vs._sparse_embeddings is custom_sparse
            assert vs.hybrid is True
            assert vs._collection.fields[1]["sparse_model"] == "default"

    def test_default_sparse_model_without_embedding_stays_non_hybrid(
        self, mock_endee_client
    ):
        """sparse_model='default' with no embedding stays non-hybrid unlike bm25."""
        with patch(
            "llama_index_endee.sparse_embeddings.EndeeModelSparse"
        ) as mock_sparse_cls:
            vs = EndeeVectorStore.from_params(
                collection_name="default_sparse_noembed_coll",
                fields=_hybrid_fields(sparse_model="default"),
            )
            mock_sparse_cls.assert_not_called()
            assert vs._sparse_embeddings is None
            assert vs.hybrid is False

    def test_add_computes_sparse_vectors_when_hybrid(self, mock_endee_client):
        """add() on a hybrid store computes and stores sparse indices/values too."""
        fake_sparse = _FakeSparseEmbeddings()
        vs = EndeeVectorStore.from_params(
            collection_name="hybrid_add_coll",
            fields=_hybrid_fields(),
            sparse_embedding=fake_sparse,
        )
        node = TextNode(text="hello", embedding=[0.1, 0.2, 0.3, 0.4], id_="h1")
        vs.add([node])
        stored = vs._collection._store["h1"]
        assert "sparse" in stored["fields"]
        assert "indices" in stored["fields"]["sparse"]
        assert "values" in stored["fields"]["sparse"]

    def _hybrid_store_with_node(self, collection_name):
        """Builds a dense+endee_bm25 hybrid store with one indexed node."""
        vs = EndeeVectorStore.from_params(
            collection_name=collection_name,
            fields=_hybrid_fields(),
            sparse_embedding=_FakeSparseEmbeddings(),
        )
        node = TextNode(text="hello world", embedding=[0.1, 0.2, 0.3, 0.4], id_="h1")
        vs.add([node])
        return vs

    def test_query_fuses_multi_field_results_with_rerank(self, mock_endee_client):
        """query() on a hybrid store calls endee_rerank with dense+sparse results."""
        vs = self._hybrid_store_with_node("rerank_coll")

        with patch("llama_index_endee.base.endee_rerank") as mock_rerank:
            mock_rerank.return_value = {"results": []}
            query = VectorStoreQuery(
                query_embedding=[0.1, 0.2, 0.3, 0.4],
                query_str="hello world",
                similarity_top_k=3,
            )
            vs.query(query)
            mock_rerank.assert_called_once()
            call_args = mock_rerank.call_args[0][0]
            assert "dense" in call_args["results"]
            assert "sparse" in call_args["results"]

    def test_single_field_query_skips_rerank(self, mock_endee_client):
        """With only the dense field searched, endee_rerank must not be invoked."""
        vs = EndeeVectorStore.from_params(
            collection_name="no_rerank_coll", dimension=DIMENSION
        )
        node = TextNode(text="hello", embedding=[0.1, 0.2, 0.3, 0.4], id_="h1")
        vs.add([node])

        with patch("llama_index_endee.base.endee_rerank") as mock_rerank:
            query = VectorStoreQuery(
                query_embedding=[0.1, 0.2, 0.3, 0.4], similarity_top_k=3
            )
            vs.query(query)
            mock_rerank.assert_not_called()

    def test_query_with_zero_result_fields_returns_empty_not_error(
        self, mock_endee_client
    ):
        """query() returns empty and skips rerank when the collection has no results."""
        # Falls through to the empty results=[] branch rather than reranking.
        collection = _mock_sparse_collection()
        collection.search.return_value = {"results": {}}
        vs = EndeeVectorStore(
            endee_collection=collection, collection_name="test", dimension=DIMENSION
        )
        with patch("llama_index_endee.base.endee_rerank") as mock_rerank:
            query = VectorStoreQuery(
                query_embedding=[0.1, 0.2, 0.3, 0.4], similarity_top_k=3
            )
            result = vs.query(query)
            mock_rerank.assert_not_called()
        assert result.nodes == []
        assert result.similarities == []
        assert result.ids == []

    def test_dense_rrf_weight_forwarded_as_field_weights_when_multi_field(
        self, mock_endee_client
    ):
        """query() forwards dense_rrf_weight into rerank_kwargs as per-field weights."""
        # dense gets the weight, sparse gets 1 - weight.
        vs = self._hybrid_store_with_node("rrf_weight_coll")

        with patch("llama_index_endee.base.endee_rerank") as mock_rerank:
            mock_rerank.return_value = {"results": []}
            query = VectorStoreQuery(
                query_embedding=[0.1, 0.2, 0.3, 0.4],
                query_str="hello world",
                similarity_top_k=3,
            )
            vs.query(query, dense_rrf_weight=0.7)
            rerank_kwargs = mock_rerank.call_args[1]
            assert rerank_kwargs["field_weights"]["dense"] == 0.7
            assert rerank_kwargs["field_weights"]["sparse"] == pytest.approx(0.3)

    def test_rrf_rank_constant_forwarded_as_rrf_k_when_multi_field(
        self, mock_endee_client
    ):
        """query() forwards rrf_rank_constant as `rrf_k` in rerank_kwargs."""
        vs = self._hybrid_store_with_node("rrf_k_coll")

        with patch("llama_index_endee.base.endee_rerank") as mock_rerank:
            mock_rerank.return_value = {"results": []}
            query = VectorStoreQuery(
                query_embedding=[0.1, 0.2, 0.3, 0.4],
                query_str="hello world",
                similarity_top_k=3,
            )
            vs.query(query, rrf_rank_constant=30)
            rerank_kwargs = mock_rerank.call_args[1]
            assert rerank_kwargs["rrf_k"] == 30

    def test_dense_rrf_weight_and_rrf_rank_constant_omitted_when_none(
        self, mock_endee_client
    ):
        """With both rrf params left None, rerank_kwargs must contain only `limit`."""
        vs = self._hybrid_store_with_node("rrf_none_coll")

        with patch("llama_index_endee.base.endee_rerank") as mock_rerank:
            mock_rerank.return_value = {"results": []}
            query = VectorStoreQuery(
                query_embedding=[0.1, 0.2, 0.3, 0.4],
                query_str="hello world",
                similarity_top_k=3,
            )
            vs.query(query)
            rerank_kwargs = mock_rerank.call_args[1]
            assert rerank_kwargs == {"limit": 3}

    def test_dense_rrf_weight_via_query_kwargs(self, mock_endee_client):
        """query() reads dense_rrf_weight from query_kwargs, like threshold params."""
        vs = self._hybrid_store_with_node("rrf_qkwargs_coll")

        with patch("llama_index_endee.base.endee_rerank") as mock_rerank:
            mock_rerank.return_value = {"results": []}
            query = VectorStoreQuery(
                query_embedding=[0.1, 0.2, 0.3, 0.4],
                query_str="hello world",
                similarity_top_k=3,
            )
            object.__setattr__(query, "query_kwargs", {"dense_rrf_weight": 0.9})
            vs.query(query)
            rerank_kwargs = mock_rerank.call_args[1]
            assert rerank_kwargs["field_weights"] == {
                "dense": 0.9,
                "sparse": pytest.approx(0.1),
            }

    def test_rrf_rank_constant_via_query_kwargs(self, mock_endee_client):
        """query() reads rrf_rank_constant from query.query_kwargs."""
        vs = self._hybrid_store_with_node("rrf_k_qkwargs_coll")

        with patch("llama_index_endee.base.endee_rerank") as mock_rerank:
            mock_rerank.return_value = {"results": []}
            query = VectorStoreQuery(
                query_embedding=[0.1, 0.2, 0.3, 0.4],
                query_str="hello world",
                similarity_top_k=3,
            )
            object.__setattr__(query, "query_kwargs", {"rrf_rank_constant": 100})
            vs.query(query)
            rerank_kwargs = mock_rerank.call_args[1]
            assert rerank_kwargs["rrf_k"] == 100

    def test_explicit_dense_rrf_weight_takes_precedence_over_query_kwargs(
        self, mock_endee_client
    ):
        """An explicit dense_rrf_weight kwarg wins over the query_kwargs value."""
        vs = self._hybrid_store_with_node("rrf_precedence_coll")

        with patch("llama_index_endee.base.endee_rerank") as mock_rerank:
            mock_rerank.return_value = {"results": []}
            query = VectorStoreQuery(
                query_embedding=[0.1, 0.2, 0.3, 0.4],
                query_str="hello world",
                similarity_top_k=3,
            )
            object.__setattr__(query, "query_kwargs", {"dense_rrf_weight": 0.2})
            vs.query(query, dense_rrf_weight=0.6)
            rerank_kwargs = mock_rerank.call_args[1]
            assert rerank_kwargs["field_weights"]["dense"] == 0.6

    def test_sparse_mode_types_supported_contains_default_and_endee_bm25(self):
        """SPARSE_MODE_TYPES_SUPPORTED must be exactly 'default' and 'endee_bm25'."""
        from llama_index_endee.constants import SPARSE_MODE_TYPES_SUPPORTED

        assert isinstance(SPARSE_MODE_TYPES_SUPPORTED, list)
        assert set(SPARSE_MODE_TYPES_SUPPORTED) == {"default", "endee_bm25"}

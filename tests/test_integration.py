"""Integration tests: all tests that require a live/local Endee server
(set `ENDEE_API_TOKEN`).

`TestVectorStoreIntegration` covers core CRUD (create-or-reuse, add/query,
delete, clear, fetch, precision/space_type/HNSW params verified via
describe(), batching, client construction, and collection lifecycle) against
a real server.

`TestFiltersIntegration` covers filter/metadata-key translation end-to-end
against a real server.

`TestSparseIntegration` covers sparse/hybrid collections end-to-end,
including both endee_bm25 and a custom sparse embedding, against a real
server.

`TestRetrieval` exercises the actual LlamaIndex framework objects
(VectorStoreIndex, VectorIndexRetriever, RetrieverQueryEngine, and
query_str back-compat) end-to-end, not just EndeeVectorStore directly.
"""

from __future__ import annotations

import os

import pytest
from conftest import ALL_PRECISIONS, ALL_SPACE_TYPES, safe_delete, uid
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
)

from llama_index_endee.base import EndeeVectorStore
from llama_index_endee.sparse_embeddings import SparseEmbeddings, SparseVector


class _DeterministicSparseEmbedding(SparseEmbeddings):
    """Deterministic hash-based sparse embedder, for testing custom sparse models."""

    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text):
        indices = sorted({hash(w) % 500 for w in text.lower().split()})
        values = [1.0 / (i + 1) for i in range(len(indices))]
        return SparseVector(indices=indices, values=values)


# ═══════════════════════════════════════════════════════════════════════════
# Vector store integration tests
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestVectorStoreIntegration:
    def test_from_params_creates_store(self, store_factory):
        """from_params() must create a store with the given dimension and space_type."""
        vs = store_factory(dimension=16, space_type="cosine")
        assert vs.dimension == 16
        assert vs.space_type == "cosine"
        assert vs.client is not None

    def test_describe_returns_collection_metadata(self, store_factory):
        """describe() must return a dict with a non-empty fields list."""
        vs = store_factory(dimension=16, space_type="cosine")
        info = vs.describe()
        assert isinstance(info, dict)
        assert "fields" in info
        assert len(info["fields"]) > 0

    def test_client_returns_endee_collection(self, store_factory):
        """client property must expose upsert, search, describe, and update_filters."""
        vs = store_factory(dimension=16, space_type="cosine")
        client = vs.client
        assert hasattr(client, "upsert")
        assert hasattr(client, "search")
        assert hasattr(client, "describe")
        assert hasattr(client, "update_filters")

    def test_default_precision_is_int8(self, store_factory):
        """A collection created without an explicit precision must default to int8."""
        vs = store_factory(dimension=16, space_type="cosine")
        assert vs.precision == "int8"

    def test_update_filters_wrapper(self, store_factory, fake_embedder):
        """update_filters() must apply a filter update without error."""
        vs = store_factory(dimension=fake_embedder.dim, space_type="cosine")
        text = "filter update test node"
        node = TextNode(text=text, embedding=fake_embedder(text), id_="upd_node")
        vs.add([node])
        result = vs.update_filters(
            [{"id": "upd_node", "filter": {"category": "updated"}}]
        )
        assert result is not None

    def test_add_and_query(self, store_factory, fake_embedder):
        """add() followed by query() must return the ids that were added."""
        vs = store_factory(dimension=fake_embedder.dim, space_type="cosine")
        texts = ["Python is great", "Rust is fast", "ML is useful"]
        nodes = [
            TextNode(text=t, embedding=fake_embedder(t), id_=f"node_{i}")
            for i, t in enumerate(texts)
        ]
        ids = vs.add(nodes)
        assert set(ids) == {n.node_id for n in nodes}

        query = VectorStoreQuery(
            query_embedding=fake_embedder("Python"), similarity_top_k=2
        )
        result = vs.query(query)
        assert len(result.ids) == len(result.nodes)

    def test_delete_is_noop_when_no_match(self, store_factory):
        """delete() with a ref_doc_id that matches nothing must not raise."""
        vs = store_factory(dimension=16, space_type="cosine")
        vs.delete("some_ref_doc_id")  # should not raise

    def test_fetch_returns_list(self, store_factory, fake_embedder):
        """fetch() against a live server must return a list."""
        vs = store_factory(dimension=fake_embedder.dim, space_type="cosine")
        text = "Fetch me"
        node = TextNode(text=text, embedding=fake_embedder(text), id_="fetch_test")
        vs.add([node])
        out = vs.fetch(["fetch_test"])
        assert isinstance(out, list)

    def test_query_with_prefilter_cardinality_threshold(
        self, store_factory, fake_embedder
    ):
        """query() with prefilter_cardinality_threshold must succeed."""
        vs = store_factory(dimension=fake_embedder.dim, space_type="cosine")
        text = "Python is great for data science"
        vs.add([TextNode(text=text, embedding=fake_embedder(text), id_="seed")])
        query = VectorStoreQuery(
            query_embedding=fake_embedder("data science"), similarity_top_k=2
        )
        result = vs.query(query, prefilter_cardinality_threshold=10000)
        assert isinstance(result.nodes, list)
        assert len(result.ids) == len(result.nodes)

    def test_query_with_filter_boost_percentage(self, store_factory, fake_embedder):
        """query() with filter_boost_percentage must return results without error."""
        vs = store_factory(dimension=fake_embedder.dim, space_type="cosine")
        text = "Python is great for data science"
        vs.add([TextNode(text=text, embedding=fake_embedder(text), id_="seed")])
        query = VectorStoreQuery(
            query_embedding=fake_embedder("data science"), similarity_top_k=2
        )
        result = vs.query(query, filter_boost_percentage=30)
        assert isinstance(result.nodes, list)

    def test_query_with_both_filter_params(self, store_factory, fake_embedder):
        """prefilter_cardinality_threshold and filter_boost_percentage combine fine."""
        vs = store_factory(dimension=fake_embedder.dim, space_type="cosine")
        text = "Python is great for data science"
        vs.add([TextNode(text=text, embedding=fake_embedder(text), id_="seed")])
        query = VectorStoreQuery(
            query_embedding=fake_embedder("data science"), similarity_top_k=2
        )
        result = vs.query(
            query, prefilter_cardinality_threshold=50000, filter_boost_percentage=20
        )
        assert isinstance(result.nodes, list)

    @pytest.mark.parametrize("precision", ALL_PRECISIONS, ids=lambda p: p)
    def test_create_collection_with_precision(self, store_factory, precision):
        """The server must actually create the field with the requested precision."""
        vs = store_factory(dimension=16, space_type="cosine", precision=precision)
        assert vs.precision == precision
        dense_field = vs.describe()["fields"][0]
        assert dense_field["params"]["precision"] == precision

    @pytest.mark.parametrize("space_type", ALL_SPACE_TYPES, ids=ALL_SPACE_TYPES)
    def test_create_collection_with_space_type(self, store_factory, space_type):
        """The server must actually create the field with the requested space_type."""
        vs = store_factory(dimension=16, space_type=space_type)
        assert vs.space_type == space_type
        dense_field = vs.describe()["fields"][0]
        assert dense_field["params"]["space_type"] == space_type

    def test_create_collection_with_custom_hnsw_params(self, store_factory):
        """The server must actually apply the requested M and ef_con HNSW params."""
        vs = store_factory(dimension=16, space_type="cosine", M=32, ef_con=256)
        dense_field = vs.describe()["fields"][0]
        assert dense_field["params"]["M"] == 32
        assert dense_field["params"]["ef_con"] == 256

    def test_batch_insert_150_documents(self, store_factory, fake_embedder):
        """add() must handle 150 documents in batches, all of them queryable after."""
        vs = store_factory(
            dimension=fake_embedder.dim, space_type="cosine", batch_size=50
        )
        nodes = [
            TextNode(
                text=f"Document number {i} about topic {i % 5}",
                embedding=fake_embedder(f"Document number {i} about topic {i % 5}"),
                id_=f"doc_{i}",
                metadata={"doc_id": str(i), "topic": str(i % 5)},
            )
            for i in range(150)
        ]
        ids = vs.add(nodes)
        assert len(ids) == 150

        query = VectorStoreQuery(
            query_embedding=fake_embedder("document topic"), similarity_top_k=10
        )
        result = vs.query(query)
        assert len(result.nodes) > 0

    def test_use_existing_collection(self, store_factory, fake_embedder):
        """A second store reusing a collection name queries data the first one added."""
        vs1 = store_factory(dimension=fake_embedder.dim, space_type="cosine")
        name = vs1.collection_name
        text = "Test document in existing collection"
        vs1.add([TextNode(text=text, embedding=fake_embedder(text), id_="doc1")])

        vs2 = EndeeVectorStore.from_params(
            endee_client=vs1._client,
            collection_name=name,
            dimension=fake_embedder.dim,
            space_type="cosine",
        )
        query = VectorStoreQuery(
            query_embedding=fake_embedder("test document"), similarity_top_k=1
        )
        result = vs2.query(query)
        assert len(result.nodes) > 0

    @pytest.mark.parametrize("top_k", [1, 3, 5])
    def test_query_respects_similarity_top_k(self, store_factory, fake_embedder, top_k):
        """query() must never return more nodes than similarity_top_k."""
        vs = store_factory(dimension=fake_embedder.dim, space_type="cosine")
        texts = [f"programming topic number {i}" for i in range(6)]
        nodes = [
            TextNode(text=t, embedding=fake_embedder(t), id_=f"topk_{i}")
            for i, t in enumerate(texts)
        ]
        vs.add(nodes)
        query = VectorStoreQuery(
            query_embedding=fake_embedder("programming"), similarity_top_k=top_k
        )
        result = vs.query(query)
        assert len(result.nodes) <= top_k

    def test_query_with_custom_ef_search(self, store_factory, fake_embedder):
        """query() with a custom ef_search must still return results."""
        vs = store_factory(dimension=fake_embedder.dim, space_type="cosine")
        text = "machine learning models need data"
        vs.add([TextNode(text=text, embedding=fake_embedder(text), id_="ef_seed")])
        query = VectorStoreQuery(
            query_embedding=fake_embedder("machine learning"), similarity_top_k=3
        )
        result = vs.query(query, ef_search=256)
        assert isinstance(result.nodes, list)
        assert len(result.nodes) > 0

    def test_query_similarity_scores_in_valid_cosine_range(
        self, store_factory, fake_embedder
    ):
        """Cosine similarity scores returned by query() must fall within [-1.0, 1.0]."""
        vs = store_factory(dimension=fake_embedder.dim, space_type="cosine")
        text = "Python programming for data science"
        vs.add([TextNode(text=text, embedding=fake_embedder(text), id_="score_seed")])
        query = VectorStoreQuery(
            query_embedding=fake_embedder("Python programming"), similarity_top_k=3
        )
        result = vs.query(query)
        assert len(result.similarities) > 0
        for score in result.similarities:
            assert -1.0 <= score <= 1.0

    def test_delete_removes_matching_ref_doc_id_entries(
        self, store_factory, fake_embedder
    ):
        """delete(ref_doc_id) must remove matching objects, not just avoid raising."""
        from llama_index.core.schema import NodeRelationship, RelatedNodeInfo

        vs = store_factory(dimension=fake_embedder.dim, space_type="cosine")
        text = "node scheduled for deletion"
        node = TextNode(text=text, embedding=fake_embedder(text), id_="to_delete")
        node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
            node_id="doc_to_delete"
        )
        vs.add([node])

        fetched_before = vs.fetch(["to_delete"])
        assert len(fetched_before) == 1

        vs.delete("doc_to_delete")

        fetched_after = vs.fetch(["to_delete"])
        assert fetched_after == []

    def test_empty_query_returns_result_without_raising(
        self, store_factory, fake_embedder
    ):
        """query() with no embedding must not raise."""
        vs = store_factory(dimension=fake_embedder.dim, space_type="cosine")
        query = VectorStoreQuery(similarity_top_k=3)  # no embedding
        result = vs.query(query)  # must not raise
        assert isinstance(result.nodes, list)

    def test_multi_vector_field_add_objects_and_multi_field_search(
        self, store_factory, fake_embedder
    ):
        """A multi_vector field must accept add_objects() and yield per-field hits."""
        dim = fake_embedder.dim
        fields = [
            {
                "name": "dense",
                "type": "vector",
                "params": {
                    "dimension": dim,
                    "space_type": "cosine",
                    "precision": "int8",
                },
            },
            {
                "name": "chunks",
                "type": "multi_vector",
                "params": {
                    "dimension": dim,
                    "space_type": "cosine",
                    "precision": "float16",
                    "pooling": "mean",
                },
            },
        ]
        vs = store_factory(fields=fields)

        text = "Consensus algorithms in distributed systems"
        dense_vec = fake_embedder(text)
        chunk_vecs = [
            fake_embedder("consensus"),
            fake_embedder("distributed"),
            fake_embedder("systems"),
        ]
        vs.add_objects(
            [
                {
                    "id": "mv1",
                    "meta": {"text": text},
                    "filter": {},
                    "fields": {"dense": dense_vec, "chunks": chunk_vecs},
                }
            ]
        )

        raw = vs.multi_field_search(
            fields={
                "dense": {"query": dense_vec, "limit": 3},
                "chunks": {"query": chunk_vecs, "limit": 3},
            }
        )
        assert "dense" in raw["results"]
        assert "chunks" in raw["results"]

    def test_multi_field_search_forwards_filter_and_thresholds_live(
        self, store_factory, fake_embedder
    ):
        """multi_field_search() must accept filter/prefilter/boost on a real server."""
        dim = fake_embedder.dim
        vs = store_factory(dimension=dim, space_type="cosine")
        text = "Filtered multi field search"
        vec = fake_embedder(text)
        node = TextNode(
            text=text, embedding=vec, id_="mf_filter", metadata={"category": "ai"}
        )
        vs.add([node])

        raw = vs.multi_field_search(
            fields={"dense": {"query": vec, "limit": 3}},
            filter=[{"category": {"$eq": "ai"}}],
            prefilter_cardinality_threshold=1000,
            filter_boost_percentage=20,
        )
        assert "dense" in raw["results"]

    def test_clear_deletes_the_collection(self, store_factory, live_client):
        """clear() must actually delete the collection from the live server."""
        vs = store_factory(dimension=16, space_type="cosine")
        name = vs.collection_name
        vs.clear()
        names = [
            c.get("name") if isinstance(c, dict) else c
            for c in live_client.list_collections()
        ]
        assert name not in names

    def test_delete_vector_removes_a_single_object(self, store_factory, fake_embedder):
        """delete_vector() must remove exactly the object with the given id."""
        vs = store_factory(dimension=fake_embedder.dim, space_type="cosine")
        node = TextNode(
            text="single object to delete",
            embedding=fake_embedder("single object to delete"),
            id_="solo_object",
        )
        vs.add([node])
        vs.delete_vector("solo_object")
        assert vs.fetch(["solo_object"]) == []

    def test_force_recreate_deletes_existing_data(self, store_factory, fake_embedder):
        """force_recreate=True on an existing collection must wipe prior data."""
        vs = store_factory(dimension=fake_embedder.dim, space_type="cosine")
        name = vs.collection_name
        node = TextNode(
            text="first generation data",
            embedding=fake_embedder("first generation data"),
            id_="gen1",
        )
        vs.add([node])
        assert vs.describe()["fields"][0]["element_count"] == 1

        recreated = store_factory(
            collection_name=name,
            dimension=fake_embedder.dim,
            space_type="cosine",
            force_recreate=True,
        )
        assert recreated.describe()["fields"][0]["element_count"] == 0

    def test_endee_collection_override_uses_the_given_collection(
        self, store_factory, live_client, fake_embedder
    ):
        """endee_collection= must skip lookup and use the pre-fetched Collection."""
        vs = store_factory(dimension=fake_embedder.dim, space_type="cosine")
        raw_collection = live_client.get_collection(name=vs.collection_name)

        reused = EndeeVectorStore(
            endee_client=live_client,
            collection_name=vs.collection_name,
            endee_collection=raw_collection,
        )
        assert reused.describe()["name"] == vs.collection_name

    def test_init_builds_client_from_api_token_and_base_url(self, live_client):
        """A store built from api_token/base_url (not endee_client=) must connect."""
        name = uid()
        vs = EndeeVectorStore.from_params(
            api_token=os.environ["ENDEE_API_TOKEN"],
            base_url=os.environ.get("ENDEE_BASE_URL"),
            collection_name=name,
            dimension=16,
            space_type="cosine",
            force_recreate=True,
        )
        try:
            assert vs.describe()["name"] == name
        finally:
            safe_delete(live_client, name)


# ═══════════════════════════════════════════════════════════════════════════
# Filters integration tests
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestFiltersIntegration:
    # ── Fixture ──────────────────────────────────────────────────────────

    @pytest.fixture
    def indexed_store(
        self, store_factory, sample_documents, fake_embedder, fake_embed_model
    ):
        """A live-server store pre-populated with sample_documents via fake_embedder."""
        from llama_index.core import Settings, StorageContext, VectorStoreIndex

        vs = store_factory(dimension=fake_embedder.dim, space_type="cosine")

        Settings.llm = None
        storage_context = StorageContext.from_defaults(vector_store=vs)
        VectorStoreIndex.from_documents(
            sample_documents,
            storage_context=storage_context,
            embed_model=fake_embed_model,
        )

        yield vs

    def test_single_eq_filter_returns_only_matching_category(
        self, indexed_store, fake_embedder
    ):
        """A single EQ filter must return only nodes matching that category."""
        ai_filter = MetadataFilter(
            key="category", value="ai", operator=FilterOperator.EQ
        )
        query = VectorStoreQuery(
            query_embedding=fake_embedder("What is machine learning?"),
            similarity_top_k=5,
            filters=MetadataFilters(filters=[ai_filter]),
        )
        result = indexed_store.query(query)
        assert len(result.nodes) > 0
        for node in result.nodes:
            assert "ai" in str(node.metadata.get("category", "")).lower()

    def test_multiple_eq_filters_returns_only_matching_all(
        self, indexed_store, fake_embedder
    ):
        """Multiple EQ filters combined must return only nodes matching all of them."""
        category_filter = MetadataFilter(
            key="category", value="programming", operator=FilterOperator.EQ
        )
        difficulty_filter = MetadataFilter(
            key="difficulty", value="beginner", operator=FilterOperator.EQ
        )
        query = VectorStoreQuery(
            query_embedding=fake_embedder(
                "What programming language is good for beginners?"
            ),
            similarity_top_k=5,
            filters=MetadataFilters(filters=[category_filter, difficulty_filter]),
        )
        result = indexed_store.query(query)
        assert len(result.nodes) > 0
        for node in result.nodes:
            assert "programming" in str(node.metadata.get("category", "")).lower()
            assert "beginner" in str(node.metadata.get("difficulty", "")).lower()

    def test_in_filter_returns_only_matching_languages(
        self, indexed_store, fake_embedder
    ):
        """An IN filter must return only nodes whose language matches a listed value."""
        in_filter = MetadataFilter(
            key="language", value=["python", "javascript"], operator=FilterOperator.IN
        )
        query = VectorStoreQuery(
            query_embedding=fake_embedder("What web technologies are discussed?"),
            similarity_top_k=5,
            filters=MetadataFilters(filters=[in_filter]),
        )
        result = indexed_store.query(query)
        assert len(result.nodes) > 0
        for node in result.nodes:
            assert str(node.metadata.get("language", "")).lower() in (
                "python",
                "javascript",
            )

    def test_ne_operator_raises_value_error(self, indexed_store, fake_embedder):
        """Querying with an NE filter against a live index must raise ValueError."""
        ne_filter = MetadataFilter(
            key="difficulty", value="beginner", operator=FilterOperator.NE
        )
        query = VectorStoreQuery(
            query_embedding=fake_embedder("What advanced topics are covered?"),
            similarity_top_k=3,
            filters=MetadataFilters(filters=[ne_filter]),
        )
        with pytest.raises(ValueError, match="Unsupported filter operator"):
            indexed_store.query(query)

    def test_nonexistent_filter_key_returns_empty_not_a_crash(
        self, indexed_store, fake_embedder
    ):
        """Filtering on a metadata key no node has must return empty, not raise."""
        invalid_filter = MetadataFilter(
            key="non_existent", value="something", operator=FilterOperator.EQ
        )
        query = VectorStoreQuery(
            query_embedding=fake_embedder("What will I get?"),
            similarity_top_k=2,
            filters=MetadataFilters(filters=[invalid_filter]),
        )
        result = indexed_store.query(query)  # must not raise
        assert result.nodes == []


# ═══════════════════════════════════════════════════════════════════════════
# Sparse integration tests
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestSparseIntegration:
    def test_hybrid_collection_end_to_end(self, store_factory, fake_embedder):
        """A dense+endee_bm25 hybrid collection must accept add() and return hits."""
        pytest.importorskip("endee_model")
        fields = [
            {
                "name": "dense",
                "type": "vector",
                "params": {
                    "dimension": fake_embedder.dim,
                    "space_type": "cosine",
                    "precision": "int8",
                },
            },
            {"name": "sparse", "type": "sparse", "sparse_model": "endee_bm25"},
        ]
        vs = store_factory(fields=fields)
        assert vs.hybrid is True

        text = "Vector databases optimize similarity search."
        node = TextNode(text=text, embedding=fake_embedder(text), id_="hybrid1")
        vs.add([node])

        query = VectorStoreQuery(
            query_embedding=fake_embedder("similarity search"),
            query_str="similarity search",
            similarity_top_k=3,
        )
        result = vs.query(query)
        assert isinstance(result.nodes, list)

    def test_default_sparse_model_hybrid_search_fuses_results(
        self, store_factory, fake_embedder
    ):
        """A custom (non-bm25) sparse_embedding must drive real RRF fusion live."""
        fields = [
            {
                "name": "dense",
                "type": "vector",
                "params": {
                    "dimension": fake_embedder.dim,
                    "space_type": "cosine",
                    "precision": "int8",
                },
            },
            {"name": "sparse", "type": "sparse", "sparse_model": "default"},
        ]
        vs = store_factory(
            fields=fields, sparse_embedding=_DeterministicSparseEmbedding()
        )
        assert vs.hybrid is True

        text = "Kubernetes orchestrates containers at scale."
        node = TextNode(text=text, embedding=fake_embedder(text), id_="hybrid_default")
        vs.add([node])

        query = VectorStoreQuery(
            query_embedding=fake_embedder("container orchestration"),
            query_str="Kubernetes orchestrates containers at scale.",
            similarity_top_k=3,
        )
        result = vs.query(query, dense_rrf_weight=0.5, rrf_rank_constant=60)
        assert "hybrid_default" in result.ids


# ═══════════════════════════════════════════════════════════════════════════
# Retrieval integration tests (exercises real LlamaIndex framework objects)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestRetrieval:
    # ── Fixture ──────────────────────────────────────────────────────────

    @pytest.fixture
    def retrieval_index(
        self, store_factory, sample_documents, fake_embedder, fake_embed_model
    ):
        """A live-server VectorStoreIndex over sample_documents, for retrieval tests."""
        vector_store = store_factory(dimension=fake_embedder.dim, space_type="cosine")
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Disable LLM to focus on vector store testing.
        Settings.llm = None

        index = VectorStoreIndex.from_documents(
            sample_documents,
            storage_context=storage_context,
            embed_model=fake_embed_model,
        )

        yield vector_store, index, fake_embed_model

    def test_custom_retriever_with_metadata_filter(self, retrieval_index):
        """A filtered VectorIndexRetriever must return matching, scored nodes only."""
        _vector_store, index, embed_model = retrieval_index
        ai_filter = MetadataFilter(
            key="category", value="ai", operator=FilterOperator.EQ
        )
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=3,
            filters=MetadataFilters(filters=[ai_filter]),
        )

        nodes = retriever.retrieve("What is deep learning?")

        assert len(nodes) > 0
        assert len(nodes) <= 3
        for node in nodes:
            assert node.score is not None
            assert "ai" in str(node.node.metadata.get("category", "")).lower()
            assert node.node.text

    def test_custom_query_engine_with_metadata_filter(self, retrieval_index):
        """A query engine on a filtered retriever returns only matching source nodes."""
        _vector_store, index, embed_model = retrieval_index
        ai_filter = MetadataFilter(
            key="category", value="ai", operator=FilterOperator.EQ
        )
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=3,
            filters=MetadataFilters(filters=[ai_filter]),
        )
        query_engine = RetrieverQueryEngine.from_args(retriever=retriever, verbose=True)

        response = query_engine.query(
            "Explain the difference between machine learning and deep learning"
        )

        assert hasattr(response, "response")
        source_nodes = response.source_nodes
        assert len(source_nodes) > 0
        for node in source_nodes:
            assert "ai" in str(node.metadata.get("category", "")).lower()

    def test_direct_vectorstore_query_with_filter(self, retrieval_index):
        """A direct filtered VectorStoreQuery must return matching nodes with scores."""
        vector_store, _index, embed_model = retrieval_index
        db_filter = MetadataFilter(
            key="category", value="database", operator=FilterOperator.EQ
        )
        query = VectorStoreQuery(
            query_embedding=embed_model.get_text_embedding(
                "What are vector databases?"
            ),
            similarity_top_k=2,
            filters=MetadataFilters(filters=[db_filter]),
        )

        result = vector_store.query(query)

        assert len(result.nodes) > 0
        assert len(result.nodes) <= 2
        assert len(result.nodes) == len(result.similarities)
        for node, score in zip(result.nodes, result.similarities):
            assert "database" in str(node.metadata.get("category", "")).lower()
            assert isinstance(score, (int, float))

    # ── query.query_str back-compat ─────────────────────────────────────────

    def test_query_with_query_str_is_accepted(self, retrieval_index):
        """VectorStoreQuery must accept query_str alongside query_embedding."""
        vector_store, _index, embed_model = retrieval_index
        query_text = "Python programming language"
        query = VectorStoreQuery(
            query_embedding=embed_model.get_text_embedding(query_text),
            query_str=query_text,
            similarity_top_k=3,
        )
        assert query.query_str == query_text

        result = vector_store.query(query)
        assert len(result.nodes) > 0

    def test_query_without_query_str_still_works(self, retrieval_index):
        """query() must still work when query_str is omitted."""
        vector_store, _index, embed_model = retrieval_index
        query = VectorStoreQuery(
            query_embedding=embed_model.get_text_embedding("test query"),
            similarity_top_k=3,
        )
        assert query.query_str is None

        result = vector_store.query(query)
        assert len(result.nodes) > 0

    def test_query_str_attribute_access_is_safe_via_getattr(self, retrieval_index):
        """getattr(query, 'query_str', None) must be safe whether or not it was set."""
        _vector_store, _index, embed_model = retrieval_index
        emb = embed_model.get_text_embedding("test")

        query_with_str = VectorStoreQuery(
            query_embedding=emb, query_str="test text", similarity_top_k=1
        )
        assert getattr(query_with_str, "query_str", None) == "test text"

        query_without_str = VectorStoreQuery(query_embedding=emb, similarity_top_k=1)
        assert getattr(query_without_str, "query_str", None) is None

"""Integration tests: all tests that require a live/local Endee server
(set `ENDEE_API_TOKEN`).

`TestVectorStoreIntegration` covers live CRUD, config validation (incl.
mismatch detection on reconnect), filter assertions, client construction,
collection lifecycle, and factory-method coverage for `EndeeVectorStore`
against a real server.

`TestMultiFieldIntegration` covers a live collection with separate title,
content, and keywords fields.

`TestSparseIntegration` covers live hybrid/BM25 auto-detection and search
against a real server.

`TestMultiVectorIntegration` covers a live collection with a dense field
and a multi_vector field.
"""

from __future__ import annotations

import os

import pytest
from conftest import (
    ALL_SPACE_TYPES,
    DENSE_FIELD,
    DIMENSION,
    METADATAS,
    TEXTS,
    FakeRawSparseModel,
    safe_delete,
    uid,
)
from langchain_core.documents import Document

from langchain_endee import EndeeVectorStore, RetrievalMode
from langchain_endee.vectorstores import EndeeVectorStoreError


def _new_simple_store(live_client, fake_embedder, name, **kwargs):
    """Build a simple-mode (dimension=) live store, defaulting dimension/recreate."""
    kwargs.setdefault("dimension", DIMENSION)
    kwargs.setdefault("force_recreate", True)
    return EndeeVectorStore(
        embedding=fake_embedder,
        endee_client=live_client,
        collection_name=name,
        **kwargs,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Vector store integration tests
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestVectorStoreIntegration:
    # ── Fixture ──────────────────────────────────────────────────────────

    @pytest.fixture(scope="class")
    def vector_store_dense_store(self, live_client, fake_embedder):
        """Own live dense-only collection with TEXTS, shared across the class."""
        name = uid()
        store = EndeeVectorStore(
            embedding=fake_embedder,
            endee_client=live_client,
            collection_name=name,
            fields=[DENSE_FIELD],
            force_recreate=True,
        )
        ids = store.add_texts(
            texts=TEXTS, metadatas=METADATAS, ids=[f"d{i}" for i in range(len(TEXTS))]
        )
        yield store, ids
        safe_delete(live_client, name)

    # ── Properties ────────────────────────────────────────────────────────

    def test_properties(self, vector_store_dense_store):
        """embeddings, client, and collection must be non-None on a connected store."""
        store, _ = vector_store_dense_store
        assert store.embeddings is not None
        assert store.client is not None
        assert store.collection is not None

    def test_field_map(self, vector_store_dense_store):
        """field_map must expose the dense field with type 'vector'."""
        store, _ = vector_store_dense_store
        fm = store.field_map
        assert "dense" in fm
        assert fm["dense"]["type"] == "vector"

    # ── similarity_search ─────────────────────────────────────────────────

    def test_similarity_search(self, vector_store_dense_store):
        """similarity_search must return up to k Document results."""
        store, _ = vector_store_dense_store
        results = store.similarity_search("programming language", k=3)
        assert 0 < len(results) <= 3
        assert isinstance(results[0], Document)

    def test_similarity_search_with_score(self, vector_store_dense_store):
        """similarity_search_with_score must return (Document, float) pairs."""
        store, _ = vector_store_dense_store
        results = store.similarity_search_with_score("machine learning", k=3)
        assert len(results) > 0
        doc, score = results[0]
        assert isinstance(doc, Document)
        assert isinstance(score, float)

    def test_similarity_search_by_object(self, vector_store_dense_store, fake_embedder):
        """similarity_search_by_object must return results for a raw vector."""
        store, _ = vector_store_dense_store
        vec = fake_embedder.embed_query("database")
        results = store.similarity_search_by_object(vec, k=2)
        assert len(results) > 0

    def test_similarity_search_by_object_with_score(
        self, vector_store_dense_store, fake_embedder
    ):
        """Must return (Document, float) pairs given a raw embedding vector."""
        store, _ = vector_store_dense_store
        vec = fake_embedder.embed_query("neural networks")
        results = store.similarity_search_by_object_with_score(vec, k=2)
        assert len(results) > 0
        _, score = results[0]
        assert isinstance(score, float)

    def test_ef_parameter(self, vector_store_dense_store):
        """similarity_search must accept an ef override and still return results."""
        store, _ = vector_store_dense_store
        results = store.similarity_search("Python", k=2, ef=256)
        assert len(results) > 0

    def test_prefilter_and_boost(self, vector_store_dense_store):
        """similarity_search must accept prefilter_cardinality_threshold and boost."""
        store, _ = vector_store_dense_store
        results = store.similarity_search(
            "database",
            k=2,
            filter=[{"category": {"$eq": "database"}}],
            prefilter_cardinality_threshold=1_000,
            filter_boost_percentage=20,
        )
        assert results is not None

    # ── get_by_ids ────────────────────────────────────────────────────────

    def test_get_by_ids(self, vector_store_dense_store):
        """get_by_ids must return documents for existing ids."""
        store, ids = vector_store_dense_store
        docs = store.get_by_ids(ids[:2])
        assert len(docs) == 2

    def test_get_by_ids_empty(self, vector_store_dense_store):
        """get_by_ids must return an empty list when given no ids."""
        store, _ = vector_store_dense_store
        assert store.get_by_ids([]) == []

    def test_get_by_ids_nonexistent(self, vector_store_dense_store):
        """get_by_ids must return an empty list for an id that does not exist."""
        store, _ = vector_store_dense_store
        assert store.get_by_ids(["fake_id_does_not_exist"]) == []

    # ── update_filters ────────────────────────────────────────────────────

    def test_update_filters(self, vector_store_dense_store):
        """update_filters must return a dict reporting an 'updated' count."""
        store, ids = vector_store_dense_store
        result = store.update_filters(
            [
                {
                    "id": ids[0],
                    "filter": {"category": "scripting", "language": "python"},
                },
            ]
        )
        assert isinstance(result, dict)
        assert "updated" in result

    def test_update_filters_empty_raises(self, vector_store_dense_store):
        """update_filters must raise ValueError when given an empty list."""
        store, _ = vector_store_dense_store
        with pytest.raises(ValueError):
            store.update_filters([])

    # ── Filters ───────────────────────────────────────────────────────────

    @pytest.fixture(scope="class")
    def filter_store(self, live_client, fake_embedder):
        """Own live dense-only collection with TEXTS, isolated from other tests."""
        name = uid()
        store = EndeeVectorStore(
            embedding=fake_embedder,
            endee_client=live_client,
            collection_name=name,
            fields=[DENSE_FIELD],
            force_recreate=True,
        )
        store.add_texts(
            texts=TEXTS, metadatas=METADATAS, ids=[f"d{i}" for i in range(len(TEXTS))]
        )
        yield store
        safe_delete(live_client, name)

    def test_filter_eq(self, filter_store):
        """The $eq operator must restrict results to docs matching the exact value."""
        results = filter_store.similarity_search(
            "language", k=5, filter=[{"category": {"$eq": "programming"}}]
        )
        assert len(results) > 0
        for doc in results:
            assert doc.metadata.get("category") == "programming"

    def test_filter_in(self, filter_store):
        """The $in operator must restrict results to docs whose value is in the list."""
        results = filter_store.similarity_search(
            "technology",
            k=5,
            filter=[{"difficulty": {"$in": ["beginner", "advanced"]}}],
        )
        assert len(results) > 0
        for doc in results:
            assert doc.metadata.get("difficulty") in ["beginner", "advanced"]

    def test_filter_multiple(self, filter_store):
        """Multiple filter dicts must combine with AND logic on a live server."""
        results = filter_store.similarity_search(
            "learning",
            k=5,
            filter=[{"category": {"$eq": "ai"}}, {"difficulty": {"$eq": "advanced"}}],
        )
        for doc in results:
            assert doc.metadata.get("category") == "ai"
            assert doc.metadata.get("difficulty") == "advanced"

    def test_filter_no_match(self, filter_store):
        """A filter matching no documents must return zero results on a live server."""
        results = filter_store.similarity_search(
            "anything", k=5, filter=[{"category": {"$eq": "nonexistent"}}]
        )
        assert len(results) == 0

    # ── as_retriever ──────────────────────────────────────────────────────

    def test_retriever(self, vector_store_dense_store):
        """as_retriever must return a retriever that yields documents via invoke."""
        store, _ = vector_store_dense_store
        retriever = store.as_retriever(search_kwargs={"k": 2})
        docs = retriever.invoke("machine learning")
        assert len(docs) > 0

    def test_retriever_with_filter(self, vector_store_dense_store):
        """A retriever built with a filter in search_kwargs must only return matches."""
        store, _ = vector_store_dense_store
        retriever = store.as_retriever(
            search_kwargs={"k": 3, "filter": [{"category": {"$eq": "ai"}}]}
        )
        docs = retriever.invoke("learning")
        for doc in docs:
            assert doc.metadata.get("category") == "ai"

    # ── Delete (own collections, independent) ────────────────────────────

    def test_delete_by_ids(self, live_client, fake_embedder):
        """delete(ids=...) on a live collection must remove only the specified ids."""
        name = uid()
        store = EndeeVectorStore(
            embedding=fake_embedder,
            endee_client=live_client,
            collection_name=name,
            fields=[DENSE_FIELD],
            force_recreate=True,
        )
        ids = store.add_texts(texts=TEXTS[:3], metadatas=METADATAS[:3])
        assert store.delete(ids=[ids[0]]) is True
        assert store.get_by_ids([ids[0]]) == []
        assert len(store.get_by_ids(ids[1:])) == 2

    def test_delete_by_filter(self, live_client, fake_embedder):
        """delete(filter=...) on a live collection must remove all matching entries."""
        name = uid()
        store = EndeeVectorStore(
            embedding=fake_embedder,
            endee_client=live_client,
            collection_name=name,
            fields=[DENSE_FIELD],
            force_recreate=True,
        )
        store.add_texts(texts=TEXTS, metadatas=METADATAS)
        assert store.delete(filter=[{"category": {"$eq": "programming"}}]) is True
        remaining = store.similarity_search(
            "programming", k=10, filter=[{"category": {"$eq": "programming"}}]
        )
        assert len(remaining) == 0

    def test_delete_no_params_raises(self, live_client, fake_embedder):
        """delete must raise ValueError when called without ids or filter."""
        name = uid()
        store = EndeeVectorStore(
            embedding=fake_embedder,
            endee_client=live_client,
            collection_name=name,
            fields=[DENSE_FIELD],
            force_recreate=True,
        )
        with pytest.raises(ValueError):
            store.delete()

    # ── Factory methods ───────────────────────────────────────────────────

    def test_from_texts(self, live_client, fake_embedder):
        """from_texts against a live server must create a searchable collection."""
        name = uid()
        store = EndeeVectorStore.from_texts(
            texts=TEXTS[:3],
            embedding=fake_embedder,
            metadatas=METADATAS[:3],
            endee_client=live_client,
            collection_name=name,
            dimension=DIMENSION,
            force_recreate=True,
        )
        assert len(store.similarity_search("Python", k=2)) > 0

    def test_from_documents(self, live_client, fake_embedder):
        """from_documents against a live server must create a searchable collection."""
        name = uid()
        docs = [
            Document(page_content=t, metadata=m)
            for t, m in zip(TEXTS[:3], METADATAS[:3])
        ]
        store = EndeeVectorStore.from_documents(
            documents=docs,
            embedding=fake_embedder,
            endee_client=live_client,
            collection_name=name,
            dimension=DIMENSION,
            force_recreate=True,
        )
        assert len(store.similarity_search("Rust", k=2)) > 0

    def test_from_existing_collection(self, live_client, fake_embedder):
        """from_existing_collection must reconnect and allow searching it."""
        name = uid()
        EndeeVectorStore.from_texts(
            texts=TEXTS[:2],
            embedding=fake_embedder,
            metadatas=METADATAS[:2],
            endee_client=live_client,
            collection_name=name,
            dimension=DIMENSION,
            force_recreate=True,
        )
        store2 = EndeeVectorStore.from_existing_collection(
            collection_name=name,
            embedding=fake_embedder,
            endee_client=live_client,
        )
        assert store2.collection_name == name
        assert len(store2.similarity_search("Python", k=2)) > 0

    # ── Collection config / force_recreate ───────────────────────────────

    def test_force_recreate(self, live_client, fake_embedder):
        """force_recreate must wipe existing data when reusing a collection name."""
        name = uid()
        store1 = EndeeVectorStore(
            embedding=fake_embedder,
            endee_client=live_client,
            collection_name=name,
            fields=[DENSE_FIELD],
            force_recreate=True,
        )
        store1.add_texts(texts=TEXTS[:2])
        store2 = EndeeVectorStore(
            embedding=fake_embedder,
            endee_client=live_client,
            collection_name=name,
            fields=[DENSE_FIELD],
            force_recreate=True,
        )
        assert len(store2.similarity_search("Python", k=5)) == 0

    def test_custom_ef_con(self, live_client, fake_embedder):
        """Custom ef_con and M values passed via fields= must show up in describe()."""
        name = uid()
        store = EndeeVectorStore(
            embedding=fake_embedder,
            endee_client=live_client,
            collection_name=name,
            fields=[
                {
                    "name": "dense",
                    "type": "vector",
                    "params": {
                        "dimension": DIMENSION,
                        "space_type": "cosine",
                        "precision": "int8",
                        "M": 32,
                        "ef_con": 256,
                    },
                }
            ],
            force_recreate=True,
        )
        info = store.collection.describe()
        dense = next(f for f in info["fields"] if f["type"] == "vector")
        assert dense["params"]["ef_con"] == 256
        assert dense["params"]["M"] == 32

    def test_missing_embedding_raises(self, live_client):
        """EndeeVectorStore must raise ValueError when embedding is None."""
        with pytest.raises(ValueError):
            EndeeVectorStore(embedding=None, collection_name="x", fields=[DENSE_FIELD])

    def test_missing_collection_name_raises(self, live_client, fake_embedder):
        """EndeeVectorStore must raise ValueError when collection_name is None."""
        with pytest.raises(ValueError):
            EndeeVectorStore(
                embedding=fake_embedder, collection_name=None, fields=[DENSE_FIELD]
            )

    def test_invalid_params_raise(self, live_client, fake_embedder):
        """A negative dimension field param must raise EndeeVectorStoreError."""
        name = uid()
        with pytest.raises(EndeeVectorStoreError):
            EndeeVectorStore(
                embedding=fake_embedder,
                endee_client=live_client,
                collection_name=name,
                fields=[
                    {
                        "name": "dense",
                        "type": "vector",
                        "params": {
                            "dimension": -1,
                            "space_type": "cosine",
                            "precision": "int8",
                        },
                    }
                ],
            )

    def test_init_builds_client_from_api_token_and_base_url(
        self, live_client, fake_embedder
    ):
        """A store built from api_token/base_url (not endee_client=) must connect."""
        name = uid()
        store = EndeeVectorStore(
            embedding=fake_embedder,
            collection_name=name,
            fields=[DENSE_FIELD],
            api_token=os.environ["ENDEE_API_TOKEN"],
            base_url=os.environ.get("ENDEE_BASE_URL"),
            force_recreate=True,
        )
        try:
            assert store.collection.describe()["name"] == name
        finally:
            safe_delete(live_client, name)

    @pytest.mark.parametrize("space_type", ALL_SPACE_TYPES)
    def test_simple_mode_with_each_space_type(
        self, live_client, fake_embedder, space_type
    ):
        """Simple-mode (dimension=) construction must work with every space_type."""
        name = uid()
        store = _new_simple_store(
            live_client, fake_embedder, name, space_type=space_type
        )
        try:
            store.add_texts(texts=["Rust guarantees memory safety."])
            results = store.similarity_search("memory safety", k=3)
            assert len(results) > 0
        finally:
            safe_delete(live_client, name)

    def test_validate_collection_config_raises_on_dimension_mismatch(
        self, live_client, fake_embedder
    ):
        """Reconnecting with a different dimension must raise, based on describe()."""
        name = uid()
        _new_simple_store(live_client, fake_embedder, name)
        try:
            with pytest.raises(EndeeVectorStoreError, match="dimension"):
                _new_simple_store(
                    live_client,
                    fake_embedder,
                    name,
                    dimension=DIMENSION + 1,
                    force_recreate=False,
                )
        finally:
            safe_delete(live_client, name)

    def test_validate_collection_config_does_not_warn_on_default_precision(
        self, live_client, fake_embedder, caplog
    ):
        """Reconnecting with the default (enum) precision must not warn falsely."""
        name = uid()
        _new_simple_store(live_client, fake_embedder, name)
        try:
            with caplog.at_level("WARNING"):
                _new_simple_store(
                    live_client, fake_embedder, name, force_recreate=False
                )
            assert "precision" not in caplog.text
        finally:
            safe_delete(live_client, name)

    def test_validate_collection_config_does_not_warn_on_matching_string_precision(
        self, live_client, fake_embedder, caplog
    ):
        """Reconnecting with a matching plain-string precision must not warn."""
        name = uid()
        _new_simple_store(live_client, fake_embedder, name, precision="int8")
        try:
            with caplog.at_level("WARNING"):
                _new_simple_store(
                    live_client,
                    fake_embedder,
                    name,
                    precision="int8",
                    force_recreate=False,
                )
            assert "precision" not in caplog.text
        finally:
            safe_delete(live_client, name)

    def test_validate_collection_config_warns_on_real_precision_mismatch(
        self, live_client, fake_embedder, caplog
    ):
        """Reconnecting with a genuinely different precision must still warn."""
        name = uid()
        _new_simple_store(live_client, fake_embedder, name, precision="int8")
        try:
            with caplog.at_level("WARNING"):
                _new_simple_store(
                    live_client,
                    fake_embedder,
                    name,
                    precision="float32",
                    force_recreate=False,
                )
            assert "precision" in caplog.text
        finally:
            safe_delete(live_client, name)

    def test_validate_collection_config_passes_on_matching_reconnect(
        self, live_client, fake_embedder
    ):
        """Reconnecting with a matching config must not raise."""
        name = uid()
        _new_simple_store(live_client, fake_embedder, name, space_type="cosine")
        try:
            store = _new_simple_store(
                live_client,
                fake_embedder,
                name,
                space_type="cosine",
                force_recreate=False,
            )
            assert store.collection is not None
        finally:
            safe_delete(live_client, name)

    def test_simple_mode_with_sparse_embedding_enables_hybrid_search(
        self, live_client, fake_embedder, fake_sparse_embedding
    ):
        """Simple-mode (no fields=) construction with sparse_embedding goes hybrid."""
        name = uid()
        store = _new_simple_store(
            live_client,
            fake_embedder,
            name,
            sparse_embedding=fake_sparse_embedding,
            retrieval_mode=RetrievalMode.HYBRID,
        )
        try:
            field_map = store.field_map
            sparse_field = field_map[store.sparse_field_name]
            assert sparse_field["params"]["sparse_model"] == "default"

            store.add_texts(texts=["Kubernetes orchestrates containers."])
            results = store.similarity_search("Kubernetes containers", k=3)
            assert len(results) > 0
        finally:
            safe_delete(live_client, name)

    def test_custom_payload_keys_round_trip_live(self, live_client, fake_embedder):
        """Custom content/metadata payload keys must round-trip on a real server."""
        name = uid()
        store = EndeeVectorStore(
            embedding=fake_embedder,
            endee_client=live_client,
            collection_name=name,
            fields=[DENSE_FIELD],
            content_payload_key="body",
            metadata_payload_key="meta",
            force_recreate=True,
        )
        try:
            store.add_texts(
                texts=["Payload key round trip test."], metadatas=[{"tag": "x"}]
            )
            results = store.similarity_search("Payload key round trip", k=1)
            assert len(results) > 0
            assert results[0].metadata.get("tag") == "x"
        finally:
            safe_delete(live_client, name)

    def test_add_texts_batches_across_multiple_upserts(
        self, live_client, fake_embedder
    ):
        """add_texts with a small batch_size must upsert everything across batches."""
        name = uid()
        store = EndeeVectorStore(
            embedding=fake_embedder,
            endee_client=live_client,
            collection_name=name,
            fields=[DENSE_FIELD],
            force_recreate=True,
        )
        try:
            texts = [f"batching document number {i}" for i in range(12)]
            ids = store.add_texts(texts=texts, batch_size=5)
            assert len(ids) == 12
            info = store.collection.describe()
            assert info["total_elements"] == 12
        finally:
            safe_delete(live_client, name)


# ═══════════════════════════════════════════════════════════════════════════
# Multi-field integration tests
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestMultiFieldIntegration:
    # ── Fixture ──────────────────────────────────────────────────────────

    @pytest.fixture(scope="class")
    def multi_field_store(self, live_client, fake_embedder):
        """Live title/content/keywords collection, shared by the multi-field tests."""
        name = uid()
        store = EndeeVectorStore(
            embedding=fake_embedder,
            endee_client=live_client,
            collection_name=name,
            fields=[
                {
                    "name": "title",
                    "type": "vector",
                    "params": {
                        "dimension": DIMENSION,
                        "space_type": "cosine",
                        "precision": "int8",
                    },
                },
                {
                    "name": "content",
                    "type": "vector",
                    "params": {
                        "dimension": DIMENSION,
                        "space_type": "cosine",
                        "precision": "int8",
                    },
                },
                {"name": "keywords", "type": "sparse", "sparse_model": "default"},
            ],
            dense_field_name="title",
            sparse_field_name="keywords",
            force_recreate=True,
        )
        objects = []
        for i, (text, meta) in enumerate(zip(TEXTS, METADATAS)):
            title_vec = fake_embedder.embed_query(text.split()[0])
            content_vec = fake_embedder.embed_query(text)
            indices = sorted(set(hash(w) % 500 for w in text.lower().split()[:5]))
            values = [1.0 / (j + 1) for j in range(len(indices))]
            objects.append(
                {
                    "id": f"mf{i}",
                    "meta": {"text": text, "metadata": meta},
                    "filter": meta,
                    "fields": {
                        "title": title_vec,
                        "content": content_vec,
                        "keywords": {"indices": indices, "values": values},
                    },
                }
            )
        ids = store.add_objects(objects)
        yield store, ids
        safe_delete(live_client, name)

    def test_add_objects_returns_ids(self, multi_field_store):
        """add_objects must return ids in the same order the objects were added."""
        store, ids = multi_field_store
        assert ids == [f"mf{i}" for i in range(len(TEXTS))]

    def test_field_map(self, multi_field_store):
        """field_map must contain the title, content, and keywords fields."""
        store, _ = multi_field_store
        fm = store.field_map
        assert "title" in fm
        assert "content" in fm
        assert "keywords" in fm

    def test_standard_search_uses_primary_field(self, multi_field_store):
        """similarity_search must use dense_field_name as the primary search field."""
        store, _ = multi_field_store
        results = store.similarity_search("Python", k=3)
        assert len(results) > 0

    def test_multi_field_search_raw(self, multi_field_store, fake_embedder):
        """multi_field_search must return raw results keyed by each searched field."""
        store, _ = multi_field_store
        raw = store.multi_field_search(
            fields={
                "title": {"query": fake_embedder.embed_query("Python"), "limit": 3},
                "content": {
                    "query": fake_embedder.embed_query("programming"),
                    "limit": 3,
                },
            }
        )
        assert "title" in raw["results"]
        assert "content" in raw["results"]

    def test_multi_field_search_with_rerank(self, multi_field_store, fake_embedder):
        """multi_field_search_with_rerank returns reranked pairs capped at `limit`."""
        store, _ = multi_field_store
        results = store.multi_field_search_with_rerank(
            fields={
                "title": {"query": fake_embedder.embed_query("machine"), "limit": 10},
                "content": {
                    "query": fake_embedder.embed_query("learning"),
                    "limit": 10,
                },
            },
            limit=3,
            field_weights={"title": 0.4, "content": 0.6},
        )
        assert 0 < len(results) <= 3
        doc, score = results[0]
        assert isinstance(doc, Document)
        assert isinstance(score, float)

    def test_multi_field_search_with_filter(self, multi_field_store, fake_embedder):
        """multi_field_search must apply the filter to every returned hit."""
        store, _ = multi_field_store
        raw = store.multi_field_search(
            fields={
                "content": {"query": fake_embedder.embed_query("learning"), "limit": 5}
            },
            filter=[{"category": {"$eq": "ai"}}],
        )
        for hit in raw["results"].get("content", []):
            assert hit.get("filter", {}).get("category") == "ai"

    def test_get_by_ids(self, multi_field_store):
        """get_by_ids must return documents for existing multi-field ids."""
        store, _ = multi_field_store
        assert len(store.get_by_ids(["mf0", "mf1"])) == 2

    def test_delete_object(self, live_client, fake_embedder):
        """delete(ids=...) on a multi-field collection must remove only that object."""
        name = uid()
        store = EndeeVectorStore(
            embedding=fake_embedder,
            endee_client=live_client,
            collection_name=name,
            fields=[
                {
                    "name": "vec",
                    "type": "vector",
                    "params": {
                        "dimension": DIMENSION,
                        "space_type": "cosine",
                        "precision": "int8",
                    },
                }
            ],
            dense_field_name="vec",
            force_recreate=True,
        )
        vec = fake_embedder.embed_query("hello")
        store.add_objects(
            [
                {
                    "id": "x1",
                    "meta": {"text": "hello"},
                    "filter": {},
                    "fields": {"vec": vec},
                },
                {
                    "id": "x2",
                    "meta": {"text": "world"},
                    "filter": {},
                    "fields": {"vec": vec},
                },
            ]
        )
        store.delete(ids=["x1"])
        assert store.get_by_ids(["x1"]) == []
        assert len(store.get_by_ids(["x2"])) == 1


# ═══════════════════════════════════════════════════════════════════════════
# Sparse integration tests
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestSparseIntegration:
    # ── Fixture ──────────────────────────────────────────────────────────

    @pytest.fixture(scope="class")
    def hybrid_store(self, live_client, fake_embedder, fake_sparse_embedding):
        """Own live hybrid (dense + sparse) collection, shared across the class."""
        name = uid()
        store = EndeeVectorStore(
            embedding=fake_embedder,
            endee_client=live_client,
            collection_name=name,
            fields=[
                DENSE_FIELD,
                {"name": "sparse", "type": "sparse", "sparse_model": "default"},
            ],
            retrieval_mode=RetrievalMode.HYBRID,
            sparse_embedding=fake_sparse_embedding,
            force_recreate=True,
        )
        ids = store.add_texts(texts=TEXTS, metadatas=METADATAS)
        yield store, ids
        safe_delete(live_client, name)

    def test_hybrid_search(self, hybrid_store):
        """similarity_search on a hybrid store must fuse dense and sparse results."""
        store, _ = hybrid_store
        results = store.similarity_search("neural networks", k=3)
        assert len(results) > 0

    def test_hybrid_search_with_score(self, hybrid_store):
        """similarity_search_with_score on a hybrid store must return score pairs."""
        store, _ = hybrid_store
        results = store.similarity_search_with_score("vector embeddings", k=3)
        assert len(results) > 0
        _, score = results[0]
        assert isinstance(score, float)

    def test_hybrid_search_with_filter(self, hybrid_store):
        """similarity_search on a hybrid collection must respect the given filter."""
        store, _ = hybrid_store
        results = store.similarity_search(
            "learning", k=5, filter=[{"category": {"$eq": "ai"}}]
        )
        for doc in results:
            assert doc.metadata.get("category") == "ai"

    def test_hybrid_search_by_object(
        self, hybrid_store, fake_embedder, fake_sparse_embedding
    ):
        """Must return results given explicit dense and sparse vectors."""
        store, _ = hybrid_store
        dense_vec = fake_embedder.embed_query("deep learning")
        sparse_vec = fake_sparse_embedding.embed_query("deep learning")
        results = store.similarity_search_by_object_with_score(
            embedding=dense_vec,
            k=3,
            sparse_indices=sparse_vec.indices,
            sparse_values=sparse_vec.values,
        )
        assert len(results) > 0

    def test_hybrid_rrf_tuning(self, hybrid_store):
        """similarity_search_with_score must accept custom RRF tuning params."""
        store, _ = hybrid_store
        results = store.similarity_search_with_score(
            "Python programming",
            k=3,
            rrf_rank_constant=60,
            dense_rrf_weight=0.7,
        )
        assert len(results) > 0

    def test_hybrid_retriever(self, hybrid_store):
        """as_retriever on a hybrid store must return a retriever usable via invoke."""
        store, _ = hybrid_store
        retriever = store.as_retriever(search_kwargs={"k": 2})
        docs = retriever.invoke("similarity search")
        assert len(docs) > 0

    def test_hybrid_get_by_ids(self, hybrid_store):
        """get_by_ids must return the document for an id from a hybrid collection."""
        store, ids = hybrid_store
        docs = store.get_by_ids([ids[0]])
        assert len(docs) == 1

    def test_hybrid_properties(self, hybrid_store):
        """A hybrid store reports mode HYBRID with non-None sparse_embeddings."""
        store, _ = hybrid_store
        assert store.retrieval_mode == RetrievalMode.HYBRID
        assert store.sparse_embeddings is not None

    def test_endee_bm25_auto_detect(self, live_client, fake_embedder):
        """A live endee_bm25 sparse field must auto-detect HYBRID mode and search."""
        name = uid()
        store = EndeeVectorStore(
            embedding=fake_embedder,
            endee_client=live_client,
            collection_name=name,
            fields=[
                DENSE_FIELD,
                {"name": "sparse", "type": "sparse", "sparse_model": "endee_bm25"},
            ],
            force_recreate=True,
        )
        assert store.retrieval_mode == RetrievalMode.HYBRID
        assert store.sparse_embeddings is not None

        store.add_texts(texts=TEXTS[:3], metadatas=METADATAS[:3])
        results = store.similarity_search("Python", k=2)
        assert len(results) > 0

    def test_raw_sparse_model_auto_wrap(self, live_client, fake_embedder):
        """A raw sparse model in a live store must auto-wrap and stay searchable."""
        name = uid()
        raw_model = FakeRawSparseModel()
        store = EndeeVectorStore(
            embedding=fake_embedder,
            endee_client=live_client,
            collection_name=name,
            fields=[
                DENSE_FIELD,
                {"name": "sparse", "type": "sparse", "sparse_model": "default"},
            ],
            retrieval_mode=RetrievalMode.HYBRID,
            sparse_embedding=raw_model,
            force_recreate=True,
        )
        assert store.sparse_embeddings is not None

        store.add_texts(texts=TEXTS[:3], metadatas=METADATAS[:3])
        results = store.similarity_search("Python", k=2)
        assert len(results) > 0

    def test_from_existing_endee_bm25_auto_reconnect(self, live_client, fake_embedder):
        """Reconnecting to a live endee_bm25 collection must auto-detect HYBRID mode."""
        name = uid()
        store1 = EndeeVectorStore(
            embedding=fake_embedder,
            endee_client=live_client,
            collection_name=name,
            fields=[
                DENSE_FIELD,
                {"name": "sparse", "type": "sparse", "sparse_model": "endee_bm25"},
            ],
            force_recreate=True,
        )
        store1.add_texts(texts=TEXTS[:3], metadatas=METADATAS[:3])

        store2 = EndeeVectorStore.from_existing_collection(
            collection_name=name,
            embedding=fake_embedder,
            endee_client=live_client,
        )
        assert store2.retrieval_mode == RetrievalMode.HYBRID
        results = store2.similarity_search("Python", k=2)
        assert len(results) > 0


# ═══════════════════════════════════════════════════════════════════════════
# Multi-vector field integration tests
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestMultiVectorIntegration:
    # ── Fixture ──────────────────────────────────────────────────────────

    @pytest.fixture(scope="class")
    def multi_vector_store(self, live_client, fake_embedder):
        """Own live dense + multi_vector collection, shared across the class."""
        name = uid()
        fields_def = [
            DENSE_FIELD,
            {
                "name": "chunks",
                "type": "multi_vector",
                "params": {
                    "dimension": DIMENSION,
                    "space_type": "cosine",
                    "precision": "float16",
                    "pooling": "mean",
                },
            },
        ]
        store = EndeeVectorStore(
            embedding=fake_embedder,
            endee_client=live_client,
            collection_name=name,
            fields=fields_def,
            force_recreate=True,
        )
        objects = [
            {
                "id": f"mv{i}",
                "meta": {"text": text},
                "filter": meta,
                "fields": {
                    "dense": fake_embedder.embed_query(text),
                    "chunks": [
                        fake_embedder.embed_query(word) for word in text.split()[:3]
                    ],
                },
            }
            for i, (text, meta) in enumerate(zip(TEXTS, METADATAS))
        ]
        ids = store.add_objects(objects)
        yield store, ids
        safe_delete(live_client, name)

    def test_add_objects_and_multi_field_search_round_trip(
        self, multi_vector_store, fake_embedder
    ):
        """multi_field_search on a multi_vector field must return per-field results."""
        store, ids = multi_vector_store
        raw = store.multi_field_search(
            fields={
                "dense": {
                    "query": fake_embedder.embed_query("programming"),
                    "limit": 5,
                },
                "chunks": {
                    "query": [
                        fake_embedder.embed_query("python"),
                        fake_embedder.embed_query("language"),
                    ],
                    "limit": 5,
                },
            }
        )
        assert "dense" in raw["results"]
        assert "chunks" in raw["results"]
        assert len(raw["results"]["dense"]) > 0
        assert any(hit["id"] in ids for hit in raw["results"]["dense"])

    def test_field_map_reports_multi_vector_field(self, multi_vector_store):
        """field_map must expose the chunks field with type 'multi_vector'."""
        store, _ = multi_vector_store
        fm = store.field_map
        assert fm["chunks"]["type"] == "multi_vector"

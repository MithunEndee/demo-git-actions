"""Integration tests: all tests that require a live/local Endee server
(set `ENDEE_API_TOKEN`).

`TestVectorStoreIntegration` covers the full dense lifecycle, including
filtered search/delete, client construction, and collection lifecycle
(force_recreate, reconnect) against a live server. Each test sets up its
own state and can run independently, in any order.

`TestSparseIntegration` covers hybrid dense+sparse collections, including
a user-supplied sparse embedding, against a live server.

`TestMultiVectorIntegration` covers multi-vector field coverage
(add_objects + multi_field_search) against a live server.
"""

from __future__ import annotations

import os
import random
import uuid

import pytest
from conftest import (
    ALL_PRECISIONS,
    ALL_SPACE_TYPES,
    LIVE_DIM,
    LIVE_EMBEDDER_CONFIG,
    safe_delete,
    uid,
)
from endee import rerank

from crewai_endee.sparse_embeddings import SparseEmbeddings, SparseVector
from crewai_endee.vector_store import EndeeVectorStore


class _DeterministicSparseEmbedding(SparseEmbeddings):
    """Deterministic hash-based sparse embedder, for testing custom sparse models."""

    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text):
        indices = sorted({hash(w) % 500 for w in text.lower().split()})
        values = [1.0 / (i + 1) for i in range(len(indices))]
        return SparseVector(indices=indices, values=values)


def _dense_field_live(dim, space_type="cosine"):
    """Dense field config with a runtime dimension, unlike the fixed DENSE_FIELD."""
    return {
        "name": "dense",
        "type": "vector",
        "params": {"dimension": dim, "space_type": space_type, "precision": "int8"},
    }


def _embed_one(store, text):
    """Embed one text with store.embedder() and return a plain list of floats."""
    emb = store.embedder([text])[0]
    return emb.tolist() if hasattr(emb, "tolist") else emb


def _connect_dense_live(live_client, name, force_recreate=False):
    """Connect to (or create) a dense-only live collection with the given name."""
    return EndeeVectorStore(
        type=name,
        embedder_config=LIVE_EMBEDDER_CONFIG,
        endee_client=live_client,
        fields=[_dense_field_live(LIVE_DIM)],
        force_recreate=force_recreate,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Vector store
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestVectorStoreIntegration:
    """Live-server tests; each provisions its own state, none depend on run order."""

    EMBEDDER = LIVE_EMBEDDER_CONFIG
    DIM = LIVE_DIM

    @pytest.fixture
    def dense_store(self, live_client):
        """Fresh dense-only EndeeVectorStore, cleaned up after the test."""
        name = uid()
        store = EndeeVectorStore(
            type=name,
            embedder_config=self.EMBEDDER,
            endee_client=live_client,
            fields=[_dense_field_live(self.DIM)],
            force_recreate=True,
        )
        yield store
        safe_delete(live_client, name)

    @pytest.fixture
    def populated_dense_store(self, dense_store):
        """dense_store pre-populated with three programming-language documents."""
        dense_store.save(
            "Python is dynamically typed.", {"lang": "Python", "category": "scripting"}
        )
        dense_store.save(
            "Go has built-in concurrency.", {"lang": "Go", "category": "systems"}
        )
        dense_store.save(
            "Rust guarantees memory safety.", {"lang": "Rust", "category": "systems"}
        )
        return dense_store

    @pytest.mark.parametrize("precision", ALL_PRECISIONS)
    def test_create_collection_with_each_precision(self, live_client, precision):
        """Every Precision value must round-trip via create_collection()/describe()."""
        name = uid()
        store = EndeeVectorStore(
            type=name,
            embedder_config=self.EMBEDDER,
            endee_client=live_client,
            fields=[
                {
                    "name": "dense",
                    "type": "vector",
                    "params": {
                        "dimension": self.DIM,
                        "space_type": "cosine",
                        "precision": precision,
                    },
                },
            ],
            force_recreate=True,
        )
        try:
            info = store.describe()
            dense_field = next(f for f in info["fields"] if f["name"] == "dense")
            assert dense_field["params"]["precision"] == precision
        finally:
            safe_delete(live_client, name)

    def test_ensure_collection(self, dense_store):
        """ensure_collection() must return a non-None collection."""
        assert dense_store.ensure_collection() is not None

    def test_describe(self, dense_store):
        """describe() must report the store's own collection name."""
        info = dense_store.describe()
        assert info["name"] == dense_store.type

    def test_save_and_describe(self, populated_dense_store):
        """describe() must report total_elements matching the number saved."""
        info = populated_dense_store.describe()
        assert info["total_elements"] == 3

    def test_search(self, populated_dense_store):
        """search() against a live server must return non-empty results with content."""
        results = populated_dense_store.search("memory safe language", limit=3)
        assert len(results) > 0
        assert results[0]["content"]

    def test_search_with_filter(self, populated_dense_store):
        """search() with a category filter must return only the matching documents."""
        results = populated_dense_store.search(
            "language",
            limit=5,
            filter=[{"category": {"$eq": "systems"}}],
        )
        assert len(results) > 0
        for r in results:
            assert r["metadata"].get("category") == "systems"

    def test_search_score_threshold(self, populated_dense_store):
        """search() with score_threshold must only return results at or above it."""
        results = populated_dense_store.search(
            "concurrency", limit=5, score_threshold=0.4
        )
        assert all(r["score"] >= 0.4 for r in results)

    def test_search_ef_search(self, populated_dense_store):
        """search() must accept a custom ef_search value and still return a list."""
        results = populated_dense_store.search("typing", limit=2, ef_search=256)
        assert isinstance(results, list)

    def test_search_include_vectors(self, populated_dense_store):
        """search() with include_vectors=True must return the full dense vector."""
        results = populated_dense_store.search("Python", limit=1, include_vectors=True)
        assert "vectors" in results[0]
        assert "dense" in results[0]["vectors"]
        assert len(results[0]["vectors"]["dense"]) == self.DIM

    def test_get_objects(self, populated_dense_store):
        """get_objects() must return a list containing the matching object."""
        sid = populated_dense_store.search("Go", limit=1)[0]["id"]
        objs = populated_dense_store.get_objects([sid])
        assert len(objs) == 1
        assert objs[0]["id"] == sid

    def test_get_vector(self, populated_dense_store):
        """get_vector() must return the object matching a searched id."""
        sid = populated_dense_store.search("Rust", limit=1)[0]["id"]
        obj = populated_dense_store.get_vector(sid)
        assert obj["id"] == sid

    def test_update_filters(self, populated_dense_store):
        """update_filters() must persist the new filter value for the object."""
        sid = populated_dense_store.search("Go", limit=1)[0]["id"]
        result = populated_dense_store.update_filters(
            [{"id": sid, "filter": {"reviewed": "yes"}}]
        )
        assert result["updated"] == 1
        obj = populated_dense_store.get_vector(sid)
        assert obj["filter"]["reviewed"] == "yes"

    def test_delete_vector(self, populated_dense_store):
        """delete_vector() must remove the object so it can no longer be fetched."""
        sid = populated_dense_store.search("Go", limit=1)[0]["id"]
        result = populated_dense_store.delete_vector(sid)
        assert result["deleted"] == sid
        assert populated_dense_store.get_vector(sid) == {}

    def test_delete_by_filter(self, populated_dense_store):
        """delete() with a filter must remove matches, leaving the rest searchable."""
        result = populated_dense_store.delete(
            filter=[{"category": {"$eq": "scripting"}}]
        )
        assert result["deleted"] >= 1
        remaining = populated_dense_store.search("language", limit=10)
        assert all(r["metadata"].get("category") != "scripting" for r in remaining)

    def test_reset(self, dense_store):
        """reset() against a live server must clear _collection to None."""
        dense_store.reset()
        assert dense_store._collection is None

    def test_close(self, dense_store):
        """close() against a live server must clear _collection to None."""
        dense_store.close()
        assert dense_store._collection is None

    def test_init_builds_client_from_api_token_and_base_url(self, live_client):
        """A store built from api_token/base_url (not endee_client=) must connect."""
        name = uid()
        store = EndeeVectorStore(
            type=name,
            embedder_config=self.EMBEDDER,
            fields=[_dense_field_live(self.DIM)],
            api_token=os.environ["ENDEE_API_TOKEN"],
            base_url=os.environ.get("ENDEE_BASE_URL"),
            force_recreate=True,
        )
        try:
            assert store.describe()["name"] == name
        finally:
            safe_delete(live_client, name)

    def test_force_recreate_deletes_existing_data(self, live_client):
        """force_recreate=True on an existing collection must wipe prior data."""
        name = uid()
        first = _connect_dense_live(live_client, name, force_recreate=True)
        first.save("first generation data", {"gen": "1"})
        assert first.describe()["total_elements"] == 1

        second = _connect_dense_live(live_client, name, force_recreate=True)
        try:
            assert second.describe()["total_elements"] == 0
        finally:
            safe_delete(live_client, name)

    def test_reconnect_without_force_recreate_preserves_data(self, live_client):
        """Connecting to an existing collection without force_recreate keeps data."""
        name = uid()
        first = _connect_dense_live(live_client, name, force_recreate=True)
        first.save("data that must survive a reconnect", {"gen": "1"})

        second = _connect_dense_live(live_client, name)
        try:
            assert second.describe()["total_elements"] == 1
        finally:
            safe_delete(live_client, name)

    @pytest.mark.parametrize("space_type", ALL_SPACE_TYPES)
    def test_search_with_each_space_type(self, live_client, space_type):
        """save()/search() must work with every supported space_type."""
        name = uid()
        store = EndeeVectorStore(
            type=name,
            embedder_config=self.EMBEDDER,
            endee_client=live_client,
            fields=[_dense_field_live(self.DIM, space_type)],
            force_recreate=True,
        )
        try:
            store.save("Rust guarantees memory safety.", {"lang": "Rust"})
            results = store.search("memory safety", limit=3)
            assert len(results) > 0
        finally:
            safe_delete(live_client, name)

    def test_search_forwards_prefilter_and_filter_boost_to_server(
        self, populated_dense_store
    ):
        """The server must accept the prefilter/filter-boost search parameters."""
        results = populated_dense_store.search(
            "language",
            limit=5,
            prefilter_cardinality_threshold=1000,
            filter_boost_percentage=20,
        )
        assert isinstance(results, list)


# ═══════════════════════════════════════════════════════════════════════════
# Sparse
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestSparseIntegration:
    """Hybrid (dense + sparse) coverage against a live/local server."""

    EMBEDDER = LIVE_EMBEDDER_CONFIG
    DIM = LIVE_DIM

    @pytest.fixture
    def hybrid_store(self, live_client):
        """Fresh EndeeVectorStore with dense and endee_bm25 sparse fields."""
        name = uid()
        store = EndeeVectorStore(
            type=name,
            embedder_config=self.EMBEDDER,
            endee_client=live_client,
            fields=[
                _dense_field_live(self.DIM),
                {"name": "sparse", "type": "sparse", "sparse_model": "endee_bm25"},
            ],
            force_recreate=True,
        )
        yield store
        safe_delete(live_client, name)

    @pytest.fixture
    def default_sparse_store(self, live_client):
        """Fresh EndeeVectorStore with dense and default (non-bm25) sparse fields."""
        name = uid()
        store = EndeeVectorStore(
            type=name,
            embedder_config=self.EMBEDDER,
            endee_client=live_client,
            fields=[
                _dense_field_live(self.DIM),
                {"name": "sparse", "type": "sparse", "sparse_model": "default"},
            ],
            force_recreate=True,
        )
        yield store
        safe_delete(live_client, name)

    def test_hybrid_auto_sparse_setup(self, hybrid_store):
        """A store with an endee_bm25 sparse field must auto-configure embeddings."""
        assert hybrid_store._sparse_embeddings is not None

    def test_hybrid_save_and_search_ranks_exact_keyword_match_first(self, hybrid_store):
        """Hybrid search for an exact keyword ranks the document containing it first."""
        hybrid_store.save("Error code XJ-99-ZQ crashed the server.", {"type": "error"})
        hybrid_store.save("The weather is sunny today.", {"type": "weather"})
        results = hybrid_store.search("XJ-99-ZQ", limit=2)
        assert len(results) > 0
        assert "XJ-99-ZQ" in results[0]["content"]

    def test_hybrid_field_weights(self, hybrid_store):
        """search() with custom field_weights and rrf_k must return a results list."""
        hybrid_store.save("Error code XJ-99-ZQ crashed the server.", {"type": "error"})
        results = hybrid_store.search(
            "error",
            limit=2,
            field_weights={"dense": 0.3, "sparse": 0.7},
            rrf_k=30,
        )
        assert isinstance(results, list)

    def test_hybrid_search_include_vectors_includes_sparse(self, hybrid_store):
        """search() with include_vectors=True on a hybrid store must return sparses."""
        hybrid_store.save("Error code XJ-99-ZQ crashed the server.", {"type": "error"})
        results = hybrid_store.search("XJ-99-ZQ", limit=1, include_vectors=True)
        assert len(results) > 0
        assert "sparses" in results[0]

    def test_custom_sparse_embedding_round_trips_live(self, live_client):
        """A user-supplied sparse_embedding must drive save()/search() end to end."""
        name = uid()
        store = EndeeVectorStore(
            type=name,
            embedder_config=self.EMBEDDER,
            endee_client=live_client,
            fields=[
                _dense_field_live(self.DIM),
                {"name": "sparse", "type": "sparse", "sparse_model": "default"},
            ],
            sparse_embedding=_DeterministicSparseEmbedding(),
            force_recreate=True,
        )
        try:
            store.save("Kubernetes orchestrates containers.", {"topic": "k8s"})
            results = store.search("Kubernetes containers", limit=3)
            assert len(results) > 0
        finally:
            safe_delete(live_client, name)

    def test_default_sparse_no_auto_setup(self, default_sparse_store):
        """A store with a non-bm25 (default) sparse field leaves embeddings unset."""
        assert default_sparse_store._sparse_embeddings is None

    def test_default_sparse_add_objects_with_user_provided_vectors(
        self, default_sparse_store
    ):
        """add_objects() must accept a user sparse vector for a default sparse field."""
        emb = _embed_one(default_sparse_store, "Python ML")
        default_sparse_store.add_objects(
            [
                {
                    "id": uuid.uuid4().hex,
                    "meta": {"text": "Python ML", "metadata": {"lang": "Python"}},
                    "filter": {"lang": "Python"},
                    "fields": {
                        "dense": emb,
                        "sparse": {"indices": [10, 42], "values": [0.9, 0.5]},
                    },
                }
            ]
        )
        assert default_sparse_store.describe()["total_elements"] >= 1

    def test_default_sparse_dense_only_save_and_search_still_works(
        self, default_sparse_store
    ):
        """Plain save()/search() must work via dense when sparse is non-bm25."""
        default_sparse_store.save("Go microservices", {"lang": "Go"})
        results = default_sparse_store.search("microservices", limit=3)
        assert len(results) >= 1

    def test_default_sparse_multi_field_search_with_rerank(self, default_sparse_store):
        """multi_field_search() over dense and user sparse query fuses via rerank()."""
        default_sparse_store.save("Go microservices", {"lang": "Go"})
        emb = _embed_one(default_sparse_store, "ML")
        raw = default_sparse_store.multi_field_search(
            fields={
                "dense": {"query": emb, "limit": 3},
                "sparse": {"query": {"indices": [10], "values": [0.8]}, "limit": 3},
            }
        )
        fused = rerank(raw, limit=3)
        assert len(fused["results"]) >= 1


# ═══════════════════════════════════════════════════════════════════════════
# Multi-vector
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestMultiVectorIntegration:
    """add_objects and multi_field_search coverage over dense plus chunks fields."""

    EMBEDDER = LIVE_EMBEDDER_CONFIG
    DIM = LIVE_DIM

    @pytest.fixture
    def multi_vector_store(self, live_client):
        """Fresh EndeeVectorStore with dense and multi_vector fields, cleaned up."""
        name = uid()
        store = EndeeVectorStore(
            type=name,
            embedder_config=self.EMBEDDER,
            endee_client=live_client,
            fields=[
                _dense_field_live(self.DIM),
                {
                    "name": "chunks",
                    "type": "multi_vector",
                    "params": {
                        "dimension": self.DIM,
                        "space_type": "cosine",
                        "precision": "int8",
                        "pooling": "mean",
                    },
                },
            ],
            force_recreate=True,
        )
        yield store
        safe_delete(live_client, name)

    def test_add_objects_with_multi_vector_field(self, multi_vector_store):
        """add_objects() must accept an object with both dense and multi_vector data."""
        random.seed(42)
        text = "Distributed systems consensus"
        emb = _embed_one(multi_vector_store, text)
        mv = [[random.uniform(-1, 1) for _ in range(self.DIM)] for _ in range(3)]
        multi_vector_store.add_objects(
            [
                {
                    "id": uuid.uuid4().hex,
                    "meta": {"text": text, "metadata": {"topic": "distributed"}},
                    "filter": {"topic": "distributed"},
                    "fields": {"dense": emb, "chunks": mv},
                }
            ]
        )
        assert multi_vector_store.describe()["total_elements"] >= 1

    def test_dense_save_and_search_still_works_alongside_multi_vector_field(
        self, multi_vector_store
    ):
        """Plain save()/search() on dense works alongside a multi_vector field."""
        multi_vector_store.save("Python packaging guide", {"topic": "python"})
        results = multi_vector_store.search("packaging", limit=3)
        assert len(results) >= 1

    def test_multi_vector_include_vectors_includes_multi_vectors(
        self, multi_vector_store
    ):
        """search() with include_vectors=True must return the object's chunks too."""
        random.seed(7)
        emb = _embed_one(multi_vector_store, "consensus algorithms")
        mv = [[random.uniform(-1, 1) for _ in range(self.DIM)] for _ in range(2)]

        # Seed with the same vector as the query to guarantee a match.
        multi_vector_store.add_objects(
            [
                {
                    "id": uuid.uuid4().hex,
                    "meta": {"text": "consensus algorithms"},
                    "filter": {},
                    "fields": {"dense": emb, "chunks": mv},
                }
            ]
        )

        results = multi_vector_store.search(
            "consensus algorithms", limit=1, include_vectors=True
        )
        assert len(results) > 0
        assert "multi_vectors" in results[0]

    def test_multi_field_search_with_rerank_across_dense_and_chunks(
        self, multi_vector_store
    ):
        """multi_field_search() over dense and chunks must fuse via rerank()."""
        random.seed(99)
        emb = _embed_one(multi_vector_store, "consensus")
        mv = [[random.uniform(-1, 1) for _ in range(self.DIM)] for _ in range(2)]

        # Seed with the same vectors as the query to guarantee a match.
        multi_vector_store.add_objects(
            [
                {
                    "id": uuid.uuid4().hex,
                    "meta": {"text": "consensus"},
                    "filter": {},
                    "fields": {"dense": emb, "chunks": mv},
                }
            ]
        )

        raw = multi_vector_store.multi_field_search(
            fields={
                "dense": {"query": emb, "limit": 3},
                "chunks": {"query": mv, "limit": 3},
            }
        )
        fused = rerank(raw, limit=3, field_weights={"dense": 0.6, "chunks": 0.4})
        assert len(fused["results"]) >= 1

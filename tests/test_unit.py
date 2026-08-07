"""Unit tests: all tests that mock the `endee` client (no network).

`TestVectorStoreUnit` covers core CRUD surface of `EndeeVectorStore`:
init/config validation, add/from_texts/from_documents/from_existing_collection,
similarity search, batching/truncation, delete, update_filters, and
multi-field RRF wiring.

`TestFiltersUnit` covers filter/metadata-key translation and operator
support for `EndeeVectorStore`.

`TestSparseUnit` covers sparse/hybrid embeddings: `SparseVector` validation,
`SparseModelAdapter`, `wrap_sparse_model`, `EndeeModelSparse`, hybrid
auto-detection, RRF wiring, and async delegation
(`aembed_documents`/`aembed_query`). It mocks the `endee` client and the
`endee_model` package (no network, no real BM25 model).
"""

from __future__ import annotations

import logging
import sys
import types

import pytest
from conftest import (
    ALL_PRECISIONS,
    DENSE_FIELD,
    DIMENSION,
    METADATAS,
    TEXTS,
    FakeRawSparseModel,
    make_fake_sparse_result,
    uid,
)  # noqa: I001
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from pydantic import ValidationError

from langchain_endee import EndeeVectorStore, RetrievalMode
from langchain_endee.sparse_embeddings import (
    EndeeModelSparse,
    SparseEmbeddings,
    SparseModelAdapter,
    SparseVector,
    wrap_sparse_model,
)
from langchain_endee.vectorstores import EndeeVectorStoreError

# ═══════════════════════════════════════════════════════════════════════════
# Vector store unit tests
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestVectorStoreUnit:
    # ── Constructor validation ───────────────────────────────────────────

    def test_init_raises_on_missing_embedding(self, mock_endee_client):
        """EndeeVectorStore must raise ValueError when embedding is None."""
        with pytest.raises(ValueError):
            EndeeVectorStore(embedding=None, collection_name="x", fields=[DENSE_FIELD])

    def test_init_raises_on_missing_collection_name(
        self, mock_endee_client, fake_embedder
    ):
        """EndeeVectorStore must raise ValueError when collection_name is None."""
        with pytest.raises(ValueError):
            EndeeVectorStore(
                embedding=fake_embedder, collection_name=None, fields=[DENSE_FIELD]
            )

    def test_init_raises_on_missing_dimension_for_new_collection(
        self, mock_endee_client, fake_embedder
    ):
        """Creating a collection with no dimension/fields must raise ValueError."""
        with pytest.raises(ValueError):
            EndeeVectorStore(
                embedding=fake_embedder,
                collection_name=uid(),
                dimension=None,
                fields=None,
            )

    def test_init_propagates_network_failure(
        self, mock_endee_client, fake_embedder, monkeypatch
    ):
        """A list_collections network failure must propagate out of the constructor."""

        def _raise(*args, **kwargs):
            raise ConnectionError("could not reach Endee server")

        monkeypatch.setattr(mock_endee_client, "list_collections", _raise)
        with pytest.raises(ConnectionError):
            EndeeVectorStore(
                embedding=fake_embedder,
                collection_name=uid(),
                dimension=DIMENSION,
            )

    def test_init_with_endee_client_skips_client_creation(
        self, mock_endee_client, fake_embedder, monkeypatch
    ):
        """Passing an existing endee_client must skip EndeeClient construction."""

        def _fail_if_called(*args, **kwargs):
            raise AssertionError("EndeeClient() should not be constructed")

        monkeypatch.setattr("langchain_endee.vectorstores.EndeeClient", _fail_if_called)
        store = EndeeVectorStore(
            embedding=fake_embedder,
            collection_name=uid(),
            dimension=DIMENSION,
            endee_client=mock_endee_client,
        )
        assert store.client is mock_endee_client

    def test_init_with_fields_ignores_dimension_and_space_type(
        self, mock_endee_client, fake_embedder
    ):
        """An explicit fields= must take precedence over dimension/space_type kwargs."""
        name = uid()
        store = EndeeVectorStore(
            embedding=fake_embedder,
            collection_name=name,
            fields=[DENSE_FIELD],
            dimension=DIMENSION + 100,
            space_type="l2",
            force_recreate=True,
        )
        # The custom fields= definition wins; dimension/space_type params
        # passed alongside fields= are ignored for collection creation.
        fm = store.field_map
        assert fm["dense"]["params"]["dimension"] == DIMENSION
        assert fm["dense"]["params"]["space_type"] == "cosine"

    def test_init_force_recreate_on_nonexistent_collection_is_noop(
        self, mock_endee_client, fake_embedder, monkeypatch
    ):
        """force_recreate must not call delete_collection on a nonexistent one."""
        calls = []
        original = mock_endee_client.delete_collection

        def spy(name):
            calls.append(name)
            return original(name)

        monkeypatch.setattr(mock_endee_client, "delete_collection", spy)
        EndeeVectorStore(
            embedding=fake_embedder,
            collection_name=uid(),
            dimension=DIMENSION,
            force_recreate=True,
        )
        assert calls == []

    def test_sparse_embeddings_property_raises_when_not_set(
        self, mock_endee_client, fake_embedder
    ):
        """sparse_embeddings must raise ValueError when none is configured."""
        store = EndeeVectorStore(
            embedding=fake_embedder,
            collection_name=uid(),
            fields=[DENSE_FIELD],
            force_recreate=True,
        )
        with pytest.raises(ValueError, match="Sparse embeddings are not set"):
            _ = store.sparse_embeddings

    # ── _validate_collection_config ──────────────────────────────────────

    def test_validate_collection_config_dimension_mismatch_raises(
        self, mock_endee_client, fake_embedder
    ):
        """Reconnecting with a mismatched dimension must raise EndeeVectorStoreError."""
        name = uid()
        EndeeVectorStore(
            embedding=fake_embedder,
            collection_name=name,
            fields=[DENSE_FIELD],
            force_recreate=True,
        )
        with pytest.raises(EndeeVectorStoreError, match="dimension"):
            EndeeVectorStore(
                embedding=fake_embedder,
                collection_name=name,
                dimension=DIMENSION + 1,
                space_type="cosine",
            )

    def test_validate_collection_config_space_type_mismatch_raises(
        self, mock_endee_client, fake_embedder
    ):
        """A mismatched space_type on reconnect must raise EndeeVectorStoreError."""
        name = uid()
        EndeeVectorStore(
            embedding=fake_embedder,
            collection_name=name,
            fields=[DENSE_FIELD],
            force_recreate=True,
        )
        with pytest.raises(EndeeVectorStoreError, match="space_type"):
            EndeeVectorStore(
                embedding=fake_embedder,
                collection_name=name,
                dimension=DIMENSION,
                space_type="l2",
            )

    def test_validate_collection_config_hybrid_requested_on_dense_only_raises(
        self, mock_endee_client, fake_embedder, fake_sparse_embedding
    ):
        """Hybrid retrieval on dense-only must raise EndeeVectorStoreError."""
        from langchain_endee import RetrievalMode

        name = uid()
        EndeeVectorStore(
            embedding=fake_embedder,
            collection_name=name,
            fields=[DENSE_FIELD],
            force_recreate=True,
        )
        with pytest.raises(EndeeVectorStoreError, match="dense-only"):
            EndeeVectorStore(
                embedding=fake_embedder,
                collection_name=name,
                dimension=DIMENSION,
                retrieval_mode=RetrievalMode.HYBRID,
                sparse_embedding=fake_sparse_embedding,
            )

    def test_validate_collection_config_hybrid_requested_no_sparse_embedding_raises(
        self, mock_endee_client, fake_embedder
    ):
        """Hybrid without sparse_embedding must raise EndeeVectorStoreError."""
        from langchain_endee import RetrievalMode

        name = uid()
        EndeeVectorStore(
            embedding=fake_embedder,
            collection_name=name,
            fields=[
                DENSE_FIELD,
                {"name": "sparse", "type": "sparse", "sparse_model": "default"},
            ],
            force_recreate=True,
        )
        with pytest.raises(EndeeVectorStoreError, match="no sparse_embedding"):
            EndeeVectorStore(
                embedding=fake_embedder,
                collection_name=name,
                dimension=DIMENSION,
                retrieval_mode=RetrievalMode.HYBRID,
            )

    def test_validate_collection_config_precision_mismatch_warns_only(
        self, mock_endee_client, fake_embedder, caplog
    ):
        """A mismatched precision must log a warning rather than raise."""
        name = uid()
        EndeeVectorStore(
            embedding=fake_embedder,
            collection_name=name,
            fields=[DENSE_FIELD],
            force_recreate=True,
        )
        with caplog.at_level(logging.WARNING, logger="langchain_endee.vectorstores"):
            store = EndeeVectorStore(
                embedding=fake_embedder,
                collection_name=name,
                dimension=DIMENSION,
                precision="float32",
            )
        assert store is not None
        assert any("precision" in rec.message for rec in caplog.records)

    def test_validate_collection_config_m_and_ef_con_mismatch_warns_only(
        self, mock_endee_client, fake_embedder, caplog
    ):
        """Mismatched M and ef_con values must log warnings rather than raise."""
        name = uid()
        # Simple mode (dimension=) sets M/ef_con; raw fields= does not.
        EndeeVectorStore(
            embedding=fake_embedder,
            collection_name=name,
            dimension=DIMENSION,
            force_recreate=True,
        )
        with caplog.at_level(logging.WARNING, logger="langchain_endee.vectorstores"):
            store = EndeeVectorStore(
                embedding=fake_embedder,
                collection_name=name,
                dimension=DIMENSION,
                M=99,
                ef_con=999,
            )
        assert store is not None
        messages = " ".join(rec.message for rec in caplog.records)
        assert "M is" in messages
        assert "ef_con is" in messages

    def test_validate_collection_config_hybrid_available_dense_requested_warns_only(
        self, mock_endee_client, fake_embedder, fake_sparse_embedding, caplog
    ):
        """Reconnecting dense to a hybrid-capable collection warns, but stays dense."""
        name = uid()
        EndeeVectorStore(
            embedding=fake_embedder,
            collection_name=name,
            fields=[
                DENSE_FIELD,
                {"name": "sparse", "type": "sparse", "sparse_model": "default"},
            ],
            force_recreate=True,
        )
        with caplog.at_level(logging.WARNING, logger="langchain_endee.vectorstores"):
            store = EndeeVectorStore(
                embedding=fake_embedder,
                collection_name=name,
                dimension=DIMENSION,
            )
        assert store.retrieval_mode.value == "dense"
        messages = " ".join(rec.message for rec in caplog.records)
        assert "supports hybrid search" in messages

    def test_validate_collection_config_describe_failure_logs_and_returns(
        self, mock_endee_client, fake_embedder, monkeypatch, caplog
    ):
        """A describe() failure during config validation must warn, not raise."""
        name = uid()
        store = EndeeVectorStore(
            embedding=fake_embedder,
            collection_name=name,
            fields=[DENSE_FIELD],
            force_recreate=True,
        )

        def _raise():
            raise RuntimeError("describe exploded")

        monkeypatch.setattr(store.collection, "describe", _raise)
        with caplog.at_level(logging.WARNING, logger="langchain_endee.vectorstores"):
            # Calling directly since validate_collection_config runs at
            # construction time, before we can monkeypatch describe().
            store._validate_collection_config()
        assert any(
            "Could not fetch collection config" in r.message for r in caplog.records
        )

    # ── add_texts / from_texts / from_documents / from_existing_collection ──

    def test_add_texts_builds_correct_upsert_payload(
        self, mock_endee_client, fake_embedder
    ):
        """add_texts must upsert each text with its embedding, text, and metadata."""
        name = uid()
        store = EndeeVectorStore(
            embedding=fake_embedder,
            collection_name=name,
            fields=[DENSE_FIELD],
            force_recreate=True,
        )
        ids = store.add_texts(texts=TEXTS[:2], metadatas=METADATAS[:2], ids=["a", "b"])
        assert ids == ["a", "b"]

        stored = store.collection.get_objects(["a", "b"])
        assert len(stored) == 2
        for entry, text, meta in zip(stored, TEXTS[:2], METADATAS[:2]):
            assert entry["meta"]["text"] == text
            assert entry["meta"]["metadata"] == meta
            assert entry["filter"] == meta
            assert "dense" in entry["fields"]
            assert len(entry["fields"]["dense"]) == DIMENSION

    def test_add_texts_raises_on_id_length_mismatch(
        self, mock_endee_client, fake_embedder
    ):
        """add_texts must raise ValueError on an id/text count mismatch."""
        name = uid()
        store = EndeeVectorStore(
            embedding=fake_embedder,
            collection_name=name,
            fields=[DENSE_FIELD],
            force_recreate=True,
        )
        with pytest.raises(ValueError, match="ids"):
            store.add_texts(texts=TEXTS[:2], ids=["only-one"])

    def test_add_texts_raises_on_metadata_length_mismatch(
        self, mock_endee_client, fake_embedder
    ):
        """add_texts must raise ValueError on a metadata/text count mismatch."""
        name = uid()
        store = EndeeVectorStore(
            embedding=fake_embedder,
            collection_name=name,
            fields=[DENSE_FIELD],
            force_recreate=True,
        )
        with pytest.raises(ValueError, match="metadatas"):
            store.add_texts(texts=TEXTS[:2], metadatas=[{"only": "one"}])

    def test_from_texts_creates_and_populates_store(
        self, mock_endee_client, fake_embedder
    ):
        """from_texts must create the collection and upsert all provided texts."""
        name = uid()
        store = EndeeVectorStore.from_texts(
            texts=TEXTS[:3],
            embedding=fake_embedder,
            metadatas=METADATAS[:3],
            collection_name=name,
            dimension=DIMENSION,
            force_recreate=True,
        )
        assert (
            len(store.collection.get_objects(list(store.collection._store.keys()))) == 3
        )

    def test_from_documents_creates_and_populates_store(
        self, mock_endee_client, fake_embedder
    ):
        """from_documents must create the collection and upsert all given documents."""
        name = uid()
        docs = [
            Document(page_content=t, metadata=m)
            for t, m in zip(TEXTS[:3], METADATAS[:3])
        ]
        store = EndeeVectorStore.from_documents(
            documents=docs,
            embedding=fake_embedder,
            collection_name=name,
            dimension=DIMENSION,
            force_recreate=True,
        )
        assert len(store.collection._store) == 3

    def test_from_existing_collection_reconnects(
        self, mock_endee_client, fake_embedder
    ):
        """from_existing_collection must reconnect without re-adding existing data."""
        name = uid()
        EndeeVectorStore.from_texts(
            texts=TEXTS[:2],
            embedding=fake_embedder,
            metadatas=METADATAS[:2],
            collection_name=name,
            dimension=DIMENSION,
            force_recreate=True,
        )
        store2 = EndeeVectorStore.from_existing_collection(
            collection_name=name,
            embedding=fake_embedder,
        )
        assert store2.collection_name == name
        assert len(store2.collection._store) == 2

    def test_from_texts_raises_when_dimension_client_and_fields_all_missing(
        self, mock_endee_client, fake_embedder
    ):
        """from_texts must raise ValueError if dimension, client, fields are unset."""
        with pytest.raises(ValueError, match="dimension must be explicitly provided"):
            EndeeVectorStore.from_texts(
                texts=TEXTS[:2],
                embedding=fake_embedder,
                collection_name=uid(),
                dimension=None,
                endee_client=None,
                fields=None,
            )

    def test_create_collection_wraps_backend_error(
        self, mock_endee_client, fake_embedder, monkeypatch
    ):
        """A simple-mode creation backend error must wrap in EndeeVectorStoreError."""

        def _raise(name, fields):
            raise RuntimeError("create failed")

        monkeypatch.setattr(mock_endee_client, "create_collection", _raise)
        with pytest.raises(EndeeVectorStoreError, match="Failed to create"):
            EndeeVectorStore(
                embedding=fake_embedder,
                collection_name=uid(),
                dimension=DIMENSION,
            )

    def test_create_collection_raw_wraps_backend_error(
        self, mock_endee_client, fake_embedder, monkeypatch
    ):
        """A fields=-based creation backend error must wrap in EndeeVectorStoreError."""

        def _raise(name, fields):
            raise RuntimeError("create failed")

        monkeypatch.setattr(mock_endee_client, "create_collection", _raise)
        with pytest.raises(EndeeVectorStoreError, match="Failed to create"):
            EndeeVectorStore(
                embedding=fake_embedder,
                collection_name=uid(),
                fields=[DENSE_FIELD],
            )

    @pytest.mark.parametrize("precision", ALL_PRECISIONS)
    def test_precision_is_passed_through_to_field_config(
        self, mock_endee_client, fake_embedder, precision
    ):
        """Every supported precision value must reach the field config unchanged."""
        name = uid(f"precision_{precision}")
        store = EndeeVectorStore(
            embedding=fake_embedder,
            collection_name=name,
            dimension=DIMENSION,
            precision=precision,
            force_recreate=True,
        )
        assert (
            store.field_map[store.dense_field_name]["params"]["precision"] == precision
        )

    # ── add_texts empty inputs ─────────────────────────────────────────────

    def test_add_texts_with_empty_texts_returns_empty_list(
        self, mock_endee_client, fake_embedder
    ):
        """add_texts must return an empty list and add nothing for empty inputs."""
        name = uid()
        store = EndeeVectorStore(
            embedding=fake_embedder,
            collection_name=name,
            fields=[DENSE_FIELD],
            force_recreate=True,
        )
        ids = store.add_texts(texts=[], metadatas=[], ids=[])
        assert ids == []
        assert len(store.collection._store) == 0

    def test_add_texts_chunks_across_multiple_batches(
        self, mock_endee_client, fake_embedder
    ):
        """add_texts must upsert all texts even across multiple batch_size calls."""
        name = uid()
        store = EndeeVectorStore(
            embedding=fake_embedder,
            collection_name=name,
            fields=[DENSE_FIELD],
            force_recreate=True,
        )
        ids = store.add_texts(texts=TEXTS, metadatas=METADATAS, batch_size=2)
        assert len(ids) == len(TEXTS)
        assert len(store.collection._store) == len(TEXTS)

    # ── _validate_batch_size ──────────────────────────────────────────────

    def test_validate_batch_size_at_max_is_allowed(
        self, mock_endee_client, fake_embedder
    ):
        """_validate_batch_size must allow a size exactly at MAX_VECTORS_PER_BATCH."""
        from endee import constants

        name = uid()
        store = EndeeVectorStore(
            embedding=fake_embedder,
            collection_name=name,
            fields=[DENSE_FIELD],
            force_recreate=True,
        )
        assert store._validate_batch_size(constants.MAX_VECTORS_PER_BATCH) == (
            constants.MAX_VECTORS_PER_BATCH
        )

    def test_validate_batch_size_one_over_max_raises(
        self, mock_endee_client, fake_embedder
    ):
        """_validate_batch_size must raise ValueError one over MAX_VECTORS_PER_BATCH."""
        from endee import constants

        name = uid()
        store = EndeeVectorStore(
            embedding=fake_embedder,
            collection_name=name,
            fields=[DENSE_FIELD],
            force_recreate=True,
        )
        with pytest.raises(ValueError):
            store._validate_batch_size(constants.MAX_VECTORS_PER_BATCH + 1)

    def test_validate_batch_size_one_under_max_is_allowed(
        self, mock_endee_client, fake_embedder
    ):
        """_validate_batch_size must allow a size one under MAX_VECTORS_PER_BATCH."""
        from endee import constants

        name = uid()
        store = EndeeVectorStore(
            embedding=fake_embedder,
            collection_name=name,
            fields=[DENSE_FIELD],
            force_recreate=True,
        )
        assert store._validate_batch_size(constants.MAX_VECTORS_PER_BATCH - 1) == (
            constants.MAX_VECTORS_PER_BATCH - 1
        )

    def test_upsert_batch_logs_and_reraises_on_backend_error(
        self, mock_endee_client, fake_embedder, monkeypatch, caplog
    ):
        """A backend error during upsert must log the error, then re-raise it."""
        name = uid()
        store = EndeeVectorStore(
            embedding=fake_embedder,
            collection_name=name,
            fields=[DENSE_FIELD],
            force_recreate=True,
        )

        def _raise(entries):
            raise RuntimeError("upsert exploded")

        monkeypatch.setattr(store.collection, "upsert", _raise)
        with caplog.at_level(logging.ERROR, logger="langchain_endee.vectorstores"):
            with pytest.raises(RuntimeError, match="upsert exploded"):
                store.add_texts(texts=TEXTS[:1], metadatas=METADATAS[:1])
        assert any("Error upserting batch" in r.message for r in caplog.records)

    # ── EMBEDDING_MODEL_LIMITS / _truncate_text / _detect_embedding_model_type

    @pytest.mark.parametrize(
        ("class_name", "expected_type", "expected_limit"),
        [
            ("OpenAIEmbeddings", "openai", 8191),
            ("CohereEmbeddings", "cohere", 512),
            ("HuggingFaceEmbeddings", "huggingface", 512),
            ("SomeRandomEmbeddings", "default", 512),
        ],
        ids=["openai", "cohere", "huggingface", "default"],
    )
    def test_detect_embedding_model_type_and_truncation_limit(
        self, mock_endee_client, class_name, expected_type, expected_limit
    ):
        """Type/limit detection keys off class name; truncation stops at the limit."""
        # Covers both _detect_embedding_model_type (via class name) and
        # _truncate_text (at-limit text is untouched, over-limit text is cut).
        embedding_cls = type(
            class_name,
            (Embeddings,),
            {
                "embed_documents": lambda self, texts: [
                    [0.0] * DIMENSION for _ in texts
                ],
                "embed_query": lambda self, text: [0.0] * DIMENSION,
            },
        )
        embedding = embedding_cls()

        name = uid(f"detect_{expected_type}")
        store = EndeeVectorStore(
            embedding=embedding,
            collection_name=name,
            fields=[DENSE_FIELD],
            force_recreate=True,
        )
        assert store.embedding_model_type == expected_type
        assert store.max_text_length == expected_limit

        # Exactly at the limit (estimated_tokens == max_tokens) -> no truncation.
        at_limit_text = "a" * (expected_limit * 4)
        assert store._truncate_text(at_limit_text) == at_limit_text

        # One token over the limit -> truncated to max_chars.
        over_limit_text = "a" * ((expected_limit + 1) * 4)
        truncated = store._truncate_text(over_limit_text)
        expected_max_chars = int(expected_limit * 4 * 0.9)
        assert len(truncated) == expected_max_chars
        assert truncated != over_limit_text

    # ── similarity_search_with_score / similarity_search_by_object_with_score

    def test_similarity_search_by_object_with_score_orders_by_cosine(
        self, mock_endee_client, fake_embedder
    ):
        """similarity_search_by_object_with_score must rank by cosine similarity."""
        name = uid()
        store = EndeeVectorStore(
            embedding=fake_embedder,
            collection_name=name,
            fields=[DENSE_FIELD],
            force_recreate=True,
        )

        vec_a = [1.0] + [0.0] * (DIMENSION - 1)
        vec_b = [0.0, 1.0] + [0.0] * (DIMENSION - 2)
        vec_c = [0.9, 0.1] + [0.0] * (DIMENSION - 2)  # closer to a than b

        store.add_objects(
            [
                {
                    "id": "a",
                    "meta": {"text": "doc a"},
                    "filter": {},
                    "fields": {"dense": vec_a},
                },
                {
                    "id": "b",
                    "meta": {"text": "doc b"},
                    "filter": {},
                    "fields": {"dense": vec_b},
                },
                {
                    "id": "c",
                    "meta": {"text": "doc c"},
                    "filter": {},
                    "fields": {"dense": vec_c},
                },
            ]
        )

        results = store.similarity_search_by_object_with_score(vec_a, k=3)
        ids_in_order = [doc.metadata["_id"] for doc, _ in results]
        assert ids_in_order == ["a", "c", "b"]

        scores = [score for _, score in results]
        assert scores[0] == pytest.approx(1.0)
        assert scores[0] > scores[1] > scores[2]

    def test_similarity_search_with_score_matches_computed_ranking(
        self, mock_endee_client, fake_embedder
    ):
        """similarity_search_with_score must match an independent cosine ranking."""
        import numpy as np

        name = uid()
        store = EndeeVectorStore(
            embedding=fake_embedder,
            collection_name=name,
            fields=[DENSE_FIELD],
            force_recreate=True,
        )
        ids = store.add_texts(
            texts=TEXTS, metadatas=METADATAS, ids=[f"d{i}" for i in range(len(TEXTS))]
        )

        query = "vector database similarity"
        qvec = np.array(fake_embedder.embed_query(query))
        expected_scores = {}
        for doc_id, text in zip(ids, TEXTS):
            v = np.array(fake_embedder.embed_documents([text])[0])
            expected_scores[doc_id] = float(
                np.dot(qvec, v) / (np.linalg.norm(qvec) * np.linalg.norm(v))
            )
        expected_order = sorted(expected_scores, key=expected_scores.get, reverse=True)

        results = store.similarity_search_with_score(query, k=len(TEXTS))
        actual_order = [doc.metadata["_id"] for doc, _ in results]
        assert actual_order == expected_order
        for doc, score in results:
            assert score == pytest.approx(expected_scores[doc.metadata["_id"]])

    @pytest.mark.parametrize(
        "sparse_kwargs",
        [
            {"sparse_indices": [1, 2, 3]},
            {"sparse_values": [0.1, 0.2, 0.3]},
        ],
        ids=["only_indices", "only_values"],
    )
    def test_similarity_search_by_object_asymmetric_sparse_input_falls_back_to_dense(
        self,
        mock_endee_client,
        fake_embedder,
        fake_sparse_embedding,
        caplog,
        sparse_kwargs,
    ):
        """An indices-only or values-only input warns and falls back to dense-only."""
        from langchain_endee import RetrievalMode

        name = uid()
        store = EndeeVectorStore(
            embedding=fake_embedder,
            collection_name=name,
            fields=[
                DENSE_FIELD,
                {"name": "sparse", "type": "sparse", "sparse_model": "default"},
            ],
            retrieval_mode=RetrievalMode.HYBRID,
            sparse_embedding=fake_sparse_embedding,
            force_recreate=True,
        )
        store.add_texts(texts=TEXTS[:2], metadatas=METADATAS[:2])
        vec = fake_embedder.embed_query("python")
        with caplog.at_level(logging.WARNING, logger="langchain_endee.vectorstores"):
            results = store.similarity_search_by_object(vec, k=2, **sparse_kwargs)
        assert isinstance(results, list)
        assert any(
            "Falling back to dense-only search" in r.message for r in caplog.records
        )

    def test_results_to_docs_skips_entries_with_no_text(
        self, mock_endee_client, fake_embedder, caplog
    ):
        """_results_to_docs must skip entries lacking text, and log a warning."""
        name = uid()
        store = EndeeVectorStore(
            embedding=fake_embedder,
            collection_name=name,
            fields=[DENSE_FIELD],
            force_recreate=True,
        )
        raw_results = [
            {"id": "has-text", "meta": {"text": "hello"}, "similarity": 0.9},
            {"id": "no-text", "meta": {}, "similarity": 0.5},
        ]
        with caplog.at_level(logging.WARNING, logger="langchain_endee.vectorstores"):
            docs = store._results_to_docs(raw_results)
        assert len(docs) == 1
        assert docs[0][0].metadata["_id"] == "has-text"
        assert any("Skipping" in r.message for r in caplog.records)

    # ── delete() ──────────────────────────────────────────────────────────

    def test_delete_raises_when_neither_ids_nor_filter_given(
        self, mock_endee_client, fake_embedder
    ):
        """delete must raise ValueError when called without ids or filter."""
        name = uid()
        store = EndeeVectorStore(
            embedding=fake_embedder,
            collection_name=name,
            fields=[DENSE_FIELD],
            force_recreate=True,
        )
        with pytest.raises(ValueError):
            store.delete()

    def test_delete_by_ids_removes_matching(self, mock_endee_client, fake_embedder):
        """delete(ids=...) must remove only those ids, leaving the rest intact."""
        name = uid()
        store = EndeeVectorStore(
            embedding=fake_embedder,
            collection_name=name,
            fields=[DENSE_FIELD],
            force_recreate=True,
        )
        ids = store.add_texts(texts=TEXTS[:3], metadatas=METADATAS[:3])
        assert store.delete(ids=[ids[0]]) is True
        assert store.get_by_ids([ids[0]]) == []
        assert len(store.get_by_ids(ids[1:])) == 2

    def test_delete_by_ids_returns_false_if_any_fail(
        self, mock_endee_client, fake_embedder
    ):
        """delete(ids=...) must return False if any id fails to delete."""
        name = uid()
        store = EndeeVectorStore(
            embedding=fake_embedder,
            collection_name=name,
            fields=[DENSE_FIELD],
            force_recreate=True,
        )
        ids = store.add_texts(texts=TEXTS[:2], metadatas=METADATAS[:2])
        result = store.delete(ids=[ids[0], "does-not-exist"])
        assert result is False
        # The valid id was still deleted despite the overall False.
        assert store.get_by_ids([ids[0]]) == []

    def test_delete_by_filter_calls_delete_by_filter(
        self, mock_endee_client, fake_embedder
    ):
        """delete(filter=...) must remove all entries matching the filter."""
        name = uid()
        store = EndeeVectorStore(
            embedding=fake_embedder,
            collection_name=name,
            fields=[DENSE_FIELD],
            force_recreate=True,
        )
        store.add_texts(texts=TEXTS, metadatas=METADATAS)
        assert store.delete(filter=[{"category": {"$eq": "programming"}}]) is True
        remaining = store.similarity_search(
            "programming", k=10, filter=[{"category": {"$eq": "programming"}}]
        )
        assert remaining == []

    def test_delete_by_filter_returns_false_on_backend_error(
        self, mock_endee_client, fake_embedder, monkeypatch, caplog
    ):
        """delete(filter=...) must return False and log if delete_by_filter fails."""
        name = uid()
        store = EndeeVectorStore(
            embedding=fake_embedder,
            collection_name=name,
            fields=[DENSE_FIELD],
            force_recreate=True,
        )
        store.add_texts(texts=TEXTS[:2], metadatas=METADATAS[:2])

        def _raise(filter):  # noqa: A002
            raise RuntimeError("delete_by_filter exploded")

        monkeypatch.setattr(store.collection, "delete_by_filter", _raise)
        with caplog.at_level(logging.ERROR, logger="langchain_endee.vectorstores"):
            result = store.delete(filter=[{"category": {"$eq": "programming"}}])
        assert result is False
        assert any("Error during deletion" in r.message for r in caplog.records)

    # ── get_by_ids / update_filters ──────────────────────────────────────

    def test_get_by_ids_returns_empty_on_backend_error(
        self, mock_endee_client, fake_embedder, monkeypatch, caplog
    ):
        """get_by_ids must return empty and log a warning if get_objects fails."""
        name = uid()
        store = EndeeVectorStore(
            embedding=fake_embedder,
            collection_name=name,
            fields=[DENSE_FIELD],
            force_recreate=True,
        )
        ids = store.add_texts(texts=TEXTS[:1], metadatas=METADATAS[:1])

        def _raise(ids):
            raise RuntimeError("get_objects exploded")

        monkeypatch.setattr(store.collection, "get_objects", _raise)
        with caplog.at_level(logging.WARNING, logger="langchain_endee.vectorstores"):
            docs = store.get_by_ids(ids)
        assert docs == []
        assert any("Error retrieving documents" in r.message for r in caplog.records)

    def test_get_by_ids_empty_list_returns_empty(
        self, mock_endee_client, fake_embedder
    ):
        """get_by_ids must return an empty list when given an empty list of ids."""
        name = uid()
        store = EndeeVectorStore(
            embedding=fake_embedder,
            collection_name=name,
            fields=[DENSE_FIELD],
            force_recreate=True,
        )
        assert store.get_by_ids([]) == []

    def test_update_filters_raises_on_empty_updates(
        self, mock_endee_client, fake_embedder
    ):
        """update_filters must raise ValueError when given an empty list of updates."""
        name = uid()
        store = EndeeVectorStore(
            embedding=fake_embedder,
            collection_name=name,
            fields=[DENSE_FIELD],
            force_recreate=True,
        )
        with pytest.raises(ValueError):
            store.update_filters([])

    def test_update_filters_wraps_backend_error(
        self, mock_endee_client, fake_embedder, monkeypatch
    ):
        """A backend error in update_filters must wrap in EndeeVectorStoreError."""
        name = uid()
        store = EndeeVectorStore(
            embedding=fake_embedder,
            collection_name=name,
            fields=[DENSE_FIELD],
            force_recreate=True,
        )

        def _raise(updates):
            raise RuntimeError("backend exploded")

        monkeypatch.setattr(store.collection, "update_filters", _raise)
        with pytest.raises(EndeeVectorStoreError):
            store.update_filters([{"id": "x", "filter": {"a": 1}}])

    # ── add_objects / multi_field_search / multi_field_search_with_rerank ──

    def test_add_objects_returns_ids(self, mock_endee_client, fake_embedder):
        """add_objects must return the ids of the objects that were added."""
        name = uid()
        store = EndeeVectorStore(
            embedding=fake_embedder,
            collection_name=name,
            fields=[DENSE_FIELD],
            force_recreate=True,
        )
        objects = [
            {
                "id": "o1",
                "meta": {"text": "hello"},
                "filter": {},
                "fields": {"dense": [0.0] * DIMENSION},
            },
            {
                "id": "o2",
                "meta": {"text": "world"},
                "filter": {},
                "fields": {"dense": [1.0] * DIMENSION},
            },
        ]
        ids = store.add_objects(objects)
        assert ids == ["o1", "o2"]

    def test_add_objects_with_multi_vector_field_stores_all_vectors(
        self, mock_endee_client, fake_embedder
    ):
        """add_objects must store every vector in a multi_vector list unchanged."""
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
        name = uid()
        store = EndeeVectorStore(
            embedding=fake_embedder,
            collection_name=name,
            fields=fields_def,
            force_recreate=True,
        )
        chunk_vectors = [[0.1] * DIMENSION, [0.2] * DIMENSION, [0.3] * DIMENSION]
        store.add_objects(
            [
                {
                    "id": "mv1",
                    "meta": {"text": "chunked doc"},
                    "filter": {},
                    "fields": {
                        "dense": [0.5] * DIMENSION,
                        "chunks": chunk_vectors,
                    },
                },
            ]
        )
        stored = store.collection.get_objects(["mv1"])[0]
        assert stored["fields"]["chunks"] == chunk_vectors
        assert len(stored["fields"]["chunks"]) == 3
        assert all(len(v) == DIMENSION for v in stored["fields"]["chunks"])

    def test_multi_field_search_returns_per_field_results(
        self, mock_endee_client, fake_embedder
    ):
        """multi_field_search must return results keyed by each searched field name."""
        fields_def = [
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
        ]
        name = uid()
        store = EndeeVectorStore(
            embedding=fake_embedder,
            collection_name=name,
            fields=fields_def,
            dense_field_name="title",
            force_recreate=True,
        )
        store.add_objects(
            [
                {
                    "id": "m1",
                    "meta": {"text": "t"},
                    "filter": {},
                    "fields": {
                        "title": [1.0] * DIMENSION,
                        "content": [0.5] * DIMENSION,
                    },
                },
            ]
        )
        raw = store.multi_field_search(
            fields={
                "title": {"query": [1.0] * DIMENSION, "limit": 3},
                "content": {"query": [0.5] * DIMENSION, "limit": 3},
            }
        )
        assert "title" in raw["results"]
        assert "content" in raw["results"]

    def test_multi_field_search_with_rerank_forwards_field_weights_and_rrf_k(
        self, mock_endee_client, fake_embedder, monkeypatch
    ):
        """multi_field_search_with_rerank must forward field_weights, rrf_k, limit."""
        fields_def = [
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
        ]
        name = uid()
        store = EndeeVectorStore(
            embedding=fake_embedder,
            collection_name=name,
            fields=fields_def,
            dense_field_name="title",
            force_recreate=True,
        )
        store.add_objects(
            [
                {
                    "id": "m1",
                    "meta": {"text": "t"},
                    "filter": {},
                    "fields": {
                        "title": [1.0] * DIMENSION,
                        "content": [0.5] * DIMENSION,
                    },
                },
            ]
        )

        captured = {}

        def fake_rerank(raw, **kwargs):
            captured.update(kwargs)
            return {"results": []}

        monkeypatch.setattr("langchain_endee.vectorstores.endee_rerank", fake_rerank)

        store.multi_field_search_with_rerank(
            fields={
                "title": {"query": [1.0] * DIMENSION, "limit": 10},
                "content": {"query": [0.5] * DIMENSION, "limit": 10},
            },
            limit=5,
            field_weights={"title": 0.4, "content": 0.6},
            rrf_k=42,
        )
        assert captured["field_weights"] == {"title": 0.4, "content": 0.6}
        assert captured["rrf_k"] == 42
        assert captured["limit"] == 5

    def test_multi_field_search_with_rerank_omits_field_weights_when_not_given(
        self, mock_endee_client, fake_embedder, monkeypatch
    ):
        """Without field_weights or rrf_k given, both fall back to their defaults."""
        fields_def = [
            {
                "name": "title",
                "type": "vector",
                "params": {
                    "dimension": DIMENSION,
                    "space_type": "cosine",
                    "precision": "int8",
                },
            },
        ]
        name = uid()
        store = EndeeVectorStore(
            embedding=fake_embedder,
            collection_name=name,
            fields=fields_def,
            dense_field_name="title",
            force_recreate=True,
        )

        captured = {}

        def fake_rerank(raw, **kwargs):
            captured.update(kwargs)
            return {"results": []}

        monkeypatch.setattr("langchain_endee.vectorstores.endee_rerank", fake_rerank)

        store.multi_field_search_with_rerank(
            fields={"title": {"query": [1.0] * DIMENSION, "limit": 10}},
        )
        assert "field_weights" not in captured
        assert captured["rrf_k"] == 60
        assert captured["limit"] == 10


# ═══════════════════════════════════════════════════════════════════════════
# Filters unit tests
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestFiltersUnit:
    def _make_store(self, mock_endee_client, fake_embedder, suffix):
        name = uid(suffix)
        return EndeeVectorStore(
            embedding=fake_embedder,
            collection_name=name,
            fields=[DENSE_FIELD],
            force_recreate=True,
        )

    # ── delete(filter=...) ────────────────────────────────────────────────

    def test_delete_by_filter_forwards_filter_to_collection(
        self, mock_endee_client, fake_embedder, monkeypatch
    ):
        """delete(filter=...) must forward the filter to collection.delete_by_filter."""
        store = self._make_store(mock_endee_client, fake_embedder, "delfilterfwd")
        store.add_texts(texts=TEXTS, metadatas=METADATAS)

        captured = {}
        original = store.collection.delete_by_filter

        def spy(filter):  # noqa: A002
            captured["filter"] = filter
            return original(filter)

        monkeypatch.setattr(store.collection, "delete_by_filter", spy)

        expected_filter = [{"category": {"$eq": "programming"}}]
        store.delete(filter=expected_filter)
        assert captured["filter"] == expected_filter

    def test_delete_by_filter_removes_only_matching(
        self, mock_endee_client, fake_embedder
    ):
        """delete(filter=...) must remove matches only, leaving others untouched."""
        store = self._make_store(mock_endee_client, fake_embedder, "delfiltermatch")
        store.add_texts(texts=TEXTS, metadatas=METADATAS)
        store.delete(filter=[{"category": {"$eq": "programming"}}])
        remaining_ids = list(store.collection._store)
        remaining_categories = {
            store.collection._store[i]["filter"]["category"] for i in remaining_ids
        }
        assert "programming" not in remaining_categories
        assert len(remaining_ids) == 3

    # ── filter forwarding through similarity_search_by_object_with_score ──

    def test_similarity_search_forwards_filter_to_search_kwargs(
        self, mock_endee_client, fake_embedder, monkeypatch
    ):
        """similarity_search must forward the filter argument to collection.search."""
        store = self._make_store(mock_endee_client, fake_embedder, "searchfilterfwd")
        store.add_texts(texts=TEXTS, metadatas=METADATAS)

        captured = {}
        original = store.collection.search

        def spy(**kwargs):
            captured.update(kwargs)
            return original(**kwargs)

        monkeypatch.setattr(store.collection, "search", spy)

        expected_filter = [{"category": {"$eq": "ai"}}]
        store.similarity_search("learning", k=5, filter=expected_filter)
        assert captured["filter"] == expected_filter

    def test_similarity_search_omits_filter_key_when_not_given(
        self, mock_endee_client, fake_embedder, monkeypatch
    ):
        """similarity_search must omit the 'filter' kwarg when none is given."""
        store = self._make_store(mock_endee_client, fake_embedder, "searchnofilter")
        store.add_texts(texts=TEXTS, metadatas=METADATAS)

        captured = {}
        original = store.collection.search

        def spy(**kwargs):
            captured.update(kwargs)
            return original(**kwargs)

        monkeypatch.setattr(store.collection, "search", spy)

        store.similarity_search("learning", k=5)
        assert "filter" not in captured

    # ── Operator support ($eq, $in, multiple filters) ─────────────────────

    def test_filter_eq(self, mock_endee_client, fake_embedder):
        """The $eq operator must restrict results to docs matching the exact value."""
        store = self._make_store(mock_endee_client, fake_embedder, "eq")
        store.add_texts(texts=TEXTS, metadatas=METADATAS)
        results = store.similarity_search(
            "language", k=5, filter=[{"category": {"$eq": "programming"}}]
        )
        assert len(results) > 0
        for doc in results:
            assert doc.metadata.get("category") == "programming"

    def test_filter_in(self, mock_endee_client, fake_embedder):
        """The $in operator must restrict results to docs whose value is in the list."""
        store = self._make_store(mock_endee_client, fake_embedder, "in")
        store.add_texts(texts=TEXTS, metadatas=METADATAS)
        results = store.similarity_search(
            "technology",
            k=5,
            filter=[{"difficulty": {"$in": ["beginner", "advanced"]}}],
        )
        assert len(results) > 0
        for doc in results:
            assert doc.metadata.get("difficulty") in ["beginner", "advanced"]

    def test_filter_multiple_is_and_logic(self, mock_endee_client, fake_embedder):
        """Multiple filter dicts in the list must combine with AND logic."""
        store = self._make_store(mock_endee_client, fake_embedder, "multi")
        store.add_texts(texts=TEXTS, metadatas=METADATAS)
        results = store.similarity_search(
            "learning",
            k=5,
            filter=[{"category": {"$eq": "ai"}}, {"difficulty": {"$eq": "advanced"}}],
        )
        for doc in results:
            assert doc.metadata.get("category") == "ai"
            assert doc.metadata.get("difficulty") == "advanced"

    def test_filter_no_match_returns_empty(self, mock_endee_client, fake_embedder):
        """A filter matching no documents must return an empty result list."""
        store = self._make_store(mock_endee_client, fake_embedder, "nomatch")
        store.add_texts(texts=TEXTS, metadatas=METADATAS)
        results = store.similarity_search(
            "anything", k=5, filter=[{"category": {"$eq": "nonexistent"}}]
        )
        assert results == []

    def test_unsupported_operator_raises(self, mock_endee_client, fake_embedder):
        """An unsupported filter operator must raise ValueError."""
        store = self._make_store(mock_endee_client, fake_embedder, "badop")
        store.add_texts(texts=TEXTS, metadatas=METADATAS)
        with pytest.raises(ValueError, match="does not support operator"):
            store.similarity_search(
                "language", k=5, filter=[{"category": {"$ne": "programming"}}]
            )


# ═══════════════════════════════════════════════════════════════════════════
# Sparse unit tests
# ═══════════════════════════════════════════════════════════════════════════


class FakeSparseModel:
    """Stand-in for `endee_model.SparseModel`."""

    def __init__(
        self,
        model_name=None,
        cache_dir=None,
        k=1.2,
        b=0.75,
        avg_len=256.0,
        language="english",
        **kwargs,
    ):
        self.model_name = model_name

    def embed(self, texts):
        for t in texts:
            indices = sorted(set(hash(w) % 1000 for w in t.split()[:10]))
            values = [1.0 / (i + 1) for i in range(len(indices))]
            yield make_fake_sparse_result(indices, values)

    def query_embed(self, text):
        yield from self.embed([text])


@pytest.mark.unit
class TestSparseUnit:
    # ── Fixture ──────────────────────────────────────────────────────────

    @pytest.fixture
    def fake_endee_model_module(self, monkeypatch):
        """Patch sys.modules for endee_model with a fake; no real package needed."""
        fake_module = types.ModuleType("endee_model")
        fake_module.SparseModel = FakeSparseModel
        monkeypatch.setitem(sys.modules, "endee_model", fake_module)
        return fake_module

    # ── SparseVector ──────────────────────────────────────────────────────

    def test_sparse_vector_valid_construction(self):
        """SparseVector must store the given indices and values unchanged."""
        sv = SparseVector(indices=[1, 5, 9], values=[0.1, 0.2, 0.3])
        assert sv.indices == [1, 5, 9]
        assert sv.values == [0.1, 0.2, 0.3]

    def test_sparse_vector_rejects_extra_fields(self):
        """SparseVector must raise ValidationError for an unexpected extra field."""
        with pytest.raises(ValidationError):
            SparseVector(indices=[1], values=[0.1], extra_field="nope")

    def test_sparse_vector_requires_indices_and_values(self):
        """SparseVector must raise ValidationError when values is missing."""
        with pytest.raises(ValidationError):
            SparseVector(indices=[1, 2])

    # ── SparseModelAdapter ────────────────────────────────────────────────

    def test_sparse_model_adapter_embed_documents(self):
        """embed_documents returns one SparseVector per text with matched lengths."""
        adapter = SparseModelAdapter(FakeRawSparseModel())
        vecs = adapter.embed_documents(["python programming", "rust systems"])
        assert len(vecs) == 2
        for v in vecs:
            assert isinstance(v, SparseVector)
            assert len(v.indices) == len(v.values)

    def test_sparse_model_adapter_embed_query(self):
        """SparseModelAdapter.embed_query must return a single SparseVector."""
        adapter = SparseModelAdapter(FakeRawSparseModel())
        vec = adapter.embed_query("python")
        assert isinstance(vec, SparseVector)

    # ── wrap_sparse_model ─────────────────────────────────────────────────

    def test_wrap_sparse_model_passes_through_sparse_embeddings(
        self, fake_sparse_embedding
    ):
        """wrap_sparse_model must return a SparseEmbeddings instance unchanged."""
        wrapped = wrap_sparse_model(fake_sparse_embedding)
        assert wrapped is fake_sparse_embedding

    def test_wrap_sparse_model_wraps_raw_model(self):
        """wrap_sparse_model must wrap a raw sparse model in a SparseModelAdapter."""
        raw = FakeRawSparseModel()
        wrapped = wrap_sparse_model(raw)
        assert isinstance(wrapped, SparseModelAdapter)

    def test_wrap_sparse_model_rejects_invalid_object(self):
        """wrap_sparse_model rejects objects that are neither model nor embeddings."""
        with pytest.raises(TypeError):
            wrap_sparse_model(object())

    # ── EndeeModelSparse ──────────────────────────────────────────────────

    def test_endee_model_sparse_raises_without_package(self, monkeypatch):
        """EndeeModelSparse must raise ImportError when endee_model is unavailable."""
        monkeypatch.setitem(sys.modules, "endee_model", None)
        with pytest.raises(ImportError):
            EndeeModelSparse()

    def test_endee_model_sparse_embed_documents(self, fake_endee_model_module):
        """embed_documents must return one SparseVector per input text."""
        model = EndeeModelSparse()
        vecs = model.embed_documents(["python programming", "rust systems"])
        assert len(vecs) == 2
        for v in vecs:
            assert isinstance(v, SparseVector)

    def test_endee_model_sparse_embed_query(self, fake_endee_model_module):
        """EndeeModelSparse.embed_query must return a SparseVector."""
        model = EndeeModelSparse()
        vec = model.embed_query("python")
        assert isinstance(vec, SparseVector)

    # ── Hybrid / retrieval-mode auto-detection ───────────────────────────

    def test_endee_bm25_field_auto_switches_to_hybrid(
        self, mock_endee_client, fake_embedder, fake_endee_model_module
    ):
        """A sparse_model='endee_bm25' field auto-switches retrieval_mode to HYBRID."""
        # It must also configure sparse_embeddings as an EndeeModelSparse instance.
        name = uid()
        store = EndeeVectorStore(
            embedding=fake_embedder,
            collection_name=name,
            fields=[
                DENSE_FIELD,
                {"name": "sparse", "type": "sparse", "sparse_model": "endee_bm25"},
            ],
            force_recreate=True,
        )
        assert store.retrieval_mode == RetrievalMode.HYBRID
        assert isinstance(store.sparse_embeddings, EndeeModelSparse)

    def test_raw_sparse_model_is_auto_wrapped(self, mock_endee_client, fake_embedder):
        """A raw sparse model as sparse_embedding auto-wraps in SparseModelAdapter."""
        name = uid()
        raw_model = FakeRawSparseModel()
        store = EndeeVectorStore(
            embedding=fake_embedder,
            collection_name=name,
            fields=[
                DENSE_FIELD,
                {"name": "sparse", "type": "sparse", "sparse_model": "default"},
            ],
            retrieval_mode=RetrievalMode.HYBRID,
            sparse_embedding=raw_model,
            force_recreate=True,
        )
        assert isinstance(store.sparse_embeddings, SparseModelAdapter)

    def test_from_existing_endee_bm25_auto_reconnect(
        self, mock_endee_client, fake_embedder, fake_endee_model_module
    ):
        """Reconnecting to an endee_bm25 sparse field must auto-detect HYBRID mode."""
        name = uid()
        EndeeVectorStore(
            embedding=fake_embedder,
            collection_name=name,
            fields=[
                DENSE_FIELD,
                {"name": "sparse", "type": "sparse", "sparse_model": "endee_bm25"},
            ],
            force_recreate=True,
        )
        store2 = EndeeVectorStore.from_existing_collection(
            collection_name=name,
            embedding=fake_embedder,
        )
        assert store2.retrieval_mode == RetrievalMode.HYBRID

    def test_hybrid_search_by_object_with_score(
        self, mock_endee_client, fake_embedder, fake_sparse_embedding
    ):
        """Hybrid by-object search with score must run RRF fusion without raising."""
        name = uid()
        store = EndeeVectorStore(
            embedding=fake_embedder,
            collection_name=name,
            fields=[
                DENSE_FIELD,
                {"name": "sparse", "type": "sparse", "sparse_model": "default"},
            ],
            retrieval_mode=RetrievalMode.HYBRID,
            sparse_embedding=fake_sparse_embedding,
            force_recreate=True,
        )
        store.add_texts(texts=TEXTS, metadatas=METADATAS)
        dense_vec = fake_embedder.embed_query("deep learning")
        sparse_vec = fake_sparse_embedding.embed_query("deep learning")
        results = store.similarity_search_by_object_with_score(
            embedding=dense_vec,
            k=3,
            sparse_indices=sparse_vec.indices,
            sparse_values=sparse_vec.values,
        )
        # The fake backend returns no sparse hits, but fusion must still return a list.
        assert isinstance(results, list)

    # ── _detect_sparse_model / simple-mode (dimension=) collection creation ─

    def test_simple_mode_creation_with_sparse_embedding_uses_default_sparse_model(
        self, mock_endee_client, fake_embedder, fake_sparse_embedding
    ):
        """Simple-mode with a plain sparse_embedding creates a 'default' sparse."""
        name = uid()
        store = EndeeVectorStore(
            embedding=fake_embedder,
            collection_name=name,
            dimension=DIMENSION,
            retrieval_mode=RetrievalMode.HYBRID,
            sparse_embedding=fake_sparse_embedding,
            force_recreate=True,
        )
        fm = store.field_map
        assert fm["sparse"]["type"] == "sparse"
        assert fm["sparse"]["sparse_model"] == "default"

    def test_simple_mode_creation_with_endee_model_sparse_uses_endee_bm25(
        self, mock_endee_client, fake_embedder, fake_endee_model_module
    ):
        """An EndeeModelSparse embedding in simple mode creates an endee_bm25 field."""
        name = uid()
        store = EndeeVectorStore(
            embedding=fake_embedder,
            collection_name=name,
            dimension=DIMENSION,
            retrieval_mode=RetrievalMode.HYBRID,
            sparse_embedding=EndeeModelSparse(),
            force_recreate=True,
        )
        fm = store.field_map
        assert fm["sparse"]["type"] == "sparse"
        assert fm["sparse"]["sparse_model"] == "endee_bm25"

    def test_simple_mode_creation_without_sparse_embedding_has_no_sparse_field(
        self, mock_endee_client, fake_embedder
    ):
        """Simple-mode without a sparse_embedding must create no sparse field."""
        name = uid()
        store = EndeeVectorStore(
            embedding=fake_embedder,
            collection_name=name,
            dimension=DIMENSION,
            force_recreate=True,
        )
        fm = store.field_map
        assert "sparse" not in fm

    # ── Async delegation (aembed_documents / aembed_query) ────────────────

    @pytest.mark.asyncio
    async def test_aembed_documents_delegates_to_sync(self, fake_sparse_embedding):
        """aembed_documents must match the sync embed_documents SparseVectors."""
        texts = ["python programming", "rust systems", "machine learning"]
        expected = fake_sparse_embedding.embed_documents(texts)
        actual = await fake_sparse_embedding.aembed_documents(texts)
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            assert a.indices == e.indices
            assert a.values == e.values

    @pytest.mark.asyncio
    async def test_aembed_query_delegates_to_sync(self, fake_sparse_embedding):
        """aembed_query must return the same SparseVector as sync embed_query."""
        expected = fake_sparse_embedding.embed_query("python programming")
        actual = await fake_sparse_embedding.aembed_query("python programming")
        assert actual.indices == expected.indices
        assert actual.values == expected.values

    @pytest.mark.asyncio
    async def test_aembed_documents_uses_run_in_executor(self):
        """The default aembed_documents delegates once to sync embed_documents."""
        # One call with the full batch, not one call per text - proves it
        # goes through run_in_executor rather than looping and awaiting per item.
        calls = []

        class TrackedSparse(SparseEmbeddings):
            def embed_documents(self, texts):
                calls.append(("embed_documents", texts))
                return [SparseVector(indices=[0], values=[1.0]) for _ in texts]

            def embed_query(self, text):
                calls.append(("embed_query", text))
                return SparseVector(indices=[0], values=[1.0])

        instance = TrackedSparse()
        result = await instance.aembed_documents(["a", "b"])
        assert len(result) == 2
        assert calls == [("embed_documents", ["a", "b"])]

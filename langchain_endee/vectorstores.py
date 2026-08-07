from __future__ import annotations

import logging
import uuid
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    Sequence,
)

from endee import Endee as EndeeClient
from endee import Precision, constants, rerank as endee_rerank
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

if TYPE_CHECKING:
    from langchain_endee.sparse_embeddings import SparseEmbeddings
logger = logging.getLogger(__name__)


class EndeeVectorStoreError(Exception):
    """`EndeeVectorStore` related exceptions."""


class RetrievalMode(str, Enum):
    """Modes for retrieving vectors from Endee."""

    DENSE = "dense"
    HYBRID = "hybrid"


class EndeeVectorStore(VectorStore):
    """Endee vector store integration for LangChain.

    Provides ANN search via HNSW with multiple distance metrics, metadata
    filtering ($eq, $in, $range), configurable precision levels,
    optional hybrid search (dense + sparse), and multi-field collections.

    Uses the Endee v2 Collections API with typed fields.

    **Simple mode** (default): one dense field, optional sparse field.
    **Multi-field mode**: pass ``fields=`` to create collections with
    multiple vector, sparse, or multi_vector fields.
    """

    # Maximum token limits for common embedding models
    EMBEDDING_MODEL_LIMITS = {
        "openai": 8191,  # text-embedding-3-small/large, text-embedding-ada-002
        "cohere": 512,  # embed-english-v3.0, embed-multilingual-v3.0
        "huggingface": 512,  # Most sentence-transformers models
        "default": 512,  # Conservative default
    }

    CONTENT_KEY: str = "text"
    METADATA_KEY: str = "metadata"
    DENSE_FIELD_NAME: str = "dense"
    SPARSE_FIELD_NAME: str = "sparse"

    def __init__(
        self,
        embedding: Embeddings,
        api_token: str | None = None,
        base_url: str | None = None,
        collection_name: str | None = None,
        max_text_length: int | None = None,
        embedding_model_type: str | None = None,
        endee_client: EndeeClient | None = None,
        retrieval_mode: RetrievalMode = RetrievalMode.DENSE,
        dimension: int | None = None,
        space_type: str = "cosine",
        precision: str = Precision.INT8,
        M: int = constants.DEFAULT_M,
        ef_con: int = constants.DEFAULT_EF_CON,
        content_payload_key: str = CONTENT_KEY,
        sparse_embedding: SparseEmbeddings | None = None,
        metadata_payload_key: str = METADATA_KEY,
        dense_field_name: str = DENSE_FIELD_NAME,
        sparse_field_name: str = SPARSE_FIELD_NAME,
        fields: list[dict[str, Any]] | None = None,
        force_recreate: bool = False,  # noqa: FBT001, FBT002
        validate_collection_config: bool = True,  # noqa: FBT001, FBT002
    ) -> None:
        """Initialize EndeeVectorStore.

        Args:
            embedding: LangChain embedding function (used for the primary
                dense field and the standard ``add_texts``/``similarity_search``
                methods).
            api_token: Endee API token (optional for local deployment).
            collection_name: Name of the Endee collection.
            max_text_length: Max text length in tokens (auto-detected if None).
            embedding_model_type: ``"openai"``, ``"cohere"``, ``"huggingface"``,
                or ``"default"`` (auto-detected if None).
            endee_client: Existing Endee client (overrides ``api_token``).
            retrieval_mode: ``RetrievalMode.DENSE`` or ``RetrievalMode.HYBRID``.
            dimension: Vector dimension (required for new collections in
                simple mode; ignored when ``fields`` is provided).
            space_type: Distance metric — ``"cosine"``, ``"l2"``, or ``"ip"``.
            precision: Quantisation level (``Precision`` enum).
            M: HNSW bi-directional links per node. Default: 16.
            ef_con: HNSW construction quality. Default: 128.
            sparse_embedding: Sparse embedding model for hybrid search.
            dense_field_name: Name of the primary dense vector field.
                Default: ``"dense"``.
            sparse_field_name: Name of the sparse vector field.
                Default: ``"sparse"``.
            fields: **Multi-field mode.** Raw field definitions passed
                directly to ``create_collection``. When provided, the
                ``dimension``/``space_type``/``precision``/``M``/``ef_con``
                params are ignored for collection creation. Example::

                    fields=[
                        {"name": "title",   "type": "vector",
                         "params": {"dimension": 384, "space_type": "cosine",
                                    "precision": "int8"}},
                        {"name": "content", "type": "vector",
                         "params": {"dimension": 768, "space_type": "cosine",
                                    "precision": "int8"}},
                        {"name": "keywords","type": "sparse",
                         "sparse_model": "default"},
                        {"name": "colbert", "type": "multi_vector",
                         "params": {"dimension": 128, "space_type": "cosine",
                                    "precision": "float16",
                                    "pooling": "mean"}},
                    ]
            force_recreate: Delete and recreate collection if it exists.
            validate_collection_config: Validate dimension on connect.
        """
        if embedding is None:
            msg = "Embedding function must be provided"
            raise ValueError(msg)

        self._embeddings = embedding
        self.content_payload_key = content_payload_key
        self.metadata_payload_key = metadata_payload_key
        self.space_type = space_type
        self.precision = precision
        self.dimension = dimension
        self.M = M
        self.ef_con = ef_con
        self.retrieval_mode = retrieval_mode
        if sparse_embedding is not None:
            from langchain_endee.sparse_embeddings import wrap_sparse_model

            self._sparse_embeddings = wrap_sparse_model(sparse_embedding)
        else:
            self._sparse_embeddings = None
        self.base_url = base_url
        self.dense_field_name = dense_field_name
        self.sparse_field_name = sparse_field_name
        self._custom_fields = fields

        self._init_client(api_token, base_url, endee_client)

        if collection_name is None:
            msg = "collection_name must be provided"
            raise ValueError(msg)

        self.collection_name = collection_name
        self._collection = self._connect_collection(
            collection_name,
            dimension,
            space_type,
            precision,
            M,
            ef_con,
            sparse_embedding,
            force_recreate,
        )

        if validate_collection_config and self._custom_fields is None:
            self._validate_collection_config()

        self._setup_text_length(embedding_model_type, max_text_length)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def embeddings(self) -> Embeddings:
        """The dense embedding model."""
        return self._embeddings

    @property
    def client(self) -> EndeeClient:
        """The underlying Endee client."""
        return self._client

    @property
    def sparse_embeddings(self) -> SparseEmbeddings:
        """The sparse embedding model. Raises ``ValueError`` if not set."""
        if self._sparse_embeddings is None:
            msg = (
                "Sparse embeddings are not set. "
                "Pass sparse_embedding= to the constructor."
            )
            raise ValueError(msg)
        return self._sparse_embeddings

    @property
    def collection(self) -> Any:
        """The underlying Endee collection object."""
        return self._collection

    @property
    def field_map(self) -> dict[str, dict[str, Any]]:
        """``{field_name: {type, params, ...}}`` for every field in the collection."""
        out: dict[str, dict[str, Any]] = {}
        for f in self._collection.fields:
            out[f["name"]] = f
        return out

    # ── Client init ───────────────────────────────────────────────────────────

    def _init_client(
        self,
        api_token: str | None,
        base_url: str | None,
        endee_client: EndeeClient | None,
    ) -> None:
        if endee_client is None:
            if api_token is None:
                self._client = EndeeClient()
            else:
                self._client = EndeeClient(token=api_token)
        else:
            self._client = endee_client
        if base_url:
            self._client.set_base_url(base_url)

    # ── Collection connect / create ───────────────────────────────────────────

    def _connect_collection(
        self,
        collection_name: str,
        dimension: int | None,
        space_type: str,
        precision: str,
        M: int,  # noqa: N803
        ef_con: int,
        sparse_embedding: SparseEmbeddings | None,
        force_recreate: bool,
    ) -> Any:
        collection_list = self._client.list_collections()
        collection_exists = any(
            c.get("name") == collection_name for c in collection_list
        )

        if collection_exists and force_recreate:
            logger.info(f"Deleting existing collection: {collection_name}")
            self._client.delete_collection(collection_name)
            collection_exists = False

        if not collection_exists:
            if self._custom_fields is not None:
                # Multi-field mode — use raw fields directly
                self._create_collection_raw(collection_name, self._custom_fields)
            else:
                # Simple mode — auto-build fields from params
                if dimension is None:
                    msg = (
                        f"Collection '{collection_name}' does not exist and "
                        "dimension is not provided. Please provide dimension "
                        "to create a new collection."
                    )
                    raise ValueError(msg)

                logger.info(f"Creating new Endee collection: {collection_name}")
                sparse_model = self._detect_sparse_model(sparse_embedding)
                self._create_collection(
                    name=collection_name,
                    dimension=dimension,
                    space_type=space_type,
                    precision=precision,
                    M=M,
                    ef_con=ef_con,
                    sparse_model=sparse_model,
                )
        else:
            logger.info(f"Using existing Endee collection: {collection_name}")

        collection = self._client.get_collection(name=collection_name)
        self._detect_field_names(collection)
        return collection

    def _detect_field_names(self, collection: Any) -> None:
        """Auto-detect primary dense/sparse field names from metadata.

        When a sparse field with ``sparse_model="endee_bm25"`` is found and
        no ``sparse_embedding`` was provided, automatically creates an
        ``EndeeModelSparse`` instance and sets ``retrieval_mode=HYBRID``.
        """
        dense_set = False
        sparse_set = False
        for f in collection.fields:
            ftype = f.get("type", "")
            fname = f.get("name", "")
            if ftype == "vector" and not dense_set:
                self.dense_field_name = fname
                dense_set = True
            elif ftype == "sparse" and not sparse_set:
                self.sparse_field_name = fname
                sparse_set = True

                # Auto-create EndeeModelSparse for endee_bm25 fields
                sparse_model = f.get("sparse_model") or f.get("params", {}).get(
                    "sparse_model", ""
                )
                if sparse_model == "endee_bm25" and self._sparse_embeddings is None:
                    from langchain_endee.sparse_embeddings import EndeeModelSparse

                    self._sparse_embeddings = EndeeModelSparse()
                    self.retrieval_mode = RetrievalMode.HYBRID
                    logger.info(
                        "Auto-created EndeeModelSparse for "
                        f"sparse field '{fname}' (endee_bm25)"
                    )

    def _create_collection(
        self,
        name: str,
        dimension: int,
        space_type: str,
        precision: str,
        M: int,  # noqa: N803
        ef_con: int,
        sparse_model: str | None = None,
    ) -> None:
        """Create a collection in simple mode (one dense + optional sparse)."""
        try:
            fields: list[dict[str, Any]] = [
                {
                    "name": self.dense_field_name,
                    "type": "vector",
                    "params": {
                        "dimension": dimension,
                        "space_type": space_type,
                        "precision": precision,
                        "M": M,
                        "ef_con": ef_con,
                    },
                }
            ]
            if sparse_model is not None:
                fields.append(
                    {
                        "name": self.sparse_field_name,
                        "type": "sparse",
                        "sparse_model": sparse_model,
                    }
                )

            self._client.create_collection(name=name, fields=fields)
        except Exception as e:
            msg = f"Failed to create Endee collection '{name}': {e}"
            raise EndeeVectorStoreError(msg) from e

    def _create_collection_raw(
        self,
        name: str,
        fields: list[dict[str, Any]],
    ) -> None:
        """Create a collection in multi-field mode (raw fields)."""
        try:
            self._client.create_collection(name=name, fields=fields)
        except Exception as e:
            msg = f"Failed to create Endee collection '{name}': {e}"
            raise EndeeVectorStoreError(msg) from e

    # ── Validation ────────────────────────────────────────────────────────────

    def _validate_collection_config(self) -> None:
        """Validate that the collection config matches the vector store settings."""
        try:
            info = self._collection.describe()
        except Exception as e:
            logger.warning(f"Could not fetch collection config: {e}")
            return

        fields = info.get("fields", [])
        dense_field = None
        sparse_field = None
        for f in fields:
            ftype = f.get("type", "")
            fname = f.get("name", "")
            if ftype == "vector" and fname == self.dense_field_name:
                dense_field = f
            elif ftype == "sparse" and fname == self.sparse_field_name:
                sparse_field = f

        if dense_field is None:
            for f in fields:
                if f.get("type") == "vector":
                    dense_field = f
                    break
        if sparse_field is None:
            for f in fields:
                if f.get("type") == "sparse":
                    sparse_field = f
                    break

        errors: list[str] = []
        params = dense_field.get("params", {}) if dense_field else {}

        if self.dimension is not None and params.get("dimension") != self.dimension:
            errors.append(
                f"dimension: collection has {params.get('dimension')}, "
                f"expected {self.dimension}"
            )

        if params.get("space_type") and params["space_type"] != self.space_type:
            errors.append(
                f"space_type: collection has '{params['space_type']}', "
                f"expected '{self.space_type}'"
            )

        collection_is_hybrid = sparse_field is not None
        user_wants_hybrid = self.retrieval_mode == RetrievalMode.HYBRID

        if user_wants_hybrid and not collection_is_hybrid:
            errors.append(
                "retrieval_mode is HYBRID but the collection is dense-only. "
                "Recreate the collection with a sparse_embedding to enable "
                "hybrid search"
            )
        if user_wants_hybrid and self._sparse_embeddings is None:
            errors.append(
                "retrieval_mode is HYBRID but no sparse_embedding was provided"
            )

        if errors:
            msg = (
                "Collection config mismatch:\n  "
                + "\n  ".join(errors)
                + "\nSet force_recreate=True to recreate the collection."
            )
            raise EndeeVectorStoreError(msg)

        if collection_is_hybrid and not user_wants_hybrid:
            sparse_model = sparse_field.get("sparse_model", "unknown")
            logger.warning(
                f"Collection '{info.get('name')}' supports hybrid search "
                f"(sparse_model='{sparse_model}') but "
                "retrieval_mode is DENSE. Pass retrieval_mode=RetrievalMode.HYBRID "
                "and a sparse_embedding to use hybrid search."
            )

        self._warn_tuning_mismatch(params)

    def _warn_tuning_mismatch(self, params: dict) -> None:
        # Normalize a Precision enum member to its string; leave a string as is.
        precision = getattr(self.precision, "value", self.precision)
        if params.get("precision") and params["precision"] != precision:
            logger.warning(
                f"Collection precision is '{params['precision']}', "
                f"expected '{precision}'. "
                "The existing collection precision will be used."
            )
        if params.get("M") and params["M"] != self.M:
            logger.warning(
                f"Collection M is {params['M']}, expected {self.M}. "
                "The existing collection M will be used."
            )
        if params.get("ef_con") and params["ef_con"] != self.ef_con:
            logger.warning(
                f"Collection ef_con is {params['ef_con']}, expected {self.ef_con}. "
                "The existing collection ef_con will be used."
            )

    # ── Text-length helpers ───────────────────────────────────────────────────

    def _setup_text_length(
        self,
        embedding_model_type: str | None,
        max_text_length: int | None,
    ) -> None:
        if embedding_model_type is None:
            embedding_model_type = self._detect_embedding_model_type(self._embeddings)
        self.embedding_model_type = embedding_model_type

        if max_text_length is None:
            self.max_text_length = self.EMBEDDING_MODEL_LIMITS.get(
                embedding_model_type, self.EMBEDDING_MODEL_LIMITS["default"]
            )
        else:
            self.max_text_length = max_text_length

    def _detect_embedding_model_type(self, embedding: Embeddings) -> str:
        class_name = embedding.__class__.__name__.lower()
        module_name = embedding.__class__.__module__.lower()
        if "openai" in class_name or "openai" in module_name:
            return "openai"
        elif "cohere" in class_name or "cohere" in module_name:
            return "cohere"
        elif (
            "huggingface" in class_name
            or "huggingface" in module_name
            or "sentence" in class_name
            or "transformers" in module_name
        ):
            return "huggingface"
        else:
            return "default"

    @staticmethod
    def _detect_sparse_model(
        sparse_embedding: SparseEmbeddings | None,
    ) -> str | None:
        if sparse_embedding is None:
            return None
        from langchain_endee.sparse_embeddings import EndeeModelSparse

        if isinstance(sparse_embedding, EndeeModelSparse):
            return "endee_bm25"
        return "default"

    def _estimate_tokens(self, text: str) -> int:
        return len(text) // 4

    def _truncate_text(self, text: str, max_tokens: int | None = None) -> str:
        if max_tokens is None:
            max_tokens = self.max_text_length
        estimated_tokens = self._estimate_tokens(text)
        if estimated_tokens <= max_tokens:
            return text
        max_chars = int(max_tokens * 4 * 0.9)
        truncated = text[:max_chars]
        logger.warning(
            f"Text truncated from ~{estimated_tokens} to ~{max_tokens} tokens "
            f"(model type: {self.embedding_model_type})"
        )
        return truncated

    def _validate_batch_size(self, batch_size: int) -> int:
        if batch_size > constants.MAX_VECTORS_PER_BATCH:
            msg = (
                f"batch_size ({batch_size}) cannot exceed "
                f"{constants.MAX_VECTORS_PER_BATCH} (Endee's maximum batch size)"
            )
            raise ValueError(msg)
        return batch_size

    # ── Prepare / embed helpers ───────────────────────────────────────────────

    def _prepare_texts_and_metadata(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None,
        ids: list[str] | None,
    ) -> tuple[list[str], list[str], list[dict]]:
        texts = list(texts)
        ids = list(ids) if ids else [str(uuid.uuid4()) for _ in texts]
        metadatas = metadatas or [{} for _ in texts]
        if len(texts) != len(ids):
            msg = (
                f"Number of texts ({len(texts)}) must match "
                f"number of ids ({len(ids)})"
            )
            raise ValueError(msg)
        if len(texts) != len(metadatas):
            msg = (
                f"Number of texts ({len(texts)}) must match "
                f"number of metadatas ({len(metadatas)})"
            )
            raise ValueError(msg)
        processed_texts = [self._truncate_text(text) for text in texts]
        return processed_texts, ids, metadatas

    def _generate_embeddings_in_batches(
        self,
        texts: list[str],
        embedding_chunk_size: int,
    ) -> list[list[float]]:
        embeddings = []
        for i in range(0, len(texts), embedding_chunk_size):
            sub_texts = texts[i : i + embedding_chunk_size]
            sub_embeddings = self.embeddings.embed_documents(sub_texts)
            embeddings.extend(sub_embeddings)
        return embeddings

    def _generate_sparse_embeddings_in_batches(
        self,
        texts: list[str],
        embedding_chunk_size: int,
    ) -> tuple[list[list[int]], list[list[float]]]:
        all_indices: list[list[int]] = []
        all_values: list[list[float]] = []
        for i in range(0, len(texts), embedding_chunk_size):
            sub_texts = texts[i : i + embedding_chunk_size]
            sparse_vectors = self.sparse_embeddings.embed_documents(sub_texts)
            for sv in sparse_vectors:
                all_indices.append(sv.indices)
                all_values.append(sv.values)
        return all_indices, all_values

    # ── Upsert helpers ────────────────────────────────────────────────────────

    def _build_upsert_entries(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        texts: list[str],
        metadatas: list[dict],
        sparse_indices: list[list[int]] | None = None,
        sparse_values: list[list[float]] | None = None,
    ) -> list[dict[str, Any]]:
        """Build object dicts for the primary dense (+ optional sparse) field."""
        entries = []
        for i, (entry_id, embedding, text, metadata) in enumerate(
            zip(ids, embeddings, texts, metadatas, strict=False)
        ):
            meta = {
                self.content_payload_key: text,
                self.metadata_payload_key: metadata,
            }
            filter_data = dict(metadata.items())

            fields: dict[str, Any] = {
                self.dense_field_name: embedding,
            }
            if sparse_indices is not None and sparse_values is not None:
                fields[self.sparse_field_name] = {
                    "indices": sparse_indices[i],
                    "values": sparse_values[i],
                }

            entries.append({
                "id": entry_id,
                "meta": meta,
                "filter": filter_data,
                "fields": fields,
            })
        return entries

    def _upsert_batch(self, entries: list[dict[str, Any]]) -> None:
        try:
            self._collection.upsert(entries)
        except Exception as e:
            logger.error(f"Error upserting batch of {len(entries)} entries: {e}")
            raise

    # ══════════════════════════════════════════════════════════════════════════
    # Standard LangChain API (single-field)
    # ══════════════════════════════════════════════════════════════════════════

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        *,
        ids: list[str] | None = None,
        batch_size: int = constants.MAX_VECTORS_PER_BATCH,
        embedding_chunk_size: int = 100,
        **kwargs: Any,
    ) -> list[str]:
        """Add texts using the primary dense embedding.

        Returns:
            List of document IDs.
        """
        batch_size = self._validate_batch_size(batch_size)
        processed_texts, ids, metadatas = self._prepare_texts_and_metadata(
            texts, metadatas, ids
        )

        for i in range(0, len(processed_texts), batch_size):
            batch_end = i + batch_size
            chunk_texts = processed_texts[i:batch_end]
            chunk_ids = ids[i:batch_end]
            chunk_metadatas = metadatas[i:batch_end]

            embeddings = self._generate_embeddings_in_batches(
                chunk_texts, embedding_chunk_size
            )

            sparse_indices = None
            sparse_values = None
            if self.retrieval_mode == RetrievalMode.HYBRID:
                sparse_indices, sparse_values = (
                    self._generate_sparse_embeddings_in_batches(
                        chunk_texts, embedding_chunk_size
                    )
                )

            entries = self._build_upsert_entries(
                chunk_ids, embeddings, chunk_texts, chunk_metadatas,
                sparse_indices=sparse_indices, sparse_values=sparse_values,
            )
            self._upsert_batch(entries)

        logger.info(f"Successfully added {len(ids)} texts to vector store")
        return ids

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict] | None = None,
        *,
        ids: list[str] | None = None,
        api_token: str | None = None,
        base_url: str | None = None,
        collection_name: str | None = None,
        endee_client: EndeeClient | None = None,
        dimension: int | None = None,
        space_type: str = "cosine",
        precision: str = Precision.INT8,
        M: int = constants.DEFAULT_M,
        ef_con: int = constants.DEFAULT_EF_CON,
        content_payload_key: str = CONTENT_KEY,
        metadata_payload_key: str = METADATA_KEY,
        batch_size: int = constants.MAX_VECTORS_PER_BATCH,
        retrieval_mode: RetrievalMode = RetrievalMode.DENSE,
        sparse_embedding: SparseEmbeddings | None = None,
        embedding_chunk_size: int = 100,
        fields: list[dict[str, Any]] | None = None,
        force_recreate: bool = False,  # noqa: FBT001, FBT002
        validate_collection_config: bool = True,  # noqa: FBT001, FBT002
        **kwargs: Any,
    ) -> EndeeVectorStore:
        """Create an EndeeVectorStore from raw texts."""
        if dimension is None and endee_client is None and fields is None:
            msg = "dimension must be explicitly provided when creating a new collection"
            raise ValueError(msg)

        endee = cls(
            embedding=embedding, api_token=api_token, base_url=base_url,
            collection_name=collection_name, endee_client=endee_client,
            dimension=dimension, space_type=space_type, precision=precision,
            M=M, retrieval_mode=retrieval_mode, sparse_embedding=sparse_embedding,
            ef_con=ef_con, content_payload_key=content_payload_key,
            metadata_payload_key=metadata_payload_key,
            fields=fields, force_recreate=force_recreate,
            validate_collection_config=validate_collection_config,
        )
        endee.add_texts(
            texts=texts, metadatas=metadatas, ids=ids,
            batch_size=batch_size, embedding_chunk_size=embedding_chunk_size,
            **kwargs,
        )
        return endee

    @classmethod
    def from_documents(
        cls,
        documents: list[Document],
        embedding: Embeddings,
        ids: Sequence[str] | None = None,
        api_token: str | None = None,
        base_url: str | None = None,
        collection_name: str | None = None,
        endee_client: EndeeClient | None = None,
        dimension: int | None = None,
        space_type: str = "cosine",
        precision: str = Precision.INT8,
        retrieval_mode: RetrievalMode = RetrievalMode.DENSE,
        sparse_embedding: SparseEmbeddings | None = None,
        M: int = constants.DEFAULT_M,
        ef_con: int = constants.DEFAULT_EF_CON,
        content_payload_key: str = CONTENT_KEY,
        metadata_payload_key: str = METADATA_KEY,
        batch_size: int = constants.MAX_VECTORS_PER_BATCH,
        embedding_chunk_size: int = 100,
        fields: list[dict[str, Any]] | None = None,
        force_recreate: bool = False,  # noqa: FBT001, FBT002
        validate_collection_config: bool = True,  # noqa: FBT001, FBT002
        **kwargs: Any,
    ) -> EndeeVectorStore:
        """Create an EndeeVectorStore from LangChain Document objects."""
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return cls.from_texts(
            texts=texts, embedding=embedding, metadatas=metadatas, ids=ids,
            api_token=api_token, base_url=base_url,
            collection_name=collection_name, endee_client=endee_client,
            dimension=dimension, space_type=space_type,
            retrieval_mode=retrieval_mode, sparse_embedding=sparse_embedding,
            precision=precision, M=M, ef_con=ef_con,
            content_payload_key=content_payload_key,
            metadata_payload_key=metadata_payload_key,
            batch_size=batch_size, embedding_chunk_size=embedding_chunk_size,
            fields=fields, force_recreate=force_recreate,
            validate_collection_config=validate_collection_config,
            **kwargs,
        )

    @classmethod
    def from_existing_collection(
        cls,
        collection_name: str,
        embedding: Embeddings,
        api_token: str | None = None,
        base_url: str | None = None,
        endee_client: EndeeClient | None = None,
        retrieval_mode: RetrievalMode = RetrievalMode.DENSE,
        sparse_embedding: SparseEmbeddings | None = None,
        content_payload_key: str = CONTENT_KEY,
        metadata_payload_key: str = METADATA_KEY,
        validate_collection_config: bool = True,
        # noqa: FBT001, FBT002
    ) -> EndeeVectorStore:
        """Connect to an existing Endee collection without adding data."""
        return cls(
            embedding=embedding, api_token=api_token,
            collection_name=collection_name, endee_client=endee_client,
            retrieval_mode=retrieval_mode, sparse_embedding=sparse_embedding,
            content_payload_key=content_payload_key,
            metadata_payload_key=metadata_payload_key,
            validate_collection_config=validate_collection_config,
            base_url=base_url,
        )

    # ── Search ────────────────────────────────────────────────────────────────

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: list[dict[str, Any]] | None = None,  # noqa: A002
        ef: int = constants.DEFAULT_EF_SEARCH,
        prefilter_cardinality_threshold: int | None = None,
        filter_boost_percentage: int | None = None,
        rrf_rank_constant: int | None = None,
        dense_rrf_weight: float | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs most similar to query."""
        docs_and_scores = self.similarity_search_with_score(
            query, k=k, filter=filter, ef=ef,
            prefilter_cardinality_threshold=prefilter_cardinality_threshold,
            filter_boost_percentage=filter_boost_percentage,
            rrf_rank_constant=rrf_rank_constant,
            dense_rrf_weight=dense_rrf_weight, **kwargs,
        )
        return [doc for doc, _ in docs_and_scores]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: list[dict[str, Any]] | None = None,  # noqa: A002
        ef: int = constants.DEFAULT_EF_SEARCH,
        prefilter_cardinality_threshold: int | None = None,
        filter_boost_percentage: int | None = None,
        rrf_rank_constant: int | None = None,
        dense_rrf_weight: float | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Return docs most similar to query, with similarity scores."""
        embedding = self.embeddings.embed_query(query)

        sparse_indices = None
        sparse_values = None
        if self.retrieval_mode == RetrievalMode.HYBRID:
            sparse_vector = self.sparse_embeddings.embed_query(query)
            sparse_indices = sparse_vector.indices
            sparse_values = sparse_vector.values

        return self.similarity_search_by_object_with_score(
            embedding, k=k, filter=filter, ef=ef,
            prefilter_cardinality_threshold=prefilter_cardinality_threshold,
            filter_boost_percentage=filter_boost_percentage,
            sparse_indices=sparse_indices, sparse_values=sparse_values,
            rrf_rank_constant=rrf_rank_constant,
            dense_rrf_weight=dense_rrf_weight,
        )

    def similarity_search_by_object(
        self,
        embedding: list[float],
        k: int = 4,
        filter: list[dict[str, Any]] | None = None,  # noqa: A002
        ef: int = constants.DEFAULT_EF_SEARCH,
        prefilter_cardinality_threshold: int | None = None,
        filter_boost_percentage: int | None = None,
        sparse_indices: list[int] | None = None,
        sparse_values: list[float] | None = None,
        rrf_rank_constant: int | None = None,
        dense_rrf_weight: float | None = None,
    ) -> list[Document]:
        """Return docs most similar to a pre-computed embedding."""
        if self.retrieval_mode == RetrievalMode.HYBRID and (
            sparse_indices is None or sparse_values is None
        ):
            logger.warning(
                "retrieval_mode is HYBRID but sparse_indices/sparse_values were "
                "not provided. Falling back to dense-only search."
            )
        docs_and_scores = self.similarity_search_by_object_with_score(
            embedding, k=k, filter=filter, ef=ef,
            prefilter_cardinality_threshold=prefilter_cardinality_threshold,
            filter_boost_percentage=filter_boost_percentage,
            sparse_indices=sparse_indices, sparse_values=sparse_values,
            rrf_rank_constant=rrf_rank_constant,
            dense_rrf_weight=dense_rrf_weight,
        )
        return [doc for doc, _ in docs_and_scores]

    def similarity_search_by_object_with_score(
        self,
        embedding: list[float],
        k: int = 4,
        filter: list[dict[str, Any]] | None = None,  # noqa: A002
        ef: int = constants.DEFAULT_EF_SEARCH,
        prefilter_cardinality_threshold: int | None = None,
        filter_boost_percentage: int | None = None,
        sparse_indices: list[int] | None = None,
        sparse_values: list[float] | None = None,
        rrf_rank_constant: int | None = None,
        dense_rrf_weight: float | None = None,
    ) -> list[tuple[Document, float]]:
        """Core search method — all other search methods delegate here."""
        is_hybrid = sparse_indices is not None and sparse_values is not None

        search_fields: dict[str, Any] = {}
        if is_hybrid:
            fetch_limit = max(k * 3, 50)
            search_fields[self.dense_field_name] = {
                "query": embedding, "limit": fetch_limit, "ef_search": ef,
            }
            search_fields[self.sparse_field_name] = {
                "query": {"indices": sparse_indices, "values": sparse_values},
                "limit": fetch_limit,
            }
        else:
            search_fields[self.dense_field_name] = {
                "query": embedding, "limit": k, "ef_search": ef,
            }

        search_kwargs: dict[str, Any] = {
            "fields": search_fields, "ef_search": ef,
        }
        if filter is not None:
            search_kwargs["filter"] = filter
        if prefilter_cardinality_threshold is not None:
            search_kwargs["prefilter_cardinality_threshold"] = (
                prefilter_cardinality_threshold
            )
        if filter_boost_percentage is not None:
            search_kwargs["filter_boost_percentage"] = filter_boost_percentage

        raw_results = self._collection.search(**search_kwargs)

        if is_hybrid:
            rerank_kwargs: dict[str, Any] = {"limit": k}
            if dense_rrf_weight is not None:
                rerank_kwargs["field_weights"] = {
                    self.dense_field_name: dense_rrf_weight,
                    self.sparse_field_name: 1.0 - dense_rrf_weight,
                }
            if rrf_rank_constant is not None:
                rerank_kwargs["rrf_k"] = rrf_rank_constant
            fused = endee_rerank(raw_results, **rerank_kwargs)
            results = fused["results"]
        else:
            per_field = raw_results.get("results", {})
            results = per_field.get(self.dense_field_name, [])

        return self._results_to_docs(results)

    def _results_to_docs(
        self, results: list[dict[str, Any]]
    ) -> list[tuple[Document, float]]:
        """Convert raw search results to ``(Document, score)`` tuples."""
        docs = []
        for res in results:
            meta = res.get("meta", {})
            text = meta.get(self.content_payload_key, "")
            metadata = meta.get(self.metadata_payload_key, {})

            metadata["_id"] = res.get("id")
            if "filter" in res:
                metadata["_filter"] = res["filter"]

            score = res.get("similarity", 0.0)

            if not text:
                logger.warning(
                    f"Found document with no `{self.content_payload_key}` key. "
                    "Skipping."
                )
                continue
            docs.append((Document(page_content=text, metadata=metadata), score))
        return docs

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def delete(
        self,
        ids: list[str] | None = None,
        filter: list[dict[str, Any]] | None = None,  # noqa: A002
        **kwargs: Any,
    ) -> bool | None:
        """Delete by IDs or filter. Returns True on success, False on error."""
        if ids is None and filter is None:
            msg = "Either ids or filter must be provided"
            raise ValueError(msg)

        try:
            if ids is not None:
                failed = []
                for doc_id in ids:
                    try:
                        self._collection.delete_object(doc_id)
                    except Exception as e:
                        logger.warning(f"Error deleting object {doc_id}: {e}")
                        failed.append(doc_id)
                if failed:
                    logger.error(f"Failed to delete {len(failed)} object(s)")
                    return False
            elif filter is not None:
                self._collection.delete_by_filter(filter=filter)
            return True
        except Exception as e:
            logger.error(f"Error during deletion: {e}")
            return False

    def get_by_ids(self, ids: Sequence[str]) -> list[Document]:
        """Retrieve documents by their IDs."""
        if not ids:
            return []

        docs = []
        try:
            results = self._collection.get_objects(list(ids))
            for result in results:
                meta = result.get("meta", {})
                text = meta.get(self.content_payload_key, "")
                metadata = meta.get(self.metadata_payload_key, {})
                metadata["_id"] = result.get("id")
                if "filter" in result:
                    metadata["_filter"] = result["filter"]
                if text:
                    docs.append(Document(page_content=text, metadata=metadata))
        except Exception as e:
            logger.warning(f"Error retrieving documents: {e}")
        return docs

    def update_filters(self, updates: list[dict]) -> dict:
        """Update filter metadata for objects by ID (no re-embedding).

        Args:
            updates: List of ``{"id": str, "filter": dict}`` dicts.

        Returns:
            Server response dict, e.g. ``{"updated": 3}``.
        """
        if not updates:
            msg = "updates must be a non-empty list"
            raise ValueError(msg)
        try:
            return self._collection.update_filters(updates)
        except Exception as e:
            msg = f"Failed to update filters: {e}"
            raise EndeeVectorStoreError(msg) from e

    # ══════════════════════════════════════════════════════════════════════════
    # Multi-field API
    # ══════════════════════════════════════════════════════════════════════════

    def add_objects(
        self,
        objects: list[dict[str, Any]],
        *,
        batch_size: int = constants.MAX_VECTORS_PER_BATCH,
    ) -> list[str]:
        """Upsert objects with arbitrary per-field data.

        Use this instead of ``add_texts`` when you need to supply vectors
        for multiple fields, sparse fields, or multi_vector fields.

        Each object dict must have at least ``"id"`` and ``"fields"``.
        ``"meta"`` and ``"filter"`` are optional.

        Args:
            objects: List of object dicts. Example::

                [
                    {
                        "id": "doc1",
                        "meta": {"text": "hello world"},
                        "filter": {"category": "greeting"},
                        "fields": {
                            "title":   [0.1, 0.2, ...],            # vector
                            "content": [0.3, 0.4, ...],            # vector
                            "keywords": {"indices": [1,2],         # sparse
                                         "values": [0.9, 0.4]},
                            "colbert": [[0.1, ...], [0.2, ...]],   # multi_vector
                        },
                    },
                ]
            batch_size: Max objects per upsert call. Default: 10000.

        Returns:
            List of upserted object IDs.
        """
        batch_size = self._validate_batch_size(batch_size)
        all_ids = []
        for i in range(0, len(objects), batch_size):
            batch = objects[i : i + batch_size]
            self._collection.upsert(batch)
            all_ids.extend(obj["id"] for obj in batch)
        return all_ids

    def multi_field_search(
        self,
        fields: dict[str, Any],
        *,
        filter: list[dict[str, Any]] | None = None,  # noqa: A002
        ef_search: int = constants.DEFAULT_EF_SEARCH,
        prefilter_cardinality_threshold: int | None = None,
        filter_boost_percentage: int | None = None,
    ) -> dict[str, Any]:
        """Search across one or more fields and return raw per-field results.

        This is a thin wrapper around ``collection.search()`` that gives you
        full control over which fields to query, each with its own query
        vector and limit.

        Args:
            fields: Per-field search config. Each value must be a dict with
                at least a ``"query"`` key::

                    {
                        "title":   {"query": title_vec,   "limit": 20},
                        "content": {"query": content_vec, "limit": 20},
                        "keywords":{"query": {"indices":[..], "values":[..]},
                                    "limit": 20},
                        "colbert": {"query": [[0.1,..],[0.2,..]],
                                    "limit": 10},
                    }
            filter: Metadata filter list (AND logic).
            ef_search: Default ef_search for all fields.
            prefilter_cardinality_threshold: Brute-force threshold (1k-1M).
            filter_boost_percentage: Expand candidate pool (0-100).

        Returns:
            Raw per-field results::

                {"results": {"title": [hit, ...], "content": [hit, ...], ...}}

            Each *hit* is ``{"id", "similarity", "meta", "filter"}``.

        Example:
            >>> raw = store.multi_field_search(fields={
            ...     "title":   {"query": title_vec, "limit": 20},
            ...     "content": {"query": content_vec, "limit": 20},
            ... })
            >>> # fuse with weighted RRF
            >>> from endee import rerank
            >>> fused = rerank(raw, limit=10,
            ...     field_weights={"title": 0.4, "content": 0.6})
        """
        search_kwargs: dict[str, Any] = {
            "fields": fields,
            "ef_search": ef_search,
        }
        if filter is not None:
            search_kwargs["filter"] = filter
        if prefilter_cardinality_threshold is not None:
            search_kwargs["prefilter_cardinality_threshold"] = (
                prefilter_cardinality_threshold
            )
        if filter_boost_percentage is not None:
            search_kwargs["filter_boost_percentage"] = filter_boost_percentage

        return self._collection.search(**search_kwargs)

    def multi_field_search_with_rerank(
        self,
        fields: dict[str, Any],
        *,
        limit: int = 10,
        field_weights: dict[str, float] | None = None,
        rrf_k: int = 60,
        filter: list[dict[str, Any]] | None = None,  # noqa: A002
        ef_search: int = constants.DEFAULT_EF_SEARCH,
        prefilter_cardinality_threshold: int | None = None,
        filter_boost_percentage: int | None = None,
    ) -> list[tuple[Document, float]]:
        """Search multiple fields and fuse with RRF, returning Documents.

        Convenience method that combines ``multi_field_search`` +
        ``endee.rerank`` + Document conversion in one call.

        Args:
            fields: Per-field search config (same as ``multi_field_search``).
            limit: Number of final results after fusion.
            field_weights: Per-field RRF weights, must sum to 1.0.
                Defaults to equal weighting.
            rrf_k: RRF rank constant. Default: 60.
            filter: Metadata filter list.
            ef_search: Default ef_search for all fields.
            prefilter_cardinality_threshold: Brute-force threshold (1k-1M).
            filter_boost_percentage: Expand candidate pool (0-100).

        Returns:
            List of ``(Document, score)`` tuples sorted by fused score.

        Example:
            >>> results = store.multi_field_search_with_rerank(
            ...     fields={
            ...         "title":   {"query": title_vec,   "limit": 30},
            ...         "content": {"query": content_vec, "limit": 30},
            ...         "keywords":{"query": sparse_q,    "limit": 30},
            ...     },
            ...     limit=10,
            ...     field_weights={"title": 0.3, "content": 0.5,
            ...                    "keywords": 0.2},
            ... )
        """
        raw = self.multi_field_search(
            fields=fields, filter=filter, ef_search=ef_search,
            prefilter_cardinality_threshold=prefilter_cardinality_threshold,
            filter_boost_percentage=filter_boost_percentage,
        )

        rerank_kwargs: dict[str, Any] = {"limit": limit, "rrf_k": rrf_k}
        if field_weights is not None:
            rerank_kwargs["field_weights"] = field_weights

        fused = endee_rerank(raw, **rerank_kwargs)
        return self._results_to_docs(fused["results"])

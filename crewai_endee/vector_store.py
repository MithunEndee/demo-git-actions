"""EndeeVectorStore — CrewAI RAGStorage backed by the Endee vector database.

Uses the same ``fields=`` configuration as ``Endee.create_collection()``.
Supports dense-only, hybrid (dense + sparse), and multi-field collections.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from crewai.rag.embeddings.factory import build_embedder
from crewai.rag.storage.base_rag_storage import BaseRAGStorage
from endee import Endee as EndeeClient
from endee import constants, rerank as endee_rerank

if TYPE_CHECKING:
    from crewai_endee.sparse_embeddings import SparseEmbeddings

_logger = logging.getLogger(__name__)

_MAX_TEXT_BYTES = 8192

CONTENT_KEY = "text"
METADATA_KEY = "metadata"


class EndeeVectorStore(BaseRAGStorage):
    """Endee vector store integration for CrewAI.

    All collections are created via ``fields=`` — the same format as
    ``Endee.create_collection()``.

    Example — dense only::

        store = EndeeVectorStore(
            type="my_collection",
            embedder_config=embedder_config,
            fields=[
                {"name": "dense", "type": "vector",
                 "params": {"dimension": 384, "space_type": "cosine",
                            "precision": "int8"}},
            ],
        )

    Example — hybrid (endee_bm25 auto-detected)::

        store = EndeeVectorStore(
            type="my_collection",
            embedder_config=embedder_config,
            fields=[
                {"name": "dense", "type": "vector",
                 "params": {"dimension": 384, "space_type": "cosine",
                            "precision": "int8"}},
                {"name": "sparse", "type": "sparse",
                 "sparse_model": "endee_bm25"},
            ],
        )

    Example — multi-field::

        store = EndeeVectorStore(
            type="my_collection",
            embedder_config=embedder_config,
            fields=[
                {"name": "dense", "type": "vector",
                 "params": {"dimension": 384, "space_type": "cosine",
                            "precision": "int8"}},
                {"name": "chunks", "type": "multi_vector",
                 "params": {"dimension": 128, "space_type": "cosine",
                            "precision": "float16", "pooling": "mean"}},
                {"name": "keywords", "type": "sparse",
                 "sparse_model": "default"},
            ],
        )
    """

    CONTENT_KEY: str = CONTENT_KEY
    METADATA_KEY: str = METADATA_KEY

    def __init__(
        self,
        type: str,
        embedder_config: Any,
        fields: List[Dict[str, Any]],
        allow_reset: bool = True,
        crew: Any = None,
        api_token: Optional[str] = None,
        base_url: Optional[str] = None,
        endee_client: Optional[EndeeClient] = None,
        sparse_embedding: Optional[SparseEmbeddings] = None,
        content_payload_key: str = CONTENT_KEY,
        metadata_payload_key: str = METADATA_KEY,
        force_recreate: bool = False,
    ) -> None:
        """Initialise an Endee-backed vector store.

        Args:
            type: Collection name.
            embedder_config: Configuration dict for the dense embedder.
            fields: Collection field definitions — same format as
                ``Endee.create_collection(fields=...)``. Must contain at
                least one ``"vector"`` type field. Example::

                    fields=[
                        {"name": "dense",    "type": "vector",
                         "params": {"dimension": 384, "space_type": "cosine",
                                    "precision": "int8", "M": 16, "ef_con": 128}},
                        {"name": "sparse",   "type": "sparse",
                         "sparse_model": "endee_bm25"},
                        {"name": "colbert",  "type": "multi_vector",
                         "params": {"dimension": 128, "space_type": "cosine",
                                    "precision": "float16", "pooling": "mean"}},
                    ]

            allow_reset: Whether ``reset()`` is permitted.
            crew: CrewAI crew instance (optional).
            api_token: Endee API token (optional for local deployment).
            base_url: Custom Endee API base URL.
            endee_client: Existing Endee client (overrides ``api_token``).
            sparse_embedding: Sparse embedding model for hybrid search.
                When a sparse field with ``endee_bm25`` is present and
                this is ``None``, an ``EndeeModelSparse`` is created
                automatically.
            content_payload_key: Metadata key for document text.
            metadata_payload_key: Metadata key for user metadata.
            force_recreate: Delete and recreate collection if it exists.
        """
        self.type = type
        self.crew = crew
        self.embedder = build_embedder(embedder_config)
        self.content_payload_key = content_payload_key
        self.metadata_payload_key = metadata_payload_key
        self._fields = fields
        self._collection = None

        # Sparse embedding setup
        if sparse_embedding is not None:
            from crewai_endee.sparse_embeddings import wrap_sparse_model

            self._sparse_embeddings = wrap_sparse_model(sparse_embedding)
        else:
            self._sparse_embeddings = None

        # Detect field roles from fields config
        self.dense_field_name = None
        self.sparse_field_name = None
        self._detect_field_roles(fields)

        # Client init
        self._init_client(api_token, base_url, endee_client)

        # Collection connect / create
        self._collection = self._connect_collection(type, fields, force_recreate)

        super().__init__(type, allow_reset, embedder_config, crew)

    # ── Field role detection ─────────────────────────────────────────────────

    def _detect_field_roles(self, fields: List[Dict[str, Any]]) -> None:
        """Auto-detect dense and sparse field names from field configs."""
        for f in fields:
            ftype = f.get("type")
            fname = f.get("name")

            if ftype == "vector" and self.dense_field_name is None:
                self.dense_field_name = fname

            elif ftype == "sparse" and self.sparse_field_name is None:
                self.sparse_field_name = fname

        if self.dense_field_name is None:
            raise ValueError(
                "fields must contain at least one 'vector' type field"
            )

    # ── Client init ──────────────────────────────────────────────────────────

    def _init_client(
        self,
        api_token: Optional[str],
        base_url: Optional[str],
        endee_client: Optional[EndeeClient],
    ) -> None:
        if endee_client is not None:
            self._client = endee_client
        elif api_token is not None:
            self._client = EndeeClient(token=api_token)
        else:
            self._client = EndeeClient()
        if base_url:
            self._client.set_base_url(base_url)

    # ── Collection connect / create ──────────────────────────────────────────

    def _connect_collection(
        self,
        collection_name: str,
        fields: List[Dict[str, Any]],
        force_recreate: bool,
    ) -> Any:
        collection_list = self._client.list_collections()
        collection_exists = any(
            c.get("name") == collection_name for c in collection_list
        )

        if collection_exists and force_recreate:
            _logger.info("Deleting existing collection: %s", collection_name)
            self._client.delete_collection(collection_name)
            collection_exists = False

        if not collection_exists:
            _logger.info("Creating collection: %s", collection_name)
            self._client.create_collection(name=collection_name, fields=fields)

        collection = self._client.get_collection(name=collection_name)
        self._auto_setup_sparse(collection)
        return collection

    def _auto_setup_sparse(self, collection: Any) -> None:
        """If the sparse field is ``endee_bm25`` and no sparse_embedding was
        provided, auto-create ``EndeeModelSparse``."""
        if self._sparse_embeddings is not None or self.sparse_field_name is None:
            return

        for f in collection.fields:
            if f.get("name") != self.sparse_field_name:
                continue
            sparse_model = f.get("sparse_model") or f.get("params", {}).get(
                "sparse_model"
            )
            if sparse_model == "endee_bm25":
                from crewai_endee.sparse_embeddings import EndeeModelSparse

                self._sparse_embeddings = EndeeModelSparse()
                _logger.info(
                    "Auto-created EndeeModelSparse for sparse field '%s'",
                    f.get("name"),
                )
            break

    # ── Collection lifecycle ─────────────────────────────────────────────────

    def ensure_collection(self):
        """Verify the collection exists and print status.

        Returns:
            The Endee ``Collection`` object.
        """
        mode = "hybrid" if self._sparse_embeddings else "dense"
        print(f"\n[Endee] Collection '{self.type}' ready ({mode} mode).\n")
        return self._collection

    ensure_index = ensure_collection

    def reset(self) -> None:
        """Delete the entire collection and reset internal state."""
        try:
            self._client.delete_collection(self.type)
            _logger.info("Reset/deleted collection: %s", self.type)
        except Exception as exc:
            _logger.info(
                "Collection '%s' did not exist or could not be deleted: %s",
                self.type, exc,
            )
        self._collection = None

    def describe(self) -> dict:
        """Return collection metadata."""
        return self._collection.describe()

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        if self._client is not None:
            try:
                self._client.close_session()
            except AttributeError:
                try:
                    self._client.close_client()
                except AttributeError:
                    pass
            self._collection = None
            _logger.info("Closed Endee connection for '%s'", self.type)

    # ── CRUD operations ──────────────────────────────────────────────────────

    def save(self, value: str, metadata: Dict[str, Any]) -> None:
        """Embed *value* and upsert it into the collection.

        Populates the primary dense field (and sparse field if hybrid).

        Args:
            value: Text content to store.
            metadata: Associated metadata dictionary.
        """
        value = _truncate_text(value)
        embedding = self.embedder([value])[0]
        if hasattr(embedding, "tolist"):
            embedding = embedding.tolist()

        meta = {
            self.content_payload_key: value,
            self.metadata_payload_key: metadata,
        }
        filter_data = {
            k: v for k, v in metadata.items() if isinstance(v, (str, int, float))
        }

        fields_data: Dict[str, Any] = {
            self.dense_field_name: embedding,
        }

        if self._sparse_embeddings is not None and self.sparse_field_name is not None:
            sv = self._sparse_embeddings.embed_documents([value])[0]
            fields_data[self.sparse_field_name] = {
                "indices": sv.indices,
                "values": sv.values,
            }

        entry = {
            "id": uuid.uuid4().hex,
            "meta": meta,
            "filter": filter_data,
            "fields": fields_data,
        }

        self._collection.upsert([entry])
        _logger.debug("Saved document id=%s", entry["id"])

    def search(
        self,
        query: str,
        limit: int = 3,
        filter: Optional[List[Dict[str, Any]]] = None,
        score_threshold: Optional[float] = None,
        ef_search: Optional[int] = None,
        include_vectors: bool = False,
        prefilter_cardinality_threshold: Optional[int] = None,
        filter_boost_percentage: Optional[int] = None,
        field_weights: Optional[Dict[str, float]] = None,
        rrf_k: int = 60,
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors.

        Queries the primary dense field (and sparse field if hybrid).

        Args:
            query: Natural-language search query.
            limit: Maximum number of results.
            filter: MongoDB-style metadata filters.
            score_threshold: Minimum similarity score to include. Omit to
                include all results.
            ef_search: HNSW ef parameter (default: 128, max: 1024).
            include_vectors: Fetch and include raw vector data.
            prefilter_cardinality_threshold: Brute-force pre-filter threshold.
            filter_boost_percentage: Candidate pool expansion (0-100).
            field_weights: Per-field RRF weights (must sum to 1.0).
            rrf_k: RRF rank constant (default: 60).

        Returns:
            List of result dicts with ``id``, ``content``, ``context``,
            ``metadata``, and ``score``.
        """
        query = _truncate_text(query)
        embedding = self.embedder([query])[0]
        if hasattr(embedding, "tolist"):
            embedding = embedding.tolist()

        search_fields: Dict[str, Any] = {
            self.dense_field_name: {
                "query": embedding,
                "limit": limit,
            },
        }

        if self._sparse_embeddings is not None and self.sparse_field_name is not None:
            sv = self._sparse_embeddings.embed_query(query)
            search_fields[self.sparse_field_name] = {
                "query": {"indices": sv.indices, "values": sv.values},
                "limit": limit,
            }

        search_kwargs: Dict[str, Any] = {"fields": search_fields}
        if filter:
            search_kwargs["filter"] = filter
        if ef_search is not None:
            search_kwargs["ef_search"] = ef_search
        if prefilter_cardinality_threshold is not None:
            search_kwargs["prefilter_cardinality_threshold"] = (
                prefilter_cardinality_threshold
            )
        if filter_boost_percentage is not None:
            search_kwargs["filter_boost_percentage"] = filter_boost_percentage

        try:
            raw_results = self._collection.search(**search_kwargs)
        except Exception as exc:
            _logger.error("Error querying Endee: %s", exc)
            return []

        per_field = raw_results.get("results", {})
        num_fields = len(per_field)

        if num_fields > 1:
            fused = endee_rerank(
                raw_results,
                name="rrf",
                limit=limit,
                field_weights=field_weights,
                rrf_k=rrf_k,
            )
            results = fused.get("results", [])
        elif num_fields == 1:
            field_name = next(iter(per_field))
            results = per_field[field_name]
        else:
            results = []

        output = []
        for r in results:
            if score_threshold is not None and r.get("similarity", 0) < score_threshold:
                continue
            meta = r.get("meta", {})
            item: Dict[str, Any] = {
                "id": r.get("id", ""),
                "metadata": meta.get(self.metadata_payload_key, {}),
                "context": meta.get(self.content_payload_key, ""),
                "content": meta.get(self.content_payload_key, ""),
                "score": r.get("similarity", 0),
            }
            output.append(item)

        if include_vectors and output:
            ids = [item["id"] for item in output]
            try:
                full_objects = self._collection.get_objects(ids)
                obj_map = {obj["id"]: obj for obj in full_objects}
                for item in output:
                    full = obj_map.get(item["id"], {})
                    item["vectors"] = full.get("vectors", {})
                    if full.get("sparses"):
                        item["sparses"] = full["sparses"]
                    if full.get("multi_vectors"):
                        item["multi_vectors"] = full["multi_vectors"]
            except Exception as exc:
                _logger.warning("Failed to fetch vectors: %s", exc)

        return output

    # ── Single-object helpers ────────────────────────────────────────────────

    def get_objects(self, ids: List[str]) -> List[Dict[str, Any]]:
        """Retrieve objects by ID."""
        return self._collection.get_objects(ids)

    def get_vector(self, id: str) -> Dict[str, Any]:
        """Retrieve a single object by ID."""
        results = self.get_objects([id])
        return results[0] if results else {}

    def delete_vector(self, id: str) -> Dict[str, Any]:
        """Delete a single object by ID."""
        result = self._collection.delete_object(id)
        _logger.info("Deleted object id=%s: %s", id, result)
        return result

    def delete(self, filter: Any) -> Dict[str, Any]:
        """Delete all objects matching a metadata filter."""
        if isinstance(filter, dict):
            filter = [filter]
        result = self._collection.delete_by_filter(filter)
        _logger.info("Deleted objects with filter=%s: %s", filter, result)
        return result

    def update_filters(self, updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Update filter metadata on existing objects."""
        result = self._collection.update_filters(updates=updates)
        _logger.debug("update_filters result: %s", result)
        return result

    # ── Multi-field API ──────────────────────────────────────────────────────

    def add_objects(self, objects: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Upsert arbitrary per-field data (multi-field mode).

        Each object: ``{"id": str, "meta": dict, "filter": dict,
        "fields": {"field_name": value, ...}}``.
        """
        return self._collection.upsert(objects)

    def multi_field_search(
        self,
        fields: Dict[str, Any],
        filter: Optional[List[Dict[str, Any]]] = None,
        ef_search: int = constants.DEFAULT_EF_SEARCH,
    ) -> Dict[str, Any]:
        """Search multiple fields, return raw per-field results.

        ``fields`` maps field names to ``{"query": ..., "limit": ...}``.
        Returns ``{"results": {field_name: [hit, ...], ...}}``.
        """
        kwargs: Dict[str, Any] = {"fields": fields, "ef_search": ef_search}
        if filter:
            kwargs["filter"] = filter
        return self._collection.search(**kwargs)

    # ── Internal ─────────────────────────────────────────────────────────────

    def _sanitize_role(self, role: str) -> str:
        return role.replace(" ", "_").replace("/", "_").lower()


def _truncate_text(text: str) -> str:
    """Truncate *text* to fit within ``_MAX_TEXT_BYTES`` UTF-8 bytes."""
    encoded = text.encode("utf-8")
    if len(encoded) <= _MAX_TEXT_BYTES:
        return text
    _logger.debug("Text truncated to %d bytes", _MAX_TEXT_BYTES)
    return encoded[:_MAX_TEXT_BYTES].decode("utf-8", errors="ignore")

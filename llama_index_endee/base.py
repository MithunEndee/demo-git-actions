"""Endee Vector Store integration for LlamaIndex.

Implements ``BasePydanticVectorStore`` using the Endee v2 Collections API.
Supports dense, hybrid (dense + sparse), metadata-filtered, and
multi-field collections with RRF-tuned search.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import (
    DEFAULT_TEXT_KEY,
    legacy_metadata_dict_to_node,
    metadata_dict_to_node,
    node_to_metadata_dict,
)

from .constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EF_SEARCH,
    MAX_VECTORS_PER_BATCH,
    REVERSE_OPERATOR_MAP,
    SUPPORTED_FILTER_OPERATORS,
)

if TYPE_CHECKING:
    from .sparse_embeddings import SparseEmbeddings

try:
    from endee import Endee
    from endee import rerank as endee_rerank
except ImportError as e:
    raise ImportError(
        "Could not import endee. Please install it with `pip install endee`."
    ) from e

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FILTERABLE_FIELDS = (
    "file_name",
    "doc_id",
    "category",
    "difficulty",
    "language",
    "field",
    "type",
    "feature",
)

DENSE_FIELD_NAME = "dense"
SPARSE_FIELD_NAME = "sparse"


class EndeeVectorStore(BasePydanticVectorStore):
    """LlamaIndex vector store backed by the Endee vector database.

    Uses the Endee v2 Collections API with typed fields.

    **Simple mode** (default): one dense field, optional sparse field.
    **Multi-field mode**: pass ``fields=`` to create collections with
    multiple vector, sparse, or multi_vector fields.

    Examples:
        >>> # Dense only
        >>> store = EndeeVectorStore.from_params(
        ...     api_token="...", collection_name="my_collection",
        ...     dimension=384,
        ... )
        >>> # Hybrid with BM25
        >>> from llama_index_endee import EndeeModelSparse
        >>> store = EndeeVectorStore.from_params(
        ...     api_token="...", collection_name="bm25_collection",
        ...     dimension=384,
        ...     sparse_embedding=EndeeModelSparse(),
        ... )
        >>> # Multi-field mode
        >>> store = EndeeVectorStore.from_params(
        ...     api_token="...", collection_name="multi_collection",
        ...     fields=[
        ...         {"name": "dense", "type": "vector",
        ...          "params": {"dimension": 384, "space_type": "cosine",
        ...                     "precision": "int8"}},
        ...         {"name": "sparse", "type": "sparse",
        ...          "sparse_model": "endee_bm25"},
        ...     ],
        ... )
    """

    stores_text: bool = True
    flat_metadata: bool = False
    api_token: Optional[str] = None
    base_url: Optional[str] = None
    collection_name: Optional[str] = None
    space_type: Optional[str] = "cosine"
    dimension: Optional[int] = None
    precision: Optional[str] = "int8"
    text_key: str = DEFAULT_TEXT_KEY
    batch_size: int = DEFAULT_BATCH_SIZE
    remove_text_from_metadata: bool = False
    dense_field_name: str = DENSE_FIELD_NAME
    sparse_field_name: str = SPARSE_FIELD_NAME
    hybrid: bool = False

    _collection: Any = PrivateAttr()
    _client: Any = PrivateAttr()
    _sparse_embeddings: Optional[Any] = PrivateAttr(default=None)
    _custom_fields: Optional[List[Dict[str, Any]]] = PrivateAttr(default=None)

    def __init__(  # noqa: D107
        self,
        api_token: Optional[str] = None,
        base_url: Optional[str] = None,
        collection_name: Optional[str] = None,
        space_type: Optional[str] = "cosine",
        dimension: Optional[int] = None,
        text_key: str = DEFAULT_TEXT_KEY,
        batch_size: int = DEFAULT_BATCH_SIZE,
        remove_text_from_metadata: bool = False,
        precision: Optional[str] = "int8",
        M: Optional[int] = None,
        ef_con: Optional[int] = None,
        sparse_embedding: Optional[SparseEmbeddings] = None,
        dense_field_name: str = DENSE_FIELD_NAME,
        sparse_field_name: str = SPARSE_FIELD_NAME,
        fields: Optional[List[Dict[str, Any]]] = None,
        endee_client: Optional[Any] = None,
        endee_collection: Optional[Any] = None,
        force_recreate: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            collection_name=collection_name,
            api_token=api_token,
            base_url=base_url,
            space_type=space_type,
            dimension=dimension,
            precision=precision,
            text_key=text_key,
            batch_size=batch_size,
            remove_text_from_metadata=remove_text_from_metadata,
            dense_field_name=dense_field_name,
            sparse_field_name=sparse_field_name,
        )

        # Set private attrs AFTER super().__init__ (Pydantic resets PrivateAttr
        # defaults during __init__, so assignments before super() are lost)
        if sparse_embedding is not None:
            from .sparse_embeddings import wrap_sparse_model

            self._sparse_embeddings = wrap_sparse_model(sparse_embedding)
        else:
            self._sparse_embeddings = None

        self._custom_fields = fields

        # Init client
        self._init_client(api_token, base_url, endee_client)

        # Connect / create collection
        if endee_collection is not None:
            self._collection = endee_collection
        else:
            self._collection = self._connect_collection(
                collection_name,
                dimension,
                space_type,
                precision,
                M=M,
                ef_con=ef_con,
                sparse_embedding=sparse_embedding,
                force_recreate=force_recreate,
            )

        # Update hybrid flag
        self.hybrid = self._sparse_embeddings is not None

    # ------------------------------------------------------------------
    # Client init
    # ------------------------------------------------------------------

    def _init_client(
        self,
        api_token: Optional[str],
        base_url: Optional[str],
        endee_client: Optional[Any],
    ) -> None:
        if endee_client is not None:
            self._client = endee_client
        elif api_token:
            self._client = Endee(token=api_token)
        else:
            self._client = Endee()
        if base_url:
            self._client.set_base_url(base_url)

    # ------------------------------------------------------------------
    # Collection connect / create
    # ------------------------------------------------------------------

    def _connect_collection(
        self,
        collection_name: Optional[str],
        dimension: Optional[int],
        space_type: Optional[str],
        precision: Optional[str],
        M: Optional[int] = None,
        ef_con: Optional[int] = None,
        sparse_embedding: Optional[SparseEmbeddings] = None,
        force_recreate: bool = False,
    ) -> Any:
        collection_list = self._client.list_collections()
        collection_exists = any(
            c.get("name") == collection_name for c in collection_list
        )

        if collection_exists and force_recreate:
            logger.info("Deleting existing collection: %s", collection_name)
            self._client.delete_collection(collection_name)
            collection_exists = False

        if not collection_exists:
            if self._custom_fields is not None:
                # Multi-field mode: use raw fields directly
                self._create_collection_raw(collection_name, self._custom_fields)
            else:
                # Simple mode: auto-build fields from params
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
            logger.info("Using existing collection: %s", collection_name)

        collection = self._client.get_collection(name=collection_name)
        self._detect_field_names(collection)
        return collection

    def _detect_field_names(self, collection: Any) -> None:
        """Auto-detect primary dense/sparse field names from collection metadata.

        When a sparse field with ``sparse_model="endee_bm25"`` is found and
        no ``sparse_embedding`` was provided, automatically creates an
        ``EndeeModelSparse`` instance.
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
                sparse_model = (
                    f.get("sparse_model")
                    or f.get("params", {}).get("sparse_model")
                    or ""
                )
                if sparse_model == "endee_bm25" and self._sparse_embeddings is None:
                    from .sparse_embeddings import EndeeModelSparse

                    self._sparse_embeddings = EndeeModelSparse()
                    self.hybrid = True
                    logger.info(
                        "Auto-created EndeeModelSparse for "
                        "sparse field '%s' (endee_bm25)",
                        fname,
                    )

    def _create_collection(
        self,
        name: Optional[str],
        dimension: Optional[int],
        space_type: Optional[str],
        precision: Optional[str],
        M: Optional[int] = None,
        ef_con: Optional[int] = None,
        sparse_model: Optional[str] = None,
    ) -> None:
        """Create a collection in simple mode (one dense + optional sparse)."""
        params: Dict[str, Any] = {
            "dimension": dimension,
            "space_type": space_type,
            "precision": precision,
        }
        if M is not None:
            params["M"] = M
        if ef_con is not None:
            params["ef_con"] = ef_con

        field_list: List[Dict[str, Any]] = [
            {
                "name": self.dense_field_name,
                "type": "vector",
                "params": params,
            }
        ]
        if sparse_model is not None:
            field_list.append(
                {
                    "name": self.sparse_field_name,
                    "type": "sparse",
                    "sparse_model": sparse_model,
                }
            )

        mode = "hybrid" if sparse_model else "dense"
        logger.info("Creating %s collection '%s' (dim=%s)", mode, name, dimension)
        self._client.create_collection(name=name, fields=field_list)

    def _create_collection_raw(
        self,
        name: Optional[str],
        fields: List[Dict[str, Any]],
    ) -> None:
        """Create a collection in multi-field mode (raw fields)."""
        logger.info("Creating collection '%s' (multi-field)", name)
        self._client.create_collection(name=name, fields=fields)

    @staticmethod
    def _detect_sparse_model(
        sparse_embedding: Optional[SparseEmbeddings],
    ) -> Optional[str]:
        """Detect the sparse_model type from the embedding object."""
        if sparse_embedding is None:
            return None
        from .sparse_embeddings import EndeeModelSparse

        if isinstance(sparse_embedding, EndeeModelSparse):
            return "endee_bm25"
        return "default"

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_params(
        cls,
        api_token: Optional[str] = None,
        base_url: Optional[str] = None,
        collection_name: Optional[str] = None,
        dimension: Optional[int] = None,
        space_type: str = "cosine",
        precision: Optional[str] = "int8",
        M: Optional[int] = None,
        ef_con: Optional[int] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        sparse_embedding: Optional[SparseEmbeddings] = None,
        dense_field_name: str = DENSE_FIELD_NAME,
        sparse_field_name: str = SPARSE_FIELD_NAME,
        fields: Optional[List[Dict[str, Any]]] = None,
        endee_client: Optional[Any] = None,
        endee_collection: Optional[Any] = None,
        force_recreate: bool = False,
    ) -> EndeeVectorStore:
        """Recommended way to create an EndeeVectorStore.

        Creates a new Endee collection or reconnects to an existing one,
        reads back the actual config from the backend, and returns a
        fully configured store instance.

        Args:
            api_token: Endee API token. None for local server.
            base_url: Override the Endee server URL.
            collection_name: Name of the Endee collection.
            dimension: Vector dimension. Required for new collections in
                simple mode; ignored when ``fields`` is provided.
            space_type: Distance metric (``"cosine"``, ``"l2"``, or ``"ip"``).
            precision: Vector precision
                (``"float32"``, ``"float16"``, ``"int16"``, ``"int8"``,
                ``"binary"``).
            M: HNSW bi-directional links per node.
            ef_con: HNSW construction quality.
            batch_size: Vectors per upsert batch.
            sparse_embedding: Sparse embedding model for hybrid search.
                Pass ``EndeeModelSparse()`` for BM25 or a
                ``SparseModelAdapter``-wrapped SPLADE model.
            dense_field_name: Name of the primary dense vector field.
            sparse_field_name: Name of the sparse vector field.
            fields: **Multi-field mode.** Raw field definitions passed
                directly to ``create_collection``. Example::

                    fields=[
                        {"name": "dense", "type": "vector",
                         "params": {"dimension": 384, "space_type": "cosine",
                                    "precision": "int8"}},
                        {"name": "sparse", "type": "sparse",
                         "sparse_model": "endee_bm25"},
                    ]
            endee_client: Existing Endee client (overrides ``api_token``).
            endee_collection: Pre-existing Endee Collection object.
                Skips collection creation/lookup when provided.
            force_recreate: Delete and recreate collection if it exists.
        """
        return cls(
            api_token=api_token,
            base_url=base_url,
            collection_name=collection_name,
            dimension=dimension,
            space_type=space_type,
            precision=precision,
            M=M,
            ef_con=ef_con,
            batch_size=batch_size,
            sparse_embedding=sparse_embedding,
            dense_field_name=dense_field_name,
            sparse_field_name=sparse_field_name,
            fields=fields,
            endee_client=endee_client,
            endee_collection=endee_collection,
            force_recreate=force_recreate,
        )

    @classmethod
    def class_name(cls) -> str:
        """Return the class name for LlamaIndex registration."""
        return "EndeeVectorStore"

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------

    def add(self, nodes: List[BaseNode], **add_kwargs: Any) -> List[str]:
        """Add nodes with batching, deduplication, and optional sparse encoding."""
        use_sparse = self.hybrid

        # Deduplicate by node ID (last occurrence wins)
        seen: Dict[str, int] = {}
        for idx, node in enumerate(nodes):
            seen[node.node_id] = idx
        nodes = [nodes[i] for i in sorted(seen.values())]

        # Compute sparse vectors in batch if hybrid mode
        sparse_indices: List[List[int]] = []
        sparse_values: List[List[float]] = []
        if use_sparse and self._sparse_embeddings is not None:
            texts = [node.get_content() for node in nodes]
            if texts:
                sparse_vectors = self._sparse_embeddings.embed_documents(texts)
                sparse_indices = [sv.indices for sv in sparse_vectors]
                sparse_values = [sv.values for sv in sparse_vectors]
            else:
                sparse_indices = [[] for _ in nodes]
                sparse_values = [[] for _ in nodes]

        # Build upsert entries
        ids: List[str] = []
        entries: List[Dict[str, Any]] = []
        for i, node in enumerate(nodes):
            metadata = node_to_metadata_dict(node)
            filter_data = self._extract_filter_fields(node, metadata)

            fields_data: Dict[str, Any] = {
                self.dense_field_name: node.get_embedding(),
            }
            if use_sparse and sparse_indices:
                fields_data[self.sparse_field_name] = {
                    "indices": sparse_indices[i],
                    "values": sparse_values[i],
                }

            entry: Dict[str, Any] = {
                "id": node.node_id,
                "meta": metadata,
                "filter": filter_data,
                "fields": fields_data,
            }

            ids.append(node.node_id)
            entries.append(entry)

        # Batch upsert
        batch_size = min(self.batch_size, MAX_VECTORS_PER_BATCH)
        for i in range(0, len(entries), batch_size):
            self._collection.upsert(entries[i : i + batch_size])

        return ids

    @staticmethod
    def _extract_filter_fields(
        node: BaseNode, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract filterable metadata fields for the Endee filter dict."""
        filter_data: Dict[str, Any] = {}
        ref_id = getattr(node, "ref_doc_id", None) or metadata.get("ref_doc_id")
        if ref_id is not None:
            filter_data["ref_doc_id"] = ref_id
        for field in _FILTERABLE_FIELDS:
            if field in metadata:
                filter_data[field] = metadata[field]
        return filter_data

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Delete all objects matching a ref_doc_id filter."""
        self._collection.delete_by_filter([{"ref_doc_id": {"$eq": ref_doc_id}}])

    def delete_vector(self, vector_id: str, **delete_kwargs: Any) -> Any:
        """Delete a single object by ID."""
        return self._collection.delete_object(vector_id)

    def clear(self) -> None:
        """Delete the entire collection and all its objects."""
        self._client.delete_collection(name=self.collection_name)

    @property
    def client(self) -> Any:
        """The underlying Endee Collection object for direct SDK access."""
        return self._collection

    def describe(self) -> Dict[str, Any]:
        """Collection metadata (name, fields, count, etc.)."""
        try:
            return self._collection.describe()
        except Exception as e:
            logger.error("Failed to describe collection: %s", e)
            return {}

    def fetch(self, ids: List[str]) -> List[Dict[str, Any]]:
        """Fetch full object data by IDs."""
        try:
            return self._collection.get_objects(ids)
        except Exception as e:
            logger.warning("Failed to fetch objects: %s", e)
            return []

    def update_filters(self, updates: List[Dict[str, Any]]) -> Any:
        """Replace filter metadata on objects.

        Each update: ``{"id": ..., "filter": {...}}``.
        """
        return self._collection.update_filters(updates)

    # ------------------------------------------------------------------
    # Multi-field API
    # ------------------------------------------------------------------

    def add_objects(
        self,
        objects: List[Dict[str, Any]],
        batch_size: Optional[int] = None,
    ) -> List[str]:
        """Upsert objects with arbitrary per-field data (multi-field mode).

        Each object: ``{"id": str, "meta": dict, "filter": dict,
        "fields": {"field_name": value, ...}}``.
        """
        if batch_size is None:
            batch_size = min(self.batch_size, MAX_VECTORS_PER_BATCH)
        all_ids: List[str] = []
        for i in range(0, len(objects), batch_size):
            batch = objects[i : i + batch_size]
            self._collection.upsert(batch)
            all_ids.extend(obj["id"] for obj in batch)
        return all_ids

    def multi_field_search(
        self,
        fields: Dict[str, Any],
        filter: Optional[List[Dict[str, Any]]] = None,
        ef_search: int = DEFAULT_EF_SEARCH,
        prefilter_cardinality_threshold: Optional[int] = None,
        filter_boost_percentage: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Search multiple fields, return raw per-field results.

        ``fields`` maps field names to ``{"query": ..., "limit": ...}``.
        Returns ``{"results": {field_name: [hit, ...], ...}}``.
        """
        kwargs: Dict[str, Any] = {"fields": fields, "ef_search": ef_search}
        if filter:
            kwargs["filter"] = filter
        if prefilter_cardinality_threshold is not None:
            kwargs["prefilter_cardinality_threshold"] = (
                prefilter_cardinality_threshold
            )
        if filter_boost_percentage is not None:
            kwargs["filter_boost_percentage"] = filter_boost_percentage
        return self._collection.search(**kwargs)

    # ------------------------------------------------------------------
    # Query internals
    # ------------------------------------------------------------------

    def _process_filters(
        self, query: VectorStoreQuery
    ) -> Optional[List[Dict[str, Any]]]:
        """Convert LlamaIndex MetadataFilters -> Endee API filter format."""
        if query.filters is None:
            return None

        filters: Dict[str, Dict[str, Any]] = {}
        for item in query.filters.filters:
            if hasattr(item, "key") and hasattr(item, "operator"):
                if item.operator not in SUPPORTED_FILTER_OPERATORS:
                    raise ValueError(
                        f"Unsupported filter operator: {item.operator}. "
                        f"Supported: {SUPPORTED_FILTER_OPERATORS}"
                    )
                filters.setdefault(item.key, {})[
                    REVERSE_OPERATOR_MAP[item.operator]
                ] = item.value
            elif isinstance(item, dict):
                for key, op_dict in item.items():
                    if isinstance(op_dict, dict):
                        for op, val in op_dict.items():
                            filters.setdefault(key, {})[op] = val
            else:
                raise ValueError(f"Unsupported filter format: {type(item).__name__}")

        return [{k: v} for k, v in filters.items()] if filters else None

    # ------------------------------------------------------------------
    # Result conversion (Endee dict -> LlamaIndex node)
    # ------------------------------------------------------------------

    def _create_node_from_legacy_metadata(
        self, metadata: Dict[str, Any], node_id: str
    ) -> BaseNode:
        """Reconstruct a TextNode from stored _node_content JSON."""
        metadata_dict, node_info, relationships = legacy_metadata_dict_to_node(
            metadata=metadata,
            text_key=self.text_key,
        )
        try:
            node_content = json.loads(metadata.get("_node_content", "{}"))
        except json.JSONDecodeError:
            node_content = {}

        node = TextNode(
            text=node_content.get(self.text_key, ""),
            metadata=metadata_dict,
            relationships=relationships,
            id_=node_id,
        )
        for key, val in node_info.items():
            if hasattr(node, key):
                setattr(node, key, val)
        return node

    def _process_single_result(
        self, result: Dict[str, Any]
    ) -> Tuple[BaseNode, float, str]:
        """Convert one Endee result dict into (node, score, id)."""
        node_id = result["id"]
        score = result.get("similarity", result.get("score", 0.0))
        metadata = result.get("meta", {})

        node = (
            metadata_dict_to_node(
                metadata=metadata, text=metadata.pop(self.text_key, None), id_=node_id
            )
            if self.flat_metadata
            else self._create_node_from_legacy_metadata(metadata, node_id)
        )
        if "vector" in result:
            node.embedding = result["vector"]
        return node, score, node_id

    def _process_query_results(
        self, results: List[Dict[str, Any]]
    ) -> Tuple[List[BaseNode], List[float], List[str]]:
        """Convert a list of Endee result dicts to LlamaIndex format."""
        nodes: List[BaseNode] = []
        similarities: List[float] = []
        ids: List[str] = []
        for result in results:
            try:
                node, score, node_id = self._process_single_result(result)
                nodes.append(node)
                similarities.append(score)
                ids.append(node_id)
            except Exception as e:
                logger.warning("Skipping result %s: %s", result.get("id", "?"), e)
        return nodes, similarities, ids

    # ------------------------------------------------------------------
    # Public query
    # ------------------------------------------------------------------

    def query(
        self,
        query: VectorStoreQuery,
        ef_search: int = DEFAULT_EF_SEARCH,
        prefilter_cardinality_threshold: Optional[int] = None,
        filter_boost_percentage: Optional[int] = None,
        dense_rrf_weight: Optional[float] = None,
        rrf_rank_constant: Optional[int] = None,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """Query the collection for the top-k most similar nodes.

        Args:
            query: LlamaIndex VectorStoreQuery (embedding, query_str,
                filters, top_k).
            ef_search: HNSW search breadth (1-1024, default 128).
            prefilter_cardinality_threshold: Switch to brute-force below
                this count.
            filter_boost_percentage: Expand candidate pool by this %
                when filtering.
            dense_rrf_weight: RRF weight for dense vs sparse fusion
                (0.0-1.0).
            rrf_rank_constant: RRF rank constant (>=1, default 60).
            **kwargs: Interface compatibility (unused).
        """
        use_sparse = self.hybrid

        # Allow passing tuning params via vector_store_kwargs
        extra: Dict[str, Any] = getattr(query, "query_kwargs", {}) or {}
        if prefilter_cardinality_threshold is None:
            prefilter_cardinality_threshold = extra.get(
                "prefilter_cardinality_threshold"
            )
        if filter_boost_percentage is None:
            filter_boost_percentage = extra.get("filter_boost_percentage")
        if dense_rrf_weight is None:
            dense_rrf_weight = extra.get("dense_rrf_weight")
        if rrf_rank_constant is None:
            rrf_rank_constant = extra.get("rrf_rank_constant")

        filter_for_api = self._process_filters(query)
        query_embedding = query.query_embedding
        top_k = query.similarity_top_k

        # Build search fields
        search_fields: Dict[str, Any] = {
            self.dense_field_name: {
                "query": query_embedding,
                "limit": top_k,
            },
        }

        # Add sparse field if hybrid
        if use_sparse and self._sparse_embeddings is not None:
            query_text = getattr(query, "query_str", None)
            if query_text:
                sv = self._sparse_embeddings.embed_query(query_text)
                search_fields[self.sparse_field_name] = {
                    "query": {
                        "indices": sv.indices,
                        "values": [float(v) for v in sv.values],
                    },
                    "limit": top_k,
                }

        # Build search kwargs
        search_kwargs: Dict[str, Any] = {
            "fields": search_fields,
            "ef_search": ef_search,
        }
        if filter_for_api is not None:
            search_kwargs["filter"] = filter_for_api
        if prefilter_cardinality_threshold is not None:
            search_kwargs["prefilter_cardinality_threshold"] = (
                prefilter_cardinality_threshold
            )
        if filter_boost_percentage is not None:
            search_kwargs["filter_boost_percentage"] = filter_boost_percentage

        try:
            raw_results = self._collection.search(**search_kwargs)
        except Exception as e:
            logger.error("Query failed: %s", e)
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])

        # Process results: handle per-field format
        per_field = raw_results.get("results", {})
        num_fields = len(per_field)

        if num_fields > 1:
            # Multi-field: fuse with RRF
            rerank_kwargs: Dict[str, Any] = {"limit": top_k}
            if dense_rrf_weight is not None:
                rerank_kwargs["field_weights"] = {
                    self.dense_field_name: dense_rrf_weight,
                    self.sparse_field_name: 1.0 - dense_rrf_weight,
                }
            if rrf_rank_constant is not None:
                rerank_kwargs["rrf_k"] = rrf_rank_constant
            fused = endee_rerank(raw_results, **rerank_kwargs)
            results = fused.get("results", [])
        elif num_fields == 1:
            field_name = next(iter(per_field))
            results = per_field[field_name]
        else:
            results = []

        nodes, similarities, ids = self._process_query_results(results)
        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)

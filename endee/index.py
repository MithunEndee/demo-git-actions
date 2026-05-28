"""
Index Module for Endee Vector Database Client

This module provides the Index class for performing vector operations including
upsert, query, delete, and retrieval on vector indices. It supports both dense
and hybrid (dense + sparse) vector operations.
"""

from typing import Any, Dict, List

import msgpack
import numpy as np
import orjson
import requests

from ._pydantic_compat import to_dict
from .compression import json_unzip, json_zip
from .constants import (
    CHECKSUM,
    COUNT_FIELD,
    DEFAULT_DENSE_RRF_WEIGHT,
    DEFAULT_EF_SEARCH,
    DEFAULT_FILTER_BOOST_PERCENTAGE,
    DEFAULT_PREFILTER_CARDINALITY_THRESHOLD,
    DEFAULT_RRF_RANK_CONSTANT,
    DEFAULT_TOPK,
    DIMENSION_FIELD,
    EF_CONSTRUCTION_FIELD,
    IS_HYBRID_FIELD,
    MAX_CONNECTIONS_FIELD,
    MAX_KEY_BYTES,
    MAX_VALUE_BYTES,
    MAX_VECTORS_PER_BATCH,
    NAME_FIELD,
    PRECISION_FIELD,
    SPACE_TYPE_FIELD,
    SPARSE_MODEL_FIELD,
)
from .exceptions import raise_exception
from .schema import (
    FilterUpdateRequest,
    IndexMetadata,
    IndexRebuildRequest,
    QueryRequest,
    VectorItem,
)


class Index:
    """
    Vector index for storing and querying embeddings.

    Provides operations for upserting vectors, performing similarity searches,
    deleting vectors, and retrieving vector metadata. Supports both dense-only
    and hybrid (dense + sparse) vector configurations.

    Attributes:
        name (str): Index name
        token (str): Authentication token
        url (str): Base API URL
        version (int): API version
        checksum (str): Defult Checksum value:-1
        lib_token (str): Library authentication token
        count (int): Total number of vectors in the index
        space_type (str): Distance metric ('cosine', 'l2', 'ip')
        dimension (int): Dense vector dimensionality
        precision (str): Vector precision type
        M (int): HNSW M parameter (bi-directional links per node)
        sparse_model (str): Sparse vector model (default/endee_bm25)
        session_client_manager: Shared HTTP session/client manager
    """

    def __init__(
        self,
        name: str,
        token: str,
        url: str,
        version: int = 1,
        params=None,
        session_client_manager=None,
    ):
        """
        Initialize an Index object.

        Args:
            name: Index name
            token: Authentication token
            url: Base API URL
            version: API version (default: 1)
            params: Index parameters from server (metadata)
            session_client_manager: Shared SessionManager or ClientManager
                from parent Endee client
        """
        metadata = IndexMetadata(**params)
        self.name = name
        self.token = token
        self.url = url
        self.version = version
        self.checksum = CHECKSUM
        self.lib_token = metadata.lib_token
        self.count = metadata.total_elements
        self.space_type = metadata.space_type
        self.dimension = metadata.dimension
        self.precision = metadata.precision
        self.M = metadata.M
        self.ef_con = metadata.ef_con
        self.sparse_model = metadata.sparse_model

        # Use shared HTTP manager from Endee client
        self.session_client_manager = session_client_manager

    def _get_session_client(self) -> requests.Session:
        """
        Get either session or client based on manager type.

        Retrieves the appropriate HTTP client (requests.Session or httpx.Client)
        from the shared manager passed from the parent Endee client.

        Returns:
            requests.Session or httpx.Client: HTTP client for API requests

        Raises:
            ValueError: If manager doesn't have required methods
        """
        if hasattr(self.session_client_manager, "get_session"):
            return self.session_client_manager.get_session()
        elif hasattr(self.session_client_manager, "get_client"):
            return self.session_client_manager.get_client()
        else:
            raise ValueError(
                "Invalid session manager. Initialize Index using "
                "Endee_client_object.get_index(index_name)"
            )

    @property
    def is_hybrid(self):
        """
        Check if index supports hybrid (dense + sparse) vectors.

        Returns:
            bool: True if index has sparse dimension > 0, False otherwise
        """
        return self.sparse_model != "None"

    def __str__(self):
        """
        String representation of the Index.

        Returns:
            str: Index name
        """
        return self.name

    def _validate_and_prepare_vectors(self, raw_vectors: List[List[float]]):
        """
        Validate and prepare vectors from raw input lists.

        Args:
            raw_vectors: List of dense vector lists

        Returns:
            tuple: (vectors_array, norms_array, vectors_list)

        Raises:
            ValueError: If vector data is invalid
        """
        # Extract vectors
        try:
            vectors = np.asarray(raw_vectors, dtype=np.float32)
        except Exception as e:
            raise ValueError(f"Invalid vector data: {e}") from e

        # Validate vector shape
        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Expected shape (N, {self.dimension}), got {vectors.shape}"
            )

        # Validate finite values
        if not np.isfinite(vectors).all():
            raise ValueError("Vectors contain NaN or infinity")

        # Normalize vectors for cosine similarity
        n_vectors = len(raw_vectors)
        if self.space_type == "cosine":
            norms = np.sqrt(np.einsum("ij,ij->i", vectors, vectors))
            np.maximum(norms, 1e-10, out=norms)  # Prevent division by zero
            vectors /= norms[:, None]
        else:
            norms = np.ones(n_vectors, dtype=np.float32)

        return vectors, norms, vectors.tolist()

    def _build_vector_batch_item(
        self,
        item: VectorItem,
        i: int,
        norms: np.ndarray,
        vectors_list: list,
        is_hybrid: bool,
    ):
        """
        Build a single vector batch item.

        Args:
            item: Validated VectorItem
            i: Index in the batch
            norms: Array of vector norms
            vectors_list: List of vectors
            is_hybrid: Whether index is hybrid

        Returns:
            list: Vector batch item

        Raises:
            ValueError: If sparse data is invalid
        """
        # Localize functions for performance
        dumps_func = orjson.dumps
        zip_func = json_zip
        str_func = str
        float_func = float

        item_id = item.id
        item_meta = item.meta
        item_filter = item.filter
        sparse_indices = item.sparse_indices
        sparse_values = item.sparse_values

        has_sparse = sparse_indices is not None

        # XOR logic: hybrid index requires sparse data,
        # dense-only forbids it
        if has_sparse != is_hybrid:
            raise ValueError(
                "Hybrid index requires sparse data(along with dense vectors), "
                "and dense-only index forbids it."
            )

        # Build vector object: [id, meta, filter, norm, vector, ...]
        obj = [
            str_func(item_id),
            zip_func(item_meta),
            dumps_func(item_filter).decode("utf-8"),
            float_func(norms[i]),
            vectors_list[i],
        ]

        # Add sparse components for hybrid indexes
        if is_hybrid:
            obj.extend(
                (
                    sparse_indices,
                    [float_func(v) for v in sparse_values],
                )
            )

        return obj

    def _validate_filter(self, filter: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate filter keys and values against size constraints.
        This method ensures that:
        - Each filter key does not exceed `MAX_KEY_BYTES` when encoded in UTF-8.
        - Each filter value (string or list of strings) does not exceed
        `MAX_VALUE_BYTES` when encoded in UTF-8.

        Args:
            filter (Dict[str, Any]): Dictionary containing filter key-value pairs.

        Returns:
            Dict[str, Any]: The validated filter dictionary.

        Raises:
            ValueError: If a key exceeds `MAX_KEY_BYTES` or a value exceeds
                `MAX_VALUE_BYTES`.
        """
        for key, value in filter.items():
            # Validate key size
            if len(key.encode("utf-8")) > MAX_KEY_BYTES:
                raise ValueError(
                    f"Filter key size must be less than {MAX_KEY_BYTES} bytes"
                )

            # Normalize value to list for uniform validation
            values_to_check = value if isinstance(value, list) else [value]

            for v in values_to_check:
                if isinstance(v, str) and len(v.encode("utf-8")) > MAX_VALUE_BYTES:
                    raise ValueError(
                        f"Value size must not exceed {MAX_VALUE_BYTES} bytes"
                    )
        return filter

    def upsert(self, input_array):
        """
        Insert or update vectors in the index.

        Upserts a batch of vectors with their metadata and optional sparse
        components. If a vector with the same ID exists, it will be updated.

        Args:
            input_array: List of dictionaries, each containing:
                - id (str): Unique vector identifier
                - vector (list): Dense vector of length `dimension`
                - meta (dict, optional): Metadata to store with vector
                - filter (dict, optional): Filter metadata for queries
                - sparse_indices (list, optional): Sparse vector indices
                    (required for hybrid indexes)
                - sparse_values (list, optional): Sparse vector values
                    (required for hybrid indexes)

        Returns:
            Vectors inserted successfully response

        Raises:
            ValueError: If vector data is invalid, dimensions don't match,
                or sparse data is missing/present when not expected

        Example:
            >>> index.upsert([
            ...     {
            ...         "id": "vec1",
            ...         "vector": [0.1, 0.2, 0.3],
            ...         "meta": {"text": "example"},
            ...         "filter": {"category": "A"}
            ...     }
            ... ])
        """
        if len(input_array) > MAX_VECTORS_PER_BATCH:
            raise ValueError(
                f"Cannot insert more than {MAX_VECTORS_PER_BATCH} vectors at a time"
            )

        # Localize for the loop
        is_hybrid = self.is_hybrid
        seen_ids = set()
        duplicate_ids = []
        validated_items = []
        vector_item_cls = VectorItem

        # Combine validation, duplicate check, and vector extraction
        vectors_to_process = []
        for item in input_array:
            v_item = vector_item_cls(**item)
            # CHECK FOR FILTER'S KEY AND VALUE SIZE TO PREVENT MDBX BAD VALSIZE ERROR
            if v_item.filter:
                self._validate_filter(v_item.filter)
            item_id = v_item.id
            if item_id in seen_ids:
                duplicate_ids.append(item_id)
            else:
                seen_ids.add(item_id)
            validated_items.append(v_item)
            vectors_to_process.append(v_item.vector)

        if duplicate_ids:
            raise ValueError(
                f"Duplicate IDs found in input array: {sorted(duplicate_ids)}"
            )

        # Validate and prepare vectors - Passing pre-extracted vectors
        vectors, norms, vectors_list = self._validate_and_prepare_vectors(
            vectors_to_process
        )

        # Build batch - localizing method call for performance
        build_item = self._build_vector_batch_item
        vector_batch = [
            build_item(item, i, norms, vectors_list, is_hybrid)
            for i, item in enumerate(validated_items)
        ]

        serialized_data = msgpack.packb(
            vector_batch, use_bin_type=True, use_single_float=True
        )
        headers = {"Authorization": self.token, "Content-Type": "application/msgpack"}

        http_client = self._get_session_client()

        # Sending the batch to the server
        response = http_client.post(
            f"{self.url}/index/{self.name}/vector/insert",
            headers=headers,
            data=serialized_data,
        )

        if response.status_code != 200:
            raise_exception(response.status_code, response.text)

        return "Vectors inserted successfully"

    def _validate_query_params(self, query: QueryRequest):
        """
        Validate query parameters against index configuration.

        Args:
            query: Validated QueryRequest model

        Raises:
            ValueError: If parameters are invalid for this index
        """
        # Cannot use sparse query on dense-only index
        if query.sparse_indices is not None and not self.is_hybrid:
            raise ValueError(
                "Cannot perform sparse search on a dense-only index. "
                "Create the index with sparse support enabled for hybrid search."
            )

    def _prepare_dense_vector(self, vector):
        """
        Prepare and validate dense query vector.

        Args:
            vector: Input vector

        Returns:
            list: Normalized vector as list

        Raises:
            ValueError: If vector is invalid
        """
        # Convert to numpy array
        vec = np.asarray(vector, dtype=np.float32)

        # Validate shape
        if vec.shape != (self.dimension,):
            raise ValueError(
                f"Vector must have shape ({self.dimension},), got {vec.shape}"
            )

        # Validate finite values
        if not np.isfinite(vec).all():
            raise ValueError("Vector contains NaN or infinity")

        # Normalize for cosine similarity using einsum
        if self.space_type == "cosine":
            norm = np.sqrt(np.einsum("i,i->", vec, vec))
            norm = max(norm, 1e-10)  # Prevent division by zero
            vec = vec / norm

        return vec.tolist()

    def _process_query_results(self, results, top_k, include_vectors):
        """
        Process and format query results.

        Args:
            results: Raw msgpack results from server
            top_k: Number of results requested
            include_vectors: Whether to include vector data

        Returns:
            list: Processed results
        """
        processed_results = []
        results = results[:top_k]

        for result in results:
            similarity = result[0]
            vector_id = result[1]
            meta_data = result[2]
            filter_str = result[3]
            norm_value = result[4]
            vector_data = result[5] if len(result) > 5 else []

            processed = {
                "id": vector_id,
                "similarity": similarity,
                "distance": 1.0 - similarity,
                "meta": json_unzip(meta_data),
                "norm": norm_value,
            }

            # Add filter if present
            if filter_str:
                processed["filter"] = orjson.loads(filter_str)

            # Add vector data if requested
            if include_vectors and vector_data:
                processed["vector"] = list(vector_data)
            else:
                processed["vector"] = []

            processed_results.append(processed)

        return processed_results

    def query(
        self,
        vector=None,
        top_k=DEFAULT_TOPK,
        filter=None,
        ef=DEFAULT_EF_SEARCH,
        include_vectors=False,
        log=False,
        sparse_indices=None,
        sparse_values=None,
        prefilter_cardinality_threshold=DEFAULT_PREFILTER_CARDINALITY_THRESHOLD,
        filter_boost_percentage=DEFAULT_FILTER_BOOST_PERCENTAGE,
        dense_rrf_weight=DEFAULT_DENSE_RRF_WEIGHT,
        rrf_rank_constant=DEFAULT_RRF_RANK_CONSTANT,
    ):
        """
        Search for similar vectors in the index.

        Performs approximate nearest neighbor search using HNSW algorithm.
        Supports dense-only, sparse-only (hybrid indexes), or combined queries.

        Args:
            vector: Dense query vector (required for dense/hybrid search)
            top_k: Number of nearest neighbors to return (default: 10, max: 512)
            filter: Dictionary filter to apply to search results
            ef: HNSW ef_search parameter controlling search accuracy vs speed.
                Higher values improve recall but slow down queries.
                (default: 128, max: 1024)
            include_vectors: If True, include vector data in results
            log: Currently unused (for future logging support)
            sparse_indices: Sparse vector indices (for hybrid indexes)
            sparse_values: Sparse vector values (for hybrid indexes)
            prefilter_cardinality_threshold: Cardinality threshold that controls when
                filtering switches from HNSW filtered search to brute-force
                prefiltering on the matched subset.
                - 1_000     → minimum value, prefilter only for very selective filters
                - 10_000    → prefilter only when filter matches ≤10k vectors (default)
                - 1_000_000 → maximum value, prefilter for almost all filtered searches
                Range: 1,000–1,000,000.
            filter_boost_percentage: Expands the internal HNSW candidate pool by this
                percentage when a filter is active, compensating for vectors filtered
                out during traversal. For example, 20 means fetch 20% more candidates
                before applying the filter.
                Range: 0-400 (default: 0, i.e. no boost).


        Returns:
            list: List of dictionaries containing:
                - id (str): Vector identifier
                - similarity (float): Similarity score (higher = more similar)
                - distance (float): Distance score (1 - similarity)
                - meta (dict): Vector metadata
                - norm (float): L2 norm of the vector
                - filter (dict, optional): Filter metadata if present
                - vector (list, optional): Vector data if include_vectors=True

        Raises:
            ValueError: If parameters are invalid or incompatible with index type

        Example:
             >>> results = index.query(
            ...     vector=[0.1, 0.2, 0.3],
            ...     top_k=5,
            ...     filter=[{"category": {"$eq": "A"}}],
            ...     prefilter_threshold=5000,
            ...     boost_percentage=20,
            ... )
        """
        # Validate and prepare query using Pydantic
        query_params = QueryRequest(
            vector=vector,
            top_k=top_k,
            filter=filter,
            ef=ef,
            include_vectors=include_vectors,
            sparse_indices=sparse_indices,
            sparse_values=sparse_values,
            prefilter_cardinality_threshold=prefilter_cardinality_threshold,
            filter_boost_percentage=filter_boost_percentage,
            dense_rrf_weight=dense_rrf_weight,
            rrf_rank_constant=rrf_rank_constant,
        )

        # Additional index-specific validation
        self._validate_query_params(query_params)

        # Prepare search request headers
        headers = {"Authorization": f"{self.token}", "Content-Type": "application/json"}

        # Prepare search request data
        data = {
            "k": query_params.top_k,
            "ef": query_params.ef,
            "include_vectors": query_params.include_vectors,
            "dense_rrf_weight": query_params.dense_rrf_weight,
            "rrf_rank_constant": query_params.rrf_rank_constant,
        }

        # Add dense vector if provided
        if query_params.vector is not None:
            data["vector"] = self._prepare_dense_vector(query_params.vector)

        # Add sparse query if provided
        if query_params.sparse_indices is not None:
            data["sparse_indices"] = list(query_params.sparse_indices)
            data["sparse_values"] = [float(v) for v in query_params.sparse_values]

        # Add filter if provided
        if query_params.filter:
            data["filter"] = orjson.dumps(query_params.filter).decode("utf-8")

        # Sending filter_params so the server uses consistent defaults
        data["filter_params"] = {
            "prefilter_threshold": query_params.prefilter_cardinality_threshold,
            "boost_percentage": query_params.filter_boost_percentage,
        }

        url = f"{self.url}/index/{self.name}/search"

        # Make API request
        http_client = self._get_session_client()
        response = http_client.post(url, headers=headers, json=data)

        # Handle errors
        if response.status_code != 200:
            raise_exception(response.status_code, response.text)

        # Parse msgpack response
        results = msgpack.unpackb(response.content, raw=False)

        # Process and format results
        return self._process_query_results(results, query_params.top_k, include_vectors)

    def delete_vector(self, id):
        """
        Delete a single vector by ID.

        Args:
            id: Vector identifier to delete

        Returns:
            str: Success message with number of rows deleted

        Raises:
            HTTPError: If deletion fails
        """
        headers = {
            "Authorization": f"{self.token}",
        }

        url = f"{self.url}/index/{self.name}/vector/{id}/delete"

        http_client = self._get_session_client()
        response = http_client.delete(url, headers=headers)

        if response.status_code != 200:
            raise_exception(response.status_code, response.text)

        return response.text + " rows deleted"

    def delete_with_filter(self, filter):
        """
        Delete multiple vectors based on a filter.

        Deletes all vectors that match the provided filter criteria.

        Args:
            filter: Dictionary containing filter criteria

        Returns:
            str: Server response with deletion details

        Raises:
            HTTPError: If deletion fails
        """
        headers = {"Authorization": f"{self.token}", "Content-Type": "application/json"}

        data = {"filter": filter}

        url = f"{self.url}/index/{self.name}/vectors/delete"

        http_client = self._get_session_client()
        response = http_client.delete(url, headers=headers, json=data)

        if response.status_code != 200:
            raise_exception(response.status_code, response.text)

        return response.text

    def get_vector(self, id):
        """
        Retrieve a single vector by ID.

        Fetches the complete vector data including metadata, filter,
        and sparse components (for hybrid indexes).

        Args:
            id: Vector identifier to retrieve

        Returns:
            dict: Dictionary containing:
                - id (str): Vector identifier
                - meta (dict): Decrypted metadata
                - filter (str): Filter metadata
                - norm (float): L2 norm
                - vector (list): Dense vector data
                - sparse_indices (list, optional): Sparse indices for hybrid
                - sparse_values (list, optional): Sparse values for hybrid

        Raises:
            HTTPError: If retrieval fails

        Example:
            >>> vec = index.get_vector("vec1")
            >>> print(vec['meta'])
        """
        headers = {"Authorization": f"{self.token}", "Content-Type": "application/json"}

        url = f"{self.url}/index/{self.name}/vector/get"

        # Use POST method with the ID in the request body
        http_client = self._get_session_client()
        response = http_client.post(url, headers=headers, json={"id": id})

        if response.status_code != 200:
            raise_exception(response.status_code, response.text)

        # Parse the msgpack response
        # Format: [id, meta, filter, norm, vector, sparse_indices, sparse_values]
        vector_obj = msgpack.unpackb(response.content, raw=False)

        result = {
            "id": vector_obj[0],
            "meta": json_unzip(vector_obj[1]),
            "filter": orjson.loads(vector_obj[2]),
            "norm": vector_obj[3],
            "vector": list(vector_obj[4]),
        }

        # Include sparse data if present (for hybrid indexes)
        if len(vector_obj) > 5:
            result["sparse_indices"] = list(vector_obj[5]) if vector_obj[5] else []
        if len(vector_obj) > 6:
            result["sparse_values"] = list(vector_obj[6]) if vector_obj[6] else []

        return result

    def update_filters(self, updates):
        """
        Update filters for multiple vectors by ID.

        Updates the filter metadata for specified vectors without modifying
        the vector data itself or other metadata fields.

        Args:
            updates: List of dictionaries, each containing:
                - id (str): Vector identifier
                - filter (dict): New filter metadata to set

        Returns:
            str: Success message with number of filters updated

        Raises:
            ValueError: If updates parameter is invalid
            HTTPError: If update fails

        Example:
            >>> index.update_filters([
            ...     {"id": "vec1", "filter": {"category": "B"}},
            ...     {"id": "vec2", "filter": {"category": "C", "priority": 1}}
            ... ])
            '2 filters updated'
        """
        # Validate using Pydantic
        update_req = FilterUpdateRequest(updates=updates)
        validated_updates = [to_dict(item) for item in update_req.updates]

        headers = {"Authorization": f"{self.token}", "Content-Type": "application/json"}

        data = {"updates": validated_updates}

        url = f"{self.url}/index/{self.name}/filters/update"

        http_client = self._get_session_client()
        response = http_client.post(url, headers=headers, json=data)

        if response.status_code != 200:
            raise_exception(response.status_code, response.text)

        return response.text

    def _refresh_metadata(self):
        """
        Refresh index metadata from the server.

        Fetches the latest index metadata from the server and updates all
        instance attributes including count, M, ef_con, and other properties.

        Returns:
            dict: Dictionary containing updated metadata

        Raises:
            APIException: If request fails
            NotFoundException: If index does not exist
            ServerException: If server encounters an error
        """
        headers = {
            "Authorization": f"{self.token}",
            "Content-Type": "application/json",
        }

        url = f"{self.url}/index/{self.name}/info"

        http_client = self._get_session_client()
        response = http_client.get(url, headers=headers)

        if response.status_code != 200:
            raise_exception(response.status_code, response.text)

        # Parse and validate metadata
        metadata_dict = response.json()
        metadata = IndexMetadata(**metadata_dict)

        # Update instance attributes
        self.count = metadata.total_elements
        self.space_type = metadata.space_type
        self.dimension = metadata.dimension
        self.precision = metadata.precision
        self.M = metadata.M
        self.ef_con = metadata.ef_con
        self.sparse_model = metadata.sparse_model
        self.lib_token = metadata.lib_token

        return {
            "count": self.count,
            "space_type": self.space_type,
            "dimension": self.dimension,
            "precision": self.precision,
            "M": self.M,
            "ef_con": self.ef_con,
            "sparse_model": self.sparse_model,
            "is_hybrid": self.is_hybrid,
        }

    def rebuild(self, M: int = None, ef_con: int = None):
        """
        Rebuild the index with new HNSW parameters.

        Note: Only M and ef_con parameters can be changed during rebuild. Other index
        properties (dimension, space_type, precision, sparse_model) cannot be modified.

        Args:
            M: bi-directional links per node.
            ef_con: candidate list size during construction.

        Returns:
            dict: Dictionary containing:
                - status (str): Rebuild status
                - previous_config (dict): Previous M and ef_con values
                - new_config (dict): New M and ef_con values
                - total_vectors (int): Total number of vectors to rebuild

        Raises:
            ValueError: If index is empty (count == 0) or parameters are invalid
            APIException: If request fails due to invalid parameters
            ServerException: If server encounters an error

        Example:
            >>> # Rebuild with higher quality parameters
            >>> result = index.rebuild(M=32, ef_con=256)
            >>> print(f"Status: {result['status']}")
            >>> print(f"Total vectors: {result['total_vectors']}")
        """
        # Refresh metadata to get current vector count from server
        self._refresh_metadata()

        # Client-side validation: index must have vectors
        if self.count == 0:
            raise ValueError(
                "Cannot rebuild empty index. Index must contain at least one vector."
            )

        rebuild_params = IndexRebuildRequest(M=M, ef_con=ef_con)

        headers = {
            "Authorization": f"{self.token}",
            "Content-Type": "application/json",
        }

        data = {
            "M": rebuild_params.M,
            "ef_con": rebuild_params.ef_con,
        }

        url = f"{self.url}/index/{self.name}/rebuild"

        http_client = self._get_session_client()
        response = http_client.post(url, headers=headers, json=data)

        if response.status_code != 202:
            raise_exception(response.status_code, response.text)

        # Parse JSON response
        response_data = response.json()

        # Update instance attributes with new values
        self.M = rebuild_params.M
        self.ef_con = rebuild_params.ef_con

        # Return response data
        return response_data

    def rebuild_status(self):
        """
        Get the current rebuild operation status.

        Queries the server for the status of any ongoing or completed rebuild
        operation for this index. Returns detailed information about progress
        and vectors processed.

        Returns:
            dict: Dictionary containing:
                - status (str): Current status - "in_progress", "completed",
                    "failed", or "idle"
                - vectors_processed (int, optional): Number of vectors processed
                - total_vectors (int, optional): Total number of vectors
                - percent_complete (float, optional): Completion percentage (0.0-100.0)

        Raises:
            APIException: If request fails
            NotFoundException: If index does not exist
            ServerException: If server encounters an error

        Example:
            >>> status = index.rebuild_status()
            >>> print(f"Status: {status['status']}")
            >>> if status['status'] == 'in_progress':
            ...     print(f"Progress: {status['percent_complete']}%")
            ...     processed = status['vectors_processed']
            ...     total = status['total_vectors']
            ...     print(f"Processed: {processed}/{total}")
        """
        # Prepare request
        headers = {
            "Authorization": f"{self.token}",
            "Content-Type": "application/json",
        }

        url = f"{self.url}/index/{self.name}/rebuild/status"

        # Make API request
        http_client = self._get_session_client()
        response = http_client.get(url, headers=headers)

        # Handle errors
        if response.status_code != 200:
            raise_exception(response.status_code, response.text)

        # Parse and return JSON response
        return response.json()

    def describe(self):
        """
        Get index metadata and configuration.

        Returns a dictionary containing all index properties including
        name, dimensions, space type, and HNSW parameters.

        Returns:
            dict: Dictionary containing:
                - name (str): Index name
                - space_type (str): Distance metric
                - dimension (int): Dense vector dimensionality
                - sparse_model (str): Sparse vector model
                - is_hybrid (bool): Whether index supports sparse vectors
                - count (int): Total number of vectors
                - precision (str): Vector precision type
                - M (int): HNSW M parameter

        Example:
            >>> info = index.describe()
            >>> print(f"Index has {info['count']} vectors")
        """
        data = {
            NAME_FIELD: self.name,
            SPACE_TYPE_FIELD: self.space_type,
            DIMENSION_FIELD: self.dimension,
            SPARSE_MODEL_FIELD: self.sparse_model,
            IS_HYBRID_FIELD: self.is_hybrid,
            COUNT_FIELD: self.count,
            PRECISION_FIELD: self.precision,
            MAX_CONNECTIONS_FIELD: self.M,
            EF_CONSTRUCTION_FIELD: self.ef_con,
        }
        return data

"""
Endee Client Library

This module provides the main client interface for interacting with the Endee
vector database service. It includes session management, index operations.
"""

import os
from functools import lru_cache

import httpx
import requests

from endee._pydantic_compat import to_dict
from endee.constants import (
    CHECKSUM,
    DEFAULT_EF_CON,
    DEFAULT_M,
    HTTP_HTTPX_1_1_LIBRARY,
    HTTP_HTTPX_2_LIBRARY,
    HTTP_METHODS_ALLOWED,
    HTTP_PROTOCOL,
    HTTP_REQUESTS_LIBRARY,
    HTTP_STATUS_CODES,
    HTTPS_PROTOCOL,
    HTTPX_MAX_CONNECTIONS,
    HTTPX_MAX_KEEPALIVE_CONNECTIONS,
    HTTPX_MAX_RETRIES,
    HTTPX_TIMEOUT_SEC,
    LOCAL_BASE_URL,
    LOCAL_REGION,
    SESSION_MAX_RETRIES,
    SESSION_POOL_CONNECTIONS,
    SESSION_POOL_MAXSIZE,
    Precision,
)
from endee.exceptions import raise_exception
from endee.index import Index
from endee.schema import IndexCreateRequest, IndexMetadata


class SessionManager:
    """
    Centralized session manager with a shared requests.Session.

    This class manages HTTP session pooling and connection reuse for the
    requests library. It ensures thread-safety by tracking the process ID
    and creating new sessions when forking occurs.

    Attributes:
        pool_connections (int): Number of connection pools to cache. Each pool
            maintains connections to a single host. Default: SESSION_POOL_CONNECTIONS
        pool_maxsize (int): Maximum number of connections to save in each pool.
            Controls how many connections can be reused per host.
            Default: SESSION_POOL_MAXSIZE
        max_retries (int): Maximum number of retry attempts for failed requests.
            Retries use exponential backoff. Default: SESSION_MAX_RETRIES
        pool_block (bool): If True, blocks when no connections available in pool.
            If False, raises exception instead. Default: True
    """

    def __init__(
        self,
        pool_connections: int = SESSION_POOL_CONNECTIONS,
        pool_maxsize: int = SESSION_POOL_MAXSIZE,
        max_retries: int = SESSION_MAX_RETRIES,
        pool_block: bool = True,
    ):
        """
        Initialize the SessionManager.

        Args:
            pool_connections: Number of connection pools to cache
            pool_maxsize: Maximum connections per pool
            max_retries: Maximum retry attempts for failed requests
            pool_block: Whether to block when pool is full
        """
        self.pool_connections = pool_connections
        self.pool_maxsize = pool_maxsize
        self.max_retries = max_retries
        self.pool_block = pool_block
        self._session: requests.Session | None = None
        self._pid = None

    def __getstate__(self):
        """
        Prepare object state for pickling.

        Removes session and PID to ensure clean state after unpickling.

        Returns:
            dict: Object state without session and PID
        """
        state = self.__dict__.copy()
        state["_session"] = None
        state["_pid"] = None
        return state

    def get_session(self) -> requests.Session:
        """
        Get or create the shared session.

        Creates a new session if none exists or if the process ID has changed
        (indicating a fork). Configures connection pooling and retry logic.

        Returns:
            requests.Session: Configured session with connection pooling
        """
        pid = os.getpid()
        if self._session is None or self._pid != pid:
            session = requests.Session()

            # Configure adapter with connection pooling and retries
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=self.pool_connections,
                pool_maxsize=self.pool_maxsize,
                max_retries=requests.adapters.Retry(
                    total=self.max_retries,
                    backoff_factor=0.5,
                    status_forcelist=HTTP_STATUS_CODES,
                    allowed_methods=HTTP_METHODS_ALLOWED,
                ),
                pool_block=self.pool_block,
            )

            session.mount(HTTP_PROTOCOL, adapter)
            session.mount(HTTPS_PROTOCOL, adapter)

            self._session = session
            self._pid = pid

        return self._session

    def close_session(self):
        """
        Close the shared session.

        Properly closes all connections in the session pool and resets
        the session state.
        """
        if self._session is not None:
            self._session.close()
            self._session = None
            self._pid = None


class ClientManager:
    """
    Centralized client manager with a shared httpx.Client.

    This class manages HTTP client pooling for the httpx library. It supports
    both HTTP/1.1 and HTTP/2 protocols and ensures thread-safety through
    process ID tracking.

    Attributes:
        max_connections (int): Maximum total connections across all hosts.
            Controls overall connection limit. Default: HTTPX_MAX_CONNECTIONS
        max_keepalive_connections (int): Maximum idle connections to keep alive.
            Idle connections are reused for subsequent requests.
            Default: HTTPX_MAX_KEEPALIVE_CONNECTIONS
        max_retries (int): Maximum retry attempts for failed requests.
            Default: HTTPX_MAX_RETRIES
        timeout (float): Request timeout in seconds. Default: HTTPX_TIMEOUT_SEC
        http2 (bool): Whether to enable HTTP/2 protocol. Default: False
    """

    def __init__(
        self,
        max_connections: int = HTTPX_MAX_CONNECTIONS,
        max_keepalive_connections: int = HTTPX_MAX_KEEPALIVE_CONNECTIONS,
        max_retries: int = HTTPX_MAX_RETRIES,
        timeout: float = HTTPX_TIMEOUT_SEC,
        enable_http2: bool = False,
    ):
        """
        Initialize the ClientManager.

        Args:
            max_connections: Maximum total connections
            max_keepalive_connections: Maximum idle keepalive connections
            max_retries: Maximum retry attempts
            timeout: Request timeout in seconds
            enable_http2: Enable HTTP/2 protocol
        """
        self.max_connections = max_connections
        self.max_keepalive_connections = max_keepalive_connections
        self.max_retries = max_retries
        self.timeout = timeout
        self.http2 = enable_http2
        self._client: httpx.Client | None = None
        self._pid = None

    def __getstate__(self):
        """
        Prepare object state for pickling.

        Removes client and PID to ensure clean state after unpickling.

        Returns:
            dict: Object state without client and PID
        """
        state = self.__dict__.copy()
        state["_client"] = None
        state["_pid"] = None
        return state

    def get_client(self) -> httpx.Client:
        """
        Get or create the shared httpx client.

        Creates a new client if none exists or if the process ID has changed.
        Configures connection limits, retry logic, and HTTP/2 support.

        Returns:
            httpx.Client: Configured client with connection pooling
        """
        pid = os.getpid()

        if self._client is None or self._pid != pid:
            limits = httpx.Limits(
                max_connections=self.max_connections,
                max_keepalive_connections=self.max_keepalive_connections,
            )

            transport = httpx.HTTPTransport(retries=self.max_retries)

            self._client = httpx.Client(
                http2=self.http2,
                limits=limits,
                transport=transport,
                timeout=self.timeout,
            )
            self._pid = pid

        return self._client

    def close_client(self):
        """
        Close the shared httpx client.

        Properly closes all connections and resets the client state.
        """
        if self._client is not None:
            self._client.close()
            self._client = None
            self._pid = None


class Endee:
    """
    Main client for interacting with the Endee vector database service.

    This class provides the primary interface for creating, listing, and
    managing vector indexes. It supports both
    requests and httpx HTTP libraries with configurable connection pooling.

    Attributes:
        token (str | None): Authentication token format
        region (str): Service region (extracted from token or default)
        base_url (str): Base URL for API endpoints
        version (int): API version
        library (str): HTTP library to use ('requests', 'httpx1.1', or 'httpx2')
    """

    def __init__(
        self, token: str | None = None, http_library: str = HTTP_REQUESTS_LIBRARY
    ):
        """
        Initialize the Endee client.

        Args:
            token: Authentication token. If None, uses local configuration.
            http_library: HTTP library to use. Options: 'requests' (default),
                'httpx1.1' (HTTP/1.1), or 'httpx2' (HTTP/2)
                'requests' is default as per our benchmark reports qps and p99
                latency values are almost similar with requests and httpx, so
                we consider using requests in our furthe beta and production
                Endee client.

        Raises:
            ValueError: If unsupported http_library is provided
        """
        self.token = token
        self.region = LOCAL_REGION
        self.base_url = LOCAL_BASE_URL

        # Parse token to extract region if present
        if token:
            token_parts = self.token.split(":")
            if len(token_parts) > 2:
                self.base_url = f"https://{token_parts[2]}.endee.io/api/v1"
                self.token = f"{token_parts[0]}:{token_parts[1]}"

        self.version = 1
        self.library = http_library

        # Initialize appropriate session/client manager based on library choice
        if self.library == HTTP_REQUESTS_LIBRARY:
            # Centralized session manager - shared across all Index objects
            self.session_manager = SessionManager(
                pool_connections=10, pool_maxsize=10, max_retries=3
            )
        elif self.library == HTTP_HTTPX_1_1_LIBRARY:
            # httpx.Client based manager for HTTP/1.1
            self.client_manager = ClientManager(
                max_connections=10, max_keepalive_connections=10, max_retries=3
            )
        elif self.library == HTTP_HTTPX_2_LIBRARY:
            # httpx.Client based manager for HTTP/2
            self.client_manager = ClientManager(
                http2=True,
                max_connections=10,
                max_keepalive_connections=10,
                max_retries=3,
            )
        else:
            raise ValueError(
                "Unsupported library. Only 'requests', 'httpx1.1' and "
                "'httpx2' are supported."
            )

    def _get_session(self) -> requests.Session:
        """
        Get session from the centralized session manager.

        Returns:
            requests.Session: Configured session with connection pooling
        """
        return self.session_manager.get_session()

    def close_session(self):
        """Close the shared requests session and cleanup connections."""
        self.session_manager.close_session()

    def _get_client(self) -> httpx.Client:
        """
        Get client from the centralized client manager.

        Returns:
            httpx.Client: Configured client with connection pooling
        """
        return self.client_manager.get_client()

    def close_client(self):
        """Close the shared httpx client and cleanup connections."""
        self.client_manager.close_client()

    def __str__(self):
        """
        String representation of the Endee client.

        Returns:
            str: The authentication token
        """
        return self.token

    def set_token(self, token: str):
        """
        Set the authentication token.

        Args:
            token: Authentication token to set
        """
        self.token = token
        self.region = self.token.split(":")[1]

    def set_base_url(self, base_url: str):
        """
        Set the base URL for API endpoints.

        Args:
            base_url: Base URL to use for API requests
        """
        self.base_url = base_url

    def create_index(
        self,
        name: str,
        dimension: int,
        space_type: str,
        M: int = DEFAULT_M,
        ef_con: int = DEFAULT_EF_CON,
        precision: str | Precision = Precision.INT8,
        version: int | None = None,
        sparse_model: str | None = None,
    ):
        """
        Create a new vector index.

        Args:
            name: Index name (alphanumeric and underscores only, max length
                defined by MAX_INDEX_NAME_LENGTH_ALLOWED)
            dimension: Vector dimensionality (min 2, max MAX_DIMENSION_ALLOWED)
            space_type: Distance metric ('cosine', 'euclidean', 'ip', etc.)
            M: HNSW parameter - number of bi-directional links per node.
                Higher values improve recall but increase memory usage.
                Default: DEFAULT_M
            ef_con: HNSW construction parameter - size of dynamic candidate list.
                Higher values improve index quality but slow construction.
                Default: DEFAULT_EF_CON
            precision: Vector precision type (Precision.BINARY2,
                Precision.INT8, Precision.INT16, Precision.FLOAT16,
                Precision.FLOAT32). Default: Precision.INT16
            version: API version (optional, uses client version if not specified)
            sparse_model: Sparse index mode. Use "default" to enable sparse
                search without IDF, "endee_bm25" to use
                endee_bm25_server_idf, or omit/"None" for dense-only indexes.

        Returns:
            str: Success message

        Raises:
            ValueError: If parameters are invalid
            HTTPError: If API request fails
        """
        # Validate parameters using Pydantic
        request_data = IndexCreateRequest(
            name=name,
            dimension=dimension,
            space_type=space_type,
            M=M,
            ef_con=ef_con,
            precision=precision,
            version=version,
            sparse_model=sparse_model,
        )

        # Prepare request headers and data
        headers = {"Authorization": f"{self.token}", "Content-Type": "application/json"}
        data = {
            "index_name": request_data.name,
            "dim": request_data.dimension,
            "space_type": request_data.space_type,
            "M": request_data.M,
            "ef_con": request_data.ef_con,
            "checksum": CHECKSUM,
            "precision": request_data.precision,
            "version": request_data.version,
        }

        # Only include sparse_model if it's not None
        if request_data.sparse_model is not None:
            data["sparse_model"] = request_data.sparse_model

        url = f"{self.base_url}/index/create"

        # Make API request using appropriate library
        if self.library == HTTP_REQUESTS_LIBRARY:
            session = self._get_session()
            response = session.post(url, headers=headers, json=data)
        else:  # httpx1.1 or httpx2
            client = self._get_client()
            response = client.post(url, headers=headers, json=data)

        # Handle response
        if response.status_code != 200:
            raise_exception(response.status_code, response.text)

        return "Index created successfully"

    def list_indexes(self):
        """
        List all indexes in the current account.

        Returns:
            list: List of index metadata dictionaries

        Raises:
            HTTPError: If API request fails
        """
        headers = {
            "Authorization": f"{self.token}",
        }

        url = f"{self.base_url}/index/list"

        # Make API request using appropriate library
        if self.library == HTTP_REQUESTS_LIBRARY:
            session = self._get_session()
            response = session.get(url, headers=headers)
        else:  # httpx1.1 or httpx2
            client = self._get_client()
            response = client.get(url, headers=headers)

        # Handle response
        if response.status_code != 200:
            raise_exception(response.status_code, response.text)

        indexes = response.json()
        return indexes

    def delete_index(self, name: str):
        """
        Delete an index.

        Args:
            name: Name of the index to delete

        Returns:
            str: Success message

        Raises:
            HTTPError: If API request fails

        Note:
            TODO - Clear the index from LRU cache when deleted
        """
        headers = {
            "Authorization": f"{self.token}",
        }

        url = f"{self.base_url}/index/{name}/delete"

        # Make API request using appropriate library
        if self.library == HTTP_REQUESTS_LIBRARY:
            session = self._get_session()
            response = session.delete(url, headers=headers)
        else:  # httpx1.1 or httpx2
            client = self._get_client()
            response = client.delete(url, headers=headers)

        # Handle response
        if response.status_code != 200:
            raise_exception(response.status_code, response.text)

        return f"Index {name} deleted successfully"

    @lru_cache(maxsize=10)  # noqa: B019
    def get_index(self, name: str):
        """
        Get an index object for performing vector operations.

        Retrieves index metadata from the server and creates an Index object.
        Results are cached using LRU cache (max 10 entries) for performance.

        Args:
            name: Name of the index to retrieve

        Returns:
            Index: Index object for vector operations

        Raises:
            HTTPError: If API request fails
        """
        headers = {"Authorization": f"{self.token}", "Content-Type": "application/json"}

        url = f"{self.base_url}/index/{name}/info"

        # Get index details from the server
        if self.library == HTTP_REQUESTS_LIBRARY:
            session = self._get_session()
            response = session.get(url, headers=headers)
        else:  # httpx1.1 or httpx2
            client = self._get_client()
            response = client.get(url, headers=headers)

        # Handle response
        if response.status_code != 200:
            raise_exception(response.status_code, response.text)

        data = response.json()

        # Validate index metadata
        metadata = IndexMetadata(**data)

        # Create Index object with appropriate manager
        if self.library == HTTP_REQUESTS_LIBRARY:
            idx = Index(
                name=name,
                token=self.token,
                url=self.base_url,
                version=self.version,
                params=to_dict(metadata, by_alias=True),
                session_client_manager=self.session_manager,
            )
        else:
            idx = Index(
                name=name,
                token=self.token,
                url=self.base_url,
                version=self.version,
                params=to_dict(metadata, by_alias=True),
                session_client_manager=self.client_manager,
            )

        return idx

    def __del__(self):
        """
        Cleanup sessions or client on object deletion.

        Ensures proper cleanup of HTTP connections when the Endee object
        is garbage collected.
        """
        try:
            if self.library == HTTP_REQUESTS_LIBRARY:
                self.close_session()
            else:
                self.close_client()
        except Exception:
            # Silently ignore cleanup errors during garbage collection
            pass

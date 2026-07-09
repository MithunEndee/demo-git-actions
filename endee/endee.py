"""
Endee Client Library

Main client interface for the Endee vector database service (v2 Collections API).
"""

import os

import httpx
import requests

from endee.collection import Collection
from endee.constants import (
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
from endee.utils import validate_collection_name

# Accepted values for database-admin operations (mirror the server; fail fast).
_VALID_DB_TYPES = {"starter", "pro", "scale", "enterprise"}
_VALID_TOKEN_TYPES = {"rw", "r"}


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
    Main client for the Endee vector database service (v2 Collections API).

    Attributes:
        token (str | None): Authentication token
        base_url (str): Base URL for v1 API endpoints (v2 URL is derived automatically)
        library (str): HTTP library — 'requests' (default), 'httpx1.1', or 'httpx2'
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
        self.version = 2

        # Parse token to extract region if present
        if token:
            token_parts = self.token.split(":")
            if len(token_parts) > 2:
                self.base_url = f"https://{token_parts[2]}.endee.io/api/v2"
                self.token = f"{token_parts[0]}:{token_parts[1]}"

        self.library = http_library

        # Initialize appropriate session/client manager based on library choice
        if self.library == HTTP_REQUESTS_LIBRARY:
            # Centralized session manager - shared across all Collection objects
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
                enable_http2=True,
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
            base_url: Base URL to use for API requests (should include /api/v2)
        """
        self.base_url = base_url

    # ── internal HTTP helpers (shared by every method below) ───────────────────
    # The client supports two backends (requests / httpx); these two helpers are
    # the single place that knows how to dispatch to either, attach auth, and
    # parse/raise. Every method below is built on them.

    def _request(self, method: str, path: str, json=None):
        """Send an authenticated request via the active backend; return the raw response.

        Content-Type is set only when there is a JSON body. ``path`` is appended
        to ``base_url`` (which already includes ``/api/v2``).
        """
        headers = {"Authorization": self.token}
        if json is not None:
            headers["Content-Type"] = "application/json"
        url = f"{self.base_url}{path}"
        if self.library == HTTP_REQUESTS_LIBRARY:
            return self._get_session().request(method, url, headers=headers, json=json)
        return self._get_client().request(method, url, headers=headers, json=json)

    def _call(self, method: str, path: str, json=None, ok=(200,)) -> dict:
        """``_request`` + status check + JSON parse. Returns the parsed body."""
        resp = self._request(method, path, json=json)
        if resp.status_code not in ok:
            raise_exception(resp.status_code, resp.text)
        return resp.json()

    # ── Collection API (v2) ──────────────────────────────────────────────────

    def create_collection(
        self,
        name: str,
        fields: list,
    ) -> dict:
        """
        Create a new collection with typed fields.

        Each field is a plain dict sent through to the server as-is; the server
        validates it and returns a 400 with a clear message on anything invalid.
        Use the server-native keys exactly:

            {"name": <str>, "type": "vector"|"sparse"|"multi_vector",
             "params": {...}, "sparse_model": <str>}

        Field types:
            - "vector"       — single dense vector field
                params: dimension (int), space_type (str), precision (str),
                        optional M (int), ef_con (int)
            - "sparse"       — sparse vector field
                sparse_model: "default" or "endee_bm25" (required, top-level)
            - "multi_vector" — multiple dense vectors per object
                params: dimension, space_type, precision, pooling ("mean"|"max"),
                        optional M, ef_con

        Args:
            name:   Collection name (no leading '__', no '/')
            fields: List of field dicts in the shape above.

        Returns:
            dict: Collection metadata from the server

        Example:
            >>> client.create_collection("my_docs", fields=[
            ...     {"name": "dense", "type": "vector",
            ...      "params": {"dimension": 768, "space_type": "cosine",
            ...                 "precision": "int8"}},
            ...     {"name": "keywords", "type": "sparse",
            ...      "sparse_model": "default"},
            ... ])
        """
        validate_collection_name(name)

        for f in fields:
            if not isinstance(f, dict):
                raise ValueError(
                    f"Each field must be a dict, got {type(f).__name__}"
                )

        return self._call(
            "POST", "/collection",
            json={"name": name, "fields": fields}, ok=(200, 201),
        )

    def list_collections(self) -> list:
        """
        List all collections for the current user.

        Returns:
            list: List of collection metadata dicts
        """
        return self._call("GET", "/collection").get("collections", [])

    def get_collection(self, name: str) -> "Collection":
        """
        Get a Collection object for performing operations.

        Fetches collection metadata and returns a Collection instance.

        Args:
            name: Collection name

        Returns:
            Collection: Collection object for vector operations
        """
        metadata = self._call("GET", f"/collection/{name}")
        manager = (
            self.session_manager
            if self.library == HTTP_REQUESTS_LIBRARY
            else self.client_manager
        )
        return Collection(
            name=name,
            token=self.token,
            v2_url=self.base_url,
            metadata=metadata,
            session_client_manager=manager,
        )

    def delete_collection(self, name: str) -> dict:
        """
        Delete a collection and all its data.

        Args:
            name: Collection name to delete

        Returns:
            dict: {"message": "Collection deleted"}
        """
        return self._call("DELETE", f"/collection/{name}")

    # ══════════════════════════════════════════════════════════════════════════
    # Database administration (require the ROOT token).
    #
    #     admin = Endee(token=ROOT_TOKEN); admin.set_base_url(".../api/v2")
    #     db_token = admin.create_database("alice")   # db_type defaults to "enterprise"
    #
    # create_database / create_token return the new db_token STRING; list_* return
    # a list; the remaining methods return the server's JSON (dict).
    # ══════════════════════════════════════════════════════════════════════════

    def create_database(self, db_name: str, db_type: str = "enterprise") -> str:
        """Create a database (default tier ``enterprise``) and return its new db token
        (``"db_name:secret"``)."""
        if not db_name:
            raise ValueError("db_name is required")
        dt = str(db_type).lower()
        if dt not in _VALID_DB_TYPES:
            raise ValueError(f"db_type must be one of {sorted(_VALID_DB_TYPES)}")
        return self._call(
            "POST", "/admin/dbs", json={"db_name": db_name, "db_type": dt}, ok=(200, 201)
        )["db_token"]

    def list_databases(self) -> list:
        """List all databases. Returns a list of ``{db_name, db_type, is_active, created_at}``."""
        return self._call("GET", "/admin/dbs").get("dbs", [])

    def get_database(self, db_name: str) -> dict:
        """Get a single database's info (root or the db itself)."""
        if not db_name:
            raise ValueError("db_name is required")
        return self._call("GET", f"/dbs/{db_name}/info")

    def delete_database(self, db_name: str) -> dict:
        """Delete a database and all its data."""
        if not db_name:
            raise ValueError("db_name is required")
        return self._call("DELETE", f"/admin/dbs/{db_name}")

    def activate_database(self, db_name: str) -> dict:
        """Activate a previously deactivated database."""
        return self._call("POST", f"/admin/dbs/{db_name}/activate")

    def deactivate_database(self, db_name: str) -> dict:
        """Deactivate a database (blocks its tokens without deleting data)."""
        return self._call("POST", f"/admin/dbs/{db_name}/deactivate")

    def set_database_type(self, db_name: str, db_type: str) -> dict:
        """Change a database's tier (db_type)."""
        dt = str(db_type).lower()
        if dt not in _VALID_DB_TYPES:
            raise ValueError(f"db_type must be one of {sorted(_VALID_DB_TYPES)}")
        return self._call("PUT", f"/admin/dbs/{db_name}/type", json={"db_type": dt})

    # ── admin view of collections across databases ─────────────────────────────

    def list_db_collections(self, db_name: str) -> list:
        """List the collection names in a specific database."""
        if not db_name:
            raise ValueError("db_name is required")
        return self._call("GET", f"/admin/dbs/{db_name}/collection").get("collections", [])

    def list_all_collections(self) -> list:
        """List all collections across all databases, grouped by db."""
        return self._call("GET", "/admin/collection").get("collections", [])

    def delete_db_collection(self, db_name: str, collection_name: str) -> dict:
        """Delete a collection inside a specific database."""
        if not db_name or not collection_name:
            raise ValueError("db_name and collection_name are required")
        return self._call("DELETE", f"/admin/dbs/{db_name}/collection/{collection_name}")

    # ── database token management ──────────────────────────────────────────────

    def create_token(self, db_name: str, name: str, token_type: str = "rw") -> str:
        """Create a named token for a database and return the new db token string.

        token_type: ``"rw"`` (read-write, default) or ``"r"`` (read-only).
        """
        if not db_name or not name:
            raise ValueError("db_name and name are required")
        tt = str(token_type).lower()
        if tt not in _VALID_TOKEN_TYPES:
            raise ValueError(f"token_type must be one of {sorted(_VALID_TOKEN_TYPES)}")
        return self._call(
            "POST", f"/admin/dbs/{db_name}/tokens",
            json={"name": name, "token_type": tt}, ok=(200, 201),
        )["db_token"]

    def list_tokens(self, db_name: str) -> list:
        """List a database's tokens (names + types; secrets are not returned)."""
        if not db_name:
            raise ValueError("db_name is required")
        return self._call("GET", f"/admin/dbs/{db_name}/tokens").get("tokens", [])

    def delete_token(self, db_name: str, name: str) -> dict:
        """Delete a database token by name."""
        if not db_name or not name:
            raise ValueError("db_name and name are required")
        return self._call("DELETE", f"/admin/dbs/{db_name}/tokens/{name}")

    # ══════════════════════════════════════════════════════════════════════════
    # Self-service token management (operate on the CALLER's own db; use a db token)
    # ══════════════════════════════════════════════════════════════════════════

    def create_my_token(self, name: str, token_type: str = "rw") -> str:
        """Create a new token for your own database and return the new db token string."""
        if not name:
            raise ValueError("name is required")
        tt = str(token_type).lower()
        if tt not in _VALID_TOKEN_TYPES:
            raise ValueError(f"token_type must be one of {sorted(_VALID_TOKEN_TYPES)}")
        return self._call(
            "POST", "/tokens", json={"name": name, "token_type": tt}, ok=(200, 201)
        )["db_token"]

    def list_my_tokens(self) -> list:
        """List your own database's tokens."""
        return self._call("GET", "/tokens").get("tokens", [])

    def delete_my_token(self, name: str) -> dict:
        """Delete one of your own database's tokens by name."""
        if not name:
            raise ValueError("name is required")
        return self._call("DELETE", f"/tokens/{name}")

    # ══════════════════════════════════════════════════════════════════════════
    # Server info
    # ══════════════════════════════════════════════════════════════════════════

    def health(self) -> dict:
        """Server health check. Returns ``{status, timestamp}``."""
        return self._call("GET", "/health")

    def stats(self) -> dict:
        """Server stats. Returns ``{version, uptime, total_requests}``."""
        return self._call("GET", "/stats")

    # ══════════════════════════════════════════════════════════════════════════
    # Backups (per-database; use a db token, or the root token with db=<name>)
    # ══════════════════════════════════════════════════════════════════════════

    def list_backups(self) -> list:
        """List this database's backups."""
        return self._call("GET", "/backup")

    def backup_info(self, backup_name: str) -> dict:
        """Get metadata for one backup."""
        if not backup_name:
            raise ValueError("backup_name is required")
        return self._call("GET", f"/backup/{backup_name}/info")

    def active_backup(self) -> dict:
        """Get the in-progress backup status for this database. ``{active: bool, ...}``."""
        return self._call("GET", "/backup/active")

    def restore_backup(self, backup_name: str, target_collection_name: str) -> dict:
        """Restore a backup into a new collection."""
        if not backup_name or not target_collection_name:
            raise ValueError("backup_name and target_collection_name are required")
        resp = self._request(
            "POST", f"/backup/{backup_name}/restore",
            json={"target_collection_name": target_collection_name},
        )
        if resp.status_code not in (200, 201):
            raise_exception(resp.status_code, resp.text)
        try:
            return resp.json()
        except ValueError:
            return {"message": resp.text}

    def delete_backup(self, backup_name: str) -> dict:
        """Delete a backup."""
        if not backup_name:
            raise ValueError("backup_name is required")
        resp = self._request("DELETE", f"/backup/{backup_name}")
        if resp.status_code not in (200, 204):
            raise_exception(resp.status_code, resp.text)
        return {"message": (resp.text or "Backup deleted successfully")}

    def download_backup(self, backup_name: str, dest_path: str, db_name: str = None) -> str:
        """Download a backup as a .tar to ``dest_path``. Returns the path written.

        Uses query-param auth (the download endpoint takes ``?token=``). With the
        root token, pass ``db_name`` to target a specific database's backup.
        """
        if not backup_name or not dest_path:
            raise ValueError("backup_name and dest_path are required")
        params = {"token": self.token}
        if db_name:
            params["db"] = db_name
        url = f"{self.base_url}/backup/{backup_name}/download"
        if self.library == HTTP_REQUESTS_LIBRARY:
            resp = self._get_session().get(url, params=params)
        else:
            resp = self._get_client().get(url, params=params)
        if resp.status_code != 200:
            raise_exception(resp.status_code, resp.text)
        with open(dest_path, "wb") as fh:
            fh.write(resp.content)
        return dest_path

    def upload_backup(self, file_path: str) -> dict:
        """Upload a backup ``.tar`` file (multipart) into this database."""
        fname = os.path.basename(file_path)
        if not fname.endswith(".tar"):
            raise ValueError("backup file must be a .tar")
        with open(file_path, "rb") as fh:
            content = fh.read()
        url = f"{self.base_url}/backup/upload"
        headers = {"Authorization": self.token}
        files = {"backup": (fname, content, "application/x-tar")}
        if self.library == HTTP_REQUESTS_LIBRARY:
            resp = self._get_session().post(url, headers=headers, files=files)
        else:
            resp = self._get_client().post(url, headers=headers, files=files)
        if resp.status_code not in (200, 201):
            raise_exception(resp.status_code, resp.text)
        try:
            return resp.json()
        except ValueError:
            return {"message": resp.text}

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

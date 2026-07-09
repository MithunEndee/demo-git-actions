"""
Collection Module for Endee Vector Database Client (v2 API)
"""

import json
from typing import Any, Dict, List, Optional, Union

import msgpack
import numpy as np
import orjson

from .compression import json_unzip, json_zip
from .constants import (
    DEFAULT_EF_SEARCH,
    DEFAULT_FILTER_BOOST_PERCENTAGE,
    DEFAULT_PREFILTER_CARDINALITY_THRESHOLD,
    DEFAULT_TOPK,
    MAX_EF_SEARCH_ALLOWED,
    MAX_FILTER_BOOST_PERCENTAGE,
    MAX_KEY_BYTES,
    MAX_PREFILTER_CARDINALITY_THRESHOLD,
    MAX_TOP_K_ALLOWED,
    MAX_VALUE_BYTES,
    MAX_VECTORS_PER_BATCH,
    MIN_PREFILTER_CARDINALITY_THRESHOLD,
)
from .exceptions import raise_exception


# Reserved meta key. For cosine fields the client sends unit vectors (the server
# stores no norms), so we stash each vector's norm here at upsert time; get_objects
# multiplies the unit vector by it to reconstruct the original vector. Hidden from
# the meta the user gets back.
_NORMS_KEY = "internal_"


# ── result decode helpers (0.1.27-style: msgpack response, json_unzip meta) ──

def _decode_meta(m):
    """Best-effort decode of a result's meta into a Python object.

    Handles the 0.1.27 wire form (zlib+orjson bytes via json_zip), plain JSON
    bytes/str, and already-decoded dict/list (current JSON server).
    """
    if m is None or m == "" or m == b"":
        return {}
    if isinstance(m, (bytes, bytearray)):
        b = bytes(m)
        try:
            return json_unzip(b)            # zlib + orjson (matches json_zip on upsert)
        except Exception:
            try:
                return orjson.loads(b)      # plain JSON bytes
            except Exception:
                return b
    if isinstance(m, str):
        try:
            return orjson.loads(m)
        except Exception:
            return m
    return m                                 # already a dict/list


def _decode_filter(f):
    """Decode a result filter (JSON string/bytes) into a Python object."""
    if not f:
        return {}
    if isinstance(f, (bytes, bytearray)):
        f = bytes(f).decode("utf-8", "replace")
    if isinstance(f, str):
        try:
            return orjson.loads(f)
        except Exception:
            return f
    return f


# ── search response helpers ───────────────────────────────────────────────────

def _decode_object_meta(int_id: int, objects_map: dict) -> dict:
    """Decode one ObjectMeta entry from the SearchResult objects map.

    Wire layout (MSGPACK_DEFINE order):
        ObjectMeta = [str_id, meta_bytes, filter_str]
    """
    obj = objects_map.get(int_id)
    if obj is None:
        return {"id": str(int_id), "meta": {}, "filter": {}}
    meta = _decode_meta(obj[1]) if len(obj) > 1 else {}
    if isinstance(meta, dict):
        meta.pop(_NORMS_KEY, None)   # search results never expose the internal norms key
    return {
        "id":     obj[0],
        "meta":   meta,
        "filter": _decode_filter(obj[2]) if len(obj) > 2 else {},
    }


def _decode_field_hits(hits, objects_map, limit) -> list:
    """Decode a field's ranked hits ([[int_id, score], ...]) into result dicts."""
    out = []
    for hit in (hits or [])[:limit]:
        d = _decode_object_meta(hit[0], objects_map)
        d["similarity"] = float(hit[1])
        out.append(d)
    return out


def _decode_object(obj) -> dict:
    """Decode one full object from a get-objects ObjectBatch.

    Wire layout (MSGPACK_DEFINE order):
        Object = [id, meta, filter, vectors, sparses, multi_vectors]
        vectors       = { name: [float, ...] }          flat (unit) dense vector
        sparses       = { name: [indices, values] }
        multi_vectors = { name: [[float, ...], ...] }

    For cosine fields the server stores unit vectors; we multiply them by the norms
    carried in meta (``internal_``) to return the original vectors. The ``internal_``
    entry is then stripped from the returned meta (it's an internal detail).
    """
    meta = _decode_meta(obj[1]) if len(obj) > 1 else {}
    norms = meta.get(_NORMS_KEY) or {} if isinstance(meta, dict) else {}

    vectors = {k: list(v) for k, v in (obj[3] or {}).items()} if len(obj) > 3 else {}
    multi_vectors = ({k: [list(x) for x in v] for k, v in (obj[5] or {}).items()}
                     if len(obj) > 5 else {})

    # Rebuild originals: unit_vector * norm (cosine fields only; others have no norm).
    for fname, vec in vectors.items():
        n = norms.get(fname)
        if isinstance(n, (int, float)):
            vectors[fname] = [x * n for x in vec]
    for fname, members in multi_vectors.items():
        ns = norms.get(fname)
        if isinstance(ns, list):
            multi_vectors[fname] = [
                [x * ns[i] for x in m] if i < len(ns) else list(m)
                for i, m in enumerate(members)
            ]

    if isinstance(meta, dict):
        meta.pop(_NORMS_KEY, None)   # hide the internal norms key from the user

    return {
        "id":     obj[0],
        "meta":   meta,
        "filter": _decode_filter(obj[2]) if len(obj) > 2 else {},
        "vectors": vectors,
        "sparses": {k: {"indices": list(v[0]), "values": list(v[1])}
                    for k, v in (obj[4] or {}).items()} if len(obj) > 4 else {},
        "multi_vectors": multi_vectors,
    }


# ── normalization helpers (cosine-only, mirrors endee 0.1.27) ────────────────

def _normalize_dense(vec, space_type: str):
    """L2-normalize a dense vector for cosine; return (vector_list, norm).

    Non-cosine spaces are sent unchanged with norm 1.0 — identical to the
    0.1.27 client's _normalize_vector behavior.
    """
    v = np.asarray(vec, dtype=np.float32)
    if v.ndim != 1:
        raise ValueError("dense vector must be a 1-D list of floats")
    if not np.isfinite(v).all():
        raise ValueError("dense vector contains NaN or infinity")
    if space_type == "cosine":
        norm = float(np.sqrt(float(np.dot(v, v))))
        norm = max(norm, 1e-10)          # guard against zero vectors
        v = v / norm
        return v.tolist(), norm
    return v.tolist(), 1.0


def _normalize_multi(vectors, space_type: str):
    """L2-normalize each sub-vector for cosine; return (vectors_list, norms_list)."""
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("multi_vector field must be a list of equal-length vectors")
    if not np.isfinite(arr).all():
        raise ValueError("multi_vector contains NaN or infinity")
    if space_type == "cosine":
        norms = np.sqrt(np.einsum("ij,ij->i", arr, arr))
        np.maximum(norms, 1e-10, out=norms)   # guard against zero vectors
        arr = arr / norms[:, None]
        return arr.tolist(), [float(n) for n in norms]
    return arr.tolist(), [1.0] * arr.shape[0]


def _extract_sparse(data):
    """Accept {indices/values}, {sparse_indices/sparse_values}, or {position/values}."""
    if not isinstance(data, dict):
        raise ValueError("sparse field expects a dict with indices and values")
    indices = data.get("indices", data.get("sparse_indices", data.get("position", [])))
    values = data.get("values", data.get("sparse_values", []))
    indices = [int(i) for i in indices]
    values = [float(v) for v in values]
    if len(indices) != len(values):
        raise ValueError(
            f"sparse indices and values must match in length "
            f"({len(indices)} vs {len(values)})"
        )
    return indices, values


def _validate_filter(flt) -> None:
    """Filter key/value size guards (mirrors endee 0.1.27 _validate_filter).

    Keeps keys ≤ MAX_KEY_BYTES and string values ≤ MAX_VALUE_BYTES (UTF-8),
    preventing MDBX 'bad valsize' errors server-side.
    """
    if not isinstance(flt, dict):
        return
    for key, value in flt.items():
        if len(str(key).encode("utf-8")) > MAX_KEY_BYTES:
            raise ValueError(f"Filter key must be ≤ {MAX_KEY_BYTES} bytes: {key!r}")
        for v in (value if isinstance(value, list) else [value]):
            if isinstance(v, str) and len(v.encode("utf-8")) > MAX_VALUE_BYTES:
                raise ValueError(f"Filter value must be ≤ {MAX_VALUE_BYTES} bytes")


def _resolve_field_limit(field_name: str, limit) -> int:
    """Resolve a field's ``limit`` (max hits to return for this field).

    A missing limit defaults to ``DEFAULT_TOPK``; the value must be an integer in
    ``[1, MAX_TOP_K_ALLOWED]``.
    """
    if limit is None:
        return DEFAULT_TOPK
    if isinstance(limit, bool) or not isinstance(limit, int) or not (1 <= limit <= MAX_TOP_K_ALLOWED):
        raise ValueError(
            f"Search field '{field_name}': limit must be an integer between "
            f"1 and {MAX_TOP_K_ALLOWED}."
        )
    return limit


def _build_field_search_entry(field_name: str, field_data,
                              ef_search: int):
    """Build one entry for the fields array sent to the server.

    ``field_data`` must be a config dict of the form
    ``{"query": ..., "limit"?: ..., "ef_search"?: ...}`` — ``query`` is required.
    Returns ``(entry, field_limit)`` so the caller can truncate decoded hits to
    the same per-field limit.
    """
    if not (isinstance(field_data, dict) and "query" in field_data):
        raise ValueError(
            f"Search field '{field_name}' must be a dict of the form "
            f"{{'query': ..., 'limit'?: ..., 'ef_search'?: ...}}."
        )

    cfg: dict = dict(field_data)
    query = cfg.pop("query")
    limit = cfg.pop("limit", None)
    ef = cfg.pop("ef_search", None)

    field_limit = _resolve_field_limit(field_name, limit)
    entry = dict(cfg)                                    # any extra keys pass through
    entry["query"] = query
    entry["limit"] = field_limit
    entry["ef_search"] = ef_search if ef is None else ef
    return {field_name: entry}, field_limit


class Collection:
    """
    A collection in the Endee vector database (v2 API).

    Obtain via ``Endee.get_collection()`` or after ``Endee.create_collection()``.
    """

    def __init__(self, name: str, token: str, v2_url: str,
                 metadata: dict, session_client_manager):
        self.name = name
        self.token = token
        self.v2_url = v2_url
        self.fields = metadata.get("fields", [])
        self.created_at = metadata.get("created_at")
        self.layout_version = metadata.get("layout_version", 1)
        self.session_client_manager = session_client_manager

    def _http(self):
        if hasattr(self.session_client_manager, "get_session"):
            return self.session_client_manager.get_session()
        if hasattr(self.session_client_manager, "get_client"):
            return self.session_client_manager.get_client()
        raise ValueError("Invalid session manager. Obtain via Endee.get_collection().")

    def __str__(self):
        return self.name

    def _field_map(self) -> Dict[str, Dict[str, Any]]:
        """name → {type, space_type, dimension} from the collection metadata."""
        idx: Dict[str, Dict[str, Any]] = {}
        for f in self.fields:
            params = f.get("params", {}) or {}
            idx[f["name"]] = {
                "type": f.get("type", "vector"),
                "space_type": params.get("space_type", "cosine"),
                "dimension": params.get("dimension", 0),
            }
        return idx

    # ── upsert ────────────────────────────────────────────────────────────────

    def upsert(self, objects: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Insert or update objects in the collection.

        Vectors are L2-normalized client-side for ``cosine`` fields (the server
        stores no norms, so a fetched vector comes back normalized), then the
        batch is serialized with msgpack into the server ``ObjectBatch`` layout
        and POSTed as ``application/msgpack``.

        Wire layout (matching the server structs). vectors/sparses/multi_vectors
        are msgpack MAPS keyed by field name:
            ObjectBatch  = [ objects ]
            Object       = [ id, meta, filter, vectors, sparses, multi_vectors ]
            vectors      = { field_name: [float, ...] }        flat dense vector
            sparses      = { field_name: [indices, values] }   Sparse
            multi_vectors= { field_name: [[float, ...], ...] } list of vectors

        Parameters
        ----------
        objects : list of dict
            Each dict:
            - ``id`` (str) — unique identifier
            - ``meta`` (any, optional) — JSON payload (zlib-compressed on the wire)
            - ``filter`` (dict | str, optional) — filter tags e.g. ``{"category":"news"}``
            - ``fields`` (dict, optional) — field_name → field_data:
                - Dense   : ``"embedding": [0.1, 0.2, ...]``
                - Sparse  : ``"keywords": {"indices": [10, 42], "values": [0.9, 0.4]}``
                - Multi-v : ``"multivec": [[0.1, ...], [0.2, ...]]``

        Returns
        -------
        dict  ``{"upserted": <count>}``
        """
        # Batch-size guard (mirrors endee 0.1.27).
        if not isinstance(objects, list):
            raise ValueError("objects must be a list")
        if len(objects) > MAX_VECTORS_PER_BATCH:
            raise ValueError(
                f"Cannot upsert more than {MAX_VECTORS_PER_BATCH} objects at a time "
                f"(got {len(objects)})"
            )

        field_map = self._field_map()
        wire_objects = []
        seen_ids: set = set()

        for obj in objects:
            oid = str(obj["id"])
            if not oid:
                raise ValueError("object id must be a non-empty string")
            if oid in seen_ids:
                raise ValueError(f"Duplicate id in batch: {oid!r}")
            seen_ids.add(oid)

            # filter → JSON string (server stores it as std::string)
            flt = obj.get("filter")
            if flt is None:
                filter_str = ""
            elif isinstance(flt, str):
                filter_str = flt
            else:
                _validate_filter(flt)
                filter_str = orjson.dumps(flt).decode("utf-8")

            # field_name -> field struct (msgpack maps); the name is the map key,
            # so the field structs themselves are name-less. For cosine fields we
            # also collect the per-vector norm(s) to stash in meta (see below).
            vectors: Dict[str, list] = {}
            sparses: Dict[str, list] = {}
            multi_vectors: Dict[str, list] = {}
            norms: Dict[str, Any] = {}

            for fname, fdata in (obj.get("fields") or {}).items():
                cfg = field_map.get(fname)
                if cfg is None:
                    raise ValueError(
                        f"Unknown field '{fname}'. Collection fields: "
                        f"{list(field_map)}"
                    )
                ftype = cfg["type"]
                dim = cfg.get("dimension") or 0
                space = cfg.get("space_type", "cosine")
                if ftype == "vector":
                    vec, norm = _normalize_dense(fdata, space)
                    if dim and len(vec) != dim:
                        raise ValueError(
                            f"Field '{fname}': expected dimension {dim}, got {len(vec)}"
                        )
                    vectors[fname] = vec                         # flat (unit) dense vector
                    if space == "cosine":
                        norms[fname] = norm
                elif ftype == "sparse":
                    indices, values = _extract_sparse(fdata)
                    sparses[fname] = [indices, values]           # Sparse = [indices, values]
                elif ftype == "multi_vector":
                    vecs, mnorms = _normalize_multi(fdata, space)
                    if dim and any(len(v) != dim for v in vecs):
                        raise ValueError(
                            f"Field '{fname}': every multi_vector must have dimension {dim}"
                        )
                    multi_vectors[fname] = vecs                  # list of (unit) vectors
                    if space == "cosine":
                        norms[fname] = mnorms                    # one norm per member
                else:
                    raise ValueError(f"Field '{fname}' has unknown type '{ftype}'")

            # meta → zlib(orjson) bytes. Stash cosine norms under the reserved
            # `internal_` key so get_objects can rebuild the original vectors.
            raw_meta = obj.get("meta")
            if isinstance(raw_meta, (bytes, bytearray)):
                meta_bytes = bytes(raw_meta)                     # pre-encoded: cannot inject norms
            elif isinstance(raw_meta, dict) or raw_meta is None:
                meta_dict = dict(raw_meta) if isinstance(raw_meta, dict) else {}
                if norms:
                    meta_dict[_NORMS_KEY] = norms
                meta_bytes = json_zip(meta_dict)
            else:
                meta_bytes = json_zip(raw_meta)                  # non-dict meta: send as-is

            # Object = [id, meta, filter, vectors, sparses, multi_vectors]
            wire_objects.append(
                [oid, meta_bytes, filter_str, vectors, sparses, multi_vectors]
            )

        # ObjectBatch = [objects]  (MSGPACK_DEFINE(objects) → single-element array)
        payload = msgpack.packb(
            [wire_objects], use_bin_type=True, use_single_float=True
        )

        headers = {"Authorization": self.token, "Content-Type": "application/msgpack"}
        resp = self._http().post(
            f"{self.v2_url}/collection/{self.name}/objects",
            headers=headers,
            data=payload,
        )
        if resp.status_code != 200:
            raise_exception(resp.status_code, resp.text)
        try:
            return resp.json()
        except ValueError:
            return {"status": resp.text}

    # ── search ────────────────────────────────────────────────────────────────

    def search(
        self,
        fields: Dict[str, Any],
        filter: Optional[List[Dict[str, Any]]] = None,
        ef_search: int = DEFAULT_EF_SEARCH,
        prefilter_cardinality_threshold: Optional[int] = None,
        filter_boost_percentage: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Search the collection across one or more fields in a single request.

        Each field is queried with the unified config
        ``{"query": ..., "limit"?: ..., "ef_search"?: ...}``, where ``limit`` is
        the max number of hits to return **for that field** (defaults to
        ``DEFAULT_TOPK``). The result is always one ranked list per field:
        ``{"results": {field_name: [hit, ...], ...}}``. Use :func:`endee.rerank`
        to fuse these into a single ranked list.

        Each ``hit`` is ``{"id", "similarity", "meta", "filter"}``.

        Parameters
        ----------
        fields : dict
            field_name → per-field config dict ``{"query": ..., "limit"?: ...,
            "ef_search"?: ...}``. ``query`` is required and takes one of:

            * **Dense**   — ``[0.1, 0.2, ...]``
            * **Sparse**  — ``{"indices": [10, 42], "values": [0.9, 0.4]}``
            * **Multi-v** — ``[[0.1, ...], [0.2, ...]]``

            To search a single field, pass just that one field.
        filter : list of dict, optional
            Filter expressions e.g. ``[{"category": {"$eq": "news"}}]``.
        ef_search : int
            Default HNSW ef_search parameter (default 128, max 1024); a field's
            own ``ef_search`` overrides it.
        prefilter_cardinality_threshold : int, optional
            Filtered search only. Cardinality below which the search switches from
            HNSW filtered search to brute-force prefiltering. Range
            1,000–1,000,000. When set (with/without ``filter_boost_percentage``)
            it is sent to the server as ``filter_params``.
        filter_boost_percentage : float, optional
            Filtered search only. Percentage by which to expand the HNSW candidate
            pool before filtering, to compensate for filtered-out candidates.
            Range 0–100 (0 = no boost, 100 = double the pool).

        Returns
        -------
        dict  ``{"results": {field_name: [hit, ...], ...}}``
        """
        if not fields:
            raise ValueError("search requires at least one field")
        if not (0 < ef_search <= MAX_EF_SEARCH_ALLOWED):
            raise ValueError(f"ef_search must be between 1 and {MAX_EF_SEARCH_ALLOWED}")

        # Optional filter-tuning params (filtered search only). Validate ranges up
        # front; they're only sent to the server when explicitly provided.
        if prefilter_cardinality_threshold is not None:
            t = prefilter_cardinality_threshold
            if not (MIN_PREFILTER_CARDINALITY_THRESHOLD <= t
                    <= MAX_PREFILTER_CARDINALITY_THRESHOLD):
                raise ValueError(
                    f"prefilter_cardinality_threshold must be between "
                    f"{MIN_PREFILTER_CARDINALITY_THRESHOLD} and "
                    f"{MAX_PREFILTER_CARDINALITY_THRESHOLD}"
                )
        if filter_boost_percentage is not None:
            b = filter_boost_percentage
            if not (0 <= b <= MAX_FILTER_BOOST_PERCENTAGE):
                raise ValueError(
                    f"filter_boost_percentage must be between 0 and "
                    f"{MAX_FILTER_BOOST_PERCENTAGE}"
                )

        # L2-normalize query vectors for cosine fields (mirrors upsert).
        field_map = self._field_map()
        fields_array = []
        field_limits: Dict[str, int] = {}
        for fname, fdata in fields.items():
            if not (isinstance(fdata, dict) and "query" in fdata):
                raise ValueError(
                    f"Search field '{fname}' must be a dict of the form "
                    f"{{'query': ..., 'limit'?: ..., 'ef_search'?: ...}}."
                )
            cfg = dict(fdata)
            query = cfg["query"]
            fld = field_map.get(fname)
            if fld:
                ftype = fld["type"]
                space_type = fld.get("space_type", "cosine")
                if ftype == "vector" and space_type == "cosine":
                    if (isinstance(query, (list, tuple)) and query
                            and not isinstance(query[0], (list, tuple))):
                        query, _ = _normalize_dense(query, space_type)
                elif ftype == "multi_vector" and space_type == "cosine":
                    if (isinstance(query, (list, tuple)) and query
                            and isinstance(query[0], (list, tuple))):
                        query, _ = _normalize_multi(query, space_type)
            cfg["query"] = query
            entry, field_limit = _build_field_search_entry(fname, cfg, ef_search)
            fields_array.append(entry)
            field_limits[fname] = field_limit

        payload: dict = {"fields": fields_array}
        if filter is not None:
            payload["filter"] = filter

        # Filter tuning: send filter_params only when the caller tunes at least
        # one knob, defaulting the other so the server gets a complete object.
        if (prefilter_cardinality_threshold is not None
                or filter_boost_percentage is not None):
            payload["filter_params"] = {
                "prefilter_threshold": (
                    prefilter_cardinality_threshold
                    if prefilter_cardinality_threshold is not None
                    else DEFAULT_PREFILTER_CARDINALITY_THRESHOLD
                ),
                "boost_percentage": (
                    filter_boost_percentage
                    if filter_boost_percentage is not None
                    else DEFAULT_FILTER_BOOST_PERCENTAGE
                ),
            }

        headers = {"Authorization": self.token, "Content-Type": "application/json"}
        resp = self._http().post(
            f"{self.v2_url}/collection/{self.name}/search",
            headers=headers,
            json=payload,
        )
        if resp.status_code != 200:
            raise_exception(resp.status_code, resp.text)

        # Parse msgpack SearchResult: MSGPACK_DEFINE(objects, results)
        #   raw[0] = objects_map  {int32_id → ObjectMeta array}
        #   raw[1] = results_map  {field_name → [SearchHit array, ...]}
        # strict_map_key=False: the objects map is keyed by integer (internal) ids.
        raw = msgpack.unpackb(resp.content, raw=False, strict_map_key=False)
        objects_map = raw[0]   # {int_id: [str_id, meta_bytes, filter_str, ...]}
        results_map = raw[1]   # {field_name: [[int_id, score], ...]}

        # Always per-field results, unfused. Each field is truncated to its own
        # limit (already applied server-side via the per-field limit).
        per_field = {
            fname: _decode_field_hits(
                results_map.get(fname, []), objects_map, field_limits[fname])
            for fname in fields
        }
        return {"results": per_field}

    # ── delete object ─────────────────────────────────────────────────────────

    def delete_object(self, id: str) -> Dict[str, Any]:
        """Delete a single object by ID. Returns ``{"deleted": "<id>"}``."""
        resp = self._http().delete(
            f"{self.v2_url}/collection/{self.name}/objects/{id}",
            headers={"Authorization": self.token},
        )
        if resp.status_code != 200:
            raise_exception(resp.status_code, resp.text)
        return resp.json()

    # ── describe ──────────────────────────────────────────────────────────────

    def describe(self) -> Dict[str, Any]:
        """Return collection metadata. ``{"name", "fields", "created_at", "layout_version"}``"""
        resp = self._http().get(
            f"{self.v2_url}/collection/{self.name}",
            headers={"Authorization": self.token},
        )
        if resp.status_code != 200:
            raise_exception(resp.status_code, resp.text)
        return resp.json()

    # ── get objects by id ──────────────────────────────────────────────────────

    def get_objects(self, ids: List[str]) -> List[Dict[str, Any]]:
        """Fetch full objects (meta, filter, vectors) by id.

        ``POST .../objects/query {"ids":[...]}`` → msgpack ``ObjectBatch``. Returns a
        list of ``{id, meta, filter, vectors, sparses, multi_vectors}``. Ids that do
        not exist are skipped by the server. (0.1.27: ``get_vector``.)
        """
        if not isinstance(ids, (list, tuple)) or not ids:
            raise ValueError("get_objects requires a non-empty list of ids")
        resp = self._http().post(
            f"{self.v2_url}/collection/{self.name}/objects/query",
            headers={"Authorization": self.token, "Content-Type": "application/json"},
            json={"ids": [str(i) for i in ids]},
        )
        if resp.status_code != 200:
            raise_exception(resp.status_code, resp.text)
        batch = msgpack.unpackb(resp.content, raw=False)        # ObjectBatch = [objects]
        if isinstance(batch, list):
            objects = batch[0] if batch else []
        else:
            objects = batch.get("objects", []) if isinstance(batch, dict) else []
        return [_decode_object(o) for o in (objects or [])]

    # ── delete by filter ───────────────────────────────────────────────────────

    def delete_by_filter(self, filter: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Delete all objects matching a filter. Returns ``{"deleted": <count>}``.

        ``filter`` uses the array format, e.g. ``[{"category": {"$eq": "news"}}]``.
        (0.1.27: ``delete_with_filter``.)
        """
        if not isinstance(filter, list):
            raise ValueError("filter must be an array, e.g. [{'field': {'$op': value}}]")
        resp = self._http().request(
            "DELETE",
            f"{self.v2_url}/collection/{self.name}/objects",
            headers={"Authorization": self.token, "Content-Type": "application/json"},
            json={"filter": filter},
        )
        if resp.status_code != 200:
            raise_exception(resp.status_code, resp.text)
        return resp.json()

    # ── update filters ─────────────────────────────────────────────────────────

    def update_filters(self, updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Update filter tags on existing objects. Returns ``{"updated": <count>}``.

        ``updates`` is a list of ``{"id": <str>, "filter": <dict>}``.
        (0.1.27: ``update_filters``.)
        """
        if not isinstance(updates, list) or not updates:
            raise ValueError("updates must be a non-empty list of {'id', 'filter'} dicts")
        resp = self._http().post(
            f"{self.v2_url}/collection/{self.name}/filters",
            headers={"Authorization": self.token, "Content-Type": "application/json"},
            json={"updates": updates},
        )
        if resp.status_code != 200:
            raise_exception(resp.status_code, resp.text)
        return resp.json()

    # ── rebuild ────────────────────────────────────────────────────────────────

    def rebuild(self, fields: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Rebuild one or more dense vector fields' HNSW graphs (async).

        Pass a list of field specs in the server-native shape; each dict is sent
        through as-is::

            [{"field": "embedding", "M": 20, "ef_con": 200},
             {"field": "colbert",   "M": 12, "ef_con": 120}]

        Only ``M``/``ef_con`` may change; dimension/space_type/precision are
        immutable. Any sparse field in the list is skipped server-side (only
        dense ``vector``/``multi_vector`` fields are rebuilt). Returns the async
        status body, e.g. ``{"fields_total": 2, "total_objects": 2,
        "status": "in_progress"}``. Poll progress with :meth:`rebuild_status`.
        """
        if not isinstance(fields, list) or not fields:
            raise ValueError("rebuild requires a non-empty list of field specs")
        for f in fields:
            if not isinstance(f, dict):
                raise ValueError(
                    f"Each field spec must be a dict, got {type(f).__name__}"
                )
            if not f.get("field"):
                raise ValueError("Each field spec must include a 'field' name")
        resp = self._http().post(
            f"{self.v2_url}/collection/{self.name}/rebuild",
            headers={"Authorization": self.token, "Content-Type": "application/json"},
            json={"fields": fields},
        )
        if resp.status_code not in (200, 202):
            raise_exception(resp.status_code, resp.text)
        return resp.json()

    def rebuild_status(self) -> Dict[str, Any]:
        """Poll rebuild progress. Returns the raw server status object.

        For a multi-field rebuild the body reports overall progress plus the
        latest field touched, e.g.::

            {"field": "colbert", "fields_done": 2, "fields_total": 2,
             "objects_processed": 2, "total_objects": 2,
             "percent_complete": 100.0, "status": "completed",
             "new_config": {"M": 12, "ef_con": 120},
             "previous_config": {"M": 12, "ef_con": 120},
             "started_at": ..., "completed_at": ...}

        ``field`` is the most recently rebuilt field; ``status`` is
        ``in_progress``/``completed``/``failed`` (``idle`` when none is running).
        """
        resp = self._http().get(
            f"{self.v2_url}/collection/{self.name}/rebuild/status",
            headers={"Authorization": self.token},
        )
        if resp.status_code != 200:
            raise_exception(resp.status_code, resp.text)
        return resp.json()

    # ── maintenance ────────────────────────────────────────────────────────────

    def shrink(self) -> Dict[str, Any]:
        """Defragment the collection's storage in place.

        Returns ``{"status": "ok", "reclaimed_bytes": <int>}``.
        """
        resp = self._http().post(
            f"{self.v2_url}/collection/{self.name}/shrink",
            headers={"Authorization": self.token},
        )
        if resp.status_code != 200:
            raise_exception(resp.status_code, resp.text)
        return resp.json()

    def create_backup(self, name: str) -> Dict[str, Any]:
        """Start an async backup of this collection.

        Returns ``{"backup_name": <str>, "status": "in_progress"}``. Poll
        :meth:`Endee.active_backup` / :meth:`Endee.list_backups` for completion.
        """
        if not name:
            raise ValueError("backup name is required")
        resp = self._http().post(
            f"{self.v2_url}/collection/{self.name}/backup",
            headers={"Authorization": self.token, "Content-Type": "application/json"},
            json={"name": name},
        )
        if resp.status_code not in (200, 201, 202):
            raise_exception(resp.status_code, resp.text)
        return resp.json()

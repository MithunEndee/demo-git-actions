# Endee Python Client - Functional Test Suite

This directory contains the end-to-end functional tests for the [Endee](https://endee.io) Python client.
The suite validates the full lifecycle of the Endee v2 Collections API — collection management,
object operations, searching, filtering, sparse/hybrid/multi-vector collections, bulk operations,
maintenance, server info, token management, every HTTP backend, and admin database management.

> **Note:** An active Endee API token is required to run tests against a live server.

---

## Running the tests

### Option 1 — Shell script (macOS / Linux) *recommended*

```bash
# Show all available options
./tests/run_tests.sh --help

# Run the full test suite
./tests/run_tests.sh --token <your_token>

# Override the API base URL (local server, staging, etc.)
./tests/run_tests.sh --token <your_token> --base-url http://localhost:8080/api/v2

# Include admin tests (requires a root token)
./tests/run_tests.sh --token <your_token> --root-token <root_token> --base-url http://localhost:8080/api/v2
```

### Option 2 — Manual (all platforms)

**Prerequisites:** Python 3.9 or higher.

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv && source .venv/bin/activate   # macOS/Linux
# python -m venv .venv && .venv\Scripts\activate     # Windows

# 2. Install the package and test dependencies
pip install -e .
pip install pytest numpy

# 3. Set environment variables
export ENDEE_TOKEN=<your_token>
export NDD_ROOT_TOKEN=<root_token>                 # optional; enables admin tests
export ENDEE_BASE_URL=http://localhost:8080/api/v2   # optional

# 4. Run from the repo root
pytest tests/                      # full suite
pytest tests/ -v                   # verbose output
pytest tests/test_searching.py     # single file
pytest tests/ -k test_filter_eq    # keyword filter
```

---

## Test files

| File | Coverage |
|------|----------|
| `test_collection_management.py` | `create_collection` (all precisions, space-types, HNSW params, dict/to_dict flows, int8e), `list_collections`, `get_collection`, `describe`, `delete_collection`, hybrid creation |
| `test_object_operations.py` | `upsert` (single, batch, overwrite, meta/filter, NaN/inf guards, duplicate-ID batch rejection, batch-size limit, unknown field) and `delete_object` |
| `test_get_objects.py` | `get_objects` — return shape, meta/filter/vector round-trips, non-existent IDs, mixed IDs, sparse and multi_vector collections |
| `test_searching.py` | `search` — result structure, `limit`, `ef_search`, `include_vectors`, meta round-trip, parameter bounds, client-side validation |
| `test_filtering.py` | `$eq`, `$in`, `$range`, `$gt`, `$gte`, `$lt`, `$lte` operators; multi-condition AND; filter correctness; sorted results |
| `test_delete_by_filter.py` | `delete_by_filter` — return shape, count accuracy, `$eq`/`$in`/`$range`, AND conditions, no-match, corpus integrity |
| `test_update_filters.py` | `update_filters` — return shape, count, values reflected in search, new keys, batch updates, numeric values, idempotency |
| `test_sparse.py` | Sparse-only collections — full upsert/search/delete/describe lifecycle, meta round-trip, filters |
| `test_hybrid_search.py` | Dense-only, sparse-only, and RRF hybrid search; per-field limits; filters; `include_vectors`; `delete_object`; field config variants |
| `test_multi_vector.py` | ColBERT-style `multi_vector` fields — pooling aliases, all precision/space-type combos, upsert, search, `get_objects` (meta/filter/multi_vectors round-trip, mixed IDs), `delete_by_filter` ($eq/$in/AND), `update_filters` (reflected in search/get_objects, idempotent), `shrink`, `rebuild` (wait for completion), `create_backup` (wait + verify in list), mixed dense + multi_vector RRF |
| `test_multi_field_search.py` | Multi-field search without reranker (per-field dict format) and with `reranker='rrf'`; `field_weights`; `rrf_k`; per-field query dict; filters; three-field (dense + sparse + multi_vector) |
| `test_rerank.py` | Standalone `rerank()` — return shape, limit, similarity ordering, `field_weights`, `rrf_k`, error cases, deduplication across fields |
| `test_filter_params.py` | `filter_params` on search — `prefilter_cardinality_threshold` and `filter_boost_percentage` accepted and validated |
| `test_rebuild.py` | `rebuild` — initial response shape, `new_config`/`previous_config`/`total_objects`, custom M/ef_con, all field combos, collection remains searchable during/after; `rebuild_status` — all status values, `objects_processed`, `percent_complete`, timestamps; multi_vector field rebuild |
| `test_shrink.py` | `shrink` — response shape, no-error on empty and populated collections, after delete; dense and multi_vector collections |
| `test_backup.py` | `create_backup` (response shape, `status=in_progress`, empty-name guard, multi_vector), `active_backup` (bool field, True-while-running, False-after-done), `list_backups`, `backup_info`, `restore_backup` (creates collection, is searchable, accepts upsert), `delete_backup` (removes from list, double-delete raises) — full lifecycle; all async waits use `wait_for_backup()` |
| `test_server_info.py` | `health` and `stats` — return shapes, required keys, repeated calls |
| `test_token_management.py` | `create_my_token`, `list_my_tokens`, `delete_my_token` — full lifecycle, rw/r types, duplicate conflict, client-side validation |
| `test_http_libraries.py` | requests (default), httpx1.1, and httpx2 backends — full CRUD cycle, `get_objects`, session/client manager lifecycle, `set_base_url` |
| `test_error_handling.py` | Client-side validation (schema, name rules, NaN/inf, duplicate IDs in batch, filter key/value size limits, search parameters, admin methods), server-side errors (401/404/409), `raise_exception` unit tests, pydantic model rejection |
| `test_admin.py` | Admin methods (root token required, skipped otherwise) — database lifecycle (create/get/list/delete), activate/deactivate, tier changes, list_db_collections, list_all_collections, delete_db_collection, admin token management (create_token/list_tokens/delete_token), client-side validation |

---

## Support files

| File | Purpose |
|------|---------|
| `conftest.py` | Session-scoped `client` fixture; function-scoped empty/populated fixtures for all collection types; `verify_server_and_cleanup` autouse fixture (fails fast on missing token or unreachable server, removes stale test collections) |
| `helpers.py` | Shared constants (`DIM`, `N_VECTORS`, `ALL_PRECISIONS` including `int8e`, etc.), vector generators (`dense_vec`, `binary_vec`, `sparse_vec`, `multi_vec`), field config builders (return dicts), object builders, and test utilities |
| `run_tests.sh` | Shell script — creates a virtualenv, installs dependencies, and runs the suite |
| `pytest.ini` *(repo root)* | Configures pytest to discover tests in `tests/` |

---

## Field builder note

The helper functions `make_dense_field()`, `make_sparse_field()`, and `make_mv_field()` now return
**plain dicts** (via `CollectionFieldConfig.to_dict()`). Pass them directly to
`create_collection(fields=[...])`. If you construct a `CollectionFieldConfig` pydantic model
manually, call `.to_dict()` before passing it to `create_collection()`.

```python
# Correct — using helpers (returns dict)
client.create_collection("my_col", fields=[make_dense_field(dim=128)])

# Correct — using to_dict() explicitly
cfg = CollectionFieldConfig(name="dense", type="vector",
                            params=CollectionFieldParams(dimension=128,
                                                        space_type="cosine",
                                                        precision="int8"))
client.create_collection("my_col", fields=[cfg.to_dict()])

# Incorrect — raises ValueError
client.create_collection("my_col", fields=[cfg])  # ← ValueError
```

---

## Known server-side gaps

Tests marked `@pytest.mark.xfail` document expected behaviour the server does not yet enforce.
They will not fail the suite and will automatically start passing once the server adds validation.

| Test | File | Notes |
|------|------|-------|
| `test_upsert_empty_id_raises` | `test_error_handling.py` | Server accepts empty string IDs |
| `test_delete_object_twice_raises_not_found` | `test_error_handling.py` | Server returns 500 instead of 404 on double-delete |
| `test_create_collection_duplicate_field_names_raises` | `test_error_handling.py` | Server may not reject duplicate field names |
| `test_search_with_empty_vector_raises` | `test_error_handling.py` | Server may not validate empty search vector |
| `test_upsert_multi_vector_empty_vectors_raises` | `test_error_handling.py` | Server may not reject empty vector list for multi_vector |
| `test_upsert_multi_vector_inconsistent_dimensions_raises` | `test_error_handling.py` | Server may not validate consistent multi_vector dimensions |
| `test_search_rrf_single_field_does_not_error` | `test_searching.py` | Server behaviour with single-field RRF is unspecified |
| `test_filter_in_empty_list_returns_empty` | `test_filtering.py` | Server behaviour for `$in: []` is unspecified |
| `test_multi_vector_search_include_vectors_true` | `test_multi_vector.py` | Server does not return vector data for multi_vector with `include_vectors=True` |

---

## What is **not** tested

The following are out of scope regardless of token type:

- `download_backup`, `upload_backup` — require filesystem access and a real `.tar` backup file

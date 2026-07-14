# Endee Python Client - Functional Test Suite

End-to-end functional tests for the [Endee](https://endee.io) Python client (v2 Collections API).
Tests run against a live server and cover the full lifecycle: collection management, object
operations, vector search (dense, sparse, hybrid, multi-vector), filtering, bulk operations,
backup/restore, maintenance, token management, all HTTP backends, and admin database management.

---

## Prerequisites

- Python 3.9 or higher
- A running Endee server (cloud or local)
- A valid `ENDEE_TOKEN` (database-level API token)
- `NDD_ROOT_TOKEN` (root token - optional, only needed for admin tests)

---

## Running the tests

### Shell script (macOS / Linux) - recommended

The script handles venv creation, dependency installation, and environment setup automatically.

```bash
# Show all options and examples
./tests/run_tests.sh --help

# Full suite against the cloud endpoint
./tests/run_tests.sh --token <api_token>

# Full suite against a local server
./tests/run_tests.sh --token <api_token> --base-url http://localhost:8080/api/v2

# Include admin tests (requires a root token)
./tests/run_tests.sh --token <api_token> --root-token <root_token>

# Full local run with admin tests
./tests/run_tests.sh --token <api_token> --root-token <root_token> \
  --base-url http://localhost:8080/api/v2

# Single file or keyword filter
./tests/run_tests.sh --token <api_token> -- tests/test_backup.py
./tests/run_tests.sh --token <api_token> -- -k test_filter_eq
```

### Manual setup (all platforms)

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv && source .venv/bin/activate   # macOS / Linux
# python -m venv .venv && .venv\Scripts\activate     # Windows

# 2. Install the package and test dependencies
pip install -e .
pip install pytest pytest-timeout numpy

# 3. Set environment variables (see table below)
export ENDEE_TOKEN=<api_token>
export ENDEE_BASE_URL=http://localhost:8080/api/v2   # optional; defaults to cloud
export NDD_ROOT_TOKEN=<root_token>                   # optional; enables admin tests

# 4. Run from the repo root
pytest tests/                       # full suite
pytest tests/ -v                    # verbose output
pytest tests/test_backup.py         # single file
pytest tests/ -k test_filter_eq     # keyword filter
pytest tests/ -x                    # stop on first failure
```

---

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ENDEE_TOKEN` | Yes | Database-level API token. Used by all non-admin tests. |
| `ENDEE_BASE_URL` | No | Server base URL (e.g. `http://localhost:8080/api/v2`). Defaults to the cloud endpoint derived from the token. |
| `NDD_ROOT_TOKEN` | No | Root/admin token. If omitted, `test_admin.py` is skipped automatically. |

---

## Test files

Tests are grouped by feature area. Each file is self-contained with its own fixtures and cleanup.

### Core collection and object operations

| File | What is tested |
|------|----------------|
| `test_collection_management.py` | `create_collection` (all precisions, space types, HNSW params, dict and to_dict flows), `list_collections`, `get_collection`, `describe`, `delete_collection`, hybrid creation |
| `test_object_operations.py` | `upsert` (single, batch, overwrite, meta/filter, NaN/inf rejection, duplicate-ID detection, batch size limit, unknown field) and `delete_object` |
| `test_get_objects.py` | `get_objects` - return shape, meta/filter/vector round-trips, non-existent IDs, mixed IDs, sparse and multi_vector collections |

### Search and filtering

| File | What is tested |
|------|----------------|
| `test_searching.py` | `search` - result structure, ordering, `limit`, `ef_search`, meta round-trips, parameter bounds, client-side validation |
| `test_filtering.py` | Filter operators: `$eq`, `$in`, `$range`, `$gt`, `$gte`, `$lt`, `$lte`; multi-condition AND; per-result value correctness; sorted results. Post-filter ANN non-determinism is marked `xfail(strict=False)`. |
| `test_filter_params.py` | `prefilter_cardinality_threshold` and `filter_boost_percentage` - accepted value ranges and client-side validation |
| `test_delete_by_filter.py` | `delete_by_filter` - return shape, count accuracy, all filter operators, AND conditions, no-match case, corpus integrity |
| `test_update_filters.py` | `update_filters` - return shape, count, updated values reflected in search, new key addition, batch updates, numeric values, idempotency |

### Field types

| File | What is tested |
|------|----------------|
| `test_sparse.py` | Sparse-only collections - collection creation (default and BM25 models), upsert, search (result structure, limit, ordering, `ef_search`), meta round-trips, `delete_object`, `describe` |
| `test_hybrid_search.py` | Dense+sparse (hybrid) collections - dense-only, sparse-only, and RRF hybrid search; per-field limits; filters; `delete_object`; field config variants |
| `test_multi_vector.py` | ColBERT-style `multi_vector` fields - all pooling methods and precision/space-type combos, upsert, search, `get_objects`, `delete_by_filter`, `update_filters`, `shrink`, `rebuild`, `create_backup`, mixed dense+multi_vector RRF |
| `test_multi_field_search.py` | Multi-field search - per-field dict result format, RRF fusion, `field_weights`, `rrf_k`, per-field limits, `ef_search`, filters, three-field collections |
| `test_rerank.py` | Standalone `rerank()` - return shape, limit, similarity ordering, `field_weights`, `rrf_k`, deduplication across fields, error cases |

### Maintenance and backup

| File | What is tested |
|------|----------------|
| `test_rebuild.py` | `rebuild` - response shape, `new_config`/`previous_config`/`total_objects`, custom M/ef_con, all field types, collection stays searchable during rebuild; `rebuild_status` - status values, `objects_processed`, `percent_complete`, timestamps |
| `test_shrink.py` | `shrink` - response shape, no error on empty and populated collections, after deletions; dense and multi_vector collections |
| `test_backup.py` | `create_backup`, `active_backup`, `list_backups`, `backup_info`, `restore_backup`, `delete_backup`, `download_backup`, `upload_backup` - full lifecycle including async wait pattern, sparse/multi_vector/multi-field collection backups, download+upload roundtrip |

### Infrastructure and admin

| File | What is tested |
|------|----------------|
| `test_server_info.py` | `health` and `stats` - return shapes, required keys, repeated calls |
| `test_token_management.py` | `create_my_token`, `list_my_tokens`, `delete_my_token` - full lifecycle, rw/r types, duplicate conflict, client-side validation |
| `test_http_libraries.py` | requests (default), httpx HTTP/1.1, and httpx HTTP/2 backends - full CRUD cycle, `get_objects`, session/client manager lifecycle, `set_base_url` |
| `test_error_handling.py` | Client-side validation (schema rules, name constraints, NaN/inf, batch duplicates, filter key/value size limits, search bounds, admin methods), server-side error mapping (401/404/409/500), `raise_exception` unit tests |
| `test_admin.py` | Admin methods - database lifecycle (create/get/list/delete), activate/deactivate, tier changes, `list_db_collections`, `list_all_collections`, `delete_db_collection`, admin token management. **Skipped automatically if `NDD_ROOT_TOKEN` is not set.** |

---

## Support files

| File | Purpose |
|------|---------|
| `conftest.py` | Session-scoped `client` fixture; function-scoped `empty_collection`, `populated_collection`, and equivalents for sparse, hybrid, and multi_vector types. The `verify_server_and_cleanup` autouse fixture fails fast on a missing token or unreachable server and removes stale test collections from previous interrupted runs. |
| `helpers.py` | Shared constants (`DIM`, `N_VECTORS`, `ALL_PRECISIONS`, field names), vector generators (`dense_vec`, `binary_vec`, `sparse_vec`, `multi_vec`), field config builders (`make_dense_field`, `make_sparse_field`, `make_mv_field`), object builders (`make_item`, `make_sparse_item`, `make_mv_item`), and test utilities (`uid`, `safe_delete`, `get_collection_names`, `parse_filter_field`, `q`, `q_sparse`, `results`). |
| `run_tests.sh` | Shell wrapper - creates a virtualenv, installs dependencies, sets environment variables, and invokes pytest. Run with `--help` for full usage and examples. |

---

## Design notes

**Unique names:** Every test resource is created with a unique name via `uid("prefix")` (a short UUID suffix). This prevents collisions between concurrent or interrupted runs.

**Async operations:** `create_backup` and `rebuild` return immediately with `status=in_progress`. Tests that need completion poll via `wait_for_backup()` or `wait_for_rebuild()` until the operation appears in the relevant list endpoint or status field.

**Cleanup:** All tests use `try/finally` blocks (or fixture teardown) to delete collections and backups after each test, even if the test fails. Backup tests also clean up local files with `shutil.rmtree`.

**xfail (post-filter ANN):** Tests in `test_filtering.py` that assert a minimum result count or a specific ID from a filtered ANN search are marked `@_XFAIL_ANN` with `strict=False`. HNSW traversal checks the filter bitmap per-node; on a 50-object corpus the bounded ef window may exit before visiting all matching nodes. These tests pass most of the time and are expected to occasionally fail - `strict=False` means a pass is not treated as an error.

# Endee-LlamaIndex Test Suite

Two kinds of tests, selected via pytest markers. Unit tests mock the `endee`
client and need nothing beyond the package itself. Integration tests run
against a live or local Endee server. They cover the `EndeeVectorStore`
wrapper, plus the LlamaIndex framework integration itself, end to end.

---

## Prerequisites

- Python 3.9 or higher
- For **unit** tests: nothing else. `pip install -e .` plus test deps is enough.
- For **integration** tests: a running Endee server (cloud or local) and a
  valid `ENDEE_API_TOKEN`.

---

## Running the tests

### Shell script (macOS / Linux, recommended)

The script handles venv creation, dependency installation, and environment
setup automatically.

```bash
# Show all options and examples
./tests/run_tests.sh --help

# Unit tests only (fast, no server needed): the common local/CI case
./tests/run_tests.sh --unit

# Full suite (unit + integration) against a live server
./tests/run_tests.sh --token <api_token>

# Full suite against a local server
./tests/run_tests.sh --token <api_token> --base-url http://localhost:8080/api/v2

# Integration tests only
./tests/run_tests.sh --token <api_token> --integration

# Single file or keyword filter
./tests/run_tests.sh --unit -- tests/test_unit.py
./tests/run_tests.sh --unit -- -k test_add
```

### Manual setup (all platforms)

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv && source .venv/bin/activate   # macOS / Linux
# python -m venv .venv && .venv\Scripts\activate      # Windows

# 2. Install the package and test dependencies
pip install -e .
pip install pytest pytest-mock pytest-timeout numpy

# 3. Unit tests: no further setup needed
pytest -m unit

# 4. Integration tests: set environment variables first (see table below)
export ENDEE_API_TOKEN=<api_token>
export ENDEE_BASE_URL=http://localhost:8080/api/v2   # optional; defaults to cloud
pytest -m integration
```

---

## Environment variables

| Variable | Required | Description |
|----------|----------|--------------|
| `ENDEE_API_TOKEN` | Only for integration tests | Endee API token. Integration tests skip automatically if unset or invalid. |
| `ENDEE_BASE_URL` | No | Server base URL (e.g. `http://localhost:8080/api/v2`). Defaults to the cloud endpoint derived from the token. |

---

## Test files

Split by kind rather than by concern: `test_unit.py` holds every mocked test class, `test_integration.py` holds every live test class plus `TestRetrieval`, selected via the `unit`/`integration` pytest markers.

### `test_unit.py`

| Class | What is tested |
|-------|-----------------|
| `TestVectorStoreUnit` | init/create-or-reuse (incl. `force_recreate`, `endee_client=`/`endee_collection=` overrides), `add()` node dedup by `node_id`, `delete`/`delete_vector`/`clear`, `describe`/`fetch` error-fallback behavior, query-result round-tripping, `constants.py`'s `Precision` fallback and filter-operator maps, empty-query handling, network-failure propagation. |
| `TestFiltersUnit` | `_extract_filter_fields` allowlist (only `file_name`/`doc_id`/`category`/`difficulty`/`language`/`field`/`type`/`feature`/`ref_doc_id` promoted to `filter`), `_process_filters` (`EQ`/`IN` supported, `NE` raises `ValueError`). |
| `TestSparseUnit` | `SparseVector`, `SparseModelAdapter`, `wrap_sparse_model`, `EndeeModelSparse`, hybrid auto-detection wiring. |

### `test_integration.py`

| Class | What is tested |
|-------|-----------------|
| `TestVectorStoreIntegration` | Live collection-config matrix (precision × HNSW × space type, verified via describe()), batch insert, `from_params`, client construction, collection lifecycle (force_recreate, endee_collection=), clear/delete_vector, and the empty-query contract against a real server. |
| `TestFiltersIntegration` | Live `$eq`/`$in`/invalid-filter-key assertions against a real server. |
| `TestSparseIntegration` | Live sparse/hybrid collection coverage, including both endee_bm25 and a custom sparse embedding, against a real server. |
| `TestRetrieval` | Real `VectorStoreIndex`, `VectorIndexRetriever`, `RetrieverQueryEngine` (with `Settings.llm = None`), and `query.query_str` back-compat, against a real server. |

---

## Support files

| File | Purpose |
|------|---------|
| `conftest.py` | `MockEndee`/`MockEndeeCollection`: a mock backend (unit tests). `mock_endee_client`, `fake_embedder`, `sample_documents` fixtures. `live_client`: a real `Endee` client, skips if no/invalid token. `store_factory`: creates `EndeeVectorStore` instances against the live server and deletes them on teardown, shared by both test files. `uid()`/`safe_delete()` helpers, plus the autouse session-scoped `_cleanup_stale_collections` stale-sweep fixture. |
| `run_tests.sh` | Shell wrapper that creates a virtualenv, installs dependencies, sets environment variables, and invokes pytest. Run with `--help` for full usage and examples. |

---

## Design notes

**One file per kind of test:** `test_unit.py` for unit tests, `test_integration.py` for integration tests, grouped into classes by concern. `TestRetrieval` only exists at the integration level, since it exercises real LlamaIndex framework objects end to end rather than mocking anything.

**Plain pytest style:** plain `assert` and fixtures only, no `unittest.TestCase`. `@pytest.mark.parametrize` covers input matrices.

**Independent tests:** no test depends on another running first.

**Unique collection names:** every integration test uses a unique name via `uid()`, prefixed `llamaindex_test_` plus a random hex suffix, never fixed, so runs never collide.

**Cleanup:** `store_factory` deletes its own collection via teardown. `_cleanup_stale_collections` sweeps leftovers from interrupted runs.

# Endee-LangChain Test Suite

There are two kinds of tests, selected via pytest markers. Unit tests mock the `endee` client, so they need nothing beyond the package itself. Integration tests run against a live or local Endee server and cover the `EndeeVectorStore` wrapper end to end.

---

## Prerequisites

- Python 3.9 or higher
- For **unit** tests: nothing else. `pip install -e .` plus the test deps is enough.
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
./tests/run_tests.sh --unit -- -k test_similarity_search
```

### Manual setup (all platforms)

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv && source .venv/bin/activate   # macOS / Linux
# python -m venv .venv && .venv\Scripts\activate      # Windows

# 2. Install the package and test dependencies
pip install -e .
pip install pytest pytest-mock pytest-timeout pytest-asyncio numpy

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
| `ENDEE_API_TOKEN` | Only for integration tests | Endee API token. Integration tests skip automatically if unset. |
| `ENDEE_BASE_URL` | No | Server base URL (e.g. `http://localhost:8080/api/v2`). Defaults to the cloud endpoint derived from the token. |

---

## Test files

Tests are split into two files by kind, selected via the `unit`/`integration` pytest markers. Each file holds several concern-classes, one per area of `EndeeVectorStore`.

### `test_unit.py`

| Class | What is tested |
|-------|-----------------|
| `TestVectorStoreUnit` | Constructor validation, `_validate_collection_config` mismatch/warn paths, `add_texts`/`from_texts`/`from_documents`/`from_existing_collection`, batch-size boundary, embedding-provider truncation matrix (openai/cohere/huggingface/default), cosine-ranked similarity search, `delete`, `update_filters`, `add_objects`/multi-field RRF wiring, network-failure propagation. |
| `TestFiltersUnit` | Filter forwarding/translation, `$eq`/`$in`/multi-filter, unsupported-operator error. |
| `TestSparseUnit` | `SparseVector`, `SparseModelAdapter`, `wrap_sparse_model`, `EndeeModelSparse`, hybrid auto-detection, async `aembed_documents`/`aembed_query`. |

### `test_integration.py`

| Class | What is tested |
|-------|-----------------|
| `TestVectorStoreIntegration` | Live CRUD, config validation (incl. mismatch detection on reconnect), filter assertions, client construction, collection lifecycle, and factory-method coverage against a real server. |
| `TestMultiFieldIntegration` | Live collection with separate title, content, and keywords fields. |
| `TestSparseIntegration` | Live hybrid/BM25 auto-detect against a real server. |
| `TestMultiVectorIntegration` | Live collection with a dense field and a `multi_vector` field, exercised via `multi_field_search`. |

All tests in `test_unit.py` mock the `endee` client (no network); all tests in `test_integration.py` require a live/local Endee server.

---

## Support files

| File | Purpose |
|------|---------|
| `conftest.py` | `MockEndee`/`MockEndeeCollection`: a mock backend (unit tests). `mock_endee_client`, `fake_embedder`, `fake_sparse_embedding` fixtures. `live_client`: a real `Endee` client that skips if no token is set. `uid()`/`safe_delete()`: helpers for integration fixtures that generate unique collection names (truncated to fit the server's 48-char limit) and delete collections silently. `_cleanup_stale_collections`: an autouse session fixture that sweeps leftover test collections from a previous interrupted run, plus a final sweep at session end. |
| `run_tests.sh` | A shell wrapper that creates a virtualenv, installs dependencies, sets environment variables, and invokes pytest. Run with `--help` for full usage and examples. |

---

## Design notes

**One file per kind of test:** `test_unit.py` for unit tests, `test_integration.py` for integration tests, grouped into classes by concern. Filters get their own class only at the unit level; at the integration level that coverage folds into `TestVectorStoreIntegration`.

**Plain pytest style:** plain `assert` and fixtures only, no `unittest.TestCase`. `@pytest.mark.parametrize` covers input matrices.

**Independent tests:** no test depends on another running first.

**Unique collection names:** every integration test uses a unique name via `uid()`, prefixed `langchain_test_` plus a random hex suffix, never fixed, so runs never collide.

**Cleanup:** fixtures delete their own collection via teardown. `_cleanup_stale_collections` sweeps leftovers at session start and end.

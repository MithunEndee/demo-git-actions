# Endee-CrewAI Test Suite

There are two kinds of tests, selected via pytest markers. Unit tests mock
the `endee` client, so they need nothing beyond the package itself.
Integration tests run against a live or local Endee server and cover the
`EndeeVectorStore` wrapper end to end.

---

## Prerequisites

- Python 3.10 or higher (`<3.14`, per `setup.py`)
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
./tests/run_tests.sh --unit -- -k test_save
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
| `ENDEE_API_TOKEN` | Only for integration tests | Endee API token. Integration tests skip automatically if unset. |
| `ENDEE_BASE_URL` | No | Server base URL (e.g. `http://localhost:8080/api/v2`). Defaults to the cloud endpoint derived from the token. |

---

## Test files

Tests are split into two files by kind, not by concern: `test_unit.py` (mocked, no server) and `test_integration.py` (live server). The `unit`/`integration` pytest markers select between them. Within each file, concerns live in their own class.

### `test_unit.py`

| Class | What is tested |
|-------|-----------------|
| `TestVectorStoreUnit` | Core CRUD on `EndeeVectorStore`. Covers init/create-or-reuse (incl. `force_recreate`), `save` (incl. the text-truncation boundary), `search` (single-field and multi-field RRF fusion), `get_objects`/`get_vector`/`delete_vector`, `update_filters`, `reset`, `describe`, `close` (session/client fallback), sparse auto-detection wiring, and network-failure propagation. |
| `TestFiltersUnit` | Filter-list normalization (single dict to list) and filter-data construction from primitive-typed metadata. |
| `TestSparseUnit` | `SparseVector`, `SparseModelAdapter`, `wrap_sparse_model`, `EndeeModelSparse`, and hybrid search RRF wiring. |

### `test_integration.py`

| Class | What is tested |
|-------|-----------------|
| `TestVectorStoreIntegration` | The full dense lifecycle, including filtered search/delete, client construction (api_token/base_url), and collection lifecycle (force_recreate, reconnect), against a live server, split into independent tests. |
| `TestSparseIntegration` | Hybrid dense+sparse collections, including a user-supplied sparse embedding, against a live server. |
| `TestMultiVectorIntegration` | Multi-vector field coverage (`add_objects` + `multi_field_search`) against a live server. |

---

## Support files

| File | Purpose |
|------|---------|
| `conftest.py` | `MockEndee`/`MockEndeeCollection`: a mock backend (unit tests). `mock_endee_client`, `fake_embedder`, `make_store` fixtures. `live_client`, a real `Endee` client that skips if no token. `uid()`/`safe_delete()`: unique collection naming and silent-delete helpers for integration fixtures. `_cleanup_stale_collections`: an autouse session fixture that sweeps leftover test collections from a previous interrupted run. |
| `run_tests.sh` | A shell wrapper that creates a virtualenv, installs dependencies, sets environment variables, and invokes pytest. Run with `--help` for full usage and examples. |

---

## Design notes

**One file per kind of test:** `test_unit.py` for unit tests, `test_integration.py` for integration tests, each grouped into classes by concern.

**Plain pytest style:** plain `assert` and fixtures only, no `unittest.TestCase`. `@pytest.mark.parametrize` covers input matrices.

**Independent tests:** no test depends on another running first; each sets up its own state.

**Unique collection names:** every integration test uses a unique name via `uid()`, prefixed `crewai_test_` plus a random hex suffix, never fixed, so runs never collide.

**Cleanup:** fixtures delete their own collection after each test. `_cleanup_stale_collections` clears leftovers from interrupted runs.

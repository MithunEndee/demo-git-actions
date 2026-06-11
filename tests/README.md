# Functional Tests

End-to-end tests for the Endee Python client against Endee Serverless.

**Requirements:** An `ENDEE_TOKEN` from [app.endee.io](https://app.endee.io).

---

## Running locally

### Option 1 - Shell script (recommended)

The `run_tests.sh` script handles everything - it creates a `.venv` virtual environment
at the repo root (if one does not already exist), installs all dependencies into it,
and runs pytest. It can be invoked from any directory.

```bash
# Full test suite
./tests/run_tests.sh --token user:mytoken:us-east

# With an explicit base URL
./tests/run_tests.sh --token user:mytoken:us-east --base-url http://0.0.0.0:8080/api/v1

# Delete .venv and reinstall everything before running
./tests/run_tests.sh --token user:mytoken:us-east --clean

# Run a specific test file
./tests/run_tests.sh --token user:mytoken:us-east -- tests/test_query_basic.py

# Run tests matching a keyword
./tests/run_tests.sh --token user:mytoken:us-east -- -k test_filter_eq

# Show help
./tests/run_tests.sh --help
```

Anything after `--` is passed directly to pytest, so all standard pytest flags work.

The virtual environment is created once at `.venv/` and reused on subsequent runs.
Pass `--clean` to wipe and recreate it from scratch.

---

### Option 2 - Manual

**Step 1: Install dependencies**

```bash
# From source (development)
pip install -e .

# Or from PyPI
pip install endee

pip install pytest pytest-html pytest-timeout numpy
```

**Step 2: Set environment variables**

```bash
export ENDEE_TOKEN=user:mytoken:us-east

# Optional - only needed to override the URL derived from the token
export ENDEE_BASE_URL=http://0.0.0.0:8080/api/v1
```

**Step 3: Run pytest from the repo root**

```bash
pytest tests/
```

---

## Running via GitHub Actions

Trigger manually: **Actions - Functional Tests - Run workflow**

| Input | Required | Description |
|-------|----------|-------------|
| `token` | No | Endee Serverless API token. Leave blank to use the `ENDEE_TOKEN` repository secret. |
| `base_url` | No | API base URL override. Leave blank to derive from the token. |

Results appear on the Actions run summary page. A full HTML report and JUnit XML
are uploaded as artifacts (retained for 7 days).

To enable automatic runs on pull requests, uncomment the `pull_request` trigger in
`.github/workflows/functional_test.yml` and add the following under
**Settings - Secrets and variables - Actions**:

| Secret | Required | Description |
|--------|----------|-------------|
| `ENDEE_TOKEN` | Yes | Endee Serverless API token. |
| `ENDEE_BASE_URL` | No | Override the API base URL. Leave unset to derive it from the token. |

---

## Test files

| File | What it covers |
|------|----------------|
| `test_index_management.py` | Create, list, describe, delete indexes; all precision x space type combinations; HNSW params; hybrid indexes |
| `test_vector_operations.py` | Upsert, get, update filters, delete by ID, delete by filter |
| `test_query_basic.py` | Query result structure, `top_k`, `ef`, `include_vectors`, meta round-trip |
| `test_query_filters.py` | `$eq`, `$in`, `$range` operators; combined filters; `filter_boost_percentage`; `prefilter_cardinality_threshold` |
| `test_hybrid_search.py` | Dense-only, sparse-only, and full hybrid queries; result structure; `top_k`/`ef`; filter operators; `filter_boost_percentage`; RRF weights; `get_vector`; `update_filters`; `delete_vector`; `delete_with_filter` |
| `test_error_handling.py` | Client-side `ValueError` for invalid inputs; server-side `ConflictException` / `NotFoundException`; batch and dimension constraints; token authentication errors |

---

## Support files

| File | Purpose |
|------|---------|
| `conftest.py` | pytest fixtures (`client`, `empty_index`, `populated_index`, `empty_hybrid_index`, `populated_hybrid_index`) |
| `helpers.py` | Shared constants (`DIM`, `N_VECTORS`, ...) and generators (`dense_vec`, `sparse_vec`, `make_item`, `get_index_names`, ...) |
| `run_tests.sh` | Shell script for running the suite locally with token and optional URL |
| `pytest.ini` *(repo root)* | Sets `testpaths = tests` and `pythonpath = tests` so `helpers` is importable without installation |

---

# Endee Python Client - Functional Test Suite

This directory contains the end-to-end functional tests for the [Endee](https://endee.io) Python client.
The suite validates the full lifecycle of the Endee Serverless vector database - index management, vector operations, querying, filtering, hybrid search, and error handling.

> **Note:** An active Endee Serverless API token is required to run these tests.

---

## Running the tests

Choose the method that fits your situation:

| Method | Best for |
|--------|----------|
| [Run locally](#running-locally) | Development and debugging on your machine |
| [GitHub Actions](#running-via-github-actions) | Automated runs on pull requests, or triggered manually |

---

## Running locally

Two options are available. **Option 1 is recommended** - it handles everything automatically.
Option 2 gives you full manual control, and also works on Windows.

### Option 1 - Shell script (macOS / Linux)

The script creates a virtual environment, installs dependencies, and runs the tests.

```bash
# Show all available options and examples
./tests/run_tests.sh --help

# Run the full test suite
./tests/run_tests.sh --token <your_token>

# Override the API base URL
./tests/run_tests.sh --token <your_token> --base-url http://localhost:8080/api/v1
```

### Option 2 - Manual (all platforms)

**Prerequisites:** Python 3.9 or higher must be installed.

**Step 1: Create and activate a virtual environment**

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

**Step 2: Install dependencies**

```bash
pip install -e .
pip install pytest numpy
```

**Step 3: Set environment variables**

```bash
# macOS / Linux
export ENDEE_TOKEN=<your_token>
export ENDEE_BASE_URL=http://localhost:8080/api/v1   # optional

# Windows
set ENDEE_TOKEN=<your_token>
set ENDEE_BASE_URL=http://localhost:8080/api/v1       # optional
```

**Step 4: Run from the repo root**

```bash
# Run the full test suite
pytest tests/

# Run a specific test file
pytest tests/test_querying.py

# Run tests matching a keyword
pytest tests/ -k test_filter_eq
```

> **Tip:** Add `-v` to any command above to see each test name and its result as it runs (e.g. `pytest tests/ -v`).

---

## Running via GitHub Actions

The workflow supports both automatic and manual runs.

**Automatic:** Tests run on every pull request to `main` or `master`. No setup needed - the workflow picks up the token from the repository secrets automatically.

**Manual:** Go to **Actions -> Functional Tests -> Run workflow**. You can optionally provide a token and base URL directly - if left blank, the repository secrets are used.

| Input | Required | Description |
|-------|----------|-------------|
| `token` | No | API token. Leave blank to use the `ENDEE_TOKEN` repository secret. |
| `base_url` | No | API base URL override. Leave blank to derive from the token. |

The following secrets are already configured under **Settings -> Secrets and variables -> Actions**:

| Secret | Required | Description |
|--------|----------|-------------|
| `ENDEE_TOKEN` | Yes | Endee Serverless API token. |
| `ENDEE_BASE_URL` | No | Override the API base URL. Leave unset to derive from the token. |

Results appear on the Actions run summary page. A full HTML report and test results are uploaded as artifacts and kept for 7 days.

---

## Test files

Each file covers a specific area of the client API. All tests run against Endee Serverless and clean up after themselves.

| File | What it covers |
|------|----------------|
| `test_index_management.py` | Create, list, describe, and delete indexes; all precision and space type combinations; HNSW params; hybrid indexes |
| `test_vector_operations.py` | Upsert, get, update filters, delete by ID, delete by filter |
| `test_querying.py` | Query result structure, `top_k`, `ef`, `include_vectors`, meta round-trip |
| `test_filtering.py` | `$eq`, `$in`, `$range` operators; combined filters; `filter_boost_percentage`; `prefilter_cardinality_threshold` |
| `test_hybrid_search.py` | Dense-only, sparse-only, and full hybrid queries; RRF weights; filter operators; vector and filter update; delete operations |
| `test_error_handling.py` | Invalid inputs, duplicate indexes, dimension mismatches, batch limits, and authentication errors |

---

## Support files

These files are not test cases - they provide shared setup and utilities used across the test suite.

| File | Purpose |
|------|---------|
| `conftest.py` | Shared pytest fixtures: `client`, `empty_index`, `populated_index`, `empty_hybrid_index`, `populated_hybrid_index` |
| `helpers.py` | Constants (`DIM`, `N_VECTORS`, ...) and vector generators (`dense_vec`, `sparse_vec`, ...) |
| `run_tests.sh` | Shell script for running the suite locally with automatic virtual environment setup and cleanup |
| `pytest.ini` *(repo root)* | Tells pytest to look in the `tests/` directory by default |

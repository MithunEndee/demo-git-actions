# Functional Tests

End-to-end tests for the Endee Python client. Tests run against a live Endee
Server — either the hosted cloud or a local OSS Docker container.

---

## Running locally

**Install dependencies**

```bash
# From source (development)
pip install -e .

# Or from PyPI
pip install endee

pip install pytest pytest-html pytest-timeout numpy
```

**OSS mode** (no token — spins up a local Docker container)

```bash
# Start the server
docker run -d \
  --ulimit nofile=100000:100000 \
  -p 8080:8080 \
  -e NDD_AUTH_TOKEN="" \
  -e NDD_NUM_THREADS=2 \
  --name endee-oss \
  endeeio/endee-server:latest

# Run tests
pytest
```

**Cloud mode** (token from app.endee.io)

```bash
export ENDEE_TOKEN=your_token_here
pytest
```

**Override the server URL** (optional)

```bash
export ENDEE_BASE_URL=http://0.0.0.0:8080/api/v1
pytest
```

---

## Running via GitHub Actions

Trigger manually: **Actions → Functional Tests → Run workflow**

| Input | Description |
|-------|-------------|
| `token` | API token from app.endee.io. Leave empty for OSS mode. |
| `base_url` | Optional URL override. Defaults to cloud (from token) or `http://127.0.0.1:8080/api/v1` (OSS). |

Results appear directly on the Actions run summary page. A full HTML report
and JUnit XML are also uploaded as artifacts.

---

## Test files

| File | What it covers |
|------|----------------|
| `test_index_management.py` | Create, list, describe, delete indexes; all precision × space type combinations; HNSW params; hybrid indexes |
| `test_vector_operations.py` | Upsert, get, update filters, delete by ID, delete by filter |
| `test_query_basic.py` | Query result structure, `top_k`, `ef`, `include_vectors`, meta round-trip |
| `test_query_filters.py` | `$eq`, `$in`, `$range` operators; combined filters; `filter_boost_percentage`; `prefilter_cardinality_threshold` |
| `test_hybrid_search.py` | Dense-only, sparse-only, and full hybrid queries; result structure; `top_k`/`ef`; `$eq`/`$in`/`$range` filters; `filter_boost_percentage`; RRF weights; `get_vector`; `update_filters`; `delete_vector`; `delete_with_filter` |
| `test_error_handling.py` | Client-side `ValueError` for invalid inputs; server-side `ConflictException` / `NotFoundException`; batch and dimension constraints |
| `test_serverless.py` | **Serverless/cloud only** (`ENDEE_TOKEN` required — skipped in OSS mode). INT8E precision: index creation × all space types and dimensions; upsert; query; `get_vector`; `update_filters`; `delete_vector`; `delete_with_filter`. Rebuild: trigger, response shape, config changes, HNSW param combinations, `rebuild_status` polling. Token/auth: invalid token, empty token, `set_token`, `AuthenticationException` |

---

## Support files

| File | Purpose |
|------|---------|
| `conftest.py` | pytest fixtures (`client`, `empty_index`, `populated_index`, `empty_hybrid_index`, `populated_hybrid_index`) |
| `helpers.py` | Shared constants (`DIM`, `N_VECTORS`, …) and generators (`dense_vec`, `sparse_vec`, `make_item`, `get_index_names`, …) |
| `pytest.ini` *(repo root)* | Sets `testpaths = tests` and `pythonpath = tests` so `helpers` is importable without installation |

---

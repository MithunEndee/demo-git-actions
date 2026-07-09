# Endee - Python Client

Endee is a high-performance vector database with three retrieval modes you can mix
**per object and per query**:

- **Dense vector search** - semantic ANN over embeddings (HNSW).
- **Keyword search** - sparse embeddings (BM25-style inverted index).
- **Multi-vector search** - many vectors per object, pooled for retrieval and reranked.

A single object can hold a dense field, a sparse field, and a multi-vector field at
once, and a single query can hit any subset - returned per-field or fused with RRF.


## Concepts

```
Database                      ← top level; one database = one token, fully isolated
└── Collection                ← a named set of typed fields
    └── Object                ← one record
        ├── id                ← unique string key
        ├── meta              ← arbitrary JSON, returned with results
        ├── filter            ← tags used to filter searches
        └── fields            ← the vectors, by field name:
              embedding   (dense)         [0.1, 0.2, ...]
              keywords    (sparse)        {indices, values}
              multivec     (multi_vector)  [[...], [...]]
```

- **Database** - top-level namespace; created by the **root token**, addressed by a **db token**. Databases are isolated from each other.
- **Collection** - declares named, typed fields (`vector` / `sparse` / `multi_vector`) and holds objects.
- **Object** - one item in a collection:
  - **`id`** - unique string identifier (insert-or-replace key).
  - **`meta`** - arbitrary JSON payload, echoed back in search results.
  - **`filter`** - key/value tags queried with filter operators (`$eq`, `$in`, `$range`, …).
  - **`fields`** - the actual vectors, one entry per field you populate.

---

## Install

```bash
pip install endee
```

Python ≥ 3.9. Depends on `numpy`, `msgpack`, `orjson`, `requests`, `httpx`.

---

## Quick start

Authentication is **always on** - every request needs a token.

```python
from endee import Endee

# 1) Admin: create a database with the ROOT token.
#    If the server was started WITHOUT NDD_ROOT_TOKEN, the root token is the
#    insecure dev default "unsafe_root_token".
admin = Endee(token="unsafe_root_token")
admin.set_base_url("http://localhost:8080/api/v2")
db_token = admin.create_database("alice")        # returns the db token, e.g. "alice:9f3..."

# 2) App: everything else uses the DB token.
client = Endee(token=db_token)
client.set_base_url("http://localhost:8080/api/v2")

# 3) Create a collection with a dense field.
client.create_collection(
    name="products",
    fields=[
        {"name": "embedding", "type": "vector",
         "params": {"dimension": 8, "space_type": "cosine", "precision": "int8"}},
    ],
)
collection = client.get_collection("products")

# 4) Insert objects (dense vector per object).
collection.upsert([
    {"id": "p1", "meta": {"name": "Wireless Headphones"}, "filter": {"category": "electronics"},
     "fields": {"embedding": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]}},
    {"id": "p2", "meta": {"name": "Running Shoes"}, "filter": {"category": "footwear"},
     "fields": {"embedding": [0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.2, 0.3]}},
])

# 5) Search.
hits = collection.search(fields={"embedding": {"query": [0.2, 0.2, 0.3, 0.3, 0.4, 0.5, 0.6, 0.7], "limit": 3}})
for h in hits["results"]["embedding"]:
    print(h["id"], round(h["similarity"], 4), h["meta"])
```

> **Tokens:** the **root token** (server's `NDD_ROOT_TOKEN`, or `unsafe_root_token`
> in dev) administers databases; a **db token** (`db_name:secret`, returned by
> `create_database`) does all data work. A missing/invalid token → 401; a read-only
> token on a write → 403.

---

## Field types

Add fields at collection-creation time. Each is a plain dict:

| Type | Keys | Value on upsert | `query` value |
|------|------|-----------------|---------------|
| `vector` | `params`: `dimension`, `space_type`, `precision` | `[0.1, 0.2, ...]` | `[0.2, ...]` |
| `sparse` | `sparse_model` (top-level) | `{"indices":[...], "values":[...]}` | `{"indices":[...], "values":[...]}` |
| `multi_vector` | `params`: vector params **+ `pooling`** (`"mean"`/`"max"`) | `[[...], [...]]` | `[[...], [...]]` |

On **upsert**, the value sits directly under the field name. On **search**, the
`query` value is wrapped in a per-field config: `{"query": <value>, "limit"?: ...,
"ef_search"?: ...}` (see [Search](#search)).

`space_type`: `cosine` (default, normalized client-side), `l2`, `ip`.
`precision`: `float32`, `float16`, `int16`, `int8`, `int8e`, `binary`.

### Adding a sparse (keyword) field

```python
client.create_collection("docs", fields=[
    {"name": "embedding", "type": "vector",
     "params": {"dimension": 8, "space_type": "cosine", "precision": "int8"}},
    {"name": "keywords", "type": "sparse", "sparse_model": "default"},
])

collection.upsert([{
    "id": "d1",
    "fields": {
        "embedding": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "keywords":  {"indices": [3, 17, 42], "values": [0.9, 0.5, 0.2]},   # term ids + weights
    },
}])
```

### Adding a multi-vector field

A multi-vector field stores several vectors per object, pooled into one vector for
retrieval. Two **pooling strategies** are supported:

- **`mean`** (default) - elementwise average of the members.
- **`max`** - elementwise maximum of the members.

```python
client.create_collection("passages", fields=[
    {"name": "embedding", "type": "vector",
     "params": {"dimension": 8, "space_type": "cosine", "precision": "int8"}},
    {"name": "multivec", "type": "multi_vector",
     "params": {"dimension": 8, "space_type": "cosine", "precision": "int8", "pooling": "mean"}},
])

collection.upsert([{
    "id": "x1",
    "fields": {
        "embedding": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "multivec":   [[0.1]*8, [0.2]*8, [0.3]*8],   # one vector per token/passage
    },
}])
```

You can set any subset of fields on a given object; `upsert` is insert-or-replace by `id`.

---

## Search

Every field is queried with the unified config
`{"query": ..., "limit"?: ..., "ef_search"?: ...}` - `query` is required, and
`limit` is the max hits to return **for that field** (default 10, max 4096).
There's no overall result limit and no single-field special case: `search`
**always** returns one ranked list per field, keyed by field name:
`{"results": {field: [hit, ...], ...}}`. To search a single field, just pass that
one field. A hit is `{"id", "similarity", "meta", "filter"}`; results carry
`meta`/`filter` but not the stored vectors - use
[`get_objects`](#object-operations) to fetch those.

### Single field

```python
res = collection.search(fields={"embedding": {"query": [0.2]*8, "limit": 5}})
res["results"]["embedding"]   # → [ {id, similarity, meta, filter}, ... ]
```

### Multiple fields (per-field results)

Query several fields at once → each field's own ranked list (scores are per-field
and not comparable across fields). Limits are **per field**:

```python
res = collection.search(
    fields={
        "embedding": {"query": [0.2]*8, "limit": 50},
        "keywords":  {"query": {"indices": [3, 17], "values": [0.8, 0.4]}, "limit": 50},
    },
)
res["results"]["embedding"]   # → [ {id, similarity, ...}, ... ]
res["results"]["keywords"]    # → [ {id, similarity, ...}, ... ]
```

### Fusing fields with RRF (`rerank`)

Reranking is a **separate step**: pass the `search` result into `rerank` to fuse
the per-field lists into a single ranked list via Reciprocal Rank Fusion.

```python
from endee import rerank

res = collection.search(fields={
    "embedding": {"query": [0.2]*8, "limit": 50},
    "keywords":  {"query": {"indices": [3, 17], "values": [0.8, 0.4]}, "limit": 50},
})
fused = rerank(
    res,
    limit=10,
    field_weights={"embedding": 0.6, "keywords": 0.4},   # optional, must sum to 1.0
    rrf_k=60,                                             # optional RRF smoothing constant k (default 60)
)
fused["results"]   # → [ {id, similarity (fused score), meta, filter}, ... ]
```

`rrf_k` is the RRF **smoothing parameter** k: a hit at rank `r` (1-based) in field
`f` contributes `field_weights[f] / (rrf_k + r)` to its fused score. Larger `k`
flattens the contribution gap between ranks (rank position matters less); smaller
`k` sharpens it (top ranks dominate). Defaults to `60`.

### Multi-vector

```python
collection.search(fields={"multivec": {"query": [[0.1]*8, [0.2]*8], "limit": 5}})
# multi-vector participates in multi-field search + rerank exactly like above.
```

### Filters

`filter` is an array of conditions (AND-combined):

```python
collection.search(
    fields={"embedding": {"query": [0.2]*8, "limit": 5}},
    filter=[{"category": {"$eq": "electronics"}}, {"price": {"$range": [10, 100]}}],
)
```

Operators: `$eq`, `$in`, `$range`, `$gt`, `$gte`, `$lt`, `$lte`.

**Filter tuning** (filtered search only) - two optional `search` params trade
speed for recall when a `filter` is active:

```python
collection.search(
    fields={"embedding": {"query": [0.2]*8, "limit": 5}},
    filter=[{"category": {"$eq": "electronics"}}],
    prefilter_cardinality_threshold=5000,   # ≤ this many matches → brute-force prefilter (1,000–1,000,000)
    filter_boost_percentage=25,             # expand HNSW candidate pool by 25% (0–100)
)
```

**Return-shape summary**

| Step | `results` |
|------|-----------|
| `search` (any number of fields) | `{field: [hit, ...], ...}` |
| `rerank(search_result, ...)` | `[hit, ...]` (fused) |

---

## Object operations

```python
collection.get_objects(["p1", "p2"])     # full objects: {id, meta, filter, vectors, sparses, multi_vectors}
collection.delete_object("p1")           # {"deleted": "p1"}
collection.delete_by_filter([{"category": {"$eq": "footwear"}}])   # {"deleted": <count>}
collection.update_filters([{"id": "p2", "filter": {"category": "sale"}}])   # {"updated": <count>}
```

## Collection maintenance

```python
collection.describe()                              # {name, fields, created_at, layout_version}

# Rebuild one or more dense fields' HNSW graphs in a single async call.
# Pass a list of field specs (server-native keys); only M/ef_con may change.
# Any sparse field in the list is skipped server-side.
collection.rebuild([
    {"field": "embedding", "M": 20, "ef_con": 200},
    {"field": "colbert",   "M": 12, "ef_con": 120},
])   # {"fields_total": 2, "total_objects": ..., "status": "in_progress"}

# Poll progress; reports the latest rebuilt field plus overall counts.
collection.rebuild_status()   # {field, fields_done, fields_total, percent_complete, status, ...}

collection.shrink()                                # reclaim disk
client.list_collections(); client.delete_collection("products")
```

## Backups (per collection)

A backup captures **one collection**; restoring it creates a **new collection**.
Backups are created on the collection; they're listed/restored/managed at the
database level (the database owns its backups).

```python
collection.create_backup("nightly")                    # back up THIS collection as a backup named "nightly" (async)
client.active_backup()                                  # status of the running backup; poll until it finishes
client.list_backups()                                   # list all backups in this database
client.backup_info("nightly")                           # metadata for the backup named "nightly"
client.restore_backup("nightly", "products_restored")   # restore backup "nightly" into a new collection "products_restored"
client.download_backup("nightly", "/tmp/nightly.tar")   # download backup "nightly" to the local file "/tmp/nightly.tar"
client.upload_backup("/tmp/nightly.tar")                # upload the local .tar file back into this database
client.delete_backup("nightly")                         # delete the backup named "nightly"
```

## Token management

**Admin (root token)** - manage databases and their tokens:

```python
admin.list_databases()                                       # list all databases
admin.get_database("alice")                                  # info for database "alice"
admin.deactivate_database("alice")                           # disable database "alice" (its tokens stop working; data kept)
admin.activate_database("alice")                             # re-enable database "alice"
admin.delete_database("alice")                               # delete database "alice" and ALL its data
admin.create_token("alice", name="ingest", token_type="rw")  # mint token "ingest" for db "alice"; returns the new db token (string)
admin.list_tokens("alice")                                   # list the token names/types of database "alice"
admin.delete_token("alice", "ingest")                        # delete the token named "ingest" from database "alice"
admin.list_db_collections("alice")                           # list collections in database "alice"
admin.list_all_collections()                                 # list collections across all databases
```

**Self-service (db token)** - a database manages its own tokens, no root needed:

```python
client.create_my_token("worker", token_type="r")            # mint token "worker" for YOUR db; returns the new db token (string)
client.list_my_tokens()                                     # list your db's token names/types
client.delete_my_token("worker")                            # delete your db's token named "worker"
```

`token_type`: `rw` (read-write, default) or `r` (read-only - 403 on writes).

## Server info

```python
client.health()    # {"status": "ok", ...}
client.stats()     # {"version", "uptime", "total_requests"}
```

---

## Reference

- **Limits (client-side, fail fast):** ≤ 10,000 objects per `upsert`; `limit` 1-4096;
  `ef_search` 1-1024; filter key ≤ 128 B, string value ≤ 1024 B; vector dim must
  match the field.
- **Errors:** non-2xx raise typed exceptions from `endee` - `EndeeException` (base),
  `APIException`, `AuthenticationException` (401), `ForbiddenException` (403),
  `NotFoundException` (404), `ConflictException` (409), `ServerException` (5xx).
  Bad inputs raise `ValueError` before any network call.

```python
from endee import EndeeException, NotFoundException
try:
    collection.upsert(objects)
except NotFoundException as e:
    ...
except EndeeException as e:
    ...
```

---

## Examples (`scripts/`)

- **`demo.py`** - full collection walkthrough. `python3 scripts/demo.py --token alice:xxxx`
- **`demo_admin.py`** - database administration. `python3 scripts/demo_admin.py --root-token <root>`
- **`test_filters.py`** - filter-operator checks. **`test_bulk_100k.py`** - bulk ingest + recall.

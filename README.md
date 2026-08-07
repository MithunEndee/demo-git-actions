# crewai-endee

**Endee vector database integration for CrewAI agent memory**

`crewai-endee` connects [Endee](https://github.com/endee-io/endee) to [CrewAI](https://crewai.com), giving your agents persistent memory with dense, hybrid, and multi-field retrieval.

Uses the same `fields=` configuration as the [Endee Python client](https://github.com/endee-io/endee)'s `create_collection()`.

---

## Installation

Requires **Python 3.10–3.13**.

```bash
pip install crewai-endee
```

This installs `endee`, `endee_model`, and `crewai` automatically.

---

## Quick Start

```python
from crewai_endee import EndeeVectorStore

store = EndeeVectorStore(
    type="my_collection",
    embedder_config={
        "provider": "sentence-transformer",
        "config": {"model_name": "all-MiniLM-L6-v2"},
    },
    fields=[
        {"name": "dense", "type": "vector",
         "params": {"dimension": 384, "space_type": "cosine", "precision": "int8"}},
    ],
)

store.save("Go is a statically typed language by Google.", {"lang": "Go"})
results = store.search("static typing", limit=3)
```

---

## Connect to Endee

### With API token

Sign up at [endee.io](https://endee.io) and get your token. See the [Endee docs](https://docs.endee.io/quick-start) for details.

```python
store = EndeeVectorStore(
    type="my_collection",
    embedder_config=embedder_config,
    api_token="YOUR_ENDEE_API_TOKEN",
    fields=[
        {"name": "dense", "type": "vector",
         "params": {"dimension": 384, "space_type": "cosine", "precision": "int8"}},
    ],
)
```

### Without API token (local)

Run the open-source Endee server locally. See [github.com/endee-io/endee](https://github.com/endee-io/endee) for setup. Omit `api_token`:

```python
store = EndeeVectorStore(
    type="my_collection",
    embedder_config=embedder_config,
    fields=[
        {"name": "dense", "type": "vector",
         "params": {"dimension": 384, "space_type": "cosine", "precision": "int8"}},
    ],
)
```

---

## Dense Mode

```python
from crewai_endee import EndeeVectorStore

store = EndeeVectorStore(
    type="demo_dense",
    embedder_config=embedder_config,
    fields=[
        {"name": "dense", "type": "vector",
         "params": {"dimension": 384, "space_type": "cosine", "precision": "int8"}},
    ],
    force_recreate=True,
)

store.save("Python is a dynamic language.", {"lang": "Python"})
results = store.search("dynamic typing", limit=3)
```

---

## Hybrid Mode (endee_bm25 — auto-encoded)

Add a sparse field with `"sparse_model": "endee_bm25"`. The BM25 sparse encoder is created automatically — no extra setup needed:

```python
store = EndeeVectorStore(
    type="demo_hybrid",
    embedder_config=embedder_config,
    fields=[
        {"name": "dense", "type": "vector",
         "params": {"dimension": 384, "space_type": "cosine", "precision": "int8"}},
        {"name": "sparse", "type": "sparse",
         "sparse_model": "endee_bm25"},
    ],
    force_recreate=True,
)

store.save("Go has native concurrency.", {"lang": "Go"})

# Hybrid search — dense similarity + BM25 keyword matching, fused via RRF
results = store.search("concurrency", limit=3)

# Tune fusion weights
results = store.search(
    "concurrency", limit=3,
    field_weights={"dense": 0.3, "sparse": 0.7},
    rrf_k=30,
)
```

---

## Hybrid Mode (default sparse — user-provided vectors)

Use `"sparse_model": "default"` and provide your own sparse vectors via `add_objects()` and `multi_field_search()`:

```python
import uuid
from endee import rerank

store = EndeeVectorStore(
    type="demo_hybrid_default",
    embedder_config=embedder_config,
    fields=[
        {"name": "dense", "type": "vector",
         "params": {"dimension": 384, "space_type": "cosine", "precision": "int8"}},
        {"name": "sparse", "type": "sparse", "sparse_model": "default"},
    ],
    force_recreate=True,
)

# Upsert with user-provided sparse vectors
store.add_objects([{
    "id": uuid.uuid4().hex,
    "meta": {"text": "Python ML libraries", "metadata": {"lang": "Python"}},
    "filter": {"lang": "Python"},
    "fields": {
        "dense": store.embedder(["Python ML libraries"])[0].tolist(),
        "sparse": {"indices": [10, 42, 99], "values": [0.9, 0.4, 0.7]},
    },
}])

# Search both fields with user-provided sparse query
raw = store.multi_field_search(fields={
    "dense":  {"query": store.embedder(["ML"])[0].tolist(), "limit": 3},
    "sparse": {"query": {"indices": [10, 42], "values": [0.8, 0.5]}, "limit": 3},
})
fused = rerank(raw, limit=3, field_weights={"dense": 0.5, "sparse": 0.5})
```

---

## Multi-Vector Mode

Add a `multi_vector` field for per-chunk or ColBERT-style embeddings:

```python
store = EndeeVectorStore(
    type="demo_multi_vector",
    embedder_config=embedder_config,
    fields=[
        {"name": "dense", "type": "vector",
         "params": {"dimension": 384, "space_type": "cosine", "precision": "int8"}},
        {"name": "chunks", "type": "multi_vector",
         "params": {"dimension": 384, "space_type": "cosine",
                    "precision": "int8", "pooling": "mean"}},
    ],
    force_recreate=True,
)

# Upsert with multi-vector data via add_objects
store.add_objects([{
    "id": uuid.uuid4().hex,
    "meta": {"text": "Long article about distributed systems.", "metadata": {}},
    "filter": {},
    "fields": {
        "dense": store.embedder(["Long article about distributed systems."])[0].tolist(),
        "chunks": [[0.1, ...], [0.3, ...], [0.5, ...]],  # pre-computed chunk embeddings
    },
}])

# Dense-only search still works
results = store.search("distributed systems", limit=3)

# Multi-field search with rerank
raw = store.multi_field_search(fields={
    "dense":  {"query": store.embedder(["consensus"])[0].tolist(), "limit": 3},
    "chunks": {"query": [[0.1, ...], [0.3, ...]], "limit": 3},
})
fused = rerank(raw, limit=3, field_weights={"dense": 0.6, "chunks": 0.4})
```

---

## Search with Filters

Endee uses MongoDB-style operator syntax:

```python
# Filter by exact match
results = store.search("web language", limit=3, filter=[{"lang": {"$eq": "Python"}}])

# Filter with score threshold
results = store.search("systems", limit=3, score_threshold=0.3)

# Include raw vectors in results
results = store.search("Python", limit=1, include_vectors=True)

# HNSW tuning
results = store.search("query", limit=3, ef_search=256)

# Filtered search tuning
results = store.search(
    "query", limit=3,
    filter=[{"category": {"$eq": "systems"}}],
    prefilter_cardinality_threshold=5000,
    filter_boost_percentage=50,
)
```

Supported operators: `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`

---

## Collection Operations

```python
# Describe collection metadata
info = store.describe()

# Retrieve objects by ID
objects = store.get_objects(["id1", "id2"])

# Retrieve a single object by ID
obj = store.get_vector("some_id")

# Update filter metadata without re-embedding
store.update_filters([{"id": "some_id", "filter": {"reviewed": "true"}}])

# Delete a single object by ID
store.delete_vector("some_id")

# Delete all objects matching a filter
store.delete(filter=[{"category": {"$eq": "outdated"}}])

# Delete the entire collection
store.reset()

# Close the connection
store.close()
```

---

## CrewAI Integration

`EndeeVectorStore` extends CrewAI's `BaseRAGStorage`. Wire it into a Crew via `ShortTermMemory` and `EntityMemory`:

```python
from crewai import LLM, Agent, Crew, Process, Task
from crewai.memory.short_term.short_term_memory import ShortTermMemory
from crewai.memory.entity.entity_memory import EntityMemory
from crewai_endee import EndeeVectorStore

embedder_config = {
    "provider": "sentence-transformer",
    "config": {"model_name": "all-MiniLM-L6-v2"},
}

fields = [
    {"name": "dense", "type": "vector",
     "params": {"dimension": 384, "space_type": "cosine", "precision": "int8"}},
]

stm_store = EndeeVectorStore(
    type="crew_short_term",
    embedder_config=embedder_config,
    fields=fields,
)

entity_store = EndeeVectorStore(
    type="crew_entity",
    embedder_config=embedder_config,
    fields=fields,
)

short_term_memory = ShortTermMemory(storage=stm_store)
entity_memory = EntityMemory(storage=entity_store)

llm = LLM(model="gemini/gemini-2.5-flash", api_key=GOOGLE_API_KEY)

agent = Agent(
    role="Software Analyst",
    goal="Extract programming language characteristics",
    backstory="You study programming language design.",
    llm=llm,
)

task = Task(
    description="Analyse key characteristics of Python, Java, and Go.",
    expected_output="Structured summary of each language.",
    agent=agent,
)

crew = Crew(
    agents=[agent],
    tasks=[task],
    process=Process.sequential,
    memory=True,
    short_term_memory=short_term_memory,
    entity_memory=entity_memory,
    embedder=embedder_config,
    verbose=True,
)

result = crew.kickoff()
```

---

## API Reference

### Constructor

```python
EndeeVectorStore(
    type: str,                          # Collection name (required)
    embedder_config: dict,              # Dense embedder config (required)
    fields: list[dict],                 # Field definitions (required)
    api_token: str = None,              # Endee API token
    base_url: str = None,               # Custom API URL
    endee_client: EndeeClient = None,   # Pre-existing client
    sparse_embedding: SparseEmbeddings = None,  # Custom sparse model
    content_payload_key: str = "text",
    metadata_payload_key: str = "metadata",
    force_recreate: bool = False,       # Delete and recreate if exists
)
```

### Field Types

```python
# Dense vector
{"name": "dense", "type": "vector",
 "params": {"dimension": 384, "space_type": "cosine",
            "precision": "int8", "M": 16, "ef_con": 128}}

# Sparse (endee_bm25 — auto-encoded)
{"name": "sparse", "type": "sparse", "sparse_model": "endee_bm25"}

# Sparse (default — user provides vectors)
{"name": "sparse", "type": "sparse", "sparse_model": "default"}

# Multi-vector
{"name": "chunks", "type": "multi_vector",
 "params": {"dimension": 128, "space_type": "cosine",
            "precision": "float16", "pooling": "mean"}}
```

### Methods

| Method | Description |
|--------|-------------|
| `save(value, metadata)` | Embed text and upsert (dense + auto sparse) |
| `search(query, limit, filter, ...)` | Search with optional filters and RRF fusion |
| `add_objects(objects)` | Upsert arbitrary per-field data |
| `multi_field_search(fields, filter)` | Search multiple fields, raw per-field results |
| `ensure_collection()` | Verify collection exists |
| `describe()` | Collection metadata |
| `get_objects(ids)` | Retrieve objects by ID list |
| `get_vector(id)` | Retrieve single object by ID |
| `update_filters(updates)` | Update filter metadata without re-embedding |
| `delete_vector(id)` | Delete single object by ID |
| `delete(filter)` | Delete by metadata filter |
| `reset()` | Delete entire collection |
| `close()` | Close HTTP connection |

### Search Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `query` | *(required)* | Natural-language search query |
| `limit` | `3` | Max results |
| `filter` | `None` | `[{"field": {"$op": value}}]` |
| `score_threshold` | `0` | Minimum similarity score |
| `ef_search` | `None` | HNSW ef (default 128, max 1024) |
| `include_vectors` | `False` | Fetch raw vector data |
| `field_weights` | `None` | Per-field RRF weights (sum to 1.0) |
| `rrf_k` | `60` | RRF rank constant |
| `prefilter_cardinality_threshold` | `None` | Brute-force pre-filter threshold |
| `filter_boost_percentage` | `None` | Candidate pool expansion (0-100) |

### Exports

```python
from crewai_endee import (
    EndeeVectorStore,
    EndeeModelSparse,
    SparseEmbeddings,
    SparseVector,
    Precision,
    rerank,
)
```

---

Full Endee documentation: [docs.endee.io](https://docs.endee.io) | GitHub: [endee-io/endee](https://github.com/endee-io/endee) | CrewAI docs: [docs.crewai.com](https://docs.crewai.com)

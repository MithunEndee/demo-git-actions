# Endee LlamaIndex Integration

LlamaIndex vector store integration for [Endee](https://github.com/endee-io/endee).

For Endee setup, features, and server docs see [docs.endee.io](https://docs.endee.io/quick-start).

**Sections:** [Setup](#1-setup) | [Dense](#2-dense-search) | [Hybrid](#3-hybrid-search) | [Multi-Field](#4-multi-field--multi-vector) | [Filters](#5-filters) | [RAG Pipeline](#6-rag-pipeline)

---

## 1. Setup

### Install

```bash
pip install llama-index-vector-stores-endee
```

Pick an embedding model:

```bash
# Option A: Local (no API key)
pip install llama-index-embeddings-huggingface sentence-transformers

# Option B: OpenAI
pip install llama-index-embeddings-openai
```

### Create a Collection

Collections are created with `fields=` — the same pattern as the Python client. Each field has a name, type, and params.

```python
import os
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index_endee import EndeeVectorStore

Settings.embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
DIMENSION = 384

# Or OpenAI:
# from llama_index.embeddings.openai import OpenAIEmbedding
# Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
# DIMENSION = 1536

# Dense-only collection (single vector field)
vector_store = EndeeVectorStore.from_params(
    api_token=os.getenv("ENDEE_API_TOKEN"),   # from app.endee.io (None for local)
    collection_name="my_collection",
    fields=[
        {
            "name": "dense",
            "type": "vector",
            "params": {
                "dimension": DIMENSION,
                "space_type": "cosine",
                "precision": "int8",
            },
        },
    ],
    force_recreate=True,
)
```

### Endee Local (Docker)

Run Endee locally — no token needed. See [GitHub](https://github.com/endee-io/endee) for setup.

```bash
docker run -p 8000:8080 -v endee-data:/data endee-oss:latest
```

```python
vector_store = EndeeVectorStore.from_params(
    collection_name="local_collection",
    fields=[
        {"name": "dense", "type": "vector",
         "params": {"dimension": DIMENSION, "space_type": "cosine", "precision": "int8"}},
    ],
    base_url="http://localhost:8000/api/v2",
)
```

### Ingest Documents

```python
from llama_index.core import Document, StorageContext, VectorStoreIndex

documents = [
    Document(text="Python is a high-level programming language known for readability.",
             metadata={"topic": "programming", "language": "python"}),
    Document(text="Machine learning gives systems the ability to learn from data.",
             metadata={"topic": "ai", "field": "ml"}),
    Document(text="Vector databases store embeddings for fast similarity search.",
             metadata={"topic": "database", "type": "vector"}),
]

storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
```

### Reconnect to an Existing Collection

`from_params` auto-detects field names from an existing collection — no data loss:

```python
vector_store = EndeeVectorStore.from_params(
    api_token="your-token",
    collection_name="my_existing_collection",
)
index = VectorStoreIndex.from_vector_store(vector_store)
```

---

## 2. Dense Search

```python
# as_retriever
results = index.as_retriever(similarity_top_k=3).retrieve("Tell me about vector databases")
for node in results:
    print(f"{node.get_score():.4f} | {node.text}")

# Direct VectorStoreQuery
from llama_index.core.vector_stores.types import VectorStoreQuery

q_emb = Settings.embed_model.get_text_embedding("vector databases")
result = vector_store.query(VectorStoreQuery(query_embedding=q_emb, similarity_top_k=3))

# Search tuning
result = vector_store.query(
    VectorStoreQuery(query_embedding=q_emb, similarity_top_k=10),
    ef_search=256,
    prefilter_cardinality_threshold=5_000,
    filter_boost_percentage=20,
)
```

### Loading many documents

```python
from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader("./data").load_data()

storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
print(f"Indexed {len(documents)} documents")
```

---

## 3. Hybrid Search

Create a collection with both `vector` and `sparse` fields. Sparse vectors are auto-encoded via `EndeeModelSparse` (BM25).

```python
from llama_index_endee import EndeeVectorStore, EndeeModelSparse

sparse = EndeeModelSparse()  # Native BM25

hybrid_store = EndeeVectorStore.from_params(
    api_token="your-token",
    collection_name="hybrid_collection",
    fields=[
        {"name": "dense", "type": "vector",
         "params": {"dimension": DIMENSION, "space_type": "cosine", "precision": "int8"}},
        {"name": "sparse", "type": "sparse", "sparse_model": "endee_bm25"},
    ],
    sparse_embedding=sparse,
    force_recreate=True,
)
```

`add()` automatically encodes sparse vectors alongside dense:

```python
from llama_index.core.schema import TextNode

nodes = [
    TextNode(text="The error code is XJ-99-ZQ and it crashed the server.",
             embedding=embed_model.get_text_embedding("The error code is XJ-99-ZQ..."),
             metadata={"type": "error_log"}),
]
hybrid_store.add(nodes)
```

Query with `query_str` to enable sparse matching:

```python
q_emb = embed_model.get_text_embedding("XJ-99-ZQ")
result = hybrid_store.query(VectorStoreQuery(
    query_embedding=q_emb,
    query_str="XJ-99-ZQ",   # used for BM25 sparse encoding
    similarity_top_k=3,
))
```

### RRF Tuning

```python
result = hybrid_store.query(
    query,
    dense_rrf_weight=0.3,    # 0.3 dense + 0.7 sparse
    rrf_rank_constant=30,
)
```

| `dense_rrf_weight` | Effect |
|-------------------|--------|
| `1.0` | Dense only |
| `0.5` | Balanced (default) |
| `0.0` | Sparse only |

---

## 4. Multi-Field & Multi-Vector

### Multiple Dense Fields

Use `fields=` with multiple `vector` entries, then `add_objects()` and `multi_field_search()`:

```python
store = EndeeVectorStore.from_params(
    api_token="your-token",
    collection_name="multi_field",
    fields=[
        {"name": "title",   "type": "vector",
         "params": {"dimension": 384, "space_type": "cosine", "precision": "int8"}},
        {"name": "content", "type": "vector",
         "params": {"dimension": 768, "space_type": "cosine", "precision": "int8"}},
        {"name": "keywords","type": "sparse", "sparse_model": "default"},
    ],
    force_recreate=True,
)

# Upsert with per-field data
store.add_objects([{
    "id": "doc1",
    "meta": {"text": "...", "metadata": {...}},
    "filter": {"topic": "ai"},
    "fields": {
        "title":   title_vec,
        "content": content_vec,
        "keywords": {"indices": [10, 42], "values": [0.9, 0.4]},
    },
}])

# Search + fuse with weighted RRF
from llama_index_endee import rerank

raw = store.multi_field_search(
    fields={
        "title":   {"query": title_vec,   "limit": 20},
        "content": {"query": content_vec, "limit": 20},
    },
)
fused = rerank(raw, limit=10, field_weights={"title": 0.4, "content": 0.6})
```

### Multi-Vector (ColBERT-style)

A `multi_vector` field stores N vectors per object (one per token/chunk):

```python
store = EndeeVectorStore.from_params(
    api_token="your-token",
    collection_name="colbert_collection",
    fields=[
        {"name": "dense",   "type": "vector",
         "params": {"dimension": 384, "space_type": "cosine", "precision": "int8"}},
        {"name": "colbert", "type": "multi_vector",
         "params": {"dimension": 128, "space_type": "cosine",
                    "precision": "float16", "pooling": "mean"}},
    ],
    force_recreate=True,
)

# Upsert: colbert field gets a list of vectors
store.add_objects([{
    "id": "doc1",
    "meta": {"text": "..."},
    "filter": {"topic": "ai"},
    "fields": {
        "dense":   [0.1, 0.2, ...],                  # 1 vector
        "colbert": [[0.1, ...], [0.2, ...], ...],     # N vectors
    },
}])

# Search: query is also a list of vectors
raw = store.multi_field_search(
    fields={"colbert": {"query": [[q1], [q2], [q3]], "limit": 10}},
)

# Or fuse dense + ColBERT
raw = store.multi_field_search(
    fields={
        "dense":   {"query": dense_vec,  "limit": 10},
        "colbert": {"query": token_vecs, "limit": 10},
    },
)
fused = rerank(raw, limit=5, field_weights={"dense": 0.5, "colbert": 0.5})
```

---

## 5. Filters

Pass `filters` to `as_retriever()` or `query()` — they are converted and forwarded to the Endee API.

```python
from llama_index.core.vector_stores.types import MetadataFilters, MetadataFilter, FilterOperator

# EQ — exact match
filters = MetadataFilters(
    filters=[MetadataFilter(key="topic", value="ai", operator=FilterOperator.EQ)]
)
results = index.as_retriever(similarity_top_k=3, filters=filters).retrieve("machine learning")

# IN — match any in list
filters = MetadataFilters(
    filters=[MetadataFilter(key="topic", value=["ai", "database"], operator=FilterOperator.IN)]
)
results = index.as_retriever(similarity_top_k=3, filters=filters).retrieve("vector search")

# Multiple filters (AND logic)
filters = MetadataFilters(filters=[
    MetadataFilter(key="topic", value="database", operator=FilterOperator.EQ),
    MetadataFilter(key="type", value="vector", operator=FilterOperator.EQ),
])
```

Supported operators: `EQ` and `IN`.

### CRUD Operations

```python
# Fetch objects by ID
objects = vector_store.fetch(["node-id-1", "node-id-2"])

# Update filter metadata (no re-embedding)
vector_store.update_filters([
    {"id": "node-id-1", "filter": {"topic": "updated", "priority": 1}},
])

# Delete by ID
vector_store.delete_vector("node-id-1")

# Delete by ref_doc_id filter
vector_store.delete(ref_doc_id="doc-uuid")

# Delete entire collection
vector_store.clear()

# Collection metadata
info = vector_store.describe()

# Direct access to Endee Collection object
collection = vector_store.client
```

---

## 6. RAG Pipeline

```python
from llama_index.core import Settings, VectorStoreIndex
from llama_index.llms.openai import OpenAI

Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)

index = VectorStoreIndex.from_vector_store(vector_store)
query_engine = index.as_query_engine(similarity_top_k=3)

response = query_engine.query("How does vector search work?")
print(response)
```

With metadata filters:

```python
from llama_index.core.vector_stores.types import MetadataFilters, MetadataFilter, FilterOperator

retriever = index.as_retriever(
    similarity_top_k=3,
    filters=MetadataFilters(filters=[
        MetadataFilter(key="topic", value="database", operator=FilterOperator.EQ),
    ]),
)

from llama_index.core.query_engine import RetrieverQueryEngine
query_engine = RetrieverQueryEngine.from_args(retriever=retriever)
response = query_engine.query("Explain vector similarity search")
```

---

## Field Types

| Type | Shape per object | Use case |
|---|---|---|
| `vector` | `[float, ...]` | Standard single-embedding (sentence-transformers, OpenAI) |
| `sparse` | `{indices: [int], values: [float]}` | BM25 / SPLADE keyword matching |
| `multi_vector` | `[[float, ...], ...]` | Token-level (ColBERT), chunk-level embeddings |

## `from_params` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `collection_name` | `str` | *required* | Name of the Endee collection |
| `fields` | `list[dict]` | `None` | Field definitions (same as Python client) |
| `api_token` | `str \| None` | `None` | From [app.endee.io](https://app.endee.io) (None for local) |
| `base_url` | `str \| None` | `None` | API base URL (e.g. `http://localhost:8000/api/v2`) |
| `dimension` | `int \| None` | `None` | Vector dimension (simple mode only, ignored with `fields=`) |
| `space_type` | `str` | `"cosine"` | Distance metric: `cosine`, `l2`, `ip` |
| `precision` | `str` | `"int8"` | Quantisation: `float32`, `float16`, `int16`, `int8`, `binary` |
| `M` | `int \| None` | `None` | HNSW bi-directional links per node |
| `ef_con` | `int \| None` | `None` | HNSW construction quality |
| `sparse_embedding` | `SparseEmbeddings \| None` | `None` | Sparse model for hybrid search |
| `dense_field_name` | `str` | `"dense"` | Primary dense field name |
| `sparse_field_name` | `str` | `"sparse"` | Sparse field name |
| `force_recreate` | `bool` | `False` | Delete and recreate collection if exists |

## Exports

```python
from llama_index_endee import (
    EndeeVectorStore,   # Main vector store class
    SparseEmbeddings,   # ABC for custom sparse models
    SparseVector,       # Sparse vector data class
    EndeeModelSparse,   # BM25 sparse encoder (endee_model)
    Precision,          # Precision enum
    rerank,             # RRF fusion for multi-field results
)
```

## Links

- [Endee Documentation](https://docs.endee.io)
- [Endee Server](https://app.endee.io)
- [LlamaIndex Docs](https://docs.llamaindex.ai/)

## License

MIT License

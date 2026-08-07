# Endee LangChain Integration

LangChain vector store integration for [Endee](https://github.com/endee-io/endee).

For Endee setup, features, and server docs see [docs.endee.io](https://docs.endee.io/quick-start).

**Sections:** [Setup](#1-setup) | [Dense](#2-dense-search) | [Hybrid](#3-hybrid-search) | [Multi-Field](#4-multi-field--multi-vector) | [Filters](#5-filters) | [RAG Chain](#6-rag-chain)

---

## 1. Setup

### Install

```bash
pip install langchain-endee endee endee-model
```

Pick an embedding model:

```bash
# Option A: Local (no API key)
pip install langchain-huggingface sentence-transformers

# Option B: OpenAI
pip install langchain-openai
```

For hybrid search with SPLADE (optional):

```bash
pip install fastembed
```

### Create a Collection

Collections are created with `fields=` — the same pattern as the Python client. Each field has a name, type, and params.

```python
from langchain_endee import EndeeVectorStore, RetrievalMode
from langchain_core.documents import Document

from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
DIMENSION = 384

# Or OpenAI:
# from langchain_openai import OpenAIEmbeddings
# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# DIMENSION = 1536

# Dense-only collection (single vector field)
vector_store = EndeeVectorStore(
    embedding=embeddings,
    api_token="your-token",       # from app.endee.io (None for local)
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
vector_store = EndeeVectorStore(
    embedding=embeddings,
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
documents = [
    Document(
        page_content="Python is a high-level programming language known for readability.",
        metadata={"topic": "programming", "language": "python"},
    ),
    Document(
        page_content="Machine learning gives systems the ability to learn from data.",
        metadata={"topic": "ai", "field": "ml"},
    ),
    Document(
        page_content="Vector databases store embeddings for fast similarity search.",
        metadata={"topic": "database", "type": "vector"},
    ),
]
```

#### `add_texts()` — insert into an existing store

```python
ids = vector_store.add_texts(
    texts=[doc.page_content for doc in documents],
    metadatas=[doc.metadata for doc in documents],
)
```

#### `from_texts()` / `from_documents()` — create + insert in one call

```python
store = EndeeVectorStore.from_texts(
    texts=["Python is great.", "Rust is fast."],
    metadatas=[{"lang": "python"}, {"lang": "rust"}],
    embedding=embeddings,
    api_token="your-token",
    collection_name="my_collection",
    dimension=DIMENSION,
    force_recreate=True,
)
```

### Reconnect to an Existing Collection

```python
vector_store = EndeeVectorStore.from_existing_collection(
    collection_name="my_collection",
    embedding=embeddings,
    api_token="your-token",
)
```

---

## 2. Dense Search

```python
# similarity_search
results = vector_store.similarity_search(query="How does RAG work?", k=3)

# similarity_search_with_score
scored = vector_store.similarity_search_with_score(query="neural networks", k=3)

# similarity_search_by_object
query_vec = embeddings.embed_query("programming language safety")
results = vector_store.similarity_search_by_object(embedding=query_vec, k=2)

# Search tuning
results = vector_store.similarity_search(
    query="vector search", k=10, ef=256,
    filter=[{"topic": {"$eq": "database"}}],
    prefilter_cardinality_threshold=5_000,
    filter_boost_percentage=20,
)

# as_retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
docs = retriever.invoke("What are vector databases used for?")
```

---

## 3. Hybrid Search

Create a collection with both `vector` and `sparse` fields:

```python
from langchain_endee import EndeeModelSparse

sparse = EndeeModelSparse()  # Native BM25

hybrid_store = EndeeVectorStore(
    embedding=embeddings,
    api_token="your-token",
    collection_name="hybrid_collection",
    fields=[
        {"name": "dense", "type": "vector",
         "params": {"dimension": DIMENSION, "space_type": "cosine", "precision": "int8"}},
        {"name": "sparse", "type": "sparse", "sparse_model": "default"},
    ],
    retrieval_mode=RetrievalMode.HYBRID,
    sparse_embedding=sparse,
    force_recreate=True,
)
```

All search methods automatically use both dense and sparse:

```python
results = hybrid_store.similarity_search("vector database semantic search", k=3)
```

### RRF Tuning

```python
results = hybrid_store.similarity_search_with_score(
    query="vector database semantic search",
    k=3,
    rrf_rank_constant=60,
    dense_rrf_weight=0.7,
)
```

---

## 4. Multi-Field & Multi-Vector

### Multiple Dense Fields

Use `fields=` with multiple `vector` entries, then `add_objects()` and `multi_field_search_with_rerank()`:

```python
store = EndeeVectorStore(
    embedding=embeddings,
    api_token="your-token",
    collection_name="multi_field",
    fields=[
        {"name": "title",   "type": "vector",
         "params": {"dimension": 384, "space_type": "cosine", "precision": "int8"}},
        {"name": "content", "type": "vector",
         "params": {"dimension": 768, "space_type": "cosine", "precision": "int8"}},
        {"name": "keywords","type": "sparse", "sparse_model": "default"},
    ],
    dense_field_name="title",   # primary field for similarity_search()
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
results = store.multi_field_search_with_rerank(
    fields={
        "title":   {"query": title_vec,   "limit": 20},
        "content": {"query": content_vec, "limit": 20},
    },
    limit=10,
    field_weights={"title": 0.4, "content": 0.6},
)
```

### Multi-Vector (ColBERT-style)

A `multi_vector` field stores N vectors per object (one per token/chunk):

```python
store = EndeeVectorStore(
    embedding=embeddings,
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
results = store.multi_field_search_with_rerank(
    fields={
        "dense":   {"query": dense_vec,  "limit": 10},
        "colbert": {"query": token_vecs, "limit": 10},
    },
    limit=5,
    field_weights={"dense": 0.5, "colbert": 0.5},
)
```

### Manual Rerank

```python
from langchain_endee import rerank

raw = store.multi_field_search(fields={...})
fused = rerank(raw, limit=10, field_weights={"title": 0.3, "content": 0.7}, rrf_k=60)
```

---

## 5. Filters

Pass filters as a list of dicts (AND logic). See [Endee docs](https://docs.endee.io) for filter operators (`$eq`, `$in`, `$range`).

```python
# $eq
results = vector_store.similarity_search(
    query="learning from data", k=5,
    filter=[{"topic": {"$eq": "ai"}}],
)

# Multiple filters (AND)
results = vector_store.similarity_search(
    query="safe languages", k=5,
    filter=[
        {"topic": {"$eq": "programming"}},
        {"language": {"$in": ["python", "rust"]}},
    ],
)

# Retriever with filters
retriever = vector_store.as_retriever(
    search_kwargs={"k": 3, "filter": [{"topic": {"$eq": "ai"}}]},
)

# get_by_ids / update_filters / delete
docs = vector_store.get_by_ids(["id1", "id2"])

vector_store.update_filters([
    {"id": "id1", "filter": {"topic": "updated", "priority": 1}},
])

vector_store.delete(ids=["id1", "id2"])
vector_store.delete(filter=[{"status": {"$eq": "expired"}}])
```

---

## 6. RAG Chain

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


retriever = vector_store.as_retriever(search_kwargs={"k": 3})
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_template(
    "Answer the question based only on the context below.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke("How does vector search work?")
print(answer)
```

---

## Field Types

| Type | Shape per object | Use case |
|---|---|---|
| `vector` | `[float, ...]` | Standard single-embedding (sentence-transformers, OpenAI) |
| `sparse` | `{indices: [int], values: [float]}` | BM25 / SPLADE keyword matching |
| `multi_vector` | `[[float, ...], ...]` | Token-level (ColBERT), chunk-level embeddings |

## Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `embedding` | `Embeddings` | *required* | LangChain embedding function |
| `collection_name` | `str` | *required* | Name of the Endee collection |
| `fields` | `list[dict]` | `None` | Field definitions (same as Python client) |
| `api_token` | `str \| None` | `None` | From [app.endee.io](https://app.endee.io) (None for local) |
| `base_url` | `str \| None` | `None` | API base URL (e.g. `http://localhost:8000/api/v2`) |
| `retrieval_mode` | `RetrievalMode` | `DENSE` | `DENSE` or `HYBRID` |
| `sparse_embedding` | `SparseEmbeddings \| None` | `None` | Sparse model for hybrid search |
| `dense_field_name` | `str` | `"dense"` | Primary dense field for `similarity_search()` |
| `sparse_field_name` | `str` | `"sparse"` | Sparse field for hybrid search |
| `force_recreate` | `bool` | `False` | Delete and recreate collection if exists |
| `validate_collection_config` | `bool` | `True` | Validate dimension/config on connect |

## Links

- [Endee Documentation](https://docs.endee.io)
- [Endee GitHub](https://github.com/endee-io/endee)
- [Endee Server](https://app.endee.io)

## License

MIT License

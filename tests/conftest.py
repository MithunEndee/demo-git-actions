"""Shared pytest fixtures for the endee-llamaindex test suite.

Provides:
- `MockEndee`/`MockEndeeCollection`: a mock backend (unit tests).
- `mock_endee_client`: patches `llama_index_endee.base.Endee` with a
  factory that returns a fresh `MockEndee()` per test.
- `fake_embedder`/`fake_embed_model`: a deterministic, dependency-free
  embedding callable and its `BaseEmbedding` wrapper.
- `live_client`/`store_factory`: a real `endee.Endee` client and a factory
  for live `EndeeVectorStore` instances, skipping if `ENDEE_API_TOKEN` is
  unset (integration tests).
- `sample_documents`: a shared set of `Document` objects with rich
  metadata, used by filter tests.
"""

from __future__ import annotations

import hashlib
import os
import re
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from endee.exceptions import NotFoundException
from llama_index.core import Document
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import PrivateAttr


def _load_dotenv():
    try:
        from dotenv import load_dotenv

        root = Path(__file__).resolve().parents[1]  # project root
        load_dotenv(root / ".env")
    except ImportError:
        pass  # python-dotenv not installed


_load_dotenv()


# ═══════════════════════════════════════════════════════════════════════════
# Integration test helpers (collection naming + cleanup)
# ═══════════════════════════════════════════════════════════════════════════

_UID_PREFIX = "llamaindex_test"
_STALE_PATTERN = re.compile(rf"^{_UID_PREFIX}_[0-9a-f]{{10}}$")

ALL_PRECISIONS = ["float32", "float16", "int8", "int16"]
ALL_SPACE_TYPES = ["cosine", "l2", "ip"]


def uid(prefix: str = _UID_PREFIX) -> str:
    """Unique collection name for integration tests (fits the 48-char limit)."""
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def safe_delete(client, name: str) -> None:
    """Delete a collection silently - used in fixture teardown."""
    try:
        client.delete_collection(name)
    except Exception:
        pass


class MockEndeeCollection:
    """Mock collection mimicking the Endee Collection API."""

    def __init__(self, name, fields):
        self.name = name
        self.fields = fields
        self._store = {}

    @staticmethod
    def _op_matches(op, field_value, val):
        if op == "$eq":
            if isinstance(field_value, list):
                return val in field_value
            return field_value == val
        if op == "$in":
            values = val if isinstance(val, list) else [val]
            if isinstance(field_value, list):
                return any(v in field_value for v in values)
            return field_value in values
        raise ValueError(f"MockEndeeCollection does not support operator {op!r}")

    def _matches_filter(self, entry, filter_list):
        entry_filter = entry.get("filter", {})
        for f in filter_list:
            for field_name, ops in f.items():
                field_value = entry_filter.get(field_name)
                for op, val in ops.items():
                    if not self._op_matches(op, field_value, val):
                        return False
        return True

    def upsert(self, objects):
        for entry in objects:
            self._store[entry["id"]] = entry
        return {"upserted": len(objects)}

    def search(
        self,
        fields,
        filter=None,
        ef_search=128,
        prefilter_cardinality_threshold=None,
        filter_boost_percentage=None,
    ):
        per_field_results = {}
        for field_name, field_query in fields.items():
            query_data = field_query.get("query")
            limit = field_query.get("limit", 10)

            if isinstance(query_data, dict) and "indices" in query_data:
                # No BM25/sparse scoring here: always no hits for a sparse
                # field. Assert on call args instead for ranking behavior.
                per_field_results[field_name] = []
                continue

            q = np.array(query_data, dtype=float)
            q_norm = np.linalg.norm(q)

            hits = []
            for vid, entry in self._store.items():
                if filter and not self._matches_filter(entry, filter):
                    continue
                vector = entry.get("fields", {}).get(field_name)
                if vector is None:
                    continue
                v = np.array(vector, dtype=float)
                v_norm = np.linalg.norm(v)
                similarity = (
                    0.0
                    if q_norm == 0 or v_norm == 0
                    else float(np.dot(q, v) / (q_norm * v_norm))
                )
                hits.append(
                    {
                        "id": vid,
                        "similarity": similarity,
                        "meta": entry.get("meta", {}),
                        "filter": entry.get("filter", {}),
                    }
                )
            hits.sort(key=lambda h: h["similarity"], reverse=True)
            per_field_results[field_name] = hits[:limit]
        return {"results": per_field_results}

    def describe(self):
        return {
            "name": self.name,
            "fields": self.fields,
            "num_objects": len(self._store),
        }

    def update_filters(self, updates):
        for upd in updates:
            vid = upd["id"]
            if vid in self._store:
                self._store[vid]["filter"] = upd["filter"]
        return {"updated": len(updates)}

    def get_objects(self, ids):
        return [self._store[vid] for vid in ids if vid in self._store]

    def delete_by_filter(self, filter_list):
        to_delete = [
            vid
            for vid, entry in self._store.items()
            if self._matches_filter(entry, filter_list)
        ]
        for vid in to_delete:
            del self._store[vid]
        return {"deleted": len(to_delete)}

    def delete_object(self, object_id):
        if object_id not in self._store:
            raise NotFoundException(f"Object '{object_id}' not found")
        del self._store[object_id]
        return {"deleted": 1}


class MockEndee:
    """Mock Endee client mimicking the real SDK without network calls."""

    def __init__(self, token=None):
        self.token = token
        self._collections = {}

    def list_collections(self):
        return [{"name": n} for n in self._collections]

    def get_collection(self, name):
        if name not in self._collections:
            raise NotFoundException(f"Collection '{name}' not found")
        return self._collections[name]

    def create_collection(self, name, fields):
        self._collections[name] = MockEndeeCollection(name=name, fields=fields)
        return {"created": name}

    def delete_collection(self, name):
        self._collections.pop(name, None)
        return {"deleted": name}

    def set_base_url(self, url):
        pass

    def close_session(self):
        pass

    def close_client(self):
        pass


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def mock_endee_client():
    """Patch llama_index_endee.base.Endee with a fresh MockEndee per test."""
    fake = MockEndee()
    with patch("llama_index_endee.base.Endee", side_effect=lambda *a, **kw: fake):
        yield fake


@pytest.fixture
def fake_embedder():
    """Deterministic hash-based embedding function, dimension 16, no ML download."""
    dim = 16

    def _embed(text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [(digest[i % len(digest)] / 255.0) * 2.0 - 1.0 for i in range(dim)]

    _embed.dim = dim
    return _embed


class FakeEmbedModel(BaseEmbedding):
    """Wraps the fake_embedder callable in a real BaseEmbedding subclass."""

    # VectorStoreIndex.from_documents() calls resolve_embed_model(), which
    # does a strict isinstance(embed_model, BaseEmbedding) check, so a
    # duck-typed object with the right method names is not enough here.

    _embed_fn: Any = PrivateAttr()

    def __init__(self, embed_fn, **kwargs):
        super().__init__(**kwargs)
        self._embed_fn = embed_fn

    def _get_text_embedding(self, text):
        return self._embed_fn(text)

    def _get_query_embedding(self, query):
        return self._embed_fn(query)

    async def _aget_text_embedding(self, text):
        return self._embed_fn(text)

    async def _aget_query_embedding(self, query):
        return self._embed_fn(query)


@pytest.fixture
def fake_embed_model(fake_embedder):
    """Real BaseEmbedding subclass wrapping fake_embedder, for direct index use."""
    return FakeEmbedModel(fake_embedder)


@pytest.fixture(scope="session", autouse=True)
def _cleanup_stale_collections():
    """Removes leftover collections from an interrupted run; no-op without a token."""
    token = os.environ.get("ENDEE_API_TOKEN", "")
    if not token:
        yield
        return
    try:
        from endee import Endee

        base_url = os.environ.get("ENDEE_BASE_URL")
        client = Endee(token=token)
        if base_url:
            client.set_base_url(base_url)
        for coll in client.list_collections():
            name = coll.get("name") if isinstance(coll, dict) else coll
            if name and _STALE_PATTERN.match(name):
                safe_delete(client, name)
    except Exception:
        pass
    yield


@pytest.fixture(scope="session")
def live_client():
    """Real endee.Endee client fixture; skips if no valid token is set."""
    from endee import Endee

    token = os.environ.get("ENDEE_API_TOKEN", "")
    if not token:
        pytest.skip("ENDEE_API_TOKEN not set; skipping integration test")

    client = Endee(token=token)
    base_url = os.environ.get("ENDEE_BASE_URL")
    if base_url:
        client.set_base_url(base_url)

    try:
        client.list_collections()
    except Exception as e:
        pytest.skip(
            f"ENDEE_API_TOKEN is invalid or expired, or service is "
            f"unreachable: {e}. Update it to run integration tests."
        )

    return client


@pytest.fixture
def store_factory(live_client):
    """Creates live-server EndeeVectorStore instances, tracking names for cleanup."""
    from llama_index_endee.base import EndeeVectorStore

    created = []

    def _make(**kwargs):
        name = kwargs.pop("collection_name", uid())
        created.append(name)
        return EndeeVectorStore.from_params(
            endee_client=live_client, collection_name=name, **kwargs
        )

    yield _make

    for name in created:
        safe_delete(live_client, name)


@pytest.fixture
def sample_documents():
    """Rich category/language/difficulty metadata Document objects."""
    return [
        Document(
            text=(
                "Python is a high-level, interpreted programming language "
                "known for its readability and simplicity."
            ),
            metadata={
                "category": "programming",
                "language": "python",
                "difficulty": "beginner",
            },
        ),
        Document(
            text=(
                "JavaScript is a versatile language for web development "
                "with advanced features like async/await."
            ),
            metadata={
                "category": "programming",
                "language": "javascript",
                "difficulty": "intermediate",
            },
        ),
        Document(
            text=(
                "Rust provides memory safety without garbage collection "
                "using ownership system."
            ),
            metadata={
                "category": "programming",
                "language": "rust",
                "difficulty": "advanced",
            },
        ),
        Document(
            text=(
                "Machine learning algorithms learn patterns from data "
                "to make predictions."
            ),
            metadata={
                "category": "ai",
                "field": "machine_learning",
                "difficulty": "intermediate",
            },
        ),
        Document(
            text=(
                "Deep learning uses neural networks with multiple layers "
                "for complex pattern recognition."
            ),
            metadata={
                "category": "ai",
                "field": "deep_learning",
                "difficulty": "advanced",
            },
        ),
        Document(
            text=(
                "Vector databases optimize similarity search for high-dimensional data."
            ),
            metadata={
                "category": "database",
                "type": "vector",
                "feature": "similarity_search",
            },
        ),
        Document(
            text=(
                "Time-series databases are optimized for sequential "
                "temporal data storage."
            ),
            metadata={
                "category": "database",
                "type": "time_series",
                "feature": "temporal_storage",
            },
        ),
        # category stays a string: the server rejects list-valued filter fields.
        # languages/technologies are safe as lists since they're not filterable.
        Document(
            text="Building a real-time ML pipeline with Python and Vector DB.",
            metadata={
                "category": "programming",
                "languages": ["python"],
                "technologies": ["ml", "vector_db"],
                "difficulty": "advanced",
            },
        ),
        Document(
            text="Implementing secure encryption in distributed databases.",
            metadata={
                "category": "database",
                "field": "cryptography",
                "difficulty": "advanced",
                "feature": "encryption",
            },
        ),
    ]

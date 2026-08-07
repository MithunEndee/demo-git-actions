"""Shared pytest fixtures for the endee-langchain test suite.

Provides:
- `MockEndee`/`MockEndeeCollection`: a mock backend (unit tests).
- `mock_endee_client`: patches `langchain_endee.vectorstores.EndeeClient`
  with a factory that returns a fresh `MockEndee()` per test.
- `fake_embedder`: a deterministic hash-based dense `Embeddings`, with no
  ML model.
- `fake_sparse_embedding`/`fake_raw_sparse_model`: deterministic sparse
  embedding fakes.
- `live_client`: a real `endee.Endee` client that skips if
  `ENDEE_API_TOKEN` is unset (integration tests).
"""

from __future__ import annotations

import os
import re
import uuid

import numpy as np
import pytest
from endee.exceptions import NotFoundException
from langchain_core.embeddings import Embeddings

from langchain_endee.sparse_embeddings import SparseEmbeddings, SparseVector

# ═══════════════════════════════════════════════════════════════════════════
# Integration-test helpers (shared collection naming + cleanup)
# ═══════════════════════════════════════════════════════════════════════════


_UID_PREFIX = "langchain_test"
_STALE_PATTERN = re.compile(rf"^{_UID_PREFIX}_[0-9a-f]{{10}}$")


def uid(prefix: str = _UID_PREFIX) -> str:
    """Unique collection name for integration tests (fits the 48-char limit)."""
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def safe_delete(client, name: str) -> None:
    """Delete a collection silently, for use in fixture teardown."""
    try:
        client.delete_collection(name)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════
# Mock Endee backend (unit tests)
# ═══════════════════════════════════════════════════════════════════════════


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
                # This fake has no BM25 scoring, so sparse fields always
                # return no hits; assert on call args instead of results.
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

    def delete_by_filter(self, filter):  # noqa: A002
        to_delete = [
            vid
            for vid, entry in self._store.items()
            if self._matches_filter(entry, filter)
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
# Shared collection field and document data, used by unit and integration tests
# ═══════════════════════════════════════════════════════════════════════════

DIMENSION = 16

ALL_PRECISIONS = ["float32", "float16", "int16", "int8", "binary"]
ALL_SPACE_TYPES = ["cosine", "l2", "ip"]

DENSE_FIELD = {
    "name": "dense",
    "type": "vector",
    "params": {"dimension": DIMENSION, "space_type": "cosine", "precision": "int8"},
}

TEXTS = [
    "Python is a high-level programming language.",
    "Rust is a systems language focused on safety.",
    "Machine learning enables systems to learn from data.",
    "Deep learning uses neural networks with multiple layers.",
    "Vector databases store embeddings for similarity search.",
]

METADATAS = [
    {"category": "programming", "language": "python", "difficulty": "beginner"},
    {"category": "programming", "language": "rust", "difficulty": "advanced"},
    {"category": "ai", "field": "ml", "difficulty": "intermediate"},
    {"category": "ai", "field": "dl", "difficulty": "advanced"},
    {"category": "database", "type": "vector", "difficulty": "intermediate"},
]


# ═══════════════════════════════════════════════════════════════════════════
# Deterministic fake embedders (no ML model download)
# ═══════════════════════════════════════════════════════════════════════════


class FakeEmbeddings(Embeddings):
    """Deterministic hash-based embeddings, no ML model download."""

    def __init__(self, dimension: int = DIMENSION):
        self.dimension = dimension

    def embed_documents(self, texts):
        return [
            [(hash(t) + i) % 100 / 100.0 for i in range(self.dimension)] for t in texts
        ]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


class FakeSparseEmbeddings(SparseEmbeddings):
    """Deterministic sparse embeddings, no ML model download."""

    def embed_documents(self, texts):
        vecs = []
        for t in texts:
            indices = sorted(set(hash(w) % 1000 for w in t.split()[:10]))
            values = [1.0 / (i + 1) for i in range(len(indices))]
            vecs.append(SparseVector(indices=indices, values=values))
        return vecs

    def embed_query(self, text):
        return self.embed_documents([text])[0]


class _TolistArray(list):
    """A list that also exposes .tolist(), like fastembed/endee_model's outputs."""

    def tolist(self):
        return list(self)


def make_fake_sparse_result(indices, values):
    """Build an object exposing `.indices.tolist()` / `.values.tolist()`."""

    class _Result:
        def __init__(self):
            self.indices = _TolistArray(indices)
            self.values = _TolistArray(values)

    return _Result()


class FakeRawSparseModel:
    """Fastembed-style fake (.embed/.query_embed), not a SparseEmbeddings subclass."""

    def embed(self, texts):
        for t in texts:
            indices = sorted(set(hash(w) % 1000 for w in t.split()[:10]))
            values = [1.0 / (i + 1) for i in range(len(indices))]
            yield make_fake_sparse_result(indices, values)

    def query_embed(self, text):
        yield from self.embed([text])


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def mock_endee_client(monkeypatch):
    """Patch EndeeClient with a factory returning a fresh MockEndee() per test."""
    # Stores built within the same test share this instance; other tests get
    # an isolated one.
    instance = MockEndee()
    monkeypatch.setattr(
        "langchain_endee.vectorstores.EndeeClient",
        lambda *args, **kwargs: instance,
    )
    return instance


@pytest.fixture(scope="session")
def fake_embedder():
    """Session-scoped since it's stateless and integration fixtures need this scope."""
    return FakeEmbeddings()


@pytest.fixture(scope="session")
def fake_sparse_embedding():
    """Session-scoped for the same reason as fake_embedder above."""
    return FakeSparseEmbeddings()


@pytest.fixture
def fake_raw_sparse_model():
    return FakeRawSparseModel()


@pytest.fixture(scope="session", autouse=True)
def _cleanup_stale_collections():
    """Removes leftover collections from a prior interrupted test run."""
    # No-op if no token is set, since unit tests must still run without a live server.
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
        pass  # server unreachable; integration tests will skip anyway via live_client
    yield


@pytest.fixture(scope="session")
def live_client():
    """Real endee.Endee client; skips if ENDEE_API_TOKEN is unset."""
    token = os.environ.get("ENDEE_API_TOKEN", "")
    if not token:
        pytest.skip("ENDEE_API_TOKEN not set; skipping integration test")

    from endee import Endee

    base_url = os.environ.get("ENDEE_BASE_URL")
    client = Endee(token=token)
    if base_url:
        client.set_base_url(base_url)

    yield client

    # Safety net for tests that build a collection without a cleanup fixture.
    for coll in client.list_collections():
        name = coll.get("name") if isinstance(coll, dict) else coll
        if name and _STALE_PATTERN.match(name):
            safe_delete(client, name)

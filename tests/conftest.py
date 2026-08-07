"""Shared pytest fixtures for the endee-crewai test suite.

Provides:
- `MockEndee`/`MockEndeeCollection`: a mock backend (unit tests).
- `mock_endee_client`: patches `crewai_endee.vector_store.EndeeClient`
  with a factory that returns a fresh `MockEndee()` per test.
- `fake_embedder`: a deterministic, dependency-free embedding callable.
- `make_store`: builds an `EndeeVectorStore` against the mocked client.
- `live_client`: a real `endee.Endee` client that skips if
  `ENDEE_API_TOKEN` is unset (integration tests).
"""

from __future__ import annotations

import hashlib
import os
import re
import uuid

import numpy as np
import pytest
from crewai.rag.embeddings.providers.custom.embedding_callable import (
    CustomEmbeddingFunction,
)
from endee import Endee
from endee.exceptions import NotFoundException

from crewai_endee.vector_store import EndeeVectorStore

ALL_PRECISIONS = ["float32", "float16", "int16", "int8", "binary"]
ALL_SPACE_TYPES = ["cosine", "l2", "ip"]

# ═══════════════════════════════════════════════════════════════════════════
# Deterministic embedder for integration tests
# ═══════════════════════════════════════════════════════════════════════════

LIVE_DIM = 384


class _DeterministicEmbeddingFunction(CustomEmbeddingFunction):
    """Deterministic embedder that skips downloading a real ML model."""

    def __call__(self, input):
        vectors = []
        for text in input:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            repeated = (digest * (LIVE_DIM // len(digest) + 1))[:LIVE_DIM]
            vectors.append([b / 255.0 for b in repeated])
        return vectors


LIVE_EMBEDDER_CONFIG = {
    "provider": "custom",
    "config": {"embedding_callable": _DeterministicEmbeddingFunction},
}


# ═══════════════════════════════════════════════════════════════════════════
# Integration-test collection naming & cleanup helpers
# ═══════════════════════════════════════════════════════════════════════════


_UID_PREFIX = "crewai_test"
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
                # Sparse query: this fake has no BM25/similarity scoring, so
                # it always returns no hits. Assert on call args instead.
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
def mock_endee_client(monkeypatch):
    """Patch EndeeClient with a fresh MockEndee factory so tests don't share state."""
    monkeypatch.setattr("crewai_endee.vector_store.EndeeClient", MockEndee)
    return MockEndee


@pytest.fixture
def fake_embedder():
    """Deterministic embedding callable: embed(texts) returns one vector per text."""
    dim = 16

    def _vector_for(text: str):
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [b / 255.0 for b in digest[:dim]]

    def embed(texts):
        return [_vector_for(t) for t in texts]

    return embed


@pytest.fixture
def make_store(monkeypatch, mock_endee_client, fake_embedder):
    """Factory for an EndeeVectorStore with defaults callers can override via kwargs."""
    monkeypatch.setattr(
        "crewai_endee.vector_store.build_embedder", lambda cfg: fake_embedder
    )

    def _make(**kwargs):
        defaults = dict(
            type="test_collection",
            embedder_config={"provider": "fake", "config": {}},
            fields=[
                {
                    "name": "dense",
                    "type": "vector",
                    "params": {
                        "dimension": 16,
                        "space_type": "cosine",
                        "precision": "float32",
                    },
                },
            ],
        )
        defaults.update(kwargs)
        return EndeeVectorStore(**defaults)

    return _make


@pytest.fixture(scope="session", autouse=True)
def _cleanup_stale_collections():
    """Removes leftover collections from a prior interrupted test run."""
    # No-op if no token is set, since unit tests must still run without a live server.
    token = os.environ.get("ENDEE_API_TOKEN")
    if not token:
        yield
        return
    try:
        base_url = os.environ.get("ENDEE_BASE_URL")
        client = Endee(token=token)
        if base_url:
            client.set_base_url(base_url)
        existing = client.list_collections()
        for coll in existing:
            name = coll.get("name") if isinstance(coll, dict) else coll
            if name and _STALE_PATTERN.match(name):
                safe_delete(client, name)
    except Exception:
        pass  # server unreachable; integration tests will skip anyway via live_client
    yield


@pytest.fixture
def live_client():
    """Real endee.Endee client. Skips if ENDEE_API_TOKEN is unset."""
    token = os.getenv("ENDEE_API_TOKEN")
    if not token:
        pytest.skip("ENDEE_API_TOKEN not set, skipping integration test")
    base_url = os.getenv("ENDEE_BASE_URL")
    client = Endee(token=token)
    if base_url:
        client.set_base_url(base_url)
    return client

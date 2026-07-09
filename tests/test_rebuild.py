"""
Tests for Collection.rebuild() and Collection.rebuild_status().

rebuild() is ASYNC on the server: it returns 202 immediately with
status="in_progress". Tests that need to assert on completed state
use wait_for_rebuild(), which polls rebuild_status() until done.
"""

import time

import pytest
from helpers import (
    DENSE_FIELD,
    MV_FIELD,
    dense_vec,
    make_mv_field,
    multi_vec,
    safe_delete,
    uid,
)


# -- polling helper ------------------------------------------------------------


def wait_for_rebuild(collection, timeout: int = 120) -> dict:
    """Poll rebuild_status() until status is 'completed' or 'idle'."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = collection.rebuild_status()
        s = status.get("status")
        if s in ("completed", "idle"):
            return status
        if s == "failed":
            raise RuntimeError(f"Rebuild failed: {status.get('error')}")
        time.sleep(0.5)
    raise TimeoutError(f"Rebuild did not complete within {timeout}s")


# -- rebuild - initial response ------------------------------------------------


def test_rebuild_returns_dict(populated_collection):
    """rebuild() must return a dict."""
    _, collection = populated_collection
    result = collection.rebuild([{"field": DENSE_FIELD, "M": 8, "ef_con": 64}])
    assert isinstance(result, dict)
    wait_for_rebuild(collection)


def test_rebuild_initial_status_is_in_progress(populated_collection):
    """rebuild() response must contain status='in_progress'."""
    _, collection = populated_collection
    result = collection.rebuild([{"field": DENSE_FIELD, "M": 8, "ef_con": 64}])
    assert result.get("status") == "in_progress", (
        f"Expected status='in_progress', got: {result}"
    )
    wait_for_rebuild(collection)


def test_rebuild_response_has_total_objects(populated_collection):
    """rebuild() response must contain total_objects count."""
    _, collection = populated_collection
    result = collection.rebuild([{"field": DENSE_FIELD, "M": 8, "ef_con": 64}])
    assert "total_objects" in result, f"Missing 'total_objects' in: {result}"
    assert isinstance(result["total_objects"], int)
    wait_for_rebuild(collection)


def test_rebuild_response_has_new_config(populated_collection):
    """rebuild() response must contain new_config dict."""
    _, collection = populated_collection
    result = collection.rebuild([{"field": DENSE_FIELD, "M": 8, "ef_con": 64}])
    assert "new_config" in result, f"Missing 'new_config' in: {result}"
    assert isinstance(result["new_config"], dict)
    wait_for_rebuild(collection)


def test_rebuild_response_has_previous_config(populated_collection):
    """rebuild() response must contain previous_config dict."""
    _, collection = populated_collection
    result = collection.rebuild([{"field": DENSE_FIELD, "M": 8, "ef_con": 64}])
    assert "previous_config" in result, f"Missing 'previous_config' in: {result}"
    wait_for_rebuild(collection)


def test_rebuild_with_custom_m(populated_collection):
    """rebuild() with a custom M parameter must succeed."""
    _, collection = populated_collection
    result = collection.rebuild([{"field": DENSE_FIELD, "M": 8}])
    assert isinstance(result, dict)
    wait_for_rebuild(collection)


def test_rebuild_custom_m_reflected_in_new_config(populated_collection):
    """rebuild() with M=8 must reflect M=8 in new_config."""
    _, collection = populated_collection
    result = collection.rebuild([{"field": DENSE_FIELD, "M": 8}])
    new_cfg = result.get("new_config", {})
    assert new_cfg.get("M") == 8, f"Expected M=8 in new_config: {new_cfg}"
    wait_for_rebuild(collection)


def test_rebuild_with_custom_ef_con(populated_collection):
    """rebuild() with a custom ef_con parameter must succeed."""
    _, collection = populated_collection
    result = collection.rebuild([{"field": DENSE_FIELD, "ef_con": 64}])
    assert isinstance(result, dict)
    wait_for_rebuild(collection)


def test_rebuild_with_both_hnsw_params(populated_collection):
    """rebuild() with both M and ef_con must succeed."""
    _, collection = populated_collection
    result = collection.rebuild([{"field": DENSE_FIELD, "M": 8, "ef_con": 64}])
    assert isinstance(result, dict)
    wait_for_rebuild(collection)


@pytest.mark.parametrize("m,ef_con", [(4, 32), (8, 64), (32, 256)])
def test_rebuild_various_hnsw_params_accepted(populated_collection, m, ef_con):
    """rebuild() must accept various valid HNSW parameter combinations."""
    _, collection = populated_collection
    result = collection.rebuild([{"field": DENSE_FIELD, "M": m, "ef_con": ef_con}])
    assert isinstance(result, dict)
    wait_for_rebuild(collection)


# -- rebuild - collection still usable during/after rebuild -------------------


def test_rebuild_collection_still_searchable_during(populated_collection):
    """After triggering rebuild(), search must still return results immediately."""
    _, collection = populated_collection
    collection.rebuild([{"field": DENSE_FIELD, "M": 8, "ef_con": 64}])
    results = collection.search(
        fields={DENSE_FIELD: {"query": dense_vec(), "limit": 5}}
    )["results"][DENSE_FIELD]
    assert isinstance(results, list)
    assert len(results) > 0
    wait_for_rebuild(collection)


def test_rebuild_collection_searchable_after_completion(populated_collection):
    """After rebuild() completes, search must still return results."""
    _, collection = populated_collection
    collection.rebuild([{"field": DENSE_FIELD, "M": 8, "ef_con": 64}])
    wait_for_rebuild(collection)
    results = collection.search(
        fields={DENSE_FIELD: {"query": dense_vec(), "limit": 5}}
    )["results"][DENSE_FIELD]
    assert isinstance(results, list)
    assert len(results) > 0


# -- rebuild_status - response shape ------------------------------------------


def test_rebuild_status_returns_dict(populated_collection):
    """rebuild_status() must return a dict."""
    _, collection = populated_collection
    result = collection.rebuild_status()
    assert isinstance(result, dict)


def test_rebuild_status_has_status_key(populated_collection):
    """rebuild_status() must contain a 'status' key."""
    _, collection = populated_collection
    result = collection.rebuild_status()
    assert "status" in result, f"Missing 'status' key in: {result}"


def test_rebuild_status_valid_values(populated_collection):
    """rebuild_status()['status'] must be one of the known values."""
    _, collection = populated_collection
    result = collection.rebuild_status()
    assert result["status"] in ("idle", "in_progress", "completed", "failed"), (
        f"Unexpected status value: {result['status']}"
    )


def test_rebuild_status_on_empty_collection_is_idle(empty_collection):
    """rebuild_status() on a fresh collection must be 'idle' or 'completed'."""
    _, collection = empty_collection
    result = collection.rebuild_status()
    assert result.get("status") in ("idle", "completed"), (
        f"Expected 'idle' or 'completed' on fresh collection, got: {result}"
    )


def test_rebuild_status_in_progress_after_trigger(populated_collection):
    """rebuild_status() called immediately after rebuild() must report 'in_progress'."""
    _, collection = populated_collection
    collection.rebuild([{"field": DENSE_FIELD, "M": 8, "ef_con": 64}])
    status = collection.rebuild_status()
    assert status.get("status") in ("in_progress", "completed"), (
        f"Unexpected status right after rebuild(): {status}"
    )
    wait_for_rebuild(collection)


def test_rebuild_status_has_objects_processed(populated_collection):
    """rebuild_status() during/after rebuild must have 'objects_processed' key."""
    _, collection = populated_collection
    collection.rebuild([{"field": DENSE_FIELD, "M": 8, "ef_con": 64}])
    status = collection.rebuild_status()
    assert "objects_processed" in status, f"Missing 'objects_processed' in: {status}"
    assert isinstance(status["objects_processed"], int)
    wait_for_rebuild(collection)


def test_rebuild_status_has_total_objects(populated_collection):
    """rebuild_status() must have 'total_objects' key."""
    _, collection = populated_collection
    collection.rebuild([{"field": DENSE_FIELD, "M": 8, "ef_con": 64}])
    status = collection.rebuild_status()
    assert "total_objects" in status, f"Missing 'total_objects' in: {status}"
    assert isinstance(status["total_objects"], int)
    wait_for_rebuild(collection)


def test_rebuild_status_has_percent_complete(populated_collection):
    """rebuild_status() must have a 'percent_complete' field."""
    _, collection = populated_collection
    collection.rebuild([{"field": DENSE_FIELD, "M": 8, "ef_con": 64}])
    status = collection.rebuild_status()
    assert "percent_complete" in status, f"Missing 'percent_complete' in: {status}"
    wait_for_rebuild(collection)


def test_rebuild_status_completed_has_started_at(populated_collection):
    """A completed rebuild_status() must include 'started_at' timestamp."""
    _, collection = populated_collection
    collection.rebuild([{"field": DENSE_FIELD, "M": 8, "ef_con": 64}])
    status = wait_for_rebuild(collection)
    assert "started_at" in status, f"Missing 'started_at' in completed status: {status}"


def test_rebuild_status_completed_has_completed_at(populated_collection):
    """A completed rebuild_status() must include 'completed_at' timestamp."""
    _, collection = populated_collection
    collection.rebuild([{"field": DENSE_FIELD, "M": 8, "ef_con": 64}])
    status = wait_for_rebuild(collection)
    assert "completed_at" in status, (
        f"Missing 'completed_at' in completed status: {status}"
    )


def test_rebuild_status_completed_percent_is_100(populated_collection):
    """A completed rebuild_status() must report percent_complete == 100."""
    _, collection = populated_collection
    collection.rebuild([{"field": DENSE_FIELD, "M": 8, "ef_con": 64}])
    status = wait_for_rebuild(collection)
    assert status["percent_complete"] == 100, (
        f"Expected percent_complete == 100, got: {status}"
    )


def test_rebuild_status_completed_objects_match(populated_collection):
    """A completed rebuild must have objects_processed == total_objects."""
    _, collection = populated_collection
    collection.rebuild([{"field": DENSE_FIELD, "M": 8, "ef_con": 64}])
    status = wait_for_rebuild(collection)
    assert status["objects_processed"] == status["total_objects"], (
        f"objects_processed {status['objects_processed']} != "
        f"total_objects {status['total_objects']}"
    )


# -- rebuild - multi_vector field ----------------------------------------------


def test_rebuild_multi_vector_field(client):
    """rebuild() on a multi_vector field must succeed and return in_progress."""
    name = uid("rbd_mv")
    try:
        client.create_collection(name=name, fields=[make_mv_field()])
        collection = client.get_collection(name)
        collection.upsert([{"id": f"mv_{i}", "fields": {MV_FIELD: multi_vec(seed=i)}} for i in range(10)])
        result = collection.rebuild([{"field": MV_FIELD, "M": 8, "ef_con": 64}])
        assert isinstance(result, dict)
        assert result.get("status") == "in_progress"
        wait_for_rebuild(collection)
    finally:
        safe_delete(client, name)


def test_rebuild_multi_vector_completes_successfully(client):
    """rebuild() on a multi_vector field must reach 'completed' status."""
    name = uid("rbd_mv2")
    try:
        client.create_collection(name=name, fields=[make_mv_field()])
        collection = client.get_collection(name)
        collection.upsert([{"id": f"mv_{i}", "fields": {MV_FIELD: multi_vec(seed=i)}} for i in range(20)])
        collection.rebuild([{"field": MV_FIELD, "M": 8, "ef_con": 64}])
        status = wait_for_rebuild(collection)
        assert status["status"] in ("completed", "idle")
    finally:
        safe_delete(client, name)


def test_rebuild_multi_vector_searchable_after_completion(client):
    """After rebuild() on a multi_vector field, search must still return results."""
    name = uid("rbd_mv3")
    try:
        client.create_collection(name=name, fields=[make_mv_field()])
        collection = client.get_collection(name)
        collection.upsert([{"id": f"mv_{i}", "fields": {MV_FIELD: multi_vec(seed=i)}} for i in range(20)])
        collection.rebuild([{"field": MV_FIELD, "M": 8, "ef_con": 64}])
        wait_for_rebuild(collection, timeout=300)
        results = collection.search(
            fields={MV_FIELD: {"query": multi_vec(seed=0), "limit": 5}}
        )["results"][MV_FIELD]
        assert isinstance(results, list)
        assert len(results) > 0
    finally:
        safe_delete(client, name)

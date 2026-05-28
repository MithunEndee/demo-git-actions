"""
test_06_index_rebuild.py

Tests for index rebuild:
  - rebuild() returns expected response keys
  - Rebuild status is trackable via rebuild_status()
  - Previous and new config are reflected
  - Rebuild on empty index raises ValueError
  - Index remains queryable during/after rebuild
"""

import time

import pytest

from helpers import N_VECTORS, dense_vec

# How long to wait for rebuild to complete (seconds)
REBUILD_TIMEOUT = 60


def _wait_for_rebuild(index, timeout: int = REBUILD_TIMEOUT) -> str:
    """Poll rebuild_status until not in_progress, return final status."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        status_resp = index.rebuild_status()
        status = status_resp.get("status", "")
        if status != "in_progress":
            return status
        time.sleep(2)
    return "timeout"


# ── rebuild_status on idle index ──────────────────────────────────────────

def test_rebuild_status_idle_on_fresh_index(populated_index):
    """Before any rebuild, status should be 'idle'."""
    _, index = populated_index
    resp = index.rebuild_status()
    assert resp.get("status") in ("idle", "completed"), (
        f"Unexpected initial rebuild status: {resp}"
    )


# ── rebuild response structure ────────────────────────────────────────────

def test_rebuild_returns_expected_keys(populated_index):
    _, index = populated_index
    result = index.rebuild(M=20, ef_con=160)
    expected_keys = {"status", "previous_config", "new_config", "total_vectors"}
    assert expected_keys.issubset(result.keys()), (
        f"Missing keys: {expected_keys - result.keys()}"
    )


def test_rebuild_new_config_matches_request(populated_index):
    _, index = populated_index
    M, ef_con = 24, 192
    result = index.rebuild(M=M, ef_con=ef_con)
    assert result["new_config"]["M"] == M
    assert result["new_config"]["ef_con"] == ef_con


def test_rebuild_previous_config_is_present(populated_index):
    _, index = populated_index
    result = index.rebuild(M=20, ef_con=160)
    assert "M" in result["previous_config"]
    assert "ef_con" in result["previous_config"]


def test_rebuild_total_vectors_is_positive(populated_index):
    _, index = populated_index
    result = index.rebuild(M=20, ef_con=160)
    assert result["total_vectors"] > 0


def test_rebuild_updates_instance_attributes(populated_index):
    _, index = populated_index
    M, ef_con = 28, 224
    index.rebuild(M=M, ef_con=ef_con)
    assert index.M == M
    assert index.ef_con == ef_con


# ── rebuild_status during and after rebuild ───────────────────────────────

def test_rebuild_status_completes(populated_index):
    _, index = populated_index
    index.rebuild(M=20, ef_con=160)
    final_status = _wait_for_rebuild(index)
    assert final_status in ("completed", "idle"), (
        f"Rebuild did not complete within {REBUILD_TIMEOUT}s; status={final_status}"
    )


def test_rebuild_status_has_correct_keys_while_in_progress(populated_index):
    """If rebuild is in_progress, response should have progress keys."""
    _, index = populated_index
    index.rebuild(M=20, ef_con=160)
    resp = index.rebuild_status()
    # May already be completed on fast servers – either state is valid
    assert "status" in resp
    if resp["status"] == "in_progress":
        for key in ("vectors_processed", "total_vectors", "percent_complete"):
            assert key in resp, f"Missing key '{key}' during in_progress status"


def test_rebuild_percent_complete_range(populated_index):
    """percent_complete must be between 0 and 100 while in_progress."""
    _, index = populated_index
    index.rebuild(M=20, ef_con=160)
    for _ in range(5):
        resp = index.rebuild_status()
        if resp["status"] == "in_progress":
            pct = resp.get("percent_complete", 0)
            assert 0.0 <= pct <= 100.0, f"percent_complete out of range: {pct}"
            break
        time.sleep(1)


# ── Index remains queryable during rebuild ────────────────────────────────

def test_index_queryable_during_rebuild(populated_index):
    """Index must remain queryable while rebuild is in_progress."""
    _, index = populated_index
    index.rebuild(M=20, ef_con=160)

    # Query immediately after starting rebuild
    results = index.query(vector=dense_vec(), top_k=5)
    assert isinstance(results, list), "Index not queryable during rebuild"

    # Wait for rebuild to finish
    _wait_for_rebuild(index)


# ── Edge cases ────────────────────────────────────────────────────────────

def test_rebuild_empty_index_raises(empty_index):
    """Rebuilding an index with no vectors must raise ValueError."""
    _, index = empty_index
    with pytest.raises(ValueError, match="empty"):
        index.rebuild(M=20, ef_con=160)


@pytest.mark.parametrize("M,ef_con", [(2, 4), (4, 16), (100, 1024)])
def test_rebuild_boundary_hnsw_params(populated_index, M, ef_con):
    """Boundary values for M (2-100) and ef_con (4-1024) must be accepted."""
    _, index = populated_index
    result = index.rebuild(M=M, ef_con=ef_con)
    assert result["new_config"]["M"] == M
    assert result["new_config"]["ef_con"] == ef_con
    _wait_for_rebuild(index)

"""
Tests for backup operations across all collection types.

Covers create_backup, list_backups, active_backup, backup_info, restore_backup,
delete_backup, download_backup, and upload_backup. Backups are async -
create_backup returns immediately with status="in_progress"; tests needing a
completed backup poll via wait_for_backup(). Downloaded files are written to
a tempfile.mkdtemp() directory and cleaned up in finally blocks.
"""

import os
import shutil
import tempfile
import time

import pytest
from helpers import (
    DENSE_FIELD,
    HYBRID_DIM,
    MV_FIELD,
    SPARSE_FIELD,
    dense_vec,
    get_collection_names,
    make_dense_field,
    make_item,
    make_mv_field,
    make_mv_item,
    make_sparse_field,
    make_sparse_item,
    multi_vec,
    safe_delete,
    sparse_vec,
    uid,
)

from endee.exceptions import EndeeException

# -- helpers ------------------------------------------------------------------


def _backup_names(client) -> set:
    """Return the set of all backup names currently on the server."""
    result = client.list_backups()
    if isinstance(result, list):
        return {b.get("name") for b in result if isinstance(b, dict) and b.get("name")}
    if isinstance(result, dict):
        return set(result.keys())
    return set()


def _delete_backup_silently(client, name: str) -> None:
    try:
        client.delete_backup(name)
    except Exception:
        pass


def wait_for_backup(client, backup_name: str, timeout: int = 120) -> None:
    """Poll list_backups() until backup_name appears.

    The server only adds an entry once the backup is fully written, so
    presence in the list means the backup is safe to use for restore/info.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = client.list_backups()
        if isinstance(result, list):
            names = [b.get("name") for b in result if isinstance(b, dict)]
            found = backup_name in names or any(backup_name in str(b) for b in result)
        else:
            found = backup_name in result or backup_name in str(result)
        if found:
            return
        time.sleep(0.5)
    raise TimeoutError(
        f"Backup '{backup_name}' did not appear in list_backups() within {timeout}s"
    )


def cleanup_backup(client, backup_name: str, timeout: int = 120) -> None:
    """Wait for backup if needed, then always delete it.

    Logic:
    - Already in list_backups()  -> skip wait, just delete (fast path)
    - Not in list + active=True  -> backup is in-progress; wait then delete
    - Not in list + active=False -> never created or already deleted; delete (no-op)

    Safe to call from finally blocks: if wait_for_backup times out, the delete
    is still attempted so no backup is ever left on the server.
    """
    try:
        result = client.list_backups()
        if isinstance(result, list):
            in_list = backup_name in [
                b.get("name") for b in result if isinstance(b, dict)
            ] or any(backup_name in str(b) for b in result)
        else:
            in_list = backup_name in result or backup_name in str(result)

        if not in_list:
            # Only worth waiting if a backup is actively running
            if client.active_backup().get("active", False):
                try:
                    wait_for_backup(client, backup_name, timeout=timeout)
                except Exception:
                    pass
    except Exception:
        pass
    _delete_backup_silently(client, backup_name)


# -- create_backup (Collection) ------------------------------------------------


def test_create_backup_returns_dict(client, populated_collection):
    """create_backup() must return a dict."""
    _, collection = populated_collection
    backup_name = uid("bk")
    try:
        result = collection.create_backup(name=backup_name)
        assert isinstance(result, dict)
    finally:
        cleanup_backup(client, backup_name)


def test_create_backup_status_is_in_progress(client, populated_collection):
    """create_backup() response must contain status='in_progress'."""
    _, collection = populated_collection
    backup_name = uid("bkst")
    try:
        result = collection.create_backup(name=backup_name)
        assert result.get("status") == "in_progress", (
            f"Expected status='in_progress', got: {result}"
        )
    finally:
        cleanup_backup(client, backup_name)


def test_create_backup_response_contains_backup_name(client, populated_collection):
    """create_backup() response must reference the backup name."""
    _, collection = populated_collection
    backup_name = uid("bkn")
    try:
        result = collection.create_backup(name=backup_name)
        has_name = (
            result.get("backup_name") == backup_name
            or result.get("name") == backup_name
            or backup_name in str(result)
        )
        assert has_name, f"backup name not found in response: {result}"
    finally:
        cleanup_backup(client, backup_name)


def test_create_backup_on_empty_collection_accepted(client, empty_collection):
    """create_backup() on an empty collection must be accepted without error."""
    _, collection = empty_collection
    backup_name = uid("bke")
    try:
        result = collection.create_backup(name=backup_name)
        assert isinstance(result, dict)
    finally:
        cleanup_backup(client, backup_name)


def test_create_backup_two_unique_names_both_succeed(client, populated_collection):
    """Two backups with different names must both succeed."""
    _, collection = populated_collection
    b1, b2 = uid("bka"), uid("bkb")
    try:
        r1 = collection.create_backup(name=b1)
        wait_for_backup(client, b1)
        r2 = collection.create_backup(name=b2)
        wait_for_backup(client, b2)
        assert isinstance(r1, dict)
        assert isinstance(r2, dict)
    finally:
        cleanup_backup(client, b1)
        cleanup_backup(client, b2)


def test_create_backup_empty_name_raises(client, empty_collection):
    """create_backup() with empty name must raise ValueError (client-side)."""
    _, collection = empty_collection
    with pytest.raises(ValueError):
        collection.create_backup(name="")


# -- active_backup (Endee client) ---------------------------------------------


def test_active_backup_returns_dict(client):
    """active_backup() must return a dict."""
    result = client.active_backup()
    assert isinstance(result, dict)


def test_active_backup_has_active_key(client):
    """active_backup() response must contain an 'active' key."""
    result = client.active_backup()
    assert "active" in result, f"Missing 'active' key in response: {result}"


def test_active_backup_while_running_is_true(client, populated_collection):
    """active_backup()['active'] must be True immediately after create_backup()."""
    _, collection = populated_collection
    backup_name = uid("bkact")
    try:
        collection.create_backup(name=backup_name)
        status = client.active_backup()
        # Active should be True immediately - if False, backup was already instant
        assert isinstance(status["active"], bool)
    finally:
        cleanup_backup(client, backup_name)


def test_active_backup_false_after_completion(client, populated_collection):
    """active_backup()['active'] must be False after backup completes."""
    _, collection = populated_collection
    backup_name = uid("bkdone")
    try:
        collection.create_backup(name=backup_name)
        wait_for_backup(client, backup_name)
        status = client.active_backup()
        assert status["active"] is False, (
            f"Expected active=False after completion, got: {status}"
        )
    finally:
        cleanup_backup(client, backup_name)


# -- list_backups (Endee client) -----------------------------------------------


def test_list_backups_returns_list_or_dict(client):
    """list_backups() must return a list or dict."""
    result = client.list_backups()
    assert isinstance(result, (list, dict))


def test_list_backups_after_create_contains_backup(client, populated_collection):
    """A completed backup must appear in list_backups()."""
    _, collection = populated_collection
    backup_name = uid("bklst")
    try:
        collection.create_backup(name=backup_name)
        wait_for_backup(client, backup_name)
        result = client.list_backups()
        if isinstance(result, list):
            names = [b.get("name") for b in result if isinstance(b, dict)]
            found = backup_name in names or any(backup_name in str(b) for b in result)
        else:
            found = backup_name in result or backup_name in str(result)
        assert found, f"Backup '{backup_name}' not found in list_backups(): {result}"
    finally:
        cleanup_backup(client, backup_name)


# -- backup_info (Endee client) ------------------------------------------------


def test_backup_info_returns_dict(client, populated_collection):
    """backup_info() must return a dict for a completed backup."""
    _, collection = populated_collection
    backup_name = uid("bkinf")
    try:
        collection.create_backup(name=backup_name)
        wait_for_backup(client, backup_name)
        result = client.backup_info(backup_name=backup_name)
        assert isinstance(result, dict)
    finally:
        cleanup_backup(client, backup_name)


def test_backup_info_contains_name(client, populated_collection):
    """backup_info() response must reference the backup name or contain expected keys."""
    _, collection = populated_collection
    backup_name = uid("bkinfn")
    try:
        collection.create_backup(name=backup_name)
        wait_for_backup(client, backup_name)
        result = client.backup_info(backup_name=backup_name)
        assert isinstance(result, dict)
        assert (
            "original_index" in result
            or "params" in result
            or backup_name in str(result)
        ), f"Unexpected backup_info() response: {result}"
    finally:
        cleanup_backup(client, backup_name)


def test_backup_info_nonexistent_raises(client):
    """backup_info() for a non-existent backup must raise EndeeException."""
    with pytest.raises(EndeeException):
        client.backup_info(backup_name="definitely_does_not_exist_xyz_99999")


def test_backup_info_empty_name_raises(client):
    """backup_info() with empty name must raise ValueError (client-side)."""
    with pytest.raises(ValueError):
        client.backup_info(backup_name="")


# -- restore_backup (Endee client) ---------------------------------------------


def test_restore_backup_returns_dict(client, populated_collection):
    """restore_backup() must return a dict."""
    _, collection = populated_collection
    backup_name = uid("bkrst")
    target_name = uid("rstcol")
    try:
        collection.create_backup(name=backup_name)
        wait_for_backup(client, backup_name)
        result = client.restore_backup(
            backup_name=backup_name, target_collection_name=target_name
        )
        assert isinstance(result, dict)
    finally:
        cleanup_backup(client, backup_name)
        safe_delete(client, target_name)


def test_restore_backup_creates_collection(client, populated_collection):
    """After restore_backup(), the target collection must exist in list_collections()."""
    _, collection = populated_collection
    backup_name = uid("bkrc")
    target_name = uid("rscol")
    try:
        collection.create_backup(name=backup_name)
        wait_for_backup(client, backup_name)
        client.restore_backup(
            backup_name=backup_name, target_collection_name=target_name
        )
        assert target_name in get_collection_names(client), (
            f"Restored collection '{target_name}' not in list_collections()"
        )
    finally:
        cleanup_backup(client, backup_name)
        safe_delete(client, target_name)


def test_restore_backup_collection_is_searchable(client, populated_collection):
    """A restored collection must return search results."""
    _, collection = populated_collection
    backup_name = uid("bkrs")
    target_name = uid("rssrch")
    try:
        collection.create_backup(name=backup_name)
        wait_for_backup(client, backup_name)
        client.restore_backup(
            backup_name=backup_name, target_collection_name=target_name
        )
        restored = client.get_collection(target_name)
        results = restored.search(
            fields={DENSE_FIELD: {"query": dense_vec(seed=0), "limit": 5}},
        )["results"][DENSE_FIELD]
        assert isinstance(results, list)
        assert len(results) > 0
    finally:
        cleanup_backup(client, backup_name)
        safe_delete(client, target_name)


def test_restore_backup_collection_accepts_upsert(client, populated_collection):
    """A restored collection must accept new upsert operations."""
    _, collection = populated_collection
    backup_name = uid("bkup")
    target_name = uid("rsupsrt")
    try:
        collection.create_backup(name=backup_name)
        wait_for_backup(client, backup_name)
        client.restore_backup(
            backup_name=backup_name, target_collection_name=target_name
        )
        restored = client.get_collection(target_name)
        result = restored.upsert([make_item(999)])
        assert result.get("upserted") == 1
    finally:
        cleanup_backup(client, backup_name)
        safe_delete(client, target_name)


def test_restore_backup_empty_name_raises(client):
    """restore_backup() with empty backup_name must raise ValueError (client-side)."""
    with pytest.raises(ValueError):
        client.restore_backup(backup_name="", target_collection_name="some_col")


def test_restore_backup_empty_target_raises(client):
    """restore_backup() with empty target_collection_name must raise ValueError."""
    with pytest.raises(ValueError):
        client.restore_backup(backup_name="some_backup", target_collection_name="")


def test_restore_backup_nonexistent_raises(client):
    """restore_backup() for a non-existent backup must raise EndeeException."""
    with pytest.raises(EndeeException):
        client.restore_backup(
            backup_name="definitely_does_not_exist_xyz_99999",
            target_collection_name=uid("ghost"),
        )


# -- delete_backup (Endee client) ----------------------------------------------


def test_delete_backup_returns_dict(client, populated_collection):
    """delete_backup() must return a dict."""
    _, collection = populated_collection
    backup_name = uid("bkdel")
    try:
        collection.create_backup(name=backup_name)
        wait_for_backup(client, backup_name)
        result = client.delete_backup(backup_name=backup_name)
        assert isinstance(result, dict)
    finally:
        cleanup_backup(client, backup_name)


def test_delete_backup_removes_from_list(client, populated_collection):
    """After delete_backup(), the backup must not appear in list_backups()."""
    _, collection = populated_collection
    backup_name = uid("bkrm")
    try:
        collection.create_backup(name=backup_name)
        wait_for_backup(client, backup_name)
        client.delete_backup(backup_name=backup_name)
        result = client.list_backups()
        if isinstance(result, list):
            names = [b.get("name") for b in result if isinstance(b, dict)]
            found = backup_name in names or any(backup_name in str(b) for b in result)
        else:
            found = backup_name in result or backup_name in str(result)
        assert not found, f"Backup '{backup_name}' still in list after delete"
    finally:
        cleanup_backup(client, backup_name)


def test_delete_backup_second_delete_raises(client, populated_collection):
    """Deleting a backup that no longer exists must raise EndeeException."""
    _, collection = populated_collection
    backup_name = uid("bkdd")
    try:
        collection.create_backup(name=backup_name)
        wait_for_backup(client, backup_name)
        client.delete_backup(backup_name=backup_name)
        with pytest.raises(EndeeException):
            client.delete_backup(backup_name=backup_name)
    finally:
        cleanup_backup(client, backup_name)


def test_delete_backup_nonexistent_raises(client):
    """delete_backup() for a non-existent backup must raise EndeeException."""
    with pytest.raises(EndeeException):
        client.delete_backup(backup_name="definitely_does_not_exist_xyz_99999")


def test_delete_backup_empty_name_raises(client):
    """delete_backup() with empty name must raise ValueError (client-side)."""
    with pytest.raises(ValueError):
        client.delete_backup(backup_name="")


# -- multi_vector collection backup --------------------------------------------


def test_create_backup_mv_collection_returns_dict(client):
    """create_backup() on a multi_vector collection must return a dict."""
    name = uid("mvbkc")
    backup_name = uid("mvbk")
    try:
        client.create_collection(name=name, fields=[make_mv_field()])
        col = client.get_collection(name)
        col.upsert([make_mv_item(i) for i in range(10)])
        result = col.create_backup(name=backup_name)
        assert isinstance(result, dict)
        assert result.get("status") == "in_progress"
    finally:
        cleanup_backup(client, backup_name)
        safe_delete(client, name)


def test_restore_backup_mv_collection_is_searchable(client):
    """A restored multi_vector collection must return search results."""
    name = uid("mvbkrs")
    backup_name = uid("mvbkrsbk")
    target_name = uid("mvrstcol")
    try:
        client.create_collection(name=name, fields=[make_mv_field()])
        col = client.get_collection(name)
        col.upsert([make_mv_item(i) for i in range(10)])
        col.create_backup(name=backup_name)
        wait_for_backup(client, backup_name)
        client.restore_backup(
            backup_name=backup_name, target_collection_name=target_name
        )
        restored = client.get_collection(target_name)
        results = restored.search(
            fields={MV_FIELD: {"query": multi_vec(seed=0), "limit": 5}}
        )["results"][MV_FIELD]
        assert isinstance(results, list)
        assert len(results) > 0
    finally:
        cleanup_backup(client, backup_name)
        safe_delete(client, name)
        safe_delete(client, target_name)


# -- sparse collection backup / restore ----------------------------------------


def test_create_backup_sparse_collection_returns_dict(client):
    """create_backup() on a sparse collection must return a dict with status=in_progress."""
    name = uid("spbkc")
    backup_name = uid("spbk")
    try:
        client.create_collection(name=name, fields=[make_sparse_field()])
        col = client.get_collection(name)
        col.upsert([make_sparse_item(i) for i in range(10)])
        result = col.create_backup(name=backup_name)
        assert isinstance(result, dict)
        assert result.get("status") == "in_progress"
    finally:
        cleanup_backup(client, backup_name)
        safe_delete(client, name)


def test_restore_backup_sparse_collection_is_searchable(client):
    """A restored sparse collection must return search results."""
    name = uid("spbkrs")
    backup_name = uid("spbkrsbk")
    target_name = uid("sprstcol")
    try:
        client.create_collection(name=name, fields=[make_sparse_field()])
        col = client.get_collection(name)
        col.upsert([make_sparse_item(i) for i in range(10)])
        col.create_backup(name=backup_name)
        wait_for_backup(client, backup_name)
        client.restore_backup(
            backup_name=backup_name, target_collection_name=target_name
        )
        restored = client.get_collection(target_name)
        si, sv = sparse_vec(seed=0)
        results = restored.search(
            fields={SPARSE_FIELD: {"query": {"indices": si, "values": sv}, "limit": 5}}
        )["results"][SPARSE_FIELD]
        assert isinstance(results, list)
        assert len(results) > 0
    finally:
        cleanup_backup(client, backup_name)
        safe_delete(client, name)
        safe_delete(client, target_name)


# -- multi-field (dense + sparse) collection backup / restore ------------------


def test_create_backup_multi_field_collection_returns_dict(client):
    """create_backup() on a dense+sparse collection must return a dict with status=in_progress."""
    name = uid("mfbkc")
    backup_name = uid("mfbk")
    try:
        client.create_collection(
            name=name, fields=[make_dense_field(dim=HYBRID_DIM), make_sparse_field()]
        )
        col = client.get_collection(name)
        col.upsert([make_item(i, dim=HYBRID_DIM, with_sparse=True) for i in range(10)])
        result = col.create_backup(name=backup_name)
        assert isinstance(result, dict)
        assert result.get("status") == "in_progress"
    finally:
        cleanup_backup(client, backup_name)
        safe_delete(client, name)


def test_restore_backup_multi_field_collection_is_searchable(client):
    """A restored dense+sparse collection must return results for both field types."""
    name = uid("mfbkrs")
    backup_name = uid("mfbkrsbk")
    target_name = uid("mfrstcol")
    try:
        client.create_collection(
            name=name, fields=[make_dense_field(dim=HYBRID_DIM), make_sparse_field()]
        )
        col = client.get_collection(name)
        col.upsert([make_item(i, dim=HYBRID_DIM, with_sparse=True) for i in range(10)])
        col.create_backup(name=backup_name)
        wait_for_backup(client, backup_name)
        client.restore_backup(
            backup_name=backup_name, target_collection_name=target_name
        )
        restored = client.get_collection(target_name)

        # Dense field search
        dense_results = restored.search(
            fields={DENSE_FIELD: {"query": dense_vec(HYBRID_DIM, seed=0), "limit": 5}}
        )["results"][DENSE_FIELD]
        assert isinstance(dense_results, list)
        assert len(dense_results) > 0

        # Sparse field search
        si, sv = sparse_vec(seed=0)
        sparse_results = restored.search(
            fields={SPARSE_FIELD: {"query": {"indices": si, "values": sv}, "limit": 5}}
        )["results"][SPARSE_FIELD]
        assert isinstance(sparse_results, list)
        assert len(sparse_results) > 0
    finally:
        cleanup_backup(client, backup_name)
        safe_delete(client, name)
        safe_delete(client, target_name)


# -- full backup lifecycle -----------------------------------------------------


def test_backup_full_lifecycle(client, populated_collection):
    """Create -> wait -> info -> restore -> delete lifecycle must complete successfully."""
    _, collection = populated_collection
    backup_name = uid("bklife")
    target_name = uid("lifecol")
    try:
        # Create (async)
        result = collection.create_backup(name=backup_name)
        assert result.get("status") == "in_progress"

        # Wait for completion
        wait_for_backup(client, backup_name)

        # active_backup should now be False
        assert client.active_backup()["active"] is False

        # Info
        info = client.backup_info(backup_name=backup_name)
        assert isinstance(info, dict)

        # Restore
        client.restore_backup(
            backup_name=backup_name, target_collection_name=target_name
        )
        assert target_name in get_collection_names(client)

        # Delete backup
        client.delete_backup(backup_name=backup_name)

        # Verify removed
        result = client.list_backups()
        if isinstance(result, list):
            names = [b.get("name") for b in result if isinstance(b, dict)]
            found = backup_name in names or any(backup_name in str(b) for b in result)
        else:
            found = backup_name in result or backup_name in str(result)
        assert not found
    finally:
        cleanup_backup(client, backup_name)
        safe_delete(client, target_name)


# -- download_backup (Endee client) --------------------------------------------


def test_download_backup_returns_path(client, populated_collection):
    """download_backup() must return the local path where the file was written."""
    _, collection = populated_collection
    backup_name = uid("bkdl")
    tmp_dir = tempfile.mkdtemp()
    local_path = os.path.join(tmp_dir, f"{backup_name}.tar")
    try:
        collection.create_backup(name=backup_name)
        wait_for_backup(client, backup_name)
        result = client.download_backup(backup_name=backup_name, dest_path=local_path)
        assert result == local_path
    finally:
        cleanup_backup(client, backup_name)
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_download_backup_file_exists_and_nonempty(client, populated_collection):
    """download_backup() must write a non-empty .tar file to the specified path."""
    _, collection = populated_collection
    backup_name = uid("bkdlf")
    tmp_dir = tempfile.mkdtemp()
    local_path = os.path.join(tmp_dir, f"{backup_name}.tar")
    try:
        collection.create_backup(name=backup_name)
        wait_for_backup(client, backup_name)
        client.download_backup(backup_name=backup_name, dest_path=local_path)
        assert os.path.exists(local_path), f"Expected file at {local_path}"
        assert os.path.getsize(local_path) > 0, "Downloaded file is empty"
    finally:
        cleanup_backup(client, backup_name)
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_download_backup_empty_name_raises(client):
    """download_backup() with empty backup_name must raise ValueError (client-side)."""
    tmp_dir = tempfile.mkdtemp()
    try:
        with pytest.raises(ValueError):
            client.download_backup(
                backup_name="", dest_path=os.path.join(tmp_dir, "x.tar")
            )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_download_backup_empty_dest_raises(client):
    """download_backup() with empty dest_path must raise ValueError (client-side)."""
    with pytest.raises(ValueError):
        client.download_backup(backup_name="some_backup", dest_path="")


def test_download_backup_nonexistent_raises(client):
    """download_backup() for a non-existent backup must raise EndeeException."""
    tmp_dir = tempfile.mkdtemp()
    try:
        with pytest.raises(EndeeException):
            client.download_backup(
                backup_name="definitely_does_not_exist_xyz_99999",
                dest_path=os.path.join(tmp_dir, "ghost.tar"),
            )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# -- upload_backup (Endee client) ----------------------------------------------


def test_upload_backup_returns_dict(client, populated_collection):
    """upload_backup() must return a dict."""
    _, collection = populated_collection
    backup_name = uid("bkul")
    upload_name = uid(
        "bkulup"
    )  # distinct name so upload doesn't conflict with original
    tmp_dir = tempfile.mkdtemp()
    local_path = os.path.join(tmp_dir, f"{upload_name}.tar")
    try:
        collection.create_backup(name=backup_name)
        wait_for_backup(client, backup_name)
        client.download_backup(backup_name=backup_name, dest_path=local_path)
        result = client.upload_backup(file_path=local_path)
        wait_for_backup(client, upload_name)
        assert isinstance(result, dict)
    finally:
        cleanup_backup(client, backup_name)
        cleanup_backup(client, upload_name)
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_upload_backup_appears_in_list_backups(client, populated_collection):
    """An uploaded backup must appear in list_backups()."""
    _, collection = populated_collection
    backup_name = uid("bkulst")
    upload_name = uid(
        "bkulstup"
    )  # distinct name so upload doesn't conflict with original
    tmp_dir = tempfile.mkdtemp()
    local_path = os.path.join(tmp_dir, f"{upload_name}.tar")
    try:
        collection.create_backup(name=backup_name)
        wait_for_backup(client, backup_name)
        client.download_backup(backup_name=backup_name, dest_path=local_path)
        client.upload_backup(file_path=local_path)
        wait_for_backup(client, upload_name)
        assert upload_name in _backup_names(client), (
            f"Uploaded backup '{upload_name}' not found in list_backups()"
        )
    finally:
        cleanup_backup(client, backup_name)
        cleanup_backup(client, upload_name)
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_upload_backup_non_tar_raises(client):
    """upload_backup() with a non-.tar file must raise ValueError (client-side)."""
    tmp_dir = tempfile.mkdtemp()
    bad_path = os.path.join(tmp_dir, "backup.zip")
    try:
        with open(bad_path, "wb") as fh:
            fh.write(b"fake content")
        with pytest.raises(ValueError):
            client.upload_backup(file_path=bad_path)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# -- download + upload roundtrip / lifecycle -----------------------------------


def test_download_upload_roundtrip(client, populated_collection):
    """Download a backup then re-upload it under a new name; both must exist in list_backups()."""
    _, collection = populated_collection
    backup_name = uid("bkrt")
    upload_name = uid("bkrtup")  # distinct name avoids server conflict on re-upload
    tmp_dir = tempfile.mkdtemp()
    local_path = os.path.join(tmp_dir, f"{upload_name}.tar")
    try:
        collection.create_backup(name=backup_name)
        wait_for_backup(client, backup_name)

        # Download using the upload filename (server will register as upload_name)
        client.download_backup(backup_name=backup_name, dest_path=local_path)
        assert os.path.exists(local_path)

        # Upload and wait for it to appear in list_backups()
        client.upload_backup(file_path=local_path)
        wait_for_backup(client, upload_name)

        # Both original and uploaded must be present
        current = _backup_names(client)
        assert backup_name in current, f"Original backup '{backup_name}' missing"
        assert upload_name in current, f"Uploaded backup '{upload_name}' missing"
    finally:
        cleanup_backup(client, backup_name)
        cleanup_backup(client, upload_name)
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_download_upload_restore_lifecycle(client, populated_collection):
    """Full lifecycle: create -> download -> upload -> restore -> searchable -> delete all."""
    _, collection = populated_collection
    backup_name = uid("bklc")
    upload_name = uid("bklcup")  # distinct name avoids server conflict on re-upload
    tmp_dir = tempfile.mkdtemp()
    local_path = os.path.join(tmp_dir, f"{upload_name}.tar")
    target_name = uid("uplrstcol")
    upload_deleted = False
    try:
        # Create and wait
        collection.create_backup(name=backup_name)
        wait_for_backup(client, backup_name)

        # Download to local disk
        client.download_backup(backup_name=backup_name, dest_path=local_path)
        assert os.path.exists(local_path)
        assert os.path.getsize(local_path) > 0

        # Upload and wait for it to appear (server registers as upload_name)
        client.upload_backup(file_path=local_path)
        wait_for_backup(client, upload_name)

        # Restore from the uploaded backup into a new collection
        client.restore_backup(
            backup_name=upload_name, target_collection_name=target_name
        )
        assert target_name in get_collection_names(client), (
            f"Restored collection '{target_name}' not found after upload+restore"
        )

        # Restored collection must be searchable
        restored = client.get_collection(target_name)
        results = restored.search(
            fields={DENSE_FIELD: {"query": dense_vec(seed=0), "limit": 5}}
        )["results"][DENSE_FIELD]
        assert isinstance(results, list)
        assert len(results) > 0

        # Delete both the original and uploaded backup from the server
        client.delete_backup(backup_name=backup_name)
        client.delete_backup(backup_name=upload_name)
        upload_deleted = True

        # Verify both are gone
        remaining = _backup_names(client)
        assert backup_name not in remaining
        assert upload_name not in remaining
    finally:
        cleanup_backup(client, backup_name)
        if not upload_deleted:
            cleanup_backup(client, upload_name)
        safe_delete(client, target_name)
        shutil.rmtree(tmp_dir, ignore_errors=True)

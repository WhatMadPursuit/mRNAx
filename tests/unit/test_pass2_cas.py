import json
import re

import pytest

from mrnax.cloud import cas_store
from mrnax.cloud.cas_store import ExecutionCASStore, StaleLeaseConflictError
from mrnax.cloud.drive_rest_v3 import DriveRestV3Client, InMemoryDriveV3Transport


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(cas_store.time, "time", lambda: now[0])
    drive = DriveRestV3Client(InMemoryDriveV3Transport())
    root = drive.create_folder("root", "runs")
    snapshots = drive.create_folder("root", "receipts")
    store = ExecutionCASStore(tmp_path / "live.db", drive_verifier=drive,
                              commits_snapshot_folder_id=snapshots, runs_root_id=root)
    execution = store.submit_execution("submission", "digest", "intent")
    lease = store.acquire_or_renew_lease(execution.execution_id, "worker1", ttl_seconds=5)
    return now, drive, root, snapshots, store, lease


def seal_attempt(drive, root, lease):
    folder = drive.ensure_attempt_dir(root, lease.execution_id, lease.attempt_id)
    artifact = drive.upload_verified_bytes(folder, "result.bin", b"result")
    seal = drive.write_attempt_seal(folder, execution_id=lease.execution_id, attempt_id=lease.attempt_id,
                                    lease_epoch=lease.lease_epoch, receipts=[artifact])
    request = dict(execution_id=lease.execution_id, attempt_id=lease.attempt_id, lease_epoch=lease.lease_epoch,
                   sealed_file_id=seal.file_id, sealed_sha256=seal.sha256, lease_token=lease.lease_token)
    return request, artifact.file_id


@pytest.mark.parametrize("recover_via", ["takeover", "status", "retry"])
def test_lost_ack_elects_same_winner_in_live_and_cold_store(workspace, tmp_path, monkeypatch, recover_via):
    now, drive, root, snapshots, store, lease = workspace
    request, _ = seal_attempt(drive, root, lease)
    upload = drive.upload_verified_bytes

    def drop_ack(*args, **kwargs):
        landed = upload(*args, **kwargs)
        if args[1].startswith("COMMIT_RECEIPT_"):
            raise TimeoutError("ack lost after publication")
        return landed

    monkeypatch.setattr(drive, "upload_verified_bytes", drop_ack)
    with pytest.raises(TimeoutError):
        store.commit_attempt_cas(**request)
    now[0] += 100
    restarted = ExecutionCASStore(store.db_path, drive_verifier=drive, commits_snapshot_folder_id=snapshots)
    if recover_via == "takeover":
        with pytest.raises(StaleLeaseConflictError, match="already COMMITTED"):
            restarted.acquire_or_renew_lease(lease.execution_id, "worker2")
    elif recover_via == "retry":
        assert restarted.commit_attempt_cas(**request).winning_attempt_id == lease.attempt_id
    live = restarted.get_execution(lease.execution_id)
    cold = ExecutionCASStore(tmp_path / "cold.db", drive_verifier=drive, commits_snapshot_folder_id=snapshots)
    assert cold.reconcile_from_drive_seals(drive, root) == 1
    restored = cold.get_execution(lease.execution_id)
    for row in (live, restored):
        assert (row.status, row.lease_epoch, row.winning_attempt_id, row.winning_sealed_sha256) == (
            "COMMITTED", lease.lease_epoch, lease.attempt_id, request["sealed_sha256"])


def test_matching_receipt_name_with_changed_bytes_rejected(workspace):
    _, drive, root, snapshots, store, lease = workspace
    request, _ = seal_attempt(drive, root, lease)
    store.commit_attempt_cas(**request)
    [entry] = drive.transport.list_children(parent_id=snapshots, name_exact=f"COMMIT_RECEIPT_{lease.execution_id}.json")
    payload = json.loads(drive.transport.download_bytes(entry["id"]))
    payload["winning_sealed_sha256"] = "0" * 64
    drive.upload_verified_bytes(snapshots, entry["name"], json.dumps(payload).encode())
    with pytest.raises(StaleLeaseConflictError, match="Conflicting receipt contents"):
        store.commit_attempt_cas(**request)
    with pytest.raises(StaleLeaseConflictError, match="Conflicting Drive receipt contents"):
        store.reconcile_from_drive_seals(drive, root)


@pytest.mark.parametrize("use_drive_verifier", [True, False])
def test_unset_snapshot_folder_restores_local_committed_winner(workspace, tmp_path, use_drive_verifier):
    _, drive, root, _, _, _ = workspace
    store = ExecutionCASStore(tmp_path / "local.db", drive_verifier=drive if use_drive_verifier else None)
    row = store.submit_execution("sub-local", "d", "intent")
    lease = store.acquire_or_renew_lease(row.execution_id, "worker")
    request, _ = seal_attempt(drive, root, lease)
    store.commit_attempt_cas(**request)
    recovered = ExecutionCASStore(tmp_path / "restored.db")
    assert recovered.reconcile_from_drive_seals(drive, root) == 1
    assert recovered.get_execution(row.execution_id).winning_sealed_file_id == request["sealed_file_id"]


@pytest.mark.parametrize("time_at_upload", [1005.0, 1100.0])
def test_delayed_drive_receipt_cannot_commit_or_restore_expired_lease(
    workspace, tmp_path, monkeypatch, time_at_upload
):
    now, drive, root, snapshots, store, lease = workspace
    request, _ = seal_attempt(drive, root, lease)
    upload = drive.upload_verified_bytes

    def slow_upload(*args, **kwargs):
        if args[1].startswith("COMMIT_RECEIPT_"):
            now[0] = time_at_upload
        return upload(*args, **kwargs)

    monkeypatch.setattr(drive, "upload_verified_bytes", slow_upload)
    with pytest.raises(StaleLeaseConflictError, match="expired"):
        store.commit_attempt_cas(**request)
    assert store.get_execution(lease.execution_id).status == "RUNNING"
    recovered = ExecutionCASStore(tmp_path / "cold.db", drive_verifier=drive, commits_snapshot_folder_id=snapshots)
    assert recovered.reconcile_from_drive_seals(drive, root) == 0
    retry = store.acquire_or_renew_lease(lease.execution_id, "worker2")
    request, _ = seal_attempt(drive, root, retry)
    assert store.commit_attempt_cas(**request).winning_attempt_id == retry.attempt_id
    assert recovered.reconcile_from_drive_seals(drive, root) == 1
    assert recovered.get_execution(lease.execution_id).winning_attempt_id == retry.attempt_id


@pytest.mark.parametrize("bad_parent", ["missing", "outside", "shared_outside", "duplicate", "other_root"])
def test_commit_requires_positive_designated_attempt_folder(workspace, monkeypatch, bad_parent):
    _, drive, root, _, store, lease = workspace
    other = drive.create_folder("root", "unrelated")
    request, artifact = seal_attempt(drive, other if bad_parent == "other_root" else root, lease)
    metadata = drive.transport.get_metadata
    parent = metadata(artifact)["parents"][0]
    if bad_parent == "duplicate":
        attempts = metadata(parent)["parents"][0]
        drive.transport.create_file(parent_id=attempts, name=lease.attempt_id, mime_type=metadata(parent)["mimeType"])

    def modified(file_id):
        meta = metadata(file_id)
        if bad_parent == "missing" and file_id == artifact:
            del meta["parents"]
        elif bad_parent == "outside" and file_id == artifact:
            meta["parents"] = [other]
        elif bad_parent == "shared_outside" and file_id in {request["sealed_file_id"], artifact}:
            meta["parents"] = [other]
        return meta

    monkeypatch.setattr(drive.transport, "get_metadata", modified)
    with pytest.raises(StaleLeaseConflictError, match="folder|ancestry|runs root"):
        store.commit_attempt_cas(**request)


def test_cas_attempts_retain_full_execution_uuid(workspace):
    *_, lease = workspace
    assert re.fullmatch(r"att_exec_[0-9a-f]{32}_001", lease.attempt_id)


def test_gateway_restores_local_winner_at_startup(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from mrnax.cloud.gateway import create_production_app

    store = ExecutionCASStore(tmp_path / "prior.db")
    execution = store.submit_execution("gateway-local", "d", "intent")
    lease = store.acquire_or_renew_lease(execution.execution_id, "worker")
    store.commit_attempt_cas(execution_id=execution.execution_id, attempt_id=lease.attempt_id,
                             lease_epoch=lease.lease_epoch, lease_token=lease.lease_token,
                             sealed_file_id="local_seal", sealed_sha256="a" * 64)
    monkeypatch.setenv("MRNAX_GATEWAY_ACCESS_TOKEN", "test-only-token")
    monkeypatch.setenv("MRNAX_STATE_ROOT", str(tmp_path))
    monkeypatch.setenv("MRNAX_GDRIVE_BIN", str(tmp_path / "no-drive"))
    monkeypatch.delenv("MRNAX_COMMITS_SNAPSHOT_FOLDER_ID", raising=False)
    with TestClient(create_production_app()) as client:
        response = client.get(f"/v1/executions/{execution.execution_id}",
                              headers={"X-Mrnax-Access-Token": "test-only-token"})
    assert response.status_code == 200
    assert response.json()["winning_attempt_id"] == lease.attempt_id


def test_inflight_drive_write_arriving_after_takeover_is_revoked(workspace, tmp_path, monkeypatch):
    now, drive, root, snapshots, store, lease = workspace
    request, _ = seal_attempt(drive, root, lease)
    upload = drive.upload_verified_bytes
    delayed = []

    def timeout_before_landing(*args, **kwargs):
        if args[1].startswith("COMMIT_RECEIPT_"):
            delayed.append((args, kwargs))
            raise TimeoutError("POST in flight")
        return upload(*args, **kwargs)

    monkeypatch.setattr(drive, "upload_verified_bytes", timeout_before_landing)
    with pytest.raises(TimeoutError):
        store.commit_attempt_cas(**request)
    monkeypatch.setattr(drive, "upload_verified_bytes", upload)
    # Replay advances the coordinator clock while the remote clock remains at its earlier value.
    retry = store.acquire_or_renew_lease(lease.execution_id, "worker2", now_ts=now[0] + 100)
    args, kwargs = delayed.pop()
    upload(*args, **kwargs)
    request, _ = seal_attempt(drive, root, retry)
    assert store.commit_attempt_cas(**request, now_ts=now[0] + 101).winning_attempt_id == retry.attempt_id
    restored = ExecutionCASStore(tmp_path / "cold.db", drive_verifier=drive, commits_snapshot_folder_id=snapshots)
    assert restored.restore_committed_receipts() == 1
    assert restored.get_execution(lease.execution_id).winning_attempt_id == retry.attempt_id


def test_remote_publication_deadline_uses_drive_clock(workspace, tmp_path, monkeypatch):
    from datetime import datetime, timezone

    now, drive, root, snapshots, store, lease = workspace
    request, _ = seal_attempt(drive, root, lease)
    create = drive.transport.create_file

    def skewed_clock(**kwargs):
        if kwargs["name"].startswith("COMMIT_RECEIPT_"):
            now[0] += 10
        entry = create(**kwargs)
        created = datetime.fromtimestamp(now[0] - 60, timezone.utc).isoformat()
        drive.transport._nodes[entry["id"]]["createdTime"] = created
        entry["createdTime"] = created
        return entry

    monkeypatch.setattr(drive.transport, "create_file", skewed_clock)
    with pytest.raises(StaleLeaseConflictError, match="expired"):
        store.commit_attempt_cas(**request)
    restored = ExecutionCASStore(tmp_path / "cold.db", drive_verifier=drive, commits_snapshot_folder_id=snapshots)
    assert restored.restore_committed_receipts() == 0


def test_legacy_local_winner_migrates_to_journal_for_restore_in_new_location(tmp_path):
    path = tmp_path / "old" / "control.db"
    journal = tmp_path / "backup" / "receipts.db"
    store = ExecutionCASStore(path, local_receipts_path=journal)
    row = store.submit_execution("legacy", "legacy-digest", "intent")
    lease = store.acquire_or_renew_lease(row.execution_id, "worker")
    with store._connect() as conn:
        conn.execute("ALTER TABLE executions DROP COLUMN receipt_payload")
        conn.execute("""UPDATE executions SET status='COMMITTED', winning_attempt_id=?, winning_sealed_file_id=?,
                        winning_sealed_sha256=? WHERE execution_id=?""",
                     (lease.attempt_id, "archive", "a" * 64, row.execution_id))
    assert not journal.exists()
    ExecutionCASStore(path, local_receipts_path=journal)
    cold = ExecutionCASStore(tmp_path / "replacement" / "cold.db", local_receipts_path=journal)
    assert cold.restore_committed_receipts() == 1
    assert cold.restore_committed_receipts() == 0
    winner = cold.get_execution(row.execution_id)
    assert (winner.winning_attempt_id, winner.winning_sealed_file_id, winner.winning_sealed_sha256) == (
        lease.attempt_id, "archive", "a" * 64)


def test_drive_receipt_metadata_follows_all_http_pages(monkeypatch):
    import httpx
    from mrnax.cloud.drive_rest_v3 import HttpDriveV3Transport

    pages = []

    def page(request):
        token = request.url.params.get("pageToken")
        assert "createdTime" in request.url.params["fields"]
        pages.append(token)
        if token is None:
            return httpx.Response(200, json={"files": [], "nextPageToken": "next"})
        return httpx.Response(200, json={"files": [{"id": "receipt", "createdTime": "2026-01-01T00:00:00Z"}]})

    client = httpx.Client(transport=httpx.MockTransport(page))
    monkeypatch.setattr(httpx, "Client", lambda **_: client)
    drive = HttpDriveV3Transport(lambda: "fixture-token")
    assert drive.list_children(parent_id="receipts") == [
        {"id": "receipt", "createdTime": "2026-01-01T00:00:00Z"}]
    assert pages == [None, "next"]

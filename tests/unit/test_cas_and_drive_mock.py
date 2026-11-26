"""TDD Contract Tests FT-08 .. FT-10 for Drive REST v3 Sealing, SQLite WAL CAS & Disaster Recovery."""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

import pytest

from mrnax.cloud.cas_store import (
    ExecutionCASStore,
    IdempotencyPayloadMismatchError,
    StaleLeaseConflictError,
)
from mrnax.cloud.colab_worker import ResumableCPUOptimizerWorker
from mrnax.cloud.drive_rest_v3 import DriveRestV3Client, InMemoryDriveV3Transport


def test_cas_commit_rejects_stale_zombie_epoch_with_409_ft08(tmp_path: Path) -> None:
    """FT-08: Zombie worker (epoch=1) wakes up after recovery worker (epoch=2) committed -> rejected (409)."""
    store = ExecutionCASStore(tmp_path / "control_plane.sqlite3")
    transport = InMemoryDriveV3Transport()
    drive = DriveRestV3Client(transport=transport)
    runs_root = drive.create_folder(parent_id="root", name="runs")

    row = store.submit_execution(
        submission_id="sub_zombie_test",
        request_digest="req_sha256_001",
        intent_id="int_001",
    )

    # Worker 1 acquires lease (epoch=1)
    lease_1 = store.acquire_or_renew_lease(row.execution_id, worker_id="colab_vm_1", ttl_seconds=10, now_ts=1000.0)
    assert lease_1.lease_epoch == 1
    att1_dir = drive.ensure_attempt_dir(runs_root, row.execution_id, lease_1.attempt_id)
    r1 = drive.upload_verified_bytes(att1_dir, "result.json", b'{"from":"worker1"}')

    # Simulate timeout at t=1020 -> Worker 2 takes over -> epoch increments to 2!
    lease_2 = store.acquire_or_renew_lease(row.execution_id, worker_id="colab_vm_2", ttl_seconds=60, now_ts=1020.0)
    assert lease_2.lease_epoch == 2
    assert lease_2.attempt_id != lease_1.attempt_id

    att2_dir = drive.ensure_attempt_dir(runs_root, row.execution_id, lease_2.attempt_id)
    r2 = drive.upload_verified_bytes(att2_dir, "result.json", b'{"from":"worker2_winner"}')
    seal_2 = drive.write_attempt_seal(
        att2_dir,
        execution_id=row.execution_id,
        attempt_id=lease_2.attempt_id,
        lease_epoch=lease_2.lease_epoch,
        receipts=[r2],
    )

    # Worker 2 commits first -> HTTP 200 equivalent
    commit_ok = store.commit_attempt_cas(
        execution_id=row.execution_id,
        attempt_id=lease_2.attempt_id,
        lease_epoch=lease_2.lease_epoch,
        sealed_file_id=seal_2.file_id,
        sealed_sha256=seal_2.sha256,
    )
    assert commit_ok.status == "COMMITTED"
    assert commit_ok.winning_attempt_id == lease_2.attempt_id

    # Zombie Worker 1 wakes up, seals its own isolated directory (zero overwrite on Worker 2!), tries to commit -> 409!
    seal_1 = drive.write_attempt_seal(
        att1_dir,
        execution_id=row.execution_id,
        attempt_id=lease_1.attempt_id,
        lease_epoch=lease_1.lease_epoch,
        receipts=[r1],
    )
    with pytest.raises(StaleLeaseConflictError) as exc_info:
        store.commit_attempt_cas(
            execution_id=row.execution_id,
            attempt_id=lease_1.attempt_id,
            lease_epoch=lease_1.lease_epoch,
            sealed_file_id=seal_1.file_id,
            sealed_sha256=seal_1.sha256,
        )
    assert exc_info.value.http_status == 409

    # Verify winning manifest remained untouched AND disaster-recovery reconciliation agrees
    final_row = store.get_execution(row.execution_id)
    assert final_row.winning_attempt_id == lease_2.attempt_id
    assert final_row.winning_sealed_file_id == seal_2.file_id

    # Simulate total SQLite DB loss -> reconstruct from Drive SEALED.json records (picks highest sealed epoch!)
    recovered_store = ExecutionCASStore(tmp_path / "recovered_after_power_loss.sqlite3")
    recovered_store.reconcile_from_drive_seals(drive, runs_root)
    rec_row = recovered_store.get_execution(row.execution_id)
    assert rec_row.status == "COMMITTED"
    assert rec_row.winning_attempt_id == lease_2.attempt_id
    assert rec_row.winning_sealed_file_id == seal_2.file_id


def test_submission_idempotency_vs_explicit_rerun_ft09(tmp_path: Path) -> None:
    """FT-09: Same submission_id + same digest -> same execution_id; mismatched digest -> error; explicit rerun -> new execution_id."""
    store = ExecutionCASStore(tmp_path / "idem.sqlite3")

    e1 = store.submit_execution("sub_retry_01", "digest_abc", "int_001")
    e1_retry = store.submit_execution("sub_retry_01", "digest_abc", "int_001")
    assert e1_retry.execution_id == e1.execution_id

    with pytest.raises(IdempotencyPayloadMismatchError):
        store.submit_execution("sub_retry_01", "digest_DIFFERENT", "int_001")

    e2_explicit_rerun = store.submit_execution("sub_user_rerun_02", "digest_abc", "int_001")
    assert e2_explicit_rerun.execution_id != e1.execution_id


def test_prng_checkpoint_bit_exact_cpu_resume_ft10() -> None:
    """FT-10: CPU trajectory interrupted at step 5 and resumed in a fresh worker matches uninterrupted 10-step run bit-exactly."""
    transport_uninterrupted = InMemoryDriveV3Transport()
    drive_a = DriveRestV3Client(transport=transport_uninterrupted)
    root_a = drive_a.create_folder("root", "runs_a")
    att_a = drive_a.ensure_attempt_dir(root_a, "exec_bit", "att_continuous")

    worker_continuous = ResumableCPUOptimizerWorker(seed=20261127, total_steps=10, checkpoint_every=5)
    result_continuous = worker_continuous.run_with_drive_checkpoints(drive_a, att_a)

    # Interrupted run: stops hard after step 5 checkpoint written to Drive
    transport_resumed = InMemoryDriveV3Transport()
    drive_b = DriveRestV3Client(transport=transport_resumed)
    root_b = drive_b.create_folder("root", "runs_b")
    att_b1 = drive_b.ensure_attempt_dir(root_b, "exec_bit", "att_part1")

    worker_part1 = ResumableCPUOptimizerWorker(seed=20261127, total_steps=10, checkpoint_every=5)
    with pytest.raises(RuntimeError, match="Simulated Colab preemption"):
        worker_part1.run_with_drive_checkpoints(drive_b, att_b1, simulate_interrupt_at_step=5)

    # Fresh worker loads checkpoint from att_b1 on Drive and writes final output to att_b2
    att_b2 = drive_b.ensure_attempt_dir(root_b, "exec_bit", "att_part2")
    worker_part2 = ResumableCPUOptimizerWorker(seed=999999, total_steps=10, checkpoint_every=5)
    result_resumed = worker_part2.run_with_drive_checkpoints(
        drive_b, att_b2, resume_from_attempt_folder_id=att_b1
    )

    assert result_resumed == result_continuous
    assert hashlib.sha256(str(result_resumed).encode()).hexdigest() == hashlib.sha256(
        str(result_continuous).encode()
    ).hexdigest()

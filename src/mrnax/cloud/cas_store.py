"""SQLite WAL persistent control-plane task lease & single-statement CAS store.

Configured with `journal_mode=WAL` and `synchronous=FULL` for power-loss durability on VPS/local,
with cryptographic per-attempt `lease_token`, Drive `SEALED.json` checksum verification prior to
CAS commit, and authoritative `COMMIT_RECEIPT` disaster-recovery snapshots.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from mrnax.cloud.drive_rest_v3 import DRIVE_FOLDER_MIME, DrivePermissionError, DriveRestV3Client


class StaleLeaseConflictError(RuntimeError):
    """Raised when a stale, expired, unauthenticated, or unverified worker attempts to commit (HTTP 409)."""

    http_status: int = 409


class IdempotencyPayloadMismatchError(RuntimeError):
    """Raised when a duplicate submission_id carries a different request_digest (HTTP 409)."""

    http_status: int = 409


@dataclass(frozen=True)
class ExecutionRow:
    execution_id: str
    submission_id: str
    request_digest: str
    intent_id: str
    status: str
    lease_epoch: int
    current_attempt_id: str | None
    current_worker_id: str | None
    lease_expires_at: float | None
    winning_attempt_id: str | None
    winning_sealed_file_id: str | None
    winning_sealed_sha256: str | None


@dataclass(frozen=True)
class LeaseGrant:
    execution_id: str
    attempt_id: str
    worker_id: str
    lease_epoch: int
    lease_token: str
    lease_expires_at: float
    previous_attempt_id: str | None


@dataclass(frozen=True)
class CommitReceipt:
    execution_id: str
    status: str
    winning_attempt_id: str
    winning_sealed_file_id: str
    lease_epoch: int


class ExecutionCASStore:
    """Crash-safe SQLite WAL execution lease & CAS coordinator."""

    def __init__(
        self,
        db_path: Path,
        *,
        drive_verifier: DriveRestV3Client | None = None,
        commits_snapshot_folder_id: str | None = None,
    ) -> None:
        self.db_path = db_path
        self.drive_verifier = drive_verifier
        self.commits_snapshot_folder_id = commits_snapshot_folder_id
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=FULL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS executions (
                    execution_id TEXT PRIMARY KEY,
                    submission_id TEXT NOT NULL UNIQUE,
                    request_digest TEXT NOT NULL,
                    intent_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    lease_epoch INTEGER NOT NULL DEFAULT 0,
                    current_attempt_id TEXT,
                    current_worker_id TEXT,
                    current_lease_token TEXT,
                    lease_expires_at REAL,
                    winning_attempt_id TEXT,
                    winning_sealed_file_id TEXT,
                    winning_sealed_sha256 TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS attempts (
                    attempt_id TEXT PRIMARY KEY,
                    execution_id TEXT NOT NULL REFERENCES executions(execution_id),
                    worker_id TEXT NOT NULL,
                    lease_epoch INTEGER NOT NULL,
                    lease_token TEXT NOT NULL,
                    started_at REAL NOT NULL,
                    sealed_file_id TEXT,
                    commit_outcome TEXT
                );
                """
            )
            conn.execute("COMMIT;")

    @staticmethod
    def _row_to_exec(row: sqlite3.Row) -> ExecutionRow:
        return ExecutionRow(
            execution_id=row["execution_id"],
            submission_id=row["submission_id"],
            request_digest=row["request_digest"],
            intent_id=row["intent_id"],
            status=row["status"],
            lease_epoch=int(row["lease_epoch"]),
            current_attempt_id=row["current_attempt_id"],
            current_worker_id=row["current_worker_id"],
            lease_expires_at=row["lease_expires_at"],
            winning_attempt_id=row["winning_attempt_id"],
            winning_sealed_file_id=row["winning_sealed_file_id"],
            winning_sealed_sha256=row["winning_sealed_sha256"],
        )

    def get_execution(self, execution_id: str) -> ExecutionRow:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM executions WHERE execution_id = ?", (execution_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown execution_id '{execution_id}'")
            return self._row_to_exec(row)

    def submit_execution(
        self, submission_id: str, request_digest: str, intent_id: str, *, now_ts: float | None = None
    ) -> ExecutionRow:
        ts = time.time() if now_ts is None else now_ts
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            try:
                existing = conn.execute(
                    "SELECT * FROM executions WHERE submission_id = ?", (submission_id,)
                ).fetchone()
                if existing is not None:
                    if existing["request_digest"] != request_digest:
                        conn.execute("ROLLBACK;")
                        raise IdempotencyPayloadMismatchError(
                            f"submission_id '{submission_id}' already exists with a different request_digest"
                        )
                    conn.execute("COMMIT;")
                    return self._row_to_exec(existing)

                exec_id = f"exec_{uuid.uuid4().hex[:12]}"
                conn.execute(
                    """
                    INSERT INTO executions (
                        execution_id, submission_id, request_digest, intent_id,
                        status, lease_epoch, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, 'QUEUED', 0, ?, ?)
                    """,
                    (exec_id, submission_id, request_digest, intent_id, ts, ts),
                )
                created = conn.execute(
                    "SELECT * FROM executions WHERE execution_id = ?", (exec_id,)
                ).fetchone()
                conn.execute("COMMIT;")
                return self._row_to_exec(created)
            except Exception:
                if conn.in_transaction:
                    conn.execute("ROLLBACK;")
                raise

    def acquire_or_renew_lease(
        self,
        execution_id: str,
        worker_id: str,
        *,
        ttl_seconds: int = 300,
        now_ts: float | None = None,
    ) -> LeaseGrant:
        ts = time.time() if now_ts is None else now_ts
        expires = ts + ttl_seconds
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            try:
                row = conn.execute(
                    "SELECT * FROM executions WHERE execution_id = ?", (execution_id,)
                ).fetchone()
                if row is None:
                    conn.execute("ROLLBACK;")
                    raise KeyError(f"Unknown execution_id '{execution_id}'")
                if row["status"] == "COMMITTED":
                    conn.execute("ROLLBACK;")
                    raise StaleLeaseConflictError(f"Execution '{execution_id}' is already COMMITTED")

                cur_worker = row["current_worker_id"]
                cur_exp = row["lease_expires_at"]
                cur_attempt = row["current_attempt_id"]
                cur_epoch = int(row["lease_epoch"])
                cur_token = row["current_lease_token"]

                if row["status"] == "RUNNING" and cur_exp is not None and ts < float(cur_exp):
                    if cur_worker == worker_id and cur_attempt is not None and cur_token is not None:
                        conn.execute(
                            "UPDATE executions SET lease_expires_at = ?, updated_at = ? WHERE execution_id = ?",
                            (expires, ts, execution_id),
                        )
                        conn.execute("COMMIT;")
                        return LeaseGrant(
                            execution_id=execution_id,
                            attempt_id=cur_attempt,
                            worker_id=worker_id,
                            lease_epoch=cur_epoch,
                            lease_token=str(cur_token),
                            lease_expires_at=expires,
                            previous_attempt_id=None,
                        )
                    conn.execute("ROLLBACK;")
                    raise StaleLeaseConflictError(
                        f"Execution '{execution_id}' is actively leased by '{cur_worker}' until {cur_exp}"
                    )

                new_epoch = cur_epoch + 1
                new_attempt = f"att_{execution_id[-6:]}_{new_epoch:03d}"
                new_token = f"lzk_{secrets.token_urlsafe(18)}"
                conn.execute(
                    """
                    UPDATE executions
                       SET status = 'RUNNING',
                           lease_epoch = ?,
                           current_attempt_id = ?,
                           current_worker_id = ?,
                           current_lease_token = ?,
                           lease_expires_at = ?,
                           updated_at = ?
                     WHERE execution_id = ?
                    """,
                    (new_epoch, new_attempt, worker_id, new_token, expires, ts, execution_id),
                )
                conn.execute(
                    """
                    INSERT INTO attempts (attempt_id, execution_id, worker_id, lease_epoch, lease_token, started_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (new_attempt, execution_id, worker_id, new_epoch, new_token, ts),
                )
                conn.execute("COMMIT;")
                return LeaseGrant(
                    execution_id=execution_id,
                    attempt_id=new_attempt,
                    worker_id=worker_id,
                    lease_epoch=new_epoch,
                    lease_token=new_token,
                    lease_expires_at=expires,
                    previous_attempt_id=cur_attempt,
                )
            except Exception:
                if conn.in_transaction:
                    conn.execute("ROLLBACK;")
                raise

    def _verify_drive_seal_before_commit(
        self,
        *,
        execution_id: str,
        attempt_id: str,
        lease_epoch: int,
        sealed_file_id: str,
        sealed_sha256: str,
    ) -> None:
        if self.drive_verifier is None:
            return
        try:
            raw_seal = self.drive_verifier.transport.download_bytes(sealed_file_id)
        except DrivePermissionError as exc:
            raise StaleLeaseConflictError(f"Drive seal '{sealed_file_id}' unreadable: {exc}") from exc
        actual_sha = hashlib.sha256(raw_seal).hexdigest()
        if actual_sha != sealed_sha256:
            raise StaleLeaseConflictError(
                f"Drive seal sha256 mismatch: expected {sealed_sha256}, got {actual_sha}"
            )
        doc = json.loads(raw_seal.decode("utf-8"))
        if (
            doc.get("execution_id") != execution_id
            or doc.get("attempt_id") != attempt_id
            or int(doc.get("lease_epoch", -1)) != lease_epoch
        ):
            raise StaleLeaseConflictError("Drive seal metadata does not match execution/attempt/epoch")
        for art in doc.get("artifacts", []):
            meta = self.drive_verifier.transport.get_metadata(str(art["file_id"]))
            if meta.get("md5Checksum") != art.get("md5"):
                raise StaleLeaseConflictError(f"Drive seal artifact md5 mismatch for {art.get('name')}")

    def commit_attempt_cas(
        self,
        *,
        execution_id: str,
        attempt_id: str,
        lease_epoch: int,
        sealed_file_id: str,
        sealed_sha256: str,
        lease_token: str | None = None,
        now_ts: float | None = None,
    ) -> CommitReceipt:
        """Verify Drive seal + lease token + expiry, then run single-statement atomic CAS commit."""
        ts = time.time() if now_ts is None else now_ts
        self._verify_drive_seal_before_commit(
            execution_id=execution_id,
            attempt_id=attempt_id,
            lease_epoch=lease_epoch,
            sealed_file_id=sealed_file_id,
            sealed_sha256=sealed_sha256,
        )

        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            try:
                cur_row = conn.execute(
                    "SELECT status, lease_epoch, current_attempt_id, current_lease_token, lease_expires_at, winning_attempt_id, submission_id, request_digest, intent_id FROM executions WHERE execution_id = ?",
                    (execution_id,),
                ).fetchone()
                if cur_row is None:
                    conn.execute("ROLLBACK;")
                    raise StaleLeaseConflictError(f"CAS rejected: unknown execution '{execution_id}'")

                if lease_token is not None and cur_row["current_lease_token"] != lease_token:
                    conn.execute("ROLLBACK;")
                    raise StaleLeaseConflictError("CAS rejected: invalid worker lease_token")

                if cur_row["lease_expires_at"] is not None and ts > float(cur_row["lease_expires_at"]):
                    conn.execute(
                        "UPDATE attempts SET sealed_file_id = ?, commit_outcome = 'EXPIRED_409' WHERE attempt_id = ?",
                        (sealed_file_id, attempt_id),
                    )
                    conn.execute("COMMIT;")
                    raise StaleLeaseConflictError(
                        f"CAS rejected: lease expired at {cur_row['lease_expires_at']} (now={ts})"
                    )

                cur = conn.execute(
                    """
                    UPDATE executions
                       SET status = 'COMMITTED',
                           winning_attempt_id = ?,
                           winning_sealed_file_id = ?,
                           winning_sealed_sha256 = ?,
                           updated_at = ?
                     WHERE execution_id = ?
                       AND lease_epoch = ?
                       AND current_attempt_id = ?
                       AND status = 'RUNNING'
                       AND (lease_expires_at IS NULL OR lease_expires_at >= ?)
                    """,
                    (
                        attempt_id,
                        sealed_file_id,
                        sealed_sha256,
                        ts,
                        execution_id,
                        lease_epoch,
                        attempt_id,
                        ts,
                    ),
                )
                if cur.rowcount == 0:
                    conn.execute(
                        "UPDATE attempts SET sealed_file_id = ?, commit_outcome = 'REJECTED_409' WHERE attempt_id = ?",
                        (sealed_file_id, attempt_id),
                    )
                    conn.execute("COMMIT;")
                    raise StaleLeaseConflictError(
                        f"CAS rejected for attempt '{attempt_id}' (epoch={lease_epoch}): "
                        f"status={cur_row['status']}, current_epoch={cur_row['lease_epoch']}, winner={cur_row['winning_attempt_id']}"
                    )

                conn.execute(
                    "UPDATE attempts SET sealed_file_id = ?, commit_outcome = 'COMMITTED_200' WHERE attempt_id = ?",
                    (sealed_file_id, attempt_id),
                )
                conn.execute("COMMIT;")
            except Exception:
                if conn.in_transaction:
                    conn.execute("ROLLBACK;")
                raise

        if self.drive_verifier is not None and self.commits_snapshot_folder_id is not None:
            receipt_doc = {
                "schema": "mrnax.commit_receipt.v1",
                "execution_id": execution_id,
                "submission_id": cur_row["submission_id"],
                "request_digest": cur_row["request_digest"],
                "intent_id": cur_row["intent_id"],
                "winning_attempt_id": attempt_id,
                "lease_epoch": lease_epoch,
                "winning_sealed_file_id": sealed_file_id,
                "winning_sealed_sha256": sealed_sha256,
                "committed_at": ts,
            }
            self.drive_verifier.upload_verified_bytes(
                self.commits_snapshot_folder_id,
                f"COMMIT_RECEIPT_{execution_id}.json",
                (json.dumps(receipt_doc, indent=2, sort_keys=True) + "\n").encode("utf-8"),
                mime_type="application/json",
            )

        return CommitReceipt(
            execution_id=execution_id,
            status="COMMITTED",
            winning_attempt_id=attempt_id,
            winning_sealed_file_id=sealed_file_id,
            lease_epoch=lease_epoch,
        )

    def reconcile_from_drive_seals(self, drive: DriveRestV3Client, runs_root_id: str) -> int:
        """Disaster recovery: restore COMMITTED rows ONLY from authoritative Control Plane COMMIT_RECEIPT snapshots
        when `commits_snapshot_folder_id` is configured; never promote uncommitted worker seals over RUNNING tasks."""
        reconciled = 0
        if self.commits_snapshot_folder_id is not None:
            receipt_files = [
                f
                for f in drive.transport.list_children(parent_id=self.commits_snapshot_folder_id)
                if str(f.get("name", "")).startswith("COMMIT_RECEIPT_")
            ]
            for rf in receipt_files:
                doc = json.loads(drive.transport.download_bytes(str(rf["id"])).decode("utf-8"))
                exec_id = str(doc["execution_id"])
                epoch = int(doc["lease_epoch"])
                att_id = str(doc["winning_attempt_id"])
                seal_fid = str(doc["winning_sealed_file_id"])
                sha = str(doc["winning_sealed_sha256"])
                ts = float(doc.get("committed_at", time.time()))
                with self._connect() as conn:
                    conn.execute("BEGIN IMMEDIATE;")
                    existing = conn.execute(
                        "SELECT status, lease_epoch FROM executions WHERE execution_id = ?", (exec_id,)
                    ).fetchone()
                    if existing is None:
                        conn.execute(
                            """
                            INSERT INTO executions (
                                execution_id, submission_id, request_digest, intent_id,
                                status, lease_epoch, current_attempt_id, winning_attempt_id,
                                winning_sealed_file_id, winning_sealed_sha256, created_at, updated_at
                            ) VALUES (?, ?, ?, ?, 'COMMITTED', ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                exec_id,
                                str(doc.get("submission_id", f"reconciled_{exec_id}")),
                                str(doc.get("request_digest", "reconciled")),
                                str(doc.get("intent_id", "reconciled")),
                                epoch,
                                att_id,
                                att_id,
                                seal_fid,
                                sha,
                                ts,
                                ts,
                            ),
                        )
                        reconciled += 1
                    elif existing["status"] != "COMMITTED" and epoch >= int(existing["lease_epoch"]):
                        conn.execute(
                            """
                            UPDATE executions
                               SET status = 'COMMITTED',
                                   lease_epoch = ?,
                                   winning_attempt_id = ?,
                                   winning_sealed_file_id = ?,
                                   winning_sealed_sha256 = ?,
                                   updated_at = ?
                             WHERE execution_id = ?
                            """,
                            (epoch, att_id, seal_fid, sha, ts, exec_id),
                        )
                        reconciled += 1
                    conn.execute("COMMIT;")
            return reconciled

        # Legacy cold-DB bootstrap when no separate commits folder configured (ONLY inserts missing rows, never overrides RUNNING)
        exec_folders = [
            f for f in drive.transport.list_children(parent_id=runs_root_id)
            if f.get("mimeType") == DRIVE_FOLDER_MIME
        ]
        for ef in exec_folders:
            exec_id = str(ef["name"])
            with self._connect() as conn:
                if conn.execute("SELECT 1 FROM executions WHERE execution_id = ?", (exec_id,)).fetchone():
                    continue
            attempts_dirs = [
                c for c in drive.transport.list_children(parent_id=str(ef["id"]), name_exact="attempts")
                if c.get("mimeType") == DRIVE_FOLDER_MIME
            ]
            if not attempts_dirs:
                continue
            att_folders = [
                a for a in drive.transport.list_children(parent_id=str(attempts_dirs[0]["id"]))
                if a.get("mimeType") == DRIVE_FOLDER_MIME
            ]
            best_seal: tuple[int, str, str, str] | None = None
            for af in att_folders:
                for sf in drive.transport.list_children(parent_id=str(af["id"]), name_exact="SEALED.json"):
                    raw = drive.transport.download_bytes(str(sf["id"]))
                    doc = json.loads(raw.decode("utf-8"))
                    epoch = int(doc["lease_epoch"])
                    att_id = str(doc["attempt_id"])
                    sha = hashlib.sha256(raw).hexdigest()
                    if best_seal is None or epoch > best_seal[0]:
                        best_seal = (epoch, att_id, str(sf["id"]), sha)
            if best_seal is not None:
                epoch, att_id, seal_fid, sha = best_seal
                ts = time.time()
                with self._connect() as conn:
                    conn.execute("BEGIN IMMEDIATE;")
                    conn.execute(
                        """
                        INSERT INTO executions (
                            execution_id, submission_id, request_digest, intent_id,
                            status, lease_epoch, current_attempt_id, winning_attempt_id,
                            winning_sealed_file_id, winning_sealed_sha256, created_at, updated_at
                        ) VALUES (?, ?, 'reconciled_from_drive', 'reconciled', 'COMMITTED', ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (exec_id, f"reconciled_{exec_id}", epoch, att_id, att_id, seal_fid, sha, ts, ts),
                    )
                    conn.execute("COMMIT;")
                    reconciled += 1
        return reconciled

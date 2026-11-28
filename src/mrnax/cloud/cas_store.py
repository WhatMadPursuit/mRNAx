"""SQLite WAL persistent control-plane task lease & single-statement CAS store.

Configured with `journal_mode=WAL` and `synchronous=FULL` for power-loss durability on VPS/local,
with cryptographic per-attempt `lease_token`, Drive `SEALED.json` checksum verification prior to
CAS commit, and authoritative `COMMIT_RECEIPT` disaster-recovery snapshots.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import secrets
import sqlite3
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from mrnax.cloud.drive_rest_v3 import DRIVE_FOLDER_MIME, DrivePermissionError, DriveRestV3Client


class StaleLeaseConflictError(RuntimeError):
    """Raised when a stale, expired, unauthenticated, or unverified worker attempts to commit (HTTP 409)."""

    http_status: int = 409


class _ExpiredLeaseConflict(StaleLeaseConflictError):
    """Lease ended before the reservation or receipt publication."""


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
        require_drive_verifier: bool = False,
        runs_root_id: str | None = None,
        local_receipts_path: Path | None = None,
    ) -> None:
        self.db_path = db_path
        self.drive_verifier = drive_verifier
        self.commits_snapshot_folder_id = commits_snapshot_folder_id
        self.require_drive_verifier = require_drive_verifier
        self.runs_root_id = runs_root_id
        self.local_receipts_path = local_receipts_path or db_path.parent / "control_plane_commit_receipts.sqlite3"
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
            if "receipt_payload" not in {row["name"] for row in conn.execute("PRAGMA table_info(executions)")}:
                conn.execute("ALTER TABLE executions ADD COLUMN receipt_payload BLOB")
            self._archive_legacy_local_winners(conn)
            conn.execute("COMMIT;")

    def _canonical_v1_receipt_bytes(self, payload: bytes) -> bytes:
        doc = json.loads(payload)
        fields = (
            "execution_id",
            "submission_id",
            "request_digest",
            "intent_id",
            "winning_attempt_id",
            "lease_epoch",
            "winning_sealed_file_id",
            "winning_sealed_sha256",
            "committed_at",
        )
        v1_doc = {field: doc[field] for field in fields}
        v1_doc["schema"] = "mrnax.commit_receipt.v1"
        return self._receipt_bytes(v1_doc)

    def _same_committed_winner(self, a: bytes, b: bytes) -> bool:
        if a == b:
            return True
        try:
            return self._canonical_v1_receipt_bytes(a) == self._canonical_v1_receipt_bytes(b)
        except Exception:
            return False

    def _archive_legacy_local_winners(self, conn: sqlite3.Connection) -> None:
        for row in conn.execute("SELECT * FROM executions WHERE status='COMMITTED'"):
            payload = row["receipt_payload"]
            if payload is None:
                fields = ("execution_id", "submission_id", "request_digest", "intent_id", "winning_attempt_id",
                          "lease_epoch", "winning_sealed_file_id", "winning_sealed_sha256")
                doc = {field: row[field] for field in fields}
                doc.update(schema="mrnax.commit_receipt.v1", committed_at=row["updated_at"])
                payload = self._receipt_bytes(doc)
                conn.execute("UPDATE executions SET receipt_payload=? WHERE execution_id=?", (payload, row["execution_id"]))
            if self.commits_snapshot_folder_id is not None:
                if self.drive_verifier is not None:
                    existing = self._find_receipt(row["execution_id"], payload)
                    if existing is None:
                        payload = self._canonical_v1_receipt_bytes(payload)
                        conn.execute(
                            "UPDATE executions SET receipt_payload=? WHERE execution_id=?",
                            (payload, row["execution_id"]),
                        )
                        conn.commit()
                        if self.local_receipts_path.exists():
                            with self._local_receipts() as local:
                                local.execute("BEGIN IMMEDIATE")
                                local.execute(
                                    "INSERT OR REPLACE INTO receipts VALUES (?, ?)",
                                    (row["execution_id"], payload),
                                )
                        conn.execute("BEGIN IMMEDIATE")
                        self.drive_verifier.upload_verified_bytes(
                            self.commits_snapshot_folder_id,
                            self._receipt_name(row["execution_id"]),
                            payload,
                            mime_type="application/json",
                        )
                    elif not self._same_committed_winner(existing, payload):
                        raise StaleLeaseConflictError(f"Conflicting receipt contents for {row['execution_id']}")
                    elif existing != payload:
                        conn.execute(
                            "UPDATE executions SET receipt_payload=? WHERE execution_id=?",
                            (existing, row["execution_id"]),
                        )
                        if self.local_receipts_path.exists():
                            with self._local_receipts() as local:
                                local.execute("BEGIN IMMEDIATE")
                                local.execute(
                                    "INSERT OR REPLACE INTO receipts VALUES (?, ?)",
                                    (row["execution_id"], existing),
                                )
            else:
                with self._local_receipts() as local:
                    local.execute("INSERT OR IGNORE INTO receipts VALUES (?, ?)", (row["execution_id"], payload))
                    archived = local.execute("SELECT payload FROM receipts WHERE execution_id=?",
                                             (row["execution_id"],)).fetchone()[0]
                    if archived != payload:
                        raise StaleLeaseConflictError(f"Conflicting local receipt contents for {row['execution_id']}")

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
            if row["status"] == "RUNNING" and row["receipt_payload"] is not None:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute("SELECT * FROM executions WHERE execution_id=?", (execution_id,)).fetchone()
                if row["status"] == "RUNNING" and row["receipt_payload"] is not None:
                    landed = self._find_receipt(execution_id, row["receipt_payload"])
                    if landed is not None:
                        self._finalize(conn, row, landed)
                        row = conn.execute("SELECT * FROM executions WHERE execution_id=?", (execution_id,)).fetchone()
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

                exec_id = f"exec_{uuid.uuid4().hex}"
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
        lease_token: str | None = None,
        ttl_seconds: int = 300,
        now_ts: float | None = None,
    ) -> LeaseGrant:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            try:
                row = conn.execute(
                    "SELECT * FROM executions WHERE execution_id = ?", (execution_id,)
                ).fetchone()
                if row is None:
                    conn.execute("ROLLBACK;")
                    raise KeyError(f"Unknown execution_id '{execution_id}'")
                if row["status"] == "RUNNING" and row["receipt_payload"] is not None:
                    landed = self._find_receipt(execution_id, row["receipt_payload"])
                    if landed is not None:
                        self._finalize(conn, row, landed)
                        conn.commit()
                        raise StaleLeaseConflictError(f"Execution '{execution_id}' is already COMMITTED")
                ts = time.time() if now_ts is None else now_ts
                expires = ts + ttl_seconds
                if row["status"] == "COMMITTED":
                    conn.execute("ROLLBACK;")
                    raise StaleLeaseConflictError(f"Execution '{execution_id}' is already COMMITTED")

                cur_worker = row["current_worker_id"]
                cur_exp = row["lease_expires_at"]
                cur_attempt = row["current_attempt_id"]
                cur_epoch = int(row["lease_epoch"])
                cur_token = row["current_lease_token"]

                if row["status"] == "RUNNING" and cur_exp is not None and ts < float(cur_exp):
                    if (
                        cur_worker == worker_id
                        and cur_attempt is not None
                        and cur_token is not None
                        and lease_token is not None
                        and hmac.compare_digest(str(cur_token), lease_token)
                    ):
                        if row["receipt_payload"] is not None:
                            raise StaleLeaseConflictError("A commit is pending for this lease; retry the commit")
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
                        f"Execution '{execution_id}' is actively leased by '{cur_worker}' until {cur_exp} (valid lease_token required to renew)"
                    )

                if row["receipt_payload"] is not None and self.commits_snapshot_folder_id is not None:
                    self._revoke_inflight_receipt(row["receipt_payload"])
                    ts = time.time() if now_ts is None else now_ts
                    expires = ts + ttl_seconds
                new_epoch = cur_epoch + 1
                new_attempt = f"att_{execution_id}_{new_epoch:03d}"
                new_token = f"lzk_{secrets.token_urlsafe(18)}"
                conn.execute(
                    """
                    UPDATE executions
                       SET status = 'RUNNING',
                           lease_epoch = ?,
                           current_attempt_id = ?,
                           current_worker_id = ?,
                           current_lease_token = ?,
                           receipt_payload = NULL,
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
            if self.require_drive_verifier:
                raise StaleLeaseConflictError(
                    "CAS rejected: production control plane requires Drive seal verification, "
                    "but drive_verifier is not configured."
                )
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
        raw_epoch = doc.get("lease_epoch")
        if (
            doc.get("execution_id") != execution_id
            or doc.get("attempt_id") != attempt_id
            or isinstance(raw_epoch, bool)
            or not isinstance(raw_epoch, int)
            or raw_epoch != lease_epoch
        ):
            raise StaleLeaseConflictError("Drive seal metadata does not match execution/attempt/epoch")
        arts = doc.get("artifacts")
        if not isinstance(arts, list) or len(arts) == 0:
            raise StaleLeaseConflictError("Drive seal must list at least one verified artifact")
        seal_meta = self.drive_verifier.transport.get_metadata(sealed_file_id)
        if seal_meta.get("name") != "SEALED.json":
            raise StaleLeaseConflictError("Drive seal must be named SEALED.json")
        parent_ids = seal_meta.get("parents")
        if not isinstance(parent_ids, list) or len(parent_ids) != 1:
            raise StaleLeaseConflictError("Drive seal is missing its unique attempt folder")
        attempt_folder = parent_ids[0]
        folder_id = attempt_folder
        for expected_name in (attempt_id, "attempts", execution_id):
            folder = self.drive_verifier.transport.get_metadata(folder_id)
            parents = folder.get("parents")
            if (folder.get("name") != expected_name or folder.get("mimeType") != DRIVE_FOLDER_MIME
                    or not isinstance(parents, list) or len(parents) != 1):
                raise StaleLeaseConflictError("Drive seal is outside the designated attempt folder")
            matches = [entry["id"] for entry in self.drive_verifier.transport.list_children(
                parent_id=parents[0], name_exact=expected_name
            ) if entry.get("mimeType") == DRIVE_FOLDER_MIME]
            if matches != [folder_id]:
                raise StaleLeaseConflictError("Drive attempt folder ancestry is ambiguous")
            folder_id = parents[0]
        if self.runs_root_id is not None and folder_id != self.runs_root_id:
            raise StaleLeaseConflictError("Drive seal is outside the designated runs root")
        if self.runs_root_id is None:
            transport = self.drive_verifier.transport
            if self.commits_snapshot_folder_id is not None:
                snapshot_parents = transport.get_metadata(self.commits_snapshot_folder_id).get("parents")
                if not isinstance(snapshot_parents, list) or len(snapshot_parents) != 1:
                    raise StaleLeaseConflictError("Missing workspace parent for the configured Drive receipt folder")
                workspace_id = snapshot_parents[0]
            elif self.require_drive_verifier:
                raise StaleLeaseConflictError("Configure MRNAX_DRIVE_RUNS_ROOT_ID to identify the attempt workspace")
            else:
                workspace_id = "root"
            roots = [f["id"] for f in transport.list_children(parent_id=workspace_id, name_exact="runs")
                     if f.get("mimeType") == DRIVE_FOLDER_MIME]
            if roots != [folder_id]:
                raise StaleLeaseConflictError("Drive seal is outside the designated workspace runs folder")
        for art in arts:
            fid = str(art.get("file_id") or "").strip()
            expected_md5 = str(art.get("md5") or "").strip()
            if not fid or not expected_md5:
                raise StaleLeaseConflictError(f"Drive seal artifact '{art.get('name')}' missing file_id or md5")
            meta = self.drive_verifier.transport.get_metadata(fid)
            actual_md5 = str(meta.get("md5Checksum") or "").strip()
            if not actual_md5 or actual_md5 != expected_md5:
                raise StaleLeaseConflictError(f"Drive seal artifact md5 mismatch for {art.get('name')}")
            if meta.get("parents") != [attempt_folder] or art.get("parent_id") != attempt_folder:
                raise StaleLeaseConflictError(
                    f"Drive seal artifact '{art.get('name')}' does not reside in the sealed attempt folder"
                )

    @staticmethod
    def _receipt_bytes(doc: dict) -> bytes:
        return (json.dumps(doc, indent=2, sort_keys=True) + "\n").encode()

    @staticmethod
    def _receipt_name(execution_id: str) -> str:
        return f"COMMIT_RECEIPT_{execution_id}.json"

    @staticmethod
    def _require_valid_lease(row: sqlite3.Row, now: float) -> None:
        if row["lease_expires_at"] is None or now >= row["lease_expires_at"]:
            raise _ExpiredLeaseConflict(f"CAS rejected: lease expired at {row['lease_expires_at']} (now={now})")

    @staticmethod
    def _drive_created_at(metadata: dict) -> float:
        stamp = metadata.get("createdTime")
        if not isinstance(stamp, str):
            raise StaleLeaseConflictError("Missing Drive creation time")
        try:
            created = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        except ValueError as exc:
            raise StaleLeaseConflictError("Invalid Drive creation time") from exc
        if created.tzinfo is None:
            raise StaleLeaseConflictError("Missing Drive timestamp timezone")
        return created.timestamp()

    def _publication_deadline(self, row: sqlite3.Row, clock: Callable[[], float]) -> float:
        if self.commits_snapshot_folder_id is None:
            return row["lease_expires_at"]
        if self.drive_verifier is None:
            raise StaleLeaseConflictError("Cannot timestamp a Drive commit without a Drive verifier")
        # A server-timestamped prepare binds remaining lease duration to the same clock as the receipt.
        stamp = self.drive_verifier.upload_verified_bytes(
            self.commits_snapshot_folder_id, f"COMMIT_WINDOW_{row['current_attempt_id']}.txt",
            row["current_attempt_id"].encode(), mime_type="text/plain",
        )
        server_time = self._drive_created_at(self.drive_verifier.transport.get_metadata(stamp.file_id))
        local_now = clock()
        self._require_valid_lease(row, local_now)
        return server_time + (row["lease_expires_at"] - local_now)

    @staticmethod
    def _revocation_name(payload: bytes) -> str:
        return f"REVOKED_RECEIPT_{hashlib.sha256(payload).hexdigest()}.json"

    def _revoke_inflight_receipt(self, payload: bytes) -> None:
        # Persist the fence before regrant so a delayed POST cannot win even after SQLite is lost.
        name = self._revocation_name(payload)
        transport = self.drive_verifier.transport
        entries = transport.list_children(parent_id=self.commits_snapshot_folder_id, name_exact=name)
        if entries:
            if any(transport.download_bytes(entry["id"]) != payload for entry in entries):
                raise StaleLeaseConflictError("Conflicting Drive revocation contents")
            return
        self.drive_verifier.upload_verified_bytes(self.commits_snapshot_folder_id, name, payload,
                                                 mime_type="application/json")

    def _drive_receipts(self, drive: DriveRestV3Client, folder: str, execution_id: str | None, expected_payload: bytes | None = None) -> dict[str, bytes]:
        """Drive creation time fences delayed writes even after the publisher crashes."""
        name = self._receipt_name(execution_id) if execution_id else None
        expected_epoch = json.loads(expected_payload)["lease_epoch"] if expected_payload is not None else None
        winners: dict[str, bytes] = {}
        entries = drive.transport.list_children(parent_id=folder, name_exact=name)
        revocations: dict[str, list[dict]] = {}
        for entry in entries:
            if entry["name"].startswith("REVOKED_RECEIPT_"):
                revocations.setdefault(entry["name"], []).append(entry)
        for entry in entries:
            if not entry["name"].startswith("COMMIT_RECEIPT_"):
                continue
            raw = drive.transport.download_bytes(entry["id"])
            try:
                doc = json.loads(raw)
                key = doc["execution_id"]
                if entry["name"] != self._receipt_name(key):
                    raise ValueError("filename and execution ID differ")
                if doc["schema"] == "mrnax.commit_receipt.v2":
                    revoked_name = self._revocation_name(raw)
                    if execution_id is not None:
                        fences = drive.transport.list_children(parent_id=folder, name_exact=revoked_name)
                    else:
                        fences = revocations.get(revoked_name, [])
                    if fences:
                        if any(drive.transport.download_bytes(fence["id"]) != raw for fence in fences):
                            raise StaleLeaseConflictError("Conflicting Drive revocation contents")
                        continue
                    meta = entry if entry.get("createdTime") else drive.transport.get_metadata(entry["id"])
                    created_at = self._drive_created_at(meta)
                    deadline = doc["publish_before_utc"]
                    if not math.isfinite(deadline):
                        raise ValueError("invalid Drive publication deadline")
                    if created_at >= deadline:
                        continue
                elif doc["schema"] != "mrnax.commit_receipt.v1":
                    raise ValueError("unknown schema")
                if (
                    expected_epoch is not None
                    and doc["lease_epoch"] >= expected_epoch
                    and expected_payload is not None
                    and not self._same_committed_winner(expected_payload, raw)
                ):
                    raise StaleLeaseConflictError(f"Conflicting receipt contents for {key}")
            except (ValueError, AttributeError, KeyError, TypeError) as exc:
                raise StaleLeaseConflictError(f"Invalid Drive commit receipt {entry['name']}: {exc}") from exc
            if key in winners and winners[key] != raw:
                raise StaleLeaseConflictError(f"Conflicting Drive receipt contents for {key}")
            winners[key] = raw
        return winners

    def _local_receipts(self) -> sqlite3.Connection:
        self.local_receipts_path.parent.mkdir(parents=True, exist_ok=True)
        local = sqlite3.connect(self.local_receipts_path, timeout=30.0)
        local.execute("PRAGMA synchronous=FULL")
        local.execute("CREATE TABLE IF NOT EXISTS receipts (execution_id TEXT PRIMARY KEY, payload BLOB NOT NULL)")
        return local

    def _find_receipt(self, execution_id: str, expected_payload: bytes | None = None) -> bytes | None:
        if self.commits_snapshot_folder_id is not None:
            if self.drive_verifier is None:
                raise StaleLeaseConflictError("Drive receipt folder is configured without a Drive verifier")
            return self._drive_receipts(self.drive_verifier, self.commits_snapshot_folder_id, execution_id, expected_payload).get(execution_id)
        if not self.local_receipts_path.exists():
            return None
        with self._local_receipts() as local:
            row = local.execute("SELECT payload FROM receipts WHERE execution_id = ?", (execution_id,)).fetchone()
            return row[0] if row else None

    def _publish(self, row: sqlite3.Row, payload: bytes, clock: Callable[[], float]) -> bytes:
        key = row["execution_id"]
        existing = self._find_receipt(key, payload)
        if existing is not None:
            if existing != payload:
                raise StaleLeaseConflictError(f"Conflicting receipt contents for {key}")
            return existing

        if self.commits_snapshot_folder_id is not None:
            self._require_valid_lease(row, clock())
            self.drive_verifier.upload_verified_bytes(
                self.commits_snapshot_folder_id, self._receipt_name(key), payload, mime_type="application/json"
            )
            landed = self._find_receipt(key, payload)
            if landed is None:
                raise _ExpiredLeaseConflict("Drive receipt was not published before the lease expired")
            if landed != payload:
                raise StaleLeaseConflictError(f"Conflicting receipt contents for {key}")
            return landed

        with self._local_receipts() as local:
            local.execute("BEGIN IMMEDIATE")
            self._require_valid_lease(row, clock())
            local.execute("INSERT OR IGNORE INTO receipts VALUES (?, ?)", (key, payload))
            landed = local.execute("SELECT payload FROM receipts WHERE execution_id = ?", (key,)).fetchone()[0]
            if landed != payload:
                raise StaleLeaseConflictError(f"Conflicting local receipt contents for {key}")
        return landed

    @staticmethod
    def _finalize(conn: sqlite3.Connection, row: sqlite3.Row, payload: bytes) -> None:
        if row["receipt_payload"] != payload:
            raise StaleLeaseConflictError(f"Receipt contents differ from the reserved commit for {row['execution_id']}")
        doc = json.loads(payload)
        conn.execute(
            """UPDATE executions SET status='COMMITTED', winning_attempt_id=?, winning_sealed_file_id=?,
                    winning_sealed_sha256=?, updated_at=? WHERE execution_id=?""",
            (doc["winning_attempt_id"], doc["winning_sealed_file_id"], doc["winning_sealed_sha256"],
             doc["committed_at"], row["execution_id"]),
        )
        conn.execute(
            "UPDATE attempts SET sealed_file_id=?, commit_outcome='COMMITTED_200' WHERE attempt_id=?",
            (doc["winning_sealed_file_id"], doc["winning_attempt_id"]),
        )

    @contextmanager
    def _commit_connection(
        self, execution_id: str, attempt_id: str, lease_token: str | None, sealed_file_id: str
    ) -> Iterator[sqlite3.Connection]:
        with self._connect() as conn:
            try:
                yield conn
            except StaleLeaseConflictError as exc:
                conn.rollback()
                outcome = "EXPIRED_409" if isinstance(exc, _ExpiredLeaseConflict) else "REJECTED_409"
                conn.execute(
                    """UPDATE attempts SET commit_outcome=?, sealed_file_id=?
                       WHERE execution_id=? AND attempt_id=? AND lease_token=?
                       AND COALESCE(commit_outcome, '') != 'COMMITTED_200'""",
                    (outcome, sealed_file_id, execution_id, attempt_id, lease_token),
                )
                raise

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
        try:
            self._verify_drive_seal_before_commit(
                execution_id=execution_id, attempt_id=attempt_id, lease_epoch=lease_epoch,
                sealed_file_id=sealed_file_id, sealed_sha256=sealed_sha256,
            )
        except (DrivePermissionError, ValueError, TypeError, AttributeError) as exc:
            raise StaleLeaseConflictError(f"Cannot verify Drive seal or attempt folder: {exc}") from exc
        clock = time.time if now_ts is None else lambda: now_ts
        receipt = CommitReceipt(execution_id, "COMMITTED", attempt_id, sealed_file_id, lease_epoch)
        request = {
            "execution_id": execution_id, "winning_attempt_id": attempt_id, "lease_epoch": lease_epoch,
            "winning_sealed_file_id": sealed_file_id, "winning_sealed_sha256": sealed_sha256,
        }
        with self._commit_connection(execution_id, attempt_id, lease_token, sealed_file_id) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM executions WHERE execution_id=?", (execution_id,)).fetchone()
            if row is None:
                raise StaleLeaseConflictError(f"CAS rejected: unknown execution '{execution_id}'")
            if row["current_lease_token"] and (
                not lease_token or not hmac.compare_digest(row["current_lease_token"], lease_token)
            ):
                raise StaleLeaseConflictError("CAS rejected: invalid worker lease_token")
            if row["current_attempt_id"] != attempt_id or row["lease_epoch"] != lease_epoch:
                raise StaleLeaseConflictError(f"CAS rejected: stale attempt {attempt_id}")
            payload = row["receipt_payload"]
            if payload is None and row["status"] == "COMMITTED":
                # Pre-journal winners already have an immutable decision in this database.
                if (row["winning_attempt_id"], row["winning_sealed_file_id"], row["winning_sealed_sha256"]) != (
                    attempt_id, sealed_file_id, sealed_sha256
                ):
                    raise StaleLeaseConflictError("Commit differs from the existing winner")
                doc = {**request, "schema": "mrnax.commit_receipt.v1", "committed_at": row["updated_at"]}
            elif payload is None:
                if row["status"] != "RUNNING":
                    raise StaleLeaseConflictError(f"CAS rejected: execution is {row['status']}")
                ts = clock()
                self._require_valid_lease(row, ts)
                deadline = self._publication_deadline(row, clock)
                doc = {**request, "schema": "mrnax.commit_receipt.v2", "committed_at": ts,
                       "publish_before_utc": deadline}
            if payload is None:
                doc.update({key: row[key] for key in ("submission_id", "request_digest", "intent_id")})
                payload = self._receipt_bytes(doc)
                conn.execute("UPDATE executions SET receipt_payload=? WHERE execution_id=?", (payload, execution_id))
            else:
                expected = json.loads(payload)
                if any(expected.get(key) != value for key, value in request.items()):
                    raise StaleLeaseConflictError("Commit differs from the reserved receipt")
            conn.commit()

            # The reservation survives an ambiguous network failure. A lease takeover reconciles it first.
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM executions WHERE execution_id=?", (execution_id,)).fetchone()
            if row["receipt_payload"] != payload:
                raise StaleLeaseConflictError("CAS rejected: commit reservation was superseded")
            if row["status"] == "COMMITTED":
                existing = self._find_receipt(execution_id, payload)
                if existing is None:
                    if self.commits_snapshot_folder_id is not None:
                        if self.drive_verifier is None:
                            raise StaleLeaseConflictError("Drive receipt folder is configured without a Drive verifier")
                        payload = self._canonical_v1_receipt_bytes(payload)
                        conn.execute(
                            "UPDATE executions SET receipt_payload=? WHERE execution_id=?",
                            (payload, execution_id),
                        )
                        conn.commit()
                        if self.local_receipts_path.exists():
                            with self._local_receipts() as local:
                                local.execute("BEGIN IMMEDIATE")
                                local.execute("INSERT OR REPLACE INTO receipts VALUES (?, ?)", (execution_id, payload))
                        conn.execute("BEGIN IMMEDIATE")
                        self.drive_verifier.upload_verified_bytes(
                            self.commits_snapshot_folder_id,
                            self._receipt_name(execution_id),
                            payload,
                            mime_type="application/json",
                        )
                        existing = self._find_receipt(execution_id, payload)
                    else:
                        with self._local_receipts() as local:
                            local.execute("BEGIN IMMEDIATE")
                            local.execute("INSERT OR IGNORE INTO receipts VALUES (?, ?)", (execution_id, payload))
                        existing = self._find_receipt(execution_id, payload)
                if existing is None or not self._same_committed_winner(existing, payload):
                    raise StaleLeaseConflictError("Conflicting receipt contents for committed winner")
                if existing != payload:
                    conn.execute(
                        "UPDATE executions SET receipt_payload=? WHERE execution_id=?",
                        (existing, execution_id),
                    )
                return receipt
            self._finalize(conn, row, self._publish(row, payload, clock))
        return receipt

    def reconcile_from_drive_seals(self, drive: DriveRestV3Client, runs_root_id: str) -> int:
        """Compatibility entry point for receipt recovery in previously deployed workers."""
        return self.restore_committed_receipts(drive=drive)

    def restore_committed_receipts(self, *, drive: DriveRestV3Client | None = None) -> int:
        """Restore the configured authority. The local journal is independent of the execution database."""
        if self.commits_snapshot_folder_id:
            drive = drive or self.drive_verifier
            if drive is None:
                raise StaleLeaseConflictError("Cannot recover Drive receipts without a Drive verifier")
            winners = self._drive_receipts(drive, self.commits_snapshot_folder_id, None)
        elif self.local_receipts_path.exists():
            with self._local_receipts() as local:
                winners = dict(local.execute("SELECT execution_id, payload FROM receipts"))
        else:
            winners = {}

        reconciled = 0
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            for key, payload in winners.items():
                doc = json.loads(payload)
                identity = (doc["lease_epoch"], doc["winning_attempt_id"], doc["winning_sealed_file_id"],
                            doc["winning_sealed_sha256"])
                row = conn.execute("SELECT * FROM executions WHERE execution_id=?", (key,)).fetchone()
                if row is not None and row["status"] == "COMMITTED":
                    if identity != (row["lease_epoch"], row["winning_attempt_id"], row["winning_sealed_file_id"],
                                    row["winning_sealed_sha256"]) or row["receipt_payload"] not in (None, payload):
                        raise StaleLeaseConflictError(f"Recovered receipt disagrees with committed winner {key}")
                    continue
                common = (doc["submission_id"], doc["request_digest"], doc["intent_id"])
                if row is None:
                    conn.execute(
                        """INSERT INTO executions (execution_id, submission_id, request_digest, intent_id, status,
                           created_at, updated_at) VALUES (?, ?, ?, ?, 'QUEUED', ?, ?)""",
                        (key, *common, doc["committed_at"], doc["committed_at"]),
                    )
                elif common != (row["submission_id"], row["request_digest"], row["intent_id"]) or (
                    row["lease_epoch"] == doc["lease_epoch"] and row["receipt_payload"] not in (None, payload)
                ):
                    raise StaleLeaseConflictError(f"Recovered receipt differs from local reservation {key}")
                conn.execute(
                    """UPDATE executions SET status='COMMITTED', lease_epoch=?, winning_attempt_id=?,
                       winning_sealed_file_id=?, winning_sealed_sha256=?, current_attempt_id=?, current_worker_id=NULL,
                       current_lease_token=NULL, lease_expires_at=NULL, receipt_payload=?, updated_at=? WHERE execution_id=?""",
                    (*identity, doc["winning_attempt_id"], payload, doc["committed_at"], key),
                )
                conn.execute("UPDATE attempts SET sealed_file_id=?, commit_outcome='COMMITTED_200' WHERE attempt_id=?",
                             (doc["winning_sealed_file_id"], doc["winning_attempt_id"]))
                reconciled += 1
        return reconciled

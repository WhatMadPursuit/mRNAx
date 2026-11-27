"""Google Drive REST API v3 client (ZERO FUSE `drive.mount`) with server md5Checksum verification."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

import httpx

DRIVE_FOLDER_MIME = "application/vnd.google-apps.folder"


class DriveIntegrityMismatchError(RuntimeError):
    """Raised when Google Drive server md5Checksum disagrees with local payload digest."""


class DrivePermissionError(RuntimeError):
    """Raised on HTTP 403 / 404 from Google Drive REST API v3."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(f"HTTP {status_code}: {message}")
        self.status_code = status_code


@dataclass(frozen=True)
class DriveFileReceipt:
    file_id: str
    name: str
    parent_id: str
    mime_type: str
    bytes: int
    md5: str
    sha256: str


@dataclass(frozen=True)
class PermissionAuditResult:
    workspace_read_write_ok: bool
    data_hub_read_ok: bool
    outside_canary_unreadable: bool
    outside_canary_http_status: int
    details: str


class DriveV3TransportProtocol(Protocol):
    def create_file(
        self, *, parent_id: str, name: str, mime_type: str, content: bytes | None = None
    ) -> dict[str, Any]: ...
    def list_children(self, *, parent_id: str, name_exact: str | None = None) -> list[dict[str, Any]]: ...
    def get_metadata(self, file_id: str) -> dict[str, Any]: ...
    def download_bytes(self, file_id: str) -> bytes: ...
    def delete_file(self, file_id: str) -> None: ...


class InMemoryDriveV3Transport:
    """Deterministic test transport modeling Google Drive's duplicate-name-allowed ID semantics."""

    def __init__(
        self,
        *,
        readable_ids: set[str] | None = None,
        writable_ids: set[str] | None = None,
        forbidden_ids: dict[str, int] | None = None,
    ) -> None:
        self._nodes: dict[str, dict[str, Any]] = {
            "root": {"id": "root", "name": "root", "mimeType": DRIVE_FOLDER_MIME, "parents": []}
        }
        self._blobs: dict[str, bytes] = {}
        self._readable = readable_ids
        self._writable = writable_ids
        self._forbidden = forbidden_ids or {}

    def _check_read(self, file_id: str) -> None:
        if file_id in self._forbidden:
            raise DrivePermissionError(self._forbidden[file_id], f"Access denied for {file_id}")
        if self._readable is not None and file_id not in self._readable and file_id not in self._nodes:
            raise DrivePermissionError(404, f"File not found: {file_id}")
        if file_id not in self._nodes:
            raise DrivePermissionError(404, f"File not found: {file_id}")

    def _check_write(self, parent_id: str) -> None:
        if parent_id in self._forbidden:
            raise DrivePermissionError(self._forbidden[parent_id], f"Write denied for {parent_id}")
        if self._writable is not None and parent_id not in self._writable:
            raise DrivePermissionError(403, f"Insufficient file permissions on {parent_id}")

    def create_file(
        self, *, parent_id: str, name: str, mime_type: str, content: bytes | None = None
    ) -> dict[str, Any]:
        self._check_read(parent_id)
        self._check_write(parent_id)
        fid = f"drv_{uuid.uuid4().hex[:16]}"
        meta: dict[str, Any] = {
            "id": fid,
            "name": name,
            "mimeType": mime_type,
            "parents": [parent_id],
            "createdTime": datetime.fromtimestamp(time.time(), timezone.utc).isoformat(),
        }
        if content is not None and mime_type != DRIVE_FOLDER_MIME:
            meta["md5Checksum"] = hashlib.md5(content).hexdigest()
            meta["size"] = str(len(content))
            self._blobs[fid] = bytes(content)
        self._nodes[fid] = meta
        if self._writable is not None and parent_id in self._writable:
            self._writable.add(fid)
        if self._readable is not None:
            self._readable.add(fid)
        return dict(meta)

    def list_children(self, *, parent_id: str, name_exact: str | None = None) -> list[dict[str, Any]]:
        self._check_read(parent_id)
        out = []
        for meta in self._nodes.values():
            if parent_id in meta.get("parents", []):
                if name_exact is None or meta["name"] == name_exact:
                    out.append(dict(meta))
        return out

    def get_metadata(self, file_id: str) -> dict[str, Any]:
        self._check_read(file_id)
        return dict(self._nodes[file_id])

    def download_bytes(self, file_id: str) -> bytes:
        self._check_read(file_id)
        return self._blobs.get(file_id, b"")

    def delete_file(self, file_id: str) -> None:
        meta = self.get_metadata(file_id)
        for p in meta.get("parents", []):
            self._check_write(p)
        self._nodes.pop(file_id, None)
        self._blobs.pop(file_id, None)


class HttpDriveV3Transport:
    """Production HTTPS Google Drive REST API v3 transport (zero FUSE)."""

    FILES_API = "https://www.googleapis.com/drive/v3/files"
    UPLOAD_API = "https://www.googleapis.com/upload/drive/v3/files"

    def __init__(self, token_provider: Callable[[], str], timeout: float = 60.0) -> None:
        self._token_provider = token_provider
        self._timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._token_provider()}"}

    def _raise_for_drive_error(self, resp: httpx.Response) -> None:
        if resp.status_code in {401, 403, 404}:
            raise DrivePermissionError(resp.status_code, resp.text[:300])
        resp.raise_for_status()

    def create_file(
        self, *, parent_id: str, name: str, mime_type: str, content: bytes | None = None
    ) -> dict[str, Any]:
        metadata = {"name": name, "mimeType": mime_type, "parents": [parent_id]}
        with httpx.Client(timeout=self._timeout) as client:
            if mime_type == DRIVE_FOLDER_MIME or content is None:
                resp = client.post(
                    self.FILES_API,
                    params={"fields": "id,name,mimeType,parents,md5Checksum,size,createdTime", "supportsAllDrives": "true"},
                    headers=self._headers(),
                    json=metadata,
                )
                self._raise_for_drive_error(resp)
                return resp.json()

            boundary = f"mrnax_boundary_{uuid.uuid4().hex}"
            meta_bytes = json.dumps(metadata).encode("utf-8")
            body = (
                f"--{boundary}\r\n"
                "Content-Type: application/json; charset=UTF-8\r\n\r\n"
            ).encode("utf-8") + meta_bytes + (
                f"\r\n--{boundary}\r\n"
                f"Content-Type: {mime_type}\r\n\r\n"
            ).encode("utf-8") + content + f"\r\n--{boundary}--\r\n".encode("utf-8")

            headers = self._headers()
            headers["Content-Type"] = f"multipart/related; boundary={boundary}"
            resp = client.post(
                self.UPLOAD_API,
                params={
                    "uploadType": "multipart",
                    "fields": "id,name,mimeType,parents,md5Checksum,size,createdTime",
                    "supportsAllDrives": "true",
                },
                headers=headers,
                content=body,
            )
            self._raise_for_drive_error(resp)
            return resp.json()

    def list_children(self, *, parent_id: str, name_exact: str | None = None) -> list[dict[str, Any]]:
        q_parts = [f"'{parent_id}' in parents", "trashed = false"]
        if name_exact is not None:
            escaped = name_exact.replace("'", "\\'")
            q_parts.append(f"name = '{escaped}'")
        all_files: list[dict[str, Any]] = []
        page_token: str | None = None
        with httpx.Client(timeout=self._timeout) as client:
            while True:
                params: dict[str, Any] = {
                    "q": " and ".join(q_parts),
                    "fields": "nextPageToken,files(id,name,mimeType,parents,md5Checksum,size,createdTime)",
                    "pageSize": 200,
                    "supportsAllDrives": "true",
                    "includeItemsFromAllDrives": "true",
                }
                if page_token:
                    params["pageToken"] = page_token
                resp = client.get(self.FILES_API, params=params, headers=self._headers())
                self._raise_for_drive_error(resp)
                body = resp.json()
                all_files.extend(body.get("files", []))
                page_token = body.get("nextPageToken")
                if not page_token:
                    break
        return all_files

    def get_metadata(self, file_id: str) -> dict[str, Any]:
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.get(
                f"{self.FILES_API}/{file_id}",
                params={
                    "fields": "id,name,mimeType,parents,md5Checksum,size,createdTime,capabilities(canEdit)",
                    "supportsAllDrives": "true",
                },
                headers=self._headers(),
            )
            self._raise_for_drive_error(resp)
            return resp.json()

    def download_bytes(self, file_id: str) -> bytes:
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.get(
                f"{self.FILES_API}/{file_id}",
                params={"alt": "media", "supportsAllDrives": "true"},
                headers=self._headers(),
            )
            self._raise_for_drive_error(resp)
            return resp.content

    def delete_file(self, file_id: str) -> None:
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.delete(
                f"{self.FILES_API}/{file_id}",
                params={"supportsAllDrives": "true"},
                headers=self._headers(),
            )
            self._raise_for_drive_error(resp)


class GdriveCliDriveV3Transport:
    """Managed-auth Google Drive REST v3 transport delegating HTTPS API calls to bare `gdrive` CLI (zero FUSE)."""

    _RATE_LIMIT_TOKENS = ("Quota exceeded", "rateLimitExceeded", "userRateLimitExceeded", "429")
    _BACKOFF_SCHEDULE = (2.0, 5.0, 10.0, 20.0, 30.0)

    def __init__(self, gdrive_bin: str = "gdrive", timeout: float = 60.0) -> None:
        resolved = shutil.which(gdrive_bin)
        if not resolved:
            raise FileNotFoundError(f"gdrive CLI '{gdrive_bin}' not found on PATH")
        self._bin = resolved
        self._timeout = timeout

    @classmethod
    def _is_rate_limit(cls, err: str) -> bool:
        return any(tok in err for tok in cls._RATE_LIMIT_TOKENS)

    def _run_with_backoff(self, cmd: list[str]) -> subprocess.CompletedProcess[str]:
        for attempt, delay in enumerate((*self._BACKOFF_SCHEDULE, None)):
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
            if proc.returncode == 0:
                return proc
            err = (proc.stderr or proc.stdout).strip()
            if delay is not None and self._is_rate_limit(err):
                time.sleep(delay)
                continue
            return proc
        raise RuntimeError(f"Unreachable backoff loop for {cmd}")

    def _run_json(self, args: list[str]) -> Any:
        proc = self._run_with_backoff([self._bin, *args, "--json"])
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout).strip()
            if "403" in err and not self._is_rate_limit(err):
                raise DrivePermissionError(403, err[:300])
            if "404" in err or "notFound" in err:
                raise DrivePermissionError(404, err[:300])
            raise RuntimeError(f"gdrive {' '.join(args)} failed ({proc.returncode}): {err[:300]}")
        out = proc.stdout.strip()
        if not out or out == "null":
            return None
        if out[0] in "{[":
            return json.loads(out)
        m = re.search(r"\bID:\s*([A-Za-z0-9_-]+)", out)
        if m:
            return {"id": m.group(1)}
        raise RuntimeError(f"Unrecognized gdrive output for {' '.join(args)}: {out[:200]}")

    def create_file(
        self, *, parent_id: str, name: str, mime_type: str, content: bytes | None = None
    ) -> dict[str, Any]:
        if mime_type == DRIVE_FOLDER_MIME or content is None:
            res = self._run_json(["mutate", "mkdir", name, "--parent", parent_id])
            meta = dict(res) if isinstance(res, dict) else {"id": str(res), "name": name}
            meta.setdefault("mimeType", DRIVE_FOLDER_MIME)
            meta.setdefault("parents", [parent_id])
            return meta

        with tempfile.TemporaryDirectory(prefix="mrnax_drv_") as td:
            src_path = Path(td) / name
            src_path.write_bytes(content)
            res = self._run_json(
                ["mutate", "upload", str(src_path), "--parent", parent_id, "--mime-type", mime_type]
            )
            fid = str(res["id"])
            # Download back from Google Drive REST v3 server to verify landed payload md5Checksum
            landed = self.download_bytes(fid)
            md5_hex = hashlib.md5(landed).hexdigest()
            size_str = str(len(landed))
            meta = dict(res)
            meta["md5Checksum"] = md5_hex
            meta["size"] = size_str
            meta.setdefault("parents", [parent_id])
            return meta

    def list_children(self, *, parent_id: str, name_exact: str | None = None) -> list[dict[str, Any]]:
        rows = self._run_json(["readonly", "ls", parent_id]) or []
        out: list[dict[str, Any]] = []
        for r in rows:
            if name_exact is None or r.get("name") == name_exact:
                item = dict(r)
                item.setdefault("parents", [parent_id])
                out.append(item)
        return out

    def get_metadata(self, file_id: str) -> dict[str, Any]:
        meta = dict(self._run_json(["readonly", "info", file_id]))
        if meta.get("mimeType") != DRIVE_FOLDER_MIME and "md5Checksum" not in meta:
            landed = self.download_bytes(file_id)
            md5_hex = hashlib.md5(landed).hexdigest()
            size_str = str(len(landed))
            meta["md5Checksum"] = md5_hex
            meta["size"] = size_str
        return meta

    def download_bytes(self, file_id: str) -> bytes:
        with tempfile.TemporaryDirectory(prefix="mrnax_dl_") as td:
            dst = Path(td) / "payload.bin"
            proc = self._run_with_backoff([self._bin, "readonly", "download", file_id, "--out", str(dst)])
            if proc.returncode != 0:
                err = (proc.stderr or proc.stdout).strip()
                if "403" in err and not self._is_rate_limit(err):
                    raise DrivePermissionError(403, err[:300])
                if "404" in err or "notFound" in err:
                    raise DrivePermissionError(404, err[:300])
                raise RuntimeError(f"gdrive download {file_id} failed: {err[:300]}")
            return dst.read_bytes()

    def delete_file(self, file_id: str) -> None:
        proc = self._run_with_backoff([self._bin, "mutate", "trash", file_id])
        if proc.returncode != 0:
            raise RuntimeError(f"gdrive trash {file_id} failed: {(proc.stderr or proc.stdout)[:300]}")


class DriveRestV3Client:
    """High-integrity Google Drive v3 client for isolated per-attempt checkpoints and SEALED.json."""

    def __init__(self, transport: DriveV3TransportProtocol) -> None:
        self.transport = transport

    def create_folder(self, parent_id: str, name: str) -> str:
        existing = [
            f for f in self.transport.list_children(parent_id=parent_id, name_exact=name)
            if f.get("mimeType") == DRIVE_FOLDER_MIME
        ]
        if existing:
            return str(existing[0]["id"])
        created = self.transport.create_file(parent_id=parent_id, name=name, mime_type=DRIVE_FOLDER_MIME)
        return str(created["id"])

    def ensure_attempt_dir(self, runs_root_id: str, execution_id: str, attempt_id: str) -> str:
        exec_dir = self.create_folder(runs_root_id, execution_id)
        attempts_dir = self.create_folder(exec_dir, "attempts")
        return self.create_folder(attempts_dir, attempt_id)

    def upload_verified_bytes(
        self,
        parent_folder_id: str,
        name: str,
        payload: bytes,
        mime_type: str = "application/octet-stream",
    ) -> DriveFileReceipt:
        local_md5 = hashlib.md5(payload).hexdigest()
        local_sha256 = hashlib.sha256(payload).hexdigest()
        meta = self.transport.create_file(
            parent_id=parent_folder_id, name=name, mime_type=mime_type, content=payload
        )
        remote_md5 = meta.get("md5Checksum")
        if remote_md5 != local_md5:
            raise DriveIntegrityMismatchError(
                f"Drive md5Checksum mismatch for '{name}': local={local_md5} vs remote={remote_md5}"
            )
        return DriveFileReceipt(
            file_id=str(meta["id"]),
            name=name,
            parent_id=parent_folder_id,
            mime_type=mime_type,
            bytes=len(payload),
            md5=local_md5,
            sha256=local_sha256,
        )

    def write_attempt_seal(
        self,
        attempt_folder_id: str,
        *,
        execution_id: str,
        attempt_id: str,
        lease_epoch: int,
        receipts: list[DriveFileReceipt],
    ) -> DriveFileReceipt:
        # Re-verify every receipt exists on Drive with matching md5Checksum before sealing
        for r in receipts:
            remote_meta = self.transport.get_metadata(r.file_id)
            if remote_meta.get("md5Checksum") != r.md5:
                raise DriveIntegrityMismatchError(
                    f"Pre-seal verification failed on {r.name} ({r.file_id})"
                )
        seal_doc = {
            "schema": "mrnax.attempt_seal.v1",
            "execution_id": execution_id,
            "attempt_id": attempt_id,
            "lease_epoch": lease_epoch,
            "sealed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "artifacts": [asdict(r) for r in receipts],
        }
        payload = (json.dumps(seal_doc, indent=2, sort_keys=True) + "\n").encode("utf-8")
        return self.upload_verified_bytes(
            attempt_folder_id, "SEALED.json", payload, mime_type="application/json"
        )

    def verify_permission_boundary(
        self,
        workspace_folder_id: str,
        readonly_hub_id: str,
        outside_canary_file_id: str,
    ) -> PermissionAuditResult:
        # 1. Verify workspace read/write
        probe_bytes = f"mrnax_perm_probe_{uuid.uuid4().hex}".encode("utf-8")
        receipt = self.upload_verified_bytes(workspace_folder_id, "_perm_probe.txt", probe_bytes, "text/plain")
        readback = self.transport.download_bytes(receipt.file_id)
        ws_ok = readback == probe_bytes
        self.transport.delete_file(receipt.file_id)

        # 2. Verify Data Hub read access
        hub_meta = self.transport.get_metadata(readonly_hub_id)
        hub_ok = bool(hub_meta.get("id"))

        # 3. Verify outside-project canary is strictly unreadable (HTTP 403 or 404)
        canary_unreadable = False
        canary_status = 200
        try:
            self.transport.get_metadata(outside_canary_file_id)
        except DrivePermissionError as exc:
            canary_status = exc.status_code
            canary_unreadable = exc.status_code in {403, 404}

        return PermissionAuditResult(
            workspace_read_write_ok=ws_ok,
            data_hub_read_ok=hub_ok,
            outside_canary_unreadable=canary_unreadable,
            outside_canary_http_status=canary_status,
            details=(
                f"workspace_rw={ws_ok}, hub_ro={hub_ok}, "
                f"outside_canary_blocked={canary_unreadable} (HTTP {canary_status})"
            ),
        )

#!/usr/bin/env python3
"""Standalone Colab Linux x86_64 Heavy SOTA Worker (LinearDesign + EternaFold) with Drive Sealing & Vultr CAS Keepalive.

Design invariants:
- Pure Python 3.10+ standard library only (runs on ephemeral Colab VMs with zero extra installs).
- Strict track isolation: `commercial_clean` immediately refuses LinearDesign and EternaFold (exit code 2),
  and conflicting CLI vs `--job-json` compliance tracks fail closed before lease acquisition or extraction.
- Pinned SHA-256 verification and streaming single-FD tar path/link validation before extracting into a fresh run dir.
- Secret-free execution: reads `MRNAX_GATEWAY_ACCESS_TOKEN` / `MRNAX_DRIVE_ACCESS_TOKEN` strictly from env,
  validates against control characters and private-IP SSRF destinations, and scrubs tokens from all logs/JSON.
- Production Drive-first CAS commit: writes immutable `output_json` BEFORE commit, uploads result + `SEALED.json`
  under `<runs_root_id>/<execution_id>/attempts/<attempt_id>/SEALED.json`, and commits `(sealed_file_id, sha256(SEALED.json))`.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import hmac
import http.client
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import ssl
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request
import uuid

PINNED_LINEARDESIGN_TAR_SHA256 = "949802af5f0833bc2ec69dfe27a0dcaf01e4f8450d9242bd13672b3221a9fff3"
PINNED_ETERNAFOLD_TAR_SHA256 = "84fe61a28c4a4fc4b5bc1be7736cd888fd59e57c875efd81f8e1f0911187e1db"

_STANDARD_AA_OPT_STOP_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWYacdefghiklmnpqrstvwy]+[*]?$", re.ASCII)
_RNA_UPPER_RE = re.compile(r"^[ACGU]+$")
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_SEQ_LINE_RE = re.compile(r"(?:^|[\r\n])[ \t]*mRNA sequence:[ \t]*([ACGUacgu]+)[ \t]*(?=\r?\n|$)")
_ENERGY_CAI_RE = re.compile(
    r"mRNA folding free energy:\s*([-+]?\d+(?:\.\d+)?)\s*kcal/mol;\s*mRNA CAI:\s*([-+]?\d+(?:\.\d+)?)"
)
_REDACT_PATTERNS = re.compile(r"(?:tok_[A-Za-z0-9_-]{8,}|lzk_[A-Za-z0-9_-]{8,}|Bearer\s+\S+)", re.IGNORECASE)

_MAX_TAR_MEMBERS = 5000
_MAX_TAR_BYTES = 512 * 1024 * 1024
_MAX_HTTP_RESPONSE_BYTES = 2 * 1024 * 1024
_MAX_ETERNAFOLD_POST_BYTES = 2 * 1024 * 1024
_MAX_SUBPROCESS_TIMEOUT_SEC = 300.0
_DRIVE_FOLDER_MIME = "application/vnd.google-apps.folder"
_CAS_COMMIT_ALLOWLIST_KEYS = (
    "execution_id",
    "status",
    "winning_attempt_id",
    "winning_sealed_file_id",
    "winning_sealed_sha256",
    "lease_epoch",
)

_CODON_TABLE_1: dict[str, str] = {
    "UUU": "F", "UUC": "F", "UUA": "L", "UUG": "L",
    "UCU": "S", "UCC": "S", "UCA": "S", "UCG": "S",
    "UAU": "Y", "UAC": "Y", "UAA": "*", "UAG": "*",
    "UGU": "C", "UGC": "C", "UGA": "*", "UGG": "W",
    "CUU": "L", "CUC": "L", "CUA": "L", "CUG": "L",
    "CCU": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAU": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "CGU": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AUU": "I", "AUC": "I", "AUA": "I", "AUG": "M",
    "ACU": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "AAU": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "AGU": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GUU": "V", "GUC": "V", "GUA": "V", "GUG": "V",
    "GCU": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "GAU": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "GGU": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Reject HTTP redirects so credentials or CAS calls are never forwarded."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        raise urllib.error.HTTPError(req.full_url, code, f"Redirects are disallowed ({code})", headers, fp)


def _redact_secrets(text: str, *secrets_to_scrub: str) -> str:
    out = str(text)
    for sec in secrets_to_scrub:
        s = (sec or "").strip()
        if s and len(s) >= 4:
            out = out.replace(s, "[REDACTED]")
    return _REDACT_PATTERNS.sub("[REDACTED]", out)


def _redact_obj(obj: Any, *secrets_to_scrub: str) -> Any:
    if isinstance(obj, str):
        return _redact_secrets(obj, *secrets_to_scrub)
    if isinstance(obj, dict):
        return {
            _redact_secrets(str(k), *secrets_to_scrub): _redact_obj(v, *secrets_to_scrub)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_redact_obj(x, *secrets_to_scrub) for x in obj]
    return obj


def _validate_env_token(raw: str, *, env_name: str) -> str:
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in raw):
        raise ValueError(f"{env_name} contains invalid control characters")
    return raw.strip()


def _is_loopback_host(host: str) -> bool:
    return host.lower() in ("127.0.0.1", "localhost", "::1")


def _is_forbidden_non_global_ip(ip_obj: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return (
        (not ip_obj.is_global)
        or ip_obj.is_private
        or ip_obj.is_loopback
        or ip_obj.is_link_local
        or ip_obj.is_multicast
        or ip_obj.is_reserved
        or ip_obj.is_unspecified
    )


def validate_gateway_url(url: str) -> str:
    """Validate gateway URL: HTTPS required unless localhost/127.0.0.1; disallow credentials and non-global IPs."""
    raw = url.strip().rstrip("/")
    parsed = urllib.parse.urlsplit(raw)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Unsupported gateway URL scheme '{parsed.scheme}'")
    if parsed.username or parsed.password:
        raise ValueError("Gateway URL must not embed credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("Gateway URL must not contain query parameters or fragments")
    host = (parsed.hostname or "").strip().lower()
    if not host:
        raise ValueError("Gateway URL missing hostname")
    if host in ("169.254.169.254", "metadata.google.internal", "metadata"):
        raise ValueError(f"Blocked metadata host in gateway URL: '{host}'")
    try:
        ip_obj = ipaddress.ip_address(host)
        if not ip_obj.is_loopback and _is_forbidden_non_global_ip(ip_obj):
            raise ValueError(f"Blocked non-global/private gateway IP: '{host}'")
    except ValueError as exc:
        if "Blocked" in str(exc):
            raise
    is_loopback = _is_loopback_host(host)
    if parsed.scheme == "http" and not is_loopback:
        raise ValueError("Plain HTTP gateway URL is only permitted for localhost/127.0.0.1; use HTTPS.")

    allowed_raw = os.environ.get("MRNAX_ALLOWED_GATEWAY_ORIGINS", "").strip()
    if allowed_raw:
        allowed_set = {item.strip().rstrip("/").lower() for item in allowed_raw.split(",") if item.strip()}
        origin = f"{parsed.scheme}://{parsed.netloc}".lower()
        if origin not in allowed_set:
            raise ValueError(f"Gateway origin '{origin}' is not in MRNAX_ALLOWED_GATEWAY_ORIGINS")
    return raw


def _resolve_safe_gateway_peer(base_url: str) -> tuple[str, int, str]:
    """Resolve gateway host once, reject any non-global/shared/private IP, and return (host, port, pinned_ip)."""
    parsed = urllib.parse.urlsplit(base_url)
    host = (parsed.hostname or "").strip().lower()
    is_https = parsed.scheme == "https"
    port = parsed.port or (443 if is_https else 80)
    if _is_loopback_host(host):
        return host, port, ("127.0.0.1" if host == "localhost" else host)
    infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    if not infos:
        raise ValueError(f"Unable to resolve gateway host '{host}'")
    verified_ips: list[str] = []
    for _fam, _stype, _proto, _canon, sockaddr in infos:
        ip_str = str(sockaddr[0]).split("%", 1)[0]
        ip_obj = ipaddress.ip_address(ip_str)
        if _is_forbidden_non_global_ip(ip_obj):
            raise ValueError(f"Blocked resolved non-global gateway IP '{ip_str}' for '{host}'")
        verified_ips.append(ip_str)
    return host, port, verified_ips[0]


def normalize_protein_sequence(protein_seq: str) -> str:
    stripped = protein_seq.strip()
    if not stripped or not stripped.isascii() or not _STANDARD_AA_OPT_STOP_RE.fullmatch(stripped):
        raise ValueError("Protein sequence must contain only canonical 20 standard amino acids (optional trailing '*').")
    if stripped.endswith("*"):
        stripped = stripped[:-1]
    if len(stripped) < 1 or len(stripped) > 5000:
        raise ValueError(f"Protein length ({len(stripped)} AA) must be within [1, 5000].")
    return stripped.upper()


def parse_lambdas(raw: str | list[float]) -> tuple[float, ...]:
    if isinstance(raw, list):
        if not raw:
            raise ValueError("At least one lambda value must be provided")
        items = [float(x) for x in raw]
    else:
        parts = [p.strip() for p in str(raw).split(",") if p.strip()]
        if not parts:
            raise ValueError("At least one lambda value must be provided")
        items = [float(p) for p in parts]
    out: list[float] = []
    for val in items:
        if not math.isfinite(val) or val < 0.0 or val > 100.0:
            raise ValueError(f"Lambda value {val!r} must be finite and within [0.0, 100.0]")
        out.append(val)
    return tuple(out)


def translate_rna_exact(cds_rna: str) -> str:
    rna = cds_rna.strip().upper().replace("T", "U")
    if len(rna) % 3 != 0 or not _RNA_UPPER_RE.fullmatch(rna):
        raise ValueError("CDS RNA must be valid uppercase A/C/G/U with length divisible by 3")
    try:
        from Bio.Seq import Seq  # type: ignore[import-untyped]

        return str(Seq(rna).translate(table=1, to_stop=False))
    except ImportError:
        return "".join(_CODON_TABLE_1[rna[i : i + 3]] for i in range(0, len(rna), 3))


def verify_exact_aa_translation(cds_rna: str, expected_protein: str) -> bool:
    rna = cds_rna.strip().upper().replace("T", "U")
    if len(rna) != 3 * len(expected_protein):
        return False
    try:
        return translate_rna_exact(rna) == expected_protein
    except (ValueError, KeyError):
        return False


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def verify_and_extract_tarball(
    tar_path: Path,
    dest_dir: Path,
    *,
    expected_sha256: str,
) -> Path:
    """Verify pinned SHA-256 and validate every tar member incrementally on ONE file handle before extraction."""
    if not tar_path.is_file():
        raise FileNotFoundError(f"Archive not found: '{tar_path}'")
    expected_norm = expected_sha256.strip().lower()
    if not _SHA256_HEX_RE.fullmatch(expected_norm):
        raise ValueError("Invalid expected SHA-256 digest")

    dest_root = dest_dir.resolve()
    validated_count = 0
    total_bytes = 0

    with tar_path.open("rb") as fh:
        h = hashlib.sha256()
        compressed_bytes = 0
        while True:
            chunk = fh.read(65536)
            if not chunk:
                break
            compressed_bytes += len(chunk)
            if compressed_bytes > _MAX_TAR_BYTES:
                raise ValueError(f"Archive '{tar_path.name}' exceeds compressed safety limit ({_MAX_TAR_BYTES} bytes)")
            h.update(chunk)
        actual_sha = h.hexdigest()
        if not hmac.compare_digest(actual_sha, expected_norm):
            raise ValueError(
                f"SHA-256 mismatch for '{tar_path.name}': got {actual_sha}, expected {expected_norm}"
            )
        fh.seek(0)

        # Pass 1: validate every member header incrementally on the same open file handle before writing anything
        with tarfile.open(fileobj=fh, mode="r:*") as tf:
            while True:
                member = tf.next()
                if member is None:
                    break
                validated_count += 1
                if validated_count > _MAX_TAR_MEMBERS:
                    raise ValueError(f"Archive '{tar_path.name}' exceeds member limit (> {_MAX_TAR_MEMBERS})")
                raw_name = member.name
                if not raw_name or "\x00" in raw_name or "\\" in raw_name:
                    raise ValueError(f"Unsafe tar member name: {raw_name!r}")
                if raw_name.startswith("/") or (len(raw_name) >= 2 and raw_name[1] == ":"):
                    raise ValueError(f"Absolute path forbidden in tar archive: {raw_name!r}")
                if ".." in Path(raw_name).parts:
                    raise ValueError(f"Path traversal ('..') forbidden in tar archive: {raw_name!r}")
                if member.issym() or member.islnk():
                    raise ValueError(f"Symlink/hardlink forbidden in tar archive: {raw_name!r}")
                if member.isdev() or not (member.isdir() or member.isreg()):
                    raise ValueError(f"Unsupported non-regular tar member type for {raw_name!r}")
                if member.size < 0:
                    raise ValueError(f"Negative tar member size for {raw_name!r}")
                total_bytes += member.size
                if total_bytes > _MAX_TAR_BYTES:
                    raise ValueError(f"Archive '{tar_path.name}' exceeds uncompressed safety limit ({_MAX_TAR_BYTES} bytes)")
                target_path = (dest_root / raw_name).resolve()
                if target_path != dest_root and not target_path.is_relative_to(dest_root):
                    raise ValueError(f"Tar member escapes destination directory: {raw_name!r}")

        # Pass 2: create dest_root and extract using the exact same verified file handle
        dest_root.mkdir(parents=True, exist_ok=True)
        fh.seek(0)
        with tarfile.open(fileobj=fh, mode="r:*") as tf:
            while True:
                member = tf.next()
                if member is None:
                    break
                target_path = (dest_root / member.name).resolve()
                if member.isdir():
                    target_path.mkdir(parents=True, exist_ok=True)
                    continue
                target_path.parent.mkdir(parents=True, exist_ok=True)
                src_fh = tf.extractfile(member)
                if src_fh is None:
                    raise ValueError(f"Failed to read tar member payload: {member.name!r}")
                with src_fh, target_path.open("wb") as out_fh:
                    remaining = member.size
                    while True:
                        buf = src_fh.read(min(65536, max(1, remaining)))
                        if not buf:
                            break
                        remaining -= len(buf)
                        if remaining < 0:
                            raise ValueError(f"Tar member '{member.name}' payload exceeded declared size")
                        out_fh.write(buf)
                mode = 0o755 if (member.mode & 0o111) else 0o644
                os.chmod(target_path, mode)
    return dest_root


def _clean_subprocess_env(bin_path: Path) -> dict[str, str]:
    """Minimal secret-free environment; prepend binary directory to LD_LIBRARY_PATH on Linux."""
    env: dict[str, str] = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "LC_ALL": "C",
        "LANG": "C",
    }
    if sys.platform.startswith("linux"):
        bin_dir = str(bin_path.resolve().parent)
        prev_ld = os.environ.get("LD_LIBRARY_PATH", "").strip()
        env["LD_LIBRARY_PATH"] = f"{bin_dir}:{prev_ld}" if prev_ld else bin_dir
    return env


def _locate_lineardesign_binary(root_dir: Path) -> tuple[Path, Path]:
    bin_path: Path | None = None
    for cand in sorted(root_dir.rglob("lineardesign")):
        if cand.is_file():
            bin_path = cand.resolve()
            break
    if bin_path is None:
        raise FileNotFoundError(f"Executable 'lineardesign' not found under '{root_dir}'")
    if not os.access(bin_path, os.X_OK):
        os.chmod(bin_path, 0o755)
    runtime_dir = bin_path.parent
    if not (
        (runtime_dir / "coding_wheel.txt").is_file()
        and (runtime_dir / "codon_usage_freq_table_human.csv").is_file()
    ):
        for wheel in sorted(root_dir.rglob("coding_wheel.txt")):
            if (wheel.parent / "codon_usage_freq_table_human.csv").is_file():
                runtime_dir = wheel.parent.resolve()
                break
    return bin_path, runtime_dir


def _locate_eternafold_binary(root_dir: Path) -> tuple[Path, Path]:
    bin_path: Path | None = None
    for cand in sorted(root_dir.rglob("contrafold")):
        if cand.is_file():
            bin_path = cand.resolve()
            break
    if bin_path is None:
        raise FileNotFoundError(f"Executable 'contrafold' not found under '{root_dir}'")
    if not os.access(bin_path, os.X_OK):
        os.chmod(bin_path, 0o755)
    params_path: Path | None = None
    for p_cand in sorted(root_dir.rglob("EternaFoldParams*")):
        if p_cand.is_file():
            params_path = p_cand.resolve()
            break
    if params_path is None:
        raise FileNotFoundError(f"EternaFoldParams file not found under '{root_dir}'")
    return bin_path, params_path


def run_lineardesign_binary(
    bin_path: Path,
    runtime_dir: Path,
    sense_aa: str,
    *,
    lambda_val: float,
    timeout_sec: float = _MAX_SUBPROCESS_TIMEOUT_SEC,
) -> dict[str, Any]:
    proc = subprocess.run(
        [str(bin_path), "-l", f"{lambda_val:g}"],
        input=f"{sense_aa}\n",
        cwd=str(runtime_dir),
        env=_clean_subprocess_env(bin_path),
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        check=True,
    )
    seq_m = _SEQ_LINE_RE.search(proc.stdout)
    if seq_m is None:
        raise RuntimeError(f"LinearDesign stdout missing 'mRNA sequence:' line (lambda={lambda_val})")
    cds_rna = seq_m.group(1).upper().replace("T", "U")
    if not verify_exact_aa_translation(cds_rna, sense_aa):
        raise RuntimeError(f"LinearDesign candidate failed 100% AA translation verification (lambda={lambda_val})")
    ec_m = _ENERGY_CAI_RE.search(proc.stdout)
    mfe_kcal = float(ec_m.group(1)) if ec_m else None
    cai_val = float(ec_m.group(2)) if ec_m else None
    return {
        "lambda": lambda_val,
        "cds_rna": cds_rna,
        "aa_length": len(sense_aa),
        "aa_exact_match": True,
        "mfe_kcal_mol": mfe_kcal,
        "cai": cai_val,
    }


def _read_bounded_bytes(path: Path, max_bytes: int, *, label: str) -> bytes:
    chunks: list[bytes] = []
    seen = 0
    with path.open("rb") as fh:
        while True:
            blk = fh.read(65536)
            if not blk:
                break
            seen += len(blk)
            if seen > max_bytes:
                raise RuntimeError(f"{label} output exceeded byte limit ({max_bytes} bytes)")
            chunks.append(blk)
    return b"".join(chunks)


def run_eternafold_aup(
    contrafold_bin: Path,
    params_path: Path,
    rna_seq: str,
    *,
    timeout_sec: float = 60.0,
) -> dict[str, Any]:
    seq_norm = rna_seq.strip().upper().replace("T", "U")
    sub = seq_norm[: min(len(seq_norm), 400)]
    if len(sub) < 6 or not _RNA_UPPER_RE.fullmatch(sub):
        raise ValueError("Invalid RNA sequence for EternaFold evaluation")
    with tempfile.TemporaryDirectory(prefix="mrnax_ef_") as td:
        fa = Path(td) / "in.fa"
        post = Path(td) / "post.txt"
        fa.write_text(f">seq\n{sub}\n", encoding="utf-8")
        subprocess.run(
            [str(contrafold_bin), "predict", str(fa), "--params", str(params_path), "--posteriors", "0.0001", str(post)],
            env=_clean_subprocess_env(contrafold_bin),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout_sec,
            check=True,
        )
        raw_bytes = _read_bounded_bytes(post, _MAX_ETERNAFOLD_POST_BYTES, label="EternaFold posteriors")
        paired = [0.0] * len(sub)
        positions = 0
        for line in raw_bytes.decode("utf-8").splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) < 2 or not parts[0].isdigit():
                raise RuntimeError("Malformed EternaFold posteriors line")
            idx = positions
            if idx >= len(sub) or int(parts[0]) != idx + 1:
                raise RuntimeError("Unexpected EternaFold posteriors indexing")
            positions += 1
            for tok in parts[2:]:
                if ":" not in tok:
                    continue
                j_s, p_s = tok.split(":", 1)
                j = int(j_s) - 1
                prob = float(p_s)
                if 0 <= j < len(sub) and 0.0 <= prob <= 1.0:
                    paired[idx] += prob
                    paired[j] += prob
        if positions != len(sub):
            raise RuntimeError("Incomplete EternaFold posteriors table")
        unpaired = [max(0.0, min(1.0, 1.0 - p)) for p in paired]
        return {
            "evaluated_length_nt": len(sub),
            "eternafold_aup": round(sum(unpaired) / len(unpaired), 6),
        }


class _StdlibDriveRestClient:
    """Minimal pure-stdlib Google Drive REST v3 sealer matching DriveRestV3Client's seal contract."""

    FILES_API = "https://www.googleapis.com/drive/v3/files"
    UPLOAD_API = "https://www.googleapis.com/upload/drive/v3/files"

    def __init__(self, oauth_token: str, *, timeout_sec: float = 30.0) -> None:
        self._token = _validate_env_token(oauth_token, env_name="MRNAX_DRIVE_ACCESS_TOKEN")
        if not self._token:
            raise ValueError("MRNAX_DRIVE_ACCESS_TOKEN is empty")
        self.timeout_sec = timeout_sec
        self._opener = urllib.request.build_opener(_NoRedirectHandler())

    def _request_json(self, method: str, url: str, *, body: bytes | None = None, content_type: str | None = None) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/json",
        }
        if content_type:
            headers["Content-Type"] = content_type
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        with self._opener.open(req, timeout=self.timeout_sec) as resp:
            raw = resp.read(_MAX_HTTP_RESPONSE_BYTES + 1)
            if len(raw) > _MAX_HTTP_RESPONSE_BYTES:
                raise RuntimeError("Drive REST response exceeded byte cap")
            decoded = json.loads(raw.decode("utf-8"))
            if not isinstance(decoded, dict):
                raise RuntimeError("Drive REST response must be a JSON object")
            return decoded

    def create_folder(self, parent_id: str, name: str) -> str:
        q = f"'{parent_id}' in parents and trashed = false and name = '{name.replace(chr(39), '')}' and mimeType = '{_DRIVE_FOLDER_MIME}'"
        qs = urllib.parse.urlencode(
            {"q": q, "fields": "files(id,name)", "supportsAllDrives": "true", "includeItemsFromAllDrives": "true"}
        )
        listed = self._request_json("GET", f"{self.FILES_API}?{qs}")
        files = listed.get("files") or []
        if files:
            return str(files[0]["id"])
        meta_bytes = json.dumps({"name": name, "mimeType": _DRIVE_FOLDER_MIME, "parents": [parent_id]}).encode("utf-8")
        created = self._request_json(
            "POST",
            f"{self.FILES_API}?fields=id,name,mimeType,parents&supportsAllDrives=true",
            body=meta_bytes,
            content_type="application/json; charset=UTF-8",
        )
        return str(created["id"])

    def ensure_attempt_dir(self, runs_root_id: str, execution_id: str, attempt_id: str) -> str:
        exec_dir = self.create_folder(runs_root_id, execution_id)
        attempts_dir = self.create_folder(exec_dir, "attempts")
        return self.create_folder(attempts_dir, attempt_id)

    def upload_verified_bytes(
        self, parent_folder_id: str, name: str, payload: bytes, mime_type: str = "application/json"
    ) -> dict[str, Any]:
        local_md5 = hashlib.md5(payload).hexdigest()
        local_sha256 = hashlib.sha256(payload).hexdigest()
        boundary = f"mrnax_colab_{uuid.uuid4().hex}"
        meta_bytes = json.dumps({"name": name, "mimeType": mime_type, "parents": [parent_folder_id]}).encode("utf-8")
        body = (
            f"--{boundary}\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n".encode("utf-8")
            + meta_bytes
            + f"\r\n--{boundary}\r\nContent-Type: {mime_type}\r\n\r\n".encode("utf-8")
            + payload
            + f"\r\n--{boundary}--\r\n".encode("utf-8")
        )
        res = self._request_json(
            "POST",
            f"{self.UPLOAD_API}?uploadType=multipart&fields=id,name,parents,md5Checksum,size&supportsAllDrives=true",
            body=body,
            content_type=f"multipart/related; boundary={boundary}",
        )
        if res.get("md5Checksum") != local_md5:
            raise RuntimeError(f"Drive md5Checksum mismatch for '{name}'")
        return {
            "file_id": str(res["id"]),
            "name": name,
            "parent_id": parent_folder_id,
            "mime_type": mime_type,
            "bytes": len(payload),
            "md5": local_md5,
            "sha256": local_sha256,
        }

    def write_attempt_seal(
        self,
        attempt_folder_id: str,
        *,
        execution_id: str,
        attempt_id: str,
        lease_epoch: int,
        receipts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        seal_doc = {
            "schema": "mrnax.attempt_seal.v1",
            "execution_id": execution_id,
            "attempt_id": attempt_id,
            "lease_epoch": lease_epoch,
            "sealed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "artifacts": receipts,
        }
        payload = (json.dumps(seal_doc, indent=2, sort_keys=True) + "\n").encode("utf-8")
        return self.upload_verified_bytes(attempt_folder_id, "SEALED.json", payload, mime_type="application/json")


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """Connects TCP directly to the verified public IP while preserving TLS SNI and hostname certificate checks."""

    def __init__(
        self,
        host: str,
        port: int | None = None,
        *,
        pinned_ip: str,
        expected_host: str,
        expected_port: int,
        timeout: float = 15.0,
        context: ssl.SSLContext | None = None,
    ) -> None:
        exp_host_norm = expected_host.lower()
        exp_port_int = int(expected_port)
        parsed_auth = urllib.parse.urlsplit(f"//{host}")
        req_host_only = (parsed_auth.hostname or "").lower()
        try:
            req_port = parsed_auth.port
        except ValueError as exc:
            raise ValueError("HTTPS connection used invalid authority port") from exc
        eff_port = exp_port_int if port is None else int(port)
        if req_host_only != exp_host_norm or eff_port != exp_port_int or (req_port is not None and int(req_port) != exp_port_int):
            raise ValueError("HTTPS connection escaped pinned verified gateway origin")
        super().__init__(exp_host_norm, port=exp_port_int, timeout=timeout, context=context)
        self._pinned_ip = pinned_ip
        self._expected_host = exp_host_norm
        self._expected_port = exp_port_int

    def connect(self) -> None:
        if self.host.lower() != self._expected_host or int(self.port) != self._expected_port:
            raise ValueError("HTTPS connection escaped pinned verified gateway origin")
        raw_sock = socket.create_connection((self._pinned_ip, self._expected_port), self.timeout, self.source_address)
        try:
            self.sock = self._context.wrap_socket(raw_sock, server_hostname=self._expected_host)
        except Exception:
            raw_sock.close()
            raise


class _PinnedHTTPSHandler(urllib.request.HTTPSHandler):
    def __init__(self, *, expected_host: str, expected_port: int, pinned_ip: str) -> None:
        super().__init__(context=ssl.create_default_context())
        self._expected_host = expected_host
        self._expected_port = expected_port
        self._pinned_ip = pinned_ip

    def https_open(self, req: urllib.request.Request) -> http.client.HTTPResponse:
        return self.do_open(
            lambda host, timeout=15.0, **_kw: _PinnedHTTPSConnection(
                host,
                port=self._expected_port,
                pinned_ip=self._pinned_ip,
                expected_host=self._expected_host,
                expected_port=self._expected_port,
                timeout=timeout,
                context=self._context,
            ),
            req,
        )


class VultrCASClient:
    """Minimal stdlib HTTP client for Vultr CAS execution submission, lease keepalive, and commit."""

    def __init__(self, gateway_url: str, *, access_token: str = "", timeout_sec: float = 15.0) -> None:
        self.base_url = validate_gateway_url(gateway_url)
        self._access_token = _validate_env_token(access_token, env_name="MRNAX_GATEWAY_ACCESS_TOKEN")
        self.timeout_sec = timeout_sec
        host = (urllib.parse.urlsplit(self.base_url).hostname or "").lower()
        self._is_loopback = _is_loopback_host(host)

    def _open_pinned_json(
        self, method: str, path: str, body: dict[str, Any] | None = None, *, timeout_sec: float | None = None
    ) -> dict[str, Any]:
        exp_host, exp_port, pinned_ip = _resolve_safe_gateway_peer(self.base_url)
        handlers: list[Any] = [_NoRedirectHandler()]
        if self._is_loopback:
            handlers.append(urllib.request.ProxyHandler({}))
        elif urllib.parse.urlsplit(self.base_url).scheme == "https":
            handlers.append(_PinnedHTTPSHandler(expected_host=exp_host, expected_port=exp_port, pinned_ip=pinned_ip))
        opener = urllib.request.build_opener(*handlers)
        url = f"{self.base_url}{path}"
        data = json.dumps(body, separators=(",", ":")).encode("utf-8") if body is not None else None
        headers: dict[str, str] = {"Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        if self._access_token:
            headers["X-MRNAX-Access-Token"] = self._access_token
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        eff_to = self.timeout_sec if timeout_sec is None else max(0.5, float(timeout_sec))
        with opener.open(req, timeout=eff_to) as resp:
            raw = resp.read(_MAX_HTTP_RESPONSE_BYTES + 1)
            if len(raw) > _MAX_HTTP_RESPONSE_BYTES:
                raise RuntimeError("Gateway CAS response exceeded byte cap")
            decoded = json.loads(raw.decode("utf-8"))
            if not isinstance(decoded, dict):
                raise RuntimeError("Gateway CAS response must be a JSON object")
            return decoded

    @property
    def is_loopback(self) -> bool:
        return self._is_loopback

    def post_json(self, path: str, body: dict[str, Any], *, timeout_sec: float | None = None) -> dict[str, Any]:
        return self._open_pinned_json("POST", path, body, timeout_sec=timeout_sec)

    def get_execution(self, execution_id: str) -> dict[str, Any]:
        safe_exec = urllib.parse.quote(execution_id, safe="")
        return self._open_pinned_json("GET", f"/v1/executions/{safe_exec}")

    def submit_execution(self, *, submission_id: str, request_digest: str, intent_id: str) -> dict[str, Any]:
        return self.post_json(
            "/v1/executions",
            {
                "submission_id": submission_id,
                "request_digest": request_digest,
                "intent_id": intent_id,
            },
        )

    def acquire_or_renew_lease(
        self,
        execution_id: str,
        *,
        worker_id: str,
        ttl_seconds: int,
        lease_token: str | None = None,
        timeout_sec: float | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"worker_id": worker_id, "ttl_seconds": ttl_seconds}
        if lease_token is not None:
            payload["lease_token"] = lease_token
        safe_exec = urllib.parse.quote(execution_id, safe="")
        return self.post_json(f"/v1/executions/{safe_exec}/leases", payload, timeout_sec=timeout_sec)

    def commit_attempt(
        self,
        *,
        execution_id: str,
        attempt_id: str,
        lease_epoch: int,
        lease_token: str,
        sealed_file_id: str,
        sealed_sha256: str,
    ) -> dict[str, Any]:
        safe_exec = urllib.parse.quote(execution_id, safe="")
        safe_att = urllib.parse.quote(attempt_id, safe="")
        return self.post_json(
            f"/v1/executions/{safe_exec}/attempts/{safe_att}:commit",
            {
                "lease_epoch": lease_epoch,
                "lease_token": lease_token,
                "sealed_file_id": sealed_file_id,
                "sealed_sha256": sealed_sha256,
            },
        )


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tf:
        tmp_path = Path(tf.name)
        try:
            tf.write(content)
            tf.flush()
            os.fsync(tf.fileno())
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
    os.replace(tmp_path, path)


def _sanitize_commit_receipt(raw: dict[str, Any], *, secrets_to_scrub: tuple[str, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in _CAS_COMMIT_ALLOWLIST_KEYS:
        if k in raw:
            out[k] = _redact_obj(raw[k], *secrets_to_scrub)
    return out


_RAW_SUPPORT_TABLES = ("coding_wheel.txt", "codon_usage_freq_table_human.csv")


def _hash_support_tables(src_dir: Path) -> dict[str, str]:
    digests: dict[str, str] = {}
    for tbl in _RAW_SUPPORT_TABLES:
        cand = src_dir / tbl
        if cand.is_file():
            h = hashlib.sha256()
            with cand.open("rb") as fh:
                while True:
                    blk = fh.read(65536)
                    if not blk:
                        break
                    h.update(blk)
            digests[tbl] = h.hexdigest()
    return digests


def _copy_verified_raw_binary(
    src_bin: Path,
    dest_dir: Path,
    *,
    expected_sha256: str,
    expected_support_sha256s: dict[str, str] | None = None,
) -> tuple[Path, Path]:
    """Read, SHA-256 verify, and stage raw binary + runtime tables via single open FDs."""
    exp_bin_sha = expected_sha256.strip().lower()
    if not _SHA256_HEX_RE.fullmatch(exp_bin_sha):
        raise ValueError("--lineardesign-bin-sha256 must be a valid 64-char hex SHA-256")
    dest_dir.mkdir(parents=True, exist_ok=True)
    exe_dir = dest_dir / "_bin"
    exe_dir.mkdir(parents=True, exist_ok=True)
    staged_bin = exe_dir / src_bin.name
    h = hashlib.sha256()
    with src_bin.open("rb") as src_fh, staged_bin.open("wb") as dst_fh:
        while True:
            blk = src_fh.read(65536)
            if not blk:
                break
            h.update(blk)
            dst_fh.write(blk)
    actual_sha = h.hexdigest()
    if not hmac.compare_digest(actual_sha, exp_bin_sha):
        staged_bin.unlink(missing_ok=True)
        raise ValueError(
            f"LinearDesign binary SHA-256 mismatch: got {actual_sha}, expected {exp_bin_sha}"
        )
    os.chmod(staged_bin, 0o755)
    for tbl in _RAW_SUPPORT_TABLES:
        cand_tbl = src_bin.parent / tbl
        if cand_tbl.is_file():
            dst_tbl = dest_dir / tbl
            th = hashlib.sha256()
            with cand_tbl.open("rb") as sfh, dst_tbl.open("wb") as dfh:
                while True:
                    blk = sfh.read(65536)
                    if not blk:
                        break
                    th.update(blk)
                    dfh.write(blk)
            if expected_support_sha256s and tbl in expected_support_sha256s:
                if not hmac.compare_digest(th.hexdigest(), expected_support_sha256s[tbl]):
                    dst_tbl.unlink(missing_ok=True)
                    raise ValueError(f"LinearDesign runtime table '{tbl}' changed after pinning")
    return staged_bin.resolve(), dest_dir.resolve()


def _seal_to_drive(
    drive_client: Any,
    *,
    drive_runs_root_id: str,
    execution_id: str,
    attempt_id: str,
    lease_epoch: int,
    artifact_name: str,
    artifact_bytes: bytes,
) -> tuple[str, str]:
    attempt_folder_id = drive_client.ensure_attempt_dir(drive_runs_root_id, execution_id, attempt_id)
    art_receipt = drive_client.upload_verified_bytes(
        attempt_folder_id,
        artifact_name,
        artifact_bytes,
        mime_type="application/json",
    )
    seal_receipt = drive_client.write_attempt_seal(
        attempt_folder_id,
        execution_id=execution_id,
        attempt_id=attempt_id,
        lease_epoch=lease_epoch,
        receipts=[art_receipt],
    )
    if isinstance(seal_receipt, dict):
        return str(seal_receipt["file_id"]), str(seal_receipt["sha256"])
    return str(seal_receipt.file_id), str(seal_receipt.sha256)


def run_worker(
    *,
    protein_seq: str,
    compliance_track: str = "research_max",
    lambdas: tuple[float, ...] = (1.0, 2.0),
    include_eternafold: bool = False,
    lineardesign_tar: Path | None = None,
    eternafold_tar: Path | None = None,
    lineardesign_bin: Path | None = None,
    lineardesign_bin_sha256: str | None = None,
    work_dir: Path | None = None,
    gateway_url: str | None = None,
    execution_id: str | None = None,
    attempt_id: str | None = None,
    worker_id: str = "colab_heavy_sota_worker",
    ttl_seconds: int = 120,
    submission_id: str | None = None,
    request_digest: str | None = None,
    intent_id: str | None = None,
    sealed_file_id: str | None = None,
    sealed_sha256: str | None = None,
    drive_runs_root_id: str | None = None,
    drive_client: Any = None,
    defer_commit: bool = False,
    output_json: Path | None = None,
    commit_receipt_json: Path | None = None,
    heartbeat_interval_sec: float | None = None,
) -> dict[str, Any]:
    """Execute Colab heavy SOTA workload with strict clean-room isolation, pinned tar verification, Drive seal, and CAS keepalive."""
    if compliance_track not in ("research_max", "commercial_clean"):
        raise ValueError(f"Unsupported compliance_track '{compliance_track}'")
    if compliance_track == "commercial_clean":
        raise PermissionError(
            "Track isolation blocked: LinearDesign and EternaFold are strictly disallowed under 'commercial_clean'."
        )
    if not lambdas:
        raise ValueError("At least one lambda value must be provided")

    sense_aa = normalize_protein_sequence(protein_seq)
    eff_ttl = min(600, max(15, int(ttl_seconds)))
    eff_interval = (
        max(5.0, eff_ttl / 2.0)
        if heartbeat_interval_sec is None
        else max(0.01, float(heartbeat_interval_sec))
    )
    access_token = _validate_env_token(
        os.environ.get("MRNAX_GATEWAY_ACCESS_TOKEN", ""), env_name="MRNAX_GATEWAY_ACCESS_TOKEN"
    )
    drive_token = _validate_env_token(
        os.environ.get("MRNAX_DRIVE_ACCESS_TOKEN", ""), env_name="MRNAX_DRIVE_ACCESS_TOKEN"
    )
    scrub_secrets = (access_token, drive_token)

    cas_client = VultrCASClient(gateway_url, access_token=access_token) if gateway_url else None
    eff_drive_runs_root = (drive_runs_root_id or os.environ.get("MRNAX_DRIVE_RUNS_ROOT_ID", "")).strip() or None
    eff_drive_client = drive_client
    if eff_drive_client is None and eff_drive_runs_root and drive_token:
        eff_drive_client = _StdlibDriveRestClient(drive_token)

    if sealed_file_id and not sealed_sha256:
        raise ValueError("--sealed-file-id requires matching --sealed-sha256 of the uploaded SEALED.json")
    if sealed_sha256 and not _SHA256_HEX_RE.fullmatch(sealed_sha256.strip().lower()):
        raise ValueError("--sealed-sha256 must be a 64-char lowercase hex SHA-256 digest")
    if (
        output_json is not None
        and commit_receipt_json is not None
        and output_json.resolve() == commit_receipt_json.resolve()
    ):
        raise ValueError("--commit-receipt-json must be distinct from --output-json so the sealed artifact stays immutable")
    if defer_commit and output_json is None:
        raise ValueError("--defer-commit requires --output-json so the computed artifact is persisted for external sealing")

    if cas_client is not None and not defer_commit and not cas_client.is_loopback:
        if eff_drive_client is None and not (sealed_file_id and sealed_sha256):
            raise ValueError(
                "Remote HTTPS gateway commit requires Drive sealing (--drive-runs-root-id + MRNAX_DRIVE_ACCESS_TOKEN "
                "or drive_client) or explicit (--sealed-file-id + --sealed-sha256); pass --defer-commit for compute-only."
            )

    eff_exec_id = (execution_id or "").strip() or None
    eff_intent_id = (intent_id or "").strip() or "int_colab_sota"
    ld_identity = (
        PINNED_LINEARDESIGN_TAR_SHA256
        if lineardesign_tar is not None
        else (lineardesign_bin_sha256 or "").strip().lower()
    )
    raw_support_shas: dict[str, str] = {}
    if lineardesign_tar is None and lineardesign_bin is not None:
        raw_bin_parent = lineardesign_bin.resolve().parent
        if raw_bin_parent.is_dir():
            raw_support_shas = _hash_support_tables(raw_bin_parent)
    ef_identity = PINNED_ETERNAFOLD_TAR_SHA256 if include_eternafold else None
    eff_req_dig = (request_digest or "").strip() or hashlib.sha256(
        json.dumps(
            {
                "protein": sense_aa,
                "lambdas": list(lambdas),
                "ef": include_eternafold,
                "track": compliance_track,
                "intent_id": eff_intent_id,
                "ld_sha": ld_identity,
                "ld_support_sha": raw_support_shas,
                "ef_sha": ef_identity,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    if cas_client is not None and not defer_commit and eff_exec_id is None:
        eff_sub_id = (submission_id or "").strip() or f"sub_colab_{eff_req_dig[:16]}"
        sub_row = cas_client.submit_execution(
            submission_id=eff_sub_id,
            request_digest=eff_req_dig,
            intent_id=eff_intent_id,
        )
        eff_exec_id = str(sub_row["execution_id"])
        if str(sub_row.get("status") or "") == "COMMITTED":
            raise RuntimeError(
                f"Execution '{eff_exec_id}' (submission_id='{eff_sub_id}') is already COMMITTED "
                f"(winning_attempt_id={sub_row.get('winning_attempt_id')})."
            )
    elif cas_client is not None and not defer_commit and eff_exec_id is not None:
        existing_exec = cas_client.get_execution(eff_exec_id)
        if str(existing_exec.get("status") or "") == "COMMITTED":
            raise RuntimeError(
                f"Execution '{eff_exec_id}' is already COMMITTED "
                f"(winning_attempt_id={existing_exec.get('winning_attempt_id')})."
            )
        if request_digest is not None and str(existing_exec.get("request_digest") or "") != request_digest.strip():
            raise ValueError(
                f"Execution '{eff_exec_id}' request_digest mismatch: stored '{existing_exec.get('request_digest')}', "
                f"got '{request_digest.strip()}'"
            )
        if intent_id is not None and existing_exec.get("intent_id") and str(existing_exec.get("intent_id")) != intent_id.strip():
            raise ValueError(
                f"Execution '{eff_exec_id}' intent_id mismatch: stored '{existing_exec.get('intent_id')}', "
                f"got '{intent_id.strip()}'"
            )

    lease_grant: dict[str, Any] | None = None
    if cas_client is not None and not defer_commit and eff_exec_id is not None:
        lease_grant = cas_client.acquire_or_renew_lease(
            eff_exec_id,
            worker_id=worker_id,
            ttl_seconds=eff_ttl,
        )

    stop_event = threading.Event()
    renewal_count = 0
    renewal_lock = threading.Lock()
    renewal_errors: list[str] = []
    lease_superseded = False
    active_grant: dict[str, Any] | None = dict(lease_grant) if lease_grant is not None else None

    def _heartbeat_loop() -> None:
        nonlocal renewal_count, lease_superseded
        assert cas_client is not None and eff_exec_id is not None
        hb_http_timeout = min(15.0, max(2.0, eff_interval))
        while not stop_event.wait(timeout=eff_interval):
            with renewal_lock:
                assert active_grant is not None
                cur_tok = str(active_grant["lease_token"])
                cur_att = str(active_grant["attempt_id"])
                cur_epoch = int(active_grant["lease_epoch"])
            try:
                renewed = cas_client.acquire_or_renew_lease(
                    eff_exec_id,
                    worker_id=worker_id,
                    ttl_seconds=eff_ttl,
                    lease_token=cur_tok,
                    timeout_sec=hb_http_timeout,
                )
                with renewal_lock:
                    assert active_grant is not None
                    new_att = str(renewed.get("attempt_id") or "")
                    new_epoch = int(renewed.get("lease_epoch", -1))
                    if new_att != cur_att or new_epoch != cur_epoch:
                        lease_superseded = True
                        renewal_errors.append(
                            f"Lease superseded during heartbeat: attempt {cur_att} (epoch {cur_epoch}) -> {new_att} (epoch {new_epoch})"
                        )
                        break
                    active_grant.update(renewed)
                    renewal_count += 1
            except Exception as exc:
                with renewal_lock:
                    renewal_errors.append(_redact_secrets(f"{type(exc).__name__}: {exc}", *scrub_secrets))
                break

    hb_thread: threading.Thread | None = None
    if lease_grant is not None:
        hb_thread = threading.Thread(target=_heartbeat_loop, name="mrnax-cas-keepalive", daemon=True)
        hb_thread.start()

    try:
        with tempfile.TemporaryDirectory(prefix="mrnax_colab_sota_") as tmp_td:
            eff_work = work_dir.resolve() if work_dir is not None else Path(tmp_td)
            eff_work.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="verified_run_", dir=eff_work) as isolated_dir_raw:
                isolated_dir = Path(isolated_dir_raw)
                if lineardesign_tar is not None:
                    ld_dir = verify_and_extract_tarball(
                        lineardesign_tar,
                        isolated_dir / "lineardesign",
                        expected_sha256=PINNED_LINEARDESIGN_TAR_SHA256,
                    )
                    ld_bin, ld_runtime = _locate_lineardesign_binary(ld_dir)
                elif lineardesign_bin is not None:
                    raw_ld_bin = lineardesign_bin.resolve()
                    if not raw_ld_bin.is_file():
                        raise FileNotFoundError(f"LinearDesign binary not found: '{raw_ld_bin}'")
                    if not lineardesign_bin_sha256:
                        raise ValueError("--lineardesign-bin requires pinned --lineardesign-bin-sha256")
                    ld_bin, ld_runtime = _copy_verified_raw_binary(
                        raw_ld_bin,
                        isolated_dir / "pinned_bin",
                        expected_sha256=lineardesign_bin_sha256,
                        expected_support_sha256s=raw_support_shas,
                    )
                else:
                    raise FileNotFoundError("Provide --lineardesign-tar (pinned SHA-256) or --lineardesign-bin + --lineardesign-bin-sha256.")

                candidates: list[dict[str, Any]] = []
                for lam in lambdas:
                    candidates.append(
                        run_lineardesign_binary(ld_bin, ld_runtime, sense_aa, lambda_val=lam)
                    )

                ef_metrics: list[dict[str, Any]] = []
                if include_eternafold:
                    if eternafold_tar is None:
                        raise FileNotFoundError("--include-eternafold requires --eternafold-tar with pinned SHA-256.")
                    ef_dir = verify_and_extract_tarball(
                        eternafold_tar,
                        isolated_dir / "eternafold",
                        expected_sha256=PINNED_ETERNAFOLD_TAR_SHA256,
                    )
                    ef_bin, ef_params = _locate_eternafold_binary(ef_dir)
                    for cand in candidates:
                        ef_metrics.append(
                            {
                                "lambda": cand["lambda"],
                                **run_eternafold_aup(ef_bin, ef_params, cand["cds_rna"]),
                            }
                        )

        with renewal_lock:
            renewals_snapshot = renewal_count
            errors_snapshot = list(renewal_errors)
            superseded_snapshot = lease_superseded
            grant_snapshot = dict(active_grant) if active_grant is not None else None

        if superseded_snapshot:
            raise RuntimeError(errors_snapshot[-1] if errors_snapshot else "CAS lease expired and was superseded")

        eff_attempt_id = str(grant_snapshot["attempt_id"]) if grant_snapshot is not None else (attempt_id or "")
        eff_lease_epoch = int(grant_snapshot["lease_epoch"]) if grant_snapshot is not None else None
        result_core: dict[str, Any] = {
            "status": "COMPUTED",
            "compliance_track": compliance_track,
            "worker_id": worker_id,
            "protein_length_aa": len(sense_aa),
            "candidates": candidates,
            "eternafold_metrics": ef_metrics,
            "execution_id": eff_exec_id,
            "attempt_id": eff_attempt_id or None,
            "lease_epoch": eff_lease_epoch,
            "heartbeat_renewals": renewals_snapshot,
            "heartbeat_errors": errors_snapshot,
        }
        immutable_bytes = (json.dumps(result_core, indent=2, sort_keys=True) + "\n").encode("utf-8")
        result_sha256 = hashlib.sha256(immutable_bytes).hexdigest()

        # Persist immutable computed output BEFORE Drive upload / CAS commit so a rejected commit never discards results
        if output_json is not None:
            _atomic_write_bytes(output_json, immutable_bytes)

        result_out: dict[str, Any] = {**result_core, "result_sha256": result_sha256}

        if cas_client is not None and eff_exec_id is not None and grant_snapshot is not None and not defer_commit:
            if eff_drive_client is not None and eff_drive_runs_root is not None:
                eff_sealed_id, eff_sealed_sha256 = _seal_to_drive(
                    eff_drive_client,
                    drive_runs_root_id=eff_drive_runs_root,
                    execution_id=eff_exec_id,
                    attempt_id=eff_attempt_id,
                    lease_epoch=int(grant_snapshot["lease_epoch"]),
                    artifact_name=output_json.name if output_json is not None else "colab_sota_result.json",
                    artifact_bytes=immutable_bytes,
                )
            elif sealed_file_id and sealed_sha256:
                eff_sealed_id = sealed_file_id.strip()
                eff_sealed_sha256 = sealed_sha256.strip().lower()
            elif cas_client.is_loopback:
                eff_sealed_id = (
                    str(output_json.resolve())
                    if output_json is not None
                    else f"local://result/{eff_exec_id}/{eff_attempt_id}"
                )
                eff_sealed_sha256 = result_sha256
            else:
                raise ValueError("Cannot commit to remote gateway without Drive SEALED.json or explicit sealed digest")

            with renewal_lock:
                if lease_superseded:
                    raise RuntimeError(
                        renewal_errors[-1] if renewal_errors else "CAS lease superseded during Drive sealing"
                    )
                commit_grant = dict(active_grant) if active_grant is not None else grant_snapshot

            raw_commit_receipt = cas_client.commit_attempt(
                execution_id=eff_exec_id,
                attempt_id=str(commit_grant["attempt_id"]),
                lease_epoch=int(commit_grant["lease_epoch"]),
                lease_token=str(commit_grant["lease_token"]),
                sealed_file_id=eff_sealed_id,
                sealed_sha256=eff_sealed_sha256,
            )
            with renewal_lock:
                result_out["heartbeat_renewals"] = renewal_count
                result_out["heartbeat_errors"] = list(renewal_errors)
            clean_receipt = _sanitize_commit_receipt(raw_commit_receipt, secrets_to_scrub=scrub_secrets)
            result_out["status"] = str(clean_receipt.get("status", "COMMITTED"))
            result_out["sealed_file_id"] = eff_sealed_id
            result_out["sealed_sha256"] = eff_sealed_sha256
            result_out["cas_commit"] = clean_receipt

            receipt_target = commit_receipt_json or (
                output_json.with_name(f"{output_json.stem}.commit_receipt.json")
                if output_json is not None
                else None
            )
            if receipt_target is not None:
                receipt_doc = {
                    "execution_id": eff_exec_id,
                    "attempt_id": str(commit_grant["attempt_id"]),
                    "status": result_out["status"],
                    "result_sha256": result_sha256,
                    "sealed_file_id": eff_sealed_id,
                    "sealed_sha256": eff_sealed_sha256,
                    "cas_commit": clean_receipt,
                }
                _atomic_write_bytes(
                    receipt_target,
                    (json.dumps(receipt_doc, indent=2, sort_keys=True) + "\n").encode("utf-8"),
                )

        return result_out
    finally:
        stop_event.set()
        if hb_thread is not None:
            hb_thread.join(timeout=5.0)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Colab Heavy SOTA Worker (LinearDesign + EternaFold) with Drive Sealing & Vultr CAS Keepalive"
    )
    p.add_argument("--protein-seq", default="", help="Primary target protein sequence (canonical AA)")
    p.add_argument("--job-json", type=Path, default=None, help="Optional JSON job descriptor file")
    p.add_argument(
        "--compliance-track",
        default=None,
        choices=("research_max", "commercial_clean"),
        help="Compliance track (defaults to research_max unless set by --job-json; conflicts fail closed)",
    )
    p.add_argument("--lambdas", default="1.0,2.0", help="Comma-separated LinearDesign lambda weights")
    p.add_argument("--include-eternafold", action="store_true", help="Also run EternaFold AUP scoring")
    p.add_argument("--lineardesign-tar", type=Path, default=None, help="Path to pinned LinearDesign Linux tarball")
    p.add_argument("--eternafold-tar", type=Path, default=None, help="Path to pinned EternaFold Linux tarball")
    p.add_argument("--lineardesign-bin", type=Path, default=None, help="Optional pre-extracted local binary path")
    p.add_argument("--lineardesign-bin-sha256", default=None, help="Required SHA-256 pin when using --lineardesign-bin")
    p.add_argument("--work-dir", type=Path, default=None, help="Extraction/working directory")
    p.add_argument("--gateway-url", default=None, help="Vultr control-plane HTTPS/localhost URL")
    p.add_argument("--execution-id", default=None, help="Pre-submitted CAS execution_id")
    p.add_argument("--attempt-id", default=None, help="Optional attempt_id hint")
    p.add_argument("--worker-id", default="colab_heavy_sota_worker", help="CAS worker identifier")
    p.add_argument("--ttl-seconds", type=int, default=120, help="CAS lease TTL in seconds (clamped [15, 600])")
    p.add_argument("--submission-id", default=None, help="Optional CAS submission_id for auto-submit")
    p.add_argument("--request-digest", default=None, help="Optional CAS request_digest for auto-submit")
    p.add_argument("--intent-id", default=None, help="Intent ID for CAS auto-submit")
    p.add_argument("--drive-runs-root-id", default=None, help="Drive runs root folder ID for SEALED.json upload")
    p.add_argument("--sealed-file-id", default=None, help="Pre-uploaded Drive SEALED.json file ID")
    p.add_argument("--sealed-sha256", default=None, help="SHA-256 of pre-uploaded Drive SEALED.json")
    p.add_argument("--defer-commit", action="store_true", help="Run compute only and leave CAS commit to external sealer")
    p.add_argument("--output-json", type=Path, default=None, help="Immutable output artifact JSON path")
    p.add_argument("--commit-receipt-json", type=Path, default=None, help="Optional CAS commit receipt JSON path")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    scrub_secrets = (
        os.environ.get("MRNAX_GATEWAY_ACCESS_TOKEN", ""),
        os.environ.get("MRNAX_DRIVE_ACCESS_TOKEN", ""),
    )

    protein_seq = args.protein_seq
    cli_track: str | None = args.compliance_track
    job_track: str | None = None
    lambdas_raw: str | list[float] = args.lambdas
    include_eternafold = bool(args.include_eternafold)

    if args.job_json is not None:
        job_doc = json.loads(args.job_json.read_text(encoding="utf-8"))
        if not isinstance(job_doc, dict):
            print("Error: --job-json must decode to a JSON object", file=sys.stderr)
            return 1
        protein_seq = str(job_doc.get("protein_seq") or job_doc.get("protein_sequence") or protein_seq)
        if "compliance_track" in job_doc and job_doc["compliance_track"] is not None:
            job_track = str(job_doc["compliance_track"]).strip()
            if job_track not in ("research_max", "commercial_clean"):
                print(
                    _redact_secrets(f"Error: unsupported compliance_track '{job_track}' in --job-json", *scrub_secrets),
                    file=sys.stderr,
                )
                return 2
        if "lambdas" in job_doc:
            lambdas_raw = job_doc["lambdas"]
        if "include_eternafold" in job_doc:
            include_eternafold = bool(job_doc["include_eternafold"])

    if args.defer_commit and args.output_json is None:
        print("Error: --defer-commit requires --output-json so the computed artifact is persisted for external sealing", file=sys.stderr)
        return 1

    if cli_track is not None and job_track is not None and cli_track != job_track:
        print(
            f"PermissionError: Conflicting compliance_track between CLI ('{cli_track}') and --job-json ('{job_track}')",
            file=sys.stderr,
        )
        return 2

    compliance_track = cli_track or job_track or "research_max"

    if not protein_seq.strip():
        print("Error: --protein-seq (or --job-json protein_seq) is required", file=sys.stderr)
        return 1

    try:
        lam_tuple = parse_lambdas(lambdas_raw)
        result = run_worker(
            protein_seq=protein_seq,
            compliance_track=compliance_track,
            lambdas=lam_tuple,
            include_eternafold=include_eternafold,
            lineardesign_tar=args.lineardesign_tar,
            eternafold_tar=args.eternafold_tar,
            lineardesign_bin=args.lineardesign_bin,
            lineardesign_bin_sha256=args.lineardesign_bin_sha256,
            work_dir=args.work_dir,
            gateway_url=args.gateway_url,
            execution_id=args.execution_id,
            attempt_id=args.attempt_id,
            worker_id=args.worker_id,
            ttl_seconds=args.ttl_seconds,
            submission_id=args.submission_id,
            request_digest=args.request_digest,
            intent_id=args.intent_id,
            sealed_file_id=args.sealed_file_id,
            sealed_sha256=args.sealed_sha256,
            drive_runs_root_id=args.drive_runs_root_id,
            defer_commit=bool(args.defer_commit),
            output_json=args.output_json,
            commit_receipt_json=args.commit_receipt_json,
        )
    except PermissionError as exc:
        print(_redact_secrets(f"PermissionError: {exc}", *scrub_secrets), file=sys.stderr)
        return 2
    except Exception as exc:
        print(_redact_secrets(f"Error: {type(exc).__name__}: {exc}", *scrub_secrets), file=sys.stderr)
        return 1

    summary = {
        "status": result["status"],
        "candidates_count": len(result["candidates"]),
        "result_sha256": result["result_sha256"],
        "execution_id": result["execution_id"],
        "heartbeat_renewals": result["heartbeat_renewals"],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

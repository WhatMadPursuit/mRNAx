#!/usr/bin/env python3
"""Standalone Colab Linux x86_64 Heavy SOTA Worker (LinearDesign + EternaFold) with Vultr CAS Lease Keepalive.

Design invariants:
- Pure Python 3.10+ standard library only (runs on ephemeral Colab VMs with zero extra installs).
- Strict track isolation: `commercial_clean` immediately refuses LinearDesign and EternaFold (exit code 2).
- Pinned SHA-256 verification and strict tar path/link validation before extracting any archive.
- Secret-free execution: reads `MRNAX_GATEWAY_ACCESS_TOKEN` exclusively from environment, never CLI argv/logs/output.
- Bounded no-redirect HTTPS/localhost CAS client with background lease keepalive and `finally:` cleanup.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import ipaddress
import json
import math
import os
from pathlib import Path
import re
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

PINNED_LINEARDESIGN_TAR_SHA256 = "949802af5f0833bc2ec69dfe27a0dcaf01e4f8450d9242bd13672b3221a9fff3"
PINNED_ETERNAFOLD_TAR_SHA256 = "84fe61a28c4a4fc4b5bc1be7736cd888fd59e57c875efd81f8e1f0911187e1db"

_STANDARD_AA_OPT_STOP_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+[*]?$")
_RNA_UPPER_RE = re.compile(r"^[ACGU]+$")
_SEQ_LINE_RE = re.compile(r"(?:^|[\r\n])[ \t]*mRNA sequence:[ \t]*([ACGUacgu]+)[ \t]*(?=\r?\n|$)")
_ENERGY_CAI_RE = re.compile(
    r"mRNA folding free energy:\s*([-+]?\d+(?:\.\d+)?)\s*kcal/mol;\s*mRNA CAI:\s*([-+]?\d+(?:\.\d+)?)"
)

_MAX_TAR_MEMBERS = 5000
_MAX_TAR_BYTES = 512 * 1024 * 1024
_MAX_HTTP_RESPONSE_BYTES = 2 * 1024 * 1024
_MAX_ETERNAFOLD_POST_BYTES = 2 * 1024 * 1024
_MAX_SUBPROCESS_TIMEOUT_SEC = 300.0

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


def validate_gateway_url(url: str) -> str:
    """Validate gateway URL: HTTPS required unless localhost/127.0.0.1; disallow credentials and metadata IPs."""
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
        if ip_obj.is_link_local or ip_obj.is_multicast:
            raise ValueError(f"Blocked non-routable/link-local gateway IP: '{host}'")
    except ValueError as exc:
        if "Blocked" in str(exc):
            raise
    is_loopback = host in ("127.0.0.1", "localhost", "::1")
    if parsed.scheme == "http" and not is_loopback:
        raise ValueError("Plain HTTP gateway URL is only permitted for localhost/127.0.0.1; use HTTPS.")
    return raw


def normalize_protein_sequence(protein_seq: str) -> str:
    cleaned = protein_seq.strip().upper()
    if not cleaned or not _STANDARD_AA_OPT_STOP_RE.fullmatch(cleaned):
        raise ValueError("Protein sequence must contain only canonical 20 standard amino acids (optional trailing '*').")
    if cleaned.endswith("*"):
        cleaned = cleaned[:-1]
    if len(cleaned) < 1 or len(cleaned) > 5000:
        raise ValueError(f"Protein length ({len(cleaned)} AA) must be within [1, 5000].")
    return cleaned


def parse_lambdas(raw: str | list[float]) -> tuple[float, ...]:
    if isinstance(raw, list):
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
    """Verify pinned SHA-256 and validate every tar member before safely extracting into `dest_dir`."""
    if not tar_path.is_file():
        raise FileNotFoundError(f"Archive not found: '{tar_path}'")
    expected_norm = expected_sha256.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_norm):
        raise ValueError("Invalid expected SHA-256 digest")
    actual_sha = compute_file_sha256(tar_path)
    if not hmac.compare_digest(actual_sha, expected_norm):
        raise ValueError(
            f"SHA-256 mismatch for '{tar_path.name}': got {actual_sha}, expected {expected_norm}"
        )

    dest_root = dest_dir.resolve()
    dest_root.mkdir(parents=True, exist_ok=True)
    validated_members: list[tuple[tarfile.TarInfo, Path]] = []
    total_bytes = 0

    with tarfile.open(tar_path, mode="r:*") as tf:
        members = tf.getmembers()
        if len(members) > _MAX_TAR_MEMBERS:
            raise ValueError(f"Archive '{tar_path.name}' exceeds member limit ({len(members)} > {_MAX_TAR_MEMBERS})")
        for member in members:
            raw_name = member.name
            if not raw_name or "\x00" in raw_name or "\\" in raw_name:
                raise ValueError(f"Unsafe tar member name: {raw_name!r}")
            if raw_name.startswith("/") or (len(raw_name) >= 2 and raw_name[1] == ":"):
                raise ValueError(f"Absolute path forbidden in tar archive: {raw_name!r}")
            parts = Path(raw_name).parts
            if ".." in parts:
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
            validated_members.append((member, target_path))

        for member, target_path in validated_members:
            if member.isdir():
                target_path.mkdir(parents=True, exist_ok=True)
                continue
            target_path.parent.mkdir(parents=True, exist_ok=True)
            src_fh = tf.extractfile(member)
            if src_fh is None:
                raise ValueError(f"Failed to read tar member payload: {member.name!r}")
            with src_fh, target_path.open("wb") as out_fh:
                while True:
                    buf = src_fh.read(65536)
                    if not buf:
                        break
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
        raw_bytes = post.read_bytes()
        if len(raw_bytes) > _MAX_ETERNAFOLD_POST_BYTES:
            raise RuntimeError("EternaFold posteriors output exceeded byte limit")
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


class VultrCASClient:
    """Minimal stdlib HTTP client for Vultr CAS execution submission, lease keepalive, and commit."""

    def __init__(self, gateway_url: str, *, access_token: str = "", timeout_sec: float = 15.0) -> None:
        self.base_url = validate_gateway_url(gateway_url)
        self._access_token = access_token.strip()
        self.timeout_sec = timeout_sec
        host = (urllib.parse.urlsplit(self.base_url).hostname or "").lower()
        handlers: list[Any] = [_NoRedirectHandler()]
        if host in ("127.0.0.1", "localhost", "::1"):
            handlers.append(urllib.request.ProxyHandler({}))
        self._opener = urllib.request.build_opener(*handlers)

    def post_json(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        data = json.dumps(body, separators=(",", ":")).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._access_token:
            headers["X-MRNAX-Access-Token"] = self._access_token
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with self._opener.open(req, timeout=self.timeout_sec) as resp:
            raw = resp.read(_MAX_HTTP_RESPONSE_BYTES + 1)
            if len(raw) > _MAX_HTTP_RESPONSE_BYTES:
                raise RuntimeError("Gateway CAS response exceeded byte cap")
            decoded = json.loads(raw.decode("utf-8"))
            if not isinstance(decoded, dict):
                raise RuntimeError("Gateway CAS response must be a JSON object")
            return decoded

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
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"worker_id": worker_id, "ttl_seconds": ttl_seconds}
        if lease_token is not None:
            payload["lease_token"] = lease_token
        safe_exec = urllib.parse.quote(execution_id, safe="")
        return self.post_json(f"/v1/executions/{safe_exec}/leases", payload)

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


def run_worker(
    *,
    protein_seq: str,
    compliance_track: str = "research_max",
    lambdas: tuple[float, ...] = (1.0, 2.0),
    include_eternafold: bool = False,
    lineardesign_tar: Path | None = None,
    eternafold_tar: Path | None = None,
    lineardesign_bin: Path | None = None,
    work_dir: Path | None = None,
    gateway_url: str | None = None,
    execution_id: str | None = None,
    attempt_id: str | None = None,
    worker_id: str = "colab_heavy_sota_worker",
    ttl_seconds: int = 120,
    submission_id: str | None = None,
    request_digest: str | None = None,
    intent_id: str = "int_colab_sota",
    sealed_file_id: str | None = None,
    output_json: Path | None = None,
    heartbeat_interval_sec: float | None = None,
) -> dict[str, Any]:
    """Execute Colab heavy SOTA workload with strict clean-room isolation, pinned tar verification, and CAS keepalive."""
    if compliance_track not in ("research_max", "commercial_clean"):
        raise ValueError(f"Unsupported compliance_track '{compliance_track}'")
    if compliance_track == "commercial_clean":
        raise PermissionError(
            "Track isolation blocked: LinearDesign and EternaFold are strictly disallowed under 'commercial_clean'."
        )

    sense_aa = normalize_protein_sequence(protein_seq)
    eff_ttl = min(600, max(15, int(ttl_seconds)))
    eff_interval = (
        max(5.0, eff_ttl / 2.0)
        if heartbeat_interval_sec is None
        else max(0.01, float(heartbeat_interval_sec))
    )
    access_token = os.environ.get("MRNAX_GATEWAY_ACCESS_TOKEN", "").strip()
    cas_client = VultrCASClient(gateway_url, access_token=access_token) if gateway_url else None

    eff_exec_id = (execution_id or "").strip() or None
    if cas_client is not None and eff_exec_id is None:
        eff_sub_id = (submission_id or "").strip() or f"sub_colab_{hashlib.sha256(sense_aa.encode('ascii')).hexdigest()[:12]}"
        eff_req_dig = (request_digest or "").strip() or hashlib.sha256(
            json.dumps({"protein": sense_aa, "lambdas": list(lambdas), "ef": include_eternafold}, sort_keys=True).encode("utf-8")
        ).hexdigest()
        sub_row = cas_client.submit_execution(
            submission_id=eff_sub_id,
            request_digest=eff_req_dig,
            intent_id=intent_id,
        )
        eff_exec_id = str(sub_row["execution_id"])

    lease_grant: dict[str, Any] | None = None
    if cas_client is not None and eff_exec_id is not None:
        lease_grant = cas_client.acquire_or_renew_lease(
            eff_exec_id,
            worker_id=worker_id,
            ttl_seconds=eff_ttl,
        )

    stop_event = threading.Event()
    renewal_count = 0
    renewal_lock = threading.Lock()
    renewal_errors: list[str] = []

    def _heartbeat_loop() -> None:
        nonlocal renewal_count
        assert cas_client is not None and eff_exec_id is not None and lease_grant is not None
        tok = str(lease_grant["lease_token"])
        while not stop_event.wait(timeout=eff_interval):
            try:
                cas_client.acquire_or_renew_lease(
                    eff_exec_id,
                    worker_id=worker_id,
                    ttl_seconds=eff_ttl,
                    lease_token=tok,
                )
                with renewal_lock:
                    renewal_count += 1
            except Exception as exc:
                with renewal_lock:
                    renewal_errors.append(str(exc))
                break

    hb_thread: threading.Thread | None = None
    if lease_grant is not None:
        hb_thread = threading.Thread(target=_heartbeat_loop, name="mrnax-cas-keepalive", daemon=True)
        hb_thread.start()

    try:
        with tempfile.TemporaryDirectory(prefix="mrnax_colab_sota_") as tmp_td:
            eff_work = work_dir.resolve() if work_dir is not None else Path(tmp_td)
            eff_work.mkdir(parents=True, exist_ok=True)

            if lineardesign_tar is not None:
                ld_dir = verify_and_extract_tarball(
                    lineardesign_tar,
                    eff_work / "lineardesign",
                    expected_sha256=PINNED_LINEARDESIGN_TAR_SHA256,
                )
                ld_bin, ld_runtime = _locate_lineardesign_binary(ld_dir)
            elif lineardesign_bin is not None:
                ld_bin = lineardesign_bin.resolve()
                if not ld_bin.is_file():
                    raise FileNotFoundError(f"LinearDesign binary not found: '{ld_bin}'")
                ld_runtime = ld_bin.parent
            else:
                raise FileNotFoundError("Provide --lineardesign-tar (pinned SHA-256) or --lineardesign-bin.")

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
                    eff_work / "eternafold",
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

        eff_attempt_id = str(lease_grant["attempt_id"]) if lease_grant is not None else (attempt_id or "")
        result_core: dict[str, Any] = {
            "status": "COMPUTED",
            "compliance_track": compliance_track,
            "worker_id": worker_id,
            "protein_length_aa": len(sense_aa),
            "candidates": candidates,
            "eternafold_metrics": ef_metrics,
            "execution_id": eff_exec_id,
            "attempt_id": eff_attempt_id or None,
            "lease_epoch": lease_grant["lease_epoch"] if lease_grant is not None else None,
            "heartbeat_renewals": renewals_snapshot,
            "heartbeat_errors": errors_snapshot,
        }
        canonical_bytes = (json.dumps(result_core, indent=2, sort_keys=True) + "\n").encode("utf-8")
        result_sha256 = hashlib.sha256(canonical_bytes).hexdigest()
        result_core["result_sha256"] = result_sha256

        if cas_client is not None and eff_exec_id is not None and lease_grant is not None:
            eff_sealed_id = (
                sealed_file_id
                or (str(output_json.resolve()) if output_json is not None else f"colab://result/{eff_exec_id}/{eff_attempt_id}")
            )
            commit_receipt = cas_client.commit_attempt(
                execution_id=eff_exec_id,
                attempt_id=eff_attempt_id,
                lease_epoch=int(lease_grant["lease_epoch"]),
                lease_token=str(lease_grant["lease_token"]),
                sealed_file_id=eff_sealed_id,
                sealed_sha256=result_sha256,
            )
            result_core["status"] = str(commit_receipt.get("status", "COMMITTED"))
            result_core["cas_commit"] = commit_receipt

        if output_json is not None:
            final_bytes = (json.dumps(result_core, indent=2, sort_keys=True) + "\n").encode("utf-8")
            _atomic_write_bytes(output_json, final_bytes)

        return result_core
    finally:
        stop_event.set()
        if hb_thread is not None:
            hb_thread.join(timeout=5.0)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Colab Heavy SOTA Worker (LinearDesign + EternaFold) with Vultr CAS Keepalive")
    p.add_argument("--protein-seq", default="", help="Primary target protein sequence (canonical AA)")
    p.add_argument("--job-json", type=Path, default=None, help="Optional JSON job descriptor file")
    p.add_argument("--compliance-track", default="research_max", choices=("research_max", "commercial_clean"))
    p.add_argument("--lambdas", default="1.0,2.0", help="Comma-separated LinearDesign lambda weights")
    p.add_argument("--include-eternafold", action="store_true", help="Also run EternaFold AUP scoring")
    p.add_argument("--lineardesign-tar", type=Path, default=None, help="Path to pinned LinearDesign Linux tarball")
    p.add_argument("--eternafold-tar", type=Path, default=None, help="Path to pinned EternaFold Linux tarball")
    p.add_argument("--lineardesign-bin", type=Path, default=None, help="Optional pre-extracted local binary path")
    p.add_argument("--work-dir", type=Path, default=None, help="Extraction/working directory")
    p.add_argument("--gateway-url", default=None, help="Vultr control-plane HTTPS/localhost URL")
    p.add_argument("--execution-id", default=None, help="Pre-submitted CAS execution_id")
    p.add_argument("--attempt-id", default=None, help="Optional attempt_id hint")
    p.add_argument("--worker-id", default="colab_heavy_sota_worker", help="CAS worker identifier")
    p.add_argument("--ttl-seconds", type=int, default=120, help="CAS lease TTL in seconds (clamped [15, 600])")
    p.add_argument("--submission-id", default=None, help="Optional CAS submission_id for auto-submit")
    p.add_argument("--request-digest", default=None, help="Optional CAS request_digest for auto-submit")
    p.add_argument("--intent-id", default="int_colab_sota", help="Intent ID for CAS auto-submit")
    p.add_argument("--sealed-file-id", default=None, help="Drive SEALED.json file ID or artifact URI")
    p.add_argument("--output-json", type=Path, default=None, help="Output JSON artifact path")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    protein_seq = args.protein_seq
    compliance_track = args.compliance_track
    lambdas_raw: str | list[float] = args.lambdas
    include_eternafold = bool(args.include_eternafold)

    if args.job_json is not None:
        job_doc = json.loads(args.job_json.read_text(encoding="utf-8"))
        if not isinstance(job_doc, dict):
            print("Error: --job-json must decode to a JSON object", file=sys.stderr)
            return 1
        protein_seq = str(job_doc.get("protein_seq") or job_doc.get("protein_sequence") or protein_seq)
        compliance_track = str(job_doc.get("compliance_track") or compliance_track)
        if "lambdas" in job_doc:
            lambdas_raw = job_doc["lambdas"]
        if "include_eternafold" in job_doc:
            include_eternafold = bool(job_doc["include_eternafold"])

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
            output_json=args.output_json,
        )
    except PermissionError as exc:
        print(f"PermissionError: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Error: {type(exc).__name__}: {exc}", file=sys.stderr)
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

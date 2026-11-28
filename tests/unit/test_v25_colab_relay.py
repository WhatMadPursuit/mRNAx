"""Unit & CAS integration tests for Vultr Compute-Tier status & Colab Heavy SOTA Worker."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tarfile
import threading
import time
from typing import Any

from Bio.Data import CodonTable
from fastapi.testclient import TestClient
import pytest
import uvicorn

from mrnax.cloud.cas_store import ExecutionCASStore
from mrnax.cloud.gateway import create_control_plane_app

REPO_ROOT = Path(__file__).resolve().parents[2]
SEED_DIR = REPO_ROOT / "registry_seed"
WORKER_SCRIPT = REPO_ROOT / "scripts" / "colab_heavy_sota_worker.py"

if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import colab_heavy_sota_worker as worker_mod  # noqa: E402


def _make_tar_gz(tar_path: Path, entries: list[tuple[str, bytes, int, str | None]]) -> str:
    """Create a tar.gz archive from `(name, data, mode, linkname)` entries and return its SHA-256."""
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "w:gz") as tf:
        for name, data, mode, linkname in entries:
            ti = tarfile.TarInfo(name=name)
            ti.mode = mode
            if linkname is not None:
                ti.type = tarfile.SYMTYPE
                ti.linkname = linkname
                ti.size = 0
                tf.addfile(ti)
            else:
                ti.size = len(data)
                tf.addfile(ti, io.BytesIO(data))
    return hashlib.sha256(tar_path.read_bytes()).hexdigest()


def test_system_status_exposes_compute_tier_and_pinned_assets(tmp_path: Path) -> None:
    app = create_control_plane_app(
        ExecutionCASStore(tmp_path / "status_cp.sqlite3"),
        projects_root=tmp_path / "projects",
        seed_dir=SEED_DIR,
    )
    client = TestClient(app)
    resp = client.get("/v1/system/status")
    assert resp.status_code == 200
    body = resp.json()

    # Preserve existing keys alongside compute_tier
    assert body["status"] == "online"
    assert body["default_profile"] == "research_max"
    assert body["supported_modalities"] == ["linear_mrna", "circrna", "sarna"]
    assert "available_engines" in body

    ct = body["compute_tier"]
    assert ct["gateway_role"] == "lightweight_relay_and_keepalive_control_plane"
    assert ct["heavy_compute_backend"] == "google_colab_and_drive"
    assert ct["vps_heavy_binaries_disabled_by_policy"] is True
    assert ct["lease_keepalive_endpoints"] == [
        "/v1/executions",
        "/v1/executions/{execution_id}/leases",
        "/v1/executions/{execution_id}/attempts/{attempt_id}:commit",
    ]
    assets = ct["colab_drive_sota_assets"]
    assert assets["lineardesign_linux_x86_64"] == {
        "drive_file_id": "1ZHpUFjRylZ2pju0w-tOBa5SOgmmNomvQ",
        "sha256": worker_mod.PINNED_LINEARDESIGN_TAR_SHA256,
        "license_track": "research_max",
    }
    assert assets["eternafold_v131_linux_x86_64"] == {
        "drive_file_id": "1Aio8lOmikCCMuA4KhbJhwnvlc4EDA_jn",
        "sha256": worker_mod.PINNED_ETERNAFOLD_TAR_SHA256,
        "license_track": "commercial_clearance_pending",
    }
    assert assets["surrogate_v2_folder"] == {
        "drive_folder_id": "1rIEgLRuJg10ZJPpHSo_rC-IDK2XQFzW6",
        "license_track": "commercial_clean",
    }
    assert assets["codontransformer_rc005"] == {
        "hf_repo": "adibvafa/CodonTransformer",
        "license_track": "commercial_clean",
    }


def test_colab_worker_commercial_clean_refusal_sha256_tar_and_url_guards(tmp_path: Path) -> None:
    short_prot = "MVSKGEELFTGVVPILVELD"

    # 1. Commercial-clean refusal (in-process PermissionError AND CLI exit code 2)
    with pytest.raises(PermissionError, match="commercial_clean"):
        worker_mod.run_worker(
            protein_seq=short_prot,
            compliance_track="commercial_clean",
            include_eternafold=True,
        )
    proc_cc = subprocess.run(
        [
            sys.executable,
            str(WORKER_SCRIPT),
            "--protein-seq",
            short_prot,
            "--compliance-track",
            "commercial_clean",
            "--include-eternafold",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc_cc.returncode == 2
    assert "PermissionError" in proc_cc.stderr

    # 2. Tampered tarball fails SHA-256 verification before tarfile opening/extraction
    good_tar = tmp_path / "good.tar.gz"
    expected_sha = _make_tar_gz(good_tar, [("bin/hello.txt", b"hello", 0o644, None)])
    tampered_tar = tmp_path / "tampered.tar.gz"
    raw = bytearray(good_tar.read_bytes())
    raw[-1] ^= 0xFF
    tampered_tar.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        worker_mod.verify_and_extract_tarball(tampered_tar, tmp_path / "out_tamper", expected_sha256=expected_sha)
    assert not (tmp_path / "out_tamper").exists()

    # 3. Malicious tar traversal (`../escape`) and symlink are rejected with zero extracted files
    escape_tar = tmp_path / "escape.tar.gz"
    escape_sha = _make_tar_gz(
        escape_tar,
        [
            ("safe.txt", b"ok", 0o644, None),
            ("../outside_escape.txt", b"pwned", 0o644, None),
        ],
    )
    with pytest.raises(ValueError, match="traversal"):
        worker_mod.verify_and_extract_tarball(escape_tar, tmp_path / "out_escape", expected_sha256=escape_sha)
    assert not (tmp_path / "outside_escape.txt").exists()
    assert not (tmp_path / "out_escape" / "safe.txt").exists()

    sym_tar = tmp_path / "symlink.tar.gz"
    sym_sha = _make_tar_gz(sym_tar, [("bad_link", b"", 0o777, "/etc/passwd")])
    with pytest.raises(ValueError, match="Symlink"):
        worker_mod.verify_and_extract_tarball(sym_tar, tmp_path / "out_sym", expected_sha256=sym_sha)
    assert not (tmp_path / "out_sym" / "bad_link").exists()

    abs_tar = tmp_path / "abs.tar.gz"
    abs_sha = _make_tar_gz(abs_tar, [("/tmp/evil.txt", b"x", 0o644, None)])
    with pytest.raises(ValueError, match="Absolute path"):
        worker_mod.verify_and_extract_tarball(abs_tar, tmp_path / "out_abs", expected_sha256=abs_sha)

    # 4. Gateway URL guardrails (HTTPS or localhost/127.0.0.1 only; zero creds; zero metadata IPs)
    assert worker_mod.validate_gateway_url("https://relay.example.com/") == "https://relay.example.com"
    assert worker_mod.validate_gateway_url("http://127.0.0.1:8080") == "http://127.0.0.1:8080"
    assert worker_mod.validate_gateway_url("http://localhost:9000") == "http://localhost:9000"
    for bad_url in (
        "http://relay.example.com",
        "http://169.254.169.254/latest/meta-data/",
        "https://169.254.169.254/latest",
        "https://user:pass@relay.example.com",
        "ftp://localhost:8080",
        "https://relay.example.com?token=leak",
    ):
        with pytest.raises(ValueError):
            worker_mod.validate_gateway_url(bad_url)

    # 5. Built-in stdlib codon table matches NCBI Translation Table 1 across all 64 codons
    ncbi_t1 = CodonTable.unambiguous_rna_by_id[1]
    assert len(worker_mod._CODON_TABLE_1) == 64
    for codon, aa in worker_mod._CODON_TABLE_1.items():
        if codon in ncbi_t1.stop_codons:
            assert aa == "*"
        else:
            assert aa == ncbi_t1.forward_table[codon]
    assert worker_mod.verify_exact_aa_translation("AUGGUGAGCAAG", "MVSK") is True
    assert worker_mod.verify_exact_aa_translation("AUGGUGAGCAAG", "MVSA") is False


def test_colab_worker_full_cas_lifecycle_heartbeat_and_zero_secret_leakage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admin_secret = "tok_ColabRelaySecretKeepPrivate987654"
    store = ExecutionCASStore(tmp_path / "relay_cas.sqlite3")
    app = create_control_plane_app(
        store,
        projects_root=tmp_path / "projects",
        seed_dir=SEED_DIR,
        access_token=admin_secret,
    )

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(16)
    port = int(sock.getsockname()[1])
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    srv_thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    srv_thread.start()
    deadline = time.monotonic() + 10.0
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.02)
    assert server.started

    try:
        # Build pinned tarballs containing stubbed LinearDesign and EternaFold executables
        snap = json.loads((SEED_DIR / "egfp_reporter_snapshot.json").read_text(encoding="utf-8"))
        egfp_prot: str = snap["mammalian_reporter_egfp"]["protein_sequence"][:20]
        egfp_cds: str = snap["mammalian_reporter_egfp"]["reference_cds_rna_no_stop"][:60]

        ld_stub_py = (
            f"#!{sys.executable}\n"
            "import sys, time\n"
            "_ = sys.stdin.read()\n"
            "lam = sys.argv[sys.argv.index('-l') + 1] if '-l' in sys.argv else '1.0'\n"
            "time.sleep(0.18)\n"
            f"print('mRNA sequence:  {egfp_cds}')\n"
            "print(f'mRNA folding free energy: -42.50 kcal/mol; mRNA CAI: 0.880')\n"
        ).encode("utf-8")
        ld_tar = tmp_path / "lineardesign_stub.tar.gz"
        ld_sha = _make_tar_gz(
            ld_tar,
            [
                ("bin/lineardesign", ld_stub_py, 0o755, None),
                ("bin/coding_wheel.txt", b"wheel\n", 0o644, None),
                ("bin/codon_usage_freq_table_human.csv", b"codon,freq\n", 0o644, None),
            ],
        )

        ef_stub_py = (
            f"#!{sys.executable}\n"
            "import sys\n"
            "fa_path = sys.argv[2]\n"
            "post_path = sys.argv[sys.argv.index('--posteriors') + 2]\n"
            "seq = [ln.strip() for ln in open(fa_path) if not ln.startswith('>')][0]\n"
            "lines = [f'{i + 1} {ch}' for i, ch in enumerate(seq)]\n"
            "open(post_path, 'w').write('\\n'.join(lines) + '\\n')\n"
        ).encode("utf-8")
        ef_tar = tmp_path / "eternafold_stub.tar.gz"
        ef_sha = _make_tar_gz(
            ef_tar,
            [
                ("bin/contrafold", ef_stub_py, 0o755, None),
                ("bin/EternaFoldParams.v1", b"params\n", 0o644, None),
            ],
        )

        monkeypatch.setattr(worker_mod, "PINNED_LINEARDESIGN_TAR_SHA256", ld_sha)
        monkeypatch.setattr(worker_mod, "PINNED_ETERNAFOLD_TAR_SHA256", ef_sha)
        monkeypatch.setenv("MRNAX_GATEWAY_ACCESS_TOKEN", admin_secret)

        # Pre-submit execution via authenticated HTTP gateway endpoint
        import httpx

        base_url = f"http://127.0.0.1:{port}"
        with httpx.Client(trust_env=False, timeout=5.0) as local_http:
            sub_resp = local_http.post(
                f"{base_url}/v1/executions",
                headers={"X-MRNAX-Access-Token": admin_secret},
                json={
                    "submission_id": "sub_colab_relay_01",
                    "request_digest": "req_dig_01",
                    "intent_id": "int_colab_01",
                },
            )
        assert sub_resp.status_code == 200
        exec_id = sub_resp.json()["execution_id"]

        out_json = tmp_path / "colab_out.json"
        res = worker_mod.run_worker(
            protein_seq=egfp_prot,
            compliance_track="research_max",
            lambdas=(1.0, 2.0),
            include_eternafold=True,
            lineardesign_tar=ld_tar,
            eternafold_tar=ef_tar,
            work_dir=tmp_path / "work",
            gateway_url=base_url,
            execution_id=exec_id,
            worker_id="colab_l4_worker_01",
            ttl_seconds=30,
            output_json=out_json,
            heartbeat_interval_sec=0.05,
        )
        assert res["status"] == "COMMITTED"
        assert res["execution_id"] == exec_id
        assert res["heartbeat_renewals"] >= 2
        assert len(res["candidates"]) == 2
        assert all(c["aa_exact_match"] is True for c in res["candidates"])
        assert len(res["eternafold_metrics"]) == 2
        assert all(m["eternafold_aup"] == 1.0 for m in res["eternafold_metrics"])

        # Verify persisted CAS store state and zero secret leakage in output artifact
        saved_text = out_json.read_text(encoding="utf-8")
        assert admin_secret not in saved_text
        exec_row = store.get_execution(exec_id)
        assert exec_row.status == "COMMITTED"
        assert exec_row.winning_attempt_id == res["attempt_id"]
        assert exec_row.winning_sealed_sha256 == res["result_sha256"]

        # Verify standalone CLI subprocess execution via --job-json (auto-submit + pinned binary sha256 + immutable artifact)
        cli_extract = worker_mod.verify_and_extract_tarball(
            ld_tar, tmp_path / "cli_bin_extract", expected_sha256=ld_sha
        )
        cli_bin = cli_extract / "bin" / "lineardesign"
        cli_bin_sha = worker_mod.compute_file_sha256(cli_bin)
        cli_out = tmp_path / "colab_cli_out.json"
        cli_receipt = tmp_path / "colab_cli_out.commit_receipt.json"
        job_json_path = tmp_path / "job_spec.json"
        job_json_path.write_text(
            json.dumps({"protein_seq": egfp_prot, "compliance_track": "research_max", "lambdas": [1.0]}),
            encoding="utf-8",
        )
        proc_cli = subprocess.run(
            [
                sys.executable,
                str(WORKER_SCRIPT),
                "--job-json",
                str(job_json_path),
                "--lineardesign-bin",
                str(cli_bin),
                "--lineardesign-bin-sha256",
                cli_bin_sha,
                "--gateway-url",
                base_url,
                "--submission-id",
                "sub_colab_cli_auto_01",
                "--output-json",
                str(cli_out),
            ],
            env={
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "MRNAX_GATEWAY_ACCESS_TOKEN": admin_secret,
            },
            capture_output=True,
            text=True,
            timeout=15.0,
            check=True,
        )
        assert admin_secret not in proc_cli.stdout
        assert admin_secret not in proc_cli.stderr
        assert admin_secret not in cli_out.read_text(encoding="utf-8")
        cli_doc = json.loads(cli_out.read_text(encoding="utf-8"))
        assert cli_doc["status"] == "COMPUTED"
        assert cli_receipt.is_file()
        receipt_doc = json.loads(cli_receipt.read_text(encoding="utf-8"))
        assert receipt_doc["status"] == "COMMITTED"
        assert receipt_doc["result_sha256"] == hashlib.sha256(cli_out.read_bytes()).hexdigest()
        assert receipt_doc["cas_commit"]["status"] == "COMMITTED"
        assert receipt_doc["cas_commit"]["winning_sealed_file_id"] == str(cli_out.resolve())

        # Missing access token against authenticated gateway fails closed with HTTP 401
        import urllib.error
        import urllib.request

        monkeypatch.delenv("MRNAX_GATEWAY_ACCESS_TOKEN", raising=False)
        with pytest.raises(urllib.error.HTTPError) as auth_exc:
            worker_mod.run_worker(
                protein_seq=egfp_prot,
                lineardesign_bin=cli_bin,
                lineardesign_bin_sha256=cli_bin_sha,
                gateway_url=base_url,
                submission_id="sub_unauth_fail",
            )
        assert auth_exc.value.code == 401

        # Redirects are strictly rejected by _NoRedirectHandler
        dummy_req = urllib.request.Request(f"{base_url}/v1/executions", method="POST")
        with pytest.raises(urllib.error.HTTPError) as redir_exc:
            worker_mod._NoRedirectHandler().redirect_request(
                dummy_req, io.BytesIO(b""), 302, "Found", {}, "https://evil.example/steal"
            )
        assert redir_exc.value.code == 302
    finally:
        server.should_exit = True
        srv_thread.join(timeout=5.0)
        sock.close()


def test_colab_worker_production_drive_seal_commit_and_security_hardening(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mrnax.cloud.drive_rest_v3 import DRIVE_FOLDER_MIME, DriveRestV3Client, InMemoryDriveV3Transport

    admin_secret = "tok_ProdSealSecret99887766"
    drv_transport = InMemoryDriveV3Transport()
    runs_root_id = drv_transport.create_file(parent_id="root", name="runs", mime_type=DRIVE_FOLDER_MIME)["id"]
    commits_id = drv_transport.create_file(parent_id="root", name="commits", mime_type=DRIVE_FOLDER_MIME)["id"]
    drive_client = DriveRestV3Client(drv_transport)

    store = ExecutionCASStore(
        tmp_path / "prod_cas.sqlite3",
        drive_verifier=drive_client,
        commits_snapshot_folder_id=commits_id,
        runs_root_id=runs_root_id,
        require_drive_verifier=True,
    )
    app = create_control_plane_app(
        store,
        projects_root=tmp_path / "projects",
        seed_dir=SEED_DIR,
        access_token=admin_secret,
    )

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(16)
    port = int(sock.getsockname()[1])
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    srv_thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    srv_thread.start()
    deadline = time.monotonic() + 10.0
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.02)
    assert server.started

    try:
        snap = json.loads((SEED_DIR / "egfp_reporter_snapshot.json").read_text(encoding="utf-8"))
        egfp_prot: str = snap["mammalian_reporter_egfp"]["protein_sequence"][:20]
        egfp_cds: str = snap["mammalian_reporter_egfp"]["reference_cds_rna_no_stop"][:60]

        ld_stub_py = (
            f"#!{sys.executable}\n"
            "import sys\n"
            "_ = sys.stdin.read()\n"
            f"print('mRNA sequence:  {egfp_cds}')\n"
            "print('mRNA folding free energy: -41.00 kcal/mol; mRNA CAI: 0.890')\n"
        ).encode("utf-8")
        ld_tar = tmp_path / "ld_prod.tar.gz"
        ld_sha = _make_tar_gz(
            ld_tar,
            [
                ("bin/lineardesign", ld_stub_py, 0o755, None),
                ("bin/coding_wheel.txt", b"wheel\n", 0o644, None),
                ("bin/codon_usage_freq_table_human.csv", b"codon,freq\n", 0o644, None),
            ],
        )
        monkeypatch.setattr(worker_mod, "PINNED_LINEARDESIGN_TAR_SHA256", ld_sha)
        monkeypatch.setenv("MRNAX_GATEWAY_ACCESS_TOKEN", admin_secret)

        # Reused --work-dir containing a rogue pre-existing executable must NOT hijack verified extraction
        reused_work = tmp_path / "reused_work"
        rogue_dir = reused_work / "lineardesign" / "bin"
        rogue_dir.mkdir(parents=True, exist_ok=True)
        (rogue_dir / "lineardesign").write_text("#!/bin/sh\necho HIJACKED\nexit 9\n", encoding="utf-8")
        (rogue_dir / "lineardesign").chmod(0o755)

        out_art = tmp_path / "prod_result.json"
        res = worker_mod.run_worker(
            protein_seq=egfp_prot,
            compliance_track="research_max",
            lambdas=(1.0,),
            lineardesign_tar=ld_tar,
            work_dir=reused_work,
            gateway_url=f"http://127.0.0.1:{port}",
            drive_runs_root_id=runs_root_id,
            drive_client=drive_client,
            output_json=out_art,
        )
        assert res["status"] == "COMMITTED"
        assert res["sealed_file_id"] and res["sealed_sha256"]
        assert hashlib.sha256(out_art.read_bytes()).hexdigest() == res["result_sha256"]
        row = store.get_execution(res["execution_id"])
        assert row.status == "COMMITTED"
        assert row.winning_sealed_file_id == res["sealed_file_id"]
        assert row.winning_sealed_sha256 == res["sealed_sha256"]

        # Replaying an already-committed submission fails early before acquiring a doomed lease
        with pytest.raises(RuntimeError, match="already COMMITTED"):
            worker_mod.run_worker(
                protein_seq=egfp_prot,
                compliance_track="research_max",
                lambdas=(1.0,),
                lineardesign_tar=ld_tar,
                gateway_url=f"http://127.0.0.1:{port}",
                drive_runs_root_id=runs_root_id,
                drive_client=drive_client,
            )

        # Conflicting CLI vs --job-json compliance_track fails closed (exit code 2) before lease/extraction
        conflict_job = tmp_path / "conflict_job.json"
        conflict_job.write_text(
            json.dumps({"protein_seq": egfp_prot, "compliance_track": "research_max", "lambdas": [1.0]}),
            encoding="utf-8",
        )
        proc_conflict = subprocess.run(
            [
                sys.executable,
                str(WORKER_SCRIPT),
                "--job-json",
                str(conflict_job),
                "--compliance-track",
                "commercial_clean",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc_conflict.returncode == 2
        assert "Conflicting compliance_track" in proc_conflict.stderr

        # Empty lambdas list, newline token, private IP gateway, and unsealed HTTPS commit fail closed
        with pytest.raises(ValueError, match="At least one lambda"):
            worker_mod.parse_lambdas([])
        monkeypatch.setenv("MRNAX_GATEWAY_ACCESS_TOKEN", "bad\nsecret")
        with pytest.raises(ValueError, match="control characters"):
            worker_mod.run_worker(protein_seq=egfp_prot, lineardesign_tar=ld_tar)
        monkeypatch.setenv("MRNAX_GATEWAY_ACCESS_TOKEN", admin_secret)
        with pytest.raises(ValueError, match="Blocked private"):
            worker_mod.validate_gateway_url("https://10.0.0.5:8443")
        with pytest.raises(ValueError, match="Remote HTTPS gateway commit requires Drive sealing"):
            worker_mod.run_worker(
                protein_seq=egfp_prot,
                lineardesign_tar=ld_tar,
                gateway_url="https://149-248-39-200.sslip.io:8443",
            )
    finally:
        server.should_exit = True
        srv_thread.join(timeout=5.0)
        sock.close()

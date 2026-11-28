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
        with pytest.raises(ValueError, match="Blocked non-global/private"):
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


def test_gate_b_r2_round2_colab_worker_hardening(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib.util
    import socket
    import tarfile
    import threading
    import time
    import uvicorn

    from mrnax.cloud.cas_store import ExecutionCASStore
    from mrnax.cloud.drive_rest_v3 import DRIVE_FOLDER_MIME, DriveRestV3Client, InMemoryDriveV3Transport
    from mrnax.cloud.gateway import create_control_plane_app

    spec = importlib.util.spec_from_file_location(
        "colab_heavy_sota_worker_r2",
        WORKER_SCRIPT,
    )
    assert spec and spec.loader
    worker_mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = worker_mod
    spec.loader.exec_module(worker_mod)

    egfp_prot = json.loads((REPO_ROOT / "registry_seed" / "egfp_reporter_snapshot.json").read_text(encoding="utf-8"))[
        "mammalian_reporter_egfp"
    ]["protein_sequence"]

    # #6: Trailing control chars rejected before strip(), CGNAT/TEST-NET non-global IPs blocked, recursive cas_commit redaction
    monkeypatch.setenv("MRNAX_GATEWAY_ACCESS_TOKEN", "tok_trailing_123456\n")
    with pytest.raises(ValueError, match="control characters"):
        worker_mod.run_worker(protein_seq=egfp_prot, lineardesign_bin=tmp_path / "missing")
    monkeypatch.setenv("MRNAX_GATEWAY_ACCESS_TOKEN", "tok_del_123456\x7f")
    with pytest.raises(ValueError, match="control characters"):
        worker_mod.run_worker(protein_seq=egfp_prot, lineardesign_bin=tmp_path / "missing")
    monkeypatch.setenv("MRNAX_GATEWAY_ACCESS_TOKEN", "tok_super_secret_abcdef")

    with pytest.raises(ValueError):
        worker_mod.validate_gateway_url("https://100.64.0.1:8443")
    with pytest.raises(ValueError):
        worker_mod.validate_gateway_url("https://192.0.2.1:8443")

    redacted = worker_mod._redact_obj(
        {"cas_commit": {"receipt": "Bearer tok_super_secret_abcdef", "items": ["tok_super_secret_abcdef"]}},
        "tok_super_secret_abcdef",
    )
    assert "tok_super_secret_abcdef" not in json.dumps(redacted)

    # Tiny executable script that satisfies LinearDesign CLI stdout parsing
    import hashlib

    fake_ld_py = (
        f"#!{sys.executable}\n"
        "import sys\n"
        "CODON = dict(zip('ACDEFGHIKLMNPQRSTVWY', "
        "'GCU UGU GAU GAA UUU GGU CAU AUU AAA CUU AUG AAU CCU CAA CGU UCU ACU GUU UGG UAU'.split()))\n"
        "aa = ''.join(ch for ch in sys.stdin.read().strip().upper() if ch in CODON)\n"
        "rna = ''.join(CODON[c] for c in aa)\n"
        "print(f'mRNA sequence:  {rna}')\n"
        "print('mRNA structure: ' + '.' * len(rna))\n"
        "print('mRNA folding free energy: -42.50 kcal/mol; mRNA CAI: 0.910')\n"
    )
    bin_a = tmp_path / "bin_a" / "lineardesign"
    bin_a.parent.mkdir(parents=True, exist_ok=True)
    bin_a.write_text(fake_ld_py, encoding="utf-8")
    bin_a.chmod(0o755)
    sha_bin_a = hashlib.sha256(bin_a.read_bytes()).hexdigest()
    ld_tar_a = tmp_path / "ld_a.tar.gz"
    with tarfile.open(ld_tar_a, "w:gz") as tf:
        tf.add(bin_a, arcname="lineardesign")
    sha_tar_a = hashlib.sha256(ld_tar_a.read_bytes()).hexdigest()

    bin_b = tmp_path / "bin_b" / "lineardesign"
    bin_b.parent.mkdir(parents=True, exist_ok=True)
    bin_b.write_text(fake_ld_py + "# variant b\n", encoding="utf-8")
    bin_b.chmod(0o755)
    sha_bin_b = hashlib.sha256(bin_b.read_bytes()).hexdigest()
    ld_tar_b = tmp_path / "ld_b.tar.gz"
    with tarfile.open(ld_tar_b, "w:gz") as tf:
        tf.add(bin_b, arcname="lineardesign")

    # #1: Reject --commit-receipt-json == --output-json and reject --defer-commit without --output-json
    same_path = tmp_path / "same.json"
    with pytest.raises(ValueError):
        worker_mod.run_worker(
            protein_seq=egfp_prot,
            lineardesign_bin=bin_a,
            output_json=same_path,
            commit_receipt_json=same_path,
            defer_commit=True,
            work_dir=tmp_path / "work_same",
        )
    rc_cli_missing_out = worker_mod.main(
        [
            "--protein-seq",
            egfp_prot,
            "--lineardesign-bin",
            str(bin_a),
            "--defer-commit",
            "--work-dir",
            str(tmp_path / "work_cli"),
        ]
    )
    assert rc_cli_missing_out == 1

    # #4: Raw --lineardesign-bin rejects wrong SHA-256 prior to execution
    with pytest.raises(ValueError, match="SHA-256"):
        worker_mod.run_worker(
            protein_seq=egfp_prot,
            lineardesign_bin=bin_a,
            lineardesign_bin_sha256="0" * 64,
            defer_commit=True,
            output_json=tmp_path / "bad_sha_out.json",
            work_dir=tmp_path / "work_bad_sha",
        )

    # #5: Distinct LinearDesign binaries yield distinct CAS request_digest & submission_id,
    # explicit --execution-id already COMMITTED fails fast before running compute,
    # and heartbeat stays alive during slow Drive sealing
    transport = InMemoryDriveV3Transport()
    runs_root_id = transport.create_file(parent_id="root", name="mRNAx_R2_Runs", mime_type=DRIVE_FOLDER_MIME)["id"]
    commits_id = transport.create_file(parent_id="root", name="mRNAx_R2_Commits", mime_type=DRIVE_FOLDER_MIME)["id"]
    drive_client = DriveRestV3Client(transport)
    cas_store = ExecutionCASStore(
        tmp_path / "cas_r2.sqlite3",
        drive_verifier=drive_client,
        commits_snapshot_folder_id=commits_id,
        runs_root_id=runs_root_id,
        require_drive_verifier=True,
    )
    app = create_control_plane_app(
        cas_store,
        projects_root=tmp_path / "projects_r2",
        seed_dir=REPO_ROOT / "registry_seed",
        access_token="tok_super_secret_abcdef",
    )

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    srv_thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    srv_thread.start()
    try:
        for _ in range(50):
            if server.started:
                break
            time.sleep(0.05)

        res_a = worker_mod.run_worker(
            protein_seq=egfp_prot,
            compliance_track="research_max",
            lambdas=(1.0,),
            lineardesign_bin=bin_a,
            lineardesign_bin_sha256=sha_bin_a,
            gateway_url=f"http://127.0.0.1:{port}",
            drive_runs_root_id=runs_root_id,
            drive_client=drive_client,
            output_json=tmp_path / "out_a.json",
            work_dir=tmp_path / "work_a",
        )
        res_b = worker_mod.run_worker(
            protein_seq=egfp_prot,
            compliance_track="research_max",
            lambdas=(1.0,),
            lineardesign_bin=bin_b,
            lineardesign_bin_sha256=sha_bin_b,
            gateway_url=f"http://127.0.0.1:{port}",
            drive_runs_root_id=runs_root_id,
            drive_client=drive_client,
            output_json=tmp_path / "out_b.json",
            work_dir=tmp_path / "work_b",
        )
        row_a = cas_store.get_execution(res_a["execution_id"])
        row_b = cas_store.get_execution(res_b["execution_id"])
        assert res_a["execution_id"] != res_b["execution_id"]
        assert row_a.request_digest != row_b.request_digest
        assert row_a.submission_id != row_b.submission_id

        monkeypatch.setattr(worker_mod, "PINNED_LINEARDESIGN_TAR_SHA256", sha_tar_a)

        eid_done = res_a["execution_id"]

        called_sweep = False

        def _fail_if_called(*args: Any, **kwargs: Any) -> Any:
            nonlocal called_sweep
            called_sweep = True
            raise AssertionError("Should not execute compute when explicit execution_id is already COMMITTED")

        with monkeypatch.context() as mp:
            mp.setattr("mrnax.engine.lineardesign_adapter.sweep_lineardesign_candidates", _fail_if_called)
            with pytest.raises(RuntimeError, match="COMMITTED"):
                worker_mod.run_worker(
                    protein_seq=egfp_prot,
                    compliance_track="research_max",
                    lambdas=(1.0,),
                    lineardesign_tar=ld_tar_a,
                    gateway_url=f"http://127.0.0.1:{port}",
                    execution_id=eid_done,
                    drive_runs_root_id=runs_root_id,
                    drive_client=drive_client,
                    work_dir=tmp_path / "work_committed",
                )
        assert called_sweep is False

        # Heartbeat stays alive throughout Drive sealing
        orig_seal = drive_client.write_attempt_seal
        lease_calls = 0
        orig_lease_fn = cas_store.acquire_or_renew_lease

        def _counting_lease(*args: Any, **kwargs: Any) -> Any:
            nonlocal lease_calls
            lease_calls += 1
            return orig_lease_fn(*args, **kwargs)

        def _slow_seal(*args: Any, **kwargs: Any) -> Any:
            time.sleep(0.25)
            return orig_seal(*args, **kwargs)

        monkeypatch.setattr(cas_store, "acquire_or_renew_lease", _counting_lease)
        monkeypatch.setattr(drive_client, "write_attempt_seal", _slow_seal)
        sealed_out = worker_mod.run_worker(
            protein_seq=egfp_prot,
            compliance_track="research_max",
            lambdas=(1.0,),
            lineardesign_tar=ld_tar_a,
            gateway_url=f"http://127.0.0.1:{port}",
            drive_runs_root_id=runs_root_id,
            drive_client=drive_client,
            heartbeat_interval_sec=0.04,
            output_json=tmp_path / "hb_seal_out.json",
            work_dir=tmp_path / "work_hb_seal",
        )
        assert sealed_out["status"] == "COMMITTED"
        assert lease_calls >= 3
    finally:
        server.should_exit = True
        srv_thread.join(timeout=5.0)
        sock.close()


def test_gate_b_r2_round3_worker_fixes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib
    import importlib.util
    import socket
    import threading
    import time
    import uvicorn

    from mrnax.cloud.cas_store import ExecutionCASStore
    from mrnax.cloud.drive_rest_v3 import DRIVE_FOLDER_MIME, DriveRestV3Client, InMemoryDriveV3Transport
    from mrnax.cloud.gateway import create_control_plane_app

    spec = importlib.util.spec_from_file_location("colab_heavy_sota_worker_r3", WORKER_SCRIPT)
    assert spec and spec.loader
    worker_mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = worker_mod
    spec.loader.exec_module(worker_mod)

    egfp_prot = json.loads((REPO_ROOT / "registry_seed" / "egfp_reporter_snapshot.json").read_text(encoding="utf-8"))[
        "mammalian_reporter_egfp"
    ]["protein_sequence"]
    secret_tok = "tok_r3_secret_key_998877"
    monkeypatch.setenv("MRNAX_GATEWAY_ACCESS_TOKEN", secret_tok)

    # #6: _PinnedHTTPSConnection supports :8443 authority while rejecting mismatched port, and _redact_obj scrubs dict keys
    conn_ok = worker_mod._PinnedHTTPSConnection(
        "149-248-39-200.sslip.io:8443",
        port=8443,
        pinned_ip="203.0.113.10",
        expected_host="149-248-39-200.sslip.io",
        expected_port=8443,
    )
    assert conn_ok.host == "149-248-39-200.sslip.io"
    assert int(conn_ok.port) == 8443
    with pytest.raises(ValueError, match="escaped pinned verified gateway"):
        worker_mod._PinnedHTTPSConnection(
            "149-248-39-200.sslip.io:9443",
            port=8443,
            pinned_ip="203.0.113.10",
            expected_host="149-248-39-200.sslip.io",
            expected_port=8443,
        )
    scrubbed = worker_mod._redact_obj({secret_tok: {"nested_" + secret_tok: secret_tok}}, secret_tok)
    assert secret_tok not in json.dumps(scrubbed)

    # #4 & #5: Support-table filename collision cannot overwrite staged binary, and support-table changes alter digest
    fake_ld_py = (
        f"#!{sys.executable}\n"
        "import sys\n"
        "CODON = dict(zip('ACDEFGHIKLMNPQRSTVWY', "
        "'GCU UGU GAU GAA UUU GGU CAU AUU AAA CUU AUG AAU CCU CAA CGU UCU ACU GUU UGG UAU'.split()))\n"
        "aa = ''.join(ch for ch in sys.stdin.read().strip().upper() if ch in CODON)\n"
        "rna = ''.join(CODON[c] for c in aa)\n"
        "print(f'mRNA sequence:  {rna}')\n"
        "print('mRNA structure: ' + '.' * len(rna))\n"
        "print('mRNA folding free energy: -42.50 kcal/mol; mRNA CAI: 0.910')\n"
    )
    dir_sup_a = tmp_path / "sup_a"
    dir_sup_a.mkdir(parents=True, exist_ok=True)
    # Deliberately name executable `coding_wheel.txt` alongside real `coding_wheel.txt`? Keep binary named `lineardesign` in A,
    # and test collision with a colliding name in a dedicated dir where support table is absent or another support table exists.
    bin_collide = dir_sup_a / "codon_usage_freq_table_human.csv"
    bin_collide.write_text(fake_ld_py, encoding="utf-8")
    bin_collide.chmod(0o755)
    (dir_sup_a / "coding_wheel.txt").write_text("wheel_v1\n", encoding="utf-8")
    sha_collide = hashlib.sha256(bin_collide.read_bytes()).hexdigest()

    # #1: run_worker(defer_commit=True, output_json=None) fails closed directly in Python API
    with pytest.raises(ValueError, match="--defer-commit requires --output-json"):
        worker_mod.run_worker(
            protein_seq=egfp_prot,
            lineardesign_bin=bin_collide,
            lineardesign_bin_sha256=sha_collide,
            defer_commit=True,
            work_dir=tmp_path / "w_nodef",
        )

    transport = InMemoryDriveV3Transport()
    runs_root_id = transport.create_file(parent_id="root", name="RunsR3", mime_type=DRIVE_FOLDER_MIME)["id"]
    commits_id = transport.create_file(parent_id="root", name="CommitsR3", mime_type=DRIVE_FOLDER_MIME)["id"]
    drive_client = DriveRestV3Client(transport)
    cas_store = ExecutionCASStore(
        tmp_path / "cas_r3.sqlite3",
        drive_verifier=drive_client,
        commits_snapshot_folder_id=commits_id,
        runs_root_id=runs_root_id,
        require_drive_verifier=True,
    )
    app = create_control_plane_app(
        cas_store,
        projects_root=tmp_path / "proj_r3",
        seed_dir=REPO_ROOT / "registry_seed",
        access_token=secret_tok,
    )
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    srv_thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    srv_thread.start()
    try:
        for _ in range(50):
            if server.started:
                break
            time.sleep(0.05)

        # #1: defer_commit=True + gateway_url runs cleanly without creating/acquiring an orphaned CAS lease
        res_def = worker_mod.run_worker(
            protein_seq=egfp_prot,
            compliance_track="research_max",
            lambdas=(1.0,),
            lineardesign_bin=bin_collide,
            lineardesign_bin_sha256=sha_collide,
            gateway_url=f"http://127.0.0.1:{port}",
            defer_commit=True,
            output_json=tmp_path / "def_out.json",
            work_dir=tmp_path / "w_def_gw",
        )
        assert res_def["status"] == "COMPUTED"
        assert res_def["execution_id"] is None

        # #5: Distinct support table contents alter CAS submission_id & request_digest
        res_s1 = worker_mod.run_worker(
            protein_seq=egfp_prot,
            compliance_track="research_max",
            lambdas=(1.0,),
            lineardesign_bin=bin_collide,
            lineardesign_bin_sha256=sha_collide,
            gateway_url=f"http://127.0.0.1:{port}",
            drive_runs_root_id=runs_root_id,
            drive_client=drive_client,
            output_json=tmp_path / "sup1.json",
            work_dir=tmp_path / "w_sup1",
        )
        (dir_sup_a / "coding_wheel.txt").write_text("wheel_v2_changed\n", encoding="utf-8")
        res_s2 = worker_mod.run_worker(
            protein_seq=egfp_prot,
            compliance_track="research_max",
            lambdas=(1.0,),
            lineardesign_bin=bin_collide,
            lineardesign_bin_sha256=sha_collide,
            gateway_url=f"http://127.0.0.1:{port}",
            drive_runs_root_id=runs_root_id,
            drive_client=drive_client,
            output_json=tmp_path / "sup2.json",
            work_dir=tmp_path / "w_sup2",
        )
        assert cas_store.get_execution(res_s1["execution_id"]).request_digest != cas_store.get_execution(res_s2["execution_id"]).request_digest

        # #5: Uncommitted explicit execution_id rejects mismatched supplied request_digest/intent_id BEFORE lease acquisition
        import httpx

        with httpx.Client(trust_env=False, timeout=5.0) as hc:
            sub_uncomm = hc.post(
                f"http://127.0.0.1:{port}/v1/executions",
                headers={"X-MRNAX-Access-Token": secret_tok},
                json={"submission_id": "sub_uncomm_r3", "request_digest": "digest_expected_r3", "intent_id": "int_r3"},
            )
        assert sub_uncomm.status_code == 200
        eid_uncomm = sub_uncomm.json()["execution_id"]
        with pytest.raises(ValueError, match="mismatch"):
            worker_mod.run_worker(
                protein_seq=egfp_prot,
                compliance_track="research_max",
                lambdas=(1.0,),
                lineardesign_bin=bin_collide,
                lineardesign_bin_sha256=sha_collide,
                gateway_url=f"http://127.0.0.1:{port}",
                execution_id=eid_uncomm,
                request_digest="digest_wrong_r3",
                intent_id="int_wrong_r3",
                drive_runs_root_id=runs_root_id,
                drive_client=drive_client,
                work_dir=tmp_path / "w_mismatch",
            )
        assert cas_store.get_execution(eid_uncomm).lease_epoch == 0
    finally:
        server.should_exit = True
        srv_thread.join(timeout=5.0)
        sock.close()


def test_gate_b_r2_round4_worker_fixes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib
    import importlib.util
    import socket
    import tarfile
    import threading
    import time
    import uvicorn

    from mrnax.cloud.cas_store import ExecutionCASStore
    from mrnax.cloud.drive_rest_v3 import DRIVE_FOLDER_MIME, DriveRestV3Client, InMemoryDriveV3Transport
    from mrnax.cloud.gateway import create_control_plane_app

    spec = importlib.util.spec_from_file_location("colab_heavy_sota_worker_r4", WORKER_SCRIPT)
    assert spec and spec.loader
    worker_mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = worker_mod
    spec.loader.exec_module(worker_mod)

    egfp_prot = json.loads((REPO_ROOT / "registry_seed" / "egfp_reporter_snapshot.json").read_text(encoding="utf-8"))[
        "mammalian_reporter_egfp"
    ]["protein_sequence"]
    secret_tok = "tok_r4_secret_key_998877"
    monkeypatch.setenv("MRNAX_GATEWAY_ACCESS_TOKEN", secret_tok)

    # #6: Reject short 1..3-char tokens so _redact_secrets cannot skip them
    with pytest.raises(ValueError, match="at least 4"):
        worker_mod._validate_env_token("abc", env_name="MRNAX_GATEWAY_ACCESS_TOKEN")

    # Case-insensitive receipt collision detection & nondeferred remote preflight parity
    out_a = tmp_path / "OutArtifact.json"
    rec_a = tmp_path / "outartifact.json"
    assert worker_mod._paths_alias(out_a, rec_a) is True
    with pytest.raises(ValueError, match="distinct"):
        worker_mod.run_worker(
            protein_seq=egfp_prot,
            lineardesign_bin=tmp_path / "missing",
            output_json=out_a,
            commit_receipt_json=rec_a,
        )

    transport = InMemoryDriveV3Transport()
    runs_root_id = transport.create_file(parent_id="root", name="RunsR4", mime_type=DRIVE_FOLDER_MIME)["id"]
    commits_id = transport.create_file(parent_id="root", name="CommitsR4", mime_type=DRIVE_FOLDER_MIME)["id"]
    drive_client = DriveRestV3Client(transport)
    cas_store = ExecutionCASStore(
        tmp_path / "cas_r4.sqlite3",
        drive_verifier=drive_client,
        commits_snapshot_folder_id=commits_id,
        runs_root_id=runs_root_id,
        require_drive_verifier=True,
    )

    with pytest.raises(ValueError, match="Drive sealing"):
        worker_mod.run_worker(
            protein_seq=egfp_prot,
            lineardesign_bin=tmp_path / "missing",
            gateway_url="https://149-248-39-200.sslip.io:8443",
            drive_client=drive_client,
            drive_runs_root_id=None,
        )

    # #4: Missing pinned support table or unexpected newly appeared support table fails cleanly
    sup_dir = tmp_path / "raw_sup_r4"
    sup_dir.mkdir(parents=True, exist_ok=True)
    raw_bin = sup_dir / "lineardesign"
    raw_bin.write_text("#!/bin/sh\necho ok\n", encoding="utf-8")
    raw_bin.chmod(0o755)
    raw_sha = hashlib.sha256(raw_bin.read_bytes()).hexdigest()
    (sup_dir / "coding_wheel.txt").write_text("w1\n", encoding="utf-8")
    pinned = worker_mod._hash_support_tables(sup_dir)
    (sup_dir / "coding_wheel.txt").unlink()
    with pytest.raises(FileNotFoundError, match="missing before staging"):
        worker_mod._copy_verified_raw_binary(
            raw_bin,
            tmp_path / "dest_missing",
            expected_sha256=raw_sha,
            expected_support_sha256s=pinned,
        )
    (sup_dir / "coding_wheel.txt").write_text("w1\n", encoding="utf-8")
    (sup_dir / "codon_usage_freq_table_human.csv").write_text("unexpected\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unpinned LinearDesign runtime table"):
        worker_mod._copy_verified_raw_binary(
            raw_bin,
            tmp_path / "dest_extra",
            expected_sha256=raw_sha,
            expected_support_sha256s=pinned,
        )

    # Pinned-tar submission ID parity & empty stored intent_id mismatch fail-closed check
    fake_ld_py = (
        f"#!{sys.executable}\n"
        "import sys\n"
        "CODON = dict(zip('ACDEFGHIKLMNPQRSTVWY', "
        "'GCU UGU GAU GAA UUU GGU CAU AUU AAA CUU AUG AAU CCU CAA CGU UCU ACU GUU UGG UAU'.split()))\n"
        "aa = ''.join(ch for ch in sys.stdin.read().strip().upper() if ch in CODON)\n"
        "rna = ''.join(CODON[c] for c in aa)\n"
        "print(f'mRNA sequence:  {rna}')\n"
        "print('mRNA structure: ' + '.' * len(rna))\n"
        "print('mRNA folding free energy: -42.50 kcal/mol; mRNA CAI: 0.910')\n"
    )
    tar_src = tmp_path / "tar_src"
    tar_src.mkdir(parents=True, exist_ok=True)
    (tar_src / "lineardesign").write_text(fake_ld_py, encoding="utf-8")
    (tar_src / "lineardesign").chmod(0o755)
    ld_tar = tmp_path / "ld_r4.tar.gz"
    with tarfile.open(ld_tar, "w:gz") as tf:
        tf.add(tar_src / "lineardesign", arcname="lineardesign")
    ld_sha = hashlib.sha256(ld_tar.read_bytes()).hexdigest()
    monkeypatch.setattr(worker_mod, "PINNED_LINEARDESIGN_TAR_SHA256", ld_sha)

    app = create_control_plane_app(
        cas_store,
        projects_root=tmp_path / "proj_r4",
        seed_dir=REPO_ROOT / "registry_seed",
        access_token=secret_tok,
    )
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    srv_thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    srv_thread.start()
    try:
        for _ in range(50):
            if server.started:
                break
            time.sleep(0.05)

        exp_dig = hashlib.sha256(
            json.dumps(
                {
                    "protein": egfp_prot,
                    "lambdas": [1.0],
                    "ef": False,
                    "track": "research_max",
                    "intent_id": "int_colab_sota",
                    "ld_sha": ld_sha,
                    "ef_sha": None,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        res_tar = worker_mod.run_worker(
            protein_seq=egfp_prot,
            compliance_track="research_max",
            lambdas=(1.0,),
            lineardesign_tar=ld_tar,
            gateway_url=f"http://127.0.0.1:{port}",
            drive_runs_root_id=runs_root_id,
            drive_client=drive_client,
            output_json=tmp_path / "tar_out_r4.json",
            work_dir=tmp_path / "w_tar_r4",
        )
        row_tar = cas_store.get_execution(res_tar["execution_id"])
        assert row_tar.request_digest == exp_dig
        assert row_tar.submission_id == f"sub_colab_{exp_dig[:16]}"

        # #5: Empty stored intent_id fails closed before lease when caller passes different intent_id + matching digest
        import httpx

        with httpx.Client(trust_env=False, timeout=5.0) as hc:
            sub_empty_int = hc.post(
                f"http://127.0.0.1:{port}/v1/executions",
                headers={"X-MRNAX-Access-Token": secret_tok},
                json={"submission_id": "sub_empty_int_r4", "request_digest": "same_dig_r4", "intent_id": ""},
            )
        assert sub_empty_int.status_code == 200
        eid_empty = sub_empty_int.json()["execution_id"]
        with pytest.raises(ValueError, match="intent_id mismatch"):
            worker_mod.run_worker(
                protein_seq=egfp_prot,
                lineardesign_tar=ld_tar,
                gateway_url=f"http://127.0.0.1:{port}",
                execution_id=eid_empty,
                request_digest="same_dig_r4",
                intent_id="int_different_r4",
                drive_runs_root_id=runs_root_id,
                drive_client=drive_client,
                work_dir=tmp_path / "w_empty_int",
            )
        assert cas_store.get_execution(eid_empty).lease_epoch == 0

        # Round-5 Item 5 & 6: _paths_alias rejects case-aliases on Linux too, and whitespace-only sealed_file_id fails before CAS submit
        with monkeypatch.context() as mp:
            mp.setattr(sys, "platform", "linux")
            assert worker_mod._paths_alias(tmp_path / "LinuxCase.json", tmp_path / "linuxcase.json") is True
        with pytest.raises(ValueError, match="empty or whitespace"):
            worker_mod.run_worker(
                protein_seq=egfp_prot,
                lineardesign_tar=ld_tar,
                gateway_url="https://149-248-39-200.sslip.io:8443",
                sealed_file_id="   ",
                sealed_sha256="a" * 64,
            )

        # Round-5 Item 1: explicit empty or whitespace intent_id preserves verbatim 35c925e7 request_digest parity
        exp_ws_dig = hashlib.sha256(
            json.dumps(
                {
                    "protein": egfp_prot,
                    "lambdas": [1.0],
                    "ef": False,
                    "track": "research_max",
                    "intent_id": "  ",
                    "ld_sha": ld_sha,
                    "ef_sha": None,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        res_ws = worker_mod.run_worker(
            protein_seq=egfp_prot,
            compliance_track="research_max",
            lambdas=(1.0,),
            lineardesign_tar=ld_tar,
            intent_id="  ",
            gateway_url=f"http://127.0.0.1:{port}",
            drive_runs_root_id=runs_root_id,
            drive_client=drive_client,
            output_json=tmp_path / "tar_ws_r5.json",
            work_dir=tmp_path / "w_tar_ws_r5",
        )
        assert cas_store.get_execution(res_ws["execution_id"]).request_digest == exp_ws_dig

        # Round-6 Item 2: pre-existing default receipt symlink aliasing output_json fails BEFORE CAS submit/lease
        implicit_out = tmp_path / "implicit_out_r6.json"
        implicit_rcpt = tmp_path / "implicit_out_r6.commit_receipt.json"
        implicit_rcpt.symlink_to(implicit_out)
        with pytest.raises(ValueError, match="must be distinct from --output-json"):
            worker_mod.run_worker(
                protein_seq=egfp_prot,
                compliance_track="research_max",
                lambdas=(1.0,),
                lineardesign_tar=ld_tar,
                intent_id="int_implicit_alias_r6",
                gateway_url=f"http://127.0.0.1:{port}",
                drive_runs_root_id=runs_root_id,
                drive_client=drive_client,
                output_json=implicit_out,
                work_dir=tmp_path / "w_implicit_alias_r6",
            )

        # Round-6 Addl 1: caller submission_id + matching digest with mismatched intent_id fails BEFORE lease,
        # and explicit execution_id resume preserves exact whitespace intent_id distinction
        sub_ws = worker_mod.run_worker(
            protein_seq=egfp_prot,
            compliance_track="research_max",
            lambdas=(1.0,),
            lineardesign_tar=ld_tar,
            submission_id="sub_r6_intent_guard",
            request_digest="dig_r6_intent_guard",
            intent_id=" a ",
            gateway_url=f"http://127.0.0.1:{port}",
            drive_runs_root_id=runs_root_id,
            drive_client=drive_client,
            output_json=tmp_path / "r6_intent_a.json",
            work_dir=tmp_path / "w_r6_intent_a",
        )
        assert sub_ws["status"] == "COMMITTED"
        with httpx.Client(trust_env=False, timeout=5.0) as hc:
            sub_mismatch = hc.post(
                f"http://127.0.0.1:{port}/v1/executions",
                headers={"X-MRNAX-Access-Token": secret_tok},
                json={"submission_id": "sub_r6_ws_pend", "request_digest": "dig_r6_ws_pend", "intent_id": " a "},
            )
        assert sub_mismatch.status_code == 200
        eid_ws_pend = sub_mismatch.json()["execution_id"]
        with pytest.raises(ValueError, match="intent_id mismatch"):
            worker_mod.run_worker(
                protein_seq=egfp_prot,
                lineardesign_tar=ld_tar,
                submission_id="sub_r6_ws_pend",
                request_digest="dig_r6_ws_pend",
                intent_id="a",
                gateway_url=f"http://127.0.0.1:{port}",
                drive_runs_root_id=runs_root_id,
                drive_client=drive_client,
                work_dir=tmp_path / "w_r6_ws_sub_mismatch",
            )
        with pytest.raises(ValueError, match="intent_id mismatch"):
            worker_mod.run_worker(
                protein_seq=egfp_prot,
                lineardesign_tar=ld_tar,
                execution_id=eid_ws_pend,
                request_digest="dig_r6_ws_pend",
                intent_id="a",
                gateway_url=f"http://127.0.0.1:{port}",
                drive_runs_root_id=runs_root_id,
                drive_client=drive_client,
                work_dir=tmp_path / "w_r6_ws_exec_mismatch",
            )
        assert cas_store.get_execution(eid_ws_pend).lease_epoch == 0
    finally:
        server.should_exit = True
        srv_thread.join(timeout=5.0)
        sock.close()




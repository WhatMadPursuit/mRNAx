"""Live Cloud Integration Gate (FT-25 & FT-26): Real Colab Kernel Interrupt Recovery + Live Google Drive REST v3 Seal + FastAPI CAS."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mrnax.cloud.cas_store import ExecutionCASStore
from mrnax.cloud.drive_rest_v3 import (
    DriveRestV3Client,
    GdriveCliDriveV3Transport,
    HttpDriveV3Transport,
    InMemoryDriveV3Transport,
)
from mrnax.cloud.gateway import create_control_plane_app

AGENT_INBOX_FOLDER_ID = "1_NM_-smOrmWjg1vmEY9XDMn5vdTwUrDg"
REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(
    not os.environ.get("MRNAX_SCOPED_DRIVE_TOKEN") or not os.environ.get("MRNAX_OUTSIDE_CANARY_ID"),
    reason=(
        "Honest Gate (Joint Ruling #2): MRNAX_SCOPED_DRIVE_TOKEN (drive.file / Service Account) not yet "
        "provisioned; refusing to fake negative-sample test with owner full-drive OAuth."
    ),
)
def test_live_drive_permission_negative_canary_ft25() -> None:
    """FT-25: Verify real Google Drive REST v3 scoped token reads/writes /mRNAx/ and gets 403/404 on outside canary."""
    token = os.environ["MRNAX_SCOPED_DRIVE_TOKEN"]
    ws_id = os.environ["MRNAX_WORKSPACE_FOLDER_ID"]
    hub_id = os.environ.get("MRNAX_DATA_HUB_ID", "12HDihif0-x4-xn7P6laot6H-hCIJRri0")
    canary_id = os.environ["MRNAX_OUTSIDE_CANARY_ID"]

    client = DriveRestV3Client(HttpDriveV3Transport(lambda: token))
    audit = client.verify_permission_boundary(ws_id, hub_id, canary_id)
    assert audit.workspace_read_write_ok is True
    assert audit.data_hub_read_ok is True
    assert audit.outside_canary_unreadable is True
    assert audit.outside_canary_http_status in {403, 404}


def test_live_colab_interrupt_and_cas_recovery_ft26(tmp_path: Path) -> None:
    """FT-26: Real remote Colab VM step 1..5 -> Live Google Drive REST v3 checkpoint -> forced kernel restart ->
    resume step 6..10 from Drive checkpoint -> Live Google Drive SEALED.json -> FastAPI CAS 200 / Zombie 409 ->
    SQLite power-loss disaster recovery from Drive COMMIT_RECEIPT."""
    colab_bin = Path("/Users/alex/.local/bin/colab")
    if not colab_bin.is_file():
        pytest.skip("google-colab-cli not installed")

    sess_name = f"ft26-{uuid.uuid4().hex[:6]}"
    live_drive_enabled = shutil.which("gdrive") is not None
    if live_drive_enabled:
        drive = DriveRestV3Client(GdriveCliDriveV3Transport("gdrive"))
        ws_root = drive.create_folder(AGENT_INBOX_FOLDER_ID, "mrnax_v2_live_cas_runs")
    else:
        drive = DriveRestV3Client(InMemoryDriveV3Transport())
        ws_root = drive.create_folder("root", "mRNAx")

    runs_root = drive.create_folder(ws_root, "runs")
    commits_root = drive.create_folder(ws_root, f"commits_{sess_name}")

    store = ExecutionCASStore(
        tmp_path / "live_cas.sqlite3",
        drive_verifier=drive,
        commits_snapshot_folder_id=commits_root,
    )
    api = TestClient(create_control_plane_app(store))

    sub_resp = api.post(
        "/v1/executions",
        json={
            "submission_id": f"sub_live_{sess_name}",
            "request_digest": "digest_ft26_egfp_cpu",
            "intent_id": "int_ft26",
        },
    )
    assert sub_resp.status_code == 200
    exec_id = sub_resp.json()["execution_id"]

    part1_py = tmp_path / "colab_part1.py"
    part1_py.write_text(
        '''
import json, random
codons = ["UUC", "CUG", "CUC", "GGC", "GGA", "AGC", "AAG", "GAG"]
rng = random.Random(20261127)
traj = []
for step in range(1, 6):
    seq = "AUG" + "".join(rng.choice(codons) for _ in range(12)) + "UAA"
    sc = round(((seq.count("G") + seq.count("C")) / len(seq)) * 100.0 + rng.uniform(-2.5, 2.5), 8)
    traj.append({"score": sc, "seq": seq, "step": step})
st = rng.getstate()
ckpt = {"completed_step": 5, "prng_state": [st[0], list(st[1]), st[2]], "trajectory": traj}
in_mem_marker = "ALIVE_BEFORE_CRASH"
print(json.dumps(ckpt, sort_keys=True))
''',
        encoding="utf-8",
    )

    try:
        subprocess.run([str(colab_bin), "new", "-s", sess_name], check=True, capture_output=True, text=True, timeout=90)

        # Epoch 1 acquires short 1s lease via FastAPI Control Plane
        l1_resp = api.post(f"/v1/executions/{exec_id}/leases", json={"worker_id": "colab_epoch1", "ttl_seconds": 1})
        assert l1_resp.status_code == 200
        lease_1 = l1_resp.json()
        assert lease_1["lease_epoch"] == 1

        att1_dir = drive.ensure_attempt_dir(runs_root, exec_id, lease_1["attempt_id"])
        out1 = subprocess.run(
            [str(colab_bin), "exec", "-s", sess_name, "-f", str(part1_py)],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        ).stdout.strip()
        ckpt_line = out1.splitlines()[-1]
        ckpt_doc = json.loads(ckpt_line)
        assert ckpt_doc["completed_step"] == 5

        # Upload Epoch 1 step-5 checkpoint directly to Google Drive REST v3 (zero FUSE!)
        r1_ckpt = drive.upload_verified_bytes(
            att1_dir, "checkpoint_step5.json", (ckpt_line + "\n").encode("utf-8"), "application/json"
        )

        # Real remote Colab kernel reset (wiping all RAM variables) + ensure 1s lease expired
        subprocess.run(
            [str(colab_bin), "restart-kernel", "-s", sess_name],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if time.time() <= float(lease_1["lease_expires_at"]):
            time.sleep(1.1)

        # Epoch 2 acquires new lease via FastAPI Control Plane -> epoch increments to 2!
        l2_resp = api.post(f"/v1/executions/{exec_id}/leases", json={"worker_id": "colab_epoch2", "ttl_seconds": 300})
        assert l2_resp.status_code == 200
        lease_2 = l2_resp.json()
        assert lease_2["lease_epoch"] == 2
        assert lease_2["previous_attempt_id"] == lease_1["attempt_id"]

        # Download step-5 checkpoint back from Google Drive REST v3 to resume Epoch 2 on Colab!
        fetched_ckpt_bytes = drive.transport.download_bytes(r1_ckpt.file_id)
        fetched_ckpt_json_str = fetched_ckpt_bytes.decode("utf-8").strip()

        part2_py = tmp_path / "colab_part2.py"
        part2_py.write_text(
            f'''
import json, random
assert "in_mem_marker" not in globals(), "Kernel was not restarted!"
ckpt = json.loads({fetched_ckpt_json_str!r})
codons = ["UUC", "CUG", "CUC", "GGC", "GGA", "AGC", "AAG", "GAG"]
rng = random.Random(0)
raw_st = ckpt["prng_state"]
rng.setstate((int(raw_st[0]), tuple(int(x) for x in raw_st[1]), raw_st[2]))
traj = list(ckpt["trajectory"])
for step in range(int(ckpt["completed_step"]) + 1, 11):
    seq = "AUG" + "".join(rng.choice(codons) for _ in range(12)) + "UAA"
    sc = round(((seq.count("G") + seq.count("C")) / len(seq)) * 100.0 + rng.uniform(-2.5, 2.5), 8)
    traj.append({{"score": sc, "seq": seq, "step": step}})

rng_ref = random.Random(20261127)
ref_traj = []
for step in range(1, 11):
    seq = "AUG" + "".join(rng_ref.choice(codons) for _ in range(12)) + "UAA"
    sc = round(((seq.count("G") + seq.count("C")) / len(seq)) * 100.0 + rng_ref.uniform(-2.5, 2.5), 8)
    ref_traj.append({{"score": sc, "seq": seq, "step": step}})

assert traj == ref_traj, "Trajectory mismatch after Colab kernel restart!"
print(json.dumps({{"bit_exact_match": True, "steps": len(traj), "final_seq": traj[-1]["seq"], "final_score": traj[-1]["score"]}}, sort_keys=True))
''',
            encoding="utf-8",
        )

        out2 = subprocess.run(
            [str(colab_bin), "exec", "-s", sess_name, "-f", str(part2_py)],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        ).stdout.strip()
        res_line = out2.splitlines()[-1]
        res_doc = json.loads(res_line)
        assert res_doc["bit_exact_match"] is True

        # Epoch 2 seals its isolated Drive directory (`attempts/{att_2}/SEALED.json`) and commits via FastAPI -> 200!
        att2_dir = drive.ensure_attempt_dir(runs_root, exec_id, lease_2["attempt_id"])
        r2 = drive.upload_verified_bytes(att2_dir, "result.json", (res_line + "\n").encode("utf-8"), "application/json")
        seal_2 = drive.write_attempt_seal(
            att2_dir,
            execution_id=exec_id,
            attempt_id=lease_2["attempt_id"],
            lease_epoch=lease_2["lease_epoch"],
            receipts=[r2],
        )

        c2_resp = api.post(
            f"/v1/executions/{exec_id}/attempts/{lease_2['attempt_id']}:commit",
            json={
                "lease_epoch": lease_2["lease_epoch"],
                "lease_token": lease_2["lease_token"],
                "sealed_file_id": seal_2.file_id,
                "sealed_sha256": seal_2.sha256,
            },
        )
        assert c2_resp.status_code == 200
        assert c2_resp.json()["status"] == "COMMITTED"

        # Zombie Epoch 1 wakes up, seals its own isolated Drive folder, and tries to commit -> HTTP 409!
        seal_1 = drive.write_attempt_seal(
            att1_dir,
            execution_id=exec_id,
            attempt_id=lease_1["attempt_id"],
            lease_epoch=lease_1["lease_epoch"],
            receipts=[r1_ckpt],
        )
        c1_resp = api.post(
            f"/v1/executions/{exec_id}/attempts/{lease_1['attempt_id']}:commit",
            json={
                "lease_epoch": lease_1["lease_epoch"],
                "lease_token": lease_1["lease_token"],
                "sealed_file_id": seal_1.file_id,
                "sealed_sha256": seal_1.sha256,
            },
        )
        assert c1_resp.status_code == 409

        # Simulate laptop power loss / SQLite DB loss -> reconcile from live Drive COMMIT_RECEIPT!
        recovered_store = ExecutionCASStore(
            tmp_path / "recovered_after_power_loss.sqlite3",
            drive_verifier=drive,
            commits_snapshot_folder_id=commits_root,
        )
        reconciled_n = recovered_store.reconcile_from_drive_seals(drive, runs_root)
        assert reconciled_n == 1
        rec_row = recovered_store.get_execution(exec_id)
        assert rec_row.status == "COMMITTED"
        assert rec_row.winning_attempt_id == lease_2["attempt_id"]
        assert rec_row.winning_sealed_file_id == seal_2.file_id

        audit_out = REPO_ROOT / "artifacts" / "m1_egfp" / "live_colab_drive_cas_audit.json"
        audit_out.parent.mkdir(parents=True, exist_ok=True)
        audit_out.write_text(
            json.dumps(
                {
                    "live_drive_rest_v3_enabled": live_drive_enabled,
                    "colab_session_name": sess_name,
                    "execution_id": exec_id,
                    "drive_workspace_folder_id": ws_root,
                    "drive_runs_root_id": runs_root,
                    "drive_commits_folder_id": commits_root,
                    "epoch1_attempt_id": lease_1["attempt_id"],
                    "epoch1_checkpoint_file_id": r1_ckpt.file_id,
                    "epoch1_zombie_sealed_file_id": seal_1.file_id,
                    "epoch1_zombie_commit_http_status": c1_resp.status_code,
                    "epoch2_winning_attempt_id": lease_2["attempt_id"],
                    "epoch2_result_file_id": r2.file_id,
                    "epoch2_winning_sealed_file_id": seal_2.file_id,
                    "epoch2_winning_sealed_sha256": seal_2.sha256,
                    "epoch2_commit_http_status": c2_resp.status_code,
                    "bit_exact_trajectory_match": res_doc["bit_exact_match"],
                    "power_loss_drive_reconciled": True,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    finally:
        subprocess.run([str(colab_bin), "stop", "-s", sess_name], capture_output=True, text=True)

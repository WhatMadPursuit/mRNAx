"""TDD contract tests for the full-loop Web UI Harness API (`src/mrnax/cloud/gateway.py`)."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from mrnax.cloud.cas_store import ExecutionCASStore
from mrnax.cloud.gateway import create_control_plane_app

REPO_ROOT = Path(__file__).resolve().parents[2]
SEED_DIR = REPO_ROOT / "registry_seed"


def test_web_harness_full_iterative_loop_and_exports(tmp_path: Path) -> None:
    store = ExecutionCASStore(tmp_path / "cp.sqlite3")
    app = create_control_plane_app(
        store,
        projects_root=tmp_path / "projects",
        seed_dir=SEED_DIR,
        access_token="tok_secret_unit_test",
    )
    client = TestClient(
        app,
        headers={"Origin": "https://whatmadpursuit.github.io"},
    )

    # 1. System status + CORS check
    st = client.get("/v1/system/status")
    assert st.status_code == 200
    assert st.headers.get("access-control-allow-origin") == "https://whatmadpursuit.github.io"
    st_json = st.json()
    assert st_json["default_profile"] == "research_max"
    assert st_json["viennarna_available"] is True

    # 2. Presets check
    presets = client.get("/v1/presets").json()
    egfp_seq = presets["egfp_reporter"]["protein_sequence"]
    assert len(egfp_seq) == 239
    assert egfp_seq.startswith("MVSKG")

    # 3. Negative input test: missing initiator M -> blocked before optimization!
    bad_turn = client.post(
        "/v1/projects/proj_neg/turn",
        json={
            "mode": "initial",
            "user_utterance": "Design truncated GFP without M",
            "protein_sequence": egfp_seq[1:],
        },
    )
    assert bad_turn.status_code == 200
    bad_body = bad_turn.json()
    assert bad_body["status"] == "NEEDS_CLARIFICATION"
    assert bad_body["intent"]["status"] == "needs_protein_confirmation"
    assert len(bad_body["blocking_questions"]) >= 1

    # 4. Negative input test: objective conflict -> blocked before optimization!
    conflict_turn = client.post(
        "/v1/projects/proj_conflict/turn",
        json={
            "mode": "initial",
            "user_utterance": "Give me a single construct with maximal GC stems AND completely unfolded CDS",
            "protein_sequence": egfp_seq,
            "conflicting_single_pick": True,
        },
    )
    assert conflict_turn.status_code == 200
    assert conflict_turn.json()["status"] == "NEEDS_CLARIFICATION"
    assert conflict_turn.json()["intent"]["status"] == "needs_conflict_clarification"

    # 5. Turn 1: Real eGFP Protein-Only design (BsaI + m1psi + liver detargeting)
    t1 = client.post(
        "/v1/projects/proj_web_loop/turn",
        json={
            "mode": "initial",
            "user_utterance": "设计 eGFP 报告蛋白 mRNA，m1Psi 修饰，肝脏去靶向，避开 BsaI",
            "protein_sequence": egfp_seq,
            "accession_or_source": "GenBank:U55762.1",
            "uridine_chemistry": "m1psi",
            "restriction_enzymes": ["BsaI"],
            "detarget_liver": True,
            "compliance_track": "research_max",
        },
    )
    assert t1.status_code == 200
    b1 = t1.json()
    assert b1["status"] == "COMPLETED"
    assert b1["intent"]["intent_version"] == 1
    assert len(b1["candidates"]) >= 2

    # 6. Turn 2: Single-item iterative refinement (BsaI -> BspQI) inheriting all locks!
    t2 = client.post(
        "/v1/projects/proj_web_loop/turn",
        json={
            "mode": "refine",
            "user_utterance": "只把内切酶从 BsaI 改成避开 BspQI，其余锁定项全部继承",
            "restriction_enzymes": ["BspQI"],
        },
    )
    assert t2.status_code == 200
    b2 = t2.json()
    assert b2["status"] == "COMPLETED"
    assert b2["intent"]["intent_version"] == 2
    assert b2["intent"]["parent_intent_id"] == b1["intent"]["intent_id"]
    assert b2["intent_diff"]["protein_lock_preserved"] is True
    assert b2["intent_diff"]["v1_enzymes"] == ["BsaI"]
    assert b2["intent_diff"]["v2_enzymes"] == ["BspQI"]

    # 7. Export 4-file INSDC bundle for top recommended candidate
    top_id = b2["report"]["recommended_bundle_ids"][0]
    exp = client.get(f"/v1/projects/proj_web_loop/bundles/{top_id}/export")
    assert exp.status_code == 200
    files = exp.json()["files"]
    assert set(files.keys()) >= {
        "full_rna.fasta",
        "cds_only.fasta",
        "construct.gb",
        "chemistry_metadata.json",
    }
    assert "LOCUS" in files["construct.gb"]

    # 8. Entry 2: Protein + Reference Construct R
    t_ref = client.post(
        "/v1/projects/proj_with_r/turn",
        json={
            "mode": "initial",
            "user_utterance": "基于参考构建 R 生成保守最小改动方案和探索方案，我更看重后期信号",
            "protein_sequence": egfp_seq,
            "use_preset_reference_r": True,
            "prioritize_late_signal": True,
        },
    )
    assert t_ref.status_code == 200
    b_ref = t_ref.json()
    assert b_ref["status"] == "COMPLETED"
    assert b_ref["intent"]["entry_mode"] == "protein_with_reference_r"
    derivations = {c["derivation_action"] for c in b_ref["candidates"]}
    assert "locked_refinement" in derivations
    assert any(c["bundle_id"].endswith("_minA") for c in b_ref["candidates"])

    # 9. Privileged LLM chat endpoint rejects unauthenticated public callers (401)
    unauth = client.post(
        "/v1/projects/proj_web_loop/agent-chat",
        json={"message": "测试未授权调用", "provider": "deepseek"},
    )
    assert unauth.status_code == 401

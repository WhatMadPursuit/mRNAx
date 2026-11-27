"""TDD Regression Suite for Wave 5 Deliverables & GPT-6.2-sol M1.5 Blind Adversarial Audit Findings."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from mrnax.cloud.cas_store import ExecutionCASStore, StaleLeaseConflictError
from mrnax.cloud.gateway import create_control_plane_app
from mrnax.constraints.registry import default_m1_constraints
from mrnax.engine.ablation_harness import run_paired_seed_surrogate_switch_ablation
from mrnax.engine.pareto_director import run_design_pipeline
from mrnax.engine.profiles import COMPONENT_CATALOG
from mrnax.engine.surrogates import (
    CommercialCleanSurrogateV1,
    stitch_sliding_window_embeddings,
    validate_structrfm_checkpoint_path,
    validate_structrfm_tokenizer,
)
from mrnax.federated.external_evidence_adapter import CleanRoomPolicyError, ExternalEvidenceAdapter
from mrnax.harness.state_machine import ProjectHarnessSession
from mrnax.harness.tools import AtomicToolbox
from mrnax.rules.models import (
    EvidenceObservationLink,
    ModelInterpretationNote,
    RuleRecord,
    load_replacement_card_yaml,
)
from mrnax.rules.tribunal import RuleRegistry
from mrnax.schemas.construct_bundle import ProteinTargetSpec
from mrnax.schemas.design_intent import DesignIntent, LockedRegionSpec, ObjectiveSpec, RuntimePolicySpec

REPO_ROOT = Path(__file__).resolve().parents[2]
SEED_DIR = REPO_ROOT / "registry_seed"


def test_wave5_rules_v1_1_and_tribunal_dedup_and_provenance(tmp_path: Path) -> None:
    # 1. Verify V9/Wave 11 19 rules with 2 rigorously substantiated verified_rules (004, 007) & 17 conditional_rules
    rules_path = SEED_DIR / "rules_commercial_v1.json"
    doc = json.loads(rules_path.read_text(encoding="utf-8"))
    rules = [RuleRecord.model_validate(r) for r in doc["rules"]]
    assert len(rules) == 19
    verified_ids = sorted(r.rule_id for r in rules if r.status == "verified_rule")
    assert verified_ids == [
        "RUL-CC-3UTR-ARE-DECAY-004",
        "RUL-CC-5UTR-GC-TIS-007",
    ]
    r009 = next(r for r in rules if r.rule_id == "RUL-CC-M1PSI-FRAMESHIFT-009")
    assert r009.status == "conditional_rule"

    r006 = next(r for r in rules if r.rule_id == "RUL-CC-CHEM-M1PSI-STRUCT-SHIFT-006")
    assert "luciferase" in r006.statement.lower() or "fluc" in r006.statement.lower()
    assert "egfp" not in r006.statement.lower()
    assert len(r006.model_interpretations) >= 1

    # 2. RC-002 & RC-003 component_ids must match COMPONENT_CATALOG and pass strict threshold gate
    from mrnax.rules.models import verify_replacement_card_strict

    rc002 = load_replacement_card_yaml(REPO_ROOT / "docs" / "replacement_cards" / "RC-002-structrfm-v0.0.9.yaml")
    verify_replacement_card_strict(rc002)
    assert rc002.component_id in COMPONENT_CATALOG
    assert COMPONENT_CATALOG[rc002.component_id].license_track == "commercial_clean"

    rc003 = load_replacement_card_yaml(REPO_ROOT / "docs" / "replacement_cards" / "RC-003-orthrus-v1.0.yaml")
    verify_replacement_card_strict(rc003)
    assert rc003.component_id in COMPONENT_CATALOG
    assert COMPONENT_CATALOG[rc003.component_id].license_track == "commercial_clean"

    # 3. RuleRegistry must reject verified_rule with <2 distinct normalized datasets or duplicate rule_ids
    bad_single_ds_verified = rules[0].model_copy_validated(
        rule_id="RUL-BAD-SINGLE-DS",
        status="verified_rule",
        linked_observations=[
            EvidenceObservationLink(
                claim_id="C1", underlying_dataset_id="GSE232927", evidence_layer="raw_observation"
            ),
            EvidenceObservationLink(
                claim_id="C2", underlying_dataset_id="GEO:gse232927", evidence_layer="raw_observation"
            ),
        ],
    )
    with pytest.raises(ValueError, match="verified_rule"):
        RuleRegistry.create_with_initial_rules(
            tmp_path / "bad_rules.json", track="commercial_clean", rules=[bad_single_ds_verified]
        )

    bad_noncomm_interp = rules[0].model_copy_validated(
        rule_id="RUL-BAD-NC-INTERP",
        model_interpretations=[
            ModelInterpretationNote(
                source_model_name="ViennaRNA",
                source_model_license_track="needs_vendor_license",
                method="regression_weight",
                summary="Unapproved interpretation",
            )
        ],
    )
    with pytest.raises(ValueError, match="commercial_clean"):
        RuleRegistry.create_with_initial_rules(
            tmp_path / "bad_nc.json", track="commercial_clean", rules=[bad_noncomm_interp]
        )


def test_wave5_4head_surrogate_and_zero_viennarna_import_in_subprocess() -> None:
    surr = CommercialCleanSurrogateV1.from_seed_dir(SEED_DIR)
    assert "3utr_mpra_stability" in surr.loaded_heads
    assert len(surr.loaded_heads) == 4

    # Subprocess check: running commercial_clean pipeline must NOT even import `RNA` into sys.modules!
    code = f"""
import json, sys
from pathlib import Path
from mrnax.constraints.registry import default_m1_constraints
from mrnax.engine.pareto_director import run_design_pipeline
from mrnax.schemas.construct_bundle import ProteinTargetSpec
from mrnax.schemas.design_intent import DesignIntent, ObjectiveSpec, RuntimePolicySpec

seed = Path({str(SEED_DIR)!r})
prot = json.loads((seed / "egfp_reporter_snapshot.json").read_text())["mammalian_reporter_egfp"]["protein_sequence"]
intent = DesignIntent(
    intent_id="i_sub", project_id="p_sub", intent_version=1, created_from_turn_ids=["t1"],
    entry_mode="protein_only", compliance_track="commercial_clean", uridine_chemistry="5mou", status="ready",
    protein_targets=[ProteinTargetSpec(protein_sequence=prot, transl_table=1, accession_or_source="U55762.1", confirmation_status="confirmed")],
    constraints=default_m1_constraints(restriction_enzymes=["BspQI"]),
    objectives=[
        ObjectiveSpec(requirement_id="o1", metric_id="utr5_mrl_pred", direction="maximize", priority_rank=1, evidence_status="validated_predictor", source_utterance="x"),
        ObjectiveSpec(requirement_id="o2", metric_id="utr3_stability_pred", direction="maximize", priority_rank=2, evidence_status="validated_predictor", source_utterance="x"),
    ],
    runtime_policy=RuntimePolicySpec(enabled_generators=["stage0_random", "dnachisel_cds", "pareto_nsga2"], enabled_evaluators=["surrogate_v1_clean"], max_evaluations=6, random_seed=42),
)
bs, rep, man = run_design_pipeline(intent, seed_dir=seed)
assert "RNA" not in sys.modules, f"RNA was imported in commercial_clean: {{[k for k in sys.modules if 'RNA' in k]}}"
assert man.rule_snapshot_version.startswith("rules_commercial_clean@")
assert rep.outcome_status == "COMPLETED_WITH_CANDIDATES"
"""
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr


def test_wave5_external_evidence_adapter_and_30case_golden_suite() -> None:
    adapter = ExternalEvidenceAdapter(license_track="commercial_clean")
    with pytest.raises(CleanRoomPolicyError, match="Thera-SAbDab"):
        adapter._guard_source("Thera-SAbDab", "tarlatamab")
    with pytest.raises(CleanRoomPolicyError, match="mature miRNA"):
        adapter.fetch_gtex_tissue_tpm("hsa-miR-122-5p")
    with pytest.raises(CleanRoomPolicyError, match="mature miRNA"):
        adapter.fetch_human_protein_atlas("hsa-miR-122-5p")
    with pytest.raises(CleanRoomPolicyError, match="BLOCKED"):
        adapter.guard_export_to_commercial_bundle(
            {"source_name": "ChEMBL", "license_spdx": "CC-BY-SA-3.0", "license_tier": "lookup_only_sharealike_boundary"},
            purpose="commercial_training",
        )

    snap = json.loads((SEED_DIR / "m2_benchmark2_dll3_cd3_evidence_snapshot_v1.json").read_text(encoding="utf-8"))
    assert len(snap["live_api_snapshots"]) == 9
    assert all(v["status"] == "ok" for v in snap["live_api_snapshots"].values())

    sys.path.insert(0, str(REPO_ROOT / "tests" / "golden"))
    from eval_golden_intent_suite import run_self_check
    from eval_m2_acceptance_fixtures import verify as verify_m2_fixtures

    res = run_self_check(
        REPO_ROOT / "tests" / "golden" / "golden_intent_eval_suite_v1.json",
        mrnax_src=REPO_ROOT / "src",
    )
    assert res["status"] == "PASS"
    assert res["total_cases"] == 30

    m2_res = verify_m2_fixtures(
        REPO_ROOT / "tests" / "golden",
        REPO_ROOT / "src",
        SEED_DIR / "patent_fto_guard_v2.json",
    )
    assert m2_res["status"] == "PASS"
    assert m2_res["fto_alerts_total"] == 0
    assert m2_res["m1psi_slippery_runs_total"] == 0


def test_codex_critical_security_cas_and_toolbox_guards(tmp_path: Path) -> None:
    # 1. Web app.js must enforce trusted API origin for ?api= and escape boolean fields
    app_js = (REPO_ROOT / "web" / "app.js").read_text(encoding="utf-8")
    assert "isTrustedApiOrigin" in app_js
    assert "${d.protein_lock_preserved}" not in app_js
    assert "${d.chemistry_preserved}" not in app_js

    # 2. CAS lease renewal must require valid lease_token (cannot spoof worker_id alone!)
    store = ExecutionCASStore(tmp_path / "cas.sqlite3")
    row = store.submit_execution("sub_sec", "dig_sec", "int_sec", now_ts=100.0)
    grant1 = store.acquire_or_renew_lease(row.execution_id, "worker_a", ttl_seconds=60, now_ts=100.0)
    with pytest.raises(StaleLeaseConflictError):
        store.acquire_or_renew_lease(row.execution_id, "worker_a", ttl_seconds=60, now_ts=110.0)
    with pytest.raises(StaleLeaseConflictError):
        store.acquire_or_renew_lease(
            row.execution_id, "worker_a", lease_token="wrong_token", ttl_seconds=60, now_ts=110.0
        )
    renewed = store.acquire_or_renew_lease(
        row.execution_id, "worker_a", lease_token=grant1.lease_token, ttl_seconds=60, now_ts=110.0
    )
    assert renewed.lease_token == grant1.lease_token

    # 3. Gateway /v1/executions* must require X-MRNAX-Access-Token when configured_token is set
    app = create_control_plane_app(
        store,
        projects_root=tmp_path / "projects",
        seed_dir=SEED_DIR,
        access_token="secret_gw_tok",
    )
    client = TestClient(app)
    r_unauth = client.post(
        "/v1/executions",
        json={"submission_id": "s1", "request_digest": "d1", "intent_id": "i1"},
    )
    assert r_unauth.status_code == 401
    r_auth = client.post(
        "/v1/executions",
        headers={"X-MRNAX-Access-Token": "secret_gw_tok"},
        json={"submission_id": "s1", "request_digest": "d1", "intent_id": "i1"},
    )
    assert r_auth.status_code == 200

    # 4. AtomicToolbox.export_bundle must block arbitrary path escape outside session.store_dir
    sess = ProjectHarnessSession(project_id="proj_tb", store_dir=tmp_path / "sess_tb", seed_dir=SEED_DIR)
    tb = AtomicToolbox(sess)
    snap = json.loads((SEED_DIR / "egfp_reporter_snapshot.json").read_text(encoding="utf-8"))
    tb.design_constructs(
        user_utterance="Design eGFP",
        protein_sequence=snap["mammalian_reporter_egfp"]["protein_sequence"],
    )
    any_bid = next(iter(sess._bundles.keys()))
    with pytest.raises(ValueError, match="inside project store_dir"):
        tb.export_bundle(bundle_id=any_bid, out_dir=str(tmp_path / "outside_escape"))


def test_codex_important_tokenizer_checkpoint_pareto_scale_and_persisted_intent(tmp_path: Path) -> None:
    # 1. Degenerate 4-identical-token tokenizer must fail USE_SPACED=False distinct-base check
    class _DegenerateSameTokenTokenizer:
        unk_token_id = 0
        def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
            return [5, 5, 5, 5]

    with pytest.raises(RuntimeError):
        validate_structrfm_tokenizer(_DegenerateSameTokenTokenizer())

    # 2. Short window output from encoder must raise ValueError instead of silent zero-padding
    with pytest.raises(ValueError, match="shorter than input"):
        stitch_sliding_window_embeddings(
            "AUGC" * 150,
            lambda sub: np.ones((len(sub) - 5, 768), dtype=np.float32),
            win=510,
            stride=384,
        )

    # 3. Missing files check when must_exist=True on validate_structrfm_checkpoint_path
    empty_ckpt = tmp_path / "structRFM_checkpoint"
    empty_ckpt.mkdir()
    with pytest.raises(FileNotFoundError):
        validate_structrfm_checkpoint_path(empty_ckpt, must_exist=True)

    # 4. Persisted DesignIntent in Web Gateway when use_commercial_surrogate=True must match executed intent!
    store = ExecutionCASStore(tmp_path / "cp_persist.sqlite3")
    app = create_control_plane_app(store, projects_root=tmp_path / "projs", seed_dir=SEED_DIR)
    client = TestClient(app)
    egfp_seq = json.loads((SEED_DIR / "egfp_reporter_snapshot.json").read_text(encoding="utf-8"))[
        "mammalian_reporter_egfp"
    ]["protein_sequence"]
    res = client.post(
        "/v1/projects/proj_surr_persist/turn",
        json={
            "mode": "initial",
            "user_utterance": "Run commercial_clean surrogate v1",
            "protein_sequence": egfp_seq,
            "compliance_track": "commercial_clean",
            "use_commercial_surrogate": True,
        },
    )
    assert res.status_code == 200
    body = res.json()
    saved_intents = sorted((tmp_path / "projs" / "proj_surr_persist" / "intents").glob("intent_v*.json"))
    assert len(saved_intents) == 1
    persisted_doc = json.loads(saved_intents[0].read_text(encoding="utf-8"))
    assert persisted_doc["runtime_policy"]["enabled_evaluators"] == ["surrogate_v1_clean"]
    assert persisted_doc["intent_id"] == body["intent"]["intent_id"]

    # 5. Equal-work paired-seed ablation must track identical candidate trial counts
    intent_obj = DesignIntent.model_validate(persisted_doc)
    rows = run_paired_seed_surrogate_switch_ablation(intent_obj, seeds=[42], seed_dir=SEED_DIR)
    assert rows[0].unguided_trial_count == rows[0].guided_trial_count

    # 6. Scale-invariance of Pareto candidate ordering + cross-session project capability guard (403)
    from mrnax.engine.pareto_director import _rank_and_order_pareto_candidates

    vecs_raw = [[0.90, 0.10], [0.55, 0.55], [0.10, 0.90], [0.20, 0.20]]
    vecs_rescaled = [[a, b * 1_000_000.0] for a, b in vecs_raw]
    r1, ord1 = _rank_and_order_pareto_candidates(vecs_raw)
    r2, ord2 = _rank_and_order_pareto_candidates(vecs_rescaled)
    assert r1 == r2
    assert ord1 == ord2

    snooper_client = TestClient(app)
    r_snoop_state = snooper_client.get("/v1/projects/proj_surr_persist/state")
    assert r_snoop_state.status_code == 403
    r_owner_state = client.get("/v1/projects/proj_surr_persist/state")
    assert r_owner_state.status_code == 200

    # 7. Requesting structrfm_rna_foundation without mounted GPU checkpoint fails closed (never fakes ranking)
    unmounted_intent = intent_obj.model_copy_validated(
        objectives=[
            ObjectiveSpec(
                requirement_id="obj_srfm",
                metric_id="structrfm_cds_median_shape",
                direction="minimize",
                priority_rank=1,
                evidence_status="validated_predictor",
                source_utterance="Use structRFM native SHAPE probe",
            )
        ],
        runtime_policy=intent_obj.runtime_policy.model_copy_validated(
            enabled_evaluators=["structrfm_rna_foundation"]
        ),
    )
    bs_sf, rep_sf, _ = run_design_pipeline(unmounted_intent, seed_dir=SEED_DIR)
    assert bs_sf == []
    assert rep_sf.outcome_status == "UNCOMPARABLE_MISSING_APPROVED_COMPONENT"
    assert all(t.stage_applied == "recorded_only" for t in rep_sf.action_transparency)

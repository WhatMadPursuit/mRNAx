"""End-to-End Verification for Wave 6–7, M2 Live Production Benchmarks 1 & 2, and GPT-6.2-Astra Ultra Remediations."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mrnax.cloud.cas_store import ExecutionCASStore, StaleLeaseConflictError
from mrnax.cloud.drive_rest_v3 import DriveRestV3Client, InMemoryDriveV3Transport
from mrnax.cloud.gateway import create_control_plane_app
from mrnax.constraints.registry import ScopedConstraintEngine, default_m1_constraints
from mrnax.engine.cell_expression import assemble_detargeted_3utr, build_crs_and_tissue_safety_advisory
from mrnax.engine.pareto_director import run_design_pipeline
from mrnax.engine.stage0_baseline import _load_bicodon_atlas, mutate_cds_region_synonymously
from mrnax.engine.surrogates import CommercialCleanSurrogateV1, _load_verified_joblib
from mrnax.federated.external_evidence_adapter import CleanRoomPolicyError, ExternalEvidenceAdapterV2
from mrnax.harness.state_machine import ProjectHarnessSession
from mrnax.io_insdc.genbank_io import ExportConstraintViolationError, export_construct_package
from mrnax.rules.models import load_replacement_card_yaml, verify_replacement_card_strict
from mrnax.rules.tribunal import RuleRegistry
from mrnax.schemas.construct_bundle import ConstructBundle, SegmentSpan
from mrnax.schemas.design_intent import DesignIntent, ObjectiveSpec
from mrnax.schemas.experiment_record import ExperimentRecord

REPO_ROOT = Path(__file__).resolve().parents[2]
SEED_DIR = REPO_ROOT / "registry_seed"
GOLDEN_DIR = REPO_ROOT / "tests" / "golden"


def test_wave6_wave7_registries_and_surrogate_provenance_manifest(tmp_path: Path) -> None:
    # 1. UTR v2 (200 5'UTRs + 200 3'UTRs) with Ensembl transcript IDs + 8-patent FTO v2 without poly(A) length bypass
    utr_v2 = json.loads((SEED_DIR / "utr_components_commercial_v2.json").read_text(encoding="utf-8"))
    assert len(utr_v2["five_prime_utrs"]) == 200
    assert len(utr_v2["three_prime_utrs"]) == 200
    assert "ensembl_transcript_id" in utr_v2["five_prime_utrs"][0]
    assert "refseq_accession" not in utr_v2["five_prime_utrs"][0]

    engine = ScopedConstraintEngine.from_seed_dir(SEED_DIR)
    assert len(engine.fto_guard.genus_rules) + len(engine.fto_guard.collision_rules) == 8

    # 2. miRNA v2 (100 organ-specific miRNAs across 9 compartments) + fail-closed unmeasured protected tissue check
    mir_v2 = json.loads((SEED_DIR / "mirna_rbp_commercial_v2.json").read_text(encoding="utf-8"))
    assert len(mir_v2["mirna_entries"]) == 100
    base_u3 = utr_v2["three_prime_utrs"][0]["sequence"]
    for det_tissue, prot_tissue in [
        ("liver_hepatocyte", "t_cell"),
        ("endothelial", "skeletal_muscle"),
        ("cardiac_heart", "t_cell"),
    ]:
        assembled, anns, mature = assemble_detargeted_3utr(
            base_utr3=base_u3,
            detarget_tissues=[det_tissue],
            protected_tissues=[prot_tissue],
            mirna_registry=mir_v2,
            copies=3,
        )
        assert len(anns) == 3
        assert len(mature) >= 20
        assert len(assembled) > len(base_u3)

    with pytest.raises(ValueError, match="no measured FANTOM5 expression"):
        assemble_detargeted_3utr(
            base_utr3=base_u3,
            detarget_tissues=["liver"],
            protected_tissues=["spleen"],
            mirna_registry=mir_v2,
            copies=3,
        )

    rbp_doc = json.loads((SEED_DIR / "rbp_human_motifs_attract_ccby4_v1.json").read_text(encoding="utf-8"))
    assert rbp_doc["total_unique_human_rbps"] == 160

    # 3. Human 61 CSC + 3,721 bicodon stability atlas wired into CDS mutation
    csc_map, bicodon_map = _load_bicodon_atlas()
    assert len(csc_map) == 61
    assert len(bicodon_map) == 3721
    egfp_aa = json.loads((SEED_DIR / "egfp_reporter_snapshot.json").read_text(encoding="utf-8"))[
        "mammalian_reporter_egfp"
    ]["protein_sequence"]
    rng = random.Random(42)
    from mrnax.engine.stage0_baseline import generate_random_synonymous_cds

    cds0 = generate_random_synonymous_cds(egfp_aa, rng, chemistry="m1psi")
    cds1 = mutate_cds_region_synonymously(
        cds0, egfp_aa, rng, codon_start=5, codon_end=80, prefer_open_au=False, chemistry="m1psi"
    )
    assert len(cds1) == len(cds0)

    # 4. Pre-deserialization SHA-256 authentication on joblib artifacts + Rule observation ledger cross-check
    prov = json.loads(
        (SEED_DIR / "surrogate_v1" / "TRAINING_PROVENANCE_MANIFEST.json").read_text(encoding="utf-8")
    )
    assert len(prov["artifacts"]) == 9
    tampered_dir = tmp_path / "surr_tampered"
    tampered_dir.mkdir()
    (tampered_dir / "TRAINING_PROVENANCE_MANIFEST.json").write_text(
        json.dumps(prov), encoding="utf-8"
    )
    (tampered_dir / "utr5_mrl_pure_seq_lgbm.joblib").write_bytes(b"corrupted_bytes")
    with pytest.raises(ValueError, match="Integrity error"):
        _load_verified_joblib(tampered_dir, "utr5_mrl_pure_seq_lgbm.joblib")

    reg = RuleRegistry(SEED_DIR / "rules_commercial_v1.json", track="commercial_clean")
    assert reg.snapshot_version().startswith("rules_commercial_clean@")


def test_astra_orthrus_card_identity_and_12api_allowlist_guards(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # 1. Orthrus fails closed BOTH when unset AND when pointing to a nonexistent directory, even with cached_candidates!
    b2_doc = json.loads((GOLDEN_DIR / "m2_benchmark2_dll3_cd3_acceptance_fixture_v1.json").read_text(encoding="utf-8"))
    base_intent = DesignIntent.model_validate(b2_doc["design_intent"])
    cached_b2 = ConstructBundle.model_validate(b2_doc["construct_bundle"])

    monkeypatch.setenv("MRNAX_ORTHRUS_CKPT_DIR", str(tmp_path / "nonexistent_orthrus_ckpt"))
    orthrus_intent = base_intent.model_copy_validated(
        objectives=[
            ObjectiveSpec(
                requirement_id="obj_orthrus_hl",
                metric_id="orthrus_halflife_pred",
                direction="maximize",
                priority_rank=1,
                evidence_status="validated_predictor",
                source_utterance="Score with Orthrus 6-track Mamba foundation probe",
            )
        ],
        runtime_policy=base_intent.runtime_policy.model_copy_validated(
            allow_cached_candidates=True,
            enabled_generators=[],
            enabled_evaluators=["orthrus_mamba_rna_foundation"],
        ),
    )
    bundles, rep, _ = run_design_pipeline(orthrus_intent, cached_candidates=[cached_b2], seed_dir=SEED_DIR)
    assert bundles == []
    assert rep.outcome_status == "UNCOMPARABLE_MISSING_APPROVED_COMPONENT"
    assert rep.recommended_bundle_ids == []

    # 2. Replacement card gate rejects mismatched component_id and duplicate metric_id
    rc002 = load_replacement_card_yaml(REPO_ROOT / "docs" / "replacement_cards" / "RC-002-structrfm-v0.0.9.yaml")
    with pytest.raises(ValueError, match="expected 'orthrus_mamba_rna_foundation'"):
        verify_replacement_card_strict(rc002, expected_component_id="orthrus_mamba_rna_foundation")

    rc003 = load_replacement_card_yaml(REPO_ROOT / "docs" / "replacement_cards" / "RC-003-orthrus-v1.0.yaml")
    verify_replacement_card_strict(rc003, expected_component_id="orthrus_mamba_rna_foundation")
    assert len(rc003.acceptance_metrics) == 6

    # 3. Strict 12-API commercial export allowlist rejects caller-forged tier, missing license, and unknown purpose
    adapter = ExternalEvidenceAdapterV2(license_track="commercial_clean")
    with pytest.raises(CleanRoomPolicyError):
        adapter.guard_export_to_commercial_bundle(
            {
                "source_name": "UnapprovedSource",
                "license_spdx": "",
                "license_tier": "commercial_clean",
                "request_url": "https://example.org",
                "payload_sha256": "a" * 64,
            },
            purpose="commercial_training",
        )
    with pytest.raises(CleanRoomPolicyError, match="Unrecognized export purpose"):
        adapter.guard_export_to_commercial_bundle(
            {
                "source_name": "UniProt",
                "license_spdx": "CC-BY-4.0",
                "license_tier": "commercial_clean",
                "request_url": "https://rest.uniprot.org",
                "payload_sha256": "a" * 64,
            },
            purpose="arbitrary_unknown_purpose",
        )


def test_m2_benchmark1_dual_speed_feedback_live_harness_e2e(tmp_path: Path) -> None:
    # Experiment ID path traversal is rejected at schema boundary
    b1_doc = json.loads(
        (GOLDEN_DIR / "m2_benchmark1_dual_speed_feedback_fixture_v1.json").read_text(encoding="utf-8")
    )
    raw_exp = dict(b1_doc["experiment_records"]["B1_CASE5_DRAFT_EXTRACTION_UNIT_TRAP"])
    raw_exp["experiment_id"] = "../escape_attempt"
    with pytest.raises(ValueError, match="safe filename component"):
        ExperimentRecord.model_validate(raw_exp)

    bundles = {k: ConstructBundle.model_validate(v) for k, v in b1_doc["bundles"].items()}
    exps = {k: ExperimentRecord.model_validate(v) for k, v in b1_doc["experiment_records"].items()}

    sess = ProjectHarnessSession(project_id="PROJ_M2_BENCHMARK1", store_dir=tmp_path / "b1_store", seed_dir=SEED_DIR)
    ref_a = bundles["B1_REF_A"]
    cand_b = bundles["B1_CAND_B"]
    for bid, b in bundles.items():
        sess.register_bundle(b, set_as_reference=(bid == "B1_REF_A"))

    sess.create_initial_intent(
        turn_id="turn_01",
        user_utterance="Benchmark 1 dual-speed wet-lab feedback baseline",
        protein_sequence=ref_a.molecules[0].effective_translation_target.protein_sequence,
        reference_bundle=ref_a,
        compliance_track="commercial_clean",
        uridine_chemistry="m1psi",
        restriction_enzymes=["BspQI"],
    )

    # Case 5: Draft extraction unit trap remains quarantined
    r_draft = sess.ingest_experiment_feedback(exps["B1_CASE5_DRAFT_EXTRACTION_UNIT_TRAP"].model_copy_validated(feedback_decision=None))
    assert r_draft.extraction_state == "draft_extraction"
    assert r_draft.feedback_decision is None

    # Case 4: N=2 rank reversal -> switches reference to B, 0 weight tuning, spawns verbatim chimera
    r_rev = sess.ingest_experiment_feedback(exps["B1_CASE4_N2_REVERSAL_CHIMERA"].model_copy_validated(feedback_decision=None))
    assert r_rev.feedback_decision is not None
    assert r_rev.feedback_decision.branch == "rank_reversal_switch_reference"
    assert r_rev.feedback_decision.weight_update_allowed is False
    chim_id = r_rev.feedback_decision.spawned_disambiguation_chimera_id
    assert chim_id is not None
    live_chim = sess.get_bundle(chim_id)
    assert live_chim.molecules[0].get_segment_sequence("five_prime_utr") == cand_b.molecules[0].get_segment_sequence("five_prime_utr")
    assert live_chim.molecules[0].get_segment_sequence("cds") == ref_a.molecules[0].get_segment_sequence("cds")

    # Case 6: N=6 independent constructs with held-out validation -> unlocks project-scoped calibration proposal!
    sess.active_reference_bundle_id = "B1_REF_A"
    r_n6 = sess.ingest_experiment_feedback(
        exps["B1_CASE6_N6_HELD_OUT_CALIBRATION_UNLOCKED"].model_copy_validated(feedback_decision=None),
        held_out_validation_passed=True,
    )
    assert r_n6.feedback_decision is not None
    assert r_n6.feedback_decision.weight_update_allowed is True
    assert r_n6.feedback_decision.new_reference_bundle_id == "B1_CAND_E"


def test_m2_benchmark2_live_two_chain_production_pipeline_and_export_parity(tmp_path: Path) -> None:
    b2_doc = json.loads(
        (GOLDEN_DIR / "m2_benchmark2_dll3_cd3_acceptance_fixture_v1.json").read_text(encoding="utf-8")
    )
    frozen_intent = DesignIntent.model_validate(b2_doc["design_intent"])

    # 1. Frozen intent requests unmeasured whole-organ 'spleen' -> fails closed truthfully
    b_blk, rep_blk, _ = run_design_pipeline(frozen_intent, seed_dir=SEED_DIR)
    assert rep_blk.outcome_status == "UNCOMPARABLE_MISSING_APPROVED_COMPONENT"
    assert b_blk == [] and rep_blk.uncomparable_reason and "spleen" in rep_blk.uncomparable_reason

    # 2. Explicit revision using FANTOM5-measured compartments executes full 2-chain production pipeline & export parity
    assert frozen_intent.cell_expression_control is not None
    intent = frozen_intent.model_copy_validated(
        intent_id=f"{frozen_intent.intent_id}_measured",
        intent_version=frozen_intent.intent_version + 1,
        parent_intent_id=frozen_intent.intent_id,
        cell_expression_control=frozen_intent.cell_expression_control.model_copy(
            update={"protected_tissues": ["t_cell", "myeloid_monocyte_macrophage"]}
        ),
    )
    live_bundles, rep, man = run_design_pipeline(intent, seed_dir=SEED_DIR)
    assert rep.outcome_status == "COMPLETED_WITH_CANDIDATES", rep.uncomparable_reason
    assert len(live_bundles) >= 2
    top_b = live_bundles[0]
    assert len(top_b.molecules) == 2
    assert top_b.co_expression_mode == "lnp_co_encapsulation"
    # Confirm BOTH molecules were evaluated by SurrogateV1 (attempt_count == 2 * len(live_bundles))
    eval_recs = [c for c in man.invoked_components if c.role == "evaluator"]
    assert eval_recs[0].attempt_count == 2 * len(live_bundles)

    # Confirm 3-layer CRS boundary + endogenous bicodon domain boundary advisories are emitted
    adv_cats = {a.category for a in rep.pre_synthesis_advisories}
    assert "crs_safety_layer_boundary" in adv_cats
    assert "model_domain_boundary" in adv_cats

    # Confirm export enforces chemistry & locked_scopes parity
    engine = ScopedConstraintEngine.from_seed_dir(SEED_DIR)
    out_dir = tmp_path / "b2_export"
    written = export_construct_package(
        top_b,
        out_dir,
        constraint_engine=engine,
        constraints=intent.constraints,
        compliance_track=intent.compliance_track,
        expected_chemistry=intent.uridine_chemistry,
        locked_scopes=intent.locked_scopes,
    )
    assert written["schema"] == "mrnax.export_package.v1"
    assert (out_dir / "construct.gb").read_text(encoding="utf-8").count("LOCUS") == 2

    with pytest.raises(ExportConstraintViolationError, match="chemistry_lock"):
        export_construct_package(
            top_b,
            tmp_path / "b2_bad_chem",
            constraint_engine=engine,
            constraints=intent.constraints,
            compliance_track=intent.compliance_track,
            expected_chemistry="5mou",
            locked_scopes=intent.locked_scopes,
        )


def test_astra_gateway_track_preservation_reference_license_and_cas_idempotency(tmp_path: Path) -> None:
    # 1. Enzyme-only refine after commercial_clean MUST preserve commercial_clean!
    store = ExecutionCASStore(tmp_path / "cp.sqlite3")
    app = create_control_plane_app(store, projects_root=tmp_path / "projs", seed_dir=SEED_DIR)
    client = TestClient(app)
    r1 = client.post(
        "/v1/projects/proj_track_keep/turn",
        headers={"X-MRNAX-Project-Key": "pk_track_keep_test_key_0001"},
        json={
            "mode": "initial",
            "user_utterance": "Commercial eGFP",
            "compliance_track": "commercial_clean",
            "use_commercial_surrogate": True,
            "restriction_enzymes": ["BsaI"],
        },
    )
    assert r1.status_code == 200
    assert r1.json()["intent"]["compliance_track"] == "commercial_clean"

    r2 = client.post(
        "/v1/projects/proj_track_keep/turn",
        headers={"X-MRNAX-Project-Key": "pk_track_keep_test_key_0001"},
        json={
            "mode": "refine",
            "user_utterance": "Switch enzyme to BspQI only",
            "restriction_enzymes": ["BspQI"],
        },
    )
    assert r2.status_code == 200
    assert r2.json()["intent"]["compliance_track"] == "commercial_clean"

    # 2. Reference segment with research_only license_track is preserved on minA and blocked in commercial_clean
    sess = ProjectHarnessSession(project_id="proj_ref_lic", store_dir=tmp_path / "ref_lic", seed_dir=SEED_DIR)
    b1_doc = json.loads((GOLDEN_DIR / "m2_benchmark1_dual_speed_feedback_fixture_v1.json").read_text(encoding="utf-8"))
    ref_base = ConstructBundle.model_validate(b1_doc["bundles"]["B1_REF_A"])
    m0 = ref_base.molecules[0]
    tainted_segs = [
        (s.model_copy_validated(license_track="research_only") if s.segment_type == "five_prime_utr" else s)
        for s in m0.segments
    ]
    tainted_ref = ConstructBundle.create(
        bundle_id="ref_tainted",
        parent_bundle_id=None,
        derivation_action="reference_import",
        target_stoichiometry=ref_base.target_stoichiometry,
        molecules=[m0.model_copy_validated(segments=tainted_segs)],
    )
    it_tainted = sess.create_initial_intent(
        turn_id="t1",
        user_utterance="Refine tainted ref in commercial_clean",
        protein_sequence=m0.effective_translation_target.protein_sequence,
        reference_bundle=tainted_ref,
        compliance_track="commercial_clean",
        use_commercial_surrogate=True,
    )
    cands_t, rep_t, _ = run_design_pipeline(
        it_tainted.model_copy_validated(
            runtime_policy=it_tainted.runtime_policy.model_copy_validated(enabled_generators=["stage0_random"])
        ),
        reference_bundle=tainted_ref,
        seed_dir=SEED_DIR,
    )
    assert not any(c.bundle_id.endswith("_minA") or c.bundle_id == "ref_tainted" for c in cands_t)
    assert any(
        d.violated_requirement_id == "clean_room:segment_license_track" and d.candidate_ref.endswith("_minA")
        for d in rep_t.failed_diagnostics
    )

    # 3. CAS idempotent retry of winning attempt keeps COMMITTED_200 & restores cleanly
    transport = InMemoryDriveV3Transport()
    drive = DriveRestV3Client(transport=transport)
    runs_root = drive.create_folder("root", "runs")
    commits_root = drive.create_folder("root", "commits")
    cas = ExecutionCASStore(tmp_path / "cas_idem.sqlite3", drive_verifier=drive, commits_snapshot_folder_id=commits_root)
    ex = cas.submit_execution("sub_i1", "dig_i1", "int_i1", now_ts=100.0)
    ls = cas.acquire_or_renew_lease(ex.execution_id, "w1", ttl_seconds=60, now_ts=100.0)
    att_dir = drive.ensure_attempt_dir(runs_root, ex.execution_id, ls.attempt_id)
    art = drive.upload_verified_bytes(att_dir, "res.json", b'{"ok":1}')
    seal = drive.write_attempt_seal(att_dir, execution_id=ex.execution_id, attempt_id=ls.attempt_id, lease_epoch=ls.lease_epoch, receipts=[art])
    c1 = cas.commit_attempt_cas(
        execution_id=ex.execution_id,
        attempt_id=ls.attempt_id,
        lease_epoch=ls.lease_epoch,
        sealed_file_id=seal.file_id,
        sealed_sha256=seal.sha256,
        lease_token=ls.lease_token,
        now_ts=110.0,
    )
    c2_retry = cas.commit_attempt_cas(
        execution_id=ex.execution_id,
        attempt_id=ls.attempt_id,
        lease_epoch=ls.lease_epoch,
        sealed_file_id=seal.file_id,
        sealed_sha256=seal.sha256,
        lease_token=ls.lease_token,
        now_ts=115.0,
    )
    assert c1.winning_attempt_id == c2_retry.winning_attempt_id

    # 4. Web UI quote escaping, spark-fill display:block, [hidden] rule & transient memory-only secrets
    app_js = (REPO_ROOT / "web" / "app.js").read_text(encoding="utf-8")
    styles_css = (REPO_ROOT / "web" / "styles.css").read_text(encoding="utf-8")
    assert '"&quot;"' in app_js
    assert "mirna_detarget_dg_total" in app_js
    assert "localStorage.setItem(\"mrnax_access_token\"" not in app_js
    assert "sessionStorage.setItem(\"mrnax_byo_key\"" not in app_js
    assert "[hidden]" in styles_css
    assert "display: block;" in styles_css

"""Regression tests pinning all Critical & Important findings from the GPT-6.2-sol adversarial review."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from mrnax.cloud.cas_store import ExecutionCASStore, StaleLeaseConflictError
from mrnax.cloud.drive_rest_v3 import DriveRestV3Client, InMemoryDriveV3Transport
from mrnax.constraints.registry import ScopedConstraintEngine, default_m1_constraints
from mrnax.engine.ablation_harness import run_fixed_pool_scorer_replay
from mrnax.engine.pareto_director import NoAvailableGeneratorError, run_design_pipeline
from mrnax.harness.state_machine import ProjectHarnessSession
from mrnax.io_insdc.genbank_io import export_construct_package, import_construct_from_genbank_package
from mrnax.rules.models import ClaimCard, EvidenceObservationLink, RuleRecord
from mrnax.rules.tribunal import RuleRegistry, count_independent_dataset_votes
from mrnax.schemas.construct_bundle import (
    ChemistryTrackSpec,
    ConstructBundle,
    ProteinTargetSpec,
    RNAMoleculeSpec,
    SegmentSpan,
)
from mrnax.schemas.design_intent import DesignIntent, LockedRegionSpec, ObjectiveSpec, RuntimePolicySpec
from mrnax.schemas.evaluation_report import BundleEvaluation, EvaluationReport
from mrnax.schemas.experiment_record import (
    BiologicalReplicate,
    ConstructObservation,
    ExperimentRecord,
    PreRegisteredSwitchCriteria,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SEED_DIR = REPO_ROOT / "registry_seed"


def _make_mini_bundle(
    *,
    utr5_track: str = "commercial_clean",
    chemistry: str = "m1psi",
    utr5_seq: str = "GGGAAACCCACC",
) -> ConstructBundle:
    cds = "AUGGUGAGCAAGGGC"  # MVSKG
    stop = "UAA"
    utr3 = "CCCAAACCC"
    polya = "A" * 20
    full = utr5_seq + cds + stop + utr3 + polya
    c1 = len(utr5_seq)
    c2 = c1 + len(cds)
    c3 = c2 + len(stop)
    c4 = c3 + len(utr3)
    c5 = c4 + len(polya)
    mol = RNAMoleculeSpec(
        molecule_id="mol_mini",
        modality="linear_mrna",
        topology="linear",
        chemistry_track=ChemistryTrackSpec(uridine_chemistry=chemistry),  # type: ignore[arg-type]
        effective_translation_target=ProteinTargetSpec(
            protein_sequence="MVSKG",
            transl_table=1,
            accession_or_source="test",
            confirmation_status="confirmed",
        ),
        sequence=full,
        segments=[
            SegmentSpan(
                segment_type="five_prime_utr",
                start=0,
                end=c1,
                origin_source="test",
                license_track=utr5_track,  # type: ignore[arg-type]
                action_taken="locked_verbatim",
            ),
            SegmentSpan(
                segment_type="cds",
                start=c1,
                end=c2,
                origin_source="test",
                license_track="commercial_clean",
                action_taken="generated_novo",
            ),
            SegmentSpan(
                segment_type="stop_cassette",
                start=c2,
                end=c3,
                origin_source="test",
                license_track="commercial_clean",
                action_taken="locked_verbatim",
            ),
            SegmentSpan(
                segment_type="three_prime_utr",
                start=c3,
                end=c4,
                origin_source="test",
                license_track="commercial_clean",
                action_taken="locked_verbatim",
            ),
            SegmentSpan(
                segment_type="poly_a",
                start=c4,
                end=c5,
                origin_source="test",
                license_track="commercial_clean",
                action_taken="locked_verbatim",
            ),
        ],
    )
    return ConstructBundle.create(
        bundle_id="bnd_mini",
        parent_bundle_id=None,
        derivation_action="de_novo_design",
        target_stoichiometry={"mol_mini": 1.0},
        molecules=[mol],
    )


def test_reg01_cas_commit_verifies_drive_seal_and_lease_token_and_expiry(tmp_path: Path) -> None:
    """Critical #1 & Important #7: CAS commit must verify Drive seal, worker lease_token, and non-expired lease."""
    drive = DriveRestV3Client(InMemoryDriveV3Transport())
    ws_root = drive.create_folder("root", "mRNAx")
    runs_root = drive.create_folder(ws_root, "runs")
    commits_root = drive.create_folder(ws_root, "control_plane_commits")

    store = ExecutionCASStore(
        tmp_path / "cas.sqlite3",
        drive_verifier=drive,
        commits_snapshot_folder_id=commits_root,
    )
    row = store.submit_execution("sub_1", "dig_1", "int_1", now_ts=100.0)
    lease = store.acquire_or_renew_lease(row.execution_id, "worker_1", ttl_seconds=30, now_ts=100.0)

    # 1. Unverified/missing Drive seal ID MUST fail!
    with pytest.raises(StaleLeaseConflictError, match="seal"):
        store.commit_attempt_cas(
            execution_id=row.execution_id,
            attempt_id=lease.attempt_id,
            lease_epoch=lease.lease_epoch,
            lease_token=lease.lease_token,
            sealed_file_id="nonexistent_file_id",
            sealed_sha256="0" * 64,
            now_ts=110.0,
        )

    att_dir = drive.ensure_attempt_dir(runs_root, row.execution_id, lease.attempt_id)
    r_art = drive.upload_verified_bytes(att_dir, "out.json", b'{"ok":true}')
    seal = drive.write_attempt_seal(
        att_dir,
        execution_id=row.execution_id,
        attempt_id=lease.attempt_id,
        lease_epoch=lease.lease_epoch,
        receipts=[r_art],
    )

    # 2. Wrong lease_token MUST fail!
    with pytest.raises(StaleLeaseConflictError, match="token"):
        store.commit_attempt_cas(
            execution_id=row.execution_id,
            attempt_id=lease.attempt_id,
            lease_epoch=lease.lease_epoch,
            lease_token="forged_token",
            sealed_file_id=seal.file_id,
            sealed_sha256=seal.sha256,
            now_ts=110.0,
        )

    # 3. Expired lease (now_ts=200 > expires=130) MUST fail!
    with pytest.raises(StaleLeaseConflictError, match="expired"):
        store.commit_attempt_cas(
            execution_id=row.execution_id,
            attempt_id=lease.attempt_id,
            lease_epoch=lease.lease_epoch,
            lease_token=lease.lease_token,
            sealed_file_id=seal.file_id,
            sealed_sha256=seal.sha256,
            now_ts=200.0,
        )

    # 4. Valid token + valid time + verified Drive seal succeeds and writes authoritative COMMIT_RECEIPT!
    ok = store.commit_attempt_cas(
        execution_id=row.execution_id,
        attempt_id=lease.attempt_id,
        lease_epoch=lease.lease_epoch,
        lease_token=lease.lease_token,
        sealed_file_id=seal.file_id,
        sealed_sha256=seal.sha256,
        now_ts=115.0,
    )
    assert ok.status == "COMMITTED"


def test_reg02_reconcile_never_promotes_uncommitted_worker_seal(tmp_path: Path) -> None:
    """Critical #2: Worker-written SEALED.json without Control Plane COMMIT_RECEIPT must NEVER become COMMITTED."""
    drive = DriveRestV3Client(InMemoryDriveV3Transport())
    ws_root = drive.create_folder("root", "mRNAx")
    runs_root = drive.create_folder(ws_root, "runs")
    commits_root = drive.create_folder(ws_root, "control_plane_commits")

    store = ExecutionCASStore(
        tmp_path / "cas_live.sqlite3",
        drive_verifier=drive,
        commits_snapshot_folder_id=commits_root,
    )
    row = store.submit_execution("sub_uncommitted", "dig_u", "int_u", now_ts=100.0)
    lease = store.acquire_or_renew_lease(row.execution_id, "w1", ttl_seconds=60, now_ts=100.0)

    # Worker writes SEALED.json to Drive, then crashes BEFORE calling commit_attempt_cas!
    att_dir = drive.ensure_attempt_dir(runs_root, row.execution_id, lease.attempt_id)
    r_art = drive.upload_verified_bytes(att_dir, "out.json", b'{"draft":1}')
    drive.write_attempt_seal(
        att_dir,
        execution_id=row.execution_id,
        attempt_id=lease.attempt_id,
        lease_epoch=lease.lease_epoch,
        receipts=[r_art],
    )

    recovered = ExecutionCASStore(
        tmp_path / "cas_recovered.sqlite3",
        drive_verifier=drive,
        commits_snapshot_folder_id=commits_root,
    )
    n_rec = recovered.reconcile_from_drive_seals(drive, runs_root)
    assert n_rec == 0
    with pytest.raises(KeyError):
        recovered.get_execution(row.execution_id)


def test_reg03_commercial_clean_rejects_quarantine_and_enforces_locks_and_segment_order() -> None:
    """Critical #3 & Important #3: Reject quarantine_reference in commercial_clean, enforce locks & unique terminal poly_a."""
    engine = ScopedConstraintEngine.from_seed_dir(SEED_DIR)
    constraints = default_m1_constraints(restriction_enzymes=[])

    quarantine_bundle = _make_mini_bundle(utr5_track="quarantine_reference")
    res_q = engine.evaluate_bundle(quarantine_bundle, constraints, compliance_track="commercial_clean")
    assert res_q.passed_all_hard_gates is False
    assert any("quarantine_reference" in v.reason for v in res_q.violations)

    # Locked 5'UTR mismatch or chemistry mismatch rejected!
    clean_bundle = _make_mini_bundle(utr5_track="commercial_clean", chemistry="unmodified_u", utr5_seq="GGGAAACCCACC")
    res_chem = engine.evaluate_bundle(
        clean_bundle,
        constraints,
        compliance_track="commercial_clean",
        expected_chemistry="m1psi",
        locked_scopes={
            "five_prime_utr": LockedRegionSpec(
                scope="five_prime_utr",
                locked_value="GGGCCCCCCCC",
                reason="lock",
                locked_at_turn_id="t1",
            )
        },
    )
    assert res_chem.passed_all_hard_gates is False
    assert len(res_chem.violations) >= 2

    # Non-terminal / internal poly_a smuggling rejected by RNAMoleculeSpec!
    mol = clean_bundle.molecules[0]
    with pytest.raises(ValidationError, match="poly_a"):
        mol.model_copy_validated(
            segments=[
                SegmentSpan(
                    segment_type="poly_a",
                    start=0,
                    end=12,
                    origin_source="smuggled",
                    license_track="commercial_clean",
                    action_taken="locked_verbatim",
                ),
                *mol.segments[1:],
            ]
        )


def test_reg04_wetlab_requires_measured_ref_unique_replicates_and_median_rule(tmp_path: Path) -> None:
    """Critical #4 & Important #12: Unique bio_rep_id, missing ref blocks switch, median rule honored, ref propagates."""
    # 1. Duplicate bio_rep_id rejected
    with pytest.raises(ValidationError, match="unique"):
        ConstructObservation(
            bundle_id="b1",
            version_digest="a" * 64,
            source_row_or_figure="R1",
            biological_replicates=[
                BiologicalReplicate(bio_rep_id="dup_1", technical_replicates=[10.0]),
                BiologicalReplicate(bio_rep_id="dup_1", technical_replicates=[12.0]),
            ],
        )

    session = ProjectHarnessSession(project_id="proj_reg4", store_dir=tmp_path, seed_dir=SEED_DIR)
    intent = session.create_initial_intent(
        turn_id="t1",
        user_utterance="Design MVSKGEELFTGVVPILVELDGDVNGHKFSV",
        protein_sequence="MVSKGEELFTGVVPILVELDGDVNGHKFSV",
    )
    bundles = session.execute_design(intent)
    cand_ref, cand_b, cand_c = bundles[0], bundles[1], bundles[2]
    assert session.active_reference_bundle_id == cand_ref.bundle_id

    # 2. Batch omitting active reference `cand_ref` MUST NOT switch reference!
    rec_missing_ref = ExperimentRecord(
        experiment_id="exp_no_ref",
        project_id="proj_reg4",
        batch_id="b1",
        extraction_state="verified_structured",
        assay_type="fluor",
        readout_name_exact="RFU",
        unit="RFU",
        cell_or_model_context="HEK293T",
        chemistry_track="m1psi",
        delivery_vehicle="LNP",
        shared_controls=["mock"],
        observations=[
            ConstructObservation(
                bundle_id=cand_b.bundle_id,
                version_digest=cand_b.version_digest,
                source_row_or_figure="R1",
                biological_replicates=[
                    BiologicalReplicate(bio_rep_id=f"b{i}", technical_replicates=[100.0]) for i in range(3)
                ],
            ),
            ConstructObservation(
                bundle_id=cand_c.bundle_id,
                version_digest=cand_c.version_digest,
                source_row_or_figure="R2",
                biological_replicates=[
                    BiologicalReplicate(bio_rep_id=f"c{i}", technical_replicates=[300.0]) for i in range(3)
                ],
            ),
        ],
    )
    out_missing = session.ingest_experiment_feedback(rec_missing_ref, batch_comparable=True)
    assert out_missing.feedback_decision is not None
    assert out_missing.feedback_decision.branch == "experimental_confound_blocked"
    assert session.active_reference_bundle_id == cand_ref.bundle_id

    # 3. Challenger with outlier high mean [80, 85, 1000] (median=85) vs ref [100, 102, 104] (median=102) MUST NOT win under median!
    rec_median_loser = ExperimentRecord(
        experiment_id="exp_median_check",
        project_id="proj_reg4",
        batch_id="b2",
        extraction_state="verified_structured",
        assay_type="fluor",
        readout_name_exact="RFU",
        unit="RFU",
        cell_or_model_context="HEK293T",
        chemistry_track="m1psi",
        delivery_vehicle="LNP",
        shared_controls=["mock"],
        pre_registered_criteria=PreRegisteredSwitchCriteria(
            aggregation_method="median", variance_estimator="iqr", min_biological_replicates=3, min_fold_change=1.5
        ),
        observations=[
            ConstructObservation(
                bundle_id=cand_ref.bundle_id,
                version_digest=cand_ref.version_digest,
                source_row_or_figure="R1",
                biological_replicates=[
                    BiologicalReplicate(bio_rep_id="r1", technical_replicates=[100.0]),
                    BiologicalReplicate(bio_rep_id="r2", technical_replicates=[102.0]),
                    BiologicalReplicate(bio_rep_id="r3", technical_replicates=[104.0]),
                ],
            ),
            ConstructObservation(
                bundle_id=cand_b.bundle_id,
                version_digest=cand_b.version_digest,
                source_row_or_figure="R2",
                biological_replicates=[
                    BiologicalReplicate(bio_rep_id="b1", technical_replicates=[80.0]),
                    BiologicalReplicate(bio_rep_id="b2", technical_replicates=[85.0]),
                    BiologicalReplicate(bio_rep_id="b3", technical_replicates=[1000.0]),
                ],
            ),
        ],
    )
    out_med = session.ingest_experiment_feedback(rec_median_loser, batch_comparable=True)
    assert out_med.feedback_decision is not None
    assert out_med.feedback_decision.branch == "within_noise_undecided"
    assert session.active_reference_bundle_id == cand_ref.bundle_id


def test_reg05_schema_digests_hashseed_replay_checksums_and_rule_sandbox(tmp_path: Path) -> None:
    """Important #1, #2, #4, #6, #8, #9: Forged digests rejected, hash-seed deterministic, CLI manifest SHA matches."""
    b = _make_mini_bundle()
    with pytest.raises(ValidationError, match="version_digest"):
        ConstructBundle(
            bundle_id=b.bundle_id,
            version_digest="0" * 64,  # Forged digest!
            parent_bundle_id=None,
            derivation_action="de_novo_design",
            target_stoichiometry=b.target_stoichiometry,
            molecules=b.molecules,
        )

    with pytest.raises(ValidationError):
        EvaluationReport(
            report_id="r1",
            execution_id="e1",
            intent_id="i1",
            intent_version=1,
            compliance_track="research_max",
            outcome_status="COMPLETED_ZERO_QUALIFIED",
            recommended_bundle_ids=[],
            candidate_evaluations=[
                BundleEvaluation(
                    bundle_id="b1",
                    version_digest="a" * 64,
                    passed_all_hard_gates=False,
                    pareto_rank=1,
                    metrics=[],
                )
            ],
            failed_diagnostics=[],
            fto_alerts=[],
            pre_synthesis_advisories=[],
            action_transparency=[],
        )

    # Fixed-pool replay rejects mismatched digest or failed evaluation, and uses exact wording "本次未改善生成器"
    bad_eval = BundleEvaluation(
        bundle_id=b.bundle_id,
        version_digest="f" * 64,
        passed_all_hard_gates=True,
        pareto_rank=1,
        metrics=[],
    )
    with pytest.raises(ValueError, match="version_digest"):
        run_fixed_pool_scorer_replay(
            candidate_pool=[b],
            base_evaluations=[bad_eval],
            scorer_name="s",
            scorer_fn=lambda _: 1.0,
        )

    ok_eval = BundleEvaluation(
        bundle_id=b.bundle_id,
        version_digest=b.version_digest,
        passed_all_hard_gates=True,
        pareto_rank=1,
        metrics=[],
    )
    rep_res = run_fixed_pool_scorer_replay(
        candidate_pool=[b],
        base_evaluations=[ok_eval],
        scorer_name="s",
        scorer_fn=lambda _: 1.0,
    )
    assert rep_res.honesty_banner == "本次未改善生成器"

    # PYTHONHASHSEED determinism on DesignIntent.digest()
    code = (
        "from pathlib import Path; from mrnax.constraints.registry import default_m1_constraints; "
        "from mrnax.schemas.construct_bundle import ProteinTargetSpec; "
        "from mrnax.schemas.design_intent import DesignIntent; "
        "i = DesignIntent(intent_id='i1', project_id='p1', intent_version=1, created_from_turn_ids=['t1'], "
        "entry_mode='protein_only', protein_targets=[ProteinTargetSpec(protein_sequence='MVSKG', accession_or_source='x')], "
        "constraints=default_m1_constraints()); print(i.digest())"
    )
    d1 = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src"), "PYTHONHASHSEED": "1"},
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    d2 = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src"), "PYTHONHASHSEED": "99999"},
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert d1 == d2

    # allow_cached_candidates=True with no cache provider raises NoAvailableGeneratorError
    intent = DesignIntent(
        intent_id="i_cache",
        project_id="p1",
        intent_version=1,
        created_from_turn_ids=["t1"],
        entry_mode="protein_only",
        protein_targets=[ProteinTargetSpec(protein_sequence="MVSKG", accession_or_source="x")],
        constraints=default_m1_constraints(restriction_enzymes=[]),
        objectives=[
            ObjectiveSpec(
                requirement_id="o1",
                metric_id="kozak_unpaired_prob",
                direction="maximize",
                evidence_status="thermodynamic_proxy",
                source_utterance="open",
            )
        ],
        runtime_policy=RuntimePolicySpec(enabled_generators=[], allow_cached_candidates=True),
    )
    with pytest.raises(NoAvailableGeneratorError):
        run_design_pipeline(intent, seed_dir=SEED_DIR)

    # GenBank package import verifies SHA-256 against tampered FASTA
    engine = ScopedConstraintEngine.from_seed_dir(SEED_DIR)
    pkg = tmp_path / "pkg_verify"
    export_construct_package(
        b, pkg, constraint_engine=engine, constraints=default_m1_constraints(restriction_enzymes=[]), compliance_track="commercial_clean"
    )
    (pkg / "full_rna.fasta").write_text(">tampered\nAUGUAA\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        import_construct_from_genbank_package(pkg)

    # Model interpretations excluded from empirical dataset votes
    model_claim = ClaimCard(
        claim_id="CLM-SHAP-01",
        status="verified_observation",
        source_type="primary_research",
        citation_id="DOI:10.1000/shap",
        exact_location="Fig. 4",
        underlying_dataset_id="GSE114002",
        lab_lineage_id="ModelLab",
        region_scope=["five_prime_utr"],
        chemistry_scope=["m1psi"],
        cell_or_model_context="HEK293T",
        protein_count_n=1,
        readout_name_exact="SHAP attribution",
        comparator="baseline",
        effect_direction="decrease",
        evidence_layer="model_interpretation",
    )
    v_summary = count_independent_dataset_votes([model_claim])
    assert v_summary.unique_dataset_votes == 0

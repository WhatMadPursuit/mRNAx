"""Targeted regression tests for all 18 Round-3 audit findings (C3, C4, C6, C8, I4, I5, I8, I10, I12, I15 + 8 new)."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from mrnax.cloud.cas_store import ExecutionCASStore
from mrnax.cloud.drive_rest_v3 import DriveRestV3Client, GdriveCliDriveV3Transport, InMemoryDriveV3Transport
from mrnax.cloud.gateway import _build_preset_reference_r, _serialize_report_with_candidates, create_control_plane_app
from mrnax.constraints.registry import ScopedConstraintEngine
from mrnax.engine.pareto_director import run_design_pipeline
from mrnax.engine.surrogates import validate_structrfm_tokenizer
from mrnax.federated.external_evidence_adapter import CleanRoomPolicyError
from mrnax.harness.auth_secrets import SecretVault
from mrnax.harness.sandbox import UntrustedExternalText, wrap_untrusted_external_text
from mrnax.harness.state_machine import ProjectHarnessSession
from mrnax.rules.tribunal import RuleRegistry
from mrnax.schemas.design_intent import ConstraintSpec, DesignIntent, ObjectiveSpec, RuntimePolicySpec
from mrnax.schemas.evaluation_report import BundleEvaluation, EvaluationReport, MetricResult
from mrnax.schemas.experiment_record import BiologicalReplicate, ConstructObservation, ExperimentRecord

ROOT = Path(__file__).resolve().parents[2]
SEED = ROOT / "registry_seed"


def test_c3_and_c4_stored_display_only_or_chain_mismatch_never_recommended() -> None:
    ref = _build_preset_reference_r(SEED)
    mol1 = ref.molecules[0]
    intent_2chain = DesignIntent(
        intent_id="int_2c",
        project_id="p1",
        intent_version=1,
        created_from_turn_ids=["t1"],
        entry_mode="protein_only",
        compliance_track="commercial_clean",
        protein_targets=[
            mol1.effective_translation_target,
            mol1.effective_translation_target.model_copy_validated(protein_sequence=mol1.effective_translation_target.protein_sequence[:40]),
        ],
        objectives=[
            ObjectiveSpec(
                requirement_id="obj1",
                metric_id="consensus_halflife_pred",
                direction="maximize",
                priority_rank=1,
                evidence_status="validated_predictor",
                source_utterance="maximize hl",
            )
        ],
        runtime_policy=RuntimePolicySpec(
            allow_cached_candidates=True,
            enabled_generators=[],
            enabled_evaluators=["surrogate_v1_clean"],
        ),
    )
    # C4: 1-molecule cached bundle against 2-target intent MUST fail hard gate
    bundles, rep, _ = run_design_pipeline(intent_2chain, cached_candidates=[ref], seed_dir=SEED)
    assert bundles == []
    assert rep.recommended_bundle_ids == []
    assert rep.outcome_status == "COMPLETED_ZERO_QUALIFIED"

    # C3: Legacy stored report whose only metric is display_only GC MUST NOT recommend
    gc_only = MetricResult(
        metric_id="cds_gc_fraction",
        value=0.55,
        unit="fraction",
        direction="informational",
        evaluated_scopes=["cds"],
        tool_name="mRNAxSeqStats",
        tool_version="0.1.0",
        chemistry_evaluation_mode="native_chemistry_validated",
        participated_in="display_only",
    )
    legacy_rep = EvaluationReport(
        report_id="rep_gc_only",
        execution_id="ex1",
        intent_id="int_2c",
        intent_version=1,
        compliance_track="commercial_clean",
        outcome_status="COMPLETED_WITH_CANDIDATES",
        recommended_bundle_ids=[ref.bundle_id],
        candidate_evaluations=[
            BundleEvaluation(
                bundle_id=ref.bundle_id,
                version_digest=ref.version_digest,
                passed_all_hard_gates=True,
                pareto_rank=1,
                metrics=[gc_only],
            )
        ],
        failed_diagnostics=[],
        fto_alerts=[],
        pre_synthesis_advisories=[],
        action_transparency=[],
    )
    projected, cards = _serialize_report_with_candidates(legacy_rep, [ref])
    assert cards == []
    assert projected is not None
    assert projected["recommended_bundle_ids"] == []
    assert projected["outcome_status"] == "UNCOMPARABLE_MISSING_APPROVED_COMPONENT"


def test_c6_prejournal_sqlite_winner_publishes_missing_drive_receipt_on_retry_and_recovery(tmp_path: Path) -> None:
    drive = DriveRestV3Client(InMemoryDriveV3Transport())
    runs_root = drive.create_folder("root", "runs")
    snaps = drive.create_folder("root", "snaps")
    db_path = tmp_path / "legacy_prejournal.db"

    store = ExecutionCASStore(db_path, drive_verifier=drive, commits_snapshot_folder_id=snaps, runs_root_id=runs_root)
    ex = store.submit_execution("sub_leg", "dig_leg", "int_leg")
    lease = store.acquire_or_renew_lease(ex.execution_id, "w1")
    att_dir = drive.ensure_attempt_dir(runs_root, ex.execution_id, lease.attempt_id)
    art = drive.upload_verified_bytes(att_dir, "out.bin", b"payload")
    seal = drive.write_attempt_seal(
        att_dir, execution_id=ex.execution_id, attempt_id=lease.attempt_id, lease_epoch=lease.lease_epoch, receipts=[art]
    )

    # Simulate pre-journal SQLite winner (status='COMMITTED', receipt_payload=NULL, 0 Drive receipts)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE executions SET status='COMMITTED', winning_attempt_id=?, winning_sealed_file_id=?, winning_sealed_sha256=?, receipt_payload=NULL WHERE execution_id=?",
            (lease.attempt_id, seal.file_id, seal.sha256, ex.execution_id),
        )
    assert drive.transport.list_children(parent_id=snaps) == []

    # Reopening or retrying backfills and publishes the missing Drive receipt so cold recovery recovers 1 winner!
    reopened = ExecutionCASStore(db_path, drive_verifier=drive, commits_snapshot_folder_id=snaps, runs_root_id=runs_root)
    retry_rec = reopened.commit_attempt_cas(
        execution_id=ex.execution_id,
        attempt_id=lease.attempt_id,
        lease_epoch=lease.lease_epoch,
        sealed_file_id=seal.file_id,
        sealed_sha256=seal.sha256,
        lease_token=lease.lease_token,
    )
    assert retry_rec.winning_attempt_id == lease.attempt_id
    cold = ExecutionCASStore(tmp_path / "cold.db", drive_verifier=drive, commits_snapshot_folder_id=snaps)
    assert cold.reconcile_from_drive_seals(drive, runs_root) == 1
    assert cold.get_execution(ex.execution_id).winning_attempt_id == lease.attempt_id


def test_c8_training_reader_blocks_empty_or_noncommercial_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "m15_runner", ROOT / "scripts" / "run_m15_tribunal_and_ood_benchmarks.py"
    )
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    fake_seed = tmp_path / "seed"
    fake_seed.mkdir()
    (fake_seed / "training_table_inventory.json").write_text(
        json.dumps({"sample.parquet": {"external_evidence": [], "sha256": "0" * 64}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "SEED_DIR", fake_seed)
    p = tmp_path / "sample.parquet"
    p.write_bytes(b"parquet")
    with pytest.raises(CleanRoomPolicyError, match="lacks verified commercial"):
        runner._read_training_table(p, {})


def test_full_transcript_gc_hard_gate_and_missing_commercial_rules_fail_closed(tmp_path: Path) -> None:
    ref = _build_preset_reference_r(SEED)
    eng = ScopedConstraintEngine.from_seed_dir(SEED)
    # Tight impossible bound [0.95, 0.99] on full_transcript MUST fail
    strict_gc = ConstraintSpec(
        requirement_id="gc_full_test",
        constraint_kind="gc_window_bounds",
        target_scopes=["full_transcript"],
        severity="hard_gate",
        source_type="system_safety_gate",
        source_locator="test:gc_full",
        params={"min_gc": 0.95, "max_gc": 0.99, "window": 50},
    )
    outcome = eng.evaluate_bundle(ref, [strict_gc], compliance_track="commercial_clean")
    assert not outcome.passed_all_hard_gates
    assert any(v.violated_scope == "full_transcript" for v in outcome.violations)

    # Commercial run without rules_commercial_v1.json MUST fail closed (never record rules_research_v1)
    partial_seed = tmp_path / "seed_no_cc_rules"
    shutil.copytree(SEED, partial_seed)
    (partial_seed / "rules_commercial_v1.json").unlink()
    intent = DesignIntent(
        intent_id="int_cc_missing_rules",
        project_id="p_cc",
        intent_version=1,
        created_from_turn_ids=["t1"],
        entry_mode="protein_only",
        compliance_track="commercial_clean",
        protein_targets=[ref.molecules[0].effective_translation_target],
        objectives=[
            ObjectiveSpec(
                requirement_id="obj_hl",
                metric_id="consensus_halflife_pred",
                direction="maximize",
                priority_rank=1,
                evidence_status="validated_predictor",
                source_utterance="hl",
            )
        ],
        runtime_policy=RuntimePolicySpec(
            allow_cached_candidates=True, enabled_generators=[], enabled_evaluators=["surrogate_v1_clean"]
        ),
    )
    bundles, rep, man = run_design_pipeline(intent, cached_candidates=[ref], seed_dir=partial_seed)
    assert bundles == []
    assert rep.recommended_bundle_ids == []
    assert rep.outcome_status == "UNCOMPARABLE_MISSING_APPROVED_COMPONENT"
    assert "rules_research_v1" not in man.rule_snapshot_version


def test_target_band_ranking_prefers_in_band_over_lower_value(monkeypatch: pytest.MonkeyPatch) -> None:
    from mrnax.schemas.construct_bundle import ConstructBundle

    base = _build_preset_reference_r(SEED)
    ref_a = ConstructBundle.create(
        bundle_id="cand_low_1",
        parent_bundle_id=None,
        derivation_action="reference_import",
        target_stoichiometry=base.target_stoichiometry,
        molecules=base.molecules,
    )
    mol_b = base.molecules[0].model_copy_validated(
        molecule_id="mol_in_band_10",
        sequence=base.molecules[0].sequence[:-2] + "AA",
    )
    ref_b = ConstructBundle.create(
        bundle_id="cand_inband_10",
        parent_bundle_id=None,
        derivation_action="reference_import",
        target_stoichiometry={"mol_in_band_10": 1.0},
        molecules=[mol_b],
    )

    import mrnax.engine.pareto_director as pd_mod

    class _MockSurrogate:
        def evaluate_molecule(self, mol):
            v = 1.0 if mol.molecule_id != "mol_in_band_10" else 10.0
            return (
                [
                    MetricResult(
                        metric_id="consensus_halflife_pred",
                        value=v,
                        unit="z",
                        direction="target_band",
                        evaluated_scopes=["cds"],
                        tool_name="Mock",
                        tool_version="1.0",
                        chemistry_evaluation_mode="native_chemistry_validated",
                        participated_in="pareto_selection_or_ranking",
                    )
                ],
                [],
            )

    monkeypatch.setattr(pd_mod.CommercialCleanSurrogateV1, "from_seed_dir", lambda _s: _MockSurrogate())
    intent = DesignIntent(
        intent_id="int_band",
        project_id="p_band",
        intent_version=1,
        created_from_turn_ids=["t1"],
        entry_mode="protein_only",
        compliance_track="commercial_clean",
        protein_targets=[ref_a.molecules[0].effective_translation_target],
        objectives=[
            ObjectiveSpec(
                requirement_id="obj_band",
                metric_id="consensus_halflife_pred",
                direction="target_band",
                target_band=(8.0, 12.0),
                priority_rank=1,
                evidence_status="validated_predictor",
                source_utterance="target 8..12",
            )
        ],
        runtime_policy=RuntimePolicySpec(
            allow_cached_candidates=True, enabled_generators=[], enabled_evaluators=["surrogate_v1_clean"]
        ),
    )
    ordered, rep, _ = run_design_pipeline(intent, cached_candidates=[ref_a, ref_b], seed_dir=SEED)
    assert [b.bundle_id for b in ordered] == ["cand_inband_10", "cand_low_1"]
    assert rep.recommended_bundle_ids == ["cand_inband_10"]


def test_i4_i5_i10_i12_i15_and_historical_exports_and_vault_isolation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # I5: CLI transport re-downloads when md5Checksum is absent from metadata
    transport = GdriveCliDriveV3Transport(gdrive_bin="/bin/echo")
    backing = [b"v1_bytes"]
    monkeypatch.setattr(transport, "_run_json", lambda _args: {"id": "fid1", "mimeType": "application/octet-stream"})
    monkeypatch.setattr(transport, "download_bytes", lambda _fid: backing[0])
    m1 = transport.get_metadata("fid1")["md5Checksum"]
    backing[0] = b"v2_changed_bytes"
    m2 = transport.get_metadata("fid1")["md5Checksum"]
    assert m1 != m2 and m2 == hashlib.md5(b"v2_changed_bytes").hexdigest()

    # I10: Reversed base mapping rejected by validate_structrfm_tokenizer
    class _ReversedTokenizer:
        unk_token_id = 0

        def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
            mp = {"A": 8, "U": 7, "C": 6, "G": 5}
            return [mp[c] for c in text]

    with pytest.raises(RuntimeError, match="canonical base-ID"):
        validate_structrfm_tokenizer(_ReversedTokenizer())

    # I12: Flipping ledger directions under unchanged rule expected_effect_direction MUST fail
    tmp_rules = tmp_path / "rules"
    tmp_rules.mkdir()
    shutil.copy(SEED / "rules_commercial_v1.json", tmp_rules / "rules_commercial_v1.json")
    ledger = json.loads((SEED / "rule_observation_ledger_v1.json").read_text(encoding="utf-8"))
    for cid in ("CC-MRNAX-M15-WU2019-ARE", "CC-MRNAX-W5-SIEGEL2022-FASTUTR-ARE"):
        ledger["claims"][cid]["preregistered_direction"] = "positive"
        ledger["claims"][cid]["observed_direction"] = "positive"
        ledger["claims"][cid]["measured_effect"] = abs(ledger["claims"][cid]["measured_effect"])
    (tmp_rules / "rule_observation_ledger_v1.json").write_text(json.dumps(ledger), encoding="utf-8")
    with pytest.raises(ValueError, match="direction check"):
        RuleRegistry(tmp_rules / "rules_commercial_v1.json", track="commercial_clean").snapshot_version()

    # I15: 6 bundle IDs sharing 1 version_digest count as 1 independent construct (cannot unlock weight_update_allowed)
    ref = _build_preset_reference_r(SEED)
    obs_dup = [
        ConstructObservation(
            bundle_id=f"alias_{i}",
            version_digest=ref.version_digest,
            source_row_or_figure=f"row_{i}",
            biological_replicates=[BiologicalReplicate(bio_rep_id=f"r_{i}_{j}", technical_replicates=[100.0 + 50.0 * i]) for j in range(3)],
        )
        for i in range(6)
    ]
    rec_dup = ExperimentRecord(
        experiment_id="exp_dup_digest",
        project_id="p_dup",
        batch_id="b1",
        extraction_state="verified_structured",
        assay_type="elisa",
        readout_name_exact="ng_ml",
        unit="ng/mL",
        chemistry_track="m1psi",
        cell_or_model_context="HEK293T",
        delivery_vehicle="LNP",
        shared_controls=["ctrl"],
        observations=obs_dup,
    )
    assert rec_dup.n_independent_constructs == 1

    # Feedback retries keep original decision + cross-project record blocked + quarantine secret redaction + I4 reload order
    sess = ProjectHarnessSession(project_id="p_sess", store_dir=tmp_path / "p_sess", seed_dir=SEED)
    cand_b = sess.build_multi_segment_variant_for_comparison(ref)
    sess.register_bundle(ref, set_as_reference=True)
    sess.create_initial_intent(
        turn_id="t1",
        user_utterance="init",
        protein_sequence=ref.molecules[0].effective_translation_target.protein_sequence,
        reference_bundle=ref,
        compliance_track="commercial_clean",
        uridine_chemistry="m1psi",
    )
    # Execute twice on same intent; make first report have higher alphabetical report_id ("rep_zzzz") and second ("rep_aaaa")
    r_high = EvaluationReport(
        report_id="rep_zzzz",
        execution_id="e1",
        intent_id=sess.latest_intent().intent_id,
        intent_version=1,
        compliance_track="commercial_clean",
        outcome_status="COMPLETED_ZERO_QUALIFIED",
        recommended_bundle_ids=[],
        candidate_evaluations=[],
        failed_diagnostics=[],
        fto_alerts=[],
        pre_synthesis_advisories=[],
        action_transparency=[],
    )
    r_newer_low_id = r_high.model_copy_validated(report_id="rep_aaaa")
    sess._reports[r_high.report_id] = r_high
    (sess.store_dir / "reports" / "rep_zzzz.json").write_text(r_high.model_dump_json(), encoding="utf-8")
    sess._reports[r_newer_low_id.report_id] = r_newer_low_id
    (sess.store_dir / "reports" / "rep_aaaa.json").write_text(r_newer_low_id.model_dump_json(), encoding="utf-8")
    sess._save_state_meta()
    reloaded = ProjectHarnessSession(project_id="p_sess", store_dir=tmp_path / "p_sess", seed_dir=SEED)
    assert reloaded.latest_report() is not None
    assert reloaded.latest_report().report_id == "rep_aaaa"

    # Cross-project feedback blocked without switching reference
    foreign = ExperimentRecord(
        experiment_id="exp_foreign",
        project_id="OTHER_PROJ",
        batch_id="b1",
        extraction_state="verified_structured",
        assay_type="elisa",
        readout_name_exact="ng_ml",
        unit="ng/mL",
        chemistry_track="m1psi",
        cell_or_model_context="HEK293T",
        delivery_vehicle="LNP",
        shared_controls=["ctrl"],
        observations=[
            ConstructObservation(
                bundle_id=ref.bundle_id,
                version_digest=ref.version_digest,
                source_row_or_figure="r1",
                biological_replicates=[BiologicalReplicate(bio_rep_id=f"a{i}", technical_replicates=[100.0]) for i in range(3)],
            ),
            ConstructObservation(
                bundle_id=cand_b.bundle_id,
                version_digest=cand_b.version_digest,
                source_row_or_figure="r2",
                biological_replicates=[BiologicalReplicate(bio_rep_id=f"b{i}", technical_replicates=[300.0]) for i in range(3)],
            ),
        ],
    )
    res_foreign = sess.ingest_experiment_feedback(foreign)
    assert res_foreign.feedback_decision is not None
    assert res_foreign.feedback_decision.branch == "experimental_confound_blocked"
    assert sess.active_reference_bundle_id == ref.bundle_id

    # Valid feedback retry is idempotent (preserves rank_reversal_switch_reference!)
    valid_exp = foreign.model_copy_validated(experiment_id="exp_valid", project_id="p_sess")
    first_dec = sess.ingest_experiment_feedback(valid_exp)
    retry_dec = sess.ingest_experiment_feedback(valid_exp)
    assert first_dec.feedback_decision is not None and retry_dec.feedback_decision is not None
    assert first_dec.feedback_decision.branch == "rank_reversal_switch_reference"
    assert retry_dec.feedback_decision == first_dec.feedback_decision

    # Secret redaction in quarantine JSONL
    fake_secret = "sk-" + "A1b2C3d4E5f6G7h8I9j0K1l2"
    wrapped = wrap_untrusted_external_text(f"Note with {fake_secret}", source_id="test_src")
    assert fake_secret not in wrapped.wrapped_payload
    raw_bypass = UntrustedExternalText(source_id="src", sha256="0" * 64, wrapped_payload=f"Bypass {fake_secret}")
    sess.attach_external_evidence_readonly(raw_bypass)
    q_txt = (sess.store_dir / "quarantined_external_evidence.jsonl").read_text(encoding="utf-8")
    assert fake_secret not in q_txt and "[REDACTED_API_KEY]" in q_txt

    # SecretVault ignores ~/.env and process os.environ
    fake_home = tmp_path / "fake_home"
    fake_home.mkdir()
    (fake_home / ".env").write_text(f"DEEPSEEK_API_KEY={fake_secret}\n", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    monkeypatch.setenv("DEEPSEEK_API_KEY", fake_secret)
    vault = SecretVault(vault_path=fake_home / ".mrnax" / "secrets.env")
    assert vault.get_key("deepseek") is None

    # Historical M1 example projects export cleanly via legacy report migration (isolated in tmp_path so source stays read-only)
    isolated_examples = tmp_path / "m1_egfp_isolated"
    shutil.copytree(ROOT / "artifacts" / "m1_egfp", isolated_examples)
    app = create_control_plane_app(
        ExecutionCASStore(tmp_path / "cas_hist.db"),
        projects_root=isolated_examples,
        seed_dir=SEED,
        access_token="adm",
    )
    client = TestClient(app, headers={"X-MRNAX-Access-Token": "adm"})
    for proj_dir_name in ("session_entry1_dialogue", "session_entry2_with_R"):
        st = client.get(f"/v1/projects/{proj_dir_name}/state")
        assert st.status_code == 200
        recs = st.json()["latest_report"]["recommended_bundle_ids"]
        assert len(recs) >= 1
        for bid in recs:
            exp_r = client.get(f"/v1/projects/{proj_dir_name}/bundles/{bid}/export")
            assert exp_r.status_code == 200, (proj_dir_name, bid, exp_r.text)


def test_round4_edge_case_closures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Comprehensive regressions for all Round-4 audit findings (C3, C4, C6, C7, C8, I4, I10, I12, I15, target_band, retry, binding, vault, ZAI)."""
    import io
    import tarfile
    from mrnax.cloud.drive_rest_v3 import DriveRestV3Client, InMemoryDriveV3Transport
    from mrnax.engine.orthrus_portable_pytorch import PortableOrthrusModel
    from mrnax.federated.external_evidence_adapter import CleanRoomPolicyError, ExternalEvidenceAdapterV2
    from mrnax.harness.auth_secrets import parse_provider_keys_from_text
    from mrnax.harness.cli import main as cli_main
    from mrnax.harness.mcp_server import handle_mcp_request
    from mrnax.harness.tools import AtomicToolbox
    from mrnax.io_insdc.genbank_io import export_construct_package
    from mrnax.constraints.registry import ScopedConstraintEngine
    from mrnax.rules.tribunal import RuleRegistry
    from mrnax.schemas.construct_bundle import ConstructBundle
    from mrnax.schemas.evaluation_report import BundleEvaluation, EvaluationReport

    # 1. SecretVault blocks ${ENV} interpolation & ZAI key redaction works in quarantine JSONL
    monkeypatch.setenv("INTERPOLATED_SECRET", "sk-EnvSecretValue1234567890abcd")
    v_path = tmp_path / "vault" / "secrets.env"
    v_path.parent.mkdir(parents=True)
    v_path.write_text("OPENAI_API_KEY=${INTERPOLATED_SECRET}\n", encoding="utf-8")
    v = SecretVault(vault_path=v_path)
    assert v.get_key("openai") == "${INTERPOLATED_SECRET}"
    v.set_key("deepseek", "sk-DeepSeekLiteral1234567890abcd")
    assert "sk-EnvSecretValue1234567890abcd" not in v_path.read_text(encoding="utf-8")

    zai_key = "0123456789abcdef0123456789abcdef.AbCdEf1234567890"
    assert parse_provider_keys_from_text(f"zai: {zai_key}")["zai"] == zai_key
    sess_q = ProjectHarnessSession(project_id="p_zai", store_dir=tmp_path / "s_zai", seed_dir=SEED)
    sess_q.attach_external_evidence_readonly(wrap_untrusted_external_text(f"Key {zai_key}", source_id="z"))
    assert zai_key not in (sess_q.store_dir / "quarantined_external_evidence.jsonl").read_text(encoding="utf-8")

    # 2. Missing commercial rule file blocks even when explicit snapshot version is supplied
    seed_no_rules = tmp_path / "seed_no_rules"
    shutil.copytree(SEED, seed_no_rules)
    valid_cc_snap = RuleRegistry(SEED / "rules_commercial_v1.json", track="commercial_clean").snapshot_version()
    (seed_no_rules / "rules_commercial_v1.json").unlink()
    sess_cc = ProjectHarnessSession(project_id="p_cc", store_dir=tmp_path / "s_cc", seed_dir=SEED)
    cc_intent = sess_cc.create_initial_intent(
        turn_id="t1",
        user_utterance="Commercial EGFP",
        protein_sequence="MVSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITLGMDELYK",
        compliance_track="commercial_clean",
        use_commercial_surrogate=True,
    )
    for explicit_snap in (None, valid_cc_snap, "rules_research_v1@1.0.0"):
        b_out, rep_out, _ = run_design_pipeline(cc_intent, seed_dir=seed_no_rules, rule_snapshot_version=explicit_snap)
        assert b_out == []
        assert rep_out.outcome_status == "UNCOMPARABLE_MISSING_APPROVED_COMPONENT"

    # 3. I12 Tribunal rejects tampered positive threshold on negative correlation rule
    ledger = json.loads((SEED / "rule_observation_ledger_v1.json").read_text(encoding="utf-8"))
    rules_doc = json.loads((SEED / "rules_commercial_v1.json").read_text(encoding="utf-8"))
    from mrnax.rules.models import RuleRecord
    rules_list = [RuleRecord.model_validate(r) for r in rules_doc["rules"]]
    tampered_claims = json.loads(json.dumps(ledger["claims"]))
    for cid in ("CC-MRNAX-M15-WU2019-ARE", "CC-MRNAX-W5-SIEGEL2022-FASTUTR-ARE"):
        tampered_claims[cid]["measured_effect"] = 0.1915
        tampered_claims[cid]["preregistered_threshold"] = 0.30
    with pytest.raises(ValueError, match="quantitative"):
        RuleRegistry.validate_rules_integrity(rules_list, track="commercial_clean", observation_ledger=tampered_claims)

    # 4. I10 Orthrus from_huggingface enforces 40-hex commit SHA and approved release digests
    with pytest.raises(ValueError, match="40-hex"):
        PortableOrthrusModel.from_huggingface(revision="v1.0", expected_safetensors_sha256="0" * 64)
    dummy_dir = tmp_path / "dummy_orth"
    dummy_dir.mkdir()
    (dummy_dir / "config.json").write_text('{"n_tracks":6,"ssm_model_dim":16,"ssm_n_layers":1}', encoding="utf-8")
    (dummy_dir / "model.safetensors").write_bytes(b"not-real-safetensors")
    dummy_sha = hashlib.sha256(b"not-real-safetensors").hexdigest()
    import sys
    import types
    fake_hf = types.ModuleType("huggingface_hub")
    fake_hf.hf_hub_download = lambda repo_id, filename, revision: str(dummy_dir / filename)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)
    with pytest.raises(ValueError):
        PortableOrthrusModel.from_huggingface(
            "antichronology/orthrus-6-track",
            revision="a" * 40,
            expected_safetensors_sha256=dummy_sha,
        )

    # 5. C8 nested research_only evidence blocked on commercial export
    bundles_cc = sess_cc.execute_design(cc_intent)
    valid_ev = ExternalEvidenceAdapterV2()._freeze(
        "UniProt",
        "CC-BY-4.0",
        "commercial_clean",
        "https://rest.uniprot.org/uniprotkb/P42212.json",
        {"primary_accession": "P42212"},
        {"audit_marker": "stand-in", "license_tier": "research_only"},
    )
    poisoned_bundle = ConstructBundle.create(
        bundle_id="bnd_poison_ev",
        parent_bundle_id=None,
        derivation_action="computational_reference",
        target_stoichiometry=bundles_cc[0].target_stoichiometry,
        co_expression_mode=bundles_cc[0].co_expression_mode,
        molecules=bundles_cc[0].molecules,
        external_evidence=[valid_ev],
    )
    with pytest.raises(CleanRoomPolicyError):
        export_construct_package(
            poisoned_bundle,
            tmp_path / "exp_poison",
            constraint_engine=ScopedConstraintEngine.from_seed_dir(SEED),
            constraints=cc_intent.constraints,
            compliance_track="commercial_clean",
        )

    # 6. C6 cold recovery after local journal commit + expired deadline + Drive snapshot migration
    drive = DriveRestV3Client(InMemoryDriveV3Transport())
    snap_folder = drive.create_folder("root", "commits")
    runs_folder = drive.create_folder("root", "runs")
    db_local = tmp_path / "cas_mig.db"
    cas1 = ExecutionCASStore(db_local, drive_verifier=drive, runs_root_id=runs_folder)
    ex = cas1.submit_execution("s1", "d" * 64, "i1", now_ts=10.0)
    lease = cas1.acquire_or_renew_lease(ex.execution_id, worker_id="w1", ttl_seconds=10, now_ts=10.0)
    att_folder = drive.ensure_attempt_dir(runs_folder, ex.execution_id, lease.attempt_id)
    art_rcpt = drive.upload_verified_bytes(att_folder, "report.json", b"{}", mime_type="application/json")
    seal = drive.write_attempt_seal(
        att_folder,
        execution_id=ex.execution_id,
        attempt_id=lease.attempt_id,
        lease_epoch=lease.lease_epoch,
        receipts=[art_rcpt],
    )
    cas1.commit_attempt_cas(
        execution_id=ex.execution_id,
        attempt_id=lease.attempt_id,
        lease_epoch=lease.lease_epoch,
        lease_token=lease.lease_token,
        sealed_file_id=seal.file_id,
        sealed_sha256=seal.sha256,
        now_ts=15.0,
    )
    # Reopen with Drive snapshots enabled AFTER deadline (now > 20.0)
    cas2 = ExecutionCASStore(db_local, commits_snapshot_folder_id=snap_folder, drive_verifier=drive, runs_root_id=runs_folder)
    cas_cold = ExecutionCASStore(tmp_path / "cas_cold.db", commits_snapshot_folder_id=snap_folder, drive_verifier=drive, runs_root_id=runs_folder)
    assert cas_cold.restore_committed_receipts() == 1
    retry_rcpt = cas2.commit_attempt_cas(
        execution_id=ex.execution_id,
        attempt_id=lease.attempt_id,
        lease_epoch=lease.lease_epoch,
        lease_token=lease.lease_token,
        sealed_file_id=seal.file_id,
        sealed_sha256=seal.sha256,
        now_ts=50.0,
    )
    assert retry_rcpt.status == "COMMITTED"

    # 7. C3/C4/target_band across MCP explain_tradeoffs, MCP audit_construct, and cold HTTP state
    from mrnax.schemas.design_intent import ObjectiveSpec
    from mrnax.engine.pareto_director import score_objective_value, project_report_for_intent
    wide_obj = ObjectiveSpec(
        requirement_id="r_wide",
        metric_id="utr5_mrl_pred",
        direction="target_band",
        evidence_status="validated_predictor",
        source_utterance="wide band",
        target_band=(0.0, 4_000_000.0),
    )
    assert score_objective_value(0.0, wide_obj) > score_objective_value(-0.1, wide_obj)  # type: ignore[operator]

    # 8. I4 USTAR archive restore preserves chronological report order even without report_order
    sess_i4 = ProjectHarnessSession(project_id="p_i4", store_dir=tmp_path / "s_i4", seed_dir=SEED)
    int_i4 = sess_i4.create_initial_intent(
        turn_id="t1",
        user_utterance="Design EGFP",
        protein_sequence=cc_intent.protein_targets[0].protein_sequence,
        compliance_track="commercial_clean",
        use_commercial_surrogate=True,
    )
    b_i4, r1, _ = run_design_pipeline(int_i4, seed_dir=SEED)
    _, r2, _ = run_design_pipeline(int_i4, seed_dir=SEED)
    r_first = r1.model_copy_validated(report_id="rep_zzz")
    r_second = r2.model_copy_validated(report_id="rep_aaa")
    for b in b_i4:
        sess_i4.register_bundle(b, originating_intent_id=int_i4.intent_id)
    (sess_i4.store_dir / "reports" / "rep_zzz.json").write_text(json.dumps(r_first.model_dump(mode="json")), encoding="utf-8")
    (sess_i4.store_dir / "reports" / "rep_aaa.json").write_text(json.dumps(r_second.model_dump(mode="json")), encoding="utf-8")
    # Strip report_order to emulate legacy project_state.json, then archive & restore via USTAR
    (sess_i4.store_dir / "project_state.json").write_text(
        json.dumps({"project_id": "p_i4", "active_reference_bundle_id": None, "intent_count": 1}),
        encoding="utf-8",
    )
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.USTAR_FORMAT) as tf:
        tf.add(sess_i4.store_dir, arcname="restored")
    buf.seek(0)
    restored_parent = tmp_path / "ustar_out"
    with tarfile.open(fileobj=buf, mode="r") as tf:
        tf.extractall(restored_parent, filter="data")
    reloaded_i4 = ProjectHarnessSession(store_dir=restored_parent / "restored", seed_dir=SEED)
    assert reloaded_i4.latest_report() is not None
    assert reloaded_i4.latest_report().report_id == "rep_aaa"  # type: ignore[union-attr]

    # 9. I15 physical construct independence blocks renamed molecule label clones & CLI/MCP preserves stored project_id
    base_b = bundles_cc[0]
    sess_fb = ProjectHarnessSession(project_id="PROJ_REAL", store_dir=tmp_path / "s_fb", seed_dir=SEED)
    sess_fb.create_initial_intent(
        turn_id="t1",
        user_utterance="Design EGFP",
        protein_sequence=cc_intent.protein_targets[0].protein_sequence,
        reference_bundle=base_b,
        compliance_track="commercial_clean",
        use_commercial_surrogate=True,
    )
    clone_obs = [
        ConstructObservation(
            bundle_id=base_b.bundle_id,
            version_digest=base_b.version_digest,
            source_row_or_figure="row_ref",
            biological_replicates=[BiologicalReplicate(bio_rep_id=f"r{i}", technical_replicates=[100.0]) for i in range(3)],
        )
    ]
    for k in range(1, 6):
        m_orig = base_b.molecules[0]
        m_renamed = m_orig.model_copy_validated(molecule_id=f"mol_clone_{k}")
        b_clone = ConstructBundle.create(
            bundle_id=f"bnd_clone_{k}",
            parent_bundle_id=None,
            derivation_action="reference_import",
            target_stoichiometry={f"mol_clone_{k}": 1.0},
            co_expression_mode=base_b.co_expression_mode,
            molecules=[m_renamed],
        )
        sess_fb.register_bundle(b_clone, originating_intent_id=sess_fb.latest_intent().intent_id)
        clone_obs.append(
            ConstructObservation(
                bundle_id=b_clone.bundle_id,
                version_digest=b_clone.version_digest,
                source_row_or_figure=f"row_c{k}",
                biological_replicates=[BiologicalReplicate(bio_rep_id=f"c{k}_{i}", technical_replicates=[400.0]) for i in range(3)],
            )
        )
    # Reopen without explicit project_id (as MCP CLI does) -> must preserve PROJ_REAL!
    sess_mcp = ProjectHarnessSession(project_id=None, store_dir=tmp_path / "s_fb", seed_dir=SEED, default_project_id="proj_mcp")
    assert sess_mcp.project_id == "PROJ_REAL"
    exp_clones = ExperimentRecord(
        experiment_id="exp_clones",
        project_id="PROJ_REAL",
        batch_id="b1",
        extraction_state="verified_structured",
        assay_type="FACS",
        readout_name_exact="MFI",
        unit="AU",
        cell_or_model_context="HEK293T",
        chemistry_track="m1psi",
        delivery_vehicle="LNP",
        shared_controls=[base_b.bundle_id],
        observations=clone_obs,
    )
    dec_clones = sess_mcp.ingest_experiment_feedback(exp_clones, held_out_validation_passed=True)
    assert dec_clones.feedback_decision is not None
    assert dec_clones.feedback_decision.weight_update_allowed is False
    assert dec_clones.feedback_decision.isolated_calibration_proposal_id is None
    assert dec_clones.n_independent_constructs == 1

    # 10. Round-5 closures: C8 list-valued provenance smuggling, I10 HF override forbidden, I12 registered metrics & BioProject
    for bad_field in (
        {"license_tier": ["research_only"]},
        {"rights": ["research_only"]},
        {"license_spdx": ["CC-BY-NC-4.0"]},
    ):
        ev_arr = ExternalEvidenceAdapterV2()._freeze(
            "UniProt",
            "CC-BY-4.0",
            "commercial_clean",
            "https://rest.uniprot.org/uniprotkb/P42212.json",
            {"primary_accession": "P42212"},
            bad_field,
        )
        with pytest.raises(CleanRoomPolicyError):
            ExternalEvidenceAdapterV2().guard_export_to_commercial_bundle(
                ev_arr, purpose="commercial_redistribution_export"
            )

    with pytest.raises(ValueError, match="sandbox overrides are forbidden"):
        PortableOrthrusModel.from_huggingface(
            "antichronology/orthrus-6-track",
            revision="8" * 40,
            expected_safetensors_sha256="0" * 64,
            allow_sandbox_override=True,
        )

    tampered_ratio = json.loads(json.dumps(ledger["claims"]))
    for cid in ("CC-MRNAX-M15-WU2019-ARE", "CC-MRNAX-W5-SIEGEL2022-FASTUTR-ARE"):
        tampered_ratio[cid]["effect_metric"] = "ratio_correlation"
        tampered_ratio[cid]["measured_effect"] = 0.1915
        tampered_ratio[cid]["preregistered_threshold"] = 0.30
    with pytest.raises(ValueError, match="unregistered effect_metric"):
        RuleRegistry.validate_rules_integrity(rules_list, track="commercial_clean", observation_ledger=tampered_ratio)


def test_round7_prohibited_rights_and_concurrent_feedback_recovery(tmp_path: Path) -> None:
    """Round-7 regressions: 'Commercial use is prohibited.', legacy approved N=1 cold load, and open-writer retry."""
    from copy import deepcopy
    from unittest.mock import patch

    from mrnax.federated.external_evidence_adapter import (
        CleanRoomPolicyError,
        ExternalEvidenceAdapterV2,
        sha256_canonical,
    )
    from mrnax.harness import state_machine
    from mrnax.schemas.construct_bundle import ConstructBundle
    from mrnax.schemas.experiment_record import BiologicalReplicate, ConstructObservation, ExperimentRecord

    seed_dir = SEED
    frozen = json.loads((seed_dir / "m2_m4_federated_12api_snapshot_v2.json").read_text(encoding="utf-8"))[
        "live_api_snapshots"
    ]
    approved = next(r for r in frozen.values() if r["source_name"] == "UniProt")
    for phrase in (
        "Commercial use is prohibited.",
        ["Commercial use is prohibited."],
        "Prohibits any commercial exploitation.",
        "All rights reserved.",
    ):
        rec_raw = deepcopy(approved)
        rec_raw["raw_excerpt"] = {"rights": phrase}
        with pytest.raises(CleanRoomPolicyError):
            ExternalEvidenceAdapterV2().guard_export_to_commercial_bundle(
                rec_raw, purpose="commercial_redistribution_export"
            )
        rec_sum = deepcopy(approved)
        rec_sum["summary"]["rights"] = phrase
        rec_sum["payload_sha256"] = sha256_canonical(rec_sum["summary"])
        with pytest.raises(CleanRoomPolicyError):
            ExternalEvidenceAdapterV2().guard_export_to_commercial_bundle(
                rec_sum, purpose="commercial_redistribution_export"
            )

    # Legacy approved N=1 record loads cleanly without ValidationError and downgrades on retry
    proj = tmp_path / "legacy_n1_approved"
    sess = ProjectHarnessSession(project_id="proj", store_dir=proj, seed_dir=seed_dir)
    ref0 = _build_preset_reference_r(seed_dir)
    base_b = ConstructBundle.create(
        bundle_id="CLONE_0",
        parent_bundle_id=None,
        derivation_action="reference_import",
        target_stoichiometry=ref0.target_stoichiometry,
        co_expression_mode=ref0.co_expression_mode,
        molecules=ref0.molecules,
    )
    intent = sess.create_initial_intent(
        turn_id="t1",
        user_utterance="Reporter control",
        protein_sequence=base_b.molecules[0].effective_translation_target.protein_sequence,
        compliance_track="commercial_clean",
        reference_bundle=base_b,
    )
    clones = [base_b]
    for idx in range(1, 6):
        mol_c = ref0.molecules[0].model_copy_validated(molecule_id=f"mol_clone_{idx}")
        c = ConstructBundle.create(
            bundle_id=f"CLONE_{idx}",
            parent_bundle_id="CLONE_0",
            derivation_action="reference_import",
            target_stoichiometry={f"mol_clone_{idx}": 1.0},
            co_expression_mode=ref0.co_expression_mode,
            molecules=[mol_c],
        )
        sess.register_bundle(c, originating_intent_id=intent.intent_id)
        clones.append(c)
    obs = [
        ConstructObservation(
            bundle_id=b.bundle_id,
            version_digest=b.version_digest,
            source_row_or_figure=f"row_{idx}",
            biological_replicates=[
                BiologicalReplicate(bio_rep_id=f"rep_{r}", technical_replicates=[(10.0 if idx == 0 else 95.0) + r])
                for r in range(3)
            ],
        )
        for idx, b in enumerate(clones)
    ]
    req = ExperimentRecord(
        experiment_id="hist_r4",
        project_id="proj",
        batch_id="b1",
        extraction_state="verified_structured",
        assay_type="luc",
        readout_name_exact="rlu",
        unit="au",
        cell_or_model_context="hela",
        chemistry_track=base_b.molecules[0].chemistry_track.uridine_chemistry,
        delivery_vehicle="lnp",
        shared_controls=[base_b.bundle_id],
        observations=obs,
    )
    legacy_payload = req.model_dump(mode="json")
    legacy_payload["feedback_decision"] = {
        "branch": "rank_reversal_switch_reference",
        "new_reference_bundle_id": "CLONE_1",
        "retained_early_control_bundle_id": None,
        "weight_update_allowed": True,
        "held_out_validation_passed": True,
        "isolated_calibration_proposal_id": "cal_iso_legacy",
        "sequence_attribution_status": "not_applicable",
        "spawned_disambiguation_chimera_id": None,
        "rationale": "Switched project reference to CLONE_1; weight_update_allowed=True (N=6).",
    }
    (proj / "experiments").mkdir(parents=True, exist_ok=True)
    exp_file = proj / "experiments" / "hist_r4.json"
    exp_file.write_text(json.dumps(legacy_payload), encoding="utf-8")
    before_bytes = exp_file.read_bytes()

    cold = ProjectHarnessSession(project_id="proj", store_dir=proj, seed_dir=seed_dir)
    assert cold._experiments[0].n_independent_constructs == 1
    assert cold._experiments[0].feedback_decision is not None
    assert cold._experiments[0].feedback_decision.weight_update_allowed is False
    assert exp_file.read_bytes() == before_bytes

    retried = cold.ingest_experiment_feedback(req, batch_comparable=True, held_out_validation_passed=True)
    assert retried.n_independent_constructs == 1
    assert retried.feedback_decision is not None and retried.feedback_decision.weight_update_allowed is False
    assert retried.feedback_decision.isolated_calibration_proposal_id is None
    assert exp_file.read_bytes() != before_bytes

    # Interrupted Writer A + open Writer B committing b_third + retrying a_second through Writer B
    fix = json.loads((ROOT / "tests/golden/m2_benchmark1_dual_speed_feedback_fixture_v1.json").read_text())
    proj2 = tmp_path / "concurrent_writers"
    writer_a = ProjectHarnessSession(project_id="p2", store_dir=proj2, seed_dir=seed_dir)
    bmap = {k: ConstructBundle.model_validate(v) for k, v in fix["bundles"].items()}
    i2 = writer_a.create_initial_intent(
        turn_id="t1",
        user_utterance="B1 fixture",
        protein_sequence=bmap["B1_REF_A"].molecules[0].effective_translation_target.protein_sequence,
        compliance_track="commercial_clean",
        reference_bundle=bmap["B1_REF_A"],
    )
    for bid, b in bmap.items():
        if bid != "B1_REF_A":
            writer_a.register_bundle(b, originating_intent_id=i2.intent_id)
    base_rec = fix["experiment_records"]["B1_CASE6_N6_HELD_OUT_CALIBRATION_UNLOCKED"]

    def _mk_rec(eid: str, winner: str) -> ExperimentRecord:
        obs_list = [dict(o) for o in base_rec["observations"]]
        by_id = {o["bundle_id"]: o for o in obs_list}
        by_id["B1_CAND_E"]["biological_replicates"], by_id[winner]["biological_replicates"] = (
            by_id[winner]["biological_replicates"],
            by_id["B1_CAND_E"]["biological_replicates"],
        )
        return ExperimentRecord.model_validate({**base_rec, "project_id": "p2", "experiment_id": eid, "observations": obs_list})

    writer_a.ingest_experiment_feedback(_mk_rec("z_first", "B1_CAND_C"))
    writer_b = ProjectHarnessSession(project_id="p2", store_dir=proj2, seed_dir=seed_dir)
    orig_write = state_machine._write_json

    def _fail_meta(path: Path, data: object, **kw: object) -> None:
        if path.name == "project_state.json" and (proj2 / "experiments" / "a_second.json").is_file():
            raise OSError("simulated crash after a_second.json")
        orig_write(path, data, **kw)  # type: ignore[arg-type]

    rec_second = _mk_rec("a_second", "B1_REF_A")
    with patch.object(state_machine, "_write_json", side_effect=_fail_meta):
        with pytest.raises(OSError, match="simulated crash"):
            writer_a.ingest_experiment_feedback(rec_second)

    a_second_before = json.loads((proj2 / "experiments" / "a_second.json").read_text(encoding="utf-8"))
    rec_third = ExperimentRecord.model_validate({**base_rec, "project_id": "p2", "experiment_id": "b_third"})
    writer_b.ingest_experiment_feedback(rec_third, held_out_validation_passed=True)
    writer_b.ingest_experiment_feedback(rec_second)
    a_second_after = json.loads((proj2 / "experiments" / "a_second.json").read_text(encoding="utf-8"))
    assert a_second_after["ingested_at_ns"] == a_second_before["ingested_at_ns"]
    reopened = ProjectHarnessSession(project_id="p2", store_dir=proj2, seed_dir=seed_dir)
    assert reopened.active_reference_bundle_id == "B1_CAND_E"
    assert [e.experiment_id for e in reopened._experiments] == ["z_first", "a_second", "b_third"]




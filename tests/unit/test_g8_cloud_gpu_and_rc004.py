"""Gate G8 unit tests: Colab L4 GPU telemetry (Orthrus RC-003 + structRFM RC-002) & BSD-3-Clause EternaFold RC-004."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
import pytest

from mrnax.cloud.gateway import _build_preset_reference_r
from mrnax.engine.pareto_director import run_design_pipeline
from mrnax.harness.state_machine import ProjectHarnessSession
from mrnax.rules.models import (
    APPROVED_REPLACEMENT_RELEASES,
    load_replacement_card_yaml,
    verify_replacement_card_strict,
)
from mrnax.schemas.design_intent import ObjectiveSpec, RuntimePolicySpec

ROOT = Path(__file__).resolve().parents[2]
SEED = ROOT / "registry_seed"


def test_g8_cloud_gpu_verification_report_and_rc004_card_match_pinned_digests() -> None:
    report_path = ROOT / "artifacts" / "m2_cloud_gpu_verification" / "g8_cloud_gpu_and_eternafold_report.json"
    assert report_path.is_file()
    doc = json.loads(report_path.read_text(encoding="utf-8"))
    assert doc["gpu_device"] == "NVIDIA L4"

    # Orthrus-6-track (RC-003) L4 verification
    orth = doc["orthrus_rc003_verification"]
    assert orth["gpu_repeat_drift"] == 0.0
    assert orth["triton_vs_sequential_scan_max_diff"] < 1e-4
    assert orth["config_sha256"] == APPROVED_REPLACEMENT_RELEASES["orthrus_mamba_rna_foundation"].artifact_digests["config.json"]
    assert (
        orth["model_safetensors_sha256"]
        == APPROVED_REPLACEMENT_RELEASES["orthrus_mamba_rna_foundation"].artifact_digests["model.safetensors"]
    )
    assert len(orth["candidate_predictions"]) == 6

    # structRFM (RC-002) L4 verification with USE_SPACED=False and zero BPfold
    srfm = doc["structrfm_rc002_verification"]
    assert srfm["use_spaced_enforced_false"] is True
    assert srfm["zero_bpfold_dependency_verified"] is True
    assert srfm["gpu_repeat_drift"] == 0.0
    for fname, sha in APPROVED_REPLACEMENT_RELEASES["structrfm_rna_foundation"].artifact_digests.items():
        assert srfm["checkpoint_files_sha256"][fname] == sha

    # BSD-3-Clause EternaFold v1.3.1 (RC-004) verification & zero ViennaRNA linkage
    ef = doc["eternafold_v131_rc004_shootout"]
    assert ef["license_spdx"] == "BSD-3-Clause"
    assert ef["zero_viennarna_dynamic_linkage"] is True
    assert all("RNA" not in lib for lib in ef["ldd_libs"])
    assert ef["rmdb_eterna_r69_cc0_1m7"]["eternafold_v131_spearman"] > ef["rmdb_eterna_r69_cc0_1m7"]["viennarna_v280_spearman"]

    rc004 = load_replacement_card_yaml(ROOT / "docs" / "replacement_cards" / "RC-004-eternafold-v1.3.1.yaml")
    verify_replacement_card_strict(rc004, expected_component_id="eternafold_bpp_engine", expected_version="1.3.1")
    assert "6 mRNAx fixture constructs + 34 seeded random windows" in rc004.frozen_benchmark_dataset_and_protocol
    assert "-15 through +17" in rc004.frozen_benchmark_dataset_and_protocol

    seal_doc = json.loads(
        (ROOT / "artifacts" / "m2_cloud_gpu_verification" / "drive_seal_receipt.json").read_text(encoding="utf-8")
    )
    local_map = {
        "paired_ablation_report.json": ROOT / "artifacts" / "m2_generator_upgrade" / "paired_ablation_report.json",
        "g8_cloud_gpu_and_eternafold_report.json": report_path,
        "RC-004-eternafold-v1.3.1.yaml": ROOT / "docs" / "replacement_cards" / "RC-004-eternafold-v1.3.1.yaml",
    }
    for rec in seal_doc["artifact_receipts"]:
        raw = local_map[rec["name"]].read_bytes()
        assert rec["bytes"] == len(raw)
        assert rec["md5"] == hashlib.md5(raw).hexdigest()
        assert rec["sha256"] == hashlib.sha256(raw).hexdigest()


def test_eternafold_runtime_dispatch_fails_closed_when_unmounted_and_executes_when_mounted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ref = _build_preset_reference_r(SEED)
    sess = ProjectHarnessSession(project_id="g8_ef_rt", store_dir=tmp_path / "proj", seed_dir=SEED)
    base = sess.create_initial_intent(
        turn_id="t1",
        user_utterance="Evaluate cached reporter with EternaFold",
        protein_sequence=ref.molecules[0].effective_translation_target.protein_sequence,
        compliance_track="commercial_clean",
        uridine_chemistry="m1psi",
    )
    ef_only = base.model_copy_validated(
        objectives=[
            ObjectiveSpec(
                requirement_id="obj_ef_kozak",
                metric_id="eternafold_kozak_unpaired_prob",
                direction="maximize",
                priority_rank=1,
                evidence_status="validated_predictor",
                source_utterance="Maximize Kozak accessibility via EternaFold",
            )
        ],
        runtime_policy=RuntimePolicySpec(
            enabled_generators=[],
            enabled_evaluators=["eternafold_bpp_engine"],
            allow_cached_candidates=True,
        ),
    )
    # 1. Unmounted -> fail-closed UNCOMPARABLE_MISSING_APPROVED_COMPONENT (never silently skipped!)
    monkeypatch.delenv("MRNAX_ETERNAFOLD_BIN", raising=False)
    monkeypatch.delenv("MRNAX_ETERNAFOLD_PARAMS", raising=False)
    b_unm, rep_unm, man_unm = run_design_pipeline(ef_only, cached_candidates=[ref], seed_dir=SEED)
    assert b_unm == []
    assert rep_unm.outcome_status == "UNCOMPARABLE_MISSING_APPROVED_COMPONENT"
    assert "eternafold_bpp_engine" in (rep_unm.uncomparable_reason or "")
    assert man_unm.optimization_mode_audit.optimization_mode == "not_executed"

    # 2. Mounted stub contrafold + sandbox override -> executes, emits metric & ComponentInvocationRecord
    fake_bin = tmp_path / "contrafold"
    fake_bin.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, sys\n"
        "fa = pathlib.Path(sys.argv[2]).read_text().splitlines()[1].strip()\n"
        "post = pathlib.Path(sys.argv[-1])\n"
        "lines = [f'{i+1} {b} {i+2}:0.25' if i + 1 < len(fa) and i % 4 == 0 else f'{i+1} {b}' for i, b in enumerate(fa)]\n"
        "post.write_text('\\n'.join(lines) + '\\n')\n",
        encoding="utf-8",
    )
    fake_bin.chmod(fake_bin.stat().st_mode | stat.S_IXUSR)
    fake_params = tmp_path / "EternaFoldParams.v1"
    fake_params.write_bytes(b"sandbox-params")

    monkeypatch.setenv("MRNAX_ETERNAFOLD_BIN", str(fake_bin))
    monkeypatch.setenv("MRNAX_ETERNAFOLD_PARAMS", str(fake_params))

    # Without isolated_sandbox_override, mismatched params SHA-256 must fail closed
    b_bad_sha, rep_bad_sha, _ = run_design_pipeline(ef_only, cached_candidates=[ref], seed_dir=SEED)
    assert b_bad_sha == []
    assert rep_bad_sha.outcome_status == "UNCOMPARABLE_MISSING_APPROVED_COMPONENT"

    # With isolated_sandbox_override=True, executes cleanly and emits metric + evaluator invocation
    b_ok, rep_ok, man_ok = run_design_pipeline(
        ef_only, cached_candidates=[ref], seed_dir=SEED, isolated_sandbox_override=True
    )
    assert len(b_ok) == 1
    assert rep_ok.outcome_status == "COMPLETED_WITH_CANDIDATES"
    m_ids = [m.metric_id for m in rep_ok.candidate_evaluations[0].metrics]
    assert "eternafold_kozak_unpaired_prob" in m_ids
    eval_names = [c.name for c in man_ok.invoked_components if c.role == "evaluator"]
    assert any("EternaFold" in n for n in eval_names)

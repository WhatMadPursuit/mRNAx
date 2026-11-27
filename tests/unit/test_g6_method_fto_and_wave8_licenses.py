"""Gate G6 tests: Dual-FTO Method-Claim Guardrails (ADR-0003) & Wave 8 Three-Layer/Transitive License Traps."""

from __future__ import annotations

from pathlib import Path

import pytest

from mrnax.cloud.gateway import _build_preset_reference_r
from mrnax.engine.pareto_director import run_design_pipeline
from mrnax.engine.profiles import (
    COMPONENT_CATALOG,
    ComponentLicenseSpec,
    check_profile_readiness,
    resolve_default_cards_dir,
    validate_pyproject_commercial_dependencies,
    validate_three_layer_and_transitive_clean_room,
)
from mrnax.harness.state_machine import ProjectHarnessSession
from mrnax.rules.models import load_replacement_card_yaml, verify_replacement_card_strict
from mrnax.schemas.run_manifest import RunManifest

ROOT = Path(__file__).resolve().parents[2]
SEED = ROOT / "registry_seed"


def test_adr_0003_and_run_manifest_method_fto_audit(tmp_path: Path) -> None:
    adr_path = ROOT / "docs" / "adr" / "0003-method-fto-joint-optimization-and-wave8-licensing.md"
    assert adr_path.is_file()
    adr_text = adr_path.read_text(encoding="utf-8")
    assert "CN119265214B" in adr_text
    assert "US20220269951A1" in adr_text
    assert "varnaapi" in adr_text

    ref = _build_preset_reference_r(SEED)
    sess = ProjectHarnessSession(project_id="g6_fto", store_dir=tmp_path / "proj", seed_dir=SEED)
    intent = sess.create_initial_intent(
        turn_id="t1",
        user_utterance="Commercial clean non-partitioned run",
        protein_sequence=ref.molecules[0].effective_translation_target.protein_sequence,
        compliance_track="commercial_clean",
        uridine_chemistry="m1psi",
        use_commercial_surrogate=True,
    )
    bundles, report, manifest = run_design_pipeline(intent, seed_dir=SEED)
    assert bundles
    assert report.outcome_status.startswith("COMPLETED")
    audit = manifest.optimization_mode_audit
    assert audit.optimization_mode == "full_transcript_non_partitioned_pareto"
    assert audit.uses_two_zone_initiation_body_partition is False
    assert audit.uses_dfa_cfg_intersection is False
    assert audit.sequence_fto_screen_applied is True
    assert "not a legal Freedom-to-Operate opinion" in audit.counsel_disclaimer

    # Unexecuted / blocked runs must NOT claim optimization or sequence FTO screening, and must preserve legacy banner
    blocked_intent = intent.model_copy_validated(
        runtime_policy=intent.runtime_policy.model_copy_validated(enabled_evaluators=["viennarna_thermo"])
    )
    b_unavail, rep_unavail, man_unavail = run_design_pipeline(
        blocked_intent, seed_dir=SEED, legacy_pure_rerank_mode=True
    )
    assert b_unavail == []
    assert rep_unavail.outcome_status == "UNCOMPARABLE_MISSING_APPROVED_COMPONENT"
    assert man_unavail.prng_state_digest == "not_executed"
    assert man_unavail.honesty_banner == "本次未改善生成器"
    assert man_unavail.optimization_mode_audit.optimization_mode == "not_executed"
    assert man_unavail.optimization_mode_audit.uses_two_zone_initiation_body_partition is None
    assert man_unavail.optimization_mode_audit.uses_dfa_cfg_intersection is None
    assert man_unavail.optimization_mode_audit.sequence_fto_screen_applied is False

    # Historical manifests lacking `optimization_mode_audit` must hydrate as `unrecorded_legacy` with False screening
    for rel in (
        "artifacts/m1_egfp/replay_protein_only/run_manifest.json",
        "artifacts/m1_egfp/replay_protein_plus_r/run_manifest.json",
    ):
        hist_path = ROOT / rel
        if hist_path.is_file():
            hist_man = RunManifest.model_validate_json(hist_path.read_text(encoding="utf-8"))
            assert hist_man.optimization_mode_audit.optimization_mode == "unrecorded_legacy"
            assert hist_man.optimization_mode_audit.sequence_fto_screen_applied is False


def test_wave8_three_layer_and_transitive_license_traps_blocked_in_commercial_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validate_pyproject_commercial_dependencies(ROOT / "pyproject.toml")

    for bad_dep in ("varnaapi==1.1.1", "helical@https://example.invalid/a.whl", "utr.lm>=1", "UTR__LM[gpu]>=0.2"):
        bad_pyproject = tmp_path / "pyproject.toml"
        bad_pyproject.write_text(
            f'[project]\nname = "bad"\ndependencies = ["pydantic>=2.9", "{bad_dep}"]\n',
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="Forbidden non-commercial or copyleft dependency"):
            validate_pyproject_commercial_dependencies(bad_pyproject)

    # Transitive copyleft trap: MIT top-level wrapper pulling GPL-3.0 varnaapi cannot be commercial_clean
    mit_wrapper_gpl_dep = ComponentLicenseSpec(
        component_id="sneaky_wrapper",
        display_name="Sneaky MIT Wrapper",
        version="0.1.0",
        role="generator",
        code_license="MIT License",
        weights_license="n/a",
        data_license="CC0-1.0",
        license_track="commercial_clean",
        transitive_dependencies=("numpy (BSD-3-Clause)", "varnaapi==1.1.1 (GPL-3.0)"),
    )
    with pytest.raises(ValueError, match="GPL-3.0"):
        validate_three_layer_and_transitive_clean_room(mit_wrapper_gpl_dep)

    # Unverified weights trap cannot be commercial_clean
    unverified_weights = ComponentLicenseSpec(
        component_id="sneaky_weights",
        display_name="Apache Code + Unverified Weights",
        version="0.1.0",
        role="evaluator",
        code_license="Apache-2.0",
        weights_license="Unverified standalone .pt weight checkpoint",
        data_license="CC-BY-4.0",
        license_track="commercial_clean",
    )
    with pytest.raises(ValueError, match="Unverified"):
        validate_three_layer_and_transitive_clean_room(unverified_weights)

    # Every registered Wave 8 trap in COMPONENT_CATALOG is blocked in commercial_clean
    ref = _build_preset_reference_r(SEED)
    sess = ProjectHarnessSession(project_id="g6_traps", store_dir=tmp_path / "traps", seed_dir=SEED)
    base_intent = sess.create_initial_intent(
        turn_id="t1",
        user_utterance="Check Wave 8 component blocks",
        protein_sequence=ref.molecules[0].effective_translation_target.protein_sequence,
        compliance_track="commercial_clean",
    )
    trap_ids = [
        "multistructrna_varnaapi",
        "helix_mrna_helical",
        "utr_lm_gpl3",
        "ribonn_noncommercial",
        "codonbert_noncommercial",
        "gemorna_noncommercial",
        "aido_rna_noncommercial",
        "lineardesign_academic",
        "ribonanzanet_unverified_weights",
        "rhofold_plus_nc_data",
    ]
    for trap_id in trap_ids:
        assert trap_id in COMPONENT_CATALOG
        pol = base_intent.runtime_policy.model_copy_validated(enabled_evaluators=[trap_id])
        readiness = check_profile_readiness(base_intent.model_copy_validated(runtime_policy=pol))
        assert readiness.is_runnable is False
        assert readiness.uncomparable_reason is not None
        assert trap_id in readiness.uncomparable_reason

    # BSD-3-Clause EternaFold v1.3.1 (RC-004) passes strict replacement card verification
    rc004_path = resolve_default_cards_dir() / "RC-004-eternafold-v1.3.1.yaml"
    card = load_replacement_card_yaml(rc004_path)
    verify_replacement_card_strict(card, expected_component_id="eternafold_bpp_engine", expected_version="1.3.1")
    pol_ef = base_intent.runtime_policy.model_copy_validated(
        enabled_evaluators=["surrogate_v1_clean", "eternafold_bpp_engine"]
    )
    assert check_profile_readiness(base_intent.model_copy_validated(runtime_policy=pol_ef)).is_runnable is True

    # Two-hop and wrapper transitive component readiness must reject outer wrappers over restricted components
    middle_spec = ComponentLicenseSpec(
        component_id="middle_wrapper",
        display_name="Middle Clean Wrapper",
        version="0.1.0",
        role="evaluator",
        code_license="MIT License",
        weights_license="n/a",
        data_license="CC0-1.0",
        license_track="commercial_clean",
        transitive_dependencies=("utr_lm_gpl3",),
    )
    outer_spec = ComponentLicenseSpec(
        component_id="outer_wrapper",
        display_name="Outer Clean Wrapper",
        version="0.1.0",
        role="evaluator",
        code_license="MIT License",
        weights_license="n/a",
        data_license="CC0-1.0",
        license_track="commercial_clean",
        transitive_dependencies=("middle_wrapper",),
    )
    monkeypatch.setitem(COMPONENT_CATALOG, "middle_wrapper", middle_spec)
    monkeypatch.setitem(COMPONENT_CATALOG, "outer_wrapper", outer_spec)
    for wid in ("middle_wrapper", "outer_wrapper"):
        r_w = check_profile_readiness(
            base_intent.model_copy_validated(
                runtime_policy=base_intent.runtime_policy.model_copy_validated(enabled_evaluators=[wid])
            )
        )
        assert r_w.is_runnable is False

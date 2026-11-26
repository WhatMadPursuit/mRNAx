"""TDD Contract Tests FT-12 .. FT-15 for Stage 0 vs Pareto Engine, Zero-Qualified, Uncached Error & Ablations."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mrnax.constraints.registry import default_m1_constraints
from mrnax.engine.ablation_harness import run_fixed_pool_scorer_replay
from mrnax.engine.cell_expression import assemble_detargeted_3utr, evaluate_mirna_site_accessibility
from mrnax.engine.pareto_director import NoAvailableGeneratorError, run_design_pipeline
from mrnax.schemas.construct_bundle import ProteinTargetSpec
from mrnax.schemas.design_intent import (
    CellExpressionSpec,
    ConstraintSpec,
    DesignIntent,
    ObjectiveSpec,
    RuntimePolicySpec,
)

SEED_DIR = Path("/Users/alex/.gemini/antigravity/scratch/mrnax-v2/registry_seed")


def _mini_reporter_intent(
    *,
    extra_constraints: list[ConstraintSpec] | None = None,
    enabled_generators: list[str] | None = None,
    compliance_track: str = "research_max",
    detarget_liver: bool = False,
) -> DesignIntent:
    # 30-aa N-terminal segment of eGFP for fast unit testing (<1s)
    egfp_snap = json.loads((SEED_DIR / "egfp_reporter_snapshot.json").read_text(encoding="utf-8"))
    prot_30 = egfp_snap["mammalian_reporter_egfp"]["protein_sequence"][:30]
    base_constraints = default_m1_constraints(restriction_enzymes=["BspQI"])
    if extra_constraints:
        base_constraints.extend(extra_constraints)

    return DesignIntent(
        intent_id="int_engine_test",
        project_id="proj_egfp",
        intent_version=1,
        parent_intent_id=None,
        created_from_turn_ids=["turn_01"],
        entry_mode="protein_only",
        compliance_track=compliance_track,  # type: ignore[arg-type]
        uridine_chemistry="m1psi",
        status="ready",
        protein_targets=[
            ProteinTargetSpec(
                protein_sequence=prot_30,
                transl_table=1,
                accession_or_source="GenBank:U55762.1_N30",
                entry_version="AAB02572.1",
                confirmation_status="confirmed",
            )
        ],
        constraints=base_constraints,
        objectives=[
            ObjectiveSpec(
                requirement_id="obj_init_open",
                metric_id="kozak_unpaired_prob",
                direction="maximize",
                priority_rank=1,
                evidence_status="thermodynamic_proxy",
                source_utterance="Maximize 5' initiation accessibility",
            ),
            ObjectiveSpec(
                requirement_id="obj_cds_stability",
                metric_id="cds_mean_window_mfe",
                direction="minimize",
                priority_rank=2,
                evidence_status="thermodynamic_proxy",
                source_utterance="Stabilize local CDS secondary structure",
            ),
        ],
        cell_expression_control=(
            CellExpressionSpec(
                detarget_tissues=["liver_hepatocyte"],
                protected_tissues=["skeletal_muscle"],
            )
            if detarget_liver
            else None
        ),
        portfolio_tracks=["computational_reference", "exploratory_pareto"],
        runtime_policy=RuntimePolicySpec(
            budget_tier="fast",
            max_wall_seconds=20.0,
            max_evaluations=12,
            random_seed=42,
            allow_cached_candidates=False,
            enabled_generators=(
                ["stage0_random", "dnachisel_cds", "pareto_nsga2"]
                if enabled_generators is None
                else enabled_generators
            ),
            enabled_evaluators=["viennarna_thermo"],
        ),
    )


def test_zero_qualified_returns_empty_recommendations_and_diagnostics_only_ft12() -> None:
    """FT-12: Contradictory hard gate -> COMPLETED_ZERO_QUALIFIED, recommended_bundle_ids == [], diagnostics populated."""
    impossible_gc = ConstraintSpec(
        requirement_id="req_impossible_gc99",
        constraint_kind="gc_window_bounds",
        target_scopes=frozenset({"cds"}),
        severity="hard_gate",
        source_type="user_explicit",
        source_locator="turn_01",
        params={"min_gc": 0.99, "max_gc": 1.0, "window": 30},
    )
    intent = _mini_reporter_intent(extra_constraints=[impossible_gc])
    bundles, report, manifest = run_design_pipeline(intent, seed_dir=SEED_DIR)

    assert bundles == []
    assert report.outcome_status == "COMPLETED_ZERO_QUALIFIED"
    assert report.recommended_bundle_ids == []
    assert report.candidate_evaluations == []
    assert len(report.failed_diagnostics) >= 1
    assert any(d.violated_requirement_id == "req_impossible_gc99" for d in report.failed_diagnostics)
    assert manifest.execution_id == report.execution_id


def test_disabled_uncached_generator_raises_runtime_error_ft13() -> None:
    """FT-13: Disabling generators when allow_cached_candidates=False MUST raise NoAvailableGeneratorError."""
    intent = _mini_reporter_intent(enabled_generators=[])
    with pytest.raises(NoAvailableGeneratorError):
        run_design_pipeline(intent, seed_dir=SEED_DIR)


def test_fixed_pool_scorer_three_mode_replay_identity_ft14() -> None:
    """FT-14 (Joint Ruling #4): Baseline vs Display-Only order MUST be 100% identical; banner marks non-generator change."""
    intent = _mini_reporter_intent()
    bundles, report, _ = run_design_pipeline(intent, seed_dir=SEED_DIR)
    assert len(bundles) >= 2

    replay = run_fixed_pool_scorer_replay(
        candidate_pool=bundles,
        base_evaluations=report.candidate_evaluations,
        scorer_name="gc_third_codon_bonus_scorer",
        scorer_fn=lambda b: float(b.molecules[0].get_segment_sequence("cds")[2::3].count("C")),
    )
    assert replay.baseline_ordered_ids == replay.display_only_ordered_ids
    assert replay.honesty_banner == "本次未改善核心生成器"
    assert len(replay.rank_participating_ordered_ids) == len(bundles)


def test_utr_cds_cross_junction_and_mir122_rnaup_response_ft15() -> None:
    """FT-15: miR-122-5p RNAup detects seed occlusion by complementary CDS tail; m1Psi flagged as approximation."""
    mir_reg = json.loads((SEED_DIR / "mirna_rbp_commercial_v1.json").read_text(encoding="utf-8"))
    utr_reg = json.loads((SEED_DIR / "utr_components_v1.json").read_text(encoding="utf-8"))
    base_utr3 = utr_reg["three_prime_utrs"][0]["sequence"][:60]

    assembled_utr3, mir_annotations, mir122_mature = assemble_detargeted_3utr(
        base_utr3=base_utr3,
        detarget_tissues=["liver_hepatocyte"],
        protected_tissues=["skeletal_muscle"],
        mirna_registry=mir_reg,
        copies=2,
    )
    assert len(mir_annotations) == 2

    # Construct an open CDS tail vs a CDS tail containing exact reverse-complement masking the miR-122 target box!
    mir122_entry = next(m for m in mir_reg["mirna_entries"] if m["mirna_id"] == "hsa-miR-122-5p")
    target_box = mir122_entry["full_complementary_site_22nt"]
    from Bio.Seq import Seq

    occluding_tail = str(Seq(target_box).reverse_complement_rna())  # Base-pairs directly with target_box!
    open_tail = "CAGCAGCAGCAGCAGCAGCAGC"

    acc_open = evaluate_mirna_site_accessibility(
        cds_tail_rna=open_tail,
        stop_codon="UAA",
        utr3_rna=assembled_utr3,
        mirna_mature_rna=mir122_mature,
    )
    acc_occluded = evaluate_mirna_site_accessibility(
        cds_tail_rna=occluding_tail,
        stop_codon="UAA",
        utr3_rna=assembled_utr3,
        mirna_mature_rna=mir122_mature,
    )
    # Open configuration must have more favorable (more negative) total interaction free energy dG_total than occluded!
    assert acc_open.dg_total_kcal_mol < acc_occluded.dg_total_kcal_mol - 2.0

    # Also verify full pipeline with liver detargeting marks m1Psi thermo metrics as cross_chemistry_thermodynamic_approximation
    intent = _mini_reporter_intent(detarget_liver=True)
    bundles, report, _ = run_design_pipeline(intent, seed_dir=SEED_DIR)
    assert report.outcome_status == "COMPLETED_WITH_CANDIDATES"
    assert len(bundles) >= 1
    thermo_metrics = [
        m for m in report.candidate_evaluations[0].metrics if m.tool_name == "ViennaRNA"
    ]
    assert len(thermo_metrics) >= 2
    assert all(
        m.chemistry_evaluation_mode == "cross_chemistry_thermodynamic_approximation"
        for m in thermo_metrics
    )

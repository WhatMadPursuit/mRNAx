"""TDD Contract Tests FT-01 .. FT-03 for the 5 Strongly-Typed Schemas (M0)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mrnax.schemas.construct_bundle import (
    ChemistryTrackSpec,
    ConstructBundle,
    FeatureAnnotation,
    ProteinTargetSpec,
    RNAMoleculeSpec,
    SegmentSpan,
)
from mrnax.schemas.design_intent import (
    CellExpressionSpec,
    ConstraintSpec,
    DesignIntent,
    ObjectiveSpec,
    RuntimePolicySpec,
)
from mrnax.schemas.evaluation_report import (
    BundleEvaluation,
    EvaluationReport,
    FailedCandidateDiagnostic,
    MetricResult,
    RequirementActionTrace,
)
from mrnax.schemas.experiment_record import (
    BiologicalReplicate,
    ConstructObservation,
    ExperimentRecord,
    FeedbackDecisionRecord,
    PreRegisteredSwitchCriteria,
)
from mrnax.schemas.run_manifest import (
    ArtifactChecksumRecord,
    ComponentInvocationRecord,
    RunManifest,
)


def _valid_egfp_mini_molecule() -> RNAMoleculeSpec:
    # Mini reporter protein: MVSKG (5 aa -> 15 nt CDS: AUG GUG AGC AAG GGC)
    # 5'UTR (6 nt): GGGAAA
    # CDS (15 nt): AUGGUGAGCAAGGGC
    # Stop cassette (3 nt): UAA
    # 3'UTR (6 nt): UGAUAA
    # poly_a (10 nt): AAAAAAAAAA
    seq = "GGGAAAAUGGUGAGCAAGGGCUAAUGAUAAAAAAAAAAAA"
    assert len(seq) == 40
    return RNAMoleculeSpec(
        molecule_id="mol_egfp_mini",
        modality="linear_mrna",
        topology="linear",
        chemistry_track=ChemistryTrackSpec(
            uridine_chemistry="m1psi",
            cap_chemistry="cap1_cleancap_ag",
            source="user_declared",
        ),
        effective_translation_target=ProteinTargetSpec(
            protein_sequence="MVSKG",
            transl_table=1,
            accession_or_source="UniProtKB:P42212_mini",
            entry_version="196",
            confirmation_status="confirmed",
        ),
        sequence=seq,
        segments=[
            SegmentSpan(
                segment_type="five_prime_utr",
                start=0,
                end=6,
                component_id="utr5_clean_01",
                origin_source="GENCODE_v49:HBA1",
                license_track="commercial_clean",
                action_taken="selected_from_library",
            ),
            SegmentSpan(
                segment_type="cds",
                start=6,
                end=21,
                component_id=None,
                origin_source="mrnax:dnachisel",
                license_track="commercial_clean",
                action_taken="generated_novo",
            ),
            SegmentSpan(
                segment_type="stop_cassette",
                start=21,
                end=24,
                component_id="stop_uaa",
                origin_source="mrnax:standard",
                license_track="commercial_clean",
                action_taken="locked_verbatim",
            ),
            SegmentSpan(
                segment_type="three_prime_utr",
                start=24,
                end=30,
                component_id="utr3_clean_01",
                origin_source="GENCODE_v49:HBA1",
                license_track="commercial_clean",
                action_taken="selected_from_library",
            ),
            SegmentSpan(
                segment_type="poly_a",
                start=30,
                end=40,
                component_id="polya_a10",
                origin_source="mrnax:homopolymer_tail",
                license_track="commercial_clean",
                action_taken="locked_verbatim",
            ),
        ],
        annotations=[
            # Nested inside CDS [6, 21) AND overlapping 5'UTR..CDS junction [2, 10)
            FeatureAnnotation(
                feature_id="feat_sig",
                feature_key="sig_peptide",
                intervals=[(6, 15)],
                qualifiers={"note": ["nested_inside_cds"]},
            ),
            FeatureAnnotation(
                feature_id="feat_kozak",
                feature_key="regulatory",
                intervals=[(2, 10)],
                qualifiers={"regulatory_class": ["ribosome_binding_site"]},
            ),
        ],
    )


def test_construct_bundle_insdc_invariants_ft01() -> None:
    """FT-01: Segments must cover [0, L) seamlessly without overlap, CDS must match protein 100%, nested annotations allowed."""
    mol = _valid_egfp_mini_molecule()
    bundle = ConstructBundle.create(
        bundle_id="bnd_001",
        parent_bundle_id=None,
        derivation_action="de_novo_design",
        target_stoichiometry={"mol_egfp_mini": 1.0},
        co_expression_mode="single_monocistronic",
        molecules=[mol],
    )
    assert len(bundle.version_digest) == 64
    assert len(bundle.molecules[0].annotations) == 2

    # 1-nt gap in segments -> ValidationError
    bad_segments_gap = list(mol.segments)
    bad_segments_gap[1] = SegmentSpan(
        segment_type="cds",
        start=7,  # 1-nt gap at index 6!
        end=21,
        component_id=None,
        origin_source="mrnax:dnachisel",
        license_track="commercial_clean",
        action_taken="generated_novo",
    )
    with pytest.raises(ValidationError, match="contiguous"):
        mol.model_copy_validated(segments=bad_segments_gap)

    # 1-aa mismatch in translation -> ValidationError
    bad_target = ProteinTargetSpec(
        protein_sequence="MVSKR",  # G -> R mismatch!
        transl_table=1,
        accession_or_source="UniProtKB:P42212_mini",
        entry_version="196",
        confirmation_status="confirmed",
    )
    with pytest.raises(ValidationError, match="translation"):
        mol.model_copy_validated(effective_translation_target=bad_target)


def test_design_intent_requires_explicit_scopes_and_protected_tissue_ft02() -> None:
    """FT-02: Empty target_scopes rejected; detargeting without protected_tissues rejected."""
    with pytest.raises(ValidationError, match="target_scopes"):
        ConstraintSpec(
            requirement_id="req_bad_scope",
            constraint_kind="homopolymer_max",
            target_scopes=frozenset(),  # Empty scopes MUST fail!
            severity="hard_gate",
            source_type="system_safety_gate",
            source_locator="system:v1",
            params={"max_len": 7},
        )

    with pytest.raises(ValidationError, match="protected_tissues"):
        CellExpressionSpec(
            detarget_tissues=["liver_hepatocyte"],
            protected_tissues=[],  # Must declare protected tissues!
        )

    intent = DesignIntent(
        intent_id="int_001",
        project_id="proj_egfp",
        intent_version=1,
        parent_intent_id=None,
        created_from_turn_ids=["turn_01"],
        entry_mode="protein_only",
        compliance_track="research_max",
        status="ready",
        protein_targets=[
            ProteinTargetSpec(
                protein_sequence="MVSKG",
                transl_table=1,
                accession_or_source="UniProtKB:P42212_mini",
                entry_version="196",
                confirmation_status="confirmed",
            )
        ],
        reference_bundle_id=None,
        locked_scopes={},
        mutable_scopes=["five_prime_utr", "cds", "three_prime_utr"],
        constraints=[
            ConstraintSpec(
                requirement_id="req_homopoly",
                constraint_kind="homopolymer_max",
                target_scopes=frozenset({"five_prime_utr", "cds", "stop_cassette", "three_prime_utr"}),
                severity="hard_gate",
                source_type="system_safety_gate",
                source_locator="system:homopolymer_v1",
                params={"max_run": 7},
            )
        ],
        objectives=[
            ObjectiveSpec(
                requirement_id="obj_5p_access",
                metric_id="kozak_unpaired_prob",
                direction="maximize",
                priority_rank=1,
                evidence_status="thermodynamic_proxy",
                source_utterance="Maximize translation initiation accessibility",
            )
        ],
        cell_expression_control=CellExpressionSpec(
            detarget_tissues=["liver_hepatocyte"],
            protected_tissues=["skeletal_muscle"],
        ),
        portfolio_tracks=["computational_reference", "exploratory_pareto"],
        runtime_policy=RuntimePolicySpec(
            budget_tier="fast",
            max_wall_seconds=30.0,
            max_evaluations=50,
            random_seed=42,
            allow_cached_candidates=False,
            enabled_generators=["stage0_random", "dnachisel_cds", "pareto_nsga2"],
            enabled_evaluators=["viennarna_thermo"],
        ),
    )
    assert intent.intent_version == 1


def test_evaluation_report_zero_qualified_and_run_manifest_invariants() -> None:
    """Zero-qualified report cannot carry recommended_bundle_ids; m1psi approximation tracked."""
    with pytest.raises(ValidationError, match="recommended_bundle_ids"):
        EvaluationReport(
            report_id="rep_bad",
            execution_id="exec_1",
            intent_id="int_001",
            intent_version=1,
            compliance_track="research_max",
            outcome_status="COMPLETED_ZERO_QUALIFIED",
            recommended_bundle_ids=["bnd_illegal"],  # Forbidden when zero-qualified!
            candidate_evaluations=[],
            failed_diagnostics=[],
            fto_alerts=[],
            pre_synthesis_advisories=[],
            action_transparency=[],
        )

    valid_zero = EvaluationReport(
        report_id="rep_zero",
        execution_id="exec_1",
        intent_id="int_001",
        intent_version=1,
        compliance_track="research_max",
        outcome_status="COMPLETED_ZERO_QUALIFIED",
        recommended_bundle_ids=[],
        candidate_evaluations=[],
        failed_diagnostics=[
            FailedCandidateDiagnostic(
                candidate_ref="cand_raw_01",
                violated_requirement_id="req_homopoly",
                violated_scope="cds",
                transcript_coordinates=(10, 19),
                reason="9-nt A homopolymer in cds",
            )
        ],
        fto_alerts=[],
        pre_synthesis_advisories=[],
        action_transparency=[
            RequirementActionTrace(
                requirement_id="req_homopoly",
                stage_applied="filtering",
                eliminated_count=1,
                changed_candidate_ids=[],
                proof_diff_summary="Eliminated 1 candidate violating cds homopolymer limit",
            )
        ],
    )
    assert valid_zero.recommended_bundle_ids == []

    manifest = RunManifest(
        submission_id="sub_1",
        execution_id="exec_1",
        attempt_id="att_1",
        lease_epoch=1,
        git_commit_sha="deadbeef",
        python_version="3.12.15",
        platform_info="macOS-arm64",
        compliance_track="research_max",
        invoked_components=[
            ComponentInvocationRecord(
                name="ViennaRNA",
                version="2.8.0",
                role="evaluator",
                code_license="ViennaRNA Custom Disclaimer",
                weights_license="n/a",
                data_license="Turner 2004 public tables",
                license_track="needs_vendor_license",
                wall_seconds=0.12,
                attempt_count=1,
                failure_count=0,
            )
        ],
        rule_snapshot_version="rules_research_v1@1.0.0",
        isolated_sandbox_override=False,
        prng_state_digest="abc123",
        input_intent_digest="def456",
        artifacts=[
            ArtifactChecksumRecord(
                logical_name="evaluation_report",
                local_relpath="evaluation_report.json",
                drive_file_id=None,
                sha256="0" * 64,
                md5="0" * 32,
                bytes=512,
            )
        ],
        replay_cli_command="uv run mrnax replay --intent-file intent.json",
        honesty_banner="本次未改善核心生成器",
    )
    assert manifest.honesty_banner == "本次未改善核心生成器"


def test_experiment_record_replicates_and_n2_weight_lock_ft03() -> None:
    """FT-03: 1 bio sample x 3 tech wells -> n_bio == 1; N < 6 constructs forbids weight_update_allowed."""
    obs_a = ConstructObservation(
        bundle_id="bnd_A",
        version_digest="a" * 64,
        source_row_or_figure="Table1:Row2",
        biological_replicates=[
            BiologicalReplicate(bio_rep_id="dish_1", technical_replicates=[101.0, 99.5, 102.5])
        ],
    )
    assert obs_a.n_biological_replicates == 1
    assert obs_a.n_total_technical_measurements == 3

    obs_b = ConstructObservation(
        bundle_id="bnd_B",
        version_digest="b" * 64,
        source_row_or_figure="Table1:Row3",
        biological_replicates=[
            BiologicalReplicate(bio_rep_id="dish_1", technical_replicates=[210.0]),
            BiologicalReplicate(bio_rep_id="dish_2", technical_replicates=[225.0]),
            BiologicalReplicate(bio_rep_id="dish_3", technical_replicates=[218.0]),
        ],
    )
    assert obs_b.n_biological_replicates == 3

    # N=2 constructs with weight_update_allowed=True MUST raise ValidationError!
    with pytest.raises(ValidationError, match="weight_update_allowed"):
        ExperimentRecord(
            experiment_id="exp_001",
            project_id="proj_egfp",
            batch_id="batch_20261127_01",
            extraction_state="verified_structured",
            assay_type="live_cell_fluorescence",
            readout_name_exact="eGFP integrated fluorescence 24h",
            unit="RFU",
            timepoint_hours=24.0,
            cell_or_model_context="HEK293T",
            chemistry_track="m1psi",
            delivery_vehicle="Lipofectamine MessengerMAX",
            shared_controls=["mock_untreated", "bnd_R"],
            pre_registered_criteria=PreRegisteredSwitchCriteria(),
            observations=[obs_a, obs_b],
            feedback_decision=FeedbackDecisionRecord(
                branch="rank_reversal_switch_reference",
                new_reference_bundle_id="bnd_B",
                weight_update_allowed=True,  # ILLEGAL for N=2!
                isolated_calibration_proposal_id=None,
                sequence_attribution_status="multi_region_ambiguous",
                spawned_disambiguation_chimera_id="bnd_chimera_AB",
                rationale="B outperformed A > 1.5x",
            ),
        )

    # Draft extraction cannot carry an active reference switch decision!
    with pytest.raises(ValidationError, match="draft_extraction"):
        ExperimentRecord(
            experiment_id="exp_draft",
            project_id="proj_egfp",
            batch_id="batch_20261127_01",
            extraction_state="draft_extraction",
            ambiguous_fields=["unit"],
            assay_type="live_cell_fluorescence",
            readout_name_exact="eGFP fluorescence",
            unit="RFU_or_RLU?",
            timepoint_hours=24.0,
            cell_or_model_context="HEK293T",
            chemistry_track="m1psi",
            delivery_vehicle="LNP",
            shared_controls=["bnd_R"],
            pre_registered_criteria=PreRegisteredSwitchCriteria(),
            observations=[obs_a, obs_b],
            feedback_decision=FeedbackDecisionRecord(
                branch="rank_reversal_switch_reference",
                new_reference_bundle_id="bnd_B",
                weight_update_allowed=False,
                isolated_calibration_proposal_id=None,
                sequence_attribution_status="single_region_resolved",
                spawned_disambiguation_chimera_id=None,
                rationale="Premature switch from draft!",
            ),
        )

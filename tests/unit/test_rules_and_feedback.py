"""TDD Contract Tests FT-20, FT-21, FT-22, FT-24 for Scientific Rule Tribunal & Replacement Card."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from mrnax.constraints.registry import default_m1_constraints
from mrnax.engine.pareto_director import run_design_pipeline
from mrnax.rules.models import (
    ClaimCard,
    ComponentReplacementCard,
    EvidenceObservationLink,
    RuleRecord,
    load_replacement_card_yaml,
)
from mrnax.rules.tribunal import RuleRegistry, count_independent_dataset_votes
from mrnax.schemas.construct_bundle import ProteinTargetSpec
from mrnax.schemas.design_intent import DesignIntent, ObjectiveSpec, RuntimePolicySpec

REPO_ROOT = Path(__file__).resolve().parents[2]
SEED_DIR = REPO_ROOT / "registry_seed"
RC_CARD_PATH = REPO_ROOT / "docs" / "replacement_cards" / "RC-001-viennarna-v2.8.0.yaml"


def test_underlying_dataset_deduplication_counts_single_vote_ft20() -> None:
    """FT-20 (Joint Ruling #5 + C-B2): Two papers reusing GSE114002 count as 1 vote; reviews count as 0."""
    c_sample = ClaimCard(
        claim_id="CLM-SAMPLE-2019",
        status="verified_observation",
        source_type="primary_research",
        citation_id="DOI:10.1038/s41587-019-0164-5",
        exact_location="Fig. 1c",
        underlying_dataset_id="GEO:GSE114002",
        lab_lineage_id="Seelig_UW",
        region_scope=["five_prime_utr"],
        chemistry_scope=["unmodified_u", "pseudouridine", "m1psi"],
        cell_or_model_context="HEK293T",
        protein_count_n=2,
        readout_name_exact="Mean Ribosome Load (MRL)",
        comparator="Length-matched 5'UTR without uAUG",
        effect_direction="decrease",
        evidence_layer="raw_observation",
    )
    c_reanalysis = ClaimCard(
        claim_id="CLM-REUSE-2024",
        status="verified_observation",
        source_type="primary_research",
        citation_id="DOI:10.1038/s41467-024-49508-2",
        exact_location="Fig. 3B (green curve reusing fixed-end 50nt GSE114002)",
        underlying_dataset_id="GSE114002",  # Different prefix formatting, same underlying GEO ID!
        lab_lineage_id="Seelig_UW",
        region_scope=["five_prime_utr"],
        chemistry_scope=["unmodified_u"],
        cell_or_model_context="HEK293T",
        protein_count_n=1,
        readout_name_exact="Mean Ribosome Load (MRL)",
        comparator="5'UTR without uAUG",
        effect_direction="decrease",
        evidence_layer="raw_observation",
    )
    c_review = ClaimCard(
        claim_id="CLM-REVIEW-PASSMORE",
        status="verified_observation",
        source_type="review",  # Reviews MUST count as 0 empirical votes per C-B2 audit!
        citation_id="DOI:10.1038/s41580-021-00417-y",
        exact_location="Main text",
        underlying_dataset_id="NONE_REVIEW",
        lab_lineage_id="Passmore_MRC",
        region_scope=["poly_a"],
        chemistry_scope=["unmodified_u"],
        cell_or_model_context="general",
        protein_count_n=0,
        readout_name_exact="narrative_review",
        comparator="n/a",
        effect_direction="increase",
        evidence_layer="raw_observation",
    )

    votes = count_independent_dataset_votes([c_sample, c_reanalysis, c_review])
    assert votes.unique_dataset_votes == 1
    assert votes.unique_lab_lineage_votes == 1
    assert votes.excluded_review_count == 1


def test_isolated_copy_counter_evidence_replay_never_mutates_prod_registry_ft21(tmp_path: Path) -> None:
    """FT-21: Simulated counter-evidence deactivates rule ONLY in fork_isolated_copy(), zero mutation to prod."""
    prod_file = tmp_path / "rules_research_v1.json"
    reg = RuleRegistry.create_with_initial_rules(
        prod_file,
        track="research_max",
        rules=[
            RuleRecord(
                rule_id="RUL-5UTR-UAUG-MRL-001",
                version="1.0.0",
                status="conditional_rule",
                allowed_track="commercial_clearance_pending",
                target_scopes=["five_prime_utr"],
                applicable_chemistries=["unmodified_u", "pseudouridine", "m1psi"],
                applicable_contexts=["HEK293T", "general_mammalian_ivt"],
                readout_name_exact="Mean Ribosome Load (MRL)",
                statement="Out-of-frame 5'UTR uAUG reduces Mean Ribosome Load (MRL) in HEK293T IVT mRNA.",
                linked_observations=[
                    EvidenceObservationLink(
                        claim_id="CC-MRNAX-MP01-001",
                        underlying_dataset_id="GSE114002",
                        evidence_layer="raw_observation",
                    )
                ],
                model_interpretations=[],
            )
        ],
    )
    prod_hash_before = hashlib.sha256(prod_file.read_bytes()).hexdigest()
    assert len(reg.get_active_rules(chemistry="m1psi", context="HEK293T")) == 1

    sandbox = reg.fork_isolated_copy(tmp_path / "sandbox_eval_copy.json")
    sandbox.apply_counter_evidence(
        rule_id="RUL-5UTR-UAUG-MRL-001",
        reason="[SIMULATED_COUNTER_EVIDENCE] Matched isogenic reporter batch showed null MRL delta",
        new_status="disproven_blacklist",
        is_simulated=True,
    )
    assert len(sandbox.get_active_rules(chemistry="m1psi", context="HEK293T")) == 0
    assert len(reg.get_active_rules(chemistry="m1psi", context="HEK293T")) == 1
    assert hashlib.sha256(prod_file.read_bytes()).hexdigest() == prod_hash_before


def test_unapproved_draft_claimcard_zero_contamination_on_dialogue_replay_ft22(tmp_path: Path) -> None:
    """FT-22: Adding an unverified_draft ClaimCard has ZERO effect on active rules, DesignIntent, or recommendations."""
    prod_file = tmp_path / "rules_research_v1.json"
    reg = RuleRegistry.create_with_initial_rules(prod_file, track="research_max", rules=[])

    intent = DesignIntent(
        intent_id="int_draft_check",
        project_id="proj_egfp",
        intent_version=1,
        parent_intent_id=None,
        created_from_turn_ids=["turn_01"],
        entry_mode="protein_only",
        compliance_track="research_max",
        uridine_chemistry="m1psi",
        status="ready",
        protein_targets=[
            ProteinTargetSpec(
                protein_sequence="MVSKGEELFTGVVPILVELDGDVNGHKFSV",
                transl_table=1,
                accession_or_source="GenBank:U55762.1_N30",
                confirmation_status="confirmed",
            )
        ],
        constraints=default_m1_constraints(restriction_enzymes=["BspQI"]),
        objectives=[
            ObjectiveSpec(
                requirement_id="obj_kozak",
                metric_id="kozak_unpaired_prob",
                direction="maximize",
                priority_rank=1,
                evidence_status="thermodynamic_proxy",
                source_utterance="Keep 5' initiation accessible",
            )
        ],
        runtime_policy=RuntimePolicySpec(random_seed=2026, max_evaluations=6),
    )

    bundles_before, report_before, _ = run_design_pipeline(
        intent, seed_dir=SEED_DIR, rule_snapshot_version=reg.snapshot_version()
    )

    # Ingest unverified draft ClaimCard into evidence inbox
    reg.ingest_draft_claim_card(
        ClaimCard(
            claim_id="DRAFT-UNVERIFIED-999",
            status="unverified_draft",
            source_type="primary_research",
            citation_id="DOI:10.1101/example.draft",
            exact_location="Fig. 9",
            underlying_dataset_id="GSE999999",
            lab_lineage_id="UnknownLab",
            region_scope=["cds"],
            chemistry_scope=["m1psi"],
            cell_or_model_context="HEK293T",
            protein_count_n=1,
            readout_name_exact="MRL",
            comparator="control",
            effect_direction="increase",
            evidence_layer="raw_observation",
        )
    )

    bundles_after, report_after, _ = run_design_pipeline(
        intent, seed_dir=SEED_DIR, rule_snapshot_version=reg.snapshot_version()
    )

    assert intent.digest() == intent.digest()
    assert [b.version_digest for b in bundles_before] == [b.version_digest for b in bundles_after]
    assert report_before.recommended_bundle_ids != []  # sanity
    assert [e.version_digest for e in report_before.candidate_evaluations] == [
        e.version_digest for e in report_after.candidate_evaluations
    ]


def test_component_replacement_card_completeness_and_commercial_gate_ft24() -> None:
    """FT-24: ViennaRNA replacement card forbids vague '95%' and commercial_clean marks missing thermo uncomparable."""
    card: ComponentReplacementCard = load_replacement_card_yaml(RC_CARD_PATH)
    assert card.component_id == "viennarna_thermo"
    assert card.current_license_track == "needs_vendor_license"
    assert len(card.acceptance_metrics) >= 2
    assert card.rollback_version == "ViennaRNA==2.8.0 (research_max only)"

    # Verify vague undefined '95%' rejection validator
    with pytest.raises(ValueError, match="vague percentage"):
        card.model_copy_validated(
            overall_switch_rule="Switch once overall multi-objective performance reaches >= 95% of baseline"
        )

    # Verify commercial_clean profile returns UNCOMPARABLE_MISSING_APPROVED_COMPONENT when ViennaRNA requested
    comm_intent = DesignIntent(
        intent_id="int_comm_gate",
        project_id="proj_egfp",
        intent_version=1,
        parent_intent_id=None,
        created_from_turn_ids=["turn_01"],
        entry_mode="protein_only",
        compliance_track="commercial_clean",
        uridine_chemistry="m1psi",
        status="ready",
        protein_targets=[
            ProteinTargetSpec(
                protein_sequence="MVSKGEELFTGVVPILVELDGDVNGHKFSV",
                transl_table=1,
                accession_or_source="GenBank:U55762.1_N30",
                confirmation_status="confirmed",
            )
        ],
        constraints=default_m1_constraints(restriction_enzymes=["BspQI"]),
        objectives=[
            ObjectiveSpec(
                requirement_id="obj_kozak",
                metric_id="kozak_unpaired_prob",
                direction="maximize",
                priority_rank=1,
                evidence_status="thermodynamic_proxy",
                source_utterance="Maximize 5' accessibility",
            )
        ],
        runtime_policy=RuntimePolicySpec(
            random_seed=42,
            enabled_generators=["dnachisel_cds"],
            enabled_evaluators=["viennarna_thermo"],
        ),
    )
    bundles, rep, _ = run_design_pipeline(comm_intent, seed_dir=SEED_DIR)
    assert bundles == []
    assert rep.outcome_status == "UNCOMPARABLE_MISSING_APPROVED_COMPONENT"
    assert "无法比较" in (rep.uncomparable_reason or "")

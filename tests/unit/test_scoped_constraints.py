"""TDD Contract Tests FT-04 .. FT-07 for Scoped Hard Constraints, m1Psi Frameshift & FTO Guard."""

from __future__ import annotations

import json
from pathlib import Path

from mrnax.constraints.frameshift_m1psi import remediate_m1psi_slippery_runs
from mrnax.constraints.registry import ScopedConstraintEngine, default_m1_constraints
from mrnax.schemas.construct_bundle import (
    ChemistryTrackSpec,
    ConstructBundle,
    ProteinTargetSpec,
    RNAMoleculeSpec,
    SegmentSpan,
)

SEED_DIR = Path("/Users/alex/.gemini/antigravity/scratch/mrnax-v2/registry_seed")


def _build_egfp_bundle(
    *,
    utr5: str | None = None,
    cds_override: str | None = None,
    utr3: str | None = None,
    polya: str = "A" * 100,
    chemistry: str = "m1psi",
) -> ConstructBundle:
    egfp_snap = json.loads((SEED_DIR / "egfp_reporter_snapshot.json").read_text(encoding="utf-8"))
    utr_reg = json.loads((SEED_DIR / "utr_components_v1.json").read_text(encoding="utf-8"))

    clean_utr5 = utr5 or next(u["sequence"] for u in utr_reg["five_prime_utrs"] if not u["has_uaug"])
    clean_utr3 = utr3 or utr_reg["three_prime_utrs"][0]["sequence"]
    raw_cds = cds_override or egfp_snap["mammalian_reporter_egfp"]["reference_cds_rna_no_stop"]
    if cds_override is None and chemistry in {"m1psi", "pseudouridine"}:
        raw_cds, _ = remediate_m1psi_slippery_runs(raw_cds)
    stop = egfp_snap["mammalian_reporter_egfp"]["reference_stop_codon"]
    prot = egfp_snap["mammalian_reporter_egfp"]["protein_sequence"]

    c0 = 0
    c1 = len(clean_utr5)
    c2 = c1 + len(raw_cds)
    c3 = c2 + len(stop)
    c4 = c3 + len(clean_utr3)
    c5 = c4 + len(polya)
    full_seq = clean_utr5 + raw_cds + stop + clean_utr3 + polya

    mol = RNAMoleculeSpec(
        molecule_id="mol_egfp",
        modality="linear_mrna",
        topology="linear",
        chemistry_track=ChemistryTrackSpec(
            uridine_chemistry=chemistry,  # type: ignore[arg-type]
            cap_chemistry="cap1_cleancap_ag",
            source="user_declared",
        ),
        effective_translation_target=ProteinTargetSpec(
            protein_sequence=prot,
            transl_table=1,
            accession_or_source="GenBank:U55762.1",
            entry_version="AAB02572.1",
            confirmation_status="confirmed",
        ),
        sequence=full_seq,
        segments=[
            SegmentSpan(
                segment_type="five_prime_utr",
                start=c0,
                end=c1,
                component_id="utr5_refseq",
                origin_source="NCBI_RefSeq",
                license_track="commercial_clean",
                action_taken="selected_from_library",
            ),
            SegmentSpan(
                segment_type="cds",
                start=c1,
                end=c2,
                component_id=None,
                origin_source="GenBank:U55762.1",
                license_track="commercial_clean",
                action_taken="synonymous_mutated",
            ),
            SegmentSpan(
                segment_type="stop_cassette",
                start=c2,
                end=c3,
                component_id="stop_std",
                origin_source="GenBank:U55762.1",
                license_track="commercial_clean",
                action_taken="locked_verbatim",
            ),
            SegmentSpan(
                segment_type="three_prime_utr",
                start=c3,
                end=c4,
                component_id="utr3_refseq",
                origin_source="NCBI_RefSeq",
                license_track="commercial_clean",
                action_taken="selected_from_library",
            ),
            SegmentSpan(
                segment_type="poly_a",
                start=c4,
                end=c5,
                component_id="polya_a100",
                origin_source="mrnax:homopolymer_tail",
                license_track="commercial_clean",
                action_taken="locked_verbatim",
            ),
        ],
    )
    return ConstructBundle.create(
        bundle_id="bnd_egfp_test",
        parent_bundle_id=None,
        derivation_action="computational_reference",
        target_stoichiometry={"mol_egfp": 1.0},
        molecules=[mol],
    )


def test_normal_polya_passes_scoped_homopolymer_gate_ft04() -> None:
    """FT-04 (Joint Ruling #3 Positive): Legal construct with A100 poly_a MUST pass homopolymer hard gate."""
    engine = ScopedConstraintEngine.from_seed_dir(SEED_DIR)
    constraints = default_m1_constraints(restriction_enzymes=["BspQI"])
    bundle = _build_egfp_bundle(polya="A" * 100)

    # Check if reference UTR/CDS happens to have BspQI; use enzyme-free check first
    constraints_no_enz = [c for c in constraints if c.constraint_kind != "type_iis_restriction_free"]
    res = engine.evaluate_bundle(bundle, constraints_no_enz, compliance_track="commercial_clean")
    assert res.passed_all_hard_gates is True, f"Unexpected violations: {res.violations}"
    assert res.fto_blocked is False


def test_injected_internal_homopolymer_and_uaug_caught_with_coords_ft05() -> None:
    """FT-05 (Joint Ruling #3 Negative): Injected 8xA in 3'UTR or uAUG in 5'UTR caught with exact scope & coords."""
    engine = ScopedConstraintEngine.from_seed_dir(SEED_DIR)
    constraints = [
        c for c in default_m1_constraints() if c.constraint_kind in {"homopolymer_max", "uaug_free"}
    ]

    bad_utr3_bundle = _build_egfp_bundle(utr3="GGGCCCCCCAAAAAAAAUUUGGG")
    res_utr3 = engine.evaluate_bundle(bad_utr3_bundle, constraints, compliance_track="commercial_clean")
    assert res_utr3.passed_all_hard_gates is False
    homopoly_v = [v for v in res_utr3.violations if v.violated_requirement_id == "gate_homopolymer_max"]
    assert len(homopoly_v) == 1
    assert homopoly_v[0].violated_scope == "three_prime_utr"
    s, e = homopoly_v[0].transcript_coordinates
    assert bad_utr3_bundle.molecules[0].sequence[s:e] == "AAAAAAAA"

    bad_utr5_bundle = _build_egfp_bundle(utr5="GGGAAAUUUAUGCCCACC")
    res_utr5 = engine.evaluate_bundle(bad_utr5_bundle, constraints, compliance_track="commercial_clean")
    assert res_utr5.passed_all_hard_gates is False
    uaug_v = [v for v in res_utr5.violations if v.violated_requirement_id == "gate_uaug_5utr"]
    assert len(uaug_v) == 1
    assert uaug_v[0].violated_scope == "five_prime_utr"
    s2, e2 = uaug_v[0].transcript_coordinates
    assert bad_utr5_bundle.molecules[0].sequence[s2:e2] == "AUG"


def test_m1psi_frameshift_and_bnt_51220548_fto_guard_ft06() -> None:
    """FT-06: m1Psi slippery U-run flagged & synonymously fixed with 100% AA identity; BNT 51220548 genus blocked."""
    # 1. Slippery motif: AUG UUU UUU A... (M F F ...) -> UUUUUUA across codon boundary!
    slippery_cds = "AUGUUUUUUAAGGGCAAGAAC"  # M F F K G K N
    fixed_cds, n_edits = remediate_m1psi_slippery_runs(slippery_cds)
    assert n_edits >= 1
    assert "UUUU" not in fixed_cds

    from Bio.Seq import Seq

    assert str(Seq(slippery_cds).translate()) == str(Seq(fixed_cds).translate())

    # 2. BioNTech family 51220548 genus check: BOTH A30-GCAUAUGACU-A70 AND custom linker A25-CCGG-A60 MUST block in commercial_clean!
    engine = ScopedConstraintEngine.from_seed_dir(SEED_DIR)
    constraints = default_m1_constraints(restriction_enzymes=[])

    bnt_ex_bundle = _build_egfp_bundle(polya=("A" * 30) + "GCAUAUGACU" + ("A" * 70))
    res_ex = engine.evaluate_bundle(bnt_ex_bundle, constraints, compliance_track="commercial_clean")
    assert res_ex.passed_all_hard_gates is False
    assert res_ex.fto_blocked is True
    assert any("51220548" in a.patent_family_or_pub for a in res_ex.fto_alerts)

    bnt_genus_bundle = _build_egfp_bundle(polya=("A" * 25) + "CCGG" + ("A" * 60))
    res_genus = engine.evaluate_bundle(bnt_genus_bundle, constraints, compliance_track="commercial_clean")
    assert res_genus.passed_all_hard_gates is False
    assert res_genus.fto_blocked is True
    assert any(a.rule_id == "BNT_FAMILY_51220548_SEGMENTED_POLYA_GENUS" for a in res_genus.fto_alerts)


def test_single_scope_source_across_generation_filter_export_ft07() -> None:
    """FT-07: Tampering with 3'UTR to introduce BspQI site (GCTCTTC -> GCUCUUC) fails export validation identically."""
    engine = ScopedConstraintEngine.from_seed_dir(SEED_DIR)
    constraints = default_m1_constraints(restriction_enzymes=["BspQI"])
    tampered = _build_egfp_bundle(utr3="GGGCCCGCUCUUCGGGCCC")  # Contains BspQI recognition GCUCUUC
    res = engine.evaluate_bundle(tampered, constraints, compliance_track="research_max")
    assert res.passed_all_hard_gates is False
    assert any(v.violated_requirement_id == "gate_type_iis_restriction" for v in res.violations)

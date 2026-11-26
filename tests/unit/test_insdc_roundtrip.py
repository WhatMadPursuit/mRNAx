"""TDD Contract Test FT-11 for INSDC GenBank/FASTA/Chemistry lossless round-trip & export gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mrnax.constraints.frameshift_m1psi import remediate_m1psi_slippery_runs
from mrnax.constraints.registry import ScopedConstraintEngine, default_m1_constraints
from mrnax.io_insdc.genbank_io import (
    ExportConstraintViolationError,
    export_construct_package,
    import_construct_from_genbank_package,
)
from mrnax.schemas.construct_bundle import (
    ChemistryTrackSpec,
    ConstructBundle,
    FeatureAnnotation,
    ProteinTargetSpec,
    RNAMoleculeSpec,
    SegmentSpan,
)

SEED_DIR = Path("/Users/alex/.gemini/antigravity/scratch/mrnax-v2/registry_seed")


def test_genbank_nested_features_lossless_roundtrip_ft11(tmp_path: Path) -> None:
    """FT-11: Export full RNA FASTA, CDS FASTA, INSDC GenBank with nested features & chemistry JSON, then round-trip."""
    egfp_snap = json.loads((SEED_DIR / "egfp_reporter_snapshot.json").read_text(encoding="utf-8"))
    utr_reg = json.loads((SEED_DIR / "utr_components_v1.json").read_text(encoding="utf-8"))

    utr5 = next(u["sequence"] for u in utr_reg["five_prime_utrs"] if not u["has_uaug"])
    utr3 = next(u["sequence"] for u in utr_reg["three_prime_utrs"] if u["max_homopolymer_run"] <= 7)
    cds, _ = remediate_m1psi_slippery_runs(
        egfp_snap["mammalian_reporter_egfp"]["reference_cds_rna_no_stop"]
    )
    stop = egfp_snap["mammalian_reporter_egfp"]["reference_stop_codon"]
    polya = "A" * 100
    prot = egfp_snap["mammalian_reporter_egfp"]["protein_sequence"]

    c0, c1 = 0, len(utr5)
    c2 = c1 + len(cds)
    c3 = c2 + len(stop)
    c4 = c3 + len(utr3)
    c5 = c4 + len(polya)
    full_seq = utr5 + cds + stop + utr3 + polya

    mol = RNAMoleculeSpec(
        molecule_id="mol_egfp_insdc",
        modality="linear_mrna",
        topology="linear",
        chemistry_track=ChemistryTrackSpec(
            uridine_chemistry="m1psi",
            cap_chemistry="cap1_cleancap_ag",
            source="user_declared",
        ),
        effective_translation_target=ProteinTargetSpec(
            protein_sequence=prot,
            transl_table=1,
            accession_or_source="GenBank:U55762.1",
            entry_version="AAB02572.1",
            has_signal_peptide=True,
            confirmation_status="confirmed",
        ),
        sequence=full_seq,
        segments=[
            SegmentSpan(
                segment_type="five_prime_utr",
                start=c0,
                end=c1,
                component_id="utr5_clean",
                origin_source="NCBI_RefSeq",
                license_track="commercial_clean",
                action_taken="selected_from_library",
            ),
            SegmentSpan(
                segment_type="cds",
                start=c1,
                end=c2,
                component_id=None,
                origin_source="mrnax:dnachisel",
                license_track="commercial_clean",
                action_taken="synonymous_mutated",
            ),
            SegmentSpan(
                segment_type="stop_cassette",
                start=c2,
                end=c3,
                component_id="stop_std",
                origin_source="mrnax:standard",
                license_track="commercial_clean",
                action_taken="locked_verbatim",
            ),
            SegmentSpan(
                segment_type="three_prime_utr",
                start=c3,
                end=c4,
                component_id="utr3_clean",
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
        annotations=[
            FeatureAnnotation(
                feature_id="sig_n_term",
                feature_key="sig_peptide",
                intervals=[(c1, c1 + 60)],
                qualifiers={"product": ["N-terminal leader tag"]},
            ),
            FeatureAnnotation(
                feature_id="kozak_junction",
                feature_key="regulatory",
                intervals=[(max(0, c1 - 6), c1 + 4)],
                qualifiers={"regulatory_class": ["ribosome_binding_site"]},
            ),
            FeatureAnnotation(
                feature_id="mir122_box_1",
                feature_key="misc_binding",
                intervals=[(c3 + 15, c3 + 37)],
                qualifiers={"bound_moiety": ["hsa-miR-122-5p"]},
            ),
        ],
    )
    bundle = ConstructBundle.create(
        bundle_id="bnd_insdc_rt",
        parent_bundle_id=None,
        derivation_action="de_novo_design",
        target_stoichiometry={"mol_egfp_insdc": 1.0},
        molecules=[mol],
    )

    engine = ScopedConstraintEngine.from_seed_dir(SEED_DIR)
    constraints = default_m1_constraints(restriction_enzymes=[])

    pkg_dir = tmp_path / "export_pkg"
    manifest = export_construct_package(
        bundle,
        pkg_dir,
        constraint_engine=engine,
        constraints=constraints,
        compliance_track="commercial_clean",
    )
    assert (pkg_dir / "full_rna.fasta").is_file()
    assert (pkg_dir / "cds_only.fasta").is_file()
    assert (pkg_dir / "construct.gb").is_file()
    assert (pkg_dir / "chemistry_metadata.json").is_file()
    assert manifest["bundle_id"] == "bnd_insdc_rt"

    reloaded = import_construct_from_genbank_package(pkg_dir)
    assert reloaded.molecules[0].sequence == bundle.molecules[0].sequence
    assert [(s.segment_type, s.start, s.end) for s in reloaded.molecules[0].segments] == [
        (s.segment_type, s.start, s.end) for s in bundle.molecules[0].segments
    ]
    reloaded_feat_ids = {a.feature_id: a.intervals for a in reloaded.molecules[0].annotations}
    assert reloaded_feat_ids["sig_n_term"] == [(c1, c1 + 60)]
    assert reloaded_feat_ids["kozak_junction"] == [(max(0, c1 - 6), c1 + 4)]
    assert reloaded_feat_ids["mir122_box_1"] == [(c3 + 15, c3 + 37)]

    # Verify export validator catches forbidden motif with the exact same ScopedConstraintEngine
    strict_constraints = default_m1_constraints(
        restriction_enzymes=[], forbidden_motifs=[utr3[:10]]
    )
    with pytest.raises(ExportConstraintViolationError):
        export_construct_package(
            bundle,
            tmp_path / "export_blocked",
            constraint_engine=engine,
            constraints=strict_constraints,
            compliance_track="commercial_clean",
        )

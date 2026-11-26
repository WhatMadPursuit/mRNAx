"""INSDC GenBank, Full-Length RNA FASTA, CDS FASTA, and Chemistry Metadata IO with shared scope check."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import CompoundLocation, FeatureLocation, SeqFeature
from Bio.SeqRecord import SeqRecord

from mrnax.constraints.registry import ScopedConstraintEngine
from mrnax.schemas.common import ComplianceProfile
from mrnax.schemas.construct_bundle import (
    ChemistryTrackSpec,
    ConstructBundle,
    FeatureAnnotation,
    ProteinTargetSpec,
    RNAMoleculeSpec,
    SegmentSpan,
)
from mrnax.schemas.design_intent import ConstraintSpec

_SEGMENT_TO_INSDC_KEY: dict[str, str] = {
    "five_prime_utr": "5'UTR",
    "cds": "CDS",
    "stop_cassette": "misc_feature",
    "three_prime_utr": "3'UTR",
    "poly_a": "polyA_tail",
}


class ExportConstraintViolationError(RuntimeError):
    """Raised when an outgoing ConstructBundle violates an active scoped constraint at export."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_construct_package(
    bundle: ConstructBundle,
    out_dir: Path,
    *,
    constraint_engine: ScopedConstraintEngine,
    constraints: list[ConstraintSpec],
    compliance_track: ComplianceProfile,
) -> dict[str, Any]:
    """Re-verify with the exact shared `ScopedConstraintEngine` then export INSDC 4-file package."""
    outcome = constraint_engine.evaluate_bundle(
        bundle,
        constraints,
        compliance_track=compliance_track,
    )
    if not outcome.passed_all_hard_gates:
        reasons = "; ".join(f"{v.violated_requirement_id}@{v.violated_scope}:{v.reason}" for v in outcome.violations)
        raise ExportConstraintViolationError(f"Export blocked by shared ScopedConstraintEngine: {reasons}")

    out_dir.mkdir(parents=True, exist_ok=True)
    full_fasta_path = out_dir / "full_rna.fasta"
    cds_fasta_path = out_dir / "cds_only.fasta"
    gb_path = out_dir / "construct.gb"
    chem_path = out_dir / "chemistry_metadata.json"

    full_fasta_lines: list[str] = []
    cds_fasta_lines: list[str] = []
    gb_records: list[SeqRecord] = []

    for mol in bundle.molecules:
        full_fasta_lines.append(
            f">{bundle.bundle_id}|{mol.molecule_id}|len={len(mol.sequence)}|chem={mol.chemistry_track.uridine_chemistry}\n{mol.sequence}"
        )
        cds_seg = mol.get_segment("cds")
        if cds_seg is not None:
            cds_seq = mol.sequence[cds_seg.start : cds_seg.end]
            cds_fasta_lines.append(
                f">{bundle.bundle_id}|{mol.molecule_id}|cds|{cds_seg.start}..{cds_seg.end}\n{cds_seq}"
            )

        # GenBank stores DNA/RNA sequence; INSDC mRNA convention uses A/C/G/T in LOCUS DNA/mRNA
        gb_seq = Seq(mol.sequence.replace("U", "T"))
        rec = SeqRecord(
            gb_seq,
            id=mol.molecule_id[:16],
            name=mol.molecule_id[:16],
            description=f"mRNAx ConstructBundle {bundle.bundle_id} ({mol.molecule_id})",
            annotations={
                "molecule_type": "mRNA",
                "topology": mol.topology,
                "data_file_division": "SYN",
            },
        )

        # Layer 1: Contiguous segments
        for seg in mol.segments:
            quals: dict[str, list[str]] = {
                "mrnax_layer": ["segment"],
                "mrnax_segment_type": [seg.segment_type],
                "mrnax_origin_source": [seg.origin_source],
                "mrnax_license_track": [seg.license_track],
                "mrnax_action_taken": [seg.action_taken],
            }
            if seg.component_id is not None:
                quals["mrnax_component_id"] = [seg.component_id]
            if seg.segment_type == "cds":
                quals["transl_table"] = [str(mol.effective_translation_target.transl_table)]
                quals["translation"] = [mol.effective_translation_target.protein_sequence]
            rec.features.append(
                SeqFeature(
                    location=FeatureLocation(seg.start, seg.end, strand=1),
                    type=_SEGMENT_TO_INSDC_KEY[seg.segment_type],
                    qualifiers=quals,
                )
            )

        # Layer 2: Nested / overlapping functional annotations
        for ann in mol.annotations:
            locs = [FeatureLocation(s, e, strand=ann.strand) for s, e in ann.intervals]
            loc = locs[0] if len(locs) == 1 else CompoundLocation(locs)
            ann_quals = {k: list(v) for k, v in ann.qualifiers.items()}
            ann_quals["mrnax_layer"] = ["annotation"]
            ann_quals["mrnax_feature_id"] = [ann.feature_id]
            rec.features.append(SeqFeature(location=loc, type=ann.feature_key, qualifiers=ann_quals))

        gb_records.append(rec)

    full_fasta_path.write_text("\n".join(full_fasta_lines) + "\n", encoding="utf-8")
    cds_fasta_path.write_text("\n".join(cds_fasta_lines) + "\n", encoding="utf-8")
    with gb_path.open("w", encoding="utf-8") as handle:
        SeqIO.write(gb_records, handle, "genbank")

    chem_manifest: dict[str, Any] = {
        "schema": "mrnax.export_package.v1",
        "bundle_id": bundle.bundle_id,
        "version_digest": bundle.version_digest,
        "parent_bundle_id": bundle.parent_bundle_id,
        "derivation_action": bundle.derivation_action,
        "target_stoichiometry": bundle.target_stoichiometry,
        "co_expression_mode": bundle.co_expression_mode,
        "coordinate_conventions": {
            "json_ir_segments_and_annotations": "0-based half-open [start, end)",
            "insdc_genbank_feature_table": "1-based closed start..end",
            "fasta_cds_header": "0-based half-open start..end on full RNA transcript",
        },
        "molecules": [
            {
                "molecule_id": m.molecule_id,
                "modality": m.modality,
                "topology": m.topology,
                "chemistry_track": m.chemistry_track.model_dump(mode="json"),
                "effective_translation_target": m.effective_translation_target.model_dump(mode="json"),
            }
            for m in bundle.molecules
        ],
        "pre_synthesis_advisories": [a.model_dump(mode="json") for a in outcome.advisories],
        "fto_alerts": [a.model_dump(mode="json") for a in outcome.fto_alerts],
        "files": {
            "full_rna.fasta": {"sha256": _sha256_file(full_fasta_path), "bytes": full_fasta_path.stat().st_size},
            "cds_only.fasta": {"sha256": _sha256_file(cds_fasta_path), "bytes": cds_fasta_path.stat().st_size},
            "construct.gb": {"sha256": _sha256_file(gb_path), "bytes": gb_path.stat().st_size},
        },
    }
    chem_path.write_text(json.dumps(chem_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return chem_manifest


def import_construct_from_genbank_package(pkg_dir: Path) -> ConstructBundle:
    """Reconstruct ConstructBundle losslessly from exported `construct.gb` + `chemistry_metadata.json` after SHA-256 verification."""
    chem_meta = json.loads((pkg_dir / "chemistry_metadata.json").read_text(encoding="utf-8"))
    for fname, file_rec in chem_meta.get("files", {}).items():
        fpath = pkg_dir / fname
        if not fpath.is_file():
            raise ValueError(f"Missing package file '{fname}'")
        actual_sha = _sha256_file(fpath)
        if actual_sha != file_rec["sha256"]:
            raise ValueError(
                f"Package SHA-256 verification failed for '{fname}': expected {file_rec['sha256']}, got {actual_sha}"
            )
    mol_meta_list = chem_meta["molecules"]
    gb_records = list(SeqIO.parse(str(pkg_dir / "construct.gb"), "genbank"))
    if len(gb_records) != len(mol_meta_list):
        raise ValueError("Molecule count mismatch between construct.gb and chemistry_metadata.json")

    molecules: list[RNAMoleculeSpec] = []
    for rec, m_meta in zip(gb_records, mol_meta_list, strict=True):
        rna_seq = str(rec.seq).upper().replace("T", "U")
        segments: list[SegmentSpan] = []
        annotations: list[FeatureAnnotation] = []

        for feat in rec.features:
            layer = feat.qualifiers.get("mrnax_layer", [""])[0]
            if layer == "segment":
                comp_list = feat.qualifiers.get("mrnax_component_id")
                segments.append(
                    SegmentSpan(
                        segment_type=feat.qualifiers["mrnax_segment_type"][0],  # type: ignore[arg-type]
                        start=int(feat.location.start),
                        end=int(feat.location.end),
                        component_id=comp_list[0] if comp_list else None,
                        origin_source=feat.qualifiers["mrnax_origin_source"][0],
                        license_track=feat.qualifiers["mrnax_license_track"][0],  # type: ignore[arg-type]
                        action_taken=feat.qualifiers["mrnax_action_taken"][0],  # type: ignore[arg-type]
                    )
                )
            elif layer == "annotation":
                parts = feat.location.parts if isinstance(feat.location, CompoundLocation) else [feat.location]
                intervals = [(int(p.start), int(p.end)) for p in parts]
                strand = -1 if feat.location.strand == -1 else 1
                cleaned_quals = {
                    k: list(v)
                    for k, v in feat.qualifiers.items()
                    if k not in {"mrnax_layer", "mrnax_feature_id"}
                }
                annotations.append(
                    FeatureAnnotation(
                        feature_id=feat.qualifiers["mrnax_feature_id"][0],
                        feature_key=feat.type,
                        intervals=intervals,
                        strand=strand,  # type: ignore[arg-type]
                        qualifiers=cleaned_quals,
                    )
                )

        segments.sort(key=lambda s: s.start)
        molecules.append(
            RNAMoleculeSpec(
                molecule_id=m_meta["molecule_id"],
                modality=m_meta["modality"],
                topology=m_meta["topology"],
                chemistry_track=ChemistryTrackSpec.model_validate(m_meta["chemistry_track"]),
                effective_translation_target=ProteinTargetSpec.model_validate(
                    m_meta["effective_translation_target"]
                ),
                sequence=rna_seq,
                segments=segments,
                annotations=annotations,
            )
        )

    reconstructed = ConstructBundle.create(
        bundle_id=chem_meta["bundle_id"],
        parent_bundle_id=chem_meta.get("parent_bundle_id"),
        derivation_action=chem_meta["derivation_action"],
        target_stoichiometry=chem_meta["target_stoichiometry"],
        co_expression_mode=chem_meta["co_expression_mode"],
        molecules=molecules,
    )
    if reconstructed.version_digest != chem_meta["version_digest"]:
        raise ValueError("Reconstructed ConstructBundle version_digest disagrees with chemistry_metadata.json")
    return reconstructed

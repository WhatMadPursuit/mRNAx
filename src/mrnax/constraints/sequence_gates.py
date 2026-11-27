"""Biological and manufacturing sequence hard gates with explicit segment-level scopes."""

from __future__ import annotations

import re

from Bio.Seq import Seq

from mrnax.constraints.frameshift_m1psi import find_m1psi_slippery_spans
from mrnax.schemas.common import ConstraintScope
from mrnax.schemas.construct_bundle import ConstructBundle, RNAMoleculeSpec
from mrnax.schemas.design_intent import ConstraintSpec
from mrnax.schemas.evaluation_report import AdvisoryNote, FailedCandidateDiagnostic

RESTRICTION_SITES_RNA: dict[str, tuple[str, str]] = {
    "BspQI": ("GCUCUUC", "GAAGAGC"),
    "SapI": ("GCUCUUC", "GAAGAGC"),
    "BsaI": ("GGUCUC", "GAGACC"),
    "BsmBI": ("CGUCUC", "GAGACG"),
    "BbsI": ("GAAGAC", "GUCUUC"),
    "EcoRI": ("GAAUUC", "GAAUUC"),
    "XbaI": ("UCUAGA", "UCUAGA"),
}

_CLASS_II_T7_TERMINATOR_RNA = "AUCUGUU"


def _iter_scoped_slices(
    mol: RNAMoleculeSpec, scopes: frozenset[ConstraintScope]
) -> list[tuple[ConstraintScope, int, str]]:
    """Return (scope_name, global_offset, substring) for all requested scopes on a molecule."""
    slices: list[tuple[ConstraintScope, int, str]] = []
    if "full_transcript" in scopes:
        slices.append(("full_transcript", 0, mol.sequence))
    if "three_prime_terminal" in scopes:
        tail_start = max(0, len(mol.sequence) - 45)
        slices.append(("three_prime_terminal", tail_start, mol.sequence[tail_start:]))
    for seg in mol.segments:
        if seg.segment_type in scopes:
            slices.append((seg.segment_type, seg.start, mol.sequence[seg.start : seg.end]))
    return slices


def check_homopolymer(
    mol: RNAMoleculeSpec, spec: ConstraintSpec, candidate_ref: str
) -> list[FailedCandidateDiagnostic]:
    max_run = int(spec.params.get("max_run", 7))
    pattern = re.compile(rf"(A{{{max_run + 1},}}|C{{{max_run + 1},}}|G{{{max_run + 1},}}|U{{{max_run + 1},}})")
    out: list[FailedCandidateDiagnostic] = []
    for scope, offset, sub in _iter_scoped_slices(mol, spec.target_scopes):
        for m in pattern.finditer(sub):
            out.append(
                FailedCandidateDiagnostic(
                    candidate_ref=candidate_ref,
                    violated_requirement_id=spec.requirement_id,
                    violated_scope=scope,
                    transcript_coordinates=(offset + m.start(), offset + m.end()),
                    reason=f"Homopolymer '{m.group(0)}' (len={len(m.group(0))} > {max_run}) in {scope}",
                )
            )
    return out


def check_uaug(
    mol: RNAMoleculeSpec, spec: ConstraintSpec, candidate_ref: str
) -> list[FailedCandidateDiagnostic]:
    out: list[FailedCandidateDiagnostic] = []
    for scope, offset, sub in _iter_scoped_slices(mol, spec.target_scopes):
        start = 0
        while True:
            idx = sub.find("AUG", start)
            if idx < 0:
                break
            out.append(
                FailedCandidateDiagnostic(
                    candidate_ref=candidate_ref,
                    violated_requirement_id=spec.requirement_id,
                    violated_scope=scope,
                    transcript_coordinates=(offset + idx, offset + idx + 3),
                    reason=f"Unexpected upstream start codon uAUG in {scope} at {offset + idx}",
                )
            )
            start = idx + 1
    return out


def check_m1psi_frameshift(
    mol: RNAMoleculeSpec, spec: ConstraintSpec, candidate_ref: str
) -> list[FailedCandidateDiagnostic]:
    if mol.chemistry_track.uridine_chemistry not in {"m1psi", "pseudouridine"}:
        return []
    out: list[FailedCandidateDiagnostic] = []
    for scope, offset, sub in _iter_scoped_slices(mol, spec.target_scopes):
        for s, e, motif in find_m1psi_slippery_spans(sub):
            out.append(
                FailedCandidateDiagnostic(
                    candidate_ref=candidate_ref,
                    violated_requirement_id=spec.requirement_id,
                    violated_scope=scope,
                    transcript_coordinates=(offset + s, offset + e),
                    reason=f"Mulroney 2024 m1Psi/Psi +1 frameshift slippery U-run '{motif}' in {scope}",
                )
            )
    return out


_RC005_ORIGIN = re.compile(r"(?<![a-z0-9])(?:codontransformer|rc[-_]?005)(?![a-z0-9])", re.I)


def has_rc005_cds_provenance(bundle: ConstructBundle) -> bool:
    return any(
        _RC005_ORIGIN.search(tag)
        for mol in bundle.molecules
        for seg in mol.segments if seg.segment_type == "cds"
        for tag in (seg.origin_source, seg.component_id or "")
    )


def check_rc005_chemistry(bundle: ConstructBundle, candidate_ref: str) -> list[FailedCandidateDiagnostic]:
    diagnostics = []
    for mol in bundle.molecules:
        if mol.chemistry_track.uridine_chemistry not in {"m1psi", "pseudouridine"}:
            continue
        for _, offset, cds in _iter_scoped_slices(mol, frozenset({"cds"})):
            spans = find_m1psi_slippery_spans(cds, include_uuuc=True)
            if spans:
                start, end, motif = spans[0]
                diagnostics.append(FailedCandidateDiagnostic(
                    candidate_ref=candidate_ref,
                    violated_requirement_id="rc005:m1psi_uuuc_u4_slippery_free",
                    violated_scope="cds",
                    transcript_coordinates=(offset + start, offset + end),
                    reason=f"RC-005 final admission blocked slippery motif '{motif}' on molecule '{mol.molecule_id}'.",
                ))
    return diagnostics


def check_restriction_sites(
    mol: RNAMoleculeSpec, spec: ConstraintSpec, candidate_ref: str
) -> list[FailedCandidateDiagnostic]:
    enzymes: list[str] = list(spec.params.get("enzymes", []))
    custom_motifs: list[str] = list(spec.params.get("forbidden_motifs", []))
    out: list[FailedCandidateDiagnostic] = []
    for scope, offset, sub in _iter_scoped_slices(mol, spec.target_scopes):
        for enz in enzymes:
            if enz not in RESTRICTION_SITES_RNA:
                raise ValueError(f"Unsupported restriction enzyme '{enz}'")
            fwd, rev = RESTRICTION_SITES_RNA[enz]
            for site in {fwd, rev}:
                idx = sub.find(site)
                if idx >= 0:
                    out.append(
                        FailedCandidateDiagnostic(
                            candidate_ref=candidate_ref,
                            violated_requirement_id=spec.requirement_id,
                            violated_scope=scope,
                            transcript_coordinates=(offset + idx, offset + idx + len(site)),
                            reason=f"Forbidden {enz} recognition site '{site}' in {scope}",
                        )
                    )
        for raw_m in custom_motifs:
            motif = raw_m.upper().replace("T", "U")
            idx = sub.find(motif)
            if idx >= 0:
                out.append(
                    FailedCandidateDiagnostic(
                        candidate_ref=candidate_ref,
                        violated_requirement_id=spec.requirement_id,
                        violated_scope=scope,
                        transcript_coordinates=(offset + idx, offset + idx + len(motif)),
                        reason=f"Forbidden motif '{motif}' in {scope}",
                    )
                )
    return out


def check_t7_terminator(
    mol: RNAMoleculeSpec, spec: ConstraintSpec, candidate_ref: str
) -> list[FailedCandidateDiagnostic]:
    out: list[FailedCandidateDiagnostic] = []
    for scope, offset, sub in _iter_scoped_slices(mol, spec.target_scopes):
        idx = sub.find(_CLASS_II_T7_TERMINATOR_RNA)
        if idx >= 0:
            out.append(
                FailedCandidateDiagnostic(
                    candidate_ref=candidate_ref,
                    violated_requirement_id=spec.requirement_id,
                    violated_scope=scope,
                    transcript_coordinates=(offset + idx, offset + idx + len(_CLASS_II_T7_TERMINATOR_RNA)),
                    reason=f"Class II T7 pause/termination signal '{_CLASS_II_T7_TERMINATOR_RNA}' in {scope}",
                )
            )
    return out


def _has_exact_reverse_complement_kmer(seq_a: str, seq_b: str, k: int) -> tuple[int, int] | None:
    if len(seq_a) < k or len(seq_b) < k:
        return None
    kmers_b = {seq_b[j : j + k] for j in range(len(seq_b) - k + 1)}
    for i in range(len(seq_a) - k + 1):
        rc = str(Seq(seq_a[i : i + k]).reverse_complement_rna())
        if rc in kmers_b:
            return (i, i + k)
    return None


def check_dsrna_and_3p_loopback(
    bundle: ConstructBundle, spec: ConstraintSpec, candidate_ref: str
) -> list[FailedCandidateDiagnostic]:
    max_stem = int(spec.params.get("max_perfect_stem_bp", 25))
    loopback_k = int(spec.params.get("loopback_3p_bp", 10))
    out: list[FailedCandidateDiagnostic] = []

    for mol in bundle.molecules:
        if "full_transcript" in spec.target_scopes and len(mol.sequence) >= 2 * max_stem:
            # Check non-overlapping intramolecular perfect reverse complement >= max_stem
            seq = mol.sequence
            kmer_pos: dict[str, int] = {}
            for i in range(len(seq) - max_stem + 1):
                sub = seq[i : i + max_stem]
                rc = str(Seq(sub).reverse_complement_rna())
                if rc in kmer_pos and (i - kmer_pos[rc]) >= max_stem:
                    out.append(
                        FailedCandidateDiagnostic(
                            candidate_ref=candidate_ref,
                            violated_requirement_id=spec.requirement_id,
                            violated_scope="full_transcript",
                            transcript_coordinates=(i, i + max_stem),
                            reason=f"Intramolecular perfect dsRNA stem >= {max_stem} bp detected",
                        )
                    )
                    break
                kmer_pos.setdefault(sub, i)

        if "three_prime_terminal" in spec.target_scopes and len(mol.sequence) >= 30:
            # Check if the very 3'-OH end (last loopback_k nt) base-pairs perfectly inside upstream 60 nt
            end_kmer = mol.sequence[-loopback_k:]
            upstream_win = mol.sequence[max(0, len(mol.sequence) - 80) : -loopback_k]
            rc_end = str(Seq(end_kmer).reverse_complement_rna())
            if rc_end in upstream_win:
                out.append(
                    FailedCandidateDiagnostic(
                        candidate_ref=candidate_ref,
                        violated_requirement_id=spec.requirement_id,
                        violated_scope="three_prime_terminal",
                        transcript_coordinates=(len(mol.sequence) - loopback_k, len(mol.sequence)),
                        reason=f"IVT 3' loop-back self-priming risk ({loopback_k}-bp terminal complement)",
                    )
                )

    if "inter_chain" in spec.target_scopes and len(bundle.molecules) > 1:
        for idx_a in range(len(bundle.molecules)):
            for idx_b in range(idx_a + 1, len(bundle.molecules)):
                hit = _has_exact_reverse_complement_kmer(
                    bundle.molecules[idx_a].sequence,
                    bundle.molecules[idx_b].sequence,
                    max_stem,
                )
                if hit is not None:
                    out.append(
                        FailedCandidateDiagnostic(
                            candidate_ref=candidate_ref,
                            violated_requirement_id=spec.requirement_id,
                            violated_scope="inter_chain",
                            transcript_coordinates=hit,
                            reason=f"Inter-chain antisense duplex >= {max_stem} bp between molecules",
                        )
                    )
    return out


def collect_soft_advisories(bundle: ConstructBundle) -> list[AdvisoryNote]:
    """Appendix A non-blocking pre-synthesis reminders (5' cap AG/GG, PAS notes)."""
    notes: list[AdvisoryNote] = []
    for mol in bundle.molecules:
        cap = mol.chemistry_track.cap_chemistry
        dinuc = mol.sequence[:2]
        if cap == "cap1_cleancap_ag" and dinuc != "AG":
            notes.append(
                AdvisoryNote(
                    advisory_id="adv_cap_cleancap_ag",
                    category="cap_5prime_dinucleotide",
                    message=(
                        f"CleanCap AG typically initiates with 5'-AG (current transcript starts with 5'-{dinuc}); "
                        "confirm T7 promoter +1/+2 initiation dinucleotide before IVT template synthesis."
                    ),
                    transcript_coordinates=(0, 2),
                )
            )
        elif cap == "cap1_arca_gg" and dinuc != "GG":
            notes.append(
                AdvisoryNote(
                    advisory_id="adv_cap_arca_gg",
                    category="cap_5prime_dinucleotide",
                    message=f"ARCA capping typically uses 5'-GG initiation (current starts with 5'-{dinuc}).",
                    transcript_coordinates=(0, 2),
                )
            )

        cds_seg = mol.get_segment("cds")
        if cds_seg is not None:
            cds_seq = mol.sequence[cds_seg.start : cds_seg.end]
            pas_idx = cds_seq.find("AAUAAA")
            if pas_idx >= 0:
                notes.append(
                    AdvisoryNote(
                        advisory_id="adv_cds_internal_pas",
                        category="cds_internal_pas",
                        message="Internal canonical AAUAAA hexamer inside CDS (soft advisory only for cytoplasmic IVT mRNA).",
                        transcript_coordinates=(cds_seg.start + pas_idx, cds_seg.start + pas_idx + 6),
                    )
                )

        utr3_seg = mol.get_segment("three_prime_utr")
        if utr3_seg is not None:
            utr3_seq = mol.sequence[utr3_seg.start : utr3_seg.end]
            pas3 = utr3_seq.find("AAUAAA")
            if pas3 >= 0:
                notes.append(
                    AdvisoryNote(
                        advisory_id="adv_utr3_pas_retained",
                        category="utr3_canonical_pas",
                        message="Endogenous AAUAAA present in 3'UTR (normal in human globin/housekeeping UTRs; non-blocking).",
                        transcript_coordinates=(utr3_seg.start + pas3, utr3_seg.start + pas3 + 6),
                    )
                )
    return notes

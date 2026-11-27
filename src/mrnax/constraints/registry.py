"""Single-source-of-truth Scoped Constraint Engine shared by Generation, Filtering, and Export."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mrnax.constraints.fto_guard import SequenceFTOGuard
from mrnax.constraints.sequence_gates import (
    check_dsrna_and_3p_loopback,
    check_homopolymer,
    check_m1psi_frameshift,
    check_restriction_sites,
    check_t7_terminator,
    check_uaug,
    collect_soft_advisories,
)
from mrnax.schemas.common import ComplianceProfile, UridineChemistry
from mrnax.schemas.construct_bundle import ConstructBundle
from mrnax.schemas.design_intent import ConstraintSpec, LockedRegionSpec
from mrnax.schemas.evaluation_report import AdvisoryNote, FailedCandidateDiagnostic, FTOAlert


@dataclass(frozen=True)
class ConstraintEvaluationOutcome:
    passed_all_hard_gates: bool
    fto_blocked: bool
    violations: list[FailedCandidateDiagnostic]
    fto_alerts: list[FTOAlert]
    advisories: list[AdvisoryNote]


def default_m1_constraints(
    *,
    max_homopolymer_run: int = 7,
    restriction_enzymes: list[str] | None = None,
    forbidden_motifs: list[str] | None = None,
) -> list[ConstraintSpec]:
    """Canonical M1 drug-grade hard constraints with explicit segment scopes."""
    specs: list[ConstraintSpec] = [
        ConstraintSpec(
            requirement_id="gate_uaug_5utr",
            constraint_kind="uaug_free",
            target_scopes=frozenset({"five_prime_utr"}),
            severity="hard_gate",
            source_type="system_safety_gate",
            source_locator="mrnax:safety:uaug_v1",
            params={},
        ),
        ConstraintSpec(
            requirement_id="gate_homopolymer_max",
            constraint_kind="homopolymer_max",
            # Explicitly excludes `poly_a` segment so legal poly(A) tails never false-positive!
            target_scopes=frozenset({"five_prime_utr", "cds", "stop_cassette", "three_prime_utr"}),
            severity="hard_gate",
            source_type="system_safety_gate",
            source_locator="mrnax:safety:homopolymer_v1",
            params={"max_run": max_homopolymer_run},
        ),
        ConstraintSpec(
            requirement_id="gate_m1psi_frameshift",
            constraint_kind="m1psi_slippery_free",
            target_scopes=frozenset({"cds"}),
            severity="hard_gate",
            source_type="system_safety_gate",
            source_locator="doi:10.1038/s41586-023-06800-3 (Mulroney et al. 2024)",
            params={},
        ),
        ConstraintSpec(
            requirement_id="gate_t7_terminator",
            constraint_kind="t7_class2_terminator_free",
            target_scopes=frozenset({"five_prime_utr", "cds", "stop_cassette", "three_prime_utr"}),
            severity="hard_gate",
            source_type="system_safety_gate",
            source_locator="mrnax:ivt:t7_terminator_v1",
            params={},
        ),
        ConstraintSpec(
            requirement_id="gate_dsrna_and_loopback",
            constraint_kind="dsrna_and_3p_loopback",
            target_scopes=frozenset({"full_transcript", "three_prime_terminal", "inter_chain"}),
            severity="hard_gate",
            source_type="system_safety_gate",
            source_locator="mrnax:safety:dsrna_loopback_v1",
            params={"max_perfect_stem_bp": 25, "loopback_3p_bp": 10},
        ),
    ]
    enz = ["BspQI"] if restriction_enzymes is None else restriction_enzymes
    motifs = forbidden_motifs or []
    if enz or motifs:
        specs.append(
            ConstraintSpec(
                requirement_id="gate_type_iis_restriction",
                constraint_kind="type_iis_restriction_free",
                target_scopes=frozenset({"full_transcript"}),
                severity="hard_gate",
                source_type="user_explicit" if restriction_enzymes is not None else "system_safety_gate",
                source_locator="mrnax:ivt:linearization_enzymes_v1",
                params={"enzymes": enz, "forbidden_motifs": motifs},
            )
        )
    return specs


class ScopedConstraintEngine:
    """Executes identical `target_scopes` logic across generation, filtering, and export."""

    def __init__(self, fto_guard: SequenceFTOGuard, *, seed_dir: Path | None = None) -> None:
        self.fto_guard = fto_guard
        self.seed_dir = seed_dir

    @classmethod
    def from_seed_dir(cls, seed_dir: Path) -> ScopedConstraintEngine:
        v2_path = seed_dir / "patent_fto_guard_v2.json"
        if not v2_path.is_file():
            raise FileNotFoundError(
                f"Mandatory patent_fto_guard_v2.json missing at '{v2_path}'; refusing silent downgrade to v1."
            )
        return cls(fto_guard=SequenceFTOGuard.from_json_path(v2_path), seed_dir=seed_dir)

    def evaluate_bundle(
        self,
        bundle: ConstructBundle,
        constraints: list[ConstraintSpec],
        *,
        compliance_track: ComplianceProfile,
        candidate_ref: str | None = None,
        expected_chemistry: UridineChemistry | None = None,
        locked_scopes: dict[str, LockedRegionSpec] | None = None,
        expected_protein_targets: list[Any] | None = None,
    ) -> ConstraintEvaluationOutcome:
        ref_id = candidate_ref or bundle.bundle_id
        violations: list[FailedCandidateDiagnostic] = []

        if expected_protein_targets is not None and len(expected_protein_targets) >= 1:
            if len(bundle.molecules) != len(expected_protein_targets):
                first_len = len(bundle.molecules[0].sequence) if bundle.molecules else 0
                violations.append(
                    FailedCandidateDiagnostic(
                        candidate_ref=ref_id,
                        violated_requirement_id="intent:target_chain_count",
                        violated_scope="full_transcript",
                        transcript_coordinates=(0, first_len),
                        reason=(
                            f"Candidate has {len(bundle.molecules)} molecule(s), but DesignIntent requires "
                            f"{len(expected_protein_targets)} target chain(s)"
                        ),
                    )
                )
            else:
                for idx, (mol, tgt) in enumerate(zip(bundle.molecules, expected_protein_targets, strict=True)):
                    exp_seq = getattr(tgt, "protein_sequence", None) or (
                        tgt.get("protein_sequence") if isinstance(tgt, dict) else None
                    )
                    actual_translated = mol.translate_cds_protein()
                    if exp_seq and (
                        mol.effective_translation_target.protein_sequence != exp_seq
                        or actual_translated != exp_seq
                    ):
                        violations.append(
                            FailedCandidateDiagnostic(
                                candidate_ref=ref_id,
                                violated_requirement_id=f"intent:target_chain_{idx + 1}_protein",
                                violated_scope="cds",
                                transcript_coordinates=(0, len(mol.sequence)),
                                reason=f"Chain {idx + 1} translated CDS protein does not match DesignIntent target",
                            )
                        )

        # Enforce Clean-Room segment license provenance, expected chemistry, and explicit locked_scopes
        for mol_idx, mol in enumerate(bundle.molecules):
            if expected_chemistry is not None and mol.chemistry_track.uridine_chemistry != expected_chemistry:
                violations.append(
                    FailedCandidateDiagnostic(
                        candidate_ref=ref_id,
                        violated_requirement_id="intent:chemistry_lock",
                        violated_scope="full_transcript",
                        transcript_coordinates=(0, len(mol.sequence)),
                        reason=(
                            f"Candidate uridine_chemistry '{mol.chemistry_track.uridine_chemistry}' "
                            f"contradicts required intent chemistry '{expected_chemistry}'"
                        ),
                    )
                )
            for seg in mol.segments:
                if compliance_track == "commercial_clean" and seg.license_track != "commercial_clean":
                    violations.append(
                        FailedCandidateDiagnostic(
                            candidate_ref=ref_id,
                            violated_requirement_id="clean_room:segment_license_track",
                            violated_scope=seg.segment_type,
                            transcript_coordinates=(seg.start, seg.end),
                            reason=(
                                f"Segment '{seg.segment_type}' has license_track='{seg.license_track}', "
                                "which is forbidden in commercial_clean"
                            ),
                        )
                    )
            for scope_name, lock_spec in (locked_scopes or {}).items():
                if scope_name == "protein_sequence":
                    expected_aa = lock_spec.locked_value
                    if len(bundle.molecules) > 1 and "|" in expected_aa:
                        parts = expected_aa.split("|")
                        expected_aa = parts[mol_idx] if mol_idx < len(parts) else "__MISMATCH__"
                    if mol.effective_translation_target.protein_sequence != expected_aa:
                        violations.append(
                            FailedCandidateDiagnostic(
                                candidate_ref=ref_id,
                                violated_requirement_id="intent:lock:protein_sequence",
                                violated_scope="cds",
                                transcript_coordinates=(0, len(mol.sequence)),
                                reason="Translated protein violates locked protein_sequence",
                            )
                        )
                else:
                    seg = mol.get_segment(scope_name)  # type: ignore[arg-type]
                    actual_sub = "" if seg is None else mol.sequence[seg.start : seg.end]
                    if actual_sub != lock_spec.locked_value:
                        coords = (0, len(mol.sequence)) if seg is None else (seg.start, seg.end)
                        violations.append(
                            FailedCandidateDiagnostic(
                                candidate_ref=ref_id,
                                violated_requirement_id=f"intent:lock:{scope_name}",
                                violated_scope=scope_name,  # type: ignore[arg-type]
                                transcript_coordinates=coords,
                                reason=f"Segment '{scope_name}' violates explicit user lock",
                            )
                        )

        for spec in constraints:
            kind = spec.constraint_kind
            spec_violations: list[FailedCandidateDiagnostic] = []
            if kind == "homopolymer_max":
                for mol in bundle.molecules:
                    spec_violations.extend(check_homopolymer(mol, spec, ref_id))
            elif kind == "uaug_free":
                for mol in bundle.molecules:
                    spec_violations.extend(check_uaug(mol, spec, ref_id))
            elif kind == "m1psi_slippery_free":
                for mol in bundle.molecules:
                    spec_violations.extend(check_m1psi_frameshift(mol, spec, ref_id))
            elif kind == "type_iis_restriction_free":
                for mol in bundle.molecules:
                    spec_violations.extend(check_restriction_sites(mol, spec, ref_id))
            elif kind == "t7_class2_terminator_free":
                for mol in bundle.molecules:
                    spec_violations.extend(check_t7_terminator(mol, spec, ref_id))
            elif kind == "dsrna_and_3p_loopback":
                spec_violations.extend(check_dsrna_and_3p_loopback(bundle, spec, ref_id))
            elif kind == "gc_window_bounds":
                min_gc = float(spec.params.get("min_gc", 0.30))
                max_gc = float(spec.params.get("max_gc", 0.75))
                win = int(spec.params.get("window", 50))
                for mol in bundle.molecules:
                    regions_to_scan: list[tuple[str, int, str]] = []
                    if "full_transcript" in spec.target_scopes:
                        regions_to_scan.append(("full_transcript", 0, mol.sequence))
                    if "cds_utr_junction" in spec.target_scopes:
                        u5_seg = mol.get_segment("five_prime_utr")
                        cds_seg = mol.get_segment("cds")
                        u3_seg = mol.get_segment("three_prime_utr")
                        if u5_seg is not None and cds_seg is not None:
                            j1_s = max(u5_seg.start, cds_seg.start - win)
                            j1_e = min(cds_seg.end, cds_seg.start + win)
                            regions_to_scan.append(("cds_utr_junction", j1_s, mol.sequence[j1_s:j1_e]))
                        if cds_seg is not None and u3_seg is not None:
                            j2_s = max(cds_seg.start, cds_seg.end - win)
                            j2_e = min(u3_seg.end, u3_seg.start + win)
                            regions_to_scan.append(("cds_utr_junction", j2_s, mol.sequence[j2_s:j2_e]))
                    for seg in mol.segments:
                        if seg.segment_type in spec.target_scopes:
                            regions_to_scan.append((seg.segment_type, seg.start, mol.sequence[seg.start : seg.end]))
                    for scope_lbl, base_offset, sub in regions_to_scan:
                        if not sub:
                            continue
                        for i in range(0, max(1, len(sub) - win + 1)):
                            chunk = sub[i : i + min(win, len(sub))]
                            gc = (chunk.count("G") + chunk.count("C")) / len(chunk)
                            if gc < min_gc or gc > max_gc:
                                spec_violations.append(
                                    FailedCandidateDiagnostic(
                                        candidate_ref=ref_id,
                                        violated_requirement_id=spec.requirement_id,
                                        violated_scope=scope_lbl,  # type: ignore[arg-type]
                                        transcript_coordinates=(base_offset + i, base_offset + i + len(chunk)),
                                        reason=f"GC fraction {gc:.3f} outside [{min_gc:.2f}, {max_gc:.2f}] in {scope_lbl}",
                                    )
                                )
                                break
            else:
                raise ValueError(f"Unknown constraint_kind '{kind}'")

            if spec.severity == "hard_gate":
                violations.extend(spec_violations)

        fto_alerts = self.fto_guard.scan_bundle(bundle, compliance_track)
        fto_blocked = any(a.blocked for a in fto_alerts)
        for alert in fto_alerts:
            if alert.blocked:
                violations.append(
                    FailedCandidateDiagnostic(
                        candidate_ref=ref_id,
                        violated_requirement_id=f"fto:{alert.rule_id}",
                        violated_scope=alert.matched_scope,
                        transcript_coordinates=alert.transcript_coordinates,
                        reason=f"Patent FTO guard blocked ({alert.patent_family_or_pub}): {alert.summary}",
                    )
                )

        advisories = collect_soft_advisories(bundle)
        return ConstraintEvaluationOutcome(
            passed_all_hard_gates=(len(violations) == 0),
            fto_blocked=fto_blocked,
            violations=violations,
            fto_alerts=fto_alerts,
            advisories=advisories,
        )

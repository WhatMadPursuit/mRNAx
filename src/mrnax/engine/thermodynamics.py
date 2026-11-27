"""ViennaRNA v2.8.0 sliding-window, cross-junction, and RNAup accessibility evaluator."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from mrnax.schemas.common import ChemistryEvalMode, UridineChemistry
from mrnax.schemas.construct_bundle import RNAMoleculeSpec
from mrnax.schemas.evaluation_report import AdvisoryNote, MetricResult

RT_37C_KCAL_MOL = 0.616320775  # R * 310.15 K in kcal/mol


def _rna() -> Any:
    """Lazy-load ViennaRNA (`RNA`) only when research_max thermodynamics is actually invoked."""
    import RNA  # type: ignore[import-untyped]

    return RNA


@dataclass(frozen=True)
class MiRNAAccessibilityScore:
    dg_duplex_kcal_mol: float
    dg_open_kcal_mol: float
    dg_total_kcal_mol: float
    seed_8mer_unpaired_prob: float


def chemistry_eval_mode_for_vienna(uridine_chemistry: UridineChemistry) -> ChemistryEvalMode:
    if uridine_chemistry == "unmodified_u":
        return "native_chemistry_validated"
    if uridine_chemistry in {"m1psi", "pseudouridine"}:
        return "cross_chemistry_thermodynamic_approximation"
    return "not_evaluated"


def compute_kozak_unpaired_probability(mol: RNAMoleculeSpec) -> float:
    """Compute mean ensemble unpaired probability around translation initiation [-15, +18] via pfl_fold_up."""
    cds_seg = mol.get_segment("cds")
    if cds_seg is None:
        return 0.0
    win_start = max(0, cds_seg.start - 35)
    win_end = min(len(mol.sequence), cds_seg.start + 55)
    sub = mol.sequence[win_start:win_end]
    if len(sub) < 12:
        return 0.5

    w_size = min(len(sub), 70)
    l_span = min(len(sub), 50)
    up = _rna().pfl_fold_up(sub, 1, l_span, w_size)

    rel_aug = cds_seg.start - win_start
    probe_lo = max(0, rel_aug - 15)
    probe_hi = min(len(sub), rel_aug + 18)
    probs = [float(up[pos + 1][1]) for pos in range(probe_lo, probe_hi)]
    return round(sum(probs) / max(1, len(probs)), 6)


def compute_cds_sliding_window_mfe(cds_rna: str, *, window: int = 120, step: int = 40) -> float:
    """Mean length-normalized local MFE (kcal/mol per 100 nt; more negative = stronger local structure)."""
    if len(cds_rna) <= window:
        _, mfe = _rna().fold(cds_rna)
        return round(float(mfe) * (100.0 / max(1, len(cds_rna))), 4)

    vals: list[float] = []
    for i in range(0, len(cds_rna) - window + 1, step):
        chunk = cds_rna[i : i + window]
        _, mfe = _rna().fold(chunk)
        vals.append(float(mfe) * (100.0 / window))
    return round(sum(vals) / len(vals), 4)


def compute_junction_interference_mfe(mol: RNAMoleculeSpec) -> float:
    """Evaluate 5'UTR..CDS and CDS..3'UTR junction folding energy."""
    cds_seg = mol.get_segment("cds")
    if cds_seg is None:
        return 0.0
    j5 = mol.sequence[max(0, cds_seg.start - 30) : min(len(mol.sequence), cds_seg.start + 35)]
    j3 = mol.sequence[max(0, cds_seg.end - 35) : min(len(mol.sequence), cds_seg.end + 45)]
    _, e5 = _rna().fold(j5)
    _, e3 = _rna().fold(j3)
    return round(float(e5) + float(e3), 4)


def compute_rnaup_mirna_interaction(
    context_rna: str,
    target_site_start_in_context: int,
    target_site_len: int,
    mirna_mature_rna: str,
) -> MiRNAAccessibilityScore:
    """Exact statistical-mechanical RNAup interaction: dG_total = dG_duplex - RT * ln(P_unpaired)."""
    target_sub = context_rna[
        target_site_start_in_context : target_site_start_in_context + target_site_len
    ]
    dup = _rna().duplexfold(target_sub, mirna_mature_rna)
    dg_duplex = float(dup.energy)

    u_max = min(target_site_len, 24)
    w_size = min(len(context_rna), 120)
    l_span = min(len(context_rna), 90)
    up = _rna().pfl_fold_up(context_rna, u_max, l_span, w_size)

    end_1based = target_site_start_in_context + target_site_len
    p_site_unpaired = max(float(up[end_1based][u_max]), 1e-12)
    dg_open = -RT_37C_KCAL_MOL * math.log(p_site_unpaired)

    # 8-mer seed binds at the 3' end of the target box (opposite miRNA 5' positions 1..8)
    seed_u = min(8, u_max)
    p_seed_unpaired = max(float(up[end_1based][seed_u]), 1e-12)

    return MiRNAAccessibilityScore(
        dg_duplex_kcal_mol=round(dg_duplex, 4),
        dg_open_kcal_mol=round(dg_open, 4),
        dg_total_kcal_mol=round(dg_duplex + dg_open, 4),
        seed_8mer_unpaired_prob=round(p_seed_unpaired, 6),
    )


def evaluate_molecule_thermodynamics(
    mol: RNAMoleculeSpec,
    *,
    mirna_mature_for_detarget: str | None = None,
    include_full_mfe: bool = False,
) -> tuple[list[MetricResult], list[AdvisoryNote]]:
    chem_mode = chemistry_eval_mode_for_vienna(mol.chemistry_track.uridine_chemistry)
    advisories: list[AdvisoryNote] = []
    if chem_mode == "cross_chemistry_thermodynamic_approximation":
        advisories.append(
            AdvisoryNote(
                advisory_id="adv_vienna_m1psi_approx",
                category="chemistry_approximation",
                message=(
                    f"Chemistry '{mol.chemistry_track.uridine_chemistry}' evaluated with standard Turner 2004 "
                    "unmodified-U nearest-neighbor parameters (cross_chemistry_thermodynamic_approximation)."
                ),
            )
        )

    kozak_p = compute_kozak_unpaired_probability(mol)
    cds_seq = mol.get_segment_sequence("cds")
    cds_mfe = compute_cds_sliding_window_mfe(cds_seq)
    junc_mfe = compute_junction_interference_mfe(mol)

    metrics = [
        MetricResult(
            metric_id="kozak_unpaired_prob",
            value=kozak_p,
            unit="probability",
            direction="maximize",
            evaluated_scopes=["five_prime_utr", "cds"],
            tool_name="ViennaRNA",
            tool_version=_rna().__version__,
            chemistry_evaluation_mode=chem_mode,
            participated_in="generation_guidance",
        ),
        MetricResult(
            metric_id="cds_mean_window_mfe",
            value=cds_mfe,
            unit="kcal/mol/100nt",
            direction="minimize",
            evaluated_scopes=["cds"],
            tool_name="ViennaRNA",
            tool_version=_rna().__version__,
            chemistry_evaluation_mode=chem_mode,
            participated_in="pareto_selection_or_ranking",
        ),
        MetricResult(
            metric_id="junction_folding_mfe",
            value=junc_mfe,
            unit="kcal/mol",
            direction="maximize",
            evaluated_scopes=["five_prime_utr", "cds", "three_prime_utr"],
            tool_name="ViennaRNA",
            tool_version=_rna().__version__,
            chemistry_evaluation_mode=chem_mode,
            participated_in="pareto_selection_or_ranking",
        ),
    ]

    if include_full_mfe:
        _, full_e = _rna().fold(mol.sequence)
        metrics.append(
            MetricResult(
                metric_id="full_transcript_mfe",
                value=round(float(full_e), 3),
                unit="kcal/mol",
                direction="informational",
                evaluated_scopes=["full_transcript"],
                tool_name="ViennaRNA",
                tool_version=_rna().__version__,
                chemistry_evaluation_mode=chem_mode,
                participated_in="display_only",
            )
        )

    if mirna_mature_for_detarget is not None:
        mir_anns = [a for a in mol.annotations if a.feature_key == "misc_binding"]
        if mir_anns:
            s_start, s_end = mir_anns[0].intervals[0]
            ctx_lo = max(0, s_start - 55)
            ctx_hi = min(len(mol.sequence), s_end + 35)
            acc = compute_rnaup_mirna_interaction(
                mol.sequence[ctx_lo:ctx_hi],
                s_start - ctx_lo,
                s_end - s_start,
                mirna_mature_for_detarget,
            )
            metrics.append(
                MetricResult(
                    metric_id="mirna_detarget_dg_total",
                    value=acc.dg_total_kcal_mol,
                    unit="kcal/mol",
                    direction="minimize",
                    evaluated_scopes=["cds", "three_prime_utr"],
                    tool_name="ViennaRNA",
                    tool_version=_rna().__version__,
                    chemistry_evaluation_mode=chem_mode,
                    participated_in="generation_guidance",
                    notes=f"seed_8mer_unpaired_prob={acc.seed_8mer_unpaired_prob:.4f}",
                )
            )

    return metrics, advisories

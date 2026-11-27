"""Stage 0 baselines: Random Synonymous Sampler & DNA Chisel MIT constraint/codon optimizer."""

from __future__ import annotations

import functools
import json
import logging
import random
from pathlib import Path

from dnachisel import (
    AvoidPattern,
    DnaOptimizationProblem,
    EnforceGCContent,
    EnforceTranslation,
    CodonOptimize,
)

from mrnax.constraints.frameshift_m1psi import remediate_m1psi_slippery_runs
from mrnax.constraints.sequence_gates import RESTRICTION_SITES_RNA
from mrnax.schemas.common import UridineChemistry
from mrnax.schemas.design_intent import ConstraintSpec

logging.getLogger("proglog").setLevel(logging.ERROR)

# Standard human-preferred synonymous RNA codons (excluding rare/slippery runs where possible)
AA_TO_SYNONYMOUS_RNA_CODONS: dict[str, tuple[str, ...]] = {
    "A": ("GCC", "GCU", "GCA", "GCG"),
    "C": ("UGC", "UGU"),
    "D": ("GAC", "GAU"),
    "E": ("GAG", "GAA"),
    "F": ("UUC", "UUU"),
    "G": ("GGC", "GGA", "GGG", "GGU"),
    "H": ("CAC", "CAU"),
    "I": ("AUC", "AUU", "AUA"),
    "K": ("AAG", "AAA"),
    "L": ("CUG", "CUC", "CUU", "CUA", "UUG", "UUA"),
    "M": ("AUG",),
    "N": ("AAC", "AAU"),
    "P": ("CCC", "CCU", "CCA", "CCG"),
    "Q": ("CAG", "CAA"),
    "R": ("CGG", "CGC", "AGA", "AGG", "CGU", "CGA"),
    "S": ("AGC", "UCC", "UCU", "UCA", "UCG", "AGU"),
    "T": ("ACC", "ACU", "ACA", "ACG"),
    "V": ("GUG", "GUC", "GUU", "GUA"),
    "W": ("UGG",),
    "Y": ("UAC", "UAU"),
}


def generate_random_synonymous_cds(
    protein_seq: str,
    rng: random.Random,
    *,
    chemistry: UridineChemistry,
    slowdown_codon_ranges: list[tuple[int, int]] | None = None,
) -> str:
    slowdown = slowdown_codon_ranges or []
    codons: list[str] = []
    for idx, aa in enumerate(protein_seq):
        choices = AA_TO_SYNONYMOUS_RNA_CODONS[aa]
        in_slowdown = any(lo <= idx < hi for lo, hi in slowdown)
        if in_slowdown and len(choices) > 1:
            # Prefer sub-maximal / moderate elongation codons in co-translational folding slowdown zones
            codons.append(choices[-1])
        else:
            codons.append(rng.choice(choices))
    cds_rna = "".join(codons)
    if chemistry in {"m1psi", "pseudouridine"}:
        cds_rna, _ = remediate_m1psi_slippery_runs(cds_rna)
    return cds_rna


@functools.lru_cache(maxsize=2)
def _load_bicodon_atlas() -> tuple[dict[str, float], dict[str, float]]:
    from mrnax.engine.profiles import resolve_default_seed_dir

    atlas_path = resolve_default_seed_dir() / "human_codon_bicodon_stability_atlas_v1.json"
    doc = json.loads(atlas_path.read_text(encoding="utf-8"))
    csc = {k: float(v["csc_spearman"]) for k, v in doc["codon_stabilization_coefficients"].items()}
    bic = {k: float(v["stability_top10_vs_bot10_log2"]) for k, v in doc["bicodon_scores"].items()}
    return csc, bic


def _extract_forbidden_rna_motifs(
    constraints: list[ConstraintSpec] | None,
    chemistry: UridineChemistry,
    *,
    junction_only: bool = False,
) -> tuple[str, ...]:
    motifs: list[str] = ["AUCUGUU"]  # Class II T7 terminator
    if not junction_only and chemistry in {"m1psi", "pseudouridine"}:
        motifs.append("UUUU")  # RUL-CC-M1PSI-FRAMESHIFT-009 slippery run
    max_run: int | None = None if junction_only else 7
    for spec in constraints or []:
        if junction_only:
            if "full_transcript" not in spec.target_scopes:
                continue
        elif "cds" not in spec.target_scopes and "full_transcript" not in spec.target_scopes:
            continue
        if spec.constraint_kind == "homopolymer_max":
            run_cfg = int(spec.params.get("max_run", spec.params.get("max_len", 7)))
            max_run = run_cfg if max_run is None else min(max_run, run_cfg)
        elif spec.constraint_kind == "type_iis_restriction_free":
            for enz in spec.params.get("enzymes", []):
                if enz in RESTRICTION_SITES_RNA:
                    fwd, rev = RESTRICTION_SITES_RNA[enz]
                    motifs.append(fwd)
                    motifs.append(rev)
            for raw_m in spec.params.get("forbidden_motifs", []):
                if raw_m:
                    motifs.append(str(raw_m).upper().replace("T", "U"))
    if max_run is not None:
        for base in "ACGU":
            motifs.append(base * (max_run + 1))
    return tuple(dict.fromkeys(motifs))


def propose_non_partitioned_bicodon_cds(
    cds_rna: str,
    protein_seq: str,
    rng: random.Random,
    *,
    chemistry: UridineChemistry = "m1psi",
    constraints: list[ConstraintSpec] | None = None,
    slowdown_codon_ranges: list[tuple[int, int]] | None = None,
    utr5_tail: str = "",
    stop_head: str = "UGA",
    gc_weight: float = 0.025,
    temperature: float = 0.08,
    target_gc: float = 0.565,
    mutation_prob: float = 0.60,
) -> str:
    """Continuous full-CDS non-partitioned GC-homeostatic bicodon + CSC + RBP proposal operator.

    Scores both `(prev_c -> c)` and `(c -> next_c)` across all 3,721 human bicodons and 61 CSC
    values in a continuous pass over `[1..n_codons)` without any two-zone initiation/body
    partition (`CN119265214B` Method-FTO separation) or DFA-CFG lattice intersection, maintaining
    endogenous human GC homeostasis (~54-59%) and preventing Type IIS sites, homopolymers, and
    m1Psi slippery runs inline.
    """
    import math

    codons = [cds_rna[i : i + 3] for i in range(0, len(cds_rna), 3)]
    n_codons = len(protein_seq)
    if len(codons) != n_codons or n_codons <= 1:
        return cds_rna
    csc_map, bic_map = _load_bicodon_atlas()
    forbidden_cds = _extract_forbidden_rna_motifs(constraints, chemistry, junction_only=False)
    forbidden_junc = _extract_forbidden_rna_motifs(constraints, chemistry, junction_only=True)
    slowdown = slowdown_codon_ranges or []
    u5_ctx = utr5_tail[-12:] if utr5_tail else ""
    st_ctx = stop_head[:12] if stop_head else ""

    gc_count = sum(c.count("G") + c.count("C") for c in codons)
    total_nt = max(3, 3 * n_codons)

    for idx in range(1, n_codons):
        aa = protein_seq[idx]
        choices = AA_TO_SYNONYMOUS_RNA_CODONS[aa]
        if len(choices) <= 1:
            continue
        if any(lo <= idx < hi for lo, hi in slowdown):
            continue
        cur_c = codons[idx]
        cds_window = "".join(codons[max(0, idx - 4) : min(n_codons, idx + 5)])
        cur_local = (
            (u5_ctx if idx <= 4 else "")
            + cds_window
            + (st_ctx if idx >= n_codons - 4 else "")
        )
        must_fix = any(pat in cds_window for pat in forbidden_cds) or any(
            pat in cur_local for pat in forbidden_junc
        )
        if not must_fix and rng.random() > mutation_prob:
            continue

        prev_c = codons[idx - 1]
        next_c = codons[idx + 1] if idx + 1 < n_codons else st_ctx[:3]
        cds_left = "".join(codons[max(0, idx - 4) : idx])
        cds_right = "".join(codons[idx + 1 : min(n_codons, idx + 5)])
        left_flank = (u5_ctx if idx <= 4 else "") + cds_left
        right_flank = cds_right + (st_ctx if idx >= n_codons - 4 else "")
        cur_gc_frac = gc_count / total_nt
        cur_c_gc = cur_c.count("G") + cur_c.count("C")

        scored: list[tuple[float, str]] = []
        for c in choices:
            cds_rna_win = cds_left + c + cds_right
            local_rna = left_flank + c + right_flank
            hard_pen = (
                -1000.0
                if (
                    any(pat in cds_rna_win for pat in forbidden_cds)
                    or any(pat in local_rna for pat in forbidden_junc)
                )
                else 0.0
            )
            rbp_pen = 0.0
            if "AUUUA" in local_rna:
                rbp_pen -= 0.45
            if "UGUA" in local_rna:
                rbp_pen -= 0.15
            if c == prev_c:
                rbp_pen -= 0.16  # Avoid repetitive mono-codon runs
            if c == next_c:
                rbp_pen -= 0.10
            c_gc = c.count("G") + c.count("C")
            proj_gc = cur_gc_frac + (c_gc - cur_c_gc) / total_nt
            gc_homeo = -4.5 * abs(proj_gc - target_gc) + gc_weight * c_gc
            val = (
                hard_pen
                + rbp_pen
                + 1.25 * csc_map.get(c, 0.0)
                + 0.22 * bic_map.get(f"{prev_c}-{c}", 0.0)
                + (0.22 * bic_map.get(f"{c}-{next_c}", 0.0) if len(next_c) == 3 else 0.0)
                + gc_homeo
            )
            scored.append((val, c))
        scored.sort(key=lambda item: -item[0])
        valid_scored = [(s, c) for s, c in scored if s > -500.0] or [scored[0]]
        if not must_fix and temperature > 1e-5 and len(valid_scored) > 1:
            top_k = valid_scored[: min(3, len(valid_scored))]
            max_s = top_k[0][0]
            weights = [math.exp(max(-20.0, (s - max_s) / temperature)) for s, _ in top_k]
            chosen = rng.choices([c for _, c in top_k], weights=weights, k=1)[0]
        else:
            chosen = valid_scored[0][1]
        gc_count += (chosen.count("G") + chosen.count("C")) - cur_c_gc
        codons[idx] = chosen

    out = "".join(codons)
    if chemistry in {"m1psi", "pseudouridine"}:
        out, _ = remediate_m1psi_slippery_runs(out)
    return out


def mutate_cds_region_synonymously(
    cds_rna: str,
    protein_seq: str,
    rng: random.Random,
    *,
    codon_start: int,
    codon_end: int,
    prefer_open_au: bool = False,
    chemistry: UridineChemistry = "m1psi",
    legacy_single_boundary: bool = False,
    constraints: list[ConstraintSpec] | None = None,
) -> str:
    codons = [cds_rna[i : i + 3] for i in range(0, len(cds_rna), 3)]
    lo = max(0, codon_start)
    hi = min(len(protein_seq), codon_end)
    csc_map, bic_map = _load_bicodon_atlas()
    forbidden = () if legacy_single_boundary else _extract_forbidden_rna_motifs(constraints, chemistry)
    for idx in range(lo, hi):
        aa = protein_seq[idx]
        choices = AA_TO_SYNONYMOUS_RNA_CODONS[aa]
        if len(choices) <= 1:
            continue
        if prefer_open_au:
            # Prefer less G-heavy synonymous codons near 5' Kozak TIS to keep initiation open
            ranked = sorted(choices, key=lambda c: (c.count("G"), c.count("C")))
            codons[idx] = ranked[0] if rng.random() < 0.7 else rng.choice(ranked[:2])
        elif legacy_single_boundary:
            prev_c = codons[idx - 1] if idx > 0 else ""
            ranked = sorted(
                choices,
                key=lambda c: -(
                    csc_map.get(c, 0.0)
                    + 0.08 * bic_map.get(f"{prev_c}-{c}", 0.0)
                    + 0.02 * (c.count("G") + c.count("C"))
                ),
            )
            codons[idx] = ranked[0] if rng.random() < 0.55 else rng.choice(choices)
        else:
            prev_c = codons[idx - 1] if idx > 0 else ""
            next_c = codons[idx + 1] if idx + 1 < len(codons) else ""
            left_flank = "".join(codons[max(0, idx - 3) : idx])
            right_flank = "".join(codons[idx + 1 : min(len(codons), idx + 4)])
            if csc_map:
                ranked = sorted(
                    choices,
                    key=lambda c: -(
                        (-1000.0 if any(p in (left_flank + c + right_flank) for p in forbidden) else 0.0)
                        + csc_map.get(c, 0.0)
                        + 0.18 * bic_map.get(f"{prev_c}-{c}", 0.0)
                        + (0.18 * bic_map.get(f"{c}-{next_c}", 0.0) if len(next_c) == 3 else 0.0)
                        + 0.02 * (c.count("G") + c.count("C"))
                    ),
                )
            else:
                ranked = sorted(choices, key=lambda c: -(c.count("G") + c.count("C")))
            codons[idx] = ranked[0] if rng.random() < 0.85 else rng.choice(ranked[:2])
    out = "".join(codons)
    if chemistry in {"m1psi", "pseudouridine"}:
        out, _ = remediate_m1psi_slippery_runs(out)
    return out


def generate_dnachisel_cds(
    protein_seq: str,
    constraints: list[ConstraintSpec],
    *,
    seed: int,
    chemistry: UridineChemistry,
) -> str:
    """Solve CDS constraints & human codon optimization via MIT-licensed DNA Chisel."""
    rng = random.Random(seed)
    init_rna = generate_random_synonymous_cds(protein_seq, rng, chemistry=chemistry)
    init_dna = init_rna.replace("U", "T")

    dc_constraints = [
        EnforceTranslation(location=(0, len(init_dna))),
        EnforceGCContent(mini=0.35, maxi=0.72, window=min(60, len(init_dna))),
        AvoidPattern("ATCTGTT"),  # Class II T7 terminator
    ]
    if chemistry in {"m1psi", "pseudouridine"}:
        dc_constraints.append(AvoidPattern("TTTT"))

    for spec in constraints:
        if "cds" not in spec.target_scopes and "full_transcript" not in spec.target_scopes:
            continue
        if spec.constraint_kind == "homopolymer_max":
            m_run = int(spec.params.get("max_run", 7)) + 1
            for base in "ACGT":
                dc_constraints.append(AvoidPattern(base * m_run))
        elif spec.constraint_kind == "type_iis_restriction_free":
            for enz in spec.params.get("enzymes", []):
                if enz in RESTRICTION_SITES_RNA:
                    fwd, rev = RESTRICTION_SITES_RNA[enz]
                    dc_constraints.append(AvoidPattern(fwd.replace("U", "T")))
                    dc_constraints.append(AvoidPattern(rev.replace("U", "T")))
            for raw_m in spec.params.get("forbidden_motifs", []):
                if raw_m:
                    dc_constraints.append(AvoidPattern(str(raw_m).upper().replace("U", "T")))

    problem = DnaOptimizationProblem(
        sequence=init_dna,
        constraints=dc_constraints,
        objectives=[CodonOptimize(species="h_sapiens", location=(0, len(init_dna)))],
        logger=None,
    )
    problem.max_random_iters = 1500
    try:
        problem.resolve_constraints()
        problem.optimize()
        solved_rna = str(problem.sequence).upper().replace("T", "U")
    except Exception:
        solved_rna = init_rna

    if chemistry in {"m1psi", "pseudouridine"}:
        solved_rna, _ = remediate_m1psi_slippery_runs(solved_rna)
    return solved_rna

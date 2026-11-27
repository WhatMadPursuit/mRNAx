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


def mutate_cds_region_synonymously(
    cds_rna: str,
    protein_seq: str,
    rng: random.Random,
    *,
    codon_start: int,
    codon_end: int,
    prefer_open_au: bool = False,
    chemistry: UridineChemistry = "m1psi",
) -> str:
    codons = [cds_rna[i : i + 3] for i in range(0, len(cds_rna), 3)]
    lo = max(0, codon_start)
    hi = min(len(protein_seq), codon_end)
    csc_map, bic_map = _load_bicodon_atlas()
    for idx in range(lo, hi):
        aa = protein_seq[idx]
        choices = AA_TO_SYNONYMOUS_RNA_CODONS[aa]
        if len(choices) <= 1:
            continue
        if prefer_open_au:
            # Prefer less G-heavy synonymous codons near 5' Kozak TIS to keep initiation open
            ranked = sorted(choices, key=lambda c: (c.count("G"), c.count("C")))
            codons[idx] = ranked[0] if rng.random() < 0.7 else rng.choice(ranked[:2])
        else:
            prev_c = codons[idx - 1] if idx > 0 else ""
            if csc_map:
                ranked = sorted(
                    choices,
                    key=lambda c: -(
                        csc_map.get(c, 0.0)
                        + 0.08 * bic_map.get(f"{prev_c}-{c}", 0.0)
                        + 0.02 * (c.count("G") + c.count("C"))
                    ),
                )
            else:
                ranked = sorted(choices, key=lambda c: -(c.count("G") + c.count("C")))
            codons[idx] = ranked[0] if rng.random() < 0.75 else rng.choice(choices)
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
                fwd, rev = RESTRICTION_SITES_RNA[enz]
                dc_constraints.append(AvoidPattern(fwd.replace("U", "T")))
                dc_constraints.append(AvoidPattern(rev.replace("U", "T")))

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

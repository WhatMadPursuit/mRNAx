"""Mulroney et al. (Nature 2024) & Poulis-Rodnina (2026) m1Psi / Psi +1 ribosomal frameshifting slippery guard."""

from __future__ import annotations

import re

from Bio.Seq import Seq

_SLIPPERY_U_RUN_RE = re.compile(r"U{4,}")
_SLIPPERY_M1PSI_STRICT_RE = re.compile(r"U{4,}|UUUC")


def find_m1psi_slippery_spans(
    cds_rna: str, *, include_uuuc: bool = False
) -> list[tuple[int, int, str]]:
    """Return 0-based (start, end, motif) spans of >=4 consecutive uridines (plus `UUUC` when `include_uuuc=True`)."""
    seq = cds_rna.upper().replace("T", "U")
    pat = _SLIPPERY_M1PSI_STRICT_RE if include_uuuc else _SLIPPERY_U_RUN_RE
    return [(m.start(), m.end(), m.group(0)) for m in pat.finditer(seq)]


def remediate_m1psi_slippery_runs(
    cds_rna: str, *, include_uuuc: bool = False
) -> tuple[str, int]:
    """Deterministically break all >=4-U runs (and `UUUC` when `include_uuuc=True`) while keeping AA 100% identical.

    In NCBI Genetic Code Table 1, every NNU codon is strictly synonymous with NNC,
    and UUA/UUG (Leu) are strictly synonymous with CUG/CUC. Moreover, in any reading frame,
    every `UUUC` tetramer has at least one codon ending in `U` (`NNU -> NNC`), turning
    `UUU+Cxx -> UUC+Cxx` (`UUCC`), `.UU+UC. -> .UC+UC.`, or `..U+UUC -> ..C+UUC` with
    zero amino-acid changes.
    """
    seq = cds_rna.upper().replace("T", "U")
    if len(seq) % 3 != 0:
        raise ValueError(f"CDS length ({len(seq)}) must be divisible by 3")

    codons = [seq[i : i + 3] for i in range(0, len(seq), 3)]
    edits = 0

    for _pass in range(4):
        current = "".join(codons)
        spans = find_m1psi_slippery_spans(current, include_uuuc=include_uuuc)
        if not spans:
            break
        touched_codon_indices: set[int] = set()
        for start, end, _ in spans:
            for pos in range(start, end):
                touched_codon_indices.add(pos // 3)
        for idx in sorted(touched_codon_indices):
            c = codons[idx]
            new_c = c
            if c in {"UUA", "UUG"}:
                new_c = "CUG"
            elif c.endswith("U"):
                new_c = c[:2] + "C"
            if new_c != c:
                codons[idx] = new_c
                edits += 1

    remediated = "".join(codons)
    if str(Seq(seq).translate(table=1, to_stop=False)) != str(
        Seq(remediated).translate(table=1, to_stop=False)
    ):
        raise RuntimeError("Internal invariant violation: synonymous m1Psi remediation altered AA")
    return remediated, edits

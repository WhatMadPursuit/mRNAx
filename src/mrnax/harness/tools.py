"""The 14 deterministic atomic tools of `mrnax_core`, shared by BYO-LLM Chat, CLI, SDK & MCP."""

from __future__ import annotations

import csv
import hashlib
import html
import importlib.util
import io
import json
import math
import os
import re
import shutil
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr, field_validator

from mrnax.constraints.frameshift_m1psi import find_m1psi_slippery_spans
from mrnax.constraints.registry import ScopedConstraintEngine, default_m1_constraints
from mrnax.constraints.sequence_gates import RESTRICTION_SITES_RNA
from mrnax.engine.lineardesign_adapter import local_heavy_binaries_disabled
from mrnax.engine.pareto_director import project_report_for_intent
from mrnax.engine.profiles import resolve_default_seed_dir
from mrnax.federated.external_evidence_adapter import ExternalEvidenceAdapterV2
from mrnax.harness.state_machine import (
    InitialTurnInput,
    ProjectHarnessSession,
    RefineTurnInput,
    paired_chains,
    parse_bounded_json_int,
)
from mrnax.io_insdc.genbank_io import export_construct_package
from mrnax.schemas.common import ComplianceProfile
from mrnax.schemas.construct_bundle import ConstructBundle
from mrnax.schemas.design_intent import DesignIntent
from mrnax.schemas.evaluation_report import BundleEvaluation
from mrnax.schemas.experiment_record import ExperimentRecord

_WAVE1_FALLBACK_DIR = Path(
    "/Users/alex/.gemini/antigravity/brain/4db4d24c-55e0-4968-869d-a93a1c9c7b7b/scratch/wave1_data_rules"
)
_WAVE1B_FALLBACK_DIR = Path(
    "/Users/alex/.gemini/antigravity/brain/4db4d24c-55e0-4968-869d-a93a1c9c7b7b/scratch/wave1_sota_models"
)
_VALID_RNA_RE = re.compile(r"^[ACGU]+$")
_G4_MOTIF_RE = re.compile(r"G{3,}[ACGU]{1,7}G{3,}[ACGU]{1,7}G{3,}[ACGU]{1,7}G{3,}")
_HOMOPOLYMER_6_RE = re.compile(r"(A{6,}|C{6,}|G{6,}|U{6,})")

# Standard Homo sapiens relative synonymous codon adaptiveness weights (Kazusa / NCBI human CDS)
_HUMAN_CODON_WEIGHTS: dict[str, float] = {
    "UUU": 0.85, "UUC": 1.00, "UUA": 0.19, "UUG": 0.32,
    "CUU": 0.33, "CUC": 0.49, "CUA": 0.18, "CUG": 1.00,
    "AUU": 0.77, "AUC": 1.00, "AUA": 0.36, "AUG": 1.00,
    "GUU": 0.39, "GUC": 0.52, "GUA": 0.25, "GUG": 1.00,
    "UCU": 0.79, "UCC": 0.91, "UCA": 0.63, "UCG": 0.23,
    "AGU": 0.63, "AGC": 1.00, "CCU": 0.91, "CCC": 1.00,
    "CCA": 0.88, "CCG": 0.34, "ACU": 0.70, "ACC": 1.00,
    "ACA": 0.78, "ACG": 0.31, "GCU": 0.68, "GCC": 1.00,
    "GCA": 0.58, "GCG": 0.28, "UAU": 0.79, "UAC": 1.00,
    "CAU": 0.72, "CAC": 1.00, "CAA": 0.36, "CAG": 1.00,
    "AAU": 0.89, "AAC": 1.00, "AAA": 0.75, "AAG": 1.00,
    "GAU": 0.85, "GAC": 1.00, "GAA": 0.72, "GAG": 1.00,
    "UGU": 0.82, "UGC": 1.00, "UGG": 1.00, "CGU": 0.38,
    "CGC": 0.86, "CGA": 0.52, "CGG": 0.95, "AGA": 1.00,
    "AGG": 1.00, "GGU": 0.47, "GGC": 1.00, "GGA": 0.74,
    "GGG": 0.74, "UAA": 1.00, "UAG": 0.80, "UGA": 1.00,
}

_LINEARIZATION_DNA_ADAPTERS: dict[str, tuple[str, str]] = {
    "BspQI": ("GAAGAGC", "Type-IIS BspQI reverse-oriented 3' scarless runoff site (5'-GAAGAGC-3' downstream of poly(A))"),
    "SapI": ("GAAGAGC", "Type-IIS SapI reverse-oriented 3' scarless runoff site (5'-GAAGAGC-3' downstream of poly(A))"),
    "BsaI": ("GAGACC", "Type-IIS BsaI reverse-oriented 3' scarless runoff site (5'-GAGACC-3' downstream of poly(A))"),
    "BsmBI": ("GAGACG", "Type-IIS BsmBI reverse-oriented 3' scarless runoff site (5'-GAGACG-3' downstream of poly(A))"),
    "BbsI": ("GTCTTC", "Type-IIS BbsI reverse-oriented 3' scarless runoff site (5'-GTCTTC-3' downstream of poly(A))"),
    "EcoRI": ("GAATTC", "Standard EcoRI runoff site (leaves 5'-AATT overhang metadata note)"),
    "XbaI": ("TCTAGA", "Standard XbaI runoff site"),
}


def resolve_registry_file(seed_dir: Path | None, filename: str) -> Path | None:
    """Resolve v2.5 registry file from `seed_dir / filename` first, falling back to Wave 1A or default seed_dir."""
    candidates: list[Path] = []
    if seed_dir is not None:
        candidates.append(seed_dir / filename)
    env_wave1 = os.environ.get("MRNAX_WAVE1_DATA_DIR", "").strip()
    if env_wave1:
        candidates.append(Path(env_wave1) / filename)
    candidates.append(_WAVE1_FALLBACK_DIR / filename)
    candidates.append(resolve_default_seed_dir() / filename)
    for p in candidates:
        if p.is_file():
            return p
    return None


def load_registry_json(seed_dir: Path | None, filename: str) -> dict[str, Any] | None:
    path = resolve_registry_file(seed_dir, filename)
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def detect_available_engines(seed_dir: Path | None = None) -> dict[str, dict[str, Any]]:
    """Lightweight filesystem/importspec check for RNA engines without importing torch or transformers."""
    eff_seed = seed_dir or resolve_default_seed_dir()
    policy_disabled = local_heavy_binaries_disabled()
    vienna_ok = importlib.util.find_spec("RNA") is not None
    ld_bin = None
    ef_bin = None
    if not policy_disabled:
        ld_candidates = [
            Path(os.environ["MRNAX_LINEARDESIGN_BIN"]) if os.environ.get("MRNAX_LINEARDESIGN_BIN") else None,
            Path.home() / ".mrnax" / "bin" / "lineardesign",
            _WAVE1B_FALLBACK_DIR / "bin" / "darwin_arm64" / "lineardesign",
            Path(shutil.which("lineardesign")) if shutil.which("lineardesign") else None,
        ]
        ld_bin = next((p for p in ld_candidates if p is not None and p.is_file() and os.access(p, os.X_OK)), None)

        ef_candidates = [
            Path(os.environ["MRNAX_ETERNAFOLD_BIN"]) if os.environ.get("MRNAX_ETERNAFOLD_BIN") else None,
            Path.home() / ".mrnax" / "bin" / "contrafold",
            _WAVE1B_FALLBACK_DIR / "bin" / "darwin_arm64" / "contrafold",
            Path(shutil.which("contrafold")) if shutil.which("contrafold") else None,
        ]
        ef_bin = next((p for p in ef_candidates if p is not None and p.is_file() and os.access(p, os.X_OK)), None)

    from mrnax.engine.surrogates import has_surrogate_v2_artifacts

    surr_v1_ok = (eff_seed / "surrogate_v1").is_dir()
    surr_v2_ok = has_surrogate_v2_artifacts(eff_seed)
    return {
        "viennarna": {"available": vienna_ok, "license_track": "research_max"},
        "lineardesign": {
            "available": ld_bin is not None,
            "disabled_by_policy": policy_disabled,
            "license_track": "research_max",
        },
        "eternafold": {
            "available": ef_bin is not None,
            "disabled_by_policy": policy_disabled,
            "license_track": "commercial_clearance_pending",
        },
        "surrogate_v1_clean": {"available": surr_v1_ok, "license_track": "commercial_clean"},
        "surrogate_v2": {"available": surr_v2_ok, "license_track": "commercial_clean"},
    }


def _compute_human_cai(cds_rna: str) -> float:
    codons = [cds_rna[i : i + 3] for i in range(0, len(cds_rna) - 2, 3) if len(cds_rna[i : i + 3]) == 3]
    coding = [c for c in codons if c not in ("UAA", "UAG", "UGA") and c in _HUMAN_CODON_WEIGHTS]
    if not coding:
        return 0.0
    log_sum = sum(math.log(max(1e-6, _HUMAN_CODON_WEIGHTS[c])) for c in coding)
    return round(math.exp(log_sum / len(coding)), 4)


def _scan_rna_motifs_fallback(
    seq: str,
    *,
    five_prime_end: int | None,
    cds_start: int | None,
    cds_end: int | None,
    three_prime_start: int | None,
    three_prime_end: int | None,
    seed_dir: Path | None,
) -> list[dict[str, Any]]:
    motifs: list[dict[str, Any]] = []
    utr5_hi = five_prime_end if five_prime_end is not None else (cds_start if cds_start is not None else 0)
    if utr5_hi > 0:
        utr5_sub = seq[:utr5_hi]
        pos = 0
        while True:
            idx = utr5_sub.find("AUG", pos)
            if idx < 0:
                break
            in_frame = (cds_start is not None) and ((cds_start - idx) % 3 == 0)
            motifs.append(
                {
                    "kind": "uaug",
                    "label": "5'UTR uAUG" + (" (in-frame)" if in_frame else " (out-of-frame)"),
                    "start": idx,
                    "end": idx + 3,
                    "severity": "critical",
                    "detail": "Upstream start codon in 5'UTR competes with canonical initiation.",
                }
            )
            pos = idx + 1

    c_lo = cds_start if cds_start is not None else 0
    c_hi = cds_end if cds_end is not None else len(seq)
    if c_hi > c_lo:
        cds_sub = seq[c_lo:c_hi]
        for s, e, m_str in find_m1psi_slippery_spans(cds_sub):
            motifs.append(
                {
                    "kind": "uuuu_frameshift",
                    "label": f"Slippery U-run ({m_str})",
                    "start": c_lo + s,
                    "end": c_lo + e,
                    "severity": "high",
                    "detail": "Mulroney et al. 2024 m1Psi +1 ribosomal frameshifting motif.",
                }
            )

    non_polya_hi = three_prime_end if three_prime_end is not None else len(seq)
    # Exclude terminal poly(A) run if coordinates were not explicitly passed
    if three_prime_end is None:
        stripped = seq.rstrip("A")
        if len(seq) - len(stripped) >= 20:
            non_polya_hi = len(stripped)
    body_seq = seq[:non_polya_hi]

    for m in _HOMOPOLYMER_6_RE.finditer(body_seq):
        run = m.group(0)
        motifs.append(
            {
                "kind": "homopolymer",
                "label": f"Homopolymer {run[0]}x{len(run)}",
                "start": m.start(),
                "end": m.end(),
                "severity": "high" if len(run) >= 8 else "warning",
                "detail": f"Mononucleotide run of length {len(run)} outside poly(A).",
            }
        )

    for m in _G4_MOTIF_RE.finditer(body_seq):
        motifs.append(
            {
                "kind": "g_quadruplex",
                "label": "Putative RNA G-quadruplex",
                "start": m.start(),
                "end": m.end(),
                "severity": "warning",
                "detail": "Four proximal G-tracts may stall 43S scanning or reverse transcription.",
            }
        )

    pos_t7 = 0
    while True:
        idx = body_seq.find("AUCUGUU", pos_t7)
        if idx < 0:
            break
        motifs.append(
            {
                "kind": "t7_terminator",
                "label": "Class-II T7 terminator (AUCUGUU)",
                "start": idx,
                "end": idx + 7,
                "severity": "critical",
                "detail": "May cause premature abortive runoff during T7 in vitro transcription.",
            }
        )
        pos_t7 = idx + 1

    seen_enz_spans: set[tuple[str, int, int]] = set()
    for enz, (fwd, rev) in RESTRICTION_SITES_RNA.items():
        for site in {fwd, rev}:
            p_e = 0
            while True:
                idx = seq.find(site, p_e)
                if idx < 0:
                    break
                key = (enz, idx, idx + len(site))
                if key not in seen_enz_spans:
                    seen_enz_spans.add(key)
                    motifs.append(
                        {
                            "kind": "restriction_site",
                            "label": f"{enz} site ({site})",
                            "start": idx,
                            "end": idx + len(site),
                            "severity": "warning",
                            "detail": f"Internal {enz} recognition site conflicts with plasmid linearization if {enz} is selected.",
                        }
                    )
                p_e = idx + 1

    u3_lo = three_prime_start if three_prime_start is not None else (cds_end if cds_end is not None else 0)
    u3_hi = non_polya_hi
    if u3_hi > u3_lo:
        u3_sub = seq[u3_lo:u3_hi]
        p_are = 0
        while True:
            idx = u3_sub.find("AUUUA", p_are)
            if idx < 0:
                break
            motifs.append(
                {
                    "kind": "are_motif",
                    "label": "3'UTR ARE pentamer (AUUUA)",
                    "start": u3_lo + idx,
                    "end": u3_lo + idx + 5,
                    "severity": "warning",
                    "detail": "AU-rich element recruits deadenylase/decay factors (ZFP36/TTP).",
                }
            )
            p_are = idx + 1

    mir_atlas = load_registry_json(seed_dir, "tissue_mirna_atlas_v1.json")
    if isinstance(mir_atlas, dict):
        for entry in mir_atlas.get("mirna_entries", []):
            if not isinstance(entry, dict):
                continue
            mir_id = str(entry.get("mirna_id") or "")
            tissue = str(entry.get("tissue_group") or "")
            full_site = str(entry.get("mrna_full_complementary_ago2_cleavage_site_rna") or "")
            seed_8 = str(entry.get("mrna_target_site_8mer_rna") or "")
            matched_spans: list[tuple[int, int]] = []
            if len(full_site) >= 18:
                p_m = 0
                while True:
                    idx = seq.find(full_site, p_m)
                    if idx < 0:
                        break
                    matched_spans.append((idx, idx + len(full_site)))
                    motifs.append(
                        {
                            "kind": "mirna_site",
                            "label": f"{mir_id} Ago2 cleavage site ({tissue})",
                            "start": idx,
                            "end": idx + len(full_site),
                            "severity": "info",
                            "detail": f"Full complementary {mir_id} detargeting cassette for {tissue}.",
                        }
                    )
                    p_m = idx + 1
            if len(seed_8) == 8:
                p_s = 0
                while True:
                    idx = body_seq.find(seed_8, p_s)
                    if idx < 0:
                        break
                    if not any(lo <= idx and idx + 8 <= hi for lo, hi in matched_spans):
                        motifs.append(
                            {
                                "kind": "mirna_site",
                                "label": f"{mir_id} 8mer seed match ({tissue})",
                                "start": idx,
                                "end": idx + 8,
                                "severity": "info",
                                "detail": f"Canonical 8mer target site for {mir_id} ({tissue}).",
                            }
                        )
                    p_s = idx + 1

    motifs.sort(key=lambda item: (int(item["start"]), int(item["end"]), str(item["kind"])))
    return motifs


_STRUCTURE_JOB_SEM = threading.BoundedSemaphore(4)


class StructureProfilerBusyError(RuntimeError):
    """Raised when concurrent RNA structure profiler invocations saturate the process-wide bound."""


def run_structure_and_motif_analysis(
    sequence: str,
    *,
    five_prime_end: int | None = None,
    cds_start: int | None = None,
    cds_end: int | None = None,
    three_prime_start: int | None = None,
    three_prime_end: int | None = None,
    modality: str = "linear_mrna",
    compliance_track: str = "research_max",
    include_eternafold: bool = False,
    seed_dir: Path | None = None,
) -> dict[str, Any]:
    """Analyze RNA secondary structure, 5' initiation base-pair arcs, sliding tracks, and sequence motifs.

    Delegates to Wave 2A's `mrnax.engine.structure_profile.analyze_rna_structure` when present,
    with a complete ViennaRNA + Python fallback keeping peak RSS well below 180 MiB.
    """
    if compliance_track not in ("research_max", "commercial_clean"):
        raise ValueError(f"Unsupported compliance_track '{compliance_track}'")
    if compliance_track == "commercial_clean" and include_eternafold:
        raise PermissionError(
            "EternaFold v1.3.1 has license_track='commercial_clearance_pending' and cannot run under commercial_clean."
        )
    norm_seq = sequence.strip().upper().replace("T", "U")
    if len(norm_seq) < 3 or len(norm_seq) > 20_000:
        raise ValueError(f"RNA sequence length ({len(norm_seq)}) must be between 3 and 20000 nt")
    if not _VALID_RNA_RE.fullmatch(norm_seq):
        raise ValueError("Sequence must contain only valid nucleotide characters (A, C, G, U/T)")
    if modality not in ("linear_mrna", "circrna", "sarna"):
        raise ValueError(f"Unsupported modality '{modality}'")

    n = len(norm_seq)
    for cname, cval in (
        ("five_prime_end", five_prime_end),
        ("cds_start", cds_start),
        ("cds_end", cds_end),
        ("three_prime_start", three_prime_start),
        ("three_prime_end", three_prime_end),
    ):
        if cval is not None and (cval < 0 or cval > n):
            raise ValueError(f"Coordinate '{cname}' ({cval}) out of bounds [0, {n}]")

    if not _STRUCTURE_JOB_SEM.acquire(timeout=3.0):
        raise StructureProfilerBusyError("Structure profiler concurrency limit reached; please retry shortly.")
    try:
        try:
            from mrnax.engine.structure_profile import analyze_rna_structure  # type: ignore[import-not-found]
        except ImportError:
            analyze_rna_structure = None

        if callable(analyze_rna_structure):
            res = dict(
                analyze_rna_structure(
                    norm_seq,
                    five_prime_end=five_prime_end,
                    cds_start=cds_start,
                    cds_end=cds_end,
                    three_prime_start=three_prime_start,
                    three_prime_end=three_prime_end,
                    modality=modality,
                    compliance_track=compliance_track,
                    include_eternafold=include_eternafold,
                    seed_dir=seed_dir,
                )
            )
            init_win = res.get("initiation_window")
            if "dot_bracket" not in res and isinstance(init_win, dict) and isinstance(init_win.get("dot_bracket"), str):
                res["dot_bracket"] = init_win["dot_bracket"]
            return res

        # Clean deterministic fallback using ViennaRNA (when permitted by track) + Python scanner
        gc_pct = round(((norm_seq.count("G") + norm_seq.count("C")) / n) * 100.0, 2)
        u_pct = round((norm_seq.count("U") / n) * 100.0, 2)
        eff_aug = cds_start if cds_start is not None else (five_prime_end if five_prime_end is not None else max(0, norm_seq.find("AUG")))
        eff_cds_end = cds_end if cds_end is not None else n
        cds_slice = norm_seq[eff_aug:eff_cds_end] if eff_cds_end > eff_aug else norm_seq
        cai_val = _compute_human_cai(cds_slice)

        win_start = max(0, eff_aug - 30)
        win_end = min(n, eff_aug + 45)
        win_seq = norm_seq[win_start:win_end]
        aug_in_win = eff_aug - win_start

        engines_used: list[str] = []
        dot_bracket_full = "." * min(n, 600)
        init_db = "." * len(win_seq)
        arcs: list[dict[str, Any]] = []
        mfe_val = 0.0
        ens_div = 0.0
        aup_mean = 0.5
        kozak_up = 0.5

        window_nt = min(120, n)
        step_nt = max(10, min(40, window_nt // 3 or 1))
        positions: list[int] = []
        local_mfe: list[float] = []
        local_gc: list[float] = []
        local_u: list[float] = []
        local_up: list[float] = []

        if compliance_track != "commercial_clean" and importlib.util.find_spec("RNA") is not None:
            import RNA  # type: ignore[import-untyped]

            engines_used.append("ViennaRNA")
            # Bound global O(N^3) fold to <= 2500 nt so 20k nt saRNA stays < 50 MiB RSS and fast
            if n <= 2500:
                ss_full, e_full = RNA.fold(norm_seq)
                dot_bracket_full = str(ss_full)
                mfe_val = round(float(e_full), 2)
            else:
                ss_head, _ = RNA.fold(norm_seq[:600])
                dot_bracket_full = str(ss_head)

            if len(win_seq) >= 6:
                fc = RNA.fold_compound(win_seq)
                init_ss, _ = fc.mfe()
                init_db = str(init_ss)
                fc.pf()
                ens_div = round(float(fc.mean_bp_distance()), 4)
                bpp = fc.bpp()
                wlen = len(win_seq)
                paired_sum = [0.0] * wlen
                for i in range(1, wlen + 1):
                    row = bpp[i]
                    for j in range(i + 1, wlen + 1):
                        p = float(row[j])
                        if p > 0.0:
                            paired_sum[i - 1] += p
                            paired_sum[j - 1] += p
                        if p >= 0.15:
                            arcs.append({"i": i - 1, "j": j - 1, "prob": round(p, 4)})
                unpaired_win = [max(0.0, min(1.0, 1.0 - ps)) for ps in paired_sum]
                k_lo = max(0, aug_in_win - 15)
                k_hi = min(wlen, aug_in_win + 18)
                if k_hi > k_lo:
                    kozak_up = round(sum(unpaired_win[k_lo:k_hi]) / (k_hi - k_lo), 4)

            w_pfl = min(n, 80)
            l_pfl = min(w_pfl, 60)
            up_table = RNA.pfl_fold_up(norm_seq, 1, w_pfl, l_pfl) if n >= 12 else None
            if up_table is not None:
                per_nt_up = [float(up_table[idx + 1][1]) for idx in range(n)]
                aup_mean = round(sum(per_nt_up) / n, 4)
            else:
                per_nt_up = [0.5] * n

            if n <= window_nt:
                _, chunk_e = RNA.fold(norm_seq)
                positions.append(0)
                local_mfe.append(round(float(chunk_e), 2))
                local_gc.append(gc_pct)
                local_u.append(u_pct)
                local_up.append(round(sum(per_nt_up) / max(1, len(per_nt_up)), 4))
            else:
                for start_i in range(0, n - window_nt + 1, step_nt):
                    chunk = norm_seq[start_i : start_i + window_nt]
                    _, chunk_e = RNA.fold(chunk)
                    positions.append(start_i)
                    local_mfe.append(round(float(chunk_e), 2))
                    local_gc.append(round(((chunk.count("G") + chunk.count("C")) / window_nt) * 100.0, 2))
                    local_u.append(round((chunk.count("U") / window_nt) * 100.0, 2))
                    local_up.append(round(sum(per_nt_up[start_i : start_i + window_nt]) / window_nt, 4))
                if n > 2500 and local_mfe:
                    mfe_val = round((sum(local_mfe) / len(local_mfe)) * (n / window_nt), 2)
        else:
            engines_used.append("heuristic_sequence_scanner")
            for start_i in range(0, max(1, n - window_nt + 1), step_nt):
                chunk = norm_seq[start_i : start_i + window_nt]
                clen = max(1, len(chunk))
                positions.append(start_i)
                local_mfe.append(0.0)
                local_gc.append(round(((chunk.count("G") + chunk.count("C")) / clen) * 100.0, 2))
                local_u.append(round((chunk.count("U") / clen) * 100.0, 2))
                local_up.append(0.5)

        motifs = _scan_rna_motifs_fallback(
            norm_seq,
            five_prime_end=five_prime_end,
            cds_start=cds_start,
            cds_end=cds_end,
            three_prime_start=three_prime_start,
            three_prime_end=three_prime_end,
            seed_dir=seed_dir,
        )
        return {
            "length_nt": n,
            "modality": modality,
            "full_gc_pct": gc_pct,
            "full_u_pct": u_pct,
            "cai": cai_val,
            "mfe_kcal_mol": mfe_val,
            "mfe_per_nt": round(mfe_val / max(1, n), 4),
            "ensemble_diversity": ens_div,
            "aup_mean": aup_mean,
            "kozak_unpaired_prob": kozak_up,
            "dot_bracket": dot_bracket_full,
            "initiation_window": {
                "start": win_start,
                "end": win_end,
                "aug_index_in_window": aug_in_win,
                "sequence": win_seq,
                "dot_bracket": init_db,
                "arcs": arcs,
            },
            "sliding_tracks": {
                "window_nt": window_nt,
                "step_nt": step_nt,
                "positions": positions,
                "local_mfe": local_mfe,
                "local_gc_pct": local_gc,
                "local_u_pct": local_u,
                "local_unpaired_prob": local_up,
            },
            "motifs": motifs,
            "engines_used": engines_used,
        }
    finally:
        _STRUCTURE_JOB_SEM.release()


def query_scientific_rules(
    *,
    seed_dir: Path | None = None,
    modality: str | None = None,
    region: str | None = None,
    grade: str | None = None,
    query: str | None = None,
    include_debunked: bool = True,
) -> dict[str, Any]:
    doc = load_registry_json(seed_dir, "rules_scientific_v2.json")
    if doc is None:
        doc = load_registry_json(seed_dir, "rules_commercial_v1.json") or {"rules": []}
    raw_rules = doc.get("rules") or []
    mod_q = modality.strip().lower() if modality and modality.strip() else None
    reg_q = region.strip().lower() if region and region.strip() else None
    grd_q = grade.strip().lower() if grade and grade.strip() else None
    txt_q = query.strip().lower() if query and query.strip() else None

    matched: list[dict[str, Any]] = []
    for r in raw_rules:
        if not isinstance(r, dict):
            continue
        ev_grade = str(r.get("evidence_grade") or r.get("tier") or "").lower()
        if not include_debunked and ev_grade == "debunked_myth":
            continue
        if grd_q and ev_grade != grd_q:
            continue
        if mod_q:
            mods = r.get("modality")
            mod_list = [str(x).lower() for x in mods] if isinstance(mods, list) else [str(mods or "linear_mrna").lower()]
            if mod_q not in mod_list:
                continue
        if reg_q:
            r_reg = str(r.get("region") or r.get("scope") or "").lower()
            if reg_q not in r_reg and r_reg != reg_q:
                continue
        if txt_q:
            blob = json.dumps(
                {
                    "id": r.get("rule_id"),
                    "zh": r.get("title_zh"),
                    "en": r.get("title_en"),
                    "eff": r.get("effect_size_summary"),
                    "key": r.get("machine_check_key"),
                    "fail": r.get("failure_modes"),
                },
                ensure_ascii=False,
            ).lower()
            if txt_q not in blob:
                continue
        matched.append(r)

    return {
        "rulebook_id": doc.get("rulebook_id", "rules_commercial_v1"),
        "schema_version": doc.get("schema_version", "1.0"),
        "total_available": len(raw_rules),
        "total_matched": len(matched),
        "filters": {
            "modality": mod_q,
            "region": region.strip() if region else None,
            "grade": grd_q,
            "q": query.strip() if query else None,
            "include_debunked": include_debunked,
        },
        "rules": matched,
    }


def compare_construct_against_clinical_benchmarks(
    *,
    seed_dir: Path | None = None,
    bundle: ConstructBundle | None = None,
    sequence: str | None = None,
    modality: str | None = None,
    chemistry: str | None = None,
) -> dict[str, Any]:
    if bundle is None and not sequence:
        raise ValueError("Provide either bundle or sequence for clinical benchmark comparison")

    if bundle is not None:
        mol = bundle.molecules[0]
        rna_seq = mol.sequence
        eff_mod = modality or mol.modality
        eff_chem = chemistry or mol.chemistry_track.uridine_chemistry
        u5 = mol.get_segment_sequence("five_prime_utr")
        cds = mol.get_segment_sequence("cds")
        u3 = mol.get_segment_sequence("three_prime_utr")
        pa = mol.get_segment_sequence("poly_a")
        cand_id = bundle.bundle_id
    else:
        assert sequence is not None
        rna_seq = sequence.strip().upper().replace("T", "U")
        if len(rna_seq) < 3 or len(rna_seq) > 20_000 or not _VALID_RNA_RE.fullmatch(rna_seq):
            raise ValueError("Sequence must be 3..20000 valid RNA/DNA nucleotides (A/C/G/U/T)")
        eff_mod = modality or "linear_mrna"
        eff_chem = chemistry or "m1psi"
        u5, cds, u3, pa = "", rna_seq, "", ""
        cand_id = "ad_hoc_sequence"

    gc_full = round((rna_seq.count("G") + rna_seq.count("C")) / max(1, len(rna_seq)), 4)
    gc_cds = round((cds.count("G") + cds.count("C")) / max(1, len(cds)), 4) if cds else gc_full
    codons = [cds[i : i + 3] for i in range(0, len(cds) - 2, 3) if len(cds[i : i + 3]) == 3]
    gc3_cds = (
        round(sum(1 for c in codons if c[2] in ("G", "C")) / max(1, len(codons)), 4)
        if codons
        else None
    )
    u4_runs = len(find_m1psi_slippery_spans(cds)) if cds else 0
    uaug_cnt = 0
    if u5:
        p = 0
        while True:
            idx = u5.find("AUG", p)
            if idx < 0:
                break
            uaug_cnt += 1
            p = idx + 1

    cand_stats = {
        "candidate_id": cand_id,
        "modality": eff_mod,
        "chemistry": eff_chem,
        "full_length_nt": len(rna_seq),
        "utr5_length_nt": len(u5),
        "cds_length_nt": len(cds),
        "utr3_length_nt": len(u3),
        "polya_length_nt": len(pa),
        "gc_full_transcript": gc_full,
        "gc_cds": gc_cds,
        "gc3_cds": gc3_cds,
        "utr5_uaug_count": uaug_cnt,
        "cds_m1psi_u4_plus_run_count": u4_runs,
        "cai_human": _compute_human_cai(cds) if cds else 0.0,
    }

    bench_doc = load_registry_json(seed_dir, "clinical_mrna_benchmarks_v1.json") or {"benchmarks": []}
    raw_benchmarks = bench_doc.get("benchmarks") or []
    comparisons: list[dict[str, Any]] = []
    for b in raw_benchmarks:
        if not isinstance(b, dict):
            continue
        if modality and str(b.get("modality") or "") != modality:
            continue
        b_stats = b.get("quantitative_sequence_stats") or {}
        b_gc_full = b_stats.get("gc_full_transcript")
        b_gc_cds = b_stats.get("gc_cds")
        b_gc3 = b_stats.get("gc3_cds")
        comparisons.append(
            {
                "benchmark_id": b.get("benchmark_id"),
                "construct_name": b.get("construct_name"),
                "developer": b.get("developer"),
                "modality": b.get("modality"),
                "chemistry": b.get("chemistry"),
                "clinical_or_reference_status": b.get("clinical_or_reference_status"),
                "license_track": b.get("license_track"),
                "fto_and_governance_notice": b.get("fto_and_governance_notice"),
                "architecture_summary": b.get("architecture_summary"),
                "benchmark_stats": b_stats,
                "deltas_candidate_minus_benchmark": {
                    "gc_full_delta": round(gc_full - float(b_gc_full), 4) if isinstance(b_gc_full, (int, float)) else None,
                    "gc_cds_delta": round(gc_cds - float(b_gc_cds), 4) if isinstance(b_gc_cds, (int, float)) else None,
                    "gc3_cds_delta": (
                        round(gc3_cds - float(b_gc3), 4)
                        if (gc3_cds is not None and isinstance(b_gc3, (int, float)))
                        else None
                    ),
                    "cds_u4_run_delta": (
                        u4_runs - int(b_stats["cds_m1psi_u4_plus_run_count"])
                        if isinstance(b_stats.get("cds_m1psi_u4_plus_run_count"), int)
                        else None
                    ),
                },
            }
        )

    return {
        "registry_id": bench_doc.get("registry_id", "clinical_mrna_benchmarks_v1"),
        "candidate_stats": cand_stats,
        "benchmarks_compared": len(comparisons),
        "comparisons": comparisons,
    }


def render_ivt_synthesis_order(
    bundle: ConstructBundle,
    intent: DesignIntent | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Generate T7/CleanCap plasmid DNA template & scarless linearization CSV order sheet."""
    enzymes: list[str] = []
    if intent is not None:
        for c in intent.constraints:
            if c.constraint_kind == "type_iis_restriction_free":
                enzymes = [str(e) for e in c.params.get("enzymes", [])]
                break
    lin_enzyme = enzymes[0] if enzymes else "BspQI"
    lin_adapter_dna, lin_note = _LINEARIZATION_DNA_ADAPTERS.get(
        lin_enzyme, ("GAAGAGC", f"Linearization site for {lin_enzyme}")
    )

    rows: list[dict[str, Any]] = []
    for mol in bundle.molecules:
        rna = mol.sequence
        cap = mol.chemistry_track.cap_chemistry
        # Invariant: template_dna_5to3 MUST match the verified linear IVT transcript 1:1
        # (for circRNA: the verified linear PIE ribozyme precursor whose spliced exon product equals mol.sequence).
        from mrnax.io_insdc.genbank_io import ExportConstraintViolationError

        if mol.modality == "circrna" or mol.topology == "circular":
            e2_ann = next((a for a in mol.annotations if a.feature_id == "pie_exon2_scar" and a.intervals), None)
            e1_ann = next((a for a in mol.annotations if a.feature_id == "pie_exon1_scar" and a.intervals), None)
            if e2_ann is None or e1_ann is None:
                raise ExportConstraintViolationError(
                    f"Circular molecule '{mol.molecule_id}' lacks PIE exon-scar annotations required to build a verified linear IVT precursor."
                )
            pie_status = (e2_ann.qualifiers.get("pie_precursor_status") or [""])[0]
            up_arm = (e2_ann.qualifiers.get("pie_upstream_arm_rna") or [""])[0].strip().upper()
            dn_arm = (e1_ann.qualifiers.get("pie_downstream_arm_rna") or [""])[0].strip().upper()
            raw_prec_len = (e2_ann.qualifiers.get("ivt_precursor_length_nt") or [""])[0].strip()
            e2_lo, e2_hi = e2_ann.intervals[0]
            e1_lo, e1_hi = e1_ann.intervals[0]
            e2_scar = rna[e2_lo:e2_hi]
            e1_scar = rna[e1_lo:e1_hi]
            if (
                pie_status != "verified_split_intron_arms"
                or e2_lo != 0
                or e1_hi != len(rna)
                or e2_hi > e1_lo
                or not up_arm
                or not dn_arm
                or len(up_arm) <= len(e2_scar)
                or len(dn_arm) <= len(e1_scar)
                or not _VALID_RNA_RE.fullmatch(up_arm)
                or not _VALID_RNA_RE.fullmatch(dn_arm)
                or not up_arm.endswith(e2_scar)
                or not dn_arm.startswith(e1_scar)
                or not raw_prec_len.isdigit()
            ):
                raise ExportConstraintViolationError(
                    f"Circular molecule '{mol.molecule_id}' lacks verified split PIE intron precursor arms (status='{pie_status or 'missing'}'); refusing to export mature circular sequence as a linear IVT plasmid."
                )
            internal_body = rna[e2_hi:e1_lo]
            ivt_rna = up_arm + internal_body + dn_arm
            expected_prec_len = int(raw_prec_len)
            spliced_check = up_arm[-len(e2_scar) :] + internal_body + dn_arm[: len(e1_scar)]
            if len(ivt_rna) != expected_prec_len or spliced_check != rna:
                raise ExportConstraintViolationError(
                    f"Circular molecule '{mol.molecule_id}' PIE precursor reconstruction failed length or splice-scar round-trip check."
                )
            core_dna = ivt_rna.replace("U", "T")
            promoter_dna = "TAATACGACTCACTATA"
            promoter_type = "T7_Uncapped_Circular_or_Custom"
            leader_check = (
                f"PASS (verified PIE linear IVT precursor {len(ivt_rna)} nt splices to {len(rna)} nt mature circRNA)"
            )
            if core_dna.replace("T", "U") != ivt_rna:
                raise ExportConstraintViolationError("Internal invariant failure: exported PIE IVT precursor diverged.")
        else:
            ivt_rna = rna
            core_dna = ivt_rna.replace("U", "T")
            promoter_dna = "TAATACGACTCACTATA"
            if cap == "cap1_cleancap_ag":
                if rna.startswith("AG"):
                    promoter_type = "T7_CleanCap_AG_Phi6.5"
                    leader_check = "PASS (transcript starts with 5'-AG matching cap1_cleancap_ag)"
                elif mol.modality == "sarna" and rna.startswith("AU"):
                    promoter_type = "T7_CleanCap_AU_Alphavirus"
                    leader_check = "PASS (saRNA starts with 5'-AU for alphavirus CleanCap AU replication)"
                else:
                    promoter_type = "T7_CleanCap_AG_Incompatible_Prefix"
                    leader_check = (
                        f"ADVISORY: bundle declares cap_chemistry='cap1_cleancap_ag' while transcript starts with 5'-{rna[:2]}; "
                        "approved transcript emitted verbatim without prepended bases — revise and re-validate bundle for 5'-AG or enzymatic Cap-1"
                    )
            elif cap == "cap1_arca_gg":
                promoter_type = "T7_ARCA_GG_Standard"
                if rna.startswith("GG"):
                    leader_check = "PASS (transcript starts with 5'-GG matching cap1_arca_gg)"
                else:
                    leader_check = (
                        f"ADVISORY: bundle declares cap_chemistry='cap1_arca_gg' while transcript starts with 5'-{rna[:2]}; "
                        "approved transcript emitted verbatim without prepended bases"
                    )
            elif cap == "cap1_enzymatic":
                promoter_type = "T7_Vaccinia_Enzymatic_Cap1"
                leader_check = (
                    "PASS (post-transcriptional enzymatic Cap-1 compatible with 5'-purine start)"
                    if rna[:1] in ("G", "A")
                    else f"ADVISORY: transcript starts with pyrimidine 5'-{rna[:1]} (lower T7 yield; preserved verbatim)"
                )
            else:
                promoter_type = "T7_Standard_Exact_Transcript"
                leader_check = (
                    "PASS (transcript starts with 5'-GGG)"
                    if rna.startswith("GGG")
                    else "ADVISORY: approved transcript emitted verbatim without prepended bases"
                )
            if core_dna.replace("T", "U") != rna:
                raise ExportConstraintViolationError("Internal invariant failure: exported IVT template diverged from approved bundle.")

        full_cassette_dna = promoter_dna + core_dna + lin_adapter_dna
        gc_pct = round(((full_cassette_dna.count("G") + full_cassette_dna.count("C")) / max(1, len(full_cassette_dna))) * 100.0, 2)
        cds_seg = mol.get_segment("cds")
        pa_seg = mol.get_segment("poly_a")
        rows.append(
            {
                "bundle_id": bundle.bundle_id,
                "version_digest": bundle.version_digest,
                "molecule_id": mol.molecule_id,
                "modality": mol.modality,
                "topology": mol.topology,
                "uridine_chemistry": mol.chemistry_track.uridine_chemistry,
                "cap_chemistry": cap,
                "promoter_type": promoter_type,
                "promoter_dna_5to3": promoter_dna,
                "five_prime_leader_check": leader_check,
                "cds_length_nt": (cds_seg.end - cds_seg.start) if cds_seg else 0,
                "full_rna_length_nt": len(ivt_rna),
                "mature_rna_length_nt": len(rna),
                "poly_a_length_nt": (pa_seg.end - pa_seg.start) if pa_seg else 0,
                "linearization_enzyme": lin_enzyme,
                "linearization_site_dna_5to3": lin_adapter_dna,
                "linearization_note": lin_note,
                "template_dna_5to3": core_dna,
                "full_ivt_insert_cassette_dna_5to3": full_cassette_dna,
                "cassette_gc_pct": gc_pct,
                "cassette_sha256": hashlib.sha256(full_cassette_dna.encode("utf-8")).hexdigest(),
            }
        )

    buf = io.StringIO()
    fieldnames = list(rows[0].keys()) if rows else ["bundle_id", "molecule_id"]
    writer = csv.DictWriter(buf, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
    return buf.getvalue(), rows


def render_qc_dossier_html(
    bundle: ConstructBundle,
    *,
    intent: DesignIntent | None,
    constraint_engine: ScopedConstraintEngine,
    evaluation: BundleEvaluation | None = None,
    seed_dir: Path | None = None,
) -> str:
    """Render a clean, printable HTML5 scientific QC & compliance dossier (zero external assets, XSS-safe)."""
    eff_constraints = intent.constraints if intent is not None else default_m1_constraints()
    eff_track = intent.compliance_track if intent is not None else "research_max"
    from mrnax.io_insdc.genbank_io import resolve_expected_chemistry_for_bundle

    raw_chem = intent.uridine_chemistry if intent is not None else bundle.molecules[0].chemistry_track.uridine_chemistry
    eff_chem = resolve_expected_chemistry_for_bundle(bundle, raw_chem)
    outcome = constraint_engine.evaluate_bundle(
        bundle,
        eff_constraints,
        compliance_track=eff_track,
        expected_chemistry=eff_chem,
        locked_scopes=intent.locked_scopes if intent is not None else None,
        expected_protein_targets=intent.protein_targets if intent is not None else None,
    )
    clin = compare_construct_against_clinical_benchmarks(seed_dir=seed_dir, bundle=bundle)
    esc = html.escape

    seg_rows_html: list[str] = []
    for mol in bundle.molecules:
        for s in mol.segments:
            sub = mol.sequence[s.start : s.end]
            gc = round(((sub.count("G") + sub.count("C")) / max(1, len(sub))) * 100.0, 2)
            seg_rows_html.append(
                f"<tr><td>{esc(mol.molecule_id)}</td><td>{esc(s.segment_type)}</td>"
                f"<td>{s.start}..{s.end}</td><td>{s.end - s.start}</td><td>{gc}%</td>"
                f"<td>{esc(s.component_id or '-')}</td><td>{esc(s.license_track)}</td>"
                f"<td>{esc(s.action_taken)}</td></tr>"
            )

    metric_rows_html: list[str] = []
    if evaluation is not None:
        for m in evaluation.metrics:
            val_str = f"{m.value:.4f}" if isinstance(m.value, (int, float)) and math.isfinite(m.value) else "N/A"
            metric_rows_html.append(
                f"<tr><td>{esc(m.metric_id)}</td><td>{esc(val_str)}</td><td>{esc(m.unit)}</td>"
                f"<td>{esc(m.tool_name)} ({esc(m.tool_version)})</td>"
                f"<td>{esc(m.chemistry_evaluation_mode)}</td></tr>"
            )

    fto_rows = (
        "".join(
            f"<li><strong>{esc(a.rule_id)}</strong> [{esc(a.matched_scope)} {a.transcript_coordinates[0]}..{a.transcript_coordinates[1]}]: "
            f"{esc(a.summary)} (blocked={a.blocked})</li>"
            for a in outcome.fto_alerts
        )
        or "<li>Zero sequence-level FTO pattern collisions detected in patent_fto_guard_v2.json.</li>"
    )
    adv_rows = (
        "".join(f"<li><strong>{esc(a.category)}</strong>: {esc(a.message)}</li>" for a in outcome.advisories)
        or "<li>No pre-synthesis soft advisories raised.</li>"
    )
    gate_badge = "PASS" if outcome.passed_all_hard_gates else "BLOCKED"
    cstats = clin["candidate_stats"]

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>mRNAx v2.5 Scientific QC Dossier — {esc(bundle.bundle_id)}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; color: #111827; margin: 32px auto; max-width: 980px; line-height: 1.5; }}
  h1 {{ font-size: 22px; border-bottom: 2px solid #111827; padding-bottom: 8px; margin-bottom: 16px; }}
  h2 {{ font-size: 15px; text-transform: uppercase; letter-spacing: 0.04em; color: #374151; margin-top: 24px; border-bottom: 1px solid #e5e7eb; padding-bottom: 4px; }}
  .meta-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; background: #f9fafb; border: 1px solid #e5e7eb; padding: 12px; font-size: 13px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 8px; }}
  th, td {{ border: 1px solid #d1d5db; padding: 6px 8px; text-align: left; }}
  th {{ background: #f3f4f6; font-weight: 600; }}
  .status-pass {{ color: #065f46; font-weight: 700; }}
  .status-fail {{ color: #991b1b; font-weight: 700; }}
  .notice {{ font-size: 12px; color: #4b5563; background: #f9fafb; border-left: 3px solid #6b7280; padding: 8px 12px; margin-top: 16px; }}
  @media print {{ body {{ margin: 12mm; }} }}
</style>
</head>
<body>
<h1>mRNAx v2.5 Construct Quality Control &amp; Compliance Dossier</h1>
<div class="meta-grid">
  <div><strong>Bundle ID:</strong> {esc(bundle.bundle_id)}</div>
  <div><strong>Compliance Track:</strong> {esc(eff_track)}</div>
  <div><strong>Hard Safety Gates:</strong> <span class="{'status-pass' if outcome.passed_all_hard_gates else 'status-fail'}">{gate_badge}</span></div>
  <div><strong>Uridine Chemistry:</strong> {esc(eff_chem)}</div>
  <div><strong>Co-Expression Mode:</strong> {esc(bundle.co_expression_mode)}</div>
  <div><strong>Transcript Length:</strong> {cstats['full_length_nt']} nt (GC {round(cstats['gc_full_transcript'] * 100, 2)}%)</div>
  <div style="grid-column: span 3;"><strong>Version Digest (SHA-256):</strong> <code>{esc(bundle.version_digest)}</code></div>
</div>

<h2>1. Segment Architecture &amp; Clean-Room Provenance</h2>
<table>
  <thead><tr><th>Molecule</th><th>Segment</th><th>Coords (0-based)</th><th>Length (nt)</th><th>GC%</th><th>Component ID</th><th>License Track</th><th>Action</th></tr></thead>
  <tbody>{''.join(seg_rows_html)}</tbody>
</table>

<h2>2. Quantitative Sequence &amp; Evaluation Metrics</h2>
<p style="font-size:13px;">Human CAI: <strong>{cstats['cai_human']}</strong> | CDS GC3: <strong>{cstats['gc3_cds']}</strong> | 5'UTR uAUG Count: <strong>{cstats['utr5_uaug_count']}</strong> | CDS m1Psi U4+ Slippery Runs: <strong>{cstats['cds_m1psi_u4_plus_run_count']}</strong></p>
{f'<table><thead><tr><th>Metric</th><th>Value</th><th>Unit</th><th>Producer</th><th>Chemistry Mode</th></tr></thead><tbody>{"".join(metric_rows_html)}</tbody></table>' if metric_rows_html else '<p style="font-size:13px;">Standalone bundle audit (no session Pareto report attached).</p>'}

<h2>3. Dual-FTO Screening &amp; Pre-Synthesis Advisories</h2>
<ul style="font-size:13px;">{fto_rows}</ul>
<ul style="font-size:13px;">{adv_rows}</ul>

<div class="notice">
  <strong>Scientific &amp; Legal Governance Boundary:</strong> Sequence-level pattern screening verifies absence of cataloged blocked patent motifs and enforces 3-layer Clean-Room component licensing ({esc(eff_track)}), but does not constitute a jurisdiction-specific Freedom-to-Operate legal opinion. Separate computational surrogate predictions and unmodified-U thermodynamic approximations from wet-lab assay evidence.
</div>
</body>
</html>
"""


class _SearchProteinAssetsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: StrictStr


class _AuditConstructInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    bundle_id: StrictStr
    compliance_track: ComplianceProfile | None = None


class _CompareBundlesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    bundle_id_a: StrictStr
    bundle_id_b: StrictStr


class _IngestExperimentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    record_json: dict[str, Any]
    batch_comparable: StrictBool = True
    held_out_validation_passed: StrictBool = False


class _ExplainTradeoffsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    report_id: StrictStr | None = None


class _GetJobStatusInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _ExportBundleInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    bundle_id: StrictStr
    out_dir: StrictStr = ""
    include_dossier: StrictBool = False


class _ResolveTargetProteinInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: StrictStr = Field(min_length=1, max_length=200)
    allow_network: StrictBool = True


class _AnalyzeStructureAndMotifsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    bundle_id: StrictStr | None = None
    sequence: StrictStr | None = Field(default=None, min_length=3, max_length=20000)
    molecule_index: int = Field(default=0, ge=0, le=16)
    modality: Literal["linear_mrna", "circrna", "sarna"] = "linear_mrna"
    include_eternafold: StrictBool = False

    @field_validator("molecule_index", mode="before")
    @classmethod
    def _validate_mol_idx(cls, v: Any) -> int:
        return parse_bounded_json_int(v, field_name="molecule_index", min_val=0, max_val=16)


class _QueryDesignRulesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    modality: StrictStr | None = None
    region: StrictStr | None = None
    grade: StrictStr | None = None
    query: StrictStr | None = Field(default=None, max_length=200)
    include_debunked: StrictBool = True


class _SearchLiteratureAndTrialsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: StrictStr = Field(min_length=1, max_length=300)
    max_results: int = Field(default=5, ge=1, le=20)
    allow_network: StrictBool = True

    @field_validator("max_results", mode="before")
    @classmethod
    def _validate_max_results(cls, v: Any) -> int:
        return parse_bounded_json_int(v, field_name="max_results", min_val=1, max_val=20)


class _BenchmarkAgainstClinicalInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    bundle_id: StrictStr | None = None
    sequence: StrictStr | None = Field(default=None, min_length=3, max_length=20000)
    modality: Literal["linear_mrna", "circrna", "sarna"] | None = None
    chemistry: Literal["m1psi", "pseudouridine", "unmodified_u"] | None = None


def _tool_json_schema(model_cls: type[BaseModel]) -> dict[str, Any]:
    schema = dict(model_cls.model_json_schema())
    schema.pop("title", None)
    return schema


class AtomicToolbox:
    """Exposes the 14 canonical `mrnax_core` tools with JSON-Schema specs for tool-calling LLMs & MCP."""

    def __init__(
        self,
        session: ProjectHarnessSession,
        *,
        turn_id_prefix: str | None = None,
        expected_parent_intent_id: str | None = None,
        default_compliance_track: ComplianceProfile | None = None,
        forced_commercial_clean: bool = False,
        default_modality: str | None = None,
        default_scaffold_id: str | None = None,
        default_protein_sequence: str | None = None,
        default_additional_protein_sequences: list[str] | None = None,
        default_accession_or_source: str | None = None,
    ) -> None:
        self.session = session
        self.constraint_engine = ScopedConstraintEngine.from_seed_dir(session.seed_dir)
        self.turn_id_prefix = turn_id_prefix
        self.expected_parent_intent_id = expected_parent_intent_id
        self.default_compliance_track = default_compliance_track
        self.forced_commercial_clean = bool(forced_commercial_clean)
        self.default_modality = default_modality
        self.default_scaffold_id = default_scaffold_id
        self.default_protein_sequence = default_protein_sequence
        self.default_additional_protein_sequences = default_additional_protein_sequences
        self.default_accession_or_source = default_accession_or_source
        self._call_counter = 0

    def _next_turn_id(self) -> str | None:
        if not self.turn_id_prefix:
            return None
        self._call_counter += 1
        return f"{self.turn_id_prefix}_{len(self.session._intents) + 1:02d}_{self._call_counter:02d}"

    # 1. search_protein_assets
    def search_protein_assets(self, query: str) -> dict[str, Any]:
        snap = json.loads(
            (self.session.seed_dir / "egfp_reporter_snapshot.json").read_text(encoding="utf-8")
        )
        q = query.strip().lower()
        if any(k in q for k in ("egfp", "gfp", "p42212", "u55762")):
            return {
                "matched": True,
                "uniprot_reference": snap["uniprot_reference"],
                "mammalian_reporter_egfp": snap["mammalian_reporter_egfp"],
            }
        return {"matched": False, "note": f"No local cached snapshot for '{query}'; query UniProt/NCBI."}

    # 2. audit_construct
    def audit_construct(
        self, bundle_id: str, compliance_track: str | None = None
    ) -> dict[str, Any]:
        if compliance_track is not None and compliance_track not in ("research_max", "commercial_clean"):
            raise ValueError(f"Unsupported compliance_track '{compliance_track}'")
        with self.session.current_state():
            bundle = self.session.get_bundle(bundle_id)
            intent = self.session.intent_for_bundle(bundle_id)
        from mrnax.io_insdc.genbank_io import resolve_expected_chemistry_for_bundle

        effective_track = intent.compliance_track if compliance_track is None else compliance_track
        eff_chem = resolve_expected_chemistry_for_bundle(bundle, intent.uridine_chemistry)
        outcome = self.constraint_engine.evaluate_bundle(
            bundle,
            intent.constraints,
            compliance_track=effective_track,
            expected_chemistry=eff_chem,
            locked_scopes=intent.locked_scopes,
            expected_protein_targets=intent.protein_targets,
        )
        return {
            "bundle_id": bundle.bundle_id,
            "version_digest": bundle.version_digest,
            "passed_all_hard_gates": outcome.passed_all_hard_gates,
            "fto_blocked": outcome.fto_blocked,
            "violations": [v.model_dump(mode="json") for v in outcome.violations],
            "fto_alerts": [a.model_dump(mode="json") for a in outcome.fto_alerts],
            "advisories": [n.model_dump(mode="json") for n in outcome.advisories],
        }

    # 3. design_constructs
    def design_constructs(
        self,
        *,
        user_utterance: str,
        protein_sequence: str | None = None,
        additional_protein_sequences: list[str] | None = None,
        accession_or_source: str = "GenBank:U55762.1",
        entry_version: str | None = None,
        compliance_track: str | None = None,
        uridine_chemistry: str | None = None,
        restriction_enzymes: list[str] | None = None,
        locked_five_prime_utr: str | None = None,
        detarget_liver: bool | None = None,
        prioritize_late_signal: bool = False,
        conflicting_single_pick: bool = False,
        reference_bundle_id: str | None = None,
        use_preset_reference_r: bool = False,
        use_commercial_surrogate: bool | None = None,
        modality: str | None = None,
        scaffold_id: str | None = None,
        random_seed: int = 42,
    ) -> dict[str, Any]:
        if self.forced_commercial_clean:
            if compliance_track is not None and compliance_track != "commercial_clean":
                raise PermissionError(
                    "Commercial-clean request context forbids escalating compliance_track away from 'commercial_clean'."
                )
            eff_track = "commercial_clean"
            eff_use_surrogate = True if use_commercial_surrogate is None else use_commercial_surrogate
        else:
            eff_track = compliance_track or self.default_compliance_track
            eff_use_surrogate = use_commercial_surrogate
        eff_prot = protein_sequence if protein_sequence is not None else self.default_protein_sequence
        eff_extra = (
            additional_protein_sequences
            if additional_protein_sequences is not None
            else self.default_additional_protein_sequences
        )
        llm_switched_primary = (
            protein_sequence is not None
            and self.default_protein_sequence is not None
            and protein_sequence.strip().upper() != self.default_protein_sequence.strip().upper()
        )
        if accession_or_source != "GenBank:U55762.1":
            eff_acc = accession_or_source
        elif llm_switched_primary:
            eff_acc = "user_supplied"
        else:
            eff_acc = self.default_accession_or_source or accession_or_source
        eff_mod = modality if modality is not None else self.default_modality
        eff_scaf = (
            scaffold_id
            if scaffold_id is not None
            else (
                self.default_scaffold_id
                if (modality is None or modality == self.default_modality)
                else None
            )
        )
        intent = self.session.create_turn_intent_canonical(
            user_utterance=user_utterance,
            protein_sequence=eff_prot,
            additional_protein_sequences=eff_extra,
            accession_or_source=eff_acc,
            entry_version=entry_version,
            compliance_track=eff_track,  # type: ignore[arg-type]
            uridine_chemistry=uridine_chemistry,  # type: ignore[arg-type]
            restriction_enzymes=restriction_enzymes,
            locked_five_prime_utr=locked_five_prime_utr,
            detarget_liver=detarget_liver,
            prioritize_late_signal=prioritize_late_signal,
            conflicting_single_pick=conflicting_single_pick,
            reference_bundle_id=reference_bundle_id,
            use_preset_reference_r=use_preset_reference_r,
            use_commercial_surrogate=eff_use_surrogate,
            modality=eff_mod,  # type: ignore[arg-type]
            scaffold_id=eff_scaf,
            random_seed=random_seed,
            turn_id=self._next_turn_id(),
        )
        self.expected_parent_intent_id = intent.intent_id
        if intent.status != "ready":
            return {
                "intent_id": intent.intent_id,
                "intent_version": intent.intent_version,
                "status": intent.status,
                "blocking_questions": intent.blocking_questions,
            }
        bundles = self.session.execute_design(intent)
        latest_rep = self.session.latest_report(intent.intent_id)
        assert latest_rep is not None
        return {
            "intent_id": intent.intent_id,
            "intent_version": intent.intent_version,
            "outcome_status": latest_rep.outcome_status,
            "recommended_bundle_ids": latest_rep.recommended_bundle_ids,
            "qualified_count": len(bundles),
            "action_transparency": [t.model_dump(mode="json") for t in latest_rep.action_transparency],
        }

    # 4. refine_construct
    def refine_construct(
        self,
        *,
        user_utterance: str,
        updated_restriction_enzymes: list[str] | None = None,
        restriction_enzymes: list[str] | None = None,
        locked_five_prime_utr: str | None = None,
        lock_cds_from_bundle_id: str | None = None,
        unlock_scopes: list[str] | None = None,
        uridine_chemistry: str | None = None,
        compliance_track: str | None = None,
        detarget_liver: bool | None = None,
        use_commercial_surrogate: bool | None = None,
        modality: str | None = None,
        scaffold_id: str | None = None,
    ) -> dict[str, Any]:
        if self.forced_commercial_clean:
            if compliance_track is not None and compliance_track != "commercial_clean":
                raise PermissionError(
                    "Commercial-clean request context forbids escalating compliance_track away from 'commercial_clean'."
                )
            eff_refine_track = "commercial_clean"
            eff_refine_surr = True if use_commercial_surrogate is None else use_commercial_surrogate
        else:
            eff_refine_track = compliance_track
            eff_refine_surr = use_commercial_surrogate
        eff_mod = modality if modality is not None else self.default_modality
        eff_scaf = (
            scaffold_id
            if scaffold_id is not None
            else (
                self.default_scaffold_id
                if (modality is None or modality == self.default_modality)
                else None
            )
        )
        intent_v = self.session.refine_turn_intent_canonical(
            user_utterance=user_utterance,
            restriction_enzymes=restriction_enzymes,
            updated_restriction_enzymes=updated_restriction_enzymes,
            locked_five_prime_utr=locked_five_prime_utr,
            lock_cds_from_bundle_id=lock_cds_from_bundle_id,
            unlock_scopes=unlock_scopes,  # type: ignore[arg-type]
            uridine_chemistry=uridine_chemistry,  # type: ignore[arg-type]
            compliance_track=eff_refine_track,  # type: ignore[arg-type]
            detarget_liver=detarget_liver,
            use_commercial_surrogate=eff_refine_surr,
            modality=eff_mod,  # type: ignore[arg-type]
            scaffold_id=eff_scaf,
            turn_id=self._next_turn_id(),
            expected_parent_intent_id=self.expected_parent_intent_id,
        )
        self.expected_parent_intent_id = intent_v.intent_id
        if intent_v.status != "ready":
            return {
                "intent_id": intent_v.intent_id,
                "intent_version": intent_v.intent_version,
                "parent_intent_id": intent_v.parent_intent_id,
                "status": intent_v.status,
                "blocking_questions": intent_v.blocking_questions,
            }
        bundles = self.session.execute_design(intent_v)
        latest_rep = self.session.latest_report(intent_v.intent_id)
        assert latest_rep is not None
        return {
            "intent_id": intent_v.intent_id,
            "intent_version": intent_v.intent_version,
            "parent_intent_id": intent_v.parent_intent_id,
            "inherited_locks": list(intent_v.locked_scopes.keys()),
            "outcome_status": latest_rep.outcome_status,
            "recommended_bundle_ids": latest_rep.recommended_bundle_ids,
            "qualified_count": len(bundles),
        }

    # 5. compare_bundles
    def compare_bundles(self, bundle_id_a: str, bundle_id_b: str) -> dict[str, Any]:
        with self.session.current_state():
            ba = self.session.get_bundle(bundle_id_a)
            bb = self.session.get_bundle(bundle_id_b)
        comparisons = []
        totals: dict[str, dict[str, Any]] = {}
        for ma, mb in paired_chains(ba, bb):
            diffs = {}
            for scope in ("five_prime_utr", "cds", "stop_cassette", "three_prime_utr", "poly_a"):
                sa = ma.get_segment_sequence(scope)
                sb = mb.get_segment_sequence(scope)
                diffs[scope] = {"identical": sa == sb, "len_a": len(sa), "len_b": len(sb)}
                total = totals.setdefault(scope, {"identical": True, "len_a": 0, "len_b": 0})
                total["identical"] &= sa == sb
                total["len_a"] += len(sa)
                total["len_b"] += len(sb)
            comparisons.append(
                {
                    "molecule_id_a": ma.molecule_id,
                    "molecule_id_b": mb.molecule_id,
                    "segment_comparison": diffs,
                }
            )
        return {
            "bundle_a": bundle_id_a,
            "bundle_b": bundle_id_b,
            "segment_comparison": totals,
            "chain_comparisons": comparisons,
        }

    # 6. ingest_experiment
    def ingest_experiment(
        self, record_json: dict[str, Any], batch_comparable: bool = True, held_out_validation_passed: bool = False
    ) -> dict[str, Any]:
        if type(held_out_validation_passed) is not bool or type(batch_comparable) is not bool:
            raise ValueError("Feedback validation flags must be booleans")
        rec = ExperimentRecord.model_validate(record_json)
        updated = self.session.ingest_experiment_feedback(
            rec,
            batch_comparable=batch_comparable,
            held_out_validation_passed=held_out_validation_passed,
            request_turn_id=self._next_turn_id(),
        )
        return updated.model_dump(mode="json")

    # 7. explain_tradeoffs
    def explain_tradeoffs(self, report_id: str | None = None) -> dict[str, Any]:
        if report_id is not None and not isinstance(report_id, str):
            raise ValueError("report_id must be a string or null")
        with self.session.current_state():
            rep = (
                self.session.report_for_read(self.session._reports[report_id])
                if report_id is not None
                else self.session.latest_report()
            )
            if rep is None:
                raise ValueError("No evaluation report available in session")
            intent = next((i for i in self.session._intents if i.intent_id == rep.intent_id), None)
            projected = project_report_for_intent(
                rep,
                self.session._bundles,
                intent,
                seed_dir=self.session.seed_dir,
                disallow_sandbox_override=True,
            )
        return {
            "report_id": projected.report_id,
            "outcome_status": projected.outcome_status,
            "recommended_bundle_ids": projected.recommended_bundle_ids,
            "candidate_evaluations": [e.model_dump(mode="json") for e in projected.candidate_evaluations],
            "action_transparency": [t.model_dump(mode="json") for t in projected.action_transparency],
            "pre_synthesis_advisories": [a.model_dump(mode="json") for a in projected.pre_synthesis_advisories],
        }

    # 8. get_job_status
    def get_job_status(self) -> dict[str, Any]:
        with self.session.current_state():
            return {
                "project_id": self.session.project_id,
                "intent_versions": len(self.session._intents),
                "bundle_count": len(self.session._bundles),
                "active_reference_bundle_id": self.session.active_reference_bundle_id,
            }

    # 9. export_bundle
    def export_bundle(self, bundle_id: str, out_dir: str = "", include_dossier: bool = False) -> dict[str, Any]:
        if not isinstance(out_dir, str):
            raise ValueError("out_dir must be a string")
        if type(include_dossier) is not bool:
            raise ValueError("include_dossier must be a boolean")
        with self.session.current_state():
            b = self.session.get_bundle(bundle_id)
            intent = self.session.intent_for_export(b)
            base_store = self.session.store_dir.resolve()
            raw_target = Path(out_dir) if out_dir else (base_store / "exports" / bundle_id)
            unres = raw_target if raw_target.is_absolute() else (base_store / raw_target)
            if unres.is_symlink() or any(p.is_symlink() for p in unres.parents if p != base_store.parent):
                raise ValueError(f"Export out_dir '{out_dir}' must not traverse symlinks")
            target_dir = unres.resolve()
            if not target_dir.is_relative_to(base_store) or target_dir == base_store:
                raise ValueError(
                    f"Export out_dir '{out_dir}' must resolve inside project store_dir '{base_store}'"
                )
            for reserved_name in ("intents", "bundles", "reports", "experiments", "manifests"):
                reserved_dir = (base_store / reserved_name).resolve()
                if target_dir == reserved_dir or target_dir.is_relative_to(reserved_dir):
                    raise ValueError(
                        f"Export out_dir '{out_dir}' cannot target protected project directory '{reserved_name}'"
                    )
            checked_files = [
                "full_rna.fasta",
                "cds_only.fasta",
                "construct.gb",
                "chemistry_metadata.json",
                "ivt_synthesis_order.csv",
                "qc_dossier.html",
            ]
            for fn in checked_files:
                cand_f = target_dir / fn
                if cand_f.is_symlink():
                    raise ValueError(f"Refusing to overwrite or remove symlinked export target '{cand_f}'")
            target_dir.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".mrnax_exp_stage_", dir=target_dir.parent) as stg_raw:
                stg_dir = Path(stg_raw) / "pkg"
                stg_dir.mkdir(parents=True, exist_ok=True)
                manifest = export_construct_package(
                    b,
                    stg_dir,
                    constraint_engine=self.constraint_engine,
                    constraints=intent.constraints,
                    compliance_track=intent.compliance_track,
                    expected_chemistry=intent.uridine_chemistry,
                    locked_scopes=intent.locked_scopes,
                    expected_protein_targets=intent.protein_targets,
                )
                staged_names = [
                    "full_rna.fasta",
                    "cds_only.fasta",
                    "construct.gb",
                    "chemistry_metadata.json",
                ]
                if include_dossier:
                    ev_match: BundleEvaluation | None = None
                    try:
                        rep = self.session.latest_report(intent.intent_id)
                        if rep is not None:
                            ev_match = next((e for e in rep.candidate_evaluations if e.bundle_id == b.bundle_id), None)
                    except ValueError:
                        ev_match = None
                    csv_text, _ = render_ivt_synthesis_order(b, intent=intent)
                    html_text = render_qc_dossier_html(
                        b,
                        intent=intent,
                        constraint_engine=self.constraint_engine,
                        evaluation=ev_match,
                        seed_dir=self.session.seed_dir,
                    )
                    csv_path = stg_dir / "ivt_synthesis_order.csv"
                    html_path = stg_dir / "qc_dossier.html"
                    csv_path.write_text(csv_text, encoding="utf-8")
                    html_path.write_text(html_text, encoding="utf-8")
                    manifest["files"]["ivt_synthesis_order.csv"] = {
                        "sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
                        "bytes": csv_path.stat().st_size,
                    }
                    manifest["files"]["qc_dossier.html"] = {
                        "sha256": hashlib.sha256(html_path.read_bytes()).hexdigest(),
                        "bytes": html_path.stat().st_size,
                    }
                    (stg_dir / "chemistry_metadata.json").write_text(
                        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                    )
                    staged_names.extend(["ivt_synthesis_order.csv", "qc_dossier.html"])

                target_dir.mkdir(parents=True, exist_ok=True)
                for fn in checked_files:
                    cand_f = target_dir / fn
                    if cand_f.is_symlink():
                        raise ValueError(f"Refusing to overwrite or remove symlinked export target '{cand_f}'")

                bak_dir = Path(stg_raw) / "bak"
                bak_dir.mkdir(parents=True, exist_ok=True)
                prior_files: set[str] = set()
                for fn in checked_files:
                    cur_f = target_dir / fn
                    if cur_f.is_file():
                        shutil.copy2(cur_f, bak_dir / fn)
                        prior_files.add(fn)
                try:
                    for fn in staged_names:
                        os.replace(stg_dir / fn, target_dir / fn)
                    if not include_dossier:
                        (target_dir / "ivt_synthesis_order.csv").unlink(missing_ok=True)
                        (target_dir / "qc_dossier.html").unlink(missing_ok=True)
                except Exception:
                    for fn in checked_files:
                        dst_f = target_dir / fn
                        bak_f = bak_dir / fn
                        if fn in prior_files and bak_f.is_file():
                            os.replace(bak_f, dst_f)
                        elif fn not in prior_files:
                            dst_f.unlink(missing_ok=True)
                    raise
            return manifest

    # 10. resolve_target_protein
    def resolve_target_protein(self, query: str, allow_network: bool = True) -> dict[str, Any]:
        if type(allow_network) is not bool:
            raise ValueError("allow_network must be a boolean")
        adapter = ExternalEvidenceAdapterV2(license_track="commercial_clean")
        return adapter.resolve_target_protein(query, allow_network=allow_network)

    # 11. analyze_structure_and_motifs
    def analyze_structure_and_motifs(
        self,
        *,
        bundle_id: str | None = None,
        sequence: str | None = None,
        molecule_index: int = 0,
        modality: str = "linear_mrna",
        include_eternafold: bool = False,
    ) -> dict[str, Any]:
        if bundle_id is None and sequence is None:
            raise ValueError("Provide either bundle_id or sequence")
        if bundle_id is not None:
            with self.session.current_state():
                b = self.session.get_bundle(bundle_id)
                try:
                    b_intent = self.session.intent_for_bundle(bundle_id)
                    eff_track = b_intent.compliance_track
                except Exception:
                    eff_track = "commercial_clean"
            if self.forced_commercial_clean:
                eff_track = "commercial_clean"
            if molecule_index < 0 or molecule_index >= len(b.molecules):
                raise ValueError(f"molecule_index {molecule_index} out of range for bundle '{bundle_id}'")
            mol = b.molecules[molecule_index]
            u5 = mol.get_segment("five_prime_utr")
            cds = mol.get_segment("cds")
            u3 = mol.get_segment("three_prime_utr")
            res = run_structure_and_motif_analysis(
                mol.sequence,
                five_prime_end=u5.end if u5 else None,
                cds_start=cds.start if cds else None,
                cds_end=cds.end if cds else None,
                three_prime_start=u3.start if u3 else None,
                three_prime_end=u3.end if u3 else None,
                modality=mol.modality,
                compliance_track=eff_track,
                include_eternafold=include_eternafold,
                seed_dir=self.session.seed_dir,
            )
            res["bundle_id"] = b.bundle_id
            res["molecule_id"] = mol.molecule_id
            return res
        assert sequence is not None
        with self.session.current_state():
            seq_track = (
                "commercial_clean"
                if self.forced_commercial_clean
                else (
                    self.session._intents[-1].compliance_track
                    if self.session._intents
                    else "commercial_clean"
                )
            )
        return run_structure_and_motif_analysis(
            sequence,
            modality=modality,
            compliance_track=seq_track,
            include_eternafold=include_eternafold,
            seed_dir=self.session.seed_dir,
        )

    # 12. query_design_rules
    def query_design_rules(
        self,
        *,
        modality: str | None = None,
        region: str | None = None,
        grade: str | None = None,
        query: str | None = None,
        include_debunked: bool = True,
    ) -> dict[str, Any]:
        return query_scientific_rules(
            seed_dir=self.session.seed_dir,
            modality=modality,
            region=region,
            grade=grade,
            query=query,
            include_debunked=include_debunked,
        )

    # 13. search_literature_and_trials
    def search_literature_and_trials(
        self,
        *,
        query: str,
        max_results: int = 5,
        allow_network: bool = True,
    ) -> dict[str, Any]:
        adapter = ExternalEvidenceAdapterV2(license_track="commercial_clean")
        return adapter.search_literature_and_trials(query, max_results=max_results, allow_network=allow_network)

    # 14. benchmark_against_clinical
    def benchmark_against_clinical(
        self,
        *,
        bundle_id: str | None = None,
        sequence: str | None = None,
        modality: str | None = None,
        chemistry: str | None = None,
    ) -> dict[str, Any]:
        if bundle_id is not None:
            with self.session.current_state():
                b = self.session.get_bundle(bundle_id)
            return compare_construct_against_clinical_benchmarks(
                seed_dir=self.session.seed_dir,
                bundle=b,
                modality=modality,
                chemistry=chemistry,
            )
        return compare_construct_against_clinical_benchmarks(
            seed_dir=self.session.seed_dir,
            sequence=sequence,
            modality=modality,
            chemistry=chemistry,
        )

    _TOOL_INPUT_MODELS: dict[str, type[BaseModel]] = {
        "search_protein_assets": _SearchProteinAssetsInput,
        "audit_construct": _AuditConstructInput,
        "design_constructs": InitialTurnInput,
        "refine_construct": RefineTurnInput,
        "compare_bundles": _CompareBundlesInput,
        "ingest_experiment": _IngestExperimentInput,
        "explain_tradeoffs": _ExplainTradeoffsInput,
        "get_job_status": _GetJobStatusInput,
        "export_bundle": _ExportBundleInput,
        "resolve_target_protein": _ResolveTargetProteinInput,
        "analyze_structure_and_motifs": _AnalyzeStructureAndMotifsInput,
        "query_design_rules": _QueryDesignRulesInput,
        "search_literature_and_trials": _SearchLiteratureAndTrialsInput,
        "benchmark_against_clinical": _BenchmarkAgainstClinicalInput,
    }
    _ALLOWED_TOOL_NAMES: frozenset[str] = frozenset(_TOOL_INPUT_MODELS.keys())

    def openai_tool_schemas(self) -> list[dict[str, Any]]:
        specs = [
            (
                "search_protein_assets",
                "Look up verified protein sequence & version snapshot (e.g. eGFP P42212 / U55762.1).",
            ),
            (
                "design_constructs",
                "Create versioned DesignIntent and run real mRNAx optimization engine.",
            ),
            (
                "refine_construct",
                "Modify a single constraint/lock on the active session while inheriting all other locked items.",
            ),
            (
                "explain_tradeoffs",
                "Fetch deterministic metrics, Pareto ranks, and action_transparency for the latest run.",
            ),
            (
                "compare_bundles",
                "Compare segment-by-segment differences between two ConstructBundles.",
            ),
            (
                "audit_construct",
                "Run shared ScopedConstraintEngine and sequence_fto_guard on a ConstructBundle.",
            ),
            (
                "ingest_experiment",
                "Record an experiment with its batch comparability and held-out validation result.",
            ),
            (
                "get_job_status",
                "Return current session project_id, intent_versions, bundle_count, and active_reference_bundle_id.",
            ),
            (
                "export_bundle",
                "Export full RNA FASTA, CDS FASTA, INSDC GenBank, chemistry_metadata.json (plus optional IVT CSV & QC dossier).",
            ),
            (
                "resolve_target_protein",
                "Resolve gene symbols, UniProt IDs, or RefSeq accessions via UniProt, AlphaFold DB, HPA, GTEx, and ClinicalTrials.gov.",
            ),
            (
                "analyze_structure_and_motifs",
                "Compute 2D initiation base-pair probability arcs, dot-bracket, sliding-window folding tracks, and sequence motifs.",
            ),
            (
                "query_design_rules",
                "Query quantitative RNA sequence design rules and debunked traps filtered by modality, region, grade, or keyword.",
            ),
            (
                "search_literature_and_trials",
                "Search EuropePMC open-access papers (with license firewall) and ClinicalTrials.gov studies.",
            ),
            (
                "benchmark_against_clinical",
                "Compare a candidate bundle or RNA sequence quantitatively against published clinical mRNA benchmarks.",
            ),
        ]
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": desc,
                    "parameters": _tool_json_schema(self._TOOL_INPUT_MODELS[name]),
                },
            }
            for name, desc in specs
        ]

    def dispatch_tool_call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        model_cls = self._TOOL_INPUT_MODELS.get(name)
        if model_cls is None:
            raise ValueError(f"Disallowed or unknown tool '{name}'")
        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be a JSON object")
        parsed = model_cls.model_validate(arguments)
        fn = getattr(self, name, None)
        if fn is None or not callable(fn):
            raise ValueError(f"Unknown tool '{name}'")
        return fn(**parsed.model_dump())

"""2D RNA Secondary Structure, Initiation Base-Pair Arcs, Sliding Tracks & Motif Profiler."""

from __future__ import annotations

import functools
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any

from mrnax.constraints.sequence_gates import RESTRICTION_SITES_RNA
from mrnax.engine.profiles import resolve_default_seed_dir
from mrnax.engine.surrogates import CODON_W

_STOP_CODONS_RNA = frozenset({"UAA", "UAG", "UGA"})
_CLASS_II_T7_RNA = "AUCUGUU"
_G4_RE = re.compile(r"G{3,}[ACGU]{1,7}G{3,}[ACGU]{1,7}G{3,}[ACGU]{1,7}G{3,}")
_HOMOPOLYMER_RE = re.compile(r"(A{6,}|C{6,}|G{6,}|U{6,})")
_UUUU_RE = re.compile(r"U{4,}")

_WAVE1B_DARWIN_EF_BIN = Path(
    "/Users/alex/.gemini/antigravity/brain/4db4d24c-55e0-4968-869d-a93a1c9c7b7b/"
    "scratch/wave1_sota_models/bin/darwin_arm64/contrafold"
)
_WAVE1B_EF_PARAMS = Path(
    "/Users/alex/.gemini/antigravity/brain/4db4d24c-55e0-4968-869d-a93a1c9c7b7b/"
    "scratch/wave1_sota_models/params/EternaFoldParams.v1"
)
_WAVE1A_FALLBACK_ATLAS = Path(
    "/Users/alex/.gemini/antigravity/brain/4db4d24c-55e0-4968-869d-a93a1c9c7b7b/"
    "scratch/wave1_data_rules/tissue_mirna_atlas_v1.json"
)


def _normalize_rna(sequence: str) -> str:
    seq = re.sub(r"[^ACGTUacgtu]", "", str(sequence)).upper().replace("T", "U")
    if not seq:
        raise ValueError("RNA sequence must contain at least one valid nucleotide (A, C, G, U).")
    return seq


def _try_vienna() -> Any | None:
    try:
        import RNA  # type: ignore[import-untyped]

        return RNA
    except Exception:
        return None


def discover_eternafold_binary() -> tuple[Path, Path] | None:
    """Locate executable `contrafold` binary and `EternaFoldParams.v1` parameters."""
    bin_cands: list[Path] = []
    env_bin = os.environ.get("MRNAX_ETERNAFOLD_BIN", "").strip()
    if env_bin:
        bin_cands.append(Path(env_bin).expanduser())
    bin_cands.append(Path.home() / ".mrnax" / "bin" / "contrafold")
    bin_cands.append(_WAVE1B_DARWIN_EF_BIN)
    which_cf = shutil.which("contrafold")
    if which_cf:
        bin_cands.append(Path(which_cf))

    param_cands: list[Path] = []
    env_params = os.environ.get("MRNAX_ETERNAFOLD_PARAMS", "").strip()
    if env_params:
        param_cands.append(Path(env_params).expanduser())
    param_cands.append(Path.home() / ".mrnax" / "bin" / "EternaFoldParams.v1")
    param_cands.append(_WAVE1B_EF_PARAMS)

    resolved_bin = next((b.resolve() for b in bin_cands if b.is_file() and os.access(b, os.X_OK)), None)
    resolved_params = next((p.resolve() for p in param_cands if p.is_file()), None)
    if resolved_bin is not None and resolved_params is not None:
        return resolved_bin, resolved_params
    return None


@functools.lru_cache(maxsize=4)
def _load_tissue_mirna_entries(seed_dir_str: str) -> tuple[dict[str, Any], ...]:
    seed_dir = Path(seed_dir_str)
    for path in (
        seed_dir / "tissue_mirna_atlas_v1.json",
        _WAVE1A_FALLBACK_ATLAS,
        seed_dir / "mirna_rbp_commercial_v2.json",
    ):
        if path.is_file():
            doc = json.loads(path.read_text(encoding="utf-8"))
            return tuple(doc.get("mirna_entries", []))
    return ()


def _validate_coord_int(name: str, val: int | None, n: int, *, allow_end: bool = True) -> int | None:
    if val is None:
        return None
    if isinstance(val, bool) or not isinstance(val, int):
        raise ValueError(f"Coordinate '{name}' must be an integer, got {val!r}")
    upper = n if allow_end else (n - 1)
    if val < 0 or val > upper:
        raise ValueError(f"Coordinate '{name}' ({val}) out of bounds [0, {upper}]")
    return val


def _resolve_region_coordinates(
    seq: str,
    *,
    five_prime_end: int | None,
    cds_start: int | None,
    cds_end: int | None,
    three_prime_start: int | None,
    three_prime_end: int | None,
) -> tuple[int, int, int, int, int]:
    n = len(seq)
    v_u5_end = _validate_coord_int("five_prime_end", five_prime_end, n, allow_end=True)
    v_c_start = _validate_coord_int("cds_start", cds_start, n, allow_end=False)
    v_c_end = _validate_coord_int("cds_end", cds_end, n, allow_end=True)
    v_u3_start = _validate_coord_int("three_prime_start", three_prime_start, n, allow_end=True)
    v_u3_end = _validate_coord_int("three_prime_end", three_prime_end, n, allow_end=True)

    if v_u5_end is not None and v_c_start is not None and v_u5_end > v_c_start:
        raise ValueError(f"Invalid coordinates: five_prime_end ({v_u5_end}) > cds_start ({v_c_start})")
    if v_c_start is not None and v_c_end is not None and v_c_end <= v_c_start:
        raise ValueError(f"Invalid half-open CDS interval [{v_c_start}, {v_c_end}): cds_end must exceed cds_start")
    if v_c_end is not None and v_u3_start is not None and v_c_end > v_u3_start:
        raise ValueError(f"Invalid coordinates: cds_end ({v_c_end}) > three_prime_start ({v_u3_start})")
    if v_u3_start is not None and v_u3_end is not None and v_u3_end < v_u3_start:
        raise ValueError(f"Invalid coordinates: three_prime_end ({v_u3_end}) < three_prime_start ({v_u3_start})")
    if v_c_start is not None and v_u3_start is not None and v_u3_start <= v_c_start:
        raise ValueError(f"Invalid coordinates: three_prime_start ({v_u3_start}) <= cds_start ({v_c_start})")

    if v_c_start is not None:
        c_start = v_c_start
        u5_end = c_start if v_u5_end is None else v_u5_end
    elif v_u5_end is not None:
        u5_end = v_u5_end
        c_start = min(u5_end, max(0, n - 1))
    else:
        if seq.startswith("AUG"):
            u5_end = 0
            c_start = 0
        else:
            search_hi = min(n, 900)
            aug_hits = [m.start() for m in re.finditer("AUG", seq[:search_hi])]
            kozak_hit = next(
                (p for p in aug_hits if p >= 3 and seq[p - 3] in "AG"),
                aug_hits[0] if aug_hits else 0,
            )
            u5_end = kozak_hit
            c_start = kozak_hit

    if v_c_end is not None:
        c_end = v_c_end
    elif v_u3_start is not None:
        cand_end = v_u3_start
        if cand_end - c_start >= 3 and seq[cand_end - 3 : cand_end] in _STOP_CODONS_RNA:
            c_end = cand_end - 3
        else:
            c_end = cand_end
    else:
        c_end = n
        for q in range(c_start + 3, n - 2, 3):
            if seq[q : q + 3] in _STOP_CODONS_RNA:
                c_end = q
                break

    if v_u3_start is not None:
        u3_start = v_u3_start
    else:
        u3_start = c_end + 3 if seq[c_end : c_end + 3] in _STOP_CODONS_RNA else c_end
        u3_start = min(n, u3_start)

    if v_u3_end is not None:
        u3_end = v_u3_end
    else:
        polya_m = re.search(r"A{20,}$", seq)
        if polya_m is not None and polya_m.start() >= u3_start:
            u3_end = polya_m.start()
        else:
            u3_end = n

    return u5_end, c_start, c_end, u3_start, u3_end


def _compute_cai(cds_rna: str) -> float | None:
    dna = cds_rna.replace("U", "T")
    n_cod = len(dna) // 3
    if n_cod < 1:
        return None
    logs = [
        math.log(CODON_W[dna[3 * i : 3 * i + 3]])
        for i in range(n_cod)
        if dna[3 * i : 3 * i + 3] in CODON_W
    ]
    return round(math.exp(sum(logs) / len(logs)), 4) if logs else None


def _run_eternafold_window_metrics(
    seq: str,
    cds_start: int,
    contrafold_bin: Path,
    ef_params: Path,
) -> dict[str, float] | None:
    win_lo = max(0, cds_start - 35)
    win_hi = min(len(seq), cds_start + 55)
    sub = seq[win_lo:win_hi]
    n_len = len(sub)
    if n_len < 6:
        return None
    with tempfile.TemporaryDirectory() as td:
        fa = Path(td) / "in.fa"
        post = Path(td) / "post.txt"
        fa.write_text(f">seq\n{sub}\n", encoding="utf-8")
        try:
            subprocess.run(
                [
                    str(contrafold_bin),
                    "predict",
                    str(fa),
                    "--params",
                    str(ef_params),
                    "--posteriors",
                    "0.0001",
                    str(post),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30,
            )
        except (subprocess.SubprocessError, OSError):
            return None
        if not post.is_file():
            return None
        paired = [0.0] * n_len
        upper: dict[tuple[int, int], float] = {}
        lower: dict[tuple[int, int], float] = {}
        positions = 0
        for line in post.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) < 2 or not re.fullmatch(r"[0-9]+", parts[0]):
                return None
            i = positions
            base_ch = parts[1].upper().replace("T", "U")
            if i >= n_len or parts[0] != str(i + 1) or base_ch != sub[i]:
                return None
            positions += 1
            for tok in parts[2:]:
                if ":" not in tok:
                    return None
                j_s, p_s = tok.split(":", 1)
                if not re.fullmatch(r"[1-9][0-9]*", j_s) or len(j_s) > len(str(n_len)):
                    return None
                j = int(j_s) - 1
                try:
                    p = float(p_s)
                except ValueError:
                    return None
                if not (0 <= j < n_len) or j == i or not math.isfinite(p) or p < 0.0 or p > 1.0:
                    return None
                triangle, pair = (upper, (i, j)) if i < j else (lower, (j, i))
                if pair in triangle:
                    return None
                triangle[pair] = p
        if positions != n_len or (lower and lower != upper):
            return None
        for (i, j), prob in upper.items():
            paired[i] += prob
            paired[j] += prob
        if any(pv > 1.001 for pv in paired):
            return None
        unpaired = [max(0.0, min(1.0, 1.0 - pv)) for pv in paired]
        rel_aug = cds_start - win_lo
        k_lo = max(0, rel_aug - 15)
        k_hi = min(n_len, rel_aug + 18)
        if k_hi <= k_lo:
            return None
        k_vals = unpaired[k_lo:k_hi]
        return {
            "kozak_unpaired_prob": round(sum(k_vals) / len(k_vals), 5),
            "aup_mean": round(sum(unpaired) / n_len, 5),
        }


def _scan_motifs(
    seq: str,
    *,
    five_prime_end: int,
    cds_start: int,
    cds_end: int,
    three_prime_start: int,
    three_prime_end: int,
    modality: str,
    seed_dir: Path,
) -> list[dict[str, Any]]:
    motifs: list[dict[str, Any]] = []
    n = len(seq)

    # 1. uAUG / uORF in 5' leader [0, five_prime_end)
    if five_prime_end >= 3:
        u5 = seq[:five_prime_end]
        for m in re.finditer("AUG", u5):
            p = m.start()
            stop_end = None
            for q in range(p + 3, len(u5) - 2, 3):
                if u5[q : q + 3] in _STOP_CODONS_RNA:
                    stop_end = q + 3
                    break
            in_frame_with_cds = (cds_start - p) % 3 == 0
            sev = "info" if modality == "circrna" else "error"
            if stop_end is not None:
                motifs.append(
                    {
                        "kind": "uorf",
                        "label": "uORF",
                        "start": p,
                        "end": stop_end,
                        "severity": sev,
                        "detail": (
                            f"Upstream open reading frame [{p}, {stop_end}) in 5' leader "
                            f"({'in-frame' if in_frame_with_cds else 'out-of-frame'} with main AUG)."
                        ),
                    }
                )
            else:
                motifs.append(
                    {
                        "kind": "uaug",
                        "label": "uAUG",
                        "start": p,
                        "end": p + 3,
                        "severity": sev,
                        "detail": (
                            f"Upstream AUG codon at {p} in 5' leader "
                            f"({'in-frame' if in_frame_with_cds else 'out-of-frame'} with main AUG)."
                        ),
                    }
                )

    # 2. UUUU slippery frameshift runs inside CDS [cds_start, cds_end)
    if cds_end > cds_start:
        cds_sub = seq[cds_start:cds_end]
        for m in _UUUU_RE.finditer(cds_sub):
            s = cds_start + m.start()
            e = cds_start + m.end()
            motifs.append(
                {
                    "kind": "uuuu_frameshift",
                    "label": f"Poly-U ({len(m.group(0))}nt)",
                    "start": s,
                    "end": e,
                    "severity": "warning",
                    "detail": f"Slippery {m.group(0)} tract in CDS (Mulroney et al. 2024 m1Psi +1 ribosomal frameshift risk).",
                }
            )

    # 3. Homopolymers >= 6 outside poly(A) tail [0, three_prime_end)
    non_polya = seq[:three_prime_end]
    for m in _HOMOPOLYMER_RE.finditer(non_polya):
        run_s = m.group(0)
        motifs.append(
            {
                "kind": "homopolymer",
                "label": f"{run_s[0]}x{len(run_s)}",
                "start": m.start(),
                "end": m.end(),
                "severity": "error" if len(run_s) >= 8 else "warning",
                "detail": f"Homopolymer run {run_s[0]}x{len(run_s)} (IVT polymerase slippage / synthesis risk).",
            }
        )

    # 4. Canonical G-quadruplex motifs
    for m in _G4_RE.finditer(seq):
        motifs.append(
            {
                "kind": "g_quadruplex",
                "label": "G4 motif",
                "start": m.start(),
                "end": m.end(),
                "severity": "warning",
                "detail": "Putative RNA G-quadruplex (G3+N1-7)x4 stalling ribosome scanning or elongation.",
            }
        )

    # 5. Class II T7 pause/termination signal
    start_pos = 0
    while True:
        idx = non_polya.find(_CLASS_II_T7_RNA, start_pos)
        if idx < 0:
            break
        motifs.append(
            {
                "kind": "t7_terminator",
                "label": "T7 Class-II Terminator",
                "start": idx,
                "end": idx + len(_CLASS_II_T7_RNA),
                "severity": "error",
                "detail": f"Class II T7 RNA polymerase premature termination motif '{_CLASS_II_T7_RNA}'.",
            }
        )
        start_pos = idx + 1

    # 6. Restriction enzyme recognition sites (deduping SapI/BspQI identical recognition site)
    seen_enz_spans: set[tuple[int, int, str]] = set()
    enz_labels_by_site: dict[str, list[str]] = {}
    for enz, (fwd, rev) in RESTRICTION_SITES_RNA.items():
        for site in {fwd, rev}:
            enz_labels_by_site.setdefault(site, []).append(enz)
    for site, enz_names in sorted(enz_labels_by_site.items()):
        joined_label = "/".join(sorted(enz_names))
        s_idx = 0
        while True:
            hit = seq.find(site, s_idx)
            if hit < 0:
                break
            key = (hit, hit + len(site), joined_label)
            if key not in seen_enz_spans:
                seen_enz_spans.add(key)
                motifs.append(
                    {
                        "kind": "restriction_site",
                        "label": joined_label,
                        "start": hit,
                        "end": hit + len(site),
                        "severity": "warning",
                        "detail": f"Recognition site '{site}' for {joined_label} template linearization/assembly.",
                    }
                )
            s_idx = hit + 1

    # 7. AU-rich element (ARE) decay motifs in 3'UTR (or full sequence when no distinct 3'UTR)
    are_lo = three_prime_start if three_prime_end > three_prime_start else 0
    are_hi = three_prime_end if three_prime_end > three_prime_start else n
    are_region = seq[are_lo:are_hi]
    for pat, lbl in (("UUAUUUAUU", "ARE nonamer"), ("AUUUA", "ARE pentamer")):
        s_idx = 0
        while True:
            hit = are_region.find(pat, s_idx)
            if hit < 0:
                break
            g_lo = are_lo + hit
            g_hi = g_lo + len(pat)
            if pat == "AUUUA" and any(
                m["kind"] == "are_motif" and m["start"] <= g_lo and g_hi <= m["end"]
                for m in motifs
            ):
                s_idx = hit + 1
                continue
            motifs.append(
                {
                    "kind": "are_motif",
                    "label": lbl,
                    "start": g_lo,
                    "end": g_hi,
                    "severity": "warning",
                    "detail": f"Destabilizing {lbl} ('{pat}') recruiting ZFP36/tristetraprolin deadenylation.",
                }
            )
            s_idx = hit + 1

    # 8. Tissue-specific miRNA full-complement & 8-mer seed matches (tissue_mirna_atlas_v1.json)
    mirna_entries = _load_tissue_mirna_entries(str(seed_dir.resolve()))
    recorded_mirna_spans: list[tuple[int, int, str]] = []
    for entry in mirna_entries:
        mir_id = str(entry.get("mirna_id", "miRNA"))
        tissue = str(
            entry.get("tissue_group")
            or entry.get("fantom5_compartment")
            or entry.get("top_enriched_tissue")
            or "tissue"
        )
        full_site = str(
            entry.get("mrna_full_complementary_ago2_cleavage_site_rna")
            or entry.get("full_complementary_site_22nt")
            or ""
        ).upper()
        seed8 = str(
            entry.get("mrna_target_site_8mer_rna") or entry.get("seed_8mer_site") or ""
        ).upper()

        if len(full_site) >= 18:
            s_idx = 0
            while True:
                hit = seq.find(full_site, s_idx)
                if hit < 0:
                    break
                end_h = hit + len(full_site)
                recorded_mirna_spans.append((hit, end_h, mir_id))
                motifs.append(
                    {
                        "kind": "mirna_site",
                        "label": f"{mir_id} (Ago2 22nt)",
                        "start": hit,
                        "end": end_h,
                        "severity": "info",
                        "detail": f"Full complementary Ago2 cleavage switch for {mir_id} ({tissue}).",
                    }
                )
                s_idx = hit + 1

        if len(seed8) >= 7:
            s_idx = 0
            while True:
                hit = seq.find(seed8, s_idx)
                if hit < 0:
                    break
                end_h = hit + len(seed8)
                if not any(
                    mid == mir_id and lo <= hit and end_h <= hi
                    for lo, hi, mid in recorded_mirna_spans
                ):
                    motifs.append(
                        {
                            "kind": "mirna_site",
                            "label": f"{mir_id} (8mer seed)",
                            "start": hit,
                            "end": end_h,
                            "severity": "info",
                            "detail": f"Canonical 8mer seed match ('{seed8}') for {mir_id} ({tissue}).",
                        }
                    )
                s_idx = hit + 1

    motifs.sort(key=lambda item: (int(item["start"]), int(item["end"]), str(item["kind"])))
    return motifs


def analyze_rna_structure(
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
    """Compute 2D initiation base-pair arcs, ensemble metrics, sliding tracks, and motif coordinates."""
    seq = _normalize_rna(sequence)
    n = len(seq)
    eff_seed = seed_dir or resolve_default_seed_dir()
    u5_end, c_start, c_end, u3_start, u3_end = _resolve_region_coordinates(
        seq,
        five_prime_end=five_prime_end,
        cds_start=cds_start,
        cds_end=cds_end,
        three_prime_start=three_prime_start,
        three_prime_end=three_prime_end,
    )

    full_gc_pct = round(100.0 * (seq.count("G") + seq.count("C")) / n, 2)
    full_u_pct = round(100.0 * seq.count("U") / n, 2)
    cai_val = _compute_cai(seq[c_start:c_end]) if c_end - c_start >= 3 else None

    rna_mod = _try_vienna() if compliance_track != "commercial_clean" else None
    engines_used: list[str] = []

    init_lo = max(0, c_start - 30)
    init_hi = min(n, c_start + 45)
    init_seq = seq[init_lo:init_hi]
    aug_in_win = c_start - init_lo
    arcs: list[dict[str, Any]] = []
    init_dot_bracket = "." * len(init_seq)

    if rna_mod is not None:
        engines_used.append("ViennaRNA")
        # 1. Initiation window [-30, +45] exact partition function & BPP arcs (prob >= 0.15)
        if len(init_seq) >= 4:
            fc_init = rna_mod.fold_compound(init_seq)
            init_ss, init_dg = fc_init.mfe()
            init_dot_bracket = str(init_ss)
            fc_init.exp_params_rescale(init_dg)
            fc_init.pf()
            bpp_init = fc_init.bpp()
            wlen = len(init_seq)
            for i in range(1, wlen + 1):
                row_i = bpp_init[i]
                for j in range(i + 1, wlen + 1):
                    prob = float(row_i[j])
                    if prob >= 0.15:
                        arcs.append({"i": i - 1, "j": j - 1, "prob": round(prob, 4)})

        # 2. Kozak unpaired probability [-15, +18] via local scanning accessibility
        k_win_lo = max(0, c_start - 35)
        k_win_hi = min(n, c_start + 55)
        k_sub = seq[k_win_lo:k_win_hi]
        if len(k_sub) >= 12:
            w_k = min(len(k_sub), 70)
            l_k = min(w_k, 50)
            up_k = rna_mod.pfl_fold_up(k_sub, 1, w_k, l_k)
            rel_aug = c_start - k_win_lo
            p_lo = max(0, rel_aug - 15)
            p_hi = min(len(k_sub), rel_aug + 18)
            k_probs = [float(up_k[pos + 1][1]) for pos in range(p_lo, p_hi)]
            kozak_unpaired_prob = round(sum(k_probs) / max(1, len(k_probs)), 6)
        else:
            kozak_unpaired_prob = 0.5

        # 3. Full-sequence MFE, ensemble diversity, and per-nucleotide unpaired probabilities
        is_circ = modality == "circrna"
        if n <= 1800:
            if is_circ:
                md = rna_mod.md()
                md.circ = 1
                fc_full = rna_mod.fold_compound(seq, md)
            else:
                fc_full = rna_mod.fold_compound(seq)
            _, full_dg = fc_full.mfe()
            fc_full.exp_params_rescale(full_dg)
            fc_full.pf()
            ens_div = float(fc_full.mean_bp_distance())
            bpp_full = fc_full.bpp()
            paired_full = [0.0] * n
            for i in range(1, n + 1):
                row_i = bpp_full[i]
                for j in range(i + 1, n + 1):
                    p = float(row_i[j])
                    if p > 0.0:
                        paired_full[i - 1] += p
                        paired_full[j - 1] += p
            unpaired_per_nt = [max(0.0, min(1.0, 1.0 - pv)) for pv in paired_full]
        else:
            w_g = min(n, 120)
            l_g = min(w_g, 80)
            up_global = rna_mod.pfl_fold_up(seq, 1, w_g, l_g)
            unpaired_per_nt = [max(0.0, min(1.0, float(up_global[idx + 1][1]))) for idx in range(n)]
            if n <= 3500:
                if is_circ:
                    md = rna_mod.md()
                    md.circ = 1
                    _, full_dg = rna_mod.fold_compound(seq, md).mfe()
                else:
                    _, full_dg = rna_mod.fold(seq)
            else:
                chunk_w = 400
                chunk_s = 200
                starts = list(range(0, n - chunk_w + 1, chunk_s))
                last_cov = starts[-1] + chunk_w if starts else 0
                w_sum = sum(float(rna_mod.fold(seq[i : i + chunk_w])[1]) for i in starts)
                w_tot = float(len(starts) * chunk_w)
                if last_cov < n:
                    rem_nt = n - last_cov
                    w_sum += float(rna_mod.fold(seq[n - chunk_w :])[1]) * (rem_nt / chunk_w)
                    w_tot += float(rem_nt)
                full_dg = w_sum * (n / max(1.0, w_tot))
            # Local pairing entropy proxy for ultra-long constructs (>1800 nt)
            ens_div = sum(4.0 * u * (1.0 - u) for u in unpaired_per_nt) * 0.25

        aup_mean = round(sum(unpaired_per_nt) / n, 5)
        dg_total = round(float(full_dg), 3)
        ensemble_diversity = round(float(ens_div), 4)
    else:
        engines_used.append("mRNAxPureSeqApprox")
        unpaired_per_nt = [0.65 if ch in "AU" else 0.35 for ch in seq]
        aup_mean = round(sum(unpaired_per_nt) / n, 5)
        kozak_slice = unpaired_per_nt[max(0, c_start - 15) : min(n, c_start + 18)] or unpaired_per_nt
        kozak_unpaired_prob = round(sum(kozak_slice) / max(1, len(kozak_slice)), 6)
        dg_total = round(-0.42 * (seq.count("G") + seq.count("C")), 3)
        ensemble_diversity = round(0.12 * n, 4)

    # 4. Sliding-window tracks
    win_nt = min(n, 60 if n < 180 else 120)
    step_nt = max(10, win_nt // 3)
    positions = list(range(0, max(1, n - win_nt + 1), step_nt))
    if n > win_nt and positions[-1] != n - win_nt:
        positions.append(n - win_nt)

    local_dg_list: list[float] = []
    local_gc_list: list[float] = []
    local_u_list: list[float] = []
    local_unp_list: list[float] = []
    for pos in positions:
        chunk = seq[pos : pos + win_nt]
        clen = max(1, len(chunk))
        if rna_mod is not None:
            _, ch_dg = rna_mod.fold(chunk)
            local_dg_list.append(round(float(ch_dg), 3))
        else:
            local_dg_list.append(round(-0.42 * (chunk.count("G") + chunk.count("C")), 3))
        local_gc_list.append(round(100.0 * (chunk.count("G") + chunk.count("C")) / clen, 2))
        local_u_list.append(round(100.0 * chunk.count("U") / clen, 2))
        unp_slice = unpaired_per_nt[pos : pos + clen]
        local_unp_list.append(round(sum(unp_slice) / max(1, len(unp_slice)), 4))

    # 5. Optional EternaFold v1.3.1 cross-check
    eternafold_summary: dict[str, float] | None = None
    if include_eternafold:
        ef_found = discover_eternafold_binary()
        if ef_found is not None:
            ef_bin, ef_params = ef_found
            eternafold_summary = _run_eternafold_window_metrics(seq, c_start, ef_bin, ef_params)
            if eternafold_summary is not None:
                engines_used.append("EternaFold")

    # 6. Exact 0-based motif scan
    motifs = _scan_motifs(
        seq,
        five_prime_end=u5_end,
        cds_start=c_start,
        cds_end=c_end,
        three_prime_start=u3_start,
        three_prime_end=u3_end,
        modality=modality,
        seed_dir=eff_seed,
    )

    result: dict[str, Any] = {
        "length_nt": n,
        "modality": modality,
        "full_gc_pct": full_gc_pct,
        "full_u_pct": full_u_pct,
        "cai": cai_val,
        "mfe_kcal_mol": dg_total,
        "mfe_per_nt": round(dg_total / max(1, n), 4),
        "ensemble_diversity": ensemble_diversity,
        "aup_mean": aup_mean,
        "kozak_unpaired_prob": kozak_unpaired_prob,
        "initiation_window": {
            "start": init_lo,
            "end": init_hi,
            "aug_index_in_window": aug_in_win,
            "sequence": init_seq,
            "dot_bracket": init_dot_bracket,
            "arcs": arcs,
        },
        "sliding_tracks": {
            "window_nt": win_nt,
            "step_nt": step_nt,
            "positions": positions,
            "local_mfe": local_dg_list,
            "local_gc_pct": local_gc_list,
            "local_u_pct": local_u_list,
            "local_unpaired_prob": local_unp_list,
        },
        "motifs": motifs,
        "engines_used": engines_used,
        "regions": {
            "five_prime_end": u5_end,
            "cds_start": c_start,
            "cds_end": c_end,
            "three_prime_start": u3_start,
            "three_prime_end": u3_end,
        },
    }
    if eternafold_summary is not None:
        result["eternafold_metrics"] = eternafold_summary
    return result


__all__ = [
    "analyze_rna_structure",
    "discover_eternafold_binary",
]

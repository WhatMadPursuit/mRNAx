"""CPU-fast pure-sequence feature extractors for surrogate_v2 (zero GPL / zero ViennaRNA / zero PyTorch)."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable

import numpy as np

STOP_CODONS = {"TAA", "TAG", "TGA"}

HUMAN_CODON_FREQ: dict[str, float] = {
    "TTT": 17.6, "TTC": 20.3, "TTA": 7.7, "TTG": 12.9, "CTT": 13.2, "CTC": 19.6, "CTA": 7.2, "CTG": 39.6,
    "ATT": 16.0, "ATC": 20.8, "ATA": 7.5, "ATG": 22.0, "GTT": 11.0, "GTC": 14.5, "GTA": 7.1, "GTG": 28.1,
    "TCT": 15.2, "TCC": 17.7, "TCA": 12.2, "TCG": 4.4, "CCT": 17.5, "CCC": 19.8, "CCA": 16.9, "CCG": 6.9,
    "ACT": 13.1, "ACC": 18.9, "ACA": 15.1, "ACG": 6.1, "GCT": 18.4, "GCC": 27.7, "GCA": 15.8, "GCG": 7.4,
    "TAT": 12.2, "TAC": 15.3, "TAA": 1.0, "TAG": 0.8, "CAT": 10.9, "CAC": 15.1, "CAA": 12.3, "CAG": 34.2,
    "AAT": 17.0, "AAC": 19.1, "AAA": 24.4, "AAG": 31.9, "GAT": 21.8, "GAC": 25.1, "GAA": 29.0, "GAG": 39.6,
    "TGT": 10.6, "TGC": 12.6, "TGA": 1.6, "TGG": 13.2, "CGT": 4.5, "CGC": 10.4, "CGA": 6.2, "CGG": 11.4,
    "AGT": 12.1, "AGC": 19.5, "AGA": 12.2, "AGG": 12.0, "GGT": 10.8, "GGC": 22.2, "GGA": 16.5, "GGG": 16.5,
}
AA_CODONS: dict[str, list[str]] = {
    "F": ["TTT", "TTC"], "L": ["TTA", "TTG", "CTT", "CTC", "CTA", "CTG"], "I": ["ATT", "ATC", "ATA"],
    "M": ["ATG"], "V": ["GTT", "GTC", "GTA", "GTG"], "S": ["TCT", "TCC", "TCA", "TCG", "AGT", "AGC"],
    "P": ["CCT", "CCC", "CCA", "CCG"], "T": ["ACT", "ACC", "ACA", "ACG"], "A": ["GCT", "GCC", "GCA", "GCG"],
    "Y": ["TAT", "TAC"], "H": ["CAT", "CAC"], "Q": ["CAA", "CAG"], "N": ["AAT", "AAC"],
    "K": ["AAA", "AAG"], "D": ["GAT", "GAC"], "E": ["GAA", "GAG"], "C": ["TGT", "TGC"],
    "W": ["TGG"], "R": ["CGT", "CGC", "CGA", "CGG", "AGA", "AGG"], "G": ["GGT", "GGC", "GGA", "GGG"],
}
CODON_W: dict[str, float] = {
    c: HUMAN_CODON_FREQ[c] / max(HUMAN_CODON_FREQ[x] for x in cs)
    for cs in AA_CODONS.values()
    for c in cs
}
CODON_TO_AA: dict[str, str] = {c: aa for aa, cs in AA_CODONS.items() for c in cs}
for _st in STOP_CODONS:
    CODON_TO_AA[_st] = "_"

SENSE_CODONS_SORTED: list[str] = sorted(CODON_W.keys())
AAS_SORTED: list[str] = sorted(AA_CODONS.keys())
DINUCLEOTIDES: list[str] = [x + y for x in "ACGT" for y in "ACGT"]
TRINUCLEOTIDES: list[str] = [x + y + z for x in "ACGT" for y in "ACGT" for z in "ACGT"]
RC_MAP = str.maketrans("ACGT", "TGCA")


def clean_dna(seq: str) -> str:
    if not isinstance(seq, str):
        return ""
    return re.sub(r"[^ACGTUacgtu]", "", seq).upper().replace("U", "T")


def _revcomp_dna(s: str) -> str:
    return s.translate(RC_MAP)[::-1]


def _stem_complementarity_density(s: str, k: int = 5) -> float:
    """Fast pure-Python self-folding propensity proxy: fraction of k-mers whose revcomp appears >= 4 nt away."""
    L = len(s)
    if L < 2 * k + 4:
        return 0.0
    pos_map: dict[str, int] = {}
    for i in range(L - k + 1):
        km = s[i : i + k]
        if km not in pos_map:
            pos_map[km] = i
    hits = 0
    total = L - k + 1
    for i in range(total):
        rc = _revcomp_dna(s[i : i + k])
        j = pos_map.get(rc)
        if j is not None and abs(j - i) >= k + 3:
            hits += 1
    return hits / max(total, 1)


def _window_gc_stats(s: str, w: int = 10) -> tuple[float, float]:
    L = len(s)
    if L <= w:
        gc = (s.count("G") + s.count("C")) / max(L, 1)
        return gc, gc
    vals = [(s[i : i + w].count("G") + s[i : i + w].count("C")) / w for i in range(L - w + 1)]
    return max(vals), min(vals)


def _cai_from_codons(codons: list[str]) -> float:
    logs = [math.log(CODON_W[c]) for c in codons if c in CODON_W]
    return math.exp(sum(logs) / len(logs)) if logs else 0.5


def extract_utr5_v2(seq: str) -> dict[str, float]:
    s = clean_dna(seq)
    raw_len = len(s)
    L = max(raw_len, 1)
    atg_pos = [m.start() for m in re.finditer("ATG", s)]
    n_uaug = len(atg_pos)
    n_uaug_inframe = sum(1 for p in atg_pos if (raw_len - p) % 3 == 0)
    n_uaug_oof = n_uaug - n_uaug_inframe
    n_strong_uaug = 0
    n_uorf = 0
    n_uorf_inframe = 0
    n_uorf_oof = 0
    max_uorf_len = 0
    for p in atg_pos:
        if (p >= 3 and s[p - 3] in "AG") or (p + 3 < raw_len and s[p + 3] == "G"):
            n_strong_uaug += 1
        stop_q = None
        for q in range(p + 3, raw_len - 2, 3):
            if s[q : q + 3] in STOP_CODONS:
                stop_q = q
                break
        if stop_q is not None:
            n_uorf += 1
            max_uorf_len = max(max_uorf_len, stop_q - p + 3)
            if (raw_len - p) % 3 == 0:
                n_uorf_inframe += 1
            else:
                n_uorf_oof += 1

    # Unterminated uAUGs (uORFs extending past canonical AUG into CDS)
    n_unterminated_uaug = n_uaug - n_uorf
    n_unterminated_oof = sum(
        1 for p in atg_pos if (raw_len - p) % 3 != 0 and not any(s[q : q + 3] in STOP_CODONS for q in range(p + 3, raw_len - 2, 3))
    )

    d_cnt = Counter(s[i : i + 2] for i in range(max(raw_len - 1, 0)))
    t_cnt = Counter(s[i : i + 3] for i in range(max(raw_len - 2, 0)))
    d_den = max(raw_len - 1, 1)
    t_den = max(raw_len - 2, 1)
    max_wgc, min_wgc = _window_gc_stats(s, w=10)

    # Positional 5 bins (0..4)
    feats: dict[str, float] = {
        "len": float(raw_len),
        "log_len": math.log1p(raw_len),
        "a_frac": s.count("A") / L,
        "c_frac": s.count("C") / L,
        "g_frac": s.count("G") / L,
        "u_frac": s.count("T") / L,
        "gc": (s.count("G") + s.count("C")) / L,
        "gc_head15": (s[:15].count("G") + s[:15].count("C")) / max(len(s[:15]), 1),
        "gc_tail15": (s[-15:].count("G") + s[-15:].count("C")) / max(len(s[-15:]), 1),
        "u_tail15": s[-15:].count("T") / max(len(s[-15:]), 1),
        "a_tail15": s[-15:].count("A") / max(len(s[-15:]), 1),
        "c_tail15": s[-15:].count("C") / max(len(s[-15:]), 1),
        "g_tail15": s[-15:].count("G") / max(len(s[-15:]), 1),
        "n_uaug": float(n_uaug),
        "has_uaug": float(n_uaug > 0),
        "n_uaug_inframe": float(n_uaug_inframe),
        "n_uaug_oof": float(n_uaug_oof),
        "n_strong_uaug": float(n_strong_uaug),
        "n_uorf": float(n_uorf),
        "n_uorf_inframe": float(n_uorf_inframe),
        "n_uorf_oof": float(n_uorf_oof),
        "n_unterminated_uaug": float(n_unterminated_uaug),
        "n_unterminated_oof": float(n_unterminated_oof),
        "max_uorf_len": float(max_uorf_len),
        "first_uaug_pos_norm": (atg_pos[0] / L) if atg_pos else 1.0,
        "last_uaug_dist": float(raw_len - atg_pos[-1]) if atg_pos else float(raw_len),
        "kozak_m3_purine": float(raw_len >= 3 and s[-3] in "AG"),
        "kozak_m3_a": float(raw_len >= 3 and s[-3] == "A"),
        "kozak_m3_g": float(raw_len >= 3 and s[-3] == "G"),
        "kozak_m1_c": float(raw_len >= 1 and s[-1] == "C"),
        "kozak_m2_c": float(raw_len >= 2 and s[-2] == "C"),
        "kozak_m4_c": float(raw_len >= 4 and s[-4] == "C"),
        "kozak_m5_c": float(raw_len >= 5 and s[-5] == "C"),
        "kozak_m6_g": float(raw_len >= 6 and s[-6] == "G"),
        "kozak_tail6_c": sum(1 for ch in s[-6:] if ch == "C") / min(L, 6),
        "kozak_gccacc_match": sum(1 for a, b in zip(s[-6:].rjust(6, "N"), "GCCACC") if a == b) / 6.0,
        "has_g4": float(bool(re.search(r"G{3,}.{1,7}G{3,}.{1,7}G{3,}.{1,7}G{3,}", s))),
        "has_g2_quad": float(bool(re.search(r"G{2,}.{1,5}G{2,}.{1,5}G{2,}.{1,5}G{2,}", s))),
        "max_poly_g": float(max((len(m.group(0)) for m in re.finditer(r"G+", s)), default=0)),
        "max_poly_u": float(max((len(m.group(0)) for m in re.finditer(r"T+", s)), default=0)),
        "max_poly_c": float(max((len(m.group(0)) for m in re.finditer(r"C+", s)), default=0)),
        "max_poly_a": float(max((len(m.group(0)) for m in re.finditer(r"A+", s)), default=0)),
        "n_poly_u4": float(len(re.findall(r"T{4,}", s))),
        "n_poly_g3": float(len(re.findall(r"G{3,}", s))),
        "n_poly_c4": float(len(re.findall(r"C{4,}", s))),
        "cpg": d_cnt["CG"] / d_den,
        "upa": d_cnt["TA"] / d_den,
        "max_gc_w10": max_wgc,
        "min_gc_w10": min_wgc,
        "stem_comp_5mer": _stem_complementarity_density(s, k=5),
        "stem_comp_4mer": _stem_complementarity_density(s, k=4),
    }

    for b_idx in range(5):
        lo = (b_idx * len(s)) // 5
        hi = ((b_idx + 1) * len(s)) // 5
        sub = s[lo:hi]
        sl = max(len(sub), 1)
        feats[f"bin{b_idx}_a"] = sub.count("A") / sl
        feats[f"bin{b_idx}_c"] = sub.count("C") / sl
        feats[f"bin{b_idx}_g"] = sub.count("G") / sl
        feats[f"bin{b_idx}_u"] = sub.count("T") / sl
        feats[f"bin{b_idx}_gc"] = (sub.count("G") + sub.count("C")) / sl
        feats[f"bin{b_idx}_atg"] = float("ATG" in sub)

    for d in DINUCLEOTIDES:
        feats[f"di_{d}"] = d_cnt[d] / d_den
    for t in TRINUCLEOTIDES:
        feats[f"tri_{t}"] = t_cnt[t] / t_den
    return feats


def extract_cds_v2(seq: str) -> dict[str, float]:
    s = clean_dna(seq)
    L = max(len(s), 1)
    n_cod = max(L // 3, 1)
    cod_list = [s[3 * i : 3 * i + 3] for i in range(L // 3)]
    pos1 = s[0 : (L // 3) * 3 : 3]
    pos2 = s[1 : (L // 3) * 3 : 3]
    pos3 = s[2 : (L // 3) * 3 : 3]
    cai = _cai_from_codons(cod_list)
    denom = max(L - 1, 1)
    cods = Counter(cod_list)
    aa_cnt = Counter(CODON_TO_AA.get(c, "X") for c in cod_list)

    # Regional ramp/body/tail (first 30 codons, mid, last 30 codons)
    ramp_c = cod_list[:30]
    mid_c = cod_list[30:-30] if len(cod_list) > 60 else cod_list
    tail_c = cod_list[-30:]
    ramp_s = "".join(ramp_c)
    mid_s = "".join(mid_c)
    tail_s = "".join(tail_c)

    # Codon-boundary dinucleotides (pos3 of codon i -> pos1 of codon i+1)
    junc_cnt = Counter(cod_list[i][2] + cod_list[i + 1][0] for i in range(max(len(cod_list) - 1, 0)) if len(cod_list[i]) == 3 and len(cod_list[i + 1]) == 3)
    junc_den = max(len(cod_list) - 1, 1)

    # Rare & optimal codon streak lengths
    max_rare_run = 0
    cur_rare = 0
    max_opt_run = 0
    cur_opt = 0
    for c in cod_list:
        w = CODON_W.get(c, 0.5)
        if w < 0.35:
            cur_rare += 1
            max_rare_run = max(max_rare_run, cur_rare)
        else:
            cur_rare = 0
        if w > 0.85:
            cur_opt += 1
            max_opt_run = max(max_opt_run, cur_opt)
        else:
            cur_opt = 0

    last_cod = cod_list[-1] if cod_list else ""
    d_cnt = Counter(s[i : i + 2] for i in range(max(L - 1, 0)))

    feats: dict[str, float] = {
        "len_nt": float(L),
        "log_len_nt": math.log1p(L),
        "a_frac": s.count("A") / L,
        "c_frac": s.count("C") / L,
        "g_frac": s.count("G") / L,
        "u_frac": s.count("T") / L,
        "gc": (s.count("G") + s.count("C")) / L,
        "gc1": (pos1.count("G") + pos1.count("C")) / max(len(pos1), 1),
        "gc2": (pos2.count("G") + pos2.count("C")) / max(len(pos2), 1),
        "gc3": (pos3.count("G") + pos3.count("C")) / max(len(pos3), 1),
        "u3": pos3.count("T") / max(len(pos3), 1),
        "a3": pos3.count("A") / max(len(pos3), 1),
        "c3": pos3.count("C") / max(len(pos3), 1),
        "g3": pos3.count("G") / max(len(pos3), 1),
        "cai": float(cai),
        "gc_ramp30": (s[:30].count("G") + s[:30].count("C")) / max(len(s[:30]), 1),
        "u_ramp30": s[:30].count("T") / max(len(s[:30]), 1),
        "gc_ramp90": (ramp_s.count("G") + ramp_s.count("C")) / max(len(ramp_s), 1),
        "gc3_ramp90": sum(1 for c in ramp_c if len(c) == 3 and c[2] in "GC") / max(len(ramp_c), 1),
        "cai_ramp90": _cai_from_codons(ramp_c),
        "gc3_mid": sum(1 for c in mid_c if len(c) == 3 and c[2] in "GC") / max(len(mid_c), 1),
        "cai_mid": _cai_from_codons(mid_c),
        "gc3_tail90": sum(1 for c in tail_c if len(c) == 3 and c[2] in "GC") / max(len(tail_c), 1),
        "u_tail90": tail_s.count("T") / max(len(tail_s), 1),
        "cpg": d_cnt["CG"] / denom,
        "upa": d_cnt["TA"] / denom,
        "n_polyu4": float(len(re.findall(r"T{4,}", s))),
        "n_uuuc": float(len(re.findall(r"TTTC", s))),
        "n_polya4": float(len(re.findall(r"A{4,}", s))),
        "n_polyc4": float(len(re.findall(r"C{4,}", s))),
        "n_polyg4": float(len(re.findall(r"G{4,}", s))),
        "max_rare_codon_run": float(max_rare_run),
        "max_opt_codon_run": float(max_opt_run),
        "stop_is_taa": float(last_cod == "TAA"),
        "stop_is_tag": float(last_cod == "TAG"),
        "stop_is_tga": float(last_cod == "TGA"),
    }
    for d in DINUCLEOTIDES:
        feats[f"di_{d}"] = d_cnt[d] / denom
        feats[f"junc_{d}"] = junc_cnt[d] / junc_den
    for aa in AAS_SORTED:
        feats[f"aa_{aa}"] = aa_cnt[aa] / n_cod
    for cod in SENSE_CODONS_SORTED:
        feats[f"cod_{cod}"] = cods[cod] / n_cod
        aa = CODON_TO_AA[cod]
        n_syn = len(AA_CODONS[aa])
        aa_tot = aa_cnt[aa]
        feats[f"rscu_{cod}"] = (cods[cod] * n_syn / aa_tot) if aa_tot > 0 else 1.0
    return feats


def extract_utr3_v2(seq: str) -> dict[str, float]:
    s = clean_dna(seq)
    L = max(len(s), 1)
    d_cnt = Counter(s[i : i + 2] for i in range(max(L - 1, 0)))
    t_cnt = Counter(s[i : i + 3] for i in range(max(L - 2, 0)))
    d_den = max(L - 1, 1)
    t_den = max(L - 2, 1)
    prox = s[:200]
    dist = s[-200:]
    feats: dict[str, float] = {
        "len": float(L),
        "log_len": math.log1p(L),
        "a_frac": s.count("A") / L,
        "c_frac": s.count("C") / L,
        "g_frac": s.count("G") / L,
        "u_frac": s.count("T") / L,
        "gc": (s.count("G") + s.count("C")) / L,
        "au_frac": (s.count("A") + s.count("T")) / L,
        "are_auuua_cnt": float(len(re.findall("ATTTA", s))),
        "are_auuua_density": len(re.findall("ATTTA", s)) / (L / 1000.0),
        "are_nonamer_cnt": float(len(re.findall("TTATTTATT", s))),
        "are_uauuuau_cnt": float(len(re.findall("TATTTAT", s))),
        "pumilio_cnt": float(len(re.findall(r"TGTA[ACGT]ATA", s))),
        "cde_roquin_cnt": float(len(re.findall("TTGTC", s))),
        "pas_aauaaa_cnt": float(len(re.findall("AATAAA", s))),
        "poly_u4_cnt": float(len(re.findall(r"T{4,}", s))),
        "poly_c4_cnt": float(len(re.findall(r"C{4,}", s))),
        "poly_g4_cnt": float(len(re.findall(r"G{4,}", s))),
        "poly_a4_cnt": float(len(re.findall(r"A{4,}", s))),
        "c_rich_ccuc_cnt": float(len(re.findall("CCTC|CTCC|CCCT", s))),
        "max_poly_u": float(max((len(m.group(0)) for m in re.finditer("T+", s)), default=0)),
        "max_poly_g": float(max((len(m.group(0)) for m in re.finditer("G+", s)), default=0)),
        "cpg": d_cnt["CG"] / d_den,
        "upa": d_cnt["TA"] / d_den,
        "prox200_gc": (prox.count("G") + prox.count("C")) / max(len(prox), 1),
        "dist200_gc": (dist.count("G") + dist.count("C")) / max(len(dist), 1),
        "prox200_are": float(len(re.findall("ATTTA", prox))),
        "dist200_are": float(len(re.findall("ATTTA", dist))),
    }
    for d in DINUCLEOTIDES:
        feats[f"di_{d}"] = d_cnt[d] / d_den
    for t in TRINUCLEOTIDES:
        feats[f"tri_{t}"] = t_cnt[t] / t_den
    return feats


def extract_full_tx_v2(utr5_seq: str, cds_seq: str, utr3_seq: str) -> dict[str, float]:
    u5 = clean_dna(utr5_seq)
    cds = clean_dna(cds_seq)
    u3 = clean_dna(utr3_seq)
    tot = max(len(u5) + len(cds) + len(u3), 1)

    feats: dict[str, float] = {
        "total_len": float(tot),
        "log_total_len": math.log1p(tot),
        "utr5_frac": len(u5) / tot,
        "cds_frac": len(cds) / tot,
        "utr3_frac": len(u3) / tot,
        "utr3_to_cds_ratio": len(u3) / max(len(cds), 1),
    }

    u5f = extract_utr5_v2(u5)
    for k, v in u5f.items():
        if not k.startswith("tri_") and not k.startswith("bin"):
            feats[f"u5_{k}"] = v

    cdsf = extract_cds_v2(cds)
    for k, v in cdsf.items():
        feats[f"cds_{k}"] = v

    u3f = extract_utr3_v2(u3)
    for k, v in u3f.items():
        feats[f"u3_{k}"] = v

    # Cross-junction 5'UTR-CDS & CDS-3'UTR features
    j5 = (u5[-30:] + cds[:30]) if (u5 or cds) else ""
    j3 = (cds[-30:] + u3[:30]) if (cds or u3) else ""
    feats["j5_gc"] = (j5.count("G") + j5.count("C")) / max(len(j5), 1)
    feats["j3_gc"] = (j3.count("G") + j3.count("C")) / max(len(j3), 1)
    feats["j5_stem5"] = _stem_complementarity_density(j5, k=5) if len(j5) >= 20 else 0.0
    feats["j3_stem5"] = _stem_complementarity_density(j3, k=5) if len(j3) >= 20 else 0.0
    return feats


def extract_circrna_ires_v2(seq: str) -> dict[str, float]:
    s = clean_dna(seq)
    L = max(len(s), 1)
    d_cnt = Counter(s[i : i + 2] for i in range(max(L - 1, 0)))
    t_cnt = Counter(s[i : i + 3] for i in range(max(L - 2, 0)))
    d_den = max(L - 1, 1)
    t_den = max(L - 2, 1)
    h60 = s[:60]
    t60 = s[-60:]
    max_wgc, min_wgc = _window_gc_stats(s, w=20)
    feats: dict[str, float] = {
        "len": float(L),
        "log_len": math.log1p(L),
        "a_frac": s.count("A") / L,
        "c_frac": s.count("C") / L,
        "g_frac": s.count("G") / L,
        "u_frac": s.count("T") / L,
        "gc": (s.count("G") + s.count("C")) / L,
        "gc_head60": (h60.count("G") + h60.count("C")) / max(len(h60), 1),
        "gc_tail60": (t60.count("G") + t60.count("C")) / max(len(t60), 1),
        "u_tail60": t60.count("T") / max(len(t60), 1),
        "max_gc_w20": max_wgc,
        "min_gc_w20": min_wgc,
        "stem_comp_5mer": _stem_complementarity_density(s, k=5),
        "stem_comp_6mer": _stem_complementarity_density(s, k=6),
        "n_atg": float(len(re.findall("ATG", s))),
        "n_atg_density": len(re.findall("ATG", s)) / (L / 100.0),
        "n_polyu4": float(len(re.findall(r"T{4,}", s))),
        "n_polya4": float(len(re.findall(r"A{4,}", s))),
        "n_polyc4": float(len(re.findall(r"C{4,}", s))),
        "n_polyg4": float(len(re.findall(r"G{4,}", s))),
        "n_are": float(len(re.findall("ATTTA", s))),
        "n_ccuc": float(len(re.findall("CCTC|CTCC|CCCT", s))),
        "has_g4": float(bool(re.search(r"G{3,}.{1,7}G{3,}.{1,7}G{3,}.{1,7}G{3,}", s))),
        "cpg": d_cnt["CG"] / d_den,
        "upa": d_cnt["TA"] / d_den,
        # IRES / eIF4G-like / pyrimidine-tract (YN-rich) motifs
        "y_tract_len_max": float(max((len(m.group(0)) for m in re.finditer(r"[CT]+", s)), default=0)),
        "u_rich_pct": (s.count("C") + s.count("T")) / L,
    }
    for d in DINUCLEOTIDES:
        feats[f"di_{d}"] = d_cnt[d] / d_den
    for t in TRINUCLEOTIDES:
        feats[f"tri_{t}"] = t_cnt[t] / t_den
    return feats


def extract_circrna_rbp_v2(orf_dna: str) -> dict[str, float]:
    s = clean_dna(orf_dna)
    L = max(len(s), 1)
    n_cod = L // 3
    aas = []
    for i in range(n_cod):
        aa = CODON_TO_AA.get(s[3 * i : 3 * i + 3], "X")
        if aa == "_":
            break
        if aa != "X":
            aas.append(aa)
    prot = "".join(aas)
    PL = max(len(prot), 1)
    aa_cnt = Counter(prot)
    pos_ch = aa_cnt["K"] + aa_cnt["R"]
    neg_ch = aa_cnt["D"] + aa_cnt["E"]
    arom = aa_cnt["F"] + aa_cnt["W"] + aa_cnt["Y"]
    hydr = aa_cnt["I"] + aa_cnt["L"] + aa_cnt["V"] + aa_cnt["M"] + aa_cnt["F"]
    diso = aa_cnt["P"] + aa_cnt["E"] + aa_cnt["S"] + aa_cnt["Q"] + aa_cnt["K"] + aa_cnt["A"] + aa_cnt["G"]
    feats: dict[str, float] = {
        "orf_len_nt": float(L),
        "prot_len": float(PL),
        "log_prot_len": math.log1p(PL),
        "orf_gc": (s.count("G") + s.count("C")) / L,
        "pos_charge_frac": pos_ch / PL,
        "neg_charge_frac": neg_ch / PL,
        "net_charge_per_res": (pos_ch - neg_ch) / PL,
        "aromatic_frac": arom / PL,
        "hydrophobic_frac": hydr / PL,
        "disorder_frac": diso / PL,
        "rg_motif_density": len(re.findall(r"RG", prot)) / (PL / 100.0),
        "rgg_motif_density": len(re.findall(r"RGG", prot)) / (PL / 100.0),
        "rs_motif_density": len(re.findall(r"RS|SR", prot)) / (PL / 100.0),
        "yg_motif_density": len(re.findall(r"[YF]G|G[YF]", prot)) / (PL / 100.0),
        "qn_rich_frac": (aa_cnt["Q"] + aa_cnt["N"]) / PL,
    }
    for aa in AAS_SORTED:
        feats[f"prot_aa_{aa}"] = aa_cnt[aa] / PL
    return feats


def extract_rmdb_seq_v2(seq: str) -> dict[str, float]:
    s = clean_dna(seq)
    L = max(len(s), 1)
    d_cnt = Counter(s[i : i + 2] for i in range(max(L - 1, 0)))
    t_cnt = Counter(s[i : i + 3] for i in range(max(L - 2, 0)))
    d_den = max(L - 1, 1)
    t_den = max(L - 2, 1)
    max_wgc, min_wgc = _window_gc_stats(s, w=12)
    feats: dict[str, float] = {
        "len": float(L),
        "a_frac": s.count("A") / L,
        "c_frac": s.count("C") / L,
        "g_frac": s.count("G") / L,
        "u_frac": s.count("T") / L,
        "gc": (s.count("G") + s.count("C")) / L,
        "au": (s.count("A") + s.count("T")) / L,
        "max_gc_w12": max_wgc,
        "min_gc_w12": min_wgc,
        "stem_comp_4mer": _stem_complementarity_density(s, k=4),
        "stem_comp_5mer": _stem_complementarity_density(s, k=5),
        "stem_comp_6mer": _stem_complementarity_density(s, k=6),
        "n_tetraloop_gnra": float(len(re.findall(r"G[ACGT][AG]A", s))),
        "n_tetraloop_uncg": float(len(re.findall(r"T[ACGT]CG", s))),
        "max_poly_a": float(max((len(m.group(0)) for m in re.finditer("A+", s)), default=0)),
        "max_poly_u": float(max((len(m.group(0)) for m in re.finditer("T+", s)), default=0)),
        "max_poly_g": float(max((len(m.group(0)) for m in re.finditer("G+", s)), default=0)),
        "max_poly_c": float(max((len(m.group(0)) for m in re.finditer("C+", s)), default=0)),
    }
    for d in DINUCLEOTIDES:
        feats[f"di_{d}"] = d_cnt[d] / d_den
    for t in TRINUCLEOTIDES:
        feats[f"tri_{t}"] = t_cnt[t] / t_den
    return feats


def seq_to_kmer_tokens(seq: str, ks: tuple[int, ...] = (4, 5)) -> str:
    s = clean_dna(seq)
    toks: list[str] = []
    for k in ks:
        if len(s) >= k:
            toks.extend(f"k{k}_{s[i : i + k]}" for i in range(len(s) - k + 1))
    return " ".join(toks)


def cds_to_bicodon_tokens(cds: str) -> str:
    s = clean_dna(cds)
    nc = len(s) // 3
    if nc < 2:
        return ""
    cods = [s[3 * i : 3 * i + 3] for i in range(nc)]
    return " ".join(f"{cods[i]}_{cods[i + 1]}" for i in range(nc - 1))


def dict_list_to_matrix(dicts: Iterable[dict[str, float]], cols: list[str]) -> np.ndarray:
    if not cols:
        raise ValueError("Feature column specification 'cols' must be non-empty.")
    rows: list[list[float]] = []
    for row_idx, d in enumerate(dicts):
        missing = [c for c in cols if c not in d]
        if missing:
            raise KeyError(
                f"Feature drift in row {row_idx}: missing {len(missing)} declared feature(s): {missing[:5]}"
            )
        vals: list[float] = []
        for c in cols:
            raw_val = d[c]
            if isinstance(raw_val, bool) or not isinstance(raw_val, (int, float, np.floating, np.integer)):
                raise ValueError(f"Non-numeric feature '{c}'={raw_val!r} in row {row_idx}")
            v = float(raw_val)
            v32 = np.float32(v)
            if not math.isfinite(v) or not bool(np.isfinite(v32)):
                raise ValueError(f"Out-of-range or non-finite feature '{c}'={raw_val!r} in row {row_idx}")
            vals.append(float(v32))
        rows.append(vals)
    if not rows:
        return np.empty((0, len(cols)), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)

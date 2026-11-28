"""M1.5 Axis-Routed Commercial-Clean Surrogate Jury (`surrogate_v1_clean`) & Native MIT `structRFM` Loader.

Clean-Room Invariants:
- 100% CC-BY 4.0 / Apache-2.0 / MIT / public-GEO provenance (`LICENSE_RECORD_m15_commercial_clean_v1.json`).
- ZERO ViennaRNA imports, features, or distilled labels in `commercial_clean` models.
- ZERO `BPfold` / GPL dependencies when loading MIT `structRFM` (`checkpoints/structRFM_checkpoint` via native HuggingFace).
- Enforces `USE_SPACED=False` (unspaced `"AUGC"`, never `"A U G C"` `[UNK]` trap), `win=510 / stride=384` long-transcript
  stitching, and strict per-chemistry SHAPE-MaP probe routing (`U` vs `moU` vs `mpU`/`m1psi`).
"""

from __future__ import annotations

import functools
import hashlib
import io
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Protocol
from weakref import WeakSet

import numpy as np

from mrnax.engine.checkpoint_identity import verify_checkpoint, verify_release_directory_artifacts
from mrnax.schemas.construct_bundle import ConstructBundle, RNAMoleculeSpec
from mrnax.schemas.evaluation_report import AdvisoryNote, MetricResult

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
DINUCLEOTIDES: list[str] = [x + y for x in "ACGT" for y in "ACGT"]
TRINUCLEOTIDES_3UTR: list[str] = [
    "ATT", "TTA", "TAT", "TTT", "AAA", "CCC", "GGG", "GCG",
    "CGC", "TGT", "GTA", "AAT", "ATA", "CAA", "TTG",
]
SENSE_CODONS_SORTED: list[str] = sorted(CODON_W.keys())


def clean_dna(seq: str) -> str:
    return re.sub(r"[^ACGTUacgtu]", "", str(seq)).upper().replace("U", "T")


def extract_5utr_features(seq: str) -> dict[str, float]:
    s = clean_dna(seq)
    L = max(len(s), 1)
    atg_pos = [m.start() for m in re.finditer("ATG", s)]
    n_uaug = len(atg_pos)
    n_uaug_inframe = sum(1 for p in atg_pos if (L - p) % 3 == 0)
    n_uaug_oof = n_uaug - n_uaug_inframe
    n_strong_uaug = sum(
        1 for p in atg_pos if (p >= 3 and s[p - 3] in "AG") or (p + 3 < L and s[p + 3] == "G")
    )
    n_uorf = sum(
        1 for p in atg_pos if any(s[q : q + 3] in STOP_CODONS for q in range(p + 3, L - 2, 3))
    )
    d_cnt = Counter(s[i : i + 2] for i in range(max(L - 1, 0)))
    denom = max(L - 1, 1)
    feats: dict[str, float] = {
        "len": float(L),
        "a_frac": s.count("A") / L,
        "c_frac": s.count("C") / L,
        "g_frac": s.count("G") / L,
        "u_frac": s.count("T") / L,
        "gc": (s.count("G") + s.count("C")) / L,
        "gc_head15": (s[:15].count("G") + s[:15].count("C")) / max(len(s[:15]), 1),
        "gc_tail15": (s[-15:].count("G") + s[-15:].count("C")) / max(len(s[-15:]), 1),
        "n_uaug": float(n_uaug),
        "has_uaug": float(n_uaug > 0),
        "n_uaug_inframe": float(n_uaug_inframe),
        "n_uaug_oof": float(n_uaug_oof),
        "n_strong_uaug": float(n_strong_uaug),
        "n_uorf": float(n_uorf),
        "kozak_m3_purine": float(L >= 3 and s[-3] in "AG"),
        "kozak_m3_a": float(L >= 3 and s[-3] == "A"),
        "kozak_tail6_c": sum(1 for ch in s[-6:] if ch == "C") / min(L, 6),
        "has_g4": float(bool(re.search(r"G{3,}.{1,7}G{3,}.{1,7}G{3,}.{1,7}G{3,}", s))),
        "max_poly_g": float(max((len(m.group(0)) for m in re.finditer(r"G+", s)), default=0)),
        "max_poly_u": float(max((len(m.group(0)) for m in re.finditer(r"T+", s)), default=0)),
    }
    for d in DINUCLEOTIDES:
        feats[f"di_{d}"] = d_cnt[d] / denom
    return feats


def extract_cds_features(seq: str) -> dict[str, float]:
    s = clean_dna(seq)
    L = max(len(s), 1)
    n_cod = L // 3
    pos1 = s[0 : n_cod * 3 : 3]
    pos2 = s[1 : n_cod * 3 : 3]
    pos3 = s[2 : n_cod * 3 : 3]
    logs = [
        math.log(CODON_W[s[3 * i : 3 * i + 3]])
        for i in range(n_cod)
        if s[3 * i : 3 * i + 3] in CODON_W
    ]
    cai = math.exp(sum(logs) / len(logs)) if logs else 0.5
    denom = max(L - 1, 1)
    cods = Counter(s[3 * i : 3 * i + 3] for i in range(n_cod))
    feats: dict[str, float] = {
        "len_nt": float(L),
        "a_frac": s.count("A") / L,
        "c_frac": s.count("C") / L,
        "g_frac": s.count("G") / L,
        "u_frac": s.count("T") / L,
        "gc": (s.count("G") + s.count("C")) / L,
        "gc1": (pos1.count("G") + pos1.count("C")) / max(len(pos1), 1),
        "gc2": (pos2.count("G") + pos2.count("C")) / max(len(pos2), 1),
        "gc3": (pos3.count("G") + pos3.count("C")) / max(len(pos3), 1),
        "u3": pos3.count("T") / max(len(pos3), 1),
        "cai": float(cai),
        "gc_ramp30": (s[:30].count("G") + s[:30].count("C")) / max(len(s[:30]), 1),
        "u_ramp30": s[:30].count("T") / max(len(s[:30]), 1),
        "cpg": sum(1 for i in range(L - 1) if s[i : i + 2] == "CG") / denom,
        "upa": sum(1 for i in range(L - 1) if s[i : i + 2] == "TA") / denom,
        "n_polyu4": float(len(re.findall(r"T{4,}", s))),
    }
    for cod in SENSE_CODONS_SORTED:
        feats[f"cod_{cod}"] = cods[cod] / max(n_cod, 1)
    return feats


@functools.lru_cache(maxsize=2048)
def _halflife_u5_subfeats(utr5_seq: str) -> tuple[float, float, float]:
    u5 = clean_dna(utr5_seq)
    u5_len = max(len(u5), 1)
    return (
        float(len(u5)),
        (u5.count("G") + u5.count("C")) / u5_len,
        float(len(re.findall("ATG", u5))),
    )


@functools.lru_cache(maxsize=2048)
def _halflife_cds_subfeats(cds_seq: str) -> tuple[float, float, float, float, float, float, tuple[tuple[str, float], ...]]:
    cds = clean_dna(cds_seq)
    c_len = max(len(cds), 1)
    nc = c_len // 3
    pos3 = cds[2 : nc * 3 : 3]
    logs = [
        math.log(CODON_W[cds[3 * i : 3 * i + 3]])
        for i in range(nc)
        if cds[3 * i : 3 * i + 3] in CODON_W
    ]
    cai = math.exp(sum(logs) / len(logs)) if logs else 0.5
    cods = Counter(cds[3 * i : 3 * i + 3] for i in range(nc))
    denom_nc = max(nc, 1)
    cod_pairs = tuple((f"cod_{cod}", cods[cod] / denom_nc) for cod in SENSE_CODONS_SORTED)
    return (
        float(len(cds)),
        math.log1p(len(cds)),
        (cds.count("G") + cds.count("C")) / c_len,
        (pos3.count("G") + pos3.count("C")) / max(len(pos3), 1),
        cds.count("T") / c_len,
        float(cai),
        cod_pairs,
    )


@functools.lru_cache(maxsize=2048)
def _halflife_u3_subfeats(utr3_seq: str) -> tuple[float, float, float, float, float, float, float, float, float]:
    u3 = clean_dna(utr3_seq)
    u3_len = max(len(u3), 1)
    are_cnt = len(re.findall("ATTTA", u3))
    return (
        float(len(u3)),
        math.log1p(len(u3)),
        (u3.count("G") + u3.count("C")) / u3_len,
        u3.count("T") / u3_len,
        float(are_cnt),
        are_cnt / (u3_len / 1000.0),
        float(len(re.findall(r"TGTA[ACGT]ATA", u3))),
        float(len(re.findall("AATAAA", u3))),
        sum(1 for i in range(max(len(u3) - 1, 0)) if u3[i : i + 2] == "CG") / u3_len,
    )


def extract_full_tx_halflife_features(utr5_seq: str, cds_seq: str, utr3_seq: str) -> dict[str, float]:
    utr5_len, u5_gc, u5_uaug = _halflife_u5_subfeats(utr5_seq)
    cds_len, log_cds_len, cds_gc, cds_gc3, cds_u, cds_cai, cod_pairs = _halflife_cds_subfeats(cds_seq)
    (
        utr3_len,
        log_utr3_len,
        u3_gc,
        u3_u,
        u3_are_cnt,
        u3_are_density,
        u3_pumilio_cnt,
        u3_pas_cnt,
        u3_cpg,
    ) = _halflife_u3_subfeats(utr3_seq)
    feats: dict[str, float] = {
        "cds_len": cds_len,
        "utr5_len": utr5_len,
        "utr3_len": utr3_len,
        "log_cds_len": log_cds_len,
        "log_utr3_len": log_utr3_len,
        "cds_gc": cds_gc,
        "cds_gc3": cds_gc3,
        "cds_u": cds_u,
        "cds_cai": cds_cai,
        "u5_gc": u5_gc,
        "u5_uaug": u5_uaug,
        "u3_gc": u3_gc,
        "u3_u": u3_u,
        "u3_are_cnt": u3_are_cnt,
        "u3_are_density": u3_are_density,
        "u3_pumilio_cnt": u3_pumilio_cnt,
        "u3_pas_cnt": u3_pas_cnt,
        "u3_cpg": u3_cpg,
    }
    feats.update(cod_pairs)
    return feats


def extract_3utr_features(seq: str) -> dict[str, float]:
    s = clean_dna(seq)
    L = max(len(s), 1)
    d_cnt = Counter(s[i : i + 2] for i in range(max(L - 1, 0)))
    t_cnt = Counter(s[i : i + 3] for i in range(max(L - 2, 0)))
    d_den = max(L - 1, 1)
    t_den = max(L - 2, 1)
    feats: dict[str, float] = {
        "len": float(L),
        "a_frac": s.count("A") / L,
        "c_frac": s.count("C") / L,
        "g_frac": s.count("G") / L,
        "u_frac": s.count("T") / L,
        "gc": (s.count("G") + s.count("C")) / L,
        "au_frac": (s.count("A") + s.count("T")) / L,
        "are_auuua_cnt": float(len(re.findall("ATTTA", s))),
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
    }
    for d in DINUCLEOTIDES:
        feats[f"di_{d}"] = d_cnt[d] / d_den
    for t in TRINUCLEOTIDES_3UTR:
        feats[f"tri_{t}"] = t_cnt[t] / t_den
    return feats


_SUPPORTED_PROVENANCE_SCHEMAS = frozenset(
    {
        "mrnax.surrogate_training_provenance.v2.0",
        "mrnax.surrogate_training_provenance.v1.2",
    }
)


_APPROVED_ARTIFACT_LICENSE_TOKENS = (
    "mit",
    "apache-2.0",
    "bsd",
    "cc-by-4.0",
    "cc by 4.0",
    "cc0",
    "public domain",
)
_FORBIDDEN_ARTIFACT_LICENSE_RE = re.compile(
    r"(?:non[-\s]?commercial|cc[-\s]?by[-\s]?nc|cc[-\s]?by[-\s]?sa|sharealike|\b(?:a?gpl|lgpl)\b|research[-\s_]?only|proprietary|pending|quarantine)",
    re.IGNORECASE,
)


def _assert_permissive_artifact_license(lic_raw: Any, *, filename: str) -> None:
    lic_s = str(lic_raw or "").strip()
    if not lic_s:
        raise PermissionError(
            f"Refusing to deserialize '{filename}' in commercial_clean loader: missing artifact license declaration"
        )
    low = lic_s.lower()
    if _FORBIDDEN_ARTIFACT_LICENSE_RE.search(low) or not any(tok in low for tok in _APPROVED_ARTIFACT_LICENSE_TOKENS):
        raise PermissionError(
            f"Refusing to deserialize '{filename}' in commercial_clean loader: prohibited or unrecognized artifact license '{lic_s}'"
        )


def extract_circrna_ires_interval(mol: RNAMoleculeSpec) -> str:
    """Resolve the canonical circRNA viral IRES slice across compound intervals & validate sense strand."""
    u5_seq = mol.get_segment_sequence("five_prime_utr")
    ires_ann = next(
        (a for a in mol.annotations if a.feature_id.startswith("viral_ires_") and a.intervals),
        None,
    )
    if ires_ann is not None:
        if ires_ann.strand != 1:
            raise ValueError(
                f"circRNA annotation '{ires_ann.feature_id}' must lie on positive (+1) sense strand, got {ires_ann.strand}"
            )
        seq_len = len(mol.sequence)
        pieces: list[str] = []
        prev_hi = 0
        for lo, hi in ires_ann.intervals:
            if lo < prev_hi or hi <= lo or hi > seq_len:
                raise ValueError(
                    f"Invalid compound interval ({lo}, {hi}) on '{ires_ann.feature_id}' for molecule length {seq_len}"
                )
            pieces.append(mol.sequence[lo:hi])
            prev_hi = hi
        return "".join(pieces)
    return u5_seq[-600:]


def _load_verified_joblib(
    weights_dir: Path,
    filename: str,
    *,
    require_commercial_clean: bool = True,
) -> dict[str, Any]:
    """Authenticate joblib artifact size, SHA-256, and license track against TRAINING_PROVENANCE_MANIFEST.json."""
    manifest_path = weights_dir / "TRAINING_PROVENANCE_MANIFEST.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Refusing to deserialize '{filename}': missing provenance manifest '{manifest_path}'"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"Refusing to deserialize '{filename}': provenance manifest must be a JSON object")
    by_name = {
        str(item["filename"]): item
        for item in manifest.get("artifacts", [])
        if isinstance(item, dict) and "filename" in item
    }
    if filename not in by_name:
        raise ValueError(f"Refusing to deserialize '{filename}': not registered in TRAINING_PROVENANCE_MANIFEST.json")
    expected = by_name[filename]
    schema_id = manifest.get("schema")
    if require_commercial_clean:
        top_track = manifest.get("compliance_track")
        item_track = expected.get("license_track")
        if top_track is not None and str(top_track) != "commercial_clean":
            raise PermissionError(
                f"Refusing to deserialize '{filename}' in commercial_clean loader: manifest compliance_track='{top_track}'"
            )
        if item_track is not None and str(item_track) != "commercial_clean":
            raise PermissionError(
                f"Refusing to deserialize '{filename}' in commercial_clean loader: artifact license_track='{item_track}'"
            )
        if schema_id not in _SUPPORTED_PROVENANCE_SCHEMAS:
            raise PermissionError(
                f"Refusing to deserialize '{filename}' in commercial_clean loader: unsupported or missing manifest schema '{schema_id}'"
            )
        if top_track != "commercial_clean":
            raise PermissionError(
                f"Refusing to deserialize '{filename}' in commercial_clean loader: missing manifest compliance_track='commercial_clean'"
            )
        if manifest.get("zero_viennarna_features_or_distilled_labels") is not True:
            raise PermissionError(
                f"Refusing to deserialize '{filename}' in commercial_clean loader: zero_viennarna_features_or_distilled_labels must be true"
            )
        if manifest.get("zero_gpl_or_noncommercial_weights") is not True:
            raise PermissionError(
                f"Refusing to deserialize '{filename}' in commercial_clean loader: zero_gpl_or_noncommercial_weights must be true"
            )
        if schema_id == "mrnax.surrogate_training_provenance.v2.0" and item_track != "commercial_clean":
            raise PermissionError(
                f"Refusing to deserialize '{filename}' in commercial_clean loader: v2.0 artifact missing license_track='commercial_clean'"
            )
        _assert_permissive_artifact_license(expected.get("license"), filename=filename)
        if expected.get("uses_viennarna_features_or_labels") is not False:
            raise PermissionError(
                f"Refusing to deserialize '{filename}' in commercial_clean loader: uses_viennarna_features_or_labels must be literal false"
            )
    decl_feats = expected.get("n_features") if "n_features" in expected else expected.get("feature_count")
    if isinstance(decl_feats, bool) or not isinstance(decl_feats, int) or decl_feats <= 0:
        raise ValueError(
            f"Refusing to deserialize '{filename}': manifest must declare a positive integer n_features / feature_count"
        )
    fpath = weights_dir / filename
    raw_bytes = fpath.read_bytes()
    if len(raw_bytes) != int(expected["bytes"]):
        raise ValueError(
            f"Integrity error on '{filename}': byte size {len(raw_bytes)} != expected {expected['bytes']}"
        )
    actual_sha = hashlib.sha256(raw_bytes).hexdigest()
    if actual_sha != str(expected["sha256"]):
        raise ValueError(
            f"Integrity error on '{filename}': sha256 {actual_sha} != expected {expected['sha256']}"
        )
    if sys.platform == "darwin":
        from mrnax.engine._darwin_joblib_worker import get_darwin_joblib_client

        loaded = dict(
            get_darwin_joblib_client().load_verified_bytes(
                f"{weights_dir.resolve()}::{filename}::{actual_sha}", raw_bytes
            )
        )
    else:
        import joblib

        loaded = dict(joblib.load(io.BytesIO(raw_bytes)))

    loaded["_manifest_n_features"] = decl_feats
    loaded["_manifest_filename"] = filename
    loaded["_manifest_schema"] = str(schema_id or "")
    loaded["_manifest_model_name"] = str(expected.get("model_name") or Path(filename).stem)
    return loaded


class CommercialCleanSurrogateV1:
    """Axis-routed, zero-ViennaRNA commercial-clean jury (`surrogate_v1_clean`, Wave 5 4-head edition).

    Axes are kept strictly separate (never blindly 50/50 diluted):
    - Axis 1 (`utr5_mrl_pred`): `utr5_mrl_pure_seq_lgbm.joblib` (GSE232927 CC-BY-4.0)
    - Axis 2 (`cds_stability_pred`): `cds_codon_stability_pure_seq_lgbm.joblib` (Wu 2019 CC-BY-4.0)
    - Axis 3 (`consensus_halflife_pred`): `saluki_consensus_halflife_pure_seq_lgbm.joblib` (Agarwal & Kelley 2022 Apache-2.0 / CC-BY-4.0)
    - Axis 4 (`utr3_stability_pred`): `utr3_stability_pure_seq_lgbm.joblib` (Siegel et al. 2022 G3 CC-BY-4.0 / fastUTR MPRA)
    """

    def __init__(self, weights_dir: Path) -> None:
        self.weights_dir = Path(weights_dir)
        self._utr5_bundle: dict[str, Any] = _load_verified_joblib(
            self.weights_dir, "utr5_mrl_pure_seq_lgbm.joblib"
        )
        self._cds_bundle: dict[str, Any] = _load_verified_joblib(
            self.weights_dir, "cds_codon_stability_pure_seq_lgbm.joblib"
        )
        self._hl_bundle: dict[str, Any] = _load_verified_joblib(
            self.weights_dir, "saluki_consensus_halflife_pure_seq_lgbm.joblib"
        )
        utr3_path = self.weights_dir / "utr3_stability_pure_seq_lgbm.joblib"
        self._utr3_bundle: dict[str, Any] | None = (
            _load_verified_joblib(self.weights_dir, "utr3_stability_pure_seq_lgbm.joblib")
            if utr3_path.is_file()
            else None
        )

    @property
    def loaded_heads(self) -> list[str]:
        heads = ["5utr_mrl", "cds_codon_stability", "saluki_full_tx_halflife"]
        if self._utr3_bundle is not None:
            heads.append("3utr_mpra_stability")
        return heads

    @classmethod
    @functools.lru_cache(maxsize=4)
    def from_seed_dir(cls, seed_dir: Path) -> CommercialCleanSurrogateV1:
        return cls(Path(seed_dir) / "surrogate_v1")

    @staticmethod
    def _predict_bundle(bundle: dict[str, Any], X: np.ndarray) -> np.ndarray:
        expected_w = bundle.get("_manifest_n_features")
        if X.ndim != 2:
            raise ValueError(f"Expected 2D feature matrix for '{bundle.get('_manifest_filename')}', got shape={X.shape}")
        if isinstance(expected_w, bool) or not isinstance(expected_w, int) or expected_w <= 0:
            raise ValueError(f"Missing verified manifest feature width for '{bundle.get('_manifest_filename')}'")
        if X.shape[1] != expected_w:
            raise ValueError(
                f"Feature width drift for '{bundle.get('_manifest_filename')}': assembled {X.shape[1]} columns != manifest n_features={expected_w}"
            )
        return np.asarray(bundle["model"].predict(X), dtype=np.float64)

    def score_5utr_mrl(self, utr5_seqs: list[str]) -> np.ndarray:
        if not utr5_seqs:
            return np.empty(0, dtype=np.float64)
        cols = tuple(self._utr5_bundle["features"])
        n_rows, n_cols = len(utr5_seqs), len(cols)
        X = np.fromiter(
            (f[c] for f in (extract_5utr_features(s) for s in utr5_seqs) for c in cols),
            dtype=np.float32,
            count=n_rows * n_cols,
        ).reshape(n_rows, n_cols)
        return self._predict_bundle(self._utr5_bundle, X)

    def score_cds_stability(self, cds_seqs: list[str]) -> np.ndarray:
        if not cds_seqs:
            return np.empty(0, dtype=np.float64)
        cols = tuple(self._cds_bundle["features"])
        n_rows, n_cols = len(cds_seqs), len(cols)
        X = np.fromiter(
            (f[c] for f in (extract_cds_features(s) for s in cds_seqs) for c in cols),
            dtype=np.float32,
            count=n_rows * n_cols,
        ).reshape(n_rows, n_cols)
        return self._predict_bundle(self._cds_bundle, X)

    def score_consensus_halflife_batch(
        self, triplets: list[tuple[str, str, str]]
    ) -> np.ndarray:
        if not triplets:
            return np.empty(0, dtype=np.float64)
        cols = tuple(self._hl_bundle["features"])
        n_rows, n_cols = len(triplets), len(cols)
        X = np.fromiter(
            (
                f[c]
                for f in (extract_full_tx_halflife_features(u5, cds, u3) for u5, cds, u3 in triplets)
                for c in cols
            ),
            dtype=np.float32,
            count=n_rows * n_cols,
        ).reshape(n_rows, n_cols)
        return self._predict_bundle(self._hl_bundle, X)

    def score_consensus_halflife(self, utr5_seq: str, cds_seq: str, utr3_seq: str) -> float:
        return float(self.score_consensus_halflife_batch([(utr5_seq, cds_seq, utr3_seq)])[0])

    def score_3utr_stability(self, utr3_seqs: list[str]) -> np.ndarray:
        if not utr3_seqs:
            return np.empty(0, dtype=np.float64)
        if self._utr3_bundle is None:
            raise RuntimeError("3'UTR MPRA stability head (utr3_stability_pure_seq_lgbm.joblib) not loaded.")
        cols = tuple(self._utr3_bundle["features"])
        n_rows, n_cols = len(utr3_seqs), len(cols)
        X = np.fromiter(
            (f[c] for f in (extract_3utr_features(s) for s in utr3_seqs) for c in cols),
            dtype=np.float32,
            count=n_rows * n_cols,
        ).reshape(n_rows, n_cols)
        return self._predict_bundle(self._utr3_bundle, X)

    def score_molecule_axes(self, mol: RNAMoleculeSpec) -> dict[str, float]:
        u5 = mol.get_segment_sequence("five_prime_utr")
        cds = mol.get_segment_sequence("cds")
        u3 = mol.get_segment_sequence("three_prime_utr")
        mrl = float(self.score_5utr_mrl([u5])[0])
        stab = float(self.score_cds_stability([cds])[0])
        hl = self.score_consensus_halflife(u5, cds, u3)
        out = {
            "utr5_mrl_pred": round(mrl, 6),
            "cds_stability_pred": round(stab, 6),
            "consensus_halflife_pred": round(hl, 6),
        }
        if self._utr3_bundle is not None:
            out["utr3_stability_pred"] = round(float(self.score_3utr_stability([u3])[0]), 6)
        return out

    def score_bundle_composite(self, bundle: ConstructBundle) -> float:
        """Deterministic scalar ranker ONLY for single-axis fixed-pool scorer replay (`run_fixed_pool_scorer_replay`)."""
        axes = self.score_molecule_axes(bundle.molecules[0])
        return round(
            axes["utr5_mrl_pred"]
            + axes["cds_stability_pred"]
            + axes["consensus_halflife_pred"]
            + axes.get("utr3_stability_pred", 0.0),
            6,
        )

    def evaluate_molecule_metrics(
        self, mol: RNAMoleculeSpec, *, participated_in: str = "pareto_selection_or_ranking"
    ) -> list[MetricResult]:
        axes = self.score_molecule_axes(mol)
        schema_chem_mode = (
            "native_chemistry_validated"
            if mol.chemistry_track.uridine_chemistry == "unmodified_u"
            else "cross_chemistry_thermodynamic_approximation"
        )
        res = [
            MetricResult(
                metric_id="utr5_mrl_pred",
                value=axes["utr5_mrl_pred"],
                unit="predicted_mrl",
                direction="maximize",
                evaluated_scopes=["five_prime_utr"],
                tool_name="CommercialCleanSurrogateV1:utr5_mrl_lgbm",
                tool_version="1.1.0",
                chemistry_evaluation_mode=schema_chem_mode,  # type: ignore[arg-type]
                participated_in=participated_in,  # type: ignore[arg-type]
            ),
            MetricResult(
                metric_id="cds_stability_pred",
                value=axes["cds_stability_pred"],
                unit="predicted_stability_slope",
                direction="maximize",
                evaluated_scopes=["cds"],
                tool_name="CommercialCleanSurrogateV1:cds_stability_lgbm",
                tool_version="1.1.0",
                chemistry_evaluation_mode=schema_chem_mode,  # type: ignore[arg-type]
                participated_in=participated_in,  # type: ignore[arg-type]
            ),
            MetricResult(
                metric_id="consensus_halflife_pred",
                value=axes["consensus_halflife_pred"],
                unit="predicted_halflife_zscore",
                direction="maximize",
                evaluated_scopes=["five_prime_utr", "cds", "three_prime_utr"],
                tool_name="CommercialCleanSurrogateV1:saluki_consensus_lgbm",
                tool_version="1.1.0",
                chemistry_evaluation_mode=schema_chem_mode,  # type: ignore[arg-type]
                participated_in=participated_in,  # type: ignore[arg-type]
            ),
        ]
        if "utr3_stability_pred" in axes:
            res.append(
                MetricResult(
                    metric_id="utr3_stability_pred",
                    value=axes["utr3_stability_pred"],
                    unit="predicted_3utr_mpra_stability",
                    direction="maximize",
                    evaluated_scopes=["three_prime_utr"],
                    tool_name="CommercialCleanSurrogateV1:fastutr_3utr_lgbm",
                    tool_version="1.1.0",
                    chemistry_evaluation_mode=schema_chem_mode,  # type: ignore[arg-type]
                    participated_in=participated_in,  # type: ignore[arg-type]
                )
            )
        return res

    def evaluate_molecule(
        self, mol: RNAMoleculeSpec
    ) -> tuple[list[MetricResult], list[AdvisoryNote]]:
        metrics = self.evaluate_molecule_metrics(mol)
        advisories: list[AdvisoryNote] = []
        chem = mol.chemistry_track.uridine_chemistry
        if chem != "unmodified_u":
            advisories.append(
                AdvisoryNote(
                    advisory_id="adv_m15_seq_surrogate_chem_domain",
                    category="chemistry_approximation",
                    message=(
                        f"Pure-sequence surrogates (GSE232927 / Wu2019 / Saluki / fastUTR) were trained on endogenous/unmodified-U "
                        f"assays and applied to '{chem}'. Enable structRFM chemistry-specific SHAPE-MaP probe ('mpU' for m1psi; "
                        f"RUL-CC-CHEM-M1PSI-STRUCT-SHIFT-006) for modified-RNA structure shifts."
                    ),
                )
            )
        u3_len = len(clean_dna(mol.get_segment_sequence("three_prime_utr")))
        if self._utr3_bundle is not None and not (130 <= u3_len <= 190):
            advisories.append(
                AdvisoryNote(
                    advisory_id="adv_w5_fastutr_length_domain",
                    category="model_domain_boundary",
                    message=(
                        f"3'UTR length ({u3_len} nt) is outside fastUTR MPRA training window (150-170 nt); "
                        "prefer consensus_halflife_pred (Saluki full-transcript PC1) for primary ranking."
                    ),
                )
            )
        advisories.append(
            AdvisoryNote(
                advisory_id="adv_w11_dual_utr_sensitivity_and_crosscell_caution",
                category="model_domain_boundary",
                message=(
                    "Descriptive sensitivity caution (RUL-013/014): across the 8 prose-grouped HeLa non-significant dual-UTR pairs "
                    "in PMC12355064, pure-sequence surrogate spreads are 0.908 (utr5_mrl_pred), 0.166 (utr3_stability_pred), and "
                    "1.296 (saluki_nluc) — descriptive sensitivity spreads only, NOT a formal equivalence margin; single-nt UTR "
                    "variant deltas also exhibit low cross-cell transfer in Su 2025 (HEK293T vs SH-SY5Y rho=0.0786)."
                ),
            )
        )
        return metrics, advisories


_SURROGATE_V2_HEAD_CANDIDATES: dict[str, tuple[str, ...]] = {
    "utr5": ("utr5_mrl_pure_seq_v2.joblib", "utr5_mrl_pure_seq_lgbm.joblib"),
    "cds": ("cds_codon_stability_v2.joblib", "cds_codon_stability_pure_seq_lgbm.joblib"),
    "hl": ("saluki_consensus_halflife_v2.joblib", "saluki_consensus_halflife_pure_seq_lgbm.joblib"),
    "utr3": ("utr3_stability_pure_seq_v2.joblib", "utr3_stability_pure_seq_lgbm.joblib"),
    "isoform_mrl": ("full_tx_isoform_mrl_v2.joblib",),
    "rmdb_aup": ("rmdb_aup_hydrolysis_proxy_v2.joblib",),
    "circ_ires": ("circrna_ires_translation_v2.joblib",),
}


def has_surrogate_v2_artifacts(seed_dir: Path) -> bool:
    """Return True when `seed_dir / 'surrogate_v2'` has a verified provenance manifest and core 3-head models."""
    v2_dir = Path(seed_dir) / "surrogate_v2"
    if not (v2_dir / "TRAINING_PROVENANCE_MANIFEST.json").is_file():
        return False
    for req_key in ("utr5", "cds", "hl"):
        if not any((v2_dir / fname).is_file() for fname in _SURROGATE_V2_HEAD_CANDIDATES[req_key]):
            return False
    return True


class CommercialCleanSurrogateV2(CommercialCleanSurrogateV1):
    """Wave 1B `surrogate_v2` commercial-clean jury with stacked Ridge+VotingRegressor heads & circRNA IRES scorer."""

    def __init__(self, weights_dir: Path, *, fallback_v1_dir: Path | None = None) -> None:
        self.weights_dir = Path(weights_dir)
        resolved_v1 = (
            Path(fallback_v1_dir)
            if fallback_v1_dir is not None
            else (self.weights_dir.parent / "surrogate_v1")
        )
        self._utr5_bundle = self._load_first_available("utr5", required=True)  # type: ignore[assignment]
        self._cds_bundle = self._load_first_available("cds", required=True)  # type: ignore[assignment]
        self._hl_bundle = self._load_first_available("hl", required=True)  # type: ignore[assignment]
        self._utr3_bundle = self._load_first_available("utr3", required=False)
        if (
            self._utr3_bundle is None
            and (resolved_v1 / "utr3_stability_pure_seq_lgbm.joblib").is_file()
            and (resolved_v1 / "TRAINING_PROVENANCE_MANIFEST.json").is_file()
        ):
            self._utr3_bundle = _load_verified_joblib(
                resolved_v1, "utr3_stability_pure_seq_lgbm.joblib"
            )
        self._isoform_mrl_bundle = self._load_first_available("isoform_mrl", required=False)
        self._rmdb_aup_bundle = self._load_first_available("rmdb_aup", required=False)
        self._circ_ires_bundle = self._load_first_available("circ_ires", required=False)

    @classmethod
    @functools.lru_cache(maxsize=4)
    def from_seed_dir(cls, seed_dir: Path) -> CommercialCleanSurrogateV2:
        resolved = Path(seed_dir)
        return cls(resolved / "surrogate_v2", fallback_v1_dir=resolved / "surrogate_v1")

    def _load_first_available(self, head_key: str, *, required: bool) -> dict[str, Any] | None:
        for fname in _SURROGATE_V2_HEAD_CANDIDATES[head_key]:
            if (self.weights_dir / fname).is_file():
                return _load_verified_joblib(self.weights_dir, fname)
        if required:
            raise FileNotFoundError(
                f"Missing required surrogate_v2 head '{head_key}' in '{self.weights_dir}'"
            )
        return None

    @property
    def loaded_heads(self) -> list[str]:
        heads = super().loaded_heads
        if self._isoform_mrl_bundle is not None:
            heads.append("full_tx_isoform_mrl")
        if self._rmdb_aup_bundle is not None:
            heads.append("rmdb_reactivity_hydrolysis_proxy")
        if self._circ_ires_bundle is not None:
            heads.append("circrna_ires_translation")
        return heads

    def score_5utr_mrl(self, utr5_seqs: list[str]) -> np.ndarray:
        if not utr5_seqs:
            return np.empty(0, dtype=np.float64)
        if "tab_features" not in self._utr5_bundle:
            return super().score_5utr_mrl(utr5_seqs)
        from mrnax.engine.features_v2 import dict_list_to_matrix, extract_utr5_v2, seq_to_kmer_tokens

        Xm = dict_list_to_matrix(
            [extract_utr5_v2(s) for s in utr5_seqs],
            list(self._utr5_bundle["tab_features"]),
        )
        ks = tuple(int(k) for k in self._utr5_bundle.get("kmer_ks", [3, 4]))
        km = np.asarray(
            self._utr5_bundle["kmer_ridge"].predict([seq_to_kmer_tokens(s, ks=ks) for s in utr5_seqs]),
            dtype=np.float32,
        )
        return self._predict_bundle(self._utr5_bundle, np.hstack([Xm, km[:, None]]))

    def score_cds_stability(self, cds_seqs: list[str]) -> np.ndarray:
        if not cds_seqs:
            return np.empty(0, dtype=np.float64)
        if "tab_features" not in self._cds_bundle:
            return super().score_cds_stability(cds_seqs)
        from mrnax.engine.features_v2 import cds_to_bicodon_tokens, dict_list_to_matrix, extract_cds_v2

        Xm = dict_list_to_matrix(
            [extract_cds_v2(s) for s in cds_seqs],
            list(self._cds_bundle["tab_features"]),
        )
        bm = np.asarray(
            self._cds_bundle["kmer_ridge"].predict([cds_to_bicodon_tokens(s) for s in cds_seqs]),
            dtype=np.float32,
        )
        return self._predict_bundle(self._cds_bundle, np.hstack([Xm, bm[:, None]]))

    def score_consensus_halflife_batch(
        self, triplets: list[tuple[str, str, str]]
    ) -> np.ndarray:
        if not triplets:
            return np.empty(0, dtype=np.float64)
        if "tab_features" not in self._hl_bundle:
            return super().score_consensus_halflife_batch(triplets)
        from mrnax.engine.features_v2 import (
            cds_to_bicodon_tokens,
            dict_list_to_matrix,
            extract_full_tx_v2,
            seq_to_kmer_tokens,
        )

        Xm = dict_list_to_matrix(
            [extract_full_tx_v2(u5, cds, u3) for u5, cds, u3 in triplets],
            list(self._hl_bundle["tab_features"]),
        )
        m1 = np.asarray(
            self._hl_bundle["utr3_kmer_ridge"].predict(
                [seq_to_kmer_tokens(u3, ks=(4, 5)) for _, _, u3 in triplets]
            ),
            dtype=np.float32,
        )
        m2 = np.asarray(
            self._hl_bundle["cds_bicodon_ridge"].predict(
                [cds_to_bicodon_tokens(cds) for _, cds, _ in triplets]
            ),
            dtype=np.float32,
        )
        return self._predict_bundle(self._hl_bundle, np.hstack([Xm, m1[:, None], m2[:, None]]))

    def score_isoform_mrl(self, triplets: list[tuple[str, str, str]]) -> np.ndarray:
        if not triplets:
            return np.empty(0, dtype=np.float64)
        if self._isoform_mrl_bundle is None:
            raise RuntimeError("Full-transcript isoform MRL head (full_tx_isoform_mrl_v2.joblib) not loaded.")
        from mrnax.engine.features_v2 import dict_list_to_matrix, extract_full_tx_v2, seq_to_kmer_tokens

        Xm = dict_list_to_matrix(
            [extract_full_tx_v2(u5, cds, u3) for u5, cds, u3 in triplets],
            list(self._isoform_mrl_bundle["tab_features"]),
        )
        m1 = np.asarray(
            self._isoform_mrl_bundle["utr3_kmer_ridge"].predict(
                [seq_to_kmer_tokens(u3, ks=(4,)) for _, _, u3 in triplets]
            ),
            dtype=np.float32,
        )
        return self._predict_bundle(self._isoform_mrl_bundle, np.hstack([Xm, m1[:, None]]))

    def score_rmdb_aup_proxy(self, seqs: list[str]) -> np.ndarray:
        if not seqs:
            return np.empty(0, dtype=np.float64)
        if self._rmdb_aup_bundle is None:
            raise RuntimeError("RMDB AUP hydrolysis proxy head (rmdb_aup_hydrolysis_proxy_v2.joblib) not loaded.")
        from mrnax.engine.features_v2 import dict_list_to_matrix, extract_rmdb_seq_v2, seq_to_kmer_tokens

        Xm = dict_list_to_matrix(
            [extract_rmdb_seq_v2(s) for s in seqs],
            list(self._rmdb_aup_bundle["tab_features"]),
        )
        km = np.asarray(
            self._rmdb_aup_bundle["kmer_ridge"].predict(
                [seq_to_kmer_tokens(s, ks=(4,)) for s in seqs]
            ),
            dtype=np.float32,
        )
        return self._predict_bundle(self._rmdb_aup_bundle, np.hstack([Xm, km[:, None]]))

    def score_circrna_ires(self, ires_seqs: list[str]) -> np.ndarray:
        if not ires_seqs:
            return np.empty(0, dtype=np.float64)
        if self._circ_ires_bundle is None:
            raise RuntimeError("circRNA IRES translation head (circrna_ires_translation_v2.joblib) not loaded.")
        from mrnax.engine.features_v2 import dict_list_to_matrix, extract_circrna_ires_v2

        Xm = dict_list_to_matrix(
            [extract_circrna_ires_v2(s) for s in ires_seqs],
            list(self._circ_ires_bundle["features"]),
        )
        return self._predict_bundle(self._circ_ires_bundle, Xm)

    def score_molecule_axes(self, mol: RNAMoleculeSpec) -> dict[str, float]:
        out = super().score_molecule_axes(mol)
        u5 = mol.get_segment_sequence("five_prime_utr")
        cds = mol.get_segment_sequence("cds")
        u3 = mol.get_segment_sequence("three_prime_utr")
        if self._isoform_mrl_bundle is not None:
            out["full_tx_isoform_mrl_pred"] = round(
                float(self.score_isoform_mrl([(u5, cds, u3)])[0]), 6
            )
        if self._rmdb_aup_bundle is not None:
            out["rmdb_aup_hydrolysis_proxy_pred"] = round(
                float(self.score_rmdb_aup_proxy([u5 + cds + u3])[0]), 6
            )
        if mol.modality == "circrna" and self._circ_ires_bundle is not None:
            ires_slice = extract_circrna_ires_interval(mol)
            out["circrna_ires_translation_pred"] = round(
                float(self.score_circrna_ires([ires_slice])[0]), 6
            )
        return out

    @staticmethod
    def _bundle_provenance(bundle: dict[str, Any], default_v1_tag: str) -> tuple[str, str]:
        schema = str(bundle.get("_manifest_schema") or "")
        mdl_name = str(bundle.get("_manifest_model_name") or default_v1_tag)
        if schema == "mrnax.surrogate_training_provenance.v2.0" or "tab_features" in bundle:
            return f"CommercialCleanSurrogateV2:{mdl_name}", "2.0.0"
        return f"CommercialCleanSurrogateV1:{default_v1_tag}", "1.1.0"

    def evaluate_molecule_metrics(
        self, mol: RNAMoleculeSpec, *, participated_in: str = "pareto_selection_or_ranking"
    ) -> list[MetricResult]:
        axes = self.score_molecule_axes(mol)
        schema_chem_mode = (
            "native_chemistry_validated"
            if mol.chemistry_track.uridine_chemistry == "unmodified_u"
            else "cross_chemistry_thermodynamic_approximation"
        )
        u5_tool, u5_ver = self._bundle_provenance(self._utr5_bundle, "utr5_mrl_lgbm")
        cds_tool, cds_ver = self._bundle_provenance(self._cds_bundle, "cds_stability_lgbm")
        hl_tool, hl_ver = self._bundle_provenance(self._hl_bundle, "saluki_consensus_lgbm")
        res = [
            MetricResult(
                metric_id="utr5_mrl_pred",
                value=axes["utr5_mrl_pred"],
                unit="predicted_mrl",
                direction="maximize",
                evaluated_scopes=["five_prime_utr"],
                tool_name=u5_tool,
                tool_version=u5_ver,
                chemistry_evaluation_mode=schema_chem_mode,  # type: ignore[arg-type]
                participated_in=participated_in,  # type: ignore[arg-type]
            ),
            MetricResult(
                metric_id="cds_stability_pred",
                value=axes["cds_stability_pred"],
                unit="predicted_stability_slope",
                direction="maximize",
                evaluated_scopes=["cds"],
                tool_name=cds_tool,
                tool_version=cds_ver,
                chemistry_evaluation_mode=schema_chem_mode,  # type: ignore[arg-type]
                participated_in=participated_in,  # type: ignore[arg-type]
            ),
            MetricResult(
                metric_id="consensus_halflife_pred",
                value=axes["consensus_halflife_pred"],
                unit="predicted_halflife_zscore",
                direction="maximize",
                evaluated_scopes=["five_prime_utr", "cds", "three_prime_utr"],
                tool_name=hl_tool,
                tool_version=hl_ver,
                chemistry_evaluation_mode=schema_chem_mode,  # type: ignore[arg-type]
                participated_in=participated_in,  # type: ignore[arg-type]
            ),
        ]
        if "utr3_stability_pred" in axes and self._utr3_bundle is not None:
            u3_tool, u3_ver = self._bundle_provenance(self._utr3_bundle, "fastutr_3utr_lgbm")
            res.append(
                MetricResult(
                    metric_id="utr3_stability_pred",
                    value=axes["utr3_stability_pred"],
                    unit="predicted_3utr_mpra_stability",
                    direction="maximize",
                    evaluated_scopes=["three_prime_utr"],
                    tool_name=u3_tool,
                    tool_version=u3_ver,
                    chemistry_evaluation_mode=schema_chem_mode,  # type: ignore[arg-type]
                    participated_in=participated_in,  # type: ignore[arg-type]
                )
            )
        if "full_tx_isoform_mrl_pred" in axes and self._isoform_mrl_bundle is not None:
            iso_tool, iso_ver = self._bundle_provenance(self._isoform_mrl_bundle, "full_tx_isoform_mrl_v2")
            res.append(
                MetricResult(
                    metric_id="full_tx_isoform_mrl_pred",
                    value=axes["full_tx_isoform_mrl_pred"],
                    unit="predicted_isoform_mrl",
                    direction="maximize",
                    evaluated_scopes=["five_prime_utr", "cds", "three_prime_utr"],
                    tool_name=iso_tool,
                    tool_version=iso_ver,
                    chemistry_evaluation_mode=schema_chem_mode,  # type: ignore[arg-type]
                    participated_in="display_only",
                )
            )
        if "rmdb_aup_hydrolysis_proxy_pred" in axes and self._rmdb_aup_bundle is not None:
            aup_tool, aup_ver = self._bundle_provenance(self._rmdb_aup_bundle, "rmdb_aup_hydrolysis_proxy_v2")
            res.append(
                MetricResult(
                    metric_id="rmdb_aup_hydrolysis_proxy_pred",
                    value=axes["rmdb_aup_hydrolysis_proxy_pred"],
                    unit="predicted_reactivity_aup",
                    direction="minimize",
                    evaluated_scopes=["five_prime_utr", "cds", "three_prime_utr"],
                    tool_name=aup_tool,
                    tool_version=aup_ver,
                    chemistry_evaluation_mode=schema_chem_mode,  # type: ignore[arg-type]
                    participated_in="display_only",
                )
            )
        if "circrna_ires_translation_pred" in axes and self._circ_ires_bundle is not None:
            ci_tool, ci_ver = self._bundle_provenance(self._circ_ires_bundle, "circrna_ires_translation_v2")
            res.append(
                MetricResult(
                    metric_id="circrna_ires_translation_pred",
                    value=axes["circrna_ires_translation_pred"],
                    unit="predicted_ires_log2_rel_expr",
                    direction="maximize",
                    evaluated_scopes=["five_prime_utr"],
                    tool_name=ci_tool,
                    tool_version=ci_ver,
                    chemistry_evaluation_mode=schema_chem_mode,  # type: ignore[arg-type]
                    participated_in="display_only",
                )
            )
        return res


@functools.lru_cache(maxsize=8)
def load_surrogate_jury(seed_dir: Path, *, prefer_v2: bool = True) -> CommercialCleanSurrogateV1:
    """Load `surrogate_v2` when verified artifacts exist under `seed_dir / 'surrogate_v2'`, else fall back to v1."""
    resolved = Path(seed_dir)
    if prefer_v2 and has_surrogate_v2_artifacts(resolved):
        return CommercialCleanSurrogateV2.from_seed_dir(resolved)
    return CommercialCleanSurrogateV1.from_seed_dir(resolved)


# ============================================================================
# Native MIT `structRFM` HuggingFace Loader & Chemistry-Specific Probe Router
# ============================================================================


class TokenizerLike(Protocol):
    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]: ...


def validate_structrfm_checkpoint_path(
    checkpoint_dir: Path,
    *,
    must_exist: bool = False,
    expected_weights_sha256: str | None = None,
    allow_sandbox_override: bool = False,
) -> Path:
    """Gotcha 1: Must load base `checkpoints/structRFM_checkpoint` (NEVER `structRFM_for_SPL`) with verified release digests."""
    p = Path(checkpoint_dir)
    if p.name != "structRFM_checkpoint" or "structRFM_for_SPL" in str(p):
        raise ValueError(
            f"Invalid structRFM checkpoint '{p}': must select base 'structRFM_checkpoint' "
            "(NOT 'splice_site_prediction/structRFM_for_SPL')."
        )
    if must_exist:
        cfg = p / "config.json"
        wt_file = (
            (p / "model.safetensors")
            if (p / "model.safetensors").is_file()
            else ((p / "pytorch_model.bin") if (p / "pytorch_model.bin").is_file() else None)
        )
        if not p.is_dir() or not cfg.is_file() or wt_file is None:
            raise FileNotFoundError(
                f"structRFM checkpoint directory '{p}' is missing config.json or model weights (model.safetensors / pytorch_model.bin)"
            )
        verify_release_directory_artifacts(
            p,
            "structrfm_rna_foundation",
            allow_sandbox_override=allow_sandbox_override,
            weights_sha256_override=expected_weights_sha256,
            weights_env_var="MRNAX_STRUCTRFM_WEIGHTS_SHA256",
        )
    return p


def prep_structrfm_unspaced_rna(seq: str, max_nt: int | None = None) -> str:
    """Gotcha 2 (`USE_SPACED=False`): Return clean unspaced uppercase RNA (`AUGC`, NEVER `A U G C`)."""
    cleaned = re.sub(r"[^ACGTUacgtu]", "", str(seq)).upper().replace("T", "U")
    return cleaned[:max_nt] if max_nt is not None else cleaned


_APPROVED_STRUCTRFM_BASE_IDS: dict[str, int] = {"A": 5, "U": 6, "C": 7, "G": 8}


def validate_structrfm_tokenizer(tokenizer: TokenizerLike) -> None:
    """Gotcha 2: Verify exact canonical base token IDs (A=5, U=6, C=7, G=8) under `USE_SPACED=False`."""
    ids = tokenizer.encode("AUGC", add_special_tokens=False)
    rev_ids = tokenizer.encode("CGUA", add_special_tokens=False)
    unk_id = getattr(tokenizer, "unk_token_id", 0)
    expected_fwd = [_APPROVED_STRUCTRFM_BASE_IDS[b] for b in "AUGC"]
    expected_rev = [_APPROVED_STRUCTRFM_BASE_IDS[b] for b in "CGUA"]
    if (
        ids != expected_fwd
        or rev_ids != expected_rev
        or (unk_id is not None and unk_id in ids)
    ):
        raise RuntimeError(
            f"structRFM tokenizer failed USE_SPACED=False canonical base-ID check: "
            f"encode('AUGC')={ids} (expected {expected_fwd}), encode('CGUA')={rev_ids} (expected {expected_rev})."
        )


def stitch_sliding_window_embeddings(
    seq: str,
    window_encoder: Callable[[str], np.ndarray],
    *,
    win: int = 510,
    stride: int = 384,
) -> np.ndarray:
    """Gotcha 3: Stitch per-nucleotide embeddings for transcripts >510 nt using overlapping windows (`win=510`, `stride=384`)."""
    if win > 510:
        raise ValueError(f"structRFM max safe nucleotide window is 510 (max_position_embeddings=514); got win={win}")
    if stride < 1 or stride > win:
        raise ValueError(f"Invalid stride={stride} for win={win}")

    s = prep_structrfm_unspaced_rna(seq)
    L = len(s)
    if L == 0:
        raise ValueError("Cannot encode empty RNA sequence")

    if L <= win:
        h = np.asarray(window_encoder(s), dtype=np.float32)
        if h.ndim != 2 or h.shape[0] < L:
            raise ValueError(
                f"Window encoder output length ({h.shape[0] if h.ndim >= 1 else 0}) is shorter than input window ({L})"
            )
        return h[:L]

    first_h = np.asarray(window_encoder(s[:win]), dtype=np.float32)
    if first_h.ndim != 2 or first_h.shape[0] < win:
        raise ValueError(
            f"Window encoder output length ({first_h.shape[0] if first_h.ndim >= 1 else 0}) is shorter than input window ({win})"
        )
    dim = int(first_h.shape[1])
    accum = np.zeros((L, dim), dtype=np.float32)
    counts = np.zeros((L, 1), dtype=np.float32)

    accum[:win] += first_h[:win]
    counts[:win] += 1.0

    starts = list(range(stride, max(stride, L - win + 1), stride))
    last_start = L - win
    if not starts or starts[-1] != last_start:
        if last_start > 0:
            starts.append(last_start)

    for st in starts:
        sub = s[st : st + win]
        h = np.asarray(window_encoder(sub), dtype=np.float32)
        if h.ndim != 2 or h.shape[0] < len(sub):
            raise ValueError(
                f"Window encoder output length ({h.shape[0] if h.ndim >= 1 else 0}) is shorter than input window ({len(sub)})"
            )
        n = len(sub)
        accum[st : st + n] += h[:n]
        counts[st : st + n] += 1.0

    return accum / np.maximum(counts, 1.0)


_NATIVE_PROBE_KEYS = {
    "unmodified_u": "U",
    "5mou": "moU",
    "m1psi": "mpU",
}

STRUCTRFM_NATIVE_CHEMISTRIES = frozenset(_NATIVE_PROBE_KEYS)
_CHEM_TO_PROBE_KEY = {**_NATIVE_PROBE_KEYS, **{key.lower(): key for key in _NATIVE_PROBE_KEYS.values()}}


_LOADER_AUTHENTICATED_STRUCTRFM: WeakSet[StructRFMNativeScorer] = WeakSet()


def is_loader_authenticated_structrfm(obj: Any) -> bool:
    """True ONLY if `obj` was loaded via `StructRFMNativeScorer.from_checkpoint_and_seed_dir(..., allow_sandbox_override=False)`."""
    return isinstance(obj, StructRFMNativeScorer) and (obj in _LOADER_AUTHENTICATED_STRUCTRFM)


class StructRFMNativeScorer:
    """Native HuggingFace `structRFM` (MIT, zero `BPfold`) encoder + chemistry-specific SHAPE-MaP probe router."""

    def __init__(
        self,
        *,
        probes_dir: Path,
        tokenizer: Any = None,
        model: Any = None,
        device: str = "cpu",
    ) -> None:
        self.probes_dir = Path(probes_dir)
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self._utr5_probe: dict[str, Any] = _load_verified_joblib(
            self.probes_dir, "structrfm_5utr_mrl_probe.joblib"
        )
        self._shape_probes: dict[str, dict[str, Any]] = {
            "U": _load_verified_joblib(self.probes_dir, "structrfm_shapemap_U_probe.joblib"),
            "moU": _load_verified_joblib(self.probes_dir, "structrfm_shapemap_moU_probe.joblib"),
            "mpU": _load_verified_joblib(self.probes_dir, "structrfm_shapemap_mpU_probe.joblib"),
        }

    @classmethod
    def probes_only_from_seed_dir(cls, seed_dir: Path) -> StructRFMNativeScorer:
        return cls(probes_dir=Path(seed_dir) / "surrogate_v1")

    @classmethod
    def from_checkpoint_and_seed_dir(
        cls,
        checkpoint_dir: Path,
        seed_dir: Path,
        *,
        device: str | None = None,
        allow_sandbox_override: bool = False,
    ) -> StructRFMNativeScorer:
        validated_ckpt = validate_structrfm_checkpoint_path(
            checkpoint_dir, must_exist=True, allow_sandbox_override=allow_sandbox_override
        )
        import torch  # type: ignore[import-not-found]
        from transformers import AutoTokenizer, BertForMaskedLM  # type: ignore[import-not-found]

        resolved_dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        tok = AutoTokenizer.from_pretrained(str(validated_ckpt), local_files_only=True)
        validate_structrfm_tokenizer(tok)
        mdl = BertForMaskedLM.from_pretrained(
            str(validated_ckpt), output_hidden_states=True, local_files_only=True,
            use_safetensors=(validated_ckpt / "model.safetensors").is_file(),
        )
        mdl.eval().to(resolved_dev)
        inst = cls(
            probes_dir=Path(seed_dir) / "surrogate_v1",
            tokenizer=tok,
            model=mdl,
            device=resolved_dev,
        )
        if not allow_sandbox_override:
            _LOADER_AUTHENTICATED_STRUCTRFM.add(inst)
        return inst

    def predict_shapemap_from_hidden(
        self, hidden_states: np.ndarray, *, uridine_chemistry: str
    ) -> np.ndarray:
        """Gotcha 4: Strictly route per-nt hidden states `[L, 768]` to the matching chemistry probe (`U`, `moU`, `mpU`)."""
        key = _CHEM_TO_PROBE_KEY.get(uridine_chemistry.strip().lower())
        if key is None:
            raise ValueError(
                f"Unsupported uridine_chemistry '{uridine_chemistry}' for structRFM SHAPE-MaP probe; "
                f"supported chemistries: {sorted(_CHEM_TO_PROBE_KEY.keys())}. Refusing cross-chemistry fallback!"
            )
        probe = self._shape_probes[key]
        X = probe["scaler"].transform(np.asarray(hidden_states, dtype=np.float32))
        return np.asarray(probe["model"].predict(X), dtype=np.float32)

    def encode_per_nt(self, seq: str) -> np.ndarray:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "structRFM base encoder weights not loaded; initialize via `from_checkpoint_and_seed_dir` on GPU worker."
            )
        import torch  # type: ignore[import-not-found]

        def _encode_window(subseq: str) -> np.ndarray:
            enc = self.tokenizer(
                subseq, return_tensors="pt", truncation=True, max_length=514
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            use_amp = self.device.startswith("cuda")
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                h = self.model(**enc).hidden_states[-1][0, 1 : 1 + len(subseq), :]
            return h.float().cpu().numpy()

        return stitch_sliding_window_embeddings(seq, _encode_window, win=510, stride=384)

    def predict_per_nt_shapemap(self, seq: str, *, uridine_chemistry: str) -> np.ndarray:
        hidden = self.encode_per_nt(seq)
        return self.predict_shapemap_from_hidden(hidden, uridine_chemistry=uridine_chemistry)

    def evaluate_molecule(
        self, mol: RNAMoleculeSpec, *, participated_in: str = "pareto_selection_or_ranking"
    ) -> tuple[list[MetricResult], list[AdvisoryNote]]:
        chem = mol.chemistry_track.uridine_chemistry
        cds_seq = mol.get_segment_sequence("cds")
        shape_arr = self.predict_per_nt_shapemap(cds_seq, uridine_chemistry=chem)
        med_shape = round(float(np.median(shape_arr)), 6)
        return (
            [
                MetricResult(
                    metric_id="structrfm_cds_median_shape",
                    value=med_shape,
                    unit="normalized_shape_reactivity",
                    direction="minimize",
                    evaluated_scopes=["cds"],
                    tool_name="StructRFMNativeScorer:mauger2019_probe",
                    tool_version="0.0.9",
                    chemistry_evaluation_mode="native_chemistry_validated",
                    participated_in=participated_in,  # type: ignore[arg-type]
                )
            ],
            [],
        )

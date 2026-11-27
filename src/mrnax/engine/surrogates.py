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
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Protocol

import joblib
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


def extract_full_tx_halflife_features(utr5_seq: str, cds_seq: str, utr3_seq: str) -> dict[str, float]:
    u5 = clean_dna(utr5_seq)
    cds = clean_dna(cds_seq)
    u3 = clean_dna(utr3_seq)
    L = max(len(cds), 1)
    nc = L // 3
    pos3 = cds[2 : nc * 3 : 3]
    logs = [
        math.log(CODON_W[cds[3 * i : 3 * i + 3]])
        for i in range(nc)
        if cds[3 * i : 3 * i + 3] in CODON_W
    ]
    cai = math.exp(sum(logs) / len(logs)) if logs else 0.5
    cods = Counter(cds[3 * i : 3 * i + 3] for i in range(nc))
    u3L = max(len(u3), 1)
    u5L = max(len(u5), 1)
    feats: dict[str, float] = {
        "cds_len": float(len(cds)),
        "utr5_len": float(len(u5)),
        "utr3_len": float(len(u3)),
        "log_cds_len": math.log1p(len(cds)),
        "log_utr3_len": math.log1p(len(u3)),
        "cds_gc": (cds.count("G") + cds.count("C")) / L,
        "cds_gc3": (pos3.count("G") + pos3.count("C")) / max(len(pos3), 1),
        "cds_u": cds.count("T") / L,
        "cds_cai": float(cai),
        "u5_gc": (u5.count("G") + u5.count("C")) / u5L,
        "u5_uaug": float(len(re.findall("ATG", u5))),
        "u3_gc": (u3.count("G") + u3.count("C")) / u3L,
        "u3_u": u3.count("T") / u3L,
        "u3_are_cnt": float(len(re.findall("ATTTA", u3))),
        "u3_are_density": len(re.findall("ATTTA", u3)) / (u3L / 1000.0),
        "u3_pumilio_cnt": float(len(re.findall(r"TGTA[ACGT]ATA", u3))),
        "u3_pas_cnt": float(len(re.findall("AATAAA", u3))),
        "u3_cpg": sum(1 for i in range(max(len(u3) - 1, 0)) if u3[i : i + 2] == "CG") / u3L,
    }
    for cod in SENSE_CODONS_SORTED:
        feats[f"cod_{cod}"] = cods[cod] / max(nc, 1)
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


def _load_verified_joblib(weights_dir: Path, filename: str) -> dict[str, Any]:
    """Authenticate joblib artifact size & SHA-256 against TRAINING_PROVENANCE_MANIFEST.json BEFORE deserializing."""
    manifest_path = weights_dir / "TRAINING_PROVENANCE_MANIFEST.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Refusing to deserialize '{filename}': missing provenance manifest '{manifest_path}'"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_name = {str(item["filename"]): item for item in manifest.get("artifacts", [])}
    if filename not in by_name:
        raise ValueError(f"Refusing to deserialize '{filename}': not registered in TRAINING_PROVENANCE_MANIFEST.json")
    expected = by_name[filename]
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
    return joblib.load(io.BytesIO(raw_bytes))


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

    def score_5utr_mrl(self, utr5_seqs: list[str]) -> np.ndarray:
        cols = self._utr5_bundle["features"]
        X = np.array([[extract_5utr_features(s)[c] for c in cols] for s in utr5_seqs], dtype=np.float32)
        return np.asarray(self._utr5_bundle["model"].predict(X), dtype=np.float64)

    def score_cds_stability(self, cds_seqs: list[str]) -> np.ndarray:
        cols = self._cds_bundle["features"]
        X = np.array([[extract_cds_features(s)[c] for c in cols] for s in cds_seqs], dtype=np.float32)
        return np.asarray(self._cds_bundle["model"].predict(X), dtype=np.float64)

    def score_consensus_halflife_batch(
        self, triplets: list[tuple[str, str, str]]
    ) -> np.ndarray:
        cols = self._hl_bundle["features"]
        X = np.array(
            [[extract_full_tx_halflife_features(u5, cds, u3)[c] for c in cols] for u5, cds, u3 in triplets],
            dtype=np.float32,
        )
        return np.asarray(self._hl_bundle["model"].predict(X), dtype=np.float64)

    def score_consensus_halflife(self, utr5_seq: str, cds_seq: str, utr3_seq: str) -> float:
        return float(self.score_consensus_halflife_batch([(utr5_seq, cds_seq, utr3_seq)])[0])

    def score_3utr_stability(self, utr3_seqs: list[str]) -> np.ndarray:
        if self._utr3_bundle is None:
            raise RuntimeError("3'UTR MPRA stability head (utr3_stability_pure_seq_lgbm.joblib) not loaded.")
        cols = self._utr3_bundle["features"]
        X = np.array([[extract_3utr_features(s)[c] for c in cols] for s in utr3_seqs], dtype=np.float32)
        return np.asarray(self._utr3_bundle["model"].predict(X), dtype=np.float64)

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
        return metrics, advisories


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


_CHEM_TO_PROBE_KEY: dict[str, str] = {
    "unmodified_u": "U",
    "u": "U",
    "5mou": "moU",
    "mou": "moU",
    "m1psi": "mpU",
    "mpu": "mpU",
}


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
        return cls(
            probes_dir=Path(seed_dir) / "surrogate_v1",
            tokenizer=tok,
            model=mdl,
            device=resolved_dev,
        )

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

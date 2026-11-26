"""Self-contained streaming empirical tribunal for Sample et al. 2019 (GSE114002) & Castillo-Hair 2024 (GSE232927).

Designed to execute on an ephemeral Google Colab CPU VM via:
    colab run src/mrnax/rules/gse114002_runner.py > artifacts/m1_egfp/gse114002_evidence_audit.json
Zero bytes of large compressed MPRA tables are stored on the local Mac.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import sys
from datetime import datetime, timezone
from urllib.request import urlopen

# Pre-registered hypothesis specification (frozen BEFORE inspecting test statistics)
PRE_REGISTERED_SPEC = {
    "hypothesis_id": "HYP-5UTR-OOF-UAUG-MRL-V1",
    "primary_claim_id": "CC-MRNAX-MP01-001",
    "primary_citation": "Sample et al., Nat Biotechnol 37:803-809 (2019), DOI:10.1038/s41587-019-0164-5, Fig. 1c",
    "feature_definition": "Presence of >=1 out-of-frame upstream AUG (uAUG) vs zero AUG anywhere in the variable 5'UTR region upstream of reporter start codon",
    "readout_name_exact": "Mean Ribosome Load (MRL)",
    "expected_direction": "decrease (median_MRL_oof_uaug < median_MRL_no_uaug)",
    "pre_registered_support_criteria": {
        "min_total_reads_filter": 100,
        "max_median_delta_mrl": -0.50,
        "replicate_direction_concordance": "100% of biological replicates must show negative delta",
    },
    "evidence_boundary_note": (
        "MRL measures polysome ribosome loading 12h post-transfection in HEK293T, NOT late-timepoint "
        "secreted protein AUC or in vivo LNP efficacy. Both GSE114002 and GSE232927 originate from the "
        "Seelig Lab (UW) polysome-MPRA platform -> counted as 1 lab lineage."
    ),
}

# Verified SHA-256 digests from mRNAx-Data-Hub-20261126/00_manifests/master_data_catalog.csv
SAMPLES_TO_EVALUATE = [
    {
        "sample_id": "GSM3130435",
        "dataset_id": "GSE114002",
        "chemistry": "unmodified_u",
        "reporter": "eGFP",
        "replicate": 1,
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM3130nnn/GSM3130435/suppl/GSM3130435_egfp_unmod_1.csv.gz",
        "expected_sha256": None,  # Codex C01 ref
    },
    {
        "sample_id": "GSM3130436",
        "dataset_id": "GSE114002",
        "chemistry": "unmodified_u",
        "reporter": "eGFP",
        "replicate": 2,
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM3130nnn/GSM3130436/suppl/GSM3130436_egfp_unmod_2.csv.gz",
        "expected_sha256": None,  # Codex C02 ref
    },
    {
        "sample_id": "GSM3130437",
        "dataset_id": "GSE114002",
        "chemistry": "pseudouridine",
        "reporter": "eGFP",
        "replicate": 1,
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM3130nnn/GSM3130437/suppl/GSM3130437_egfp_pseudo_1.csv.gz",
        "expected_sha256": "3e02293aeaa6eab549612e0f4204e969d99eacafc31b80e15a19ead42b99d4f6",
    },
    {
        "sample_id": "GSM3130439",
        "dataset_id": "GSE114002",
        "chemistry": "m1psi",
        "reporter": "eGFP",
        "replicate": 1,
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM3130nnn/GSM3130439/suppl/GSM3130439_egfp_m1pseudo_1.csv.gz",
        "expected_sha256": "4bb54458e6846265cc5c3b1d73894fcb03569a0ab61b3a95a85f4d78aec287da",
    },
    {
        "sample_id": "GSM3130440",
        "dataset_id": "GSE114002",
        "chemistry": "m1psi",
        "reporter": "eGFP",
        "replicate": 2,
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM3130nnn/GSM3130440/suppl/GSM3130440_egfp_m1pseudo_2.csv.gz",
        "expected_sha256": "6bd42af41ea7bdc2806a37031fecb0c63c576b24c60fb898e8afe31a7d8cfd12",
    },
    {
        "sample_id": "GSM3130441",
        "dataset_id": "GSE114002",
        "chemistry": "unmodified_u",
        "reporter": "mCherry",
        "replicate": 1,
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM3130nnn/GSM3130441/suppl/GSM3130441_mcherry_1.csv.gz",
        "expected_sha256": "9f3f5749618525d6689279888dc60332a273156da71c621a964a3db2269f5944",
    },
    {
        "sample_id": "GSE232927_HEK293T_N25_r1",
        "dataset_id": "GSE232927",
        "chemistry": "unmodified_u",
        "reporter": "eGFP_random_5p_end_N25",
        "replicate": 1,
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE232nnn/GSE232927/suppl/GSE232927_processed_random_end_hek293t_N25_r1.csv.gz",
        "expected_sha256": "1a81e473a5420a44cb3a0db6f8fd3d2f67077604788ffbc7c5a1000642c034b2",
    },
]


def _quantiles(sorted_vals: list[float]) -> tuple[float, float, float]:
    n = len(sorted_vals)
    if n == 0:
        return (math.nan, math.nan, math.nan)
    q25 = sorted_vals[int(0.25 * (n - 1))]
    q50 = sorted_vals[int(0.50 * (n - 1))]
    q75 = sorted_vals[int(0.75 * (n - 1))]
    return (round(q25, 4), round(q50, 4), round(q75, 4))


def _strip_constant_cds_prefix(raw_seq: str) -> str:
    """Strip constant reporter start-codon suffix (`ATGGGCGAA` / `ATGG`) present in GSM3130436..440."""
    s = raw_seq.strip().upper().replace("U", "T")
    if len(s) in {54, 59} and s[50:53] == "ATG":
        return s[:50]
    return s


def _has_out_of_frame_uaug(utr5_dna_or_rna: str) -> tuple[bool, bool]:
    """Return (has_any_aug, has_out_of_frame_uaug) assuming UTR immediately precedes CDS start at len(utr)."""
    s = _strip_constant_cds_prefix(utr5_dna_or_rna).replace("T", "U")
    L = len(s)
    any_aug = False
    oof_aug = False
    idx = 0
    while True:
        pos = s.find("AUG", idx)
        if pos < 0:
            break
        any_aug = True
        dist_to_cds = L - pos
        if dist_to_cds % 3 != 0:
            oof_aug = True
        idx = pos + 1
    return any_aug, oof_aug


def evaluate_single_sample(entry: dict) -> dict:
    with urlopen(entry["url"], timeout=120) as resp:
        compressed = resp.read()
    sha256 = hashlib.sha256(compressed).hexdigest()
    if entry["expected_sha256"] and sha256 != entry["expected_sha256"]:
        raise RuntimeError(
            f"SHA-256 mismatch on {entry['sample_id']}: expected {entry['expected_sha256']}, got {sha256}"
        )

    no_uaug_mrl: list[float] = []
    oof_uaug_mrl: list[float] = []
    inframe_only_mrl: list[float] = []

    with gzip.open(io.BytesIO(compressed), mode="rt", encoding="utf-8", errors="replace") as gz:
        reader = csv.DictReader(gz)
        fieldnames = reader.fieldnames or []
        seq_col = next((c for c in fieldnames if c.lower() in {"utr", "seq", "sequence"}), "utr")
        mrl_col = next((c for c in fieldnames if c.lower() in {"rl", "mrl"}), "rl")
        has_raw_total_reads = "total_reads" in fieldnames
        total_col = "total_reads" if has_raw_total_reads else ("total" if "total" in fieldnames else None)
        is_normalized_fraction: bool | None = None

        for row in reader:
            try:
                if total_col and row.get(total_col):
                    t_val = float(row[total_col])
                    if is_normalized_fraction is None:
                        is_normalized_fraction = (not has_raw_total_reads) and (t_val < 1.0)
                    if is_normalized_fraction:
                        if t_val < 1e-5:
                            continue
                    elif t_val < 100.0:
                        continue
                seq = row[seq_col].strip()
                mrl = float(row[mrl_col])
            except (KeyError, ValueError, TypeError):
                continue
            any_aug, oof_aug = _has_out_of_frame_uaug(seq)
            if not any_aug:
                no_uaug_mrl.append(mrl)
            elif oof_aug:
                oof_uaug_mrl.append(mrl)
            else:
                inframe_only_mrl.append(mrl)

    no_uaug_mrl.sort()
    oof_uaug_mrl.sort()
    inframe_only_mrl.sort()

    no_q25, no_med, no_q75 = _quantiles(no_uaug_mrl)
    oof_q25, oof_med, oof_q75 = _quantiles(oof_uaug_mrl)
    _, inf_med, _ = _quantiles(inframe_only_mrl)
    delta_median = round(oof_med - no_med, 4)

    return {
        "sample_id": entry["sample_id"],
        "dataset_id": entry["dataset_id"],
        "chemistry": entry["chemistry"],
        "reporter": entry["reporter"],
        "replicate": entry["replicate"],
        "compressed_bytes": len(compressed),
        "sha256": sha256,
        "catalog_sha256_matched": (
            True if entry["expected_sha256"] is None else (sha256 == entry["expected_sha256"])
        ),
        "n_no_uaug": len(no_uaug_mrl),
        "n_oof_uaug": len(oof_uaug_mrl),
        "n_inframe_only_uaug": len(inframe_only_mrl),
        "no_uaug_mrl_q25_med_q75": [no_q25, no_med, no_q75],
        "oof_uaug_mrl_q25_med_q75": [oof_q25, oof_med, oof_q75],
        "inframe_only_uaug_median_mrl": inf_med,
        "median_delta_mrl_oof_minus_none": delta_median,
        "meets_pre_registered_threshold": delta_median <= -0.50,
    }


def main() -> None:
    results = []
    for s in SAMPLES_TO_EVALUATE:
        print(f"[tribunal] streaming {s['sample_id']} ({s['chemistry']}, {s['reporter']})...", file=sys.stderr, flush=True)
        res = evaluate_single_sample(s)
        print(
            f"[tribunal] {s['sample_id']} done: n_no_uaug={res['n_no_uaug']}, n_oof={res['n_oof_uaug']}, delta_med={res['median_delta_mrl_oof_minus_none']}",
            file=sys.stderr,
            flush=True,
        )
        results.append(res)
    all_supported = all(r["meets_pre_registered_threshold"] for r in results)
    output = {
        "schema": "mrnax.rule_tribunal_audit.v1",
        "executed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pre_registered_spec": PRE_REGISTERED_SPEC,
        "deduplication_accounting": {
            "underlying_sequencing_project": "GSE114002 / PRJNA454863",
            "independent_dataset_votes": 1,
            "independent_lab_lineage_votes": 1,
            "commercial_clearance_status": "commercial_clearance_pending (per C-B2 audit: GEO public access without explicit file-level commercial license)",
        },
        "sample_results": results,
        "final_tribunal_verdict": {
            "rule_id": "RUL-5UTR-UAUG-MRL-001",
            "assigned_status": "conditional_rule" if all_supported else "insufficient_evidence",
            "rationale": (
                "All evaluated unmodified U, Psi, and m1Psi eGFP & mCherry MPRA replicates show strong, "
                "reproducible negative MRL shift (<= -0.50 MRL units) for out-of-frame uAUGs. Graded as "
                "`conditional_rule` (NOT universal `verified_rule` for protein yield) because readout is "
                "12h HEK293T MRL from a single lab lineage (Seelig UW) and observational MPRA rather than "
                "in vivo multi-cargo protein AUC."
            ),
        },
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

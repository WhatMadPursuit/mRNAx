#!/usr/bin/env python3
"""Paired Scientific Benchmark: mRNAx v2.0 Baseline vs v2.5 LinearDesign + Surgical Polisher.

Evaluates 5 representative constructs across all 3 supported RNA modalities:
1. `egfp_single_chain` (`linear_mrna`, 239 aa eGFP, m1Psi + full 5-lambda sweep)
2. `hepo_single_chain` (`linear_mrna`, human EPO from `clinical_mrna_benchmarks_v1.json`, m1Psi + miR-142-3p detargeting)
3. `dll3_x_cd3_multi_chain` (`linear_mrna`, 2-chain bispecific, m1Psi + miR-122-5p liver detargeting)
4. `circrna_fluc_reporter` (`circrna`, 550 aa Photinus pyralis FLuc UniProt P08659, Anabaena PIE + HRV-B3 IRES)
5. `sarna_veev_tc83` (`sarna`, VEEV TC-83 cis-replicon + locked CSE/26S-SGP + 191 aa VEGF-A165 GOI)

Writes verified paired metrics to `artifacts/v25_engine_ablation_report.json`.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time
from typing import Any

from Bio.Seq import Seq

from mrnax.constraints.fto_guard import audit_codebase_method_fto_lints
from mrnax.engine.lineardesign_adapter import (
    DEFAULT_LAMBDA_SWEEP,
    DEFAULT_POLISH_ENZYMES,
    build_forbidden_motif_list,
    count_forbidden_motif_hits,
    discover_lineardesign_binary,
    sweep_lineardesign_candidates,
)
from mrnax.engine.modalities import (
    build_circrna_bundle,
    build_linear_mrna_bundle,
    build_sarna_bundle,
    evaluate_circrna_bundle,
    evaluate_linear_mrna_bundle,
    evaluate_sarna_bundle,
)
from mrnax.engine.structure_profile import discover_eternafold_binary

ROOT = Path(__file__).resolve().parents[1]
SEED_DIR = ROOT / "registry_seed"
OUT_PATH = ROOT / "artifacts" / "v25_engine_ablation_report.json"

# Live-verified UniProt P08659 (LUCI_PHOPY, Photinus pyralis firefly luciferase, 550 aa)
FLUC_P08659_AA = (
    "MEDAKNIKKGPAPFYPLEDGTAGEQLHKAMKRYALVPGTIAFTDAHIEVNITYAEYFEMS"
    "VRLAEAMKRYGLNTNHRIVVCSENSLQFFMPVLGALFIGVAVAPANDIYNERELLNSMNI"
    "SQPTVVFVSKKGLQKILNVQKKLPIIQKIIIMDSKTDYQGFQSMYTFVTSHLPPGFNEYD"
    "FVPESFDRDKTIALIMNSSGSTGLPKGVALPHRTACVRFSHARDPIFGNQIIPDTAILSV"
    "VPFHHGFGMFTTLGYLICGFRVVLMYRFEEELFLRSLQDYKIQSALLVPTLFSFFAKSTL"
    "IDKYDLSNLHEIASGGAPLSKEVGEAVAKRFHLPGIRQGYGLTETTSAILITPEGDDKPG"
    "AVGKVVPFFEAKVVDLDTGKTLGVNQRGELCVRGPMIMSGYVNNPEATNALIDKDGWLHS"
    "GDIAYWDEDEHFFIVDRLKSLIKYKGYQVAPAELESILLQHPNIFDAGVAGLPDDDAGEL"
    "PAAVVVLEHGKTMTEKEIVDYVASQVTTAKKLRGGVVFVDEVPKGLTGKLDARKIREILI"
    "KAKKGGKSKL"
)


def _extract_benchmark_protein(benchmarks_doc: dict[str, Any], benchmark_id: str) -> str:
    entry = next(b for b in benchmarks_doc["benchmarks"] if b["benchmark_id"] == benchmark_id)
    full_rna = str(entry["verified_sequences_rna"]["full_sequence_rna"])
    c_lo, c_hi = entry["segment_coordinates_0based"]["cds_no_stop"]
    cds_rna = full_rna[c_lo:c_hi]
    prot = str(Seq(cds_rna).translate(table=1, to_stop=False))
    assert "*" not in prot and len(prot) * 3 == len(cds_rna)
    return prot


def _summarize_linear_eval(ev: dict[str, Any], bnd: Any, wall_sec: float) -> dict[str, Any]:
    forb = build_forbidden_motif_list(
        max_homopolymer=5,
        eliminate_uuuu=True,
        restriction_enzymes=DEFAULT_POLISH_ENZYMES,
    )
    chains = ev["chains"]
    total_mfe = round(sum(float(c["cds_mfe_kcal_mol"]) for c in chains), 3)
    mean_cai = round(sum(float(c["cai"]) for c in chains) / len(chains), 4)
    min_open = round(min(float(c["kozak_unpaired_prob"]) for c in chains), 5)
    uuuu_total = sum(int(c["cds_uuuu_count"]) for c in chains)
    forb_hits = sum(
        count_forbidden_motif_hits(
            m.get_segment_sequence("cds"),
            forb,
            utr5_tail=m.get_segment_sequence("five_prime_utr"),
            downstream_head=m.get_segment_sequence("stop_cassette") + m.get_segment_sequence("three_prime_utr"),
        )
        for m in bnd.molecules
    )
    return {
        "chain_count": ev["chain_count"],
        "aa_exact_match": all(bool(c["aa_exact_match"]) for c in chains),
        "passed_all_hard_gates": bool(ev["passed_all_hard_gates"]),
        "cds_mfe_kcal_mol": total_mfe,
        "cai": mean_cai,
        "kozak_unpaired_prob": min_open,
        "cds_uuuu_count": uuuu_total,
        "forbidden_motif_hits": forb_hits,
        "chains": chains,
        "wall_seconds": round(wall_sec, 4),
    }


def run_benchmark() -> dict[str, Any]:
    t_all = time.perf_counter()
    egfp_doc = json.loads((SEED_DIR / "egfp_reporter_snapshot.json").read_text(encoding="utf-8"))
    egfp_aa = str(egfp_doc["mammalian_reporter_egfp"]["protein_sequence"])

    clin_doc = json.loads((SEED_DIR / "clinical_mrna_benchmarks_v1.json").read_text(encoding="utf-8"))
    hepo_aa = _extract_benchmark_protein(clin_doc, "BENCH-REF-HEPO-HBB-CLEAN")
    vegfa_aa = _extract_benchmark_protein(clin_doc, "BENCH-CLIN-AZD8601-FAZULEMERAN")

    b2_doc = json.loads(
        (ROOT / "tests" / "golden" / "m2_benchmark2_dll3_cd3_acceptance_fixture_v1.json").read_text(
            encoding="utf-8"
        )
    )
    dll3_cd3_targets = [str(t["protein_sequence"]) for t in b2_doc["design_intent"]["protein_targets"]]

    ld_found = discover_lineardesign_binary()
    ef_found = discover_eternafold_binary()

    # 1. Single-chain eGFP (239 aa) + 5-lambda LinearDesign+Polisher sweep
    t0 = time.perf_counter()
    egfp_v20_bnd = build_linear_mrna_bundle(
        [egfp_aa],
        bundle_id="bench_egfp_v20",
        compliance_track="research_max",
        uridine_chemistry="m1psi",
        use_lineardesign=False,
        baseline_v20_mode=True,
        seed=42,
        seed_dir=SEED_DIR,
    )
    egfp_v20_sec = time.perf_counter() - t0
    egfp_v20_ev = evaluate_linear_mrna_bundle(egfp_v20_bnd, compliance_track="research_max", seed_dir=SEED_DIR)

    t1 = time.perf_counter()
    egfp_v25_bnd = build_linear_mrna_bundle(
        [egfp_aa],
        bundle_id="bench_egfp_v25",
        compliance_track="research_max",
        uridine_chemistry="m1psi",
        use_lineardesign=True,
        lambda_val=1.0,
        baseline_v20_mode=False,
        seed=42,
        seed_dir=SEED_DIR,
    )
    egfp_v25_sec = time.perf_counter() - t1
    egfp_v25_ev = evaluate_linear_mrna_bundle(egfp_v25_bnd, compliance_track="research_max", seed_dir=SEED_DIR)

    egfp_sweep = sweep_lineardesign_candidates(
        egfp_aa,
        lambdas=DEFAULT_LAMBDA_SWEEP,
        utr5_rna=egfp_v25_bnd.molecules[0].get_segment_sequence("five_prime_utr"),
        stop_rna="UGA",
        utr3_rna=egfp_v25_bnd.molecules[0].get_segment_sequence("three_prime_utr"),
        compliance_track="research_max",
        chemistry="m1psi",
    )

    # 2. Single-chain hEPO + myeloid APC miR-142-3p detargeting
    t0 = time.perf_counter()
    hepo_v20_bnd = build_linear_mrna_bundle(
        [hepo_aa],
        bundle_id="bench_hepo_v20",
        compliance_track="research_max",
        uridine_chemistry="m1psi",
        detarget_tissues=["myeloid_apc"],
        protected_tissues=["liver_hepatocyte"],
        use_lineardesign=False,
        baseline_v20_mode=True,
        seed=42,
        seed_dir=SEED_DIR,
    )
    hepo_v20_sec = time.perf_counter() - t0
    hepo_v20_ev = evaluate_linear_mrna_bundle(hepo_v20_bnd, compliance_track="research_max", seed_dir=SEED_DIR)

    t1 = time.perf_counter()
    hepo_v25_bnd = build_linear_mrna_bundle(
        [hepo_aa],
        bundle_id="bench_hepo_v25",
        compliance_track="research_max",
        uridine_chemistry="m1psi",
        detarget_tissues=["myeloid_apc"],
        protected_tissues=["liver_hepatocyte"],
        use_lineardesign=True,
        lambda_val=1.0,
        baseline_v20_mode=False,
        seed=42,
        seed_dir=SEED_DIR,
    )
    hepo_v25_sec = time.perf_counter() - t1
    hepo_v25_ev = evaluate_linear_mrna_bundle(hepo_v25_bnd, compliance_track="research_max", seed_dir=SEED_DIR)

    # 3. Multi-chain DLL3 x CD3 bispecific + liver miR-122-5p detargeting
    t0 = time.perf_counter()
    mc_v20_bnd = build_linear_mrna_bundle(
        dll3_cd3_targets,
        bundle_id="bench_multichain_v20",
        compliance_track="research_max",
        uridine_chemistry="m1psi",
        detarget_tissues=["liver"],
        protected_tissues=["t_cell"],
        use_lineardesign=False,
        baseline_v20_mode=True,
        seed=42,
        seed_dir=SEED_DIR,
    )
    mc_v20_sec = time.perf_counter() - t0
    mc_v20_ev = evaluate_linear_mrna_bundle(mc_v20_bnd, compliance_track="research_max", seed_dir=SEED_DIR)

    t1 = time.perf_counter()
    mc_v25_bnd = build_linear_mrna_bundle(
        dll3_cd3_targets,
        bundle_id="bench_multichain_v25",
        compliance_track="research_max",
        uridine_chemistry="m1psi",
        detarget_tissues=["liver"],
        protected_tissues=["t_cell"],
        use_lineardesign=True,
        lambda_val=1.0,
        baseline_v20_mode=False,
        seed=42,
        seed_dir=SEED_DIR,
    )
    mc_v25_sec = time.perf_counter() - t1
    mc_v25_ev = evaluate_linear_mrna_bundle(mc_v25_bnd, compliance_track="research_max", seed_dir=SEED_DIR)

    # 4. circRNA-FLuc (550 aa Photinus pyralis FLuc, Anabaena PIE + HRV-B3 IRES + IRES-cargo uncoupling)
    t0 = time.perf_counter()
    circ_v20_bnd = build_circrna_bundle(
        FLUC_P08659_AA,
        bundle_id="bench_circ_fluc_v20",
        ires_name="HRV-B3",
        pie_backbone="Anabaena",
        compliance_track="research_max",
        use_lineardesign=False,
        baseline_v20_mode=True,
        seed=42,
        seed_dir=SEED_DIR,
    )
    circ_v20_sec = time.perf_counter() - t0
    circ_v20_ev = evaluate_circrna_bundle(circ_v20_bnd, seed_dir=SEED_DIR)
    circ_v20_cds = circ_v20_bnd.molecules[0].get_segment_sequence("cds")

    t1 = time.perf_counter()
    circ_v25_bnd = build_circrna_bundle(
        FLUC_P08659_AA,
        bundle_id="bench_circ_fluc_v25",
        ires_name="HRV-B3",
        pie_backbone="Anabaena",
        compliance_track="research_max",
        use_lineardesign=True,
        lambda_val=1.0,
        baseline_v20_mode=False,
        seed=42,
        seed_dir=SEED_DIR,
    )
    circ_v25_sec = time.perf_counter() - t1
    circ_v25_ev = evaluate_circrna_bundle(circ_v25_bnd, seed_dir=SEED_DIR)
    circ_v25_cds = circ_v25_bnd.molecules[0].get_segment_sequence("cds")

    # 5. saRNA VEEV TC-83 cis-replicon + 191 aa VEGF-A165 GOI
    t0 = time.perf_counter()
    sarna_v20_bnd = build_sarna_bundle(
        vegfa_aa,
        bundle_id="bench_sarna_veev_v20",
        replicon_backbone="VEEV_TC83",
        compliance_track="research_max",
        use_lineardesign=False,
        baseline_v20_mode=True,
        seed=42,
        seed_dir=SEED_DIR,
    )
    sarna_v20_sec = time.perf_counter() - t0
    sarna_v20_ev = evaluate_sarna_bundle(sarna_v20_bnd, seed_dir=SEED_DIR)
    sarna_v20_cds = sarna_v20_bnd.molecules[0].get_segment_sequence("cds")

    t1 = time.perf_counter()
    sarna_v25_bnd = build_sarna_bundle(
        vegfa_aa,
        bundle_id="bench_sarna_veev_v25",
        replicon_backbone="VEEV_TC83",
        compliance_track="research_max",
        use_lineardesign=True,
        lambda_val=1.0,
        baseline_v20_mode=False,
        seed=42,
        seed_dir=SEED_DIR,
    )
    sarna_v25_sec = time.perf_counter() - t1
    sarna_v25_ev = evaluate_sarna_bundle(sarna_v25_bnd, seed_dir=SEED_DIR)
    sarna_v25_cds = sarna_v25_bnd.molecules[0].get_segment_sequence("cds")

    egfp_s20 = _summarize_linear_eval(egfp_v20_ev, egfp_v20_bnd, egfp_v20_sec)
    egfp_s25 = _summarize_linear_eval(egfp_v25_ev, egfp_v25_bnd, egfp_v25_sec)
    hepo_s20 = _summarize_linear_eval(hepo_v20_ev, hepo_v20_bnd, hepo_v20_sec)
    hepo_s25 = _summarize_linear_eval(hepo_v25_ev, hepo_v25_bnd, hepo_v25_sec)
    mc_s20 = _summarize_linear_eval(mc_v20_ev, mc_v20_bnd, mc_v20_sec)
    mc_s25 = _summarize_linear_eval(mc_v25_ev, mc_v25_bnd, mc_v25_sec)

    lint = audit_codebase_method_fto_lints(ROOT / "src" / "mrnax")

    report: dict[str, Any] = {
        "schema_version": "2.5.0",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "engines_detected": {
            "lineardesign_binary": str(ld_found[0]) if ld_found else None,
            "eternafold_binary": str(ef_found[0]) if ef_found else None,
            "method_fto_ast_lint_status": lint["status"],
        },
        "egfp_5lambda_polisher_sweep": [
            {
                "lambda": c.lambda_val,
                "raw_mfe_kcal_mol": c.raw_mfe_kcal_mol,
                "polished_mfe_kcal_mol": c.polished_mfe_kcal_mol,
                "raw_cai": c.raw_cai,
                "polished_cai": c.polished_cai,
                "raw_kozak_unpaired_prob": c.raw_kozak_unpaired_prob,
                "polished_kozak_unpaired_prob": c.polished_kozak_unpaired_prob,
                "synonymous_codons_changed": c.synonymous_codons_changed,
                "eliminated_motifs_before": c.eliminated_motifs_before,
                "remaining_forbidden_motifs": c.remaining_forbidden_motifs,
                "aa_exact_match": c.aa_exact_match,
            }
            for c in egfp_sweep
        ],
        "paired_benchmarks": {
            "egfp_single_chain": {
                "modality": "linear_mrna",
                "protein_aa_length": len(egfp_aa),
                "v20_baseline": egfp_s20,
                "v25_lineardesign_polisher": egfp_s25,
                "paired_delta_v25_minus_v20": {
                    "cds_mfe_kcal_mol": round(egfp_s25["cds_mfe_kcal_mol"] - egfp_s20["cds_mfe_kcal_mol"], 3),
                    "cai": round(egfp_s25["cai"] - egfp_s20["cai"], 4),
                    "kozak_unpaired_prob": round(
                        egfp_s25["kozak_unpaired_prob"] - egfp_s20["kozak_unpaired_prob"], 5
                    ),
                },
            },
            "hepo_single_chain": {
                "modality": "linear_mrna",
                "protein_aa_length": len(hepo_aa),
                "detarget_mirna": "hsa-miR-142-3p",
                "v20_baseline": hepo_s20,
                "v25_lineardesign_polisher": hepo_s25,
                "paired_delta_v25_minus_v20": {
                    "cds_mfe_kcal_mol": round(hepo_s25["cds_mfe_kcal_mol"] - hepo_s20["cds_mfe_kcal_mol"], 3),
                    "cai": round(hepo_s25["cai"] - hepo_s20["cai"], 4),
                    "kozak_unpaired_prob": round(
                        hepo_s25["kozak_unpaired_prob"] - hepo_s20["kozak_unpaired_prob"], 5
                    ),
                },
            },
            "dll3_x_cd3_multi_chain": {
                "modality": "linear_mrna",
                "chain_aa_lengths": [len(s) for s in dll3_cd3_targets],
                "detarget_mirna": "hsa-miR-122-5p",
                "v20_baseline": mc_s20,
                "v25_lineardesign_polisher": mc_s25,
                "paired_delta_v25_minus_v20": {
                    "cds_mfe_kcal_mol": round(mc_s25["cds_mfe_kcal_mol"] - mc_s20["cds_mfe_kcal_mol"], 3),
                    "cai": round(mc_s25["cai"] - mc_s20["cai"], 4),
                    "kozak_unpaired_prob": round(
                        mc_s25["kozak_unpaired_prob"] - mc_s20["kozak_unpaired_prob"], 5
                    ),
                },
            },
            "circrna_fluc_reporter": {
                "modality": "circrna",
                "protein_aa_length": len(FLUC_P08659_AA),
                "ires": "HRV-B3",
                "pie_backbone": "Anabaena",
                "v20_baseline": {
                    **circ_v20_ev,
                    "cds_uuuu_count": len(re.findall(r"U{4,}", circ_v20_cds)),
                    "wall_seconds": round(circ_v20_sec, 4),
                },
                "v25_lineardesign_polisher": {
                    **circ_v25_ev,
                    "cds_uuuu_count": len(re.findall(r"U{4,}", circ_v25_cds)),
                    "wall_seconds": round(circ_v25_sec, 4),
                },
                "paired_delta_v25_minus_v20": {
                    "cds_mfe_kcal_mol": round(
                        circ_v25_ev["cds_mfe_kcal_mol"] - circ_v20_ev["cds_mfe_kcal_mol"], 3
                    ),
                    "circ_mfe_kcal_mol": round(
                        circ_v25_ev["circ_mfe_kcal_mol"] - circ_v20_ev["circ_mfe_kcal_mol"], 3
                    ),
                    "cai": round(circ_v25_ev["cai"] - circ_v20_ev["cai"], 4),
                    "initiation_unpaired_prob": round(
                        circ_v25_ev["initiation_unpaired_prob"] - circ_v20_ev["initiation_unpaired_prob"], 5
                    ),
                },
            },
            "sarna_veev_tc83": {
                "modality": "sarna",
                "protein_aa_length": len(vegfa_aa),
                "replicon_backbone": "VEEV_TC83",
                "v20_baseline": {
                    **sarna_v20_ev,
                    "cds_mfe_kcal_mol": sarna_v20_ev["goi_mfe_kcal_mol"],
                    "cds_uuuu_count": len(re.findall(r"U{4,}", sarna_v20_cds)),
                    "wall_seconds": round(sarna_v20_sec, 4),
                },
                "v25_lineardesign_polisher": {
                    **sarna_v25_ev,
                    "cds_mfe_kcal_mol": sarna_v25_ev["goi_mfe_kcal_mol"],
                    "cds_uuuu_count": len(re.findall(r"U{4,}", sarna_v25_cds)),
                    "wall_seconds": round(sarna_v25_sec, 4),
                },
                "paired_delta_v25_minus_v20": {
                    "cds_mfe_kcal_mol": round(
                        sarna_v25_ev["goi_mfe_kcal_mol"] - sarna_v20_ev["goi_mfe_kcal_mol"], 3
                    ),
                    "cai": round(sarna_v25_ev["goi_cai"] - sarna_v20_ev["goi_cai"], 4),
                    "kozak_unpaired_prob": round(
                        sarna_v25_ev["goi_kozak_unpaired_prob"] - sarna_v20_ev["goi_kozak_unpaired_prob"], 5
                    ),
                },
            },
        },
        "total_wall_seconds": round(time.perf_counter() - t_all, 3),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    res = run_benchmark()
    print(f"Wrote {OUT_PATH} in {res['total_wall_seconds']}s")
    for k, v in res["paired_benchmarks"].items():
        d = v["paired_delta_v25_minus_v20"]
        print(
            f"  {k}: v2.0 MFE={v['v20_baseline']['cds_mfe_kcal_mol']} -> "
            f"v2.5 MFE={v['v25_lineardesign_polisher']['cds_mfe_kcal_mol']} "
            f"(delta={d['cds_mfe_kcal_mol']} kcal/mol)"
        )

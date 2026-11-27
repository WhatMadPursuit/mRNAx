"""Gate G7: Paired 5-Seed Equal-Budget 3-Arm Generator Inner-Loop Ablation (1-Chain EGFP + 2-Chain DLL3xCD3).

Compares under identical `CommercialCleanSurrogateV1` evaluation and identical per-seed candidate/trial budgets:
  1. `arm1_stage0_baseline`: Stage 0 (`stage0_random` + `dnachisel_cds`, no `pareto_nsga2`)
  2. `arm2_legacy_generator_pure_rerank`: Legacy unguided generator + Surrogate v2 pure rerank (`honesty_banner = "本次未改善生成器"`)
  3. `arm3_upgraded_inner_loop_generator`: Non-partitioned full-transcript bicodon/CSC + 200x200 UTR + ATtRACT 160-RBP + `gc_tail15` inner-loop guided generator + Surrogate v2
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

from mrnax.cloud.gateway import _build_preset_reference_r
from mrnax.engine.pareto_director import run_design_pipeline
from mrnax.engine.stage0_baseline import _load_bicodon_atlas
from mrnax.engine.surrogates import CommercialCleanSurrogateV1
from mrnax.harness.state_machine import ProjectHarnessSession
from mrnax.schemas.construct_bundle import ConstructBundle
from mrnax.schemas.design_intent import DesignIntent

REPO_ROOT = Path(__file__).resolve().parents[1]
SEED_DIR = REPO_ROOT / "registry_seed"
SEEDS = [42, 101, 2026, 777, 9001]


def _mean_bicodon_score(cds_rna: str, csc_map: dict[str, float], bic_map: dict[str, float]) -> float:
    codons = [cds_rna[i : i + 3] for i in range(0, len(cds_rna), 3)]
    if len(codons) < 2:
        return 0.0
    csc_mean = sum(csc_map.get(c, 0.0) for c in codons[1:]) / max(1, len(codons) - 1)
    bic_mean = sum(bic_map.get(f"{codons[i]}-{codons[i + 1]}", 0.0) for i in range(len(codons) - 1)) / (
        len(codons) - 1
    )
    return round(csc_mean + 0.25 * bic_mean, 6)


def _score_bundle_multi_axis(
    bundle: ConstructBundle,
    surrogate: CommercialCleanSurrogateV1,
    csc_map: dict[str, float],
    bic_map: dict[str, float],
) -> dict[str, float]:
    mrls: list[float] = []
    cds_stabs: list[float] = []
    u3_stabs: list[float] = []
    hls: list[float] = []
    atlas_scores: list[float] = []
    are_counts: list[int] = []
    for mol in bundle.molecules:
        u5 = mol.get_segment_sequence("five_prime_utr")
        cds = mol.get_segment_sequence("cds")
        u3 = mol.get_segment_sequence("three_prime_utr")
        mrls.append(float(surrogate.score_5utr_mrl([u5])[0]))
        cds_stabs.append(float(surrogate.score_cds_stability([cds])[0]))
        u3_stabs.append(float(surrogate.score_3utr_stability([u3])[0]))
        hls.append(float(surrogate.score_consensus_halflife(u5, cds, u3)))
        atlas_scores.append(_mean_bicodon_score(cds, csc_map, bic_map))
        are_counts.append(u3.count("AUUUA"))
    # Use worst-chain (min) for multi-chain bottleneck honesty + mean across chains
    return {
        "worst_chain_utr5_mrl": round(min(mrls), 5),
        "worst_chain_cds_stability": round(min(cds_stabs), 5),
        "worst_chain_utr3_stability": round(min(u3_stabs), 5),
        "worst_chain_halflife": round(min(hls), 5),
        "mean_chain_halflife": round(sum(hls) / len(hls), 5),
        "mean_chain_cds_stability": round(sum(cds_stabs) / len(cds_stabs), 5),
        "mean_bicodon_atlas_index": round(sum(atlas_scores) / len(atlas_scores), 5),
        "total_utr3_are_motifs": float(sum(are_counts)),
    }


def _evaluate_benchmark_across_seeds(
    benchmark_id: str,
    base_intent: DesignIntent,
    surrogate: CommercialCleanSurrogateV1,
    csc_map: dict[str, float],
    bic_map: dict[str, float],
) -> dict[str, Any]:
    seed_rows: list[dict[str, Any]] = []
    for sd in SEEDS:
        # Arm 1: Stage 0 only
        pol_s0 = base_intent.runtime_policy.model_copy_validated(
            random_seed=sd,
            enabled_generators=["stage0_random", "dnachisel_cds"],
            enabled_evaluators=["surrogate_v1_clean"],
            max_evaluations=10,
        )
        t0 = time.perf_counter()
        b_s0, rep_s0, man_s0 = run_design_pipeline(
            base_intent.model_copy_validated(runtime_policy=pol_s0),
            seed_dir=SEED_DIR,
        )
        dt_s0 = round(time.perf_counter() - t0, 4)

        # Arm 2: Legacy unguided generator + Surrogate v2 Pure Rerank ("本次未改善生成器")
        pol_pareto = base_intent.runtime_policy.model_copy_validated(
            random_seed=sd,
            enabled_generators=["stage0_random", "dnachisel_cds", "pareto_nsga2"],
            enabled_evaluators=["surrogate_v1_clean"],
            max_evaluations=10,
        )
        t1 = time.perf_counter()
        b_legacy, rep_legacy, man_legacy = run_design_pipeline(
            base_intent.model_copy_validated(runtime_policy=pol_pareto),
            seed_dir=SEED_DIR,
            legacy_pure_rerank_mode=True,
        )
        dt_legacy = round(time.perf_counter() - t1, 4)
        assert man_legacy.honesty_banner == "本次未改善生成器"

        # Arm 3: Upgraded non-partitioned inner-loop guided generator + Surrogate v2
        t2 = time.perf_counter()
        b_guided, rep_guided, man_guided = run_design_pipeline(
            base_intent.model_copy_validated(runtime_policy=pol_pareto),
            seed_dir=SEED_DIR,
            legacy_pure_rerank_mode=False,
        )
        dt_guided = round(time.perf_counter() - t2, 4)
        assert man_guided.honesty_banner is None
        assert man_guided.optimization_mode_audit.optimization_mode == "full_transcript_non_partitioned_pareto"
        assert man_guided.optimization_mode_audit.uses_two_zone_initiation_body_partition is False

        attempts_legacy = sum(c.attempt_count for c in man_legacy.invoked_components if c.role == "generator")
        attempts_guided = sum(c.attempt_count for c in man_guided.invoked_components if c.role == "generator")
        if attempts_legacy != attempts_guided:
            raise AssertionError(
                f"Equal-budget violation on {benchmark_id} seed={sd}: {attempts_legacy} != {attempts_guided}"
            )
        assert not rep_s0.fto_alerts and not rep_legacy.fto_alerts and not rep_guided.fto_alerts

        def _top_metrics(bundles: list[ConstructBundle], rec_ids: list[str]) -> dict[str, float]:
            by_id = {b.bundle_id: b for b in bundles}
            top_b = by_id[rec_ids[0]] if rec_ids and rec_ids[0] in by_id else bundles[0]
            scored_all = [_score_bundle_multi_axis(b, surrogate, csc_map, bic_map) for b in bundles]
            top_m = _score_bundle_multi_axis(top_b, surrogate, csc_map, bic_map)
            return {
                **top_m,
                "pool_best_halflife": round(max(m["worst_chain_halflife"] for m in scored_all), 5),
                "pool_best_cds_stability": round(max(m["worst_chain_cds_stability"] for m in scored_all), 5),
                "pool_best_utr5_mrl": round(max(m["worst_chain_utr5_mrl"] for m in scored_all), 5),
                "qualified_count": float(len(bundles)),
            }

        m1 = _top_metrics(b_s0, rep_s0.recommended_bundle_ids)
        m2 = _top_metrics(b_legacy, rep_legacy.recommended_bundle_ids)
        m3 = _top_metrics(b_guided, rep_guided.recommended_bundle_ids)
        seed_rows.append(
            {
                "seed": sd,
                "equal_generator_attempt_budget": attempts_guided,
                "arm1_stage0_baseline": {**m1, "wall_sec": dt_s0},
                "arm2_legacy_generator_pure_rerank": {
                    **m2,
                    "honesty_banner": man_legacy.honesty_banner,
                    "wall_sec": dt_legacy,
                },
                "arm3_upgraded_inner_loop_generator": {
                    **m3,
                    "optimization_mode": man_guided.optimization_mode_audit.optimization_mode,
                    "wall_sec": dt_guided,
                },
                "paired_delta_arm3_minus_arm2": {
                    "worst_chain_halflife": round(m3["worst_chain_halflife"] - m2["worst_chain_halflife"], 5),
                    "worst_chain_cds_stability": round(
                        m3["worst_chain_cds_stability"] - m2["worst_chain_cds_stability"], 5
                    ),
                    "worst_chain_utr5_mrl": round(m3["worst_chain_utr5_mrl"] - m2["worst_chain_utr5_mrl"], 5),
                    "pool_best_halflife": round(m3["pool_best_halflife"] - m2["pool_best_halflife"], 5),
                    "pool_best_cds_stability": round(
                        m3["pool_best_cds_stability"] - m2["pool_best_cds_stability"], 5
                    ),
                    "mean_bicodon_atlas_index": round(
                        m3["mean_bicodon_atlas_index"] - m2["mean_bicodon_atlas_index"], 5
                    ),
                },
            }
        )

    def _agg(arm_key: str, metric: str) -> float:
        return round(statistics.mean(r[arm_key][metric] for r in seed_rows), 5)

    deltas_hl = [r["paired_delta_arm3_minus_arm2"]["pool_best_halflife"] for r in seed_rows]
    deltas_cds = [r["paired_delta_arm3_minus_arm2"]["pool_best_cds_stability"] for r in seed_rows]
    deltas_bic = [r["paired_delta_arm3_minus_arm2"]["mean_bicodon_atlas_index"] for r in seed_rows]
    return {
        "benchmark_id": benchmark_id,
        "n_seeds": len(SEEDS),
        "seeds": SEEDS,
        "n_chains": len(base_intent.protein_targets),
        "summary_means": {
            "arm1_stage0_pool_best_halflife": _agg("arm1_stage0_baseline", "pool_best_halflife"),
            "arm2_pure_rerank_pool_best_halflife": _agg("arm2_legacy_generator_pure_rerank", "pool_best_halflife"),
            "arm3_guided_generator_pool_best_halflife": _agg(
                "arm3_upgraded_inner_loop_generator", "pool_best_halflife"
            ),
            "arm1_stage0_pool_best_cds_stability": _agg("arm1_stage0_baseline", "pool_best_cds_stability"),
            "arm2_pure_rerank_pool_best_cds_stability": _agg(
                "arm2_legacy_generator_pure_rerank", "pool_best_cds_stability"
            ),
            "arm3_guided_generator_pool_best_cds_stability": _agg(
                "arm3_upgraded_inner_loop_generator", "pool_best_cds_stability"
            ),
            "arm2_pure_rerank_bicodon_atlas_index": _agg(
                "arm2_legacy_generator_pure_rerank", "mean_bicodon_atlas_index"
            ),
            "arm3_guided_generator_bicodon_atlas_index": _agg(
                "arm3_upgraded_inner_loop_generator", "mean_bicodon_atlas_index"
            ),
            "mean_paired_delta_halflife_arm3_vs_arm2": round(statistics.mean(deltas_hl), 5),
            "mean_paired_delta_cds_stability_arm3_vs_arm2": round(statistics.mean(deltas_cds), 5),
            "mean_paired_delta_bicodon_index_arm3_vs_arm2": round(statistics.mean(deltas_bic), 5),
            "win_rate_halflife_arm3_vs_arm2": f"{sum(d > 0 for d in deltas_hl)}/{len(SEEDS)}",
            "win_rate_cds_stability_arm3_vs_arm2": f"{sum(d > 0 for d in deltas_cds)}/{len(SEEDS)}",
        },
        "per_seed": seed_rows,
    }


def run_ablation_and_write_report(out_path: Path | None = None) -> dict[str, Any]:
    import tempfile

    surrogate = CommercialCleanSurrogateV1.from_seed_dir(SEED_DIR)
    csc_map, bic_map = _load_bicodon_atlas()

    with tempfile.TemporaryDirectory() as td:
        ref = _build_preset_reference_r(SEED_DIR)
        sess = ProjectHarnessSession(project_id="g7_1chain", store_dir=Path(td) / "p1", seed_dir=SEED_DIR)
        egfp_intent = sess.create_initial_intent(
            turn_id="t1",
            user_utterance="1-chain EGFP m1Psi commercial_clean BspQI liver detargeting",
            protein_sequence=ref.molecules[0].effective_translation_target.protein_sequence,
            compliance_track="commercial_clean",
            uridine_chemistry="m1psi",
            restriction_enzymes=["BspQI"],
            detarget_tissues=["liver"],
            protected_tissues=["t_cell"],
            use_commercial_surrogate=True,
        )

    b2_fixture = json.loads(
        (REPO_ROOT / "tests" / "golden" / "m2_benchmark2_dll3_cd3_acceptance_fixture_v1.json").read_text(
            encoding="utf-8"
        )
    )
    raw_b2_intent = DesignIntent.model_validate(b2_fixture["design_intent"])
    assert raw_b2_intent.cell_expression_control is not None
    dll3_cd3_intent = raw_b2_intent.model_copy_validated(
        cell_expression_control=raw_b2_intent.cell_expression_control.model_copy_validated(
            protected_tissues=["t_cell", "myeloid_monocyte_macrophage"]
        )
    )

    res_1chain = _evaluate_benchmark_across_seeds("1chain_egfp_m1psi_detarget", egfp_intent, surrogate, csc_map, bic_map)
    res_2chain = _evaluate_benchmark_across_seeds(
        "2chain_dll3_cd3_bispecific_m1psi_detarget", dll3_cd3_intent, surrogate, csc_map, bic_map
    )

    report_doc = {
        "schema_version": "m2_g7_paired_ablation_v1",
        "generated_at_utc": "2026-11-27T02:45:00Z",
        "compliance_track": "commercial_clean",
        "honesty_statement": (
            "Arm 2 (`arm2_legacy_generator_pure_rerank`) leaves candidate generation unguided and applies "
            "`CommercialCleanSurrogateV1` strictly post-hoc (`本次未改善生成器`). Arm 3 (`arm3_upgraded_inner_loop_generator`) "
            "upgrades the inner-loop proposal & selection operators (`human_codon_bicodon_stability_atlas_v1.json` 61 CSC + "
            "3,721 forward/backward bicodons, `utr_components_commercial_v2.json` 200x200 library, "
            "`rbp_human_motifs_attract_ccby4_v1.json` 160 human RBPs, `gc_tail15`, and secondary-chain inner-loop co-optimization) "
            "under an identical per-seed generator attempt budget (`uses_two_zone_initiation_body_partition=False`)."
        ),
        "benchmarks": {
            "single_chain_egfp": res_1chain,
            "two_chain_dll3_x_cd3": res_2chain,
        },
    }
    target = out_path or (REPO_ROOT / "artifacts" / "m2_generator_upgrade" / "paired_ablation_report.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report_doc


if __name__ == "__main__":
    doc = run_ablation_and_write_report()
    print(json.dumps({k: v["summary_means"] for k, v in doc["benchmarks"].items()}, indent=2, ensure_ascii=False))

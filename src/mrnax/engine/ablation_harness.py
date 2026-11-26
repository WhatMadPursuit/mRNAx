"""Role-separated ablation & attribution harnesses (Joint Ruling #4).

1. `run_fixed_pool_scorer_replay`: Pure scorer/ranker 3-group replay (Baseline / Display-Only / Rank-Participating).
2. `run_paired_seed_generator_ablation`: Paired-seed on/off ablation for search-guiding evaluators.
3. `run_three_way_upstream_attribution`: Equal-total-budget 3-group attribution when an upstream tool supplies seeds.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from mrnax.schemas.construct_bundle import ConstructBundle
from mrnax.schemas.design_intent import DesignIntent
from mrnax.schemas.evaluation_report import BundleEvaluation, MetricResult


@dataclass(frozen=True)
class FixedPoolScorerReplayResult:
    scorer_name: str
    baseline_ordered_ids: list[str]
    display_only_ordered_ids: list[str]
    rank_participating_ordered_ids: list[str]
    order_changed_in_group3: bool
    display_only_evaluations: list[BundleEvaluation]
    honesty_banner: str = "本次未改善生成器"


def run_fixed_pool_scorer_replay(
    *,
    candidate_pool: list[ConstructBundle],
    base_evaluations: list[BundleEvaluation],
    scorer_name: str,
    scorer_fn: Callable[[ConstructBundle], float],
) -> FixedPoolScorerReplayResult:
    """Replay a fixed candidate pool across 3 modes with deterministic tie-breaking."""
    eval_by_id = {e.bundle_id: e for e in base_evaluations}
    for b in candidate_pool:
        if b.bundle_id not in eval_by_id:
            raise ValueError(f"Missing base BundleEvaluation for candidate '{b.bundle_id}'")
        ev = eval_by_id[b.bundle_id]
        if ev.version_digest != b.version_digest:
            raise ValueError(
                f"Evaluation version_digest mismatch for '{b.bundle_id}': {ev.version_digest} != {b.version_digest}"
            )
        if not ev.passed_all_hard_gates:
            raise ValueError(
                f"Disqualified candidate '{b.bundle_id}' cannot enter fixed-pool ranking replay"
            )
    input_index = {b.bundle_id: idx for idx, b in enumerate(candidate_pool)}

    # Group 1: Baseline order (pareto_rank asc, original pool index asc)
    baseline_sorted = sorted(
        candidate_pool,
        key=lambda b: (eval_by_id[b.bundle_id].pareto_rank, input_index[b.bundle_id]),
    )
    baseline_ids = [b.bundle_id for b in baseline_sorted]

    # Group 2: Display-Only — compute score and attach as display_only metric WITHOUT touching selection/order
    scores: dict[str, float] = {}
    display_evals: list[BundleEvaluation] = []
    for b in candidate_pool:
        s = float(scorer_fn(b))
        scores[b.bundle_id] = s
        orig = eval_by_id[b.bundle_id]
        extra_metric = MetricResult(
            metric_id=scorer_name,
            value=s,
            unit="score",
            direction="maximize",
            evaluated_scopes=["cds"],
            tool_name=scorer_name,
            tool_version="0.1.0",
            chemistry_evaluation_mode="not_evaluated",
            participated_in="display_only",
        )
        display_evals.append(orig.model_copy_validated(metrics=[*orig.metrics, extra_metric]))

    display_sorted = sorted(
        candidate_pool,
        key=lambda b: (eval_by_id[b.bundle_id].pareto_rank, input_index[b.bundle_id]),
    )
    display_ids = [b.bundle_id for b in display_sorted]
    if baseline_ids != display_ids:
        raise AssertionError("Invariant violated: Display-Only mode altered baseline order")

    # Group 3: Score participates in ranking (within Pareto rank or primary tie-break, higher score wins)
    ranked_sorted = sorted(
        candidate_pool,
        key=lambda b: (-scores[b.bundle_id], eval_by_id[b.bundle_id].pareto_rank, input_index[b.bundle_id]),
    )
    ranked_ids = [b.bundle_id for b in ranked_sorted]

    return FixedPoolScorerReplayResult(
        scorer_name=scorer_name,
        baseline_ordered_ids=baseline_ids,
        display_only_ordered_ids=display_ids,
        rank_participating_ordered_ids=ranked_ids,
        order_changed_in_group3=(ranked_ids != baseline_ids),
        display_only_evaluations=display_evals,
        honesty_banner="本次未改善生成器",
    )


@dataclass(frozen=True)
class PairedSeedAblationRow:
    seed: int
    stage0_best_kozak: float
    stage0_best_cds_mfe: float
    pareto_best_kozak: float
    pareto_best_cds_mfe: float
    stage0_wall_sec: float
    pareto_wall_sec: float


def run_paired_seed_generator_ablation(
    intent: DesignIntent,
    seeds: list[int],
    *,
    seed_dir: Path,
) -> list[PairedSeedAblationRow]:
    from mrnax.engine.pareto_director import run_design_pipeline

    rows: list[PairedSeedAblationRow] = []
    for sd in seeds:
        pol_stage0 = intent.runtime_policy.model_copy_validated(
            random_seed=sd, enabled_generators=["stage0_random", "dnachisel_cds"]
        )
        t0 = time.perf_counter()
        _, rep_0, _ = run_design_pipeline(
            intent.model_copy_validated(runtime_policy=pol_stage0), seed_dir=seed_dir
        )
        dt0 = time.perf_counter() - t0

        pol_pareto = intent.runtime_policy.model_copy_validated(
            random_seed=sd, enabled_generators=["stage0_random", "dnachisel_cds", "pareto_nsga2"]
        )
        t1 = time.perf_counter()
        _, rep_1, _ = run_design_pipeline(
            intent.model_copy_validated(runtime_policy=pol_pareto), seed_dir=seed_dir
        )
        dt1 = time.perf_counter() - t1

        def _extract_best(evals: list[BundleEvaluation]) -> tuple[float, float]:
            best_k = max(
                (
                    float(m.value)
                    for e in evals
                    for m in e.metrics
                    if m.metric_id == "kozak_unpaired_prob" and m.value is not None
                ),
                default=0.0,
            )
            best_mfe = min(
                (
                    float(m.value)
                    for e in evals
                    for m in e.metrics
                    if m.metric_id == "cds_mean_window_mfe" and m.value is not None
                ),
                default=0.0,
            )
            return best_k, best_mfe

        k0, m0 = _extract_best(rep_0.candidate_evaluations)
        k1, m1 = _extract_best(rep_1.candidate_evaluations)
        rows.append(
            PairedSeedAblationRow(
                seed=sd,
                stage0_best_kozak=k0,
                stage0_best_cds_mfe=m0,
                pareto_best_kozak=k1,
                pareto_best_cds_mfe=m1,
                stage0_wall_sec=round(dt0, 4),
                pareto_wall_sec=round(dt1, 4),
            )
        )
    return rows


@dataclass(frozen=True)
class ThreeWayUpstreamAttributionResult:
    total_budget_seconds: float
    upstream_standalone_bundles: list[ConstructBundle]
    mrnax_independent_bundles: list[ConstructBundle]
    mrnax_from_upstream_seed_bundles: list[ConstructBundle]
    upstream_seed_cost_seconds: float
    mrnax_remaining_budget_seconds: float


def run_three_way_upstream_attribution(
    intent: DesignIntent,
    *,
    total_budget_seconds: float,
    upstream_generator: Callable[[DesignIntent, float], tuple[list[ConstructBundle], float]],
    seed_dir: Path,
) -> ThreeWayUpstreamAttributionResult:
    """Equal-total-budget 3-group attribution deducting upstream seed generation time in Group 3."""
    from mrnax.engine.pareto_director import run_design_pipeline

    # Group 1: Upstream standalone with full budget
    g1_bundles, _ = upstream_generator(intent, total_budget_seconds)

    # Group 2: mRNAx independent start with full budget
    pol_g2 = intent.runtime_policy.model_copy_validated(max_wall_seconds=total_budget_seconds)
    g2_bundles, _, _ = run_design_pipeline(
        intent.model_copy_validated(runtime_policy=pol_g2), seed_dir=seed_dir
    )

    # Group 3: Upstream generates seed first, remaining wall budget passed to mRNAx
    seed_bundles, upstream_elapsed = upstream_generator(intent, total_budget_seconds * 0.5)
    remaining = max(0.05, total_budget_seconds - upstream_elapsed)
    pol_g3 = intent.runtime_policy.model_copy_validated(max_wall_seconds=remaining)
    g3_bundles, _, _ = run_design_pipeline(
        intent.model_copy_validated(runtime_policy=pol_g3),
        reference_bundle=seed_bundles[0] if seed_bundles else None,
        seed_dir=seed_dir,
    )
    return ThreeWayUpstreamAttributionResult(
        total_budget_seconds=total_budget_seconds,
        upstream_standalone_bundles=g1_bundles,
        mrnax_independent_bundles=g2_bundles,
        mrnax_from_upstream_seed_bundles=g3_bundles,
        upstream_seed_cost_seconds=round(upstream_elapsed, 4),
        mrnax_remaining_budget_seconds=round(remaining, 4),
    )

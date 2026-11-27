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
    candidate_digests: dict[str, str]
    candidate_scores: dict[str, float]
    honesty_banner: str = "本次未改善生成器"


def run_fixed_pool_scorer_replay(
    *,
    candidate_pool: list[ConstructBundle],
    base_evaluations: list[BundleEvaluation],
    scorer_name: str,
    scorer_fn: Callable[[ConstructBundle], float],
) -> FixedPoolScorerReplayResult:
    """Replay a fixed candidate pool across 3 modes with deterministic tie-breaking."""
    if not candidate_pool:
        raise ValueError("Fixed-pool scorer replay requires at least one qualified candidate in candidate_pool")
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
        candidate_digests={b.bundle_id: b.version_digest for b in candidate_pool},
        candidate_scores={bid: round(val, 6) for bid, val in scores.items()},
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


@dataclass(frozen=True)
class PairedSeedSurrogateSwitchRow:
    seed: int
    unguided_trial_count: int
    guided_trial_count: int
    unguided_best_utr5_mrl: float
    unguided_best_cds_stability: float
    unguided_best_halflife: float
    guided_best_utr5_mrl: float
    guided_best_cds_stability: float
    guided_best_halflife: float
    unguided_wall_sec: float
    guided_wall_sec: float


def run_paired_seed_surrogate_switch_ablation(
    intent: DesignIntent,
    seeds: list[int],
    *,
    seed_dir: Path,
) -> list[PairedSeedSurrogateSwitchRow]:
    """Equal-budget paired-seed inner-loop switch ablation (surrogate OFF vs `surrogate_v1_clean` ON)."""
    from mrnax.engine.pareto_director import run_design_pipeline
    from mrnax.engine.surrogates import CommercialCleanSurrogateV1

    surrogate = CommercialCleanSurrogateV1.from_seed_dir(seed_dir)
    rows: list[PairedSeedSurrogateSwitchRow] = []

    def _measure_pool(bundles: list[ConstructBundle]) -> tuple[float, float, float]:
        if not bundles:
            return 0.0, 0.0, 0.0
        mrls = [float(surrogate.score_5utr_mrl([b.molecules[0].get_segment_sequence("five_prime_utr")])[0]) for b in bundles]
        stabs = [float(surrogate.score_cds_stability([b.molecules[0].get_segment_sequence("cds")])[0]) for b in bundles]
        hls = [
            surrogate.score_consensus_halflife(
                b.molecules[0].get_segment_sequence("five_prime_utr"),
                b.molecules[0].get_segment_sequence("cds"),
                b.molecules[0].get_segment_sequence("three_prime_utr"),
            )
            for b in bundles
        ]
        return round(max(mrls), 5), round(max(stabs), 5), round(max(hls), 5)

    for sd in seeds:
        pol_off = intent.runtime_policy.model_copy_validated(
            random_seed=sd,
            enabled_evaluators=[],
        )
        t0 = time.perf_counter()
        bundles_off, rep_off, man_off = run_design_pipeline(
            intent.model_copy_validated(runtime_policy=pol_off, objectives=[]), seed_dir=seed_dir
        )
        dt_off = time.perf_counter() - t0

        pol_on = intent.runtime_policy.model_copy_validated(
            random_seed=sd,
            enabled_evaluators=["surrogate_v1_clean"],
        )
        t1 = time.perf_counter()
        bundles_on, rep_on, man_on = run_design_pipeline(
            intent.model_copy_validated(runtime_policy=pol_on), seed_dir=seed_dir
        )
        dt_on = time.perf_counter() - t1

        if rep_off.outcome_status.startswith("UNCOMPARABLE") or rep_on.outcome_status.startswith("UNCOMPARABLE"):
            raise ValueError(
                f"Cannot run paired ablation when an arm is uncomparable ({rep_off.outcome_status} vs {rep_on.outcome_status})"
            )
        if not bundles_off or not bundles_on:
            raise ValueError(
                f"Paired ablation produced zero qualified candidates on seed={sd} ({len(bundles_off)} vs {len(bundles_on)})"
            )

        trials_off = sum(c.attempt_count for c in man_off.invoked_components if c.role == "generator")
        trials_on = sum(c.attempt_count for c in man_on.invoked_components if c.role == "generator")
        if trials_off != trials_on:
            raise AssertionError(
                f"Equal-work budget violated on seed={sd}: unguided_trial_count={trials_off} != guided_trial_count={trials_on}"
            )

        u_mrl, u_stab, u_hl = _measure_pool(bundles_off)
        g_mrl, g_stab, g_hl = _measure_pool(bundles_on)
        rows.append(
            PairedSeedSurrogateSwitchRow(
                seed=sd,
                unguided_trial_count=trials_off,
                guided_trial_count=trials_on,
                unguided_best_utr5_mrl=u_mrl,
                unguided_best_cds_stability=u_stab,
                unguided_best_halflife=u_hl,
                guided_best_utr5_mrl=g_mrl,
                guided_best_cds_stability=g_stab,
                guided_best_halflife=g_hl,
                unguided_wall_sec=round(dt_off, 4),
                guided_wall_sec=round(dt_on, 4),
            )
        )
    return rows

"""Sliding-window structure-guided CDS<->UTR alternating Pareto optimization director."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, get_args

from mrnax.constraints.frameshift_m1psi import remediate_m1psi_slippery_runs
from mrnax.constraints.fto_guard import COMMERCIAL_ZERO_GPL_REQUIREMENT, MethodFTOGuard, normalize_sense_cds
from mrnax.constraints.registry import ScopedConstraintEngine
from mrnax.constraints.sequence_gates import has_rc005_cds_provenance
from mrnax.engine.cell_expression import (
    MissingTissueMeasurement,
    select_detarget_mirna_entry,
    assemble_detargeted_3utr,
    build_crs_and_tissue_safety_advisory,
)
from mrnax.engine.lineardesign_adapter import local_heavy_binaries_disabled
from mrnax.engine.profiles import (
    COMPONENT_CATALOG,
    RC005_PARTNER_METRICS,
    audit_component_commercial_clean,
    check_profile_readiness,
    producer_covers_metric,
    rc005_partner_covers_metric,
    structrfm_unavailable_reason,
    resolve_default_seed_dir,
    resolve_tool_name_to_component_id,
)
from mrnax.engine.stage0_baseline import (
    generate_dnachisel_cds,
    generate_random_synonymous_cds,
    mutate_cds_region_synonymously,
    propose_non_partitioned_bicodon_cds,
)
from mrnax.engine.surrogates import (
    STRUCTRFM_NATIVE_CHEMISTRIES,
    CommercialCleanSurrogateV1,
    CommercialCleanSurrogateV2,
    StructRFMNativeScorer,
    has_surrogate_v2_artifacts,
)
from mrnax.engine.thermodynamics import (
    compute_cds_sliding_window_mfe,
    evaluate_molecule_thermodynamics,
)
from mrnax.rules.models import APPROVED_REPLACEMENT_RELEASES
from mrnax.rules.tribunal import RuleRegistry
from mrnax.schemas.common import LicenseTrack, SegmentScope
from mrnax.schemas.construct_bundle import (
    ChemistryTrackSpec,
    ConstructBundle,
    DerivationAction,
    FeatureAnnotation,
    RNAMoleculeSpec,
    SegmentSpan,
)
from mrnax.schemas.design_intent import BundleSegmentLock, DesignIntent, ObjectiveSpec
from mrnax.schemas.evaluation_report import (
    AdvisoryNote,
    BundleEvaluation,
    EvaluationReport,
    FailedCandidateDiagnostic,
    FTOAlert,
    MetricResult,
    RequirementActionTrace,
    serialize_report_artifact,
)
from mrnax.schemas.run_manifest import (
    ArtifactChecksumRecord,
    ComponentInvocationRecord,
    MethodFTOAuditRecord,
    RunManifest,
)


class NoAvailableGeneratorError(RuntimeError):
    """Raised when all candidate generators are disabled and `allow_cached_candidates` is False."""


class UnsupportedModalityError(RuntimeError):
    """Raised fail-closed in V1 when saRNA, taRNA, or circRNA generation is requested."""


def _git_sha(repo_dir: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_dir),
            check=True,
            capture_output=True,
            text=True,
        )
        return proc.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


@dataclass(frozen=True)
class _Origin:
    component_id: str | None
    source: str
    tier: LicenseTrack


def _build_molecule_spec(
    *,
    molecule_id: str,
    target: Any,
    intent: DesignIntent,
    origins: dict[SegmentScope, _Origin],
    utr5_seq: str,
    utr5_action: str,
    cds_seq: str,
    cds_action: str,
    stop_seq: str,
    utr3_seq: str,
    utr3_action: str,
    polya_seq: str,
    extra_utr3_annotations: list[FeatureAnnotation] | None = None,
) -> RNAMoleculeSpec:
    c0 = 0
    c1 = len(utr5_seq)
    c2 = c1 + len(cds_seq)
    c3 = c2 + len(stop_seq)
    c4 = c3 + len(utr3_seq)
    c5 = c4 + len(polya_seq)
    full_seq = utr5_seq + cds_seq + stop_seq + utr3_seq + polya_seq

    anns: list[FeatureAnnotation] = [
        FeatureAnnotation(
            feature_id="kozak_tis",
            feature_key="regulatory",
            intervals=[(max(0, c1 - 6), min(c5, c1 + 4))],
            qualifiers={"regulatory_class": ["ribosome_binding_site"]},
        )
    ]
    if extra_utr3_annotations:
        for a in extra_utr3_annotations:
            shifted = [(c3 + s, c3 + e) for s, e in a.intervals]
            anns.append(a.model_copy_validated(intervals=shifted))

    return RNAMoleculeSpec(
        molecule_id=molecule_id,
        modality="linear_mrna",
        topology="linear",
        chemistry_track=ChemistryTrackSpec(
            uridine_chemistry=intent.uridine_chemistry,
            cap_chemistry="cap1_cleancap_ag",
            source="user_declared",
        ),
        effective_translation_target=target,
        sequence=full_seq,
        segments=[
            SegmentSpan(
                segment_type="five_prime_utr",
                start=c0,
                end=c1,
                component_id=origins["five_prime_utr"].component_id,
                origin_source=origins["five_prime_utr"].source,
                license_track=origins["five_prime_utr"].tier,
                action_taken=utr5_action,  # type: ignore[arg-type]
            ),
            SegmentSpan(
                segment_type="cds",
                start=c1,
                end=c2,
                component_id=origins["cds"].component_id,
                origin_source=origins["cds"].source,
                license_track=origins["cds"].tier,
                action_taken=cds_action,  # type: ignore[arg-type]
            ),
            SegmentSpan(
                segment_type="stop_cassette",
                start=c2,
                end=c3,
                component_id=origins["stop_cassette"].component_id,
                origin_source=origins["stop_cassette"].source,
                license_track=origins["stop_cassette"].tier,
                action_taken="locked_verbatim",
            ),
            SegmentSpan(
                segment_type="three_prime_utr",
                start=c3,
                end=c4,
                component_id=origins["three_prime_utr"].component_id,
                origin_source=origins["three_prime_utr"].source,
                license_track=origins["three_prime_utr"].tier,
                action_taken=utr3_action,  # type: ignore[arg-type]
            ),
            SegmentSpan(
                segment_type="poly_a",
                start=c4,
                end=c5,
                component_id=origins["poly_a"].component_id,
                origin_source=origins["poly_a"].source,
                license_track=origins["poly_a"].tier,
                action_taken="locked_verbatim",
            ),
        ],
        annotations=anns,
    )


def _build_single_molecule_bundle(
    *,
    bundle_id: str,
    parent: ConstructBundle | None,
    derivation_action: DerivationAction,
    intent: DesignIntent,
    origins: dict[SegmentScope, _Origin],
    utr5_seq: str,
    utr5_action: str,
    cds_seq: str,
    cds_action: str,
    stop_seq: str,
    utr3_seq: str,
    utr3_action: str,
    polya_seq: str,
    extra_utr3_annotations: list[FeatureAnnotation] | None = None,
    secondary_molecules: list[RNAMoleculeSpec] | None = None,
) -> ConstructBundle:
    mol_id_0 = "mol_primary" if not secondary_molecules else "chain_01"
    mol0 = _build_molecule_spec(
        molecule_id=mol_id_0,
        target=intent.protein_targets[0],
        intent=intent,
        origins=origins,
        utr5_seq=utr5_seq,
        utr5_action=utr5_action,
        cds_seq=cds_seq,
        cds_action=cds_action,
        stop_seq=stop_seq,
        utr3_seq=utr3_seq,
        utr3_action=utr3_action,
        polya_seq=polya_seq,
        extra_utr3_annotations=extra_utr3_annotations,
    )
    all_mols = [mol0, *(secondary_molecules or [])]
    stoich = {m.molecule_id: 1.0 for m in all_mols}
    co_mode = "lnp_co_encapsulation" if len(all_mols) > 1 else "single_monocistronic"
    return ConstructBundle.create(
        bundle_id=bundle_id,
        parent_bundle_id=parent.bundle_id if parent else None,
        external_evidence=parent.external_evidence if parent else [],
        derivation_action=derivation_action,
        target_stoichiometry=stoich,
        co_expression_mode=co_mode,  # type: ignore[arg-type]
        molecules=all_mols,
    )


def _dominates(vec_a: list[float], vec_b: list[float]) -> bool:
    """Maximize all normalized objective values in `vec`."""
    return all(a >= b for a, b in zip(vec_a, vec_b, strict=True)) and any(
        a > b for a, b in zip(vec_a, vec_b, strict=True)
    )


def _assign_pareto_ranks(obj_vectors: list[list[float]]) -> list[int]:
    n = len(obj_vectors)
    ranks = [1] * n
    remaining = set(range(n))
    current_rank = 1
    while remaining:
        front = [
            i
            for i in remaining
            if not any(_dominates(obj_vectors[j], obj_vectors[i]) for j in remaining if j != i)
        ]
        if not front:
            front = list(remaining)
        for idx in front:
            ranks[idx] = current_rank
            remaining.remove(idx)
        current_rank += 1
    return ranks


def _rank_and_order_pareto_candidates(obj_vectors: list[list[float]]) -> tuple[list[int], list[int]]:
    """Assign Pareto ranks and order candidates using strictly scale-invariant rank percentiles & crowding distance."""
    n = len(obj_vectors)
    if n == 0:
        return [], []
    if not obj_vectors[0]:
        return [1] * n, list(range(n))
    ranks = _assign_pareto_ranks(obj_vectors)
    dims = len(obj_vectors[0])

    # Per-axis rank percentile in [0, 1] — invariant to ANY strictly increasing rescaling of any axis!
    quantiles = [[0.5] * dims for _ in range(n)]
    if n > 1:
        for d in range(dims):
            vals = [vec[d] for vec in obj_vectors]
            sorted_idx = sorted(range(n), key=lambda i: vals[i])
            pos = 0
            while pos < n:
                end = pos + 1
                while end < n and vals[sorted_idx[end]] == vals[sorted_idx[pos]]:
                    end += 1
                avg_rank = 0.5 * (pos + end - 1) / (n - 1)
                for k in range(pos, end):
                    quantiles[sorted_idx[k]][d] = avg_rank
                pos = end

    mean_quantile = [sum(q) / max(dims, 1) for q in quantiles]
    crowding = [0.0] * n
    by_rank: dict[int, list[int]] = {}
    for idx, rk in enumerate(ranks):
        by_rank.setdefault(rk, []).append(idx)

    for front in by_rank.values():
        if len(front) <= 2 or dims == 1:
            for idx in front:
                crowding[idx] = mean_quantile[idx]
            continue
        for d in range(dims):
            distinct_vals = sorted({round(quantiles[idx][d], 12) for idx in front})
            if len(distinct_vals) <= 1:
                continue
            min_q = distinct_vals[0]
            max_q = distinct_vals[-1]
            span = max_q - min_q
            if span <= 1e-12:
                continue
            val_to_pos = {v: k for k, v in enumerate(distinct_vals)}
            for idx_f in front:
                k = val_to_pos[round(quantiles[idx_f][d], 12)]
                if k == 0 or k == len(distinct_vals) - 1:
                    crowding[idx_f] += 2.0
                else:
                    crowding[idx_f] += (distinct_vals[k + 1] - distinct_vals[k - 1]) / span

    ordered_indices = sorted(
        range(n),
        key=lambda i: (ranks[i], -round(mean_quantile[i], 9), -round(crowding[i], 9), i),
    )
    return ranks, ordered_indices


_REPORT_NS_LOCK = __import__("threading").Lock()
_LAST_REPORT_NS = 0


def next_report_timestamp_ns() -> int:
    global _LAST_REPORT_NS
    with _REPORT_NS_LOCK:
        now = time.time_ns()
        if now <= _LAST_REPORT_NS:
            now = _LAST_REPORT_NS + 1
        _LAST_REPORT_NS = now
        return now


def score_objective_value(val: float, obj: ObjectiveSpec) -> float | None:
    """Convert raw metric value into maximize-normalized objective score; returns None if bounds are missing/invalid."""
    if not math.isfinite(val):
        return None
    if obj.direction == "maximize":
        return float(val)
    if obj.direction == "minimize":
        return -float(val)
    if obj.direction == "target_band":
        if obj.target_band is None or len(obj.target_band) != 2:
            return None
        lo_raw, hi_raw = obj.target_band
        if lo_raw is None or hi_raw is None:
            return None
        lo, hi = float(lo_raw), float(hi_raw)
        if not math.isfinite(lo) or not math.isfinite(hi) or lo > hi:
            return None
        mid = 0.5 * (lo + hi)
        span = max(hi - lo, 1.0)
        if lo <= val <= hi:
            return 0.0 if hi == lo else -0.25 * (abs(val - mid) / (0.5 * (hi - lo)))
        return -(1.0 + min(abs(val - lo), abs(val - hi)) / span)
    return None


def _is_rc005_partner_measurement(metric: MetricResult, component: str | None, *, chains: int) -> bool:
    if not rc005_partner_covers_metric(component, metric.tool_name, metric.metric_id, metric.evaluated_scopes):
        return False
    if metric.participated_in == "display_only" or metric.value is None or not math.isfinite(metric.value):
        return False
    if metric.chain_values is None:
        return chains == 1
    return len(metric.chain_values) == chains and all(math.isfinite(v) for v in metric.chain_values)


def project_report_for_intent(
    report: EvaluationReport,
    bundles_by_id: dict[str, ConstructBundle],
    intent: DesignIntent | None,
    *,
    seed_dir: Path | None = None,
    disallow_sandbox_override: bool = False,
) -> EvaluationReport:
    """Project stored EvaluationReport against active DesignIntent target proteins, chains, rules, and measured objectives."""
    def without_ranking():
        return [trace for trace in report.action_transparency if trace.stage_applied != "ranking"]

    def withdraw(reason: str) -> EvaluationReport:
        return report.model_copy_validated(
            outcome_status="UNCOMPARABLE_MISSING_APPROVED_COMPONENT", uncomparable_reason=reason,
            candidate_evaluations=[], recommended_bundle_ids=[], action_transparency=without_ranking(),
        )

    is_commercial = (
        intent is not None and intent.compliance_track == "commercial_clean"
    ) or report.compliance_track == "commercial_clean"
    if is_commercial:
        has_candidate_claims = bool(report.candidate_evaluations or report.recommended_bundle_ids)
        completed = report.outcome_status in {"COMPLETED_WITH_CANDIDATES", "COMPLETED_TIE"}
        wrong_origin = intent is not None and (
            report.intent_id != intent.intent_id or report.intent_version != intent.intent_version
            or report.compliance_track != intent.compliance_track
        )
        if wrong_origin or (has_candidate_claims and (not completed or not report.candidate_evaluations)):
            return report.model_copy_validated(
                outcome_status="UNCOMPARABLE_MISSING_APPROVED_COMPONENT",
                uncomparable_reason="Stored candidate claims lack a completed evaluation for this intent revision and track.",
                candidate_evaluations=[], recommended_bundle_ids=[], action_transparency=[],
            )
        if disallow_sandbox_override and report.isolated_sandbox_override:
            return withdraw("Sandbox-override report cannot be served as a production commercial_clean recommendation.")
        eff_seed = seed_dir or resolve_default_seed_dir()
        cc_rules_path = eff_seed / "rules_commercial_v1.json"
        if not cc_rules_path.is_file():
            return withdraw(f"Required commercial rule registry '{cc_rules_path.name}' is missing.")
        try:
            verified_snap = RuleRegistry(cc_rules_path, track="commercial_clean").snapshot_version()
        except ValueError as exc:
            return withdraw(str(exc))
        if report.rule_snapshot_version != verified_snap:
            return withdraw(
                f"Stored report rule snapshot '{report.rule_snapshot_version}' does not match active "
                f"commercial_clean registry '{verified_snap}'."
            )
        if intent is not None:
            readiness = check_profile_readiness(intent)
            if not readiness.is_runnable:
                return withdraw(readiness.uncomparable_reason or "Intent components are not approved for commercial_clean.")

    if not report.candidate_evaluations:
        return report.model_copy_validated(action_transparency=without_ranking())

    expected_proteins = [t.protein_sequence for t in intent.protein_targets] if (intent and intent.protein_targets) else None
    rankable_objs = (
        [o for o in intent.objectives if o.evidence_status != "unverified_target_for_wetlab"]
        if intent is not None
        else None
    )

    constraint_engine = None
    if is_commercial:
        try:
            constraint_engine = ScopedConstraintEngine.from_seed_dir(eff_seed)
        except (OSError, ValueError) as exc:
            return withdraw(f"Cannot verify stored candidate hard gates: {exc}")
    no_objectives = intent is not None and not intent.objectives
    surviving: list[tuple[BundleEvaluation, list[float]]] = []
    conflicting_digests = {
        e.bundle_id for e in report.candidate_evaluations
        if e.bundle_id in bundles_by_id and getattr(e, "version_digest", None) != bundles_by_id[e.bundle_id].version_digest
    }
    for ev in report.candidate_evaluations:
        bundle = bundles_by_id.get(ev.bundle_id)
        if bundle is None or ev.bundle_id in conflicting_digests or not ev.passed_all_hard_gates:
            continue
        if ev.version_digest != bundle.compute_digest(
            bundle.molecules, bundle.target_stoichiometry, bundle.co_expression_mode, bundle.external_evidence
        ):
            continue
        if expected_proteins is not None:
            if len(bundle.molecules) != len(expected_proteins):
                continue
            if [m.effective_translation_target.protein_sequence for m in bundle.molecules] != expected_proteins:
                continue
            if any(
                m.effective_translation_target.transl_table != target.transl_table
                or m.translate_cds_protein(translation_table=target.transl_table) != target.protein_sequence
                for m, target in zip(bundle.molecules, intent.protein_targets, strict=True)
            ):
                continue
        partial_ids = {
            m.metric_id
            for m in ev.metrics
            if len(bundle.molecules) > 1
            and "inter_chain" not in m.evaluated_scopes
            and (
                m.chain_values is None
                or len(m.chain_values) != len(bundle.molecules)
                or not all(math.isfinite(v) for v in m.chain_values)
            )
        }
        if partial_ids:
            continue
        m_map: dict[str, float] = {}
        conflicting_dup = False
        unapproved_tool_hit = False
        requires_rc005_partner = has_rc005_cds_provenance(bundle) or bool(
            intent and "codontransformer_cds_baseline" in intent.runtime_policy.enabled_generators
        ) or any(resolve_tool_name_to_component_id(m.tool_name) == "codontransformer_cds_baseline" for m in ev.metrics)
        has_cleared_partner_metric = False
        chemistries = {mol.chemistry_track.uridine_chemistry for mol in bundle.molecules}
        if intent is not None:
            chemistries.add(intent.uridine_chemistry)
        for m in ev.metrics:
            comp_id = resolve_tool_name_to_component_id(m.tool_name, m.metric_id)
            if is_commercial:
                if comp_id is None:
                    if m.participated_in != "display_only":
                        unapproved_tool_hit = True
                        break
                elif audit_component_commercial_clean(comp_id) is not None:
                    unapproved_tool_hit = True
                    break
            check_domain = is_commercial or comp_id is not None or requires_rc005_partner
            if check_domain and not producer_covers_metric(
                comp_id, m.tool_name, m.metric_id, m.evaluated_scopes,
                display_only=m.participated_in == "display_only", research=not is_commercial,
            ):
                unapproved_tool_hit = True
                break
            if comp_id == "structrfm_rna_foundation" and not chemistries <= STRUCTRFM_NATIVE_CHEMISTRIES:
                unapproved_tool_hit = True
                break
            if (comp_id is not None and _is_rc005_partner_measurement(m, comp_id, chains=len(bundle.molecules))
                    and (is_commercial or audit_component_commercial_clean(comp_id) is None)):
                has_cleared_partner_metric = True
            if (
                m.participated_in != "display_only"
                and m.value is not None
                and math.isfinite(m.value)
                and m.metric_id not in partial_ids
            ):
                fval = float(m.value)
                if m.metric_id in m_map and m_map[m.metric_id] != fval:
                    conflicting_dup = True
                    break
                m_map[m.metric_id] = fval
        if unapproved_tool_hit or conflicting_dup or (not m_map and not no_objectives):
            continue
        if requires_rc005_partner and not has_cleared_partner_metric:
            continue
        if constraint_engine is not None:
            from mrnax.io_insdc.genbank_io import resolve_expected_chemistry_for_bundle

            proj_chem = resolve_expected_chemistry_for_bundle(bundle, intent.uridine_chemistry) if intent else None
            checked = constraint_engine.evaluate_bundle(
                bundle, intent.constraints if intent else [], compliance_track="commercial_clean",
                expected_chemistry=proj_chem,
                locked_scopes=intent.locked_scopes if intent else None,
                expected_protein_targets=intent.protein_targets if intent else None,
                rc005_active=requires_rc005_partner,
            )
            if not checked.passed_all_hard_gates:
                continue
        if rankable_objs is not None:
            if not rankable_objs and not no_objectives:
                continue
            vec: list[float] = []
            ok = True
            for obj in rankable_objs:
                if obj.metric_id not in m_map:
                    ok = False
                    break
                sc = score_objective_value(m_map[obj.metric_id], obj)
                if sc is None:
                    ok = False
                    break
                vec.append(sc)
            if not ok:
                continue
            surviving.append((ev, vec))
        else:
            surviving.append((ev, []))

    if not surviving:
        return withdraw(
            report.uncomparable_reason or "Stored candidates lack complete objective, target-protein, or chain measurement coverage."
        )

    if rankable_objs:
        vectors = [vec for _, vec in surviving]
        ranks, order = _rank_and_order_pareto_candidates(vectors)
        recomputed_evals = [
            surviving[idx][0].model_copy_validated(
                pareto_rank=ranks[idx],
                selection_rationale=[f"Pareto rank {ranks[idx]} under {report.compliance_track} profile"],
            )
            for idx in order
        ]
        # Keep each measured row on its own card; recommendation slots identify distinct bundles.
        front = dict.fromkeys(e.bundle_id for e in recomputed_evals if e.pareto_rank == 1)
        kept_recs = [bid for bid in report.recommended_bundle_ids if bid in front]
        if not kept_recs or len(set(kept_recs)) != len(kept_recs):
            kept_recs = list(front)
        kept_recs = kept_recs[:3]
        is_tie = len(front) > 1 and all(v == vectors[0] for v in vectors[1:])
        status = "COMPLETED_TIE" if is_tie else "COMPLETED_WITH_CANDIDATES"
    else:
        recomputed_evals = [ev for ev, _ in surviving]
        valid_ids = {ev.bundle_id for ev in recomputed_evals}
        kept_recs = [] if intent is not None else list(dict.fromkeys(
            bid for bid in report.recommended_bundle_ids if bid in valid_ids
        ))
        status = "COMPLETED_WITH_CANDIDATES" if len(surviving) == 1 and report.outcome_status == "COMPLETED_TIE" else report.outcome_status

    actions = []
    filtered = len(surviving) != len(report.candidate_evaluations) or kept_recs != report.recommended_bundle_ids
    for trace in report.action_transparency:
        if trace.stage_applied != "ranking":
            actions.append(trace)
        elif kept_recs:
            ids = list(dict.fromkeys(bid for bid in trace.changed_candidate_ids if bid in kept_recs))
            if filtered or ids != trace.changed_candidate_ids:
                trace = trace.model_copy_validated(
                    changed_candidate_ids=ids, proof_diff_summary="Historical ranking restricted to eligible candidates."
                )
            actions.append(trace)
    return report.model_copy_validated(
        outcome_status=status, candidate_evaluations=recomputed_evals,
        recommended_bundle_ids=kept_recs, action_transparency=actions,
    )


def _aggregate_chain_metrics(chain_metric_lists: list[list[MetricResult]]) -> tuple[list[MetricResult], bool]:
    n_chains = len(chain_metric_lists)
    if n_chains == 1:
        return chain_metric_lists[0], True
    chain_maps = [{m.metric_id: m for m in chain} for chain in chain_metric_lists]
    merged = []
    complete = True
    for mid in dict.fromkeys(m.metric_id for chain in chain_metric_lists for m in chain):
        items = [chain[mid] for chain in chain_maps if mid in chain]
        first = items[0]
        values = [m.value for m in items if m.value is not None and math.isfinite(m.value)
                  and (m.direction, m.unit, m.participated_in) == (first.direction, first.unit, first.participated_in)]
        if len(values) != n_chains:
            complete = False
            result = first.model_copy_validated(
                value=None,
                chain_values=None,
                notes=f"incomplete_chain_coverage({len(values)}/{n_chains})",
            )
        else:
            if first.direction == "maximize":
                value = min(values)
            elif first.direction == "minimize":
                value = max(values)
            else:
                value = sum(values) / n_chains
            policy = "worst_case" if first.direction in {"maximize", "minimize"} else "mean"
            result = first.model_copy_validated(
                value=round(value, 6),
                chain_values=[round(float(v), 6) for v in values],
                notes=f"aggregated_{n_chains}_chains({policy})",
            )
        merged.append(result)
    return merged, complete


def _load_rbp_motif_sets(seed_dir: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Load destabilizing ARE/PUM and stabilizing PCBP motifs from ATtRACT 160-RBP registry."""
    rbp_path = seed_dir / "rbp_human_motifs_attract_ccby4_v1.json"
    destab: list[str] = ["AUUUA", "UUAUUUAUU", "AUUUAUUUAUUU", "UGUA"]
    stab: list[str] = ["CCCC", "UCCC", "CCCU"]
    if rbp_path.is_file():
        doc = json.loads(rbp_path.read_text(encoding="utf-8"))
        curated = doc.get("curated_mrna_engineering_rbps", {})
        for gene in ("ZFP36", "KHSRP", "ELAVL1", "PUM1", "PUM2"):
            for m in curated.get(gene, {}).get("top_motifs", []):
                seq = str(m.get("motif_rna") or "")
                if len(seq) >= 5 and ("AUUU" in seq or "UGUA" in seq):
                    destab.append(seq)
        for gene in ("PCBP1", "PCBP2", "IGF2BP1"):
            for m in curated.get(gene, {}).get("top_motifs", []):
                seq = str(m.get("motif_rna") or "")
                if 4 <= len(seq) <= 7 and seq.count("C") >= 3:
                    stab.append(seq)
    return tuple(dict.fromkeys(destab)), tuple(dict.fromkeys(stab))


def _utr5_gc_tail15_prior(seq: str) -> float:
    """RUL-CC-5UTR-GC-TIS-007: moderate/open TIS [-15..-1] GC avoids stable pre-AUG clamp."""
    tail = seq[-15:] if len(seq) >= 15 else seq
    if not tail:
        return 0.0
    gc = (tail.count("G") + tail.count("C")) / len(tail)
    if 0.33 <= gc <= 0.60:
        return 0.35
    if gc > 0.72:
        return -0.60
    return 0.10 - abs(gc - 0.47)


def _utr3_rbp_decay_prior(seq: str, destab_motifs: tuple[str, ...], stab_motifs: tuple[str, ...]) -> float:
    """RUL-CC-3UTR-ARE-DECAY-004 + ATtRACT 160-RBP decay/stabilizer prior."""
    score = 0.0
    score -= 1.25 * seq.count("AUUUA")
    score -= 1.50 * seq.count("UUAUUUAUU")
    score -= 0.25 * seq.count("UGUA")
    for m in destab_motifs:
        if len(m) >= 6 and m in seq:
            score -= 0.35
    stab_hits = sum(seq.count(m) for m in stab_motifs[:6])
    score += min(0.45, 0.08 * stab_hits)
    return score


def _valid_otool_library_listing(output: str, executable: Path) -> bool:
    if "is not an object file" in output.lower():
        return False
    header = re.compile(re.escape(str(executable)) + r"(?: \(architecture [^()]+\))?:")
    version = r"[0-9]+(?:\.[0-9]+){0,2}"
    dependency = re.compile(rf"\s+\S.* \(compatibility version {version}, current version {version}(?:, [^()]+)?\)")
    libraries = None
    for line in output.splitlines():
        if header.fullmatch(line):
            if libraries == 0:
                return False
            libraries = 0
        elif libraries is not None and line.strip():
            if not dependency.fullmatch(line):
                return False
            libraries += 1
    return bool(libraries)


def _evaluate_molecule_eternafold(
    mol: RNAMoleculeSpec,
    *,
    contrafold_bin: Path,
    ef_params: Path,
) -> tuple[list[MetricResult], list[AdvisoryNote]]:
    """Compute BSD-3-Clause EternaFold v1.3.1 Kozak initiation unpaired accessibility (RC-004)."""
    if local_heavy_binaries_disabled():
        raise PermissionError("Local heavy binary execution (EternaFold) is disabled by host policy.")
    u5 = mol.get_segment_sequence("five_prime_utr")
    cds = mol.get_segment_sequence("cds")
    stop = mol.get_segment_sequence("stop_cassette")
    u5_tail = u5[-35:]
    ctx = (u5_tail + (cds + stop)[:55]).upper().replace("T", "U")
    n_len = len(ctx)
    with tempfile.TemporaryDirectory() as td:
        fa = Path(td) / "in.fa"
        post = Path(td) / "post.txt"
        fa.write_text(f">seq\n{ctx}\n", encoding="utf-8")
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
            return [], []
        if not post.is_file():
            return [], []
        try:
            max_post_bytes = 8 * 1024 * 1024
            if post.stat().st_size > max_post_bytes:
                return [], []
            with post.open("rb") as fh_post:
                raw_post_bytes = fh_post.read(max_post_bytes + 1)
            if len(raw_post_bytes) > max_post_bytes:
                return [], []
            raw_post = raw_post_bytes.decode("utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            return [], []
        if not raw_post or n_len <= 0:
            return [], []
        paired = [0.0] * n_len
        positions = 0
        upper: dict[tuple[int, int], float] = {}
        lower: dict[tuple[int, int], float] = {}
        for line in raw_post:
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) < 2 or not re.fullmatch(r"[0-9]+", parts[0]):
                return [], []
            i = positions
            base_ch = parts[1].upper().replace("T", "U")
            if i >= n_len or parts[0] != str(i + 1) or base_ch != ctx[i]:
                return [], []
            positions += 1
            for tok in parts[2:]:
                if ":" not in tok:
                    return [], []
                j_s, p_s = tok.split(":", 1)
                if not re.fullmatch(r"[1-9][0-9]*", j_s) or len(j_s) > len(str(n_len)):
                    return [], []
                j = int(j_s) - 1
                try:
                    p = float(p_s)
                except ValueError:
                    return [], []
                if not (0 <= j < n_len) or j == i or not math.isfinite(p) or p < 0.0 or p > 1.0:
                    return [], []
                triangle, pair = (upper, (i, j)) if i < j else (lower, (j, i))
                if pair in triangle:
                    return [], []
                triangle[pair] = p
        if positions != n_len or (lower and lower != upper):
            return [], []
        for (i, j), probability in upper.items():
            paired[i] += probability
            paired[j] += probability
        if any(pv > 1.001 for pv in paired):
            return [], []
    unpaired = [max(0.0, min(1.0, 1.0 - pv)) for pv in paired]
    tis_start = max(0, len(u5_tail) - 15)
    tis_end = min(n_len, len(u5_tail) + 18)
    win_vals = unpaired[tis_start:tis_end] or unpaired
    mean_unp = round(sum(win_vals) / max(1, len(win_vals)), 5)
    chem = mol.chemistry_track.uridine_chemistry
    chem_mode = (
        "native_chemistry_validated" if chem == "unmodified_u" else "endogenous_unmodified_proxy"
    )
    advisories: list[AdvisoryNote] = []
    if chem != "unmodified_u":
        probe_guidance = (
            f"Pair with the RC-002 structRFM native SHAPE-MaP probe for '{chem}' when a GPU is available."
            if chem in STRUCTRFM_NATIVE_CHEMISTRIES
            else f"RC-002 structRFM has no native validated SHAPE-MaP probe for '{chem}'."
        )
        advisories.append(
            AdvisoryNote(
                advisory_id=f"adv_eternafold_{chem}_proxy",
                category="chemistry_approximation",
                message=(
                    "EternaFold v1.3.1 (RC-004) posteriors are calibrated on unmodified chemical-mapping data. "
                    + probe_guidance
                ),
            )
        )
    return (
        [
            MetricResult(
                metric_id="eternafold_kozak_unpaired_prob",
                value=mean_unp,
                unit="probability",
                direction="maximize",
                evaluated_scopes=["five_prime_utr", "cds"],
                tool_name="EternaFoldBSD3",
                tool_version=COMPONENT_CATALOG["eternafold_bpp_engine"].version,
                chemistry_evaluation_mode=chem_mode,  # type: ignore[arg-type]
                participated_in="pareto_selection_or_ranking",
            ),
            MetricResult(
                metric_id="eternafold_aup",
                value=round(sum(unpaired) / max(1, len(unpaired)), 5),
                unit="probability",
                direction="informational",
                evaluated_scopes=["five_prime_utr", "cds"],
                tool_name="EternaFoldBSD3",
                tool_version=COMPONENT_CATALOG["eternafold_bpp_engine"].version,
                chemistry_evaluation_mode=chem_mode,  # type: ignore[arg-type]
                participated_in="display_only",
                notes="pairing/unpaired probability [0,1]; strictly separate from kcal/mol MFE",
            ),
        ],
        advisories,
    )


def _wildtype_references_by_index(
    references: dict[int | str, str], intent: DesignIntent,
    reference_bundle: ConstructBundle | None, cached_candidates: list[ConstructBundle] | None,
) -> dict[int, str]:
    if not references:
        raise ValueError("wildtype_cds_reference requires at least one chain key")
    count = len(intent.protein_targets)
    id_indices: dict[str, set[int]] = {}
    bundles = ([reference_bundle] if reference_bundle else []) + (cached_candidates or [])
    for bundle in bundles:
        for idx, (mol, target) in enumerate(zip(bundle.molecules, intent.protein_targets)):
            actual = mol.effective_translation_target
            if (actual.protein_sequence, actual.transl_table) == (target.protein_sequence, target.transl_table):
                id_indices.setdefault(mol.molecule_id, set()).add(idx)
    if not bundles:
        for idx in range(count):
            generated_id = "mol_primary" if count == 1 else f"chain_{idx + 1:02d}"
            id_indices[generated_id] = {idx}

    resolved: dict[int, str] = {}
    for key, cds in references.items():
        indices: set[int] = set()
        if type(key) is int and 0 <= key < count:
            indices.add(key)
        elif isinstance(key, str) and key.strip():
            indices.update(id_indices.get(key, ()))
            if key.isdecimal() and int(key) < count:
                indices.add(int(key))
        if len(indices) != 1:
            problem = "Ambiguous" if indices else "Unrecognized"
            raise ValueError(f"{problem} chain key {key!r} in wildtype_cds_reference; use a molecule ID or index 0–{count - 1}")
        idx = indices.pop()
        if idx in resolved:
            raise ValueError(f"Duplicate chain key {key!r} for index {idx} in wildtype_cds_reference")
        resolved[idx] = cds
    return resolved


def run_design_pipeline(
    intent: DesignIntent,
    *,
    reference_bundle: ConstructBundle | None = None,
    cached_candidates: list[ConstructBundle] | None = None,
    wildtype_cds_reference: str | list[str] | dict[int | str, str] | None = None,
    target_deployment_jurisdictions: list[str] | None = None,
    seed_dir: Path,
    rule_snapshot_version: str | None = None,
    isolated_sandbox_override: bool = False,
    structrfm_scorer: StructRFMNativeScorer | None = None,
    codontransformer_model: Any = None,
    legacy_pure_rerank_mode: bool = False,
    modality: str | None = None,
    use_lineardesign: bool | None = None,
) -> tuple[list[ConstructBundle], EvaluationReport, RunManifest]:
    """Execute Stage 0 + Sliding-Window CDS<->UTR Pareto Optimization with shared scoped constraints."""
    exec_id = f"exec_{uuid.uuid4().hex}"
    att_id = f"att_{exec_id}_001"
    repo_root = seed_dir.parent

    if intent.status != "ready":
        raise ValueError(f"DesignIntent status is '{intent.status}' (must be 'ready' before engine run)")

    effective_modality = modality or (
        reference_bundle.molecules[0].modality
        if (reference_bundle is not None and reference_bundle.molecules)
        else intent.modality
    )
    if effective_modality not in {"linear_mrna", "circrna", "sarna"}:
        raise UnsupportedModalityError(f"Unsupported RNA modality '{effective_modality}'")

    # 1. Fail-closed check when generators are disabled and no valid cache is provided
    if not intent.runtime_policy.enabled_generators:
        if not intent.runtime_policy.allow_cached_candidates or not cached_candidates:
            raise NoAvailableGeneratorError(
                "enabled_generators is empty and no cached_candidates were supplied; refusing silent zero-candidate fallback."
            )

    resolved_rule_snapshot = rule_snapshot_version or (
        "rules_commercial_clean@missing" if intent.compliance_track == "commercial_clean" else "rules_research_v1@1.0.0"
    )

    mfto_seed_path = seed_dir / "method_fto_guard_v1.1.json"
    if not mfto_seed_path.is_file() and intent.compliance_track != "commercial_clean":
        mfto_seed_path = resolve_default_seed_dir() / "method_fto_guard_v1.1.json"
    mfto_load_error: str | None = None
    active_method_fto_guard: MethodFTOGuard | None = None
    if mfto_seed_path.is_file():
        try:
            active_method_fto_guard = MethodFTOGuard.from_json_path(mfto_seed_path)
        except (ValueError, json.JSONDecodeError) as exc:
            mfto_load_error = str(exc)

    # 2. Clean-Room profile readiness check (`commercial_clean` vs `research_max`)
    unavailable_components: dict[str, str] = {}
    if "structrfm_rna_foundation" in intent.runtime_policy.enabled_evaluators:
        reason = structrfm_unavailable_reason(intent.uridine_chemistry)
        if reason:
            unavailable_components["structrfm_rna_foundation"] = reason

    def unavailable(
        reason: str, *, method_fto_blocks: dict[str, str] | None = None,
    ) -> tuple[list[ConstructBundle], EvaluationReport, RunManifest]:
        report = EvaluationReport(
            report_id=f"rep_{exec_id.removeprefix('exec_')}",
            execution_id=exec_id,
            intent_id=intent.intent_id,
            intent_version=intent.intent_version,
            compliance_track=intent.compliance_track,
            outcome_status="UNCOMPARABLE_MISSING_APPROVED_COMPONENT",
            uncomparable_reason=reason,
            recommended_bundle_ids=[],
            candidate_evaluations=[],
            failed_diagnostics=[],
            fto_alerts=[],
            pre_synthesis_advisories=[],
            action_transparency=[
                RequirementActionTrace(
                    requirement_id=c.requirement_id,
                    stage_applied="recorded_only",
                    eliminated_count=0,
                    changed_candidate_ids=[],
                    proof_diff_summary=reason,
                )
                for c in [*intent.constraints, *intent.objectives]
            ],
            created_at_ns=next_report_timestamp_ns(),
            rule_snapshot_version=resolved_rule_snapshot,
            isolated_sandbox_override=isolated_sandbox_override,
        )
        mfto_ver = (
            f"{active_method_fto_guard.registry_id}@{active_method_fto_guard.version}"
            if active_method_fto_guard is not None
            else "method_fto_guard_v1.1@missing"
        )
        mfto_checks = (
            active_method_fto_guard.not_executed_checks(override_blocks=method_fto_blocks)
            if active_method_fto_guard is not None
            else {}
        )
        entries = []
        if intent.compliance_track == "commercial_clean":
            from mrnax.constraints.fto_guard import count_component_gpl_dependencies

            for cid in dict.fromkeys([*intent.runtime_policy.enabled_generators, *intent.runtime_policy.enabled_evaluators]):
                spec = COMPONENT_CATALOG.get(cid)
                if spec is None:
                    continue
                count = count_component_gpl_dependencies(cid, spec.code_license, spec.weights_license)
                if count:
                    entries.append(dict(asset_id=spec.display_name, version=spec.version, role=spec.role,
                                        code_license=spec.code_license, weights_license=spec.weights_license,
                                        data_license=spec.data_license, license_track=spec.license_track,
                                        transitive_gpl_count=count, admission="BLOCK"))
        manifest = RunManifest(
            submission_id=f"sub_{intent.intent_id}",
            execution_id=exec_id,
            attempt_id=att_id,
            lease_epoch=1,
            git_commit_sha=_git_sha(repo_root),
            python_version=sys.version.split()[0],
            platform_info=platform.platform(),
            compliance_track=intent.compliance_track,
            invoked_components=[],
            unavailable_components=unavailable_components,
            rule_snapshot_version=resolved_rule_snapshot,
            isolated_sandbox_override=isolated_sandbox_override,
            prng_state_digest="not_executed",
            input_intent_digest=intent.digest(),
            artifacts=[],
            replay_cli_command=f"uv run mrnax replay --intent-file <intent_{intent.intent_id}.json>",
            honesty_banner="本次未改善生成器" if legacy_pure_rerank_mode else None,
            optimization_mode_audit=MethodFTOAuditRecord(
                optimization_mode="not_executed",
                uses_two_zone_initiation_body_partition=None,
                uses_dfa_cfg_intersection=None,
                sequence_fto_screen_applied=False,
                method_fto_guard_version=mfto_ver,
                certifies_100_percent_fto=False,
                counsel_fto_signoff_required=True,
                target_deployment_jurisdictions=list(
                    target_deployment_jurisdictions
                    or (active_method_fto_guard.jurisdictions if active_method_fto_guard else ["CN", "US", "EP", "JP"])
                ),
                input_origin_type="not_executed",
                region_partition_strategy="not_executed",
                method_fto_boundary_checks=mfto_checks,
                three_layer_dependency_manifest=entries,
            ),
        )
        return [], report, manifest

    if mfto_load_error is not None:
        return unavailable(f"Invalid method_fto_guard_v1.1.json: {mfto_load_error}")
    if intent.compliance_track == "commercial_clean" and active_method_fto_guard is None:
        return unavailable(
            "Required method_fto_guard_v1.1.json is missing; refusing commercial_clean execution without Method-FTO guard."
        )

    if intent.compliance_track == "commercial_clean":
        cc_rules_path = seed_dir / "rules_commercial_v1.json"
        if not cc_rules_path.is_file():
            resolved_rule_snapshot = "rules_commercial_clean@missing"
            return unavailable(
                f"Required commercial rule registry '{cc_rules_path.name}' is missing; refusing fallback or unverified snapshot override."
            )
        try:
            verified_cc_snapshot = RuleRegistry(
                cc_rules_path, track="commercial_clean"
            ).snapshot_version()
        except ValueError as exc:
            resolved_rule_snapshot = "rules_commercial_clean@invalid"
            return unavailable(str(exc))
        if rule_snapshot_version is not None and rule_snapshot_version != verified_cc_snapshot:
            resolved_rule_snapshot = verified_cc_snapshot
            return unavailable(
                f"Requested rule_snapshot_version '{rule_snapshot_version}' does not match verified commercial rule registry '{verified_cc_snapshot}'."
            )
        resolved_rule_snapshot = verified_cc_snapshot
    elif rule_snapshot_version is None:
        resolved_rule_snapshot = "rules_research_v1@1.0.0"
    rule_snapshot_version = resolved_rule_snapshot

    readiness = check_profile_readiness(intent)
    if not readiness.is_runnable:
        return unavailable(readiness.uncomparable_reason or "Profile is unavailable")
    if use_lineardesign is True and intent.compliance_track == "commercial_clean":
        return unavailable(
            "无法比较：commercial_clean 档阻断未获纯商用产品放行的组件 [lineardesign_academic (needs_vendor_license)]"
        )

    if active_method_fto_guard is not None:
        mfto_blocks = active_method_fto_guard.evaluate_pre_execution_intent(intent)
        if mfto_blocks:
            joined = "; ".join(f"{bid}: {msg}" for bid, msg in mfto_blocks.items())
            return unavailable(
                f"无法比较：Method-FTO 护栏阻断 ({joined})",
                method_fto_blocks=mfto_blocks,
            )

    # Resolve & strictly validate per-chain CureVac wild-type CDS references across ALL chains upfront
    from Bio.Seq import Seq

    wildtype_cds_by_chain: dict[int, str] = {}
    for idx, pt in enumerate(intent.protein_targets):
        if pt.wildtype_cds_rna is not None:
            wildtype_cds_by_chain[idx] = pt.wildtype_cds_rna

    declared_wildtype_cds = dict(wildtype_cds_by_chain)
    if wildtype_cds_reference is not None:
        if isinstance(wildtype_cds_reference, str):
            if len(intent.protein_targets) != 1:
                err_note = (
                    f"Multi-chain intent ({len(intent.protein_targets)} chains) requires per-chain CureVac "
                    "wild-type CDS references (list or dict by chain index/molecule_id), not a single string."
                )
                return unavailable(
                    f"无法比较：{err_note}",
                    method_fto_blocks={"CUREVAC_US10898584B2_FAMILY_SYNONYMOUS_CYTOSINE_THRESHOLD": err_note},
                )
            wildtype_cds_by_chain[0] = wildtype_cds_reference
        elif isinstance(wildtype_cds_reference, list):
            if len(wildtype_cds_reference) != len(intent.protein_targets):
                err_note = (
                    f"wildtype_cds_reference list length ({len(wildtype_cds_reference)}) != "
                    f"target chain count ({len(intent.protein_targets)})"
                )
                return unavailable(
                    f"无法比较：{err_note}",
                    method_fto_blocks={"CUREVAC_US10898584B2_FAMILY_SYNONYMOUS_CYTOSINE_THRESHOLD": err_note},
                )
            for idx, wt_s in enumerate(wildtype_cds_reference):
                if wt_s is not None:
                    wildtype_cds_by_chain[idx] = wt_s
        elif isinstance(wildtype_cds_reference, dict):
            try:
                wildtype_cds_by_chain.update(_wildtype_references_by_index(
                    wildtype_cds_reference, intent, reference_bundle, cached_candidates,
                ))
            except ValueError as exc:
                return unavailable(
                    f"无法比较：{exc}",
                    method_fto_blocks={"CUREVAC_US10898584B2_FAMILY_SYNONYMOUS_CYTOSINE_THRESHOLD": str(exc)},
                )

    if reference_bundle is not None:
        wt_prov_re = re.compile(r"(?:^|[^a-z0-9])(?:wild[-_\s]?type|wt)(?:$|[^a-z0-9])", re.IGNORECASE)
        for idx, ref_mol in enumerate(reference_bundle.molecules):
            if idx in wildtype_cds_by_chain or idx >= len(intent.protein_targets):
                continue
            ref_cds_seg = ref_mol.get_segment("cds")
            if ref_cds_seg is not None and (
                wt_prov_re.search(ref_cds_seg.origin_source)
                or wt_prov_re.search(str(ref_cds_seg.component_id or ""))
            ):
                # Note: preset egfp_ref_R component_id is 'cds_u55762_wt_m1psi_clean' with origin_source 'GenBank:U55762.1'
                # Only treat explicit origin_source wild-type declarations as wild-type mRNA references!
                if wt_prov_re.search(ref_cds_seg.origin_source):
                    wildtype_cds_by_chain[idx] = ref_mol.get_segment_sequence("cds")

    for idx, raw_wt in list(wildtype_cds_by_chain.items()):
        pt = intent.protein_targets[idx]
        try:
            wt_norm = normalize_sense_cds(raw_wt, pt.transl_table)
            if idx in declared_wildtype_cds and wt_norm != normalize_sense_cds(
                declared_wildtype_cds[idx], pt.transl_table,
            ):
                raise ValueError("Conflicting wild-type CDS references for the same chain")
            if str(Seq(wt_norm).translate(table=pt.transl_table)) != pt.protein_sequence:
                raise ValueError("Reference CDS translation does not match the target protein")
        except ValueError as exc:
            err_note = f"Chain {idx + 1} CureVac wild-type CDS reference failed frame/length/translation validation: {exc}"
            return unavailable(
                f"无法比较：{err_note}",
                method_fto_blocks={"CUREVAC_US10898584B2_FAMILY_SYNONYMOUS_CYTOSINE_THRESHOLD": err_note},
            )
        wildtype_cds_by_chain[idx] = wt_norm

    mir_v2_path = seed_dir / "mirna_rbp_commercial_v2.json"
    mir_reg = json.loads(
        (mir_v2_path if mir_v2_path.is_file() else (seed_dir / "mirna_rbp_commercial_v1.json")).read_text(
            encoding="utf-8"
        )
    )
    control = intent.cell_expression_control
    if control and control.detarget_tissues and control.protected_tissues:
        try:
            select_detarget_mirna_entry(control.detarget_tissues, control.protected_tissues, mir_reg)
        except MissingTissueMeasurement as exc:
            return unavailable(str(exc))

    use_surrogate_v2 = "surrogate_v2" in intent.runtime_policy.enabled_evaluators
    v2_ready = (
        has_surrogate_v2_artifacts(seed_dir, allow_sandbox_override=True)
        if isolated_sandbox_override
        else has_surrogate_v2_artifacts(seed_dir)
    )
    if use_surrogate_v2 and not v2_ready:
        return unavailable(
            "无法比较：已请求 surrogate_v2 评估器，但当前 seed_dir 未包含通过 SHA-256 与许可证校验的 surrogate_v2 权重清单。"
        )
    surrogate_v2 = (
        (
            CommercialCleanSurrogateV2.from_seed_dir(seed_dir, allow_sandbox_override=True)
            if isolated_sandbox_override
            else CommercialCleanSurrogateV2.from_seed_dir(seed_dir)
        )
        if use_surrogate_v2
        else None
    )
    use_surrogate_v1 = "surrogate_v1_clean" in intent.runtime_policy.enabled_evaluators
    surrogate_v1_only = (
        (
            CommercialCleanSurrogateV1.from_seed_dir(seed_dir, allow_sandbox_override=True)
            if isolated_sandbox_override
            else CommercialCleanSurrogateV1.from_seed_dir(seed_dir)
        )
        if use_surrogate_v1
        else None
    )
    surrogate_v1 = surrogate_v2 if surrogate_v2 is not None else surrogate_v1_only

    use_orthrus = "orthrus_mamba_rna_foundation" in intent.runtime_policy.enabled_evaluators
    active_orthrus_model: Any = None
    active_orthrus_probe: dict[str, Any] | None = None
    if use_orthrus:
        orth_env = os.environ.get("MRNAX_ORTHRUS_CKPT_DIR", "").strip()
        orth_dir = Path(orth_env) if orth_env else None
        if (
            orth_dir is None
            or not (orth_dir / "config.json").is_file()
            or not (orth_dir / "model.safetensors").is_file()
        ):
            missing_orth = (
                "无法比较：已请求 orthrus_mamba_rna_foundation 评估器，但当前节点未挂载有效的 Orthrus-6-track 权重目录 "
                "(需设置 MRNAX_ORTHRUS_CKPT_DIR 包含 config.json 与 model.safetensors；参见 RC-003-orthrus-v1.0.yaml)"
            )
            return unavailable(missing_orth)
        from mrnax.engine.orthrus_portable_pytorch import PortableOrthrusModel
        from mrnax.engine.surrogates import _load_verified_joblib

        active_orthrus_model = PortableOrthrusModel.from_local_checkpoint(
            orth_dir, allow_sandbox_override=isolated_sandbox_override, device="cpu"
        )
        active_orthrus_probe = _load_verified_joblib(
            seed_dir / "surrogate_v1", "orthrus6_saluki_halflife_probe.joblib"
        )

    use_structrfm = "structrfm_rna_foundation" in intent.runtime_policy.enabled_evaluators
    active_structrfm: StructRFMNativeScorer | None = None
    if use_structrfm and "structrfm_rna_foundation" not in unavailable_components:
        if structrfm_scorer is not None:
            from mrnax.engine.surrogates import is_loader_authenticated_structrfm

            if (
                intent.compliance_track == "commercial_clean"
                and not isolated_sandbox_override
                and not is_loader_authenticated_structrfm(structrfm_scorer)
            ):
                return unavailable(
                    "无法比较：commercial_clean 拒绝未经 StructRFMNativeScorer.from_checkpoint_and_seed_dir 发布摘要校验的注入式实例。"
                )
            active_structrfm = structrfm_scorer
        else:
            ckpt_env = os.environ.get("MRNAX_STRUCTRFM_CKPT_DIR", "").strip()
            if not ckpt_env:
                missing_msg = (
                    "无法比较：已请求 structrfm_rna_foundation 评估器，但当前节点未挂载 structRFM GPU 权重 "
                    "(需设置 MRNAX_STRUCTRFM_CKPT_DIR 指向 checkpoints/structRFM_checkpoint 或在 Colab GPU 节点执行)"
                )
                return unavailable(missing_msg)
            active_structrfm = StructRFMNativeScorer.from_checkpoint_and_seed_dir(
                Path(ckpt_env), seed_dir, allow_sandbox_override=isolated_sandbox_override
            )

    requested_ef_ids = [
        cid
        for cid in ("eternafold_bpp_engine", "eternafold_structure_head")
        if cid in intent.runtime_policy.enabled_evaluators
    ]
    use_eternafold = bool(requested_ef_ids)
    active_eternafold_comp_id = requested_ef_ids[0] if requested_ef_ids else "eternafold_bpp_engine"
    active_eternafold_bin: Path | None = None
    active_eternafold_params: Path | None = None
    if use_eternafold:
        if local_heavy_binaries_disabled():
            return unavailable(
                f"无法比较：当前轻量控制面节点已禁用本地 {active_eternafold_comp_id} 重型二进制评估（MRNAX_DISABLE_LOCAL_HEAVY_BINARIES=1）。"
            )
        ef_bin_env = os.environ.get("MRNAX_ETERNAFOLD_BIN", "").strip()
        ef_params_env = os.environ.get("MRNAX_ETERNAFOLD_PARAMS", "").strip()
        ef_bin_cand = Path(ef_bin_env) if ef_bin_env else None
        ef_params_cand = Path(ef_params_env) if ef_params_env else None
        if (
            ef_bin_cand is None
            or not ef_bin_cand.is_file()
            or not os.access(ef_bin_cand, os.X_OK)
            or ef_params_cand is None
            or not ef_params_cand.is_file()
        ):
            missing_ef = (
                f"无法比较：已请求 {active_eternafold_comp_id} 评估器，但当前节点未挂载有效的 BSD-3-Clause EternaFold v1.3.1 二进制与参数文件 "
                "(需设置可执行 MRNAX_ETERNAFOLD_BIN 与 MRNAX_ETERNAFOLD_PARAMS 指向 parameters/EternaFoldParams.v1；参见 RC-004-eternafold-v1.3.1.yaml)"
            )
            return unavailable(missing_ef)
        expected_ef_sha = APPROVED_REPLACEMENT_RELEASES[active_eternafold_comp_id].artifact_digests[
            "parameters/EternaFoldParams.v1"
        ]
        actual_ef_sha = hashlib.sha256(ef_params_cand.read_bytes()).hexdigest()
        if not isolated_sandbox_override:
            if actual_ef_sha != expected_ef_sha:
                return unavailable(
                    f"无法比较：MRNAX_ETERNAFOLD_PARAMS SHA-256 ({actual_ef_sha}) 与 RC-004 冻结摘要 ({expected_ef_sha}) 不一致。"
                )
            expected_bin_sha = os.environ.get("MRNAX_ETERNAFOLD_BIN_SHA256", "").strip().lower()
            actual_bin_sha = hashlib.sha256(ef_bin_cand.read_bytes()).hexdigest()
            if not re.fullmatch(r"[0-9a-f]{64}", expected_bin_sha) or actual_bin_sha != expected_bin_sha:
                return unavailable(
                    "无法比较：非沙盒模式下启用 EternaFold 必须提供匹配的 MRNAX_ETERNAFOLD_BIN_SHA256 可执行文件摘要校验。"
                )
            try:
                link_cmd = ["otool", "-L", str(ef_bin_cand)] if os.uname().sysname == "Darwin" else ["ldd", str(ef_bin_cand)]
                link_out = subprocess.check_output(link_cmd, text=True, stderr=subprocess.STDOUT, timeout=10)
                if "vienna" in link_out.lower() or "librna" in link_out.lower():
                    return unavailable("无法比较：检测到 EternaFold 二进制动态链接了 ViennaRNA/libRNA，拒绝运行。")
                if link_cmd[0] == "otool" and not _valid_otool_library_listing(link_out, ef_bin_cand):
                    return unavailable("无法比较：EternaFold 的 otool -L 输出缺少有效的 Mach-O 动态库清单。")
            except (subprocess.SubprocessError, OSError, UnicodeDecodeError):
                return unavailable("无法比较：无法完成 EternaFold 二进制动态库纯净性 (otool/ldd) 核验。")
        active_eternafold_bin = ef_bin_cand
        active_eternafold_params = ef_params_cand

    use_codontransformer = "codontransformer_cds_baseline" in intent.runtime_policy.enabled_generators
    active_codontransformer: Any = None
    if use_codontransformer:
        from mrnax.engine.codontransformer_loader import (
            CodonTransformerNativeModel,
            is_loader_authenticated_codontransformer,
        )

        if codontransformer_model is not None:
            if (
                intent.compliance_track == "commercial_clean"
                and not isolated_sandbox_override
                and not is_loader_authenticated_codontransformer(codontransformer_model)
            ):
                return unavailable(
                    "无法比较：commercial_clean 拒绝未经 CodonTransformerNativeModel.from_local_checkpoint RC-005 SHA-256 认证的注入式实例。"
                )
            active_codontransformer = codontransformer_model
        else:
            ct_env = os.environ.get("MRNAX_CODONTRANSFORMER_CKPT_DIR", "").strip()
            if not ct_env:
                return unavailable(
                    "无法比较：已请求 codontransformer_cds_baseline 生成器，但当前节点未挂载 RC-005 CodonTransformer 权重目录 "
                    "(需设置 MRNAX_CODONTRANSFORMER_CKPT_DIR 指向 adibvafa/CodonTransformer@9744dcc920d8；参见 RC-005-codontransformer-v1.0.yaml)"
                )
            try:
                active_codontransformer = CodonTransformerNativeModel.from_local_checkpoint(
                    Path(ct_env), allow_sandbox_override=isolated_sandbox_override
                )
            except (FileNotFoundError, ValueError) as exc:
                return unavailable(f"无法比较：RC-005 CodonTransformer 权重校验失败：{exc}")

    utr_v1_reg = json.loads((seed_dir / "utr_components_v1.json").read_text(encoding="utf-8"))
    utr_v2_path = seed_dir / "utr_components_commercial_v2.json"
    if (use_surrogate_v1 or intent.compliance_track == "commercial_clean") and utr_v2_path.is_file():
        utr_v2_reg = json.loads(utr_v2_path.read_text(encoding="utf-8"))
        seen_u5: set[str] = set()
        merged_u5 = []
        for u in [*utr_v1_reg["five_prime_utrs"], *utr_v2_reg["five_prime_utrs"]]:
            if u["sequence"] not in seen_u5:
                seen_u5.add(u["sequence"])
                merged_u5.append(u)
        seen_u3: set[str] = set()
        merged_u3 = []
        for u in [*utr_v1_reg["three_prime_utrs"], *utr_v2_reg["three_prime_utrs"]]:
            if u["sequence"] not in seen_u3:
                seen_u3.add(u["sequence"])
                merged_u3.append(u)
        utr_reg = {"five_prime_utrs": merged_u5, "three_prime_utrs": merged_u3}
    else:
        utr_reg = utr_v1_reg
    constraint_engine = ScopedConstraintEngine.from_seed_dir(seed_dir)

    rng = random.Random(intent.runtime_policy.random_seed)
    prng_digest = hashlib.sha256(str(rng.getstate()).encode("utf-8")).hexdigest()

    protein_seq = intent.protein_targets[0].protein_sequence
    clean_utr5s = [
        u for u in utr_v1_reg["five_prime_utrs"] if not u["has_uaug"] and u["max_homopolymer_run"] <= 7
    ]
    clean_utr3s = [u for u in utr_v1_reg["three_prime_utrs"] if u["max_homopolymer_run"] <= 7]
    if intent.scaffold_id == "cap1_minimal_structured_3utr":
        if len(clean_utr5s) > 1:
            clean_utr5s = sorted(clean_utr5s, key=lambda u: len(str(u.get("sequence", ""))))
        if len(clean_utr3s) > 1:
            clean_utr3s = sorted(
                clean_utr3s,
                key=lambda u: 0 if "hbb" in str(u.get("component_id", "")).lower() else 1,
            )

    # Respect locked scopes
    locked_utr5 = intent.locked_scopes.get("five_prime_utr")
    locked_cds = intent.locked_scopes.get("cds")
    locked_utr3 = intent.locked_scopes.get("three_prime_utr")
    locked_polya = intent.locked_scopes.get("poly_a")

    if reference_bundle is not None:
        ref_mol = reference_bundle.molecules[0]
        default_utr5 = ref_mol.get_segment_sequence("five_prime_utr")
        default_cds = ref_mol.get_segment_sequence("cds")
        default_stop = ref_mol.get_segment_sequence("stop_cassette") or "UAA"
        default_utr3 = ref_mol.get_segment_sequence("three_prime_utr")
        default_polya = ref_mol.get_segment_sequence("poly_a") or ("A" * 100)
    else:
        default_utr5 = clean_utr5s[0]["sequence"]
        default_cds = ""
        default_stop = "UGA"
        default_utr3 = clean_utr3s[0]["sequence"]
        default_polya = "A" * 100

    if stop_lock := intent.locked_scopes.get("stop_cassette"):
        default_stop = stop_lock.locked_value
    polya_seq = locked_polya.locked_value if locked_polya else default_polya

    library_origins = {}
    for end in ("five_prime_utrs", "three_prime_utrs"):
        for row in utr_reg[end]:
            origin = _Origin(row["component_id"], row["origin_source"], row["license_track"])
            if library_origins.setdefault(origin.component_id, origin) != origin:
                raise ValueError(f"Conflicting library provenance for {origin.component_id}")

    # Every new chain uses the primary input's shared UTRs, stop and tail.
    primary = reference_bundle.molecules[0] if reference_bundle else None

    def inherited(scope: SegmentScope, default: _Origin | None = None) -> _Origin:
        ref = primary.get_segment(scope) if primary else None
        lock = intent.locked_scopes.get(scope)
        if isinstance(lock, BundleSegmentLock):
            source = lock.source_segment
            return _Origin(source.component_id, source.origin_source, source.license_track)
        if lock and (ref is None or primary.get_segment_sequence(scope) != lock.locked_value):
            return _Origin(None, f"user_lock:{lock.locked_at_turn_id}", "commercial_clearance_pending")
        if ref is not None:
            return _Origin(ref.component_id, ref.origin_source, ref.license_track)
        if default is None:
            raise ValueError(f"Missing origin for inherited segment {scope}")
        return default

    def origins_for(
        utr5_id: str | None, utr3_id: str | None, cds_source: str, annotations: list[FeatureAnnotation],
        *, minimal: bool = False, secondary: bool = False, cds_tier: LicenseTrack = "commercial_clean",
    ) -> dict[SegmentScope, _Origin]:
        u3 = inherited("three_prime_utr") if minimal or locked_utr3 else library_origins[utr3_id]
        if annotations:
            cassette = annotations[0].qualifiers
            tiers = [u3.tier, *cassette["mrnax_license_track"]]
            tier = max(tiers, key=get_args(LicenseTrack).index)
            u3 = _Origin(u3.component_id, f"{u3.source} + {cassette['mrnax_origin_source'][0]}", tier)
        return {
            "five_prime_utr": inherited("five_prime_utr") if minimal or locked_utr5 else library_origins[utr5_id],
            "three_prime_utr": u3,
            "cds": inherited("cds") if (minimal or locked_cds) and not secondary
                   else _Origin(None, cds_source, cds_tier),
            "stop_cassette": inherited("stop_cassette", _Origin(None, "mrnax:standard", "commercial_clean")),
            "poly_a": inherited("poly_a", _Origin(f"polya_a{len(polya_seq)}", "mrnax:homopolymer_tail", "commercial_clean")),
        }

    # Prepare optional CellExpressionController 3'UTR cassette
    mirna_mature: str | None = None
    def _prepare_utr3(raw_u3: str) -> tuple[str, list[FeatureAnnotation]]:
        nonlocal mirna_mature
        if locked_utr3:
            return locked_utr3.locked_value, []
        if intent.cell_expression_control and intent.cell_expression_control.detarget_tissues:
            assembled, anns, mir_mat = assemble_detargeted_3utr(
                base_utr3=raw_u3,
                detarget_tissues=intent.cell_expression_control.detarget_tissues,
                protected_tissues=intent.cell_expression_control.protected_tissues,
                mirna_registry=mir_reg,
                copies=intent.cell_expression_control.mirna_copy_range[0],
                utr3_global_start=0,
            )
            mirna_mature = mir_mat
            return assembled, anns
        return raw_u3, []

    raw_candidates: list[ConstructBundle] = list(cached_candidates or [])
    comp_records: list[ComponentInvocationRecord] = []
    exec_tag = exec_id.removeprefix("exec_")

    used_endogenous_atlas = False
    used_two_zone_partition = False

    def mutate_with_atlas(cds: str, protein: str, random_source: random.Random, **options: Any) -> str:
        nonlocal used_endogenous_atlas
        used_endogenous_atlas = True
        return mutate_cds_region_synonymously(cds, protein, random_source, **options)

    def propose_full_cds_with_atlas(cds: str, protein: str, random_source: random.Random, **options: Any) -> str:
        nonlocal used_endogenous_atlas
        used_endogenous_atlas = True
        return propose_non_partitioned_bicodon_cds(cds, protein, random_source, **options)

    from mrnax.constraints.sequence_gates import RESTRICTION_SITES_RNA

    utr5_enzyme_motifs: list[str] = []
    cds_enzyme_motifs: list[str] = []
    utr3_enzyme_motifs: list[str] = []
    full_tx_enzyme_motifs: list[str] = []
    for cspec in intent.constraints:
        if cspec.constraint_kind == "type_iis_restriction_free":
            motifs_c: list[str] = []
            for enz in cspec.params.get("enzymes", []):
                if enz in RESTRICTION_SITES_RNA:
                    fwd_e, rev_e = RESTRICTION_SITES_RNA[enz]
                    motifs_c.extend((fwd_e, rev_e))
            for raw_m in cspec.params.get("forbidden_motifs", []):
                if raw_m:
                    motifs_c.append(str(raw_m).upper().replace("T", "U"))
            if "full_transcript" in cspec.target_scopes:
                full_tx_enzyme_motifs.extend(motifs_c)
                utr5_enzyme_motifs.extend(motifs_c)
                cds_enzyme_motifs.extend(motifs_c)
                utr3_enzyme_motifs.extend(motifs_c)
            else:
                if "five_prime_utr" in cspec.target_scopes:
                    utr5_enzyme_motifs.extend(motifs_c)
                if "cds" in cspec.target_scopes:
                    cds_enzyme_motifs.extend(motifs_c)
                if "three_prime_utr" in cspec.target_scopes:
                    utr3_enzyme_motifs.extend(motifs_c)
    destab_rbp_motifs, stab_rbp_motifs = _load_rbp_motif_sets(seed_dir)

    def _is_utr5_enzyme_clean(seq: str) -> bool:
        return not any(pat in seq for pat in utr5_enzyme_motifs)

    def _is_utr3_enzyme_clean(seq: str) -> bool:
        return not any(pat in seq for pat in utr3_enzyme_motifs)

    sec_s0_seeds: dict[int, str] = {}
    sec_dc_seeds: dict[int, str] = {}
    stage0_cds_calls = 0
    dnachisel_cds_calls = 0
    pareto_cds_calls = 0

    def _make_secondary_chains(
        *,
        utr5_seq: str,
        utr5_id: str | None,
        utr5_action: str,
        utr3_seq: str,
        utr3_id: str | None,
        utr3_action: str,
        extra_anns: list[FeatureAnnotation],
        use_dnachisel: bool = False,
        seed_offset: int = 0,
    ) -> list[RNAMoleculeSpec]:
        nonlocal stage0_cds_calls, dnachisel_cds_calls, pareto_cds_calls
        if len(intent.protein_targets) <= 1:
            return []
        sec_mols: list[RNAMoleculeSpec] = []
        for ch_idx, pt in enumerate(intent.protein_targets[1:], start=2):
            sec_rng = random.Random(intent.runtime_policy.random_seed + 1009 * ch_idx + seed_offset)
            if use_dnachisel:
                sec_cds = generate_dnachisel_cds(
                    pt.protein_sequence,
                    intent.constraints,
                    seed=intent.runtime_policy.random_seed + ch_idx + seed_offset,
                    chemistry=intent.uridine_chemistry,
                )
                dnachisel_cds_calls += 1
                if seed_offset == 0:
                    sec_dc_seeds[ch_idx] = sec_cds
                sec_src = "mrnax:dnachisel_v3.2.16"
                sec_act = "generated_novo"
            elif seed_offset > 0:
                rand_sec = generate_random_synonymous_cds(
                    pt.protein_sequence,
                    sec_rng,
                    chemistry=intent.uridine_chemistry,
                    slowdown_codon_ranges=intent.co_translational_slowdown_codons,
                )
                pareto_cds_calls += 1
                dc_sec_base = sec_dc_seeds.get(ch_idx) or sec_s0_seeds.get(ch_idx) or rand_sec
                sec_trials = [rand_sec] if legacy_pure_rerank_mode else [dc_sec_base, rand_sec]
                cur_sec = rand_sec
                if len(pt.protein_sequence) > 15:
                    for t_i in range(3):
                        if legacy_pure_rerank_mode:
                            cur_sec = mutate_with_atlas(
                                cur_sec,
                                pt.protein_sequence,
                                sec_rng,
                                codon_start=10,
                                codon_end=max(11, len(pt.protein_sequence) - 5),
                                prefer_open_au=False,
                                chemistry=intent.uridine_chemistry,
                                legacy_single_boundary=True,
                            )
                        else:
                            tgt_gc = (0.45, 0.53, 0.58)[t_i % 3]
                            mut_p = (0.95, 0.55, 0.65)[t_i % 3]
                            seed_in = rand_sec if t_i == 0 else (dc_sec_base if t_i == 1 else cur_sec)
                            cur_sec = propose_full_cds_with_atlas(
                                seed_in,
                                pt.protein_sequence,
                                sec_rng,
                                chemistry=intent.uridine_chemistry,
                                constraints=intent.constraints,
                                slowdown_codon_ranges=intent.co_translational_slowdown_codons,
                                utr5_tail=utr5_seq,
                                stop_head=default_stop + utr3_seq[:9],
                                gc_weight=0.012 + 0.004 * t_i,
                                temperature=0.08 + 0.02 * t_i,
                                target_gc=tgt_gc,
                                mutation_prob=mut_p,
                            )
                        pareto_cds_calls += 1
                        sec_trials.append(cur_sec)

                if surrogate_v1 is not None and not legacy_pure_rerank_mode:
                    s_stabs = surrogate_v1.score_cds_stability(sec_trials)
                    s_hls = surrogate_v1.score_consensus_halflife_batch(
                        [(utr5_seq, tb, utr3_seq) for tb in sec_trials]
                    )
                    slot_focus = (seed_offset - 1) % 3
                    if slot_focus == 0:
                        sec_cds = sec_trials[max(range(len(sec_trials)), key=lambda i: float(s_hls[i]))]
                    elif slot_focus == 1:
                        sec_cds = sec_trials[max(range(len(sec_trials)), key=lambda i: float(s_stabs[i]))]
                    else:
                        s_vecs = [[float(s_stabs[i]), float(s_hls[i])] for i in range(len(sec_trials))]
                        _, s_ord = _rank_and_order_pareto_candidates(s_vecs)
                        sec_cds = sec_trials[s_ord[0]]
                else:
                    sec_cds = sec_trials[-1]
                sec_src = "mrnax:pareto_multi_chain"
                sec_act = "synonymous_mutated"
            else:
                sec_cds = generate_random_synonymous_cds(
                    pt.protein_sequence,
                    sec_rng,
                    chemistry=intent.uridine_chemistry,
                )
                stage0_cds_calls += 1
                sec_s0_seeds[ch_idx] = sec_cds
                sec_src = "mrnax:stage0_random"
                sec_act = "generated_novo"
            sec_mols.append(
                _build_molecule_spec(
                    molecule_id=f"chain_{ch_idx:02d}",
                    target=pt,
                    intent=intent,
                    origins=origins_for(utr5_id, utr3_id, sec_src, extra_anns, secondary=True),
                    utr5_seq=utr5_seq,
                    utr5_action=utr5_action,
                    cds_seq=sec_cds,
                    cds_action=sec_act,
                    stop_seq=default_stop,
                    utr3_seq=utr3_seq,
                    utr3_action=utr3_action,
                    polya_seq=polya_seq,
                    extra_utr3_annotations=extra_anns,
                )
            )
        return sec_mols

    # Include reference R verbatim or minimally adapted A when reference_bundle provided
    if reference_bundle is not None:
        raw_candidates.append(reference_bundle)
        if "minimal_change_vs_r" in intent.portfolio_tracks:

            min_cds = locked_cds.locked_value if locked_cds else default_cds
            if not locked_cds and intent.uridine_chemistry in {"m1psi", "pseudouridine"}:
                min_cds, _ = remediate_m1psi_slippery_runs(min_cds)
            u3_min, anns_min = _prepare_utr3(default_utr3)
            raw_candidates.append(
                _build_single_molecule_bundle(
                    bundle_id=f"bnd_{exec_tag}_minA",
                    parent=reference_bundle,
                    derivation_action="locked_refinement",
                    intent=intent,
                    origins=origins_for(None, None, "", anns_min, minimal=True),
                    utr5_seq=locked_utr5.locked_value if locked_utr5 else default_utr5,
                    utr5_action="locked_verbatim",
                    cds_seq=min_cds,
                    cds_action="locked_verbatim" if locked_cds else "synonymous_mutated",
                    stop_seq=default_stop,
                    utr3_seq=u3_min,
                    utr3_action="locked_verbatim" if not anns_min else "selected_from_library",
                    polya_seq=polya_seq,
                    extra_utr3_annotations=anns_min,
                    secondary_molecules=list(reference_bundle.molecules[1:]) if len(reference_bundle.molecules) > 1 else None,
                )
            )

    # Resolve LinearDesign generator eligibility once across all modalities, honoring explicit disablement.
    explicit_ld_requested = (use_lineardesign is not False) and (
        any(
            g in {"lineardesign_academic", "sota_open", "lineardesign"}
            for g in intent.runtime_policy.enabled_generators
        )
        or (use_lineardesign is True)
    )
    auto_ld_research = (
        use_lineardesign is None
        and intent.compliance_track == "research_max"
        and "viennarna_thermo" in intent.runtime_policy.enabled_evaluators
        and "pareto_nsga2" in intent.runtime_policy.enabled_generators
        and not locked_cds
        and not legacy_pure_rerank_mode
        and os.environ.get("MRNAX_DISABLE_AUTO_LINEARDESIGN") != "1"
    )

    # Multi-Modality Branch (`circrna` and `sarna` driven by modality_scaffolds_v1.json)
    if effective_modality in {"circrna", "sarna"} and intent.runtime_policy.enabled_generators:
        from mrnax.engine.modalities import build_circrna_bundle, build_sarna_bundle

        t0_mod = time.perf_counter()
        use_ld_mod = (
            intent.compliance_track != "commercial_clean"
            and (use_lineardesign is not False)
            and (not legacy_pure_rerank_mode)
            and (explicit_ld_requested or auto_ld_research)
        )
        v20_mod = (
            (use_lineardesign is False)
            or legacy_pure_rerank_mode
            or (
                not use_ld_mod
                and "pareto_nsga2" not in intent.runtime_policy.enabled_generators
            )
        )
        mod_built: list[ConstructBundle] = []
        scaf_choice = (intent.scaffold_id or "").lower()
        if effective_modality == "circrna":
            pie_bb = "ANABAENA" if "anabaena" in scaf_choice else "T4_TD"
            if "cvb3" in scaf_choice:
                circ_plan: tuple[tuple[str, float], ...] = (("CVB3", 1.0), ("CVB3", 2.0))
            elif "hrvb3" in scaf_choice or "hrv_b3" in scaf_choice:
                circ_plan = (("HRV-B3", 1.0), ("HRV-B3", 2.0))
            else:
                circ_plan = (("HRV-B3", 1.0), ("EV-B107", 1.0), ("CVB3", 2.0))
            for c_idx, (ires_nm, lam_v) in enumerate(circ_plan, start=1):
                ires_tag = ires_nm.lower().replace("-", "")
                lam_tag = f"_lam{lam_v:g}" if use_ld_mod else ""
                bnd_c = build_circrna_bundle(
                    intent.protein_targets[0],
                    bundle_id=f"bnd_{exec_tag}_circ_{pie_bb.lower()}_{ires_tag}{lam_tag}_{c_idx:02d}",
                    ires_name=ires_nm,
                    pie_backbone=pie_bb,
                    compliance_track=intent.compliance_track,
                    use_lineardesign=use_ld_mod,
                    lambda_val=lam_v,
                    baseline_v20_mode=v20_mod,
                    seed=intent.runtime_policy.random_seed + c_idx,
                    seed_dir=seed_dir,
                )
                raw_candidates.append(bnd_c)
                mod_built.append(bnd_c)
        else:
            if "sinv" in scaf_choice:
                sarna_plan: tuple[tuple[str, float], ...] = (("SINV", 1.0), ("SINV", 2.0))
            elif "veev" in scaf_choice:
                sarna_plan = (("VEEV_TC83", 1.0), ("VEEV_TC83", 2.0))
            else:
                sarna_plan = (("VEEV_TC83", 1.0), ("VEEV_TC83", 2.0), ("SINV", 1.0))
            for s_idx, (bb_nm, lam_v) in enumerate(sarna_plan, start=1):
                bb_tag = "veev_tc83" if "veev" in bb_nm.lower() else "sinv"
                lam_tag = f"_lam{lam_v:g}" if use_ld_mod else ""
                bnd_s = build_sarna_bundle(
                    intent.protein_targets[0],
                    bundle_id=f"bnd_{exec_tag}_sarna_{bb_tag}{lam_tag}_{s_idx:02d}",
                    replicon_backbone=bb_nm,
                    compliance_track=intent.compliance_track,
                    use_lineardesign=use_ld_mod,
                    lambda_val=lam_v,
                    baseline_v20_mode=v20_mod,
                    seed=intent.runtime_policy.random_seed + s_idx,
                    seed_dir=seed_dir,
                )
                raw_candidates.append(bnd_s)
                mod_built.append(bnd_s)
        mod_elapsed = round(time.perf_counter() - t0_mod, 5)
        if use_ld_mod:
            ld_mod_ok = sum(
                1
                for b in mod_built
                if (seg := b.molecules[0].get_segment("cds")) is not None
                and seg.origin_source.startswith("mrnax:lineardesign_polisher")
            )
            ld_mod_fb = len(mod_built) - ld_mod_ok
            spec_ld_mod = COMPONENT_CATALOG["lineardesign_academic"]
            comp_records.append(
                ComponentInvocationRecord(
                    name=spec_ld_mod.display_name,
                    version=spec_ld_mod.version,
                    role="generator",
                    code_license=spec_ld_mod.code_license,
                    weights_license=spec_ld_mod.weights_license,
                    data_license=spec_ld_mod.data_license,
                    license_track=spec_ld_mod.license_track,
                    wall_seconds=mod_elapsed,
                    attempt_count=len(mod_built),
                    failure_count=ld_mod_fb,
                )
            )
            if ld_mod_fb > 0:
                spec_fb_mod = COMPONENT_CATALOG["pareto_nsga2"]
                comp_records.append(
                    ComponentInvocationRecord(
                        name=spec_fb_mod.display_name,
                        version=spec_fb_mod.version,
                        role="generator",
                        code_license=spec_fb_mod.code_license,
                        weights_license=spec_fb_mod.weights_license,
                        data_license=spec_fb_mod.data_license,
                        license_track="needs_vendor_license",
                        wall_seconds=mod_elapsed,
                        attempt_count=ld_mod_fb,
                        failure_count=0,
                    )
                )
        else:
            spec_clean_mod = COMPONENT_CATALOG["dnachisel_cds" if v20_mod else "pareto_nsga2"]
            comp_records.append(
                ComponentInvocationRecord(
                    name=spec_clean_mod.display_name,
                    version=spec_clean_mod.version,
                    role="generator",
                    code_license=spec_clean_mod.code_license,
                    weights_license=spec_clean_mod.weights_license,
                    data_license=spec_clean_mod.data_license,
                    license_track=spec_clean_mod.license_track,
                    wall_seconds=mod_elapsed,
                    attempt_count=len(mod_built),
                    failure_count=0,
                )
            )

    # Generator 1: Stage 0 random synonymous baseline / computational reference
    if effective_modality == "linear_mrna" and "stage0_random" in intent.runtime_policy.enabled_generators:
        t0 = time.perf_counter()
        if locked_cds:
            cds_rand = locked_cds.locked_value
        else:
            cds_rand = generate_random_synonymous_cds(
                protein_seq,
                rng,
                chemistry=intent.uridine_chemistry,
                slowdown_codon_ranges=intent.co_translational_slowdown_codons,
            )
            stage0_cds_calls += 1
        u5 = locked_utr5.locked_value if locked_utr5 else clean_utr5s[0]["sequence"]
        u3, anns_u3 = _prepare_utr3(clean_utr3s[0]["sequence"])
        u5_id_s0 = None if locked_utr5 else clean_utr5s[0]["component_id"]
        u5_act_s0 = "locked_verbatim" if locked_utr5 else "selected_from_library"
        u3_id_s0 = None if locked_utr3 else clean_utr3s[0]["component_id"]
        u3_act_s0 = "locked_verbatim" if locked_utr3 else "selected_from_library"
        raw_candidates.append(
            _build_single_molecule_bundle(
                bundle_id=f"bnd_{exec_tag}_stage0_ref",
                parent=reference_bundle,
                derivation_action="computational_reference" if reference_bundle is None else "de_novo_design",
                intent=intent,
                origins=origins_for(u5_id_s0, u3_id_s0, "mrnax:stage0_random", anns_u3),
                utr5_seq=u5,
                utr5_action=u5_act_s0,
                cds_seq=cds_rand,
                cds_action="locked_verbatim" if locked_cds else "generated_novo",
                stop_seq=default_stop,
                utr3_seq=u3,
                utr3_action=u3_act_s0,
                polya_seq=polya_seq,
                extra_utr3_annotations=anns_u3,
                secondary_molecules=_make_secondary_chains(
                    utr5_seq=u5,
                    utr5_id=u5_id_s0,
                    utr5_action=u5_act_s0,
                    utr3_seq=u3,
                    utr3_id=u3_id_s0,
                    utr3_action=u3_act_s0,
                    extra_anns=anns_u3,
                ),
            )
        )
        spec_s0 = COMPONENT_CATALOG["stage0_random"]
        comp_records.append(
            ComponentInvocationRecord(
                name=spec_s0.display_name,
                version=spec_s0.version,
                role="generator",
                code_license=spec_s0.code_license,
                weights_license=spec_s0.weights_license,
                data_license=spec_s0.data_license,
                license_track=spec_s0.license_track,
                wall_seconds=round(time.perf_counter() - t0, 5),
                attempt_count=stage0_cds_calls,
                failure_count=0,
            )
        )

    # Generator 2: DNA Chisel constraint/codon baseline
    dc_seed_cds = locked_cds.locked_value if locked_cds else ""
    dc_seed_source = "mrnax:locked_cds" if locked_cds else "mrnax:dnachisel_v3.2.16"
    dc_seed_tier: LicenseTrack = inherited("cds").tier if locked_cds else "commercial_clean"
    if effective_modality == "linear_mrna" and "dnachisel_cds" in intent.runtime_policy.enabled_generators:
        t0 = time.perf_counter()
        if not dc_seed_cds:
            dc_seed_cds = generate_dnachisel_cds(
                protein_seq,
                intent.constraints,
                seed=intent.runtime_policy.random_seed,
                chemistry=intent.uridine_chemistry,
            )
            dc_seed_source = "mrnax:dnachisel_v3.2.16"
            dc_seed_tier = "commercial_clean"
            dnachisel_cds_calls += 1
        u5 = locked_utr5.locked_value if locked_utr5 else clean_utr5s[0]["sequence"]
        u3, anns_u3 = _prepare_utr3(clean_utr3s[0]["sequence"])
        u5_id_dc = None if locked_utr5 else clean_utr5s[0]["component_id"]
        u5_act_dc = "locked_verbatim" if locked_utr5 else "selected_from_library"
        u3_id_dc = None if locked_utr3 else clean_utr3s[0]["component_id"]
        u3_act_dc = "locked_verbatim" if locked_utr3 else "selected_from_library"
        raw_candidates.append(
            _build_single_molecule_bundle(
                bundle_id=f"bnd_{exec_tag}_dnachisel",
                parent=reference_bundle,
                derivation_action="de_novo_design",
                intent=intent,
                origins=origins_for(u5_id_dc, u3_id_dc, "mrnax:dnachisel_v3.2.16", anns_u3),
                utr5_seq=u5,
                utr5_action=u5_act_dc,
                cds_seq=dc_seed_cds,
                cds_action="locked_verbatim" if locked_cds else "generated_novo",
                stop_seq=default_stop,
                utr3_seq=u3,
                utr3_action=u3_act_dc,
                polya_seq=polya_seq,
                extra_utr3_annotations=anns_u3,
                secondary_molecules=_make_secondary_chains(
                    utr5_seq=u5,
                    utr5_id=u5_id_dc,
                    utr5_action=u5_act_dc,
                    utr3_seq=u3,
                    utr3_id=u3_id_dc,
                    utr3_action=u3_act_dc,
                    extra_anns=anns_u3,
                    use_dnachisel=True,
                ),
            )
        )
        spec_dc = COMPONENT_CATALOG["dnachisel_cds"]
        comp_records.append(
            ComponentInvocationRecord(
                name=spec_dc.display_name,
                version=spec_dc.version,
                role="generator",
                code_license=spec_dc.code_license,
                weights_license=spec_dc.weights_license,
                data_license=spec_dc.data_license,
                license_track=spec_dc.license_track,
                wall_seconds=round(time.perf_counter() - t0, 5),
                attempt_count=dnachisel_cds_calls,
                failure_count=0,
            )
        )

    # Generator 2B: Apache-2.0 CodonTransformer v1.0 Homo sapiens Synonymous CDS Baseline (RC-005)
    ct_seed_cds = ""
    if effective_modality == "linear_mrna" and active_codontransformer is not None:
        t0_ct = time.perf_counter()
        ct_calls = 0
        if locked_cds:
            ct_seed_cds = locked_cds.locked_value
        else:
            ct_seed_cds, _ = active_codontransformer.generate_and_score_cds(
                protein_seq,
                uridine_chemistry=intent.uridine_chemistry,
                deterministic=True,
                seed=intent.runtime_policy.random_seed,
            )
            ct_calls += 1
            if not dc_seed_cds:
                dc_seed_cds = ct_seed_cds
                dc_seed_source = "mrnax:codontransformer_v1.0.0"
                dc_seed_tier = COMPONENT_CATALOG["codontransformer_cds_baseline"].license_track
        u5_ct = locked_utr5.locked_value if locked_utr5 else clean_utr5s[0]["sequence"]
        u3_ct, anns_u3_ct = _prepare_utr3(clean_utr3s[0]["sequence"])
        u5_id_ct = None if locked_utr5 else clean_utr5s[0]["component_id"]
        u5_act_ct = "locked_verbatim" if locked_utr5 else "selected_from_library"
        u3_id_ct = None if locked_utr3 else clean_utr3s[0]["component_id"]
        u3_act_ct = "locked_verbatim" if locked_utr3 else "selected_from_library"
        sec_ct_mols: list[RNAMoleculeSpec] = []
        for ch_idx, pt in enumerate(intent.protein_targets[1:], start=2):
            sec_ct_cds, _ = active_codontransformer.generate_and_score_cds(
                pt.protein_sequence,
                uridine_chemistry=intent.uridine_chemistry,
                deterministic=True,
                seed=intent.runtime_policy.random_seed + ch_idx,
            )
            ct_calls += 1
            sec_dc_seeds.setdefault(ch_idx, sec_ct_cds)
            sec_ct_mols.append(
                _build_molecule_spec(
                    molecule_id=f"chain_{ch_idx:02d}",
                    target=pt,
                    intent=intent,
                    origins=origins_for(
                        u5_id_ct, u3_id_ct, "mrnax:codontransformer_v1.0.0", anns_u3_ct, secondary=True
                    ),
                    utr5_seq=u5_ct,
                    utr5_action=u5_act_ct,
                    cds_seq=sec_ct_cds,
                    cds_action="generated_novo",
                    stop_seq=default_stop,
                    utr3_seq=u3_ct,
                    utr3_action=u3_act_ct,
                    polya_seq=polya_seq,
                    extra_utr3_annotations=anns_u3_ct,
                )
            )
        raw_candidates.append(
            _build_single_molecule_bundle(
                bundle_id=f"bnd_{exec_tag}_codontransformer",
                parent=reference_bundle,
                derivation_action="de_novo_design",
                intent=intent,
                origins=origins_for(u5_id_ct, u3_id_ct, "mrnax:codontransformer_v1.0.0", anns_u3_ct),
                utr5_seq=u5_ct,
                utr5_action=u5_act_ct,
                cds_seq=ct_seed_cds,
                cds_action="locked_verbatim" if locked_cds else "generated_novo",
                stop_seq=default_stop,
                utr3_seq=u3_ct,
                utr3_action=u3_act_ct,
                polya_seq=polya_seq,
                extra_utr3_annotations=anns_u3_ct,
                secondary_molecules=sec_ct_mols or None,
            )
        )
        spec_ct = COMPONENT_CATALOG["codontransformer_cds_baseline"]
        comp_records.append(
            ComponentInvocationRecord(
                name=spec_ct.display_name,
                version=spec_ct.version,
                role="generator",
                code_license=spec_ct.code_license,
                weights_license=spec_ct.weights_license,
                data_license=spec_ct.data_license,
                license_track=spec_ct.license_track,
                wall_seconds=round(time.perf_counter() - t0_ct, 5),
                attempt_count=ct_calls,
                failure_count=0,
            )
        )

    # Generator 2C: LinearDesign Lattice-CFG + 5'-Opening Surgical Polisher (`research_max` / `sota_open` ONLY)
    if (
        effective_modality == "linear_mrna"
        and intent.compliance_track != "commercial_clean"
        and (explicit_ld_requested or auto_ld_research)
    ):
        from mrnax.engine.lineardesign_adapter import (
            DEFAULT_LAMBDA_SWEEP,
            is_lineardesign_available,
            sweep_lineardesign_candidates,
        )

        if explicit_ld_requested or is_lineardesign_available():
            t0_ld = time.perf_counter()
            ld_calls = 0
            ld_fallback_calls = 0
            u5_ld_row = next(
                (u for u in clean_utr5s if _is_utr5_enzyme_clean(u["sequence"])),
                clean_utr5s[0],
            )
            u3_ld_row = next(
                (u for u in clean_utr3s if _is_utr3_enzyme_clean(_prepare_utr3(u["sequence"])[0])),
                clean_utr3s[0],
            )
            u5_ld = locked_utr5.locked_value if locked_utr5 else u5_ld_row["sequence"]
            u3_ld, anns_u3_ld = _prepare_utr3(u3_ld_row["sequence"])
            u5_id_ld = None if locked_utr5 else u5_ld_row["component_id"]
            u5_act_ld = "locked_verbatim" if locked_utr5 else "selected_from_library"
            u3_id_ld = None if locked_utr3 else u3_ld_row["component_id"]
            u3_act_ld = "locked_verbatim" if locked_utr3 else "selected_from_library"

            ld_lambdas = DEFAULT_LAMBDA_SWEEP if explicit_ld_requested else (1.0, 2.0)
            ld_primary_entries: list[tuple[float, str, str, LicenseTrack]] = []
            if locked_cds:
                ld_primary_entries = [(1.0, locked_cds.locked_value, "mrnax:locked_cds", inherited("cds").tier)]
            else:
                ld_cands = sweep_lineardesign_candidates(
                    protein_seq,
                    lambdas=ld_lambdas,
                    utr5_rna=u5_ld,
                    stop_rna=default_stop,
                    utr3_rna=u3_ld,
                    compliance_track=intent.compliance_track,
                    transl_table=intent.protein_targets[0].transl_table,
                    chemistry=intent.uridine_chemistry,
                    eliminate_uuuu=True,
                    constraints=intent.constraints,
                    seed=intent.runtime_policy.random_seed,
                )
                ld_calls += len(ld_cands)
                for c in ld_cands:
                    is_cpp = c.backend_used == "lineardesign_cpp_binary"
                    if not is_cpp:
                        ld_fallback_calls += 1
                    c_src = (
                        f"mrnax:lineardesign_polisher_lam{c.lambda_val:g}"
                        if is_cpp
                        else f"mrnax:guided_bicodon_vienna_fallback_lam{c.lambda_val:g}"
                    )
                    ld_primary_entries.append((c.lambda_val, c.polished_cds_rna, c_src, "needs_vendor_license"))
                if ld_primary_entries and not dc_seed_cds:
                    chosen_ld = ld_primary_entries[min(2, len(ld_primary_entries) - 1)]
                    dc_seed_cds = chosen_ld[1]
                    dc_seed_source = chosen_ld[2]
                    dc_seed_tier = chosen_ld[3]

            sec_ld_mols: list[RNAMoleculeSpec] = []
            for ch_idx, pt in enumerate(intent.protein_targets[1:], start=2):
                sec_ld_res = sweep_lineardesign_candidates(
                    pt.protein_sequence,
                    lambdas=(1.0,),
                    utr5_rna=u5_ld,
                    stop_rna=default_stop,
                    utr3_rna=u3_ld,
                    compliance_track=intent.compliance_track,
                    transl_table=pt.transl_table,
                    chemistry=intent.uridine_chemistry,
                    eliminate_uuuu=True,
                    constraints=intent.constraints,
                    seed=intent.runtime_policy.random_seed + ch_idx,
                )
                ld_calls += 1
                sec_cand = sec_ld_res[0]
                sec_is_cpp = sec_cand.backend_used == "lineardesign_cpp_binary"
                if not sec_is_cpp:
                    ld_fallback_calls += 1
                sec_ld_src = (
                    "mrnax:lineardesign_polisher_v25"
                    if sec_is_cpp
                    else "mrnax:guided_bicodon_vienna_fallback_v25"
                )
                sec_ld_cds = sec_cand.polished_cds_rna
                sec_ld_mols.append(
                    _build_molecule_spec(
                        molecule_id=f"chain_{ch_idx:02d}",
                        target=pt,
                        intent=intent,
                        origins=origins_for(
                            u5_id_ld,
                            u3_id_ld,
                            sec_ld_src,
                            anns_u3_ld,
                            secondary=True,
                            cds_tier="needs_vendor_license",
                        ),
                        utr5_seq=u5_ld,
                        utr5_action=u5_act_ld,
                        cds_seq=sec_ld_cds,
                        cds_action="generated_novo",
                        stop_seq=default_stop,
                        utr3_seq=u3_ld,
                        utr3_action=u3_act_ld,
                        polya_seq=polya_seq,
                        extra_utr3_annotations=anns_u3_ld,
                    )
                )

            for k_idx, (lam_v, ld_cds_seq, ld_src, ld_tier) in enumerate(ld_primary_entries, start=1):
                raw_candidates.append(
                    _build_single_molecule_bundle(
                        bundle_id=f"bnd_{exec_tag}_lineardesign_{k_idx:02d}",
                        parent=reference_bundle,
                        derivation_action="de_novo_design" if not locked_cds else "locked_refinement",
                        intent=intent,
                        origins=origins_for(
                            u5_id_ld,
                            u3_id_ld,
                            ld_src,
                            anns_u3_ld,
                            cds_tier=ld_tier,
                        ),
                        utr5_seq=u5_ld,
                        utr5_action=u5_act_ld,
                        cds_seq=ld_cds_seq,
                        cds_action="locked_verbatim" if locked_cds else "generated_novo",
                        stop_seq=default_stop,
                        utr3_seq=u3_ld,
                        utr3_action=u3_act_ld,
                        polya_seq=polya_seq,
                        extra_utr3_annotations=anns_u3_ld,
                        secondary_molecules=sec_ld_mols or None,
                    )
                )
            ld_elapsed = round(time.perf_counter() - t0_ld, 5)
            spec_ld = COMPONENT_CATALOG["lineardesign_academic"]
            comp_records.append(
                ComponentInvocationRecord(
                    name=spec_ld.display_name,
                    version=spec_ld.version,
                    role="generator",
                    code_license=spec_ld.code_license,
                    weights_license=spec_ld.weights_license,
                    data_license=spec_ld.data_license,
                    license_track=spec_ld.license_track,
                    wall_seconds=ld_elapsed,
                    attempt_count=ld_calls,
                    failure_count=ld_fallback_calls,
                )
            )
            if ld_fallback_calls > 0:
                spec_fb = COMPONENT_CATALOG["pareto_nsga2"]
                comp_records.append(
                    ComponentInvocationRecord(
                        name=spec_fb.display_name,
                        version=spec_fb.version,
                        role="generator",
                        code_license=spec_fb.code_license,
                        weights_license=spec_fb.weights_license,
                        data_license=spec_fb.data_license,
                        license_track="needs_vendor_license",
                        wall_seconds=ld_elapsed,
                        attempt_count=ld_fallback_calls,
                        failure_count=0,
                    )
                )

    # Generator 3: Non-Partitioned Full-Transcript CDS<->UTR Joint Pareto Search
    if effective_modality == "linear_mrna" and "pareto_nsga2" in intent.runtime_policy.enabled_generators:
        t0 = time.perf_counter()
        if dc_seed_cds:
            base_cds = dc_seed_cds
            base_cds_source = dc_seed_source
            base_cds_tier: LicenseTrack = dc_seed_tier
        else:
            base_cds = generate_random_synonymous_cds(
                protein_seq, rng, chemistry=intent.uridine_chemistry
            )
            base_cds_source = "mrnax:stage0_random"
            base_cds_tier = "commercial_clean"
            pareto_cds_calls += 1
        n_codons = len(protein_seq)
        budget_n = max(3, min(intent.runtime_policy.max_evaluations - len(raw_candidates), 8))
        pareto_utr5s = clean_utr5s
        pareto_utr3s = clean_utr3s
        all_u5_raw = [
            u for u in utr_reg["five_prime_utrs"] if not u["has_uaug"] and u["max_homopolymer_run"] <= 7
        ] or list(utr_reg["five_prime_utrs"])
        all_u3_raw = [u for u in utr_reg["three_prime_utrs"] if u["max_homopolymer_run"] <= 7] or list(
            utr_reg["three_prime_utrs"]
        )

        def _can_prepare_utr3(raw_seq: str) -> bool:
            try:
                _prepare_utr3(raw_seq)
                return True
            except ValueError:
                return False

        def _is_utr3_usable(raw_seq: str) -> bool:
            try:
                prep_seq, _ = _prepare_utr3(raw_seq)
            except ValueError:
                return False
            return _is_utr3_enzyme_clean(prep_seq)

        all_u5 = [u for u in all_u5_raw if _is_utr5_enzyme_clean(u["sequence"])] or all_u5_raw
        all_u3_prep = [u for u in all_u3_raw if _can_prepare_utr3(u["sequence"])] or [
            u for u in clean_utr3s if _can_prepare_utr3(u["sequence"])
        ]
        all_u3 = (
            [u for u in all_u3_prep if _is_utr3_usable(u["sequence"])]
            or all_u3_prep
            or all_u3_raw
        )

        def _prepare_utr3_safe(raw_u3: str) -> tuple[str, list[FeatureAnnotation]]:
            try:
                return _prepare_utr3(raw_u3)
            except ValueError:
                return raw_u3, []

        has_vienna_inner = (
            "viennarna_thermo" in intent.runtime_policy.enabled_evaluators
            and intent.compliance_track != "commercial_clean"
        )

        # Ensure secondary probe seeds exist with identical call counts regardless of evaluator switches
        sec_probe_cds_list: list[str] = []
        for ch_idx, pt in enumerate(intent.protein_targets[1:], start=2):
            existing_sec = sec_dc_seeds.get(ch_idx) or sec_s0_seeds.get(ch_idx)
            if existing_sec is None:
                fallback_rng = random.Random(intent.runtime_policy.random_seed + 4099 * ch_idx)
                existing_sec = generate_random_synonymous_cds(
                    pt.protein_sequence,
                    fallback_rng,
                    chemistry=intent.uridine_chemistry,
                )
                pareto_cds_calls += 1
                sec_s0_seeds[ch_idx] = existing_sec
            sec_probe_cds_list.append(existing_sec)

        chosen_utr_pairs: list[tuple[dict[str, Any], dict[str, Any]]] | None = None
        if not has_vienna_inner:
            pool_rng = random.Random(intent.runtime_policy.random_seed + 31337)
            k5 = min(24, len(all_u5))
            k3 = min(24, len(all_u3))
            sampled_u5 = pool_rng.sample(all_u5, k5)
            sampled_u3 = pool_rng.sample(all_u3, k3)

            probe_rng = random.Random(intent.runtime_policy.random_seed + 7919)
            probe_trials = [base_cds]
            cur_p = base_cds
            if not locked_cds:
                for p_step in range(4):
                    if legacy_pure_rerank_mode:
                        cur_p = mutate_with_atlas(
                            cur_p,
                            protein_seq,
                            probe_rng,
                            codon_start=10,
                            codon_end=max(11, n_codons - 5),
                            prefer_open_au=False,
                            chemistry=intent.uridine_chemistry,
                            legacy_single_boundary=True,
                        )
                    else:
                        cur_p = propose_full_cds_with_atlas(
                            base_cds,
                            protein_seq,
                            probe_rng,
                            chemistry=intent.uridine_chemistry,
                            constraints=intent.constraints,
                            slowdown_codon_ranges=intent.co_translational_slowdown_codons,
                            utr5_tail=default_utr5,
                            stop_head=default_stop,
                            gc_weight=0.015,
                            temperature=0.09,
                            target_gc=0.49 + 0.03 * p_step,
                            mutation_prob=0.35 + 0.15 * p_step,
                        )
                    pareto_cds_calls += 1
                    probe_trials.append(cur_p)

            if surrogate_v1 is not None and not legacy_pure_rerank_mode:
                p_scores = surrogate_v1.score_cds_stability(probe_trials)
                best_p_idx = max(range(len(probe_trials)), key=lambda i: float(p_scores[i]))
                probe_cds = probe_trials[best_p_idx]
                base_cds = probe_cds if not locked_cds else base_cds

                all_chain_probes = [probe_cds, *sec_probe_cds_list]

                # Score full 200x200 commercial_clean UTR universe across all chains + ATtRACT 160-RBP + gc_tail15
                cand_u5_pool = all_u5
                cand_u3_pool = all_u3
                prepared_u3_seqs = [_prepare_utr3_safe(u["sequence"])[0] for u in cand_u3_pool]
                scored_mrl = surrogate_v1.score_5utr_mrl([u["sequence"] for u in cand_u5_pool])
                u5_mrl_prior = [
                    float(scored_mrl[i]) + 0.15 * _utr5_gc_tail15_prior(cand_u5_pool[i]["sequence"])
                    for i in range(len(cand_u5_pool))
                ]
                anchor_u3 = prepared_u3_seqs[0]
                u5_hl_lists = [
                    surrogate_v1.score_consensus_halflife_batch(
                        [(u["sequence"], ch_cds, anchor_u3) for u in cand_u5_pool]
                    )
                    for ch_cds in all_chain_probes
                ]
                u5_worst_hl = [
                    min(float(u5_hl_lists[c_i][i]) for c_i in range(len(all_chain_probes)))
                    for i in range(len(cand_u5_pool))
                ]
                top_mrl_u5 = sorted(range(len(cand_u5_pool)), key=lambda i: -u5_mrl_prior[i])[:8]
                top_hl_u5 = sorted(range(len(cand_u5_pool)), key=lambda i: -u5_worst_hl[i])[:8]
                top_u5_idx = list(dict.fromkeys([*top_mrl_u5, *top_hl_u5]))

                u3_raw_preds = (
                    surrogate_v1.score_3utr_stability(prepared_u3_seqs)
                    if "3utr_mpra_stability" in surrogate_v1.loaded_heads
                    else [0.0] * len(cand_u3_pool)
                )
                u3_rbp_priors = [
                    _utr3_rbp_decay_prior(prepared_u3_seqs[i], destab_rbp_motifs, stab_rbp_motifs)
                    for i in range(len(cand_u3_pool))
                ]
                anchor_u5 = cand_u5_pool[top_u5_idx[0]]["sequence"]
                u3_hl_lists = [
                    surrogate_v1.score_consensus_halflife_batch(
                        [(anchor_u5, ch_cds, prepared_u3_seqs[i]) for i in range(len(cand_u3_pool))]
                    )
                    for ch_cds in all_chain_probes
                ]
                u3_worst_hl = [
                    min(float(u3_hl_lists[c_i][i]) for c_i in range(len(all_chain_probes)))
                    + 0.25 * u3_rbp_priors[i]
                    for i in range(len(cand_u3_pool))
                ]
                top_u3_idx = sorted(range(len(cand_u3_pool)), key=lambda i: -u3_worst_hl[i])[:18]

                pair_coords = [(i5, i3) for i5 in top_u5_idx for i3 in top_u3_idx]
                chain_hl_lists = [
                    surrogate_v1.score_consensus_halflife_batch(
                        [(cand_u5_pool[i5]["sequence"], ch_cds, prepared_u3_seqs[i3]) for i5, i3 in pair_coords]
                    )
                    for ch_cds in all_chain_probes
                ]
                worst_chain_hls = [
                    min(float(chain_hl_lists[c_i][idx_p]) for c_i in range(len(all_chain_probes)))
                    for idx_p in range(len(pair_coords))
                ]
                u3_in_domain = [
                    (float(u3_raw_preds[i]) if 140 <= len(prepared_u3_seqs[i]) <= 180 else 0.15 * float(u3_raw_preds[i]))
                    + 0.30 * u3_rbp_priors[i]
                    for i in range(len(cand_u3_pool))
                ]
                pair_vecs = [
                    [float(scored_mrl[i5]), worst_chain_hls[idx_p], u3_in_domain[i3]]
                    for idx_p, (i5, i3) in enumerate(pair_coords)
                ]
                hl_only_order = sorted(range(len(pair_coords)), key=lambda idx_p: -worst_chain_hls[idx_p])
                mrl_only_order = sorted(
                    range(len(pair_coords)),
                    key=lambda idx_p: -(float(scored_mrl[pair_coords[idx_p][0]]) + 0.15 * worst_chain_hls[idx_p]),
                )
                _, pareto_pair_order = _rank_and_order_pareto_candidates(pair_vecs)
                merged_pair_order = list(
                    dict.fromkeys([*hl_only_order[:2], *mrl_only_order[:2], *pareto_pair_order])
                )
                chosen_utr_pairs = [
                    (cand_u5_pool[pair_coords[p_idx][0]], cand_u3_pool[pair_coords[p_idx][1]])
                    for p_idx in merged_pair_order[: max(budget_n, 4)]
                ]
                pareto_utr5s = [p[0] for p in chosen_utr_pairs[:3]]
                pareto_utr3s = [p[1] for p in chosen_utr_pairs[:3]]
            else:
                base_cds = probe_trials[-1] if not locked_cds else base_cds
                pareto_utr5s = sampled_u5[:3]
                pareto_utr3s = sampled_u3[:3]

        n_utr_pick = min(4, len(pareto_utr5s)) if not legacy_pure_rerank_mode else min(3, len(pareto_utr5s))
        n_utr3_pick = min(4, len(pareto_utr3s)) if not legacy_pure_rerank_mode else min(3, len(pareto_utr3s))
        utr5_pool = [locked_utr5.locked_value] if locked_utr5 else [u["sequence"] for u in pareto_utr5s[:n_utr_pick]]
        utr5_ids = [None] if locked_utr5 else [u["component_id"] for u in pareto_utr5s[:n_utr_pick]]

        utr3_pool_raw = [locked_utr3.locked_value] if locked_utr3 else [u["sequence"] for u in pareto_utr3s[:n_utr3_pick]]
        utr3_ids = [None] if locked_utr3 else [u["component_id"] for u in pareto_utr3s[:n_utr3_pick]]

        for idx in range(budget_n):
            if chosen_utr_pairs is not None:
                pair_u5, pair_u3 = chosen_utr_pairs[idx % len(chosen_utr_pairs)]
                cur_u5_seq = locked_utr5.locked_value if locked_utr5 else pair_u5["sequence"]
                cur_u5_id = None if locked_utr5 else pair_u5["component_id"]
                cur_u3_raw = locked_utr3.locked_value if locked_utr3 else pair_u3["sequence"]
                cur_u3_id = None if locked_utr3 else pair_u3["component_id"]
            else:
                u5_idx = idx % len(utr5_pool)
                u3_idx = (idx // max(1, len(utr5_pool))) % len(utr3_pool_raw)
                cur_u5_seq = utr5_pool[u5_idx]
                cur_u5_id = utr5_ids[u5_idx]
                cur_u3_raw = utr3_pool_raw[u3_idx]
                cur_u3_id = utr3_ids[u3_idx]
            u3_seq, anns_u3 = _prepare_utr3_safe(cur_u3_raw)
            cand_cds_tier: LicenseTrack = "commercial_clean"
            cand_cds_src = "mrnax:pareto_cds_utr_coopt"

            if locked_cds:
                cand_cds = locked_cds.locked_value
                cds_act = "locked_verbatim"
                cand_cds_src = "mrnax:locked_cds"
                cand_cds_tier = inherited("cds").tier
            elif has_vienna_inner:
                used_two_zone_partition = True
                cand_cds_tier = max((base_cds_tier, "needs_vendor_license"), key=get_args(LicenseTrack).index)
                cand_cds_src = (
                    "mrnax:pareto_cds_utr_coopt"
                    if base_cds_tier == "commercial_clean"
                    else f"mrnax:pareto_cds_utr_coopt+{base_cds_source}"
                )
                cand_cds = mutate_with_atlas(
                    base_cds,
                    protein_seq,
                    rng,
                    codon_start=1,
                    codon_end=min(12, n_codons),
                    prefer_open_au=(idx % 2 == 0),
                    chemistry=intent.uridine_chemistry,
                    constraints=intent.constraints,
                )
                pareto_cds_calls += 1
                if n_codons > 15:
                    trial_bodies = [cand_cds]
                    cur_body = cand_cds
                    for _trial in range(3):
                        cur_body = mutate_with_atlas(
                            cur_body,
                            protein_seq,
                            rng,
                            codon_start=12,
                            codon_end=max(13, n_codons - 10),
                            prefer_open_au=False,
                            chemistry=intent.uridine_chemistry,
                            constraints=intent.constraints,
                        )
                        pareto_cds_calls += 1
                        trial_bodies.append(cur_body)
                    mfes = [compute_cds_sliding_window_mfe(tb) for tb in trial_bodies]
                    cand_cds = trial_bodies[min(range(len(trial_bodies)), key=lambda i: mfes[i])]
                if mirna_mature is not None and n_codons > 10:
                    cand_cds = mutate_with_atlas(
                        cand_cds,
                        protein_seq,
                        rng,
                        codon_start=max(1, n_codons - 12),
                        codon_end=n_codons,
                        prefer_open_au=True,
                        chemistry=intent.uridine_chemistry,
                        constraints=intent.constraints,
                    )
                    pareto_cds_calls += 1
                cds_act = "synonymous_mutated"
            elif legacy_pure_rerank_mode:
                # Unguided legacy generator control arm ("本次未改善生成器") with 1-to-1 identical direct CDS operator calls
                cand_cds = generate_random_synonymous_cds(
                    protein_seq,
                    rng,
                    chemistry=intent.uridine_chemistry,
                    slowdown_codon_ranges=intent.co_translational_slowdown_codons,
                )
                pareto_cds_calls += 1
                if n_codons > 15:
                    for _trial in range(3):
                        cand_cds = mutate_with_atlas(
                            cand_cds,
                            protein_seq,
                            rng,
                            codon_start=10,
                            codon_end=max(11, n_codons - 5),
                            prefer_open_au=False,
                            chemistry=intent.uridine_chemistry,
                            legacy_single_boundary=True,
                        )
                        pareto_cds_calls += 1
                cds_act = "synonymous_mutated"
            else:
                # Non-partitioned continuous full-CDS proposal with strict 1-to-1 equal direct CDS operator calls
                if n_codons > 15:
                    rand_primary = generate_random_synonymous_cds(
                        protein_seq,
                        rng,
                        chemistry=intent.uridine_chemistry,
                        slowdown_codon_ranges=intent.co_translational_slowdown_codons,
                    )
                    pareto_cds_calls += 1
                    trial_bodies = [base_cds, rand_primary]
                    trial_origins: list[tuple[str, LicenseTrack]] = [
                        (base_cds_source, base_cds_tier),
                        ("mrnax:stage0_random", "commercial_clean"),
                    ]
                    cur_body = base_cds
                    for t_step in range(3):
                        seed_primary = rand_primary if t_step == 0 else (base_cds if t_step == 1 else cur_body)
                        step_tier: LicenseTrack = "commercial_clean" if t_step == 0 else base_cds_tier
                        step_src = (
                            "mrnax:pareto_cds_utr_coopt"
                            if step_tier == "commercial_clean"
                            else f"mrnax:pareto_cds_utr_coopt+{base_cds_source}"
                        )
                        cur_body = propose_full_cds_with_atlas(
                            seed_primary,
                            protein_seq,
                            rng,
                            chemistry=intent.uridine_chemistry,
                            constraints=intent.constraints,
                            slowdown_codon_ranges=intent.co_translational_slowdown_codons,
                            utr5_tail=cur_u5_seq,
                            stop_head=default_stop + u3_seq[:9],
                            gc_weight=0.010 + 0.008 * t_step,
                            temperature=0.08 + 0.02 * t_step,
                            target_gc=(0.45, 0.53, 0.58)[t_step % 3],
                            mutation_prob=(0.95, 0.50, 0.65)[t_step % 3],
                        )
                        pareto_cds_calls += 1
                        trial_bodies.append(cur_body)
                        trial_origins.append((step_src, step_tier))

                    if surrogate_v1 is not None:
                        stabs = surrogate_v1.score_cds_stability(trial_bodies)
                        hls = surrogate_v1.score_consensus_halflife_batch(
                            [(cur_u5_seq, tb, u3_seq) for tb in trial_bodies]
                        )
                        if idx % 3 == 0:
                            pick_i = max(range(len(trial_bodies)), key=lambda i: float(hls[i]))
                        elif idx % 3 == 1:
                            pick_i = max(range(len(trial_bodies)), key=lambda i: float(stabs[i]))
                        else:
                            t_vecs = [[float(stabs[i]), float(hls[i])] for i in range(len(trial_bodies))]
                            _, t_order = _rank_and_order_pareto_candidates(t_vecs)
                            pick_i = t_order[0]
                    else:
                        pick_i = len(trial_bodies) - 1
                    cand_cds = trial_bodies[pick_i]
                    cand_cds_src, cand_cds_tier = trial_origins[pick_i]
                else:
                    cand_cds = propose_full_cds_with_atlas(
                        base_cds,
                        protein_seq,
                        rng,
                        chemistry=intent.uridine_chemistry,
                        constraints=intent.constraints,
                        slowdown_codon_ranges=intent.co_translational_slowdown_codons,
                        utr5_tail=cur_u5_seq,
                        stop_head=default_stop + u3_seq[:9],
                        gc_weight=0.012,
                        temperature=0.10,
                        target_gc=(0.49, 0.54, 0.58)[idx % 3],
                        mutation_prob=(0.65, 0.35, 0.50)[idx % 3],
                    )
                    pareto_cds_calls += 1
                    cand_cds_tier = base_cds_tier
                    cand_cds_src = (
                        "mrnax:pareto_cds_utr_coopt"
                        if cand_cds_tier == "commercial_clean"
                        else f"mrnax:pareto_cds_utr_coopt+{base_cds_source}"
                    )
                cds_act = "synonymous_mutated"

            # Scoped safety check for active Type IIS restriction sites (respecting target_scopes)
            if not locked_cds:
                cds_bad = any(pat in cand_cds for pat in cds_enzyme_motifs)
                full_check = cur_u5_seq[-8:] + cand_cds + default_stop + u3_seq[:8]
                junc_bad = any(pat in full_check for pat in full_tx_enzyme_motifs)
                if (cds_bad or junc_bad) and dc_seed_cds:
                    dc_full = cur_u5_seq[-8:] + dc_seed_cds + default_stop + u3_seq[:8]
                    if not any(pat in dc_seed_cds for pat in cds_enzyme_motifs) and not any(
                        pat in dc_full for pat in full_tx_enzyme_motifs
                    ):
                        cand_cds = dc_seed_cds
                        cand_cds_src = dc_seed_source
                        cand_cds_tier = dc_seed_tier

            u5_act_p = "locked_verbatim" if locked_utr5 else "selected_from_library"
            u3_act_p = "locked_verbatim" if locked_utr3 else "selected_from_library"
            raw_candidates.append(
                _build_single_molecule_bundle(
                    bundle_id=f"bnd_{exec_tag}_pareto_{idx + 1:02d}",
                    parent=reference_bundle,
                    derivation_action="de_novo_design" if not locked_cds else "locked_refinement",
                    intent=intent,
                    origins=origins_for(
                        cur_u5_id,
                        cur_u3_id,
                        cand_cds_src,
                        anns_u3,
                        cds_tier=cand_cds_tier,
                    ),
                    utr5_seq=cur_u5_seq,
                    utr5_action=u5_act_p,
                    cds_seq=cand_cds,
                    cds_action=cds_act,
                    stop_seq=default_stop,
                    utr3_seq=u3_seq,
                    utr3_action=u3_act_p,
                    polya_seq=polya_seq,
                    extra_utr3_annotations=anns_u3,
                    secondary_molecules=_make_secondary_chains(
                        utr5_seq=cur_u5_seq,
                        utr5_id=cur_u5_id,
                        utr5_action=u5_act_p,
                        utr3_seq=u3_seq,
                        utr3_id=cur_u3_id,
                        utr3_action=u3_act_p,
                        extra_anns=anns_u3,
                        seed_offset=idx + 1,
                    ),
                )
            )
        spec_p = COMPONENT_CATALOG["pareto_nsga2"]
        comp_records.append(
            ComponentInvocationRecord(
                name=spec_p.display_name,
                version=spec_p.version,
                role="generator",
                code_license=spec_p.code_license,
                weights_license=spec_p.weights_license,
                data_license=spec_p.data_license,
                license_track=spec_p.license_track,
                wall_seconds=round(time.perf_counter() - t0, 5),
                attempt_count=pareto_cds_calls,
                failure_count=0,
            )
        )

    # Deduplicate exact sequence digests while preserving generation order
    dedup_candidates: list[ConstructBundle] = []
    seen_digests: set[str] = set()
    for b in raw_candidates:
        if b.version_digest not in seen_digests:
            seen_digests.add(b.version_digest)
            dedup_candidates.append(b)

    # Attach resolved wild-type CDS references onto emitted ConstructBundles so replay & direct export enforce CureVac!
    if wildtype_cds_by_chain:
        stamped_dedup: list[ConstructBundle] = []
        for b in dedup_candidates:
            updated_mols: list[RNAMoleculeSpec] = []
            changed_mol = False
            for idx, m in enumerate(b.molecules):
                wt_s = wildtype_cds_by_chain.get(idx)
                if wt_s is not None and m.effective_translation_target.wildtype_cds_rna is None:
                    updated_mols.append(
                        m.model_copy_validated(
                            effective_translation_target=m.effective_translation_target.model_copy_validated(
                                wildtype_cds_rna=wt_s
                            )
                        )
                    )
                    changed_mol = True
                else:
                    updated_mols.append(m)
            if changed_mol:
                stamped_dedup.append(
                    ConstructBundle.create(
                        bundle_id=b.bundle_id,
                        parent_bundle_id=b.parent_bundle_id,
                        derivation_action=b.derivation_action,
                        co_expression_mode=b.co_expression_mode,
                        target_stoichiometry=b.target_stoichiometry,
                        molecules=updated_mols,
                        external_evidence=b.external_evidence,
                    )
                )
            else:
                stamped_dedup.append(b)
        dedup_candidates = stamped_dedup

    input_origin_type = (
        "wildtype_mrna_reference"
        if wildtype_cds_by_chain or any(
            m.effective_translation_target.wildtype_cds_rna is not None
            for candidate in dedup_candidates for m in candidate.molecules
        )
        else ("reference_construct_bundle" if reference_bundle is not None else "amino_acid_only")
    )
    max_cv_any_c: float | None = None
    max_cv_nog_c: float | None = None
    performed_fto_counts = {"genus": 0, "collision": 0, "cartesian_pair": 0}

    # 3. Shared Scoped Hard-Gate Filtering (+ CureVac F4A wild-type C-optimizable codon ceiling across ALL chains)
    qualified: list[ConstructBundle] = []
    all_diagnostics: list[FailedCandidateDiagnostic] = []
    all_fto_alerts: list[FTOAlert] = []
    all_advisories: list[AdvisoryNote] = [
        AdvisoryNote(advisory_id=f"unavailable_{cid}", category="model_domain_boundary", message=message)
        for cid, message in unavailable_components.items()
    ]
    eliminated_by_req: dict[str, int] = {c.requirement_id: 0 for c in intent.constraints}

    expected_chem_gate = intent.uridine_chemistry if effective_modality == "linear_mrna" else "unmodified_u"
    for b in dedup_candidates:
        outcome = constraint_engine.evaluate_bundle(
            b,
            intent.constraints,
            compliance_track=intent.compliance_track,
            expected_chemistry=expected_chem_gate,
            locked_scopes=intent.locked_scopes,
            expected_protein_targets=intent.protein_targets,
            wildtype_cds_by_chain=wildtype_cds_by_chain,
            rc005_active=use_codontransformer,
        )
        for rule_type, count in outcome.fto_alerts.performed_counts.items():
            performed_fto_counts[rule_type] += count
        if outcome.max_curevac_any_c is not None:
            max_cv_any_c = (
                outcome.max_curevac_any_c
                if max_cv_any_c is None
                else max(max_cv_any_c, outcome.max_curevac_any_c)
            )
        if outcome.max_curevac_no_higher_g is not None:
            max_cv_nog_c = (
                outcome.max_curevac_no_higher_g
                if max_cv_nog_c is None
                else max(max_cv_nog_c, outcome.max_curevac_no_higher_g)
            )
        all_fto_alerts.extend(outcome.fto_alerts)
        cand_violations = list(outcome.violations)

        for adv in outcome.advisories:
            if adv.advisory_id == "adv_curevac_c_conversion_warn" and adv.advisory_id not in {
                a.advisory_id for a in all_advisories
            }:
                all_advisories.append(adv)

        if not cand_violations:
            qualified.append(b)
            for adv in outcome.advisories:
                if adv.advisory_id not in {a.advisory_id for a in all_advisories}:
                    all_advisories.append(adv)
        else:
            all_diagnostics.extend(cand_violations)
            violated_ids = {v.violated_requirement_id for v in cand_violations}
            for req_id in violated_ids:
                if req_id in eliminated_by_req:
                    eliminated_by_req[req_id] += 1

    if intent.cell_expression_control and intent.cell_expression_control.detarget_tissues:
        all_advisories.append(build_crs_and_tissue_safety_advisory())

    if used_endogenous_atlas:
        all_advisories.append(
            AdvisoryNote(
                advisory_id="adv_bicodon_atlas_endogenous_scope",
                category="model_domain_boundary",
                message=(
                    f"Codon/bicodon mutation guidance uses endogenous cellular unmodified-mRNA human stability coefficients "
                    f"(human_codon_bicodon_stability_atlas_v1.json, Saluki+GENCODE v49) as a heuristic prior for synthetic IVT constructs "
                    f"(declared chemistry '{intent.uridine_chemistry}'); native m1Psi slippery-run gates remain strictly enforced."
                ),
            )
        )

    # 4. Thermodynamic / Surrogate Evaluation & Pareto Non-Dominated Ranking on Qualified Candidates
    eval_wall = 0.0
    surr_wall = 0.0
    srfm_wall = 0.0
    orth_wall = 0.0
    ef_wall = 0.0
    surr_ok_mols = 0
    surr_fail_mols = 0
    srfm_ok_mols = 0
    srfm_fail_mols = 0
    orth_ok_mols = 0
    orth_fail_mols = 0
    ef_ok_mols = 0
    ef_fail_mols = 0
    use_vienna = "viennarna_thermo" in intent.runtime_policy.enabled_evaluators
    raw_metrics_per_bundle: list[list[MetricResult]] = []
    obj_vectors: list[list[float]] = []
    complete_chain_coverage = []

    def _has_finite_ranking_metric(ms: list[MetricResult]) -> bool:
        return any(
            m.participated_in != "display_only" and m.value is not None and math.isfinite(m.value)
            for m in ms
        )

    rc005_candidates = {
        b.bundle_id for b in qualified if use_codontransformer or has_rc005_cds_provenance(b)
    }
    cleared_partners = {
        cid for cid in intent.runtime_policy.enabled_evaluators if rc005_candidates
        and cid in RC005_PARTNER_METRICS and audit_component_commercial_clean(cid) is None
    }

    def valid_measurement(metrics: list[MetricResult], component: str, requires_partner: bool) -> bool:
        if not requires_partner:
            return _has_finite_ranking_metric(metrics)
        return component in cleared_partners and any(
            resolve_tool_name_to_component_id(m.tool_name, m.metric_id) == component
            and _is_rc005_partner_measurement(m, component, chains=1) for m in metrics
        )

    for b in qualified:
        requires_partner = b.bundle_id in rc005_candidates
        per_chain_lists: list[list[MetricResult]] = []
        all_chains_have_cleared_partner = True
        for mol in b.molecules:
            mol_m: list[MetricResult] = []
            chain_has_cleared_partner = False
            if use_vienna:
                t_ev = time.perf_counter()
                th_list, th_adv = evaluate_molecule_thermodynamics(
                    mol,
                    mirna_mature_for_detarget=mirna_mature,
                    include_full_mfe=False,
                )
                eval_wall += time.perf_counter() - t_ev
                mol_m.extend(th_list)
                for adv in th_adv:
                    if adv.advisory_id not in {a.advisory_id for a in all_advisories}:
                        all_advisories.append(adv)
            if surrogate_v1 is not None:
                active_surr_cid = "surrogate_v2" if surrogate_v2 is not None else "surrogate_v1_clean"
                t_sr = time.perf_counter()
                sr_list, sr_adv = surrogate_v1.evaluate_molecule(mol)
                surr_wall += time.perf_counter() - t_sr
                if valid_measurement(sr_list, active_surr_cid, requires_partner):
                    surr_ok_mols += 1
                    chain_has_cleared_partner = True
                else:
                    surr_fail_mols += 1
                mol_m.extend(sr_list)
                for adv in sr_adv:
                    if adv.advisory_id not in {a.advisory_id for a in all_advisories}:
                        all_advisories.append(adv)
            if active_structrfm is not None:
                t_sf = time.perf_counter()
                sf_list, sf_adv = active_structrfm.evaluate_molecule(mol)
                srfm_wall += time.perf_counter() - t_sf
                if valid_measurement(sf_list, "structrfm_rna_foundation", requires_partner):
                    srfm_ok_mols += 1
                    chain_has_cleared_partner = True
                else:
                    srfm_fail_mols += 1
                mol_m.extend(sf_list)
                for adv in sf_adv:
                    if adv.advisory_id not in {a.advisory_id for a in all_advisories}:
                        all_advisories.append(adv)
            if active_eternafold_bin is not None and active_eternafold_params is not None:
                t_ef = time.perf_counter()
                ef_list, ef_adv = _evaluate_molecule_eternafold(
                    mol,
                    contrafold_bin=active_eternafold_bin,
                    ef_params=active_eternafold_params,
                )
                ef_wall += time.perf_counter() - t_ef
                if _has_finite_ranking_metric(ef_list):
                    ef_ok_mols += 1
                else:
                    ef_fail_mols += 1
                mol_m.extend(ef_list)
                for adv in ef_adv:
                    if adv.advisory_id not in {a.advisory_id for a in all_advisories}:
                        all_advisories.append(adv)
            if active_codontransformer is not None:
                cds_full_for_ct = mol.get_segment_sequence("cds") + mol.get_segment_sequence("stop_cassette")
                ct_ll = float(active_codontransformer.score_cds_synonymous_loglik([cds_full_for_ct])[0])
                if math.isfinite(ct_ll):
                    mol_m.append(
                        MetricResult(
                            metric_id="codontransformer_syn_loglik",
                            value=round(ct_ll, 6),
                            unit="log_prob_per_codon",
                            direction="maximize",
                            evaluated_scopes=["cds"],
                            tool_name="CodonTransformerApache2",
                            tool_version=COMPONENT_CATALOG["codontransformer_cds_baseline"].version,
                            chemistry_evaluation_mode=(
                                "native_chemistry_validated"
                                if mol.chemistry_track.uridine_chemistry == "unmodified_u"
                                else "endogenous_unmodified_proxy"
                            ),
                            participated_in="pareto_selection_or_ranking",
                        )
                    )
            if active_orthrus_model is not None and active_orthrus_probe is not None:
                import torch  # type: ignore[import-not-found]
                from mrnax.engine.orthrus_portable_pytorch import PortableOrthrusModel

                t_or = time.perf_counter()
                arr6 = PortableOrthrusModel.encode_6track(
                    mol.get_segment_sequence("five_prime_utr"),
                    mol.get_segment_sequence("cds") + mol.get_segment_sequence("stop_cassette"),
                    mol.get_segment_sequence("three_prime_utr"),
                )
                with torch.no_grad():
                    encoded = torch.from_numpy(arr6)
                    max_len = active_orthrus_probe["max_len"]
                    if len(encoded) > max_len:
                        head = max_len // 2
                        encoded = torch.cat((encoded[:head], encoded[-(max_len - head):]))
                    x_t = encoded.unsqueeze(0)
                    lengths_t = torch.tensor([len(encoded)], dtype=torch.long)
                    emb = active_orthrus_model.representation(x_t, lengths_t).cpu().numpy()
                calibrated = active_orthrus_probe["scaler"].transform(emb)
                pred_hl = float(active_orthrus_probe["model"].predict(calibrated)[0])
                orth_wall += time.perf_counter() - t_or
                if math.isfinite(pred_hl):
                    orth_ok_mols += 1
                    if "orthrus_mamba_rna_foundation" in cleared_partners:
                        chain_has_cleared_partner = True
                    mol_m.append(
                        MetricResult(
                            metric_id="orthrus_halflife_pred",
                            value=round(pred_hl, 5),
                            unit="saluki_pc1_z",
                            direction="maximize",
                            evaluated_scopes=["five_prime_utr", "cds", "three_prime_utr"],
                            tool_name="Orthrus6TrackMIT",
                            tool_version=COMPONENT_CATALOG["orthrus_mamba_rna_foundation"].version,
                            chemistry_evaluation_mode="endogenous_unmodified_proxy",
                            participated_in="pareto_selection_or_ranking",
                        )
                    )
                else:
                    orth_fail_mols += 1
            if not chain_has_cleared_partner:
                all_chains_have_cleared_partner = False
            if not mol_m:
                cds_s = mol.get_segment_sequence("cds")
                gc_val = round((cds_s.count("G") + cds_s.count("C")) / max(1, len(cds_s)), 4)
                mol_m = [
                    MetricResult(
                        metric_id="cds_gc_fraction",
                        value=gc_val,
                        unit="fraction",
                        direction="informational",
                        evaluated_scopes=["cds"],
                        tool_name="mRNAxSeqStats",
                        tool_version="0.1.0",
                        chemistry_evaluation_mode="native_chemistry_validated",
                        participated_in="display_only",
                    )
                ]
            per_chain_lists.append(mol_m)

        m_list, complete = _aggregate_chain_metrics(per_chain_lists)
        if requires_partner and not all_chains_have_cleared_partner:
            complete = False
        complete_chain_coverage.append(complete)
        raw_metrics_per_bundle.append(m_list)

        m_map = {
            m.metric_id: m.value
            for m in m_list
            if m.value is not None and math.isfinite(m.value) and m.participated_in != "display_only"
        }
        vec: list[float] = []
        for obj in intent.objectives:
            if obj.evidence_status == "unverified_target_for_wetlab" or obj.metric_id not in m_map:
                continue
            sc = score_objective_value(float(m_map[obj.metric_id]), obj)
            if sc is not None:
                vec.append(sc)
        obj_vectors.append(vec)

    total_evaluated_mols = sum(len(b.molecules) for b in qualified)
    if use_vienna and qualified:
        spec_v = COMPONENT_CATALOG["viennarna_thermo"]
        comp_records.append(
            ComponentInvocationRecord(
                name=spec_v.display_name,
                version=spec_v.version,
                role="evaluator",
                code_license=spec_v.code_license,
                weights_license=spec_v.weights_license,
                data_license=spec_v.data_license,
                license_track=spec_v.license_track,
                wall_seconds=round(eval_wall, 5),
                attempt_count=total_evaluated_mols,
                failure_count=0,
            )
        )
    if surrogate_v1 is not None and qualified:
        active_surr_cid = "surrogate_v2" if surrogate_v2 is not None else "surrogate_v1_clean"
        spec_s = COMPONENT_CATALOG[active_surr_cid]
        comp_records.append(
            ComponentInvocationRecord(
                name=spec_s.display_name,
                version=spec_s.version,
                role="evaluator",
                code_license=spec_s.code_license,
                weights_license=spec_s.weights_license,
                data_license=spec_s.data_license,
                license_track=spec_s.license_track,
                wall_seconds=round(surr_wall, 5),
                attempt_count=surr_ok_mols + surr_fail_mols,
                failure_count=surr_fail_mols,
            )
        )
    if active_structrfm is not None and qualified:
        spec_sf = COMPONENT_CATALOG["structrfm_rna_foundation"]
        comp_records.append(
            ComponentInvocationRecord(
                name=spec_sf.display_name,
                version=spec_sf.version,
                role="evaluator",
                code_license=spec_sf.code_license,
                weights_license=spec_sf.weights_license,
                data_license=spec_sf.data_license,
                license_track=spec_sf.license_track,
                wall_seconds=round(srfm_wall, 5),
                attempt_count=srfm_ok_mols + srfm_fail_mols,
                failure_count=srfm_fail_mols,
            )
        )
    if active_orthrus_model is not None and qualified:
        spec_or = COMPONENT_CATALOG["orthrus_mamba_rna_foundation"]
        comp_records.append(
            ComponentInvocationRecord(
                name=spec_or.display_name,
                version=spec_or.version,
                role="evaluator",
                code_license=spec_or.code_license,
                weights_license=spec_or.weights_license,
                data_license=spec_or.data_license,
                license_track=spec_or.license_track,
                wall_seconds=round(orth_wall, 5),
                attempt_count=orth_ok_mols + orth_fail_mols,
                failure_count=orth_fail_mols,
            )
        )
    if active_eternafold_bin is not None and qualified:
        spec_ef = COMPONENT_CATALOG[active_eternafold_comp_id]
        comp_records.append(
            ComponentInvocationRecord(
                name=spec_ef.display_name,
                version=spec_ef.version,
                role="evaluator",
                code_license=spec_ef.code_license,
                weights_license=spec_ef.weights_license,
                data_license=spec_ef.data_license,
                license_track=spec_ef.license_track,
                wall_seconds=round(ef_wall, 5),
                attempt_count=ef_ok_mols + ef_fail_mols,
                failure_count=ef_fail_mols,
            )
        )
    if use_eternafold and qualified and ef_ok_mols == 0:
        return unavailable(
            "无法比较：EternaFold 后验输出为空、截断或格式非法，拒绝生成虚假 1.0 概率指标。"
        )

    # 5. Outcome status & deterministic Action Transparency
    action_traces: list[RequirementActionTrace] = []
    for c in intent.constraints:
        elim = eliminated_by_req.get(c.requirement_id, 0)
        action_traces.append(
            RequirementActionTrace(
                requirement_id=c.requirement_id,
                stage_applied="filtering",
                eliminated_count=elim,
                changed_candidate_ids=[],
                proof_diff_summary=(
                    f"Executed hard gate '{c.constraint_kind}' on scopes {sorted(c.target_scopes)}; "
                    f"eliminated {elim}/{len(dedup_candidates)} candidates."
                ),
            )
        )

    rankable_objectives = [
        o for o in intent.objectives if o.evidence_status != "unverified_target_for_wetlab"
    ]
    uncomparable_missing_msg: str | None = None

    if not qualified:
        outcome_status = "COMPLETED_ZERO_QUALIFIED"
        recommended_ids: list[str] = []
        bundle_evals: list[BundleEvaluation] = []
        ordered_qualified: list[ConstructBundle] = []
    else:
        if intent.objectives and not rankable_objectives:
            comparable = []
        else:
            comparable = [
                i for i, vector in enumerate(obj_vectors)
                if len(vector) == len(rankable_objectives)
                and complete_chain_coverage[i]
            ]
        if comparable:
            vectors = [obj_vectors[i] for i in comparable]
            ranks, order = _rank_and_order_pareto_candidates(vectors)
            ordered_qualified = [qualified[comparable[i]] for i in order]
            bundle_evals = [
                BundleEvaluation(
                    bundle_id=qualified[comparable[i]].bundle_id,
                    version_digest=qualified[comparable[i]].version_digest,
                    passed_all_hard_gates=True, pareto_rank=ranks[i],
                    metrics=raw_metrics_per_bundle[comparable[i]],
                    selection_rationale=[f"Pareto rank {ranks[i]} under {intent.compliance_track} profile"],
                ) for i in order
            ]
            is_tie = len(vectors) > 1 and all(v == vectors[0] for v in vectors[1:])
            outcome_status = "COMPLETED_TIE" if is_tie else "COMPLETED_WITH_CANDIDATES"
            recommended_ids = [e.bundle_id for e in bundle_evals if e.pareto_rank == 1][:3] if rankable_objectives else []
        else:
            outcome_status = "UNCOMPARABLE_MISSING_APPROVED_COMPONENT"
            missing_names = [o.metric_id for o in (rankable_objectives or intent.objectives)]
            uncomparable_missing_msg = "Missing complete measurements for requested objectives: " + ", ".join(
                missing_names
            )
            recommended_ids = []
            bundle_evals = []
            ordered_qualified = []

    for obj in intent.objectives:
        if obj.evidence_status == "unverified_target_for_wetlab":
            action_traces.append(
                RequirementActionTrace(
                    requirement_id=obj.requirement_id,
                    stage_applied="recorded_only",
                    eliminated_count=0,
                    changed_candidate_ids=[],
                    proof_diff_summary=(
                        f"Objective '{obj.metric_id}' lacks validated computational predictor; "
                        "recorded as pending wet-lab validation target."
                    ),
                )
            )
        elif not ordered_qualified:
            action_traces.append(
                RequirementActionTrace(
                    requirement_id=obj.requirement_id,
                    stage_applied="recorded_only",
                    eliminated_count=0,
                    changed_candidate_ids=[],
                    proof_diff_summary=(
                        f"Objective '{obj.metric_id}' could not be ranked with complete measurements from "
                        f"{intent.runtime_policy.enabled_evaluators}; recorded without participating in ranking."
                    ),
                )
            )
        else:
            action_traces.append(
                RequirementActionTrace(
                    requirement_id=obj.requirement_id,
                    stage_applied="ranking",
                    eliminated_count=0,
                    changed_candidate_ids=recommended_ids,
                    proof_diff_summary=(
                        f"Objective '{obj.metric_id}' ({obj.direction}) participated in Pareto selection across {len(ordered_qualified)} fully measured bundles."
                    ),
                )
            )

    report = EvaluationReport(
        report_id=f"rep_{exec_id.removeprefix('exec_')}",
        execution_id=exec_id,
        intent_id=intent.intent_id,
        intent_version=intent.intent_version,
        compliance_track=intent.compliance_track,
        outcome_status=outcome_status,  # type: ignore[arg-type]
        uncomparable_reason=uncomparable_missing_msg,
        recommended_bundle_ids=recommended_ids,
        candidate_evaluations=bundle_evals,
        failed_diagnostics=all_diagnostics,
        fto_alerts=all_fto_alerts,
        pre_synthesis_advisories=all_advisories,
        action_transparency=action_traces,
        created_at_ns=next_report_timestamp_ns(),
        rule_snapshot_version=rule_snapshot_version,
        isolated_sandbox_override=isolated_sandbox_override,
    )

    opt_mode = (
        "stage0_or_reference_only"
        if "pareto_nsga2" not in intent.runtime_policy.enabled_generators
        else (
            "research_sandbox_sliding_window"
            if used_two_zone_partition
            else "full_transcript_non_partitioned_pareto"
        )
    )
    git_sha = _git_sha(repo_root)
    executed_operators: list[str] = []
    if cached_candidates:
        executed_operators.append("cached_candidate_ingestion")
    if reference_bundle is not None:
        executed_operators.append("reference_bundle_inheritance_and_minimal_lock")
    stage0_spec_name = COMPONENT_CATALOG["stage0_random"].display_name
    dc_spec_name = COMPONENT_CATALOG["dnachisel_cds"].display_name
    pareto_spec_name = COMPONENT_CATALOG["pareto_nsga2"].display_name
    if any(c.name == stage0_spec_name and c.attempt_count > 0 for c in comp_records):
        executed_operators.append("stage0_random_synonymous_cds_constructive")
    if any(c.name == dc_spec_name and c.attempt_count > 0 for c in comp_records):
        executed_operators.append("dnachisel_scoped_constraint_cds")
    ld_spec_name = COMPONENT_CATALOG["lineardesign_academic"].display_name
    if any(c.name == ld_spec_name and c.attempt_count > c.failure_count for c in comp_records):
        executed_operators.append("lineardesign_lattice_cfg_plus_5prime_surgical_polisher")
    if active_codontransformer is not None and any(
        "CodonTransformer" in c.name and c.attempt_count > 0 for c in comp_records
    ):
        chemistry = intent.uridine_chemistry
        suffix = f"plus_{chemistry}_uuuc_u4_remediation" if chemistry in {"m1psi", "pseudouridine"} else chemistry
        executed_operators.append(f"codontransformer_v1_homo_sapiens_synonymous_mask_{suffix}")
    if "pareto_nsga2" in intent.runtime_policy.enabled_generators:
        pareto_ran_cds = any(c.name == pareto_spec_name and c.attempt_count > 0 for c in comp_records)
        if not legacy_pure_rerank_mode and not (locked_utr5 and locked_utr3):
            executed_operators.append("utr_pool_multi_axis_pareto_preselection")
        if pareto_ran_cds:
            if used_two_zone_partition:
                executed_operators.append("research_sandbox_sliding_window_cds_utr_coopt")
            elif legacy_pure_rerank_mode:
                executed_operators.append("legacy_unguided_control_cds_mutation")
            else:
                executed_operators.append("propose_non_partitioned_full_cds_bicodon")
    if dedup_candidates:
        executed_operators.append("scoped_hard_gate_and_sequence_fto_filter")
    if ordered_qualified:
        executed_operators.append("pareto_non_dominated_crowding_selection")

    seq_fto_receipt = constraint_engine.fto_guard.build_receipt(
        all_fto_alerts,
        screen_applied=bool(dedup_candidates),
        performed_counts=performed_fto_counts,
    )
    opt_audit = (
        active_method_fto_guard.build_runtime_audit_record(
            intent=intent,
            opt_mode=opt_mode,
            used_two_zone_partition=used_two_zone_partition,
            executed_operators=executed_operators,
            invoked_components=comp_records,
            seq_fto_receipt=seq_fto_receipt,
            input_origin_type=input_origin_type,
            max_curevac_any_c=max_cv_any_c,
            max_curevac_no_higher_g=max_cv_nog_c,
            curevac_error_note=next(
                (d.reason for d in all_diagnostics if d.violated_requirement_id == "method_fto:CUREVAC_INVALID_WILDTYPE_REFERENCE"),
                None,
            ),
            target_jurisdictions=target_deployment_jurisdictions,
            run_id=exec_id,
            engine_git_commit=git_sha,
        )
        if active_method_fto_guard is not None
        else MethodFTOAuditRecord(
            run_id=exec_id,
            engine_git_commit=git_sha,
            optimization_mode=opt_mode,  # type: ignore[arg-type]
            uses_two_zone_initiation_body_partition=used_two_zone_partition,
            uses_dfa_cfg_intersection=False,
            sequence_fto_screen_applied=bool(dedup_candidates),
            sequence_fto_guard_v2_receipt=seq_fto_receipt,
        )
    )
    if intent.compliance_track == "commercial_clean":
        blocked_mfto = [
            f"{bid}: {info.get('evidence_note', '')}"
            for bid, info in opt_audit.method_fto_boundary_checks.items()
            if info.get("status") == "BLOCK"
        ]
        dependency_blocks = [
            f"{entry['asset_id']} (transitive_gpl_count={entry['transitive_gpl_count']})"
            for entry in opt_audit.three_layer_dependency_manifest
            if entry["transitive_gpl_count"] > 0
        ]
        if dependency_blocks or (ordered_qualified and blocked_mfto):
            traces = list(report.action_transparency)
            if dependency_blocks:
                traces.append(RequirementActionTrace(
                    requirement_id=COMMERCIAL_ZERO_GPL_REQUIREMENT,
                    stage_applied="filtering",
                    eliminated_count=len(ordered_qualified),
                    changed_candidate_ids=[b.bundle_id for b in dedup_candidates],
                    proof_diff_summary="Commercial dependency policy blocked release: " + ", ".join(dependency_blocks),
                ))
            ordered_qualified = []
            report = report.model_copy_validated(
                outcome_status="UNCOMPARABLE_MISSING_APPROVED_COMPONENT",
                uncomparable_reason="Method-FTO post-execution block: " + "; ".join(blocked_mfto + dependency_blocks),
                recommended_bundle_ids=[],
                candidate_evaluations=[],
                action_transparency=[
                    trace.model_copy_validated(
                        changed_candidate_ids=[],
                        proof_diff_summary="Commercial release gate withdrew the ranking result.",
                    ) if trace.stage_applied == "ranking" else trace
                    for trace in traces
                ],
            )

    rep_bytes = serialize_report_artifact(report.model_dump(mode="json"))
    manifest = RunManifest(
        submission_id=f"sub_{intent.intent_id}_v{intent.intent_version}",
        execution_id=exec_id,
        attempt_id=att_id,
        lease_epoch=1,
        git_commit_sha=_git_sha(repo_root),
        python_version=sys.version.split()[0],
        platform_info=platform.platform(),
        compliance_track=intent.compliance_track,
        invoked_components=comp_records,
        unavailable_components=unavailable_components,
        rule_snapshot_version=rule_snapshot_version,
        isolated_sandbox_override=isolated_sandbox_override,
        prng_state_digest=prng_digest,
        input_intent_digest=intent.digest(),
        artifacts=[
            ArtifactChecksumRecord(
                logical_name="evaluation_report",
                local_relpath="evaluation_report.json",
                drive_file_id=None,
                sha256=hashlib.sha256(rep_bytes).hexdigest(),
                md5=hashlib.md5(rep_bytes).hexdigest(),
                bytes=len(rep_bytes),
            )
        ],
        replay_cli_command=f"uv run mrnax replay --intent-file intent_{intent.intent_id}.json",
        honesty_banner="本次未改善生成器" if legacy_pure_rerank_mode else None,
        optimization_mode_audit=opt_audit,
    )
    return ordered_qualified, report, manifest

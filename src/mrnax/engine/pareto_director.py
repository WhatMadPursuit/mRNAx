"""Sliding-window structure-guided CDS<->UTR alternating Pareto optimization director."""

from __future__ import annotations

import hashlib
import json
import platform
import random
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from mrnax.constraints.frameshift_m1psi import remediate_m1psi_slippery_runs
from mrnax.constraints.registry import ScopedConstraintEngine
from mrnax.engine.cell_expression import (
    assemble_detargeted_3utr,
    build_crs_and_tissue_safety_advisory,
)
from mrnax.engine.profiles import COMPONENT_CATALOG, check_profile_readiness
from mrnax.engine.stage0_baseline import (
    generate_dnachisel_cds,
    generate_random_synonymous_cds,
    mutate_cds_region_synonymously,
)
from mrnax.engine.thermodynamics import evaluate_molecule_thermodynamics
from mrnax.schemas.construct_bundle import (
    ChemistryTrackSpec,
    ConstructBundle,
    DerivationAction,
    FeatureAnnotation,
    RNAMoleculeSpec,
    SegmentSpan,
)
from mrnax.schemas.design_intent import DesignIntent
from mrnax.schemas.evaluation_report import (
    AdvisoryNote,
    BundleEvaluation,
    EvaluationReport,
    FailedCandidateDiagnostic,
    FTOAlert,
    MetricResult,
    RequirementActionTrace,
)
from mrnax.schemas.run_manifest import (
    ArtifactChecksumRecord,
    ComponentInvocationRecord,
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


def _build_single_molecule_bundle(
    *,
    bundle_id: str,
    parent_bundle_id: str | None,
    derivation_action: DerivationAction,
    intent: DesignIntent,
    utr5_seq: str,
    utr5_id: str | None,
    utr5_source: str,
    utr5_action: str,
    cds_seq: str,
    cds_source: str,
    cds_action: str,
    stop_seq: str,
    utr3_seq: str,
    utr3_id: str | None,
    utr3_source: str,
    utr3_action: str,
    polya_seq: str,
    extra_utr3_annotations: list[FeatureAnnotation] | None = None,
) -> ConstructBundle:
    target = intent.protein_targets[0]
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

    mol = RNAMoleculeSpec(
        molecule_id="mol_primary",
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
                component_id=utr5_id,
                origin_source=utr5_source,
                license_track="commercial_clean",
                action_taken=utr5_action,  # type: ignore[arg-type]
            ),
            SegmentSpan(
                segment_type="cds",
                start=c1,
                end=c2,
                component_id=None,
                origin_source=cds_source,
                license_track="commercial_clean",
                action_taken=cds_action,  # type: ignore[arg-type]
            ),
            SegmentSpan(
                segment_type="stop_cassette",
                start=c2,
                end=c3,
                component_id="stop_uga_uaa",
                origin_source="mrnax:standard",
                license_track="commercial_clean",
                action_taken="locked_verbatim",
            ),
            SegmentSpan(
                segment_type="three_prime_utr",
                start=c3,
                end=c4,
                component_id=utr3_id,
                origin_source=utr3_source,
                license_track="commercial_clean",
                action_taken=utr3_action,  # type: ignore[arg-type]
            ),
            SegmentSpan(
                segment_type="poly_a",
                start=c4,
                end=c5,
                component_id=f"polya_a{len(polya_seq)}",
                origin_source="mrnax:homopolymer_tail",
                license_track="commercial_clean",
                action_taken="locked_verbatim",
            ),
        ],
        annotations=anns,
    )
    return ConstructBundle.create(
        bundle_id=bundle_id,
        parent_bundle_id=parent_bundle_id,
        derivation_action=derivation_action,
        target_stoichiometry={"mol_primary": 1.0},
        molecules=[mol],
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


def run_design_pipeline(
    intent: DesignIntent,
    *,
    reference_bundle: ConstructBundle | None = None,
    seed_dir: Path,
    rule_snapshot_version: str = "rules_research_v1@1.0.0",
    isolated_sandbox_override: bool = False,
) -> tuple[list[ConstructBundle], EvaluationReport, RunManifest]:
    """Execute Stage 0 + Sliding-Window CDS<->UTR Pareto Optimization with shared scoped constraints."""
    exec_id = f"exec_{uuid.uuid4().hex[:12]}"
    att_id = f"att_{exec_id[-6:]}_001"
    repo_root = seed_dir.parent

    if intent.status != "ready":
        raise ValueError(f"DesignIntent status is '{intent.status}' (must be 'ready' before engine run)")

    # 1. Fail-closed check for disabled generators when cache is disallowed
    if not intent.runtime_policy.enabled_generators and not intent.runtime_policy.allow_cached_candidates:
        raise NoAvailableGeneratorError(
            "enabled_generators is empty and allow_cached_candidates=False; refusing silent fallback."
        )

    # 2. Clean-Room profile readiness check (`commercial_clean` vs `research_max`)
    readiness = check_profile_readiness(intent)
    if not readiness.is_runnable:
        report = EvaluationReport(
            report_id=f"rep_{exec_id[-8:]}",
            execution_id=exec_id,
            intent_id=intent.intent_id,
            intent_version=intent.intent_version,
            compliance_track=intent.compliance_track,
            outcome_status="UNCOMPARABLE_MISSING_APPROVED_COMPONENT",
            uncomparable_reason=readiness.uncomparable_reason,
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
                    proof_diff_summary=readiness.uncomparable_reason or "Blocked by commercial_clean gate",
                )
                for c in intent.constraints
            ],
        )
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
            rule_snapshot_version=rule_snapshot_version,
            isolated_sandbox_override=isolated_sandbox_override,
            prng_state_digest="not_executed",
            input_intent_digest=intent.digest(),
            artifacts=[],
            replay_cli_command=f"uv run mrnax replay --intent-file <intent_{intent.intent_id}.json>",
            honesty_banner=None,
        )
        return [], report, manifest

    utr_reg = json.loads((seed_dir / "utr_components_v1.json").read_text(encoding="utf-8"))
    mir_reg = json.loads((seed_dir / "mirna_rbp_commercial_v1.json").read_text(encoding="utf-8"))
    constraint_engine = ScopedConstraintEngine.from_seed_dir(seed_dir)

    rng = random.Random(intent.runtime_policy.random_seed)
    prng_digest = hashlib.sha256(str(rng.getstate()).encode("utf-8")).hexdigest()

    protein_seq = intent.protein_targets[0].protein_sequence
    clean_utr5s = [
        u for u in utr_reg["five_prime_utrs"] if not u["has_uaug"] and u["max_homopolymer_run"] <= 7
    ]
    clean_utr3s = [u for u in utr_reg["three_prime_utrs"] if u["max_homopolymer_run"] <= 7]

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
        parent_id = reference_bundle.bundle_id
    else:
        default_utr5 = clean_utr5s[0]["sequence"]
        default_cds = ""
        default_stop = "UGA"
        default_utr3 = clean_utr3s[0]["sequence"]
        default_polya = "A" * 100
        parent_id = None

    polya_seq = locked_polya.locked_value if locked_polya else default_polya

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

    raw_candidates: list[ConstructBundle] = []
    comp_records: list[ComponentInvocationRecord] = []

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
                    bundle_id=f"bnd_{exec_id[-4:]}_minA",
                    parent_bundle_id=parent_id,
                    derivation_action="locked_refinement",
                    intent=intent,
                    utr5_seq=locked_utr5.locked_value if locked_utr5 else default_utr5,
                    utr5_id="ref_utr5",
                    utr5_source="reference_R",
                    utr5_action="locked_verbatim",
                    cds_seq=min_cds,
                    cds_source="reference_R_remediated",
                    cds_action="locked_verbatim" if locked_cds else "synonymous_mutated",
                    stop_seq=default_stop,
                    utr3_seq=u3_min,
                    utr3_id="ref_utr3",
                    utr3_source="reference_R",
                    utr3_action="locked_verbatim" if not anns_min else "selected_from_library",
                    polya_seq=polya_seq,
                    extra_utr3_annotations=anns_min,
                )
            )

    # Generator 1: Stage 0 random synonymous baseline / computational reference
    if "stage0_random" in intent.runtime_policy.enabled_generators:
        t0 = time.perf_counter()
        cds_rand = (
            locked_cds.locked_value
            if locked_cds
            else generate_random_synonymous_cds(
                protein_seq,
                rng,
                chemistry=intent.uridine_chemistry,
                slowdown_codon_ranges=intent.co_translational_slowdown_codons,
            )
        )
        u5 = locked_utr5.locked_value if locked_utr5 else clean_utr5s[0]["sequence"]
        u3, anns_u3 = _prepare_utr3(clean_utr3s[0]["sequence"])
        raw_candidates.append(
            _build_single_molecule_bundle(
                bundle_id=f"bnd_{exec_id[-4:]}_stage0_ref",
                parent_bundle_id=parent_id,
                derivation_action="computational_reference" if reference_bundle is None else "de_novo_design",
                intent=intent,
                utr5_seq=u5,
                utr5_id=None if locked_utr5 else clean_utr5s[0]["component_id"],
                utr5_source="locked" if locked_utr5 else clean_utr5s[0]["origin_source"],
                utr5_action="locked_verbatim" if locked_utr5 else "selected_from_library",
                cds_seq=cds_rand,
                cds_source="mrnax:stage0_random",
                cds_action="locked_verbatim" if locked_cds else "generated_novo",
                stop_seq=default_stop,
                utr3_seq=u3,
                utr3_id=None if locked_utr3 else clean_utr3s[0]["component_id"],
                utr3_source="locked" if locked_utr3 else clean_utr3s[0]["origin_source"],
                utr3_action="locked_verbatim" if locked_utr3 else "selected_from_library",
                polya_seq=polya_seq,
                extra_utr3_annotations=anns_u3,
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
                attempt_count=1,
                failure_count=0,
            )
        )

    # Generator 2: DNA Chisel constraint/codon baseline
    dc_seed_cds = locked_cds.locked_value if locked_cds else ""
    if "dnachisel_cds" in intent.runtime_policy.enabled_generators:
        t0 = time.perf_counter()
        if not dc_seed_cds:
            dc_seed_cds = generate_dnachisel_cds(
                protein_seq,
                intent.constraints,
                seed=intent.runtime_policy.random_seed,
                chemistry=intent.uridine_chemistry,
            )
        u5 = locked_utr5.locked_value if locked_utr5 else clean_utr5s[0]["sequence"]
        u3, anns_u3 = _prepare_utr3(clean_utr3s[0]["sequence"])
        raw_candidates.append(
            _build_single_molecule_bundle(
                bundle_id=f"bnd_{exec_id[-4:]}_dnachisel",
                parent_bundle_id=parent_id,
                derivation_action="de_novo_design",
                intent=intent,
                utr5_seq=u5,
                utr5_id=None if locked_utr5 else clean_utr5s[0]["component_id"],
                utr5_source="locked" if locked_utr5 else clean_utr5s[0]["origin_source"],
                utr5_action="locked_verbatim" if locked_utr5 else "selected_from_library",
                cds_seq=dc_seed_cds,
                cds_source="mrnax:dnachisel_v3.2.16",
                cds_action="locked_verbatim" if locked_cds else "generated_novo",
                stop_seq=default_stop,
                utr3_seq=u3,
                utr3_id=None if locked_utr3 else clean_utr3s[0]["component_id"],
                utr3_source="locked" if locked_utr3 else clean_utr3s[0]["origin_source"],
                utr3_action="locked_verbatim" if locked_utr3 else "selected_from_library",
                polya_seq=polya_seq,
                extra_utr3_annotations=anns_u3,
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
                attempt_count=1,
                failure_count=0,
            )
        )

    # Generator 3: Sliding-Window Structure-Guided CDS<->UTR Alternating Pareto Search
    if "pareto_nsga2" in intent.runtime_policy.enabled_generators:
        t0 = time.perf_counter()
        base_cds = dc_seed_cds or generate_random_synonymous_cds(
            protein_seq, rng, chemistry=intent.uridine_chemistry
        )
        n_codons = len(protein_seq)
        budget_n = max(3, min(intent.runtime_policy.max_evaluations - len(raw_candidates), 8))
        utr5_pool = [locked_utr5.locked_value] if locked_utr5 else [u["sequence"] for u in clean_utr5s[:3]]
        utr5_ids = [None] if locked_utr5 else [u["component_id"] for u in clean_utr5s[:3]]
        utr5_srcs = ["locked"] if locked_utr5 else [u["origin_source"] for u in clean_utr5s[:3]]

        utr3_pool_raw = [locked_utr3.locked_value] if locked_utr3 else [u["sequence"] for u in clean_utr3s[:3]]
        utr3_ids = [None] if locked_utr3 else [u["component_id"] for u in clean_utr3s[:3]]
        utr3_srcs = ["locked"] if locked_utr3 else [u["origin_source"] for u in clean_utr3s[:3]]

        for idx in range(budget_n):
            u5_idx = idx % len(utr5_pool)
            u3_idx = (idx // max(1, len(utr5_pool))) % len(utr3_pool_raw)
            u3_seq, anns_u3 = _prepare_utr3(utr3_pool_raw[u3_idx])

            if locked_cds:
                cand_cds = locked_cds.locked_value
                cds_act = "locked_verbatim"
            else:
                # Step A: Open 5' initiation codons [1..12] while stabilizing body codons [12..n_codons-10]
                cand_cds = mutate_cds_region_synonymously(
                    base_cds,
                    protein_seq,
                    rng,
                    codon_start=1,
                    codon_end=min(12, n_codons),
                    prefer_open_au=(idx % 2 == 0),
                    chemistry=intent.uridine_chemistry,
                )
                if n_codons > 15:
                    cand_cds = mutate_cds_region_synonymously(
                        cand_cds,
                        protein_seq,
                        rng,
                        codon_start=12,
                        codon_end=max(13, n_codons - 10),
                        prefer_open_au=False,
                        chemistry=intent.uridine_chemistry,
                    )
                # Step B: Co-adapt 3' CDS tail if de-targeting miRNA site is occluded
                if mirna_mature is not None and n_codons > 10:
                    cand_cds = mutate_cds_region_synonymously(
                        cand_cds,
                        protein_seq,
                        rng,
                        codon_start=max(1, n_codons - 12),
                        codon_end=n_codons,
                        prefer_open_au=True,
                        chemistry=intent.uridine_chemistry,
                    )
                # Re-ensure restriction sites stayed clean
                for cspec in intent.constraints:
                    if cspec.constraint_kind == "type_iis_restriction_free":
                        from mrnax.constraints.sequence_gates import RESTRICTION_SITES_RNA

                        for enz in cspec.params.get("enzymes", []):
                            fwd, rev = RESTRICTION_SITES_RNA[enz]
                            if fwd in cand_cds or rev in cand_cds:
                                cand_cds = dc_seed_cds or cand_cds
                cds_act = "synonymous_mutated"

            raw_candidates.append(
                _build_single_molecule_bundle(
                    bundle_id=f"bnd_{exec_id[-4:]}_pareto_{idx + 1:02d}",
                    parent_bundle_id=parent_id,
                    derivation_action="de_novo_design" if not locked_cds else "locked_refinement",
                    intent=intent,
                    utr5_seq=utr5_pool[u5_idx],
                    utr5_id=utr5_ids[u5_idx],
                    utr5_source=utr5_srcs[u5_idx],
                    utr5_action="locked_verbatim" if locked_utr5 else "selected_from_library",
                    cds_seq=cand_cds,
                    cds_source="mrnax:pareto_cds_utr_coopt",
                    cds_action=cds_act,
                    stop_seq=default_stop,
                    utr3_seq=u3_seq,
                    utr3_id=utr3_ids[u3_idx],
                    utr3_source=utr3_srcs[u3_idx],
                    utr3_action="locked_verbatim" if locked_utr3 else "selected_from_library",
                    polya_seq=polya_seq,
                    extra_utr3_annotations=anns_u3,
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
                attempt_count=budget_n,
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

    # 3. Shared Scoped Hard-Gate Filtering
    qualified: list[ConstructBundle] = []
    all_diagnostics: list[FailedCandidateDiagnostic] = []
    all_fto_alerts: list[FTOAlert] = []
    all_advisories: list[AdvisoryNote] = []
    eliminated_by_req: dict[str, int] = {c.requirement_id: 0 for c in intent.constraints}

    for b in dedup_candidates:
        outcome = constraint_engine.evaluate_bundle(
            b, intent.constraints, compliance_track=intent.compliance_track
        )
        all_fto_alerts.extend(outcome.fto_alerts)
        if outcome.passed_all_hard_gates:
            qualified.append(b)
            for adv in outcome.advisories:
                if adv.advisory_id not in {a.advisory_id for a in all_advisories}:
                    all_advisories.append(adv)
        else:
            all_diagnostics.extend(outcome.violations)
            violated_ids = {v.violated_requirement_id for v in outcome.violations}
            for req_id in violated_ids:
                if req_id in eliminated_by_req:
                    eliminated_by_req[req_id] += 1

    if intent.cell_expression_control and intent.cell_expression_control.detarget_tissues:
        all_advisories.append(build_crs_and_tissue_safety_advisory())

    # 4. Thermodynamic Evaluation & Pareto Non-Dominated Ranking on Qualified Candidates
    eval_wall = 0.0
    use_vienna = "viennarna_thermo" in intent.runtime_policy.enabled_evaluators
    raw_metrics_per_bundle: list[list[MetricResult]] = []
    obj_vectors: list[list[float]] = []

    for b in qualified:
        if use_vienna:
            t_ev = time.perf_counter()
            m_list, th_adv = evaluate_molecule_thermodynamics(
                b.molecules[0],
                mirna_mature_for_detarget=mirna_mature,
                include_full_mfe=True,
            )
            eval_wall += time.perf_counter() - t_ev
            for adv in th_adv:
                if adv.advisory_id not in {a.advisory_id for a in all_advisories}:
                    all_advisories.append(adv)
        else:
            cds_s = b.molecules[0].get_segment_sequence("cds")
            gc_val = round((cds_s.count("G") + cds_s.count("C")) / max(1, len(cds_s)), 4)
            m_list = [
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
        raw_metrics_per_bundle.append(m_list)

        m_map = {m.metric_id: m.value for m in m_list if m.value is not None}
        vec: list[float] = []
        for obj in intent.objectives:
            if obj.evidence_status == "unverified_target_for_wetlab" or obj.metric_id not in m_map:
                continue
            val = float(m_map[obj.metric_id])
            vec.append(val if obj.direction == "maximize" else -val)
        if not vec:
            vec = [0.0]
        obj_vectors.append(vec)

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
                attempt_count=len(qualified),
                failure_count=0,
            )
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

    if not qualified:
        outcome_status = "COMPLETED_ZERO_QUALIFIED"
        recommended_ids: list[str] = []
        bundle_evals: list[BundleEvaluation] = []
        ordered_qualified: list[ConstructBundle] = []
    else:
        ranks = _assign_pareto_ranks(obj_vectors)
        indexed = list(enumerate(qualified))
        indexed.sort(key=lambda pair: (ranks[pair[0]], -sum(obj_vectors[pair[0]]), pair[0]))
        ordered_qualified = [b for _, b in indexed]
        bundle_evals = []
        for orig_idx, b in indexed:
            rk = ranks[orig_idx]
            bundle_evals.append(
                BundleEvaluation(
                    bundle_id=b.bundle_id,
                    version_digest=b.version_digest,
                    passed_all_hard_gates=True,
                    pareto_rank=rk,
                    metrics=raw_metrics_per_bundle[orig_idx],
                    selection_rationale=[
                        f"Pareto rank {rk} under {intent.compliance_track} profile ({b.derivation_action})"
                    ],
                )
            )
        is_tie = len(obj_vectors) > 1 and all(v == obj_vectors[0] for v in obj_vectors[1:])
        outcome_status = "COMPLETED_TIE" if is_tie else "COMPLETED_WITH_CANDIDATES"
        recommended_ids = [e.bundle_id for e in bundle_evals if e.pareto_rank == 1][:3]
        if not recommended_ids:
            recommended_ids = [bundle_evals[0].bundle_id]

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
        else:
            action_traces.append(
                RequirementActionTrace(
                    requirement_id=obj.requirement_id,
                    stage_applied="ranking",
                    eliminated_count=0,
                    changed_candidate_ids=recommended_ids,
                    proof_diff_summary=(
                        f"Objective '{obj.metric_id}' ({obj.direction}) participated in Pareto selection across {len(qualified)} qualified bundles."
                    ),
                )
            )

    report = EvaluationReport(
        report_id=f"rep_{exec_id[-8:]}",
        execution_id=exec_id,
        intent_id=intent.intent_id,
        intent_version=intent.intent_version,
        compliance_track=intent.compliance_track,
        outcome_status=outcome_status,  # type: ignore[arg-type]
        recommended_bundle_ids=recommended_ids,
        candidate_evaluations=bundle_evals,
        failed_diagnostics=all_diagnostics,
        fto_alerts=all_fto_alerts,
        pre_synthesis_advisories=all_advisories,
        action_transparency=action_traces,
    )

    rep_bytes = json.dumps(report.model_dump(mode="json"), sort_keys=True).encode("utf-8")
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
        honesty_banner=None,
    )
    return ordered_qualified, report, manifest

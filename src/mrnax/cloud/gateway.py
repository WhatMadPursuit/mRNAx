"""VPS-ready 7x24 FastAPI Control Plane & Web Harness Gateway.

Exposes:
- Atomic Execution Lease & CommitAttempt CAS endpoints (`/v1/executions*`)
- Interactive Web UI Harness loop (`/v1/system/status`, `/v1/presets`, `/v1/projects/{id}/turn`,
  `/v1/projects/{id}/agent-chat`, `/v1/projects/{id}/state`, `/v1/projects/{id}/bundles/{bid}/export`)
- CORS for `https://whatmadpursuit.github.io` + localhost development
- Quota protection (`X-MRNAX-Access-Token` or ephemeral `X-MRNAX-LLM-Key` on privileged LLM/Colab endpoints)
"""

from __future__ import annotations

import hashlib
import hmac
import importlib.metadata
import importlib.util
import json
import math
import os
import re
import secrets
import shutil
import socket
import tempfile
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import AfterValidator, BaseModel, Field

from mrnax.cloud.cas_store import (
    ExecutionCASStore,
    IdempotencyPayloadMismatchError,
    StaleLeaseConflictError,
)
from mrnax.cloud.drive_rest_v3 import DriveRestV3Client, GdriveCliDriveV3Transport
from mrnax.constraints.frameshift_m1psi import remediate_m1psi_slippery_runs
from mrnax.constraints.registry import ScopedConstraintEngine
from mrnax.engine.pareto_director import project_report_for_intent
from mrnax.engine.profiles import resolve_default_seed_dir
from mrnax.harness.auth_secrets import SecretVault
from mrnax.harness.dialogue import LiveDialogueAgent, normalize_dialogue_provider
from mrnax.harness.state_machine import (
    CanonicalTurnInput,
    ProjectHarnessSession,
    build_preset_reference_r,
    project_lock,
)
from mrnax.harness.tools import AtomicToolbox
from mrnax.federated.external_evidence_adapter import CleanRoomPolicyError
from mrnax.io_insdc.genbank_io import ExportConstraintViolationError, export_construct_package
from mrnax.schemas.common import ComplianceProfile, SegmentScope, UridineChemistry
from mrnax.schemas.construct_bundle import (
    ChemistryTrackSpec,
    ConstructBundle,
    ProteinTargetSpec,
    RNAMoleculeSpec,
    SegmentSpan,
)
from mrnax.schemas.design_intent import DesignIntent
from mrnax.schemas.evaluation_report import EvaluationReport

DEFAULT_SEED_DIR = resolve_default_seed_dir()
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
_BOOT_TS = time.time()


def _validate_safe_id(raw_id: str, label: str = "id") -> str:
    if not _SAFE_ID_RE.fullmatch(raw_id):
        raise HTTPException(status_code=400, detail=f"Invalid {label}: '{raw_id}'")
    return raw_id


def _build_preset_reference_r(seed_dir: Path) -> ConstructBundle:
    return build_preset_reference_r(seed_dir)


def _compute_intent_diff(prev: DesignIntent | None, curr: DesignIntent) -> dict[str, Any] | None:
    if prev is None:
        return None
    v1_enz = next(
        (c.params.get("enzymes", []) for c in prev.constraints if c.constraint_kind == "type_iis_restriction_free"),
        [],
    )
    v2_enz = next(
        (c.params.get("enzymes", []) for c in curr.constraints if c.constraint_kind == "type_iis_restriction_free"),
        [],
    )
    p1 = prev.locked_scopes.get("protein_sequence")
    p2 = curr.locked_scopes.get("protein_sequence")
    return {
        "v1_id": prev.intent_id,
        "v2_id": curr.intent_id,
        "v1_version": prev.intent_version,
        "v2_version": curr.intent_version,
        "parent_intent_id_matches": curr.parent_intent_id == prev.intent_id,
        "protein_lock_preserved": bool(
            p1 and p2 and p1.locked_value == p2.locked_value
            and [t.protein_sequence for t in prev.protein_targets]
            == [t.protein_sequence for t in curr.protein_targets]
        ),
        "chemistry_preserved": prev.uridine_chemistry == curr.uridine_chemistry,
        "inherited_locks": list(curr.locked_scopes.keys()),
        "v1_enzymes": v1_enz,
        "v2_enzymes": v2_enz,
        "v1_compliance_track": prev.compliance_track,
        "v2_compliance_track": curr.compliance_track,
        "v1_cell_control": prev.cell_expression_control.model_dump(mode="json") if prev.cell_expression_control else None,
        "v2_cell_control": curr.cell_expression_control.model_dump(mode="json") if curr.cell_expression_control else None,
    }


def _serialize_molecule_view(mol: Any) -> dict[str, Any]:
    segs = []
    for s in mol.segments:
        subseq = mol.sequence[s.start : s.end]
        gc = round(((subseq.count("G") + subseq.count("C")) / max(1, len(subseq))) * 100.0, 2)
        segs.append(
            {
                "segment_type": s.segment_type,
                "start": s.start,
                "end": s.end,
                "length_nt": s.end - s.start,
                "gc_pct": gc,
                "component_id": s.component_id,
                "origin_source": s.origin_source,
                "license_track": s.license_track,
                "action_taken": s.action_taken,
                "sequence_preview": subseq if len(subseq) <= 120 else f"{subseq[:60]}...{subseq[-40:]}",
            }
        )
    full_gc = round(((mol.sequence.count("G") + mol.sequence.count("C")) / max(1, len(mol.sequence))) * 100.0, 2)
    return {
        "molecule_id": mol.molecule_id,
        "target": mol.effective_translation_target.model_dump(mode="json"),
        "length_nt": len(mol.sequence),
        "full_gc_pct": full_gc,
        "uridine_chemistry": mol.chemistry_track.uridine_chemistry,
        "segments": segs,
        "full_sequence": mol.sequence,
    }


def _serialize_candidate_card(bundle: ConstructBundle, eval_map: dict[str, Any]) -> dict[str, Any]:
    mol_views = [_serialize_molecule_view(m) for m in bundle.molecules]
    primary = mol_views[0]
    ev = eval_map.get(bundle.bundle_id)
    metrics = ev.metrics if ev is not None else []
    partial_ids = {
        m.metric_id for m in metrics
        if len(mol_views) > 1
        and (
            m.value is None
            or not math.isfinite(m.value)
            or (m.notes or "").startswith("incomplete_chain_coverage(")
            or (
                m.participated_in != "display_only"
                and (
                    m.chain_values is None
                    or len(m.chain_values) != len(mol_views)
                    or not all(math.isfinite(v) for v in m.chain_values)
                )
            )
        )
    }
    ranked_valid = [
        m for m in metrics
        if m.participated_in != "display_only"
        and m.value is not None
        and math.isfinite(m.value)
        and m.metric_id not in partial_ids
    ]
    ranked_ids = {m.metric_id for m in ranked_valid}
    details = []
    for metric in metrics:
        if metric.participated_in == "display_only" and metric.metric_id in ranked_ids:
            continue
        detail = metric.model_dump(mode="json")
        if metric.metric_id in partial_ids:
            detail["value"] = None
            detail["chain_values"] = None
            if (metric.notes or "").startswith("aggregated_"):
                detail["notes"] = "incomplete_chain_coverage"
        details.append(detail)
    metrics_summary: dict[str, float] = {}
    for m in metrics:
        if (
            m.participated_in == "display_only"
            and m.value is not None
            and math.isfinite(m.value)
            and m.metric_id not in partial_ids
        ):
            metrics_summary.setdefault(m.metric_id, m.value)
    for m in ranked_valid:
        assert m.value is not None
        metrics_summary[m.metric_id] = m.value
    return {
        "bundle_id": bundle.bundle_id,
        "version_digest": bundle.version_digest,
        "derivation_action": bundle.derivation_action,
        "parent_bundle_id": bundle.parent_bundle_id,
        "co_expression_mode": bundle.co_expression_mode,
        "molecules_count": len(mol_views),
        "target_stoichiometry": bundle.target_stoichiometry,
        "molecules": mol_views,
        "length_nt": primary["length_nt"],
        "full_gc_pct": primary["full_gc_pct"],
        "uridine_chemistry": primary["uridine_chemistry"],
        "metrics": metrics_summary,
        "metric_details": details,
        "pareto_rank": ev.pareto_rank if (ev is not None and not partial_ids and len(ranked_valid) > 0) else None,
        "passed_all_hard_gates": ev.passed_all_hard_gates if ev is not None else None,
        "segments": primary["segments"],
        "full_sequence": primary["full_sequence"],
    }


def _serialize_report_with_candidates(
    report: EvaluationReport | None,
    bundles: Iterable[ConstructBundle],
    *,
    intent: DesignIntent | None = None,
    seed_dir: Path | None = None,
    disallow_sandbox_override: bool = False,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    if report is None:
        return None, []
    bundles_by_id = {b.bundle_id: b for b in bundles}
    projected = project_report_for_intent(
        report,
        bundles_by_id,
        intent,
        seed_dir=seed_dir,
        disallow_sandbox_override=disallow_sandbox_override,
    )
    evaluations = {evaluation.bundle_id: evaluation for evaluation in projected.candidate_evaluations}
    cards: list[dict[str, Any]] = []
    for ev in projected.candidate_evaluations:
        bundle = bundles_by_id.get(ev.bundle_id)
        if bundle is not None:
            card = _serialize_candidate_card(bundle, evaluations)
            if card["pareto_rank"] is not None:
                cards.append(card)
    return projected.model_dump(mode="json"), cards


class SubmitExecutionRequest(BaseModel):
    submission_id: str
    request_digest: str
    intent_id: str


class LeaseAcquireRequest(BaseModel):
    worker_id: str
    ttl_seconds: int = 300
    lease_token: str | None = None


class CommitAttemptRequest(BaseModel):
    lease_epoch: int
    lease_token: str
    sealed_file_id: str
    sealed_sha256: str


class ProjectTurnRequest(CanonicalTurnInput):
    """HTTP `/turn` request inheriting 100% of `CanonicalTurnInput` wire validation."""


class AgentChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    provider: Annotated[str, AfterValidator(normalize_dialogue_provider)] = "deepseek"
    model: str | None = None


def create_control_plane_app(
    store: ExecutionCASStore,
    *,
    projects_root: Path | None = None,
    seed_dir: Path | None = None,
    access_token: str | None = None,
) -> FastAPI:
    resolved_seed = seed_dir or Path(os.environ.get("MRNAX_SEED_DIR", str(DEFAULT_SEED_DIR)))
    resolved_projects = projects_root or Path(
        os.environ.get("MRNAX_PROJECTS_ROOT", str(store.db_path.parent / "projects"))
    )
    resolved_projects.mkdir(parents=True, exist_ok=True)
    configured_token = (
        access_token
        if access_token is not None
        else os.environ.get("MRNAX_GATEWAY_ACCESS_TOKEN", "").strip()
    )

    app = FastAPI(title="mRNAx v2 7x24 Control Plane & Web Harness Gateway", version="0.2.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "https://whatmadpursuit.github.io",
            "http://localhost:8000",
            "http://localhost:5173",
            "http://localhost:3000",
            "http://127.0.0.1:8000",
            "http://127.0.0.1:5173",
        ],
        allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=[
            "Content-Type",
            "Authorization",
            "X-MRNAX-Access-Token",
            "X-MRNAX-Project-Key",
            "X-MRNAX-LLM-Key",
            "X-MRNAX-LLM-Provider",
        ],
    )

    # Bounded LRU sliding-window IP rate limiter across all gateway endpoints
    MAX_TRACKED_IPS = 2048
    ip_buckets: OrderedDict[str, deque[float]] = OrderedDict()
    rate_lock = threading.Lock()
    def _enforce_rate_limit(req: Request, max_per_min: int = 45) -> None:
        client_ip = req.client.host if req.client else "local"
        now = time.time()
        with rate_lock:
            dq = ip_buckets.get(client_ip)
            if dq is None:
                if len(ip_buckets) >= MAX_TRACKED_IPS:
                    ip_buckets.popitem(last=False)
                dq = deque()
                ip_buckets[client_ip] = dq
            else:
                ip_buckets.move_to_end(client_ip)
            while dq and now - dq[0] > 60.0:
                dq.popleft()
            if len(dq) >= max_per_min:
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded (max {max_per_min} requests/min per IP).",
                )
            dq.append(now)

    def _is_valid_admin_token(header_token: str | None) -> bool:
        return bool(
            configured_token
            and header_token
            and hmac.compare_digest(header_token.strip(), configured_token)
        )

    def _require_execution_auth(header_token: str | None) -> None:
        if configured_token and not _is_valid_admin_token(header_token):
            raise HTTPException(
                status_code=401,
                detail="Execution control-plane endpoints require a valid X-MRNAX-Access-Token.",
            )

    def _authorize_or_bind_project(
        project_id: str,
        req: Request,
        resp: Response | None = None,
        *,
        allow_bind_new: bool = False,
        header_project_key: str | None = None,
        header_access_token: str | None = None,
    ) -> tuple[str, str | None]:
        safe_pid = _validate_safe_id(project_id, "project_id")
        is_admin = _is_valid_admin_token(header_access_token)
        proj_dir = resolved_projects / safe_pid
        owner_file = proj_dir / "owner_token_sha256.txt"
        cookie_key = req.cookies.get(f"mrnax_pk_{safe_pid}", "")
        supplied_key = (header_project_key or cookie_key or "").strip()

        if allow_bind_new:
            proj_dir.mkdir(parents=True, exist_ok=True)
        if not proj_dir.is_dir():
            raise HTTPException(status_code=404, detail=f"Unknown project '{safe_pid}'")
        with project_lock(proj_dir):
            if owner_file.is_file():
                if is_admin:
                    return safe_pid, None
                expected_sha = owner_file.read_text(encoding="utf-8").strip()
                actual_sha = hashlib.sha256(supplied_key.encode("utf-8")).hexdigest() if supplied_key else ""
                if not supplied_key or not hmac.compare_digest(actual_sha, expected_sha):
                    raise HTTPException(
                        status_code=403,
                        detail="Project access denied: missing or invalid project capability key (X-MRNAX-Project-Key).",
                    )
                return safe_pid, None

            has_existing_state = (proj_dir / "intents").is_dir() and any((proj_dir / "intents").iterdir())
            if has_existing_state:
                if is_admin:
                    return safe_pid, None
                if configured_token:
                    raise HTTPException(
                        status_code=403,
                        detail="Project access denied: pre-existing unbound project requires X-MRNAX-Access-Token.",
                    )
                return safe_pid, None

            if not allow_bind_new:
                if is_admin and proj_dir.is_dir():
                    return safe_pid, None
                raise HTTPException(status_code=404, detail=f"Unknown project '{safe_pid}'")

            # Exclusive atomic creation (`O_CREAT | O_EXCL`) even when initiated by an admin turn!
            effective_key = supplied_key if len(supplied_key) >= 16 else f"pk_{secrets.token_urlsafe(24)}"
            proj_dir.mkdir(parents=True, exist_ok=True)
            try:
                fd = os.open(str(owner_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError as exc:
                expected_sha = owner_file.read_text(encoding="utf-8").strip()
                actual_sha = hashlib.sha256(supplied_key.encode("utf-8")).hexdigest() if supplied_key else ""
                if is_admin or (supplied_key and hmac.compare_digest(actual_sha, expected_sha)):
                    return safe_pid, None
                raise HTTPException(
                    status_code=403,
                    detail="Project access denied: concurrent project ownership collision.",
                ) from exc
            try:
                os.write(fd, (hashlib.sha256(effective_key.encode("utf-8")).hexdigest() + "\n").encode("utf-8"))
            finally:
                os.close(fd)
            if resp is not None:
                resp.set_cookie(
                    key=f"mrnax_pk_{safe_pid}",
                    value=effective_key,
                    httponly=True,
                    samesite="lax",
                )
            return safe_pid, effective_key

    def _get_session(safe_pid: str) -> ProjectHarnessSession:
        return ProjectHarnessSession(
            project_id=safe_pid,
            store_dir=resolved_projects / safe_pid,
            seed_dir=resolved_seed,
        )

    def _verify_privileged_access(
        header_token: str | None, header_llm_key: str | None
    ) -> bool:
        if header_llm_key and len(header_llm_key.strip()) >= 12:
            return True
        if _is_valid_admin_token(header_token):
            return True
        return False

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/system/status")
    def system_status(
        request: Request,
        x_mrnax_access_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        _enforce_rate_limit(request, max_per_min=90)
        viennarna_ok = importlib.util.find_spec("RNA") is not None
        viennarna_ver = "unavailable"
        if viennarna_ok:
            try:
                viennarna_ver = importlib.metadata.version("ViennaRNA")
            except Exception:
                viennarna_ver = "2.6+"

        vault = SecretVault()
        prov = vault.provider_status()
        configured_providers = [k for k, v in prov.items() if v.get("configured")]
        token_valid = _is_valid_admin_token(x_mrnax_access_token)
        return {
            "status": "online",
            "hostname": socket.gethostname(),
            "uptime_seconds": round(time.time() - _BOOT_TS, 1),
            "default_profile": "research_max",
            "viennarna_available": viennarna_ok,
            "viennarna_version": viennarna_ver,
            "colab_bridge_present": Path("/opt/vps-connect/colab-run").is_file()
            or Path("/Users/alex/.local/bin/colab").is_file(),
            "drive_bridge_present": Path("/opt/vps-connect/batch2-drive").is_file(),
            "server_llm_providers_ready": configured_providers,
            "gateway_token_required_for_server_llm": bool(configured_token),
            "caller_token_authenticated": token_valid,
        }

    @app.get("/v1/presets")
    def get_presets(request: Request) -> dict[str, Any]:
        _enforce_rate_limit(request, max_per_min=90)
        snap = json.loads((resolved_seed / "egfp_reporter_snapshot.json").read_text(encoding="utf-8"))
        ref_r = _build_preset_reference_r(resolved_seed)
        return {
            "egfp_reporter": snap["mammalian_reporter_egfp"],
            "uniprot_reference": snap["uniprot_reference"],
            "preset_reference_r": {
                "bundle_id": ref_r.bundle_id,
                "version_digest": ref_r.version_digest,
                "length_nt": len(ref_r.molecules[0].sequence),
                "segments": [s.model_dump(mode="json") for s in ref_r.molecules[0].segments],
                "sequence_preview": ref_r.molecules[0].sequence[:120] + "...",
            },
            "verified_empirical_rule": {
                "rule_id": "RUL-5UTR-UAUG-MRL-001",
                "status": "conditional_rule",
                "datasets": ["GSE114002 (6 replicates)", "GSE232927 (HEK293T N25)"],
                "readout": "Mean Ribosome Load (MRL, 12h HEK293T)",
                "delta_mrl_range": "[-2.8207, -0.8870]",
                "boundary_note": "Deduplicated as 1 lab lineage (Seelig UW); supports 5'UTR out-of-frame uAUG exclusion at ribosome-loading layer, not extrapolated to in vivo late AUC.",
            },
        }

    @app.get("/v1/projects/{project_id}/state")
    def get_project_state(
        project_id: str,
        request: Request,
        x_mrnax_project_key: str | None = Header(default=None),
        x_mrnax_access_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        _enforce_rate_limit(request, max_per_min=60)
        safe_pid, _ = _authorize_or_bind_project(
            project_id,
            request,
            allow_bind_new=False,
            header_project_key=x_mrnax_project_key,
            header_access_token=x_mrnax_access_token,
        )
        with project_lock(resolved_projects / safe_pid):
            session = _get_session(safe_pid)
            intents = [i.model_dump(mode="json") for i in session._intents]
            curr_intent = session._intents[-1] if session._intents else None
            try:
                latest_rep = session.latest_report()
            except ValueError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            report_view, cands = _serialize_report_with_candidates(
                latest_rep,
                session._bundles.values(),
                intent=curr_intent,
                seed_dir=resolved_seed,
                disallow_sandbox_override=True,
            )
            return {
                "project_id": session.project_id,
                "intent_count": len(intents),
                "intents": intents,
                "latest_report": report_view,
                "candidates": cands,
            }

    @app.post("/v1/projects/{project_id}/turn")
    def execute_project_turn(
        project_id: str,
        req: ProjectTurnRequest,
        request: Request,
        response: Response,
        x_mrnax_project_key: str | None = Header(default=None),
        x_mrnax_access_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        _enforce_rate_limit(request)
        safe_pid, minted_key = _authorize_or_bind_project(
            project_id,
            request,
            response,
            allow_bind_new=(req.mode == "initial"),
            header_project_key=x_mrnax_project_key,
            header_access_token=x_mrnax_access_token,
        )
        with project_lock(resolved_projects / safe_pid):
            session = _get_session(safe_pid)
            prev_intent = session._intents[-1] if session._intents else None

            if req.mode == "refine":
                if prev_intent is None:
                    raise HTTPException(
                        status_code=400,
                        detail="Cannot run 'refine' before creating an initial DesignIntent in this project.",
                    )
                eff_enz = (
                    req.updated_restriction_enzymes
                    if req.updated_restriction_enzymes is not None
                    else req.restriction_enzymes
                )
                intent = session.refine_turn_intent_canonical(
                    user_utterance=req.user_utterance,
                    restriction_enzymes=eff_enz,
                    locked_five_prime_utr=req.locked_five_prime_utr,
                    lock_cds_from_bundle_id=req.lock_cds_from_bundle_id,
                    unlock_scopes=req.unlock_scopes,
                    uridine_chemistry=req.uridine_chemistry,
                    compliance_track=req.compliance_track,
                    detarget_liver=req.detarget_liver,
                    use_commercial_surrogate=req.use_commercial_surrogate,
                )
            else:
                intent = session.create_turn_intent_canonical(
                    user_utterance=req.user_utterance,
                    protein_sequence=req.protein_sequence,
                    accession_or_source=req.accession_or_source,
                    entry_version=req.entry_version,
                    compliance_track=req.compliance_track,
                    uridine_chemistry=req.uridine_chemistry,
                    restriction_enzymes=req.restriction_enzymes,
                    locked_five_prime_utr=req.locked_five_prime_utr,
                    detarget_liver=req.detarget_liver,
                    prioritize_late_signal=req.prioritize_late_signal,
                    conflicting_single_pick=req.conflicting_single_pick,
                    reference_bundle_id=req.reference_bundle_id,
                    use_preset_reference_r=req.use_preset_reference_r,
                    use_commercial_surrogate=req.use_commercial_surrogate,
                    random_seed=req.random_seed,
                )

            diff = _compute_intent_diff(prev_intent, intent)
            if intent.status != "ready":
                out_clarify: dict[str, Any] = {
                    "status": "NEEDS_CLARIFICATION",
                    "blocking_questions": intent.blocking_questions,
                    "intent": intent.model_dump(mode="json"),
                    "intent_diff": diff,
                    "report": None,
                    "candidates": [],
                }
                if minted_key is not None:
                    out_clarify["project_key"] = minted_key
                return out_clarify

            bundles = session.execute_design(intent)
            latest_rep = session.latest_report(intent.intent_id)
            assert latest_rep is not None
            report_view, cards = _serialize_report_with_candidates(
                latest_rep,
                bundles,
                intent=intent,
                seed_dir=resolved_seed,
                disallow_sandbox_override=True,
            )
            replay_cmd = (
                f"uv run mrnax replay --intent-file intents/intent_v{intent.intent_version:03d}_{intent.intent_id}.json "
                f"--out-dir artifacts/replay_{intent.intent_id}"
            )
            out_ok: dict[str, Any] = {
                "status": "COMPLETED",
                "blocking_questions": [],
                "intent": intent.model_dump(mode="json"),
                "intent_diff": diff,
                "report": report_view,
                "candidates": cards,
                "cli_replay_command": replay_cmd,
            }
            if minted_key is not None:
                out_ok["project_key"] = minted_key
            return out_ok

    @app.post("/v1/projects/{project_id}/agent-chat")
    def execute_agent_chat(
        project_id: str,
        req: AgentChatRequest,
        request: Request,
        response: Response,
        x_mrnax_project_key: str | None = Header(default=None),
        x_mrnax_access_token: str | None = Header(default=None),
        x_mrnax_llm_key: str | None = Header(default=None),
    ) -> dict[str, Any]:
        _enforce_rate_limit(request, max_per_min=20)
        if not _verify_privileged_access(x_mrnax_access_token, x_mrnax_llm_key):
            raise HTTPException(
                status_code=401,
                detail="Privileged LLM Agent mode requires either X-MRNAX-Access-Token or your own X-MRNAX-LLM-Key.",
            )
        safe_pid, minted_key = _authorize_or_bind_project(
            project_id,
            request,
            response,
            allow_bind_new=True,
            header_project_key=x_mrnax_project_key,
            header_access_token=x_mrnax_access_token,
        )
        with project_lock(resolved_projects / safe_pid):
            session = _get_session(safe_pid)
            prev_intent = session._intents[-1] if session._intents else None
            toolbox = AtomicToolbox(session)
            agent = LiveDialogueAgent(toolbox, provider=req.provider, model=req.model)
            reply = agent.send_user_message(
                req.message,
                explicit_api_key=x_mrnax_llm_key.strip() if x_mrnax_llm_key else None,
            )
            curr_intent = session._intents[-1] if session._intents else None
            latest_rep = session.latest_report(curr_intent.intent_id if curr_intent else None)
            report_view, cards = _serialize_report_with_candidates(
                latest_rep,
                session._bundles.values(),
                intent=curr_intent,
                seed_dir=resolved_seed,
                disallow_sandbox_override=True,
            )
            diff = (
                _compute_intent_diff(prev_intent, curr_intent)
                if (curr_intent is not None and prev_intent is not None and curr_intent.intent_id != prev_intent.intent_id)
                else None
            )
            chat_out: dict[str, Any] = {
                "status": "COMPLETED",
                "assistant_reply": reply,
                "tool_calls": [
                    {"name": tc["name"], "arguments": tc["arguments"]}
                    for tc in agent.tool_call_trace
                ],
                "intent": curr_intent.model_dump(mode="json") if curr_intent else None,
                "intent_diff": diff,
                "report": report_view,
                "candidates": cards,
            }
            if minted_key is not None:
                chat_out["project_key"] = minted_key
            return chat_out

    @app.get("/v1/projects/{project_id}/bundles/{bundle_id}/export")
    def export_project_bundle(
        project_id: str,
        bundle_id: str,
        request: Request,
        x_mrnax_project_key: str | None = Header(default=None),
        x_mrnax_access_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        _enforce_rate_limit(request, max_per_min=45)
        safe_pid, _ = _authorize_or_bind_project(
            project_id,
            request,
            allow_bind_new=False,
            header_project_key=x_mrnax_project_key,
            header_access_token=x_mrnax_access_token,
        )
        with project_lock(resolved_projects / safe_pid):
            session = _get_session(safe_pid)
            safe_bid = _validate_safe_id(bundle_id, "bundle_id")
            try:
                bundle = session.get_bundle(safe_bid)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=f"Unknown bundle '{safe_bid}'") from exc

            try:
                intent = session.intent_for_bundle(safe_bid)
            except ValueError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            if intent.compliance_track == "commercial_clean":
                for rep in session._reports.values():
                    if rep.intent_id == intent.intent_id and any(
                        e.bundle_id == safe_bid for e in rep.candidate_evaluations
                    ):
                        if rep.isolated_sandbox_override or (
                            rep.rule_snapshot_version is not None
                            and not rep.rule_snapshot_version.startswith("rules_commercial_clean@")
                        ):
                            raise HTTPException(
                                status_code=409,
                                detail="Commercial export blocked: candidate originated from an isolated_sandbox_override or non-commercial rule snapshot.",
                            )
            constraint_engine = ScopedConstraintEngine.from_seed_dir(resolved_seed)
            with tempfile.TemporaryDirectory(prefix="mrnax_web_exp_") as td:
                out_dir = Path(td)
                try:
                    manifest = export_construct_package(
                        bundle,
                        out_dir,
                        constraint_engine=constraint_engine,
                        constraints=intent.constraints,
                        compliance_track=intent.compliance_track,
                        expected_chemistry=intent.uridine_chemistry,
                        locked_scopes=intent.locked_scopes,
                        expected_protein_targets=intent.protein_targets,
                    )
                except (CleanRoomPolicyError, ExportConstraintViolationError) as exc:
                    raise HTTPException(status_code=409, detail=str(exc)) from exc
                files_payload: dict[str, str] = {}
                for fname in ("full_rna.fasta", "cds_only.fasta", "construct.gb", "chemistry_metadata.json"):
                    files_payload[fname] = (out_dir / fname).read_text(encoding="utf-8")
                return {
                    "bundle_id": bundle.bundle_id,
                    "version_digest": bundle.version_digest,
                    "export_manifest": manifest,
                    "files": files_payload,
                }

    # --- Persistent CAS Lease & Execution Endpoints (Authenticated) ---
    @app.post("/v1/executions")
    def submit_execution(
        req: SubmitExecutionRequest,
        request: Request,
        x_mrnax_access_token: str | None = Header(default=None),
    ) -> dict[str, object]:
        _enforce_rate_limit(request, max_per_min=60)
        _require_execution_auth(x_mrnax_access_token)
        try:
            row = store.submit_execution(req.submission_id, req.request_digest, req.intent_id)
            return asdict(row)
        except IdempotencyPayloadMismatchError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/v1/executions/{execution_id}/leases")
    def acquire_lease(
        execution_id: str,
        req: LeaseAcquireRequest,
        request: Request,
        x_mrnax_access_token: str | None = Header(default=None),
    ) -> dict[str, object]:
        _enforce_rate_limit(request, max_per_min=90)
        _require_execution_auth(x_mrnax_access_token)
        try:
            grant = store.acquire_or_renew_lease(
                execution_id,
                req.worker_id,
                lease_token=req.lease_token,
                ttl_seconds=req.ttl_seconds,
            )
            return asdict(grant)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except StaleLeaseConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/v1/executions/{execution_id}/attempts/{attempt_id}:commit")
    def commit_attempt(
        execution_id: str,
        attempt_id: str,
        req: CommitAttemptRequest,
        request: Request,
        x_mrnax_access_token: str | None = Header(default=None),
    ) -> dict[str, object]:
        _enforce_rate_limit(request, max_per_min=60)
        _require_execution_auth(x_mrnax_access_token)
        try:
            receipt = store.commit_attempt_cas(
                execution_id=execution_id,
                attempt_id=attempt_id,
                lease_epoch=req.lease_epoch,
                lease_token=req.lease_token,
                sealed_file_id=req.sealed_file_id,
                sealed_sha256=req.sealed_sha256,
            )
            return asdict(receipt)
        except StaleLeaseConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/v1/executions/{execution_id}")
    def get_execution(
        execution_id: str,
        request: Request,
        x_mrnax_access_token: str | None = Header(default=None),
    ) -> dict[str, object]:
        _enforce_rate_limit(request, max_per_min=90)
        _require_execution_auth(x_mrnax_access_token)
        try:
            return asdict(store.get_execution(execution_id))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except StaleLeaseConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/v1/webhooks/stage-check")
    def trigger_lightweight_stage_check(request: Request) -> dict[str, Any]:
        """Ultra-lightweight (<50ms) periodic timer / webhook health & CAS integrity check for low-spec VPS."""
        _enforce_rate_limit(request, max_per_min=30)
        with store._connect() as conn:
            sqlite_check = str(conn.execute("PRAGMA quick_check;").fetchone()[0])
            exec_total = int(conn.execute("SELECT COUNT(*) FROM executions;").fetchone()[0])

        disk_usage = shutil.disk_usage(str(store.db_path.parent))
        rules_cc_path = resolved_seed / "rules_commercial_v1.json"
        rules_cc_count = (
            len(json.loads(rules_cc_path.read_text(encoding="utf-8")).get("rules", []))
            if rules_cc_path.is_file()
            else 0
        )
        surr_dir = resolved_seed / "surrogate_v1"
        surr_4heads_ok = all(
            (surr_dir / fn).is_file()
            for fn in (
                "utr5_mrl_pure_seq_lgbm.joblib",
                "cds_codon_stability_pure_seq_lgbm.joblib",
                "saluki_consensus_halflife_pure_seq_lgbm.joblib",
                "utr3_stability_pure_seq_lgbm.joblib",
            )
        )
        receipt = {
            "checked_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "hostname": socket.gethostname(),
            "sqlite_quick_check": sqlite_check,
            "executions_count": exec_total,
            "projects_count": sum(1 for p in resolved_projects.iterdir() if p.is_dir()),
            "commercial_clean_rules_loaded": rules_cc_count,
            "surrogate_4head_present": surr_4heads_ok,
            "free_disk_gb": round(disk_usage.free / (1024**3), 2),
            "status": (
                "PASS"
                if sqlite_check.lower() == "ok" and rules_cc_count >= 8 and surr_4heads_ok
                else "WARN"
            ),
        }
        (store.db_path.parent / "latest_stage_check.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return receipt

    @app.get("/v1/webhooks/stage-check/latest")
    def get_latest_stage_check(request: Request) -> dict[str, Any]:
        _enforce_rate_limit(request, max_per_min=60)
        fp = store.db_path.parent / "latest_stage_check.json"
        if not fp.is_file():
            return {"status": "NO_CHECK_RECORDED_YET"}
        return json.loads(fp.read_text(encoding="utf-8"))

    return app


def create_production_app() -> FastAPI:
    """ASGI factory for `uvicorn mrnax.cloud.gateway:create_production_app --factory`."""
    prod_token = os.environ.get("MRNAX_GATEWAY_ACCESS_TOKEN", "").strip()
    if not prod_token:
        raise RuntimeError(
            "Refusing to start production gateway without MRNAX_GATEWAY_ACCESS_TOKEN configured."
        )
    state_root = Path(os.environ.get("MRNAX_STATE_ROOT", "/var/lib/mrnax"))
    state_root.mkdir(parents=True, exist_ok=True)

    drive_client: DriveRestV3Client | None = None
    gdrive_bin = os.environ.get("MRNAX_GDRIVE_BIN", "").strip() or shutil.which("gdrive") or ""
    if gdrive_bin and Path(gdrive_bin).is_file():
        drive_client = DriveRestV3Client(GdriveCliDriveV3Transport(gdrive_bin=gdrive_bin))

    commits_folder = os.environ.get("MRNAX_COMMITS_SNAPSHOT_FOLDER_ID", "").strip() or None
    store = ExecutionCASStore(
        state_root / "control_plane.sqlite3",
        drive_verifier=drive_client,
        commits_snapshot_folder_id=commits_folder,
        runs_root_id=os.environ.get("MRNAX_DRIVE_RUNS_ROOT_ID", "").strip() or None,
        require_drive_verifier=True,
    )
    store.restore_committed_receipts()
    return create_control_plane_app(
        store,
        projects_root=state_root / "projects",
        access_token=prod_token,
    )

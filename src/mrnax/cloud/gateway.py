"""VPS-ready 7x24 FastAPI Control Plane & Web Harness Gateway.

Exposes:
- Atomic Execution Lease & CommitAttempt CAS endpoints (`/v1/executions*`)
- Interactive Web UI Harness loop (`/v1/system/status`, `/v1/presets`, `/v1/projects/{id}/turn`,
  `/v1/projects/{id}/agent-chat`, `/v1/projects/{id}/state`, `/v1/projects/{id}/bundles/{bid}/export`)
- CORS for `https://whatmadpursuit.github.io` + localhost development
- Quota protection (`X-MRNAX-Access-Token` or ephemeral `X-MRNAX-LLM-Key` on privileged LLM/Colab endpoints)
"""

from __future__ import annotations

import hmac
import json
import os
import re
import socket
import tempfile
import time
from collections import defaultdict, deque
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from mrnax.cloud.cas_store import (
    ExecutionCASStore,
    IdempotencyPayloadMismatchError,
    StaleLeaseConflictError,
)
from mrnax.constraints.frameshift_m1psi import remediate_m1psi_slippery_runs
from mrnax.constraints.registry import ScopedConstraintEngine
from mrnax.harness.auth_secrets import SecretVault
from mrnax.harness.dialogue import LiveDialogueAgent
from mrnax.harness.state_machine import ProjectHarnessSession
from mrnax.harness.tools import AtomicToolbox
from mrnax.io_insdc.genbank_io import export_construct_package
from mrnax.schemas.common import ComplianceProfile, SegmentScope, UridineChemistry
from mrnax.schemas.construct_bundle import (
    ChemistryTrackSpec,
    ConstructBundle,
    ProteinTargetSpec,
    RNAMoleculeSpec,
    SegmentSpan,
)
from mrnax.schemas.design_intent import DesignIntent

DEFAULT_SEED_DIR = Path(__file__).resolve().parents[3] / "registry_seed"
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
_BOOT_TS = time.time()


def _validate_safe_id(raw_id: str, label: str = "id") -> str:
    if not _SAFE_ID_RE.fullmatch(raw_id):
        raise HTTPException(status_code=400, detail=f"Invalid {label}: '{raw_id}'")
    return raw_id


def _build_preset_reference_r(seed_dir: Path) -> ConstructBundle:
    snap = json.loads((seed_dir / "egfp_reporter_snapshot.json").read_text(encoding="utf-8"))
    utrs = json.loads((seed_dir / "utr_components_v1.json").read_text(encoding="utf-8"))
    hba1_u5 = next(u for u in utrs["five_prime_utrs"] if u["gene_symbol"] == "HBA1")
    hba1_u3 = next(u for u in utrs["three_prime_utrs"] if u["gene_symbol"] == "HBA1")

    u5 = hba1_u5["sequence"]
    cds_wt, _ = remediate_m1psi_slippery_runs(snap["mammalian_reporter_egfp"]["reference_cds_rna_no_stop"])
    stop = snap["mammalian_reporter_egfp"]["reference_stop_codon"]
    u3 = hba1_u3["sequence"]
    pa = "A" * 100
    prot = snap["mammalian_reporter_egfp"]["protein_sequence"]

    c1 = len(u5)
    c2 = c1 + len(cds_wt)
    c3 = c2 + len(stop)
    c4 = c3 + len(u3)
    c5 = c4 + len(pa)

    mol = RNAMoleculeSpec(
        molecule_id="mol_egfp_ref_R",
        modality="linear_mrna",
        topology="linear",
        chemistry_track=ChemistryTrackSpec(
            uridine_chemistry="m1psi",
            cap_chemistry="cap1_cleancap_ag",
            source="user_declared",
        ),
        effective_translation_target=ProteinTargetSpec(
            protein_sequence=prot,
            transl_table=1,
            accession_or_source="GenBank:U55762.1",
            entry_version="AAB02572.1",
            confirmation_status="confirmed",
        ),
        sequence=u5 + cds_wt + stop + u3 + pa,
        segments=[
            SegmentSpan(
                segment_type="five_prime_utr",
                start=0,
                end=c1,
                component_id=hba1_u5["component_id"],
                origin_source=hba1_u5["origin_source"],
                license_track="commercial_clean",
                action_taken="locked_verbatim",
            ),
            SegmentSpan(
                segment_type="cds",
                start=c1,
                end=c2,
                component_id="cds_u55762_wt_m1psi_clean",
                origin_source="GenBank:U55762.1",
                license_track="commercial_clean",
                action_taken="locked_verbatim",
            ),
            SegmentSpan(
                segment_type="stop_cassette",
                start=c2,
                end=c3,
                component_id="stop_uaa",
                origin_source="GenBank:U55762.1",
                license_track="commercial_clean",
                action_taken="locked_verbatim",
            ),
            SegmentSpan(
                segment_type="three_prime_utr",
                start=c3,
                end=c4,
                component_id=hba1_u3["component_id"],
                origin_source=hba1_u3["origin_source"],
                license_track="commercial_clean",
                action_taken="locked_verbatim",
            ),
            SegmentSpan(
                segment_type="poly_a",
                start=c4,
                end=c5,
                component_id="polya_a100",
                origin_source="mrnax:homopolymer_tail",
                license_track="commercial_clean",
                action_taken="locked_verbatim",
            ),
        ],
    )
    return ConstructBundle.create(
        bundle_id="bnd_egfp_ref_R",
        parent_bundle_id=None,
        derivation_action="reference_import",
        target_stoichiometry={"mol_egfp_ref_R": 1.0},
        molecules=[mol],
    )


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
        "protein_lock_preserved": bool(p1 and p2 and p1.locked_value == p2.locked_value),
        "chemistry_preserved": prev.uridine_chemistry == curr.uridine_chemistry,
        "inherited_locks": list(curr.locked_scopes.keys()),
        "v1_enzymes": v1_enz,
        "v2_enzymes": v2_enz,
        "v1_compliance_track": prev.compliance_track,
        "v2_compliance_track": curr.compliance_track,
        "v1_cell_control": prev.cell_expression_control.model_dump(mode="json") if prev.cell_expression_control else None,
        "v2_cell_control": curr.cell_expression_control.model_dump(mode="json") if curr.cell_expression_control else None,
    }


def _serialize_candidate_card(bundle: ConstructBundle, eval_map: dict[str, Any]) -> dict[str, Any]:
    mol = bundle.molecules[0]
    ev = eval_map.get(bundle.bundle_id)
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
        "bundle_id": bundle.bundle_id,
        "version_digest": bundle.version_digest,
        "derivation_action": bundle.derivation_action,
        "parent_bundle_id": bundle.parent_bundle_id,
        "length_nt": len(mol.sequence),
        "full_gc_pct": full_gc,
        "uridine_chemistry": mol.chemistry_track.uridine_chemistry,
        "metrics": ev.metrics if ev is not None else {},
        "pareto_rank": ev.pareto_rank if ev is not None else 1,
        "passed_all_hard_gates": ev.passed_all_hard_gates if ev is not None else True,
        "segments": segs,
        "full_sequence": mol.sequence,
    }


class SubmitExecutionRequest(BaseModel):
    submission_id: str
    request_digest: str
    intent_id: str


class LeaseAcquireRequest(BaseModel):
    worker_id: str
    ttl_seconds: int = 300


class CommitAttemptRequest(BaseModel):
    lease_epoch: int
    lease_token: str
    sealed_file_id: str
    sealed_sha256: str


class ProjectTurnRequest(BaseModel):
    mode: str = Field(default="initial", pattern="^(initial|refine)$")
    user_utterance: str = Field(min_length=1, max_length=4000)
    protein_sequence: str | None = None
    accession_or_source: str = "GenBank:U55762.1"
    entry_version: str | None = "AAB02572.1"
    compliance_track: ComplianceProfile = "research_max"
    uridine_chemistry: UridineChemistry | None = None
    restriction_enzymes: list[str] | None = None
    locked_five_prime_utr: str | None = None
    lock_cds_from_bundle_id: str | None = None
    unlock_scopes: list[SegmentScope] | None = None
    detarget_liver: bool | None = None
    prioritize_late_signal: bool = False
    conflicting_single_pick: bool = False
    use_preset_reference_r: bool = False
    custom_reference_rna_sequence: str | None = None
    random_seed: int = Field(default=42, ge=0, le=2_147_483_647)


class AgentChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    provider: str = "deepseek"
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
            "X-MRNAX-LLM-Key",
            "X-MRNAX-LLM-Provider",
        ],
    )

    # Simple sliding-window IP rate limiter for public deterministic design turns (45 req / 60s per IP)
    ip_buckets: dict[str, deque[float]] = defaultdict(deque)

    def _enforce_rate_limit(req: Request, max_per_min: int = 45) -> None:
        client_ip = req.client.host if req.client else "local"
        now = time.time()
        dq = ip_buckets[client_ip]
        while dq and now - dq[0] > 60.0:
            dq.popleft()
        if len(dq) >= max_per_min:
            raise HTTPException(status_code=429, detail="Rate limit exceeded (max 45 requests/min per IP).")
        dq.append(now)

    def _get_session(project_id: str) -> ProjectHarnessSession:
        safe_pid = _validate_safe_id(project_id, "project_id")
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
        if configured_token and header_token and hmac.compare_digest(header_token.strip(), configured_token):
            return True
        return False

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/system/status")
    def system_status(
        x_mrnax_access_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        viennarna_ok = False
        viennarna_ver = "unavailable"
        try:
            import RNA  # type: ignore[import-untyped]

            viennarna_ok = True
            viennarna_ver = getattr(RNA, "__version__", "2.8.0")
        except Exception:
            pass

        vault = SecretVault()
        prov = vault.provider_status()
        configured_providers = [k for k, v in prov.items() if v.get("configured")]
        token_valid = bool(
            configured_token
            and x_mrnax_access_token
            and hmac.compare_digest(x_mrnax_access_token.strip(), configured_token)
        )
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
    def get_presets() -> dict[str, Any]:
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
    def get_project_state(project_id: str) -> dict[str, Any]:
        session = _get_session(project_id)
        intents = [i.model_dump(mode="json") for i in session._intents]
        latest_rep = list(session._reports.values())[-1] if session._reports else None
        eval_map = (
            {e.bundle_id: e for e in latest_rep.candidate_evaluations}
            if latest_rep is not None
            else {}
        )
        cands = [
            _serialize_candidate_card(b, eval_map)
            for b in session._bundles.values()
            if b.bundle_id in eval_map
        ]
        return {
            "project_id": session.project_id,
            "intent_count": len(intents),
            "intents": intents,
            "latest_report": latest_rep.model_dump(mode="json") if latest_rep else None,
            "candidates": cands,
        }

    @app.post("/v1/projects/{project_id}/turn")
    def execute_project_turn(project_id: str, req: ProjectTurnRequest, request: Request) -> dict[str, Any]:
        _enforce_rate_limit(request)
        session = _get_session(project_id)
        prev_intent = session._intents[-1] if session._intents else None

        if req.mode == "refine":
            if prev_intent is None:
                raise HTTPException(
                    status_code=400,
                    detail="Cannot run 'refine' before creating an initial DesignIntent in this project.",
                )
            add_locks: dict[SegmentScope, str] | None = None
            if req.lock_cds_from_bundle_id:
                b_lock = session.get_bundle(req.lock_cds_from_bundle_id)
                add_locks = {"cds": b_lock.molecules[0].get_segment_sequence("cds")}
            if req.locked_five_prime_utr:
                add_locks = dict(add_locks or {})
                add_locks["five_prime_utr"] = req.locked_five_prime_utr

            detarget = None
            prot_tiss = None
            if req.detarget_liver is True:
                detarget = ["liver_hepatocyte"]
                prot_tiss = ["skeletal_muscle"]
            elif req.detarget_liver is False:
                detarget = []
                prot_tiss = []

            intent = session.apply_incremental_intent_update(
                turn_id=f"turn_{len(session._intents) + 1:02d}",
                user_utterance=req.user_utterance,
                updated_restriction_enzymes=req.restriction_enzymes,
                additional_locked_scopes=add_locks,
                unlocked_scopes=req.unlock_scopes,
                updated_chemistry=req.uridine_chemistry,
                updated_compliance_track=req.compliance_track if req.compliance_track != prev_intent.compliance_track else None,
                detarget_tissues=detarget,
                protected_tissues=prot_tiss,
            )
        else:
            snap = json.loads((resolved_seed / "egfp_reporter_snapshot.json").read_text(encoding="utf-8"))
            prot_seq = (req.protein_sequence or snap["mammalian_reporter_egfp"]["protein_sequence"]).strip()
            ref_bundle: ConstructBundle | None = None
            if req.use_preset_reference_r:
                ref_bundle = _build_preset_reference_r(resolved_seed)

            locks_init: dict[SegmentScope, str] | None = (
                {"five_prime_utr": req.locked_five_prime_utr} if req.locked_five_prime_utr else None
            )
            detarget_init = ["liver_hepatocyte"] if req.detarget_liver else None
            prot_init = ["skeletal_muscle"] if req.detarget_liver else None
            if detarget_init is None and any(
                k in req.user_utterance.lower() for k in ("mir-122", "肝脏去靶向", "避开肝脏", "detarget liver")
            ):
                detarget_init = ["liver_hepatocyte"]
                prot_init = ["skeletal_muscle"]

            intent = session.create_initial_intent(
                turn_id=f"turn_{len(session._intents) + 1:02d}",
                user_utterance=req.user_utterance,
                protein_sequence=prot_seq,
                accession_or_source=req.accession_or_source,
                entry_version=req.entry_version,
                reference_bundle=ref_bundle,
                compliance_track=req.compliance_track,
                uridine_chemistry=req.uridine_chemistry,
                locked_regions=locks_init,
                restriction_enzymes=req.restriction_enzymes,
                detarget_tissues=detarget_init,
                protected_tissues=prot_init,
                prioritize_late_signal=req.prioritize_late_signal,
                conflicting_single_pick=req.conflicting_single_pick,
                random_seed=req.random_seed,
            )

        diff = _compute_intent_diff(prev_intent, intent)
        if intent.status != "ready":
            return {
                "status": "NEEDS_CLARIFICATION",
                "blocking_questions": intent.blocking_questions,
                "intent": intent.model_dump(mode="json"),
                "intent_diff": diff,
                "report": None,
                "candidates": [],
            }

        bundles = session.execute_design(intent)
        latest_rep = list(session._reports.values())[-1]
        eval_map = {e.bundle_id: e for e in latest_rep.candidate_evaluations}
        cards = [_serialize_candidate_card(b, eval_map) for b in bundles]
        replay_cmd = (
            f"uv run mrnax replay --intent-file intents/intent_v{intent.intent_version:03d}_{intent.intent_id}.json "
            f"--out-dir artifacts/replay_{intent.intent_id}"
        )
        return {
            "status": "COMPLETED",
            "blocking_questions": [],
            "intent": intent.model_dump(mode="json"),
            "intent_diff": diff,
            "report": latest_rep.model_dump(mode="json"),
            "candidates": cards,
            "cli_replay_command": replay_cmd,
        }

    @app.post("/v1/projects/{project_id}/agent-chat")
    def execute_agent_chat(
        project_id: str,
        req: AgentChatRequest,
        request: Request,
        x_mrnax_access_token: str | None = Header(default=None),
        x_mrnax_llm_key: str | None = Header(default=None),
    ) -> dict[str, Any]:
        _enforce_rate_limit(request, max_per_min=20)
        if not _verify_privileged_access(x_mrnax_access_token, x_mrnax_llm_key):
            raise HTTPException(
                status_code=401,
                detail="Privileged LLM Agent mode requires either X-MRNAX-Access-Token or your own X-MRNAX-LLM-Key.",
            )
        session = _get_session(project_id)
        prev_intent = session._intents[-1] if session._intents else None
        toolbox = AtomicToolbox(session)
        agent = LiveDialogueAgent(toolbox, provider=req.provider, model=req.model)
        reply = agent.send_user_message(
            req.message,
            explicit_api_key=x_mrnax_llm_key.strip() if x_mrnax_llm_key else None,
        )
        curr_intent = session._intents[-1] if session._intents else None
        latest_rep = list(session._reports.values())[-1] if session._reports else None
        eval_map = (
            {e.bundle_id: e for e in latest_rep.candidate_evaluations}
            if latest_rep is not None
            else {}
        )
        cards = [
            _serialize_candidate_card(b, eval_map)
            for b in session._bundles.values()
            if b.bundle_id in eval_map
        ]
        diff = (
            _compute_intent_diff(prev_intent, curr_intent)
            if (curr_intent is not None and prev_intent is not None and curr_intent.intent_id != prev_intent.intent_id)
            else None
        )
        return {
            "status": "COMPLETED",
            "assistant_reply": reply,
            "tool_calls": [
                {"name": tc["name"], "arguments": tc["arguments"]}
                for tc in agent.tool_call_trace
            ],
            "intent": curr_intent.model_dump(mode="json") if curr_intent else None,
            "intent_diff": diff,
            "report": latest_rep.model_dump(mode="json") if latest_rep else None,
            "candidates": cards,
        }

    @app.get("/v1/projects/{project_id}/bundles/{bundle_id}/export")
    def export_project_bundle(project_id: str, bundle_id: str) -> dict[str, Any]:
        session = _get_session(project_id)
        safe_bid = _validate_safe_id(bundle_id, "bundle_id")
        try:
            bundle = session.get_bundle(safe_bid)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown bundle '{safe_bid}'") from exc

        intent = session.latest_intent()
        constraint_engine = ScopedConstraintEngine.from_seed_dir(resolved_seed)
        with tempfile.TemporaryDirectory(prefix="mrnax_web_exp_") as td:
            out_dir = Path(td)
            manifest = export_construct_package(
                bundle,
                out_dir,
                constraint_engine=constraint_engine,
                constraints=intent.constraints,
                compliance_track=intent.compliance_track,
            )
            files_payload: dict[str, str] = {}
            for fname in ("full_rna.fasta", "cds_only.fasta", "construct.gb", "chemistry_metadata.json"):
                files_payload[fname] = (out_dir / fname).read_text(encoding="utf-8")
            return {
                "bundle_id": bundle.bundle_id,
                "version_digest": bundle.version_digest,
                "export_manifest": manifest,
                "files": files_payload,
            }

    # --- Persistent CAS Lease & Execution Endpoints ---
    @app.post("/v1/executions")
    def submit_execution(req: SubmitExecutionRequest) -> dict[str, object]:
        try:
            row = store.submit_execution(req.submission_id, req.request_digest, req.intent_id)
            return asdict(row)
        except IdempotencyPayloadMismatchError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/v1/executions/{execution_id}/leases")
    def acquire_lease(execution_id: str, req: LeaseAcquireRequest) -> dict[str, object]:
        try:
            grant = store.acquire_or_renew_lease(
                execution_id, req.worker_id, ttl_seconds=req.ttl_seconds
            )
            return asdict(grant)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except StaleLeaseConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/v1/executions/{execution_id}/attempts/{attempt_id}:commit")
    def commit_attempt(
        execution_id: str, attempt_id: str, req: CommitAttemptRequest
    ) -> dict[str, object]:
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
    def get_execution(execution_id: str) -> dict[str, object]:
        try:
            return asdict(store.get_execution(execution_id))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    return app


def create_production_app() -> FastAPI:
    """ASGI factory for `uvicorn mrnax.cloud.gateway:create_production_app --factory`."""
    state_root = Path(os.environ.get("MRNAX_STATE_ROOT", "/var/lib/mrnax"))
    state_root.mkdir(parents=True, exist_ok=True)
    store = ExecutionCASStore(state_root / "control_plane.sqlite3")
    return create_control_plane_app(
        store,
        projects_root=state_root / "projects",
    )

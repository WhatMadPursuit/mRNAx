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
from collections.abc import Callable, Coroutine, Iterable
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Any, Literal

import httpx
from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.routing import APIRoute
from pydantic import AfterValidator, BaseModel, ConfigDict, Discriminator, Field, StrictBool, StrictStr, Tag, field_validator

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
from mrnax.harness.dialogue import (
    _REDACT_KEY_RE,
    DEFAULT_PROVIDER_MODELS,
    SUPPORTED_PROVIDER_MODELS,
    LiveDialogueAgent,
    UpstreamProviderProtocolError,
    normalize_dialogue_provider,
    redact_secrets_in_obj,
    redact_secrets_in_text,
)
from mrnax.harness.state_machine import (
    InitialTurnInput,
    ProjectHarnessSession,
    RefineTurnInput,
    build_preset_reference_r,
    parse_bounded_json_int,
    parse_json_float_token_bounded,
    project_lock,
)
from mrnax.harness.tools import (
    AtomicToolbox,
    compare_construct_against_clinical_benchmarks,
    detect_available_engines,
    load_registry_json,
    query_scientific_rules,
    render_ivt_synthesis_order,
    render_qc_dossier_html,
    resolve_registry_file,
    run_structure_and_motif_analysis,
)
from mrnax.federated.external_evidence_adapter import CleanRoomPolicyError, ExternalEvidenceAdapterV2
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
from mrnax.schemas.evaluation_report import BundleEvaluation, EvaluationReport

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


_METRIC_DISPLAY_LABELS: dict[str, str] = {
    "utr5_mrl_pred": "5'UTR Predicted Ribosome Load (MRL)",
    "cds_stability_pred": "CDS Codon Stability Score",
    "consensus_halflife_pred": "Predicted Transcript Half-Life",
    "utr3_stability_pred": "3'UTR Stability Score",
    "kozak_unpaired_prob": "5' Initiation Unpaired Probability",
    "cds_mean_window_mfe": "CDS Sliding-Window MFE",
    "junction_folding_mfe": "UTR-CDS Junction Folding MFE",
    "full_transcript_mfe": "Full-Transcript MFE",
    "mirna_detarget_dg_total": "miRNA Detargeting Net dG",
    "cds_gc_fraction": "CDS GC Fraction",
    "structrfm_cds_median_shape": "structRFM Median SHAPE Reactivity",
    "orthrus_halflife_pred": "Orthrus Predicted Half-Life",
    "codontransformer_syn_loglik": "CodonTransformer Synonymous Log-Likelihood",
    "eternafold_kozak_unpaired_prob": "EternaFold Initiation Unpaired Probability",
    "eternafold_aup": "EternaFold Mean Unpaired Probability (AUP)",
}


def _derive_candidate_display_meta(
    bundle: ConstructBundle,
    pareto_rank: int | None,
) -> tuple[str, str]:
    bid = bundle.bundle_id
    low = bid.lower()
    if bundle.derivation_action == "reference_import" or low.endswith("_ref_r"):
        return "Reference Construct (R)", "reference_baseline"
    if low.endswith("_mina") or bundle.derivation_action == "locked_refinement":
        return "Conservative Refinement (Minimal Edit)", "conservative_refinement"
    if low.endswith("_expb"):
        return "Exploratory Re-Optimized Variant", "exploratory_reoptimization"
    if len(bundle.molecules) > 1:
        rank_prefix = f"Pareto #{pareto_rank} · " if pareto_rank is not None else ""
        return f"{rank_prefix}Multi-Chain Co-Expression Candidate", "multi_chain_coexpression"
    if "ld" in low or "lineardesign" in low:
        rank_prefix = f"Pareto #{pareto_rank} · " if pareto_rank is not None else ""
        return f"{rank_prefix}LinearDesign + 5' Polished Candidate", "lineardesign_5prime_polished"
    if pareto_rank == 1:
        return f"Pareto #1 Candidate ({bid[-6:]})", "pareto_frontier_top"
    if pareto_rank is not None:
        return f"Pareto #{pareto_rank} Candidate ({bid[-6:]})", "pareto_balanced"
    return f"Candidate ({bid[-8:]})", "de_novo_candidate"


def _serialize_candidate_card(
    bundle: ConstructBundle,
    ev: BundleEvaluation | None,
    *,
    hard_gate_total: int = 6,
    fto_alerts: Iterable[Any] = (),
) -> dict[str, Any]:
    mol_views = [_serialize_molecule_view(m) for m in bundle.molecules]
    primary = mol_views[0]
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

    eff_rank = ev.pareto_rank if (ev is not None and not partial_ids and len(ranked_valid) > 0) else None
    passed_gates = ev.passed_all_hard_gates if ev is not None else None
    display_label, strategy_tag = _derive_candidate_display_meta(bundle, eff_rank)
    total_gates = max(1, int(hard_gate_total))
    passed_count = total_gates if passed_gates is not False else 0

    bundle_alerts = [
        a for a in fto_alerts
        if getattr(a, "bundle_id", None) in (None, bundle.bundle_id)
    ]
    all_seg_clean = all(
        s.license_track == "commercial_clean"
        for m in bundle.molecules
        for s in m.segments
    )
    if any(bool(getattr(a, "blocked", False)) for a in bundle_alerts):
        fto_badge = "fto_blocked"
    elif bundle_alerts:
        fto_badge = "fto_advisory_review"
    elif all_seg_clean:
        fto_badge = "sequence_screen_clear"
    else:
        fto_badge = "research_track_only"

    clean_summary_metrics: list[dict[str, Any]] = []
    for d in details:
        val = d.get("value")
        if not isinstance(val, (int, float)) or not math.isfinite(val):
            continue
        mid = str(d.get("metric_id") or "")
        human_lbl = _METRIC_DISPLAY_LABELS.get(
            mid, mid.replace("_", " ").strip().title()
        )
        clean_summary_metrics.append(
            {
                "label": human_lbl,
                "value": round(float(val), 4),
                "unit": d.get("unit", ""),
                "chemistry_mode": d.get("chemistry_evaluation_mode", ""),
            }
        )

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
        "pareto_rank": eff_rank,
        "passed_all_hard_gates": passed_gates,
        "segments": primary["segments"],
        "full_sequence": primary["full_sequence"],
        "display_label": display_label,
        "strategy_tag": strategy_tag,
        "short_digest": bundle.version_digest[:12],
        "gate_passed_count": passed_count,
        "gate_total_count": total_gates,
        "fto_badge": fto_badge,
        "summary_metrics": clean_summary_metrics,
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
    hard_gate_count = (
        sum(1 for c in intent.constraints if c.severity == "hard_gate")
        if intent is not None and intent.constraints
        else 6
    )
    cards: list[dict[str, Any]] = []
    for ev in projected.candidate_evaluations:
        bundle = bundles_by_id.get(ev.bundle_id)
        if bundle is not None:
            card = _serialize_candidate_card(
                bundle,
                ev,
                hard_gate_total=hard_gate_count,
                fto_alerts=projected.fto_alerts,
            )
            if card["pareto_rank"] is not None:
                cards.append(card)
    return projected.model_dump(mode="json"), cards


class SubmitExecutionRequest(BaseModel):
    submission_id: str
    request_digest: str
    intent_id: str


class LeaseAcquireRequest(BaseModel):
    worker_id: str
    ttl_seconds: int = Field(default=300, ge=1, le=86_400)
    lease_token: str | None = None

    @field_validator("ttl_seconds", mode="before")
    @classmethod
    def _validate_ttl_before(cls, v: Any) -> int:
        return parse_bounded_json_int(v, field_name="ttl_seconds", min_val=1, max_val=86_400)


class CommitAttemptRequest(BaseModel):
    lease_epoch: int = Field(ge=0, le=2_147_483_647)
    lease_token: str
    sealed_file_id: str
    sealed_sha256: str

    @field_validator("lease_epoch", mode="before")
    @classmethod
    def _validate_epoch_before(cls, v: Any) -> int:
        return parse_bounded_json_int(v, field_name="lease_epoch", min_val=0, max_val=2_147_483_647)


class InitialTurnRequest(InitialTurnInput):
    mode: Literal["initial"] = "initial"


class RefineTurnRequest(RefineTurnInput):
    mode: Literal["refine"]


def _turn_mode_discriminator(v: Any) -> str:
    if isinstance(v, dict):
        m = v.get("mode", "initial")
        return m if isinstance(m, str) else "__invalid__"
    return getattr(v, "mode", "initial")


ProjectTurnRequest = Annotated[
    Annotated[InitialTurnRequest, Tag("initial")] | Annotated[RefineTurnRequest, Tag("refine")],
    Discriminator(_turn_mode_discriminator),
]


def _format_bounded_decimal(d: Decimal, max_chars: int = 64) -> str:
    """Format a Decimal in O(1) bounded characters without expanding exponents or calling int(d)."""
    if not d.is_finite():
        return str(d)
    tup = d.as_tuple()
    sign = "-" if tup.sign else ""
    digits = "".join(str(x) for x in tup.digits[:max_chars])
    if len(tup.digits) > max_chars:
        digits += "..."
    text = f"{sign}{digits}E{tup.exponent:+d}" if abs(int(tup.exponent)) > 12 else str(d)
    return text[:max_chars]


def _sanitize_validation_errors(
    obj: Any,
    depth: int = 0,
    *,
    extra_secrets: tuple[str, ...] = (),
) -> Any:
    """Sanitize FastAPI/Pydantic validation errors without calling decimal_encoder or echoing sensitive inputs."""
    if obj is None or isinstance(obj, bool):
        return obj
    if isinstance(obj, Decimal):
        return _format_bounded_decimal(obj)
    if isinstance(obj, int):
        if -2_147_483_648 <= obj <= 2_147_483_647:
            return obj
        return "<int out of 32-bit range>"
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else str(obj)
    if isinstance(obj, str):
        scrubbed = redact_secrets_in_text(obj, extra_secrets=extra_secrets)
        return scrubbed if len(scrubbed) <= 256 else (scrubbed[:253] + "...")
    if isinstance(obj, (bytes, bytearray)):
        return f"<{len(obj)} bytes>"
    if depth > 6:
        return "<truncated>"
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in list(obj.items())[:32]:
            ks = redact_secrets_in_text(str(k)[:64], extra_secrets=extra_secrets)
            if ks in ("url", "input"):
                continue
            out[ks] = _sanitize_validation_errors(v, depth + 1, extra_secrets=extra_secrets)
        return out
    if isinstance(obj, (list, tuple)):
        return [
            _sanitize_validation_errors(x, depth + 1, extra_secrets=extra_secrets)
            for x in list(obj)[:32]
        ]
    return redact_secrets_in_text(str(obj), extra_secrets=extra_secrets)[:128]


class _DecimalJsonRequest(Request):
    async def json(self) -> Any:
        if not hasattr(self, "_json"):
            raw = await self.body()
            self._json = json.loads(raw, parse_float=parse_json_float_token_bounded)
        return self._json


class _ExactDecimalAPIRoute(APIRoute):
    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        orig = super().get_route_handler()

        async def _handler(request: Request) -> Response:
            return await orig(_DecimalJsonRequest(request.scope, request.receive))

        return _handler


def _looks_like_credential_value(value: str) -> bool:
    low = value.lower()
    if low.startswith(("sk-", "pk_", "aiza", "bearer ")):
        return True
    return bool(_REDACT_KEY_RE.search(value)) or (redact_secrets_in_text(value) != value)


class AgentChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    provider: Annotated[str, AfterValidator(normalize_dialogue_provider)] | None = None
    model: str | None = Field(default=None, max_length=100)
    client_request_id: str | None = Field(default=None, max_length=80)

    @field_validator("model")
    @classmethod
    def _validate_model(cls, v: str | None) -> str | None:
        if v is None:
            return None
        cleaned = v.strip()
        if not cleaned:
            return None
        if _looks_like_credential_value(cleaned) or not re.fullmatch(r"[A-Za-z0-9._:/-]{1,100}", cleaned):
            raise ValueError("Invalid model identifier")
        return cleaned

    @field_validator("client_request_id")
    @classmethod
    def _validate_client_request_id(cls, v: str | None) -> str | None:
        if v is None:
            return None
        cleaned = v.strip()
        if not cleaned:
            return None
        if _looks_like_credential_value(cleaned) or not _SAFE_ID_RE.fullmatch(cleaned):
            raise ValueError("Invalid client_request_id")
        return cleaned


class StructureAnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sequence: StrictStr = Field(min_length=3, max_length=20000)
    five_prime_end: int | None = None
    cds_start: int | None = None
    cds_end: int | None = None
    three_prime_start: int | None = None
    three_prime_end: int | None = None
    modality: Literal["linear_mrna", "circrna", "sarna"] = "linear_mrna"
    include_eternafold: StrictBool = False

    @field_validator("five_prime_end", "cds_start", "cds_end", "three_prime_start", "three_prime_end", mode="before")
    @classmethod
    def _validate_coord_int(cls, v: Any) -> int | None:
        if v is None:
            return None
        return parse_bounded_json_int(v, field_name="coordinate", min_val=0, max_val=20000)


class EvidenceResolveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: StrictStr = Field(min_length=1, max_length=300)
    mode: Literal["protein", "literature", "both"] = "both"
    max_results: int = Field(default=5, ge=1, le=20)
    allow_network: StrictBool = True

    @field_validator("max_results", mode="before")
    @classmethod
    def _validate_max_res_int(cls, v: Any) -> int:
        return parse_bounded_json_int(v, field_name="max_results", min_val=1, max_val=20)


class ClinicalBenchmarkCompareRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sequence: StrictStr | None = Field(default=None, min_length=3, max_length=20000)
    project_id: StrictStr | None = Field(default=None, max_length=80)
    bundle_id: StrictStr | None = Field(default=None, max_length=80)
    modality: Literal["linear_mrna", "circrna", "sarna"] | None = None
    chemistry: Literal["m1psi", "pseudouridine", "unmodified_u"] | None = None


MUTATING_AGENT_TOOLS = frozenset({"design_constructs", "refine_construct", "ingest_experiment"})


def _atomic_write_private_json(target_path: Path, payload: Any) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target_path.parent, mode="w", encoding="utf-8", delete=False) as tmp:
        staged = Path(tmp.name)
        try:
            json.dump(payload, tmp, indent=2, ensure_ascii=False)
            tmp.write("\n")
            tmp.close()
            os.chmod(staged, 0o600)
            os.replace(staged, target_path)
        finally:
            staged.unlink(missing_ok=True)


_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


def _hash_client_request_id(req_id: str) -> str:
    return hashlib.sha256(req_id.encode("utf-8")).hexdigest()


def _is_https_request(req: Request) -> bool:
    xfp = (req.headers.get("x-forwarded-proto") or "").split(",")[0].strip().lower()
    return req.url.scheme == "https" or xfp == "https"


def _experiment_file_digests(proj_dir: Path) -> dict[str, str]:
    exp_dir = proj_dir / "experiments"
    if not exp_dir.is_dir():
        return {}
    digests: dict[str, str] = {}
    for p in sorted(exp_dir.glob("*.json")):
        try:
            digests[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        except OSError:
            continue
    return digests


def _project_has_existing_state(proj_dir: Path) -> bool:
    for sub in ("intents", "bundles", "reports", "experiments"):
        d = proj_dir / sub
        if d.is_dir() and any(d.iterdir()):
            return True
    for fname in (
        "agent_chat_history.json",
        "agent_chat_requests.json",
        "turn_requests.json",
        "pending_delivery_claim.json",
        "project_state.json",
        "quarantined_external_evidence.jsonl",
    ):
        if (proj_dir / fname).is_file():
            return True
    return False


def _load_agent_chat_history(
    proj_dir: Path,
    max_turns: int = 20,
    *,
    extra_secrets: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    history_path = proj_dir / "agent_chat_history.json"
    if not history_path.is_file():
        return []
    try:
        raw = json.loads(history_path.read_text(encoding="utf-8"))
    except ValueError:
        return []
    if not isinstance(raw, list):
        return []
    entries = [redact_secrets_in_obj(item, extra_secrets=extra_secrets) for item in raw if isinstance(item, dict)]
    return entries[-max_turns:]


def _save_agent_chat_history(
    proj_dir: Path,
    entries: list[dict[str, Any]],
    *,
    max_turns: int = 30,
    extra_secrets: tuple[str, ...] = (),
) -> None:
    bounded = entries[-max_turns:]
    sanitized = [redact_secrets_in_obj(item, extra_secrets=extra_secrets) for item in bounded if isinstance(item, dict)]
    _atomic_write_private_json(proj_dir / "agent_chat_history.json", sanitized)


def _sanitize_journal_entry(entry: dict[str, Any], *, extra_secrets: tuple[str, ...] = ()) -> dict[str, Any]:
    """Scrub secrets from user/model text in journal entries without altering protocol keys or state markers."""
    out: dict[str, Any] = {}
    protected_top = (
        "status",
        "request_digest",
        "turn_id_prefix",
        "baseline_intent_id",
        "baseline_intent_count",
        "baseline_exp_digests",
        "mutating_tools_committed",
        "mutated_intent_id",
        "expected_parent_intent_id",
        "updated_at",
    )
    for k in protected_top:
        if k in entry:
            out[k] = entry[k]

    if "pending_batch_tool_calls" in entry and isinstance(entry["pending_batch_tool_calls"], list):
        clean_batch: list[dict[str, Any]] = []
        for item in entry["pending_batch_tool_calls"]:
            if not isinstance(item, dict):
                continue
            item_out: dict[str, Any] = {}
            if "tool_call_id" in item:
                item_out["tool_call_id"] = item["tool_call_id"]
            if "name" in item:
                item_out["name"] = redact_secrets_in_text(str(item["name"]), extra_secrets=extra_secrets)
            if "raw_arguments" in item:
                item_out["raw_arguments"] = redact_secrets_in_text(
                    str(item["raw_arguments"]), extra_secrets=extra_secrets
                )
            clean_batch.append(item_out)
        out["pending_batch_tool_calls"] = clean_batch

    if "tool_calls" in entry and isinstance(entry["tool_calls"], list):
        clean_tcs: list[dict[str, Any]] = []
        for tc in entry["tool_calls"]:
            if not isinstance(tc, dict):
                continue
            tc_out: dict[str, Any] = {}
            if "tool_call_id" in tc:
                tc_out["tool_call_id"] = tc["tool_call_id"]
            if "name" in tc:
                tc_out["name"] = redact_secrets_in_text(str(tc["name"]), extra_secrets=extra_secrets)
            if "arguments" in tc:
                tc_out["arguments"] = redact_secrets_in_obj(
                    tc["arguments"], extra_secrets=extra_secrets, redact_keys=False
                )
            if "result" in tc:
                tc_out["result"] = redact_secrets_in_obj(
                    tc["result"], extra_secrets=extra_secrets, redact_keys=False
                )
            clean_tcs.append(tc_out)
        out["tool_calls"] = clean_tcs

    if "baseline_llm_history" in entry and isinstance(entry["baseline_llm_history"], list):
        clean_hist: list[dict[str, str]] = []
        for msg in entry["baseline_llm_history"]:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            content = msg.get("content")
            if role in ("user", "assistant") and isinstance(content, str):
                clean_hist.append(
                    {
                        "role": str(role),
                        "content": redact_secrets_in_text(content, extra_secrets=extra_secrets),
                    }
                )
        out["baseline_llm_history"] = clean_hist

    if "response" in entry and isinstance(entry["response"], dict):
        orig_resp = entry["response"]
        clean_resp = redact_secrets_in_obj(orig_resp, extra_secrets=extra_secrets, redact_keys=False)
        for pkey in ("status", "provider", "model"):
            if pkey in orig_resp:
                clean_resp[pkey] = orig_resp[pkey]
        if isinstance(orig_resp.get("intent"), dict) and isinstance(clean_resp.get("intent"), dict):
            for ipkey in ("intent_id", "intent_version", "status"):
                if ipkey in orig_resp["intent"]:
                    clean_resp["intent"][ipkey] = orig_resp["intent"][ipkey]
        out["response"] = clean_resp

    return out


class _JournalCorruptionError(ValueError):
    """Raised when a persistent idempotency journal cannot be parsed safely."""


def _load_hashed_requests(
    req_path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], bool]:
    if not req_path.is_file():
        return {}, {}, False
    try:
        raw = json.loads(req_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise _JournalCorruptionError(
            "Corrupted idempotency request journal; refusing duplicate execution."
        ) from exc
    if not isinstance(raw, dict):
        raise _JournalCorruptionError(
            "Corrupted idempotency request journal; refusing duplicate execution."
        )
    if "schema_version" in raw:
        if raw.get("schema_version") != 2 or not isinstance(raw.get("entries"), dict):
            raise _JournalCorruptionError(
                "Corrupted idempotency request journal; refusing duplicate execution."
            )
        entries = {
            k: dict(v)
            for k, v in raw["entries"].items()
            if isinstance(k, str) and isinstance(v, dict) and _SHA256_HEX_RE.fullmatch(k)
        }
        raw_legacy_hex = raw.get("legacy_hex")
        legacy_hex_v2 = (
            {
                k: dict(v)
                for k, v in raw_legacy_hex.items()
                if isinstance(k, str) and isinstance(v, dict) and _SHA256_HEX_RE.fullmatch(k)
            }
            if isinstance(raw_legacy_hex, dict)
            else {}
        )
        return entries, legacy_hex_v2, False

    # Legacy flat journal: hash non-hex IDs now and retain unverified 64-hex keys in `legacy_hex` across saves.
    out: dict[str, dict[str, Any]] = {}
    legacy_hex: dict[str, dict[str, Any]] = {}
    for k, v in raw.items():
        if not isinstance(k, str) or not isinstance(v, dict):
            continue
        rec_copy = dict(v)
        if _SHA256_HEX_RE.fullmatch(k):
            legacy_hex[k] = rec_copy
        else:
            hashed_k = _hash_client_request_id(k)
            existing = out.get(hashed_k)
            if existing is None or (existing.get("status") != "completed" and rec_copy.get("status") == "completed"):
                out[hashed_k] = rec_copy
    return out, legacy_hex, True


def _lookup_request_record(
    entries: dict[str, dict[str, Any]],
    legacy_hex: dict[str, dict[str, Any]],
    raw_id: str | None,
    req_key: str | None,
) -> tuple[dict[str, Any] | None, bool]:
    if not req_key:
        return None, False
    rec = entries.get(req_key)
    if rec is not None:
        return rec, False
    if req_key in legacy_hex:
        upgraded = dict(legacy_hex.pop(req_key))
        entries[req_key] = upgraded
        return upgraded, True
    if raw_id and _SHA256_HEX_RE.fullmatch(raw_id) and raw_id in legacy_hex:
        upgraded = dict(legacy_hex.pop(raw_id))
        entries[req_key] = upgraded
        return upgraded, True
    return None, False


def _save_hashed_requests(
    req_path: Path,
    requests_map: dict[str, dict[str, Any]],
    *,
    legacy_hex: dict[str, dict[str, Any]] | None = None,
    max_entries: int = 50,
    extra_secrets: tuple[str, ...] = (),
) -> None:
    normalized: dict[str, dict[str, Any]] = {}
    for k, v in requests_map.items():
        if not isinstance(k, str) or not isinstance(v, dict):
            continue
        hashed_k = k if _SHA256_HEX_RE.fullmatch(k) else _hash_client_request_id(k)
        normalized[hashed_k] = v
    unresolved = [(k, v) for k, v in normalized.items() if v.get("status") != "completed"]
    completed = [(k, v) for k, v in normalized.items() if v.get("status") == "completed"]
    kept_items = completed[-max_entries:] + unresolved
    sanitized = {
        k: _sanitize_journal_entry(v, extra_secrets=extra_secrets)
        for k, v in kept_items
    }
    payload: dict[str, Any] = {"schema_version": 2, "entries": sanitized}
    if legacy_hex:
        leg_items = [
            (k, v)
            for k, v in legacy_hex.items()
            if isinstance(k, str) and isinstance(v, dict) and _SHA256_HEX_RE.fullmatch(k)
        ]
        if leg_items:
            payload["legacy_hex"] = {
                k: _sanitize_journal_entry(v, extra_secrets=extra_secrets)
                for k, v in leg_items[-max_entries:]
            }
    _atomic_write_private_json(req_path, payload)


def _load_agent_chat_requests(
    proj_dir: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], bool]:
    return _load_hashed_requests(proj_dir / "agent_chat_requests.json")


def _save_agent_chat_requests(
    proj_dir: Path,
    requests_map: dict[str, dict[str, Any]],
    *,
    legacy_hex: dict[str, dict[str, Any]] | None = None,
    max_entries: int = 50,
    extra_secrets: tuple[str, ...] = (),
) -> None:
    _save_hashed_requests(
        proj_dir / "agent_chat_requests.json",
        requests_map,
        legacy_hex=legacy_hex,
        max_entries=max_entries,
        extra_secrets=extra_secrets,
    )


def _format_tool_calls_for_recovery_prompt(
    tool_calls: list[dict[str, Any]],
    *,
    max_total_chars: int = 4000,
    extra_secrets: tuple[str, ...] = (),
) -> str:
    items = [tc for tc in tool_calls if isinstance(tc, dict)]
    if not items:
        return "[]"
    per_item = max(220, (max_total_chars - 16) // len(items))
    formatted: list[dict[str, Any]] = []
    for tc in items:
        name_s = redact_secrets_in_text(str(tc.get("name") or ""), extra_secrets=extra_secrets)
        args_raw = redact_secrets_in_text(
            json.dumps(tc.get("arguments", {}), ensure_ascii=False),
            extra_secrets=extra_secrets,
        )
        res_obj = redact_secrets_in_obj(tc.get("result", {}), extra_secrets=extra_secrets, redact_keys=False)
        res_raw = json.dumps(res_obj, ensure_ascii=False)
        budget = max(80, per_item - len(name_s) - 64)
        args_budget = min(len(args_raw), max(40, budget // 3))
        res_budget = max(60, budget - args_budget)
        formatted.append(
            {
                "name": name_s,
                "arguments": (
                    tc.get("arguments", {})
                    if len(args_raw) <= args_budget
                    else (args_raw[: args_budget - 3] + "...")
                ),
                "result": (
                    res_obj
                    if len(res_raw) <= res_budget
                    else (res_raw[: res_budget - 3] + "...")
                ),
            }
        )
    return json.dumps(formatted, ensure_ascii=False)


def _safe_upstream_status_detail(
    exc: httpx.HTTPStatusError,
    *,
    extra_secrets: tuple[str, ...] = (),
) -> str:
    code_hint = ""
    try:
        body = exc.response.json()
        if isinstance(body, dict):
            err_obj = body.get("error") if isinstance(body.get("error"), dict) else body
            raw_code = err_obj.get("code") or err_obj.get("type")
            if (
                isinstance(raw_code, str)
                and re.fullmatch(r"[a-z][a-z0-9_]{0,39}", raw_code)
                and not raw_code.startswith(("sk-", "pk_", "aiza"))
                and redact_secrets_in_text(raw_code, extra_secrets=extra_secrets) == raw_code
            ):
                code_hint = f" ({raw_code})"
    except ValueError:
        pass
    return f"Upstream LLM provider returned HTTP {exc.response.status_code}{code_hint}."


def _build_agent_project_context(
    session: ProjectHarnessSession,
    *,
    target_intent: DesignIntent | None = None,
) -> dict[str, Any] | None:
    if not session._intents:
        return None
    curr = target_intent if target_intent is not None else session._intents[-1]
    enzymes = next(
        (c.params.get("enzymes", []) for c in curr.constraints if c.constraint_kind == "type_iis_restriction_free"),
        [],
    )
    recommended_ids: list[str] = []
    try:
        rep = session.latest_report(curr.intent_id)
        if rep is not None:
            recommended_ids = list(rep.recommended_bundle_ids)
    except ValueError:
        recommended_ids = []
    return {
        "project_id": session.project_id,
        "active_intent_id": curr.intent_id,
        "active_intent_version": curr.intent_version,
        "intent_status": curr.status,
        "entry_mode": curr.entry_mode,
        "compliance_track": curr.compliance_track,
        "uridine_chemistry": curr.uridine_chemistry,
        "protein_lengths_aa": [len(t.protein_sequence) for t in curr.protein_targets],
        "enzymes": enzymes,
        "locked_scopes": list(curr.locked_scopes.keys()),
        "bundle_count": len(session._bundles),
        "recommended_bundle_ids": recommended_ids,
        "active_reference_bundle_id": session.active_reference_bundle_id,
    }


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
    app.router.route_class = _ExactDecimalAPIRoute

    def _collect_active_secrets(*extra: str | None) -> tuple[str, ...]:
        vault_secrets = [v for v in SecretVault()._read_all().values() if isinstance(v, str)]
        return tuple(
            dict.fromkeys(
                s.strip()
                for s in (*vault_secrets, configured_token, *extra)
                if isinstance(s, str) and len(s.strip()) >= 2
            )
        )

    @app.exception_handler(RequestValidationError)
    async def _bounded_validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        cookie_secrets = [v for k, v in request.cookies.items() if k.startswith("mrnax_pk_")]
        secrets_tuple = _collect_active_secrets(
            request.headers.get("X-MRNAX-LLM-Key"),
            request.headers.get("X-MRNAX-Access-Token"),
            request.headers.get("X-MRNAX-Project-Key"),
            *cookie_secrets,
        )
        return JSONResponse(
            status_code=422,
            content={"detail": _sanitize_validation_errors(exc.errors(), extra_secrets=secrets_tuple)},
        )
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
            "X-MRNAX-Idempotency-Key",
            "X-MRNAX-LLM-Key",
            "X-MRNAX-LLM-Provider",
        ],
        expose_headers=["X-MRNAX-Project-Key"],
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
        pending_claim_proof: str | None = None,
    ) -> tuple[str, str | None]:
        safe_pid = _validate_safe_id(project_id, "project_id")
        is_admin = _is_valid_admin_token(header_access_token)
        proj_dir = resolved_projects / safe_pid
        owner_file = proj_dir / "owner_token_sha256.txt"
        claim_file = proj_dir / "pending_delivery_claim.json"
        cookie_key = req.cookies.get(f"mrnax_pk_{safe_pid}", "")
        supplied_key = (header_project_key or cookie_key or "").strip()
        https_req = _is_https_request(req)

        def _parse_owner_file(path: Path) -> tuple[str, str | None]:
            lines = path.read_text(encoding="utf-8").splitlines()
            expected_sha = lines[0].strip() if lines else ""
            inline_claim = None
            for ln in lines[1:]:
                if ln.startswith("claim:"):
                    val = ln[6:].strip()
                    inline_claim = val or None
                    break
            return expected_sha, inline_claim

        def _write_owner_file_exclusive(path: Path, expected_sha: str, claim_proof: str | None) -> None:
            payload = (expected_sha + "\n" + (f"claim:{claim_proof}\n" if claim_proof else "")).encode("utf-8")
            with tempfile.NamedTemporaryFile(dir=path.parent, mode="wb", delete=False) as tmp_owner:
                staged = Path(tmp_owner.name)
                try:
                    tmp_owner.write(payload)
                    tmp_owner.close()
                    os.chmod(staged, 0o600)
                    os.link(staged, path)
                finally:
                    staged.unlink(missing_ok=True)

        def _atomic_replace_owner_file(path: Path, expected_sha: str, claim_proof: str | None) -> None:
            payload = (expected_sha + "\n" + (f"claim:{claim_proof}\n" if claim_proof else "")).encode("utf-8")
            with tempfile.NamedTemporaryFile(dir=path.parent, mode="wb", delete=False) as tmp_owner:
                staged = Path(tmp_owner.name)
                try:
                    tmp_owner.write(payload)
                    tmp_owner.close()
                    os.chmod(staged, 0o600)
                    os.replace(staged, path)
                finally:
                    staged.unlink(missing_ok=True)

        if allow_bind_new:
            proj_dir.mkdir(parents=True, exist_ok=True)
        if not proj_dir.is_dir():
            raise HTTPException(status_code=404, detail=f"Unknown project '{safe_pid}'")
        with project_lock(proj_dir):
            if owner_file.is_file():
                expected_sha, stored_claim = _parse_owner_file(owner_file)
                if stored_claim is None and claim_file.is_file():
                    try:
                        claim_doc = json.loads(claim_file.read_text(encoding="utf-8"))
                        stored_claim = str(claim_doc.get("claim_proof_sha256") or "") or None
                    except ValueError:
                        stored_claim = None
                if is_admin:
                    return safe_pid, None
                actual_sha = hashlib.sha256(supplied_key.encode("utf-8")).hexdigest() if supplied_key else ""
                if supplied_key and hmac.compare_digest(actual_sha, expected_sha):
                    if stored_claim is not None:
                        _atomic_replace_owner_file(owner_file, expected_sha, None)
                    claim_file.unlink(missing_ok=True)
                    return safe_pid, None
                # First-call BYO / turn crash recovery: retain claim proof until client demonstrates key receipt.
                if (not supplied_key) and pending_claim_proof and stored_claim:
                    if hmac.compare_digest(pending_claim_proof, stored_claim):
                        rotated_key = f"pk_{secrets.token_urlsafe(24)}"
                        rotated_sha = hashlib.sha256(rotated_key.encode("utf-8")).hexdigest()
                        _atomic_replace_owner_file(owner_file, rotated_sha, stored_claim)
                        if resp is not None:
                            resp.set_cookie(
                                key=f"mrnax_pk_{safe_pid}",
                                value=rotated_key,
                                httponly=True,
                                secure=https_req,
                                samesite="lax",
                            )
                        return safe_pid, rotated_key
                raise HTTPException(
                    status_code=403,
                    detail="Project access denied: missing or invalid project capability key (X-MRNAX-Project-Key).",
                )

            has_existing_state = _project_has_existing_state(proj_dir)
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

            # Exclusive atomic creation of both capability digest and pending claim proof together (`os.link`).
            was_server_minted = len(supplied_key) < 16
            effective_key = supplied_key if not was_server_minted else f"pk_{secrets.token_urlsafe(24)}"
            effective_sha = hashlib.sha256(effective_key.encode("utf-8")).hexdigest()
            effective_claim = pending_claim_proof if (was_server_minted and pending_claim_proof) else None
            proj_dir.mkdir(parents=True, exist_ok=True)
            try:
                _write_owner_file_exclusive(owner_file, effective_sha, effective_claim)
            except FileExistsError as exc:
                expected_sha, _ = _parse_owner_file(owner_file)
                actual_sha = hashlib.sha256(supplied_key.encode("utf-8")).hexdigest() if supplied_key else ""
                if is_admin or (supplied_key and hmac.compare_digest(actual_sha, expected_sha)):
                    return safe_pid, None
                raise HTTPException(
                    status_code=403,
                    detail="Project access denied: concurrent project ownership collision.",
                ) from exc
            if resp is not None:
                resp.set_cookie(
                    key=f"mrnax_pk_{safe_pid}",
                    value=effective_key,
                    httponly=True,
                    secure=https_req,
                    samesite="lax",
                )
            return safe_pid, effective_key

    def _is_legacy_unbound_anonymous(proj_dir: Path, header_access_token: str | None) -> bool:
        owner_file = proj_dir / "owner_token_sha256.txt"
        return (not owner_file.is_file()) and _project_has_existing_state(proj_dir) and (not _is_valid_admin_token(header_access_token))

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
        engines_status = detect_available_engines(resolved_seed)
        return {
            "status": "online",
            "hostname": socket.gethostname(),
            "uptime_seconds": round(time.time() - _BOOT_TS, 1),
            "default_profile": "research_max",
            "viennarna_available": viennarna_ok,
            "viennarna_version": viennarna_ver,
            "available_engines": engines_status,
            "supported_modalities": ["linear_mrna", "circrna", "sarna"],
            "colab_bridge_present": Path("/opt/vps-connect/colab-run").is_file()
            or Path("/Users/alex/.local/bin/colab").is_file(),
            "drive_bridge_present": Path("/opt/vps-connect/batch2-drive").is_file(),
            "server_llm_providers_ready": configured_providers,
            "default_llm_provider": "deepseek",
            "default_llm_models": dict(DEFAULT_PROVIDER_MODELS),
            "supported_llm_models": {k: list(v) for k, v in SUPPORTED_PROVIDER_MODELS.items()},
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
        proj_dir = resolved_projects / safe_pid
        chat_authorized = (proj_dir / "owner_token_sha256.txt").is_file() or _is_valid_admin_token(x_mrnax_access_token)
        cookie_cap = request.cookies.get(f"mrnax_pk_{safe_pid}")
        read_secrets = _collect_active_secrets(x_mrnax_access_token, x_mrnax_project_key, cookie_cap)
        with project_lock(proj_dir):
            session = _get_session(safe_pid)
            intents = redact_secrets_in_obj(
                [i.model_dump(mode="json") for i in session._intents],
                extra_secrets=read_secrets,
            )
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
                "quarantined_bundles": session.quarantined_bundles,
                "intent_count": len(intents),
                "intents": intents,
                "latest_report": redact_secrets_in_obj(report_view, extra_secrets=read_secrets),
                "candidates": redact_secrets_in_obj(cands, extra_secrets=read_secrets),
                "chat_history": _load_agent_chat_history(proj_dir, extra_secrets=read_secrets)
                if chat_authorized
                else [],
            }

    @app.post("/v1/projects/{project_id}/turn")
    def execute_project_turn(
        project_id: str,
        req: ProjectTurnRequest,
        request: Request,
        response: Response,
        x_mrnax_project_key: str | None = Header(default=None),
        x_mrnax_access_token: str | None = Header(default=None),
        x_mrnax_idempotency_key: str | None = Header(default=None),
    ) -> dict[str, Any]:
        _enforce_rate_limit(request)
        turn_idemp_raw = (x_mrnax_idempotency_key or "").strip() if x_mrnax_idempotency_key is not None else ""
        if x_mrnax_idempotency_key is not None:
            if (not turn_idemp_raw) or _looks_like_credential_value(turn_idemp_raw) or not _SAFE_ID_RE.fullmatch(turn_idemp_raw):
                raise HTTPException(
                    status_code=422,
                    detail="Invalid X-MRNAX-Idempotency-Key header.",
                )
        turn_req_key = _hash_client_request_id(turn_idemp_raw) if turn_idemp_raw else None
        turn_req_digest = hashlib.sha256(req.model_dump_json().encode("utf-8")).hexdigest() if turn_req_key else None
        safe_pid_preview = _validate_safe_id(project_id, "project_id")
        pre_turn_cookie = request.cookies.get(f"mrnax_pk_{safe_pid_preview}", "")
        turn_claim_proof = (
            hashlib.sha256(f"turn:{turn_req_key}:{turn_req_digest}".encode("utf-8")).hexdigest()
            if (
                req.mode == "initial"
                and turn_req_key
                and turn_req_digest
                and not _is_valid_admin_token(x_mrnax_access_token)
                and not (x_mrnax_project_key or pre_turn_cookie or "").strip()
            )
            else None
        )

        safe_pid, minted_key = _authorize_or_bind_project(
            project_id,
            request,
            response,
            allow_bind_new=(req.mode == "initial"),
            header_project_key=x_mrnax_project_key,
            header_access_token=x_mrnax_access_token,
            pending_claim_proof=turn_claim_proof,
        )
        proj_dir = resolved_projects / safe_pid
        cookie_cap = request.cookies.get(f"mrnax_pk_{safe_pid}")
        turn_secrets = _collect_active_secrets(x_mrnax_access_token, x_mrnax_project_key, cookie_cap, minted_key)

        with project_lock(proj_dir):
            turn_journal_path = proj_dir / "turn_requests.json"
            if turn_req_key:
                try:
                    turn_requests_map, turn_legacy_hex, turn_migrated = _load_hashed_requests(turn_journal_path)
                except _JournalCorruptionError as exc:
                    raise HTTPException(status_code=409, detail=str(exc)) from exc
                prior_turn, turn_lookup_migrated = _lookup_request_record(
                    turn_requests_map, turn_legacy_hex, turn_idemp_raw, turn_req_key
                )
                turn_migrated = turn_migrated or turn_lookup_migrated
            else:
                turn_requests_map, turn_legacy_hex, turn_migrated = {}, {}, False
                prior_turn = None
            if prior_turn and turn_req_digest:
                if prior_turn.get("request_digest") and prior_turn.get("request_digest") != turn_req_digest:
                    raise HTTPException(
                        status_code=409,
                        detail="Idempotency conflict: X-MRNAX-Idempotency-Key reused with a different request payload.",
                    )
                if prior_turn.get("status") == "completed" and isinstance(prior_turn.get("response"), dict):
                    if turn_migrated:
                        _save_hashed_requests(
                            turn_journal_path,
                            turn_requests_map,
                            legacy_hex=turn_legacy_hex,
                            extra_secrets=turn_secrets,
                        )
                    orig_resp = dict(prior_turn["response"])
                    cached_turn = redact_secrets_in_obj(orig_resp, extra_secrets=turn_secrets, redact_keys=False)
                    if isinstance(orig_resp.get("status"), str):
                        cached_turn["status"] = orig_resp["status"]
                    if isinstance(orig_resp.get("intent"), dict) and isinstance(cached_turn.get("intent"), dict):
                        for ipkey in ("intent_id", "intent_version", "status"):
                            if ipkey in orig_resp["intent"]:
                                cached_turn["intent"][ipkey] = orig_resp["intent"][ipkey]
                    if minted_key is not None:
                        cached_turn["project_key"] = minted_key
                    return cached_turn

            session = _get_session(safe_pid)
            owned_turn_prefix = f"turnreq_{turn_req_key[:16]}" if turn_req_key else None
            owned_turn_intent = (
                next(
                    (
                        i
                        for i in reversed(session._intents)
                        if i.created_from_turn_ids and i.created_from_turn_ids[-1].startswith(f"{owned_turn_prefix}_")
                    ),
                    None,
                )
                if (prior_turn and owned_turn_prefix)
                else None
            )

            clean_turn_utterance = redact_secrets_in_text(req.user_utterance, extra_secrets=turn_secrets)
            if owned_turn_intent is not None:
                intent = owned_turn_intent
                base_id = prior_turn.get("baseline_intent_id") if prior_turn else None
                prev_intent = (
                    next((i for i in session._intents if i.intent_id == base_id), None)
                    if isinstance(base_id, str)
                    else None
                )
            else:
                expected_turn_parent_id: str | None = None
                if prior_turn is not None and "baseline_intent_id" in prior_turn:
                    saved_base_id = prior_turn.get("baseline_intent_id")
                    prev_intent = (
                        next((i for i in session._intents if i.intent_id == saved_base_id), None)
                        if isinstance(saved_base_id, str)
                        else None
                    )
                    if isinstance(saved_base_id, str) and prev_intent is None:
                        raise HTTPException(
                            status_code=409,
                            detail=f"Concurrent revision conflict: baseline intent '{saved_base_id}' no longer present.",
                        )
                    expected_turn_parent_id = saved_base_id if isinstance(saved_base_id, str) else None
                    if not isinstance(req, RefineTurnRequest):
                        cur_head_id = session._intents[-1].intent_id if session._intents else None
                        if cur_head_id != expected_turn_parent_id:
                            raise HTTPException(
                                status_code=409,
                                detail=(
                                    f"Concurrent revision conflict: expected baseline intent '{expected_turn_parent_id}', "
                                    f"found '{cur_head_id}'."
                                ),
                            )
                else:
                    prev_intent = session._intents[-1] if session._intents else None
                    if turn_req_key and turn_req_digest:
                        turn_requests_map[turn_req_key] = {
                            "status": "in_progress",
                            "request_digest": turn_req_digest,
                            "baseline_intent_id": prev_intent.intent_id if prev_intent else None,
                        }
                        _save_hashed_requests(
                            turn_journal_path,
                            turn_requests_map,
                            legacy_hex=turn_legacy_hex,
                            extra_secrets=turn_secrets,
                        )
                assigned_turn_id = (
                    f"{owned_turn_prefix}_{len(session._intents) + 1:02d}"
                    if owned_turn_prefix
                    else None
                )

                def _cleanup_uncommitted_turn() -> None:
                    if prior_turn is None and turn_req_key and turn_req_key in turn_requests_map:
                        turn_requests_map.pop(turn_req_key, None)
                        if turn_requests_map or turn_legacy_hex:
                            _save_hashed_requests(
                                turn_journal_path,
                                turn_requests_map,
                                legacy_hex=turn_legacy_hex,
                                extra_secrets=turn_secrets,
                            )
                        else:
                            turn_journal_path.unlink(missing_ok=True)
                    if (
                        minted_key is not None
                        and not session._intents
                        and not session._bundles
                        and not (proj_dir / "agent_chat_history.json").is_file()
                        and not _experiment_file_digests(proj_dir)
                    ):
                        (proj_dir / "owner_token_sha256.txt").unlink(missing_ok=True)
                        (proj_dir / "pending_delivery_claim.json").unlink(missing_ok=True)

                try:
                    if isinstance(req, RefineTurnRequest):
                        if prev_intent is None:
                            raise HTTPException(
                                status_code=400,
                                detail="Cannot run 'refine' before creating an initial DesignIntent in this project.",
                            )
                        intent = session.refine_turn_intent_canonical(
                            user_utterance=clean_turn_utterance,
                            restriction_enzymes=req.restriction_enzymes,
                            updated_restriction_enzymes=req.updated_restriction_enzymes,
                            locked_five_prime_utr=req.locked_five_prime_utr,
                            lock_cds_from_bundle_id=req.lock_cds_from_bundle_id,
                            unlock_scopes=req.unlock_scopes,
                            uridine_chemistry=req.uridine_chemistry,
                            compliance_track=req.compliance_track,
                            detarget_liver=req.detarget_liver,
                            use_commercial_surrogate=req.use_commercial_surrogate,
                            turn_id=assigned_turn_id,
                            expected_parent_intent_id=expected_turn_parent_id,
                            redact_extra_secrets=turn_secrets,
                        )
                    else:
                        intent = session.create_turn_intent_canonical(
                            user_utterance=clean_turn_utterance,
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
                            turn_id=assigned_turn_id,
                            redact_extra_secrets=turn_secrets,
                        )
                except HTTPException:
                    _cleanup_uncommitted_turn()
                    raise
                except ValueError as exc:
                    _cleanup_uncommitted_turn()
                    msg = redact_secrets_in_text(str(exc), extra_secrets=turn_secrets)
                    code = 409 if ("Concurrent revision" in msg or "missing from project store" in msg) else 422
                    raise HTTPException(status_code=code, detail=msg) from exc

            diff = redact_secrets_in_obj(_compute_intent_diff(prev_intent, intent), extra_secrets=turn_secrets)
            safe_intent_dump = redact_secrets_in_obj(intent.model_dump(mode="json"), extra_secrets=turn_secrets)
            if intent.status != "ready":
                out_clarify: dict[str, Any] = redact_secrets_in_obj(
                    {
                        "status": "NEEDS_CLARIFICATION",
                        "blocking_questions": intent.blocking_questions,
                        "intent": safe_intent_dump,
                        "intent_diff": diff,
                        "report": None,
                        "candidates": [],
                    },
                    extra_secrets=turn_secrets,
                )
                out_clarify["status"] = "NEEDS_CLARIFICATION"
                if turn_req_key and turn_req_digest:
                    turn_requests_map[turn_req_key] = {
                        "status": "completed",
                        "request_digest": turn_req_digest,
                        "response": dict(out_clarify),
                    }
                    _save_hashed_requests(
                        turn_journal_path,
                        turn_requests_map,
                        legacy_hex=turn_legacy_hex,
                        extra_secrets=turn_secrets,
                    )
                elif turn_migrated:
                    _save_hashed_requests(
                        turn_journal_path,
                        turn_requests_map,
                        legacy_hex=turn_legacy_hex,
                        extra_secrets=turn_secrets,
                    )
                if minted_key is not None:
                    out_clarify["project_key"] = minted_key
                return out_clarify

            try:
                existing_rep = session.latest_report(intent.intent_id)
            except ValueError as exc:
                raise HTTPException(
                    status_code=409,
                    detail=redact_secrets_in_text(str(exc), extra_secrets=turn_secrets),
                ) from exc
            try:
                if existing_rep is None:
                    bundles = session.execute_design(intent)
                    latest_rep = session.latest_report(intent.intent_id)
                else:
                    latest_rep = existing_rep
                    bundles = list(session._bundles.values())
            except ValueError as exc:
                raise HTTPException(
                    status_code=409,
                    detail=redact_secrets_in_text(str(exc), extra_secrets=turn_secrets),
                ) from exc
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
            out_ok: dict[str, Any] = redact_secrets_in_obj(
                {
                    "status": "COMPLETED",
                    "blocking_questions": [],
                    "intent": safe_intent_dump,
                    "intent_diff": diff,
                    "report": report_view,
                    "candidates": cards,
                    "cli_replay_command": replay_cmd,
                },
                extra_secrets=turn_secrets,
            )
            out_ok["status"] = "COMPLETED"
            if turn_req_key and turn_req_digest:
                turn_requests_map[turn_req_key] = {
                    "status": "completed",
                    "request_digest": turn_req_digest,
                    "response": dict(out_ok),
                }
                _save_hashed_requests(turn_journal_path, turn_requests_map, extra_secrets=turn_secrets)
            elif turn_migrated:
                _save_hashed_requests(turn_journal_path, turn_requests_map, extra_secrets=turn_secrets)
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
        x_mrnax_llm_provider: str | None = Header(default=None),
    ) -> Any:
        _enforce_rate_limit(request, max_per_min=20)
        if not _verify_privileged_access(x_mrnax_access_token, x_mrnax_llm_key):
            raise HTTPException(
                status_code=401,
                detail="Privileged LLM Agent mode requires either X-MRNAX-Access-Token or your own X-MRNAX-LLM-Key.",
            )

        header_provider: str | None = None
        if x_mrnax_llm_provider is not None:
            header_provider_raw = x_mrnax_llm_provider.strip()
            if not header_provider_raw:
                raise HTTPException(
                    status_code=422,
                    detail="Unsupported X-MRNAX-LLM-Provider header.",
                )
            try:
                header_provider = normalize_dialogue_provider(header_provider_raw)
            except ValueError as exc:
                raise HTTPException(
                    status_code=422,
                    detail="Unsupported X-MRNAX-LLM-Provider header.",
                ) from exc
        if header_provider and req.provider and header_provider != req.provider:
            raise HTTPException(
                status_code=400,
                detail="Provider mismatch between X-MRNAX-LLM-Provider header and request body provider.",
            )
        effective_provider = header_provider or req.provider or "deepseek"

        cookie_cap = request.cookies.get(f"mrnax_pk_{project_id}")
        pre_secrets = _collect_active_secrets(x_mrnax_llm_key, x_mrnax_access_token, x_mrnax_project_key, cookie_cap)
        if req.model and redact_secrets_in_text(req.model, extra_secrets=pre_secrets) != req.model:
            raise HTTPException(status_code=422, detail="Invalid model identifier.")

        req_id = req.client_request_id
        req_key = _hash_client_request_id(req_id) if req_id else None
        req_digest = hashlib.sha256(
            json.dumps(
                {"message": req.message, "provider": effective_provider, "model": req.model or ""},
                sort_keys=True,
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()
        byo_clean = (x_mrnax_llm_key or "").strip()
        claim_proof = (
            hashlib.sha256(
                f"{req_key or 'no_id'}:{req_digest}:{hashlib.sha256(byo_clean.encode('utf-8')).hexdigest()}".encode("utf-8")
            ).hexdigest()
            if byo_clean and not _is_valid_admin_token(x_mrnax_access_token)
            else None
        )

        safe_pid, minted_key = _authorize_or_bind_project(
            project_id,
            request,
            response,
            allow_bind_new=True,
            header_project_key=x_mrnax_project_key,
            header_access_token=x_mrnax_access_token,
            pending_claim_proof=claim_proof,
        )
        proj_dir = resolved_projects / safe_pid
        if _is_legacy_unbound_anonymous(proj_dir, x_mrnax_access_token):
            raise HTTPException(
                status_code=403,
                detail="Project access denied: pre-existing unbound project requires X-MRNAX-Access-Token for Agent chat.",
            )
        cookie_cap = request.cookies.get(f"mrnax_pk_{safe_pid}")
        active_secrets = _collect_active_secrets(
            x_mrnax_llm_key, x_mrnax_access_token, x_mrnax_project_key, cookie_cap, minted_key
        )
        https_req = _is_https_request(request)

        def _ensure_ready_intent_evaluated(
            session: ProjectHarnessSession, target_intent: DesignIntent | None
        ) -> None:
            if target_intent is None or target_intent.status != "ready":
                return
            existing_rep = session.latest_report(target_intent.intent_id)
            if existing_rep is None:
                session.execute_design(target_intent)

        def _assemble_agent_chat_response(
            session: ProjectHarnessSession,
            *,
            target_intent: DesignIntent | None,
            baseline_intent: DesignIntent | None,
            provider_name: str,
            model_name: str,
            reply_text: str,
            tool_calls_list: list[dict[str, Any]],
        ) -> dict[str, Any]:
            try:
                latest_rep = session.latest_report(target_intent.intent_id) if target_intent is not None else None
            except ValueError as exc:
                raise HTTPException(
                    status_code=409,
                    detail=redact_secrets_in_text(str(exc), extra_secrets=active_secrets),
                ) from exc
            report_view, cards = _serialize_report_with_candidates(
                latest_rep,
                session._bundles.values(),
                intent=target_intent,
                seed_dir=resolved_seed,
                disallow_sandbox_override=True,
            )
            diff = (
                _compute_intent_diff(baseline_intent, target_intent)
                if (
                    target_intent is not None
                    and baseline_intent is not None
                    and target_intent.intent_id != baseline_intent.intent_id
                )
                else None
            )
            tool_calls_out = redact_secrets_in_obj(
                [
                    {"name": tc.get("name", ""), "arguments": tc.get("arguments", {})}
                    for tc in tool_calls_list
                    if isinstance(tc, dict)
                ],
                extra_secrets=active_secrets,
            )
            needs_clarification = bool(target_intent is not None and target_intent.status != "ready")
            out_status = "NEEDS_CLARIFICATION" if needs_clarification else "COMPLETED"
            out_payload: dict[str, Any] = {
                "status": out_status,
                "blocking_questions": (
                    list(target_intent.blocking_questions)
                    if (needs_clarification and target_intent is not None)
                    else []
                ),
                "provider": redact_secrets_in_text(provider_name, extra_secrets=active_secrets),
                "model": redact_secrets_in_text(model_name, extra_secrets=active_secrets),
                "assistant_reply": redact_secrets_in_text(reply_text, extra_secrets=active_secrets),
                "tool_calls": tool_calls_out,
                "intent": target_intent.model_dump(mode="json") if target_intent else None,
                "intent_diff": diff,
                "report": report_view,
                "candidates": cards,
            }
            if target_intent is not None and target_intent.status == "ready":
                out_payload["cli_replay_command"] = (
                    f"uv run mrnax replay --intent-file intents/intent_v{target_intent.intent_version:03d}_{target_intent.intent_id}.json "
                    f"--out-dir artifacts/replay_{target_intent.intent_id}"
                )
            sanitized_out = redact_secrets_in_obj(out_payload, extra_secrets=active_secrets)
            sanitized_out["status"] = out_status
            if isinstance(sanitized_out.get("intent"), dict) and target_intent is not None:
                sanitized_out["intent"]["status"] = target_intent.status
            return sanitized_out

        with project_lock(proj_dir):
            try:
                requests_map, legacy_hex_map, req_migrated = _load_agent_chat_requests(proj_dir)
            except _JournalCorruptionError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            if req_key:
                prior_req_record, lookup_migrated = _lookup_request_record(
                    requests_map, legacy_hex_map, req_id, req_key
                )
                req_migrated = req_migrated or lookup_migrated
            else:
                prior_req_record = None
            if prior_req_record:
                prior_digest = prior_req_record.get("request_digest")
                if prior_digest and prior_digest != req_digest:
                    raise HTTPException(
                        status_code=409,
                        detail="Idempotency conflict: client_request_id reused with a different request payload.",
                    )
            if prior_req_record and prior_req_record.get("status") == "completed":
                if req_migrated:
                    _save_agent_chat_requests(
                        proj_dir,
                        requests_map,
                        legacy_hex=legacy_hex_map,
                        extra_secrets=active_secrets,
                    )
                cached_resp = prior_req_record.get("response")
                if isinstance(cached_resp, dict):
                    out_cached = redact_secrets_in_obj(
                        dict(cached_resp), extra_secrets=active_secrets, redact_keys=False
                    )
                    for pkey in ("status", "provider", "model"):
                        if pkey in cached_resp:
                            out_cached[pkey] = cached_resp[pkey]
                    if isinstance(cached_resp.get("intent"), dict) and isinstance(out_cached.get("intent"), dict):
                        for ipkey in ("intent_id", "intent_version", "status"):
                            if ipkey in cached_resp["intent"]:
                                out_cached["intent"][ipkey] = cached_resp["intent"][ipkey]
                    if minted_key is not None:
                        out_cached["project_key"] = minted_key
                    return out_cached

            session = _get_session(safe_pid)
            intents_by_id = {i.intent_id: i for i in session._intents}
            prior_history = _load_agent_chat_history(proj_dir, max_turns=30, extra_secrets=active_secrets)

            # Crash-window 2 recovery: history was saved right before request journal marked completed.
            if req_key:
                hist_match = next(
                    (h for h in reversed(prior_history) if h.get("client_request_key") == req_key),
                    None,
                )
                if hist_match is not None:
                    matched_intent_id = hist_match.get("intent_id")
                    matched_intent = intents_by_id.get(matched_intent_id) if isinstance(matched_intent_id, str) else None
                    if matched_intent is None and isinstance(hist_match.get("intent_version"), int):
                        matched_intent = next(
                            (i for i in session._intents if i.intent_version == hist_match["intent_version"]),
                            None,
                        )
                    base_id = prior_req_record.get("baseline_intent_id") if prior_req_record else None
                    base_intent = intents_by_id.get(base_id) if isinstance(base_id, str) else None
                    chat_out = _assemble_agent_chat_response(
                        session,
                        target_intent=matched_intent,
                        baseline_intent=base_intent,
                        provider_name=str(hist_match.get("provider") or effective_provider),
                        model_name=str(hist_match.get("model") or req.model or DEFAULT_PROVIDER_MODELS[effective_provider]),
                        reply_text=str(hist_match.get("assistant_reply") or ""),
                        tool_calls_list=[tc for tc in (hist_match.get("tool_calls") or []) if isinstance(tc, dict)],
                    )
                    requests_map[req_key] = {
                        "status": "completed",
                        "request_digest": req_digest,
                        "response": dict(chat_out),
                    }
                    _save_agent_chat_requests(
                        proj_dir,
                        requests_map,
                        legacy_hex=legacy_hex_map,
                        extra_secrets=active_secrets,
                    )
                    if minted_key is not None:
                        chat_out["project_key"] = minted_key
                    return chat_out

            turn_id_prefix = (
                str(prior_req_record.get("turn_id_prefix"))
                if (prior_req_record is not None and isinstance(prior_req_record.get("turn_id_prefix"), str))
                else (f"req_{req_key[:16]}" if req_key else f"anon_{secrets.token_hex(6)}")
            )
            owned_intents = (
                [
                    i
                    for i in session._intents
                    if i.created_from_turn_ids and i.created_from_turn_ids[-1].startswith(f"{turn_id_prefix}_")
                ]
                if prior_req_record is not None
                else []
            )
            owned_exp_ids = (
                {
                    exp_id
                    for exp_id, tids in session._experiment_origins.items()
                    if any(
                        isinstance(tid, str) and tid.startswith(f"{turn_id_prefix}_")
                        for tid in (tids if isinstance(tids, list) else [tids])
                    )
                }
                if prior_req_record is not None
                else set()
            )
            baseline_intent_id = (
                prior_req_record.get("baseline_intent_id")
                if prior_req_record is not None
                else (session._intents[-1].intent_id if session._intents else None)
            )
            baseline_intent_count = (
                int(prior_req_record["baseline_intent_count"])
                if (prior_req_record is not None and "baseline_intent_count" in prior_req_record)
                else len(session._intents)
            )
            diff_prev_intent = (
                intents_by_id.get(baseline_intent_id) if isinstance(baseline_intent_id, str) else None
            )

            recovered_target_intent: DesignIntent | None = owned_intents[-1] if owned_intents else None
            if prior_req_record is not None:
                rec_mut_id = prior_req_record.get("mutated_intent_id")
                if recovered_target_intent is None and isinstance(rec_mut_id, str):
                    cand_mut = intents_by_id.get(rec_mut_id)
                    if cand_mut is not None and (
                        not cand_mut.created_from_turn_ids
                        or cand_mut.created_from_turn_ids[-1].startswith(f"{turn_id_prefix}_")
                    ):
                        recovered_target_intent = cand_mut
            exp_changed = bool(owned_exp_ids)

            prior_mutating = bool(
                prior_req_record is not None
                and (
                    prior_req_record.get("mutating_tools_committed")
                    or recovered_target_intent is not None
                    or exp_changed
                )
            )

            expected_parent_id = (
                recovered_target_intent.intent_id
                if recovered_target_intent is not None
                else baseline_intent_id
            )
            toolbox = AtomicToolbox(
                session,
                turn_id_prefix=turn_id_prefix,
                expected_parent_intent_id=expected_parent_id,
            )
            pre_intent_ids = {i.intent_id for i in session._intents}
            pre_exp_ids = set(owned_exp_ids)
            pending_batch: list[dict[str, Any]] = (
                [dict(ptc) for ptc in (prior_req_record.get("pending_batch_tool_calls") or []) if isinstance(ptc, dict)]
                if prior_req_record is not None
                else []
            )
            initial_recovered_tc: list[dict[str, Any]] = (
                [dict(tc) for tc in (prior_req_record.get("tool_calls") or []) if isinstance(tc, dict)]
                if prior_req_record is not None
                else []
            )

            # Reconcile any interrupted owned intent or pending batch tool calls (mutating or read-only).
            if prior_mutating or pending_batch:
                if recovered_target_intent is not None and recovered_target_intent in owned_intents:
                    try:
                        _ensure_ready_intent_evaluated(session, recovered_target_intent)
                    except ValueError as exc:
                        raise HTTPException(
                            status_code=409,
                            detail=redact_secrets_in_text(str(exc), extra_secrets=active_secrets),
                        ) from exc
                if pending_batch:
                    head_ptc = pending_batch[0]
                    head_name = str(head_ptc.get("name") or "")
                    raw_head_args = str(head_ptc.get("raw_arguments") or "{}")
                    parsed_head_args: Any = {}
                    try:
                        decoded_head = json.loads(raw_head_args)
                        if isinstance(decoded_head, dict):
                            parsed_head_args = redact_secrets_in_obj(decoded_head, extra_secrets=active_secrets)
                    except (json.JSONDecodeError, ValueError, TypeError):
                        decoded_head = {}
                        parsed_head_args = {}

                    recorded_intent_muts = sum(
                        1
                        for tc in initial_recovered_tc
                        if tc.get("name") in ("design_constructs", "refine_construct")
                        and not (isinstance(tc.get("result"), dict) and "error" in tc["result"])
                    )
                    recorded_exp_ids = {
                        str(tc["result"].get("experiment_id"))
                        for tc in initial_recovered_tc
                        if tc.get("name") == "ingest_experiment"
                        and isinstance(tc.get("result"), dict)
                        and "error" not in tc["result"]
                        and isinstance(tc["result"].get("experiment_id"), str)
                    }
                    head_exp_id: str | None = None
                    if head_name == "ingest_experiment" and isinstance(decoded_head, dict):
                        rec_json = decoded_head.get("record_json")
                        if isinstance(rec_json, dict) and isinstance(rec_json.get("experiment_id"), str):
                            head_exp_id = rec_json["experiment_id"]

                    head_already_committed = (
                        head_name in ("design_constructs", "refine_construct")
                        and len(owned_intents) > recorded_intent_muts
                    ) or (
                        head_name == "ingest_experiment"
                        and head_exp_id is not None
                        and head_exp_id in owned_exp_ids
                        and head_exp_id not in recorded_exp_ids
                    )
                    if head_already_committed:
                        if head_name == "ingest_experiment":
                            matched_exp = next(
                                (e for e in session._experiments if e.experiment_id == head_exp_id), None
                            )
                            head_res: dict[str, Any] = (
                                matched_exp.model_dump(mode="json")
                                if matched_exp is not None
                                else {"experiment_id": head_exp_id, "status": "committed"}
                            )
                            exp_changed = True
                        elif recovered_target_intent is not None:
                            try:
                                completed_rep = session.latest_report(recovered_target_intent.intent_id)
                            except ValueError as exc:
                                raise HTTPException(
                                    status_code=409,
                                    detail=redact_secrets_in_text(str(exc), extra_secrets=active_secrets),
                                ) from exc
                            head_res = {
                                "intent_id": recovered_target_intent.intent_id,
                                "intent_version": recovered_target_intent.intent_version,
                                "status": recovered_target_intent.status,
                                "outcome_status": completed_rep.outcome_status if completed_rep is not None else None,
                                "recommended_bundle_ids": completed_rep.recommended_bundle_ids
                                if completed_rep is not None
                                else [],
                            }
                        else:
                            head_res = {"status": "committed"}
                        initial_recovered_tc.append(
                            {
                                "tool_call_id": head_ptc.get("tool_call_id"),
                                "name": redact_secrets_in_text(head_name, extra_secrets=active_secrets),
                                "arguments": parsed_head_args if isinstance(parsed_head_args, dict) else {},
                                "result": redact_secrets_in_obj(head_res, extra_secrets=active_secrets),
                            }
                        )
                        pending_batch.pop(0)
                        prior_mutating = True
                        if req_key and prior_req_record is not None:
                            requests_map[req_key] = {
                                "status": "partial_mutating_committed",
                                "request_digest": req_digest,
                                "turn_id_prefix": turn_id_prefix,
                                "baseline_intent_id": baseline_intent_id,
                                "baseline_intent_count": baseline_intent_count,
                                "baseline_llm_history": list(prior_req_record.get("baseline_llm_history") or []),
                                "mutating_tools_committed": True,
                                "mutated_intent_id": recovered_target_intent.intent_id
                                if recovered_target_intent
                                else None,
                                "expected_parent_intent_id": toolbox.expected_parent_intent_id,
                                "pending_batch_tool_calls": list(pending_batch),
                                "tool_calls": list(initial_recovered_tc),
                            }
                            _save_agent_chat_requests(
                                proj_dir,
                                requests_map,
                                legacy_hex=legacy_hex_map,
                                extra_secrets=active_secrets,
                            )
                while pending_batch:
                    ptc = pending_batch[0]
                    fn_name = str(ptc.get("name") or "")
                    raw_args = str(ptc.get("raw_arguments") or "{}")
                    parsed_args: Any = {}
                    try:
                        decoded = json.loads(raw_args)
                        if not isinstance(decoded, dict):
                            raise ValueError("Tool arguments must decode to a JSON object")
                        parsed_args = redact_secrets_in_obj(decoded, extra_secrets=active_secrets)
                        res = toolbox.dispatch_tool_call(fn_name, parsed_args)
                    except (json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
                        err_msg = redact_secrets_in_text(str(exc), extra_secrets=active_secrets)
                        if "Concurrent revision conflict" in err_msg:
                            raise HTTPException(status_code=409, detail=err_msg) from exc
                        res = {"error": redact_secrets_in_text(f"{type(exc).__name__}: {exc}", extra_secrets=active_secrets)}
                    safe_res = redact_secrets_in_obj(res, extra_secrets=active_secrets)
                    initial_recovered_tc.append(
                        {
                            "tool_call_id": ptc.get("tool_call_id"),
                            "name": redact_secrets_in_text(fn_name, extra_secrets=active_secrets),
                            "arguments": redact_secrets_in_obj(parsed_args, extra_secrets=active_secrets)
                            if isinstance(parsed_args, dict)
                            else {},
                            "result": safe_res,
                        }
                    )
                    pending_batch.pop(0)
                    new_after_ptc = [
                        i
                        for i in session._intents
                        if i.intent_id not in pre_intent_ids
                        and i.created_from_turn_ids
                        and i.created_from_turn_ids[-1].startswith(f"{turn_id_prefix}_")
                    ]
                    if new_after_ptc:
                        recovered_target_intent = new_after_ptc[-1]
                        pre_intent_ids.update(i.intent_id for i in new_after_ptc)
                        prior_mutating = True
                        try:
                            _ensure_ready_intent_evaluated(session, recovered_target_intent)
                        except ValueError as exc:
                            raise HTTPException(
                                status_code=409,
                                detail=redact_secrets_in_text(str(exc), extra_secrets=active_secrets),
                            ) from exc
                    if fn_name == "ingest_experiment" and not (isinstance(safe_res, dict) and "error" in safe_res):
                        exp_changed = True
                        prior_mutating = True
                    if req_key and prior_req_record is not None:
                        requests_map[req_key] = {
                            "status": "partial_mutating_committed" if prior_mutating else "partial_readonly_only",
                            "request_digest": req_digest,
                            "turn_id_prefix": turn_id_prefix,
                            "baseline_intent_id": baseline_intent_id,
                            "baseline_intent_count": baseline_intent_count,
                            "baseline_llm_history": list(prior_req_record.get("baseline_llm_history") or []),
                            "mutating_tools_committed": prior_mutating,
                            "mutated_intent_id": recovered_target_intent.intent_id if recovered_target_intent else None,
                            "expected_parent_intent_id": toolbox.expected_parent_intent_id,
                            "pending_batch_tool_calls": list(pending_batch),
                            "tool_calls": list(initial_recovered_tc),
                        }
                        _save_agent_chat_requests(
                            proj_dir,
                            requests_map,
                            legacy_hex=legacy_hex_map,
                            extra_secrets=active_secrets,
                        )

            llm_history: list[dict[str, Any]] = []
            if prior_req_record is not None and isinstance(prior_req_record.get("baseline_llm_history"), list):
                for msg in prior_req_record["baseline_llm_history"]:
                    if isinstance(msg, dict) and msg.get("role") in ("user", "assistant") and isinstance(msg.get("content"), str):
                        llm_history.append({"role": msg["role"], "content": msg["content"]})
            else:
                baseline_ver = diff_prev_intent.intent_version if diff_prev_intent is not None else 0
                for item in prior_history[-10:]:
                    if prior_req_record is not None:
                        item_ver = item.get("intent_version")
                        if isinstance(item_ver, int) and item_ver > baseline_ver:
                            continue
                    u_msg = item.get("user_message")
                    a_msg = item.get("assistant_reply")
                    if isinstance(u_msg, str) and u_msg.strip():
                        llm_history.append({"role": "user", "content": u_msg})
                    if isinstance(a_msg, str) and a_msg.strip():
                        llm_history.append({"role": "assistant", "content": a_msg})

            context_intent = recovered_target_intent if prior_mutating else diff_prev_intent
            project_context = _build_agent_project_context(session, target_intent=context_intent)
            exec_tracker: dict[str, Any] = {
                "mutated": prior_mutating,
                "mutated_intent_id": recovered_target_intent.intent_id if recovered_target_intent else None,
                "pending_batch": list(pending_batch),
            }

            readonly_prior: list[dict[str, Any]] = (
                list(initial_recovered_tc)
                if (prior_req_record is not None and not prior_mutating)
                else []
            )
            recovered_tool_calls: list[dict[str, Any]] | None = None
            allow_tools = True
            message_to_send = req.message
            if prior_mutating and prior_req_record is not None:
                recovered_tool_calls = list(initial_recovered_tc)
                if not recovered_tool_calls and (recovered_target_intent is not None or exp_changed):
                    try:
                        rep_for_summary = (
                            session.latest_report(recovered_target_intent.intent_id)
                            if recovered_target_intent is not None
                            else None
                        )
                    except ValueError as exc:
                        raise HTTPException(
                            status_code=409,
                            detail=redact_secrets_in_text(str(exc), extra_secrets=active_secrets),
                        ) from exc
                    recovered_tool_calls = [
                        {
                            "name": "recovered_committed_state",
                            "arguments": {},
                            "result": {
                                "intent_id": recovered_target_intent.intent_id if recovered_target_intent else None,
                                "intent_version": recovered_target_intent.intent_version if recovered_target_intent else None,
                                "status": recovered_target_intent.status if recovered_target_intent else "committed",
                                "outcome_status": rep_for_summary.outcome_status if rep_for_summary else None,
                                "experiments_updated": exp_changed,
                            },
                        }
                    ]
                allow_tools = False
                message_to_send = (
                    f"{req.message}\n\n"
                    "[RECOVERED_TOOL_RESULTS_FROM_PRIOR_ATTEMPT]\n"
                    f"{_format_tool_calls_for_recovery_prompt(recovered_tool_calls, max_total_chars=4000, extra_secrets=active_secrets)}\n"
                    "[/RECOVERED_TOOL_RESULTS_FROM_PRIOR_ATTEMPT]"
                )
            elif readonly_prior:
                message_to_send = (
                    f"{req.message}\n\n"
                    "[PRIOR_READONLY_LOOKUPS_COMPLETED]\n"
                    f"{_format_tool_calls_for_recovery_prompt(readonly_prior, max_total_chars=3000, extra_secrets=active_secrets)}\n"
                    "[/PRIOR_READONLY_LOOKUPS_COMPLETED]"
                )

            def _record_progress(agent_obj: LiveDialogueAgent, *, force_save_readonly: bool = False) -> bool:
                owned_now = [
                    i
                    for i in session._intents
                    if i.intent_id not in pre_intent_ids
                    and i.created_from_turn_ids
                    and i.created_from_turn_ids[-1].startswith(f"{turn_id_prefix}_")
                ]
                if owned_now:
                    exec_tracker["mutated_intent_id"] = owned_now[-1].intent_id
                exp_now = {
                    exp_id
                    for exp_id, tids in session._experiment_origins.items()
                    if any(
                        isinstance(tid, str) and tid.startswith(f"{turn_id_prefix}_")
                        for tid in (tids if isinstance(tids, list) else [tids])
                    )
                }
                exp_delta = bool(exp_now - pre_exp_ids)
                trace_mut = any(
                    tc.get("name") in MUTATING_AGENT_TOOLS
                    and not (isinstance(tc.get("result"), dict) and "error" in tc["result"])
                    for tc in agent_obj.tool_call_trace
                )
                has_mutated = bool(
                    prior_mutating or exec_tracker["mutated"] or owned_now or exp_delta or trace_mut
                )
                exec_tracker["mutated"] = has_mutated
                if req_key and (has_mutated or force_save_readonly or agent_obj.tool_call_trace):
                    base_tc = recovered_tool_calls if recovered_tool_calls is not None else readonly_prior
                    combined_tc = [
                        {
                            "tool_call_id": tc.get("tool_call_id"),
                            "name": tc.get("name", ""),
                            "arguments": tc.get("arguments", {}),
                            "result": tc.get("result", {}),
                        }
                        for tc in (*base_tc, *agent_obj.tool_call_trace)
                        if isinstance(tc, dict)
                    ]
                    requests_map[req_key] = {
                        "status": "partial_mutating_committed" if has_mutated else "partial_readonly_only",
                        "request_digest": req_digest,
                        "turn_id_prefix": turn_id_prefix,
                        "baseline_intent_id": baseline_intent_id,
                        "baseline_intent_count": baseline_intent_count,
                        "baseline_llm_history": list(llm_history),
                        "mutating_tools_committed": has_mutated,
                        "mutated_intent_id": exec_tracker["mutated_intent_id"],
                        "expected_parent_intent_id": toolbox.expected_parent_intent_id,
                        "pending_batch_tool_calls": list(exec_tracker["pending_batch"]),
                        "tool_calls": combined_tc,
                    }
                    _save_agent_chat_requests(
                        proj_dir,
                        requests_map,
                        legacy_hex=legacy_hex_map,
                        extra_secrets=active_secrets,
                    )
                return has_mutated

            if req_key and prior_req_record is None:
                requests_map[req_key] = {
                    "status": "in_progress",
                    "request_digest": req_digest,
                    "turn_id_prefix": turn_id_prefix,
                    "baseline_intent_id": baseline_intent_id,
                    "baseline_intent_count": baseline_intent_count,
                    "baseline_llm_history": list(llm_history),
                    "mutating_tools_committed": False,
                    "mutated_intent_id": None,
                    "expected_parent_intent_id": toolbox.expected_parent_intent_id,
                    "pending_batch_tool_calls": [],
                    "tool_calls": [],
                }
                _save_agent_chat_requests(
                    proj_dir,
                    requests_map,
                    legacy_hex=legacy_hex_map,
                    extra_secrets=active_secrets,
                )

            agent_holder: dict[str, LiveDialogueAgent] = {}

            def _on_tool_batch_started(batch: list[dict[str, Any]]) -> None:
                exec_tracker["pending_batch"] = [dict(x) for x in batch]
                ag = agent_holder.get("agent")
                if ag is not None and req_key:
                    _record_progress(ag, force_save_readonly=True)

            def _on_tool_executed(entry: dict[str, Any]) -> None:
                done_id = str(entry.get("tool_call_id") or "")
                exec_tracker["pending_batch"] = [
                    x for x in exec_tracker["pending_batch"] if x.get("tool_call_id") != done_id
                ]
                ag = agent_holder.get("agent")
                if ag is not None:
                    _record_progress(ag, force_save_readonly=True)

            agent = LiveDialogueAgent(
                toolbox,
                provider=effective_provider,
                model=req.model,
                history=llm_history,
                project_context=project_context,
                extra_secrets=active_secrets,
                on_tool_executed=_on_tool_executed,
                on_tool_batch_started=_on_tool_batch_started,
            )
            agent_holder["agent"] = agent

            def _handle_agent_failure(err_detail: str) -> Any:
                did_mutate = _record_progress(
                    agent,
                    force_save_readonly=bool(agent.tool_call_trace or readonly_prior or recovered_tool_calls),
                )
                if did_mutate:
                    if minted_key is not None:
                        err_resp = JSONResponse(
                            status_code=502,
                            content={"detail": err_detail, "project_key": minted_key},
                            headers={"X-MRNAX-Project-Key": minted_key},
                        )
                        err_resp.set_cookie(
                            key=f"mrnax_pk_{safe_pid}",
                            value=minted_key,
                            httponly=True,
                            secure=https_req,
                            samesite="lax",
                        )
                        return err_resp
                    raise HTTPException(status_code=502, detail=err_detail)

                # Zero mutations committed: roll back newly minted project ownership so first-call BYO retries cleanly.
                # Retain baseline journal entries on retries so concurrent mutations cannot shift this request's baseline.
                can_rollback_owner = (
                    minted_key is not None
                    and not session._intents
                    and not (proj_dir / "agent_chat_history.json").is_file()
                    and not _experiment_file_digests(proj_dir)
                )
                if can_rollback_owner:
                    (proj_dir / "owner_token_sha256.txt").unlink(missing_ok=True)
                    (proj_dir / "pending_delivery_claim.json").unlink(missing_ok=True)
                    if (not _is_valid_admin_token(x_mrnax_access_token)) and req_key and req_key in requests_map:
                        requests_map.pop(req_key, None)
                        if requests_map or legacy_hex_map:
                            _save_agent_chat_requests(
                                proj_dir,
                                requests_map,
                                legacy_hex=legacy_hex_map,
                                extra_secrets=active_secrets,
                            )
                        else:
                            (proj_dir / "agent_chat_requests.json").unlink(missing_ok=True)
                raise HTTPException(status_code=502, detail=err_detail)

            try:
                reply = agent.send_user_message(
                    message_to_send,
                    explicit_api_key=x_mrnax_llm_key.strip() if x_mrnax_llm_key else None,
                    allow_tools=allow_tools,
                )
            except UpstreamProviderProtocolError:
                return _handle_agent_failure("Upstream LLM provider returned an invalid response envelope.")
            except httpx.HTTPStatusError as exc:
                return _handle_agent_failure(_safe_upstream_status_detail(exc, extra_secrets=active_secrets))
            except httpx.RequestError as exc:
                return _handle_agent_failure(f"Upstream LLM network error ({type(exc).__name__}).")
            except RuntimeError as exc:
                return _handle_agent_failure(redact_secrets_in_text(str(exc), extra_secrets=active_secrets)[:240])

            final_new_intents = [
                i
                for i in session._intents
                if i.intent_id not in pre_intent_ids
                and i.created_from_turn_ids
                and i.created_from_turn_ids[-1].startswith(f"{turn_id_prefix}_")
            ]
            if prior_mutating:
                curr_intent = (
                    recovered_target_intent
                    or (final_new_intents[-1] if final_new_intents else None)
                    or diff_prev_intent
                )
            elif final_new_intents:
                curr_intent = final_new_intents[-1]
            else:
                curr_intent = context_intent

            raw_tool_calls = (
                recovered_tool_calls
                if recovered_tool_calls is not None
                else ([*readonly_prior, *agent.tool_call_trace] if readonly_prior else agent.tool_call_trace)
            )
            chat_out = _assemble_agent_chat_response(
                session,
                target_intent=curr_intent,
                baseline_intent=diff_prev_intent,
                provider_name=agent.provider,
                model_name=agent.model,
                reply_text=reply,
                tool_calls_list=raw_tool_calls,
            )

            full_history = _load_agent_chat_history(proj_dir, max_turns=30, extra_secrets=active_secrets)
            if not req_key or not any(h.get("client_request_key") == req_key for h in full_history):
                full_history.append(
                    {
                        "turn_ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "client_request_key": req_key,
                        "provider": chat_out["provider"],
                        "model": chat_out["model"],
                        "user_message": redact_secrets_in_text(req.message, extra_secrets=active_secrets),
                        "assistant_reply": chat_out["assistant_reply"],
                        "tool_calls": chat_out["tool_calls"],
                        "intent_id": curr_intent.intent_id if curr_intent else None,
                        "intent_version": curr_intent.intent_version if curr_intent else None,
                    }
                )
                _save_agent_chat_history(
                    proj_dir, full_history, max_turns=30, extra_secrets=active_secrets
                )

            if req_key:
                requests_map[req_key] = {
                    "status": "completed",
                    "request_digest": req_digest,
                    "response": dict(chat_out),
                }
                _save_agent_chat_requests(
                    proj_dir,
                    requests_map,
                    legacy_hex=legacy_hex_map,
                    extra_secrets=active_secrets,
                )
            elif req_migrated:
                _save_agent_chat_requests(
                    proj_dir,
                    requests_map,
                    legacy_hex=legacy_hex_map,
                    extra_secrets=active_secrets,
                )
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
                intent = session.intent_for_export(bundle)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=f"Unknown bundle '{safe_bid}'") from exc
            except ValueError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
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

    def _load_authorized_bundle_for_export(
        project_id: str,
        bundle_id: str,
        request: Request,
        header_project_key: str | None,
        header_access_token: str | None,
    ) -> tuple[ProjectHarnessSession, ConstructBundle, DesignIntent, BundleEvaluation | None]:
        safe_pid, _ = _authorize_or_bind_project(
            project_id,
            request,
            allow_bind_new=False,
            header_project_key=header_project_key,
            header_access_token=header_access_token,
        )
        with project_lock(resolved_projects / safe_pid):
            session = _get_session(safe_pid)
            safe_bid = _validate_safe_id(bundle_id, "bundle_id")
            try:
                bundle = session.get_bundle(safe_bid)
                intent = session.intent_for_export(bundle)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=f"Unknown bundle '{safe_bid}'") from exc
            except ValueError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc

            constraint_engine = ScopedConstraintEngine.from_seed_dir(resolved_seed)
            try:
                if intent.compliance_track == "commercial_clean":
                    ExternalEvidenceAdapterV2().filter_and_export_commercial_records(bundle.external_evidence)
                outcome = constraint_engine.evaluate_bundle(
                    bundle,
                    intent.constraints,
                    compliance_track=intent.compliance_track,
                    expected_chemistry=intent.uridine_chemistry,
                    locked_scopes=intent.locked_scopes,
                    expected_protein_targets=intent.protein_targets,
                )
                if not outcome.passed_all_hard_gates:
                    reasons = "; ".join(
                        f"{v.violated_requirement_id}@{v.violated_scope}:{v.reason}" for v in outcome.violations
                    )
                    raise ExportConstraintViolationError(
                        f"Export blocked by shared ScopedConstraintEngine: {reasons}"
                    )
            except (CleanRoomPolicyError, ExportConstraintViolationError) as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc

            ev_match: BundleEvaluation | None = None
            try:
                rep = session.latest_report(intent.intent_id)
                if rep is not None:
                    ev_match = next((e for e in rep.candidate_evaluations if e.bundle_id == bundle.bundle_id), None)
            except ValueError:
                ev_match = None
            return session, bundle, intent, ev_match

    @app.get("/v1/projects/{project_id}/bundles/{bundle_id}/dossier")
    def get_project_bundle_dossier(
        project_id: str,
        bundle_id: str,
        request: Request,
        format: Literal["json", "html"] = "json",
        x_mrnax_project_key: str | None = Header(default=None),
        x_mrnax_access_token: str | None = Header(default=None),
    ) -> Response:
        _enforce_rate_limit(request, max_per_min=45)
        _, bundle, intent, ev_match = _load_authorized_bundle_for_export(
            project_id, bundle_id, request, x_mrnax_project_key, x_mrnax_access_token
        )
        constraint_engine = ScopedConstraintEngine.from_seed_dir(resolved_seed)
        html_doc = render_qc_dossier_html(
            bundle,
            intent=intent,
            constraint_engine=constraint_engine,
            evaluation=ev_match,
            seed_dir=resolved_seed,
        )
        secrets_tuple = _collect_active_secrets(x_mrnax_access_token, x_mrnax_project_key)
        safe_html = redact_secrets_in_text(html_doc, extra_secrets=secrets_tuple)
        if format == "html":
            return HTMLResponse(content=safe_html, status_code=200)
        return JSONResponse(
            content={
                "bundle_id": bundle.bundle_id,
                "version_digest": bundle.version_digest,
                "compliance_track": intent.compliance_track,
                "content_type": "text/html",
                "qc_dossier_html": safe_html,
            }
        )

    @app.get("/v1/projects/{project_id}/bundles/{bundle_id}/ivt-order")
    def get_project_bundle_ivt_order(
        project_id: str,
        bundle_id: str,
        request: Request,
        format: Literal["json", "csv"] = "json",
        x_mrnax_project_key: str | None = Header(default=None),
        x_mrnax_access_token: str | None = Header(default=None),
    ) -> Response:
        _enforce_rate_limit(request, max_per_min=45)
        _, bundle, intent, _ = _load_authorized_bundle_for_export(
            project_id, bundle_id, request, x_mrnax_project_key, x_mrnax_access_token
        )
        csv_text, rows = render_ivt_synthesis_order(bundle, intent=intent)
        secrets_tuple = _collect_active_secrets(x_mrnax_access_token, x_mrnax_project_key)
        safe_csv = redact_secrets_in_text(csv_text, extra_secrets=secrets_tuple)
        if format == "csv":
            return Response(content=safe_csv, media_type="text/csv; charset=utf-8", status_code=200)
        return JSONResponse(
            content={
                "bundle_id": bundle.bundle_id,
                "version_digest": bundle.version_digest,
                "content_type": "text/csv",
                "ivt_synthesis_order_csv": safe_csv,
                "rows": redact_secrets_in_obj(rows, extra_secrets=secrets_tuple),
            }
        )

    # --- Wave 2B Scientific Knowledge, Structure, Evidence & Clinical Benchmark Endpoints ---
    @app.get("/api/v1/rules")
    def api_get_rules(
        request: Request,
        modality: str | None = None,
        region: str | None = None,
        grade: str | None = None,
        q: str | None = None,
        include_debunked: bool = True,
    ) -> dict[str, Any]:
        _enforce_rate_limit(request, max_per_min=90)
        secrets_tuple = _collect_active_secrets()
        res = query_scientific_rules(
            seed_dir=resolved_seed,
            modality=modality,
            region=region,
            grade=grade,
            query=q,
            include_debunked=include_debunked,
        )
        return redact_secrets_in_obj(res, extra_secrets=secrets_tuple)

    @app.get("/api/v1/data-hub")
    def api_get_data_hub(request: Request) -> dict[str, Any]:
        _enforce_rate_limit(request, max_per_min=60)
        secrets_tuple = _collect_active_secrets()
        catalog = load_registry_json(resolved_seed, "data_hub_catalog_v2.json")
        if catalog is None:
            v9_inv = load_registry_json(resolved_seed, "v9_wetlab_benchmarks_inventory.json") or {}
            catalog = {
                "catalog_id": "v9_fallback_inventory",
                "datasets": v9_inv.get("benchmarks", []),
                "external_apis": [],
                "engines_and_models": [],
            }
        clin_doc = load_registry_json(resolved_seed, "clinical_mrna_benchmarks_v1.json") or {"benchmarks": []}
        mirna_doc = load_registry_json(resolved_seed, "tissue_mirna_atlas_v1.json")
        if mirna_doc is None:
            mirna_doc = load_registry_json(resolved_seed, "mirna_rbp_commercial_v2.json") or {}
        scaffolds_doc = load_registry_json(resolved_seed, "modality_scaffolds_v1.json") or {}

        scorecard: dict[str, Any] | None = None
        for sc_candidate in (
            resolved_seed / "surrogate_v2" / "surrogate_v2_ood_scorecard.json",
            resolved_seed / "surrogate_v2_ood_scorecard.json",
            Path("/Users/alex/.gemini/antigravity/brain/4db4d24c-55e0-4968-869d-a93a1c9c7b7b/scratch/wave1_sota_models/surrogate_v2_ood_scorecard.json"),
            Path("/Users/alex/.gemini/antigravity/brain/4db4d24c-55e0-4968-869d-a93a1c9c7b7b/scratch/wave1_sota_models/surrogate_v2/surrogate_v2_ood_scorecard.json"),
        ):
            if sc_candidate.is_file():
                try:
                    scorecard = json.loads(sc_candidate.read_text(encoding="utf-8"))
                    break
                except ValueError:
                    pass

        compact_benchmarks = [
            {
                "benchmark_id": b.get("benchmark_id"),
                "construct_name": b.get("construct_name"),
                "who_inn_name": b.get("who_inn_name"),
                "developer": b.get("developer"),
                "modality": b.get("modality"),
                "chemistry": b.get("chemistry"),
                "clinical_or_reference_status": b.get("clinical_or_reference_status"),
                "license_track": b.get("license_track"),
                "architecture_summary": b.get("architecture_summary"),
                "quantitative_sequence_stats": b.get("quantitative_sequence_stats"),
                "fto_and_governance_notice": b.get("fto_and_governance_notice"),
            }
            for b in (clin_doc.get("benchmarks") or [])
            if isinstance(b, dict)
        ]

        payload = {
            "catalog_id": catalog.get("catalog_id", "mRNAx_Data_Hub_Catalog_v2"),
            "summary_counts": catalog.get("summary_counts", {}),
            "datasets": catalog.get("datasets", []),
            "external_apis": catalog.get("external_apis", []),
            "engines_and_models": catalog.get("engines_and_models", []),
            "clinical_benchmarks": compact_benchmarks,
            "tissue_mirna_atlas": {
                "atlas_id": mirna_doc.get("atlas_id", "tissue_mirna_atlas_v1"),
                "summary_counts": mirna_doc.get("summary_counts", {}),
                "tissue_groups": mirna_doc.get("tissue_groups", {}),
                "mirna_entries": mirna_doc.get("mirna_entries", []),
            },
            "modality_scaffolds_summary": {
                "registry_id": scaffolds_doc.get("registry_id"),
                "summary_counts": scaffolds_doc.get("summary_counts", {}),
                "modality_architectures": scaffolds_doc.get("modality_architectures", {}),
            },
            "surrogate_scorecard": scorecard,
        }
        return redact_secrets_in_obj(payload, extra_secrets=secrets_tuple)

    @app.post("/api/v1/structure/analyze")
    def api_analyze_structure(
        req: StructureAnalyzeRequest,
        request: Request,
    ) -> dict[str, Any]:
        _enforce_rate_limit(request, max_per_min=30)
        secrets_tuple = _collect_active_secrets()
        try:
            res = run_structure_and_motif_analysis(
                req.sequence,
                five_prime_end=req.five_prime_end,
                cds_start=req.cds_start,
                cds_end=req.cds_end,
                three_prime_start=req.three_prime_start,
                three_prime_end=req.three_prime_end,
                modality=req.modality,
                include_eternafold=req.include_eternafold,
                seed_dir=resolved_seed,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail=redact_secrets_in_text(str(exc), extra_secrets=secrets_tuple),
            ) from exc
        return redact_secrets_in_obj(res, extra_secrets=secrets_tuple)

    @app.post("/api/v1/evidence/resolve")
    def api_resolve_evidence(
        req: EvidenceResolveRequest,
        request: Request,
    ) -> dict[str, Any]:
        _enforce_rate_limit(request, max_per_min=30)
        secrets_tuple = _collect_active_secrets()
        adapter = ExternalEvidenceAdapterV2(license_track="commercial_clean")
        out: dict[str, Any] = {"query": req.query, "mode": req.mode}
        try:
            if req.mode in ("protein", "both"):
                out["protein"] = adapter.resolve_target_protein(
                    req.query, allow_network=req.allow_network
                )
            if req.mode in ("literature", "both"):
                out["literature_and_trials"] = adapter.search_literature_and_trials(
                    req.query, max_results=req.max_results, allow_network=req.allow_network
                )
        except CleanRoomPolicyError as exc:
            raise HTTPException(
                status_code=422,
                detail=redact_secrets_in_text(str(exc), extra_secrets=secrets_tuple),
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail=redact_secrets_in_text(str(exc), extra_secrets=secrets_tuple),
            ) from exc
        return redact_secrets_in_obj(out, extra_secrets=secrets_tuple)

    @app.post("/api/v1/benchmarks/clinical")
    def api_compare_clinical_benchmarks(
        req: ClinicalBenchmarkCompareRequest,
        request: Request,
        x_mrnax_project_key: str | None = Header(default=None),
        x_mrnax_access_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        _enforce_rate_limit(request, max_per_min=45)
        secrets_tuple = _collect_active_secrets(x_mrnax_access_token, x_mrnax_project_key)
        target_bundle: ConstructBundle | None = None
        if req.bundle_id is not None or req.project_id is not None:
            if not req.project_id or not req.bundle_id:
                raise HTTPException(
                    status_code=422,
                    detail="Both project_id and bundle_id are required when comparing a stored project bundle.",
                )
            safe_pid, _ = _authorize_or_bind_project(
                req.project_id,
                request,
                allow_bind_new=False,
                header_project_key=x_mrnax_project_key,
                header_access_token=x_mrnax_access_token,
            )
            with project_lock(resolved_projects / safe_pid):
                session = _get_session(safe_pid)
                safe_bid = _validate_safe_id(req.bundle_id, "bundle_id")
                try:
                    target_bundle = session.get_bundle(safe_bid)
                except KeyError as exc:
                    raise HTTPException(status_code=404, detail=f"Unknown bundle '{safe_bid}'") from exc
                except ValueError as exc:
                    raise HTTPException(status_code=409, detail=str(exc)) from exc
        elif not req.sequence:
            raise HTTPException(
                status_code=422,
                detail="Provide either sequence or (project_id, bundle_id).",
            )

        try:
            res = compare_construct_against_clinical_benchmarks(
                seed_dir=resolved_seed,
                bundle=target_bundle,
                sequence=req.sequence,
                modality=req.modality,
                chemistry=req.chemistry,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail=redact_secrets_in_text(str(exc), extra_secrets=secrets_tuple),
            ) from exc
        return redact_secrets_in_obj(res, extra_secrets=secrets_tuple)

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

"""Deterministic Project State Machine: Intent Lineage, Lock Inheritance, Conflict Gate & Wet-Lab Loop."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import tempfile
import threading
import uuid
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from decimal import Decimal, InvalidOperation
from functools import wraps
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr, ValidationError, field_validator, model_validator

from mrnax.constraints.frameshift_m1psi import remediate_m1psi_slippery_runs
from mrnax.constraints.fto_guard import COMMERCIAL_ZERO_GPL_REQUIREMENT
from mrnax.constraints.registry import ScopedConstraintEngine, default_m1_constraints
from mrnax.engine.pareto_director import next_report_timestamp_ns, project_report_for_intent, run_design_pipeline
from mrnax.engine.profiles import check_profile_readiness
from mrnax.harness.sandbox import (
    UntrustedExternalText,
    redact_secret_markers,
    redact_secrets_in_obj,
    redact_secrets_in_text,
)
from mrnax.schemas.common import ComplianceProfile, Modality, SegmentScope, UridineChemistry
from mrnax.schemas.construct_bundle import (
    ChemistryTrackSpec,
    ConstructBundle,
    ProteinTargetSpec,
    RNAMoleculeSpec,
    SegmentSpan,
)
from mrnax.schemas.design_intent import (
    BundleSegmentLock,
    CellExpressionSpec,
    ConstraintSpec,
    DesignIntent,
    LockedRegionSpec,
    ObjectiveSpec,
    RuntimePolicySpec,
)
from mrnax.schemas.evaluation_report import EvaluationReport, serialize_report_artifact
from mrnax.schemas.experiment_record import ExperimentRecord, FeedbackDecisionRecord
from mrnax.schemas.run_manifest import ArtifactChecksumRecord, RunManifest


def parse_json_float_token_bounded(tok: str) -> Decimal:
    """Parse a JSON float token into an exact Decimal without raising InvalidOperation on extreme exponents."""
    if len(tok) > 4096:
        return Decimal("-Infinity") if tok.startswith("-") else Decimal("Infinity")
    try:
        return Decimal(tok)
    except InvalidOperation:
        return Decimal("-Infinity") if tok.startswith("-") else Decimal("Infinity")


def parse_bounded_json_int(v: Any, *, field_name: str, min_val: int, max_val: int) -> int:
    """Validate JSON Schema `type: integer` within `[min_val, max_val]` BEFORE any integer conversion."""
    if isinstance(v, bool):
        raise ValueError(f"{field_name} must be an integer, not boolean")
    if isinstance(v, int):
        if v < min_val or v > max_val:
            raise ValueError(f"{field_name} must be between {min_val} and {max_val}")
        return v
    if isinstance(v, Decimal):
        if not v.is_finite():
            raise ValueError(f"{field_name} must be a finite integer")
        min_dec = Decimal(min_val)
        max_dec = Decimal(max_val)
        if v < min_dec or v > max_dec:
            raise ValueError(f"{field_name} must be between {min_val} and {max_val}")
        if v != v.to_integral_value():
            raise ValueError(f"{field_name} must have zero fractional part")
        return int(v)
    if isinstance(v, float):
        if not math.isfinite(v):
            raise ValueError(f"{field_name} must be a finite integer")
        if v < min_val or v > max_val:
            raise ValueError(f"{field_name} must be between {min_val} and {max_val}")
        d = Decimal(str(v))
        if d < Decimal(min_val) or d > Decimal(max_val) or d != d.to_integral_value():
            raise ValueError(f"{field_name} must have zero fractional part")
        return int(d)
    raise ValueError(f"{field_name} must be a JSON integer")


def _parse_json_schema_seed(v: Any) -> int:
    """Validate JSON Schema `type: integer` in `[0, 2147483647]` before any integer expansion."""
    return parse_bounded_json_int(v, field_name="random_seed", min_val=0, max_val=2_147_483_647)


_STANDARD_AA_OPT_STOP_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+[*]?$")


def _normalize_additional_protein_sequences(
    seqs: list[str] | None,
    *,
    require_primary: str | None = None,
    check_primary: bool = False,
) -> list[str] | None:
    """Validate and normalize 1..3 additional protein chains (canonical AA, 10..5000 residues)."""
    if seqs is None:
        return None
    if check_primary and not (require_primary or "").strip():
        raise ValueError("additional_protein_sequences requires primary protein_sequence to be supplied.")
    if not isinstance(seqs, list) or len(seqs) < 1 or len(seqs) > 3:
        raise ValueError(
            "additional_protein_sequences must be a non-empty list of 1 to 3 sequences (max 4 total chains)."
        )
    normalized: list[str] = []
    for idx, raw in enumerate(seqs, start=2):
        if not isinstance(raw, str):
            raise ValueError(f"Additional chain #{idx} must be a string.")
        cleaned = raw.strip().upper()
        if not cleaned or not _STANDARD_AA_OPT_STOP_RE.fullmatch(cleaned):
            raise ValueError(
                f"Additional chain #{idx} must contain only canonical 20 standard amino acids (optional trailing '*')."
            )
        if cleaned.endswith("*"):
            cleaned = cleaned[:-1]
        if len(cleaned) < 10 or len(cleaned) > 5000:
            raise ValueError(
                f"Additional chain #{idx} length ({len(cleaned)} AA) must be within [10, 5000]."
            )
        normalized.append(cleaned)
    return normalized


class InitialTurnInput(BaseModel):
    """Strict initial-turn wire contract shared 1:1 between HTTP `mode='initial'` and MCP `design_constructs`."""

    model_config = ConfigDict(extra="forbid")

    user_utterance: StrictStr = Field(min_length=1, max_length=4000)
    protein_sequence: StrictStr | None = None
    additional_protein_sequences: list[StrictStr] | None = None
    accession_or_source: StrictStr = "GenBank:U55762.1"
    entry_version: StrictStr | None = None
    compliance_track: ComplianceProfile | None = None
    uridine_chemistry: UridineChemistry | None = None
    restriction_enzymes: list[StrictStr] | None = None
    locked_five_prime_utr: StrictStr | None = None
    detarget_liver: StrictBool | None = None
    prioritize_late_signal: StrictBool = False
    conflicting_single_pick: StrictBool = False
    reference_bundle_id: StrictStr | None = None
    use_preset_reference_r: StrictBool = False
    use_commercial_surrogate: StrictBool | None = None
    modality: Modality | None = None
    scaffold_id: StrictStr | None = Field(default=None, pattern=r"^[a-z0-9][a-z0-9_-]{1,63}$")
    random_seed: int = Field(default=42, ge=0, le=2_147_483_647)

    @field_validator("random_seed", mode="before")
    @classmethod
    def _validate_seed_before(cls, v: Any) -> int:
        return _parse_json_schema_seed(v)

    @field_validator("additional_protein_sequences")
    @classmethod
    def _validate_additional_chains(cls, v: list[str] | None) -> list[str] | None:
        return _normalize_additional_protein_sequences(v, check_primary=False)

    @model_validator(mode="after")
    def _require_primary_for_additional_chains(self) -> InitialTurnInput:
        if self.additional_protein_sequences is not None and not (self.protein_sequence or "").strip():
            raise ValueError("additional_protein_sequences requires primary protein_sequence to be supplied.")
        return self


class RefineTurnInput(BaseModel):
    """Strict refine-turn wire contract shared 1:1 between HTTP `mode='refine'` and MCP `refine_construct`."""

    model_config = ConfigDict(extra="forbid")

    user_utterance: StrictStr = Field(min_length=1, max_length=4000)
    restriction_enzymes: list[StrictStr] | None = None
    updated_restriction_enzymes: list[StrictStr] | None = None
    locked_five_prime_utr: StrictStr | None = None
    lock_cds_from_bundle_id: StrictStr | None = None
    unlock_scopes: list[SegmentScope] | None = None
    uridine_chemistry: UridineChemistry | None = None
    compliance_track: ComplianceProfile | None = None
    detarget_liver: StrictBool | None = None
    use_commercial_surrogate: StrictBool | None = None
    modality: Modality | None = None
    scaffold_id: StrictStr | None = Field(default=None, pattern=r"^[a-z0-9][a-z0-9_-]{1,63}$")


def _detect_chemistry(utterance: str, default: UridineChemistry = "m1psi") -> UridineChemistry:
    low = utterance.lower()
    if any(k in low for k in ("m1psi", "m¹ψ", "n1-methylpseudouridine", "1-甲基假尿苷", "n1-甲基假尿苷")):
        return "m1psi"
    if any(k in low for k in ("unmodified", "未修饰", "canonical u")):
        return "unmodified_u"
    if any(k in low for k in ("pseudouridine", "假尿苷", "psi")):
        return "pseudouridine"
    return default


def _detect_enzymes(utterance: str) -> list[str] | None:
    found: list[str] = []
    for enz in ("BspQI", "SapI", "BsaI", "BsmBI", "BbsI", "EcoRI", "XbaI"):
        if re.search(rf"\b{enz}\b", utterance, re.IGNORECASE):
            found.append(enz)
    return found if found else None


_held_locks = threading.local()


@contextmanager
def project_lock(store_dir: Path) -> Iterator[None]:
    path = store_dir.resolve() / ".project.lock"
    key = (os.getpid(), path)
    held = getattr(_held_locks, "paths", set())
    if key in held:
        yield
        return
    with path.open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _held_locks.paths = held | {key}
        try:
            yield
        finally:
            _held_locks.paths = held


def _with_current_state[**P, R](method: Callable[P, R]) -> Callable[P, R]:
    @wraps(method)
    def serialized(self, *args, **kwargs):
        with self.current_state():
            return method(self, *args, **kwargs)
    return serialized


def _report_bundle_ids(report: EvaluationReport) -> set[str]:
    ids = set(report.recommended_bundle_ids)
    ids.update(ev.bundle_id for ev in report.candidate_evaluations)
    ids.update(bid for trace in report.action_transparency for bid in trace.changed_candidate_ids)
    ids.update(diagnostic.candidate_ref for diagnostic in report.failed_diagnostics)
    return ids


def _manifest_verifies_report(
    manifest: RunManifest, report: EvaluationReport, raw_report: bytes, intent: DesignIntent, intent_file_sha: str,
) -> bool:
    if manifest.compliance_track != report.compliance_track or manifest.compliance_track != intent.compliance_track:
        return False
    reports = [a for a in manifest.artifacts if a.logical_name == "evaluation_report"]
    if len(reports) != 1:
        return False
    pin = reports[0]
    report_file_matches = pin.sha256 == hashlib.sha256(raw_report).hexdigest()
    intent_pins = [a.sha256 for a in manifest.artifacts if a.logical_name == "input_intent"]
    if intent_pins:
        return report_file_matches and intent_pins == [intent_file_sha] and manifest.input_intent_digest == intent.digest()
    if manifest.input_intent_digest == intent_file_sha and report_file_matches:
        return True  # Legacy receipt pinned both files directly.
    if (report.rule_snapshot_version is None or pin.local_relpath != "evaluation_report.json"
            or manifest.input_intent_digest != intent.digest()):
        return False
    # A direct pipeline run pins the exported artifact. A project can store the same JSON in compact form.
    artifact = serialize_report_artifact(json.loads(raw_report))
    return (pin.sha256 == hashlib.sha256(artifact).hexdigest() and pin.bytes == len(artifact)
            and pin.md5 == hashlib.md5(artifact).hexdigest())


def _write_json(path: Path, value: dict, *, exclusive: bool = False) -> None:
    # Publish only fully written files; link(2) gives atomic create-if-absent.
    with tempfile.NamedTemporaryFile(dir=path.parent, mode="w", encoding="utf-8", delete=False) as tmp:
        staged = Path(tmp.name)
        try:
            json.dump(value, tmp, indent=2, sort_keys=True)
            tmp.write("\n")
            tmp.close()
            if exclusive:
                os.link(staged, path)
            else:
                os.replace(staged, path)
        finally:
            staged.unlink(missing_ok=True)


_SEGMENT_SCOPES: tuple[SegmentScope, ...] = ("five_prime_utr", "cds", "stop_cassette", "three_prime_utr", "poly_a")


def paired_chains(left: ConstructBundle, right: ConstructBundle) -> list[tuple[RNAMoleculeSpec, RNAMoleculeSpec]]:
    for bundle in (left, right):
        ids = {m.molecule_id for m in bundle.molecules}
        if len(ids) != len(bundle.molecules) or ids != bundle.target_stoichiometry.keys():
            raise ValueError("Bundle chain IDs and stoichiometry disagree")
    if left.co_expression_mode != right.co_expression_mode or len(left.molecules) != len(right.molecules):
        raise ValueError("Incompatible chain count or co-expression mode")
    by_id = {m.molecule_id: m for m in right.molecules}
    pairs = []
    for a in left.molecules:
        b = by_id.get(a.molecule_id)
        if b is None:
            matches = [m for m in right.molecules if m.effective_translation_target == a.effective_translation_target]
            if len(matches) != 1:
                raise ValueError("Ambiguous chain identity across bundles")
            b = matches[0]
        if (a.effective_translation_target.protein_sequence != b.effective_translation_target.protein_sequence
                or a.chemistry_track != b.chemistry_track
                or left.target_stoichiometry[a.molecule_id] != right.target_stoichiometry[b.molecule_id]):
            raise ValueError("Incompatible target, chemistry, or chain stoichiometry")
        pairs.append((a, b))
    if len({b.molecule_id for _, b in pairs}) != len(pairs):
        raise ValueError("No one-to-one chain identity across bundles")
    return pairs


def build_preset_reference_r(seed_dir: Path) -> ConstructBundle:
    """Construct the canonical commercial_clean eGFP + HBA1 reference ConstructBundle R."""
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


class ProjectHarnessSession:
    """Stateful project service persisting turns, DesignIntent versions, ConstructBundles, and feedback."""

    def __init__(
        self,
        *,
        project_id: str | None = None,
        store_dir: Path,
        seed_dir: Path,
        default_project_id: str = "proj_default",
    ) -> None:
        self.project_id = (project_id or "").strip() or default_project_id
        self.store_dir = store_dir
        self.seed_dir = seed_dir
        self.store_dir.mkdir(parents=True, exist_ok=True)
        (self.store_dir / "intents").mkdir(exist_ok=True)
        (self.store_dir / "bundles").mkdir(exist_ok=True)
        (self.store_dir / "reports").mkdir(exist_ok=True)
        (self.store_dir / "experiments").mkdir(exist_ok=True)

        self._intents: list[DesignIntent] = []
        self._bundles: dict[str, ConstructBundle] = {}
        self.quarantined_bundles: dict[str, str] = {}
        self._reports: dict[str, EvaluationReport] = {}
        self._manifests: dict[str, RunManifest] = {}
        self._historical_release_blocks: defaultdict[str, set[str]] = defaultdict(set)
        self._experiments: list[ExperimentRecord] = []
        self._experiment_origins: dict[str, list[str]] = {}
        self._bundle_intent_map: dict[str, str] = {}
        self._ambiguous_report_intents: set[str] = set()
        self._external_notes: list[UntrustedExternalText] = []
        self.active_reference_bundle_id: str | None = None
        self._state_revision = 0
        with project_lock(self.store_dir):
            self._load_existing()

    MAX_PROJECT_TURNS: int = 120

    @contextmanager
    def current_state(self) -> Iterator[None]:
        """Hold the same revision from loading project history through the caller's last use."""
        with project_lock(self.store_dir):
            meta = self.store_dir / "project_state.json"
            if meta.is_file() and json.loads(meta.read_bytes()).get("state_revision", 0) != self._state_revision:
                self._load_existing()
            yield

    def _read_stored_report(self, raw_report: bytes) -> EvaluationReport:
        try:
            return EvaluationReport.model_validate_json(raw_report)
        except ValidationError:
            # Retain the historical association so a malformed evaluation cannot become an unevaluated import.
            data = json.loads(raw_report)
            if not isinstance(data, dict):
                raise
            candidates = data.get("candidate_evaluations")
            recommendations = data.get("recommended_bundle_ids")
            if not isinstance(candidates, list) or not isinstance(recommendations, list):
                raise
            affected = {e["bundle_id"] for e in candidates if isinstance(e, dict) and isinstance(e.get("bundle_id"), str)}
            affected.update(bid for bid in recommendations if isinstance(bid, str))
            if not affected:
                raise
            rejected = EvaluationReport.model_validate(dict(
                data, candidate_evaluations=[], recommended_bundle_ids=[], action_transparency=[],
                outcome_status="UNCOMPARABLE_MISSING_APPROVED_COMPONENT",
                uncomparable_reason="Stored report contains invalid evaluations or recommendation references.",
            ))
            for bid in affected:
                self._historical_release_blocks[bid].add(rejected.uncomparable_reason)
            return rejected

    def _load_existing(self) -> None:
        self._intents.clear()
        self._bundles.clear()
        self.quarantined_bundles.clear()
        self._reports.clear()
        self._manifests.clear()
        self._historical_release_blocks.clear()
        self._experiments.clear()
        self._experiment_origins.clear()
        self._bundle_intent_map.clear()
        self._ambiguous_report_intents.clear()
        intent_files = sorted((self.store_dir / "intents").glob("intent_v*.json"))
        intent_files_by_revision: dict[tuple[str, int], tuple[DesignIntent, str]] = {}
        for p in intent_files:
            raw_intent = p.read_bytes()
            intent = DesignIntent.model_validate_json(raw_intent)
            self._intents.append(intent)
            intent_files_by_revision[(intent.intent_id, intent.intent_version)] = (intent, hashlib.sha256(raw_intent).hexdigest())
        bundle_files = (self.store_dir / "bundles").glob("*.json")
        for p in bundle_files:
            try:
                b = ConstructBundle.model_validate_json(p.read_bytes())
            except ValidationError as exc:
                errors = exc.errors(include_context=False, include_input=False, include_url=False)
                self.quarantined_bundles[p.stem] = "; ".join(
                    f"{'.'.join(map(str, err['loc'])) or 'bundle'}: {err['msg']}" for err in errors
                )
                continue
            self._bundles[b.bundle_id] = b
        for bundle_id in self.quarantined_bundles:
            self._bundles.pop(bundle_id, None)

        # Load any persisted RunManifest files in store_dir / manifests / reports for historical provenance hydration
        manifest_candidates: list[Path] = []
        for mp in (
            *self.store_dir.glob("*manifest*.json"),
            *((self.store_dir / "manifests").glob("*.json") if (self.store_dir / "manifests").is_dir() else ()),
            *((self.store_dir / "reports").glob("*manifest*.json") if (self.store_dir / "reports").is_dir() else ()),
        ):
            if mp.is_file():
                manifest_candidates.append(mp)
        ambiguous_executions: set[str] = set()
        for mp in manifest_candidates:
            raw_manifest = mp.read_bytes()
            try:
                man = RunManifest.model_validate_json(raw_manifest)
            except ValidationError:
                try:
                    data = json.loads(raw_manifest)
                except ValueError:
                    data = None
                execution_id = data.get("execution_id") if isinstance(data, dict) else None
                ambiguous_executions.add(mp.stem)
                if isinstance(execution_id, str):
                    ambiguous_executions.add(execution_id)
                continue
            if mp.parent == self.store_dir / "manifests" and mp.stem != man.execution_id:
                # A file occupying another execution's manifest slot cannot make its report receipt-free.
                ambiguous_executions.add(mp.stem)
            previous = self._manifests.get(man.execution_id)
            if previous is not None and previous != man:
                ambiguous_executions.add(man.execution_id)
            self._manifests[man.execution_id] = man

        state_meta = self.store_dir / "project_state.json"
        meta: dict[str, Any] = {}
        has_saved_bundle_map = False
        report_order: list[str] = []
        has_saved_exp_order = False
        exp_order: list[str] = []
        if state_meta.is_file():
            meta = json.loads(state_meta.read_text(encoding="utf-8"))
            self.active_reference_bundle_id = meta.get("active_reference_bundle_id")
            if isinstance(meta.get("report_order"), list):
                report_order = [str(x) for x in meta["report_order"]]
            if isinstance(meta.get("ambiguous_report_intents"), list):
                self._ambiguous_report_intents.update(str(x) for x in meta["ambiguous_report_intents"])
            if isinstance(meta.get("experiment_order"), list):
                has_saved_exp_order = True
                exp_order = [str(x) for x in meta["experiment_order"]]
            saved_map = meta.get("bundle_intent_map")
            if isinstance(saved_map, dict):
                has_saved_bundle_map = True
                self._bundle_intent_map.update({str(k): str(v) for k, v in saved_map.items()})
            saved_exp_origins = meta.get("experiment_origins")
            if isinstance(saved_exp_origins, dict):
                for ek, ev in saved_exp_origins.items():
                    if isinstance(ev, list):
                        self._experiment_origins[str(ek)] = [str(x) for x in ev if isinstance(x, str) and x]
                    elif isinstance(ev, str) and ev:
                        self._experiment_origins[str(ek)] = [ev]
        stored_pid = meta.get("project_id") or (self._intents[0].project_id if self._intents else None)
        if isinstance(stored_pid, str) and stored_pid.strip():
            self.project_id = stored_pid.strip()

        order_rank = {rid: idx for idx, rid in enumerate(report_order)}
        loaded_reps: list[tuple[tuple[int, int, int, int, int, str], EvaluationReport]] = []
        legacy_mtime_by_intent: dict[str, list[int]] = {}
        for p in sorted((self.store_dir / "reports").glob("rep_*.json")):
            raw_report = p.read_bytes()
            rep = self._read_stored_report(raw_report)
            man = self._manifests.get(rep.execution_id)
            source = intent_files_by_revision.get((rep.intent_id, rep.intent_version))
            manifest_block = None
            if rep.execution_id in ambiguous_executions:
                manifest_block = "Producing execution has invalid or conflicting manifests."
            elif man is not None and source is None:
                manifest_block = "Producing manifest lacks its matching persisted intent revision."
            elif man is not None and source is not None:
                intent, intent_sha = source
                if not _manifest_verifies_report(man, rep, raw_report, intent, intent_sha):
                    manifest_block = "Producing manifest does not verify the stored report, intent, and compliance track."
                else:
                    manifest_block = man.commercial_release_block()
                    rep = rep.model_copy_validated(
                        rule_snapshot_version=man.rule_snapshot_version if rep.rule_snapshot_version is None else rep.rule_snapshot_version,
                        isolated_sandbox_override=rep.isolated_sandbox_override or man.isolated_sandbox_override,
                    )
            self._record_history_blocks(rep, manifest_block)
            ord_idx = order_rank.get(rep.report_id, 10**9)
            mtime_ns = p.stat().st_mtime_ns
            has_ns = 1 if rep.created_at_ns is not None else 0
            created_ns = int(rep.created_at_ns) if rep.created_at_ns is not None else 0
            if ord_idx == 10**9 and rep.created_at_ns is None:
                legacy_mtime_by_intent.setdefault(rep.intent_id, []).append(mtime_ns)
            loaded_reps.append(((ord_idx, rep.intent_version, has_ns, created_ns, mtime_ns, rep.report_id), rep))
        for iid, mtimes in legacy_mtime_by_intent.items():
            if len(mtimes) > 1 and len(set(mtimes)) < len(mtimes):
                self._ambiguous_report_intents.add(iid)
        loaded_reps.sort(key=lambda item: item[0])
        for _, rep in loaded_reps:
            self._reports[rep.report_id] = rep

        # Compute earliest witnessed intent per bundle across chronological intent history (heals legacy maps too)
        version_by_iid = {i.intent_id: i.intent_version for i in self._intents}
        reps_by_intent: dict[str, list[EvaluationReport]] = {}
        for rep in self._reports.values():
            reps_by_intent.setdefault(rep.intent_id, []).append(rep)
        earliest_witness: dict[str, str] = {}
        for intent in self._intents:
            if intent.reference_bundle_id and intent.reference_bundle_id in self._bundles:
                earliest_witness.setdefault(intent.reference_bundle_id, intent.intent_id)
            for rep in reps_by_intent.get(intent.intent_id, []):
                for ce in rep.candidate_evaluations:
                    earliest_witness.setdefault(ce.bundle_id, intent.intent_id)

        if not has_saved_bundle_map:
            self._bundle_intent_map.update(earliest_witness)
            if (
                self.active_reference_bundle_id
                and self.active_reference_bundle_id in self._bundles
                and len(self._intents) == 1
            ):
                self._bundle_intent_map.setdefault(self.active_reference_bundle_id, self._intents[0].intent_id)
        else:
            for bid, earliest_iid in earliest_witness.items():
                cur_iid = self._bundle_intent_map.get(bid)
                if (
                    cur_iid is not None
                    and cur_iid in version_by_iid
                    and version_by_iid[cur_iid] > version_by_iid[earliest_iid]
                ):
                    self._bundle_intent_map[bid] = earliest_iid

        self._reload_experiments_from_disk(meta=meta, exp_order=exp_order, has_saved_exp_order=has_saved_exp_order)
        self._state_revision = meta.get("state_revision", 0)

    def _repair_experiment_in_memory(self, exp: ExperimentRecord) -> tuple[ExperimentRecord, bool]:
        """Stamp physical construct digests in memory and downgrade legacy over-counted approvals."""
        stamped_obs = [
            o.model_copy_validated(physical_construct_digest=self._bundles[o.bundle_id].physical_construct_digest)
            if (o.physical_construct_digest is None and o.bundle_id in self._bundles)
            else o
            for o in exp.observations
        ]
        obs_changed = stamped_obs != exp.observations
        known_digests = [o.physical_construct_digest for o in stamped_obs]
        phys_n = len(set(known_digests)) if all(d is not None for d in known_digests) else None
        dec_cur = exp.feedback_decision
        needs_downgrade = bool(
            dec_cur is not None
            and phys_n is not None
            and phys_n < 6
            and (dec_cur.weight_update_allowed or dec_cur.isolated_calibration_proposal_id is not None)
        )
        if needs_downgrade and dec_cur is not None:
            new_rationale = re.sub(
                r"weight_update_allowed=True(?:\s*\(N=\d+\))?",
                f"weight_update_allowed=False (N={phys_n})",
                dec_cur.rationale,
            )
            if f"N={phys_n}" not in new_rationale:
                new_rationale = f"{new_rationale} [downgraded: physical N={phys_n} < 6]"
            dec_cur = dec_cur.model_copy_validated(
                weight_update_allowed=False,
                held_out_validation_passed=False,
                isolated_calibration_proposal_id=None,
                rationale=new_rationale,
            )
        if obs_changed or needs_downgrade:
            exp = exp.model_copy_validated(observations=stamped_obs, feedback_decision=dec_cur)
        return exp, needs_downgrade

    def _reload_experiments_from_disk(
        self,
        *,
        meta: dict[str, Any] | None = None,
        exp_order: list[str] | None = None,
        has_saved_exp_order: bool | None = None,
    ) -> None:
        initial_load = meta is not None
        prev_ids = {e.experiment_id for e in self._experiments}
        disk_advanced = False
        if meta is None:
            state_meta = self.store_dir / "project_state.json"
            if state_meta.is_file():
                raw_meta = json.loads(state_meta.read_text(encoding="utf-8"))
                meta = raw_meta if isinstance(raw_meta, dict) else {}
                disk_rev = meta.get("state_revision")
                if isinstance(disk_rev, int) and disk_rev > self._state_revision:
                    disk_advanced = True
                    self._state_revision = disk_rev
                    disk_ref = meta.get("active_reference_bundle_id")
                    if isinstance(disk_ref, str) and disk_ref in self._bundles:
                        self.active_reference_bundle_id = disk_ref
                    disk_origins = meta.get("experiment_origins")
                    if isinstance(disk_origins, dict):
                        parsed_origins: dict[str, list[str]] = {}
                        for ek, ev in disk_origins.items():
                            if isinstance(ev, list):
                                parsed_origins[str(ek)] = [str(x) for x in ev if isinstance(x, str) and x]
                            elif isinstance(ev, str) and ev:
                                parsed_origins[str(ek)] = [ev]
                        self._experiment_origins = parsed_origins
            else:
                meta = {}
        if has_saved_exp_order is None:
            has_saved_exp_order = isinstance(meta.get("experiment_order"), list)
        if exp_order is None:
            exp_order = meta.get("experiment_order") if has_saved_exp_order else []

        exp_rank = {eid: idx for idx, eid in enumerate(exp_order)}
        last_app = meta.get("last_applied_experiment_id")
        loaded_exps: list[tuple[tuple[int, int, int, int, str], ExperimentRecord]] = []
        for p in sorted((self.store_dir / "experiments").glob("*.json")):
            exp = ExperimentRecord.model_validate_json(p.read_text(encoding="utf-8"))
            exp, _ = self._repair_experiment_in_memory(exp)
            e_ord = exp_rank.get(exp.experiment_id, 10**9)
            if exp.ingested_at_ns is not None:
                sort_key = (1, int(exp.ingested_at_ns), e_ord, 0, exp.experiment_id)
            else:
                is_last = 1 if (isinstance(last_app, str) and exp.experiment_id == last_app) else 0
                sort_key = (0, e_ord, p.stat().st_mtime_ns, is_last, exp.experiment_id)
            loaded_exps.append((sort_key, exp))
        loaded_exps.sort(key=lambda item: item[0])
        self._experiments = [exp for _, exp in loaded_exps]

        saved_exp_set = set(exp_order)
        if self._experiments:
            latest_exp = self._experiments[-1]
            newly_discovered_latest = initial_load or disk_advanced or (latest_exp.experiment_id not in prev_ids)
            latest_is_unapplied = newly_discovered_latest and (
                (has_saved_exp_order and latest_exp.experiment_id not in saved_exp_set)
                or (
                    not has_saved_exp_order
                    and "last_applied_experiment_id" in meta
                    and latest_exp.experiment_id != last_app
                )
                or (
                    not has_saved_exp_order
                    and "last_applied_experiment_id" not in meta
                    and len(self._experiments) == 1
                )
            )
            if (
                latest_is_unapplied
                and latest_exp.feedback_decision is not None
                and latest_exp.feedback_decision.new_reference_bundle_id
                and latest_exp.feedback_decision.new_reference_bundle_id in self._bundles
            ):
                self.active_reference_bundle_id = latest_exp.feedback_decision.new_reference_bundle_id

    def _save_state_meta(self) -> None:
        revision = self._state_revision + 1
        meta_payload: dict[str, Any] = {
            "project_id": self.project_id,
            "active_reference_bundle_id": self.active_reference_bundle_id,
            "intent_count": len(self._intents),
            "bundle_intent_map": self._bundle_intent_map,
            "report_order": [
                rid for rid, r in self._reports.items() if r.intent_id not in self._ambiguous_report_intents
            ],
            "experiment_order": [e.experiment_id for e in self._experiments],
            "last_applied_experiment_id": self._experiments[-1].experiment_id if self._experiments else None,
            "state_revision": revision,
        }
        if self._experiment_origins:
            meta_payload["experiment_origins"] = {k: list(v) for k, v in self._experiment_origins.items()}
        if self._ambiguous_report_intents:
            meta_payload["ambiguous_report_intents"] = sorted(self._ambiguous_report_intents)
        if self.quarantined_bundles:
            meta_payload["quarantined_bundles"] = self.quarantined_bundles
        _write_json(self.store_dir / "project_state.json", meta_payload)
        self._state_revision = revision

    def _persist_intent(
        self,
        intent: DesignIntent,
        *,
        publish_reference_bundle_id: str | None = None,
    ) -> DesignIntent:
        previous = self._intents[-1] if self._intents else None
        if intent.intent_version != len(self._intents) + 1 or intent.parent_intent_id != (previous.intent_id if previous else None):
            raise ValueError("Stale intent revision or parent")
        if len(self._intents) >= self.MAX_PROJECT_TURNS:
            raise ValueError(
                f"Project '{self.project_id}' reached maximum turn history ({self.MAX_PROJECT_TURNS}); start a new session."
            )
        out = self.store_dir / "intents" / f"intent_v{intent.intent_version:03d}_{intent.intent_id}.json"
        _write_json(out, intent.model_dump(mode="json"), exclusive=True)
        self._intents.append(intent)
        if publish_reference_bundle_id is not None:
            self.active_reference_bundle_id = publish_reference_bundle_id
            self._bundle_intent_map.setdefault(publish_reference_bundle_id, intent.intent_id)
        self._save_state_meta()
        return intent

    @_with_current_state
    def register_bundle(
        self,
        bundle: ConstructBundle,
        *,
        set_as_reference: bool = False,
        originating_intent_id: str | None = None,
    ) -> None:
        if Path(bundle.bundle_id).name != bundle.bundle_id:
            raise ValueError("bundle_id must be a filename component")
        if originating_intent_id and not any(i.intent_id == originating_intent_id for i in self._intents):
            raise ValueError("Unknown originating intent")
        out = self.store_dir / "bundles" / f"{bundle.bundle_id}.json"
        try:
            _write_json(out, bundle.model_dump(mode="json"), exclusive=True)
        except FileExistsError:
            stored = ConstructBundle.model_validate_json(out.read_bytes())
            if stored != bundle:
                raise FileExistsError(f"Bundle '{bundle.bundle_id}' already exists with different content") from None
        self._bundles[bundle.bundle_id] = bundle
        if originating_intent_id is not None:
            self._bundle_intent_map.setdefault(bundle.bundle_id, originating_intent_id)
        if set_as_reference:
            self.active_reference_bundle_id = bundle.bundle_id
        self._save_state_meta()

    @_with_current_state
    def get_bundle(self, bundle_id: str) -> ConstructBundle:
        if bundle_id in self.quarantined_bundles:
            raise ValueError(f"Bundle '{bundle_id}' is quarantined: {self.quarantined_bundles[bundle_id]}")
        return self._bundles[bundle_id]

    @_with_current_state
    def latest_intent(self) -> DesignIntent:
        if not self._intents:
            raise RuntimeError("No DesignIntent exists in this session yet")
        return self._intents[-1]

    @_with_current_state
    def intent_for_bundle(self, bundle_id: str) -> DesignIntent:
        origin = self._bundle_intent_map.get(bundle_id)
        for intent in self._intents:
            if intent.intent_id == origin:
                return intent
        raise ValueError(f"Bundle '{bundle_id}' has no known originating intent")

    def _record_history_blocks(self, report: EvaluationReport, manifest_block: str | None) -> None:
        """Keep release vetoes even when a later report repeats the same immutable identity."""
        evaluated_ids = {ev.bundle_id for ev in report.candidate_evaluations}
        for bid in set(report.recommended_bundle_ids) - evaluated_ids:
            self._historical_release_blocks[bid].add("Recommendation lacks an evaluation.")
        for ev in report.candidate_evaluations:
            bundle = self._bundles.get(ev.bundle_id)
            if report.isolated_sandbox_override:
                self._historical_release_blocks[ev.bundle_id].add(
                    "Candidate originated from an isolated_sandbox_override execution."
                )
            if bundle is not None and ev.version_digest != bundle.version_digest:
                self._historical_release_blocks[ev.bundle_id].add("Stored evaluation digest differs from the bundle digest.")
        for trace in report.action_transparency:
            if trace.requirement_id == COMMERCIAL_ZERO_GPL_REQUIREMENT and trace.stage_applied == "filtering":
                for bid in trace.changed_candidate_ids:
                    self._historical_release_blocks[bid].add("Producing run recorded a copyleft dependency.")
        if manifest_block:
            for bid in _report_bundle_ids(report):
                self._historical_release_blocks[bid].add(manifest_block)

    def _release_block(self, bundle_id: str, track: ComplianceProfile) -> str | None:
        if bundle_id in self.quarantined_bundles:
            return self.quarantined_bundles[bundle_id]
        if track == "commercial_clean" and (blocks := self._historical_release_blocks.get(bundle_id)):
            return "; ".join(sorted(blocks))
        return None

    @_with_current_state
    def report_for_read(self, report: EvaluationReport) -> EvaluationReport:
        denied = {}
        claimed_ids = set(report.recommended_bundle_ids)
        claimed_ids.update(ev.bundle_id for ev in report.candidate_evaluations)
        for bid in claimed_ids:
            reason = self._release_block(bid, report.compliance_track)
            if reason:
                denied[bid] = reason
        if not denied:
            return report
        intent = next((i for i in self._intents if i.intent_id == report.intent_id), None)
        available = {bid: b for bid, b in self._bundles.items() if bid not in denied}
        view = project_report_for_intent(report, available, intent, seed_dir=self.seed_dir)
        if not view.candidate_evaluations:
            view = view.model_copy_validated(uncomparable_reason=denied[min(denied)])
        return view

    @_with_current_state
    def intent_for_export(self, bundle: ConstructBundle) -> DesignIntent:
        """Verify the bundle identity and producing reports before commercial export."""
        origin = self.intent_for_bundle(bundle.bundle_id)
        intent = origin
        if origin.compliance_track != "commercial_clean":
            active = self.latest_intent()
            if active.compliance_track != "commercial_clean" or not any(
                report.intent_id == active.intent_id and bundle.bundle_id in _report_bundle_ids(report)
                for report in self._reports.values()
            ):
                return origin
            intent = active

        history_block = self._release_block(bundle.bundle_id, intent.compliance_track)
        if history_block:
            raise ValueError(f"Commercial export blocked: {history_block}")
        eligible = False
        reasons = set()
        for report in self._reports.values():
            if report.intent_id != intent.intent_id:
                continue
            if not any(e.bundle_id == bundle.bundle_id for e in report.candidate_evaluations):
                continue
            projected = project_report_for_intent(
                report, self._bundles, intent, seed_dir=self.seed_dir, disallow_sandbox_override=True
            )
            if any(e.bundle_id == bundle.bundle_id for e in projected.candidate_evaluations):
                eligible = True
            else:
                reasons.add(projected.uncomparable_reason or projected.outcome_status)
        if not eligible and not reasons:
            if intent is not origin:
                raise ValueError("Commercial export blocked: research-origin bundle has no verified commercial evaluation.")
            if bundle.derivation_action != "reference_import":
                raise ValueError("Commercial export blocked: generated bundle has no verified evaluation.")
            readiness = check_profile_readiness(intent)
            if not readiness.is_runnable:
                raise ValueError(f"Commercial export blocked: {readiness.uncomparable_reason}")
        if reasons and not eligible:
            raise ValueError(
                f"Commercial export blocked: stored evaluation failed current commercial_clean eligibility ({'; '.join(sorted(reasons))})."
            )
        return intent

    @_with_current_state
    def latest_report(self, intent_id: str | None = None) -> EvaluationReport | None:
        target = intent_id or (self._intents[-1].intent_id if self._intents else None)
        if target in self._ambiguous_report_intents:
            raise ValueError(
                f"Ambiguous legacy report order for intent '{target}' (multiple un-ordered reports share identical filesystem timestamps)"
            )
        report = next((r for r in reversed(self._reports.values()) if r.intent_id == target), None)
        return self.report_for_read(report) if report is not None else None

    def attach_external_evidence_readonly(self, sandboxed: UntrustedExternalText) -> None:
        """Store quarantined external text; strictly forbidden from modifying any DesignIntent or lock."""
        if not sandboxed.untrusted_readonly:
            raise ValueError("External text must be marked untrusted_readonly=True")
        redacted = sandboxed.model_copy_validated(
            source_id=redact_secret_markers(sandboxed.source_id),
            wrapped_payload=redact_secret_markers(sandboxed.wrapped_payload),
        )
        self._external_notes.append(redacted)
        log_path = self.store_dir / "quarantined_external_evidence.jsonl"
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(redact_secret_markers(json.dumps(redacted.model_dump(mode="json"))) + "\n")

    @_with_current_state
    def create_initial_intent(
        self,
        *,
        turn_id: str,
        user_utterance: str,
        protein_sequence: str,
        additional_protein_sequences: list[str] | None = None,
        accession_or_source: str = "user_supplied",
        entry_version: str | None = None,
        is_secreted_without_sp: bool = False,
        reference_bundle: ConstructBundle | None = None,
        compliance_track: ComplianceProfile = "research_max",
        uridine_chemistry: UridineChemistry | None = None,
        locked_regions: dict[SegmentScope, str] | None = None,
        restriction_enzymes: list[str] | None = None,
        detarget_tissues: list[str] | None = None,
        protected_tissues: list[str] | None = None,
        prioritize_late_signal: bool = False,
        conflicting_single_pick: bool = False,
        use_commercial_surrogate: bool | None = None,
        modality: Modality | None = None,
        scaffold_id: str | None = None,
        random_seed: int = 42,
        redact_extra_secrets: Iterable[str] = (),
    ) -> DesignIntent:
        user_utterance = redact_secrets_in_text(user_utterance, extra_secrets=redact_extra_secrets)
        accession_or_source = redact_secrets_in_text(accession_or_source, extra_secrets=redact_extra_secrets)
        if entry_version is not None:
            entry_version = redact_secrets_in_text(entry_version, extra_secrets=redact_extra_secrets)
        if restriction_enzymes is not None:
            restriction_enzymes = [
                redact_secrets_in_text(x, extra_secrets=redact_extra_secrets) for x in restriction_enzymes
            ]
        if detarget_tissues is not None:
            detarget_tissues = [
                redact_secrets_in_text(x, extra_secrets=redact_extra_secrets) for x in detarget_tissues
            ]
        if protected_tissues is not None:
            protected_tissues = [
                redact_secrets_in_text(x, extra_secrets=redact_extra_secrets) for x in protected_tissues
            ]
        eff_modality: Modality = (
            reference_bundle.molecules[0].modality
            if (reference_bundle is not None and reference_bundle.molecules)
            else (modality or "linear_mrna")
        )
        eff_scaffold = (
            redact_secrets_in_text(scaffold_id, extra_secrets=redact_extra_secrets)
            if scaffold_id is not None
            else None
        )
        prot_clean = protein_sequence.strip().upper()
        norm_additional = _normalize_additional_protein_sequences(
            additional_protein_sequences,
            require_primary=prot_clean,
            check_primary=True,
        )
        if norm_additional is not None:
            if eff_modality != "linear_mrna":
                raise ValueError(
                    f"Multi-chain additional_protein_sequences is only supported for 'linear_mrna' modality, got '{eff_modality}'."
                )
            if reference_bundle is not None and len(reference_bundle.molecules) != 1 + len(norm_additional):
                raise ValueError(
                    f"additional_protein_sequences chain count ({1 + len(norm_additional)}) does not match "
                    f"reference_bundle molecules ({len(reference_bundle.molecules)})."
                )
            if prot_clean.endswith("*"):
                prot_clean = prot_clean[:-1]
        chem: UridineChemistry = (
            "unmodified_u"
            if eff_modality in ("circrna", "sarna")
            else (uridine_chemistry or _detect_chemistry(user_utterance))
        )
        enz = restriction_enzymes if restriction_enzymes is not None else (_detect_enzymes(user_utterance) or ["BspQI"])

        blocking_q: list[str] = []
        prot_status = "confirmed"
        intent_status = "ready"

        if not prot_clean.startswith("M") or is_secreted_without_sp:
            prot_status = "needs_protein_confirmation"
            intent_status = "needs_protein_confirmation"
            blocking_q.append(
                "Provided protein sequence lacks an N-terminal initiator Methionine ('M') or signal peptide. "
                "Please confirm whether to prepend initiator 'M', attach a specific signal peptide, or evaluate as-is."
            )
        elif conflicting_single_pick:
            intent_status = "needs_conflict_clarification"
            blocking_q.append(
                "Your objectives conflict for a single construct (maximal GC/stem stability vs completely unfolded CDS). "
                "Which objective has higher priority, or should mRNAx retain multiple Pareto trade-off candidates?"
            )

        ref_id = reference_bundle.bundle_id if reference_bundle else None
        if norm_additional is not None:
            targets_init = [
                ProteinTargetSpec(
                    protein_sequence=prot_clean,
                    transl_table=1,
                    accession_or_source=accession_or_source,
                    entry_version=entry_version,
                    confirmation_status=prot_status,  # type: ignore[arg-type]
                ),
                *[
                    ProteinTargetSpec(
                        protein_sequence=extra_aa,
                        transl_table=1,
                        accession_or_source=f"{accession_or_source}:chain_{idx}",
                        confirmation_status="confirmed",
                    )
                    for idx, extra_aa in enumerate(norm_additional, start=2)
                ],
            ]
            prot_lock_val = "|".join(t.protein_sequence for t in targets_init)
        elif reference_bundle is not None and len(reference_bundle.molecules) > 1:
            targets_init = [
                ProteinTargetSpec(
                    protein_sequence=(
                        prot_clean if idx == 0 else m.effective_translation_target.protein_sequence
                    ),
                    transl_table=m.effective_translation_target.transl_table,
                    accession_or_source=(
                        accession_or_source if idx == 0 else m.effective_translation_target.accession_or_source
                    ),
                    entry_version=(
                        entry_version if idx == 0 else m.effective_translation_target.entry_version
                    ),
                    confirmation_status=prot_status,  # type: ignore[arg-type]
                )
                for idx, m in enumerate(reference_bundle.molecules)
            ]
            prot_lock_val = "|".join(t.protein_sequence for t in targets_init)
        else:
            targets_init = [
                ProteinTargetSpec(
                    protein_sequence=prot_clean,
                    transl_table=1,
                    accession_or_source=accession_or_source,
                    entry_version=entry_version,
                    confirmation_status=prot_status,  # type: ignore[arg-type]
                )
            ]
            prot_lock_val = prot_clean

        locks: dict[str, LockedRegionSpec] = {
            "protein_sequence": LockedRegionSpec(
                scope="protein_sequence",
                locked_value=prot_lock_val,
                reason="100% amino-acid identity lock",
                locked_at_turn_id=turn_id,
            )
        }
        for scope_k, seq_v in (locked_regions or {}).items():
            locks[scope_k] = LockedRegionSpec(
                scope=scope_k,
                locked_value=redact_secrets_in_text(seq_v, extra_secrets=redact_extra_secrets).upper().replace("T", "U"),
                reason=f"User locked {scope_k}",
                locked_at_turn_id=turn_id,
            )

        mutable: list[SegmentScope] = [
            s for s in ("five_prime_utr", "cds", "three_prime_utr") if s not in locks
        ]

        effective_cc_surrogate = (
            compliance_track == "commercial_clean"
            if use_commercial_surrogate is None
            else bool(use_commercial_surrogate)
        )
        if effective_cc_surrogate:
            objs = [
                ObjectiveSpec(
                    requirement_id="obj_utr5_mrl_cc",
                    metric_id="utr5_mrl_pred",
                    direction="maximize",
                    priority_rank=1,
                    evidence_status="validated_predictor",
                    source_utterance=user_utterance,
                ),
                ObjectiveSpec(
                    requirement_id="obj_cds_stab_cc",
                    metric_id="cds_stability_pred",
                    direction="maximize",
                    priority_rank=2,
                    evidence_status="validated_predictor",
                    source_utterance=user_utterance,
                ),
                ObjectiveSpec(
                    requirement_id="obj_tx_hl_cc",
                    metric_id="consensus_halflife_pred",
                    direction="maximize",
                    priority_rank=2,
                    evidence_status="validated_predictor",
                    source_utterance=user_utterance,
                ),
                ObjectiveSpec(
                    requirement_id="obj_utr3_stab_cc",
                    metric_id="utr3_stability_pred",
                    direction="maximize",
                    priority_rank=3,
                    evidence_status="validated_predictor",
                    source_utterance=user_utterance,
                ),
            ]
            run_pol = RuntimePolicySpec(
                enabled_evaluators=["surrogate_v1_clean"],
                random_seed=random_seed,
            )
        else:
            objs = [
                ObjectiveSpec(
                    requirement_id="obj_5p_initiation_openness",
                    metric_id="kozak_unpaired_prob",
                    direction="maximize",
                    priority_rank=1,
                    evidence_status="thermodynamic_proxy",
                    source_utterance=user_utterance,
                ),
                ObjectiveSpec(
                    requirement_id="obj_cds_local_stability",
                    metric_id="cds_mean_window_mfe",
                    direction="minimize",
                    priority_rank=2,
                    evidence_status="thermodynamic_proxy",
                    source_utterance=user_utterance,
                ),
            ]
            run_pol = RuntimePolicySpec(random_seed=random_seed)
        if prioritize_late_signal or any(w in user_utterance for w in ("后期信号", "late signal", "48h", "half-life")):
            objs.append(
                ObjectiveSpec(
                    requirement_id="obj_late_protein_persistence",
                    metric_id="late_timepoint_protein_auc",
                    direction="maximize",
                    priority_rank=1,
                    evidence_status="unverified_target_for_wetlab",
                    source_utterance=user_utterance,
                )
            )

        tracks = (
            ["minimal_change_vs_r", "exploratory_pareto"]
            if reference_bundle is not None
            else ["computational_reference", "exploratory_pareto"]
        )

        cell_ctrl = (
            CellExpressionSpec(
                detarget_tissues=detarget_tissues,
                protected_tissues=protected_tissues or [],
            )
            if detarget_tissues
            else None
        )

        intent = DesignIntent(
            intent_id=f"int_{uuid.uuid4().hex[:8]}",
            project_id=self.project_id,
            intent_version=len(self._intents) + 1,
            parent_intent_id=self._intents[-1].intent_id if self._intents else None,
            created_from_turn_ids=[turn_id],
            entry_mode="protein_with_reference_r" if reference_bundle is not None else "protein_only",
            compliance_track=compliance_track,
            uridine_chemistry=chem,
            status=intent_status,  # type: ignore[arg-type]
            blocking_questions=blocking_q,
            protein_targets=targets_init,
            reference_bundle_id=ref_id,
            locked_scopes=locks,
            mutable_scopes=mutable,
            constraints=default_m1_constraints(restriction_enzymes=enz),
            objectives=objs,
            cell_expression_control=cell_ctrl,
            portfolio_tracks=tracks,  # type: ignore[arg-type]
            runtime_policy=run_pol,
            modality=eff_modality,
            scaffold_id=eff_scaffold,
        )
        if reference_bundle is not None:
            self.register_bundle(reference_bundle, set_as_reference=False)
        self._persist_intent(
            intent,
            publish_reference_bundle_id=(reference_bundle.bundle_id if reference_bundle is not None else None),
        )
        return intent

    def create_turn_intent_canonical(
        self,
        *,
        user_utterance: str,
        protein_sequence: str | None = None,
        additional_protein_sequences: list[str] | None = None,
        accession_or_source: str = "GenBank:U55762.1",
        entry_version: str | None = None,
        compliance_track: ComplianceProfile | None = None,
        uridine_chemistry: UridineChemistry | None = None,
        restriction_enzymes: list[str] | None = None,
        locked_five_prime_utr: str | None = None,
        detarget_liver: bool | None = None,
        prioritize_late_signal: bool = False,
        conflicting_single_pick: bool = False,
        reference_bundle_id: str | None = None,
        use_preset_reference_r: bool = False,
        use_commercial_surrogate: bool | None = None,
        modality: Modality | None = None,
        scaffold_id: str | None = None,
        random_seed: int = 42,
        turn_id: str | None = None,
        redact_extra_secrets: Iterable[str] = (),
    ) -> DesignIntent:
        """Single shared initial-turn adapter used identically by HTTP `/turn` and MCP `design_constructs`."""
        validated = InitialTurnInput.model_validate(
            {
                "user_utterance": user_utterance,
                "protein_sequence": protein_sequence,
                "additional_protein_sequences": additional_protein_sequences,
                "accession_or_source": accession_or_source,
                "entry_version": entry_version,
                "compliance_track": compliance_track,
                "uridine_chemistry": uridine_chemistry,
                "restriction_enzymes": restriction_enzymes,
                "locked_five_prime_utr": locked_five_prime_utr,
                "detarget_liver": detarget_liver,
                "prioritize_late_signal": prioritize_late_signal,
                "conflicting_single_pick": conflicting_single_pick,
                "reference_bundle_id": reference_bundle_id,
                "use_preset_reference_r": use_preset_reference_r,
                "use_commercial_surrogate": use_commercial_surrogate,
                "modality": modality,
                "scaffold_id": scaffold_id,
                "random_seed": random_seed,
            }
        )
        user_utterance = redact_secrets_in_text(validated.user_utterance, extra_secrets=redact_extra_secrets)
        protein_sequence = validated.protein_sequence
        accession_or_source = redact_secrets_in_text(validated.accession_or_source, extra_secrets=redact_extra_secrets)
        entry_version = (
            redact_secrets_in_text(validated.entry_version, extra_secrets=redact_extra_secrets)
            if validated.entry_version is not None
            else None
        )
        compliance_track = validated.compliance_track
        uridine_chemistry = validated.uridine_chemistry
        restriction_enzymes = (
            [redact_secrets_in_text(x, extra_secrets=redact_extra_secrets) for x in validated.restriction_enzymes]
            if validated.restriction_enzymes is not None
            else None
        )
        locked_five_prime_utr = (
            redact_secrets_in_text(validated.locked_five_prime_utr, extra_secrets=redact_extra_secrets)
            if validated.locked_five_prime_utr is not None
            else None
        )
        detarget_liver = validated.detarget_liver
        prioritize_late_signal = validated.prioritize_late_signal
        conflicting_single_pick = validated.conflicting_single_pick
        reference_bundle_id = validated.reference_bundle_id
        use_preset_reference_r = validated.use_preset_reference_r
        use_commercial_surrogate = validated.use_commercial_surrogate
        random_seed = validated.random_seed
        snap = json.loads((self.seed_dir / "egfp_reporter_snapshot.json").read_text(encoding="utf-8"))
        raw_prot = (protein_sequence or "").strip()
        if raw_prot:
            prot_seq = raw_prot
        else:
            low_utt = user_utterance.lower()
            names_egfp = any(k in low_utt for k in ("egfp", "gfp", "p42212", "u55762"))
            other_targets = any(
                k in low_utt for k in ("mcherry", "cas9", "luciferase", "fluc", "spike", "insulin", "epo")
            )
            accession_is_egfp = accession_or_source in ("GenBank:U55762.1", "UniProt:P42212")
            allow_preset_egfp = bool(
                accession_is_egfp
                and not other_targets
                and (use_preset_reference_r or names_egfp)
            )
            if not allow_preset_egfp:
                raise ValueError(
                    "Missing protein_sequence: supply an explicit amino-acid sequence or call search_protein_assets for a verified preset."
                )
            prot_seq = snap["mammalian_reporter_egfp"]["protein_sequence"].strip()
        ref_bundle: ConstructBundle | None = None
        if reference_bundle_id:
            ref_bundle = self.get_bundle(reference_bundle_id)
        elif use_preset_reference_r:
            ref_bundle = build_preset_reference_r(self.seed_dir)

        effective_entry_version = (
            entry_version
            if entry_version is not None
            else ("AAB02572.1" if accession_or_source == "GenBank:U55762.1" else None)
        )
        locks_init: dict[SegmentScope, str] | None = (
            {"five_prime_utr": locked_five_prime_utr} if locked_five_prime_utr else None
        )
        detarget_init = ["liver_hepatocyte"] if detarget_liver else None
        prot_init = ["skeletal_muscle"] if detarget_liver else None
        if detarget_liver is None and any(
            k in user_utterance.lower() for k in ("mir-122", "肝脏去靶向", "避开肝脏", "detarget liver")
        ):
            detarget_init = ["liver_hepatocyte"]
            prot_init = ["skeletal_muscle"]

        effective_track: ComplianceProfile = compliance_track or "research_max"
        return self.create_initial_intent(
            turn_id=turn_id or f"turn_{len(self._intents) + 1:02d}",
            user_utterance=user_utterance,
            protein_sequence=prot_seq,
            additional_protein_sequences=validated.additional_protein_sequences,
            accession_or_source=accession_or_source,
            entry_version=effective_entry_version,
            reference_bundle=ref_bundle,
            compliance_track=effective_track,
            uridine_chemistry=uridine_chemistry,
            locked_regions=locks_init,
            restriction_enzymes=restriction_enzymes,
            detarget_tissues=detarget_init,
            protected_tissues=prot_init,
            prioritize_late_signal=prioritize_late_signal,
            conflicting_single_pick=conflicting_single_pick,
            use_commercial_surrogate=use_commercial_surrogate,
            modality=validated.modality,
            scaffold_id=validated.scaffold_id,
            random_seed=random_seed,
            redact_extra_secrets=redact_extra_secrets,
        )

    def refine_turn_intent_canonical(
        self,
        *,
        user_utterance: str,
        restriction_enzymes: list[str] | None = None,
        updated_restriction_enzymes: list[str] | None = None,
        locked_five_prime_utr: str | None = None,
        lock_cds_from_bundle_id: str | None = None,
        unlock_scopes: list[SegmentScope] | None = None,
        uridine_chemistry: UridineChemistry | None = None,
        compliance_track: ComplianceProfile | None = None,
        detarget_liver: bool | None = None,
        use_commercial_surrogate: bool | None = None,
        modality: Modality | None = None,
        scaffold_id: str | None = None,
        turn_id: str | None = None,
        expected_parent_intent_id: str | None = None,
        redact_extra_secrets: Iterable[str] = (),
    ) -> DesignIntent:
        """Single shared refine-turn adapter used identically by HTTP `/turn` and MCP `refine_construct`."""
        validated = RefineTurnInput.model_validate(
            {
                "user_utterance": user_utterance,
                "restriction_enzymes": restriction_enzymes,
                "updated_restriction_enzymes": updated_restriction_enzymes,
                "locked_five_prime_utr": locked_five_prime_utr,
                "lock_cds_from_bundle_id": lock_cds_from_bundle_id,
                "unlock_scopes": unlock_scopes,
                "uridine_chemistry": uridine_chemistry,
                "compliance_track": compliance_track,
                "detarget_liver": detarget_liver,
                "use_commercial_surrogate": use_commercial_surrogate,
                "modality": modality,
                "scaffold_id": scaffold_id,
            }
        )
        user_utterance = redact_secrets_in_text(validated.user_utterance, extra_secrets=redact_extra_secrets)
        raw_enzymes = (
            validated.updated_restriction_enzymes
            if validated.updated_restriction_enzymes is not None
            else validated.restriction_enzymes
        )
        eff_enzymes = (
            [redact_secrets_in_text(x, extra_secrets=redact_extra_secrets) for x in raw_enzymes]
            if raw_enzymes is not None
            else None
        )
        locked_five_prime_utr = (
            redact_secrets_in_text(validated.locked_five_prime_utr, extra_secrets=redact_extra_secrets)
            if validated.locked_five_prime_utr is not None
            else None
        )
        lock_cds_from_bundle_id = validated.lock_cds_from_bundle_id
        unlock_scopes = validated.unlock_scopes
        uridine_chemistry = validated.uridine_chemistry
        compliance_track = validated.compliance_track
        detarget_liver = validated.detarget_liver
        use_commercial_surrogate = validated.use_commercial_surrogate
        prev_intent = self.latest_intent()
        add_locks: dict[SegmentScope, str] | None = (
            {"five_prime_utr": locked_five_prime_utr} if locked_five_prime_utr else None
        )
        detarget: list[str] | None = None
        prot_tiss: list[str] | None = None
        if detarget_liver is True:
            detarget = ["liver_hepatocyte"]
            prot_tiss = ["skeletal_muscle"]
        elif detarget_liver is False:
            detarget = []
            prot_tiss = []
        elif any(k in user_utterance.lower() for k in ("mir-122", "肝脏去靶向", "避开肝脏", "detarget liver")):
            detarget = ["liver_hepatocyte"]
            prot_tiss = ["skeletal_muscle"]

        explicit_track_change = (
            compliance_track
            if (compliance_track is not None and compliance_track != prev_intent.compliance_track)
            else None
        )
        return self.apply_incremental_intent_update(
            turn_id=turn_id or f"turn_{len(self._intents) + 1:02d}",
            user_utterance=user_utterance,
            updated_restriction_enzymes=eff_enzymes,
            additional_locked_scopes=add_locks,
            lock_cds_from_bundle_id=lock_cds_from_bundle_id,
            unlocked_scopes=unlock_scopes,
            updated_chemistry=uridine_chemistry,
            updated_compliance_track=explicit_track_change,
            detarget_tissues=detarget,
            protected_tissues=prot_tiss,
            use_commercial_surrogate=bool(use_commercial_surrogate),
            updated_modality=validated.modality,
            updated_scaffold_id=validated.scaffold_id,
            expected_parent_intent_id=expected_parent_intent_id,
            redact_extra_secrets=redact_extra_secrets,
        )

    def extract_structured_intent_deterministic(
        self,
        *,
        turn_id: str,
        user_utterance: str,
        protein_sequence: str,
        accession_or_source: str = "GenBank:U55762.1",
        reference_bundle: ConstructBundle | None = None,
    ) -> DesignIntent:
        """Deterministic synonym-invariant parser mapping equivalent phrasing to identical core constraints."""
        locked_regions: dict[SegmentScope, str] = {}
        m_utr5 = re.search(
            r"(?:5'\s*utr|five_prime_utr).*?\b([ACGTU]{8,120})\b",
            user_utterance,
            re.IGNORECASE,
        )
        if m_utr5:
            locked_regions["five_prime_utr"] = m_utr5.group(1).upper().replace("T", "U")

        detarget: list[str] | None = None
        prot_tiss: list[str] | None = None
        if any(k in user_utterance.lower() for k in ("mir-122", "肝脏去靶向", "detarget liver")):
            detarget = ["liver_hepatocyte"]
            prot_tiss = ["skeletal_muscle"]

        return self.create_initial_intent(
            turn_id=turn_id,
            user_utterance=user_utterance,
            protein_sequence=protein_sequence,
            accession_or_source=accession_or_source,
            reference_bundle=reference_bundle,
            locked_regions=locked_regions,
            detarget_tissues=detarget,
            protected_tissues=prot_tiss,
        )

    @_with_current_state
    def apply_incremental_intent_update(
        self,
        *,
        turn_id: str,
        user_utterance: str,
        updated_restriction_enzymes: list[str] | None = None,
        additional_locked_scopes: dict[SegmentScope, str] | None = None,
        lock_cds_from_bundle_id: str | None = None,
        unlocked_scopes: list[SegmentScope] | None = None,
        updated_chemistry: UridineChemistry | None = None,
        updated_compliance_track: ComplianceProfile | None = None,
        detarget_tissues: list[str] | None = None,
        protected_tissues: list[str] | None = None,
        use_commercial_surrogate: bool = False,
        updated_modality: Modality | None = None,
        updated_scaffold_id: str | None = None,
        expected_parent_intent_id: str | None = None,
        redact_extra_secrets: Iterable[str] = (),
    ) -> DesignIntent:
        """Single-item incremental edit inheriting 100% of untouched locks, constraints, and targets."""
        user_utterance = redact_secrets_in_text(user_utterance, extra_secrets=redact_extra_secrets)
        prev = self.latest_intent()
        if expected_parent_intent_id is not None and prev.intent_id != expected_parent_intent_id:
            raise ValueError(
                f"Concurrent revision conflict: expected parent intent '{expected_parent_intent_id}', found '{prev.intent_id}'"
            )
        new_locks: dict[str, LockedRegionSpec] = {}
        for l_k, l_obj in prev.locked_scopes.items():
            if l_k in (unlocked_scopes or []):
                continue
            new_locks[l_k] = l_obj.model_copy_validated(
                reason=redact_secrets_in_text(l_obj.reason, extra_secrets=redact_extra_secrets)
            )
        for s_k, s_val in (additional_locked_scopes or {}).items():
            new_locks[s_k] = LockedRegionSpec(
                scope=s_k,
                locked_value=redact_secrets_in_text(s_val, extra_secrets=redact_extra_secrets).upper().replace("T", "U"),
                reason=f"Locked at {turn_id}: {user_utterance}",
                locked_at_turn_id=turn_id,
            )

        if lock_cds_from_bundle_id is not None:
            source = self.get_bundle(lock_cds_from_bundle_id)
            molecule = source.molecules[0]
            segment = molecule.get_segment("cds")
            if segment is None:
                raise ValueError(f"Selected bundle {source.bundle_id} has no primary CDS")
            new_locks["cds"] = BundleSegmentLock(
                scope="cds", locked_value=molecule.sequence[segment.start:segment.end],
                reason=f"Locked at {turn_id}: {user_utterance}", locked_at_turn_id=turn_id,
                source_bundle_id=source.bundle_id, source_bundle_digest=source.version_digest,
                source_molecule_id=molecule.molecule_id, source_segment=segment,
            )

        mutable: list[SegmentScope] = [
            s for s in ("five_prime_utr", "cds", "three_prime_utr") if s not in new_locks
        ]

        clean_enz = (
            [redact_secrets_in_text(x, extra_secrets=redact_extra_secrets) for x in updated_restriction_enzymes]
            if updated_restriction_enzymes is not None
            else None
        )
        new_constraints: list[ConstraintSpec] = []
        replaced_enz = False
        for c in prev.constraints:
            clean_locator = (
                redact_secrets_in_text(c.source_locator, extra_secrets=redact_extra_secrets)
                if c.source_locator
                else c.source_locator
            )
            clean_params = redact_secrets_in_obj(c.params, extra_secrets=redact_extra_secrets, redact_keys=False)
            if c.constraint_kind == "type_iis_restriction_free" and clean_enz is not None:
                new_constraints.append(
                    c.model_copy_validated(
                        source_type="user_explicit",
                        source_locator=turn_id,
                        params={"enzymes": clean_enz, "forbidden_motifs": clean_params.get("forbidden_motifs", [])},
                    )
                )
                replaced_enz = True
            else:
                new_constraints.append(
                    c.model_copy_validated(source_locator=clean_locator, params=clean_params)
                )
        if clean_enz is not None and not replaced_enz:
            new_constraints.append(
                ConstraintSpec(
                    requirement_id="gate_type_iis_restriction",
                    constraint_kind="type_iis_restriction_free",
                    target_scopes=frozenset({"full_transcript"}),
                    severity="hard_gate",
                    source_type="user_explicit",
                    source_locator=turn_id,
                    params={"enzymes": clean_enz, "forbidden_motifs": []},
                )
            )

        new_cell = prev.cell_expression_control
        if detarget_tissues is not None:
            new_cell = CellExpressionSpec(
                detarget_tissues=detarget_tissues,
                protected_tissues=protected_tissues or (prev.cell_expression_control.protected_tissues if prev.cell_expression_control else []),
            )

        new_objs = prev.objectives
        new_pol = prev.runtime_policy
        if use_commercial_surrogate:
            new_objs = [
                ObjectiveSpec(
                    requirement_id="obj_utr5_mrl_cc",
                    metric_id="utr5_mrl_pred",
                    direction="maximize",
                    priority_rank=1,
                    evidence_status="validated_predictor",
                    source_utterance=user_utterance,
                ),
                ObjectiveSpec(
                    requirement_id="obj_cds_stab_cc",
                    metric_id="cds_stability_pred",
                    direction="maximize",
                    priority_rank=2,
                    evidence_status="validated_predictor",
                    source_utterance=user_utterance,
                ),
                ObjectiveSpec(
                    requirement_id="obj_tx_hl_cc",
                    metric_id="consensus_halflife_pred",
                    direction="maximize",
                    priority_rank=2,
                    evidence_status="validated_predictor",
                    source_utterance=user_utterance,
                ),
                ObjectiveSpec(
                    requirement_id="obj_utr3_stab_cc",
                    metric_id="utr3_stability_pred",
                    direction="maximize",
                    priority_rank=3,
                    evidence_status="validated_predictor",
                    source_utterance=user_utterance,
                ),
            ]
            new_pol = prev.runtime_policy.model_copy_validated(
                enabled_evaluators=["surrogate_v1_clean"]
            )

        else:
            new_objs = [
                (
                    o.model_copy_validated(
                        source_utterance=redact_secrets_in_text(
                            o.source_utterance, extra_secrets=redact_extra_secrets
                        )
                    )
                    if o.source_utterance
                    else o
                )
                for o in prev.objectives
            ]

        new_targets = [
            t.model_copy_validated(
                accession_or_source=redact_secrets_in_text(
                    t.accession_or_source, extra_secrets=redact_extra_secrets
                ),
                entry_version=(
                    redact_secrets_in_text(t.entry_version, extra_secrets=redact_extra_secrets)
                    if t.entry_version is not None
                    else None
                ),
            )
            for t in prev.protein_targets
        ]
        new_blocking_q = [
            redact_secrets_in_text(q, extra_secrets=redact_extra_secrets)
            for q in prev.blocking_questions
        ]
        eff_next_mod = updated_modality or prev.modality
        eff_next_scaf = updated_scaffold_id if updated_scaffold_id is not None else (
            prev.scaffold_id if eff_next_mod == prev.modality else None
        )
        eff_next_chem: UridineChemistry = (
            "unmodified_u"
            if eff_next_mod in ("circrna", "sarna")
            else (updated_chemistry or prev.uridine_chemistry)
        )
        next_ref_id = (
            None
            if eff_next_mod != prev.modality
            else (self.active_reference_bundle_id or prev.reference_bundle_id)
        )
        updated = prev.model_copy_validated(
            intent_id=f"int_{uuid.uuid4().hex[:8]}",
            intent_version=prev.intent_version + 1,
            parent_intent_id=prev.intent_id,
            created_from_turn_ids=[*prev.created_from_turn_ids, turn_id],
            protein_targets=new_targets,
            blocking_questions=new_blocking_q,
            reference_bundle_id=next_ref_id,
            uridine_chemistry=eff_next_chem,
            compliance_track=updated_compliance_track or prev.compliance_track,
            locked_scopes=new_locks,
            mutable_scopes=mutable,
            constraints=new_constraints,
            objectives=new_objs,
            cell_expression_control=new_cell,
            runtime_policy=new_pol,
            modality=eff_next_mod,
            scaffold_id=eff_next_scaf,
        )
        return self._persist_intent(updated)

    @_with_current_state
    def resolve_conflict_with_pareto_tracks(self, *, turn_id: str, user_utterance: str) -> DesignIntent:
        prev = self.latest_intent()
        resolved = prev.model_copy_validated(
            intent_id=f"int_{uuid.uuid4().hex[:8]}",
            intent_version=prev.intent_version + 1,
            parent_intent_id=prev.intent_id,
            created_from_turn_ids=[*prev.created_from_turn_ids, turn_id],
            status="ready",
            blocking_questions=[],
            portfolio_tracks=["computational_reference", "exploratory_pareto"],
        )
        return self._persist_intent(resolved)

    @_with_current_state
    def execute_design(self, intent: DesignIntent | None = None) -> list[ConstructBundle]:
        active_intent = intent or self.latest_intent()
        ref: ConstructBundle | None = None
        if active_intent.reference_bundle_id:
            ref = self._bundles.get(active_intent.reference_bundle_id)
            if ref is None:
                raise ValueError(
                    f"Referenced bundle '{active_intent.reference_bundle_id}' is missing from project store '{self.store_dir}'"
                )
            self._bundle_intent_map.setdefault(ref.bundle_id, active_intent.intent_id)
        bundles, report, manifest = run_design_pipeline(
            active_intent, reference_bundle=ref, seed_dir=self.seed_dir
        )
        for idx, b in enumerate(bundles):
            self.register_bundle(
                b,
                set_as_reference=(idx == 0 and self.active_reference_bundle_id is None),
                originating_intent_id=(active_intent.intent_id if b.bundle_id not in self._bundles else None),
            )
        if report.created_at_ns is None:
            report = report.model_copy_validated(created_at_ns=next_report_timestamp_ns())
        self._ambiguous_report_intents.discard(active_intent.intent_id)
        self._reports[report.report_id] = report
        if isinstance(manifest, RunManifest):
            intent_path = self.store_dir / "intents" / f"intent_v{active_intent.intent_version:03d}_{active_intent.intent_id}.json"
            if intent_path.is_file():
                content = intent_path.read_bytes()
                manifest = manifest.model_copy_validated(artifacts=[*manifest.artifacts, ArtifactChecksumRecord(
                    logical_name="input_intent", local_relpath=str(intent_path.relative_to(self.store_dir)),
                    sha256=hashlib.sha256(content).hexdigest(), md5=hashlib.md5(content).hexdigest(), bytes=len(content),
                )])
        self._manifests[manifest.execution_id] = manifest
        self._record_history_blocks(report, manifest.commercial_release_block() if isinstance(manifest, RunManifest) else None)
        if hasattr(manifest, "model_dump"):
            (self.store_dir / "manifests").mkdir(exist_ok=True)
            _write_json(self.store_dir / "manifests" / f"{manifest.execution_id}.json", manifest.model_dump(mode="json"))
        _write_json(self.store_dir / "reports" / f"{report.report_id}.json", report.model_dump(mode="json"), exclusive=True)
        self._save_state_meta()
        return bundles

    @_with_current_state
    def build_multi_segment_variant_for_comparison(self, base_bundle: ConstructBundle) -> ConstructBundle:
        """Helper producing a distinct 5'UTR + synonymous CDS variant for multi-segment attribution tests."""
        utr_reg = json.loads((self.seed_dir / "utr_components_v1.json").read_text(encoding="utf-8"))
        mol = base_bundle.molecules[0]
        cur_u5 = mol.get_segment_sequence("five_prime_utr")
        alt_u5 = next(
            u
            for u in utr_reg["five_prime_utrs"]
            if not u["has_uaug"] and u["sequence"] != cur_u5
        )
        # Flip one synonymous codon in CDS
        cds_seq = mol.get_segment_sequence("cds")
        alt_cds = cds_seq
        for i in range(3, len(cds_seq) - 3, 3):
            c = cds_seq[i : i + 3]
            if c == "GGC":
                alt_cds = cds_seq[:i] + "GGA" + cds_seq[i + 3 :]
                break
            if c == "CUG":
                alt_cds = cds_seq[:i] + "CUC" + cds_seq[i + 3 :]
                break

        spans = []
        parts = []
        cursor = 0
        for span in mol.segments:
            piece = mol.sequence[span.start:span.end]
            if span.segment_type == "five_prime_utr":
                piece = alt_u5["sequence"]
                span = span.model_copy_validated(
                    component_id=alt_u5["component_id"], origin_source=alt_u5["origin_source"],
                    license_track=alt_u5["license_track"], action_taken="selected_from_library",
                )
            elif span.segment_type == "cds" and alt_cds != piece:
                piece = alt_cds
                span = span.model_copy_validated(action_taken="synonymous_mutated")
            else:
                span = span.model_copy_validated(action_taken="locked_verbatim")
            parts.append(piece)
            spans.append(span.model_copy_validated(start=cursor, end=cursor + len(piece)))
            cursor += len(piece)
        new_mol = mol.model_copy_validated(sequence="".join(parts), segments=spans, annotations=[])
        alt_bundle = ConstructBundle.create(
            bundle_id=f"{base_bundle.bundle_id}_multi_alt",
            parent_bundle_id=base_bundle.bundle_id,
            derivation_action="de_novo_design",
            target_stoichiometry=base_bundle.target_stoichiometry,
            co_expression_mode=base_bundle.co_expression_mode,
            molecules=[new_mol, *base_bundle.molecules[1:]],
            external_evidence=base_bundle.external_evidence,
        )
        self.register_bundle(alt_bundle, originating_intent_id=self._bundle_intent_map.get(base_bundle.bundle_id))
        return alt_bundle

    def _spawn_verbatim_disambiguation_chimera(
        self, ref_a: ConstructBundle, winner_b: ConstructBundle
    ) -> ConstructBundle | None:
        """Create a zero-junction-edit pure verbatim chimera (priority: five_prime_utr > three_prime_utr > cds)."""
        pairs = paired_chains(ref_a, winner_b)
        constraint_engine = ScopedConstraintEngine.from_seed_dir(self.seed_dir)
        intent = self.latest_intent()
        for swap_scope in ("five_prime_utr", "three_prime_utr", "cds"):
            for idx, (mol_a, mol_b) in enumerate(pairs):
                if mol_a.get_segment_sequence(swap_scope) == mol_b.get_segment_sequence(swap_scope):
                    continue
                parts = []
                spans = []
                cursor = 0
                for scope in _SEGMENT_SCOPES:
                    donor = mol_b if scope == swap_scope else mol_a
                    span = donor.get_segment(scope)
                    if span is None:
                        continue
                    part = donor.sequence[span.start:span.end]
                    parts.append(part)
                    spans.append(span.model_copy_validated(
                        start=cursor, end=cursor + len(part),
                        action_taken="chimera_spliced" if scope == swap_scope else "locked_verbatim",
                    ))
                    cursor += len(part)
                molecules = list(ref_a.molecules)
                molecules[idx] = mol_a.model_copy_validated(sequence="".join(parts), segments=spans, annotations=[])
                chimera = ConstructBundle.create(
                    bundle_id=f"bnd_chimera_{uuid.uuid4().hex}",
                    parent_bundle_id=winner_b.bundle_id,
                    derivation_action="disambiguation_chimera",
                    target_stoichiometry=ref_a.target_stoichiometry,
                    co_expression_mode=ref_a.co_expression_mode,
                    molecules=molecules,
                    external_evidence=[*ref_a.external_evidence, *(r for r in winner_b.external_evidence if r not in ref_a.external_evidence)],
                )
                check = constraint_engine.evaluate_bundle(
                    chimera, intent.constraints, compliance_track=intent.compliance_track,
                    expected_chemistry=intent.uridine_chemistry, locked_scopes=intent.locked_scopes,
                )
                if check.passed_all_hard_gates:
                    self.register_bundle(chimera, originating_intent_id=intent.intent_id)
                    return chimera

        return None

    @_with_current_state
    def ingest_experiment_feedback(
        self,
        record: ExperimentRecord,
        *,
        batch_comparable: bool = True,
        held_out_validation_passed: bool = False,
        request_turn_id: str | None = None,
    ) -> ExperimentRecord:
        """Deterministic 4-branch wet-lab feedback handler (Joint Ruling #6)."""
        exp_dir = (self.store_dir / "experiments").resolve()
        out_path = (exp_dir / f"{record.experiment_id}.json").resolve()
        if out_path.parent != exp_dir:
            raise ValueError(f"Experiment output path '{out_path}' escapes project experiments directory")

        if record.project_id != self.project_id:
            return record.model_copy_validated(
                feedback_decision=FeedbackDecisionRecord(
                    branch="experimental_confound_blocked",
                    new_reference_bundle_id=None,
                    weight_update_allowed=False,
                    rationale=(
                        f"Experiment project_id '{record.project_id}' does not match session project "
                        f"'{self.project_id}'; refusing cross-project mutation."
                    ),
                )
            )

        self._reload_experiments_from_disk()
        existing_exp = next((e for e in self._experiments if e.experiment_id == record.experiment_id), None)
        if existing_exp is not None:
            same_payload = existing_exp.model_dump(
                mode="json",
                exclude={"feedback_decision", "ingested_at_ns", "observations"},
            ) == record.model_dump(
                mode="json",
                exclude={"feedback_decision", "ingested_at_ns", "observations"},
            ) and [
                o.model_dump(mode="json", exclude={"physical_construct_digest"}) for o in existing_exp.observations
            ] == [
                o.model_dump(mode="json", exclude={"physical_construct_digest"}) for o in record.observations
            ]
            if same_payload and (
                existing_exp.extraction_state == "draft_extraction" or existing_exp.feedback_decision is not None
            ):
                # Re-verify physical construct independence on legacy cross-version retries
                existing_exp, _ = self._repair_experiment_in_memory(existing_exp)
                if out_path.is_file():
                    disk_payload = json.loads(out_path.read_text(encoding="utf-8"))
                    disk_dec = disk_payload.get("feedback_decision") if isinstance(disk_payload, dict) else None
                    disk_needs_downgrade = bool(
                        isinstance(disk_dec, dict)
                        and existing_exp.n_independent_constructs < 6
                        and (
                            disk_dec.get("weight_update_allowed") is True
                            or disk_dec.get("isolated_calibration_proposal_id") is not None
                        )
                    )
                    if disk_needs_downgrade:
                        _write_json(out_path, existing_exp.model_dump(mode="json"))
                self._experiments = [
                    existing_exp if e.experiment_id == existing_exp.experiment_id else e
                    for e in self._experiments
                ]
                if (
                    existing_exp.feedback_decision is not None
                    and existing_exp.feedback_decision.new_reference_bundle_id
                    and self._experiments
                    and existing_exp.experiment_id == self._experiments[-1].experiment_id
                ):
                    self.active_reference_bundle_id = existing_exp.feedback_decision.new_reference_bundle_id
                state_meta_path = self.store_dir / "project_state.json"
                persisted_meta = (
                    json.loads(state_meta_path.read_text(encoding="utf-8"))
                    if state_meta_path.is_file()
                    else {}
                )
                persisted_exp_order = persisted_meta.get("experiment_order") or []
                persisted_origins = persisted_meta.get("experiment_origins") or {}
                persisted_list = persisted_origins.get(existing_exp.experiment_id)
                persisted_has_turn = (
                    (isinstance(persisted_list, list) and request_turn_id in persisted_list)
                    or persisted_list == request_turn_id
                )
                if request_turn_id:
                    writers = self._experiment_origins.setdefault(record.experiment_id, [])
                    if request_turn_id not in writers:
                        writers.append(request_turn_id)
                if (
                    persisted_meta.get("active_reference_bundle_id") != self.active_reference_bundle_id
                    or existing_exp.experiment_id not in persisted_exp_order
                    or (request_turn_id and not persisted_has_turn)
                    or (
                        self._experiments
                        and persisted_meta.get("last_applied_experiment_id") != self._experiments[-1].experiment_id
                    )
                ):
                    self._save_state_meta()
                return existing_exp
            if not same_payload and existing_exp.extraction_state != "draft_extraction":
                raise ValueError(
                    f"Experiment '{record.experiment_id}' already exists with different observations or metadata"
                )

        def save(decision: FeedbackDecisionRecord | None) -> ExperimentRecord:
            stamped = [
                o.model_copy_validated(physical_construct_digest=self._bundles[o.bundle_id].physical_construct_digest)
                if o.bundle_id in self._bundles
                else o
                for o in record.observations
            ]
            updated = record.model_copy_validated(
                observations=stamped,
                feedback_decision=decision,
                ingested_at_ns=record.ingested_at_ns or next_report_timestamp_ns(),
            )
            _write_json(out_path, updated.model_dump(mode="json"))
            merged = [e for e in self._experiments if e.experiment_id != updated.experiment_id]
            merged.append(updated)
            merged.sort(
                key=lambda item: (
                    1 if item.ingested_at_ns is not None else 0,
                    int(item.ingested_at_ns) if item.ingested_at_ns is not None else 0,
                )
            )
            self._experiments = merged
            if request_turn_id:
                writers = self._experiment_origins.setdefault(updated.experiment_id, [])
                if request_turn_id not in writers:
                    writers.append(request_turn_id)
            if (
                decision is not None
                and decision.new_reference_bundle_id
                and self._experiments
                and updated.experiment_id == self._experiments[-1].experiment_id
            ):
                self.active_reference_bundle_id = decision.new_reference_bundle_id
            self._save_state_meta()
            return updated

        if record.extraction_state == "draft_extraction":
            return save(None)

        if not batch_comparable or not record.shared_controls:
            dec = FeedbackDecisionRecord(
                branch="experimental_confound_blocked",
                new_reference_bundle_id=None,
                weight_update_allowed=False,
                rationale="Cross-batch comparison or missing shared control; reference & weights preserved.",
            )
            return save(dec)

        # Verify all observed construct IDs & digests match registered session bundles
        for obs in record.observations:
            if obs.bundle_id not in self._bundles:
                dec = FeedbackDecisionRecord(
                    branch="experimental_confound_blocked",
                    new_reference_bundle_id=None,
                    weight_update_allowed=False,
                    rationale=f"Observed bundle '{obs.bundle_id}' is not registered in this project.",
                )
                return save(dec)
            if self._bundles[obs.bundle_id].version_digest != obs.version_digest:
                dec = FeedbackDecisionRecord(
                    branch="experimental_confound_blocked",
                    new_reference_bundle_id=None,
                    weight_update_allowed=False,
                    rationale=f"Version digest mismatch on '{obs.bundle_id}'; cannot align wet-lab observation.",
                )
                return save(dec)

        if self.active_reference_bundle_id is not None and self.active_reference_bundle_id not in {
            obs.bundle_id for obs in record.observations
        }:
            dec = FeedbackDecisionRecord(
                branch="experimental_confound_blocked",
                new_reference_bundle_id=None,
                weight_update_allowed=False,
                rationale=(
                    f"Active reference '{self.active_reference_bundle_id}' was not measured in batch "
                    f"'{record.batch_id}'; refusing unanchored reference switch."
                ),
            )
            return save(dec)

        # Compute biological replicate summary & interval per pre_registered_criteria
        crit = record.pre_registered_criteria
        stats: list[tuple[str, float, float, float, int]] = []  # (bundle_id, center, lo, hi, n_bio)
        for obs in record.observations:
            unit_vals = sorted(b.unit_mean for b in obs.biological_replicates)
            n_bio = len(unit_vals)
            if crit.aggregation_method == "median":
                mid = n_bio // 2
                center = unit_vals[mid] if n_bio % 2 == 1 else 0.5 * (unit_vals[mid - 1] + unit_vals[mid])
            elif crit.aggregation_method == "geometric_mean":
                center = math.exp(sum(math.log(max(x, 1e-12)) for x in unit_vals) / n_bio)
            else:
                center = sum(unit_vals) / n_bio

            if crit.variance_estimator == "iqr" and n_bio >= 3:
                q25 = unit_vals[int(0.25 * (n_bio - 1))]
                q75 = unit_vals[int(0.75 * (n_bio - 1))]
                lo, hi = q25, q75
            elif n_bio > 1:
                mean_v = sum(unit_vals) / n_bio
                sd = math.sqrt(sum((x - mean_v) ** 2 for x in unit_vals) / (n_bio - 1))
                half_w = sd if crit.variance_estimator == "sd" else (1.96 * sd / math.sqrt(n_bio))
                lo, hi = center - half_w, center + half_w
            else:
                half_w = abs(center) * 0.25
                lo, hi = center - half_w, center + half_w
            stats.append((obs.bundle_id, center, lo, hi, n_bio))

        stats.sort(key=lambda t: -t[1])
        best_id, best_center, best_lo, _, best_nbio = stats[0]
        cur_ref_id = self.active_reference_bundle_id or stats[-1][0]
        ref_stat = next(s for s in stats if s[0] == cur_ref_id)
        _, ref_center, _, ref_hi, ref_nbio = ref_stat

        fold = best_center / max(1e-9, ref_center)
        ci_non_overlap = best_lo > ref_hi

        if (
            best_id == cur_ref_id
            or best_nbio < crit.min_biological_replicates
            or ref_nbio < crit.min_biological_replicates
            or fold < crit.min_fold_change
            or (crit.require_non_overlapping_ci and not ci_non_overlap)
        ):
            dec = FeedbackDecisionRecord(
                branch="within_noise_undecided",
                new_reference_bundle_id=None,
                weight_update_allowed=False,
                rationale=(
                    f"Difference within measurement noise or below criteria (fold={fold:.2f}x, "
                    f"ci_non_overlap={ci_non_overlap}); retaining reference {cur_ref_id}."
                ),
            )
            return save(dec)

        ref_bundle = self._bundles[cur_ref_id]
        win_bundle = self._bundles[best_id]
        try:
            pairs = paired_chains(ref_bundle, win_bundle)
        except ValueError as exc:
            return save(FeedbackDecisionRecord(branch="experimental_confound_blocked", rationale=str(exc)))

        changes = sum(
            a.get_segment_sequence(scope) != b.get_segment_sequence(scope)
            for a, b in pairs for scope in _SEGMENT_SCOPES
        )
        attr_status = ("multi_region_ambiguous" if changes > 1 else
                       "single_region_resolved" if changes == 1 else "not_applicable")
        chimera = (
            self._spawn_verbatim_disambiguation_chimera(ref_bundle, win_bundle)
            if attr_status == "multi_region_ambiguous" else None
        )
        followup_note = (
            " No verbatim follow-up met the active constraints."
            if attr_status == "multi_region_ambiguous" and chimera is None else ""
        )

        phys_digests = {self._bundles[obs.bundle_id].physical_construct_digest for obs in record.observations}
        n_ind = min(record.n_independent_constructs, len(phys_digests))
        allow_cal = bool(n_ind >= 6 and held_out_validation_passed)
        iso_cal_id = f"cal_iso_{uuid.uuid4().hex[:8]}" if allow_cal else None

        # Check kinetic divergence: earlier timepoint (<24h) favored cur_ref_id while late (>=24h) favors best_id
        is_kinetic_divergence = False
        if record.timepoint_hours is not None and record.timepoint_hours >= 24.0:
            for prev_exp in self._experiments:
                if (
                    prev_exp.extraction_state == "verified_structured"
                    and prev_exp.feedback_decision is not None
                    and prev_exp.feedback_decision.branch != "experimental_confound_blocked"
                    and all(getattr(prev_exp, field) == getattr(record, field) for field in (
                        "project_id", "batch_id", "assay_type", "readout_name_exact", "unit",
                        "cell_or_model_context", "chemistry_track", "delivery_vehicle",
                    ))
                    and set(prev_exp.shared_controls) == set(record.shared_controls)
                    and prev_exp.timepoint_hours is not None
                    and prev_exp.timepoint_hours < 24.0
                ):
                    early_means = {
                        o.bundle_id: sum(b.unit_mean for b in o.biological_replicates) / len(o.biological_replicates)
                        for o in prev_exp.observations
                    }
                    if cur_ref_id in early_means and best_id in early_means and early_means[cur_ref_id] > early_means[best_id]:
                        is_kinetic_divergence = True
                        break

        dec = FeedbackDecisionRecord(
            branch="kinetic_divergence_update" if is_kinetic_divergence else "rank_reversal_switch_reference",
            new_reference_bundle_id=best_id,
            retained_early_control_bundle_id=cur_ref_id if is_kinetic_divergence else None,
            weight_update_allowed=allow_cal,
            held_out_validation_passed=allow_cal,
            isolated_calibration_proposal_id=iso_cal_id,
            sequence_attribution_status=attr_status,  # type: ignore[arg-type]
            spawned_disambiguation_chimera_id=chimera.bundle_id if chimera else None,
            rationale=(
                f"Empirical readout ({record.readout_name_exact}) favored {best_id} over {cur_ref_id} "
                f"by {fold:.2f}x with non-overlapping CIs (n_bio={best_nbio}). Switched project reference to {best_id}; "
                f"weight_update_allowed={allow_cal} (N={n_ind}).{followup_note}"
            ),
        )
        return save(dec)

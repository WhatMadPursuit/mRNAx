"""Schema 2: Versioned Requirement Sheet (`DesignIntent`)."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Literal, Self

from pydantic import Field, SerializerFunctionWrapHandler, field_serializer, field_validator, model_serializer, model_validator

from mrnax.schemas.common import ComplianceProfile, ConstraintScope, Modality, SegmentScope, UridineChemistry
from mrnax.schemas.construct_bundle import FrozenSchemaModel, ProteinTargetSpec, SegmentSpan


class LockedRegionSpec(FrozenSchemaModel):
    scope: SegmentScope | Literal["protein_sequence"]
    locked_value: str
    reason: str
    locked_at_turn_id: str


class BundleSegmentLock(LockedRegionSpec):
    source_bundle_id: str
    source_bundle_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_molecule_id: str
    source_segment: SegmentSpan

    @model_validator(mode="after")
    def _check_source_scope(self) -> Self:
        if self.scope != self.source_segment.segment_type:
            raise ValueError("Selected segment does not match lock scope")
        if len(self.locked_value) != self.source_segment.end - self.source_segment.start:
            raise ValueError("Selected segment length does not match lock value")
        return self


class ConstraintSpec(FrozenSchemaModel):
    requirement_id: str = Field(min_length=1)
    constraint_kind: str = Field(min_length=1)
    target_scopes: frozenset[ConstraintScope]
    severity: Literal["hard_gate", "advisory"] = "hard_gate"
    source_type: Literal["user_explicit", "system_safety_gate", "approved_rule"]
    source_locator: str = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("target_scopes")
    @classmethod
    def _require_non_empty_scopes(cls, v: frozenset[ConstraintScope]) -> frozenset[ConstraintScope]:
        if not v:
            raise ValueError("target_scopes must not be empty; every constraint requires explicit scopes")
        return v

    @field_serializer("target_scopes", when_used="json")
    def _serialize_sorted_scopes(self, v: frozenset[ConstraintScope]) -> list[ConstraintScope]:
        return sorted(v)


class ObjectiveSpec(FrozenSchemaModel):
    requirement_id: str = Field(min_length=1)
    metric_id: str = Field(min_length=1)
    direction: Literal["maximize", "minimize", "target_band"]
    priority_rank: int | None = Field(default=None, ge=1)
    evidence_status: Literal[
        "validated_predictor",
        "thermodynamic_proxy",
        "unverified_target_for_wetlab",
    ]
    source_utterance: str = Field(min_length=1)
    target_band: tuple[float, float] | None = None

    @model_validator(mode="after")
    def _validate_target_band_bounds(self) -> Self:
        if self.target_band is not None:
            lo, hi = self.target_band
            if lo is None or hi is None or not math.isfinite(lo) or not math.isfinite(hi) or lo > hi:
                raise ValueError(f"target_band must contain finite ordered bounds (lo <= hi), got {self.target_band!r}")
        return self


class CellExpressionSpec(FrozenSchemaModel):
    detarget_tissues: list[str] = Field(default_factory=list)
    protected_tissues: list[str] = Field(default_factory=list)
    mirna_copy_range: tuple[int, int] = (2, 4)
    spacer_nt_range: tuple[int, int] = (8, 16)
    min_stop_offset_nt: int = Field(default=15, ge=15)

    @model_validator(mode="after")
    def _require_protected_when_detargeting(self) -> Self:
        if self.detarget_tissues and not self.protected_tissues:
            raise ValueError(
                "protected_tissues must be declared whenever detarget_tissues is non-empty"
            )
        return self


class RuntimePolicySpec(FrozenSchemaModel):
    budget_tier: Literal["fast", "standard"] = "fast"
    max_wall_seconds: float = Field(default=60.0, gt=0.0)
    max_evaluations: int = Field(default=40, ge=1)
    random_seed: int = Field(default=42, ge=0, le=2_147_483_647)
    allow_cached_candidates: bool = False
    enabled_generators: list[str] = Field(
        default_factory=lambda: ["stage0_random", "dnachisel_cds", "pareto_nsga2"]
    )
    enabled_evaluators: list[str] = Field(default_factory=lambda: ["viennarna_thermo"])


class DesignIntent(FrozenSchemaModel):
    intent_id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    intent_version: int = Field(ge=1)
    parent_intent_id: str | None = None
    created_from_turn_ids: list[str] = Field(min_length=1)
    entry_mode: Literal[
        "biology_concept",
        "protein_only",
        "protein_with_reference_r",
        "experiment_feedback",
    ]
    compliance_track: ComplianceProfile = "research_max"
    uridine_chemistry: UridineChemistry = "m1psi"
    status: Literal[
        "ready",
        "needs_protein_confirmation",
        "needs_conflict_clarification",
        "draft_pending_confirmation",
    ] = "ready"
    blocking_questions: list[str] = Field(default_factory=list)
    protein_targets: list[ProteinTargetSpec] = Field(min_length=1)
    reference_bundle_id: str | None = None
    locked_scopes: dict[str, BundleSegmentLock | LockedRegionSpec] = Field(default_factory=dict)
    mutable_scopes: list[SegmentScope] = Field(
        default_factory=lambda: ["five_prime_utr", "cds", "three_prime_utr"]
    )
    constraints: list[ConstraintSpec] = Field(default_factory=list)
    objectives: list[ObjectiveSpec] = Field(default_factory=list)
    cell_expression_control: CellExpressionSpec | None = None
    co_translational_slowdown_codons: list[tuple[int, int]] = Field(default_factory=list)
    portfolio_tracks: list[
        Literal[
            "minimal_change_vs_r",
            "exploratory_pareto",
            "computational_reference",
            "disambiguation_chimera",
        ]
    ] = Field(default_factory=lambda: ["exploratory_pareto"])
    runtime_policy: RuntimePolicySpec = Field(default_factory=RuntimePolicySpec)
    modality: Modality = "linear_mrna"
    scaffold_id: str | None = Field(default=None, pattern=r"^[a-z0-9][a-z0-9_-]{1,63}$")

    @model_serializer(mode="wrap")
    def _omit_default_modality_and_scaffold(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, Any]:
        out: dict[str, Any] = handler(self)
        if out.get("modality") == "linear_mrna":
            out.pop("modality", None)
        if out.get("scaffold_id") is None:
            out.pop("scaffold_id", None)
        return out

    def digest(self) -> str:
        def _canon(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: _canon(v) for k, v in sorted(obj.items())}
            if isinstance(obj, (set, frozenset)):
                return sorted(_canon(x) for x in obj)
            if isinstance(obj, list):
                return [_canon(x) for x in obj]
            return obj

        # Normalize ConstraintSpec.target_scopes frozenset order deterministically across PYTHONHASHSEED
        raw = self.model_dump(mode="python")
        canonical = json.dumps(_canon(raw), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

"""Schema 5: Dual-Speed Wet-Lab Observation & Pre-Registered Criteria (`ExperimentRecord`)."""

from __future__ import annotations

import re
from typing import Any, Literal, Self

from pydantic import Field, field_validator, model_validator

from mrnax.schemas.construct_bundle import FrozenSchemaModel


class BiologicalReplicate(FrozenSchemaModel):
    """One independent biological experimental unit containing >=1 technical measurements."""

    bio_rep_id: str = Field(min_length=1)
    technical_replicates: list[float] = Field(min_length=1)

    @property
    def unit_mean(self) -> float:
        return sum(self.technical_replicates) / len(self.technical_replicates)


class ConstructObservation(FrozenSchemaModel):
    bundle_id: str = Field(min_length=1)
    version_digest: str = Field(min_length=64, max_length=64)
    physical_construct_digest: str | None = Field(default=None, min_length=64, max_length=64)
    source_row_or_figure: str = Field(min_length=1)
    biological_replicates: list[BiologicalReplicate] = Field(min_length=1)

    @model_validator(mode="after")
    def _require_unique_bio_rep_ids(self) -> Self:
        rep_ids = [b.bio_rep_id for b in self.biological_replicates]
        if len(rep_ids) != len(set(rep_ids)):
            raise ValueError("biological_replicates must have unique bio_rep_id identifiers")
        return self

    @property
    def n_biological_replicates(self) -> int:
        return len(self.biological_replicates)

    @property
    def n_total_technical_measurements(self) -> int:
        return sum(len(b.technical_replicates) for b in self.biological_replicates)


class PreRegisteredSwitchCriteria(FrozenSchemaModel):
    aggregation_method: Literal["mean", "median", "geometric_mean"] = "mean"
    variance_estimator: Literal["sd", "ci95", "iqr"] = "ci95"
    min_biological_replicates: int = Field(default=3, ge=2)
    min_fold_change: float = Field(default=1.5, gt=1.0)
    require_non_overlapping_ci: bool = True


class FeedbackDecisionRecord(FrozenSchemaModel):
    branch: Literal[
        "experimental_confound_blocked",
        "within_noise_undecided",
        "kinetic_divergence_update",
        "rank_reversal_switch_reference",
    ]
    new_reference_bundle_id: str | None = None
    retained_early_control_bundle_id: str | None = None
    weight_update_allowed: bool = False
    held_out_validation_passed: bool = False
    isolated_calibration_proposal_id: str | None = None
    sequence_attribution_status: Literal[
        "single_region_resolved",
        "multi_region_ambiguous",
        "not_applicable",
    ] = "not_applicable"
    spawned_disambiguation_chimera_id: str | None = None
    rationale: str = Field(min_length=1)


class ExperimentRecord(FrozenSchemaModel):
    experiment_id: str = Field(min_length=1, max_length=128)
    project_id: str = Field(min_length=1)

    @field_validator("experiment_id")
    @classmethod
    def _validate_safe_experiment_id(cls, v: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", v):
            raise ValueError(
                "experiment_id must be a single safe filename component matching [A-Za-z0-9][A-Za-z0-9._-]{0,127}"
            )
        return v
    batch_id: str = Field(min_length=1)
    extraction_state: Literal["verified_structured", "draft_extraction"]
    ambiguous_fields: list[str] = Field(default_factory=list)
    raw_extracted_snapshot: dict[str, Any] = Field(default_factory=dict)
    user_corrected_snapshot: dict[str, Any] = Field(default_factory=dict)
    assay_type: str = Field(min_length=1)
    readout_name_exact: str = Field(min_length=1)
    unit: str = Field(min_length=1)
    timepoint_hours: float | None = Field(default=None, ge=0.0)
    cell_or_model_context: str = Field(min_length=1)
    chemistry_track: str = Field(min_length=1)
    delivery_vehicle: str = Field(min_length=1)
    shared_controls: list[str] = Field(default_factory=list)
    pre_registered_criteria: PreRegisteredSwitchCriteria = Field(
        default_factory=PreRegisteredSwitchCriteria
    )
    observations: list[ConstructObservation] = Field(min_length=1)
    feedback_decision: FeedbackDecisionRecord | None = None
    ingested_at_ns: int | None = None

    @property
    def n_independent_constructs(self) -> int:
        return len({obs.physical_construct_digest or obs.version_digest for obs in self.observations})

    @model_validator(mode="after")
    def _enforce_statistical_honesty_and_draft_quarantine(self) -> Self:
        if self.extraction_state == "draft_extraction" and self.feedback_decision is not None:
            if (
                self.feedback_decision.new_reference_bundle_id is not None
                or self.feedback_decision.weight_update_allowed
            ):
                raise ValueError(
                    "draft_extraction records with unresolved ambiguity cannot switch reference or update weights"
                )
        if self.feedback_decision is not None and self.feedback_decision.weight_update_allowed:
            if self.n_independent_constructs < 6:
                raise ValueError(
                    f"weight_update_allowed must be False when independent constructs N={self.n_independent_constructs} < 6"
                )
            if not self.feedback_decision.held_out_validation_passed:
                raise ValueError(
                    "weight_update_allowed requires held_out_validation_passed=True on independent un-fitted candidates"
                )
        return self

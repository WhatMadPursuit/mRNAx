"""Schema 5: Dual-Speed Wet-Lab Observation & Pre-Registered Criteria (`ExperimentRecord`)."""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import Field, model_validator

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
    source_row_or_figure: str = Field(min_length=1)
    biological_replicates: list[BiologicalReplicate] = Field(min_length=1)

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
    isolated_calibration_proposal_id: str | None = None
    sequence_attribution_status: Literal[
        "single_region_resolved",
        "multi_region_ambiguous",
        "not_applicable",
    ] = "not_applicable"
    spawned_disambiguation_chimera_id: str | None = None
    rationale: str = Field(min_length=1)


class ExperimentRecord(FrozenSchemaModel):
    experiment_id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
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

    @property
    def n_independent_constructs(self) -> int:
        return len({obs.bundle_id for obs in self.observations})

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
        if (
            self.feedback_decision is not None
            and self.feedback_decision.weight_update_allowed
            and self.n_independent_constructs < 6
        ):
            raise ValueError(
                f"weight_update_allowed must be False when independent constructs N={self.n_independent_constructs} < 6"
            )
        return self

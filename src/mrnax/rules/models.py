"""Models for Scientific ClaimCards, 4-State Rule Records, Deduplicated Votes, and Replacement Cards."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import Field, model_validator

from mrnax.schemas.common import ConstraintScope, LicenseTrack, RuleStatus, UridineChemistry
from mrnax.schemas.construct_bundle import FrozenSchemaModel

_VAGUE_PERCENT_RE = re.compile(r"\b\d{1,3}(?:\.\d+)?%")


class ClaimCard(FrozenSchemaModel):
    claim_id: str = Field(min_length=1)
    status: Literal["unverified_draft", "verified_observation", "retracted_or_invalid"]
    source_type: Literal["primary_research", "patent_example", "patent_claim", "regulatory_epar", "review"]
    citation_id: str = Field(min_length=1)
    exact_location: str = Field(min_length=1)
    underlying_dataset_id: str = Field(min_length=1)
    lab_lineage_id: str = Field(min_length=1)
    region_scope: list[ConstraintScope] = Field(min_length=1)
    chemistry_scope: list[UridineChemistry] = Field(min_length=1)
    cell_or_model_context: str = Field(min_length=1)
    protein_count_n: int = Field(ge=0)
    readout_name_exact: str = Field(min_length=1)
    comparator: str = Field(min_length=1)
    effect_direction: Literal["increase", "decrease", "null_or_inconclusive", "predictive_fit"]
    effect_size_summary: str = ""
    caveats_or_counter_evidence: str = ""
    evidence_layer: Literal["raw_observation", "model_interpretation"] = "raw_observation"


class EvidenceObservationLink(FrozenSchemaModel):
    claim_id: str
    underlying_dataset_id: str
    evidence_layer: Literal["raw_observation", "model_interpretation"]


class ModelInterpretationNote(FrozenSchemaModel):
    """Kept strictly separated from raw experimental observations (Clean-Room & Joint Ruling #5)."""

    source_model_name: str
    source_model_license_track: LicenseTrack
    method: Literal["SHAP", "ISM", "TF_MoDISco", "regression_weight"]
    summary: str


class PreRegisteredRuleTestSpec(FrozenSchemaModel):
    hypothesis_id: str
    target_feature: str
    readout_name_exact: str
    applicable_chemistries: list[UridineChemistry]
    expected_direction: Literal["increase", "decrease"]
    min_abs_median_delta: float = Field(gt=0.0)
    max_mann_whitney_p: float = Field(gt=0.0, lt=1.0)
    variance_estimator: str = "bootstrap_iqr_and_replicate_concordance"


class RuleAuditEvent(FrozenSchemaModel):
    timestamp_utc: str
    previous_status: RuleStatus
    new_status: RuleStatus
    reason: str
    is_simulated_sandbox_only: bool = False


class RuleRecord(FrozenSchemaModel):
    rule_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    status: RuleStatus
    allowed_track: LicenseTrack
    target_scopes: list[ConstraintScope] = Field(min_length=1)
    applicable_chemistries: list[UridineChemistry] = Field(min_length=1)
    applicable_contexts: list[str] = Field(min_length=1)
    readout_name_exact: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    linked_observations: list[EvidenceObservationLink] = Field(default_factory=list)
    model_interpretations: list[ModelInterpretationNote] = Field(default_factory=list)
    audit_history: list[RuleAuditEvent] = Field(default_factory=list)


class ReplacementMetricThreshold(FrozenSchemaModel):
    metric_id: str = Field(min_length=1)
    definition_and_scale: str = Field(min_length=1)
    direction: Literal["higher_is_better", "lower_is_better", "closer_to_zero"]
    frozen_baseline_reference: str = Field(min_length=1)
    max_acceptable_degradation: str = Field(min_length=1)


class ComponentReplacementCard(FrozenSchemaModel):
    card_id: str = Field(min_length=1)
    component_id: str = Field(min_length=1)
    component_name_and_version: str = Field(min_length=1)
    role_in_pipeline: str = Field(min_length=1)
    current_license_track: LicenseTrack
    code_rights_basis: str = Field(min_length=1)
    weights_rights_basis: str = Field(min_length=1)
    training_and_parameter_data_rights_basis: str = Field(min_length=1)
    runtime_dependency_rights_basis: str = Field(min_length=1)
    derived_assets_clean_room_rule: str = Field(min_length=1)
    commercial_replacement_paths: list[str] = Field(min_length=1)
    frozen_benchmark_dataset_and_protocol: str = Field(min_length=1)
    acceptance_metrics: list[ReplacementMetricThreshold] = Field(min_length=2)
    max_compute_cost_budget: str = Field(min_length=1)
    overall_switch_rule: str = Field(min_length=1)
    rollback_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def _forbid_vague_overall_percentage(self) -> Self:
        if _VAGUE_PERCENT_RE.search(self.overall_switch_rule):
            raise ValueError(
                "overall_switch_rule contains a vague percentage; use explicit per-metric bounds in acceptance_metrics"
            )
        return self


def load_replacement_card_yaml(path: Path) -> ComponentReplacementCard:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ComponentReplacementCard.model_validate(raw)

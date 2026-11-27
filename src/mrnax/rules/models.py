"""Models for Scientific ClaimCards, 4-State Rule Records, Deduplicated Votes, and Replacement Cards."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Literal, NamedTuple, Self

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
    expected_effect_direction: Literal["positive", "negative"] | None = None
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
    measured_value: float | None = None
    numeric_min_bound: float | None = None
    numeric_max_bound: float | None = None


class ComponentReplacementCard(FrozenSchemaModel):
    card_id: str = Field(min_length=1)
    component_id: str = Field(min_length=1)
    approved_software_version: str | None = None
    approved_artifact_digests: dict[str, str] = Field(default_factory=dict)
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
        for m in self.acceptance_metrics:
            for num_v, fname in (
                (m.measured_value, "measured_value"),
                (m.numeric_min_bound, "numeric_min_bound"),
                (m.numeric_max_bound, "numeric_max_bound"),
            ):
                if num_v is not None and not math.isfinite(num_v):
                    raise ValueError(
                        f"Replacement card '{self.card_id}' metric '{m.metric_id}' has non-finite {fname}={num_v}"
                    )
            if m.measured_value is not None:
                if m.numeric_min_bound is not None and m.measured_value < m.numeric_min_bound:
                    raise ValueError(
                        f"Replacement card '{self.card_id}' metric '{m.metric_id}' failed switch threshold: "
                        f"measured {m.measured_value} < numeric_min_bound {m.numeric_min_bound}"
                    )
                if m.numeric_max_bound is not None and m.measured_value > m.numeric_max_bound:
                    raise ValueError(
                        f"Replacement card '{self.card_id}' metric '{m.metric_id}' failed switch threshold: "
                        f"measured {m.measured_value} > numeric_max_bound {m.numeric_max_bound}"
                    )
        return self


class ApprovedComponentRelease(NamedTuple):
    name: str
    version: str
    artifact_digests: dict[str, str]


APPROVED_REPLACEMENT_RELEASES = {
    "structrfm_rna_foundation": ApprovedComponentRelease(
        "structRFM",
        "0.0.9",
        {
            "config.json": "d362b6ed07057d24d5ec585711ef4d14b86f81747293cef8705c566e0af667c8",
            "model.safetensors": "e40d8ab5115a93b551bf4b2c8e6a52b179a6d5296f29a2906bcfe487ed258ace",
            "special_tokens_map.json": "311de3f4eed9d76a43bf0d71f10e62e086ca65ccce9f15d5da0d2098bf519ecc",
            "tokenizer.json": "2048853d0d43e493f2e44b59a0f247d874dc3feebbebe947402940f7b931afcc",
            "tokenizer_config.json": "c47120c0385c12a01bf1cefb1cc4bb689f0840dd61152a5f8b08b497793d6df4",
        },
    ),
    "orthrus_mamba_rna_foundation": ApprovedComponentRelease(
        "Orthrus",
        "1.0.0",
        {
            "config.json": "22ef4964de7757d76f6aecb09620483682e025c699b356132fe69d24f934011e",
            "model.safetensors": "0b3a297a429307a97693a5bbb68af13312bc4fcd580e6580cd3d6e59eeba5bd7",
        },
    ),
}


_MANDATORY_CARD_METRICS_BY_COMPONENT: dict[str, set[str]] = {
    "structrfm_rna_foundation": {
        "moderna_shapemap_per_nt_spearman_m1psi",
        "moderna_shapemap_chemistry_specificity_gain",
        "gpu_determinism_max_abs_diff",
        "gpu_l4_peak_vram_mb",
        "gpu_l4_latency_ms_per_seq",
    },
    "orthrus_mamba_rna_foundation": {
        "saluki_full13642_human_halflife_ood_spearman",
        "moderna_shapemap_per_nt_spearman_m1psi",
        "triton_scan_parity_max_diff",
        "triton_gpu_determinism_drift",
        "triton_l4_peak_vram_mb",
        "triton_l4_latency_ms_per_seq",
    },
}


def verify_replacement_card_strict(
    card: ComponentReplacementCard,
    *,
    expected_component_id: str | None = None,
    expected_version: str | None = None,
) -> None:
    """Strict runtime gate for active commercial_clean replacement components."""
    if expected_component_id is not None and card.component_id != expected_component_id:
        raise ValueError(
            f"Replacement card '{card.card_id}' declares component_id='{card.component_id}', "
            f"expected '{expected_component_id}'"
        )
    approved = APPROVED_REPLACEMENT_RELEASES.get(card.component_id)
    if approved is None:
        raise ValueError(f"Replacement card '{card.card_id}' has no approved component release")
    if expected_version is not None and approved.version != expected_version:
        raise ValueError(f"Requested catalog version '{expected_version}' differs from approved version '{approved.version}'")
    if card.approved_software_version != approved.version:
        raise ValueError(f"Replacement card '{card.card_id}' must declare approved_software_version='{approved.version}'")
    if card.component_name_and_version != f"{approved.name} {approved.version}":
        raise ValueError(f"Replacement card '{card.card_id}' component name and version disagree with its release")
    if approved.artifact_digests and card.approved_artifact_digests != approved.artifact_digests:
        raise ValueError(
            f"Replacement card '{card.card_id}' approved_artifact_digests do not match pinned release digests for '{card.component_id}'"
        )
    if card.current_license_track != "commercial_clean":
        raise ValueError(
            f"Replacement card '{card.card_id}' has current_license_track='{card.current_license_track}' (expected 'commercial_clean')"
        )
    metric_ids = [m.metric_id for m in card.acceptance_metrics]
    if len(metric_ids) != len(set(metric_ids)):
        raise ValueError(
            f"Replacement card '{card.card_id}' contains duplicate acceptance_metrics metric_id entries"
        )
    required_ids = _MANDATORY_CARD_METRICS_BY_COMPONENT.get(card.component_id)
    if required_ids is not None:
        missing = sorted(required_ids - set(metric_ids))
        if missing:
            raise ValueError(
                f"Replacement card '{card.card_id}' for '{card.component_id}' is missing mandatory metrics: {missing}"
            )
    for m in card.acceptance_metrics:
        if m.measured_value is None or not math.isfinite(m.measured_value):
            raise ValueError(
                f"Replacement card '{card.card_id}' metric '{m.metric_id}' is missing a finite measured_value"
            )
        if m.numeric_min_bound is None and m.numeric_max_bound is None:
            raise ValueError(
                f"Replacement card '{card.card_id}' metric '{m.metric_id}' must specify numeric_min_bound or numeric_max_bound"
            )


def load_replacement_card_yaml(path: Path) -> ComponentReplacementCard:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ComponentReplacementCard.model_validate(raw)

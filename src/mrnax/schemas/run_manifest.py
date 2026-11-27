"""Schema 4: Reproducibility, Provenance & CAS Execution Record (`RunManifest`)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from mrnax.schemas.common import ComplianceProfile, LicenseTrack
from mrnax.schemas.construct_bundle import FrozenSchemaModel


class ComponentInvocationRecord(FrozenSchemaModel):
    name: str
    version: str
    role: Literal["generator", "evaluator", "ranker", "controller", "exporter"]
    code_license: str
    weights_license: str
    data_license: str
    license_track: LicenseTrack
    wall_seconds: float = Field(ge=0.0)
    attempt_count: int = Field(ge=0)
    failure_count: int = Field(ge=0)


class ArtifactChecksumRecord(FrozenSchemaModel):
    logical_name: str
    local_relpath: str
    drive_file_id: str | None = None
    sha256: str = Field(min_length=64, max_length=64)
    md5: str = Field(min_length=32, max_length=32)
    bytes: int = Field(ge=0)


class MethodFTOAuditRecord(FrozenSchemaModel):
    """Dual-FTO algorithmic execution & runtime receipt provenance (`method_fto_guard_v1.1` + ADR-0003)."""

    overall_decision: str | None = Field(default=None, exclude_if=lambda value: value is None)
    run_id: str | None = None
    timestamp_utc: str = ""
    commercial_tier: str | None = None
    engine_git_commit: str | None = None
    optimization_mode: Literal[
        "full_transcript_non_partitioned_pareto",
        "stage0_or_reference_only",
        "research_sandbox_sliding_window",
        "not_executed",
        "unrecorded_legacy",
    ] = "unrecorded_legacy"
    uses_two_zone_initiation_body_partition: bool | None = None
    uses_dfa_cfg_intersection: bool | None = None
    sequence_fto_screen_applied: bool = False
    method_fto_guard_version: str = "unrecorded_legacy"
    certifies_100_percent_fto: Literal[False] = False
    counsel_fto_signoff_required: Literal[True] = True
    target_deployment_jurisdictions: list[str] = Field(
        default_factory=lambda: ["CN", "US", "EP", "JP"]
    )
    input_origin_type: Literal[
        "amino_acid_only",
        "wildtype_mrna_reference",
        "reference_construct_bundle",
        "not_executed",
        "unrecorded_legacy",
    ] = "unrecorded_legacy"
    wildtype_c_optimizable_replacement_fraction: float | None = None
    wildtype_c_optimizable_no_higher_g_fraction: float | None = None
    region_partition_strategy: str = "unrecorded_legacy"
    optimization_schedule_and_operators: list[str] = Field(default_factory=list)
    objective_formulation: dict[str, Any] = Field(default_factory=dict)
    three_layer_dependency_manifest: list[dict[str, Any]] = Field(default_factory=list)
    method_fto_boundary_checks: dict[str, dict[str, str]] = Field(default_factory=dict)
    sequence_fto_guard_v2_receipt: dict[str, Any] = Field(default_factory=dict)
    counsel_disclaimer: str = (
        "Algorithmic provenance & sequence-collision audit only; not a legal Freedom-to-Operate opinion "
        "and never certifies 100% FTO (review jurisdiction-specific method & composition claims before clinical/commercial filing)."
    )


class RunManifest(FrozenSchemaModel):
    overall_decision: str | None = Field(default=None, exclude_if=lambda value: value is None)
    submission_id: str = Field(min_length=1)
    execution_id: str = Field(min_length=1)
    attempt_id: str = Field(min_length=1)
    lease_epoch: int = Field(ge=1)
    git_commit_sha: str
    python_version: str
    platform_info: str
    compliance_track: ComplianceProfile
    invoked_components: list[ComponentInvocationRecord]
    unavailable_components: dict[str, str] = Field(default_factory=dict)
    rule_snapshot_version: str
    isolated_sandbox_override: bool = False
    prng_state_digest: str
    input_intent_digest: str
    artifacts: list[ArtifactChecksumRecord]
    replay_cli_command: str
    honesty_banner: str | None = None
    optimization_mode_audit: MethodFTOAuditRecord = Field(default_factory=MethodFTOAuditRecord)


    def commercial_release_block(self) -> str | None:
        audit = self.optimization_mode_audit
        if "BLOCK" in (self.overall_decision, audit.overall_decision):
            return "Producing manifest recorded an overall BLOCK decision."
        for family, check in audit.method_fto_boundary_checks.items():
            if check.get("status") == "BLOCK":
                return f"Producing manifest recorded a method-FTO block for {family}."
        for entry in audit.three_layer_dependency_manifest:
            count = entry.get("transitive_gpl_count")
            if type(count) is not int or count < 0:
                return "Producing manifest has an invalid dependency count."
            if count > 0:
                return f"Producing manifest recorded transitive_gpl_count={count} for {entry.get('asset_id')}."
        return None


__all__ = [
    "ArtifactChecksumRecord",
    "ComponentInvocationRecord",
    "MethodFTOAuditRecord",
    "RunManifest",
]

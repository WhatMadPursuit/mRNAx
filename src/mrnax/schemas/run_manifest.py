"""Schema 4: Reproducibility, Provenance & CAS Execution Record (`RunManifest`)."""

from __future__ import annotations

from typing import Literal

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
    """Dual-FTO algorithmic execution provenance (ADR-0003; distinct from SequenceFTOGuard)."""

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
    counsel_disclaimer: str = (
        "Algorithmic provenance & sequence-collision audit only; not a legal Freedom-to-Operate opinion "
        "(review jurisdiction-specific method & composition claims before clinical/commercial filing)."
    )


class RunManifest(FrozenSchemaModel):
    submission_id: str = Field(min_length=1)
    execution_id: str = Field(min_length=1)
    attempt_id: str = Field(min_length=1)
    lease_epoch: int = Field(ge=1)
    git_commit_sha: str
    python_version: str
    platform_info: str
    compliance_track: ComplianceProfile
    invoked_components: list[ComponentInvocationRecord]
    rule_snapshot_version: str
    isolated_sandbox_override: bool = False
    prng_state_digest: str
    input_intent_digest: str
    artifacts: list[ArtifactChecksumRecord]
    replay_cli_command: str
    honesty_banner: str | None = None
    optimization_mode_audit: MethodFTOAuditRecord = Field(default_factory=MethodFTOAuditRecord)


__all__ = [
    "ArtifactChecksumRecord",
    "ComponentInvocationRecord",
    "MethodFTOAuditRecord",
    "RunManifest",
]

"""Scientific Rule Tribunal: dataset deduplication, isolated sandbox counter-evidence, and draft quarantine."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from mrnax.rules.models import ClaimCard, RuleAuditEvent, RuleRecord
from mrnax.schemas.common import ComplianceProfile, RuleStatus, UridineChemistry

_DATASET_PREFIX_RE = re.compile(r"^(GEO:|ENA:|SRA:|NCBI:)", re.IGNORECASE)


def normalize_dataset_id(raw_id: str) -> str:
    cleaned = _DATASET_PREFIX_RE.sub("", raw_id.strip()).upper()
    # Canonicalize GSE114002 / PRJNA454863 / SRP144485 same underlying sequencing project!
    if cleaned in {"GSE114002", "PRJNA454863", "SRP144485"}:
        return "GSE114002_PRJNA454863"
    return cleaned


@dataclass(frozen=True)
class DeduplicatedVoteSummary:
    unique_dataset_votes: int
    unique_lab_lineage_votes: int
    excluded_review_count: int
    excluded_draft_count: int
    dataset_clusters: dict[str, list[str]]


def count_independent_dataset_votes(claims: list[ClaimCard]) -> DeduplicatedVoteSummary:
    """Count unique underlying datasets and lab lineages; reviews & unverified drafts get 0 votes."""
    excluded_reviews = 0
    excluded_drafts = 0
    clusters: dict[str, list[str]] = {}
    labs: set[str] = set()

    for c in claims:
        if c.source_type == "review":
            excluded_reviews += 1
            continue
        if c.status != "verified_observation":
            excluded_drafts += 1
            continue
        norm_ds = normalize_dataset_id(c.underlying_dataset_id)
        clusters.setdefault(norm_ds, []).append(c.claim_id)
        labs.add(c.lab_lineage_id.strip().lower())

    return DeduplicatedVoteSummary(
        unique_dataset_votes=len(clusters),
        unique_lab_lineage_votes=len(labs),
        excluded_review_count=excluded_reviews,
        excluded_draft_count=excluded_drafts,
        dataset_clusters=clusters,
    )


class RuleRegistry:
    """Dual-track rule store supporting zero-contamination draft inbox & isolated sandbox forks."""

    def __init__(
        self,
        backing_path: Path,
        *,
        track: ComplianceProfile,
        is_isolated_sandbox: bool = False,
    ) -> None:
        self.backing_path = backing_path
        self.track = track
        self.is_isolated_sandbox = is_isolated_sandbox
        self._draft_inbox: list[ClaimCard] = []

    @classmethod
    def create_with_initial_rules(
        cls, backing_path: Path, *, track: ComplianceProfile, rules: list[RuleRecord]
    ) -> RuleRegistry:
        backing_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "registry_track": track,
            "is_isolated_sandbox": False,
            "rules": [r.model_dump(mode="json") for r in rules],
            "draft_claim_inbox": [],
        }
        backing_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return cls(backing_path, track=track, is_isolated_sandbox=False)

    def _load_doc(self) -> dict:
        return json.loads(self.backing_path.read_text(encoding="utf-8"))

    def _save_doc(self, doc: dict) -> None:
        self.backing_path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def snapshot_version(self) -> str:
        doc = self._load_doc()
        rules_canonical = json.dumps(doc.get("rules", []), sort_keys=True)
        digest8 = hashlib.sha256(rules_canonical.encode("utf-8")).hexdigest()[:8]
        return f"rules_{self.track}@{digest8}"

    def fork_isolated_copy(self, sandbox_path: Path) -> RuleRegistry:
        """Create an isolated evaluation copy so simulated counter-evidence NEVER touches production."""
        doc = self._load_doc()
        doc["is_isolated_sandbox"] = True
        sandbox_path.parent.mkdir(parents=True, exist_ok=True)
        sandbox_path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return RuleRegistry(sandbox_path, track=self.track, is_isolated_sandbox=True)

    def ingest_draft_claim_card(self, card: ClaimCard) -> None:
        """Store unapproved ClaimCard in quarantined draft inbox; NEVER enters active `rules`."""
        if card.status != "unverified_draft":
            raise ValueError("Only unverified_draft ClaimCards belong in the draft inbox")
        doc = self._load_doc()
        inbox = list(doc.get("draft_claim_inbox", []))
        inbox.append(card.model_dump(mode="json"))
        doc["draft_claim_inbox"] = inbox
        self._save_doc(doc)

    def apply_counter_evidence(
        self,
        *,
        rule_id: str,
        reason: str,
        new_status: RuleStatus,
        is_simulated: bool = False,
    ) -> RuleRecord:
        if is_simulated and not self.is_isolated_sandbox:
            raise PermissionError(
                "Simulated counter-evidence may ONLY be written to an isolated sandbox fork!"
            )
        doc = self._load_doc()
        updated_rule: RuleRecord | None = None
        new_rules = []
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        for raw_r in doc.get("rules", []):
            r = RuleRecord.model_validate(raw_r)
            if r.rule_id == rule_id:
                ev = RuleAuditEvent(
                    timestamp_utc=now,
                    previous_status=r.status,
                    new_status=new_status,
                    reason=reason,
                    is_simulated_sandbox_only=is_simulated,
                )
                r = r.model_copy_validated(status=new_status, audit_history=[*r.audit_history, ev])
                updated_rule = r
            new_rules.append(r.model_dump(mode="json"))
        if updated_rule is None:
            raise KeyError(f"Unknown rule_id '{rule_id}'")
        doc["rules"] = new_rules
        self._save_doc(doc)
        return updated_rule

    def get_active_rules(
        self, *, chemistry: UridineChemistry, context: str = "general_mammalian_ivt"
    ) -> list[RuleRecord]:
        doc = self._load_doc()
        active: list[RuleRecord] = []
        for raw_r in doc.get("rules", []):
            r = RuleRecord.model_validate(raw_r)
            if r.status not in {"verified_rule", "conditional_rule"}:
                continue
            if self.track == "commercial_clean" and r.allowed_track != "commercial_clean":
                continue
            if chemistry not in r.applicable_chemistries:
                continue
            if context not in r.applicable_contexts and "general_mammalian_ivt" not in r.applicable_contexts:
                continue
            active.append(r)
        return active

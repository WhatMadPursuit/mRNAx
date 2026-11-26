"""Candidate sequence Freedom-to-Operate (FTO) collision & patent genus guard (`sequence_fto_guard`)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from Bio.Align import PairwiseAligner

from mrnax.schemas.common import ComplianceProfile
from mrnax.schemas.construct_bundle import ConstructBundle
from mrnax.schemas.evaluation_report import FTOAlert


class SequenceFTOGuard:
    """Checks candidate segments against patented clinical UTRs and BioNTech 51220548 poly(A) genus."""

    def __init__(self, registry_payload: dict[str, Any]) -> None:
        self.genus_rules = registry_payload.get("genus_grammar_rules", [])
        self.collision_rules = registry_payload.get("sequence_collision_rules", [])
        self._aligner = PairwiseAligner()
        self._aligner.mode = "local"
        self._aligner.match_score = 1.0
        self._aligner.mismatch_score = 0.0
        self._aligner.open_gap_score = -1.0
        self._aligner.extend_gap_score = -0.5

    @classmethod
    def from_json_path(cls, path: Path) -> SequenceFTOGuard:
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def scan_bundle(
        self, bundle: ConstructBundle, compliance_track: ComplianceProfile
    ) -> list[FTOAlert]:
        alerts: list[FTOAlert] = []
        for mol in bundle.molecules:
            # 1. Genus grammar rules (e.g. BioNTech family 51220548 >=20A + linker + >=20A)
            for rule in self.genus_rules:
                scope = rule["target_scope"]
                seg = mol.get_segment(scope)
                if seg is None:
                    continue
                sub = mol.sequence[seg.start : seg.end]
                min_len = int(rule.get("min_total_poly_a_length", 0))
                if len(sub) < min_len:
                    continue
                m = re.search(rule["regex_rna"], sub)
                if m:
                    blocked = (
                        bool(rule["block_in_commercial_clean"])
                        if compliance_track == "commercial_clean"
                        else bool(rule["block_in_research_max"])
                    )
                    alerts.append(
                        FTOAlert(
                            rule_id=rule["rule_id"],
                            patent_family_or_pub=f"Family {rule['patent_family_id']} ({', '.join(rule['granted_publications'])})",
                            matched_scope=scope,
                            transcript_coordinates=(seg.start + m.start(), seg.start + m.end()),
                            identity_or_pattern=f"genus_regex:{rule['regex_rna']}",
                            blocked=blocked,
                            summary=rule["summary"],
                        )
                    )

            # 2. Sequence substring / >=80% identity collision rules
            for rule in self.collision_rules:
                scope = rule["target_scope"]
                seg = mol.get_segment(scope)
                if seg is None:
                    continue
                sub = mol.sequence[seg.start : seg.end]
                ref = str(rule["sequence_rna"]).upper().replace("T", "U")
                thresh = float(rule.get("identity_threshold", 0.80))
                if not sub or not ref:
                    continue

                blocked = (
                    bool(rule.get("block_in_commercial_clean", True))
                    if compliance_track == "commercial_clean"
                    else bool(rule.get("block_in_research_max", False))
                )

                exact_idx = sub.find(ref)
                if exact_idx >= 0:
                    alerts.append(
                        FTOAlert(
                            rule_id=rule["rule_id"],
                            patent_family_or_pub=rule["patent_family_or_pub"],
                            matched_scope=scope,
                            transcript_coordinates=(seg.start + exact_idx, seg.start + exact_idx + len(ref)),
                            identity_or_pattern="exact_match:1.000",
                            blocked=blocked,
                            summary=rule["summary"],
                        )
                    )
                    continue

                if thresh < 1.0 and len(sub) >= int(len(ref) * thresh):
                    score = float(self._aligner.score(sub, ref))
                    identity = score / len(ref)
                    if identity >= thresh:
                        alerts.append(
                            FTOAlert(
                                rule_id=rule["rule_id"],
                                patent_family_or_pub=rule["patent_family_or_pub"],
                                matched_scope=scope,
                                transcript_coordinates=(seg.start, seg.end),
                                identity_or_pattern=f"local_identity:{identity:.3f}>={thresh:.2f}",
                                blocked=blocked,
                                summary=rule["summary"],
                            )
                        )
        return alerts

"""Scientific Rule Tribunal: dataset deduplication, isolated sandbox counter-evidence, and draft quarantine."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from mrnax.rules.models import ClaimCard, ModelInterpretationNote, RuleAuditEvent, RuleRecord
from mrnax.schemas.common import ComplianceProfile, LicenseTrack, RuleStatus, UridineChemistry

_RC_PIN_TO_COMPONENT_ID: dict[str, str] = {
    "rc-001": "viennarna_thermo",
    "rc-002": "structrfm_rna_foundation",
    "rc-003": "orthrus_mamba_rna_foundation",
    "rc-004": "eternafold_structure_head",
    "rc-005": "codontransformer_cds_baseline",
}


def _resolve_model_interpretation_catalog_track(
    interp: ModelInterpretationNote,
) -> tuple[str, LicenseTrack]:
    """All recorded identities must agree, including pins embedded in older display names."""
    from mrnax.engine.profiles import COMPONENT_CATALOG, resolve_default_cards_dir, resolve_tool_name_to_component_id

    identities: set[str] = set()
    if interp.source_component_id is not None:
        cid = interp.source_component_id.strip()
        if cid not in COMPONENT_CATALOG:
            raise ValueError(f"Unregistered model interpretation component '{cid}'")
        identities.add(cid)

    for field, value in (
        ("display", interp.source_model_name), ("model_id", interp.model_id),
        ("card", interp.card_pin), ("card", interp.replacement_card),
    ):
        if value is None:
            continue
        field_ids: set[str] = set()
        pins = re.findall(r"(?<![a-z0-9])rc-\d+(?![a-z0-9])", value.casefold())
        if field == "card" and not pins:
            raise ValueError(f"Unregistered replacement card '{value}'")
        for pin in pins:
            if pin not in _RC_PIN_TO_COMPONENT_ID:
                raise ValueError(f"Unregistered replacement card '{pin}'")
            field_ids.add(_RC_PIN_TO_COMPONENT_ID[pin])
        for filename in re.findall(r"(?i)\brc-[\w.-]+\.yaml\b", value):
            if not (resolve_default_cards_dir() / filename).is_file():
                raise ValueError(f"Unregistered replacement card '{filename}'")
        labels = {value, *re.findall(r"[\w.-]+", value), *re.findall(r"[a-zA-Z0-9]+", value)}
        for label in labels:
            # Legacy records append a version or assay tag with an underscore.
            for name in (label, label.partition("_")[0]):
                comp = resolve_tool_name_to_component_id(name)
                if comp in COMPONENT_CATALOG:
                    field_ids.add(comp)
                elif name.casefold() in {"surrogatev2", "surrogatev1"}:
                    field_ids.add("surrogate_v1_clean")
        if not field_ids and field == "model_id":
            raise ValueError(f"Unregistered model or card identity '{value}'")
        identities.update(field_ids)
    if len(identities) != 1:
        raise ValueError(f"Unknown or conflicting model interpretation '{interp.source_model_name}': {sorted(identities)}")
    cid, = identities
    return cid, COMPONENT_CATALOG[cid].license_track

_DATASET_PREFIX_RE = re.compile(r"^(GEO:|ENA:|SRA:|NCBI:)", re.IGNORECASE)


def normalize_dataset_id(raw_id: str) -> str:
    cleaned = _DATASET_PREFIX_RE.sub("", raw_id.strip()).upper()
    # Canonicalize verified NCBI GEO / BioProject / SRA aliases of the exact same sequencing study!
    if cleaned in {"GSE114002", "PRJNA454863", "SRP144485"}:
        return "GSE114002_PRJNA454863"
    if cleaned in {"GSE232927", "PRJNA974439"}:
        return "GSE232927"
    if cleaned in {"GSE176581", "PRJNA736738", "SRP323646"}:
        return "GSE176581"
    if cleaned in {"GSE173083", "PRJNA723762", "SRP315847"}:
        return "GSE173083"
    if cleaned in {"GSE139176", "PRJNA578697", "SRP226465"}:
        return "GSE139176"
    if cleaned.startswith("ELIFE_45396") or "WU2019" in cleaned:
        return "ELIFE_45396_WU2019"
    if "PMC8728028" in cleaned or "SIEGEL2022" in cleaned or "FASTUTR" in cleaned:
        return "PMC8728028_SIEGEL2022_FASTUTR"
    if (
        "PMID42805987" in cleaned
        or "PMC13620113" in cleaned
        or "POULIS" in cleaned
        or "S41467-026-77796-3" in cleaned
    ):
        return "PMID42805987_POULIS_RODNINA_2026"
    return cleaned


_ZERO_NULL_EFFECT_METRICS: frozenset[str] = frozenset(
    {
        "spearman_rho",
        "spearman_rho_and_causal_deletion_stabilization_68.1pct_74.8pct",
        "spearman_rho_gc_tail15_vs_rl_mean",
        "spearman_rho_gc_tail15_vs_te",
        "spearman_rho_gc_tail15_vs_top2.5pct_gfp_mpra",
        "pearson_r",
        "log2_fold_change",
        "cohen_d",
        "delta_mean",
    }
)
_UNIT_RATIO_EFFECT_METRICS: frozenset[str] = frozenset(
    {
        "fold_increase_m1psi_vs_unmodified_sfu_per_million",
        "fold_recoding_increase_at_slippery_u_rich_motifs",
        "fold_change",
        "odds_ratio",
        "hazard_ratio",
    }
)


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
        if c.status != "verified_observation" or c.evidence_layer != "raw_observation":
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
        self.is_isolated_sandbox = (
            json.loads(backing_path.read_bytes()).get("is_isolated_sandbox") is True
            if backing_path.is_file() else is_isolated_sandbox
        )
        self._draft_inbox: list[ClaimCard] = []

    @staticmethod
    def validate_rules_integrity(
        rules: list[RuleRecord],
        *,
        track: ComplianceProfile,
        observation_ledger: dict[str, dict] | None = None,
    ) -> None:
        from mrnax.engine.profiles import audit_component_commercial_clean

        seen_ids: set[str] = set()
        cleared_components: set[str] = set()
        seen_global_claims: set[str] = set()
        for r in rules:
            if r.rule_id in seen_ids:
                raise ValueError(f"Duplicate rule_id '{r.rule_id}' in RuleRegistry")
            seen_ids.add(r.rule_id)
            seen_rule_ds: set[tuple[str, str]] = set()
            for obs in r.linked_observations:
                norm_d = normalize_dataset_id(obs.underlying_dataset_id)
                key = (obs.claim_id, norm_d)
                if key in seen_rule_ds or obs.claim_id in seen_global_claims:
                    raise ValueError(
                        f"Duplicate observation link '{obs.claim_id}' ({norm_d}) in RuleRegistry"
                    )
                seen_rule_ds.add(key)
                seen_global_claims.add(obs.claim_id)
            if track == "commercial_clean":
                if r.allowed_track != "commercial_clean":
                    raise ValueError(
                        f"Rule '{r.rule_id}' has allowed_track='{r.allowed_track}' inside commercial_clean registry"
                    )
                for obs in r.linked_observations:
                    if obs.evidence_layer != "raw_observation":
                        raise ValueError(
                            f"Rule '{r.rule_id}' linked_observations must only contain raw_observation (move model outputs to model_interpretations)"
                        )
                for interp in r.model_interpretations:
                    resolved_comp_id, expected_track = _resolve_model_interpretation_catalog_track(interp)
                    name_low = interp.source_model_name.lower()
                    if interp.source_model_license_track != expected_track:
                        raise ValueError(
                            f"Rule '{r.rule_id}' model_interpretation '{interp.source_model_name}' "
                            f"(component={resolved_comp_id}) declares track='{interp.source_model_license_track}' "
                            f"but authoritative catalog track is '{expected_track}'"
                        )
                    if interp.source_model_license_track == "commercial_clean":
                        if resolved_comp_id not in cleared_components:
                            issue = audit_component_commercial_clean(resolved_comp_id)
                            if issue is not None:
                                raise ValueError(f"Rule '{r.rule_id}' model interpretation failed clearance: {issue}")
                            cleared_components.add(resolved_comp_id)
                        if "clearance_pending" in name_low or "research_only" in name_low:
                            raise ValueError(
                                f"Rule '{r.rule_id}' model_interpretation '{interp.source_model_name}' declares "
                                "clearance_pending/research_only in source_model_name but stamps source_model_license_track='commercial_clean'"
                            )
                    elif (
                        r.status == "conditional_rule"
                        and interp.source_model_license_track == "commercial_clearance_pending"
                        and expected_track == "commercial_clearance_pending"
                    ):
                        # Held candidate RC-* model notes on conditional_rules are quarantined at runtime by get_active_rules
                        pass
                    else:
                        raise ValueError(
                            f"Rule '{r.rule_id}' contains non-commercial_clean model_interpretation '{interp.source_model_name}'"
                        )
            if r.status == "verified_rule":
                raw_ds = {
                    normalize_dataset_id(obs.underlying_dataset_id)
                    for obs in r.linked_observations
                    if obs.evidence_layer == "raw_observation"
                }
                if len(raw_ds) < 2:
                    raise ValueError(
                        f"verified_rule '{r.rule_id}' requires >=2 distinct underlying raw_observation datasets; found {sorted(raw_ds)}"
                    )
                if not isinstance(observation_ledger, dict):
                    raise ValueError(f"verified_rule '{r.rule_id}' requires an observation ledger")
                if r.expected_effect_direction not in {"positive", "negative"}:
                    raise ValueError(
                        f"verified_rule '{r.rule_id}' must declare explicit expected_effect_direction ('positive' or 'negative')"
                    )
                lineages: set[str] = set()
                directions: set[str] = set()
                for obs in r.linked_observations:
                    rec = observation_ledger.get(obs.claim_id)
                    if not isinstance(rec, dict):
                        raise ValueError(
                            f"verified_rule '{r.rule_id}' references unsubstantiated claim_id '{obs.claim_id}' missing from rule_observation_ledger_v1.json"
                        )
                    if str(rec.get("underlying_dataset_id", "")).strip() != obs.underlying_dataset_id.strip():
                        raise ValueError(
                            f"verified_rule '{r.rule_id}' claim '{obs.claim_id}' dataset mismatch against observation ledger"
                        )
                    p_dir = str(rec.get("preregistered_direction", "")).strip()
                    o_dir = str(rec.get("observed_direction", "")).strip()
                    if not p_dir or p_dir != o_dir or p_dir != r.expected_effect_direction:
                        raise ValueError(
                            f"verified_rule '{r.rule_id}' claim '{obs.claim_id}' failed preregistered/rule direction check ({p_dir!r} vs {o_dir!r} vs rule {r.expected_effect_direction!r})"
                        )
                    effect = rec.get("measured_effect")
                    limit = rec.get("preregistered_threshold")
                    metric_name = str(rec.get("effect_metric", "")).strip().lower()
                    if metric_name in _UNIT_RATIO_EFFECT_METRICS:
                        is_ratio_scale = True
                    elif metric_name in _ZERO_NULL_EFFECT_METRICS:
                        is_ratio_scale = False
                    else:
                        raise ValueError(
                            f"verified_rule '{r.rule_id}' claim '{obs.claim_id}' uses unregistered effect_metric {metric_name!r}"
                        )
                    rule_text_low = f"{r.readout_name_exact} {r.statement}".lower()
                    if re.search(r"\b(spearman|rho)\b", rule_text_low) and not metric_name.startswith("spearman_rho"):
                        raise ValueError(
                            f"verified_rule '{r.rule_id}' claim '{obs.claim_id}' metric {metric_name!r} contradicts Spearman/rho correlation rule statement"
                        )
                    if re.search(r"\bpearson\b", rule_text_low) and not metric_name.startswith("pearson_r"):
                        raise ValueError(
                            f"verified_rule '{r.rule_id}' claim '{obs.claim_id}' metric {metric_name!r} contradicts Pearson correlation rule statement"
                        )
                    if is_ratio_scale and any(k in rule_text_low for k in ("spearman", "pearson", "rho=")):
                        raise ValueError(
                            f"verified_rule '{r.rule_id}' claim '{obs.claim_id}' ratio metric {metric_name!r} contradicts correlation rule statement"
                        )
                    if (
                        rec.get("rule_id") != r.rule_id
                        or rec.get("claim_id") != obs.claim_id
                        or p_dir not in {"positive", "negative"}
                        or type(effect) not in (int, float)
                        or type(limit) not in (int, float)
                        or not math.isfinite(effect)
                        or not math.isfinite(limit)
                    ):
                        raise ValueError(f"verified_rule '{r.rule_id}' claim '{obs.claim_id}' failed quantitative threshold")
                    if is_ratio_scale:
                        valid_quant = (
                            (p_dir == "positive" and limit >= 1.0 and effect > 1.0 and effect > limit)
                            or (p_dir == "negative" and 0.0 < limit <= 1.0 and 0.0 < effect < 1.0 and effect < limit)
                        )
                    else:
                        valid_quant = (
                            (p_dir == "positive" and limit >= 0.0 and effect > 0.0 and effect > limit)
                            or (p_dir == "negative" and limit <= 0.0 and effect < 0.0 and effect < limit)
                        )
                    if not valid_quant:
                        raise ValueError(
                            f"verified_rule '{r.rule_id}' claim '{obs.claim_id}' failed quantitative threshold "
                            f"(direction={p_dir!r}, metric={metric_name!r}, effect={effect!r}, threshold={limit!r})"
                        )
                    lineage = rec.get("author_lineage_group")
                    if not isinstance(lineage, str) or not lineage.strip():
                        raise ValueError(f"verified_rule '{r.rule_id}' claim '{obs.claim_id}' lacks a lineage")
                    lineages.add(lineage.strip().casefold())
                    directions.add(o_dir)
                if len(lineages) < 2:
                    raise ValueError(
                        f"verified_rule '{r.rule_id}' requires >=2 independent author_lineage_group entries in ledger; found {sorted(lineages)}"
                    )
                if len(directions) != 1:
                    raise ValueError(
                        f"verified_rule '{r.rule_id}' linked claims disagree on effect direction: {sorted(directions)}"
                    )

    @classmethod
    def create_with_initial_rules(
        cls, backing_path: Path, *, track: ComplianceProfile, rules: list[RuleRecord]
    ) -> RuleRegistry:
        registry = cls(backing_path, track=track)
        ledger = registry._ledger()
        cls.validate_rules_integrity(rules, track=track, observation_ledger=ledger["claims"] if ledger else None)
        backing_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "registry_track": track,
            "is_isolated_sandbox": False,
            "rules": [r.model_dump(mode="json") for r in rules],
            "draft_claim_inbox": [],
        }
        backing_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return cls(backing_path, track=track, is_isolated_sandbox=False)

    @property
    def _ledger_path(self) -> Path:
        if self.is_isolated_sandbox:
            return self.backing_path.with_suffix(".observations.json")
        return self.backing_path.with_name("rule_observation_ledger_v1.json")

    def _ledger(self) -> dict | None:
        if not self._ledger_path.is_file():
            return None
        ledger = json.loads(self._ledger_path.read_bytes())
        if not isinstance(ledger, dict) or not isinstance(ledger.get("claims"), dict):
            raise ValueError("Observation ledger must contain a claims mapping")
        return ledger

    def _read_snapshot(self) -> tuple[dict, dict | None]:
        doc = json.loads(self.backing_path.read_bytes())
        ledger = self._ledger()
        self.validate_rules_integrity(
            [RuleRecord.model_validate(r) for r in doc.get("rules", [])],
            track=self.track,
            observation_ledger=ledger["claims"] if ledger else None,
        )
        return doc, ledger

    def _load_doc(self) -> dict:
        doc, _ = self._read_snapshot()
        return doc

    def _save_doc(self, doc: dict) -> None:
        self.backing_path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def snapshot_version(self) -> str:
        doc, ledger = self._read_snapshot()
        canonical = json.dumps({"rules": doc["rules"], "ledger": ledger}, sort_keys=True, separators=(",", ":"))
        return f"rules_{self.track}@{hashlib.sha256(canonical.encode()).hexdigest()}"

    def fork_isolated_copy(self, sandbox_path: Path) -> RuleRegistry:
        """Copy the validated rules and the exact ledger used to validate them."""
        doc, ledger = self._read_snapshot()
        if sandbox_path.exists() or sandbox_path.with_suffix(".observations.json").exists():
            raise FileExistsError("Rule sandbox requires unused rule and observation paths")
        sandbox = RuleRegistry(sandbox_path, track=self.track, is_isolated_sandbox=True)
        doc["is_isolated_sandbox"] = True
        sandbox_path.parent.mkdir(parents=True, exist_ok=True)
        if ledger is not None:
            with sandbox._ledger_path.open("x", encoding="utf-8") as f:
                json.dump(ledger, f, sort_keys=True)
        with sandbox_path.open("x", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, sort_keys=True)
            f.write("\n")
        return sandbox

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
        doc, ledger = self._read_snapshot()
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
        self.validate_rules_integrity(
            [RuleRecord.model_validate(r) for r in new_rules],
            track=self.track,
            observation_ledger=ledger["claims"] if ledger else None,
        )
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
            if self.track == "commercial_clean":
                if r.allowed_track != "commercial_clean":
                    continue
                clean_interps = [
                    mi for mi in r.model_interpretations if mi.source_model_license_track == "commercial_clean"
                ]
                if not r.linked_observations and not clean_interps:
                    # Purely clearance_pending model-benchmark rule stays inactive until its RC-* card clears!
                    continue
                if len(clean_interps) != len(r.model_interpretations):
                    r = r.model_copy_validated(model_interpretations=clean_interps)
            if chemistry not in r.applicable_chemistries:
                continue
            if context not in r.applicable_contexts and "general_mammalian_ivt" not in r.applicable_contexts:
                continue
            active.append(r)
        return active


def load_scientific_rules_v2(seed_dir: Path | None = None) -> dict:
    """Load the 38-rule quantitative rulebook (`rules_scientific_v2.json`) from `seed_dir`."""
    from mrnax.engine.profiles import resolve_default_seed_dir

    eff_seed = Path(seed_dir) if seed_dir is not None else resolve_default_seed_dir()
    v2_path = eff_seed / "rules_scientific_v2.json"
    if not v2_path.is_file():
        raise FileNotFoundError(f"Scientific rulebook v2 not found at '{v2_path}'")
    return json.loads(v2_path.read_text(encoding="utf-8"))


def query_scientific_rules(
    *,
    seed_dir: Path | None = None,
    modality: str | None = None,
    region: str | None = None,
    evidence_grade: str | None = None,
    query: str | None = None,
    include_debunked_myths: bool = True,
) -> list[dict]:
    """Filter `rules_scientific_v2.json` across modality, region, evidence_grade, and free-text query."""
    doc = load_scientific_rules_v2(seed_dir)
    out: list[dict] = []
    mod_norm = modality.strip().lower() if modality else None
    reg_norm = region.strip().lower() if region else None
    grade_norm = evidence_grade.strip().lower() if evidence_grade else None
    q_norm = query.strip().lower() if query else None

    for rule in doc.get("rules", []):
        r_grade = str(rule.get("evidence_grade", "")).lower()
        if not include_debunked_myths and r_grade == "debunked_myth":
            continue
        if grade_norm and r_grade != grade_norm:
            continue
        if mod_norm:
            mods = rule.get("modality", [])
            mod_list = [str(m).lower() for m in mods] if isinstance(mods, list) else [str(mods).lower()]
            if mod_norm not in mod_list:
                continue
        if reg_norm and str(rule.get("region", "")).lower() != reg_norm:
            continue
        if q_norm:
            haystack = " ".join(
                str(rule.get(k, ""))
                for k in ("rule_id", "title_zh", "title_en", "effect_size_summary", "machine_check_key")
            ).lower()
            if q_norm not in haystack:
                continue
        out.append(rule)
    return out

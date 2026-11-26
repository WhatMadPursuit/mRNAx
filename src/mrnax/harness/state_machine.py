"""Deterministic Project State Machine: Intent Lineage, Lock Inheritance, Conflict Gate & Wet-Lab Loop."""

from __future__ import annotations

import json
import math
import re
import uuid
from pathlib import Path
from typing import Any

from mrnax.constraints.registry import ScopedConstraintEngine, default_m1_constraints
from mrnax.engine.pareto_director import run_design_pipeline
from mrnax.harness.sandbox import UntrustedExternalText
from mrnax.schemas.common import ComplianceProfile, SegmentScope, UridineChemistry
from mrnax.schemas.construct_bundle import ConstructBundle, ProteinTargetSpec, SegmentSpan
from mrnax.schemas.design_intent import (
    CellExpressionSpec,
    ConstraintSpec,
    DesignIntent,
    LockedRegionSpec,
    ObjectiveSpec,
    RuntimePolicySpec,
)
from mrnax.schemas.evaluation_report import EvaluationReport
from mrnax.schemas.experiment_record import ExperimentRecord, FeedbackDecisionRecord
from mrnax.schemas.run_manifest import RunManifest


def _detect_chemistry(utterance: str, default: UridineChemistry = "m1psi") -> UridineChemistry:
    low = utterance.lower()
    if any(k in low for k in ("m1psi", "m¹ψ", "n1-methylpseudouridine", "1-甲基假尿苷", "n1-甲基假尿苷")):
        return "m1psi"
    if any(k in low for k in ("unmodified", "未修饰", "canonical u")):
        return "unmodified_u"
    if any(k in low for k in ("pseudouridine", "假尿苷", "psi")):
        return "pseudouridine"
    return default


def _detect_enzymes(utterance: str) -> list[str] | None:
    found: list[str] = []
    for enz in ("BspQI", "SapI", "BsaI", "BsmBI", "BbsI", "EcoRI", "XbaI"):
        if re.search(rf"\b{enz}\b", utterance, re.IGNORECASE):
            found.append(enz)
    return found if found else None


class ProjectHarnessSession:
    """Stateful project service persisting turns, DesignIntent versions, ConstructBundles, and feedback."""

    def __init__(
        self,
        *,
        project_id: str,
        store_dir: Path,
        seed_dir: Path,
    ) -> None:
        self.project_id = project_id
        self.store_dir = store_dir
        self.seed_dir = seed_dir
        self.store_dir.mkdir(parents=True, exist_ok=True)
        (self.store_dir / "intents").mkdir(exist_ok=True)
        (self.store_dir / "bundles").mkdir(exist_ok=True)
        (self.store_dir / "reports").mkdir(exist_ok=True)
        (self.store_dir / "experiments").mkdir(exist_ok=True)

        self._intents: list[DesignIntent] = []
        self._bundles: dict[str, ConstructBundle] = {}
        self._reports: dict[str, EvaluationReport] = {}
        self._manifests: dict[str, RunManifest] = {}
        self._external_notes: list[UntrustedExternalText] = []
        self.active_reference_bundle_id: str | None = None
        self._load_existing()

    def _load_existing(self) -> None:
        for p in sorted((self.store_dir / "intents").glob("intent_v*.json")):
            self._intents.append(DesignIntent.model_validate_json(p.read_text(encoding="utf-8")))
        for p in sorted((self.store_dir / "bundles").glob("*.json")):
            b = ConstructBundle.model_validate_json(p.read_text(encoding="utf-8"))
            self._bundles[b.bundle_id] = b
        state_meta = self.store_dir / "project_state.json"
        if state_meta.is_file():
            meta = json.loads(state_meta.read_text(encoding="utf-8"))
            self.active_reference_bundle_id = meta.get("active_reference_bundle_id")

    def _save_state_meta(self) -> None:
        (self.store_dir / "project_state.json").write_text(
            json.dumps(
                {
                    "project_id": self.project_id,
                    "active_reference_bundle_id": self.active_reference_bundle_id,
                    "intent_count": len(self._intents),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def _persist_intent(self, intent: DesignIntent) -> DesignIntent:
        self._intents.append(intent)
        out = self.store_dir / "intents" / f"intent_v{intent.intent_version:03d}_{intent.intent_id}.json"
        out.write_text(json.dumps(intent.model_dump(mode="json"), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self._save_state_meta()
        return intent

    def register_bundle(self, bundle: ConstructBundle, *, set_as_reference: bool = False) -> None:
        self._bundles[bundle.bundle_id] = bundle
        out = self.store_dir / "bundles" / f"{bundle.bundle_id}.json"
        out.write_text(json.dumps(bundle.model_dump(mode="json"), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if set_as_reference:
            self.active_reference_bundle_id = bundle.bundle_id
        self._save_state_meta()

    def get_bundle(self, bundle_id: str) -> ConstructBundle:
        return self._bundles[bundle_id]

    def latest_intent(self) -> DesignIntent:
        if not self._intents:
            raise RuntimeError("No DesignIntent exists in this session yet")
        return self._intents[-1]

    def attach_external_evidence_readonly(self, sandboxed: UntrustedExternalText) -> None:
        """Store quarantined external text; strictly forbidden from modifying any DesignIntent or lock."""
        if not sandboxed.untrusted_readonly:
            raise ValueError("External text must be marked untrusted_readonly=True")
        self._external_notes.append(sandboxed)
        log_path = self.store_dir / "quarantined_external_evidence.jsonl"
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(sandboxed.model_dump(mode="json")) + "\n")

    def create_initial_intent(
        self,
        *,
        turn_id: str,
        user_utterance: str,
        protein_sequence: str,
        accession_or_source: str = "user_supplied",
        entry_version: str | None = None,
        is_secreted_without_sp: bool = False,
        reference_bundle: ConstructBundle | None = None,
        compliance_track: ComplianceProfile = "research_max",
        uridine_chemistry: UridineChemistry | None = None,
        locked_regions: dict[SegmentScope, str] | None = None,
        restriction_enzymes: list[str] | None = None,
        detarget_tissues: list[str] | None = None,
        protected_tissues: list[str] | None = None,
        prioritize_late_signal: bool = False,
        conflicting_single_pick: bool = False,
        random_seed: int = 42,
    ) -> DesignIntent:
        prot_clean = protein_sequence.strip().upper()
        chem = uridine_chemistry or _detect_chemistry(user_utterance)
        enz = restriction_enzymes if restriction_enzymes is not None else (_detect_enzymes(user_utterance) or ["BspQI"])

        blocking_q: list[str] = []
        prot_status = "confirmed"
        intent_status = "ready"

        if not prot_clean.startswith("M") or is_secreted_without_sp:
            prot_status = "needs_protein_confirmation"
            intent_status = "needs_protein_confirmation"
            blocking_q.append(
                "Provided protein sequence lacks an N-terminal initiator Methionine ('M') or signal peptide. "
                "Please confirm whether to prepend initiator 'M', attach a specific signal peptide, or evaluate as-is."
            )
        elif conflicting_single_pick:
            intent_status = "needs_conflict_clarification"
            blocking_q.append(
                "Your objectives conflict for a single construct (maximal GC/stem stability vs completely unfolded CDS). "
                "Which objective has higher priority, or should mRNAx retain multiple Pareto trade-off candidates?"
            )

        ref_id: str | None = None
        if reference_bundle is not None:
            self.register_bundle(reference_bundle, set_as_reference=True)
            ref_id = reference_bundle.bundle_id

        locks: dict[str, LockedRegionSpec] = {
            "protein_sequence": LockedRegionSpec(
                scope="protein_sequence",
                locked_value=prot_clean,
                reason="100% amino-acid identity lock",
                locked_at_turn_id=turn_id,
            )
        }
        for scope_k, seq_v in (locked_regions or {}).items():
            locks[scope_k] = LockedRegionSpec(
                scope=scope_k,
                locked_value=seq_v.upper().replace("T", "U"),
                reason=f"User locked {scope_k}",
                locked_at_turn_id=turn_id,
            )

        mutable: list[SegmentScope] = [
            s for s in ("five_prime_utr", "cds", "three_prime_utr") if s not in locks
        ]

        objs = [
            ObjectiveSpec(
                requirement_id="obj_5p_initiation_openness",
                metric_id="kozak_unpaired_prob",
                direction="maximize",
                priority_rank=1,
                evidence_status="thermodynamic_proxy",
                source_utterance=user_utterance,
            ),
            ObjectiveSpec(
                requirement_id="obj_cds_local_stability",
                metric_id="cds_mean_window_mfe",
                direction="minimize",
                priority_rank=2,
                evidence_status="thermodynamic_proxy",
                source_utterance=user_utterance,
            ),
        ]
        if prioritize_late_signal or any(w in user_utterance for w in ("后期信号", "late signal", "48h", "half-life")):
            objs.append(
                ObjectiveSpec(
                    requirement_id="obj_late_protein_persistence",
                    metric_id="late_timepoint_protein_auc",
                    direction="maximize",
                    priority_rank=1,
                    evidence_status="unverified_target_for_wetlab",
                    source_utterance=user_utterance,
                )
            )

        tracks = (
            ["minimal_change_vs_r", "exploratory_pareto"]
            if reference_bundle is not None
            else ["computational_reference", "exploratory_pareto"]
        )

        cell_ctrl = (
            CellExpressionSpec(
                detarget_tissues=detarget_tissues,
                protected_tissues=protected_tissues or [],
            )
            if detarget_tissues
            else None
        )

        intent = DesignIntent(
            intent_id=f"int_{uuid.uuid4().hex[:8]}",
            project_id=self.project_id,
            intent_version=len(self._intents) + 1,
            parent_intent_id=self._intents[-1].intent_id if self._intents else None,
            created_from_turn_ids=[turn_id],
            entry_mode="protein_with_reference_r" if reference_bundle is not None else "protein_only",
            compliance_track=compliance_track,
            uridine_chemistry=chem,
            status=intent_status,  # type: ignore[arg-type]
            blocking_questions=blocking_q,
            protein_targets=[
                ProteinTargetSpec(
                    protein_sequence=prot_clean,
                    transl_table=1,
                    accession_or_source=accession_or_source,
                    entry_version=entry_version,
                    confirmation_status=prot_status,  # type: ignore[arg-type]
                )
            ],
            reference_bundle_id=ref_id,
            locked_scopes=locks,
            mutable_scopes=mutable,
            constraints=default_m1_constraints(restriction_enzymes=enz),
            objectives=objs,
            cell_expression_control=cell_ctrl,
            portfolio_tracks=tracks,  # type: ignore[arg-type]
            runtime_policy=RuntimePolicySpec(random_seed=random_seed),
        )
        return self._persist_intent(intent)

    def extract_structured_intent_deterministic(
        self,
        *,
        turn_id: str,
        user_utterance: str,
        protein_sequence: str,
        accession_or_source: str = "GenBank:U55762.1",
        reference_bundle: ConstructBundle | None = None,
    ) -> DesignIntent:
        """Deterministic synonym-invariant parser mapping equivalent phrasing to identical core constraints."""
        locked_regions: dict[SegmentScope, str] = {}
        m_utr5 = re.search(
            r"(?:5'\s*utr|five_prime_utr).*?\b([ACGTU]{8,120})\b",
            user_utterance,
            re.IGNORECASE,
        )
        if m_utr5:
            locked_regions["five_prime_utr"] = m_utr5.group(1).upper().replace("T", "U")

        detarget: list[str] | None = None
        prot_tiss: list[str] | None = None
        if any(k in user_utterance.lower() for k in ("mir-122", "肝脏去靶向", "detarget liver")):
            detarget = ["liver_hepatocyte"]
            prot_tiss = ["skeletal_muscle"]

        return self.create_initial_intent(
            turn_id=turn_id,
            user_utterance=user_utterance,
            protein_sequence=protein_sequence,
            accession_or_source=accession_or_source,
            reference_bundle=reference_bundle,
            locked_regions=locked_regions,
            detarget_tissues=detarget,
            protected_tissues=prot_tiss,
        )

    def apply_incremental_intent_update(
        self,
        *,
        turn_id: str,
        user_utterance: str,
        updated_restriction_enzymes: list[str] | None = None,
        additional_locked_scopes: dict[SegmentScope, str] | None = None,
        unlocked_scopes: list[SegmentScope] | None = None,
        updated_chemistry: UridineChemistry | None = None,
        updated_compliance_track: ComplianceProfile | None = None,
        detarget_tissues: list[str] | None = None,
        protected_tissues: list[str] | None = None,
    ) -> DesignIntent:
        """Single-item incremental edit inheriting 100% of untouched locks, constraints, and targets."""
        prev = self.latest_intent()
        new_locks = dict(prev.locked_scopes)
        for un_s in unlocked_scopes or []:
            new_locks.pop(un_s, None)
        for s_k, s_val in (additional_locked_scopes or {}).items():
            new_locks[s_k] = LockedRegionSpec(
                scope=s_k,
                locked_value=s_val.upper().replace("T", "U"),
                reason=f"Locked at {turn_id}: {user_utterance}",
                locked_at_turn_id=turn_id,
            )

        mutable: list[SegmentScope] = [
            s for s in ("five_prime_utr", "cds", "three_prime_utr") if s not in new_locks
        ]

        new_constraints: list[ConstraintSpec] = []
        replaced_enz = False
        for c in prev.constraints:
            if c.constraint_kind == "type_iis_restriction_free" and updated_restriction_enzymes is not None:
                new_constraints.append(
                    c.model_copy_validated(
                        source_type="user_explicit",
                        source_locator=turn_id,
                        params={"enzymes": updated_restriction_enzymes, "forbidden_motifs": c.params.get("forbidden_motifs", [])},
                    )
                )
                replaced_enz = True
            else:
                new_constraints.append(c)
        if updated_restriction_enzymes is not None and not replaced_enz:
            new_constraints.append(
                ConstraintSpec(
                    requirement_id="gate_type_iis_restriction",
                    constraint_kind="type_iis_restriction_free",
                    target_scopes=frozenset({"full_transcript"}),
                    severity="hard_gate",
                    source_type="user_explicit",
                    source_locator=turn_id,
                    params={"enzymes": updated_restriction_enzymes, "forbidden_motifs": []},
                )
            )

        new_cell = prev.cell_expression_control
        if detarget_tissues is not None:
            new_cell = CellExpressionSpec(
                detarget_tissues=detarget_tissues,
                protected_tissues=protected_tissues or (prev.cell_expression_control.protected_tissues if prev.cell_expression_control else []),
            )

        updated = prev.model_copy_validated(
            intent_id=f"int_{uuid.uuid4().hex[:8]}",
            intent_version=prev.intent_version + 1,
            parent_intent_id=prev.intent_id,
            created_from_turn_ids=[*prev.created_from_turn_ids, turn_id],
            reference_bundle_id=self.active_reference_bundle_id or prev.reference_bundle_id,
            uridine_chemistry=updated_chemistry or prev.uridine_chemistry,
            compliance_track=updated_compliance_track or prev.compliance_track,
            locked_scopes=new_locks,
            mutable_scopes=mutable,
            constraints=new_constraints,
            cell_expression_control=new_cell,
        )
        return self._persist_intent(updated)

    def resolve_conflict_with_pareto_tracks(self, *, turn_id: str, user_utterance: str) -> DesignIntent:
        prev = self.latest_intent()
        resolved = prev.model_copy_validated(
            intent_id=f"int_{uuid.uuid4().hex[:8]}",
            intent_version=prev.intent_version + 1,
            parent_intent_id=prev.intent_id,
            created_from_turn_ids=[*prev.created_from_turn_ids, turn_id],
            status="ready",
            blocking_questions=[],
            portfolio_tracks=["computational_reference", "exploratory_pareto"],
        )
        return self._persist_intent(resolved)

    def execute_design(self, intent: DesignIntent | None = None) -> list[ConstructBundle]:
        active_intent = intent or self.latest_intent()
        ref = self._bundles.get(active_intent.reference_bundle_id) if active_intent.reference_bundle_id else None
        bundles, report, manifest = run_design_pipeline(
            active_intent, reference_bundle=ref, seed_dir=self.seed_dir
        )
        for idx, b in enumerate(bundles):
            self.register_bundle(b, set_as_reference=(idx == 0 and self.active_reference_bundle_id is None))
        self._reports[report.report_id] = report
        self._manifests[manifest.execution_id] = manifest
        (self.store_dir / "reports" / f"{report.report_id}.json").write_text(
            json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return bundles

    def build_multi_segment_variant_for_comparison(self, base_bundle: ConstructBundle) -> ConstructBundle:
        """Helper producing a distinct 5'UTR + synonymous CDS variant for multi-segment attribution tests."""
        utr_reg = json.loads((self.seed_dir / "utr_components_v1.json").read_text(encoding="utf-8"))
        mol = base_bundle.molecules[0]
        cur_u5 = mol.get_segment_sequence("five_prime_utr")
        alt_u5 = next(
            u["sequence"]
            for u in utr_reg["five_prime_utrs"]
            if not u["has_uaug"] and u["sequence"] != cur_u5
        )
        # Flip one synonymous codon in CDS
        cds_seq = mol.get_segment_sequence("cds")
        alt_cds = cds_seq
        for i in range(3, len(cds_seq) - 3, 3):
            c = cds_seq[i : i + 3]
            if c == "GGC":
                alt_cds = cds_seq[:i] + "GGA" + cds_seq[i + 3 :]
                break
            if c == "CUG":
                alt_cds = cds_seq[:i] + "CUC" + cds_seq[i + 3 :]
                break

        stop_s = mol.get_segment_sequence("stop_cassette")
        u3_s = mol.get_segment_sequence("three_prime_utr")
        pa_s = mol.get_segment_sequence("poly_a")
        c1 = len(alt_u5)
        c2 = c1 + len(alt_cds)
        c3 = c2 + len(stop_s)
        c4 = c3 + len(u3_s)
        c5 = c4 + len(pa_s)
        new_mol = mol.model_copy_validated(
            sequence=alt_u5 + alt_cds + stop_s + u3_s + pa_s,
            segments=[
                SegmentSpan(
                    segment_type="five_prime_utr",
                    start=0,
                    end=c1,
                    component_id="utr5_alt",
                    origin_source="NCBI_RefSeq_alt",
                    license_track="commercial_clean",
                    action_taken="selected_from_library",
                ),
                SegmentSpan(
                    segment_type="cds",
                    start=c1,
                    end=c2,
                    component_id=None,
                    origin_source="mrnax:synonymous_alt",
                    license_track="commercial_clean",
                    action_taken="synonymous_mutated",
                ),
                SegmentSpan(
                    segment_type="stop_cassette",
                    start=c2,
                    end=c3,
                    component_id="stop",
                    origin_source="mrnax:standard",
                    license_track="commercial_clean",
                    action_taken="locked_verbatim",
                ),
                SegmentSpan(
                    segment_type="three_prime_utr",
                    start=c3,
                    end=c4,
                    component_id="utr3_same",
                    origin_source="NCBI_RefSeq",
                    license_track="commercial_clean",
                    action_taken="locked_verbatim",
                ),
                SegmentSpan(
                    segment_type="poly_a",
                    start=c4,
                    end=c5,
                    component_id="polya_same",
                    origin_source="mrnax:homopolymer_tail",
                    license_track="commercial_clean",
                    action_taken="locked_verbatim",
                ),
            ],
            annotations=[],
        )
        alt_bundle = ConstructBundle.create(
            bundle_id=f"{base_bundle.bundle_id}_multi_alt",
            parent_bundle_id=base_bundle.bundle_id,
            derivation_action="de_novo_design",
            target_stoichiometry=base_bundle.target_stoichiometry,
            molecules=[new_mol],
        )
        self.register_bundle(alt_bundle)
        return alt_bundle

    def _spawn_verbatim_disambiguation_chimera(
        self, ref_a: ConstructBundle, winner_b: ConstructBundle
    ) -> ConstructBundle:
        """Create a zero-junction-edit pure verbatim chimera (priority: five_prime_utr > three_prime_utr > cds)."""
        mol_a = ref_a.molecules[0]
        mol_b = winner_b.molecules[0]
        constraint_engine = ScopedConstraintEngine.from_seed_dir(self.seed_dir)
        active_constraints = self.latest_intent().constraints
        track = self.latest_intent().compliance_track

        for swap_scope in ("five_prime_utr", "three_prime_utr", "cds"):
            seg_a = mol_a.get_segment_sequence(swap_scope)
            seg_b = mol_b.get_segment_sequence(swap_scope)
            if seg_a == seg_b:
                continue

            parts: dict[SegmentScope, str] = {
                s: (seg_b if s == swap_scope else mol_a.get_segment_sequence(s))
                for s in ("five_prime_utr", "cds", "stop_cassette", "three_prime_utr", "poly_a")
            }
            cursor = 0
            new_segs: list[SegmentSpan] = []
            for s in ("five_prime_utr", "cds", "stop_cassette", "three_prime_utr", "poly_a"):
                sub = parts[s]
                orig_seg = mol_b.get_segment(s) if s == swap_scope else mol_a.get_segment(s)
                assert orig_seg is not None
                new_segs.append(
                    SegmentSpan(
                        segment_type=s,
                        start=cursor,
                        end=cursor + len(sub),
                        component_id=orig_seg.component_id,
                        origin_source=f"{'winner_B' if s == swap_scope else 'ref_A'}:{orig_seg.origin_source}",
                        license_track=orig_seg.license_track,
                        action_taken="chimera_spliced" if s == swap_scope else "locked_verbatim",
                    )
                )
                cursor += len(sub)

            chim_mol = mol_a.model_copy_validated(
                sequence="".join(parts[s] for s in ("five_prime_utr", "cds", "stop_cassette", "three_prime_utr", "poly_a")),
                segments=new_segs,
                annotations=[],
            )
            chimera = ConstructBundle.create(
                bundle_id=f"bnd_chimera_{swap_scope[:4]}_{uuid.uuid4().hex[:6]}",
                parent_bundle_id=winner_b.bundle_id,
                derivation_action="disambiguation_chimera",
                target_stoichiometry=ref_a.target_stoichiometry,
                molecules=[chim_mol],
            )
            check = constraint_engine.evaluate_bundle(chimera, active_constraints, compliance_track=track)
            if check.passed_all_hard_gates:
                self.register_bundle(chimera)
                return chimera

        raise RuntimeError("Could not build valid disambiguation chimera across differing segments")

    def ingest_experiment_feedback(
        self, record: ExperimentRecord, *, batch_comparable: bool = True
    ) -> ExperimentRecord:
        """Deterministic 4-branch wet-lab feedback handler (Joint Ruling #6)."""
        out_path = self.store_dir / "experiments" / f"{record.experiment_id}.json"
        if record.extraction_state == "draft_extraction":
            out_path.write_text(
                json.dumps(record.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return record

        if not batch_comparable or not record.shared_controls:
            dec = FeedbackDecisionRecord(
                branch="experimental_confound_blocked",
                new_reference_bundle_id=None,
                weight_update_allowed=False,
                rationale="Cross-batch comparison or missing shared control; reference & weights preserved.",
            )
            updated = record.model_copy_validated(feedback_decision=dec)
            out_path.write_text(json.dumps(updated.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")
            return updated

        # Verify all observed construct IDs & digests match registered session bundles
        for obs in record.observations:
            if obs.bundle_id not in self._bundles:
                dec = FeedbackDecisionRecord(
                    branch="experimental_confound_blocked",
                    new_reference_bundle_id=None,
                    weight_update_allowed=False,
                    rationale=f"Observed bundle '{obs.bundle_id}' is not registered in this project.",
                )
                updated = record.model_copy_validated(feedback_decision=dec)
                out_path.write_text(json.dumps(updated.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")
                return updated
            if self._bundles[obs.bundle_id].version_digest != obs.version_digest:
                dec = FeedbackDecisionRecord(
                    branch="experimental_confound_blocked",
                    new_reference_bundle_id=None,
                    weight_update_allowed=False,
                    rationale=f"Version digest mismatch on '{obs.bundle_id}'; cannot align wet-lab observation.",
                )
                updated = record.model_copy_validated(feedback_decision=dec)
                out_path.write_text(json.dumps(updated.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")
                return updated

        if self.active_reference_bundle_id is not None and self.active_reference_bundle_id not in {
            obs.bundle_id for obs in record.observations
        }:
            dec = FeedbackDecisionRecord(
                branch="experimental_confound_blocked",
                new_reference_bundle_id=None,
                weight_update_allowed=False,
                rationale=(
                    f"Active reference '{self.active_reference_bundle_id}' was not measured in batch "
                    f"'{record.batch_id}'; refusing unanchored reference switch."
                ),
            )
            updated = record.model_copy_validated(feedback_decision=dec)
            out_path.write_text(json.dumps(updated.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")
            return updated

        # Compute biological replicate summary & interval per pre_registered_criteria
        crit = record.pre_registered_criteria
        stats: list[tuple[str, float, float, float, int]] = []  # (bundle_id, center, lo, hi, n_bio)
        for obs in record.observations:
            unit_vals = sorted(b.unit_mean for b in obs.biological_replicates)
            n_bio = len(unit_vals)
            if crit.aggregation_method == "median":
                mid = n_bio // 2
                center = unit_vals[mid] if n_bio % 2 == 1 else 0.5 * (unit_vals[mid - 1] + unit_vals[mid])
            elif crit.aggregation_method == "geometric_mean":
                center = math.exp(sum(math.log(max(x, 1e-12)) for x in unit_vals) / n_bio)
            else:
                center = sum(unit_vals) / n_bio

            if crit.variance_estimator == "iqr" and n_bio >= 3:
                q25 = unit_vals[int(0.25 * (n_bio - 1))]
                q75 = unit_vals[int(0.75 * (n_bio - 1))]
                lo, hi = q25, q75
            elif n_bio > 1:
                mean_v = sum(unit_vals) / n_bio
                sd = math.sqrt(sum((x - mean_v) ** 2 for x in unit_vals) / (n_bio - 1))
                half_w = sd if crit.variance_estimator == "sd" else (1.96 * sd / math.sqrt(n_bio))
                lo, hi = center - half_w, center + half_w
            else:
                half_w = abs(center) * 0.25
                lo, hi = center - half_w, center + half_w
            stats.append((obs.bundle_id, center, lo, hi, n_bio))

        stats.sort(key=lambda t: -t[1])
        best_id, best_center, best_lo, _, best_nbio = stats[0]
        cur_ref_id = self.active_reference_bundle_id or stats[-1][0]
        ref_stat = next(s for s in stats if s[0] == cur_ref_id)
        _, ref_center, _, ref_hi, ref_nbio = ref_stat

        fold = best_center / max(1e-9, ref_center)
        ci_non_overlap = best_lo > ref_hi

        if (
            best_id == cur_ref_id
            or best_nbio < crit.min_biological_replicates
            or ref_nbio < crit.min_biological_replicates
            or fold < crit.min_fold_change
            or (crit.require_non_overlapping_ci and not ci_non_overlap)
        ):
            dec = FeedbackDecisionRecord(
                branch="within_noise_undecided",
                new_reference_bundle_id=None,
                weight_update_allowed=False,
                rationale=(
                    f"Difference within measurement noise or below criteria (fold={fold:.2f}x, "
                    f"ci_non_overlap={ci_non_overlap}); retaining reference {cur_ref_id}."
                ),
            )
            updated = record.model_copy_validated(feedback_decision=dec)
            out_path.write_text(json.dumps(updated.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")
            return updated

        # Rank reversal met pre-registered criteria -> switch project reference to best_id first!
        ref_bundle = self._bundles[cur_ref_id]
        win_bundle = self._bundles[best_id]
        self.active_reference_bundle_id = best_id
        self._save_state_meta()

        changed_scopes = [
            s
            for s in ("five_prime_utr", "cds", "stop_cassette", "three_prime_utr", "poly_a")
            if ref_bundle.molecules[0].get_segment_sequence(s) != win_bundle.molecules[0].get_segment_sequence(s)
        ]
        attr_status = "multi_region_ambiguous" if len(changed_scopes) >= 2 else "single_region_resolved"
        chimera_id: str | None = None
        if attr_status == "multi_region_ambiguous":
            try:
                chimera = self._spawn_verbatim_disambiguation_chimera(ref_bundle, win_bundle)
                chimera_id = chimera.bundle_id
            except RuntimeError:
                chimera_id = None

        n_ind = record.n_independent_constructs
        iso_cal_id = f"cal_iso_{uuid.uuid4().hex[:8]}" if n_ind >= 6 else None
        dec = FeedbackDecisionRecord(
            branch="rank_reversal_switch_reference",
            new_reference_bundle_id=best_id,
            weight_update_allowed=False,  # Production weights require independent held-out verification even at N>=6!
            isolated_calibration_proposal_id=iso_cal_id,
            sequence_attribution_status=attr_status,  # type: ignore[arg-type]
            spawned_disambiguation_chimera_id=chimera_id,
            rationale=(
                f"Empirical readout ({record.readout_name_exact}) favored {best_id} over {cur_ref_id} "
                f"by {fold:.2f}x with non-overlapping CIs (n_bio={best_nbio}). Switched project reference to {best_id}; "
                f"automatic weight tuning strictly disabled (N={n_ind})."
            ),
        )
        updated = record.model_copy_validated(feedback_decision=dec)
        out_path.write_text(json.dumps(updated.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")
        return updated

"""The 9 deterministic atomic tools of `mrnax_core`, shared by BYO-LLM Chat, CLI, SDK & MCP."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mrnax.constraints.registry import ScopedConstraintEngine, default_m1_constraints
from mrnax.harness.state_machine import ProjectHarnessSession
from mrnax.io_insdc.genbank_io import export_construct_package
from mrnax.schemas.experiment_record import ExperimentRecord


class AtomicToolbox:
    """Exposes the 9 canonical `mrnax_core` tools with JSON-Schema specs for tool-calling LLMs & MCP."""

    def __init__(self, session: ProjectHarnessSession) -> None:
        self.session = session
        self.constraint_engine = ScopedConstraintEngine.from_seed_dir(session.seed_dir)

    # 1. search_protein_assets
    def search_protein_assets(self, query: str) -> dict[str, Any]:
        snap = json.loads(
            (self.session.seed_dir / "egfp_reporter_snapshot.json").read_text(encoding="utf-8")
        )
        q = query.strip().lower()
        if any(k in q for k in ("egfp", "gfp", "p42212", "u55762")):
            return {
                "matched": True,
                "uniprot_reference": snap["uniprot_reference"],
                "mammalian_reporter_egfp": snap["mammalian_reporter_egfp"],
            }
        return {"matched": False, "note": f"No local cached snapshot for '{query}'; query UniProt/NCBI."}

    # 2. audit_construct
    def audit_construct(
        self, bundle_id: str, compliance_track: str = "research_max"
    ) -> dict[str, Any]:
        bundle = self.session.get_bundle(bundle_id)
        constraints = (
            self.session._intents[-1].constraints
            if self.session._intents
            else default_m1_constraints()
        )
        outcome = self.constraint_engine.evaluate_bundle(
            bundle, constraints, compliance_track=compliance_track  # type: ignore[arg-type]
        )
        return {
            "bundle_id": bundle.bundle_id,
            "version_digest": bundle.version_digest,
            "passed_all_hard_gates": outcome.passed_all_hard_gates,
            "fto_blocked": outcome.fto_blocked,
            "violations": [v.model_dump(mode="json") for v in outcome.violations],
            "fto_alerts": [a.model_dump(mode="json") for a in outcome.fto_alerts],
            "advisories": [n.model_dump(mode="json") for n in outcome.advisories],
        }

    # 3. design_constructs
    def design_constructs(
        self,
        *,
        user_utterance: str,
        protein_sequence: str,
        accession_or_source: str = "GenBank:U55762.1",
        compliance_track: str = "research_max",
        uridine_chemistry: str = "m1psi",
        restriction_enzymes: list[str] | None = None,
        locked_five_prime_utr: str | None = None,
        detarget_liver: bool = False,
        prioritize_late_signal: bool = False,
        reference_bundle_id: str | None = None,
        random_seed: int = 42,
    ) -> dict[str, Any]:
        ref_b = self.session.get_bundle(reference_bundle_id) if reference_bundle_id else None
        locks = {"five_prime_utr": locked_five_prime_utr} if locked_five_prime_utr else None
        intent = self.session.create_initial_intent(
            turn_id=f"turn_{len(self.session._intents) + 1:02d}",
            user_utterance=user_utterance,
            protein_sequence=protein_sequence,
            accession_or_source=accession_or_source,
            reference_bundle=ref_b,
            compliance_track=compliance_track,  # type: ignore[arg-type]
            uridine_chemistry=uridine_chemistry,  # type: ignore[arg-type]
            locked_regions=locks,  # type: ignore[arg-type]
            restriction_enzymes=restriction_enzymes,
            detarget_tissues=["liver_hepatocyte"] if detarget_liver else None,
            protected_tissues=["skeletal_muscle"] if detarget_liver else None,
            prioritize_late_signal=prioritize_late_signal,
            random_seed=random_seed,
        )
        if intent.status != "ready":
            return {
                "intent_id": intent.intent_id,
                "intent_version": intent.intent_version,
                "status": intent.status,
                "blocking_questions": intent.blocking_questions,
            }
        bundles = self.session.execute_design(intent)
        latest_rep = list(self.session._reports.values())[-1]
        return {
            "intent_id": intent.intent_id,
            "intent_version": intent.intent_version,
            "outcome_status": latest_rep.outcome_status,
            "recommended_bundle_ids": latest_rep.recommended_bundle_ids,
            "qualified_count": len(bundles),
            "action_transparency": [t.model_dump(mode="json") for t in latest_rep.action_transparency],
        }

    # 4. refine_construct
    def refine_construct(
        self,
        *,
        user_utterance: str,
        updated_restriction_enzymes: list[str] | None = None,
        lock_cds_from_bundle_id: str | None = None,
        detarget_liver: bool | None = None,
    ) -> dict[str, Any]:
        add_locks = None
        if lock_cds_from_bundle_id:
            b = self.session.get_bundle(lock_cds_from_bundle_id)
            add_locks = {"cds": b.molecules[0].get_segment_sequence("cds")}
        intent_v = self.session.apply_incremental_intent_update(
            turn_id=f"turn_{len(self.session._intents) + 1:02d}",
            user_utterance=user_utterance,
            updated_restriction_enzymes=updated_restriction_enzymes,
            additional_locked_scopes=add_locks,  # type: ignore[arg-type]
            detarget_tissues=(["liver_hepatocyte"] if detarget_liver else None),
            protected_tissues=(["skeletal_muscle"] if detarget_liver else None),
        )
        bundles = self.session.execute_design(intent_v)
        latest_rep = list(self.session._reports.values())[-1]
        return {
            "intent_id": intent_v.intent_id,
            "intent_version": intent_v.intent_version,
            "parent_intent_id": intent_v.parent_intent_id,
            "inherited_locks": list(intent_v.locked_scopes.keys()),
            "outcome_status": latest_rep.outcome_status,
            "recommended_bundle_ids": latest_rep.recommended_bundle_ids,
            "qualified_count": len(bundles),
        }

    # 5. compare_bundles
    def compare_bundles(self, bundle_id_a: str, bundle_id_b: str) -> dict[str, Any]:
        ba = self.session.get_bundle(bundle_id_a)
        bb = self.session.get_bundle(bundle_id_b)
        ma, mb = ba.molecules[0], bb.molecules[0]
        diffs: dict[str, dict[str, Any]] = {}
        for s in ("five_prime_utr", "cds", "stop_cassette", "three_prime_utr", "poly_a"):
            sa = ma.get_segment_sequence(s)
            sb = mb.get_segment_sequence(s)
            diffs[s] = {
                "identical": sa == sb,
                "len_a": len(sa),
                "len_b": len(sb),
            }
        return {
            "bundle_a": bundle_id_a,
            "bundle_b": bundle_id_b,
            "segment_comparison": diffs,
        }

    # 6. ingest_experiment
    def ingest_experiment(self, record_json: dict[str, Any], batch_comparable: bool = True) -> dict[str, Any]:
        rec = ExperimentRecord.model_validate(record_json)
        updated = self.session.ingest_experiment_feedback(rec, batch_comparable=batch_comparable)
        return updated.model_dump(mode="json")

    # 7. explain_tradeoffs
    def explain_tradeoffs(self, report_id: str | None = None) -> dict[str, Any]:
        rep = (
            self.session._reports[report_id]
            if report_id
            else list(self.session._reports.values())[-1]
        )
        return {
            "report_id": rep.report_id,
            "outcome_status": rep.outcome_status,
            "recommended_bundle_ids": rep.recommended_bundle_ids,
            "candidate_evaluations": [e.model_dump(mode="json") for e in rep.candidate_evaluations],
            "action_transparency": [t.model_dump(mode="json") for t in rep.action_transparency],
            "pre_synthesis_advisories": [a.model_dump(mode="json") for a in rep.pre_synthesis_advisories],
        }

    # 8. get_job_status
    def get_job_status(self) -> dict[str, Any]:
        return {
            "project_id": self.session.project_id,
            "intent_versions": len(self.session._intents),
            "bundle_count": len(self.session._bundles),
            "active_reference_bundle_id": self.session.active_reference_bundle_id,
        }

    # 9. export_bundle
    def export_bundle(self, bundle_id: str, out_dir: str) -> dict[str, Any]:
        b = self.session.get_bundle(bundle_id)
        intent = self.session.latest_intent()
        manifest = export_construct_package(
            b,
            Path(out_dir),
            constraint_engine=self.constraint_engine,
            constraints=intent.constraints,
            compliance_track=intent.compliance_track,
        )
        return manifest

    def openai_tool_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_protein_assets",
                    "description": "Look up verified protein sequence & version snapshot (e.g. eGFP P42212 / U55762.1).",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "design_constructs",
                    "description": "Create versioned DesignIntent and run real mRNAx optimization engine.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "user_utterance": {"type": "string"},
                            "protein_sequence": {"type": "string"},
                            "accession_or_source": {"type": "string"},
                            "compliance_track": {"type": "string", "enum": ["research_max", "commercial_clean"]},
                            "uridine_chemistry": {"type": "string", "enum": ["unmodified_u", "pseudouridine", "m1psi"]},
                            "restriction_enzymes": {"type": "array", "items": {"type": "string"}},
                            "locked_five_prime_utr": {"type": "string"},
                            "detarget_liver": {"type": "boolean"},
                            "prioritize_late_signal": {"type": "boolean"},
                            "reference_bundle_id": {"type": "string"},
                        },
                        "required": ["user_utterance", "protein_sequence"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "refine_construct",
                    "description": "Modify a single constraint/lock on the active session while inheriting all other locked items.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "user_utterance": {"type": "string"},
                            "updated_restriction_enzymes": {"type": "array", "items": {"type": "string"}},
                            "lock_cds_from_bundle_id": {"type": "string"},
                            "detarget_liver": {"type": "boolean"},
                        },
                        "required": ["user_utterance"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "explain_tradeoffs",
                    "description": "Fetch deterministic metrics, Pareto ranks, and action_transparency for the latest run.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "compare_bundles",
                    "description": "Compare segment-by-segment differences between two ConstructBundles.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "bundle_id_a": {"type": "string"},
                            "bundle_id_b": {"type": "string"},
                        },
                        "required": ["bundle_id_a", "bundle_id_b"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "audit_construct",
                    "description": "Run shared ScopedConstraintEngine and sequence_fto_guard on a ConstructBundle.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "bundle_id": {"type": "string"},
                            "compliance_track": {"type": "string", "enum": ["research_max", "commercial_clean"]},
                        },
                        "required": ["bundle_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "export_bundle",
                    "description": "Export full RNA FASTA, CDS FASTA, INSDC GenBank, and chemistry_metadata.json.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "bundle_id": {"type": "string"},
                            "out_dir": {"type": "string"},
                        },
                        "required": ["bundle_id", "out_dir"],
                    },
                },
            },
        ]

    def dispatch_tool_call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        fn = getattr(self, name, None)
        if fn is None or name.startswith("_"):
            raise ValueError(f"Unknown tool '{name}'")
        return fn(**arguments)

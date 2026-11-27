"""The 9 deterministic atomic tools of `mrnax_core`, shared by BYO-LLM Chat, CLI, SDK & MCP."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mrnax.constraints.registry import ScopedConstraintEngine
from mrnax.engine.pareto_director import project_report_for_intent
from mrnax.harness.state_machine import ProjectHarnessSession, paired_chains
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
        self, bundle_id: str, compliance_track: str | None = None
    ) -> dict[str, Any]:
        bundle = self.session.get_bundle(bundle_id)
        intent = self.session.intent_for_bundle(bundle_id)
        outcome = self.constraint_engine.evaluate_bundle(
            bundle,
            intent.constraints,
            compliance_track=compliance_track or intent.compliance_track,
            expected_chemistry=intent.uridine_chemistry,
            locked_scopes=intent.locked_scopes,
            expected_protein_targets=intent.protein_targets,
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
        protein_sequence: str | None = None,
        accession_or_source: str = "GenBank:U55762.1",
        entry_version: str | None = None,
        compliance_track: str | None = None,
        uridine_chemistry: str | None = None,
        restriction_enzymes: list[str] | None = None,
        locked_five_prime_utr: str | None = None,
        detarget_liver: bool | None = None,
        prioritize_late_signal: bool = False,
        conflicting_single_pick: bool = False,
        reference_bundle_id: str | None = None,
        use_preset_reference_r: bool = False,
        use_commercial_surrogate: bool | None = None,
        random_seed: int = 42,
    ) -> dict[str, Any]:
        intent = self.session.create_turn_intent_canonical(
            user_utterance=user_utterance,
            protein_sequence=protein_sequence,
            accession_or_source=accession_or_source,
            entry_version=entry_version,
            compliance_track=compliance_track,  # type: ignore[arg-type]
            uridine_chemistry=uridine_chemistry,  # type: ignore[arg-type]
            restriction_enzymes=restriction_enzymes,
            locked_five_prime_utr=locked_five_prime_utr,
            detarget_liver=detarget_liver,
            prioritize_late_signal=prioritize_late_signal,
            conflicting_single_pick=conflicting_single_pick,
            reference_bundle_id=reference_bundle_id,
            use_preset_reference_r=use_preset_reference_r,
            use_commercial_surrogate=use_commercial_surrogate,
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
        latest_rep = self.session.latest_report(intent.intent_id)
        assert latest_rep is not None
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
        restriction_enzymes: list[str] | None = None,
        locked_five_prime_utr: str | None = None,
        lock_cds_from_bundle_id: str | None = None,
        unlock_scopes: list[str] | None = None,
        uridine_chemistry: str | None = None,
        compliance_track: str | None = None,
        detarget_liver: bool | None = None,
        use_commercial_surrogate: bool | None = None,
    ) -> dict[str, Any]:
        eff_enz = updated_restriction_enzymes if updated_restriction_enzymes is not None else restriction_enzymes
        intent_v = self.session.refine_turn_intent_canonical(
            user_utterance=user_utterance,
            restriction_enzymes=eff_enz,
            locked_five_prime_utr=locked_five_prime_utr,
            lock_cds_from_bundle_id=lock_cds_from_bundle_id,
            unlock_scopes=unlock_scopes,  # type: ignore[arg-type]
            uridine_chemistry=uridine_chemistry,  # type: ignore[arg-type]
            compliance_track=compliance_track,  # type: ignore[arg-type]
            detarget_liver=detarget_liver,
            use_commercial_surrogate=use_commercial_surrogate,
        )
        if intent_v.status != "ready":
            return {
                "intent_id": intent_v.intent_id,
                "intent_version": intent_v.intent_version,
                "parent_intent_id": intent_v.parent_intent_id,
                "status": intent_v.status,
                "blocking_questions": intent_v.blocking_questions,
            }
        bundles = self.session.execute_design(intent_v)
        latest_rep = self.session.latest_report(intent_v.intent_id)
        assert latest_rep is not None
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
        comparisons = []
        totals: dict[str, dict[str, Any]] = {}
        for ma, mb in paired_chains(ba, bb):
            diffs = {}
            for scope in ("five_prime_utr", "cds", "stop_cassette", "three_prime_utr", "poly_a"):
                sa = ma.get_segment_sequence(scope)
                sb = mb.get_segment_sequence(scope)
                diffs[scope] = {"identical": sa == sb, "len_a": len(sa), "len_b": len(sb)}
                total = totals.setdefault(scope, {"identical": True, "len_a": 0, "len_b": 0})
                total["identical"] &= sa == sb
                total["len_a"] += len(sa)
                total["len_b"] += len(sb)
            comparisons.append({"molecule_id_a": ma.molecule_id, "molecule_id_b": mb.molecule_id,
                                "segment_comparison": diffs})
        return {"bundle_a": bundle_id_a, "bundle_b": bundle_id_b, "segment_comparison": totals,
                "chain_comparisons": comparisons}

    # 6. ingest_experiment
    def ingest_experiment(
        self, record_json: dict[str, Any], batch_comparable: bool = True, held_out_validation_passed: bool = False
    ) -> dict[str, Any]:
        if type(held_out_validation_passed) is not bool or type(batch_comparable) is not bool:
            raise ValueError("Feedback validation flags must be booleans")
        rec = ExperimentRecord.model_validate(record_json)
        updated = self.session.ingest_experiment_feedback(
            rec, batch_comparable=batch_comparable, held_out_validation_passed=held_out_validation_passed,
        )
        return updated.model_dump(mode="json")

    # 7. explain_tradeoffs
    def explain_tradeoffs(self, report_id: str | None = None) -> dict[str, Any]:
        rep = (
            self.session._reports[report_id]
            if report_id
            else self.session.latest_report()
        )
        if rep is None:
            raise ValueError("No evaluation report available in session")
        intent = next((i for i in self.session._intents if i.intent_id == rep.intent_id), None)
        projected = project_report_for_intent(
            rep,
            self.session._bundles,
            intent,
            seed_dir=self.session.seed_dir,
            disallow_sandbox_override=True,
        )
        return {
            "report_id": projected.report_id,
            "outcome_status": projected.outcome_status,
            "recommended_bundle_ids": projected.recommended_bundle_ids,
            "candidate_evaluations": [e.model_dump(mode="json") for e in projected.candidate_evaluations],
            "action_transparency": [t.model_dump(mode="json") for t in projected.action_transparency],
            "pre_synthesis_advisories": [a.model_dump(mode="json") for a in projected.pre_synthesis_advisories],
        }

    # 8. get_job_status
    def get_job_status(self) -> dict[str, Any]:
        return {
            "project_id": self.session.project_id,
            "intent_versions": len(self.session._intents),
            "bundle_count": len(self.session._bundles),
            "active_reference_bundle_id": self.session.active_reference_bundle_id,
        }

    _ALLOWED_TOOL_NAMES: frozenset[str] = frozenset(
        {
            "search_protein_assets",
            "audit_construct",
            "design_constructs",
            "refine_construct",
            "compare_bundles",
            "ingest_experiment",
            "explain_tradeoffs",
            "get_job_status",
            "export_bundle",
        }
    )

    # 9. export_bundle
    def export_bundle(self, bundle_id: str, out_dir: str = "") -> dict[str, Any]:
        b = self.session.get_bundle(bundle_id)
        intent = self.session.intent_for_bundle(bundle_id)
        if intent.compliance_track == "commercial_clean":
            for rep in self.session._reports.values():
                if rep.intent_id == intent.intent_id and any(
                    e.bundle_id == bundle_id for e in rep.candidate_evaluations
                ):
                    if rep.isolated_sandbox_override or (
                        rep.rule_snapshot_version is not None
                        and not rep.rule_snapshot_version.startswith("rules_commercial_clean@")
                    ):
                        raise ValueError(
                            "Commercial export blocked: candidate originated from an isolated_sandbox_override or non-commercial rule snapshot."
                        )
        base_store = self.session.store_dir.resolve()
        raw_target = Path(out_dir) if out_dir else (base_store / "exports" / bundle_id)
        unres = raw_target if raw_target.is_absolute() else (base_store / raw_target)
        if unres.is_symlink() or any(p.is_symlink() for p in unres.parents if p != base_store.parent):
            raise ValueError(f"Export out_dir '{out_dir}' must not traverse symlinks")
        target_dir = unres.resolve()
        if not target_dir.is_relative_to(base_store) or target_dir == base_store:
            raise ValueError(
                f"Export out_dir '{out_dir}' must resolve inside project store_dir '{base_store}'"
            )
        for fn in ("full_rna.fasta", "cds_only.fasta", "construct.gb", "chemistry_metadata.json"):
            cand_f = target_dir / fn
            if cand_f.is_symlink():
                raise ValueError(f"Refusing to overwrite symlinked export target '{cand_f}'")
        manifest = export_construct_package(
            b,
            target_dir,
            constraint_engine=self.constraint_engine,
            constraints=intent.constraints,
            compliance_track=intent.compliance_track,
            expected_chemistry=intent.uridine_chemistry,
            locked_scopes=intent.locked_scopes,
            expected_protein_targets=intent.protein_targets,
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
                            "entry_version": {"type": "string"},
                            "compliance_track": {"type": "string", "enum": ["research_max", "commercial_clean"]},
                            "uridine_chemistry": {"type": "string", "enum": ["unmodified_u", "pseudouridine", "m1psi"]},
                            "restriction_enzymes": {"type": "array", "items": {"type": "string"}},
                            "locked_five_prime_utr": {"type": "string"},
                            "detarget_liver": {"type": "boolean"},
                            "prioritize_late_signal": {"type": "boolean"},
                            "conflicting_single_pick": {"type": "boolean"},
                            "reference_bundle_id": {"type": "string"},
                            "use_preset_reference_r": {"type": "boolean"},
                            "use_commercial_surrogate": {"type": "boolean"},
                            "random_seed": {"type": "integer", "default": 42},
                        },
                        "required": ["user_utterance"],
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
                            "restriction_enzymes": {"type": "array", "items": {"type": "string"}},
                            "locked_five_prime_utr": {"type": "string"},
                            "lock_cds_from_bundle_id": {"type": "string"},
                            "unlock_scopes": {"type": "array", "items": {"type": "string"}},
                            "uridine_chemistry": {"type": "string", "enum": ["unmodified_u", "pseudouridine", "m1psi"]},
                            "compliance_track": {"type": "string", "enum": ["research_max", "commercial_clean"]},
                            "detarget_liver": {"type": "boolean"},
                            "use_commercial_surrogate": {"type": "boolean"},
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
                    "name": "ingest_experiment",
                    "description": "Record an experiment with its batch comparability and held-out validation result.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "record_json": {"type": "object"},
                            "batch_comparable": {"type": "boolean", "default": True},
                            "held_out_validation_passed": {"type": "boolean", "default": False},
                        },
                        "required": ["record_json"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_job_status",
                    "description": "Return current session project_id, intent_versions, bundle_count, and active_reference_bundle_id.",
                    "parameters": {"type": "object", "properties": {}},
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
        if name not in self._ALLOWED_TOOL_NAMES:
            raise ValueError(f"Disallowed or unknown tool '{name}'")
        fn = getattr(self, name, None)
        if fn is None or not callable(fn):
            raise ValueError(f"Unknown tool '{name}'")
        return fn(**arguments)

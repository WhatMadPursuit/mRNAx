#!/usr/bin/env python3
"""Live Production Engine & Schema Verifier for M2 Frozen Acceptance Fixtures (Benchmark 1 & Benchmark 2)."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

DEFAULT_DIR = Path(__file__).parent


def verify_benchmark1(fixtures_dir: Path, seed_dir: Path, fto) -> dict:
    from mrnax.constraints.frameshift_m1psi import find_m1psi_slippery_spans
    from mrnax.harness.state_machine import ProjectHarnessSession
    from mrnax.schemas.construct_bundle import ConstructBundle
    from mrnax.schemas.experiment_record import ExperimentRecord

    # 1. Verify Benchmark 1 (including live ProjectHarnessSession replay across all 6 constructs!)
    b1 = json.loads((fixtures_dir / "m2_benchmark1_dual_speed_feedback_fixture_v1.json").read_text(encoding="utf-8"))
    bundles_b1 = {k: ConstructBundle.model_validate(v) for k, v in b1["bundles"].items()}
    assert len(bundles_b1) == 6
    for bid, b in bundles_b1.items():
        assert fto.scan_bundle(b, "commercial_clean") == [], f"FTO alert on {bid}"
        for mol in b.molecules:
            assert find_m1psi_slippery_spans(mol.get_segment_sequence("cds")) == []

    ma = bundles_b1["B1_REF_A"].molecules[0]
    mb = bundles_b1["B1_CAND_B"].molecules[0]
    mc = bundles_b1["B1_CHIMERA_5UTR_B_ON_A"].molecules[0]
    assert mc.get_segment_sequence("five_prime_utr") == mb.get_segment_sequence("five_prime_utr")
    assert mc.get_segment_sequence("cds") == ma.get_segment_sequence("cds")
    assert mc.get_segment_sequence("three_prime_utr") == ma.get_segment_sequence("three_prime_utr")

    exps = {k: ExperimentRecord.model_validate(v) for k, v in b1["experiment_records"].items()}
    assert len(exps) == 7

    record_count = len(exps)
    early = exps.pop("B1_CASE3_EARLY_CONTROL")
    with tempfile.TemporaryDirectory(prefix="mrnax_m2_b1_") as td:
        for case, record in exps.items():
            sess = ProjectHarnessSession(project_id=record.project_id, store_dir=Path(td) / case, seed_dir=seed_dir)
            for bundle in bundles_b1.values():
                sess.register_bundle(bundle)
            sess.create_initial_intent(
                turn_id="turn_01",
                user_utterance="Benchmark 1 live harness verification",
                protein_sequence=ma.effective_translation_target.protein_sequence,
                reference_bundle=bundles_b1["B1_REF_A"],
                compliance_track="commercial_clean",
                uridine_chemistry="m1psi",
                restriction_enzymes=["BspQI"],
            )
            if case == "B1_CASE3_KINETIC_DIVERGENCE":
                sess.ingest_experiment_feedback(early)
                assert sess.active_reference_bundle_id == "B1_REF_A"
            result = sess.ingest_experiment_feedback(
                record.model_copy_validated(feedback_decision=None),
                held_out_validation_passed=(case == "B1_CASE6_N6_HELD_OUT_CALIBRATION_UNLOCKED"),
            )
            actual, expected = result.feedback_decision, record.feedback_decision
            if record.extraction_state == "draft_extraction":
                assert actual is None and sess.active_reference_bundle_id == "B1_REF_A"
                continue
            assert actual is not None and expected is not None
            for field in (
                "branch", "new_reference_bundle_id", "retained_early_control_bundle_id",
                "sequence_attribution_status", "weight_update_allowed", "held_out_validation_passed",
            ):
                assert getattr(actual, field) == getattr(expected, field), (case, field)
            assert sess.active_reference_bundle_id == (expected.new_reference_bundle_id or "B1_REF_A")
            assert bool(actual.isolated_calibration_proposal_id) == bool(expected.isolated_calibration_proposal_id)
            assert bool(actual.spawned_disambiguation_chimera_id) == bool(expected.spawned_disambiguation_chimera_id)
            if expected.spawned_disambiguation_chimera_id:
                assert actual.spawned_disambiguation_chimera_id is not None
                child = sess.get_bundle(actual.spawned_disambiguation_chimera_id)
                expected_child = bundles_b1[expected.spawned_disambiguation_chimera_id]
                assert [m.sequence for m in child.molecules] == [m.sequence for m in expected_child.molecules]

    return {"benchmark1_bundles_verified": len(bundles_b1), "benchmark1_experiment_records_verified": record_count}


def verify(fixtures_dir: Path, mrnax_src: Path, fto_guard_json: Path) -> dict:
    sys.path.insert(0, str(mrnax_src))
    from mrnax.constraints.frameshift_m1psi import find_m1psi_slippery_spans
    from mrnax.constraints.fto_guard import SequenceFTOGuard
    from mrnax.engine.pareto_director import run_design_pipeline
    from mrnax.schemas.construct_bundle import ConstructBundle
    from mrnax.schemas.design_intent import DesignIntent

    seed_dir = fto_guard_json.parent
    fto = SequenceFTOGuard.from_json_path(fto_guard_json)
    b1_results = verify_benchmark1(fixtures_dir, seed_dir, fto)

    # 2. Verify Benchmark 2 (both frozen bundle AND live 2-chain production run_design_pipeline!)
    b2 = json.loads((fixtures_dir / "m2_benchmark2_dll3_cd3_acceptance_fixture_v1.json").read_text(encoding="utf-8"))
    intent = DesignIntent.model_validate(b2["design_intent"])
    assert intent.digest() == b2["design_intent_digest"]
    bundle_b2 = ConstructBundle.model_validate(b2["construct_bundle"])
    assert len(bundle_b2.molecules) == 2
    assert bundle_b2.co_expression_mode == "lnp_co_encapsulation"
    assert fto.scan_bundle(bundle_b2, "commercial_clean") == []

    # 2A. Frozen Benchmark 2 requests whole-organ 'spleen', which has no FANTOM5 primary-cell sRNA measurement -> must fail closed!
    blocked_bundles, blocked_rep, blocked_man = run_design_pipeline(intent, seed_dir=seed_dir)
    assert blocked_rep.outcome_status == "UNCOMPARABLE_MISSING_APPROVED_COMPONENT"
    assert blocked_bundles == blocked_rep.recommended_bundle_ids == blocked_man.invoked_components == []
    assert blocked_rep.uncomparable_reason and "spleen" in blocked_rep.uncomparable_reason

    # 2B. With FANTOM5-measured immune compartments (t_cell + myeloid_monocyte_macrophage), live 2-chain pipeline succeeds end-to-end!
    assert intent.cell_expression_control is not None
    measured_ctrl = intent.cell_expression_control.model_copy(
        update={"protected_tissues": ["t_cell", "myeloid_monocyte_macrophage"]}
    )
    intent_measured = intent.model_copy_validated(
        intent_id=f"{intent.intent_id}_fantom5_measured",
        intent_version=intent.intent_version + 1,
        parent_intent_id=intent.intent_id,
        cell_expression_control=measured_ctrl,
    )
    live_bundles, live_rep, live_man = run_design_pipeline(intent_measured, seed_dir=seed_dir)
    assert live_rep.outcome_status == "COMPLETED_WITH_CANDIDATES", live_rep.uncomparable_reason
    assert len(live_bundles) >= 2
    for lb in live_bundles:
        assert len(lb.molecules) == 2
        assert lb.co_expression_mode == "lnp_co_encapsulation"
        assert fto.scan_bundle(lb, "commercial_clean") == []
        for mol in lb.molecules:
            assert find_m1psi_slippery_spans(mol.get_segment_sequence("cds")) == []
            u3 = mol.get_segment_sequence("three_prime_utr")
            assert u3.count("CAAACACCAUUGUCACACUCCA") == 3

    return {
        "status": "PASS",
        **b1_results,
        "benchmark2_intent_digest_verified": intent.digest()[:16],
        "benchmark2_live_generated_two_chain_bundles": len(live_bundles),
        "benchmark2_evaluated_molecule_invocations": sum(
            c.attempt_count for c in live_man.invoked_components if c.role == "evaluator"
        ),
        "fto_alerts_total": 0,
        "m1psi_slippery_runs_total": 0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures-dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--mrnax-src", type=Path, required=True)
    ap.add_argument("--fto-guard-json", type=Path, required=True)
    args = ap.parse_args()
    res = verify(args.fixtures_dir, args.mrnax_src, args.fto_guard_json)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

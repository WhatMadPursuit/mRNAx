"""Revision, immutable lineage and matched-observation regression tests."""

import ast
import json
import multiprocessing
import re
import runpy
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from mrnax.cloud.cas_store import ExecutionCASStore
from mrnax.cloud.gateway import create_control_plane_app
from mrnax.constraints.fto_guard import SequenceFTOGuard
from mrnax.harness import cli, state_machine
from mrnax.harness.mcp_server import handle_mcp_request
from mrnax.harness.state_machine import ProjectHarnessSession, project_lock
from mrnax.harness.tools import AtomicToolbox
from mrnax.schemas.construct_bundle import ConstructBundle
from mrnax.schemas.evaluation_report import BundleEvaluation, EvaluationReport, MetricResult

ROOT = Path(__file__).resolve().parents[2]
SEED = ROOT / "registry_seed"
FIXTURE = ROOT / "tests/golden/m2_benchmark1_dual_speed_feedback_fixture_v1.json"
CAP = {"X-MRNAX-Project-Key": "test_project_key_0001"}
ADMIN = {"X-MRNAX-Access-Token": "test_admin_token"}


def session(path):
    return ProjectHarnessSession(project_id="project", store_dir=path, seed_dir=SEED)


def client(path):
    return TestClient(create_control_plane_app(
        ExecutionCASStore(path / "cas.sqlite"), projects_root=path, seed_dir=SEED, access_token=ADMIN["X-MRNAX-Access-Token"],
    ))


def initial(sess, ref):
    return sess.create_initial_intent(
        turn_id="t1", user_utterance="Reporter reference", reference_bundle=ref,
        protein_sequence=ref.molecules[0].effective_translation_target.protein_sequence,
    )


def report(intent, bundles):
    fixture_metrics = [
        MetricResult(
            metric_id=o.metric_id,
            value=0.5,
            unit="units",
            direction=o.direction,
            evaluated_scopes=["cds"],
            tool_name="fixture",
            tool_version="1.0",
            chemistry_evaluation_mode="native_chemistry_validated",
            participated_in="pareto_selection_or_ranking",
        )
        for o in intent.objectives
        if o.evidence_status != "unverified_target_for_wetlab"
    ]
    return EvaluationReport(report_id="rep_" + intent.intent_id, intent_id=intent.intent_id,
        intent_version=intent.intent_version, execution_id="run", compliance_track=intent.compliance_track,
        outcome_status="COMPLETED_WITH_CANDIDATES", recommended_bundle_ids=[b.bundle_id for b in bundles],
        candidate_evaluations=[BundleEvaluation(bundle_id=b.bundle_id, version_digest=b.version_digest, pareto_rank=1, metrics=fixture_metrics) for b in bundles],
        failed_diagnostics=[], fto_alerts=[], pre_synthesis_advisories=[], action_transparency=[])


def base_reference():
    return ConstructBundle.model_validate(json.loads(FIXTURE.read_text())["bundles"]["B1_REF_A"])


def test_reference_origin_survives_reevaluation_and_unmapped_export_fails(tmp_path, monkeypatch):
    ref = base_reference()
    s = session(tmp_path / "project")
    first = initial(s, ref)
    second = s.apply_incremental_intent_update(turn_id="t2", user_utterance="Revise constraints")
    unmapped = ref.model_copy_validated(bundle_id="unmapped")
    s.register_bundle(unmapped)
    monkeypatch.setattr(state_machine, "run_design_pipeline", lambda *args, **kwargs: (
        [ref, unmapped], report(second, [ref, unmapped]), SimpleNamespace(execution_id="run"),
    ))
    s.execute_design(second)
    reloaded = session(s.store_dir)
    assert reloaded.intent_for_bundle(ref.bundle_id) == first
    with pytest.raises(ValueError, match="originating intent"):
        reloaded.intent_for_bundle(unmapped.bundle_id)
    assert client(tmp_path).get("/v1/projects/project/bundles/unmapped/export", headers=ADMIN).status_code == 409


def test_cli_replay_enforces_chemistry_and_locked_scopes(tmp_path, monkeypatch):
    ref = base_reference()
    s = session(tmp_path / "project")
    intent = initial(s, ref)
    monkeypatch.setattr(cli, "run_design_pipeline", lambda *args, **kwargs: (
        [ref], report(intent, [ref]), SimpleNamespace(model_dump=lambda **kwargs: {}),
    ))
    actual_export = cli.export_construct_package
    captured = {}

    def capture(*args, **kwargs):
        captured.update(kwargs)
        return actual_export(*args, **kwargs)

    monkeypatch.setattr(cli, "export_construct_package", capture)
    assert cli.main(["replay", "--intent-file", str(next((s.store_dir / "intents").glob("*.json"))),
                     "--out-dir", str(tmp_path / "out"), "--seed-dir", str(SEED)]) == 0
    assert captured["expected_chemistry"] == intent.uridine_chemistry
    assert captured["locked_scopes"] == intent.locked_scopes
    assert (tmp_path / "out/candidates" / ref.bundle_id / "construct.gb").is_file()


def test_state_matches_current_report_and_reloads_all_415_bundles(tmp_path):
    s = session(tmp_path / "project")
    ref = base_reference().model_copy_validated(bundle_id="a_ref")
    intent = initial(s, ref)
    candidates = [ref.model_copy_validated(bundle_id=f"b_{i}") for i in range(10)]
    for b in candidates:
        s.register_bundle(b, originating_intent_id=intent.intent_id)
    for i in range(415):
        b = ref.model_copy_validated(bundle_id=f"z_{i:04}")
        (s.store_dir / "bundles" / f"{b.bundle_id}.json").write_text(b.model_dump_json())
    rep = report(intent, candidates)
    (s.store_dir / "reports" / f"{rep.report_id}.json").write_text(rep.model_dump_json())
    read = session(s.store_dir)
    assert read.get_bundle(read.active_reference_bundle_id) == ref
    http = client(tmp_path)
    state = http.get("/v1/projects/project/state", headers=ADMIN).json()
    assert {c["bundle_id"] for c in state["candidates"]} == {b.bundle_id for b in candidates}
    assert http.get("/v1/projects/project/bundles/b_9/export", headers=ADMIN).status_code == 200
    read.apply_incremental_intent_update(turn_id="t2", user_utterance="New revision")
    assert read.latest_report(intent.intent_id) == rep
    state = http.get("/v1/projects/project/state", headers=ADMIN).json()
    assert state["intent_count"] == 2 and state["latest_report"] is None and state["candidates"] == []


def test_bundle_file_collision_is_exclusive_and_idempotent(tmp_path):
    ref = base_reference()
    sess = session(tmp_path)
    sess.register_bundle(ref)
    path = tmp_path / "bundles" / f"{ref.bundle_id}.json"
    original = path.read_bytes()
    inode = path.stat().st_ino
    sess.register_bundle(ref)
    assert path.stat().st_ino == inode
    with pytest.raises(FileExistsError):
        sess.register_bundle(ref.model_copy_validated(parent_bundle_id="other"))
    assert path.read_bytes() == original and sess.get_bundle(ref.bundle_id) == ref


def submit_turn(path, ready, start, results):
    with client(path) as http:
        ready.put(True)
        assert start.wait(20)
        res = http.post("/v1/projects/project/turn", headers=CAP,
                        json={"mode":"initial","protein_sequence":"AA","user_utterance":"Check input"})
        results.put((res.status_code, res.json()["intent"]["intent_version"]))


def test_concurrent_gateway_processes_append_distinct_revisions(tmp_path):
    http = client(tmp_path)
    first = http.post("/v1/projects/project/turn", headers=CAP,
                      json={"mode":"initial","protein_sequence":"AA","user_utterance":"Check input"})
    assert first.status_code == 200
    ctx = multiprocessing.get_context("spawn")
    ready, results, start = ctx.Queue(), ctx.Queue(), ctx.Event()
    processes = [ctx.Process(target=submit_turn, args=(tmp_path, ready, start, results)) for _ in range(2)]
    for process in processes:
        process.start()
    try:
        assert all(ready.get(timeout=20) for _ in processes)
        start.set()
        assert sorted(results.get(timeout=20) for _ in processes) == [(200, 2), (200, 3)]
    finally:
        start.set()
        for process in processes:
            process.join(20)
            if process.is_alive():
                process.terminate()
                process.join()
    read = session(tmp_path / "project")
    assert [i.intent_version for i in read._intents] == [1, 2, 3]
    assert [i.parent_intent_id for i in read._intents[1:]] == [i.intent_id for i in read._intents[:2]]
    assert json.loads((read.store_dir / "project_state.json").read_bytes())["intent_count"] == 3


def test_held_project_lock_survives_other_projects(tmp_path):
    attempted, acquired = threading.Event(), threading.Event()

    def competing_writer():
        attempted.set()
        with project_lock(tmp_path):
            acquired.set()

    with ThreadPoolExecutor() as pool:
        with project_lock(tmp_path):
            for i in range(1025):
                path = tmp_path / str(i)
                path.mkdir()
                with project_lock(path):
                    pass
            task = pool.submit(competing_writer)
            assert attempted.wait(10)
            assert not acquired.wait(0.05)
        task.result(timeout=10)
    assert acquired.is_set()


def fixture_tool(path):
    data = json.loads(FIXTURE.read_text())
    bundles = {k:ConstructBundle.model_validate(v) for k,v in data["bundles"].items()}
    s = ProjectHarnessSession(project_id="PROJ_M2_BENCHMARK1", store_dir=path, seed_dir=SEED)
    initial(s, bundles["B1_REF_A"])
    for b in bundles.values():
        s.register_bundle(b)
    return AtomicToolbox(s), data["experiment_records"], bundles


def ingest(tool, record, **flags):
    return tool.dispatch_tool_call("ingest_experiment", {"record_json": {**record, "feedback_decision":None}, **flags})["feedback_decision"]


@pytest.mark.parametrize("changed", [None, "assay_type", "batch_id", "readout_name_exact", "unit"])
def test_kinetic_replay_requires_matched_assay_and_batch(tmp_path, changed):
    tool, records, _ = fixture_tool(tmp_path)
    early = records["B1_CASE3_EARLY_CONTROL"]
    if changed:
        early[changed] += "_other"
    ingest(tool, early)
    late = ingest(tool, records["B1_CASE3_KINETIC_DIVERGENCE"])
    if changed:
        assert late["branch"] == "rank_reversal_switch_reference" and late["retained_early_control_bundle_id"] is None
    else:
        assert late["branch"] == "kinetic_divergence_update" and late["retained_early_control_bundle_id"] == "B1_REF_A"


def test_public_feedback_schema_and_held_out_result(tmp_path):
    tool, records, _ = fixture_tool(tmp_path)
    desc = handle_mcp_request(tool, {"method":"tools/list","id":1})["result"]["tools"]
    spec = next(t["inputSchema"] for t in desc if t["name"] == "ingest_experiment")
    assert spec["properties"]["held_out_validation_passed"]["type"] == "boolean"
    record = records["B1_CASE6_N6_HELD_OUT_CALIBRATION_UNLOCKED"]
    with pytest.raises(ValueError):
        ingest(tool, record, held_out_validation_passed="false")
    dec = ingest(tool, record, held_out_validation_passed=True)
    assert dec["held_out_validation_passed"] and dec["weight_update_allowed"] and dec["isolated_calibration_proposal_id"]


@pytest.mark.parametrize("mismatch", [False, True])
def test_feedback_keeps_second_chain_and_checks_stoichiometry(tmp_path, mismatch):
    tool, records, bundles = fixture_tool(tmp_path)
    a, b = bundles["B1_REF_A"], bundles["B1_CAND_B"]
    unchanged = a.molecules[0].model_copy_validated(molecule_id="first")
    ref_chains = [unchanged, a.molecules[0].model_copy_validated(molecule_id="second")]
    winner_chains = [unchanged, b.molecules[0].model_copy_validated(molecule_id="second")]
    stoich = {"first":1.0, "second":2.0}
    def pair(name, chains):
        return ConstructBundle.create(bundle_id=name, parent_bundle_id=None, derivation_action="reference_import",
                                      molecules=chains, co_expression_mode="lnp_co_encapsulation", target_stoichiometry=stoich,
                                      external_evidence=[{"source_name": name}])
    ref, winner = pair("pair_a", ref_chains), pair("pair_b", winner_chains[:1] if mismatch else winner_chains[::-1])
    tool.session.register_bundle(ref, set_as_reference=True)
    tool.session.register_bundle(winner)
    record = records["B1_CASE4_N2_REVERSAL_CHIMERA"]
    for obs, parent in zip(record["observations"], [ref, winner], strict=True):
        obs.update(bundle_id=parent.bundle_id, version_digest=parent.version_digest)
    dec = ingest(tool, record)
    if mismatch:
        assert dec["branch"] == "experimental_confound_blocked"
        assert tool.session.active_reference_bundle_id == ref.bundle_id
        return
    assert dec["sequence_attribution_status"] == "multi_region_ambiguous"
    child = tool.session.get_bundle(dec["spawned_disambiguation_chimera_id"])
    assert re.fullmatch(r"bnd_chimera_[a-f0-9]{32}", child.bundle_id)
    assert child.external_evidence == [*ref.external_evidence, *winner.external_evidence]
    comparison = tool.compare_bundles(ref.bundle_id, winner.bundle_id)
    assert comparison["chain_comparisons"][1]["molecule_id_a"] == "second"
    assert not comparison["segment_comparison"]["five_prime_utr"]["identical"]
    assert child.target_stoichiometry == stoich and child.co_expression_mode == ref.co_expression_mode
    assert [m.molecule_id for m in child.molecules] == ["first","second"]
    assert child.molecules[0] == unchanged
    assert child.molecules[1].get_segment_sequence("five_prime_utr") == b.molecules[0].get_segment_sequence("five_prime_utr")
    assert child.molecules[1].get_segment_sequence("cds") == a.molecules[0].get_segment_sequence("cds")


def test_single_definition_for_origin_and_report():
    tree = ast.parse(Path(state_machine.__file__).read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ProjectHarnessSession")
    names = [node.name for node in cls.body if isinstance(node, ast.FunctionDef)]
    for name in ("intent_for_bundle", "latest_report"):
        assert names.count(name) == 1


def test_m2_benchmark1_golden_runner_replays_independent_cases():
    runner = runpy.run_path(str(FIXTURE.parent / "eval_m2_acceptance_fixtures.py"))
    guard = SequenceFTOGuard.from_json_path(SEED / "patent_fto_guard_v2.json")
    assert runner["verify_benchmark1"](FIXTURE.parent, SEED, guard) == {
        "benchmark1_bundles_verified": 6,
        "benchmark1_experiment_records_verified": 7,
    }

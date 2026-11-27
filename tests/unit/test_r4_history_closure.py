"""Commercial release decisions from complete, immutable project history."""

import hashlib
import json
import shutil

import pytest
from fastapi.testclient import TestClient

from mrnax.cloud.cas_store import ExecutionCASStore
from mrnax.cloud.gateway import create_control_plane_app
from mrnax.constraints.fto_guard import MethodFTOGuard
from mrnax.engine.pareto_director import project_report_for_intent
from mrnax.engine.profiles import resolve_default_seed_dir
from mrnax.harness.state_machine import ProjectHarnessSession
from mrnax.harness.tools import AtomicToolbox
from mrnax.schemas.design_intent import RuntimePolicySpec


SEED = resolve_default_seed_dir()


@pytest.fixture(scope="module")
def producing_run(tmp_path_factory):
    store = tmp_path_factory.mktemp("r4-history")
    session = ProjectHarnessSession(project_id="history", store_dir=store, seed_dir=SEED)
    base = session.create_initial_intent(
        turn_id="initial", user_utterance="Reporter control", protein_sequence="M" + "ALESK" * 12,
        compliance_track="commercial_clean", uridine_chemistry="unmodified_u",
    )
    intent = session._persist_intent(base.model_copy_validated(
        intent_id="release", intent_version=2, parent_intent_id=base.intent_id,
        runtime_policy=RuntimePolicySpec(enabled_generators=["stage0_random"], enabled_evaluators=["surrogate_v1_clean"]),
    ))
    bundle, = session.execute_design(intent)
    report = session.latest_report()
    assert report.recommended_bundle_ids == [bundle.bundle_id]
    return store, intent, bundle, report


def check_history_routes(store, bundle, allowed):
    originals = {p: p.read_bytes() for folder in ("reports", "manifests") for p in (store/folder).glob("*.json")}
    session = ProjectHarnessSession(project_id="history", store_dir=store, seed_dir=SEED)
    latest = session.latest_report()
    assert latest.recommended_bundle_ids == ([bundle.bundle_id] if allowed else [])
    toolbox = AtomicToolbox(session)
    if allowed:
        toolbox.export_bundle(bundle.bundle_id)
    else:
        with pytest.raises(ValueError, match="Commercial export blocked"):
            toolbox.export_bundle(bundle.bundle_id)
    app = create_control_plane_app(ExecutionCASStore(store.parent/"cas.sqlite"), projects_root=store.parent,
                                   seed_dir=SEED, access_token="audit")
    with TestClient(app, headers={"X-MRNAX-Access-Token": "audit"}) as client:
        state = client.get("/v1/projects/history/state")
        exported = client.get(f"/v1/projects/history/bundles/{bundle.bundle_id}/export")
    assert state.status_code == 200
    public = state.json()
    expected = [bundle.bundle_id] if allowed else []
    assert public["latest_report"]["recommended_bundle_ids"] == expected
    assert [e["bundle_id"] for e in public["latest_report"]["candidate_evaluations"]] == expected
    assert [c["bundle_id"] for c in public["candidates"]] == expected
    assert all(set(t["changed_candidate_ids"]) <= set(expected) for t in public["latest_report"]["action_transparency"]
               if t["stage_applied"] == "ranking")
    assert exported.status_code == (200 if allowed else 409), exported.text
    assert (store/"exports"/bundle.bundle_id).exists() is allowed
    assert all(p.read_bytes() == original for p, original in originals.items())


@pytest.mark.parametrize("decision", [
    "clean", "dependency", "overall", "manifest-overall", "different-execution", "conflicting-history", "wrong-report-hash", "unreadable-manifest", "misfiled-execution",
    *MethodFTOGuard.BOUNDARY_IDS,
])
def test_recorded_release_decision_on_reload(producing_run, tmp_path, decision):
    original, _, bundle, report = producing_run
    store = tmp_path/"history"
    shutil.copytree(original, store)
    path = store/"manifests"/f"{report.execution_id}.json"
    manifest = json.loads(path.read_bytes())
    audit = manifest["optimization_mode_audit"]
    if decision in ("dependency", "different-execution", "conflicting-history"):
        audit["three_layer_dependency_manifest"][0]["transitive_gpl_count"] = 1
        if decision == "different-execution":
            manifest["execution_id"] = "other"
            path = store/"manifests"/"other.json"
        elif decision == "conflicting-history":
            path = store/"run_manifest.json"
    elif decision == "misfiled-execution":
        manifest["execution_id"] = "other"
    elif decision == "overall":
        audit["overall_decision"] = "BLOCK"
    elif decision == "manifest-overall":
        manifest["overall_decision"] = "BLOCK"
    elif decision == "wrong-report-hash":
        next(a for a in manifest["artifacts"] if a["logical_name"] == "evaluation_report")["sha256"] = "0" * 64
    elif decision in MethodFTOGuard.BOUNDARY_IDS:
        audit["method_fto_boundary_checks"][decision]["status"] = "BLOCK"
    if decision == "unreadable-manifest":
        path.write_bytes(b"\xff")
    else:
        path.write_text(json.dumps(manifest))
    check_history_routes(store, bundle, decision in {"clean", "different-execution"})


@pytest.mark.parametrize("case", ["clean", "changed-report", "changed-intent", "wrong-path", "pinned-files", "legacy"])
def test_compact_copy_of_a_direct_pipeline_receipt(producing_run, tmp_path, case):
    original, intent, bundle, report = producing_run
    store = tmp_path/"history"
    shutil.copytree(original, store)
    ipath = next((store/"intents").glob("*_release.json"))
    rpath = store/"reports"/f"{report.report_id}.json"
    mpath = store/"manifests"/f"{report.execution_id}.json"
    manifest = json.loads(mpath.read_bytes())
    if case != "pinned-files":
        manifest["artifacts"] = [a for a in manifest["artifacts"] if a["logical_name"] != "input_intent"]
    if case == "wrong-path":
        manifest["artifacts"][0]["local_relpath"] = "reports/explicit-filename.json"
    if case == "changed-intent":
        intent = intent.model_copy_validated(created_from_turn_ids=[*intent.created_from_turn_ids, "changed"])
    if case == "changed-report":
        report = report.model_copy_validated(created_at_ns=1)
    if case == "legacy":
        report = report.model_copy_validated(rule_snapshot_version=None)
    ipath.write_text(intent.model_dump_json())
    rpath.write_text(report.model_dump_json())
    mpath.write_text(json.dumps(manifest))
    check_history_routes(store, bundle, case == "clean")


def test_older_foreign_digest_withdraws_the_later_clean_view(producing_run, tmp_path):
    original, _, bundle, report = producing_run
    store = tmp_path/"history"
    shutil.copytree(original, store)
    ev, = report.candidate_evaluations
    foreign = report.model_copy_validated(report_id="rep_older", execution_id="older",
        candidate_evaluations=[ev.model_copy_validated(version_digest="f" * 64)], created_at_ns=1)
    (store/"reports"/"rep_older.json").write_text(foreign.model_dump_json())
    state_path = store/"project_state.json"
    state = json.loads(state_path.read_bytes())
    state["report_order"] = [foreign.report_id, report.report_id]
    state_path.write_text(json.dumps(state))
    check_history_routes(store, bundle, False)


def test_older_manifest_with_missing_intent_revision_blocks_later_claim(producing_run, tmp_path):
    original, _, bundle, report = producing_run
    store = tmp_path/"history"
    shutil.copytree(original, store)
    older = report.model_copy_validated(report_id="rep_old", execution_id="old", intent_version=5, created_at_ns=1)
    report_bytes = older.model_dump_json().encode()
    (store/"reports"/"rep_old.json").write_bytes(report_bytes)
    manifest = json.loads((store/"manifests"/f"{report.execution_id}.json").read_bytes())
    manifest["execution_id"] = older.execution_id
    manifest["optimization_mode_audit"]["run_id"] = older.execution_id
    report_pin = next(a for a in manifest["artifacts"] if a["logical_name"] == "evaluation_report")
    report_pin.update(sha256=hashlib.sha256(report_bytes).hexdigest(), md5=hashlib.md5(report_bytes).hexdigest(),
                      bytes=len(report_bytes))
    manifest["optimization_mode_audit"]["three_layer_dependency_manifest"][0]["transitive_gpl_count"] = 1
    (store/"manifests"/"old.json").write_text(json.dumps(manifest))
    state_path = store/"project_state.json"
    state = json.loads(state_path.read_bytes())
    state["report_order"] = [older.report_id, report.report_id]
    state_path.write_text(json.dumps(state))
    check_history_routes(store, bundle, False)


@pytest.mark.parametrize(("repeat", "repeated_selection"), [(1, False), (3, False), (3, True)])
def test_unique_ranked_bundles_get_the_top_three_slots(producing_run, repeat, repeated_selection):
    _, intent, bundle, report = producing_run
    second = bundle.model_copy_validated(bundle_id="second")
    withheld = bundle.model_copy_validated(bundle_id="withheld")
    ev, = report.candidate_evaluations
    history = report.model_copy_validated(candidate_evaluations=[
        *([ev] * repeat), ev.model_copy_validated(bundle_id="second"),
        ev.model_copy_validated(bundle_id="withheld", version_digest="a" * 64),
    ], recommended_bundle_ids=[bundle.bundle_id] * 3 if repeated_selection else ["withheld"])
    projected = project_report_for_intent(history, {b.bundle_id: b for b in (bundle, second, withheld)},
                                          intent, seed_dir=SEED)
    assert projected.recommended_bundle_ids == [bundle.bundle_id, "second"]
    assert [e.bundle_id for e in projected.candidate_evaluations] == [bundle.bundle_id] * repeat + ["second"]

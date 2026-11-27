"""Persisted project isolation and agreement between the state and export routes."""

import json
from functools import partial
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from mrnax.cloud.cas_store import ExecutionCASStore
from mrnax.cloud.gateway import create_control_plane_app
from mrnax.engine.pareto_director import run_design_pipeline
from mrnax.engine.profiles import resolve_default_seed_dir
from mrnax.harness.state_machine import ProjectHarnessSession, build_preset_reference_r
from mrnax.harness.tools import AtomicToolbox
from mrnax.io_insdc.genbank_io import export_construct_package
from mrnax.schemas.construct_bundle import ConstructBundle
from mrnax.schemas.design_intent import RuntimePolicySpec

SEED = resolve_default_seed_dir()


@pytest.fixture
def project(tmp_path, request):
    bundle = build_preset_reference_r(SEED)
    session = ProjectHarnessSession(project_id="history", store_dir=tmp_path / "history", seed_dir=SEED)
    track = getattr(request, "param", "commercial_clean")
    locks = {"five_prime_utr": bundle.molecules[0].get_segment_sequence("five_prime_utr")} if track == "research_max" else {}
    initial = session.create_initial_intent(
        turn_id="first", user_utterance="Reporter control",
        protein_sequence=bundle.molecules[0].effective_translation_target.protein_sequence,
        compliance_track=track, uridine_chemistry="m1psi", use_commercial_surrogate=True, locked_regions=locks,
    )
    intent = session._persist_intent(initial.model_copy_validated(
        intent_id="cached", intent_version=2, parent_intent_id=initial.intent_id, constraints=[],
        runtime_policy=RuntimePolicySpec(
            enabled_generators=[], enabled_evaluators=["surrogate_v1_clean"], allow_cached_candidates=True,
        ),
    ))
    return session, intent, bundle


def run_cached(session, intent, bundles, *, sandbox=False):
    run = partial(run_design_pipeline, cached_candidates=bundles, isolated_sandbox_override=sandbox)
    with patch("mrnax.harness.state_machine.run_design_pipeline", new=run):
        assert {b.bundle_id for b in session.execute_design(intent)} == {b.bundle_id for b in bundles}


def route_client(root):
    app = create_control_plane_app(ExecutionCASStore(root / "cas.sqlite"), projects_root=root,
                                   seed_dir=SEED, access_token="fixture")
    return TestClient(app, headers={"X-MRNAX-Access-Token": "fixture"}, raise_server_exceptions=False)


def project_json_bytes(store):
    return {p.relative_to(store): p.read_bytes() for p in store.rglob("*.json") if "exports" not in p.parts}


def test_invalid_legacy_bundle_keeps_its_identity_and_its_valid_sibling(project, tmp_path):
    session, intent, bundle = project
    original = bundle.molecules[0]
    stop = original.get_segment("stop_cassette")
    legacy_molecule = original.model_copy_validated(
        sequence=original.sequence[:stop.start] + "UAG" + original.sequence[stop.start + 3:]
    )
    legacy = ConstructBundle.create(
        bundle_id="legacy", parent_bundle_id=None, derivation_action="reference_import",
        molecules=[legacy_molecule], target_stoichiometry=bundle.target_stoichiometry,
    )
    run_cached(session, intent, [bundle, legacy])
    path = session.store_dir / "bundles" / "legacy.json"
    raw = json.loads(path.read_bytes())
    molecule = raw["molecules"][0]
    molecule["effective_translation_target"]["transl_table"] = 2
    sequence = molecule["sequence"]
    molecule["sequence"] = sequence[:stop.start] + "UGA" + sequence[stop.start + 3:]
    path.write_text(json.dumps(raw))
    before = project_json_bytes(session.store_dir)

    with route_client(tmp_path) as client:
        state = client.get("/v1/projects/history/state")
        assert state.status_code == 200
        view = state.json()
        assert set(view["quarantined_bundles"]) == {"legacy"}
        assert "stop_cassette" in view["quarantined_bundles"]["legacy"]
        assert [c["bundle_id"] for c in view["candidates"]] == [bundle.bundle_id]
        assert view["latest_report"]["recommended_bundle_ids"] == [bundle.bundle_id]
        assert client.get(f"/v1/projects/history/bundles/{bundle.bundle_id}/export").status_code == 200
        assert client.get("/v1/projects/history/bundles/legacy/export").status_code == 409
        assert client.get("/v1/projects/history/bundles/unknown/export").status_code == 404
    reopened = ProjectHarnessSession(project_id="history", store_dir=session.store_dir, seed_dir=SEED)
    assert reopened.latest_report().recommended_bundle_ids == [bundle.bundle_id]
    toolbox = AtomicToolbox(reopened)
    with pytest.raises(ValueError, match="stop_cassette"):
        toolbox.export_bundle("legacy")
    toolbox.export_bundle(bundle.bundle_id)
    assert not (session.store_dir / "exports" / "legacy").exists()
    assert project_json_bytes(session.store_dir) == before


@pytest.mark.parametrize("history,allowed", [((False, False), True), ((True, False), False), ((True,), False)])
def test_sandbox_history_agrees_across_project_state_and_both_exports(project, tmp_path, history, allowed):
    session, intent, bundle = project
    for sandbox in history:
        run_cached(session, intent, [bundle], sandbox=sandbox)
    before = project_json_bytes(session.store_dir)
    expected = [bundle.bundle_id] if allowed else []

    reopened = ProjectHarnessSession(project_id="history", store_dir=session.store_dir, seed_dir=SEED)
    assert reopened.latest_report().recommended_bundle_ids == expected
    toolbox = AtomicToolbox(reopened)
    if allowed:
        toolbox.export_bundle(bundle.bundle_id)
    else:
        with pytest.raises(ValueError, match="isolated_sandbox_override"):
            toolbox.export_bundle(bundle.bundle_id)
    with route_client(tmp_path) as client:
        state = client.get("/v1/projects/history/state")
        assert state.status_code == 200
        view = state.json()
        assert [c["bundle_id"] for c in view["candidates"]] == expected
        assert view["latest_report"]["recommended_bundle_ids"] == expected
        ranking = [a for a in view["latest_report"]["action_transparency"] if a["stage_applied"] == "ranking"]
        assert bool(ranking) == allowed
        assert client.get(f"/v1/projects/history/bundles/{bundle.bundle_id}/export").status_code == (200 if allowed else 409)
    assert project_json_bytes(session.store_dir) == before


def test_older_recommendation_without_an_evaluation_blocks_later_production(project, tmp_path):
    session, intent, bundle = project

    def legacy_run(*args, **kwargs):
        bundles, report, manifest = run_design_pipeline(*args, cached_candidates=[bundle], **kwargs)
        return bundles, report.model_copy(update={"candidate_evaluations": []}), manifest

    with patch("mrnax.harness.state_machine.run_design_pipeline", new=legacy_run):
        session.execute_design(intent)
    run_cached(session, intent, [bundle])
    before = project_json_bytes(session.store_dir)
    for reader in (session, ProjectHarnessSession(project_id="history", store_dir=session.store_dir, seed_dir=SEED)):
        assert reader.latest_report().recommended_bundle_ids == []
        with pytest.raises(ValueError, match="Commercial export blocked"):
            AtomicToolbox(reader).export_bundle(bundle.bundle_id)
    with route_client(tmp_path) as client:
        view = client.get("/v1/projects/history/state")
        assert view.status_code == 200
        assert view.json()["candidates"] == []
        assert client.get(f"/v1/projects/history/bundles/{bundle.bundle_id}/export").status_code == 409
    assert project_json_bytes(session.store_dir) == before


@pytest.mark.parametrize(("sandbox_runs", "refine", "allowed"), [
    ((False, False), True, True),
    ((True, False), True, False),
    ((False, True, False), True, False),
    ((False, True), False, False),
    ((False, True, False), False, False),
])
def test_bundle_history_is_current_across_intents_and_readers(project, tmp_path, sandbox_runs, refine, allowed):
    writer, intent, bundle = project
    run_cached(writer, intent, [bundle], sandbox=sandbox_runs[0])
    reader = ProjectHarnessSession(store_dir=writer.store_dir, seed_dir=SEED)
    toolbox = AtomicToolbox(reader)
    for sandbox in sandbox_runs[1:]:
        if refine:
            intent = writer.refine_turn_intent_canonical(user_utterance="Compare the same reporter again")
        run_cached(writer, intent, [bundle], sandbox=sandbox)
    before = project_json_bytes(writer.store_dir)
    expected = [bundle.bundle_id] if allowed else []

    if allowed:
        toolbox.export_bundle(bundle.bundle_id)
    else:
        with pytest.raises(ValueError, match="isolated_sandbox_override"):
            toolbox.export_bundle(bundle.bundle_id)
    assert len(list((reader.store_dir / "exports" / bundle.bundle_id).glob("*"))) == (4 if allowed else 0)
    assert reader.latest_report().recommended_bundle_ids == expected
    assert toolbox.explain_tradeoffs()["recommended_bundle_ids"] == expected
    assert toolbox.get_job_status()["intent_versions"] == intent.intent_version
    with route_client(tmp_path) as client:
        state = client.get("/v1/projects/history/state")
        assert state.status_code == 200
        view = state.json()
        assert [c["bundle_id"] for c in view["candidates"]] == expected
        assert view["latest_report"]["recommended_bundle_ids"] == expected
        assert sum(a["stage_applied"] == "ranking" for a in view["latest_report"]["action_transparency"]) == (4 if allowed else 0)
        assert client.get(f"/v1/projects/history/bundles/{bundle.bundle_id}/export").status_code == (200 if allowed else 409)
    assert project_json_bytes(writer.store_dir) == before


def test_open_toolbox_reads_new_bundles_intents_and_reports(project):
    writer, intent, bundle = project
    status, audit, compare, explain = (
        AtomicToolbox(ProjectHarnessSession(store_dir=writer.store_dir, seed_dir=SEED)) for _ in range(4)
    )
    child = writer.refine_turn_intent_canonical(user_utterance="Compare the same reporter again")
    run_cached(writer, child, [bundle])
    report = writer.latest_report(child.intent_id)

    assert status.get_job_status()["intent_versions"] == intent.intent_version + 1
    assert audit.audit_construct(bundle.bundle_id)["bundle_id"] == bundle.bundle_id
    assert compare.compare_bundles(bundle.bundle_id, bundle.bundle_id)["bundle_a"] == bundle.bundle_id
    assert explain.explain_tradeoffs(report.report_id)["recommended_bundle_ids"] == [bundle.bundle_id]


def test_failed_refresh_retries_the_same_revision_before_export(project):
    writer, intent, bundle = project
    run_cached(writer, intent, [bundle])
    toolbox = AtomicToolbox(ProjectHarnessSession(store_dir=writer.store_dir, seed_dir=SEED))
    run_cached(writer, intent, [bundle], sandbox=True)
    latest = writer.latest_report()
    path = writer.store_dir / "reports" / f"{latest.report_id}.json"
    raw = path.read_bytes()
    path.write_bytes(b"{")
    for _ in range(2):
        with pytest.raises(json.JSONDecodeError):
            toolbox.export_bundle(bundle.bundle_id)
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="isolated_sandbox_override"):
        toolbox.export_bundle(bundle.bundle_id)
    assert not (writer.store_dir / "exports").exists()


def test_loading_legacy_report_order_preserves_the_stored_project(project):
    writer, intent, bundle = project
    for _ in range(2):
        run_cached(writer, intent, [bundle])
    latest_id = writer.latest_report().report_id
    meta_path = writer.store_dir / "project_state.json"
    meta = json.loads(meta_path.read_bytes())
    del meta["report_order"]
    meta_path.write_text(json.dumps(meta))
    before = project_json_bytes(writer.store_dir)
    reader = ProjectHarnessSession(store_dir=writer.store_dir, seed_dir=SEED)
    assert reader.latest_report().report_id == latest_id
    assert project_json_bytes(reader.store_dir) == before


@pytest.mark.parametrize("project", ["research_max"], indirect=True)
@pytest.mark.parametrize("sandbox", [True, False])
def test_commercial_refinement_uses_commercial_export_for_research_origin(project, tmp_path, sandbox):
    writer, original, bundle = project
    run_cached(writer, original, [bundle], sandbox=sandbox)
    reader = AtomicToolbox(ProjectHarnessSession(store_dir=writer.store_dir, seed_dir=SEED))
    commercial = writer.refine_turn_intent_canonical(
        user_utterance="Evaluate the reporter for commercial release",
        compliance_track="commercial_clean", use_commercial_surrogate=True,
    )
    run_cached(writer, commercial, [bundle])
    before = project_json_bytes(writer.store_dir)
    expected = [] if sandbox else [bundle.bundle_id]
    assert writer.intent_for_bundle(bundle.bundle_id) == original
    assert commercial.locked_scopes == original.locked_scopes
    with patch("mrnax.harness.tools.export_construct_package", wraps=export_construct_package) as harness_export:
        if sandbox:
            with pytest.raises(ValueError, match="isolated_sandbox_override"):
                reader.export_bundle(bundle.bundle_id)
            harness_export.assert_not_called()
        else:
            reader.export_bundle(bundle.bundle_id)
            assert harness_export.call_args.kwargs["compliance_track"] == "commercial_clean"
            assert harness_export.call_args.kwargs["locked_scopes"] == original.locked_scopes
            assert harness_export.call_args.kwargs["constraints"] == commercial.constraints
    with route_client(tmp_path) as client:
        view = client.get("/v1/projects/history/state").json()
        assert [c["bundle_id"] for c in view["candidates"]] == view["latest_report"]["recommended_bundle_ids"] == expected
        with patch("mrnax.cloud.gateway.export_construct_package", wraps=export_construct_package) as http_export:
            response = client.get(f"/v1/projects/history/bundles/{bundle.bundle_id}/export")
            assert response.status_code == (409 if sandbox else 200)
            if sandbox:
                http_export.assert_not_called()
            else:
                assert http_export.call_args.kwargs["compliance_track"] == "commercial_clean"
                assert http_export.call_args.kwargs["locked_scopes"] == original.locked_scopes
    assert project_json_bytes(writer.store_dir) == before

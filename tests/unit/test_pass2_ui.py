import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mrnax.cloud.cas_store import ExecutionCASStore
from mrnax.cloud.gateway import (
    AgentChatRequest,
    _build_preset_reference_r,
    _serialize_candidate_card,
    create_control_plane_app,
)
from mrnax.harness.dialogue import LiveDialogueAgent
from mrnax.harness.state_machine import ProjectHarnessSession
from mrnax.schemas.construct_bundle import ConstructBundle
from mrnax.schemas.evaluation_report import BundleEvaluation, EvaluationReport, MetricResult


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def two_chain_bundle():
    first = _build_preset_reference_r(ROOT / "registry_seed").molecules[0]
    second = first.model_copy_validated(molecule_id="second")
    return ConstructBundle.create(
        bundle_id="two-chain", parent_bundle_id=None, derivation_action="reference_import",
        co_expression_mode="lnp_co_encapsulation", molecules=[first, second],
        target_stoichiometry={first.molecule_id: 1.0, second.molecule_id: 2.0},
    )


def test_card_exposes_each_molecule_and_its_target(two_chain_bundle):
    card = _serialize_candidate_card(two_chain_bundle, {})
    assert card["target_stoichiometry"] == two_chain_bundle.target_stoichiometry
    assert card["molecules_count"] == 2
    for molecule, view in zip(two_chain_bundle.molecules, card["molecules"], strict=True):
        assert view["molecule_id"] == molecule.molecule_id
        assert view["full_sequence"] == molecule.sequence
        assert view["target"] == molecule.effective_translation_target.model_dump(mode="json")
    assert card["pareto_rank"] is None


@pytest.fixture
def partial_evaluation(two_chain_bundle):
    first = MetricResult(
        metric_id="fixture", value=0.9, direction="maximize", unit="units", evaluated_scopes=["cds"],
        tool_name="fixture", tool_version="1", chemistry_evaluation_mode="native_chemistry_validated",
        participated_in="pareto_selection_or_ranking", notes="aggregated_2_chains(worst_case)",
    )
    return BundleEvaluation(
        bundle_id=two_chain_bundle.bundle_id, version_digest=two_chain_bundle.version_digest,
        passed_all_hard_gates=True, pareto_rank=1, metrics=[first, first.model_copy_validated(value=None)],
    )


def test_card_with_incomplete_chain_measurement_has_no_score_or_rank(two_chain_bundle, partial_evaluation):
    card = _serialize_candidate_card(two_chain_bundle, {two_chain_bundle.bundle_id: partial_evaluation})
    assert card["pareto_rank"] is None
    assert card["metrics"] == {}
    assert all(m["value"] is None and "aggregated_" not in m["notes"] for m in card["metric_details"])


def test_unsupported_dialogue_provider_rejected_before_dispatch(tmp_path):
    assert AgentChatRequest(message="hello", provider=" OpenAI ").provider == "openai"
    with pytest.raises(ValueError, match="Unsupported dialogue provider"):
        LiveDialogueAgent(None, provider="anthropic")
    app = create_control_plane_app(
        ExecutionCASStore(tmp_path / "cas.sqlite"), projects_root=tmp_path / "projects",
        seed_dir=ROOT / "registry_seed",
    )
    res = TestClient(app).post("/v1/projects/p/agent-chat", json={"message": "hello", "provider": "anthropic"})
    assert res.status_code == 422


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js unavailable")
def test_browser_logic_with_node():
    subprocess.run(["node", "--test", "tests/web/ui_logic.test.cjs"], cwd=ROOT, check=True)


@pytest.mark.parametrize("include_complete", [False, True])
def test_historical_partial_report_cannot_recommend_in_state_or_chat(
    tmp_path, monkeypatch, two_chain_bundle, partial_evaluation, include_complete,
):
    projects = tmp_path / "projects"
    session = ProjectHarnessSession(project_id="history", store_dir=projects / "history", seed_dir=ROOT / "registry_seed")
    intent = session.create_initial_intent(
        turn_id="first", user_utterance="Inspect historical results",
        protein_sequence=two_chain_bundle.molecules[0].effective_translation_target.protein_sequence,
        reference_bundle=two_chain_bundle,
    )
    evaluations = [partial_evaluation]
    expected_ids = []
    if include_complete:
        complete = two_chain_bundle.model_copy_validated(bundle_id="complete")
        session.register_bundle(complete, originating_intent_id=intent.intent_id)
        expected_ids.append(complete.bundle_id)
        evaluations.append(partial_evaluation.model_copy_validated(
            bundle_id=complete.bundle_id,
            metrics=[
                partial_evaluation.metrics[0].model_copy_validated(
                    metric_id="kozak_unpaired_prob", chain_values=[0.9, 0.9]
                ),
                partial_evaluation.metrics[0].model_copy_validated(
                    metric_id="cds_mean_window_mfe", direction="minimize", value=-18.0, chain_values=[-18.0, -18.0]
                ),
            ],
        ))
    report = EvaluationReport(
        report_id="rep_stored", execution_id="historical", intent_id=intent.intent_id, intent_version=intent.intent_version,
        compliance_track=intent.compliance_track, outcome_status="COMPLETED_TIE" if include_complete else "COMPLETED_WITH_CANDIDATES",
        candidate_evaluations=evaluations, recommended_bundle_ids=[e.bundle_id for e in evaluations],
        failed_diagnostics=[], fto_alerts=[], pre_synthesis_advisories=[], action_transparency=[],
    )
    report_path = session.store_dir / "reports" / "rep_stored.json"
    original = report.model_dump_json().encode()
    report_path.write_bytes(original)
    http = TestClient(create_control_plane_app(
        ExecutionCASStore(tmp_path / "cas.sqlite"), projects_root=projects, seed_dir=ROOT / "registry_seed",
        access_token="test-admin",
    ), headers={"X-MRNAX-Access-Token": "test-admin"})
    monkeypatch.setattr(LiveDialogueAgent, "send_user_message", lambda *args, **kwargs: "fixture")
    for res, key in (
        (http.get("/v1/projects/history/state"), "latest_report"),
        (http.post("/v1/projects/history/agent-chat", json={"message": "Inspect"}), "report"),
    ):
        assert res.status_code == 200
        payload = res.json()
        view = EvaluationReport.model_validate(payload[key])
        assert view.recommended_bundle_ids == expected_ids
        assert [e.bundle_id for e in view.candidate_evaluations] == expected_ids
        assert [c["bundle_id"] for c in payload["candidates"]] == expected_ids
        assert view.outcome_status == (
            "COMPLETED_WITH_CANDIDATES" if include_complete else "UNCOMPARABLE_MISSING_APPROVED_COMPONENT"
        )
    assert report_path.read_bytes() == original

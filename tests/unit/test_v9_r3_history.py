"""Round-3 release controls for persisted evidence, producer identity and wild-type receipts."""

import hashlib
import json
from copy import deepcopy

from fastapi.testclient import TestClient
import pytest

from mrnax.cloud.cas_store import ExecutionCASStore
from mrnax.cloud.gateway import _build_preset_reference_r, create_control_plane_app
from mrnax.constraints.fto_guard import MethodFTOGuard
from mrnax.constraints.registry import ScopedConstraintEngine
from mrnax.engine.pareto_director import project_report_for_intent, run_design_pipeline
from mrnax.engine.profiles import COMPONENT_CATALOG, resolve_default_seed_dir, resolve_tool_name_to_component_id
from mrnax.harness.state_machine import ProjectHarnessSession
from mrnax.harness.tools import AtomicToolbox
from mrnax.io_insdc.genbank_io import ExportConstraintViolationError, export_construct_package
from mrnax.schemas.construct_bundle import ConstructBundle
from mrnax.schemas.design_intent import RuntimePolicySpec

SEED = resolve_default_seed_dir()
F4A = MethodFTOGuard.BOUNDARY_IDS[3]


@pytest.fixture(scope="module")
def reporter(tmp_path_factory):
    session = ProjectHarnessSession(project_id="fixture", store_dir=tmp_path_factory.mktemp("reporter"), seed_dir=SEED)
    ref = _build_preset_reference_r(SEED)
    target = ref.molecules[0].effective_translation_target
    template = session.create_initial_intent(
        turn_id="t", user_utterance="Evaluate reporter", protein_sequence=target.protein_sequence,
        compliance_track="commercial_clean", uridine_chemistry="m1psi", use_commercial_surrogate=True,
    )
    intent = template.model_copy_validated(
        project_id="history", constraints=[], runtime_policy=RuntimePolicySpec(
            enabled_generators=[], enabled_evaluators=["surrogate_v1_clean"], allow_cached_candidates=True
        )
    )
    bundles, report, manifest = run_design_pipeline(intent, cached_candidates=[ref], seed_dir=SEED)
    assert report.recommended_bundle_ids == [ref.bundle_id]
    return bundles[0], intent, report, manifest


def _persist(tmp_path, bundle, intent, report_data):
    session = ProjectHarnessSession(project_id=intent.project_id, store_dir=tmp_path/intent.project_id, seed_dir=SEED)
    session._persist_intent(intent)
    session.register_bundle(bundle, originating_intent_id=intent.intent_id)
    report_path = session.store_dir/"reports"/f"{report_data['report_id']}.json"
    report_path.write_text(json.dumps(report_data, indent=3) + "\n", encoding="utf-8")
    return session, report_path


def _check_routes(tmp_path, bundle, intent, report_path, *, allowed, ranked):
    original = report_path.read_bytes()
    session = ProjectHarnessSession(project_id=intent.project_id, store_dir=report_path.parent.parent, seed_dir=SEED)
    projected = project_report_for_intent(
        session.latest_report(), {bundle.bundle_id: bundle}, intent, seed_dir=SEED, disallow_sandbox_override=True
    )
    assert [e.bundle_id for e in projected.candidate_evaluations] == ([bundle.bundle_id] if allowed else [])
    assert projected.recommended_bundle_ids == ([bundle.bundle_id] if allowed and ranked else [])
    if allowed:
        AtomicToolbox(session).export_bundle(bundle.bundle_id)
        assert len(list((session.store_dir/"exports"/bundle.bundle_id).iterdir())) == 4
    else:
        with pytest.raises((ValueError, ExportConstraintViolationError)):
            AtomicToolbox(session).export_bundle(bundle.bundle_id)
        assert not (session.store_dir/"exports"/bundle.bundle_id).exists()
    app = create_control_plane_app(ExecutionCASStore(tmp_path/"cas.sqlite"), projects_root=tmp_path,
                                   seed_dir=SEED, access_token="fixture")
    with TestClient(app, headers={"X-MRNAX-Access-Token": "fixture"}) as client:
        res = client.get(f"/v1/projects/{intent.project_id}/bundles/{bundle.bundle_id}/export")
        state = client.get(f"/v1/projects/{intent.project_id}/state")
    assert res.status_code == (200 if allowed else 409), res.text
    assert state.status_code == 200
    public = state.json()
    assert public["latest_report"] == projected.model_dump(mode="json")
    for trace in projected.action_transparency:
        if trace.stage_applied == "ranking":
            assert set(trace.changed_candidate_ids) <= set(projected.recommended_bundle_ids)
    assert original == report_path.read_bytes()
    return public


@pytest.mark.parametrize("ranked", [True, False], ids=["ranked", "no_objectives"])
@pytest.mark.parametrize("case", ["current", "foreign", "missing", "null", "stale", "restricted", "unknown"])
def test_historical_candidate_export(reporter, tmp_path, ranked, case):
    bundle, intent, report, _ = reporter
    if not ranked:
        intent = intent.model_copy_validated(objectives=[])
        report = report.model_copy_validated(recommended_bundle_ids=[])
    data = report.model_dump(mode="json")
    evaluation = data["candidate_evaluations"][0]
    if case == "foreign":
        evaluation["version_digest"] = "0" * 64
    elif case == "missing":
        del evaluation["version_digest"]
    elif case == "null":
        evaluation["version_digest"] = None
    elif case == "stale":
        data["rule_snapshot_version"] = "rules_commercial_clean@stale"
    elif case in ("restricted", "unknown"):
        evaluation["metrics"][0]["tool_name"] = "RiboNN" if case == "restricted" else "unknown surrogate"
    _, path = _persist(tmp_path, bundle, intent, data)
    _check_routes(tmp_path, bundle, intent, path, allowed=case == "current", ranked=ranked)


@pytest.mark.parametrize(("tool", "expected"), [
    ("RiboNN", "ribonn_noncommercial"), ("UTR-LM", "utr_lm_gpl3"),
    ("Helix-mRNA", "helix_mrna_helical"), ("ribonn_noncommercial", "ribonn_noncommercial"),
    ("unknown surrogate", None), ("CommercialCleanSurrogateV1:RiboNN", None), ("", None),
    ("CommercialCleanSurrogateV1:utr5_mrl_lgbm", "surrogate_v1_clean"),
])
@pytest.mark.parametrize("metric", ["utr5_mrl_pred", "structrfm_cds_median_shape", "codontransformer_masked_pll"])
def test_metric_does_not_substitute_for_producer(tool, expected, metric):
    assert resolve_tool_name_to_component_id(tool, metric) == expected


@pytest.mark.parametrize("case", ["verified", "reformatted_intent", "reformatted_report", "missing_report",
                                  "wrong_intent", "missing_intent", "missing_intent_field",
                                  "canonical_and_file", "canonical_only", "wrong_file", "wrong_canonical"])
def test_legacy_snapshot_hydration_requires_both_file_hashes(reporter, tmp_path, case):
    bundle, intent, report, manifest = reporter
    raw = report.model_dump(mode="json")
    raw.pop("rule_snapshot_version")
    session, path = _persist(tmp_path, bundle, intent, raw)
    ipath = next((session.store_dir/"intents").glob("*.json"))
    man = manifest.model_dump(mode="json")
    man["input_intent_digest"] = hashlib.sha256(ipath.read_bytes()).hexdigest()
    man["artifacts"][0]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    if case in {"canonical_and_file", "canonical_only", "wrong_file", "wrong_canonical"}:
        man["input_intent_digest"] = intent.digest() if case != "wrong_canonical" else "0" * 64
        if case != "canonical_only":
            content = ipath.read_bytes()
            man["artifacts"].append({
                "logical_name": "input_intent", "local_relpath": str(ipath.relative_to(session.store_dir)),
                "sha256": hashlib.sha256(content).hexdigest() if case != "wrong_file" else "0" * 64,
                "md5": hashlib.md5(content).hexdigest(), "bytes": len(content),
            })
    if case == "reformatted_intent":
        ipath.write_bytes(ipath.read_bytes() + b"\n")
    elif case == "reformatted_report":
        path.write_bytes(path.read_bytes() + b"\n")
    elif case == "missing_report":
        man["artifacts"] = []
    elif case == "wrong_intent":
        man["input_intent_digest"] = "0" * 64
    elif case == "missing_intent":
        man["input_intent_digest"] = ""
    elif case == "missing_intent_field":
        del man["input_intent_digest"]
    (session.store_dir/"run_manifest.json").write_text(json.dumps(man), encoding="utf-8")
    _check_routes(tmp_path, bundle, intent, path, allowed=case in {"verified", "canonical_and_file"}, ranked=True)


def _with_wildtype(bundle, reference):
    mol = bundle.molecules[0]
    mol = mol.model_copy_validated(
        effective_translation_target=mol.effective_translation_target.model_copy_validated(wildtype_cds_rna=reference)
    )
    return ConstructBundle.create(**bundle.model_dump(exclude={"version_digest", "molecules"}), molecules=[mol])


@pytest.mark.parametrize("source", ["intent", "embedded"])
@pytest.mark.parametrize("reference", ["", "   ", "NNN", None, "valid"])
def test_explicit_wildtype_through_replay_and_export(reporter, tmp_path, source, reference):
    bundle, intent, report, _ = reporter
    if reference == "valid":
        reference = bundle.molecules[0].get_segment_sequence("cds")
    if source == "embedded":
        bundle = ConstructBundle.model_validate_json(_with_wildtype(bundle, reference).model_dump_json())
    else:
        intent = intent.model_copy_validated(protein_targets=[
            intent.protein_targets[0].model_copy_validated(wildtype_cds_rna=reference)
        ])
    allowed = reference is None or reference == bundle.molecules[0].get_segment_sequence("cds")
    _, replay, manifest = run_design_pipeline(intent, cached_candidates=[bundle], seed_dir=SEED)
    receipt = manifest.optimization_mode_audit
    assert receipt.method_fto_boundary_checks[F4A]["status"] == ("PASS" if allowed else "BLOCK")
    assert bool(replay.recommended_bundle_ids) == allowed
    if allowed and reference is not None:
        assert receipt.input_origin_type == "wildtype_mrna_reference"
        assert receipt.wildtype_c_optimizable_replacement_fraction == 0.0
    out = tmp_path/"direct"
    args = dict(constraint_engine=ScopedConstraintEngine.from_seed_dir(SEED), constraints=intent.constraints,
                compliance_track="commercial_clean", expected_chemistry=intent.uridine_chemistry,
                expected_protein_targets=intent.protein_targets)
    if allowed:
        export_construct_package(bundle, out, **args)
        assert len(list(out.iterdir())) == 4
    else:
        with pytest.raises(ExportConstraintViolationError, match="CUREVAC_INVALID"):
            export_construct_package(bundle, out, **args)
        assert not out.exists()
    data = report.model_dump(mode="json")
    data["candidate_evaluations"][0]["version_digest"] = bundle.version_digest
    _, path = _persist(tmp_path, bundle, intent, data)
    _check_routes(tmp_path, bundle, intent, path, allowed=allowed, ranked=True)


@pytest.mark.parametrize("separate_reports", [False, True])
def test_conflicting_bundle_identity_cannot_borrow_a_valid_evaluation(reporter, tmp_path, separate_reports):
    bundle, intent, report, _ = reporter
    foreign = report.candidate_evaluations[0].model_copy_validated(version_digest="0" * 64)
    conflicting = report.model_copy_validated(candidate_evaluations=[foreign, *report.candidate_evaluations])
    if not separate_reports:
        _, path = _persist(tmp_path, bundle, intent, conflicting.model_dump(mode="json"))
        _check_routes(tmp_path, bundle, intent, path, allowed=False, ranked=True)
        return
    session, path = _persist(tmp_path, bundle, intent, report.model_dump(mode="json"))
    other = report.model_copy_validated(report_id="rep_other", candidate_evaluations=[foreign])
    (path.parent/"rep_other.json").write_text(other.model_dump_json(), encoding="utf-8")
    original = path.read_bytes()
    session = ProjectHarnessSession(project_id=intent.project_id, store_dir=session.store_dir, seed_dir=SEED)
    with pytest.raises(ValueError, match="digest"):
        AtomicToolbox(session).export_bundle(bundle.bundle_id)
    app = create_control_plane_app(ExecutionCASStore(tmp_path/"cas.sqlite"), projects_root=tmp_path,
                                   seed_dir=SEED, access_token="fixture")
    with TestClient(app, headers={"X-MRNAX-Access-Token": "fixture"}) as client:
        res = client.get(f"/v1/projects/{intent.project_id}/bundles/{bundle.bundle_id}/export")
    assert res.status_code == 409
    assert path.read_bytes() == original


def test_display_only_metric_allows_unranked_export(reporter, tmp_path):
    bundle, intent, report, _ = reporter
    intent = intent.model_copy_validated(objectives=[])
    metric = report.candidate_evaluations[0].metrics[0].model_copy_validated(
        metric_id="cds_gc_fraction", tool_name="mRNAxSeqStats", participated_in="display_only", evaluated_scopes=["cds"]
    )
    ev = report.candidate_evaluations[0].model_copy_validated(metrics=[metric])
    report = report.model_copy_validated(candidate_evaluations=[ev], recommended_bundle_ids=[])
    _, path = _persist(tmp_path, bundle, intent, report.model_dump(mode="json"))
    _check_routes(tmp_path, bundle, intent, path, allowed=True, ranked=False)


@pytest.mark.parametrize("held_first", [False, True])
def test_zero_objective_history_keeps_only_clean_evaluation_for_same_version(reporter, tmp_path, held_first):
    bundle, intent, report, _ = reporter
    intent = intent.model_copy_validated(objectives=[])
    data = report.model_dump(mode="json")
    data["recommended_bundle_ids"] = []
    clean, = data["candidate_evaluations"]
    held = deepcopy(clean)
    held["metrics"][0]["tool_name"] = "RiboNN"
    data["candidate_evaluations"] = [held, clean] if held_first else [clean, held]
    _, path = _persist(tmp_path, bundle, intent, data)
    public = _check_routes(tmp_path, bundle, intent, path, allowed=True, ranked=False)
    assert public["latest_report"]["candidate_evaluations"] == [clean]
    assert [card["bundle_id"] for card in public["candidates"]] == [bundle.bundle_id]
    assert "RiboNN" not in json.dumps(public["candidates"])


def test_each_public_card_reports_its_own_evaluation_for_same_version(reporter, tmp_path):
    bundle, intent, report, _ = reporter
    intent = intent.model_copy_validated(objectives=[])
    evaluation, = report.candidate_evaluations
    stability = next(metric for metric in evaluation.metrics if metric.metric_id == "cds_stability_pred")
    rows = [evaluation.model_copy_validated(metrics=[stability.model_copy_validated(value=value)])
            for value in (0.25, 0.75)]
    report = report.model_copy_validated(candidate_evaluations=rows, recommended_bundle_ids=[])
    _, path = _persist(tmp_path, bundle, intent, report.model_dump(mode="json"))
    original = path.read_bytes()
    app = create_control_plane_app(ExecutionCASStore(tmp_path / "cas.sqlite"), projects_root=tmp_path,
                                   seed_dir=SEED, access_token="fixture")
    with TestClient(app, headers={"X-MRNAX-Access-Token": "fixture"}) as client:
        response = client.get(f"/v1/projects/{intent.project_id}/state")
    assert response.status_code == 200
    state = response.json()
    assert state["latest_report"]["candidate_evaluations"] == [row.model_dump(mode="json") for row in rows]
    assert [card["metrics"][stability.metric_id] for card in state["candidates"]] == [0.25, 0.75]
    assert [card["metric_details"] for card in state["candidates"]] == [
        [row.metrics[0].model_dump(mode="json")] for row in rows
    ]
    assert path.read_bytes() == original


def test_partial_historical_ranking_withdraws_only_suppressed_candidate(reporter, tmp_path):
    bundle, intent, report, _ = reporter
    data = report.model_dump(mode="json")
    held_bundle = bundle.model_copy_validated(bundle_id="withdrawn_history_row")
    held = deepcopy(data["candidate_evaluations"][0])
    held["bundle_id"] = held_bundle.bundle_id
    held["metrics"][0]["tool_name"] = "RiboNN"
    data["candidate_evaluations"].append(held)
    data["recommended_bundle_ids"].append(held_bundle.bundle_id)
    for trace in data["action_transparency"]:
        if trace["stage_applied"] == "ranking":
            trace["changed_candidate_ids"].append(held_bundle.bundle_id)
            trace["proof_diff_summary"] += f" Selected {held_bundle.bundle_id}."
    session, path = _persist(tmp_path, bundle, intent, data)
    session.register_bundle(held_bundle, originating_intent_id=intent.intent_id)
    public = _check_routes(tmp_path, bundle, intent, path, allowed=True, ranked=True)
    ranking = [t for t in public["latest_report"]["action_transparency"] if t["stage_applied"] == "ranking"]
    assert ranking and all(t["changed_candidate_ids"] == [bundle.bundle_id] for t in ranking)
    assert held_bundle.bundle_id not in json.dumps(ranking)


@pytest.mark.parametrize(("producer", "metric_id", "scopes", "allowed"), [
    ("CommercialCleanSurrogateV1:utr5_mrl_lgbm", "cds_stability_pred", ["cds"], False),
    ("CommercialCleanSurrogateV1_utr5_mrl_lgbm", "cds_stability_pred", ["cds"], False),
    ("COMMERCIALCLEANsurrogateV1-UTR5-MRL-LGBM", "cds_stability_pred", ["cds"], False),
    ("CommercialCleanSurrogateV1_cds_stability_lgbm", "cds_stability_pred", ["five_prime_utr"], False),
    ("CommercialCleanSurrogateV1_cds_stability_lgbm", "cds_stability_pred", ["three_prime_utr"], False),
    ("CommercialCleanSurrogateV1:fastutr_3utr_lgbm", "utr3_stability_pred", ["five_prime_utr"], False),
    ("CommercialCleanSurrogateV1:saluki_consensus_lgbm", "consensus_halflife_pred", ["cds"], False),
    ("Orthrus6TrackMIT", "orthrus_halflife_pred", ["five_prime_utr"], False),
    ("StructRFMNativeScorer:mauger2019_probe", "structrfm_cds_median_shape", ["three_prime_utr"], False),
    ("CommercialCleanSurrogateV1-CDS-STABILITY-LGBM", "cds_stability_pred", ["cds"], True),
    ("CommercialCleanSurrogateV1:cds_stability_lgbm", "cds_stability_pred", ["full_transcript"], False),
    ("CommercialCleanSurrogateV1:fastutr_3utr_lgbm", "utr3_stability_pred", ["three_prime_utr"], True),
    ("CommercialCleanSurrogateV1:saluki_consensus_lgbm", "consensus_halflife_pred", ["full_transcript"], True),
    ("Orthrus6TrackMIT", "orthrus_halflife_pred", ["five_prime_utr", "cds", "three_prime_utr"], True),
    ("StructRFMNativeScorer:mauger2019_probe", "structrfm_cds_median_shape", ["cds"], True),
])
def test_historical_partner_agrees_with_producer_head_and_scope(
    reporter, tmp_path, producer, metric_id, scopes, allowed
):
    bundle, intent, report, _ = reporter
    intent = intent.model_copy_validated(objectives=[
        intent.objectives[0].model_copy_validated(metric_id="codontransformer_syn_loglik")
    ])
    ev, = report.candidate_evaluations
    codon = ev.metrics[0].model_copy_validated(
        tool_name="CodonTransformer", metric_id="codontransformer_syn_loglik", evaluated_scopes=["cds"]
    )
    partner = codon.model_copy_validated(tool_name=producer, metric_id=metric_id, evaluated_scopes=scopes)
    data = report.model_copy_validated(candidate_evaluations=[ev.model_copy_validated(metrics=[codon, partner])])
    _, path = _persist(tmp_path, bundle, intent, data.model_dump(mode="json"))
    _check_routes(tmp_path, bundle, intent, path, allowed=allowed, ranked=True)


def test_valid_partner_does_not_validate_a_contradictory_measurement(reporter, tmp_path):
    bundle, intent, report, _ = reporter
    ev, = report.candidate_evaluations
    stability = next(m for m in ev.metrics if m.metric_id == "cds_stability_pred")
    codon = stability.model_copy_validated(tool_name="CodonTransformer", metric_id="codontransformer_syn_loglik")
    contradictory = stability.model_copy_validated(tool_name="CommercialCleanSurrogateV1_utr5_mrl_lgbm")
    data = report.model_copy_validated(candidate_evaluations=[
        ev.model_copy_validated(metrics=[*ev.metrics, codon, contradictory])
    ])
    _, path = _persist(tmp_path, bundle, intent, data.model_dump(mode="json"))
    _check_routes(tmp_path, bundle, intent, path, allowed=False, ranked=True)


@pytest.mark.parametrize("case", ["orphan", "uncompleted", "wrong_revision", "wrong_track"])
def test_inconsistent_historical_associations_cannot_support_export(reporter, tmp_path, case):
    bundle, intent, report, _ = reporter
    if case == "orphan":
        changed = report.model_copy(update={"candidate_evaluations": []})
    elif case == "uncompleted":
        changed = report.model_copy(update={"outcome_status": "COMPLETED_ZERO_QUALIFIED"})
    elif case == "wrong_revision":
        changed = report.model_copy_validated(intent_version=intent.intent_version + 1)
    else:
        changed = report.model_copy_validated(compliance_track="research_max")
    projected = project_report_for_intent(changed, {bundle.bundle_id: bundle}, intent, seed_dir=SEED)
    assert projected.recommended_bundle_ids == [] and projected.candidate_evaluations == []
    _, path = _persist(tmp_path, bundle, intent, changed.model_dump(mode="json"))
    _check_routes(tmp_path, bundle, intent, path, allowed=False, ranked=True)


def test_unwitnessed_generated_candidate_requires_evaluation(reporter, tmp_path):
    ref, intent, _, _ = reporter
    session = ProjectHarnessSession(project_id=intent.project_id, store_dir=tmp_path/intent.project_id, seed_dir=SEED)
    session._persist_intent(intent)
    session.register_bundle(ref, originating_intent_id=intent.intent_id)
    generated = ref.model_copy_validated(bundle_id="generated", derivation_action="de_novo_design")
    session.register_bundle(generated, originating_intent_id=intent.intent_id)
    AtomicToolbox(session).export_bundle(ref.bundle_id)
    with pytest.raises(ValueError):
        AtomicToolbox(session).export_bundle(generated.bundle_id)
    app = create_control_plane_app(ExecutionCASStore(tmp_path/"cas.sqlite"), projects_root=tmp_path,
                                   seed_dir=SEED, access_token="fixture")
    with TestClient(app, headers={"X-MRNAX-Access-Token": "fixture"}) as client:
        assert client.get(f"/v1/projects/{intent.project_id}/bundles/{generated.bundle_id}/export").status_code == 409
        assert client.get(f"/v1/projects/{intent.project_id}/bundles/{ref.bundle_id}/export").status_code == 200


def test_current_harness_manifest_pins_persisted_intent_and_report_files(tmp_path):
    session = ProjectHarnessSession(project_id="fresh", store_dir=tmp_path/"fresh", seed_dir=SEED)
    template = session.create_initial_intent(turn_id="create", user_utterance="Reporter control",
        protein_sequence="M" + "ALESK"*12, compliance_track="commercial_clean", uridine_chemistry="unmodified_u")
    intent = session._persist_intent(template.model_copy_validated(
        intent_id="current", intent_version=2, parent_intent_id=template.intent_id, objectives=[],
        runtime_policy=RuntimePolicySpec(enabled_generators=["stage0_random"], enabled_evaluators=["surrogate_v1_clean"])
    ))
    bundles = session.execute_design(intent)
    report = session.latest_report()
    assert len(bundles) == 1 and report.recommended_bundle_ids == []
    man_path = session.store_dir/"manifests"/(report.execution_id + ".json")
    manifest = json.loads(man_path.read_bytes())
    hashes = {a["logical_name"]: a["sha256"] for a in manifest["artifacts"]}
    report_path = session.store_dir/"reports"/(report.report_id + ".json")
    intent_path = next((session.store_dir/"intents").glob("*_current.json"))
    assert hashes["evaluation_report"] == hashlib.sha256(report_path.read_bytes()).hexdigest()
    assert hashes["input_intent"] == hashlib.sha256(intent_path.read_bytes()).hexdigest()
    assert manifest["input_intent_digest"] == intent.digest()
    _check_routes(tmp_path, bundles[0], intent, report_path, allowed=True, ranked=False)


@pytest.mark.parametrize(("producer", "scopes", "allowed"), [
    ("CommercialCleanSurrogateV1:utr5_mrl_lgbm", ["five_prime_utr"], True),
    ("CommercialCleanSurrogateV1:utr5_mrl_lgbm", ["cds"], False),
    ("CommercialCleanSurrogateV1:utr5_mrl_lgbm", ["full_transcript"], False),
    ("CommercialCleanSurrogateV1:utr5_mrl_lgbm", ["full_transcript", "five_prime_utr"], False),
    ("CommercialCleanSurrogateV1:cds_stability_lgbm", ["five_prime_utr"], False),
    ("CommercialCleanSurrogateV1:fastutr_3utr_lgbm", ["five_prime_utr"], False),
    ("Orthrus6TrackMIT", ["five_prime_utr"], False),
])
def test_historical_mrl_objective_requires_its_producer_and_scope(reporter, tmp_path, producer, scopes, allowed):
    bundle, intent, report, _ = reporter
    intent = intent.model_copy_validated(objectives=[intent.objectives[0].model_copy_validated(metric_id="utr5_mrl_pred")])
    ev, = report.candidate_evaluations
    mrl = next(m for m in ev.metrics if m.metric_id == "utr5_mrl_pred")
    ev = ev.model_copy_validated(metrics=[mrl.model_copy_validated(tool_name=producer, evaluated_scopes=scopes)])
    stored = report.model_copy_validated(candidate_evaluations=[ev])
    _, path = _persist(tmp_path, bundle, intent, stored.model_dump(mode="json"))
    _check_routes(tmp_path, bundle, intent, path, allowed=allowed, ranked=True)


@pytest.mark.parametrize(("chemistry", "allowed"), [
    ("unmodified_u", True), ("m1psi", True), ("5mou", True), ("pseudouridine", False), ("custom_mod", False),
])
def test_historical_native_shape_requires_a_shipped_probe(reporter, tmp_path, chemistry, allowed):
    ref, intent, report, _ = reporter
    mol, = ref.molecules
    mol = mol.model_copy_validated(chemistry_track=mol.chemistry_track.model_copy_validated(uridine_chemistry=chemistry))
    bundle = ConstructBundle.create(bundle_id=ref.bundle_id, parent_bundle_id=None, derivation_action=ref.derivation_action,
                                    molecules=[mol], target_stoichiometry=ref.target_stoichiometry)
    intent = intent.model_copy_validated(uridine_chemistry=chemistry, objectives=[
        intent.objectives[0].model_copy_validated(metric_id="structrfm_cds_median_shape")
    ])
    ev, = report.candidate_evaluations
    shape = ev.metrics[0].model_copy_validated(tool_name="StructRFMNativeScorer:mauger2019_probe",
        metric_id="structrfm_cds_median_shape", evaluated_scopes=["cds"], chemistry_evaluation_mode="native_chemistry_validated")
    stored = report.model_copy_validated(candidate_evaluations=[
        ev.model_copy_validated(version_digest=bundle.version_digest, metrics=[shape])])
    _, path = _persist(tmp_path, bundle, intent, stored.model_dump(mode="json"))
    _check_routes(tmp_path, bundle, intent, path, allowed=allowed, ranked=True)


@pytest.mark.parametrize(("tool", "metric", "scope", "track", "eligible"), [
    ("RiboNN", "ribonn_translation_efficiency", "full_transcript", "research_max", True),
    ("UTR-LM", "utr_lm_mrl", "five_prime_utr", "research_max", True),
    ("Helix-mRNA", "helix_mrna_next_token_loss", "full_transcript", "research_max", True),
    ("CodonBERT", "codonbert_masked_codon_loss", "cds", "research_max", True),
    ("RhoFold+", "rhofold_mean_plddt", "full_transcript", "research_max", True),
    ("RiboNN", "ribonn_translation_efficiency", "cds", "research_max", False),
    ("RiboNN", "utr5_mrl_pred", "five_prime_utr", "research_max", False),
    ("RiboNN", "ribonn_translation_efficiency", "full_transcript", "commercial_clean", False),
    ("RibonanzaNet", "ribonanzanet_mean_2a3_reactivity", "full_transcript", "research_max", True),
    ("RibonanzaNet", "ribonanzanet_mean_2a3_reactivity", "full_transcript", "commercial_clean", False),
    *[(component, "archived_observation", "full_transcript", track, track == "research_max")
      for component in COMPONENT_CATALOG for track in ("research_max", "commercial_clean")],
])
def test_historical_research_observation(reporter, tmp_path, tool, metric, scope, track, eligible):
    bundle, intent, report, _ = reporter
    if track == "research_max":
        component = resolve_tool_name_to_component_id(tool)
        assert component is not None
        policy = intent.runtime_policy.model_copy_validated(enabled_evaluators=["surrogate_v1_clean", component])
        intent = intent.model_copy_validated(compliance_track=track, runtime_policy=policy)
        report = report.model_copy_validated(compliance_track=track, rule_snapshot_version="rules_research_v1@1.0.0")
    ev, = report.candidate_evaluations
    observation = ev.metrics[0].model_copy_validated(
        tool_name=tool, metric_id=metric, evaluated_scopes=[scope], value=0.5, unit="arb",
        direction="informational", participated_in="display_only",
    )
    report = report.model_copy_validated(candidate_evaluations=[ev.model_copy_validated(metrics=[*ev.metrics, observation])])
    _, path = _persist(tmp_path, bundle, intent, report.model_dump(mode="json"))
    raw = path.read_bytes()
    app = create_control_plane_app(ExecutionCASStore(tmp_path/"cas.sqlite"), projects_root=tmp_path,
                                   seed_dir=SEED, access_token="fixture")
    with TestClient(app, headers={"X-MRNAX-Access-Token": "fixture"}) as client:
        state = client.get(f"/v1/projects/{intent.project_id}/state")
        assert state.status_code == 200
        if track == "commercial_clean":
            assert client.get(f"/v1/projects/{intent.project_id}/bundles/{bundle.bundle_id}/export").status_code == 409
    assert path.read_bytes() == raw
    public = state.json()
    assert public["latest_report"]["recommended_bundle_ids"] == ([bundle.bundle_id] if eligible else [])
    assert len(public["candidates"]) == int(eligible)
    if eligible:
        assert any(m["tool_name"] == tool and m["participated_in"] == "display_only"
                   for m in public["candidates"][0]["metric_details"])
        ranked = observation.model_copy_validated(participated_in="pareto_selection_or_ranking")
        promoted = report.model_copy_validated(candidate_evaluations=[ev.model_copy_validated(metrics=[*ev.metrics, ranked])])
        projection = project_report_for_intent(promoted, {bundle.bundle_id: bundle}, intent, seed_dir=SEED)
        assert projection.recommended_bundle_ids == []

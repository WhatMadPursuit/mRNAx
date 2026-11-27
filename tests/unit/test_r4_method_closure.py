"""Genetic code, registry flag, and multichain input regressions from Round 4."""

import json

from fastapi.testclient import TestClient
import pytest

from mrnax.cloud.cas_store import ExecutionCASStore
from mrnax.cloud.gateway import _build_preset_reference_r, create_control_plane_app
from mrnax.constraints.fto_guard import SequenceFTOGuard
from mrnax.constraints.registry import ScopedConstraintEngine
from mrnax.engine.pareto_director import run_design_pipeline
from mrnax.engine.profiles import resolve_default_seed_dir
from mrnax.harness.state_machine import ProjectHarnessSession
from mrnax.harness.tools import AtomicToolbox
from mrnax.io_insdc.genbank_io import ExportConstraintViolationError, export_construct_package
from mrnax.schemas.construct_bundle import ConstructBundle
from mrnax.schemas.design_intent import RuntimePolicySpec

SEED = resolve_default_seed_dir()


def small_molecule(cds="AUGUGG", protein="MW", table=1, cassette=None):
    base = _build_preset_reference_r(SEED).molecules[0]
    segments = [base.get_segment("cds").model_copy_validated(start=0, end=len(cds))]
    if cassette:
        segments.append(base.get_segment("stop_cassette").model_copy_validated(
            start=len(cds), end=len(cds) + len(cassette)
        ))
    return base.model_copy_validated(
        sequence=cds + (cassette or ""), segments=segments, annotations=[],
        effective_translation_target=base.effective_translation_target.model_copy_validated(
            protein_sequence=protein, transl_table=table
        ),
    )


def make_bundle(*molecules):
    return ConstructBundle.create(
        bundle_id="reporter", parent_bundle_id=None, derivation_action="reference_import",
        molecules=list(molecules), target_stoichiometry={m.molecule_id: 1 for m in molecules},
        co_expression_mode="lnp_co_encapsulation" if len(molecules) > 1 else "single_monocistronic",
    )


def setup_intent(tmp_path, targets):
    draft = ProjectHarnessSession(project_id="code", store_dir=tmp_path/"draft", seed_dir=SEED)
    initial = draft.create_initial_intent(
        turn_id="t", user_utterance="Check reporter", protein_sequence=targets[0].protein_sequence,
        compliance_track="commercial_clean", uridine_chemistry="m1psi", use_commercial_surrogate=True,
    )
    intent = initial.model_copy_validated(
        protein_targets=targets, constraints=[], objectives=[], locked_scopes={},
        runtime_policy=RuntimePolicySpec(
            enabled_generators=[], enabled_evaluators=["surrogate_v1_clean"], allow_cached_candidates=True
        ),
    )
    session = ProjectHarnessSession(project_id="code", store_dir=tmp_path/"code", seed_dir=SEED)
    session._persist_intent(intent)
    return session, intent


@pytest.mark.parametrize("cds,table,requested,allow", [
    ("AUGUGG", 1, 1, True), ("AUGUGA", 2, 2, True),
    ("AUGUGA", 2, 1, False), ("AUGUGG", 2, 1, False), ("AUGUGG", 1, 2, False),
])
def test_requested_code_controls_cache_and_export(tmp_path, cds, table, requested, allow):
    mol = small_molecule(cds=cds, table=table)
    bundle = make_bundle(mol)
    expected = mol.effective_translation_target.model_copy_validated(transl_table=requested)
    session, intent = setup_intent(tmp_path, [expected])
    engine = ScopedConstraintEngine.from_seed_dir(SEED)
    kwargs = dict(constraints=[], compliance_track="commercial_clean", expected_protein_targets=[expected])
    assert engine.evaluate_bundle(bundle, **kwargs).passed_all_hard_gates == allow
    target_dict = expected.model_dump(exclude={"transl_table"} if requested == 1 else set())
    assert engine.evaluate_bundle(bundle, [], compliance_track="commercial_clean",
                                  expected_protein_targets=[target_dict]).passed_all_hard_gates == allow
    admitted, report, _ = run_design_pipeline(intent, cached_candidates=[bundle], seed_dir=SEED)
    assert bool(admitted) == allow
    if not allow:
        assert report.recommended_bundle_ids == []
    directory = tmp_path / "direct"
    if allow:
        export_construct_package(bundle, directory, constraint_engine=engine, **kwargs)
        assert len(list(directory.iterdir())) == 4
    else:
        with pytest.raises(ExportConstraintViolationError):
            export_construct_package(bundle, directory, constraint_engine=engine, **kwargs)
        assert not directory.exists()
    # A legacy report from before the code-agreement gate still claimed eligibility.
    original_intent = intent.model_copy_validated(protein_targets=[mol.effective_translation_target])
    _, old, _ = run_design_pipeline(original_intent, cached_candidates=[bundle], seed_dir=SEED)
    assert old.candidate_evaluations
    session.register_bundle(bundle, originating_intent_id=intent.intent_id)
    (session.store_dir/"reports"/f"{old.report_id}.json").write_text(old.model_dump_json())
    reopened = ProjectHarnessSession(project_id=intent.project_id, store_dir=session.store_dir, seed_dir=SEED)
    if allow:
        AtomicToolbox(reopened).export_bundle(bundle.bundle_id)
    else:
        with pytest.raises((ValueError, ExportConstraintViolationError)):
            AtomicToolbox(reopened).export_bundle(bundle.bundle_id)
    app = create_control_plane_app(ExecutionCASStore(tmp_path/"cas.sqlite"), projects_root=tmp_path,
                                   seed_dir=SEED, access_token="fixture")
    with TestClient(app, headers={"X-MRNAX-Access-Token": "fixture"}) as client:
        route = f"/v1/projects/{intent.project_id}"
        assert client.get(f"{route}/bundles/{bundle.bundle_id}/export").status_code == (200 if allow else 409)
        assert bool(client.get(f"{route}/state").json()["candidates"]) == allow
    if not allow:
        assert not (session.store_dir / "exports" / bundle.bundle_id).exists()


@pytest.mark.parametrize("table,stops", [(1, {"UAA", "UAG", "UGA"}), (2, {"UAA", "UAG", "AGA", "AGG"})])
@pytest.mark.parametrize("codon", ["UAA", "UAG", "UGA", "AGA", "AGG", "UGG"])
def test_stop_cassette_uses_declared_code(table, stops, codon):
    if codon in stops:
        assert small_molecule(table=table, cassette=codon).translate_cds_protein() == "MW"
    else:
        with pytest.raises(ValueError, match="stop_cassette"):
            small_molecule(table=table, cassette=codon)


@pytest.mark.parametrize("flag", ["true", "false", 1, 0, None])
def test_optional_research_flag_is_strict(flag):
    payload = json.loads((SEED/"patent_fto_guard_v2.json").read_bytes())
    payload["sequence_collision_rules"][0]["block_in_research_max"] = flag
    with pytest.raises(ValueError, match="block_in_research_max"):
        SequenceFTOGuard(payload)


@pytest.mark.parametrize("flag", [True, False, None])
def test_scanner_uses_validated_flag_snapshot(flag):
    payload = json.loads((SEED/"patent_fto_guard_v2.json").read_bytes())
    rule = payload["sequence_collision_rules"][0]
    rule.update(target_scope="cds", sequence_rna="AUGUGG", identity_threshold=1.0)
    if flag is None:
        rule.pop("block_in_research_max", None)
    else:
        rule["block_in_research_max"] = flag
    guard = SequenceFTOGuard(payload)
    rule["block_in_research_max"] = not flag
    hits = guard.scan_bundle(make_bundle(small_molecule()), "research_max")
    assert [h.blocked for h in hits if h.rule_id == rule["rule_id"]] == [bool(flag)]


@pytest.mark.parametrize("keys,allow", [
    (("first", "second"), True), ((0, 1), True), (("0", "1"), True),
    (("first", 1), True), ((0, "first"), False), (("", 1), False),
    (("unknown", 1), False), ((0, 2), False), ((False, 1), False), ((), False),
])
def test_reference_dictionary_resolves_unambiguous_chain_keys(tmp_path, keys, allow):
    first = small_molecule().model_copy_validated(molecule_id="first")
    second = small_molecule(cds="AUGGCU", protein="MA").model_copy_validated(molecule_id="second")
    bundle = make_bundle(first, second)
    _, intent = setup_intent(tmp_path, [m.effective_translation_target for m in bundle.molecules])
    references = dict(zip(keys, (first.sequence, second.sequence)))
    admitted, report, manifest = run_design_pipeline(
        intent, cached_candidates=[bundle], seed_dir=SEED, wildtype_cds_reference=references
    )
    assert bool(admitted) == allow, report.uncomparable_reason
    if allow:
        assert manifest.optimization_mode_audit.wildtype_c_optimizable_replacement_fraction == 0.0
    else:
        assert "key" in report.uncomparable_reason.lower()

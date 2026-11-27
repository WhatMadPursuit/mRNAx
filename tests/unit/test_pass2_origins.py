"""Exercise segment rights through the public candidate pipeline with a fluorescent reporter."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mrnax.cloud.cas_store import ExecutionCASStore
from mrnax.cloud.gateway import _build_preset_reference_r, create_control_plane_app
from mrnax.engine.pareto_director import run_design_pipeline
from mrnax.harness.state_machine import ProjectHarnessSession
from mrnax.schemas.construct_bundle import ConstructBundle
from mrnax.schemas.design_intent import BundleSegmentLock, CellExpressionSpec, DesignIntent


SEED = Path(__file__).resolve().parents[2] / "registry_seed"
GENERATORS = ["stage0_random", "dnachisel_cds", "pareto_nsga2"]
SUFFIX = {"stage0_random": "_stage0_ref", "dnachisel_cds": "_dnachisel", "pareto_nsga2": "_pareto_01"}
LOCKED = ["five_prime_utr", "cds", "three_prime_utr"]


def provenance(segment):
    return segment.component_id, segment.origin_source, segment.license_track


def reference(chains=1):
    molecule = _build_preset_reference_r(SEED).molecules[0]
    tiers = ["research_only", "needs_vendor_license", "quarantine_reference", "commercial_clearance_pending", "research_only"]
    molecules = []
    for i in range(chains):
        spans = [
            s.model_copy_validated(
                component_id=f"reference-{i}-{s.segment_type}", origin_source=f"origin-{i}-{s.segment_type}",
                license_track=tier,
            )
            for s, tier in zip(molecule.segments, tiers, strict=True)
        ]
        molecules.append(molecule.model_copy_validated(molecule_id=f"ref-{i}", segments=spans))
    return ConstructBundle.create(
        bundle_id="reference", derivation_action="reference_import", parent_bundle_id=None,
        molecules=molecules, target_stoichiometry={m.molecule_id: 1.0 for m in molecules},
        co_expression_mode="single_monocistronic" if chains == 1 else "lnp_co_encapsulation",
    )


def design_intent(tmp_path, ref, generators, *, locks=(), minimal=False, seed_dir=SEED, track="research_max"):
    primary = ref.molecules[0]
    sess = ProjectHarnessSession(project_id="origin", store_dir=tmp_path / "project", seed_dir=seed_dir)
    intent = sess.create_initial_intent(
        turn_id="origin-turn", user_utterance="Check segment provenance",
        protein_sequence=primary.effective_translation_target.protein_sequence, reference_bundle=ref,
        uridine_chemistry=primary.chemistry_track.uridine_chemistry, compliance_track=track,
        locked_regions={scope: primary.get_segment_sequence(scope) for scope in locks},
    )
    return intent.model_copy_validated(
        protein_targets=[m.effective_translation_target for m in ref.molecules],
        constraints=[], objectives=[], portfolio_tracks=["minimal_change_vs_r"] if minimal else [],
        runtime_policy=intent.runtime_policy.model_copy_validated(
            enabled_generators=generators, enabled_evaluators=[], max_evaluations=3, allow_cached_candidates=True,
        ),
    )


@pytest.mark.parametrize("generator", GENERATORS)
def test_generator_copies_source_and_tier_for_locked_and_carried_segments(tmp_path, generator):
    ref = reference()
    intent = design_intent(tmp_path, ref, [generator], locks=LOCKED)
    bundles, _, _ = run_design_pipeline(intent, reference_bundle=ref, seed_dir=SEED)
    child = next(b for b in bundles if b.bundle_id.endswith(SUFFIX[generator]))
    assert [provenance(s) for s in child.molecules[0].segments] == [provenance(s) for s in ref.molecules[0].segments]


def test_minimal_refinement_inherits_both_chains_without_relabelling(tmp_path):
    ref = reference(2)
    intent = design_intent(tmp_path, ref, [], minimal=True)
    bundles, _, _ = run_design_pipeline(intent, reference_bundle=ref, cached_candidates=[ref], seed_dir=SEED)
    child = next(b for b in bundles if b.bundle_id.endswith("_minA"))
    assert [[provenance(s) for s in m.segments] for m in child.molecules] == [
        [provenance(s) for s in m.segments] for m in ref.molecules
    ]
    assert child.molecules[1] == ref.molecules[1]


@pytest.mark.parametrize("generator", GENERATORS)
def test_generated_secondary_chains_inherit_shared_segment_provenance(tmp_path, generator):
    ref = reference(2)
    intent = design_intent(tmp_path, ref, [generator], locks=["five_prime_utr", "three_prime_utr"])
    bundles, _, _ = run_design_pipeline(intent, reference_bundle=ref, seed_dir=SEED)
    child = next(b for b in bundles if b.bundle_id.endswith(SUFFIX[generator]))
    assert len(child.molecules) == 2
    for mol in child.molecules:
        for scope in ["five_prime_utr", "stop_cassette", "three_prime_utr", "poly_a"]:
            assert mol.get_segment_sequence(scope) == ref.molecules[0].get_segment_sequence(scope)
            assert provenance(mol.get_segment(scope)) == provenance(ref.molecules[0].get_segment(scope))


def seed_copy(tmp_path, editable):
    seed = tmp_path / "seed"
    seed.mkdir()
    for file in SEED.glob("*.json"):
        if file.name != editable:
            (seed / file.name).symlink_to(file)
    return seed


def test_registry_assets_retain_selected_row_provenance_on_every_generator(tmp_path):
    seed = seed_copy(tmp_path, "utr_components_v1.json")
    registry = json.loads((SEED / "utr_components_v1.json").read_text())
    expected = {}
    for key in ["five_prime_utrs", "three_prime_utrs"]:
        for row in registry[key]:
            row.update(license_track="needs_vendor_license", origin_source="fixture:" + row["component_id"])
            expected[row["component_id"]] = row["origin_source"], row["license_track"]
    (seed / "utr_components_v1.json").write_text(json.dumps(registry))

    ref = reference()
    intent = design_intent(tmp_path, ref, GENERATORS, seed_dir=seed)
    bundles, _, _ = run_design_pipeline(intent, reference_bundle=ref, seed_dir=seed)
    children = [b for b in bundles if b.bundle_id != ref.bundle_id]
    assert all(any(b.bundle_id.endswith(SUFFIX[gen]) for b in children) for gen in GENERATORS)
    for child in children:
        for scope in ["five_prime_utr", "three_prime_utr"]:
            cid, source, tier = provenance(child.molecules[0].get_segment(scope))
            assert (source, tier) == expected[cid]


def test_public_commercial_gate_rejects_inherited_rights_and_keeps_unrelated_cache(tmp_path):
    ref = reference()
    cached = _build_preset_reference_r(SEED)
    intent = design_intent(tmp_path, ref, ["stage0_random"], minimal=True, track="commercial_clean")
    bundles, report, _ = run_design_pipeline(intent, reference_bundle=ref, cached_candidates=[cached], seed_dir=SEED)
    assert bundles == [cached]
    denied = [d for d in report.failed_diagnostics if d.violated_requirement_id == "clean_room:segment_license_track"]
    for suffix in ["_minA", "_stage0_ref"]:
        assert {"stop_cassette", "poly_a"} <= {d.violated_scope for d in denied if d.candidate_ref.endswith(suffix)}


def test_new_lock_without_a_matching_source_has_pending_clearance(tmp_path):
    ref = reference()
    intent = design_intent(tmp_path, ref, ["stage0_random"], locks=["poly_a"])
    locks = dict(intent.locked_scopes)
    locks["poly_a"] = locks["poly_a"].model_copy_validated(locked_value=locks["poly_a"].locked_value + "A")
    intent = intent.model_copy_validated(locked_scopes=locks)
    bundles, _, _ = run_design_pipeline(intent, reference_bundle=ref, seed_dir=SEED)
    child = next(b for b in bundles if b.bundle_id.endswith("_stage0_ref"))
    assert provenance(child.molecules[0].get_segment("poly_a")) == (
        None, "user_lock:origin-turn", "commercial_clearance_pending",
    )


def test_conflicting_library_source_for_same_id_is_rejected(tmp_path):
    filename = "utr_components_v1.json"
    seed = seed_copy(tmp_path, filename)
    doc = json.loads((SEED / filename).read_text())
    row = doc["five_prime_utrs"][0]
    doc["five_prime_utrs"].append({**row, "license_track": "quarantine_reference"})
    (seed / filename).write_text(json.dumps(doc))
    ref = reference()
    intent = design_intent(tmp_path, ref, ["stage0_random"])
    with pytest.raises(ValueError, match="Conflicting library provenance"):
        run_design_pipeline(intent, reference_bundle=ref, seed_dir=seed)


def test_composite_segment_inherits_restricted_registry_and_both_sources(tmp_path):
    filename = "mirna_rbp_commercial_v2.json"
    seed = seed_copy(tmp_path, filename)
    doc = json.loads((SEED / filename).read_text())
    row = {**doc["mirna_entries"][0], "mirna_id": "fixture-source", "top_enriched_tissue": "fixture-a",
           "fantom5_mean_cpm_by_tissue": {"fixture-a": 1.0, "fixture-b": 0.0}}
    doc.update(registry_id="fixture-registry", license_track="research_only", mirna_entries=[row])
    (seed / filename).write_text(json.dumps(doc))
    ref = reference()
    intent = design_intent(tmp_path, ref, ["stage0_random"], minimal=True).model_copy_validated(
        cell_expression_control=CellExpressionSpec(detarget_tissues=["fixture-a"], protected_tissues=["fixture-b"]),
    )
    bundles, _, _ = run_design_pipeline(intent, reference_bundle=ref, seed_dir=seed)
    base_rows = json.loads((SEED / "utr_components_v1.json").read_text())["three_prime_utrs"]
    registry_rows = {r["component_id"]: r["origin_source"] for r in base_rows}
    for suffix in ["_minA", "_stage0_ref"]:
        child = next(b for b in bundles if b.bundle_id.endswith(suffix))
        span = child.molecules[0].get_segment("three_prime_utr")
        base_source = (ref.molecules[0].get_segment("three_prime_utr").origin_source if suffix == "_minA"
                       else registry_rows[span.component_id])
        assert span.license_track == "research_only"
        assert base_source in span.origin_source and "fixture-registry:fixture-source" in span.origin_source


def test_selected_bundle_lock_preserves_its_origin_even_when_reference_has_same_bytes(tmp_path):
    projects = tmp_path / "projects"
    session = ProjectHarnessSession(project_id="locks", store_dir=projects / "locks", seed_dir=SEED)
    clean = _build_preset_reference_r(SEED)
    source = reference()
    initial = session.create_initial_intent(
        turn_id="first", user_utterance="Compare sources", reference_bundle=clean,
        protein_sequence=clean.molecules[0].effective_translation_target.protein_sequence,
        uridine_chemistry=clean.molecules[0].chemistry_track.uridine_chemistry,
    )
    session.register_bundle(source, originating_intent_id=initial.intent_id)
    client = TestClient(create_control_plane_app(
        ExecutionCASStore(tmp_path / "cas.sqlite"), projects_root=projects, seed_dir=SEED, access_token="test-admin",
    ))
    response = client.post("/v1/projects/locks/turn", headers={"X-MRNAX-Access-Token": "test-admin"}, json={
        "mode": "refine", "user_utterance": "Lock selected CDS", "lock_cds_from_bundle_id": source.bundle_id,
    })
    assert response.status_code == 200
    reloaded = ProjectHarnessSession(project_id="locks", store_dir=projects / "locks", seed_dir=SEED).latest_intent()
    lock = reloaded.locked_scopes["cds"]
    assert isinstance(lock, BundleSegmentLock)
    assert lock.source_bundle_digest == source.version_digest and lock.source_bundle_id == source.bundle_id
    assert lock.source_segment == source.molecules[0].get_segment("cds")
    assert lock.locked_value == clean.molecules[0].get_segment_sequence("cds")
    variants = [c for c in response.json()["candidates"] if c["bundle_id"] != clean.bundle_id]
    assert variants
    for candidate in variants:
        cds = next(s for s in candidate["segments"] if s["segment_type"] == "cds")
        assert (cds["component_id"], cds["origin_source"], cds["license_track"]) == provenance(lock.source_segment)


def test_historical_unsourced_lock_preserves_frozen_digest_and_wire_shape():
    path = SEED.parent / "artifacts/m1_egfp/session_entry1_dialogue/intents/intent_v001_int_c8c7d3f1.json"
    original = json.loads(path.read_text())
    intent = DesignIntent.model_validate(original)
    assert intent.digest() == "c280ffb1ff7c10c579f1a130cd148944874adb337ecde3a155141856de1308b1"
    assert intent.model_dump(mode="json")["locked_scopes"] == original["locked_scopes"]

"""Round-2 regressions for method receipts and registry screening."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import random
from dataclasses import replace
from email.message import Message
from pathlib import Path
from types import SimpleNamespace

import pytest

from mrnax.cloud.gateway import _build_preset_reference_r
from mrnax.constraints.fto_guard import (
    MethodFTOGuard, SequenceFTOGuard, _count_recursive_gpl_dependencies, audit_codebase_method_fto_lints,
)
from mrnax.engine.pareto_director import run_design_pipeline
from mrnax.engine.profiles import COMPONENT_CATALOG
from mrnax.harness.state_machine import ProjectHarnessSession

ROOT = Path(__file__).resolve().parents[2]
SEED = ROOT / "registry_seed"
METHOD = SEED / "method_fto_guard_v1.1.json"
SEQUENCE = SEED / "patent_fto_guard_v2.json"


@pytest.fixture
def commercial_fixture(tmp_path):
    ref = _build_preset_reference_r(SEED)
    session = ProjectHarnessSession(project_id="receipt", store_dir=tmp_path / "receipt", seed_dir=SEED)
    intent = session.create_initial_intent(
        turn_id="t", user_utterance="Receipt screening", compliance_track="commercial_clean",
        protein_sequence=ref.molecules[0].effective_translation_target.protein_sequence,
        uridine_chemistry="m1psi", use_commercial_surrogate=True,
    )
    return intent, ref


def test_all_registered_affirmative_method_events(commercial_fixture):
    guard = MethodFTOGuard.from_json_path(METHOD)
    intent, _ = commercial_fixture
    failed = []
    for boundary in guard.boundaries:
        for event in boundary["prohibited_execution_pattern"]:
            receipt = guard.build_runtime_audit_record(
                intent=intent, opt_mode="stage0_or_reference_only", used_two_zone_partition=False,
                executed_operators=[event], invoked_components=[], seq_fto_receipt={},
                input_origin_type="amino_acid_only",
            )
            if receipt.method_fto_boundary_checks[boundary["boundary_id"]]["status"] != "BLOCK":
                failed.append(event)
            if event == "uses_dfa_lattice_cfg_intersection":
                assert receipt.uses_dfa_cfg_intersection
    assert not failed


@pytest.mark.parametrize("event", ["dual_hash", *(
    event for boundary in json.loads(METHOD.read_bytes())["method_claim_boundaries"]
    for event in boundary["prohibited_execution_pattern"]
)])
def test_withdrawn_recommendation_removed_from_ranking_trace(event, commercial_fixture, monkeypatch):
    intent, ref = commercial_fixture
    intent = intent.model_copy_validated(runtime_policy=intent.runtime_policy.model_copy_validated(
        enabled_generators=[], allow_cached_candidates=True,
    ))
    _, clean, _ = run_design_pipeline(intent, cached_candidates=[ref], seed_dir=SEED)
    assert ref.bundle_id in clean.recommended_bundle_ids
    assert any(ref.bundle_id in t.changed_candidate_ids for t in clean.action_transparency)
    build = MethodFTOGuard.build_runtime_audit_record

    def inject(self, **kwargs):
        return build(self, **{**kwargs, "executed_operators": [*kwargs["executed_operators"], event]})

    monkeypatch.setattr(MethodFTOGuard, "build_runtime_audit_record", inject)
    bundles, report, _ = run_design_pipeline(intent, cached_candidates=[ref], seed_dir=SEED)
    assert report.outcome_status == "UNCOMPARABLE_MISSING_APPROVED_COMPONENT"
    assert not (bundles or report.recommended_bundle_ids or report.candidate_evaluations)
    assert all(not trace.changed_candidate_ids for trace in report.action_transparency if trace.stage_applied == "ranking")


@pytest.mark.parametrize("path", [
    ("method_claim_boundaries", 0, "prohibited_execution_pattern"),
    ("method_claim_boundaries", 0, "prohibited_execution_pattern", "scalar_weighted_sum_tie_cai_mfe"),
    ("method_claim_boundaries", 3, "patent_family", 0),
    ("method_claim_boundaries", 2, "jurisdictions_active_grant"),
    ("required_commercial_run_audit_fields", "fields", 0),
    ("required_commercial_run_audit_fields", "schema_version"),
    ("cartesian_utr_pair_rules", 0, "utr5_identity_threshold"),
    ("cartesian_utr_pair_rules", 0, "five_prime_utr_candidates"),
    ("cartesian_utr_pair_rules", 0, "three_prime_utr_candidates"),
    ("cartesian_utr_pair_rules", 0, "five_prime_utr_candidates", 0, "sequence_rna"),
    ("genus_grammar_rules", 0, "regex_rna"),
    ("sequence_collision_rules", 0, "target_scope"),
])
def test_registry_schema_rejects_incomplete_records(path):
    is_method = path[0] in {"method_claim_boundaries", "required_commercial_run_audit_fields"}
    doc = json.loads((METHOD if is_method else SEQUENCE).read_bytes())
    target = doc
    for key in path[:-1]:
        target = target[key]
    if str(path[-1]).endswith("_candidates"):
        target[path[-1]] = []
    else:
        del target[path[-1]]
    with pytest.raises(ValueError):
        (MethodFTOGuard if is_method else SequenceFTOGuard)(doc)


def test_lint_standalone_dual_hash(tmp_path):
    file = tmp_path / "actual.py"
    file.write_text("dual_hasher = 1\n")
    assert audit_codebase_method_fto_lints(tmp_path)["status"] == "PASS"
    file.write_text("dual_hash = 1\n")
    assert audit_codebase_method_fto_lints(tmp_path)["status"] == "BLOCK"


def test_builtin_asset_receipts_hash_packaged_bytes(commercial_fixture):
    intent, ref = commercial_fixture
    _, _, manifest = run_design_pipeline(intent, reference_bundle=ref, seed_dir=SEED)
    entries = {e["asset_id"]: e for e in manifest.optimization_mode_audit.three_layer_dependency_manifest}
    paths = {
        "stage0_random": ROOT / "src/mrnax/engine/stage0_baseline.py",
        "pareto_nsga2": ROOT / "src/mrnax/engine/pareto_director.py",
        "dnachisel_cds": importlib.metadata.distribution("dnachisel").locate_file(
            "dnachisel/DnaOptimizationProblem/DnaOptimizationProblem.py"
        ),
        "surrogate_v1_clean": SEED / "surrogate_v1/TRAINING_PROVENANCE_MANIFEST.json",
    }
    for cid, source_file in paths.items():
        item = entries[COMPONENT_CATALOG[cid].display_name]
        assert item["sha256"] == hashlib.sha256(source_file.read_bytes()).hexdigest()


def test_distribution_graph_crosses_catalog_cycles_and_normalizes_pep508(monkeypatch):
    base = COMPONENT_CATALOG["stage0_random"]
    root = replace(base, component_id="audit_root", display_name="Root", transitive_dependencies=("audit_mid",))
    middle = replace(base, component_id="audit_mid", display_name="Middle",
                     transitive_dependencies=("audit_root", "Outer.Pkg[test]>=1"))
    for spec in (root, middle):
        monkeypatch.setitem(COMPONENT_CATALOG, spec.component_id, spec)

    def distribution(license_name, deps):
        metadata = Message()
        metadata["License-Expression"] = license_name
        return SimpleNamespace(metadata=metadata, requires=deps)

    dists = {
        "outer-pkg": distribution("MIT", ["Leaf__Pkg>=1", "leaf.pkg", "Outer.pkg"]),
        "leaf-pkg": distribution("GPL-3.0-only", ["outer_pkg"]),
    }
    monkeypatch.setattr(importlib.metadata, "distribution", dists.__getitem__)
    assert _count_recursive_gpl_dependencies("Root", "MIT", "MIT")[0] == 1


def test_cartesian_threshold_all_single_edits_and_short_negative():
    guard = SequenceFTOGuard.from_json_path(SEQUENCE)
    rng = random.Random(7)
    for _ in range(20):
        ref = "".join(rng.choice("ACGU") for _ in range(20))
        for i in range(len(ref)):
            subst = next(c for c in "ACGU" if c != ref[i])
            for sub in (ref[:i] + ref[i+1:], ref[:i] + subst + ref[i+1:], ref[:i] + subst + ref[i:]):
                assert guard._match_identity(sub, ref, .95)[0], (ref, i, sub)
        assert not guard._match_identity(ref[:18], ref, .95)[0]
    longer = "ACGU" * 10
    assert guard._match_identity(longer[:2] + longer[4:], longer, .95)[0]
    assert not guard._match_identity(longer[3:], longer, .95)[0]

def test_cartesian_identity_uses_whole_reference_for_mixed_indels():
    guard = SequenceFTOGuard.from_json_path(SEQUENCE)
    ref = "CAGAGCAGACAACUAAGUGC"
    # Removing ref[5] and inserting another letter leaves 19 reference identities.
    candidate = ref[:5] + ref[6:10] + "C" + ref[10:]
    assert guard._match_identity(candidate, ref, .95) == (
        True, "local_identity:0.950>=0.95", (0, len(candidate)),
    )


@pytest.mark.parametrize(("license_text", "expected"), [
    ("GPLv3", 1),
    ("GPLv2+", 1),
    ("License :: OSI Approved :: GNU General Public License v3 (GPLv3)", 1),
    ("GPL-3.0-or-later WITH GCC-exception-3.1", 0),
    ("GPLv2+ and a separate library under the GCC Runtime Library Exception", 1),
])
def test_dependency_license_spellings_and_scoped_exception(license_text, expected, monkeypatch):
    metadata = Message()
    metadata["License"] = license_text
    dist = SimpleNamespace(metadata=metadata, requires=[])
    monkeypatch.setattr(importlib.metadata, "distribution", lambda name: dist)
    assert _count_recursive_gpl_dependencies("audit-dist", "MIT", "MIT")[0] == expected


def test_cartesian_rejects_sparse_reference_subsequence():
    guard = SequenceFTOGuard.from_json_path(SEQUENCE)
    ref = "ACGU" * 5
    candidate = ("A" * 10).join(ref)
    assert not guard._match_identity(candidate, ref, .95)[0]


def test_cartesian_matches_exhaustive_short_interval_oracle():
    from itertools import combinations, product

    guard = SequenceFTOGuard.from_json_path(SEQUENCE)
    for letters in product("AC", repeat=4):
        ref = "".join(letters)
        # At 75%, short intervals require three identities. A span of five
        # requires all four. Six or more can never qualify.
        triples = {"".join(triple) for triple in combinations(ref, 3)}
        for letters in product("AC", repeat=6):
            target = "".join(letters)
            expected = any(
                "".join(triple) in triples
                for start in range(4)
                for triple in combinations(target[start:start+4], 3)
            ) or any(
                "".join(quad) == ref
                for start in range(2)
                for quad in combinations(target[start:start+5], 4)
            )
            assert guard._match_identity(target, ref, .75)[0] == expected, (target, ref)


def test_mandatory_sequence_policy_refuses_shipped_v1(tmp_path):
    from mrnax.constraints.registry import ScopedConstraintEngine

    legacy_path = SEED / "patent_fto_guard_v1.json"
    installed = tmp_path / SEQUENCE.name
    installed.write_bytes(legacy_path.read_bytes())
    with pytest.raises(ValueError):
        ScopedConstraintEngine.from_seed_dir(tmp_path)
    with pytest.raises(ValueError):
        SequenceFTOGuard.from_json_path(installed, require_v2=False)
    with pytest.raises(ValueError):
        SequenceFTOGuard.from_json_path(legacy_path)
    legacy = SequenceFTOGuard.from_json_path(legacy_path, require_v2=False)
    assert legacy.registry_id == "patent_fto_guard_v1"
    assert ScopedConstraintEngine.from_seed_dir(SEED).fto_guard.registry_id == "patent_fto_guard_v2"


@pytest.mark.parametrize(("field", "value"), [
    ("registry_id", "patent_fto_guard_v1"),
    ("version", "1.0.0"),
    ("schema_version", "1.0.0"),
    ("schema_version", None),
    ("cartesian_utr_pair_rules", []),
])
def test_v2_sequence_policy_checks_identity_and_schema(tmp_path, field, value):
    path = tmp_path / SEQUENCE.name
    policy = json.loads(SEQUENCE.read_bytes())
    if value is None:
        policy.pop(field, None)
    else:
        policy[field] = value
    path.write_text(json.dumps(policy))
    with pytest.raises(ValueError):
        SequenceFTOGuard.from_json_path(path)


@pytest.mark.parametrize(("cds", "reference", "table", "protein"), [
    ("AUGGCU", "AUGGCU", 1, "MA"),
    ("AUGGCUUAA", "AUGGCU", 1, "MA"),
    ("AUGGCU", "AUGGCUUAG", 1, "MA"),
    ("AUGUGA", "AUGUGA", 2, "MW"),
    ("AUGUGAAGA", "AUGUGA", 2, "MW"),
])
def test_reference_representations_survive_pipeline_and_export(
    commercial_fixture, tmp_path, cds, reference, table, protein,
):
    from mrnax.constraints.registry import ScopedConstraintEngine
    from mrnax.io_insdc.genbank_io import export_construct_package
    from mrnax.schemas.construct_bundle import ConstructBundle

    intent, preset = commercial_fixture
    mol = preset.molecules[0]
    target = mol.effective_translation_target.model_copy_validated(
        protein_sequence=protein, transl_table=table, wildtype_cds_rna=reference,
    )
    molecule = mol.model_copy_validated(
        effective_translation_target=target, sequence=cds, annotations=[],
        segments=[mol.get_segment("cds").model_copy_validated(start=0, end=len(cds))],
    )
    bundle = ConstructBundle.create(
        bundle_id="toy", parent_bundle_id=None, derivation_action="reference_import",
        target_stoichiometry={molecule.molecule_id: 1}, molecules=[molecule],
    )
    intent = intent.model_copy_validated(
        protein_targets=[target], constraints=[], objectives=[], locked_scopes={},
        runtime_policy=intent.runtime_policy.model_copy_validated(enabled_generators=[], allow_cached_candidates=True),
    )
    candidates, report, manifest = run_design_pipeline(intent, cached_candidates=[bundle], seed_dir=SEED)
    assert [b.bundle_id for b in candidates] == [bundle.bundle_id], report.failed_diagnostics
    assert manifest.optimization_mode_audit.wildtype_c_optimizable_replacement_fraction == 0.0
    directory = tmp_path / "export"
    export_construct_package(
        bundle, directory, constraint_engine=ScopedConstraintEngine.from_seed_dir(SEED),
        constraints=[], compliance_track="commercial_clean", expected_protein_targets=[target],
    )
    assert len(list(directory.iterdir())) == 4


@pytest.mark.parametrize(("wildtype", "candidate", "table"), [
    ("", "AUG", 1), ("  ", "AUG", 1), ("AUGN", "AUG", 1),
    ("AUGGC", "AUG", 1), ("AUG", "AUGGCU", 1), ("AUGGCU", "AUGGUU", 1),
    ("AUGUAAGCU", "AUGUAAGCU", 1), ("AUGUAAUAG", "AUGUAA", 1),
    ("UAA", "UAA", 1), ("AUGAGAGCU", "AUGAGAGCU", 2),
    ("AUG", "AUG", 999),
])
def test_reference_representation_rejects_invalid_cds(wildtype, candidate, table):
    from mrnax.constraints.fto_guard import compute_curevac_c_optimizable_conversion

    with pytest.raises(ValueError):
        compute_curevac_c_optimizable_conversion(wildtype, candidate, translation_table=table)


def test_unsupported_genetic_code_is_rejected_when_declaring_target():
    from mrnax.schemas.construct_bundle import ProteinTargetSpec

    with pytest.raises(ValueError, match="Unknown NCBI translation table 999"):
        ProteinTargetSpec(protein_sequence="M", transl_table=999, accession_or_source="test")

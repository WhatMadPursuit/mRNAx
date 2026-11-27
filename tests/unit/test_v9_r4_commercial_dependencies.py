from dataclasses import replace
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from mrnax.cloud.cas_store import ExecutionCASStore
from mrnax.cloud.gateway import _build_preset_reference_r, create_control_plane_app
from mrnax.constraints.fto_guard import MethodFTOGuard, _count_recursive_gpl_dependencies
from mrnax.engine.pareto_director import run_design_pipeline
from mrnax.engine.profiles import COMPONENT_CATALOG, check_profile_readiness
from mrnax.harness.state_machine import ProjectHarnessSession
from mrnax.harness.tools import AtomicToolbox
from mrnax.schemas.design_intent import RuntimePolicySpec


SEED = Path(__file__).resolve().parents[2] / "registry_seed"


def install_metadata(directory, name, expression, *requirements):
    metadata = directory / f"{name.replace('-', '_')}-1.0.dist-info"
    metadata.mkdir()
    fields = ["Metadata-Version: 2.4", f"Name: {name}", "Version: 1.0",
              f"License-Expression: {expression}"]
    fields.extend(f"Requires-Dist: {requirement}" for requirement in requirements)
    (metadata / "METADATA").write_text("\n".join(fields) + "\n\n")


@pytest.mark.parametrize(("expression", "expected"), [
    ("GPL-3.0-only", 1),
    ("GPL-3.0-only OR MIT", 0),
    ("GPL-3.0-only AND (MIT OR BSD-3-Clause)", 1),
    ("(MIT OR BSD-3-Clause) AND GPL-3.0-only", 1),
    ("(GPL-3.0-only OR MIT) AND (Apache-2.0 OR BSD-3-Clause)", 0),
    ("(GPL-3.0-only OR AGPL-3.0-only) AND MIT", 1),
    ("MIT AND GPL-3.0-only OR Apache-2.0", 0),
    ("GPL-3.0-only WITH GCC-exception-3.1 AND LGPL-2.1-only", 1),
    ("GPL-3.0-only WITH GCC-exception-3.1", 0),
    ("GPL-3.0-only WITH Classpath-exception-2.0", 1),
    ("MIT OR (BSD-3-Clause", 1),
    ("MIT AND", 1),
    ("MIT OR OR Apache-2.0", 1),
    ("MIT GPL-3.0-only", 1),
    ("MIT WITH", 1),
    ("MIT OR Unknown-License-999", 1),
    ("MIT OR ()", 1),
    ("", 1),
])
def test_spdx_distribution_receipt(expression, expected, tmp_path, monkeypatch):
    install_metadata(tmp_path, "round4-leaf", expression)
    monkeypatch.syspath_prepend(str(tmp_path))
    count, _ = _count_recursive_gpl_dependencies("round4-leaf", "MIT", "MIT")
    assert count == expected


@pytest.fixture
def commercial_run(tmp_path):
    reference = _build_preset_reference_r(SEED)
    session = ProjectHarnessSession(project_id="audit", store_dir=tmp_path / "audit", seed_dir=SEED)
    template = session.create_initial_intent(
        turn_id="audit", user_utterance="Audit the cached reporter", compliance_track="commercial_clean",
        uridine_chemistry="m1psi", use_commercial_surrogate=True,
        protein_sequence=reference.molecules[0].effective_translation_target.protein_sequence,
    )
    intent = template.model_copy_validated(constraints=[], runtime_policy=RuntimePolicySpec(
        enabled_generators=[], enabled_evaluators=["surrogate_v1_clean"], allow_cached_candidates=True,
    ))
    return session, intent, reference


@pytest.mark.parametrize(("expression", "allowed"), [
    ("GPL-3.0-only AND (MIT OR BSD-3-Clause)", False),
    ("GPL-3.0-only OR MIT", True),
])
def test_admission_and_execution_follow_installed_descendants(
    expression, allowed, commercial_run, tmp_path, monkeypatch
):
    session, intent, reference = commercial_run
    install_metadata(tmp_path, "round4-parent", "MIT", "round4-leaf>=1")
    install_metadata(tmp_path, "round4-leaf", expression)
    monkeypatch.syspath_prepend(str(tmp_path))
    evaluator = COMPONENT_CATALOG["surrogate_v1_clean"]
    middle = replace(COMPONENT_CATALOG["stage0_random"], component_id="round4_middle",
                     transitive_dependencies=("round4-parent",))
    monkeypatch.setitem(COMPONENT_CATALOG, middle.component_id, middle)
    monkeypatch.setitem(COMPONENT_CATALOG, evaluator.component_id,
                        replace(evaluator, transitive_dependencies=(middle.component_id,)))
    readiness = check_profile_readiness(intent)
    bundles, report, manifest = run_design_pipeline(intent, cached_candidates=[reference], seed_dir=SEED)
    assert readiness.is_runnable is allowed
    if allowed:
        assert report.recommended_bundle_ids == [reference.bundle_id]
        assert len(bundles) == 1
        assert all(item["transitive_gpl_count"] == 0
                   for item in manifest.optimization_mode_audit.three_layer_dependency_manifest)
    else:
        assert "transitive_gpl_count=1" in readiness.uncomparable_reason
        assert report.outcome_status == "UNCOMPARABLE_MISSING_APPROVED_COMPONENT"
        assert manifest.invoked_components == []
        assert not (bundles or report.candidate_evaluations or report.recommended_bundle_ids)
    check_project_export(session, intent, reference, report, manifest, tmp_path, allowed=allowed)


@pytest.mark.parametrize("ranked", [True, False], ids=["ranked", "feasibility_only"])
def test_positive_execution_receipt_withdraws_commercial_release(
    ranked, commercial_run, tmp_path, monkeypatch
):
    session, intent, reference = commercial_run
    if not ranked:
        intent = intent.model_copy_validated(objectives=[])
    build = MethodFTOGuard.build_runtime_audit_record

    def recorded_copyleft(guard, **kwargs):
        audit = build(guard, **kwargs)
        first, *others = audit.three_layer_dependency_manifest
        return audit.model_copy_validated(three_layer_dependency_manifest=[
            dict(first, transitive_gpl_count=1), *others,
        ])

    monkeypatch.setattr(MethodFTOGuard, "build_runtime_audit_record", recorded_copyleft)
    assert check_profile_readiness(intent).is_runnable
    bundles, report, manifest = run_design_pipeline(intent, cached_candidates=[reference], seed_dir=SEED)
    entries = manifest.optimization_mode_audit.three_layer_dependency_manifest
    assert entries[0]["transitive_gpl_count"] == 1
    assert all(info["status"] != "BLOCK"
               for info in manifest.optimization_mode_audit.method_fto_boundary_checks.values())
    assert report.outcome_status == "UNCOMPARABLE_MISSING_APPROVED_COMPONENT"
    assert "transitive_gpl_count=1" in report.uncomparable_reason
    assert not (bundles or report.candidate_evaluations or report.recommended_bundle_ids)
    assert all(not action.changed_candidate_ids for action in report.action_transparency
               if action.stage_applied == "ranking")

    check_project_export(session, intent, reference, report, manifest, tmp_path, allowed=False)


def check_project_export(session, intent, reference, report, manifest, tmp_path, *, allowed):
    session.register_bundle(reference, originating_intent_id=intent.intent_id)
    directory = session.store_dir
    intent_path, = (directory / "intents").glob("*.json")
    intent_path.write_text(intent.model_dump_json())
    (directory / "reports" / f"{report.report_id}.json").write_text(report.model_dump_json())
    (directory / "manifests").mkdir(exist_ok=True)
    (directory / "manifests" / f"{manifest.execution_id}.json").write_text(manifest.model_dump_json())
    reloaded = ProjectHarnessSession(project_id="audit", store_dir=directory, seed_dir=SEED)
    toolbox = AtomicToolbox(reloaded)
    if allowed:
        toolbox.export_bundle(reference.bundle_id)
    else:
        with pytest.raises(ValueError, match="Commercial export blocked"):
            toolbox.export_bundle(reference.bundle_id)
    app = create_control_plane_app(ExecutionCASStore(tmp_path / "cas.sqlite"), projects_root=tmp_path,
                                   seed_dir=SEED, access_token="audit")
    with TestClient(app, headers={"X-MRNAX-Access-Token": "audit"}) as client:
        response = client.get(f"/v1/projects/audit/bundles/{reference.bundle_id}/export")
    assert response.status_code == (200 if allowed else 409)
    assert (directory / "exports" / reference.bundle_id).exists() is allowed


def test_installed_requirement_with_catalog_namesake_keeps_distribution_metadata(tmp_path, monkeypatch):
    base = COMPONENT_CATALOG["stage0_random"]
    for component_id, display, deps in [
        ("round4_start", "Initial test component", ("Display alias",)),
        ("round4_middle", "Display alias", ("round4-parent",)),
        ("round4_leaf", "Independent catalog component", ()),
    ]:
        monkeypatch.setitem(COMPONENT_CATALOG, component_id, replace(
            base, component_id=component_id, display_name=display, transitive_dependencies=deps,
        ))
    install_metadata(tmp_path, "round4-parent", "MIT", "round4-leaf")
    install_metadata(tmp_path, "round4-leaf", "GPL-3.0-only")
    monkeypatch.syspath_prepend(str(tmp_path))
    assert _count_recursive_gpl_dependencies("round4_start", "MIT", "MIT")[0] == 1


def test_admission_denial_also_blocks_registered_reference_export(commercial_run, tmp_path, monkeypatch):
    session, intent, reference = commercial_run
    generator = COMPONENT_CATALOG["stage0_random"]
    install_metadata(tmp_path, "round4-generator-dependency", "GPL-3.0-only")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setitem(COMPONENT_CATALOG, generator.component_id, replace(
        generator, transitive_dependencies=("round4-generator-dependency",),
    ))
    intent = intent.model_copy_validated(runtime_policy=intent.runtime_policy.model_copy_validated(
        enabled_generators=[generator.component_id],
    ))
    _, report, manifest = run_design_pipeline(intent, cached_candidates=[reference], seed_dir=SEED)
    assert "transitive_gpl_count=1" in report.uncomparable_reason
    assert not manifest.invoked_components
    check_project_export(session, intent, reference, report, manifest, tmp_path, allowed=False)


@pytest.mark.parametrize("root", ["joblib", "scikit-learn", "scipy", "numpy", "lightgbm"])
def test_shipped_surrogate_inspects_real_package_roots(commercial_run, tmp_path, monkeypatch, root):
    session, intent, reference = commercial_run
    install_metadata(tmp_path, root, "BSD-3-Clause", "copyleft-child")
    install_metadata(tmp_path, "copyleft-child", "GPL-3.0-only")
    monkeypatch.syspath_prepend(str(tmp_path))
    from mrnax.constraints.fto_guard import count_component_gpl_dependencies

    assert count_component_gpl_dependencies("surrogate_v1_clean", "MIT", "MIT") >= 1
    assert not check_profile_readiness(intent).is_runnable
    bundles, report, manifest = run_design_pipeline(intent, cached_candidates=[reference], seed_dir=SEED)
    assert not bundles and not report.recommended_bundle_ids and not manifest.invoked_components
    assert any(e["transitive_gpl_count"] >= 1 for e in manifest.optimization_mode_audit.three_layer_dependency_manifest)
    check_project_export(session, intent, reference, report, manifest, tmp_path, allowed=False)


@pytest.mark.parametrize(("component", "roots"), [
    ("stage0_random", {"biopython", "dnachisel"}),
    ("dnachisel_cds", {"biopython", "dnachisel"}),
    ("pareto_nsga2", {"biopython", "dnachisel", "numpy"}),
    ("surrogate_v1_clean", {"lightgbm", "joblib", "scikit-learn", "numpy", "scipy"}),
    ("structrfm_rna_foundation", {"joblib", "scikit-learn", "numpy", "scipy", "torch", "transformers"}),
    ("orthrus_mamba_rna_foundation", {"joblib", "scikit-learn", "numpy", "scipy", "torch", "safetensors", "huggingface-hub"}),
    ("codontransformer_cds_baseline", {"numpy", "torch", "transformers", "tokenizers", "safetensors", "huggingface-hub"}),
])
def test_shipped_component_package_roots(component, roots):
    assert roots <= set(COMPONENT_CATALOG[component].transitive_dependencies)


@pytest.mark.parametrize("name", ["scipy", "scikit-learn", "biopython"])
def test_reviewed_distribution_names_never_clear_changed_license_evidence(name, tmp_path, monkeypatch):
    from mrnax.constraints.fto_guard import count_component_gpl_dependencies

    if name == "biopython":
        install_metadata(tmp_path, name, "LicenseRef-Biopython-License-Agreement")
    else:
        metadata = tmp_path / f"{name.replace('-', '_')}-1.0.dist-info"
        metadata.mkdir()
        (metadata / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\nLicense: GPL-3.0-only\n"
            "Classifier: License :: OSI Approved :: BSD License\n\n"
        )
    monkeypatch.syspath_prepend(str(tmp_path))
    assert count_component_gpl_dependencies(name, "MIT", "MIT") == 1

"""Regression coverage for complete measurements, checkpoint identity and policy inputs."""
import hashlib
import json
import re
from pathlib import Path

import pytest

from mrnax.constraints.fto_guard import SequenceFTOGuard
from mrnax.engine import pareto_director as director
from mrnax.engine.surrogates import CommercialCleanSurrogateV1
from mrnax.schemas.construct_bundle import ConstructBundle
from mrnax.schemas.design_intent import DesignIntent, ObjectiveSpec, RuntimePolicySpec
from mrnax.schemas.evaluation_report import MetricResult

ROOT = Path(__file__).resolve().parents[2]
SEED = ROOT / "registry_seed"


@pytest.fixture
def cpu_torch():
    torch = pytest.importorskip("torch")
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield torch
    finally:
        torch.set_num_threads(original)


@pytest.fixture
def reporter():
    fixture = json.loads(next((ROOT / "tests/golden").glob("m2_benchmark1_*.json")).read_text())
    return ConstructBundle.model_validate(fixture["bundles"]["B1_REF_A"])


def objective(name):
    return ObjectiveSpec(requirement_id="req_" + name, metric_id=name, direction="maximize",
                         priority_rank=1, evidence_status="validated_predictor", source_utterance=name)


def cached_intent(bundle, objectives, evaluators):
    return DesignIntent(
        project_id="audit", intent_id="audit", intent_version=1, created_from_turn_ids=["one"],
        entry_mode="protein_only", compliance_track="commercial_clean", status="ready",
        uridine_chemistry=bundle.molecules[0].chemistry_track.uridine_chemistry,
        protein_targets=[m.effective_translation_target for m in bundle.molecules],
        objectives=[objective(name) for name in objectives], constraints=[],
        runtime_policy=RuntimePolicySpec(allow_cached_candidates=True, enabled_generators=[],
                                         enabled_evaluators=evaluators),
    )


def metric(value):
    return MetricResult(metric_id="measured", value=value, unit="test_units", direction="maximize",
                        evaluated_scopes=["cds"], tool_name="instrument", tool_version="1.0",
                        chemistry_evaluation_mode="native_chemistry_validated",
                        participated_in="pareto_selection_or_ranking")


@pytest.mark.parametrize("some_measured", [False, True])
def test_cached_objectives_require_complete_measurements(reporter, monkeypatch, some_measured):
    evaluators = ["surrogate_v1_clean"] if some_measured else []
    monkeypatch.setattr(CommercialCleanSurrogateV1, "evaluate_molecule", lambda self, mol: ([metric(1)], []))
    intent = cached_intent(reporter, ["measured", "missing"], evaluators)
    bundles, report, manifest = director.run_design_pipeline(intent, cached_candidates=[reporter], seed_dir=SEED)
    assert report.outcome_status == "UNCOMPARABLE_MISSING_APPROVED_COMPONENT"
    assert bundles == report.recommended_bundle_ids == []
    assert sum(r.attempt_count for r in manifest.invoked_components if r.role == "evaluator") == int(some_measured)
    assert all(trace.stage_applied == "recorded_only" for trace in report.action_transparency)
    assert re.fullmatch(r"att_exec_[a-f0-9]{32}_001", manifest.attempt_id)


def test_partial_two_chain_measurement_blocks_recommendation(reporter, monkeypatch):
    chains = [reporter.molecules[0].model_copy_validated(molecule_id=name) for name in ("first", "second")]
    bundle = ConstructBundle.create(bundle_id="pair", parent_bundle_id=None,
        derivation_action="reference_import", co_expression_mode="lnp_co_encapsulation",
        target_stoichiometry={m.molecule_id: 1 for m in chains}, molecules=chains)
    observed = []
    def measure(self, mol):
        observed.append(mol.molecule_id)
        return [metric(1 if mol.molecule_id == "first" else None)], []
    monkeypatch.setattr(CommercialCleanSurrogateV1, "evaluate_molecule", measure)
    intent = cached_intent(bundle, ["measured"], ["surrogate_v1_clean"])
    _, report, _ = director.run_design_pipeline(intent, cached_candidates=[bundle], seed_dir=SEED)
    assert observed == ["first", "second"]
    assert report.outcome_status == "UNCOMPARABLE_MISSING_APPROVED_COMPONENT"
    assert not report.recommended_bundle_ids
    metrics, complete = director._aggregate_chain_metrics([[metric(1)], [metric(None)]])
    assert not complete and metrics[0].value is None and metrics[0].notes == "incomplete_chain_coverage(1/2)"


def test_orthrus_numeric_forward_and_checkpoint_identity(tmp_path, monkeypatch, cpu_torch):
    torch = cpu_torch
    from safetensors.torch import save_file
    from mrnax.engine.orthrus_portable_pytorch import PortableOrthrusModel

    torch.manual_seed(123)
    model = PortableOrthrusModel(n_tracks=6, d_model=8, n_layers=1).eval()
    x = torch.arange(84, dtype=torch.float32).reshape(2, 7, 6)
    with torch.no_grad():
        hidden = model(x)
        pooled = model.representation(x, torch.tensor([7, 4]))
    assert hidden.shape == (2, 7, 8) and torch.isfinite(hidden).all()
    torch.testing.assert_close(pooled[1], hidden[1, :4].mean(dim=0))
    (tmp_path / "config.json").write_text(json.dumps({"n_tracks": 6, "ssm_model_dim": 8, "ssm_n_layers": 1}))
    weights = tmp_path / "model.safetensors"
    save_file(model.state_dict(), str(weights))
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    (tmp_path / "SHA256SUMS.json").write_text(json.dumps({weights.name: digest}))
    monkeypatch.delenv("MRNAX_ORTHRUS_SAFETENSORS_SHA256", raising=False)
    with pytest.raises(ValueError, match="approved"):
        PortableOrthrusModel.from_local_checkpoint(tmp_path, device="cpu")
    with pytest.raises(ValueError, match="mismatch"):
        PortableOrthrusModel.from_local_checkpoint(tmp_path, device="cpu", expected_safetensors_sha256="0"*64)
    with pytest.raises(ValueError, match="release digest"):
        PortableOrthrusModel.from_local_checkpoint(tmp_path, device="cpu", expected_safetensors_sha256=digest)
    loaded = PortableOrthrusModel.from_local_checkpoint(
        tmp_path, device="cpu", expected_safetensors_sha256=digest, allow_sandbox_override=True
    )
    torch.testing.assert_close(loaded(x), hidden)


@pytest.mark.parametrize("linker", ["C", "CG" * 120, "C" + "A" * 10 + "G" * 100])
def test_published_linker_policy_blocks_without_length_cap(reporter, linker):
    guard = SequenceFTOGuard.from_json_path(SEED / "patent_fto_guard_v2.json")
    rule = next(r for r in guard.genus_rules if r["patent_family_id"] == "51220548")
    assert "min_total_poly_a_length" not in rule and "80 nt" not in rule["summary"]
    mol = reporter.molecules[0]
    tail = mol.get_segment("poly_a")
    value = "A"*20 + linker + "A"*20
    seq = mol.sequence[:tail.start] + value
    edited = mol.model_copy_validated(sequence=seq,
        segments=[*mol.segments[:-1], tail.model_copy_validated(end=len(seq))])
    bundle = ConstructBundle.create(bundle_id="policy", parent_bundle_id=None, derivation_action="reference_import",
        target_stoichiometry={mol.molecule_id: 1}, molecules=[edited])
    hits = [hit for hit in guard.scan_bundle(bundle, "commercial_clean") if hit.rule_id == rule["rule_id"]]
    assert len(hits) == 1 and hits[0].blocked


def test_original_protection_requires_direct_measurement():
    fixture = json.loads(next((ROOT / "tests/golden").glob("m2_benchmark2_*.json")).read_text())
    intent = DesignIntent.model_validate(fixture["design_intent"])
    assert "spleen" in intent.cell_expression_control.protected_tissues
    assert intent.digest() == fixture["design_intent_digest"]
    atlas = json.loads((SEED / "mirna_rbp_commercial_v2.json").read_text())["mirna_entries"]
    assert all("spleen" not in r["fantom5_mean_cpm_by_tissue"] for r in atlas)
    bundles, report, manifest = director.run_design_pipeline(intent, seed_dir=SEED)
    assert bundles == report.recommended_bundle_ids == manifest.invoked_components == []
    assert report.outcome_status == "UNCOMPARABLE_MISSING_APPROVED_COMPONENT"
    assert "spleen" in report.uncomparable_reason


def test_unmodified_reporter_has_atlas_advisory(reporter):
    intent = cached_intent(reporter, [], []).model_copy_validated(
        uridine_chemistry="unmodified_u",
        runtime_policy=RuntimePolicySpec(enabled_generators=["pareto_nsga2"], enabled_evaluators=[],
                                         max_evaluations=3, random_seed=7),
    )
    _, report, _ = director.run_design_pipeline(intent, seed_dir=SEED)
    assert "adv_bicodon_atlas_endogenous_scope" in {a.advisory_id for a in report.pre_synthesis_advisories}


def test_orthrus_dispatch_applies_shipped_probe_and_reports_units(tmp_path, monkeypatch, reporter, cpu_torch):
    torch = cpu_torch
    from safetensors.torch import save_file
    from mrnax.engine.orthrus_portable_pytorch import PortableOrthrusModel
    from mrnax.engine.surrogates import _load_verified_joblib

    torch.manual_seed(42)
    encoder = PortableOrthrusModel(n_tracks=6, d_model=512, n_layers=1).eval()
    (tmp_path / "config.json").write_text(json.dumps({"n_tracks": 6, "ssm_model_dim": 512, "ssm_n_layers": 1}))
    weights = tmp_path / "model.safetensors"
    save_file(encoder.state_dict(), str(weights))
    monkeypatch.setenv("MRNAX_ORTHRUS_CKPT_DIR", str(tmp_path))
    monkeypatch.setenv("MRNAX_ORTHRUS_SAFETENSORS_SHA256", hashlib.sha256(weights.read_bytes()).hexdigest())
    intent = cached_intent(reporter, ["orthrus_halflife_pred"], ["orthrus_mamba_rna_foundation"])
    _, report, manifest = director.run_design_pipeline(
        intent, cached_candidates=[reporter], seed_dir=SEED, isolated_sandbox_override=True
    )
    mol = reporter.molecules[0]
    encoded = torch.from_numpy(encoder.encode_6track(mol.get_segment_sequence("five_prime_utr"),
        mol.get_segment_sequence("cds") + mol.get_segment_sequence("stop_cassette"),
        mol.get_segment_sequence("three_prime_utr"))).unsqueeze(0)
    probe = _load_verified_joblib(SEED / "surrogate_v1", "orthrus6_saluki_halflife_probe.joblib")
    hidden = encoder(encoded).mean(dim=1).detach().numpy()
    expected = probe["model"].predict(probe["scaler"].transform(hidden))[0]
    measurement, = report.candidate_evaluations[0].metrics
    assert measurement.value == pytest.approx(expected, abs=1e-5)
    assert measurement.unit == "saluki_pc1_z" and report.recommended_bundle_ids == [reporter.bundle_id]
    assert manifest.invoked_components[0].attempt_count == 1


def test_two_chain_reporters_evaluate_and_export_both_records(reporter, tmp_path):
    from Bio import SeqIO
    from mrnax.constraints.registry import ScopedConstraintEngine
    from mrnax.io_insdc.genbank_io import export_construct_package

    target = reporter.molecules[0].effective_translation_target
    intent = cached_intent(reporter, ["consensus_halflife_pred"], ["surrogate_v1_clean"]).model_copy_validated(
        protein_targets=[target, target.model_copy_validated(protein_sequence=target.protein_sequence[:30])],
        runtime_policy=RuntimePolicySpec(enabled_generators=["stage0_random"], enabled_evaluators=["surrogate_v1_clean"]),
    )
    bundles, report, manifest = director.run_design_pipeline(intent, seed_dir=SEED)
    assert report.recommended_bundle_ids == [bundles[0].bundle_id]
    assert [r.attempt_count for r in manifest.invoked_components if r.role == "evaluator"] == [2]
    export_construct_package(bundles[0], tmp_path, constraint_engine=ScopedConstraintEngine.from_seed_dir(SEED),
                             constraints=intent.constraints, compliance_track=intent.compliance_track,
                             expected_chemistry=intent.uridine_chemistry, locked_scopes=intent.locked_scopes)
    records = list(SeqIO.parse(tmp_path / "construct.gb", "genbank"))
    assert [r.id for r in records] == [m.molecule_id for m in bundles[0].molecules]
    assert len(records) == 2

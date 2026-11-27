"""Regression controls for model admission, provenance and native output validation."""
import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest

from mrnax.cloud.gateway import _build_preset_reference_r
from mrnax.constraints.registry import ScopedConstraintEngine, default_m1_constraints
from mrnax.engine.pareto_director import (
    _evaluate_molecule_eternafold,
    project_report_for_intent,
    run_design_pipeline,
)
from mrnax.engine.profiles import check_profile_readiness, resolve_default_cards_dir, resolve_default_seed_dir
from mrnax.harness.state_machine import ProjectHarnessSession
from mrnax.rules.models import APPROVED_REPLACEMENT_RELEASES
from mrnax.schemas.construct_bundle import ConstructBundle
from mrnax.schemas.design_intent import ObjectiveSpec, ProteinTargetSpec, RuntimePolicySpec
from mrnax.schemas.evaluation_report import MetricResult

SEED = resolve_default_seed_dir()


def _metric(mid, producer, scope):
    return MetricResult(metric_id=mid, tool_name=producer, tool_version="1.0.0", unit="arb",
                        direction="maximize", evaluated_scopes=[scope], value=0.5,
                        participated_in="pareto_selection_or_ranking",
                        chemistry_evaluation_mode="native_chemistry_validated")


STABILITY = _metric("structrfm_cds_median_shape", "StructRFMNativeScorer:mauger2019_probe", "cds")
MRL = _metric("utr5_mrl_pred", "CommercialCleanSurrogateV1:utr5_mrl_lgbm", "five_prime_utr")


class FixtureGenerator:
    def generate_and_score_cds(self, protein, **kwargs):
        return {"MAKL": "AUGGCUAAGCUG", "MFLE": "AUGUUCCUGGAA"}[protein], -0.35

    def score_cds_synonymous_loglik(self, sequences):
        return [-0.35] * len(sequences)


class FixturePartner:
    def __init__(self, missing_chain=None):
        self.missing_chain = missing_chain

    def evaluate_molecule(self, molecule):
        if molecule.molecule_id == self.missing_chain:
            return [MRL], []
        return [STABILITY], []


@pytest.fixture
def intent(tmp_path):
    session = ProjectHarnessSession(project_id="model", store_dir=tmp_path / "project", seed_dir=SEED)
    return session.create_initial_intent(
        turn_id="test", user_utterance="model control", protein_sequence="MAKL",
        compliance_track="commercial_clean", uridine_chemistry="unmodified_u",
    ).model_copy_validated(
        protein_targets=[ProteinTargetSpec(protein_sequence=p, accession_or_source="fixture") for p in ("MAKL", "MFLE")],
        locked_scopes={}, constraints=default_m1_constraints(restriction_enzymes=["BspQI"]),
        objectives=[ObjectiveSpec(requirement_id="score", metric_id="codontransformer_syn_loglik", direction="maximize",
                                  priority_rank=1, evidence_status="validated_predictor", source_utterance="fixture")],
        runtime_policy=RuntimePolicySpec(enabled_generators=["codontransformer_cds_baseline"],
                                         enabled_evaluators=["structrfm_rna_foundation"]),
    )


def _run_fixture(intent, partner=None):
    return run_design_pipeline(intent, seed_dir=SEED, codontransformer_model=FixtureGenerator(),
                               structrfm_scorer=partner or FixturePartner(), isolated_sandbox_override=True)


@pytest.mark.parametrize("chemistry", ["unmodified_u", "m1psi", "5mou", "custom_mod", "pseudouridine"])
def test_chemistry_operator_records_supplied_identity(intent, chemistry):
    policy = intent.runtime_policy.model_copy_validated(enabled_evaluators=["surrogate_v1_clean"])
    bundles, _, manifest = _run_fixture(intent.model_copy_validated(uridine_chemistry=chemistry, runtime_policy=policy))
    assert len(bundles) == 1
    operators = [op for op in manifest.optimization_mode_audit.optimization_schedule_and_operators
                 if op.startswith("codontransformer_")]
    suffix = (f"plus_{chemistry}_uuuc_u4_remediation"
              if chemistry in {"m1psi", "pseudouridine"} else chemistry)
    assert operators == [f"codontransformer_v1_homo_sapiens_synonymous_mask_{suffix}"]


def test_research_partner_needs_current_clearance_card(tmp_path, intent):
    research = intent.model_copy_validated(compliance_track="research_max")
    cards = tmp_path / "cards"
    shutil.copytree(resolve_default_cards_dir(), cards)
    card = next(cards.glob("RC-002-*.yaml"))
    assert check_profile_readiness(research, cards_dir=cards).is_runnable
    card.unlink()
    assert not check_profile_readiness(research, cards_dir=cards).is_runnable
    card.write_text("{}", encoding="utf-8")
    assert not check_profile_readiness(research, cards_dir=cards).is_runnable


def test_partner_requires_structure_or_stability_on_each_chain(intent):
    bundles, report, _ = _run_fixture(intent)
    [bundle] = bundles
    [ev] = report.candidate_evaluations
    codon = next(m for m in ev.metrics if m.metric_id == "codontransformer_syn_loglik")

    def project(partner):
        copy = report.model_copy_validated(
            candidate_evaluations=[ev.model_copy_validated(metrics=[codon, partner])],
            isolated_sandbox_override=False,
        )
        return project_report_for_intent(copy, {bundle.bundle_id: bundle}, intent, seed_dir=SEED,
                                         disallow_sandbox_override=True).recommended_bundle_ids

    assert project(STABILITY.model_copy_validated(chain_values=[0.4, 0.5])) == [bundle.bundle_id]
    assert project(MRL.model_copy_validated(chain_values=[0.4, 0.5])) == []
    assert project(MRL.model_copy_validated(metric_id="cds_stability_pred", chain_values=[0.4, 0.5])) == []
    assert project(STABILITY.model_copy_validated(chain_values=[0.5], evaluated_scopes=["inter_chain"])) == []
    for mol in bundle.molecules:
        _, partial, manifest = _run_fixture(intent, FixturePartner(mol.molecule_id))
        assert partial.recommended_bundle_ids == []
        component = next(c for c in manifest.invoked_components if "structrfm" in c.name.lower())
        assert (component.attempt_count, component.failure_count) == (2, 1)


@pytest.mark.parametrize("motif_cds", ["AUGUUUCUGGAA", "AUGUUUUUGGAA"])
def test_provenance_applies_chemistry_to_all_chains_on_cache_only_replay(intent, motif_cds):
    [original], _, _ = _run_fixture(intent.model_copy_validated(uridine_chemistry="m1psi"))
    first, second = original.molecules
    # Keep the RC-005 origin on chain one only. The restriction still covers chain two.
    segments = [s.model_copy_validated(origin_source="fixture") if s.segment_type == "cds" else s
                for s in second.segments]
    changed = second.model_copy_validated(sequence=second.sequence.replace("AUGUUCCUGGAA", motif_cds),
                                          segments=segments)
    candidate = ConstructBundle.create(bundle_id="cached", parent_bundle_id=None,
                                       derivation_action=original.derivation_action,
                                       target_stoichiometry=original.target_stoichiometry,
                                       co_expression_mode=original.co_expression_mode, molecules=[first, changed])
    replay = intent.model_copy_validated(
        uridine_chemistry="m1psi",
        objectives=[intent.objectives[0].model_copy_validated(metric_id="cds_stability_pred")],
        runtime_policy=RuntimePolicySpec(enabled_generators=[], enabled_evaluators=["surrogate_v1_clean"],
                                         allow_cached_candidates=True),
    )
    _, report, _ = run_design_pipeline(replay, seed_dir=SEED, cached_candidates=[original, candidate],
                                       isolated_sandbox_override=False)
    assert report.recommended_bundle_ids == [original.bundle_id]
    assert any(d.candidate_ref == candidate.bundle_id and d.violated_requirement_id.startswith("rc005:")
               and second.molecule_id in d.reason for d in report.failed_diagnostics)
    direct = ScopedConstraintEngine.from_seed_dir(SEED).evaluate_bundle(candidate, [], compliance_track="commercial_clean")
    assert not direct.passed_all_hard_gates


@pytest.mark.parametrize(("mode", "valid"), [
    ("upper", True), ("symmetric", True), ("duplicate", False), ("lower_only", False),
    ("asymmetric", False), ("invalid_utf8", False),
])
def test_posterior_pair_integrity(monkeypatch, tmp_path, mode, valid):
    molecule = _build_preset_reference_r(SEED).molecules[0]

    def write_posterior(argv, **kwargs):
        sequence = Path(argv[2]).read_text().splitlines()[1]
        rows = [f"{i} {base}" for i, base in enumerate(sequence, 1)]
        if mode != "lower_only":
            rows[1] += " 7:0.25"
        if mode == "duplicate":
            rows[1] += " 7:0.25"
        if mode in {"lower_only", "symmetric", "asymmetric"}:
            rows[6] += " 2:0.5" if mode == "asymmetric" else " 2:0.25"
        data = b"\xff" if mode == "invalid_utf8" else "\n".join(rows).encode()
        Path(argv[-1]).write_bytes(data)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(subprocess, "run", write_posterior)
    metrics, _ = _evaluate_molecule_eternafold(molecule, contrafold_bin=tmp_path / "bin", ef_params=tmp_path / "params")
    assert bool(metrics) == valid
    if valid:
        aup = next(m.value for m in metrics if m.metric_id == "eternafold_aup")
        assert aup == round(1 - 0.5 / 90, 5)


@pytest.mark.parametrize("listing", ["non_object", "header_only", "no_header", "junk_after_header", "native"])
def test_darwin_linkage_requires_macho_listing(tmp_path, monkeypatch, intent, listing):
    # Model the successful Darwin command status; some inputs yield only a diagnostic.
    import os
    from mrnax.engine import pareto_director

    binary = tmp_path / "pinned-executable"
    binary.write_bytes(b"#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    params = tmp_path / "params"
    params.write_bytes(b"test")
    row = "\t/usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1.2.3)"
    outputs = {"non_object": f"{binary}: Is Not An Object File", "header_only": f"{binary}:",
               "no_header": row, "junk_after_header": f"{binary}:\n  something",
               "native": f"{binary}:\n{row}"}
    system = os.uname()
    monkeypatch.setattr(os, "uname", lambda: os.uname_result(("Darwin", *system[1:])))
    original = subprocess.check_output
    monkeypatch.setattr(subprocess, "check_output", lambda cmd, **kw: outputs[listing] if cmd[0] == "otool" else original(cmd, **kw))
    for name, value in (("BIN", str(binary)), ("PARAMS", str(params)),
                        ("BIN_SHA256", hashlib.sha256(binary.read_bytes()).hexdigest())):
        monkeypatch.setenv("MRNAX_ETERNAFOLD_" + name, value)
    release = APPROVED_REPLACEMENT_RELEASES["eternafold_structure_head"]
    monkeypatch.setitem(APPROVED_REPLACEMENT_RELEASES, "eternafold_structure_head", release._replace(
        artifact_digests={"parameters/EternaFoldParams.v1": hashlib.sha256(params.read_bytes()).hexdigest()}))
    reached = []
    monkeypatch.setattr(pareto_director, "_evaluate_molecule_eternafold", lambda *a, **kw: (reached.append(True) or [], []))
    [candidate], _, _ = _run_fixture(intent)
    research = intent.model_copy_validated(compliance_track="research_max", runtime_policy=RuntimePolicySpec(
        enabled_generators=[], enabled_evaluators=["eternafold_structure_head"], allow_cached_candidates=True))
    _, report, _ = run_design_pipeline(research, cached_candidates=[candidate], seed_dir=SEED,
                                       isolated_sandbox_override=False)
    assert report.outcome_status == "UNCOMPARABLE_MISSING_APPROVED_COMPONENT"
    assert bool(reached) == (listing == "native")


def test_replacement_card_guidance_matches_commercial_readiness(intent):
    from mrnax.rules.models import load_replacement_card_yaml

    cards = resolve_default_cards_dir()
    rc004, rc005 = [load_replacement_card_yaml(next(cards.glob(f"{pin}-*.yaml")))
                    for pin in ("RC-004", "RC-005")]
    for partner, runnable in [(rc004.component_id, False), ("structrfm_rna_foundation", True),
                              ("surrogate_v1_clean", True)]:
        policy = intent.runtime_policy.model_copy_validated(enabled_evaluators=[partner])
        assert check_profile_readiness(intent.model_copy_validated(runtime_policy=policy)).is_runnable is runnable

    assert rc004.current_license_track == "commercial_clearance_pending"
    assert rc004.current_license_track in rc004.derived_assets_clean_room_rule
    assert "research-only" in rc004.commercial_replacement_paths[0]
    assert "hold" in rc004.overall_switch_rule.lower()
    assert "RC-004" in rc005.role_in_pipeline and "clearance hold" in rc005.role_in_pipeline
    production = " ".join(rc005.commercial_replacement_paths)
    assert "RC-004" not in production and "EternaFold" not in production
    assert "RC-002" in production and "surrogate stability" in production


@pytest.mark.parametrize(("component", "card"), [
    ("eternafold_bpp_engine", "RC-004-eternafold-v1.3.1.yaml"),
    ("eternafold_structure_head", "RC-004-eternafold-v1.3.1.yaml"),
    ("viennarna_thermo", "RC-001-viennarna-v2.8.0.yaml"),
    ("ribonn_noncommercial", None),
])
def test_commercial_refusal_points_to_held_component_card(intent, component, card):
    policy = intent.runtime_policy.model_copy_validated(enabled_evaluators=[component, "structrfm_rna_foundation"])
    ready = check_profile_readiness(intent.model_copy_validated(runtime_policy=policy))
    assert not ready.is_runnable
    message = ready.uncomparable_reason
    assert component in message
    assert {p.name for p in resolve_default_cards_dir().glob("*.yaml") if p.name in message} == (
        {card} if card else set()
    )


@pytest.mark.parametrize(("chemistry", "native"), [
    ("unmodified_u", True), ("m1psi", True), ("5mou", True), ("pseudouridine", False), ("custom_mod", False),
])
def test_eternafold_advisory_agrees_with_native_probe_readiness(intent, monkeypatch, tmp_path, chemistry, native):
    ref = _build_preset_reference_r(SEED).molecules[0]
    molecule = ref.model_copy_validated(chemistry_track=ref.chemistry_track.model_copy_validated(uridine_chemistry=chemistry))

    def posterior(argv, **kwargs):
        sequence = Path(argv[2]).read_text().splitlines()[1]
        Path(argv[-1]).write_text("\n".join(f"{i} {base}" for i, base in enumerate(sequence, 1)))
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(subprocess, "run", posterior)
    metrics, advisories = _evaluate_molecule_eternafold(molecule, contrafold_bin=tmp_path/"bin", ef_params=tmp_path/"params")
    assert len(metrics) == 2
    policy = RuntimePolicySpec(enabled_generators=[], enabled_evaluators=["structrfm_rna_foundation"])
    profile = intent.model_copy_validated(runtime_policy=policy, compliance_track="research_max", uridine_chemistry=chemistry)
    assert check_profile_readiness(profile).is_runnable == native
    if chemistry == "unmodified_u":
        assert advisories == []
    else:
        note, = advisories
        assert chemistry in note.message and "RC-002" in note.message
        assert ("Pair with" in note.message) == native
        assert ("no native validated" in note.message) == (not native)


@pytest.mark.parametrize("chemistry", ["unmodified_u", "m1psi", "5mou", "pseudouridine", "custom_mod"])
@pytest.mark.parametrize("with_alternative", [True, False])
def test_requested_chemistry_controls_live_native_probe_and_partner(intent, chemistry, with_alternative):
    native = chemistry in {"unmodified_u", "m1psi", "5mou"}
    evaluators = ["structrfm_rna_foundation"] + (["surrogate_v1_clean"] if with_alternative else [])
    profile = intent.model_copy_validated(uridine_chemistry=chemistry,
        runtime_policy=intent.runtime_policy.model_copy_validated(enabled_evaluators=evaluators))
    ready = check_profile_readiness(profile)
    assert ready.is_runnable == (native or with_alternative)
    bundles, report, manifest = _run_fixture(profile)
    assert bool(bundles) == bool(report.recommended_bundle_ids) == ready.is_runnable
    measured = [m for e in report.candidate_evaluations for m in e.metrics if m.metric_id == STABILITY.metric_id]
    assert bool(measured) == native
    if not native:
        for raw in [report.model_dump_json(), manifest.model_dump_json()]:
            assert "unsupported" in raw.lower() and chemistry in raw


def test_rc004_card_routes_modified_accessibility_to_native_probe():
    import yaml

    card = yaml.safe_load(next(resolve_default_cards_dir().glob("RC-004-*.yaml")).read_text())
    text = str(card)
    assert "orthrus_halflife_pred" in text and "endogenous_unmodified_proxy" in text
    assert "SHAPE-MaP" in text and "5mou" in text

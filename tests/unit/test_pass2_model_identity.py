import hashlib

import pytest
import yaml

from mrnax.engine.profiles import check_profile_readiness, resolve_default_cards_dir
from mrnax.engine.surrogates import StructRFMNativeScorer, validate_structrfm_checkpoint_path
from mrnax.rules.models import load_replacement_card_yaml, verify_replacement_card_strict
from mrnax.schemas.construct_bundle import ProteinTargetSpec
from mrnax.schemas.design_intent import DesignIntent, RuntimePolicySpec


@pytest.mark.parametrize(("filename", "component", "version"), [
    ("RC-002-structrfm-v0.0.9.yaml", "structrfm_rna_foundation", "0.0.9"),
    ("RC-003-orthrus-v1.0.yaml", "orthrus_mamba_rna_foundation", "1.0.0"),
    ("RC-005-codontransformer-v1.0.yaml", "codontransformer_cds_baseline", "1.0.0"),
])
def test_replacement_card_exact_release_in_direct_and_profile_gate(tmp_path, filename, component, version):
    source = resolve_default_cards_dir() / filename
    card = load_replacement_card_yaml(source)
    verify_replacement_card_strict(card, expected_component_id=component, expected_version=version)
    is_gen = component == "codontransformer_cds_baseline"
    intent = DesignIntent(
        intent_id="intent", project_id="project", intent_version=1, created_from_turn_ids=["turn"],
        entry_mode="protein_only", compliance_track="commercial_clean",
        protein_targets=[ProteinTargetSpec(protein_sequence="M", accession_or_source="fixture")],
        runtime_policy=RuntimePolicySpec(
            enabled_generators=[component] if is_gen else [],
            enabled_evaluators=["surrogate_v1_clean"] if is_gen else [component],
        ),
    )
    assert check_profile_readiness(intent).is_runnable
    for changed in (
        {"component_id": "unapproved_component"},
        {"approved_software_version": None},
        {"approved_software_version": "99.1.0"},
        {"component_name_and_version": "Unrelated 99.1.0"},
    ):
        altered = card.model_copy_validated(**changed)
        with pytest.raises(ValueError):
            verify_replacement_card_strict(altered)
        (tmp_path / filename).write_text(yaml.safe_dump(altered.model_dump(mode="json")))
        assert not check_profile_readiness(intent, cards_dir=tmp_path).is_runnable
    with pytest.raises(ValueError):
        verify_replacement_card_strict(card.model_copy_validated(approved_software_version="99.1.0"),
                                       expected_component_id=component, expected_version="99.1.0")


@pytest.mark.parametrize("weights_filename", ["model.safetensors", "pytorch_model.bin"])
def test_production_structrfm_loader_rejects_unapproved_weights(tmp_path, monkeypatch, weights_filename):
    checkpoint = tmp_path / "structRFM_checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    weights = checkpoint / weights_filename
    weights.write_bytes(b"checkpoint fixture")
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    (checkpoint / f"{weights_filename}.sha256").write_text(digest)
    monkeypatch.delenv("MRNAX_STRUCTRFM_WEIGHTS_SHA256", raising=False)
    with pytest.raises(ValueError, match="approved lowercase SHA-256"):
        StructRFMNativeScorer.from_checkpoint_and_seed_dir(checkpoint, tmp_path)
    monkeypatch.setenv("MRNAX_STRUCTRFM_WEIGHTS_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        StructRFMNativeScorer.from_checkpoint_and_seed_dir(checkpoint, tmp_path)
    monkeypatch.setenv("MRNAX_STRUCTRFM_WEIGHTS_SHA256", digest)
    with pytest.raises((ValueError, FileNotFoundError)):
        validate_structrfm_checkpoint_path(checkpoint, must_exist=True)
    assert validate_structrfm_checkpoint_path(checkpoint, must_exist=True, allow_sandbox_override=True) == checkpoint

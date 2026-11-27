import hashlib
import json
import runpy
from pathlib import Path

import pandas as pd
import pytest

from mrnax.constraints.registry import ScopedConstraintEngine
from mrnax.engine.surrogates import _load_verified_joblib
from mrnax.federated.external_evidence_adapter import CleanRoomPolicyError, ExternalEvidenceAdapterV2
from mrnax.io_insdc.genbank_io import export_construct_package, import_construct_from_genbank_package
from mrnax.schemas.construct_bundle import ConstructBundle

ROOT = Path(__file__).resolve().parents[2]
SEED = ROOT / "registry_seed"


@pytest.fixture
def records():
    snapshot = json.loads((SEED / "m2_m4_federated_12api_snapshot_v2.json").read_text())
    return {r["source_name"]: r for r in snapshot["live_api_snapshots"].values()}


@pytest.mark.parametrize("purpose", ["commercial_training", "commercial_redistribution_export"])
def test_all_frozen_source_decisions_match_at_point_of_use(records, purpose):
    gate = ExternalEvidenceAdapterV2()
    assert len(records) == 12
    rejected = {name for name, rec in records.items() if rec["license_tier"] != "commercial_clean"}
    assert rejected == {"ChEMBL", "HumanProteinAtlas"}
    for name, rec in records.items():
        if name in rejected:
            with pytest.raises(CleanRoomPolicyError):
                gate.guard_export_to_commercial_bundle(rec, purpose=purpose)
        else:
            gate.guard_export_to_commercial_bundle(rec, purpose=purpose)


@pytest.mark.parametrize("changed", [
    {"request_url": "not a URL"}, {"request_url": "ftp://rest.uniprot.org/uniprotkb/test"},
    {"request_url": "https://rest.uniprot.org/uniprotkb/bad%zz"},
    {"request_url": "https://unapproved.invalid/uniprotkb/test"},
    {"payload_sha256": "Z" * 64}, {"payload_sha256": "F" * 64}, {"payload_sha256": "0" * 64},
    {"source_name": "unregistered"}, {"license_tier": "research_only"},
])
def test_malformed_or_unapproved_attribution_is_rejected(records, changed):
    record = {**records["UniProt"], **changed}
    with pytest.raises(CleanRoomPolicyError):
        ExternalEvidenceAdapterV2().guard_export_to_commercial_bundle(record, purpose="commercial_training")


def test_shared_hostname_cannot_relabel_restricted_provider_as_metadata(records):
    mislabeled = {**records["EuropePMC"], "request_url": records["ChEMBL"]["request_url"]}
    with pytest.raises(CleanRoomPolicyError, match="allowlist"):
        ExternalEvidenceAdapterV2().guard_export_to_commercial_bundle(mislabeled, purpose="commercial_training")


def test_product_export_roundtrip_preserves_evidence_and_blocks_disallowed_source(tmp_path, records):
    fixture = json.loads((ROOT / "tests/golden/m2_benchmark1_dual_speed_feedback_fixture_v1.json").read_text())
    prior = ConstructBundle.model_validate(fixture["bundles"]["B1_REF_A"])
    fields = dict(bundle_id=prior.bundle_id, parent_bundle_id=prior.parent_bundle_id,
                  derivation_action=prior.derivation_action, target_stoichiometry=prior.target_stoichiometry,
                  co_expression_mode=prior.co_expression_mode, molecules=prior.molecules)
    allowed = [r for r in records.values() if r["license_tier"] == "commercial_clean"]
    product = ConstructBundle.create(**fields, external_evidence=allowed)
    policy = dict(constraint_engine=ScopedConstraintEngine.from_seed_dir(SEED), constraints=[],
                  compliance_track="commercial_clean")
    package = tmp_path / "approved"
    manifest = export_construct_package(product, package, **policy)
    assert manifest["external_evidence"] == allowed
    restored = import_construct_from_genbank_package(package)
    assert restored == product
    disallowed = ConstructBundle.create(**fields, external_evidence=[*allowed, records["ChEMBL"]])
    with pytest.raises(CleanRoomPolicyError):
        export_construct_package(disallowed, tmp_path / "restricted", **policy)
    assert not (tmp_path / "restricted").exists()


def test_training_table_requires_registered_bytes_and_permitted_evidence(tmp_path, monkeypatch, records):
    globals_ = runpy.run_path(str(ROOT / "scripts/run_m15_tribunal_and_ood_benchmarks.py"))
    read_training_table = globals_["_read_training_table"]
    monkeypatch.setitem(read_training_table.__globals__, "SEED_DIR", tmp_path)
    table = tmp_path / "inputs.parquet"
    pd.DataFrame({"value": [1, 2]}).to_parquet(table)
    digest = hashlib.sha256(table.read_bytes()).hexdigest()
    inventory = tmp_path / "training_table_inventory.json"
    provenance = {}
    inventory.write_text("{}")
    with pytest.raises(CleanRoomPolicyError, match="Unregistered"):
        read_training_table(table, provenance)
    for checksum, evidence, expected in [
        ("a" * 64, [records["UniProt"]], "digest"),
        (digest, [records["UniProt"], records["ChEMBL"]], "BLOCKED"),
        (digest, [records["UniProt"]], None),
    ]:
        inventory.write_text(json.dumps({table.name: {"sha256": checksum, "external_evidence": evidence}}))
        if expected:
            with pytest.raises(CleanRoomPolicyError, match=expected):
                read_training_table(table, provenance)
        else:
            assert read_training_table(table, provenance)["value"].tolist() == [1, 2]
            assert provenance == {table.name: {"sha256": digest, "bytes": table.stat().st_size}}


def test_fastutr_release_inventory_manifest_and_embedded_training_tag_agree():
    license_doc = json.loads((SEED / "LICENSE_RECORD_m15_commercial_clean_v1.json").read_text())
    directory = SEED / "surrogate_v1"
    manifest = json.loads((directory / "TRAINING_PROVENANCE_MANIFEST.json").read_text())
    filename = "utr3_stability_pure_seq_lgbm.joblib"
    head = next(row for row in license_doc["exported_model_artifacts"] if Path(row["artifact"]).name == filename)
    entry = next(row for row in manifest["artifacts"] if row["filename"] == filename)
    upstream = next(row for row in license_doc["upstream_sources"] if row["dataset_id"] == head["training_dataset"])
    assert entry["primary_dataset_id"] == upstream["dataset_id"] == "PMC8728028_Siegel2022_fastUTR_Saluki_Fig6_S7"
    assert entry["citation"] == upstream["citation"]
    assert "Siegel_testSet" in upstream["training_input"]
    artifact = _load_verified_joblib(directory, filename)
    assert entry["embedded_dataset_tag"] == artifact["dataset"] == "fastUTR_Beas2B_Jurkat_MPRA"
    assert entry["target_column_exact"] == artifact["target"] == "stability_mean_2cell"
    assert head["target_column"].split()[0] == artifact["target"]

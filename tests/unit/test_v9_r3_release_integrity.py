"""Round-2 release controls for evidence exports, interpretation identity, and worker precision."""

import json
import subprocess
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlsplit

import numpy as np
import pytest

from mrnax.engine.profiles import resolve_default_seed_dir
from mrnax.federated.external_evidence_adapter import (
    CleanRoomPolicyError,
    ExternalEvidenceAdapterV2,
    sha256_canonical,
)
from mrnax.rules.models import ModelInterpretationNote
from mrnax.rules.tribunal import RuleRegistry


def _export(adapter, summary, batch, query):
    summary = {"commercial_clean_article_eligible": True, "is_open_access": True, **summary}
    record = {
        "source_name": "EuropePMC",
        "license_spdx": "CC0-1.0_Metadata",
        "license_tier": "commercial_clean",
        "request_url": "https://www.ebi.ac.uk/europepmc/webservices/rest/search?" + urlencode({"query": query}),
        "summary": summary,
        "payload_sha256": sha256_canonical(summary),
    }
    if batch:
        assert adapter.filter_and_export_commercial_records([record]) == [record]
    else:
        adapter.guard_export_to_commercial_bundle(record, purpose="commercial_redistribution_export")


@pytest.mark.parametrize("batch", [False, True], ids=["single", "batch"])
@pytest.mark.parametrize("citation", [
    "10.7554/eLife.97682", "PMID:39964837", "PMC11835390", 39964837,
    "https://example.org/search?query=DOI%3A10.7554%2FeLife.97682",
])
def test_su_citation_exports_metadata_but_blocks_coordinate_data(citation, batch):
    adapter = ExternalEvidenceAdapterV2()
    query = str(citation)
    if query.startswith("https://"):
        query = parse_qs(urlsplit(query).query)["query"][0]
    for data in (
        {"rows": [{"chromosome": "1", "position": 1, "reference": "A", "alternate": "G"}]},
        {"chrom": ["1"], "pos": [1], "ref": ["A"], "alt": ["G"]},
        {"columns": ["#CHROM", "POS", "REF", "ALT"], "rows": [["1", 1, "A", "G"]]},
        {"columns": [{"name": "chr"}, {"name": "pos"}], "data": [["1", 1]]},
        {"variantRecords": [{"sample": "example"}]},
        {"loci": ["chr1:123:A>G"]},
    ):
        with pytest.raises(CleanRoomPolicyError):
            _export(adapter, {"citation": {"id": citation}, "tables": [data]}, batch, query)
    _export(adapter, {"citation": citation, "author": "Christopher Example",
                      "title": "Chromosome position and UTR variants in HGMD", "reference": "Article"}, batch, query)
    _export(adapter, {"citation": "PMID:399648370", "rows": [{"chrom": 1, "pos": 1}]}, batch, "PMID:399648370")


@pytest.mark.parametrize(("component", "name", "allowed"), [
    ("surrogate_v1_clean", "Independent scorer [pin: RC-004-eternafold-v1.3.1.yaml]", False),
    ("surrogate_v1_clean", "RiboNN", False),
    ("surrogate_v1_clean", "CommercialCleanSurrogateV2_RiboNN", False),
    (None, "SurrogateV2 [pin: RC-002 structRFM 0.0.9 MIT]", False),
    (None, "Renamed Structure Scorer", False),
    ("surrogate_v1_clean", "Independent scorer", True),
    ("structrfm_rna_foundation", "structRFM [pin: RC-002 structRFM 0.0.9 MIT]", True),
    (None, "CommercialCleanSurrogateV2_LightGBM", True),
])
def test_interpretation_reconciles_all_recorded_identities(component, name, allowed):
    seed = resolve_default_seed_dir()
    reg = RuleRegistry(seed / "rules_commercial_v1.json", track="commercial_clean")
    rule = next(r for r in reg.get_active_rules(chemistry="m1psi", context="m1Psi_IVT_mRNA_design")
                if r.rule_id.endswith("012"))
    note = ModelInterpretationNote(source_component_id=component, source_model_name=name,
                                   source_model_license_track="commercial_clean",
                                   method="regression_weight", summary="Synthetic provenance control")
    rules = [rule.model_copy_validated(model_interpretations=[note])]
    if allowed:
        reg.validate_rules_integrity(rules, track="commercial_clean")
    else:
        with pytest.raises(ValueError):
            reg.validate_rules_integrity(rules, track="commercial_clean")


def test_structured_interpretation_fields_reconcile_with_display_and_component():
    seed = resolve_default_seed_dir()
    reg = RuleRegistry(seed / "rules_commercial_v1.json", track="commercial_clean")
    rule = next(r for r in reg.get_active_rules(chemistry="m1psi", context="m1Psi_IVT_mRNA_design")
                if r.rule_id.endswith("012"))
    fields = dict(component_id="structrfm_rna_foundation", display_name="Independent model",
                  model_id="structrfm_rna_foundation", card_pin="RC-002",
                  replacement_card="RC-002-structrfm-v0.0.9.yaml",
                  source_model_license_track="commercial_clean", method="SHAP", summary="Audit")
    for change in ({}, {"model_id": "RiboNN"}, {"card_pin": "RC-004"}, {"card_pin": "structrfm"},
                   {"replacement_card": "RC-999"}, {"replacement_card": "RC-002-fabricated.yaml"},
                   {"display_name": "EternaFold"}):
        note = ModelInterpretationNote.model_validate(fields | change)
        rules = [rule.model_copy_validated(model_interpretations=[note])]
        if change:
            with pytest.raises(ValueError):
                reg.validate_rules_integrity(rules, track="commercial_clean")
        else:
            reg.validate_rules_integrity(rules, track="commercial_clean")


def test_interpretation_clean_identity_requires_its_clearance_card(tmp_path, monkeypatch):
    from mrnax.engine import profiles

    seed = resolve_default_seed_dir()
    reg = RuleRegistry(seed / "rules_commercial_v1.json", track="commercial_clean")
    rule = next(r for r in reg.get_active_rules(chemistry="m1psi", context="m1Psi_IVT_mRNA_design")
                if r.rule_id.endswith("012"))
    note = ModelInterpretationNote(source_component_id="structrfm_rna_foundation", source_model_name="Scorer",
                                   source_model_license_track="commercial_clean",
                                   method="SHAP", summary="Audit")
    rules = [rule.model_copy_validated(model_interpretations=[note])]
    reg.validate_rules_integrity(rules, track="commercial_clean")
    monkeypatch.setattr(profiles, "resolve_default_cards_dir", lambda: tmp_path)
    with pytest.raises(ValueError):
        reg.validate_rules_integrity(rules, track="commercial_clean")


def test_worker_preserves_native_scaler_precision_and_predictions(tmp_path: Path):
    from mrnax.engine._darwin_joblib_worker import get_darwin_joblib_client

    weights = resolve_default_seed_dir() / "surrogate_v1"
    names = ["orthrus6_saluki_halflife_probe", *[
        f"structrfm_shapemap_{chem}_probe" for chem in ("U", "moU", "mpU")
    ]]
    # Run native sklearn in its own interpreter to keep OpenMP out of the parent.
    native_script = """
import json, joblib, numpy as np, sys
from pathlib import Path
weights, dest, names = sys.argv[1:]
for name in json.loads(names):
    bundle = joblib.load(Path(weights) / (name + '.joblib'))
    for dtype in (np.float32, np.float64):
        x = np.random.default_rng(42).standard_normal((10, bundle['scaler'].n_features_in_)).astype(dtype)
        scaled = bundle['scaler'].transform(x)
        np.savez(Path(dest) / (name + dtype.__name__), x=x, scaled=scaled, y=bundle['model'].predict(scaled))
"""
    subprocess.run([sys.executable, "-c", native_script, str(weights), str(tmp_path), json.dumps(names)], check=True)
    worker = get_darwin_joblib_client()
    for name in names:
        bundle = worker.load_verified_bytes(name, (weights / (name + ".joblib")).read_bytes())
        for dtype in (np.float32, np.float64):
            with np.load(tmp_path / f"{name}{dtype.__name__}.npz") as native:
                scaled = bundle["scaler"].transform(native["x"])
                assert scaled.dtype == native["scaled"].dtype
                np.testing.assert_array_equal(scaled, native["scaled"])
                assert np.max(np.abs(bundle["model"].predict(scaled) - native["y"])) == 0.0


def _freeze_europepmc(monkeypatch, query, title):
    article = {"pmid": "39964837", "pmcid": "PMC11835390", "doi": "10.7554/eLife.97682",
               "license": "cc by", "isOpenAccess": "Y", "title": title}
    monkeypatch.setattr("mrnax.federated.external_evidence_adapter._http_json",
                        lambda url: {"resultList": {"result": [article]}})
    return ExternalEvidenceAdapterV2().check_paper_license_europepmc(query)


def _export_frozen(record, batch):
    adapter = ExternalEvidenceAdapterV2()
    if batch:
        assert adapter.filter_and_export_commercial_records([record]) == [record]
    else:
        adapter.guard_export_to_commercial_bundle(record, purpose="commercial_redistribution_export")


@pytest.mark.parametrize("batch", [False, True], ids=["single", "batch"])
@pytest.mark.parametrize("query", ["PMCID:PMC11835390", "pmid:39964837", "DOI:10.7554/ELIFE.97682",
                                  "PMC11835390", "39964837", "10.7554/eLife.97682"])
@pytest.mark.parametrize("title", ["A citation", "Descriptive studies of HGMD variants"])
def test_europepmc_frozen_article_exports_by_returned_identifier(monkeypatch, batch, query, title):
    record = _freeze_europepmc(monkeypatch, query, title)
    _export_frozen(record, batch)


@pytest.mark.parametrize("batch", [False, True], ids=["single", "batch"])
def test_europepmc_title_envelope_still_screens_data(monkeypatch, batch):
    for title, extra in [
        ("chr1:123:A>G in HGMD", {}),
        ("HGMD", {"rows": [{"chr": 1, "pos": 4, "ref": "A", "alt": "T"}]}),
        ("HGMD", {"variants": [{}]}),
        ("HGMD", {"source_id": "su2025_elife97682"}),
        ("A citation", {"source_id": "raw_HGMD"}),
        ("HGMD", {"text_excerpt": [{"chrom": 1, "pos": 4}]}),
    ]:
        record = _freeze_europepmc(monkeypatch, "PMCID:PMC11835390", title)
        record["summary"]["title_sandboxed"].update(extra)
        record["payload_sha256"] = sha256_canonical(record["summary"])
        with pytest.raises(CleanRoomPolicyError):
            _export_frozen(record, batch)


@pytest.mark.parametrize("batch", [False, True], ids=["single", "batch"])
def test_europepmc_attribution_checks_the_complete_search_identifier(monkeypatch, batch):
    record = _freeze_europepmc(monkeypatch, "PMCID:PMC11835390", "A citation")
    endpoint = record["request_url"].partition("?")[0]
    for query in (
        "query=PMCID:PMC118353901", "query=PMCID:PMC999&extra=PMCID:PMC11835390",
        "query=PMCID:PMC999&query=PMCID:PMC11835390", "query=open#PMCID:PMC11835390",
        "query=DOI:10.7554%2Felife.976820", "query=PMID:399648370", "format=json",
    ):
        with pytest.raises(CleanRoomPolicyError):
            _export_frozen({**record, "request_url": endpoint + "?" + query}, batch)
    record["summary"].update(pmid=None, doi=None, pmcid=None)
    record["payload_sha256"] = sha256_canonical(record["summary"])
    with pytest.raises(CleanRoomPolicyError):
        _export_frozen(record, batch)

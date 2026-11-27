"""Round-4 export contracts for attributed article metadata and governed variants."""

from unittest.mock import patch
from urllib.parse import urlencode

import pytest

from mrnax.federated.external_evidence_adapter import (
    CleanRoomPolicyError,
    ExternalEvidenceAdapterV2,
    sha256_canonical,
)


EUROPEPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search?"
IDENTIFIERS = (("pmid", 39964837), ("pmid", "39964837"), ("pmcid", "PMC11835390"),
               ("doi", "10.7554/eLife.97682"))


def export(record, batch):
    adapter = ExternalEvidenceAdapterV2()
    if batch:
        assert adapter.filter_and_export_commercial_records([record]) == [record]
    else:
        adapter.guard_export_to_commercial_bundle(record, purpose="commercial_redistribution_export")


@pytest.mark.parametrize("field,identifier", IDENTIFIERS)
@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("location", ["title", "annotation"])
@pytest.mark.parametrize("notation", ["c.-12A>G", "c.*12A>G", "g.123_124del", "c.12A>G",
                                      "c.23+1_23+2delinsGC", "chr1:123_124del", "n.12_13insU"])
def test_governed_variant_in_attributed_article(field, identifier, batch, location, notation):
    text = f"An HGMD study of {notation}"
    article = {field: identifier, "isOpenAccess": "Y", "license": "cc by",
               "title": text if location == "title" else "An HGMD study"}
    with patch("mrnax.federated.external_evidence_adapter._http_json",
               return_value={"resultList": {"result": [article]}}):
        record = ExternalEvidenceAdapterV2().check_paper_license_europepmc(f"{field}:{identifier}")
    if location == "annotation":
        record["summary"]["appendix"] = [{"annotations": [f"Observed {notation}"]}]
        record["payload_sha256"] = sha256_canonical(record["summary"])
    with pytest.raises(CleanRoomPolicyError, match="variant/dataset"):
        export(record, batch)


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("field,identifier", IDENTIFIERS)
@pytest.mark.parametrize("location", ["record", "summary", "citation_dict", "citation_text"])
def test_article_identifier_binds_legacy_and_producer_urls(batch, field, identifier, location):
    summary = {"commercial_clean_article_eligible": True, "is_open_access": True,
               "title": "HGMD and the history of UTR variants", "author": "Christopher Lee"}
    record = {"source_name": "EuropePMC", "license_spdx": "CC0-1.0_Metadata",
              "license_tier": "commercial_clean", "summary": summary}
    if location == "record":
        record[field] = identifier
    elif location == "summary":
        summary[field] = identifier
    elif location == "citation_dict":
        summary["citation"] = {field: identifier}
    else:
        summary["citation"] = f"{field.upper()}:{identifier}"
    record["payload_sha256"] = sha256_canonical(summary)
    record["request_url"] = EUROPEPMC + urlencode({"query": f"{field.upper()}:{identifier}"})
    export(record, batch)
    for wrong in ("unrelated", "PMID:9", "PMCID:PMC42", "DOI:10.9999/other"):
        record["request_url"] = EUROPEPMC + urlencode({"query": wrong})
        with pytest.raises(CleanRoomPolicyError, match="attribution|article|match"):
            export(record, batch)


@pytest.mark.parametrize("batch", [False, True])
def test_unrelated_referenced_article_cannot_attribute_source_query(batch):
    summary = {"commercial_clean_article_eligible": True, "is_open_access": True,
               "pmid": "39964837", "references": [{"pmid": "12345"}]}
    record = {"source_name": "EuropePMC", "license_spdx": "CC0-1.0_Metadata",
              "license_tier": "commercial_clean", "summary": summary,
              "payload_sha256": sha256_canonical(summary), "request_url": EUROPEPMC + "query=PMID:12345"}
    with pytest.raises(CleanRoomPolicyError):
        export(record, batch)


@pytest.mark.parametrize("query", ["unrelated", "PMID:39964837"])
def test_identifier_in_legacy_citation_text(query):
    summary = {"commercial_clean_article_eligible": True, "is_open_access": True,
               "citation": "Su et al. (2025), PMID: 39964837."}
    record = {"source_name": "EuropePMC", "license_spdx": "CC0-1.0_Metadata",
              "license_tier": "commercial_clean", "summary": summary,
              "payload_sha256": sha256_canonical(summary), "request_url": EUROPEPMC + "query=" + query}
    if query == "unrelated":
        with pytest.raises(CleanRoomPolicyError):
            export(record, False)
    else:
        export(record, False)

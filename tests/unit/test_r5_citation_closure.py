"""Public export decisions for governed HGVS annotations and legacy article URLs."""

from urllib.parse import urlencode

import pytest

from mrnax.federated.external_evidence_adapter import (
    CleanRoomPolicyError,
    ExternalEvidenceAdapterV2,
    sha256_canonical,
)


def article_record(citation, query, **metadata):
    summary = {
        "citation": citation,
        "title": "A history of HGMD",
        "author": "Christopher Example",
        "is_open_access": True,
        "commercial_clean_article_eligible": True,
        **metadata,
    }
    return {
        "source_name": "EuropePMC",
        "license_tier": "commercial_clean",
        "license_spdx": "CC0-1.0_Metadata",
        "request_url": "https://www.ebi.ac.uk/europepmc/webservices/rest/search?" + urlencode({"query": query}),
        "summary": summary,
        "payload_sha256": sha256_canonical(summary),
    }


def export(record, batch):
    adapter = ExternalEvidenceAdapterV2()
    if batch:
        assert adapter.filter_and_export_commercial_records([record]) == [record]
    else:
        adapter.guard_export_to_commercial_bundle(record, purpose="commercial_redistribution_export")


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("notation", [
    "g.123_124delinsN", "g.123_124delinsACGTUNRYWSMKBDHV", "c.-13_-12insN",
    "r.20_21delins(n)", "n.5_6insN[(3_7)]", "g.(100_110)_(200_210)del",
    "g.(?_100)_(200_?)dup", "m.100_(200_210)inv", "c.(-20_-10)_(*1_*5)delinsN",
    "c.23+?_24-?del", "r.(12a>u)", "g.[123G>A;345del]", "r.12_14[6]",
    "g.123CAG[23]", "g.123_124ins[NC_000022.10:g.20_24]", "c.[=];[123A>G]",
    "r.(spl?)", "r.0?", "g.123_127|gom", "r.?_123::NM_123456.1:r.50_?",
])
def test_governed_hgvs_across_export_apis(batch, notation):
    for metadata in ({"title": "HGMD report " + notation}, {"appendix": [{"annotations": [notation]}]}):
        record = article_record("PMID:39964837", "PMID:39964837", **metadata)
        with pytest.raises(CleanRoomPolicyError, match="variant/dataset"):
            export(record, batch)


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize(("url", "query"), [
    ("https://pubmed.ncbi.nlm.nih.gov/39964837/", "PMID:39964837"),
    ("https://www.ncbi.nlm.nih.gov/pubmed/39964837", "PMID:39964837"),
    ("https://pmc.ncbi.nlm.nih.gov/articles/PMC11835390/", "PMCID:PMC11835390"),
    ("https://ncbi.nlm.nih.gov/pmc/articles/PMC11835390/", "PMCID:PMC11835390"),
    ("https://europepmc.org/article/PMC/PMC11835390", "PMCID:PMC11835390"),
    ("https://europepmc.org/article/MED/39964837", "PMID:39964837"),
    ("https://doi.org/10.7554/eLife.97682", "DOI:10.7554/eLife.97682"),
    ("https://dx.doi.org/10.7554/eLife.97682", "DOI:10.7554/eLife.97682"),
])
def test_canonical_citation_path_binds_matching_query_only(batch, url, query):
    citation = url + "?query=PMID:42#PMC42"
    export(article_record(citation, query), batch)
    for wrong_query in ("unrelated", "PMID:42", "PMCID:PMC42"):
        with pytest.raises(CleanRoomPolicyError, match="article"):
            export(article_record(citation, wrong_query), batch)


@pytest.mark.parametrize("batch", [False, True])
def test_unrelated_host_or_path_cannot_supply_article_id(batch):
    for url in ("https://example.org/pubmed/39964837",
                "https://pubmed.ncbi.nlm.nih.gov/39964837/comments"):
        with pytest.raises(CleanRoomPolicyError, match="article"):
            export(article_record(url, "PMID:39964837"), batch)


@pytest.mark.parametrize("batch", [False, True])
def test_neutral_title_with_initials_and_numbers_is_bibliographic(batch):
    record = article_record("PMID:39964837", "PMID:39964837",
                            title="A history of HGMD by Christopher R. and C. Example, c.1990")
    export(record, batch)

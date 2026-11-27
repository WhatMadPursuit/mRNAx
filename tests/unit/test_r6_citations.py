"""Public citation export decisions for chromosome landmarks and bibliographic identifiers."""

from urllib.parse import urlencode

import pytest

from mrnax.federated.external_evidence_adapter import (
    CleanRoomPolicyError,
    ExternalEvidenceAdapterV2,
    sha256_canonical,
)


def export(citation, query, batch, **metadata):
    summary = {
        "citation": citation,
        "title": "HGMD records described by Christopher Example",
        "is_open_access": True,
        "commercial_clean_article_eligible": True,
        **metadata,
    }
    record = {
        "source_name": "EuropePMC",
        "license_tier": "commercial_clean",
        "license_spdx": "CC0-1.0_Metadata",
        "request_url": "https://www.ebi.ac.uk/europepmc/webservices/rest/search?" + urlencode({"query": query}),
        "summary": summary,
        "payload_sha256": sha256_canonical(summary),
    }
    adapter = ExternalEvidenceAdapterV2()
    if batch:
        assert adapter.filter_and_export_commercial_records([record]) == [record]
    else:
        adapter.guard_export_to_commercial_bundle(record, purpose="commercial_redistribution_export")


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("location", ["title", "annotation"])
@pytest.mark.parametrize("field,value", [
    ("pmid", 39964837), ("pmid", "39964837"), ("doi", "10.7554/eLife.97682"), ("pmcid", "PMC11835390"),
])
@pytest.mark.parametrize("notation", [
    "NC_000002.12:g.pter_8247756delins[NC_000011.10:g.pter_15825266]",
    "g.pter_124delinsN", "g.123_qterdelinsN",
])
def test_landmark_delins_blocks_attributed_metadata(batch, location, field, value, notation):
    metadata = {"title": notation} if location == "title" else {"appendix": [{"annotations": [notation]}]}
    with pytest.raises(CleanRoomPolicyError, match="variant/dataset"):
        export({field: value}, f"{field}:{value}", batch, **metadata)


@pytest.mark.parametrize("notation", [
    "c.PTERn>g", "n.(pter_?)_(cen_123)del", "r.cen_qterinsN", "m.CENdup2",
    "g.(pter_cen)_qterinv", "c.-?_+12conN", "r.(cen)_(*2+?_qter)delins4",
])
def test_mixed_landmark_and_uncertain_breakpoints(notation):
    with pytest.raises(CleanRoomPolicyError, match="variant/dataset"):
        export("PMID:39964837", "PMID:39964837", False, title=notation)


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("form", ["bare", "inline", "default_port", "label"])
@pytest.mark.parametrize("site,path,identifier", [
    ("pubmed.ncbi.nlm.nih.gov", "/39964837/", "PMID:39964837"),
    ("pmc.ncbi.nlm.nih.gov", "/articles/PMC11835390/", "PMCID:PMC11835390"),
    ("doi.org", "/10.7554/eLife.97682", "DOI:10.7554/eLife.97682"),
])
def test_bibliographic_identifier_binds_both_export_routes(batch, form, site, path, identifier):
    url = f"https://{site}{path}"
    citation = {
        "bare": url,
        "inline": f"Su et al. (2025). Available at <{url}>.",
        "default_port": f"https://{site}:443{path}",
        "label": f"Su et al. (2025). {identifier}.",
    }[form]
    export(citation, identifier, batch)
    with pytest.raises(CleanRoomPolicyError, match="article"):
        export(citation, "unrelated", batch)


@pytest.mark.parametrize("batch", [False, True])
def test_bibliography_collects_every_url_and_explicit_token(batch):
    citation = (
        "Available at http://WWW.NCBI.NLM.NIH.GOV:80/pubmed/1234/, "
        "(https://ncbi.nlm.nih.gov:8443/pmc/articles/PMC5678); "
        "https://dx.doi.org:443/10.9999/example?query=PMID:9876#PMC543. "
        "Other identifiers include PMID: 6789 and PMCID: PMC4321."
    )
    for query in ("PMID:1234", "PMCID:PMC5678", "DOI:10.9999/example", "PMID:6789", "PMCID:PMC4321"):
        export(citation, query, batch)
    for query in ("PMID:9876", "PMCID:PMC543", "unrelated"):
        with pytest.raises(CleanRoomPolicyError, match="article"):
            export(citation, query, batch)


@pytest.mark.parametrize("source,path,identifier", [
    ("europepmc.org", "article/MED/12345", "PMID:12345"),
    ("www.europepmc.org", "article/PMC/PMC11111", "PMCID:PMC11111"),
])
def test_europepmc_bibliographic_link(source, path, identifier):
    export(f"Source: [http://{source}:80/{path}].", identifier, False)


@pytest.mark.parametrize("url", [
    "https://pubmed.ncbi.nlm.nih.gov@example.org/1234/",
    "https://example.org@pubmed.ncbi.nlm.nih.gov:443/1234/",
    "https://pubmed.ncbi.nlm.nih.gov:invalid/1234/",
    "https://pubmed.ncbi.nlm.nih.gov:443/1234/comments?query=PMID:1234",
])
def test_bibliographic_link_requires_public_article_authority_and_path(url):
    with pytest.raises(CleanRoomPolicyError, match="article"):
        export(f"Su et al. Available at {url}.", "PMID:1234", False)


@pytest.mark.parametrize("citation", [
    "Su et al. (https://doi.org:443/10.1234/alpha(beta)).",
    "Su et al. (DOI:10.1234/alpha(beta)).",
])
def test_sentence_punctuation_preserves_balanced_doi_suffix(citation):
    export(citation, "DOI:10.1234/alpha(beta)", False)
    with pytest.raises(CleanRoomPolicyError, match="article"):
        export(citation, "DOI:10.1234/alpha(beta", False)

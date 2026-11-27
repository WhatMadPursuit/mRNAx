#!/usr/bin/env python3
"""mRNAx v2 · M2/M4 Commercial-Clean 12-API Federated External Evidence Adapter v2.

Federated live client for 12 official public biomedical APIs with strict Clean-Room & Security enforcement:
  1. UniProt REST (CC-BY-4.0) — protein sequence, signal peptide, transmembrane & domain topology
  2. Ensembl REST (Apache-2.0) — gene/canonical transcript coordinates
  3. RCSB PDB Search + Core API (CC0-1.0) — experimental 3D Fab/target structures
  4. ChEMBL REST v3 (CC-BY-SA-3.0 — STRICT lookup_only_sharealike_boundary + live chembl_db_version)
  5. OpenTargets GraphQL v4 (CC0 / Public) — disease associations & antibody tractability
  6. ClinicalTrials.gov REST v2 (US Public Domain) — clinical trial landscape (sandboxed untrusted text)
  7. GTEx Portal API v2 (Public aggregate TPM — STRICTLY blocked for mature miRNAs!)
  8. AlphaFold DB API (CC-BY-4.0) — 3D predicted confidence / pLDDT / disordered regions for co-translational slowdown
  9. Human Protein Atlas (HPA) REST API (CC-BY-SA-3.0 — STRICT lookup_only_sharealike_boundary + blocked for mature miRNAs! Subcellular localization: secreted vs membrane vs cytosolic)
 10. Europe PMC REST API (CC0 / Public metadata) — AUTOMATED Clean-Room OA & License Firewall (`isOpenAccess` / `license`)
 11. NCBI E-utilities (PubMed + GenBank/RefSeq Nucleotide, US Public Domain) — curated NM_*/PP784251 transcripts & sandboxed literature
 12. RNAcentral REST API (CC0 / RNAcentral terms) — mature miRNA / ncRNA sequence & accession verification

4 Fail-Closed Negative Guardrails enforced in code:
  - Block GTEx for mature miRNA queries
  - Block Human Protein Atlas (HPA) for mature miRNA queries
  - Block Thera-SAbDab in `commercial_clean` track
  - Block ShareAlike (`ChEMBL`, `HPA`) & pending/research_only records from commercial training or redistribution exports
"""
from __future__ import annotations

import hashlib
import json
import re
import ssl
import time
import urllib.parse
import urllib.request
from typing import Any

USER_AGENT = "mRNAx-v2-ExternalEvidenceAdapter/2.0 (commercial_clean; contact=research)"
MATURE_MIRNA_RE = re.compile(r"\b(hsa-)?(mir|let)-\d+[a-z]?(-[35]p)?\b|\bmirna\b", re.IGNORECASE)


class CleanRoomPolicyError(ValueError):
    """Raised when a query or export violates mRNAx v2 licensing or biological source boundaries."""


def utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_canonical(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def sandbox_untrusted_text(text: str, source_id: str) -> dict[str, Any]:
    """Wrap external text in a strict read-only untrusted envelope per mRNAx v2 §5.1."""
    sanitized = str(text).replace("</UNTRUSTED_DATA>", "[ESCAPED_TAG]")[:2000]
    return {
        "untrusted_external_data": True,
        "must_not_modify_locked_constraints": True,
        "source_id": source_id,
        "text_excerpt": sanitized,
    }


def _ssl_context() -> ssl.SSLContext | None:
    try:
        import certifi  # type: ignore[import-not-found]

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return None


def _http_json(url: str, method: str = "GET", payload: dict | None = None, timeout: int = 20) -> Any:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    ctx = _ssl_context()
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
        if resp.status == 204:
            return {}
        raw = resp.read().decode("utf-8").strip()
        return json.loads(raw) if raw else {}


def _http_text(url: str, timeout: int = 20) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    ctx = _ssl_context()
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
        return resp.read().decode("utf-8", errors="replace")


class ExternalEvidenceAdapterV2:
    """12-API federated, provenance-freezing biomedical adapter for mRNAx v2 M2/M4."""

    def __init__(self, license_track: str = "commercial_clean") -> None:
        self.license_track = license_track

    def _guard_source(self, source_name: str, query: str) -> None:
        if source_name == "Thera-SAbDab" and self.license_track == "commercial_clean":
            raise CleanRoomPolicyError(
                "Thera-SAbDab is classified as needs_vendor_license / research_only and is BLOCKED in commercial_clean. "
                "Use WHO INN + RCSB PDB (CC0) + ChEMBL (lookup-only) instead."
            )
        if source_name in ("GTEx", "HumanProteinAtlas") and MATURE_MIRNA_RE.search(query):
            raise CleanRoomPolicyError(
                f"{source_name} MUST NOT be queried for mature miRNA expression (query='{query}'). "
                "Use mirna_rbp_commercial_v2.json (miRBase CC0 + FANTOM5 CC-BY-4.0) instead."
            )

    # Several providers share a host. Each license covers its adapter's endpoint scope.
    _COMMERCIAL_ATTRIBUTION = {
        "UniProt": ("rest.uniprot.org", r"/uniprotkb/[^/]+", "CC-BY-4.0"),
        "Ensembl": ("rest.ensembl.org", r"/lookup/symbol/[^/]+/[^/]+", "Apache-2.0"),
        "RCSB_PDB": ("search.rcsb.org", r"/rcsbsearch/v2/query", "CC0-1.0"),
        "AlphaFoldDB": ("alphafold.ebi.ac.uk", r"/api/prediction/[^/]+", "CC-BY-4.0"),
        "OpenTargets": ("api.platform.opentargets.org", r"/api/v4/graphql", "CC0-1.0_Public_Platform"),
        "ClinicalTrials.gov": ("clinicaltrials.gov", r"/api/v2/studies", "US-Public-Domain"),
        "GTEx": ("gtexportal.org", r"/api/v2/expression/medianGeneExpression", "Public_Aggregate_TPM"),
        "EuropePMC": ("www.ebi.ac.uk", r"/europepmc/webservices/rest/search", "CC0-1.0_Metadata"),
        "NCBI_Eutils": ("eutils.ncbi.nlm.nih.gov", r"/entrez/eutils/efetch\.fcgi", "US-Public-Domain"),
        "RNAcentral": ("rnacentral.org", r"/api/v1/rna/(?:[^/]+/)?", "CC0-1.0_with_upstream_terms"),
    }
    _ALLOWED_EXPORT_PURPOSES: frozenset[str] = frozenset(
        {"internal_evidence_lookup", "commercial_training", "commercial_redistribution_export"}
    )

    _FORBIDDEN_EVIDENCE_TOKENS: tuple[str, ...] = (
        "research_only",
        "needs_vendor_license",
        "non-commercial",
        "noncommercial",
        "cc-by-nc",
        "cc by-nc",
        "cc-by-sa",
        "cc by-sa",
        "sharealike",
        "academic_only",
        "academic-only",
        "thera-sabdab",
        "agpl",
        "gpl-",
        "gplv",
    )
    _FORBIDDEN_EVIDENCE_PHRASE_RE = re.compile(
        r"("
        r"\bnot\s+(?:(?:permitted|allowed|licensed|authorized|available|eligible|intended|approved)\s+)?(?:for\s+)?commercial\b"
        r"|\bno\s+commercial\b"
        r"|\bcommercial\s+(?:(?:use|usage|exploitation|distribution|redistribution|licensing|application|deployment|purposes?|rights?|activity|activities)\s+)?"
        r"(?:(?:is|are|remains?|remain|was|were|be|being)\s+)?(?:(?:strictly|expressly|explicitly)\s+)?"
        r"(?:not\s+(?:permitted|allowed|licensed|authorized|approved|available)|prohibited|forbidden|restricted|excluded|disallowed|barred|denied|unlicensed|reserved)\b"
        r"|\b(?:prohibit(?:s|ed|ing)?|forbid(?:s|den|ding)?|disallow(?:s|ed|ing)?|restrict(?:s|ed|ing)?|exclude(?:s|d|ing)?|bar(?:s|red|ring)?)\s+(?:any\s+|all\s+|from\s+)?commercial\b"
        r"|\bwithout\s+(?:a\s+|separate\s+|prior\s+|written\s+|explicit\s+)*(?:commercial\s+)?(?:license|permission|authorization|consent|agreement)\b"
        r"|\brequires?\s+(?:a\s+|separate\s+|commercial\s+|vendor\s+)+(?:license|permission|agreement)\b"
        r"|\b(?:academic|research|educational|internal|personal|evaluation|non[-\s]?profit)\s+(?:use|purposes?|research|evaluation)\s+only\b"
        r"|\bonly\s+for\s+(?:academic|research|educational|internal|personal|non[-\s]?commercial|non[-\s]?profit)\b"
        r"|\bfor\s+(?:academic|research|educational|internal|non[-\s]?commercial|non[-\s]?profit)\s+(?:use|purposes?|research)\b"
        r"|\ball\s+rights\s+reserved\b"
        r")",
        re.IGNORECASE,
    )
    _POSITIVE_COMMERCIAL_BOOL_KEYS: frozenset[str] = frozenset(
        {
            "commercial_use",
            "commercial_allowed",
            "commercial_permitted",
            "commercial_clean",
            "commercial_clean_eligible",
            "commercial_clean_article_eligible",
            "allow_commercial",
            "is_commercial_allowed",
            "redistributable",
        }
    )
    _NEGATIVE_RESTRICTION_BOOL_KEYS: frozenset[str] = frozenset(
        {
            "research_only",
            "non_commercial",
            "noncommercial",
            "academic_only",
            "needs_vendor_license",
            "share_alike",
            "sharealike",
            "copyleft",
            "restricted",
        }
    )

    @classmethod
    def _collect_leaf_strings(cls, val: Any) -> list[str]:
        if isinstance(val, str):
            return [val]
        if isinstance(val, dict):
            out: list[str] = []
            for k, v in val.items():
                out.extend(cls._collect_leaf_strings(k))
                out.extend(cls._collect_leaf_strings(v))
            return out
        if isinstance(val, (list, tuple, set, frozenset)):
            out = []
            for item in val:
                out.extend(cls._collect_leaf_strings(item))
            return out
        return []

    @classmethod
    def _assert_no_forbidden_nested_provenance(cls, node: Any, *, src: str) -> None:
        if isinstance(node, str):
            low = node.strip().lower()
            if any(tok in low for tok in cls._FORBIDDEN_EVIDENCE_TOKENS) or cls._FORBIDDEN_EVIDENCE_PHRASE_RE.search(
                node
            ):
                raise CleanRoomPolicyError(
                    f"Nested non-commercial or copyleft marker {node!r} inside '{src}' evidence is BLOCKED in commercial_clean"
                )
            return
        if isinstance(node, dict):
            for k, v in node.items():
                key_low = str(k).strip().lower()
                if key_low in cls._POSITIVE_COMMERCIAL_BOOL_KEYS and isinstance(v, bool) and v is False:
                    raise CleanRoomPolicyError(
                        f"Nested non-commercial boolean {k}={v!r} inside '{src}' evidence is BLOCKED in commercial_clean"
                    )
                if key_low in cls._NEGATIVE_RESTRICTION_BOOL_KEYS and isinstance(v, bool) and v is True:
                    raise CleanRoomPolicyError(
                        f"Nested restricted boolean {k}={v!r} inside '{src}' evidence is BLOCKED in commercial_clean"
                    )
                if key_low in {"license_tier", "license_track", "tier"}:
                    if not isinstance(v, str) or v.strip() != "commercial_clean":
                        raise CleanRoomPolicyError(
                            f"Nested non-commercial {k}={v!r} inside '{src}' evidence is BLOCKED in commercial_clean"
                        )
                elif any(p in key_low for p in ("license", "spdx", "rights", "access_tier")):
                    leaves = cls._collect_leaf_strings(v)
                    if not leaves and v is not None and not isinstance(v, (bool, dict, list, tuple)):
                        raise CleanRoomPolicyError(
                            f"Malformed non-string rights field {k}={v!r} inside '{src}' evidence is BLOCKED in commercial_clean"
                        )
                    for leaf in leaves:
                        low = leaf.strip().lower()
                        if (
                            any(tok in low for tok in cls._FORBIDDEN_EVIDENCE_TOKENS)
                            or cls._FORBIDDEN_EVIDENCE_PHRASE_RE.search(leaf)
                            or low in {"gpl", "academic", "restricted", "proprietary"}
                        ):
                            raise CleanRoomPolicyError(
                                f"Nested incompatible license {k}={v!r} inside '{src}' evidence is BLOCKED in commercial_clean"
                            )
                cls._assert_no_forbidden_nested_provenance(v, src=src)
        elif isinstance(node, (list, tuple, set, frozenset)):
            for item in node:
                cls._assert_no_forbidden_nested_provenance(item, src=src)

    def guard_export_to_commercial_bundle(self, frozen_record: dict[str, Any], *, purpose: str) -> None:
        """Strict allowlist gate verifying source_name, approved SPDX license, purpose, attribution URL & digest."""
        if purpose not in self._ALLOWED_EXPORT_PURPOSES:
            raise CleanRoomPolicyError(
                f"Unrecognized export purpose '{purpose}'; must be one of {sorted(self._ALLOWED_EXPORT_PURPOSES)}"
            )
        src = str(frozen_record.get("source_name") or "").strip()
        spdx = str(frozen_record.get("license_spdx") or "").strip()
        tier = str(frozen_record.get("license_tier") or "").strip()
        req_url = frozen_record.get("request_url")
        sha = frozen_record.get("payload_sha256")
        try:
            url = urllib.parse.urlsplit(req_url)
            valid_url = (isinstance(req_url, str) and not re.search(r"[\s\x00-\x1f\x7f\\]|%(?![a-fA-F0-9]{2})", req_url)
                         and url.scheme in {"http", "https"} and bool(url.hostname)
                         and url.username is None and url.password is None and url.port != 0)
        except (ValueError, TypeError):
            valid_url = False
        if not src or not spdx or not valid_url or not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
            raise CleanRoomPolicyError(f"BLOCKED malformed attribution URL or SHA-256 for '{src}'")
        if "summary" in frozen_record and sha256_canonical(frozen_record["summary"]) != sha:
            raise CleanRoomPolicyError(f"BLOCKED frozen summary digest mismatch for '{src}'")
        if purpose == "internal_evidence_lookup":
            if tier in {"needs_vendor_license", "research_only"} and self.license_track == "commercial_clean":
                raise CleanRoomPolicyError(f"Source '{src}' (tier={tier}) is BLOCKED for commercial lookup")
            return
        self._assert_no_forbidden_nested_provenance(frozen_record, src=src)
        approved = self._COMMERCIAL_ATTRIBUTION.get(src)
        path = urllib.parse.unquote(url.path)
        if (approved is None or url.hostname != approved[0] or spdx != approved[2]
                or not re.fullmatch(approved[1], path) or ".." in path.split("/") or "." in path.split("/")
                or url.port not in {None, 443 if url.scheme == "https" else 80}
                or tier != "commercial_clean"):
            raise CleanRoomPolicyError(f"Source '{src}' is BLOCKED by the {purpose} allowlist")
        if not isinstance(frozen_record.get("summary"), dict):
            raise CleanRoomPolicyError(f"BLOCKED missing frozen summary for '{src}'")
        if src == "EuropePMC":
            summ = frozen_record["summary"]
            if summ.get("commercial_clean_article_eligible") is not True or summ.get("is_open_access") is not True:
                raise CleanRoomPolicyError(
                    f"EuropePMC article '{summ.get('pmcid') or summ.get('query')}' "
                    "is not commercial_clean_article_eligible"
                )
            pmcid = str(summ.get("pmcid") or "").strip()
            if pmcid and f"PMCID:{pmcid}" not in urllib.parse.unquote(url.query):
                raise CleanRoomPolicyError(
                    f"EuropePMC request_url query does not match summary PMCID '{pmcid}'"
                )

    def filter_and_export_commercial_records(
        self, records: list[dict[str, Any]], *, purpose: str = "commercial_redistribution_export"
    ) -> list[dict[str, Any]]:
        """Enforce `guard_export_to_commercial_bundle` across every record before emitting a commercial bundle."""
        verified: list[dict[str, Any]] = []
        for rec in records:
            self.guard_export_to_commercial_bundle(rec, purpose=purpose)
            verified.append(rec)
        return verified

    def _freeze(
        self,
        source_name: str,
        license_spdx: str,
        license_tier: str,
        request_url: str,
        summary: dict[str, Any],
        raw_excerpt: Any,
    ) -> dict[str, Any]:
        return {
            "source_name": source_name,
            "license_spdx": license_spdx,
            "license_tier": license_tier,
            "request_url": request_url,
            "retrieved_at_utc": utc_now_iso(),
            "payload_sha256": sha256_canonical(summary),
            "summary": summary,
            "raw_excerpt": raw_excerpt,
        }

    # 1. UniProt REST (CC-BY-4.0)
    def fetch_uniprot(self, accession_or_gene: str, organism_id: int = 9606) -> dict[str, Any]:
        self._guard_source("UniProt", accession_or_gene)
        if re.fullmatch(r"[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}", accession_or_gene):
            url = f"https://rest.uniprot.org/uniprotkb/{accession_or_gene}.json"
            entry = _http_json(url)
        else:
            q = urllib.parse.quote(f"gene_exact:{accession_or_gene} AND organism_id:{organism_id} AND reviewed:true")
            url = f"https://rest.uniprot.org/uniprotkb/search?query={q}&format=json&size=1"
            entry = _http_json(url)["results"][0]
        seq = entry.get("sequence", {}).get("value", "")
        features = entry.get("features", [])
        sig_pep = [{"start": f["location"]["start"]["value"], "end": f["location"]["end"]["value"]} for f in features if f.get("type") == "Signal"]
        tm_doms = [{"start": f["location"]["start"]["value"], "end": f["location"]["end"]["value"], "desc": f.get("description", "")} for f in features if f.get("type") == "Transmembrane"]
        ens_ids = sorted({p["value"] for x in entry.get("uniProtKBCrossReferences", []) if x.get("database") == "Ensembl" for p in x.get("properties", []) if p.get("key") == "GeneId"})
        summary = {
            "primary_accession": entry.get("primaryAccession"),
            "uniProtkbId": entry.get("uniProtkbId"),
            "protein_name": entry.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", ""),
            "aa_length": len(seq),
            "starts_with_met": seq.startswith("M"),
            "has_signal_peptide": bool(sig_pep),
            "signal_peptide_coords_1based": sig_pep,
            "transmembrane_coords_1based": tm_doms,
            "ensembl_gene_ids": ens_ids[:5],
            "sequence_sha256": hashlib.sha256(seq.encode("utf-8")).hexdigest(),
        }
        return self._freeze("UniProt", "CC-BY-4.0", "commercial_clean", url, summary, {"sequence_prefix_30aa": seq[:30]})

    # 2. Ensembl REST (Apache-2.0)
    def fetch_ensembl_gene(self, gene_symbol: str, species: str = "homo_sapiens") -> dict[str, Any]:
        self._guard_source("Ensembl", gene_symbol)
        url = f"https://rest.ensembl.org/lookup/symbol/{species}/{urllib.parse.quote(gene_symbol)}?expand=1"
        res = _http_json(url)
        summary = {
            "ensembl_gene_id": res.get("id"),
            "display_name": res.get("display_name"),
            "canonical_transcript": res.get("canonical_transcript"),
            "biotype": res.get("biotype"),
            "seq_region_name": res.get("seq_region_name"),
            "strand": res.get("strand"),
        }
        return self._freeze("Ensembl", "Apache-2.0", "commercial_clean", url, summary, summary)

    # 3. RCSB PDB (CC0-1.0)
    def fetch_rcsb_pdb_cc0(self, query_text: str, limit: int = 4) -> dict[str, Any]:
        self._guard_source("RCSB_PDB", query_text)
        search_url = "https://search.rcsb.org/rcsbsearch/v2/query"
        q_payload = {
            "query": {"type": "terminal", "service": "full_text", "parameters": {"value": query_text}},
            "return_type": "entry",
            "request_options": {"paginate": {"start": 0, "rows": limit}},
        }
        res = _http_json(search_url, method="POST", payload=q_payload) or {}
        hits = []
        for item in res.get("result_set", [])[:limit]:
            pid = item["identifier"]
            meta = _http_json(f"https://data.rcsb.org/rest/v1/core/entry/{pid}")
            hits.append({
                "pdb_id": pid,
                "title": (meta.get("struct") or {}).get("title"),
                "resolution_angstrom": ((meta.get("rcsb_entry_info") or {}).get("resolution_combined") or [None])[0],
            })
        summary = {"query": query_text, "total_hits": res.get("total_count", 0), "top_entries": hits}
        return self._freeze("RCSB_PDB", "CC0-1.0", "commercial_clean", search_url, summary, hits)

    # 4. ChEMBL REST v3 (CC-BY-SA-3.0 — STRICT lookup_only_sharealike_boundary!)
    def fetch_chembl_biologic(self, query_text: str, limit: int = 3) -> dict[str, Any]:
        self._guard_source("ChEMBL", query_text)
        chembl_ver, chembl_rel_date = "unknown", "unknown"
        try:
            st = _http_json("https://www.ebi.ac.uk/chembl/api/data/status.json")
            chembl_ver = str(st.get("chembl_db_version", "unknown"))
            chembl_rel_date = str(st.get("chembl_release_date", "unknown"))
        except Exception:
            pass
        url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/search.json?q={urllib.parse.quote(query_text)}&limit={limit}"
        res = _http_json(url)
        mols = [
            {
                "chembl_id": m.get("molecule_chembl_id"),
                "pref_name": m.get("pref_name"),
                "molecule_type": m.get("molecule_type"),
                "max_phase": m.get("max_phase"),
            }
            for m in res.get("molecules", [])[:limit]
        ]
        summary = {
            "query": query_text,
            "chembl_db_version": chembl_ver,
            "chembl_release_date": chembl_rel_date,
            "attribution_url": "https://chembl.github.io/chembl-licensing/",
            "n_returned": len(mols),
            "molecules": mols,
        }
        return self._freeze("ChEMBL", "CC-BY-SA-3.0", "lookup_only_sharealike_boundary", url, summary, {"pointer_only_no_raw_redistribution": True})

    # 5. OpenTargets GraphQL v4 (CC0 / Public)
    def fetch_opentargets(self, ensembl_gene_id: str) -> dict[str, Any]:
        self._guard_source("OpenTargets", ensembl_gene_id)
        url = "https://api.platform.opentargets.org/api/v4/graphql"
        gql = """query T($ensg: String!) { target(ensemblId: $ensg) { id approvedSymbol tractability { label modality value } associatedDiseases(page:{index:0,size:5}) { count rows { score disease { id name } } } } }"""
        res = _http_json(url, method="POST", payload={"query": gql, "variables": {"ensg": ensembl_gene_id.split(".")[0]}})
        if res.get("errors"):
            raise ValueError(f"OpenTargets GraphQL error: {res['errors']}")
        tgt = (res.get("data") or {}).get("target")
        if not tgt or not tgt.get("id"):
            raise ValueError(f"OpenTargets found no target for '{ensembl_gene_id}'")
        summary = {
            "ensembl_id": tgt.get("id"),
            "approved_symbol": tgt.get("approvedSymbol"),
            "total_associated_diseases": (tgt.get("associatedDiseases") or {}).get("count", 0),
            "antibody_tractability": [t["label"] for t in (tgt.get("tractability") or []) if t.get("modality") == "AB" and t.get("value")],
        }
        return self._freeze("OpenTargets", "CC0-1.0_Public_Platform", "commercial_clean", url, summary, summary)

    # 6. ClinicalTrials.gov REST v2 (US Public Domain)
    def fetch_clinicaltrials(self, query_term: str, limit: int = 3) -> dict[str, Any]:
        self._guard_source("ClinicalTrials.gov", query_term)
        url = f"https://clinicaltrials.gov/api/v2/studies?query.term={urllib.parse.quote(query_term)}&pageSize={limit}"
        res = _http_json(url)
        studies = []
        for st in res.get("studies", [])[:limit]:
            ps = st.get("protocolSection", {})
            idm = ps.get("identificationModule", {})
            studies.append({
                "nct_id": idm.get("nctId"),
                "brief_title_sandboxed": sandbox_untrusted_text(idm.get("briefTitle", ""), f"CTgov:{idm.get('nctId')}"),
                "overall_status": (ps.get("statusModule") or {}).get("overallStatus"),
            })
        summary = {"query": query_term, "n_returned": len(studies), "studies": studies}
        return self._freeze("ClinicalTrials.gov", "US-Public-Domain", "commercial_clean", url, summary, studies)

    # 7. GTEx Portal API v2 (Public aggregate TPM — NEVER mature miRNAs!)
    def fetch_gtex_tissue_tpm(self, gene_symbol: str) -> dict[str, Any]:
        self._guard_source("GTEx", gene_symbol)
        ref = _http_json(f"https://gtexportal.org/api/v2/reference/gene?geneId={urllib.parse.quote(gene_symbol)}")
        data_list = ref.get("data") or []
        exact = next((g for g in data_list if str(g.get("geneSymbol", "")).upper() == gene_symbol.upper()), data_list[0])
        gencode_id = exact["gencodeId"]
        exp_url = f"https://gtexportal.org/api/v2/expression/medianGeneExpression?gencodeId={urllib.parse.quote(gencode_id)}&datasetId=gtex_v8"
        rows = (_http_json(exp_url).get("data") or [])
        ranked = sorted([{"tissue": r["tissueSiteDetailId"], "median_tpm": round(float(r["median"]), 3)} for r in rows], key=lambda x: x["median_tpm"], reverse=True)
        summary = {"gene_symbol": exact.get("geneSymbol"), "gencode_id": gencode_id, "n_tissues": len(ranked), "top_5_tissues": ranked[:5], "liver_tpm": next((r["median_tpm"] for r in ranked if r["tissue"] == "Liver"), None)}
        return self._freeze("GTEx", "Public_Aggregate_TPM", "commercial_clean", exp_url, summary, ranked[:10])

    # 8. AlphaFold DB API (CC-BY-4.0 — predicted 3D domain confidence & pLDDT for co_translational_slowdown_codons!)
    def fetch_alphafold_db(self, uniprot_accession: str) -> dict[str, Any]:
        self._guard_source("AlphaFoldDB", uniprot_accession)
        url = f"https://alphafold.ebi.ac.uk/api/prediction/{urllib.parse.quote(uniprot_accession)}"
        arr = _http_json(url)
        rec = arr[0] if isinstance(arr, list) and arr else arr
        summary = {
            "uniprot_accession": rec.get("uniprotAccession"),
            "entry_id": rec.get("entryId"),
            "latest_version": rec.get("latestVersion"),
            "global_metric_plddt": rec.get("globalMetricValue"),
            "fraction_plddt_very_high": rec.get("fractionPlddtVeryHigh"),
            "fraction_plddt_very_low_disordered": rec.get("fractionPlddtVeryLow"),
            "seq_length": len(rec.get("uniprotSequence") or ""),
            "cif_url": rec.get("cifUrl"),
            "pae_doc_url": rec.get("paeDocUrl"),
        }
        return self._freeze("AlphaFoldDB", "CC-BY-4.0", "commercial_clean", url, summary, summary)

    # 9. Human Protein Atlas (HPA) REST API (CC-BY-SA-3.0 — STRICT lookup_only_sharealike_boundary + NEVER mature miRNAs!)
    def fetch_human_protein_atlas(self, ensembl_gene_id: str) -> dict[str, Any]:
        self._guard_source("HumanProteinAtlas", ensembl_gene_id)
        ensg = ensembl_gene_id.split(".")[0]
        url = f"https://www.proteinatlas.org/{urllib.parse.quote(ensg)}.json"
        rec = _http_json(url)
        summary = {
            "ensembl_id": ensg,
            "gene": rec.get("Gene"),
            "protein_class": (rec.get("Protein class") or [])[:8],
            "subcellular_main_location": rec.get("Subcellular main location") or [],
            "secretome_location": rec.get("Secretome location"),
            "rna_tissue_specificity": rec.get("RNA tissue specificity"),
            "attribution_url": "https://www.proteinatlas.org/about/licence",
            "license_note": "Human Protein Atlas is CC-BY-SA-3.0; lookup-only localization pointer, blocked from commercial training/redistribution exports.",
        }
        return self._freeze("HumanProteinAtlas", "CC-BY-SA-3.0", "lookup_only_sharealike_boundary", url, summary, {"pointer_only_no_raw_redistribution": True})

    # 10. Europe PMC REST API (CC0 / Public Metadata — Automated Clean-Room OA & License Firewall!)
    def check_paper_license_europepmc(self, query_id: str) -> dict[str, Any]:
        self._guard_source("EuropePMC", query_id)
        url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query={urllib.parse.quote(query_id)}&format=json&resultType=core&pageSize=1"
        res = _http_json(url)
        items = (res.get("resultList") or {}).get("result") or []
        if not items:
            raise ValueError(f"Europe PMC found no match for '{query_id}'")
        r = items[0]
        lic = str(r.get("license") or "").lower().strip()
        is_oa = str(r.get("isOpenAccess") or "N").upper() == "Y"
        commercial_clean_eligible = is_oa and lic in ("cc by", "cc0", "cc by 4.0", "mit", "apache-2.0")
        summary = {
            "query": query_id,
            "pmid": r.get("pmid"),
            "pmcid": r.get("pmcid"),
            "doi": r.get("doi"),
            "pub_year": r.get("pubYear"),
            "journal": ((r.get("journalInfo") or {}).get("journal") or {}).get("title"),
            "is_open_access": is_oa,
            "license_raw": lic or None,
            "commercial_clean_article_eligible": commercial_clean_eligible,
            "title_sandboxed": sandbox_untrusted_text(r.get("title", ""), f"EuropePMC:{r.get('pmid')}"),
        }
        return self._freeze("EuropePMC", "CC0-1.0_Metadata", "commercial_clean", url, summary, summary)

    # 11. NCBI E-utilities (PubMed + GenBank/RefSeq Nucleotide — US Public Domain)
    def fetch_ncbi_nucleotide_fasta(self, accession: str) -> dict[str, Any]:
        self._guard_source("NCBI_Eutils", accession)
        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id={urllib.parse.quote(accession)}&rettype=fasta&retmode=text"
        txt = _http_text(url).strip()
        lines = txt.splitlines()
        header = lines[0] if lines else ""
        seq = "".join(ln.strip() for ln in lines[1:] if not ln.startswith(">")).upper()
        summary = {
            "accession": accession,
            "header_sandboxed": sandbox_untrusted_text(header, f"NCBI:{accession}"),
            "length_nt": len(seq),
            "gc_fraction": round((seq.count("G") + seq.count("C")) / max(len(seq), 1), 4),
            "sequence_sha256": hashlib.sha256(seq.encode("utf-8")).hexdigest(),
        }
        return self._freeze("NCBI_Eutils", "US-Public-Domain", "commercial_clean", url, summary, {"prefix_60nt": seq[:60]})

    # 12. RNAcentral REST API (CC0 / RNAcentral terms — mature miRNA / ncRNA verification)
    def fetch_rnacentral_by_external_id(self, mirbase_or_urs_id: str) -> dict[str, Any]:
        self._guard_source("RNAcentral", mirbase_or_urs_id)
        if mirbase_or_urs_id.upper().startswith("URS"):
            url = f"https://rnacentral.org/api/v1/rna/{urllib.parse.quote(mirbase_or_urs_id)}/?format=json"
            rec = _http_json(url)
        else:
            url = f"https://rnacentral.org/api/v1/rna/?external_id={urllib.parse.quote(mirbase_or_urs_id)}&format=json"
            res = _http_json(url)
            rec = (res.get("results") or [{}])[0]
        summary = {
            "query_id": mirbase_or_urs_id,
            "rnacentral_id": rec.get("rnacentral_id"),
            "rna_type": rec.get("rna_type"),
            "length": rec.get("length"),
            "sequence": rec.get("sequence"),
            "md5": rec.get("md5"),
            "license_terms_url": "https://docs.rnacentral.org/license.html",
        }
        return self._freeze("RNAcentral", "CC0-1.0_with_upstream_terms", "commercial_clean", url, summary, summary)


def run_12api_smoke(out_json_path: str) -> dict[str, Any]:
    adapter = ExternalEvidenceAdapterV2(license_track="commercial_clean")
    report: dict[str, Any] = {
        "suite_id": "M2_M4_FEDERATED_12API_EVIDENCE_SNAPSHOT_V2",
        "generated_at_utc": utc_now_iso(),
        "license_track": "commercial_clean",
        "guardrail_negative_tests": {},
        "live_api_snapshots": {},
    }

    # Negative guardrails 1-3 (query-level)
    for name, fn in [
        ("block_gtex_mature_mirna", lambda: adapter.fetch_gtex_tissue_tpm("hsa-miR-122-5p")),
        ("block_hpa_mature_mirna", lambda: adapter.fetch_human_protein_atlas("hsa-miR-122-5p")),
        ("block_thera_sabdab_commercial", lambda: adapter._guard_source("Thera-SAbDab", "tarlatamab")),
    ]:
        try:
            fn()
            report["guardrail_negative_tests"][name] = {"passed": False}
        except CleanRoomPolicyError as exc:
            report["guardrail_negative_tests"][name] = {"passed": True, "blocked_reason": str(exc)}

    calls = [
        ("01_uniprot_DLL3", lambda: adapter.fetch_uniprot("Q9NYJ7")),
        ("02_ensembl_DLL3", lambda: adapter.fetch_ensembl_gene("DLL3")),
        ("03_rcsb_pdb_CD3E_Fab", lambda: adapter.fetch_rcsb_pdb_cc0("CD3 epsilon antibody", limit=3)),
        ("04_chembl_tarlatamab", lambda: adapter.fetch_chembl_biologic("tarlatamab", limit=2)),
        ("05_opentargets_DLL3", lambda: adapter.fetch_opentargets("ENSG00000090932")),
        ("06_clinicaltrials_DLL3_CD3", lambda: adapter.fetch_clinicaltrials("DLL3 CD3 T-cell engager", limit=3)),
        ("07_gtex_DLL3", lambda: adapter.fetch_gtex_tissue_tpm("DLL3")),
        ("08_alphafold_DLL3", lambda: adapter.fetch_alphafold_db("Q9NYJ7")),
        ("09_hpa_DLL3_localization", lambda: adapter.fetch_human_protein_atlas("ENSG00000090932")),
        ("10_europepmc_license_gate_siegel2022", lambda: adapter.check_paper_license_europepmc("PMCID:PMC8728028")),
        ("11_ncbi_genbank_PP784251_m1psi_ref", lambda: adapter.fetch_ncbi_nucleotide_fasta("PP784251")),
        ("12_rnacentral_miR122_5p", lambda: adapter.fetch_rnacentral_by_external_id("MIMAT0000421")),
    ]
    for key, fn in calls:
        t0 = time.time()
        try:
            snap = fn()
            snap["latency_ms"] = round((time.time() - t0) * 1000.0, 1)
            snap["status"] = "ok"
            report["live_api_snapshots"][key] = snap
            print(f"[12-API OK] {key:36s}: {snap['latency_ms']:6.1f} ms | {snap['license_spdx']}", flush=True)
        except Exception as exc:
            report["live_api_snapshots"][key] = {"status": "error", "error": str(exc), "latency_ms": round((time.time() - t0) * 1000.0, 1)}
            print(f"[12-API ERR] {key:36s}: {exc}", flush=True)

    # Negative guardrails 4 & 5 (export-level ShareAlike firewall for BOTH ChEMBL & HPA!)
    for rec_key, test_name in [
        ("04_chembl_tarlatamab", "block_chembl_ccbysa_commercial_export"),
        ("09_hpa_DLL3_localization", "block_hpa_ccbysa_commercial_export"),
    ]:
        rec = report["live_api_snapshots"].get(rec_key)
        if rec and rec.get("status") == "ok":
            try:
                adapter.guard_export_to_commercial_bundle(rec, purpose="commercial_training")
                report["guardrail_negative_tests"][test_name] = {"passed": False}
            except CleanRoomPolicyError as exc:
                report["guardrail_negative_tests"][test_name] = {"passed": True, "blocked_reason": str(exc)}

    with open(out_json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    ok_n = sum(1 for v in report["live_api_snapshots"].values() if v.get("status") == "ok")
    g_n = sum(1 for v in report["guardrail_negative_tests"].values() if v.get("passed"))
    print(f"SUMMARY: {ok_n}/{len(calls)} live APIs OK | {g_n}/{len(report['guardrail_negative_tests'])} negative guardrails PASS", flush=True)
    return report


ExternalEvidenceAdapter = ExternalEvidenceAdapterV2


if __name__ == "__main__":
    import sys
    out_p = sys.argv[1] if len(sys.argv) > 1 else "/Users/alex/.gemini/antigravity/brain/35858d54-b7a4-4a1b-9f76-1d8070d0dc6a/scratch/wave7_assets/m2_m4_federated_12api_snapshot_v2.json"
    run_12api_smoke(out_p)

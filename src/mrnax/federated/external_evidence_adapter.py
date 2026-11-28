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
from collections.abc import Iterable
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


MAX_UPSTREAM_RESPONSE_BYTES = 2 * 1024 * 1024


def sandbox_untrusted_text(text: str, source_id: str) -> dict[str, Any]:
    """Wrap external text in a strict read-only untrusted envelope per mRNAx v2 §5.1."""
    from mrnax.harness.sandbox import wrap_untrusted_external_text

    quarantined = wrap_untrusted_external_text(str(text)[:2000], source_id=str(source_id), max_chars=2000)
    return {
        "untrusted_external_data": True,
        "must_not_modify_locked_constraints": True,
        "source_id": quarantined.source_id,
        "sha256": quarantined.sha256,
        "text_excerpt": quarantined.wrapped_payload,
    }


def _ssl_context() -> ssl.SSLContext | None:
    try:
        import certifi  # type: ignore[import-not-found]

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return None


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> Any:
        raise urllib.error.HTTPError(
            req.full_url, code, f"Outbound redirect blocked by SSRF policy ({code})", headers, fp
        )


class _PinnedHTTPSConnection(urllib.request.http.client.HTTPSConnection):  # type: ignore[attr-defined]
    def __init__(self, host: str, *, pinned_ip: str, port: int = 443, timeout: int = 20, context: ssl.SSLContext | None = None) -> None:
        super().__init__(host, port=port, timeout=timeout, context=context)
        self._pinned_ip = pinned_ip

    def connect(self) -> None:
        import socket

        raw_sock = socket.create_connection((self._pinned_ip, self.port), self.timeout)
        ctx = self._context if hasattr(self, "_context") and self._context is not None else ssl.create_default_context()
        self.sock = ctx.wrap_socket(raw_sock, server_hostname=self.host)


class _PinnedHTTPSHandler(urllib.request.HTTPSHandler):
    def __init__(self, *, pinned_ip: str, port: int, context: ssl.SSLContext | None) -> None:
        super().__init__(context=context)
        self._pinned_ip = pinned_ip
        self._pinned_port = port
        self._tls_ctx = context

    def https_open(self, req: urllib.request.Request) -> Any:
        def _factory(host: str, **kwargs: Any) -> _PinnedHTTPSConnection:
            return _PinnedHTTPSConnection(
                host.split(":", 1)[0],
                pinned_ip=self._pinned_ip,
                port=self._pinned_port,
                timeout=kwargs.get("timeout", 20),
                context=self._tls_ctx,
            )

        return self.do_open(_factory, req)


def _open_verified_public_request(req: urllib.request.Request, *, timeout: int) -> Any:
    from mrnax.harness.sandbox import resolve_verified_public_peer

    _, _, port, pinned_ip = resolve_verified_public_peer(req.full_url, allow_localhost=False)
    ctx = _ssl_context()
    https_handler = _PinnedHTTPSHandler(pinned_ip=pinned_ip, port=port, context=ctx)
    opener = urllib.request.build_opener(https_handler, _NoRedirectHandler())
    return opener.open(req, timeout=timeout)


def _read_bounded_bytes(resp: Any, *, max_bytes: int = MAX_UPSTREAM_RESPONSE_BYTES) -> bytes:
    raw = resp.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise ValueError(f"Upstream response exceeded {max_bytes}-byte safety cap")
    return raw


def _http_json(url: str, method: str = "GET", payload: dict | None = None, timeout: int = 20) -> Any:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with _open_verified_public_request(req, timeout=timeout) as resp:
        if resp.status == 204:
            return {}
        raw = _read_bounded_bytes(resp).decode("utf-8").strip()
        return json.loads(raw) if raw else {}


def _http_text(url: str, timeout: int = 20) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with _open_verified_public_request(req, timeout=timeout) as resp:
        return _read_bounded_bytes(resp).decode("utf-8", errors="replace")


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
            "external_redistribution_allowed",
            "redistribution_allowed",
            "variant_dataset_redistribution_allowed",
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
            "internal_evaluation_only",
            "internal_only",
        }
    )
    _RESTRICTED_GOVERNANCE_SCOPES: frozenset[str] = frozenset(
        {
            "internal_evaluation_only",
            "internal_only",
            "restricted",
            "quarantine",
            "research_only",
            "commercial_clearance_pending",
        }
    )
    _GOVERNED_DATA_KEY = re.compile(
        r"(?:^|[^a-z0-9])(?:variants?|coords?|coordinates?|hg38|grch38|hg19|parquet|datasets?|"
        r"benchmarks?|supp(?:lement(?:ary)?)?\d*|mpra|half_life_pairs?|utr_pairs?|row_count|"
        r"drive_id|upstream_artifact)(?:$|[^a-z0-9])"
    )
    _HGVS_POSITION = r"(?:[-+*]?(?:\d+|\?)(?:[+-](?:\d+|\?))?|pter|qter|cen)"
    _HGVS_BREAKPOINT = rf"(?:{_HGVS_POSITION}|\({_HGVS_POSITION}(?:_{_HGVS_POSITION})?\))"
    _HGVS_LOCUS = rf"{_HGVS_BREAKPOINT}(?:_{_HGVS_BREAKPOINT})?"
    _HGVS_BASE = r"[acgtunrywsmkbdhv]"
    # The operation identifies governed data even when the allele is a length or an accession.
    _HGVS_EDIT = rf"""(?x:
        {_HGVS_BASE}+ > {_HGVS_BASE}+
        | (?:delins|del|ins|dup|inv|con) (?:(?:{_HGVS_BASE}|[\d_])*\b | (?=[a-z][\w.]*:[cgmnro]\.))
        | {_HGVS_BASE}* (?:= | \[(?=[\d?(]))
        | :: | \|(?:gom|lom|met=)
    )"""
    _DATA_VALUE = re.compile(
        r"(?<![a-z0-9])(?:chr(?:omosome)?(?:[0-9]+|[xym]|mt):[0-9]+|utr_variant|"
        r"variant_pair|supp\d+\b|rs\d+\b|"
        rf"[cgmnro]\.(?:\[[=?]\];\[|\[?\(?{_HGVS_LOCUS}{_HGVS_EDIT})|"
        r"r\.(?:\(?(?:spl|0)\b|\([=?]\)))|\.parquet\b", re.I
    )
    _HGMD = re.compile(r"(?<![a-z0-9])hgmd\b", re.I)
    _BIBLIOGRAPHIC_TEXT = {"title", "article_title", "abstract", "author", "author_name", "author_string", "journal"}

    @staticmethod
    def _load_restricted_inventory_index() -> tuple[frozenset[str], frozenset[str]]:
        """Return `(hard_data_tokens, restricted_publication_aliases)` from `v9_wetlab_benchmarks_inventory.json`."""
        from mrnax.engine.profiles import resolve_default_seed_dir

        inv_path = resolve_default_seed_dir() / "v9_wetlab_benchmarks_inventory.json"
        hard_tokens: set[str] = {
            "su2025",
            "su2025_elife97682",
            "su2025_elife97682_utr_variants_v1.parquet",
            "4c956ac99399c86e554e4f6e351d9f313b16b787397de7d00a1a04cd018dea95",
            "1vopzcmc6itfy6wdczecmzlxmtxte3wye",
            "elife-97682-supp1",
            "pmc11835390_su2025_elife97682_utr_mpra",
        }
        pub_aliases: set[str] = {
            "39964837",
            "pmid:39964837",
            "pmc11835390",
            "pmcid:pmc11835390",
            "10.7554/elife.97682",
            "elife.97682",
            "elife-97682",
            "elife97682",
            "articles/97682",
        }
        if inv_path.is_file():
            doc = json.loads(inv_path.read_text(encoding="utf-8"))
            for row in doc.get("benchmarks", []):
                if row.get("external_redistribution_allowed") is False or (
                    str(row.get("governance_scope") or "") in ExternalEvidenceAdapterV2._RESTRICTED_GOVERNANCE_SCOPES
                ):
                    for k in ("filename", "dataset_id", "sha256", "drive_id"):
                        val = str(row.get(k) or "").strip().lower()
                        if val:
                            hard_tokens.add(val)
                    for u in row.get("restricted_data_urls", []):
                        if u:
                            hard_tokens.add(str(u).strip().lower())
                    for alias in row.get("publication_aliases", []):
                        al = str(alias).strip().lower()
                        if al:
                            if "mpra" in al or "su2025" in al:
                                hard_tokens.add(al)
                            else:
                                pub_aliases.add(al)
        return frozenset(hard_tokens), frozenset(pub_aliases)

    @classmethod
    def _collect_leaf_strings(cls, val: Any) -> list[str]:
        if isinstance(val, str) or type(val) is int:
            return [str(val)]
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

    @staticmethod
    def _coordinate_columns(names: Iterable[Any]) -> bool:
        columns = {name.strip().casefold().lstrip("#") for name in names if isinstance(name, str)}
        locus = columns & {"chr", "chrom", "chromosome"} and columns & {"pos", "position"}
        ref = columns & {"ref", "reference", "reference_allele"}
        alt = columns & {"alt", "alternate", "alternative", "alternate_allele", "alternative_allele"}
        return bool(locus or (ref and alt))

    @classmethod
    def _has_governed_payload(cls, obj: Any, *, citation_envelope: bool = False) -> bool:
        if isinstance(obj, dict):
            fields = [(re.sub(r"([a-z])([A-Z])", r"\1_\2", str(k)).lower(), v) for k, v in obj.items()]
            if cls._coordinate_columns(key for key, _ in fields):
                return True
            for key, value in fields:
                if cls._GOVERNED_DATA_KEY.search(key):
                    return True
                if key in {"columns", "fields", "headers"} and isinstance(value, list):
                    if cls._coordinate_columns(v.get("name") for v in value if isinstance(v, dict)):
                        return True
                if isinstance(value, str) and (key in cls._BIBLIOGRAPHIC_TEXT or
                                               (citation_envelope and key == "text_excerpt")):
                    if cls._DATA_VALUE.search(value):
                        return True
                    continue
                sandboxed_citation = (
                    key.removesuffix("_sandboxed") in cls._BIBLIOGRAPHIC_TEXT
                    and isinstance(value, dict)
                    and value.get("untrusted_external_data") is True
                    and value.get("must_not_modify_locked_constraints") is True
                )
                if cls._has_governed_payload(value, citation_envelope=sandboxed_citation):
                    return True
            return False
        if isinstance(obj, (list, tuple, set, frozenset)):
            return cls._coordinate_columns(obj) or any(cls._has_governed_payload(v) for v in obj)
        return isinstance(obj, str) and bool(cls._DATA_VALUE.search(obj) or cls._HGMD.search(obj))

    _ARTICLE_ID = re.compile(
        r"(?:(?:pmid:\s*)?(?P<pmid>[0-9]+)|(?:pmcid:\s*)?(?P<pmcid>PMC[0-9]+)|"
        r"(?:doi:\s*)?(?P<doi>10\.[0-9]{4,9}/[^\s]+))", re.I | re.ASCII
    )
    _CITED_ID = re.compile(
        r"\b(?:pmid(?:\s*:\s*|\s+)([0-9]+)\b|(PMC[0-9]+)\b|(10\.[0-9]{4,9}/[^\s,;]+))", re.I | re.ASCII
    )
    _ARTICLE_LINK = re.compile(
        r"(?:(?:pubmed\.ncbi\.nlm\.nih\.gov|(?:www\.)?ncbi\.nlm\.nih\.gov/pubmed|"
        r"(?:www\.)?europepmc\.org/article/MED)/(?P<pmid>[0-9]+)/?|"
        r"(?:pmc\.ncbi\.nlm\.nih\.gov/articles|(?:www\.)?ncbi\.nlm\.nih\.gov/pmc/articles|"
        r"(?:www\.)?europepmc\.org/article/PMC)/(?P<pmcid>PMC[0-9]+)/?|"
        r"(?:dx\.)?doi\.org/(?P<doi>10\.[0-9]{4,9}/[^\s]+))", re.I | re.ASCII
    )
    _CITATION_URL = re.compile(r"""(https?://[^\s<>"']+?(?=$|[\s<>"']|[,;]https?://))""", re.I)

    @staticmethod
    def _without_citation_punctuation(token: str) -> str:
        stem = token.rstrip(""".,;)]}>"'""")
        end = len(stem)
        # A DOI can end in a balanced delimiter, even inside a parenthesized citation.
        suffix = token[end:]
        if suffix.startswith((")", "]", "}")):
            unmatched = {b: stem.count(a) - stem.count(b) for a, b in ("()", "[]", "{}")}
            for char in suffix:
                if unmatched.get(char, 0) <= 0:
                    break
                unmatched[char] -= 1
                end += 1
        return token[:end]

    @classmethod
    def _citation_ids(cls, text: str) -> set[str]:
        ids: set[str] = set()
        for part in cls._CITATION_URL.split(text):
            if not part.casefold().startswith(("http://", "https://")):
                if part == text:
                    match = cls._ARTICLE_ID.fullmatch(cls._without_citation_punctuation(text))
                    if match:
                        return {match[match.lastgroup].casefold()}
                for match in cls._CITED_ID.finditer(part):
                    ids.add(cls._without_citation_punctuation(match[match.lastindex]).casefold())
                continue
            try:
                url = urllib.parse.urlparse(cls._without_citation_punctuation(part))
                host = url.hostname or ""
                if url.username is not None or url.port == 0:
                    continue
            except ValueError:
                continue
            path = url.path + (";" + url.params if url.params else "")
            article = cls._ARTICLE_LINK.fullmatch(host + urllib.parse.unquote(path))
            if article:
                ids.add(article[article.lastgroup].casefold())
            elif part == text and path.rstrip("/").endswith("/search"):
                # Historical citations also used a search URL with the identifier in its query.
                queries = urllib.parse.parse_qs(url.query).get("query", [])
                query_id = cls._ARTICLE_ID.fullmatch(queries[0].strip()) if len(queries) == 1 else None
                if query_id:
                    ids.add(query_id[query_id.lastgroup].casefold())
        return ids

    @classmethod
    def _article_ids(cls, node: Any, *, citation: bool = False) -> set[str]:
        """Read identifiers from article metadata, excluding the request's own query."""
        ids: set[str] = set()
        if isinstance(node, dict):
            for key, value in node.items():
                key = str(key).casefold()
                if key in {"doi", "pmid", "pmcid"}:
                    match = cls._ARTICLE_ID.fullmatch(str(value).strip())
                    if match and match.lastgroup == key:
                        ids.add(match[key].casefold())
                elif key in {"citation", "article"} or (citation and key in {"id", "identifier", "identifiers"}):
                    ids.update(cls._article_ids(value, citation=True))
                elif key == "summary":
                    ids.update(cls._article_ids(value))
        elif isinstance(node, list):
            for item in node:
                ids.update(cls._article_ids(item, citation=citation))
        elif citation and isinstance(node, (str, int)):
            ids.update(cls._citation_ids(str(node).strip()))
        return ids

    @classmethod
    def _verify_europepmc_query(cls, query_string: str, record: dict[str, Any]) -> None:
        queries = urllib.parse.parse_qs(query_string, keep_blank_values=True).get("query", [])
        if len(queries) != 1 or not queries[0].strip():
            raise CleanRoomPolicyError("EuropePMC attribution requires one search query")
        query = queries[0].strip().casefold()
        ids = cls._article_ids(record)
        match = cls._ARTICLE_ID.fullmatch(query)
        if match:
            if match[match.lastgroup] in ids:
                return
            raise CleanRoomPolicyError("EuropePMC request_url query does not match the returned article")
        if ids or ":" in query or any(k in record["summary"] for k in ("doi", "pmid", "pmcid", "query")):
            raise CleanRoomPolicyError("EuropePMC request_url query has no verifiable article identifier")

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
                elif key_low in {"governance_scope", "redistribution_scope", "usage_scope"}:
                    if not isinstance(v, str) or v.strip().lower() in cls._RESTRICTED_GOVERNANCE_SCOPES:
                        raise CleanRoomPolicyError(
                            f"Nested restricted governance scope {k}={v!r} inside '{src}' evidence is BLOCKED in commercial_clean"
                        )
                elif any(p in key_low for p in ("license", "spdx", "rights", "access_tier")):
                    leaves = cls._collect_leaf_strings(v)
                    if type(v) is int or (not leaves and v is not None and not isinstance(v, (bool, dict, list, tuple))):
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

        hard_tokens, pub_aliases = self._load_restricted_inventory_index()
        all_leaves_low = [urllib.parse.unquote(s).strip().lower() for s in self._collect_leaf_strings(frozen_record) if s]
        for leaf in all_leaves_low:
            if any(ht in leaf for ht in hard_tokens):
                raise CleanRoomPolicyError(
                    f"Source '{src}' references restricted internal-evaluation dataset/artifact ({leaf!r}) "
                    "governed under A-W11 C3 (Su et al. 2025 eLife 97682); external commercial redistribution is BLOCKED."
                )

        cites_restricted_pub = any(
            re.search(r"(?<![a-z0-9])" + re.escape(pa) + r"(?![a-z0-9])", leaf)
            for pa in pub_aliases
            for leaf in all_leaves_low
        )
        if cites_restricted_pub:
            is_bibliographic_source = src in {"EuropePMC", "NCBI_Eutils"}
            has_governed_data_payload = self._has_governed_payload(
                {
                    k: v
                    for k, v in frozen_record.items()
                    if k
                    not in {
                        "source_name",
                        "license_spdx",
                        "license_tier",
                        "request_url",
                        "payload_sha256",
                        "fetched_at_utc",
                        "latency_ms",
                        "status",
                    }
                }
            )
            if not is_bibliographic_source or has_governed_data_payload:
                raise CleanRoomPolicyError(
                    f"Source '{src}' carries governed variant/dataset provenance tied to Su et al. 2025 eLife 97682 "
                    "(PMID:39964837 / DOI:10.7554/eLife.97682 / PMCID:PMC11835390); restricted to internal evaluation only (A-W11 C3)."
                )
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
            self._verify_europepmc_query(url.query, frozen_record)

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

    # ------------------------------------------------------------------
    # Wave 2B: Universal Target Protein & Literature Resolver (v2.5)
    # ------------------------------------------------------------------
    _OFFLINE_BENCHMARK_ALIASES: dict[str, str] = {
        "egfp": "EGFP",
        "gfp": "EGFP",
        "p42212": "P42212",
        "u55762": "EGFP",
        "u55762.1": "EGFP",
        "aab02572.1": "EGFP",
        "epo": "EPO",
        "hepo": "EPO",
        "p01588": "EPO",
        "nm_000799": "EPO",
        "nm_000799.4": "EPO",
        "np_000790": "EPO",
        "np_000790.2": "EPO",
        "fgf21": "FGF21",
        "hfgf21": "FGF21",
        "q9nsa1": "FGF21",
        "nm_019113": "FGF21",
        "nm_019113.4": "FGF21",
        "vegfa": "VEGFA",
        "vegf": "VEGFA",
        "p15692": "VEGFA",
        "erbb2": "ERBB2",
        "her2": "ERBB2",
        "p04626": "ERBB2",
        "fluc": "FLUC",
        "luciferase": "FLUC",
        "firefly_luciferase": "FLUC",
        "p08659": "FLUC",
        "dll3": "DLL3",
        "q9nyj7": "DLL3",
        "cd3e": "CD3E",
        "p07766": "CD3E",
    }

    # Exact UniProt / GenBank verified amino-acid sequences (zero guessed residues).
    _OFFLINE_VERIFIED_BENCHMARKS: dict[str, dict[str, Any]] = {
        "EGFP": {
            "gene_symbol": "EGFP",
            "uniprot_accession": "P42212",
            "accession_or_source": "GenBank:U55762.1",
            "entry_version": "AAB02572.1",
            "protein_name": "Enhanced Green Fluorescent Protein (mammalian reporter)",
            "organism": "Aequorea victoria (engineered mammalian reporter)",
            "protein_sequence": (
                "MVSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFK"
                "DDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLST"
                "QSALSKDPNEKRDHMVLLEFVTAAGITLGMDELYK"
            ),
            "has_signal_peptide": False,
            "signal_peptide_coords_1based": [],
            "transmembrane_coords_1based": [],
            "ensembl_gene_ids": [],
            "subcellular_locations": ["Cytosol", "Nucleoplasm"],
            "is_secreted_or_membrane": False,
            "license_spdx": "CC-BY-4.0",
        },
        "P42212": {
            "gene_symbol": "GFP",
            "uniprot_accession": "P42212",
            "accession_or_source": "UniProt:P42212",
            "entry_version": "196",
            "protein_name": "Green fluorescent protein (Aequorea victoria wild-type P42212)",
            "organism": "Aequorea victoria",
            "protein_sequence": (
                "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFK"
                "DDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLST"
                "QSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
            ),
            "has_signal_peptide": False,
            "signal_peptide_coords_1based": [],
            "transmembrane_coords_1based": [],
            "ensembl_gene_ids": [],
            "subcellular_locations": ["Cytoplasm"],
            "is_secreted_or_membrane": False,
            "license_spdx": "CC-BY-4.0",
        },
        "EPO": {
            "gene_symbol": "EPO",
            "uniprot_accession": "P01588",
            "accession_or_source": "UniProt:P01588",
            "entry_version": "1",
            "protein_name": "Erythropoietin",
            "organism": "Homo sapiens",
            "protein_sequence": (
                "MGVHECPAWLWLLLSLLSLPLGLPVLGAPPRLICDSRVLERYLLEAKEAENITTGCAEHCSLNENITVPDTKVNFYAWKRMEVGQQAVEVWQGLALLSEAVL"
                "RGQALLVNSSQPWEPLQLHVDKAVSGLRSLTTLLRALGAQKEAISPPDAASAAPLRTITADTFRKLFRVYSNFLRGKLKLYTGEACRTGDR"
            ),
            "has_signal_peptide": True,
            "signal_peptide_coords_1based": [{"start": 1, "end": 27}],
            "transmembrane_coords_1based": [],
            "ensembl_gene_ids": ["ENSG00000130427.3"],
            "subcellular_locations": ["Secreted"],
            "is_secreted_or_membrane": True,
            "license_spdx": "CC-BY-4.0",
        },
        "FGF21": {
            "gene_symbol": "FGF21",
            "uniprot_accession": "Q9NSA1",
            "accession_or_source": "UniProt:Q9NSA1",
            "entry_version": "1",
            "protein_name": "Fibroblast growth factor 21",
            "organism": "Homo sapiens",
            "protein_sequence": (
                "MDSDETGFEHSGLWVSVLAGLLLGACQAHPIPDSSPLLQFGGQVRQRYLYTDDAQQTEAHLEIREDGTVGGAADQSPESLLQLKALKPGVIQILGVKTSRFL"
                "CQRPDGALYGSLHFDPEACSFRELLLEDGYNVYQSEAHGLPLHLPGNKSPHRDPAPRGPARFLPLPGLPPALPEPPGILAPQPPDVGSSDPLSMVGPSQGRSPSYAS"
            ),
            "has_signal_peptide": True,
            "signal_peptide_coords_1based": [{"start": 1, "end": 28}],
            "transmembrane_coords_1based": [],
            "ensembl_gene_ids": ["ENSG00000105550.10"],
            "subcellular_locations": ["Secreted"],
            "is_secreted_or_membrane": True,
            "license_spdx": "CC-BY-4.0",
        },
        "VEGFA": {
            "gene_symbol": "VEGFA",
            "uniprot_accession": "P15692",
            "accession_or_source": "UniProt:P15692",
            "entry_version": "2",
            "protein_name": "Vascular endothelial growth factor A, long form",
            "organism": "Homo sapiens",
            "protein_sequence": (
                "MTDRQTDTAPSPSYHLLPGRRRTVDAAASRGQGPEPAPGGGVEGVGARGVALKLFVQLLGCSRFGGAVVRAGEAEPSGAARSASSGREEPQPEEGEEEEEKE"
                "EERGPQWRLGARKPGSWTGEAAVCADSAPAARAPQALARASGRGGRVARRGAEESGPPHSPSRRGSASRAGPGRASETMNFLLSWVHWSLALLLYLHHAKWS"
                "QAAPMAEGGGQNHHEVVKFMDVYQRSYCHPIETLVDIFQEYPDEIEYIFKPSCVPLMRCGGCCNDEGLECVPTEESNITMQIMRIKPHQGQHIGEMSFLQHN"
                "KCECRPKKDRARQEKKSVRGKGKGQKRKRKKSRYKSWSVPCGPCSERRKHLFVQDPQTCKCSCKNTDSRCKARQLELNERTCRCDKPRR"
            ),
            "has_signal_peptide": False,
            "signal_peptide_coords_1based": [],
            "transmembrane_coords_1based": [],
            "ensembl_gene_ids": ["ENSG00000112715.26"],
            "subcellular_locations": ["Secreted"],
            "is_secreted_or_membrane": True,
            "license_spdx": "CC-BY-4.0",
        },
        "ERBB2": {
            "gene_symbol": "ERBB2",
            "uniprot_accession": "P04626",
            "accession_or_source": "UniProt:P04626",
            "entry_version": "1",
            "protein_name": "Receptor tyrosine-protein kinase erbB-2",
            "organism": "Homo sapiens",
            "protein_sequence": (
                "MELAALCRWGLLLALLPPGAASTQVCTGTDMKLRLPASPETHLDMLRHLYQGCQVVQGNLELTYLPTNASLSFLQDIQEVQGYVLIAHNQVRQVPLQRLRIV"
                "RGTQLFEDNYALAVLDNGDPLNNTTPVTGASPGGLRELQLRSLTEILKGGVLIQRNPQLCYQDTILWKDIFHKNNQLALTLIDTNRSRACHPCSPMCKGSRC"
                "WGESSEDCQSLTRTVCAGGCARCKGPLPTDCCHEQCAAGCTGPKHSDCLACLHFNHSGICELHCPALVTYNTDTFESMPNPEGRYTFGASCVTACPYNYLST"
                "DVGSCTLVCPLHNQEVTAEDGTQRCEKCSKPCARVCYGLGMEHLREVRAVTSANIQEFAGCKKIFGSLAFLPESFDGDPASNTAPLQPEQLQVFETLEEITG"
                "YLYISAWPDSLPDLSVFQNLQVIRGRILHNGAYSLTLQGLGISWLGLRSLRELGSGLALIHHNTHLCFVHTVPWDQLFRNPHQALLHTANRPEDECVGEGLA"
                "CHQLCARGHCWGPGPTQCVNCSQFLRGQECVEECRVLQGLPREYVNARHCLPCHPECQPQNGSVTCFGPEADQCVACAHYKDPPFCVARCPSGVKPDLSYMP"
                "IWKFPDEEGACQPCPINCTHSCVDLDDKGCPAEQRASPLTSIISAVVGILLVVVLGVVFGILIKRRQQKIRKYTMRRLLQETELVEPLTPSGAMPNQAQMRI"
                "LKETELRKVKVLGSGAFGTVYKGIWIPDGENVKIPVAIKVLRENTSPKANKEILDEAYVMAGVGSPYVSRLLGICLTSTVQLVTQLMPYGCLLDHVRENRGR"
                "LGSQDLLNWCMQIAKGMSYLEDVRLVHRDLAARNVLVKSPNHVKITDFGLARLLDIDETEYHADGGKVPIKWMALESILRRRFTHQSDVWSYGVTVWELMTF"
                "GAKPYDGIPAREIPDLLEKGERLPQPPICTIDVYMIMVKCWMIDSECRPRFRELVSEFSRMARDPQRFVVIQNEDLGPASPLDSTFYRSLLEDDDMGDLVDA"
                "EEYLVPQQGFFCPDPAPGAGGMVHHRHRSSSTRSGGGDLTLGLEPSEEEAPRSPLAPSEGAGSDVFDGDLGMGAAKGLQSLPTHDPSPLQRYSEDPTVPLPS"
                "ETDGYVAPLTCSPQPEYVNQPDVRPQPPSPREGPLPAARPAGATLERPKTLSPGKNGVVKDVFAFGGAVENPEYLTPQGGAAPQPHPPPAFSPAFDNLYYWD"
                "QDPPERGAPPSTFKGTPTAENPEYLGLDVPV"
            ),
            "has_signal_peptide": True,
            "signal_peptide_coords_1based": [{"start": 1, "end": 22}],
            "transmembrane_coords_1based": [{"start": 653, "end": 675, "desc": "Helical"}],
            "ensembl_gene_ids": ["ENSG00000141736.15"],
            "subcellular_locations": ["Cell membrane"],
            "is_secreted_or_membrane": True,
            "license_spdx": "CC-BY-4.0",
        },
        "FLUC": {
            "gene_symbol": "FLuc",
            "uniprot_accession": "P08659",
            "accession_or_source": "UniProt:P08659",
            "entry_version": "1",
            "protein_name": "Luciferin 4-monooxygenase",
            "organism": "Photinus pyralis",
            "protein_sequence": (
                "MEDAKNIKKGPAPFYPLEDGTAGEQLHKAMKRYALVPGTIAFTDAHIEVNITYAEYFEMSVRLAEAMKRYGLNTNHRIVVCSENSLQFFMPVLGALFIGVAV"
                "APANDIYNERELLNSMNISQPTVVFVSKKGLQKILNVQKKLPIIQKIIIMDSKTDYQGFQSMYTFVTSHLPPGFNEYDFVPESFDRDKTIALIMNSSGSTGL"
                "PKGVALPHRTACVRFSHARDPIFGNQIIPDTAILSVVPFHHGFGMFTTLGYLICGFRVVLMYRFEEELFLRSLQDYKIQSALLVPTLFSFFAKSTLIDKYDL"
                "SNLHEIASGGAPLSKEVGEAVAKRFHLPGIRQGYGLTETTSAILITPEGDDKPGAVGKVVPFFEAKVVDLDTGKTLGVNQRGELCVRGPMIMSGYVNNPEAT"
                "NALIDKDGWLHSGDIAYWDEDEHFFIVDRLKSLIKYKGYQVAPAELESILLQHPNIFDAGVAGLPDDDAGELPAAVVVLEHGKTMTEKEIVDYVASQVTTAK"
                "KLRGGVVFVDEVPKGLTGKLDARKIREILIKAKKGGKSKL"
            ),
            "has_signal_peptide": False,
            "signal_peptide_coords_1based": [],
            "transmembrane_coords_1based": [],
            "ensembl_gene_ids": [],
            "subcellular_locations": ["Peroxisome", "Cytosol"],
            "is_secreted_or_membrane": False,
            "license_spdx": "CC-BY-4.0",
        },
        "DLL3": {
            "gene_symbol": "DLL3",
            "uniprot_accession": "Q9NYJ7",
            "accession_or_source": "UniProt:Q9NYJ7",
            "entry_version": "1",
            "protein_name": "Delta-like protein 3",
            "organism": "Homo sapiens",
            "protein_sequence": (
                "MVSPRMSGLLSQTVILALIFLPQTRPAGVFELQIHSFGPGPGPGAPRSPCSARLPCRLFFRVCLKPGLSEEAAESPCALGAALSARGPVYTEQPGAPAPDLP"
                "LPDGLLQVPFRDAWPGTFSFIIETWREELGDQIGGPAWSLLARVAGRRRLAAGGPWARDIQRAGAWELRFSYRARCEPPAVGTACTRLCRPRSAPSRCGPGL"
                "RPCAPLEDECEAPLVCRAGCSPEHGFCEQPGECRCLEGWTGPLCTVPVSTSSCLSPRGPSSATTGCLVPGPGPCDGNPCANGGSCSETPRSFECTCPRGFYG"
                "LRCEVSGVTCADGPCFNGGLCVGGADPDSAYICHCPPGFQGSNCEKRVDRCSLQPCRNGGLCLDLGHALRCRCRAGFAGPRCEHDLDDCAGRACANGGTCVE"
                "GGGAHRCSCALGFGGRDCRERADPCAARPCAHGGRCYAHFSGLVCACAPGYMGARCEFPVHPDGASALPAAPPGLRPGDPQRYLLPPALGLLVAAGVAGAAL"
                "LLVHVRRRGHSQDAGSRLLAGTPEPSVHALPDALNNLRTQEGSGDGPSSSVDWNRPEDVDPQGIYVISAPSIYAREVATPLFPPLHTGRAGQRQHLLFPYPS"
                "SILSVK"
            ),
            "has_signal_peptide": True,
            "signal_peptide_coords_1based": [{"start": 1, "end": 26}],
            "transmembrane_coords_1based": [{"start": 493, "end": 513, "desc": "Helical"}],
            "ensembl_gene_ids": ["ENSG00000090932.12"],
            "subcellular_locations": ["Membrane"],
            "is_secreted_or_membrane": True,
            "license_spdx": "CC-BY-4.0",
        },
        "CD3E": {
            "gene_symbol": "CD3E",
            "uniprot_accession": "P07766",
            "accession_or_source": "UniProt:P07766",
            "entry_version": "1",
            "protein_name": "T-cell surface glycoprotein CD3 epsilon chain",
            "organism": "Homo sapiens",
            "protein_sequence": (
                "MQSGTHWRVLGLCLLSVGVWGQDGNEEMGGITQTPYKVSISGTTVILTCPQYPGSEILWQHNDKNIGGDEDDKNIGSDEDHLSLKEFSELEQSGYYVCYPRG"
                "SKPEDANFYLYLRARVCENCMEMDVMSVATIVIVDICITGGLLLLVYYWSKNRKAKAKPVTRGAGAGGRQRGQNKERPPPVPNPDYEPIRKGQRDLYSGLNQRRI"
            ),
            "has_signal_peptide": True,
            "signal_peptide_coords_1based": [{"start": 1, "end": 22}],
            "transmembrane_coords_1based": [{"start": 127, "end": 152, "desc": "Helical"}],
            "ensembl_gene_ids": ["ENSG00000198851.11"],
            "subcellular_locations": ["Cell membrane"],
            "is_secreted_or_membrane": True,
            "license_spdx": "CC-BY-4.0",
        },
    }

    @staticmethod
    def _build_signal_peptide_hint(
        *,
        has_signal_peptide: bool,
        sig_coords: list[dict[str, Any]],
        tm_coords: list[dict[str, Any]],
        subcellular_locations: list[str],
        secretome_location: str | None,
    ) -> tuple[bool, str]:
        loc_blob = " ".join([*subcellular_locations, str(secretome_location or "")]).lower()
        is_sec_or_mem = bool(
            has_signal_peptide
            or tm_coords
            or any(k in loc_blob for k in ("secreted", "membrane", "extracellular", "cell surface", "vesicle"))
        )
        if has_signal_peptide and sig_coords:
            sp = sig_coords[0]
            return (
                True,
                f"Native N-terminal signal peptide detected ({sp.get('start')}..{sp.get('end')} aa). "
                "Preserve native leader or swap intentionally with a validated heterologous signal peptide; "
                "maintain moderate local codon speed around codons 30-45 for SRP co-translational ER targeting.",
            )
        if tm_coords:
            return (
                True,
                f"Transmembrane topology detected ({len(tm_coords)} helix/helices) without cleavable N-terminal signal peptide; "
                "preserve signal-anchor residues and avoid rare-codon clusters inside core structured domains.",
            )
        if is_sec_or_mem:
            return (
                True,
                "Secreted/membrane localization annotated without canonical cleavable N-terminal signal peptide "
                "(check isoform choice or non-classical secretion before locking CDS).",
            )
        return (
            False,
            "Intracellular/cytosolic target: optimize 5-prime initiation openness [-15, +30 nt] without ER leader peptide.",
        )

    def resolve_target_protein(
        self,
        query: str,
        *,
        allow_network: bool = True,
        timeout: int = 8,
    ) -> dict[str, Any]:
        """Resolve human/reporter gene symbols, UniProt IDs, or RefSeq IDs across UniProt, AlphaFold, HPA, GTEx & ClinicalTrials."""
        cleaned = query.strip()
        if not cleaned:
            raise ValueError("Protein query must be a non-empty string")
        self._guard_source("UniProt", cleaned)
        if MATURE_MIRNA_RE.search(cleaned):
            raise CleanRoomPolicyError(
                f"Mature miRNA identifier '{cleaned}' is not a protein-coding target; query tissue_mirna_atlas_v1.json instead."
            )

        from mrnax.harness.sandbox import wrap_untrusted_external_text

        norm_low = cleaned.lower()
        alias_key = self._OFFLINE_BENCHMARK_ALIASES.get(norm_low)
        # Prefer canonical engineered 239-aa mammalian eGFP reporter when user explicitly asks for eGFP / U55762
        if norm_low in {"egfp", "u55762", "u55762.1", "aab02572.1"}:
            bench = dict(self._OFFLINE_VERIFIED_BENCHMARKS["EGFP"])
            seq = str(bench["protein_sequence"])
            wrapped_name = wrap_untrusted_external_text(bench["protein_name"], source_id="GenBank:U55762.1")
            return {
                "matched": True,
                "query": cleaned,
                "resolution_mode": "verified_reporter_snapshot",
                "gene_symbol": "EGFP",
                "uniprot_accession": "P42212",
                "accession_or_source": "GenBank:U55762.1",
                "entry_version": "AAB02572.1",
                "organism": bench["organism"],
                "protein_name_sandboxed": sandbox_untrusted_text(bench["protein_name"], "GenBank:U55762.1"),
                "protein_name_quarantined": wrapped_name.model_dump(mode="json"),
                "protein_sequence": seq,
                "aa_length": len(seq),
                "starts_with_met": seq.startswith("M"),
                "sequence_sha256": hashlib.sha256(seq.encode("utf-8")).hexdigest(),
                "has_signal_peptide": False,
                "signal_peptide_coords_1based": [],
                "transmembrane_coords_1based": [],
                "disordered_regions_1based": [],
                "is_secreted_or_membrane": False,
                "signal_peptide_hint": (
                    "Cytosolic mammalian eGFP reporter (239 aa with V2 Kozak valine); no ER signal peptide required."
                ),
                "alphafold_summary": None,
                "hpa_summary": None,
                "gtex_summary": None,
                "clinical_trials": [],
                "commercial_clean_evidence": [],
                "lookup_only_evidence": [],
            }

        live_entry: dict[str, Any] | None = None
        uniprot_url = ""
        if allow_network:
            try:
                target_tok = cleaned
                if alias_key and alias_key in self._OFFLINE_VERIFIED_BENCHMARKS and not re.match(r"^(NM_|NP_|XP_)", cleaned, re.I):
                    target_tok = self._OFFLINE_VERIFIED_BENCHMARKS[alias_key]["uniprot_accession"]
                if re.fullmatch(r"[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}", target_tok):
                    uniprot_url = f"https://rest.uniprot.org/uniprotkb/{urllib.parse.quote(target_tok)}.json"
                    live_entry = _http_json(uniprot_url, timeout=timeout)
                elif re.match(r"^(NM_|NP_|XP_|XM_)[0-9]+", target_tok, re.I):
                    stem = target_tok.split(".")[0].upper()
                    q_ref = urllib.parse.quote(f"(xref:RefSeq-{stem} OR {stem}) AND reviewed:true")
                    uniprot_url = f"https://rest.uniprot.org/uniprotkb/search?query={q_ref}&format=json&size=1"
                    res_u = _http_json(uniprot_url, timeout=timeout)
                    results_u = res_u.get("results") or []
                    if not results_u:
                        q_any = urllib.parse.quote(f"{stem}")
                        uniprot_url = f"https://rest.uniprot.org/uniprotkb/search?query={q_any}&format=json&size=1"
                        results_u = (_http_json(uniprot_url, timeout=timeout) or {}).get("results") or []
                    live_entry = results_u[0] if results_u else None
                else:
                    sym_clean = re.sub(r"^h(?=[A-Z]{2,})", "", target_tok)
                    q_human = urllib.parse.quote(f"gene_exact:{sym_clean} AND organism_id:9606 AND reviewed:true")
                    uniprot_url = f"https://rest.uniprot.org/uniprotkb/search?query={q_human}&format=json&size=1"
                    results_u = (_http_json(uniprot_url, timeout=timeout) or {}).get("results") or []
                    if not results_u:
                        q_broad = urllib.parse.quote(f"(gene:{target_tok} OR {target_tok}) AND reviewed:true")
                        uniprot_url = f"https://rest.uniprot.org/uniprotkb/search?query={q_broad}&format=json&size=1"
                        results_u = (_http_json(uniprot_url, timeout=timeout) or {}).get("results") or []
                    live_entry = results_u[0] if results_u else None
            except Exception:
                live_entry = None

        if live_entry and live_entry.get("sequence", {}).get("value"):
            seq = str(live_entry["sequence"]["value"]).strip().upper()
            acc = str(live_entry.get("primaryAccession") or "")
            genes = live_entry.get("genes") or []
            gene_sym = (
                ((genes[0].get("geneName") or {}).get("value"))
                if genes and isinstance(genes[0], dict)
                else None
            ) or (alias_key or cleaned)
            prot_name = (
                (((live_entry.get("proteinDescription") or {}).get("recommendedName") or {}).get("fullName") or {}).get("value")
                or str(live_entry.get("uniProtkbId") or acc)
            )
            organism_name = str(((live_entry.get("organism") or {}).get("scientificName")) or "Unknown")
            organism_quarantined = wrap_untrusted_external_text(
                organism_name, source_id=f"UniProt:{acc}", max_chars=160
            ).model_dump(mode="json")
            seq_ver = str((live_entry.get("sequence") or {}).get("version") or "1")
            features = live_entry.get("features") or []
            sig_pep = [
                {"start": int(f["location"]["start"]["value"]), "end": int(f["location"]["end"]["value"])}
                for f in features
                if f.get("type") == "Signal" and isinstance(f.get("location"), dict)
            ]
            tm_doms = [
                {
                    "start": int(f["location"]["start"]["value"]),
                    "end": int(f["location"]["end"]["value"]),
                    "desc_quarantined": (
                        wrap_untrusted_external_text(
                            str(f.get("description") or ""), source_id=f"UniProt:{acc}", max_chars=240
                        ).model_dump(mode="json")
                        if f.get("description")
                        else None
                    ),
                }
                for f in features
                if f.get("type") == "Transmembrane" and isinstance(f.get("location"), dict)
            ]
            disordered_doms = [
                {
                    "start": int(f["location"]["start"]["value"]),
                    "end": int(f["location"]["end"]["value"]),
                    "desc_quarantined": wrap_untrusted_external_text(
                        str(f.get("description") or "Disordered"), source_id=f"UniProt:{acc}", max_chars=240
                    ).model_dump(mode="json"),
                }
                for f in features
                if f.get("type") == "Region"
                and "disordered" in str(f.get("description") or "").lower()
                and isinstance(f.get("location"), dict)
            ]
            ens_ids = sorted(
                {
                    p["value"]
                    for x in (live_entry.get("uniProtKBCrossReferences") or [])
                    if x.get("database") == "Ensembl"
                    for p in (x.get("properties") or [])
                    if p.get("key") == "GeneId" and isinstance(p.get("value"), str)
                }
            )
            refseq_ids = sorted(
                {
                    str(x.get("id"))
                    for x in (live_entry.get("uniProtKBCrossReferences") or [])
                    if x.get("database") == "RefSeq" and x.get("id")
                }
            )[:6]

            cc_evidence: list[dict[str, Any]] = []
            lookup_evidence: list[dict[str, Any]] = []
            direct_uniprot_url = f"https://rest.uniprot.org/uniprotkb/{urllib.parse.quote(acc)}.json"
            u_summary = {
                "primary_accession": acc,
                "gene_symbol": gene_sym,
                "aa_length": len(seq),
                "starts_with_met": seq.startswith("M"),
                "has_signal_peptide": bool(sig_pep),
                "signal_peptide_coords_1based": sig_pep,
                "transmembrane_coords_1based": [
                    {"start": d["start"], "end": d["end"]} for d in tm_doms
                ],
                "sequence_sha256": hashlib.sha256(seq.encode("utf-8")).hexdigest(),
            }
            cc_evidence.append(
                self._freeze("UniProt", "CC-BY-4.0", "commercial_clean", direct_uniprot_url, u_summary, {"sequence_prefix_30aa": seq[:30]})
            )

            af_summary: dict[str, Any] | None = None
            try:
                af_url = f"https://alphafold.ebi.ac.uk/api/prediction/{urllib.parse.quote(acc)}"
                af_arr = _http_json(af_url, timeout=timeout)
                af_rec = af_arr[0] if isinstance(af_arr, list) and af_arr else af_arr
                if isinstance(af_rec, dict) and af_rec.get("entryId"):
                    af_summary = {
                        "uniprot_accession": acc,
                        "entry_id": af_rec.get("entryId"),
                        "global_metric_plddt": af_rec.get("globalMetricValue"),
                        "fraction_plddt_very_high": af_rec.get("fractionPlddtVeryHigh"),
                        "fraction_plddt_very_low_disordered": af_rec.get("fractionPlddtVeryLow"),
                    }
                    cc_evidence.append(
                        self._freeze("AlphaFoldDB", "CC-BY-4.0", "commercial_clean", af_url, af_summary, af_summary)
                    )
            except Exception:
                af_summary = None

            hpa_summary: dict[str, Any] | None = None
            subcell_locs: list[str] = []
            secretome_loc: str | None = None
            if ens_ids:
                try:
                    ensg_clean = ens_ids[0].split(".")[0]
                    hpa_url = f"https://www.proteinatlas.org/{urllib.parse.quote(ensg_clean)}.json"
                    hpa_rec = _http_json(hpa_url, timeout=timeout)
                    if isinstance(hpa_rec, dict):
                        subcell_locs = [str(x) for x in (hpa_rec.get("Subcellular main location") or [])]
                        secretome_loc = str(hpa_rec.get("Secretome location")) if hpa_rec.get("Secretome location") else None
                        raw_tissue_spec = str(hpa_rec.get("RNA tissue specificity")) if hpa_rec.get("RNA tissue specificity") else None
                        subcell_joined = "; ".join(subcell_locs)
                        hpa_src = f"HPA:{ensg_clean}"
                        hpa_summary = {
                            "ensembl_id": ensg_clean,
                            "subcellular_main_location_sandboxed": (
                                sandbox_untrusted_text(subcell_joined, hpa_src) if subcell_joined else None
                            ),
                            "subcellular_main_location_quarantined": (
                                wrap_untrusted_external_text(subcell_joined, source_id=hpa_src, max_chars=240).model_dump(mode="json")
                                if subcell_joined
                                else None
                            ),
                            "secretome_location_sandboxed": (
                                sandbox_untrusted_text(secretome_loc, hpa_src) if secretome_loc else None
                            ),
                            "secretome_location_quarantined": (
                                wrap_untrusted_external_text(secretome_loc, source_id=hpa_src, max_chars=240).model_dump(mode="json")
                                if secretome_loc
                                else None
                            ),
                            "rna_tissue_specificity_sandboxed": (
                                sandbox_untrusted_text(raw_tissue_spec, hpa_src) if raw_tissue_spec else None
                            ),
                            "rna_tissue_specificity_quarantined": (
                                wrap_untrusted_external_text(raw_tissue_spec, source_id=hpa_src, max_chars=240).model_dump(mode="json")
                                if raw_tissue_spec
                                else None
                            ),
                            "license_tier": "lookup_only_sharealike_boundary",
                        }
                        lookup_evidence.append(
                            self._freeze(
                                "HumanProteinAtlas",
                                "CC-BY-SA-3.0",
                                "lookup_only_sharealike_boundary",
                                hpa_url,
                                {
                                    "ensembl_id": ensg_clean,
                                    "has_subcellular_annotation": bool(subcell_locs),
                                    "has_secretome_annotation": bool(secretome_loc),
                                    "license_tier": "lookup_only_sharealike_boundary",
                                },
                                {"pointer_only_no_raw_redistribution": True},
                            )
                        )
                except Exception:
                    hpa_summary = None

            gtex_summary: dict[str, Any] | None = None
            if organism_name == "Homo sapiens" and gene_sym:
                try:
                    ref_g = _http_json(
                        f"https://gtexportal.org/api/v2/reference/gene?geneId={urllib.parse.quote(str(gene_sym))}",
                        timeout=timeout,
                    )
                    g_list = (ref_g or {}).get("data") or []
                    if g_list:
                        exact_g = next(
                            (g for g in g_list if str(g.get("geneSymbol", "")).upper() == str(gene_sym).upper()),
                            g_list[0],
                        )
                        gencode_id = exact_g["gencodeId"]
                        exp_url = (
                            f"https://gtexportal.org/api/v2/expression/medianGeneExpression"
                            f"?gencodeId={urllib.parse.quote(gencode_id)}&datasetId=gtex_v8"
                        )
                        rows = ((_http_json(exp_url, timeout=timeout) or {}).get("data")) or []
                        ranked = sorted(
                            [{"tissue": r["tissueSiteDetailId"], "median_tpm": round(float(r["median"]), 3)} for r in rows],
                            key=lambda x: x["median_tpm"],
                            reverse=True,
                        )
                        gtex_summary = {
                            "gene_symbol": exact_g.get("geneSymbol"),
                            "gencode_id": gencode_id,
                            "top_5_tissues": ranked[:5],
                            "liver_tpm": next((r["median_tpm"] for r in ranked if r["tissue"] == "Liver"), None),
                        }
                        cc_evidence.append(
                            self._freeze("GTEx", "Public_Aggregate_TPM", "commercial_clean", exp_url, gtex_summary, ranked[:5])
                        )
                except Exception:
                    gtex_summary = None

            trials_list: list[dict[str, Any]] = []
            try:
                ct_term = f"{gene_sym} mRNA"
                ct_url = f"https://clinicaltrials.gov/api/v2/studies?query.term={urllib.parse.quote(ct_term)}&pageSize=3"
                ct_res = _http_json(ct_url, timeout=timeout) or {}
                for st in (ct_res.get("studies") or [])[:3]:
                    ps = st.get("protocolSection") or {}
                    idm = ps.get("identificationModule") or {}
                    nct_id = str(idm.get("nctId") or "")
                    btitle = str(idm.get("briefTitle") or "")
                    wrapped_bt = wrap_untrusted_external_text(btitle, source_id=f"CTgov:{nct_id}")
                    trials_list.append(
                        {
                            "nct_id": nct_id,
                            "overall_status": (ps.get("statusModule") or {}).get("overallStatus"),
                            "brief_title_sandboxed": sandbox_untrusted_text(btitle, f"CTgov:{nct_id}"),
                            "brief_title_quarantined": wrapped_bt.model_dump(mode="json"),
                        }
                    )
            except Exception:
                trials_list = []

            is_sec_mem, sp_hint = self._build_signal_peptide_hint(
                has_signal_peptide=bool(sig_pep),
                sig_coords=sig_pep,
                tm_coords=tm_doms,
                subcellular_locations=subcell_locs,
                secretome_location=secretome_loc,
            )
            wrapped_pname = wrap_untrusted_external_text(prot_name, source_id=f"UniProt:{acc}")
            return {
                "matched": True,
                "query": cleaned,
                "resolution_mode": "live_federated",
                "gene_symbol": gene_sym,
                "uniprot_accession": acc,
                "accession_or_source": f"UniProt:{acc}",
                "entry_version": seq_ver,
                "organism_quarantined": organism_quarantined,
                "protein_name_sandboxed": sandbox_untrusted_text(prot_name, f"UniProt:{acc}"),
                "protein_name_quarantined": wrapped_pname.model_dump(mode="json"),
                "protein_sequence": seq,
                "aa_length": len(seq),
                "starts_with_met": seq.startswith("M"),
                "sequence_sha256": hashlib.sha256(seq.encode("utf-8")).hexdigest(),
                "has_signal_peptide": bool(sig_pep),
                "signal_peptide_coords_1based": sig_pep,
                "transmembrane_coords_1based": tm_doms,
                "disordered_regions_1based": disordered_doms,
                "ensembl_gene_ids": ens_ids[:5],
                "refseq_ids": refseq_ids,
                "is_secreted_or_membrane": is_sec_mem,
                "signal_peptide_hint": sp_hint,
                "alphafold_summary": af_summary,
                "hpa_summary": hpa_summary,
                "gtex_summary": gtex_summary,
                "clinical_trials": trials_list,
                "commercial_clean_evidence": cc_evidence,
                "lookup_only_evidence": lookup_evidence,
            }

        # Deterministic offline fallback for verified common targets
        if alias_key and alias_key in self._OFFLINE_VERIFIED_BENCHMARKS:
            bench = dict(self._OFFLINE_VERIFIED_BENCHMARKS[alias_key])
            seq = str(bench["protein_sequence"])
            acc = str(bench["uniprot_accession"])
            is_sec_mem, sp_hint = self._build_signal_peptide_hint(
                has_signal_peptide=bool(bench["has_signal_peptide"]),
                sig_coords=list(bench["signal_peptide_coords_1based"]),
                tm_coords=list(bench["transmembrane_coords_1based"]),
                subcellular_locations=list(bench["subcellular_locations"]),
                secretome_location="Secreted" if bench["is_secreted_or_membrane"] and not bench["transmembrane_coords_1based"] else None,
            )
            wrapped_pname = wrap_untrusted_external_text(bench["protein_name"], source_id=bench["accession_or_source"])
            return {
                "matched": True,
                "query": cleaned,
                "resolution_mode": "offline_verified_snapshot",
                "gene_symbol": bench["gene_symbol"],
                "uniprot_accession": acc,
                "accession_or_source": bench["accession_or_source"],
                "entry_version": bench["entry_version"],
                "organism": bench["organism"],
                "protein_name_sandboxed": sandbox_untrusted_text(bench["protein_name"], bench["accession_or_source"]),
                "protein_name_quarantined": wrapped_pname.model_dump(mode="json"),
                "protein_sequence": seq,
                "aa_length": len(seq),
                "starts_with_met": seq.startswith("M"),
                "sequence_sha256": hashlib.sha256(seq.encode("utf-8")).hexdigest(),
                "has_signal_peptide": bool(bench["has_signal_peptide"]),
                "signal_peptide_coords_1based": list(bench["signal_peptide_coords_1based"]),
                "transmembrane_coords_1based": list(bench["transmembrane_coords_1based"]),
                "disordered_regions_1based": [],
                "ensembl_gene_ids": list(bench["ensembl_gene_ids"]),
                "refseq_ids": [],
                "is_secreted_or_membrane": is_sec_mem,
                "signal_peptide_hint": sp_hint,
                "alphafold_summary": None,
                "hpa_summary": {
                    "subcellular_main_location": list(bench["subcellular_locations"]),
                    "license_tier": "offline_curated_metadata",
                },
                "gtex_summary": None,
                "clinical_trials": [],
                "commercial_clean_evidence": [],
                "lookup_only_evidence": [],
            }

        return {
            "matched": False,
            "query": cleaned,
            "resolution_mode": "unresolved",
            "note": (
                f"Could not resolve verified protein sequence for '{cleaned}'. "
                "Please supply the exact amino-acid sequence or a reviewed UniProt / RefSeq accession."
            ),
        }

    _OFFLINE_VERIFIED_LITERATURE: tuple[dict[str, Any], ...] = (
        {
            "pmid": "31267113",
            "pmcid": "PMC7100133",
            "doi": "10.1038/s41587-019-0164-5",
            "pub_year": "2019",
            "journal": "Nature Biotechnology",
            "title": "Human 5' UTR design and variant effect prediction from a massively parallel translation assay",
            "is_open_access": True,
            "license_raw": "cc by 4.0",
            "commercial_clean_article_eligible": True,
            "keywords": ("5'utr", "5utr", "mrl", "sample", "optimus", "uaug", "m1psi", "translation"),
        },
        {
            "pmid": "38057662",
            "pmcid": "PMC10796200",
            "doi": "10.1038/s41586-023-06800-3",
            "pub_year": "2024",
            "journal": "Nature",
            "title": "N1-methylpseudouridylation of mRNA causes +1 ribosomal frameshifting",
            "is_open_access": True,
            "license_raw": "cc by 4.0",
            "commercial_clean_article_eligible": True,
            "keywords": ("m1psi", "frameshift", "mulroney", "slippery", "uuuu", "pseudouridine", "vaccine"),
        },
        {
            "pmid": "37130513",
            "pmcid": "PMC10202639",
            "doi": "10.1038/s41586-023-06127-z",
            "pub_year": "2023",
            "journal": "Nature",
            "title": "Algorithm forOptimized mRNA Design Improves Stability and Immunogenicity (LinearDesign)",
            "is_open_access": True,
            "license_raw": "cc by 4.0",
            "commercial_clean_article_eligible": True,
            "keywords": ("lineardesign", "mfe", "cai", "stability", "codon", "zhang", "cds"),
        },
        {
            "pmid": "35585242",
            "pmcid": "PMC9931579",
            "doi": "10.1038/s41467-022-30261-x",
            "pub_year": "2022",
            "journal": "Nature Communications",
            "title": "Combinatorial optimization of mRNA structure, stability, and translation for RNA-based therapeutics",
            "is_open_access": True,
            "license_raw": "cc by 4.0",
            "commercial_clean_article_eligible": True,
            "keywords": ("leppek", "hydrolysis", "in-line probing", "stability", "aup", "5'utr", "cds"),
        },
        {
            "pmid": "31767746",
            "pmcid": "PMC6900575",
            "doi": "10.1073/pnas.1909907116",
            "pub_year": "2019",
            "journal": "Proceedings of the National Academy of Sciences",
            "title": "mRNA structure regulates protein expression through changes in functional half-life",
            "is_open_access": True,
            "license_raw": "cc by 4.0",
            "commercial_clean_article_eligible": True,
            "keywords": ("mauger", "moderna", "shape-map", "m1psi", "half-life", "5'utr", "cds"),
        },
    )

    def search_literature_and_trials(
        self,
        query: str,
        max_results: int = 5,
        *,
        allow_network: bool = True,
        timeout: int = 8,
    ) -> dict[str, Any]:
        """Query EuropePMC (enforcing automated OA/license & restricted-dataset firewall) and ClinicalTrials.gov."""
        cleaned = query.strip()
        if not cleaned:
            raise ValueError("Literature/trials query must be a non-empty string")
        if isinstance(max_results, bool) or not isinstance(max_results, int) or max_results < 1 or max_results > 25:
            raise ValueError("max_results must be an integer between 1 and 25")
        self._guard_source("EuropePMC", cleaned)
        self._guard_source("ClinicalTrials.gov", cleaned)

        from mrnax.harness.sandbox import wrap_untrusted_external_text

        hard_tokens, pub_aliases = self._load_restricted_inventory_index()
        papers: list[dict[str, Any]] = []
        trials: list[dict[str, Any]] = []
        live_hit = False

        if allow_network:
            try:
                epmc_url = (
                    f"https://www.ebi.ac.uk/europepmc/webservices/rest/search"
                    f"?query={urllib.parse.quote(cleaned)}&format=json&resultType=core&pageSize={max_results}"
                )
                res_epmc = _http_json(epmc_url, timeout=timeout) or {}
                items = ((res_epmc.get("resultList") or {}).get("result")) or []
                live_hit = True
                for r in items[:max_results]:
                    lic = str(r.get("license") or "").lower().strip()
                    is_oa = str(r.get("isOpenAccess") or "N").upper() == "Y"
                    cc_ok = is_oa and lic in (
                        "cc by",
                        "cc0",
                        "cc by 4.0",
                        "cc-by",
                        "cc-by-4.0",
                        "cc0 1.0",
                        "mit",
                        "apache-2.0",
                    )
                    pmid_s = str(r.get("pmid") or "")
                    pmcid_s = str(r.get("pmcid") or "")
                    doi_s = str(r.get("doi") or "")
                    id_tokens = {pmid_s.lower(), pmcid_s.lower(), doi_s.lower()} - {""}
                    restricted_dataset_linked = bool(
                        any(tok in pub_aliases or tok in hard_tokens for tok in id_tokens)
                    )
                    title_raw = str(r.get("title") or "")
                    abstract_raw = str(r.get("abstractText") or "")[:600]
                    journal_raw = str(((r.get("journalInfo") or {}).get("journal") or {}).get("title") or "").strip()
                    src_tag = f"EuropePMC:{pmid_s or pmcid_s or doi_s or 'entry'}"
                    papers.append(
                        {
                            "pmid": pmid_s or None,
                            "pmcid": pmcid_s or None,
                            "doi": doi_s or None,
                            "pub_year": r.get("pubYear"),
                            "journal_quarantined": (
                                wrap_untrusted_external_text(journal_raw, source_id=src_tag, max_chars=240).model_dump(mode="json")
                                if journal_raw
                                else None
                            ),
                            "is_open_access": is_oa,
                            "license_raw": lic or None,
                            "commercial_clean_article_eligible": bool(cc_ok and not restricted_dataset_linked),
                            "restricted_dataset_governance_notice": (
                                "Bibliographic metadata only; linked supplementary dataset is restricted to internal evaluation."
                                if restricted_dataset_linked
                                else None
                            ),
                            "title_sandboxed": sandbox_untrusted_text(title_raw, src_tag),
                            "title_quarantined": wrap_untrusted_external_text(title_raw, source_id=src_tag).model_dump(mode="json"),
                            "abstract_quarantined": (
                                wrap_untrusted_external_text(abstract_raw, source_id=src_tag, max_chars=600).model_dump(mode="json")
                                if abstract_raw
                                else None
                            ),
                            "abstract_sandboxed": sandbox_untrusted_text(abstract_raw, src_tag) if abstract_raw else None,
                        }
                    )
            except Exception:
                pass

            try:
                ct_url = f"https://clinicaltrials.gov/api/v2/studies?query.term={urllib.parse.quote(cleaned)}&pageSize={max_results}"
                res_ct = _http_json(ct_url, timeout=timeout) or {}
                live_hit = True
                for st in (res_ct.get("studies") or [])[:max_results]:
                    ps = st.get("protocolSection") or {}
                    idm = ps.get("identificationModule") or {}
                    nct_id = str(idm.get("nctId") or "")
                    btitle = str(idm.get("briefTitle") or "")
                    conds = ", ".join((ps.get("conditionsModule") or {}).get("conditions") or [])
                    src_tag = f"CTgov:{nct_id or 'study'}"
                    trials.append(
                        {
                            "nct_id": nct_id or None,
                            "overall_status": (ps.get("statusModule") or {}).get("overallStatus"),
                            "phases": ((ps.get("designModule") or {}).get("phases")) or [],
                            "brief_title_sandboxed": sandbox_untrusted_text(btitle, src_tag),
                            "brief_title_quarantined": wrap_untrusted_external_text(btitle, source_id=src_tag).model_dump(mode="json"),
                            "conditions_quarantined": (
                                wrap_untrusted_external_text(conds, source_id=src_tag, max_chars=400).model_dump(mode="json")
                                if conds
                                else None
                            ),
                            "conditions_sandboxed": sandbox_untrusted_text(conds, src_tag) if conds else None,
                        }
                    )
            except Exception:
                pass

        if not papers and not live_hit:
            q_tokens = [t for t in re.split(r"[^a-z0-9']+", cleaned.lower()) if len(t) >= 2]
            scored = []
            for item in self._OFFLINE_VERIFIED_LITERATURE:
                hay = f"{item['title']} {' '.join(item['keywords'])} {item['pmid']} {item['pmcid']} {item['doi']}".lower()
                hits = sum(1 for t in q_tokens if t in hay)
                scored.append((hits, item))
            scored.sort(key=lambda pair: pair[0], reverse=True)
            for _, item in scored[:max_results]:
                src_tag = f"EuropePMC:{item['pmid']}"
                papers.append(
                    {
                        "pmid": item["pmid"],
                        "pmcid": item["pmcid"],
                        "doi": item["doi"],
                        "pub_year": item["pub_year"],
                        "journal_quarantined": wrap_untrusted_external_text(
                            str(item["journal"]), source_id=src_tag, max_chars=240
                        ).model_dump(mode="json"),
                        "is_open_access": item["is_open_access"],
                        "license_raw": item["license_raw"],
                        "commercial_clean_article_eligible": item["commercial_clean_article_eligible"],
                        "restricted_dataset_governance_notice": None,
                        "title_sandboxed": sandbox_untrusted_text(item["title"], src_tag),
                        "title_quarantined": wrap_untrusted_external_text(item["title"], source_id=src_tag).model_dump(mode="json"),
                        "abstract_quarantined": None,
                        "abstract_sandboxed": None,
                    }
                )

        return {
            "query": cleaned,
            "resolution_mode": "live_federated" if live_hit else "offline_verified_registry",
            "papers_count": len(papers),
            "trials_count": len(trials),
            "papers": papers,
            "clinical_trials": trials,
        }


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


def resolve_target_protein(
    query: str,
    *,
    license_track: str = "commercial_clean",
    allow_network: bool = True,
    timeout: int = 8,
) -> dict[str, Any]:
    """Module-level convenience wrapper for universal target protein resolution."""
    return ExternalEvidenceAdapterV2(license_track=license_track).resolve_target_protein(
        query, allow_network=allow_network, timeout=timeout
    )


def search_literature_and_trials(
    query: str,
    max_results: int = 5,
    *,
    license_track: str = "commercial_clean",
    allow_network: bool = True,
    timeout: int = 8,
) -> dict[str, Any]:
    """Module-level convenience wrapper for EuropePMC + ClinicalTrials.gov resolution."""
    return ExternalEvidenceAdapterV2(license_track=license_track).search_literature_and_trials(
        query, max_results=max_results, allow_network=allow_network, timeout=timeout
    )


if __name__ == "__main__":
    import sys
    out_p = sys.argv[1] if len(sys.argv) > 1 else "/Users/alex/.gemini/antigravity/brain/35858d54-b7a4-4a1b-9f76-1d8070d0dc6a/scratch/wave7_assets/m2_m4_federated_12api_snapshot_v2.json"
    run_12api_smoke(out_p)

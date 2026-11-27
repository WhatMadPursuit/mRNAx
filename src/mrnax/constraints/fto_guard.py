"""Dual-FTO Guard Suite:
1. `SequenceFTOGuard` (`patent_fto_guard_v2.json` v2.1.0): genus regexes, sequence collisions, AND
   Cartesian 5'UTR x 3'UTR pair screening (`WO2024231621A1` / `EP4704918A1` / `JP2026515056A`).
2. `MethodFTOGuard` (`method_fto_guard_v1.1.json` v1.1.0): runtime & CI algorithmic execution guardrails
   across boundaries F1 (`CN119265214B`/`JP7892088B2`), F2 (`CN118038986B`), F3 (`JP7367090B2`/`US20240394558A1`),
   F4A (`US10898584B2`/`US12533422B2`/`US20260284227A1` CureVac C-optimizable codon ceiling),
   F4B (`US12706176B1` random pool motif-count argmin & weighted-random repair), and
   F4C (`EP3551758B1`/`US11848074B2` boosted GAM + protein_per_time),
   with `certifies_100_percent_fto = False` enforced unconditionally.
"""

from __future__ import annotations

import ast
from collections import defaultdict, deque
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from functools import cache, lru_cache
import hashlib
import importlib.metadata
import json
import math
import re
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Any, Literal

from packaging.licenses import canonicalize_license_expression
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StringConstraints

from Bio.Align import PairwiseAligner
from Bio.Data import CodonTable

from mrnax.schemas.common import ComplianceProfile, SegmentScope
from mrnax.schemas.construct_bundle import ConstructBundle
from mrnax.schemas.design_intent import DesignIntent
from mrnax.schemas.evaluation_report import FTOAlert
from mrnax.schemas.run_manifest import ComponentInvocationRecord, MethodFTOAuditRecord

def normalize_sense_cds(sequence: str, translation_table: int = 1) -> str:
    """Validate a CDS and remove its optional terminal stop under the declared genetic code."""
    try:
        table = CodonTable.unambiguous_rna_by_id[translation_table]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Unknown NCBI translation table {translation_table!r}") from exc
    if not isinstance(sequence, str) or not sequence.strip():
        raise ValueError("CDS sequence must be non-empty")
    rna = sequence.strip().upper().replace("T", "U")
    if not re.fullmatch("[ACGU]+", rna):
        raise ValueError("CDS sequence contains invalid nucleotide characters")
    if len(rna) % 3:
        raise ValueError(f"Invalid CDS reading frame length: {len(rna)}")
    # As in Bio.Seq.translate(to_stop=False), dual-coding codons encode their amino acid.
    stops = set(table.stop_codons) - table.forward_table.keys()
    if rna[-3:] in stops:
        rna = rna[:-3]
    if not rna or any(rna[i:i + 3] in stops for i in range(0, len(rna), 3)):
        raise ValueError("CDS must contain sense codons and no internal stop")
    return rna


@cache
def _synonymous_families(translation_table: int) -> dict[str, tuple[str, ...]]:
    by_amino_acid: dict[str, list[str]] = defaultdict(list)
    for codon, aa in CodonTable.unambiguous_rna_by_id[translation_table].forward_table.items():
        by_amino_acid[aa].append(codon)
    return {codon: tuple(family) for family in by_amino_acid.values() for codon in family}


@dataclass(frozen=True)
class CureVacConversionStats:
    """Tracks CureVac US10898584B2 / US12533422B2 (non-higher-G) & divisional US20260284227A1 (any higher-C) conversions."""

    total_sense_codons: int
    optimizable_any_higher_c: int
    converted_any_higher_c: int
    fraction_any_higher_c: float
    optimizable_non_higher_g: int
    converted_non_higher_g: int
    fraction_higher_c_non_higher_g: float

    @property
    def effective_max_fraction(self) -> float:
        return max(self.fraction_any_higher_c, self.fraction_higher_c_non_higher_g)


def compute_curevac_c_optimizable_conversion(
    wildtype_cds_rna: str, candidate_cds_rna: str, *, translation_table: int = 1
) -> CureVacConversionStats:
    """Compute exact C-optimizable codon conversion fractions after strict frame, length, and synonymous validation."""
    wt = normalize_sense_cds(wildtype_cds_rna, translation_table)
    cand = normalize_sense_cds(candidate_cds_rna, translation_table)
    families = _synonymous_families(translation_table)
    if len(wt) != len(cand):
        raise ValueError(f"Mismatched wild-type ({len(wt)} nt) vs candidate ({len(cand)} nt) sense CDS lengths")
    n_cod = len(wt) // 3
    opt_any = 0
    conv_any = 0
    opt_nog = 0
    conv_nog = 0
    for i in range(n_cod):
        w = wt[3 * i : 3 * i + 3]
        c = cand[3 * i : 3 * i + 3]
        fam = families[w]
        if c not in fam:
            raise ValueError(
                f"Non-synonymous codon mismatch at index {i}: wild-type {w!r} vs candidate {c!r}"
            )
        w_c = w.count("C")
        w_g = w.count("G")
        c_c = c.count("C")
        c_g = c.count("G")
        if any(s.count("C") > w_c for s in fam):
            opt_any += 1
            if c_c > w_c:
                conv_any += 1
        if any(s.count("C") > w_c and s.count("G") <= w_g for s in fam):
            opt_nog += 1
            if c_c > w_c and c_g <= w_g:
                conv_nog += 1
    f_any = round(conv_any / opt_any, 6) if opt_any > 0 else 0.0
    f_nog = round(conv_nog / opt_nog, 6) if opt_nog > 0 else 0.0
    return CureVacConversionStats(
        total_sense_codons=n_cod,
        optimizable_any_higher_c=opt_any,
        converted_any_higher_c=conv_any,
        fraction_any_higher_c=f_any,
        optimizable_non_higher_g=opt_nog,
        converted_non_higher_g=conv_nog,
        fraction_higher_c_non_higher_g=f_nog,
    )


_Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
_Reference = Annotated[str, StringConstraints(pattern=r"^[ACGTUacgtu]+$")]
_Threshold = Annotated[float, Field(gt=0, le=1)]


class _ScreenRule(BaseModel):
    model_config = ConfigDict(extra="allow")

    rule_id: _Text
    summary: _Text
    block_in_commercial_clean: StrictBool


class _GenusRule(_ScreenRule):
    patent_family_id: _Text
    granted_publications: list[_Text] = Field(min_length=1)
    target_scope: SegmentScope
    regex_rna: _Text
    block_in_research_max: StrictBool


class _CollisionRule(_ScreenRule):
    patent_family_or_pub: _Text
    target_scope: SegmentScope
    sequence_rna: _Reference
    identity_threshold: _Threshold
    block_in_research_max: StrictBool = False


class _PairMember(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: _Text
    sequence_rna: _Reference


class _CartesianRule(_ScreenRule):
    patent_family_or_pub: _Text
    block_in_research_max: StrictBool
    utr5_identity_threshold: _Threshold
    utr3_identity_threshold: _Threshold
    five_prime_utr_candidates: list[_PairMember] = Field(min_length=1)
    three_prime_utr_candidates: list[_PairMember] = Field(min_length=1)


class _SequenceRegistry(BaseModel):
    registry_id: Literal["patent_fto_guard_v1"]
    version: str = Field(pattern=r"^1\.\d+\.\d+$")
    genus_grammar_rules: list[_GenusRule] = Field(min_length=1)
    sequence_collision_rules: list[_CollisionRule] = Field(min_length=1)
    cartesian_utr_pair_rules: list[_CartesianRule] = Field(default_factory=list)


class _SequenceRegistryV2(_SequenceRegistry):
    registry_id: Literal["patent_fto_guard_v2"]
    version: str = Field(pattern=r"^2\.\d+\.\d+$")
    schema_version: str = Field(pattern=r"^2\.\d+\.\d+$")
    cartesian_utr_pair_rules: list[_CartesianRule] = Field(min_length=1)


class SequenceScanAlerts(list[FTOAlert]):
    """List-compatible result carrying counters from this scan, including scans without hits."""

    __slots__ = ("_performed_counts",)

    def __init__(self, alerts: list[FTOAlert], *, genus: int, collision: int, cartesian_pair: int) -> None:
        super().__init__(alerts)
        self._performed_counts = MappingProxyType(
            {"genus": genus, "collision": collision, "cartesian_pair": cartesian_pair}
        )

    @property
    def performed_counts(self) -> Mapping[str, int]:
        return self._performed_counts


class SequenceFTOGuard:
    """Checks candidate segments against patented clinical UTRs, BioNTech poly(A) genus, and WO2024231621 UTR pairs."""

    def __init__(self, registry_payload: dict[str, Any], *, require_v2: bool = True) -> None:
        schema_type = _SequenceRegistryV2 if require_v2 else _SequenceRegistry
        schema = schema_type.model_validate(registry_payload)
        self.registry_id = schema.registry_id
        self.version = schema.version
        self.genus_rules = [rule.model_dump() for rule in schema.genus_grammar_rules]
        self.collision_rules = [rule.model_dump() for rule in schema.sequence_collision_rules]
        self.cartesian_pair_rules = [rule.model_dump() for rule in schema.cartesian_utr_pair_rules]
        for gr in self.genus_rules:
            try:
                re.compile(gr["regex_rna"])
            except re.error as exc:
                raise ValueError(f"Invalid genus regex in {gr['rule_id']}") from exc
        self._aligner = PairwiseAligner()
        self._aligner.mode = "local"
        self._aligner.match_score = 2.0
        self._aligner.mismatch_score = -1.0
        self._aligner.open_gap_score = -2.0
        self._aligner.extend_gap_score = -1.0

    @classmethod
    def from_json_path(cls, path: Path, *, require_v2: bool = True) -> SequenceFTOGuard:
        allow_v1 = not require_v2 and path.name.endswith("_v1.json")
        return cls(json.loads(path.read_text(encoding="utf-8")), require_v2=not allow_v1)

    def _match_identity(self, sub: str, ref_raw: str, thresh: float) -> tuple[bool, str, tuple[int, int]]:
        """Find a contiguous interval with LCS matches / max(reference length, span) >= threshold."""
        ref = ref_raw.upper().replace("T", "U")
        if not ref:
            return False, "", (0, 0)
        start = sub.find(ref)
        if start >= 0:
            return True, "exact_match:1.000", (start, start + len(ref))

        threshold = Fraction(str(thresh))
        min_matches = math.ceil(len(ref) * threshold)
        max_span = int(len(ref) / threshold)
        if len(sub) < min_matches:
            return False, "", (0, 0)

        positions: dict[str, int] = {}
        for i, base in enumerate(ref):
            positions[base] = positions.get(base, 0) | (1 << i)

        best_identity = 0.0
        best_span = (0, 0)
        for start in range(len(sub) - min_matches + 1):
            row = 0
            limit = min(len(sub), start + max_span)
            for end in range(start + 1, limit + 1):
                # The set bits are the unit increases along a row of the LCS
                # dynamic program. Its population count gives the match count.
                available = row | positions.get(sub[end - 1], 0)
                row = available & ~(available - ((row << 1) | 1))
                matches = row.bit_count()
                if matches + limit - end < min_matches:
                    break
                width = end - start
                denominator = max(len(ref), width)
                if matches * threshold.denominator < threshold.numerator * denominator:
                    continue
                identity = matches / denominator
                if identity > best_identity or (identity == best_identity and width < best_span[1] - best_span[0]):
                    best_identity = identity
                    best_span = (start, end)
        if best_identity:
            return True, f"local_identity:{best_identity:.3f}>={thresh:.2f}", best_span
        return False, "", (0, 0)

    def _match_collision(self, sub: str, ref_raw: str, thresh: float) -> tuple[bool, str, tuple[int, int]]:
        ref = str(ref_raw).upper().replace("T", "U")
        if not sub or not ref:
            return False, "", (0, 0)
        exact_idx = sub.find(ref)
        if exact_idx >= 0:
            return True, "exact_match:1.000", (exact_idx, exact_idx + len(ref))
        if thresh < 1.0 and len(sub) >= int(math.ceil(len(ref) * thresh)):
            alns = self._aligner.align(sub, ref)
            if len(alns) > 0:
                aln = alns[0]
                t_blocks, q_blocks = aln.aligned
                if len(t_blocks) > 0 and len(q_blocks) > 0:
                    matches = sum(
                        1
                        for (ts, te), (qs, _qe) in zip(t_blocks, q_blocks, strict=True)
                        for k in range(int(te) - int(ts))
                        if sub[int(ts) + k] == ref[int(qs) + k]
                    )
                    identity = matches / len(ref)
                    ref_cov = (int(q_blocks[-1][1]) - int(q_blocks[0][0])) / len(ref)
                    span = (int(t_blocks[0][0]), int(t_blocks[-1][1]))
                    if identity + 1e-9 >= thresh and ref_cov + 1e-9 >= thresh:
                        return True, f"local_identity:{identity:.3f}>={thresh:.2f}", span
        return False, "", (0, 0)

    def scan_bundle(
        self, bundle: ConstructBundle, compliance_track: ComplianceProfile
    ) -> SequenceScanAlerts:
        if compliance_track == "commercial_clean" and self.registry_id != "patent_fto_guard_v2":
            raise ValueError("Commercial sequence screening requires patent_fto_guard_v2")
        blocking_flag = "block_in_commercial_clean" if compliance_track == "commercial_clean" else "block_in_research_max"
        alerts: list[FTOAlert] = []
        performed_genus = 0
        performed_collision = 0
        performed_pair = 0
        for mol in bundle.molecules:
            # 1. Genus grammar rules (e.g. BioNTech family 51220548 >=20A + linker + >=20A)
            for rule in self.genus_rules:
                scope = rule["target_scope"]
                seg = mol.get_segment(scope)
                if seg is None:
                    continue
                performed_genus += 1
                sub = mol.sequence[seg.start : seg.end]
                m = re.search(rule["regex_rna"], sub)
                if m:
                    alerts.append(
                        FTOAlert(
                            rule_id=rule["rule_id"],
                            patent_family_or_pub=f"Family {rule['patent_family_id']} ({', '.join(rule['granted_publications'])})",
                            matched_scope=scope,
                            transcript_coordinates=(seg.start + m.start(), seg.start + m.end()),
                            identity_or_pattern=f"mol={mol.molecule_id}|genus_regex:{rule['regex_rna']}",
                            blocked=rule[blocking_flag],
                            summary=rule["summary"],
                            bundle_id=bundle.bundle_id,
                            molecule_id=mol.molecule_id,
                        )
                    )

            # 2. Sequence substring / >=80% identity collision rules
            for rule in self.collision_rules:
                scope = rule["target_scope"]
                seg = mol.get_segment(scope)
                if seg is None:
                    continue
                performed_collision += 1
                sub = mol.sequence[seg.start : seg.end]
                thresh = float(rule.get("identity_threshold", 0.80))
                matched, pat_str, rel_coords = self._match_collision(sub, str(rule["sequence_rna"]), thresh)
                if matched:
                    alerts.append(
                        FTOAlert(
                            rule_id=rule["rule_id"],
                            patent_family_or_pub=rule["patent_family_or_pub"],
                            matched_scope=scope,
                            transcript_coordinates=(seg.start + rel_coords[0], seg.start + rel_coords[1]),
                            identity_or_pattern=pat_str,
                            blocked=rule[blocking_flag],
                            summary=rule["summary"],
                            bundle_id=bundle.bundle_id,
                            molecule_id=mol.molecule_id,
                        )
                    )

            # 3. Cartesian 5'UTR x 3'UTR pair rules (WO2024231621A1 / EP4704918A1 / JP2026515056A)
            #    NEVER individually blocks listed 5'UTRs; triggers ONLY when paired with a listed 3'UTR on the same molecule!
            seg5 = mol.get_segment("five_prime_utr")
            seg3 = mol.get_segment("three_prime_utr")
            if seg5 is not None and seg3 is not None:
                sub5 = mol.sequence[seg5.start : seg5.end]
                sub3 = mol.sequence[seg3.start : seg3.end]
                for pair_rule in self.cartesian_pair_rules:
                    performed_pair += 1
                    t5 = float(pair_rule.get("utr5_identity_threshold", 0.90))
                    t3 = float(pair_rule.get("utr3_identity_threshold", 0.95))
                    hit5: tuple[str, str, tuple[int, int]] | None = None
                    for c5 in pair_rule.get("five_prime_utr_candidates", []):
                        ok5, p5, rel5 = self._match_identity(sub5, str(c5["sequence_rna"]), t5)
                        if ok5:
                            hit5 = (str(c5["id"]), p5, (seg5.start + rel5[0], seg5.start + rel5[1]))
                            break
                    if hit5 is None:
                        continue
                    hit3: tuple[str, str, tuple[int, int]] | None = None
                    for c3 in pair_rule.get("three_prime_utr_candidates", []):
                        ok3, p3, rel3 = self._match_identity(sub3, str(c3["sequence_rna"]), t3)
                        if ok3:
                            hit3 = (str(c3["id"]), p3, (seg3.start + rel3[0], seg3.start + rel3[1]))
                            break
                    if hit3 is None:
                        continue
                    span5 = hit5[2]
                    span3 = hit3[2]
                    alerts.append(
                        FTOAlert(
                            rule_id=pair_rule["rule_id"],
                            patent_family_or_pub=pair_rule["patent_family_or_pub"],
                            matched_scope="full_transcript",
                            transcript_coordinates=(span5[0], span3[1]),
                            identity_or_pattern=(
                                f"mol={mol.molecule_id}|cartesian_pair:{hit5[0]}({hit5[1]}@[{span5[0]},{span5[1]}))"
                                f"x{hit3[0]}({hit3[1]}@[{span3[0]},{span3[1]}))"
                            ),
                            blocked=pair_rule[blocking_flag],
                            summary=pair_rule["summary"],
                            bundle_id=bundle.bundle_id,
                            molecule_id=mol.molecule_id,
                            paired_intervals=[span5, span3],
                        )
                    )
        return SequenceScanAlerts(
            alerts, genus=performed_genus, collision=performed_collision, cartesian_pair=performed_pair,
        )

    def build_receipt(
        self,
        alerts: list[FTOAlert],
        *,
        screen_applied: bool = True,
        performed_counts: Mapping[str, int] | None = None,
    ) -> dict[str, Any]:
        counts = performed_counts
        if counts is None and isinstance(alerts, SequenceScanAlerts):
            counts = alerts.performed_counts
        if not screen_applied:
            g_chk, c_chk, p_chk = 0, 0, 0
        elif counts is not None:
            g_chk = int(counts.get("genus", 0))
            c_chk = int(counts.get("collision", 0))
            p_chk = int(counts.get("cartesian_pair", 0))
        else:
            g_chk, c_chk, p_chk = 0, 0, 0
        return {
            "registry_id": f"{self.registry_id}@{self.version}",
            "screen_applied": bool(screen_applied and (g_chk + c_chk + p_chk) > 0),
            "genus_rules_configured": len(self.genus_rules),
            "collision_rules_configured": len(self.collision_rules),
            "cartesian_pair_rules_configured": len(self.cartesian_pair_rules),
            "genus_rules_checked": g_chk,
            "collision_rules_checked": c_chk,
            "cartesian_pair_rules_checked": p_chk,
            "total_alerts": len(alerts),
            "blocked_alerts": sum(1 for a in alerts if a.blocked),
            "alert_rule_ids": sorted({a.rule_id for a in alerts}),
        }


# ============================================================================
# Method-Level FTO Guard (`method_fto_guard_v1.1.json`) & Static AST Source Lint
# ============================================================================

_FORBIDDEN_PARTITION_PARAM_RE = re.compile(
    r"(?:^|[^a-z0-9]|_)(?:init_region_len|preset_nt|tir_window|frozen_initiation_prefix)(?:$|[^a-z0-9]|_)",
    re.IGNORECASE,
)
_FORBIDDEN_F2_DUAL_HASH_RE = re.compile(
    r"(?:^|[^a-z0-9]|_)(?:dual_hash|dfa_dual_hash|dfa_backpointer_dp|fused_dicodon_energy_dp)(?:$|[^a-z0-9]|_)",
    re.IGNORECASE,
)
_FORBIDDEN_GAM_OR_PPT_RE = re.compile(
    r"(?:^|[^a-z0-9]|_)(?:protein_per_time|boosted_gam|mboost|explainableboosting)(?:$|[^a-z0-9]|_)",
    re.IGNORECASE,
)
_TIE_CAI_MFE_TRIPLE_TOKENS: frozenset[str] = frozenset(
    {
        "tie",
        "cai",
        "mfe",
        "kozak_unpaired_prob",
        "cds_mean_window_mfe",
        "junction_folding_mfe",
        "full_transcript_mfe",
    }
)


def _ast_expr_semantic_tokens(node: ast.AST) -> set[str]:
    """Extract lowercase identifier AND subscript/dict-lookup string tokens from an AST expression."""
    toks: set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            toks.add(n.id.lower())
        elif isinstance(n, ast.Attribute):
            toks.add(n.attr.lower())
        elif isinstance(n, ast.Subscript):
            sl = n.slice
            if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
                toks.add(sl.value.lower())
        elif isinstance(n, ast.Call):
            if (
                isinstance(n.func, ast.Attribute)
                and n.func.attr in {"get", "__getitem__"}
                and n.args
                and isinstance(n.args[0], ast.Constant)
                and isinstance(n.args[0].value, str)
            ):
                toks.add(n.args[0].value.lower())
    return toks


def _ast_has_additive_mfe_regularizer(tree: ast.AST) -> list[int]:
    """Detect scalar expressions of the form `mfe +/- weight * regularizer` across Names, Attributes, and Dict lookups."""
    hit_lines: list[int] = []

    def _has_mul(node: ast.AST) -> bool:
        return any(isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mult) for n in ast.walk(node))

    reg_keywords = ("cai", "reg", "reg_term", "regularizer", "tie", "codon_adaptation")
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            lnames = _ast_expr_semantic_tokens(node.left)
            rnames = _ast_expr_semantic_tokens(node.right)
            left_has_mfe = any("mfe" in nm for nm in lnames)
            right_has_mfe = any("mfe" in nm for nm in rnames)
            if left_has_mfe and not right_has_mfe and (
                _has_mul(node.right) or any(k in rnames for k in reg_keywords)
            ):
                hit_lines.append(getattr(node, "lineno", 0))
            elif right_has_mfe and not left_has_mfe and (
                _has_mul(node.left) or any(k in lnames for k in reg_keywords)
            ):
                hit_lines.append(getattr(node, "lineno", 0))
    return sorted(set(hit_lines))


def audit_codebase_method_fto_lints(src_root: Path) -> dict[str, Any]:
    """CI/static AST lint enforcing all architectural method-FTO bans across a non-empty `src_root`."""
    root = Path(src_root)
    if not root.is_dir():
        return {
            "status": "BLOCK",
            "files_scanned": 0,
            "violations": [
                {
                    "boundary_id": "LINT_INFRASTRUCTURE_EMPTY_OR_MISSING_SOURCE_ROOT",
                    "file": str(root),
                    "line": 0,
                    "reason": f"Source directory '{root}' does not exist",
                }
            ],
        }
    violations: list[dict[str, Any]] = []
    scanned = 0
    found_rel_names: set[str] = set()
    for py_file in sorted(root.rglob("*.py")):
        if py_file.name == "fto_guard.py":
            continue
        scanned += 1
        found_rel_names.add(py_file.name)
        src = py_file.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(py_file))
        docstring_const_ids = {
            id(body[0].value)
            for n in ast.walk(tree)
            if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and (body := getattr(n, "body", None))
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        }

        for node in ast.walk(tree):
            # Function / class / variable / argument / string checks across F1, F2, F4C
            ident_candidates: list[tuple[str, int]] = []
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                ident_candidates.append((node.name, node.lineno))
                for arg in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]:
                    ident_candidates.append((arg.arg, node.lineno))
            elif isinstance(node, ast.ClassDef):
                ident_candidates.append((node.name, node.lineno))
            elif isinstance(node, ast.Attribute):
                ident_candidates.append((node.attr, node.lineno))
            elif isinstance(node, ast.Name):
                ident_candidates.append((node.id, getattr(node, "lineno", 0)))
            elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstring_const_ids:
                ident_candidates.append((node.value, getattr(node, "lineno", 0)))

            for text_val, ln in ident_candidates:
                if _FORBIDDEN_PARTITION_PARAM_RE.search(text_val):
                    violations.append(
                        {
                            "boundary_id": "BAIDU_CN119265214B_STAGED_UTR5_NCDS_PARTITION_OPT",
                            "file": str(py_file),
                            "line": ln,
                            "reason": f"Forbidden initiation-partition identifier/key {text_val!r}",
                        }
                    )
                if _FORBIDDEN_F2_DUAL_HASH_RE.search(text_val):
                    violations.append(
                        {
                            "boundary_id": "BAIDU_CN118038986B_DFA_DUAL_HASH_DICODON_DP",
                            "file": str(py_file),
                            "line": ln,
                            "reason": f"Forbidden dual-hash DFA dicodon DP identifier {text_val!r}",
                        }
                    )
                if _FORBIDDEN_GAM_OR_PPT_RE.search(text_val):
                    violations.append(
                        {
                            "boundary_id": "LEIBNIZ_HANNOVER_US11848074B2_EP3551758B1_BOOSTED_GAM_PROTEIN_PER_TIME",
                            "file": str(py_file),
                            "line": ln,
                            "reason": f"Forbidden boosted-GAM / protein_per_time identifier {text_val!r}",
                        }
                    )

        for ln in _ast_has_additive_mfe_regularizer(tree):
            violations.append(
                {
                    "boundary_id": "BAIDU_USA_LINEARDESIGN_DFA_CFG_LATTICE_AND_ADDITIVE_MFE_REG",
                    "file": str(py_file),
                    "line": ln,
                    "reason": "Additive MFE + weighted regularizer scalar expression detected",
                }
            )

    if scanned == 0:
        violations.append(
            {
                "boundary_id": "LINT_INFRASTRUCTURE_EMPTY_OR_MISSING_SOURCE_ROOT",
                "file": str(root),
                "line": 0,
                "reason": f"Zero Python files scanned under '{root}'",
            }
        )
    elif root.name == "mrnax" and not {"pareto_director.py", "stage0_baseline.py", "profiles.py"} <= found_rel_names:
        violations.append(
            {
                "boundary_id": "LINT_INFRASTRUCTURE_INCOMPLETE_PACKAGE_TREE",
                "file": str(root),
                "line": 0,
                "reason": "Package tree missing mandatory core engine modules",
            }
        )

    return {
        "status": "PASS" if not violations else "BLOCK",
        "files_scanned": scanned,
        "violations": violations,
    }


def _receipt_asset(component_id: str, seed_dir: Path | None) -> Path | None:
    from mrnax.engine.profiles import _REPLACEMENT_CARD_BY_COMPONENT, resolve_default_cards_dir, resolve_default_seed_dir

    engine_dir = Path(__file__).resolve().parents[1] / "engine"
    if component_id == "viennarna_thermo":
        return Path(importlib.metadata.distribution("ViennaRNA").locate_file("RNA/RNA.py"))
    if component_id in _REPLACEMENT_CARD_BY_COMPONENT:
        return resolve_default_cards_dir() / _REPLACEMENT_CARD_BY_COMPONENT[component_id]
    if component_id == "surrogate_v1_clean":
        return (seed_dir or resolve_default_seed_dir()) / "surrogate_v1/TRAINING_PROVENANCE_MANIFEST.json"
    if component_id == "stage0_random":
        return engine_dir / "stage0_baseline.py"
    if component_id == "pareto_nsga2":
        return engine_dir / "pareto_director.py"
    if component_id == "dnachisel_cds":
        return Path(importlib.metadata.distribution("dnachisel").locate_file(
            "dnachisel/DnaOptimizationProblem/DnaOptimizationProblem.py"
        ))
    return None


COMMERCIAL_ZERO_GPL_REQUIREMENT = "commercial_license:zero_transitive_gpl"

_GPL_LICENSE = re.compile(
    r"\b(?:[al]?gpl(?:[- ]?v?\d+(?:\.\d+)*)?\b|gnu\s+(?:(?:affero|lesser|library)\s+)?general\s+public\s+license)",
    re.IGNORECASE,
)
_GCC_EXCEPTION = re.compile(
    r"\bgpl[- ]?v?\d+(?:\.\d+)*(?:-only|-or-later|\+)?\s+with\s+(?:the\s+)?"
    r"(?:gcc-exception-[\d.]+|gcc runtime library exception)",
    re.IGNORECASE,
)
_SPDX_TAG = re.compile(r"\bSPDX-License-Identifier:[ \t]*([^\r\n]*)", re.IGNORECASE)
_GPL_NAME = rf"(?:GNU\s+)?{_GPL_LICENSE.pattern}"
_GRANT_VERB = r"(?:licen[cs]ed|distributed|released|redistributed|available|provided|offered|covered)"
_GRANT_LINK = r"\s+(?:exclusively\s+|solely\s+|only\s+)?under\s+(?:(?:the\s+)?terms\s+of\s+)?(?:the\s+)?"
# Match the qualified reference itself, so a separate grant in the same notice still counts.
_EXTERNAL_OR_PAST_GRANT = re.compile(
    rf"(?:\b(?:previously|formerly|originally|never|not)\s+|"
    rf"\b(?:(?:versions?|releases?)\s+(?:prior to|before)\s+[\w.]+|(?:earlier|prior|older)\s+releases?)"
    rf"\s+(?:were|was)\s+|"
    rf"\b(?:compatible|interoperable)\s+with\s+(?:[\w-]+\s+){{0,4}}?|"
    rf"\b(?:combine|link)\s+(?:[\w-]+\s+){{0,10}}with\s+(?:other\s+)?"
    rf"(?:software|programs?|applications?|libraries|works?)\s+(?:(?:that|which)\s+)?(?:is\s+|are\s+)?)"
    rf"{_GRANT_VERB}{_GRANT_LINK}{_GPL_NAME}", re.IGNORECASE,
)
_GPL_GRANT = re.compile(
    rf"(?:\b{_GRANT_VERB}{_GRANT_LINK}|\bunder\s+(?:the\s+)?terms\s+of\s+(?:the\s+)?|"
    rf"\b(?:is|are)\s+under\s+(?:the\s+)?|\blicen[cs]e\s+for\s+(?:[\w-]+\s+){{1,8}}is\s+(?:the\s+)?)"
    rf"(?P<license>{_GPL_NAME})", re.IGNORECASE,
)
_GPL_DECLARATION = re.compile(
    rf"\b(?:is|are)\s+(?P<license>{_GPL_NAME})(?:-only|-or-later|\+)?"
    r"(?=$|[.,;](?:\s|$)|[- ]licen[cs]ed\b|\s+only\b)", re.IGNORECASE | re.MULTILINE,
)
_GPL_HEADING = re.compile(
    rf"^[ \t#*/=-]*(?:licen[cs]e[ \t]*:[ \t]*)?(?P<license>{_GPL_NAME})"
    r"(?:[ \t]+\([AL]?GPL(?:v?[\d.]+)?\))?"
    r"(?:-only|-or-later|\+)?(?:[ ,\t]+(?:version\s+|v)?\d[\w., +()-]*)?[. \t#*/=-]*$",
    re.IGNORECASE | re.MULTILINE,
)
_PACKAGE_SUBJECT = r"(?:the|this)\s+(?:(?:entire|whole)\s+)?(?:package|distribution|project)"
_PACKAGE_DUAL_GRANT = re.compile(
    rf"(?:^|(?<=[.!?]))\s*(?:{_PACKAGE_SUBJECT}\s+is\s+(?:dual[- ]licen[cs]ed|licen[cs]ed)|"
    rf"(?:at\s+your\s+option,\s*)?you\s+may\s+(?:use|redistribute)\s+{_PACKAGE_SUBJECT})"
    r"\s+under\s+(?:either\s+)?(?:the\s+)?([^\n,;]+?)\s+or\s+(?:the\s+)?([^\n,;]+?)"
    r"(?:,\s*at\s+your\s+option)?[.]?[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
_INDEPENDENT_GRANT = re.compile(
    r"\b(?:exclusively|solely|bundled|vendored|vendor|third[- ]party|helper|non-vendored)\b|"
    r"^Files:[ \t]", re.IGNORECASE | re.MULTILINE,
)


def _has_gpl_license(text: str) -> bool:
    # An exception qualifies its associated license, not the rest of a manifest.
    return _GPL_LICENSE.search(_GCC_EXCEPTION.sub("", text)) is not None


def _spdx_requires_gpl(expression: str) -> bool:
    """A conjunction requires every license; a disjunction offers a choice."""
    try:
        normalized = canonicalize_license_expression(expression)
        tokens = deque(normalized.replace("(", " ( ").replace(")", " ) ").split())

        def term() -> bool:
            token = tokens.popleft()
            if token == "(":
                value = disjunction()
                if tokens.popleft() != ")":
                    raise ValueError("Expected closing parenthesis")
                return value
            if token in {")", "AND", "OR", "WITH"}:
                raise ValueError("Expected a license identifier")
            if tokens and tokens[0] == "WITH":
                tokens.popleft()
                token += " WITH " + tokens.popleft()
            return token.startswith("LicenseRef-") or _has_gpl_license(token)

        def conjunction() -> bool:
            value = term()
            while tokens and tokens[0] == "AND":
                tokens.popleft()
                required = term()
                value = value or required
            return value

        def disjunction() -> bool:
            value = conjunction()
            while tokens and tokens[0] == "OR":
                tokens.popleft()
                alternative = conjunction()
                value = value and alternative
            return value

        return disjunction() or bool(tokens)
    except (ValueError, IndexError, RecursionError):
        return True


# Exact METADATA notices from the reviewed Darwin arm64 and Linux x86_64 wheels. ADR-0003 covers
# static and Python copyleft. SciPy lists separate dynamically linked GCC and LGPL libquadmath libraries;
# sklearn lists an Apache/LLVM libomp library / GCC runtime whose exception mentions GPL interoperability.
_REVIEWED_BINARY_NOTICES: dict[str, frozenset[str]] = {
    "scipy": frozenset({
        "c85f6c4154a28ccb2a49fe9a1cbcacb17939c14ffad7a2f63d32655610652bb7",
        "2a463073ea6816d22b763c8e2f03886f4e8bf4f2a234f3c9267401d688588d17",
    }),
    "scikit-learn": frozenset({
        "b790419e16d1f5e137f4718a231eb44e47f827c371a566aa1b473b9c2935f9b5",
        "d22525053a4374436269e2fd29b4121bc6e1634d4eea0ceae8a4d7d997406a03",
    }),
}
_BIOPYTHON_AGREEMENT_SHA256 = "11d1ebcaad2eb9f0f09a3df1deb4aca99a4da366ae9ae70f828c7447a24de453"


# NumPy ships the GPL text for its GCC-excepted libgfortran runtime. Scope the reviewed
# exception to exact Darwin/Linux notices so changing it or adding a file is checked.
_REVIEWED_LICENSE_FILES: dict[tuple[str, str], frozenset[str]] = {
    ("numpy", "LICENSE.txt"): frozenset({
        "853f260937a63b0db91c5c9ea99a5c3bbe82e88834ada124a7415502403d1600",
        "8c3433f4b8b93e30a2b54a376ef8d0576a3e1e8d14b14eb288da0a75e0c99627",
    }),
}


def _read_declared_license(dist, declared: str) -> str | None:
    # PEP 639 paths are relative to the distribution's licenses directory.
    if (not declared or "\\" in declared or ":" in declared or "\x00" in declared
            or any(part in {"", ".", ".."} for part in declared.split("/"))):
        return None
    try:
        notice = dist.read_text(f"licenses/{declared}")
        if notice is None:
            # Older wheels put declared files directly inside .dist-info.
            if any(str(f).partition("/")[0].endswith(".dist-info")
                   and str(f).partition("/")[2] == declared for f in dist.files or ()):
                notice = dist.read_text(declared)
    except (OSError, UnicodeError):
        return None
    return notice


def _gpl_families(text: str) -> frozenset[str]:
    families = set()
    for match in _GPL_LICENSE.finditer(text.casefold()):
        name = match.group()
        if "affero" in name or name.startswith("agpl"):
            families.add("AGPL")
        elif "lesser" in name or "library" in name or name.startswith("lgpl"):
            families.add("LGPL")
        else:
            families.add("GPL")
    return frozenset(families)


@dataclass(frozen=True)
class _LicenseNotice:
    copyleft: frozenset[str] = frozenset()
    package_alternatives: frozenset[str] = frozenset()
    independent_obligation: bool = False


@lru_cache(maxsize=256)
def _classify_license_notice(notice: str) -> _LicenseNotice:
    if "LicenseRef-" not in notice and not _GPL_LICENSE.search(notice) and not _SPDX_TAG.search(notice):
        return _LicenseNotice()
    copyleft: set[str] = set()
    alternatives: set[str] = set()
    independent = False

    def record(grant: re.Match[str], licenses: frozenset[str]) -> None:
        nonlocal independent
        copyleft.update(licenses)
        # A choice granted for the package cannot relicense a separately scoped component.
        context = grant.string[:grant.start()].rsplit("\n\n", 1)[-1] + grant.group()
        if not licenses or _INDEPENDENT_GRANT.search(context):
            independent = True

    for tag in _SPDX_TAG.finditer(notice):
        expression = tag[1].strip().removesuffix("*/").removesuffix("-->").strip()
        if _spdx_requires_gpl(expression):
            try:
                normalized = canonicalize_license_expression(expression)
            except ValueError:
                independent = True
            else:
                record(tag, _gpl_families(normalized))
                independent |= "LicenseRef-" in normalized

    def package_choice(match: re.Match[str]) -> str:
        options = match.groups()
        for gpl, alternative in (options, options[::-1]):
            if re.fullmatch(rf"{_GPL_NAME}(?:-only|-or-later|\+| version [\d.]+(?: only)?)?", gpl, re.I):
                license_id = re.sub(r"\s+license$", "", alternative, flags=re.I)
                if not _spdx_requires_gpl(license_id):
                    alternatives.update(_gpl_families(gpl))
                    return ""
        return match.group()

    prose = _PACKAGE_DUAL_GRANT.sub(package_choice, _SPDX_TAG.sub("", notice))
    if len(prose.strip().splitlines()) == 1:
        try:
            expression = canonicalize_license_expression(prose.strip())
        except ValueError:
            pass
        else:
            if _spdx_requires_gpl(expression):
                copyleft.update(_gpl_families(expression))
                independent |= "LicenseRef-" in expression
            return _LicenseNotice(frozenset(copyleft), frozenset(alternatives), independent)

    prose = _EXTERNAL_OR_PAST_GRANT.sub("", _GCC_EXCEPTION.sub("", prose))
    for pattern in (_GPL_HEADING, _GPL_GRANT, _GPL_DECLARATION):
        for grant in pattern.finditer(prose):
            record(grant, _gpl_families(grant["license"]))
    return _LicenseNotice(frozenset(copyleft), frozenset(alternatives), independent)


def _distribution_requires_gpl(name: str, dist, metadata) -> bool:
    expression = metadata.get("License-Expression")
    alternatives: set[str] = set()
    if expression == "LicenseRef-Biopython-License-Agreement" and name == "biopython":
        agreement = next((f for f in dist.files or () if str(f).endswith(".dist-info/licenses/LICENSE.rst")), None)
        if agreement is None:
            return True
        try:
            if hashlib.sha256(dist.locate_file(agreement).read_bytes()).hexdigest() != _BIOPYTHON_AGREEMENT_SHA256:
                return True
        except OSError:
            return True
    elif expression and expression.strip():
        if _spdx_requires_gpl(expression):
            return True
        normalized = canonicalize_license_expression(expression)
        if re.search(r"\bOR\b", normalized):
            alternatives.update(_gpl_families(normalized))
    else:
        description = metadata.get("License") or ""
        classifiers = metadata.get_all("Classifier") or ()
        if any(c.startswith("License ::") and _has_gpl_license(c) for c in classifiers):
            return True
        if (_has_gpl_license(description) and
                hashlib.sha256(description.encode()).hexdigest() not in _REVIEWED_BINARY_NOTICES.get(name, ())):
            return True

    declared = metadata.get_all("License-File") or ()
    copyleft: set[str] = set()
    independent = False
    for filename in declared:
        notice = _read_declared_license(dist, filename)
        if not notice or not notice.strip():
            return True
        reviewed = _REVIEWED_LICENSE_FILES.get((name, filename))
        if reviewed and hashlib.sha256(notice.encode()).hexdigest() in reviewed:
            continue
        evidence = _classify_license_notice(notice)
        copyleft.update(evidence.copyleft)
        alternatives.update(evidence.package_alternatives)
        independent |= evidence.independent_obligation
    if declared:
        return independent or bool(copyleft - alternatives)
    return expression is not None and not expression.strip()


def count_component_gpl_dependencies(component_id: str, code_lic: str, weights_lic: str) -> int:
    """Visit active catalog and Requires-Dist edges; count each copyleft asset once."""
    from mrnax.engine.profiles import COMPONENT_CATALOG, _normalize_pkg_name, _resolve_catalog_dependency

    catalog = {canonicalize_name(s.component_id): s for s in COMPONENT_CATALOG.values()}
    root_name = canonicalize_name(component_id)
    root = ("catalog" if root_name in catalog else "distribution", root_name)
    hits = {root} if _has_gpl_license(f"{code_lic} {weights_lic}") else set()
    # Active extras are part of the visit key: another path can enable additional edges.
    queue = deque([(root, frozenset())])
    seen: set[tuple[tuple[str, str], frozenset[str]]] = set()
    while queue:
        node, extras = queue.popleft()
        if (node, extras) in seen:
            continue
        seen.add((node, extras))
        kind, name = node
        if kind == "catalog":
            spec = catalog[name]
            if _has_gpl_license(f"{spec.code_license} {spec.weights_license}"):
                hits.add(node)
            dependencies = spec.transitive_dependencies
        else:
            if name in {"viennarna", "rna", "bpfold"}:
                hits.add(node)
            try:
                dist = importlib.metadata.distribution(name)
            except importlib.metadata.PackageNotFoundError:
                continue
            metadata = dist.metadata
            if _distribution_requires_gpl(name, dist, metadata):
                hits.add(node)
            dependencies = dist.requires or ()
        for edge in dependencies:
            annotated_copyleft = False
            try:
                requirement = Requirement(edge)
            except InvalidRequirement:
                # Catalog declarations also allow annotated asset names, e.g. "torch (BSD-3-Clause)".
                dep_name = _normalize_pkg_name(edge)
                dep_extras = frozenset()
                annotated_copyleft = _has_gpl_license(edge)
            else:
                if requirement.marker is not None and not any(
                    requirement.marker.evaluate({"extra": extra}) for extra in (extras | {""})
                ):
                    continue
                dep_name = canonicalize_name(requirement.name)
                dep_extras = frozenset(requirement.extras)
            # A Requires-Dist always names an installed distribution, even when a catalog ID matches.
            child_spec = _resolve_catalog_dependency(edge, COMPONENT_CATALOG) if kind == "catalog" else None
            if child_spec and canonicalize_name(child_spec.component_id) != name:
                child = ("catalog", canonicalize_name(child_spec.component_id))
            else:
                child = ("distribution", dep_name)
            if annotated_copyleft:
                hits.add(child)
            queue.append((child, dep_extras))
    return len(hits)


def _count_recursive_gpl_dependencies(
    component_name: str, code_lic: str, weights_lic: str, *, seed_dir: Path | None = None,
) -> tuple[int, str | None]:
    from mrnax.engine.profiles import COMPONENT_CATALOG

    root = COMPONENT_CATALOG.get(component_name) or next(
        (s for s in COMPONENT_CATALOG.values() if s.display_name == component_name), None
    )
    count = count_component_gpl_dependencies(root.component_id if root else component_name, code_lic, weights_lic)
    asset = _receipt_asset(root.component_id, seed_dir) if root else None
    digest = hashlib.sha256(asset.read_bytes()).hexdigest() if asset is not None else None
    return count, digest


class _Publication(BaseModel):
    publication: _Text


class _MethodBoundary(BaseModel):
    boundary_id: _Text
    family_code: _Text
    risk_level: _Text
    jurisdictions_active_grant: list[_Text] = Field(min_length=1)
    jurisdictions_pending: list[_Text]
    jurisdictions_ceased_or_refused: list[_Text]
    patent_family: list[_Publication] = Field(min_length=1)
    verified_claim_1_elements: list[_Text] = Field(min_length=1)
    enforcement_action_commercial_clean: _Text
    prohibited_execution_pattern: dict[_Text, StrictBool] = Field(min_length=1)
    clean_room_design_around_requirements: list[_Text] = Field(min_length=1)


class _ReceiptField(BaseModel):
    field: _Text
    type: _Text
    required: StrictBool


class _ReceiptSchema(BaseModel):
    schema_version: Literal["1.1.0"]
    fields: list[_ReceiptField] = Field(min_length=1)


class _MethodRegistry(BaseModel):
    registry_id: Literal["method_fto_guard_v1.1"]
    version: Annotated[str, StringConstraints(pattern=r"^1\.1\.\d+$")]
    certifies_100_percent_fto: StrictBool
    jurisdiction_monitoring_required: list[_Text]
    method_claim_boundaries: list[_MethodBoundary]
    sequence_and_composition_references: list[_Publication] = Field(min_length=1)
    required_commercial_run_audit_fields: _ReceiptSchema


class MethodFTOGuard:
    """Evaluates live execution plans and trajectories against `method_fto_guard_v1.1.json`."""

    BOUNDARY_IDS: tuple[str, ...] = (
        "BAIDU_CN119265214B_STAGED_UTR5_NCDS_PARTITION_OPT",
        "BAIDU_CN118038986B_DFA_DUAL_HASH_DICODON_DP",
        "BAIDU_USA_LINEARDESIGN_DFA_CFG_LATTICE_AND_ADDITIVE_MFE_REG",
        "CUREVAC_US10898584B2_FAMILY_SYNONYMOUS_CYTOSINE_THRESHOLD",
        "ASU_US12706176B1_RANDOM_SYNONYMOUS_SAMPLING_PROBLEM_COUNT_ARGMIN",
        "LEIBNIZ_HANNOVER_US11848074B2_EP3551758B1_BOOSTED_GAM_PROTEIN_PER_TIME",
    )
    REQUIRED_EVENTS = {
        "F1": {
            "hard_partition_5utr_plus_preset_n_cds_as_initiation_region",
            "forbid_config_keys_matching_init_region_len_or_preset_nt_or_tir_window",
            "sequential_stage1_initiation_opt_then_frozen_stage2_cds_body",
            "scalar_weighted_sum_tie_cai_mfe",
        },
        "F2": {
            "uses_dual_hash_table_atomic_state_dfa_dp",
            "stores_codon_graph_dp_backpointer_table",
            "fuses_subpath_free_energy_and_dicodon_score_in_single_dp_recursion",
        },
        "F3": {
            "uses_dfa_lattice_cfg_intersection",
            "invokes_lineardesign_binary_or_tables_or_api",
            "computes_or_optimizes_additive_mfe_plus_weighted_regularizer_scalar",
        },
        "F4A": {
            "cytosine_maximization_objective_or_rewrite_rule",
            "wildtype_c_optimizable_conversion_ge_070",
        },
        "F4B": {
            "massive_random_synonymous_reverse_translation_plus_problem_motif_count_argmin",
            "weighted_random_replacement_list_motif_repair",
        },
        "F4C": {
            "uses_boosted_gam_with_protein_per_time_scoring",
            "random_elongation_rate_weighted_codon_sampler",
        },
    }
    REQUIRED_PUBLICATIONS = {
        "F1": {"CN119265214B", "JP7892088B2", "US20250092387A1"},
        "F2": {"CN118038986B"},
        "F3": {"JP7367090B2", "US20220269951A1", "US20240394558A1", "EP4030438A1", "EP4682890A2_A3", "CN114974428A"},
        "F4A": {"US10898584B2", "US12533422B2", "US20260284227A1", "CA2925021C", "ES2806575T3"},
        "F4B": {"US12706176B1"},
        "F4C": {"US11848074B2", "EP3551758B1"},
    }
    MANDATORY_PUBLICATIONS = frozenset().union(*REQUIRED_PUBLICATIONS.values())
    REQUIRED_RECEIPT_FIELDS = {
        "run_id", "timestamp_utc", "commercial_tier", "target_deployment_jurisdictions",
        "engine_git_commit", "input_origin_type", "wildtype_c_optimizable_replacement_fraction",
        "region_partition_strategy", "optimization_schedule_and_operators", "objective_formulation",
        "three_layer_dependency_manifest", "method_fto_boundary_checks", "sequence_fto_guard_v2_receipt",
        "counsel_fto_signoff_required",
    }

    def __init__(self, payload: dict[str, Any]) -> None:
        policy = _MethodRegistry.model_validate(payload)
        self.registry_id = policy.registry_id
        self.version = policy.version
        self.certifies_100_percent_fto = policy.certifies_100_percent_fto
        if self.certifies_100_percent_fto:
            raise ValueError("method_fto_guard_v1.1 MUST declare certifies_100_percent_fto=false")
        self.jurisdictions = policy.jurisdiction_monitoring_required
        if not {"CN", "US", "EP", "JP"} <= set(self.jurisdictions):
            raise ValueError("method_fto_guard_v1.1 must monitor at least CN, US, EP, and JP jurisdictions")
        required_fields = {f.field for f in policy.required_commercial_run_audit_fields.fields if f.required}
        if not self.REQUIRED_RECEIPT_FIELDS <= required_fields:
            raise ValueError("Incomplete required_commercial_run_audit_fields")
        self._seed_dir: Path | None = None
        self.boundaries = payload["method_claim_boundaries"]
        expected_ids = dict(zip(self.REQUIRED_EVENTS, self.BOUNDARY_IDS, strict=True))
        self.events_by_boundary: dict[str, set[str]] = {}
        pubs = {p.publication for p in policy.sequence_and_composition_references}
        for boundary in policy.method_claim_boundaries:
            family = boundary.family_code
            bid = boundary.boundary_id
            if expected_ids.get(family) != bid or bid in self.events_by_boundary:
                raise ValueError(f"Unknown or duplicate method boundary {family!r} / {bid!r}")
            patterns = boundary.prohibited_execution_pattern
            if not self.REQUIRED_EVENTS[family] <= patterns.keys() or not all(patterns.values()):
                raise ValueError(f"Missing or disabled prohibited_execution_pattern for {family}")
            self.events_by_boundary[bid] = set(patterns)
            publications = {p.publication for p in boundary.patent_family}
            if not self.REQUIRED_PUBLICATIONS[family] <= publications:
                raise ValueError(f"Missing mandatory patent_family publication in {family}")
            pubs.update(publications)
        if set(self.BOUNDARY_IDS) != self.events_by_boundary.keys():
            raise ValueError("Incomplete method_claim_boundaries")
        self.covered_publications = frozenset(pubs)

    @classmethod
    def from_json_path(cls, path: Path) -> MethodFTOGuard:
        guard = cls(json.loads(path.read_text(encoding="utf-8")))
        guard._seed_dir = path.parent
        return guard

    @classmethod
    def from_seed_dir(cls, seed_dir: Path) -> MethodFTOGuard:
        p = Path(seed_dir) / "method_fto_guard_v1.1.json"
        if not p.is_file():
            raise FileNotFoundError(f"Mandatory method_fto_guard_v1.1.json missing at '{p}'")
        return cls.from_json_path(p)

    def not_executed_checks(self, *, override_blocks: dict[str, str] | None = None) -> dict[str, dict[str, str]]:
        checks = {
            bid: {"status": "NOT_EXECUTED", "evidence_note": "Pipeline halted prior to candidate execution."}
            for bid in self.BOUNDARY_IDS
        }
        for bid, note in (override_blocks or {}).items():
            checks[bid] = {"status": "BLOCK", "evidence_note": note}
        return checks

    def evaluate_pre_execution_intent(self, intent: DesignIntent) -> dict[str, str]:
        """Return `{boundary_id: block_reason}` for any prohibited objective/constraint/generator pattern."""
        blocks: dict[str, str] = {}
        if intent.compliance_track != "commercial_clean":
            return blocks

        for c in intent.constraints:
            for k in c.params:
                if _FORBIDDEN_PARTITION_PARAM_RE.search(str(k)):
                    blocks["BAIDU_CN119265214B_STAGED_UTR5_NCDS_PARTITION_OPT"] = (
                        f"Forbidden initiation-region split parameter '{k}' blocked under CN119265214B / JP7892088B2."
                    )

        rankable_objs = [o for o in intent.objectives if o.evidence_status != "unverified_target_for_wetlab"]
        for obj in intent.objectives:
            mid = obj.metric_id.strip().lower()
            utt = obj.source_utterance.strip().lower()
            if _FORBIDDEN_GAM_OR_PPT_RE.search(mid) or ("boosted gam" in utt and _FORBIDDEN_GAM_OR_PPT_RE.search(mid)):
                blocks["LEIBNIZ_HANNOVER_US11848074B2_EP3551758B1_BOOSTED_GAM_PROTEIN_PER_TIME"] = (
                    f"Objective metric '{obj.metric_id}' is blocked under EP3551758B1 (pure-software Cl.1/Cl.14) & US11848074B2."
                )
            if "mfe" in mid and any(tok in mid for tok in ("plus", "add", "reg", "cai", "lambda", "scalar")):
                blocks["BAIDU_USA_LINEARDESIGN_DFA_CFG_LATTICE_AND_ADDITIVE_MFE_REG"] = (
                    f"Additive MFE+regularizer scalar metric '{obj.metric_id}' is blocked under JP7367090B2 & precaution US20240394558A1."
                )
            if mid in {"cds_cytosine_fraction", "cytosine_content", "maximize_cytosine"} and obj.direction == "maximize":
                blocks["CUREVAC_US10898584B2_FAMILY_SYNONYMOUS_CYTOSINE_THRESHOLD"] = (
                    f"Cytosine-maximization objective '{obj.metric_id}' is blocked under CureVac US10898584B2 / US12533422B2 / US20260284227A1."
                )
            if mid in {"problematic_motif_count", "motif_violation_count"} and obj.direction == "minimize":
                blocks["ASU_US12706176B1_RANDOM_SYNONYMOUS_SAMPLING_PROBLEM_COUNT_ARGMIN"] = (
                    f"Motif-count argmin objective '{obj.metric_id}' is blocked under ASU US12706176B1 (use constructive hard gates)."
                )

        if rankable_objs and all(o.metric_id.strip().lower() in _TIE_CAI_MFE_TRIPLE_TOKENS for o in rankable_objs):
            blocks["BAIDU_CN119265214B_STAGED_UTR5_NCDS_PARTITION_OPT"] = (
                "Commercial runs with active objectives must include >=1 empirical axis outside {TIE, CAI, MFE} "
                "(e.g. consensus_halflife_pred, utr5_mrl_pred, cds_stability_pred, structrfm_cds_median_shape) per CN119265214B / JP7892088B2 guard."
            )
        return blocks

    def build_runtime_audit_record(
        self,
        *,
        intent: DesignIntent,
        opt_mode: str,
        used_two_zone_partition: bool,
        executed_operators: list[str],
        invoked_components: list[ComponentInvocationRecord],
        seq_fto_receipt: dict[str, Any],
        input_origin_type: str,
        max_curevac_any_c: float | None = None,
        max_curevac_no_higher_g: float | None = None,
        curevac_error_note: str | None = None,
        target_jurisdictions: list[str] | None = None,
        run_id: str | None = None,
        engine_git_commit: str | None = None,
    ) -> MethodFTOAuditRecord:
        is_cc = intent.compliance_track == "commercial_clean"
        rankable_objs = [o for o in intent.objectives if o.evidence_status != "unverified_target_for_wetlab"]
        obj_ids = [o.metric_id for o in rankable_objs]
        has_outside_triple = bool(obj_ids) and any(
            mid.lower() not in _TIE_CAI_MFE_TRIPLE_TOKENS for mid in obj_ids
        )
        ops_low = [str(op).strip().lower() for op in executed_operators]
        comps_low = [str(c.name).strip().lower() for c in invoked_components]
        evidence_tokens = " ".join([*ops_low, *comps_low, *(m.lower() for m in obj_ids)])

        partition_markers = (
            "two_zone", "sliding_window", "frozen_initiation", "init_region_len", "preset_nt",
            "hard_partition_5utr_plus_preset_n_cds_as_initiation_region",
            "sequential_stage1_initiation_opt_then_frozen_stage2_cds_body",
        )
        used_partition_ev = used_two_zone_partition or any(
            marker in op for op in ops_low for marker in partition_markers
        )
        used_f2_dual_hash = any(
            k in evidence_tokens for k in ("dual_hash", "dfa_dicodon", "backpointer_dp", "fused_dicodon_energy", "cn118038986")
        )
        used_dfa_cfg = any(
            k in evidence_tokens for k in ("dfa_cfg", "uses_dfa_lattice_cfg_intersection", "lattice_intersection", "lineardesign", "cfg_parser")
        )
        used_additive_mfe_reg = any(
            ("mfe" in m.lower() and any(t in m.lower() for t in ("plus", "add", "reg", "cai", "lambda", "scalar")))
            for m in [*obj_ids, *ops_low]
        )
        used_f4b_weighted_or_argmin = any(
            k in evidence_tokens
            for k in ("weighted_random_replacement", "weighted_random_repair", "motif_count_argmin", "problematic_motif_count", "massive_random_pool_argmin")
        )
        used_f4c_gam_or_ppt = bool(_FORBIDDEN_GAM_OR_PPT_RE.search(evidence_tokens))

        region_strategy = (
            "research_only_two_zone_sliding_window"
            if used_partition_ev
            else (
                "full_transcript_free_variables_zero_frozen_prefix"
                if opt_mode == "full_transcript_non_partitioned_pareto"
                else "stage0_whole_cds_without_partition"
            )
        )

        # F1 decision
        if used_partition_ev:
            f1_status = "BLOCK" if is_cc else "WARN"
        elif obj_ids and not has_outside_triple and is_cc:
            f1_status = "BLOCK"
        else:
            f1_status = "PASS"

        # F2 decision
        f2_status = ("BLOCK" if is_cc else "WARN") if used_f2_dual_hash else "PASS"
        f2_note = (
            "Detected prohibited dual-hash DFA / fused dicodon DP execution evidence."
            if used_f2_dual_hash
            else "Zero DFA dual-hash-table backpointer DP or fused free-energy+dicodon DP recursion."
        )

        # F3 decision
        f3_hit = used_dfa_cfg or used_additive_mfe_reg
        f3_status = ("BLOCK" if is_cc else "WARN") if f3_hit else "PASS"
        f3_note = (
            f"uses_dfa_cfg_intersection={used_dfa_cfg}; uses_additive_mfe_reg={used_additive_mfe_reg}."
        )

        # F4A CureVac decision
        eff_cv = (
            max(max_curevac_any_c or 0.0, max_curevac_no_higher_g or 0.0)
            if (max_curevac_any_c is not None or max_curevac_no_higher_g is not None)
            else None
        )
        if curevac_error_note is not None:
            f4a_status = "BLOCK"
            f4a_note = curevac_error_note
        elif eff_cv is None:
            f4a_status = "PASS"
            f4a_note = f"input_origin_type={input_origin_type}; zero cytosine-maximization objective."
        elif eff_cv >= 0.65 and is_cc:
            f4a_status = "BLOCK"
            f4a_note = (
                f"Wild-type C-optimizable codon conversion {eff_cv:.3f} >= 0.65 counsel gate "
                "(US10898584B2 / US12533422B2 / divisional US20260284227A1)."
            )
        elif eff_cv >= 0.50:
            f4a_status = "WARN"
            f4a_note = f"Wild-type C-optimizable codon conversion {eff_cv:.3f} in [0.50, 0.65) warning band (<0.70 claim threshold)."
        else:
            f4a_status = "PASS"
            f4a_note = f"Wild-type C-optimizable codon conversion {eff_cv:.3f} < 0.50 engineering ceiling."

        # F4B ASU decision
        f4b_status = ("BLOCK" if is_cc else "WARN") if used_f4b_weighted_or_argmin else "PASS"
        f4b_note = (
            "Detected prohibited weighted-random repair or motif-count argmin execution evidence."
            if used_f4b_weighted_or_argmin
            else "Constructive constraint satisfaction + deterministic motif remediation + continuous Pareto ranking (zero weighted-random repair, zero motif-count argmin)."
        )

        # F4C Leibniz Hannover decision
        f4c_status = ("BLOCK" if is_cc else "WARN") if used_f4c_gam_or_ppt else "PASS"
        f4c_note = (
            "Detected prohibited boosted-GAM / protein_per_time execution evidence."
            if used_f4c_gam_or_ppt
            else "Zero boosted GAM (mboost) scorer and zero protein_per_time feature across EP/US."
        )

        obj_mode = "pareto_non_dominated" if obj_ids else "hard_gate_feasibility_only"
        boundary_checks = {
            "BAIDU_CN119265214B_STAGED_UTR5_NCDS_PARTITION_OPT": {
                "status": f1_status,
                "evidence_note": (
                    f"region_partition_strategy={region_strategy}; uses_two_zone_partition={used_partition_ev}; "
                    f"objective_mode={obj_mode}; has_axis_outside_tie_cai_mfe={has_outside_triple}."
                ),
            },
            "BAIDU_CN118038986B_DFA_DUAL_HASH_DICODON_DP": {
                "status": f2_status,
                "evidence_note": f2_note,
            },
            "BAIDU_USA_LINEARDESIGN_DFA_CFG_LATTICE_AND_ADDITIVE_MFE_REG": {
                "status": f3_status,
                "evidence_note": f3_note,
            },
            "CUREVAC_US10898584B2_FAMILY_SYNONYMOUS_CYTOSINE_THRESHOLD": {
                "status": f4a_status,
                "evidence_note": f4a_note,
            },
            "ASU_US12706176B1_RANDOM_SYNONYMOUS_SAMPLING_PROBLEM_COUNT_ARGMIN": {
                "status": f4b_status,
                "evidence_note": f4b_note,
            },
            "LEIBNIZ_HANNOVER_US11848074B2_EP3551758B1_BOOSTED_GAM_PROTEIN_PER_TIME": {
                "status": f4c_status,
                "evidence_note": f4c_note,
            },
        }
        # Consume affirmative event names directly from the loaded, validated policy.
        # Aliases and measured thresholds above retain their more specific evidence.
        observed = {*ops_low, *comps_low, *(m.lower() for m in obj_ids)}
        for bid, patterns in self.events_by_boundary.items():
            matched = sorted(patterns & observed)
            if matched:
                boundary_checks[bid] = {
                    "status": "BLOCK" if is_cc else "WARN",
                    "evidence_note": "Observed prohibited execution event: " + ", ".join(matched),
                }
        dep_manifest = []
        for c in invoked_components:
            gpl_cnt, sha_pin = _count_recursive_gpl_dependencies(
                c.name, c.code_license, c.weights_license, seed_dir=self._seed_dir
            )
            if sha_pin is None:
                raise ValueError(f"No verified receipt asset for invoked component {c.name!r}")
            dep_manifest.append(
                {
                    "asset_id": c.name,
                    "version": c.version,
                    "role": c.role,
                    "code_license": c.code_license,
                    "weights_license": c.weights_license,
                    "data_license": c.data_license,
                    "license_track": c.license_track,
                    "transitive_gpl_count": gpl_cnt,
                    "sha256": sha_pin,
                }
            )
        return MethodFTOAuditRecord(
            run_id=run_id,
            timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            commercial_tier="commercial_clean" if is_cc else "research_only",
            engine_git_commit=engine_git_commit,
            optimization_mode=opt_mode,  # type: ignore[arg-type]
            uses_two_zone_initiation_body_partition=used_partition_ev,
            uses_dfa_cfg_intersection=used_dfa_cfg,
            sequence_fto_screen_applied=bool(seq_fto_receipt.get("screen_applied")),
            method_fto_guard_version=f"{self.registry_id}@{self.version}",
            certifies_100_percent_fto=False,
            counsel_fto_signoff_required=True,
            target_deployment_jurisdictions=list(target_jurisdictions or self.jurisdictions),
            input_origin_type=input_origin_type,  # type: ignore[arg-type]
            wildtype_c_optimizable_replacement_fraction=max_curevac_any_c,
            wildtype_c_optimizable_no_higher_g_fraction=max_curevac_no_higher_g,
            region_partition_strategy=region_strategy,
            optimization_schedule_and_operators=executed_operators,
            objective_formulation={
                "mode": obj_mode,
                "objective_metric_ids": obj_ids,
                "has_axis_outside_tie_cai_mfe": has_outside_triple,
                "uses_additive_mfe_plus_weighted_regularizer_scalar": used_additive_mfe_reg,
                "uses_boosted_gam_or_protein_per_time": used_f4c_gam_or_ppt,
            },
            three_layer_dependency_manifest=dep_manifest,
            method_fto_boundary_checks=boundary_checks,
            sequence_fto_guard_v2_receipt=seq_fto_receipt,
        )

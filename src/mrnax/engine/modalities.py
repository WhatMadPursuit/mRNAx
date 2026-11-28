"""Multi-Modality Construct Builders & Evaluators (`linear_mrna`, `circrna`, `sarna`).

Driven by `registry_seed/modality_scaffolds_v1.json` and `registry_seed/tissue_mirna_atlas_v1.json`.
Strictly enforces Clean-Room track separation (`commercial_clean` vs `research_max` / `sota_open`).
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
import random
import re
from typing import Any, get_args

from Bio.Seq import Seq

from mrnax.constraints.registry import ScopedConstraintEngine, default_m1_constraints
from mrnax.engine.lineardesign_adapter import (
    DEFAULT_POLISH_ENZYMES,
    compute_5prime_initiation_openness,
    compute_human_cai,
    polish_cds_5prime_and_constraints,
    sweep_lineardesign_candidates,
)
from mrnax.engine.profiles import resolve_default_seed_dir
from mrnax.engine.stage0_baseline import (
    AA_TO_SYNONYMOUS_RNA_CODONS,
    generate_dnachisel_cds,
    generate_random_synonymous_cds,
    propose_non_partitioned_bicodon_cds,
)
from mrnax.engine.structure_profile import analyze_rna_structure
from mrnax.engine.surrogates import (
    CommercialCleanSurrogateV2,
    extract_circrna_ires_interval,
    has_surrogate_v2_artifacts,
    load_surrogate_jury,
)
from mrnax.engine.thermodynamics import compute_rnaup_mirna_interaction
from mrnax.schemas.common import ComplianceProfile, LicenseTrack, UridineChemistry
from mrnax.schemas.construct_bundle import (
    ChemistryTrackSpec,
    ConstructBundle,
    FeatureAnnotation,
    ProteinTargetSpec,
    RNAMoleculeSpec,
    SegmentSpan,
)
from mrnax.schemas.design_intent import ConstraintSpec

_WAVE1A_FALLBACK_DIR = Path(
    "/Users/alex/.gemini/antigravity/brain/4db4d24c-55e0-4968-869d-a93a1c9c7b7b/scratch/wave1_data_rules"
)

_CLEAN_MIRNA_SPACERS = ("CAAACAAACA", "CUAACUAACU", "CAACCAAUCA")

_TISSUE_ALIAS_MAP: dict[str, str] = {
    "liver": "liver",
    "liver_hepatocyte": "liver",
    "hepatocyte": "liver",
    "myeloid": "myeloid_apc",
    "myeloid_apc": "myeloid_apc",
    "myeloid_monocyte_macrophage": "myeloid_apc",
    "apc": "myeloid_apc",
    "spleen": "myeloid_apc",
    "muscle": "muscle",
    "skeletal_muscle": "muscle",
    "cardiac_heart": "muscle",
    "heart": "muscle",
    "cns": "cns",
    "brain": "cns",
    "brain_neural": "cns",
    "endothelial": "endothelial",
    "lymphoid": "lymphoid_t_cell",
    "lymphoid_t_cell": "lymphoid_t_cell",
    "t_cell": "lymphoid_t_cell",
    "lung": "lung_epithelial",
    "lung_epithelial": "lung_epithelial",
}

_FANTOM5_COMPARTMENT_ALIAS: dict[str, str] = {
    "liver": "liver_hepatocyte",
    "liver_hepatocyte": "liver_hepatocyte",
    "hepatocyte": "liver_hepatocyte",
    "myeloid": "myeloid_monocyte_macrophage",
    "myeloid_apc": "myeloid_monocyte_macrophage",
    "myeloid_monocyte_macrophage": "myeloid_monocyte_macrophage",
    "muscle": "skeletal_muscle",
    "skeletal_muscle": "skeletal_muscle",
    "cardiac_heart": "cardiac_heart",
    "heart": "cardiac_heart",
    "cns": "brain_neural",
    "brain": "brain_neural",
    "brain_neural": "brain_neural",
    "endothelial": "endothelial",
    "t_cell": "t_cell",
    "lymphoid_t_cell": "t_cell",
    "lung": "lung_epithelial",
    "lung_epithelial": "lung_epithelial",
    "kidney": "kidney_renal",
    "kidney_renal": "kidney_renal",
}

_IRES_SCAFFOLD_BY_KEY: dict[str, str] = {
    "EV-B107": "SCAF-CIRC-IRES-EVB107-WT",
    "EVB107": "SCAF-CIRC-IRES-EVB107-WT",
    "HRV-B3": "SCAF-CIRC-IRES-HRVB3-WT",
    "HRVB3": "SCAF-CIRC-IRES-HRVB3-WT",
    "EV-A71": "SCAF-CIRC-IRES-EVA71-WT",
    "EVA71": "SCAF-CIRC-IRES-EVA71-WT",
    "CVB3": "SCAF-CIRC-IRES-CVB3-WT",
    "ICVB3": "SCAF-CIRC-IRES-CVB3-WT",
    "EMCV": "SCAF-CIRC-IRES-EMCV-WT",
}

_PIE_SCAFFOLD_BY_KEY: dict[str, str] = {
    "ANABAENA": "SCAF-CIRC-PIE-ANABAENA-TRNALEU-GROUP1",
    "SCAF-CIRC-PIE-ANABAENA-TRNALEU-GROUP1": "SCAF-CIRC-PIE-ANABAENA-TRNALEU-GROUP1",
    "T4_TD": "SCAF-CIRC-PIE-T4-TD-GROUP1",
    "T4TD": "SCAF-CIRC-PIE-T4-TD-GROUP1",
    "T4": "SCAF-CIRC-PIE-T4-TD-GROUP1",
    "SCAF-CIRC-PIE-T4-TD-GROUP1": "SCAF-CIRC-PIE-T4-TD-GROUP1",
}

_SARNA_SCAFFOLD_BY_KEY: dict[str, str] = {
    "VEEV": "SCAF-SARNA-VEEV-TC83-CIS-REPLICON",
    "VEEV_TC83": "SCAF-SARNA-VEEV-TC83-CIS-REPLICON",
    "VEEV-TC83": "SCAF-SARNA-VEEV-TC83-CIS-REPLICON",
    "TC83": "SCAF-SARNA-VEEV-TC83-CIS-REPLICON",
    "SCAF-SARNA-VEEV-TC83-CIS-REPLICON": "SCAF-SARNA-VEEV-TC83-CIS-REPLICON",
    "VEEV_WT": "SCAF-SARNA-VEEV-WT-REFSEQ-DESCRIPTOR",
    "SCAF-SARNA-VEEV-WT-REFSEQ-DESCRIPTOR": "SCAF-SARNA-VEEV-WT-REFSEQ-DESCRIPTOR",
    "SINV": "SCAF-SARNA-SINV-REFSEQ-CIS-REPLICON",
    "SINDBIS": "SCAF-SARNA-SINV-REFSEQ-CIS-REPLICON",
    "SCAF-SARNA-SINV-REFSEQ-CIS-REPLICON": "SCAF-SARNA-SINV-REFSEQ-CIS-REPLICON",
}


def _try_vienna() -> Any | None:
    try:
        import RNA  # type: ignore[import-untyped]

        return RNA
    except Exception:
        return None


@functools.lru_cache(maxsize=4)
def _cached_json(path_str: str) -> dict[str, Any]:
    return json.loads(Path(path_str).read_text(encoding="utf-8"))


def load_modality_scaffolds(seed_dir: Path | None = None) -> dict[str, Any]:
    eff = Path(seed_dir) if seed_dir is not None else resolve_default_seed_dir()
    for p in (eff / "modality_scaffolds_v1.json", _WAVE1A_FALLBACK_DIR / "modality_scaffolds_v1.json"):
        if p.is_file():
            return _cached_json(str(p.resolve()))
    raise FileNotFoundError(f"modality_scaffolds_v1.json missing in {eff}")


def load_tissue_mirna_atlas(seed_dir: Path | None = None) -> dict[str, Any]:
    eff = Path(seed_dir) if seed_dir is not None else resolve_default_seed_dir()
    for p in (eff / "tissue_mirna_atlas_v1.json", _WAVE1A_FALLBACK_DIR / "tissue_mirna_atlas_v1.json"):
        if p.is_file():
            return _cached_json(str(p.resolve()))
    raise FileNotFoundError(f"tissue_mirna_atlas_v1.json missing in {eff}")


_LICENSE_TRACK_ORDER: tuple[LicenseTrack, ...] = get_args(LicenseTrack)


def _strictest_license_track(*tiers: str) -> LicenseTrack:
    valid: list[LicenseTrack] = []
    for raw in tiers:
        if raw not in _LICENSE_TRACK_ORDER:
            raise ValueError(f"Unsupported license_track '{raw}'")
        valid.append(raw)  # type: ignore[arg-type]
    if not valid:
        return "commercial_clean"
    return max(valid, key=_LICENSE_TRACK_ORDER.index)


def _validate_scaffold_component(
    scaf: dict[str, Any],
    *,
    expected_modality: str,
    allowed_categories: tuple[str, ...],
    compliance_track: ComplianceProfile,
    allow_quarantine_reference: bool = False,
) -> LicenseTrack:
    scaf_id = str(scaf.get("scaffold_id", "<unknown>"))
    actual_mod = str(scaf.get("modality", ""))
    actual_cat = str(scaf.get("scaffold_category", ""))
    if actual_mod != expected_modality:
        raise ValueError(
            f"Scaffold '{scaf_id}' modality mismatch: expected '{expected_modality}', got '{actual_mod}'"
        )
    if allowed_categories and actual_cat not in allowed_categories:
        raise ValueError(
            f"Scaffold '{scaf_id}' category mismatch: expected one of {allowed_categories}, got '{actual_cat}'"
        )
    lic_obj = scaf.get("license")
    raw_tier = lic_obj.get("license_track") if isinstance(lic_obj, dict) else None
    if not isinstance(raw_tier, str) or raw_tier not in _LICENSE_TRACK_ORDER:
        raise PermissionError(f"Scaffold '{scaf_id}' has missing or invalid license_track '{raw_tier}'")
    tier: LicenseTrack = raw_tier  # type: ignore[assignment]
    fto_obj = scaf.get("patent_fto_tag")
    fto_blocked_cc = isinstance(fto_obj, dict) and fto_obj.get("commercial_clean_allowed") is False
    if tier == "quarantine_reference" and not allow_quarantine_reference:
        raise PermissionError(
            f"Scaffold '{scaf_id}' has license_track='quarantine_reference' and cannot be assembled into active designs."
        )
    if compliance_track == "commercial_clean" and (tier != "commercial_clean" or fto_blocked_cc):
        raise PermissionError(
            f"Scaffold '{scaf_id}' (license_track='{tier}') is blocked under compliance_track='commercial_clean'."
        )
    return tier


def _scaffold_by_id(scaffolds_doc: dict[str, Any], scaffold_id: str) -> dict[str, Any]:
    for item in scaffolds_doc.get("scaffolds", []):
        if item.get("scaffold_id") == scaffold_id:
            return item
    raise KeyError(f"Unknown scaffold_id '{scaffold_id}' in modality_scaffolds_v1.json")


def select_atlas_mirna_for_detargeting(
    detarget_tissues: list[str],
    protected_tissues: list[str] | None = None,
    *,
    seed_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Select verified tissue-specific miRNAs from `tissue_mirna_atlas_v1.json` avoiding protected tissues."""
    atlas = load_tissue_mirna_atlas(seed_dir)
    entries = {m["mirna_id"]: m for m in atlas.get("mirna_entries", [])}
    groups = atlas.get("tissue_groups", {})
    prot_list = [_FANTOM5_COMPARTMENT_ALIAS.get(t.strip().lower(), t.strip().lower()) for t in (protected_tissues or [])]

    selected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for raw_t in detarget_tissues:
        grp_key = _TISSUE_ALIAS_MAP.get(raw_t.strip().lower(), raw_t.strip().lower())
        cand_ids = list(groups.get(grp_key, []))
        if not cand_ids and raw_t in entries:
            cand_ids = [raw_t]
        chosen: dict[str, Any] | None = None
        for cid in cand_ids:
            entry = entries.get(cid)
            if entry is None:
                continue
            cpm_map = entry.get("fantom5_expression", {}).get("mean_cpm_by_tissue", {})
            conflict = False
            for pt in prot_list:
                val = cpm_map.get(pt)
                if isinstance(val, (int, float)) and float(val) > 500.0:
                    conflict = True
                    break
            if not conflict:
                chosen = entry
                break
        if chosen is None:
            raise ValueError(
                f"No safe miRNA in tissue_mirna_atlas_v1.json for detarget='{raw_t}' sparing protected={protected_tissues}"
            )
        if chosen["mirna_id"] not in seen_ids:
            seen_ids.add(chosen["mirna_id"])
            selected.append(chosen)
    return selected


def _inject_atlas_mirna_cassette(
    base_utr3: str,
    mirna_entries: list[dict[str, Any]],
    *,
    copies_per_mirna: int = 2,
) -> tuple[str, list[FeatureAnnotation]]:
    if not mirna_entries:
        return base_utr3, []
    lead = max(16, min(24, len(base_utr3)))
    prefix = base_utr3[:lead]
    suffix = base_utr3[lead:]
    cursor = len(prefix)
    parts: list[str] = [prefix]
    anns: list[FeatureAnnotation] = []
    spacer_idx = 0
    for entry in mirna_entries:
        site_rna = str(entry["mrna_full_complementary_ago2_cleavage_site_rna"]).upper()
        mir_id = str(entry["mirna_id"])
        for cp in range(copies_per_mirna):
            if spacer_idx > 0:
                sp = _CLEAN_MIRNA_SPACERS[spacer_idx % len(_CLEAN_MIRNA_SPACERS)]
                parts.append(sp)
                cursor += len(sp)
            s_lo = cursor
            s_hi = s_lo + len(site_rna)
            parts.append(site_rna)
            anns.append(
                FeatureAnnotation(
                    feature_id=f"mirna_site_{mir_id.replace('-', '_')}_{cp + 1}",
                    feature_key="misc_binding",
                    intervals=[(s_lo, s_hi)],
                    qualifiers={
                        "bound_moiety": [mir_id],
                        "mrnax_origin_source": [f"tissue_mirna_atlas_v1:{mir_id}"],
                        "mrnax_license_track": [str(entry.get("license_track", "commercial_clean"))],
                    },
                )
            )
            cursor = s_hi
            spacer_idx += 1
    parts.append(_CLEAN_MIRNA_SPACERS[0])
    parts.append(suffix)
    return "".join(parts), anns


def _optimize_cds_for_modality(
    protein_seq: str,
    *,
    utr5_rna: str,
    stop_rna: str,
    utr3_rna: str,
    compliance_track: str,
    chemistry: UridineChemistry,
    use_lineardesign: bool = True,
    lambda_val: float = 1.0,
    extra_forbidden_motifs: tuple[str, ...] = (),
    constraints: list[ConstraintSpec] | None = None,
    seed: int = 42,
    baseline_v20_mode: bool = False,
) -> tuple[str, str, dict[str, Any]]:
    """Optimize CDS respecting `commercial_clean` isolation (`LinearDesign+Polisher` vs clean bicodon/v2.0)."""
    if baseline_v20_mode:
        cds_dc = generate_dnachisel_cds(
            protein_seq,
            constraints or default_m1_constraints(),
            seed=seed,
            chemistry=chemistry,
        )
        return (
            cds_dc,
            "mrnax:v20_dnachisel_baseline",
            {"generator": "v20_dnachisel_baseline", "backend_used": "dnachisel_cds", "license_track": "commercial_clean"},
        )

    if compliance_track != "commercial_clean" and use_lineardesign:
        cands = sweep_lineardesign_candidates(
            protein_seq,
            lambdas=(lambda_val,),
            utr5_rna=utr5_rna,
            stop_rna=stop_rna,
            utr3_rna=utr3_rna,
            compliance_track=compliance_track,
            chemistry=chemistry,
            eliminate_uuuu=(chemistry in {"m1psi", "pseudouridine"} or True),
            extra_forbidden_motifs=extra_forbidden_motifs,
            constraints=constraints,
            seed=seed,
        )
        best = cands[0]
        origin_tag = (
            f"mrnax:lineardesign_polisher_lam{lambda_val:g}"
            if best.backend_used == "lineardesign_cpp_binary"
            else f"mrnax:guided_bicodon_vienna_fallback_lam{lambda_val:g}"
        )
        info = best.to_dict()
        info["license_track"] = "needs_vendor_license"
        return (
            best.polished_cds_rna,
            origin_tag,
            info,
        )

    # Commercial-clean non-partitioned full-CDS bicodon optimizer
    rng = random.Random(seed)
    init_cds = generate_dnachisel_cds(
        protein_seq,
        constraints or default_m1_constraints(),
        seed=seed,
        chemistry=chemistry,
    )
    trials = [init_cds]
    cur = init_cds
    for step_i in range(3):
        cur = propose_non_partitioned_bicodon_cds(
            cur,
            protein_seq,
            rng,
            chemistry=chemistry,
            constraints=constraints,
            utr5_tail=utr5_rna,
            stop_head=stop_rna + utr3_rna[:9],
            target_gc=(0.52, 0.55, 0.57)[step_i],
            mutation_prob=0.65,
        )
        trials.append(cur)
    chosen = trials[-1]
    return (
        chosen,
        "mrnax:clean_bicodon_optimizer",
        {"generator": "clean_bicodon_optimizer", "backend_used": "clean_bicodon_optimizer", "license_track": "commercial_clean"},
    )


# ============================================================================
# 1. Linear mRNA (Single-Chain & Multi-Chain + Tissue miRNA Detargeting)
# ============================================================================


def build_linear_mrna_bundle(
    protein_targets: list[ProteinTargetSpec] | list[str],
    *,
    bundle_id: str = "bnd_v25_linear",
    compliance_track: ComplianceProfile = "research_max",
    uridine_chemistry: UridineChemistry = "m1psi",
    utr5_scaffold_id: str = "SCAF-LIN-5UTR-LGALS1-GENCODE49",
    utr3_scaffold_id: str = "SCAF-LIN-3UTR-RPS21-GENCODE49",
    polya_scaffold_id: str = "SCAF-LIN-POLYA-HOMO-A100-CLEAN",
    detarget_tissues: list[str] | None = None,
    protected_tissues: list[str] | None = None,
    mirna_copies: int = 2,
    use_lineardesign: bool = True,
    lambda_val: float = 1.0,
    baseline_v20_mode: bool = False,
    seed: int = 42,
    seed_dir: Path | None = None,
) -> ConstructBundle:
    """Build single- or multi-chain `linear_mrna` ConstructBundle from `modality_scaffolds_v1.json`."""
    scaffolds_doc = load_modality_scaffolds(seed_dir)
    u5_scaf = _scaffold_by_id(scaffolds_doc, utr5_scaffold_id)
    u3_scaf = _scaffold_by_id(scaffolds_doc, utr3_scaffold_id)
    pa_scaf = _scaffold_by_id(scaffolds_doc, polya_scaffold_id)
    kozak_scaf = _scaffold_by_id(scaffolds_doc, "SCAF-LIN-KOZAK-VERTEBRATE-STRONG-GCCACC")

    u5_tier_base = _validate_scaffold_component(
        u5_scaf,
        expected_modality="linear_mrna",
        allowed_categories=("five_prime_utr",),
        compliance_track=compliance_track,
    )
    kozak_tier = _validate_scaffold_component(
        kozak_scaf,
        expected_modality="linear_mrna",
        allowed_categories=("kozak_template", "kozak_consensus", "five_prime_utr"),
        compliance_track=compliance_track,
    )
    u3_tier_base = _validate_scaffold_component(
        u3_scaf,
        expected_modality="linear_mrna",
        allowed_categories=("three_prime_utr",),
        compliance_track=compliance_track,
    )
    pa_tier = _validate_scaffold_component(
        pa_scaf,
        expected_modality="linear_mrna",
        allowed_categories=("poly_a_tail",),
        compliance_track=compliance_track,
    )

    u5_base = str(u5_scaf["sequence_rna"]).upper()
    if not u5_base.endswith("ACC"):
        u5_seq = u5_base + str(kozak_scaf["sequence_rna"]).upper()[:6]
        u5_tier = _strictest_license_track(u5_tier_base, kozak_tier)
    else:
        u5_seq = u5_base
        u5_tier = u5_tier_base
    if not u5_seq.startswith("AG"):
        u5_seq = "AG" + u5_seq

    u3_base = str(u3_scaf["sequence_rna"]).upper()
    mir_entries = (
        select_atlas_mirna_for_detargeting(
            detarget_tissues, protected_tissues, seed_dir=seed_dir
        )
        if detarget_tissues
        else []
    )
    for mir_e in mir_entries:
        mir_tier = str(mir_e.get("license_track", "commercial_clean"))
        if compliance_track == "commercial_clean" and mir_tier != "commercial_clean":
            raise PermissionError(
                f"miRNA '{mir_e.get('mirna_id')}' (license_track='{mir_tier}') is blocked in commercial_clean"
            )
    u3_seq, u3_anns = _inject_atlas_mirna_cassette(
        u3_base, mir_entries, copies_per_mirna=mirna_copies
    )
    u3_tier = _strictest_license_track(
        u3_tier_base,
        *(str(m.get("license_track", "commercial_clean")) for m in mir_entries),
    )
    polya_seq = str(pa_scaf["sequence_rna"]).upper()
    stop_seq = "UGA"

    norm_targets: list[ProteinTargetSpec] = [
        ProteinTargetSpec(
            protein_sequence=pt,
            transl_table=1,
            accession_or_source=f"target_chain_{idx + 1}",
            confirmation_status="confirmed",
        )
        if isinstance(pt, str)
        else pt
        for idx, pt in enumerate(protein_targets)
    ]

    constraints = default_m1_constraints(
        max_homopolymer_run=5 if not baseline_v20_mode else 7,
        restriction_enzymes=list(DEFAULT_POLISH_ENZYMES) if not baseline_v20_mode else ["BspQI"],
    )

    molecules: list[RNAMoleculeSpec] = []
    for idx, tgt in enumerate(norm_targets):
        mol_id = "mol_primary" if len(norm_targets) == 1 else f"chain_{idx + 1:02d}"
        cds_seq, cds_origin, cds_meta = _optimize_cds_for_modality(
            tgt.protein_sequence,
            utr5_rna=u5_seq,
            stop_rna=stop_seq,
            utr3_rna=u3_seq,
            compliance_track=compliance_track,
            chemistry=uridine_chemistry,
            use_lineardesign=use_lineardesign,
            lambda_val=lambda_val,
            constraints=constraints,
            seed=seed + 101 * idx,
            baseline_v20_mode=baseline_v20_mode,
        )
        cds_tier = _strictest_license_track(str(cds_meta.get("license_track", "commercial_clean")))
        c1 = len(u5_seq)
        c2 = c1 + len(cds_seq)
        c3 = c2 + len(stop_seq)
        c4 = c3 + len(u3_seq)
        c5 = c4 + len(polya_seq)
        full_seq = u5_seq + cds_seq + stop_seq + u3_seq + polya_seq

        anns = [
            FeatureAnnotation(
                feature_id="kozak_tis",
                feature_key="regulatory",
                intervals=[(max(0, c1 - 6), min(c5, c1 + 4))],
                qualifiers={"regulatory_class": ["ribosome_binding_site"]},
            ),
            *[
                a.model_copy_validated(intervals=[(c3 + s, c3 + e) for s, e in a.intervals])
                for a in u3_anns
            ],
        ]
        molecules.append(
            RNAMoleculeSpec(
                molecule_id=mol_id,
                modality="linear_mrna",
                topology="linear",
                chemistry_track=ChemistryTrackSpec(
                    uridine_chemistry=uridine_chemistry,
                    cap_chemistry="cap1_cleancap_ag",
                    source="user_declared",
                ),
                effective_translation_target=tgt,
                sequence=full_seq,
                segments=[
                    SegmentSpan(
                        segment_type="five_prime_utr",
                        start=0,
                        end=c1,
                        component_id=u5_scaf["scaffold_id"],
                        origin_source=f"modality_scaffolds_v1:{u5_scaf['scaffold_id']}",
                        license_track=u5_tier,
                        action_taken="selected_from_library",
                    ),
                    SegmentSpan(
                        segment_type="cds",
                        start=c1,
                        end=c2,
                        component_id=None,
                        origin_source=cds_origin,
                        license_track=cds_tier,
                        action_taken="generated_novo",
                    ),
                    SegmentSpan(
                        segment_type="stop_cassette",
                        start=c2,
                        end=c3,
                        component_id=None,
                        origin_source="mrnax:standard_stop",
                        license_track="commercial_clean",
                        action_taken="locked_verbatim",
                    ),
                    SegmentSpan(
                        segment_type="three_prime_utr",
                        start=c3,
                        end=c4,
                        component_id=u3_scaf["scaffold_id"],
                        origin_source=f"modality_scaffolds_v1:{u3_scaf['scaffold_id']}",
                        license_track=u3_tier,
                        action_taken="selected_from_library",
                    ),
                    SegmentSpan(
                        segment_type="poly_a",
                        start=c4,
                        end=c5,
                        component_id=pa_scaf["scaffold_id"],
                        origin_source=f"modality_scaffolds_v1:{pa_scaf['scaffold_id']}",
                        license_track=pa_tier,
                        action_taken="locked_verbatim",
                    ),
                ],
                annotations=anns,
            )
        )

    co_mode = "lnp_co_encapsulation" if len(molecules) > 1 else "single_monocistronic"
    return ConstructBundle.create(
        bundle_id=bundle_id,
        parent_bundle_id=None,
        derivation_action="de_novo_design",
        target_stoichiometry={m.molecule_id: 1.0 for m in molecules},
        co_expression_mode=co_mode,
        molecules=molecules,
    )


def evaluate_linear_mrna_bundle(
    bundle: ConstructBundle,
    *,
    compliance_track: ComplianceProfile = "research_max",
    seed_dir: Path | None = None,
) -> dict[str, Any]:
    eff_seed = Path(seed_dir) if seed_dir is not None else resolve_default_seed_dir()
    engine = ScopedConstraintEngine.from_seed_dir(eff_seed)
    gate_outcome = engine.evaluate_bundle(
        bundle,
        default_m1_constraints(restriction_enzymes=["BspQI"]),
        compliance_track=compliance_track,
    )
    atlas = load_tissue_mirna_atlas(eff_seed)
    atlas_by_id = {m["mirna_id"]: m for m in atlas.get("mirna_entries", [])}
    allow_vienna = compliance_track != "commercial_clean"
    rna_mod = _try_vienna() if allow_vienna else None

    chain_reports: list[dict[str, Any]] = []
    for mol in bundle.molecules:
        u5_seg = mol.get_segment("five_prime_utr")
        cds_seg = mol.get_segment("cds")
        u3_seg = mol.get_segment("three_prime_utr")
        prof = analyze_rna_structure(
            mol.sequence,
            five_prime_end=u5_seg.end if u5_seg else None,
            cds_start=cds_seg.start if cds_seg else None,
            cds_end=cds_seg.end if cds_seg else None,
            three_prime_start=u3_seg.start if u3_seg else None,
            three_prime_end=u3_seg.end if u3_seg else None,
            modality="linear_mrna",
            compliance_track=compliance_track,
            seed_dir=eff_seed,
        )
        mir_interactions: list[dict[str, Any]] = []
        for ann in mol.annotations:
            if ann.feature_key == "misc_binding" and ann.qualifiers.get("bound_moiety"):
                mid = ann.qualifiers["bound_moiety"][0]
                entry = atlas_by_id.get(mid)
                if entry is not None and rna_mod is not None:
                    s_lo, s_hi = ann.intervals[0]
                    ctx_lo = max(0, s_lo - 50)
                    ctx_hi = min(len(mol.sequence), s_hi + 35)
                    acc = compute_rnaup_mirna_interaction(
                        mol.sequence[ctx_lo:ctx_hi],
                        s_lo - ctx_lo,
                        s_hi - s_lo,
                        str(entry["mature_sequence_rna"]),
                    )
                    mir_interactions.append(
                        {
                            "mirna_id": mid,
                            "dg_total_kcal_mol": acc.dg_total_kcal_mol,
                            "seed_8mer_unpaired_prob": acc.seed_8mer_unpaired_prob,
                        }
                    )
        cds_seq = mol.get_segment_sequence("cds")
        cds_dg: float | None = round(float(rna_mod.fold(cds_seq)[1]), 3) if (rna_mod is not None and cds_seq) else None
        full_dg: float | None = prof["mfe_kcal_mol"] if rna_mod is not None else None
        chain_reports.append(
            {
                "molecule_id": mol.molecule_id,
                "aa_exact_match": mol.translate_cds_protein() == mol.effective_translation_target.protein_sequence,
                "length_nt": prof["length_nt"],
                "cds_mfe_kcal_mol": cds_dg,
                "full_mfe_kcal_mol": full_dg,
                "mfe_scope": "exact_vienna_mfe" if rna_mod is not None else "unavailable_without_approved_thermo_backend",
                "cai": prof["cai"],
                "kozak_unpaired_prob": prof["kozak_unpaired_prob"] if rna_mod is not None else None,
                "aup_mean": prof["aup_mean"] if rna_mod is not None else None,
                "cds_uuuu_count": len(re.findall(r"U{4,}", cds_seq)),
                "mirna_interactions": mir_interactions,
            }
        )

    return {
        "modality": "linear_mrna",
        "chain_count": len(bundle.molecules),
        "passed_all_hard_gates": gate_outcome.passed_all_hard_gates,
        "violation_count": len(gate_outcome.violations),
        "chains": chain_reports,
    }


# ============================================================================
# 2. Circular RNA (`circrna`: PIE Group-I Intron + Spacers + Viral IRES + ORF)
# ============================================================================


def _max_exact_antisense_kmer(seq_a: str, seq_b: str, *, min_k: int = 7, max_k: int = 18) -> int:
    """Find longest exact Watson-Crick reverse-complement substring between `seq_a` and `seq_b`."""
    best = 0
    for k in range(min_k, max_k + 1):
        if len(seq_a) < k or len(seq_b) < k:
            break
        b_kmers = {seq_b[j : j + k] for j in range(len(seq_b) - k + 1)}
        found = False
        for i in range(len(seq_a) - k + 1):
            rc = str(Seq(seq_a[i : i + k]).reverse_complement_rna())
            if rc in b_kmers:
                found = True
                break
        if found:
            best = k
        else:
            break
    return best


def _uncouple_cds_from_ires_synonymously(
    cds_rna: str,
    protein_seq: str,
    ires_rna: str,
    *,
    k_ban: int = 12,
    forbidden_motifs: tuple[str, ...] = (),
) -> str:
    """Eliminate >=12-nt antisense base-pairing invasions between CDS and viral IRES domains IV-VI."""
    ires_core = ires_rna[-480:] if len(ires_rna) > 480 else ires_rna
    banned_rc_kmers = {
        str(Seq(ires_core[i : i + k_ban]).reverse_complement_rna())
        for i in range(max(0, len(ires_core) - k_ban + 1))
    }
    codons = [cds_rna[3 * i : 3 * i + 3] for i in range(len(protein_seq))]
    for _pass in range(4):
        cur_cds = "".join(codons)
        hit_pos = -1
        for i in range(max(0, len(cur_cds) - k_ban + 1)):
            if cur_cds[i : i + k_ban] in banned_rc_kmers:
                hit_pos = i
                break
        if hit_pos < 0:
            break
        c_mid = min(len(codons) - 1, max(1, (hit_pos + k_ban // 2) // 3))
        replaced = False
        for idx in (c_mid, max(1, c_mid - 1), min(len(codons) - 1, c_mid + 1)):
            choices = AA_TO_SYNONYMOUS_RNA_CODONS.get(protein_seq[idx], (codons[idx],))
            for cand_c in choices:
                if cand_c == codons[idx]:
                    continue
                old_c = codons[idx]
                codons[idx] = cand_c
                local_win = "".join(codons[max(0, idx - 5) : min(len(codons), idx + 6)])
                has_ires_rc = any(
                    local_win[j : j + k_ban] in banned_rc_kmers
                    for j in range(max(0, len(local_win) - k_ban + 1))
                )
                has_forb = any(p in local_win for p in forbidden_motifs)
                if not has_ires_rc and not has_forb:
                    replaced = True
                    break
                codons[idx] = old_c
            if replaced:
                break
        if not replaced:
            break
    return "".join(codons)


def build_circrna_bundle(
    protein_target: ProteinTargetSpec | str,
    *,
    bundle_id: str = "bnd_v25_circrna",
    ires_name: str = "HRV-B3",
    pie_backbone: str = "Anabaena",
    spacer5_scaffold_id: str = "SCAF-CIRC-SPACER5-POLYAC-51NT",
    spacer3_scaffold_id: str = "SCAF-CIRC-SPACER3-RANDOM50-50NT",
    compliance_track: ComplianceProfile = "research_max",
    use_lineardesign: bool = True,
    lambda_val: float = 1.0,
    baseline_v20_mode: bool = False,
    seed: int = 42,
    seed_dir: Path | None = None,
) -> ConstructBundle:
    """Build covalently closed `circrna` ConstructBundle (PIE scar + spacers + ranked viral IRES + ORF)."""
    scaffolds_doc = load_modality_scaffolds(seed_dir)
    ires_key = _IRES_SCAFFOLD_BY_KEY.get(ires_name.strip().upper(), ires_name)
    pie_key = _PIE_SCAFFOLD_BY_KEY.get(pie_backbone.strip().upper(), pie_backbone)

    ires_scaf = _scaffold_by_id(scaffolds_doc, ires_key)
    pie_scaf = _scaffold_by_id(scaffolds_doc, pie_key)
    sp5_scaf = _scaffold_by_id(scaffolds_doc, spacer5_scaffold_id)
    sp3_scaf = _scaffold_by_id(scaffolds_doc, spacer3_scaffold_id)

    ires_tier = _validate_scaffold_component(
        ires_scaf,
        expected_modality="circrna",
        allowed_categories=("viral_ires", "ires_element"),
        compliance_track=compliance_track,
    )
    pie_tier = _validate_scaffold_component(
        pie_scaf,
        expected_modality="circrna",
        allowed_categories=("pie_group_i_intron",),
        compliance_track=compliance_track,
    )
    sp5_tier = _validate_scaffold_component(
        sp5_scaf,
        expected_modality="circrna",
        allowed_categories=("circrna_spacer_enhancer", "ires_5prime_spacer", "circ_spacer", "homology_spacer"),
        compliance_track=compliance_track,
    )
    sp3_tier = _validate_scaffold_component(
        sp3_scaf,
        expected_modality="circrna",
        allowed_categories=("circrna_spacer_enhancer", "ires_3prime_spacer", "circ_spacer", "homology_spacer"),
        compliance_track=compliance_track,
    )
    utr5_circ_tier = _strictest_license_track(pie_tier, sp5_tier, ires_tier)
    utr3_circ_tier = _strictest_license_track(sp3_tier, pie_tier)

    if pie_scaf["scaffold_id"] == "SCAF-CIRC-PIE-ANABAENA-TRNALEU-GROUP1":
        scar_27 = str(pie_scaf["verified_ligated_27nt_e2_e1_splicing_scar_rna"]).upper()
        e2_scar = scar_27[:8]
        e1_scar = scar_27[8:]
        up_arm = e2_scar
        dn_arm = e1_scar
    else:
        up_arm = str(pie_scaf["upstream_arm_3prime_intron_plus_e2_rna"]).upper()
        dn_arm = str(pie_scaf["downstream_arm_e1_plus_5prime_intron_rna"]).upper()
        e2_scar = up_arm[-14:]
        e1_scar = dn_arm[:14]

    spacer5_rna = str(sp5_scaf["sequence_rna"]).upper()
    ires_rna = str(ires_scaf["sequence_rna"]).upper()
    spacer3_rna = str(sp3_scaf["sequence_rna"]).upper()
    stop_rna = "UGA"

    utr5_composite = e2_scar + spacer5_rna + ires_rna
    utr3_composite = spacer3_rna + e1_scar

    tgt = (
        ProteinTargetSpec(
            protein_sequence=protein_target,
            transl_table=1,
            accession_or_source="circrna_orf_target",
            confirmation_status="confirmed",
        )
        if isinstance(protein_target, str)
        else protein_target
    )

    cds_seq, cds_origin, cds_meta = _optimize_cds_for_modality(
        tgt.protein_sequence,
        utr5_rna=utr5_composite[-45:],
        stop_rna=stop_rna,
        utr3_rna=utr3_composite[:30],
        compliance_track=compliance_track,
        chemistry="unmodified_u",
        use_lineardesign=use_lineardesign,
        lambda_val=lambda_val,
        seed=seed,
        baseline_v20_mode=baseline_v20_mode,
    )
    used_vendor_polisher = False
    if not baseline_v20_mode:
        cds_seq = _uncouple_cds_from_ires_synonymously(cds_seq, tgt.protein_sequence, ires_rna)
        if compliance_track != "commercial_clean":
            cds_seq, _ = polish_cds_5prime_and_constraints(
                cds_seq,
                tgt.protein_sequence,
                utr5_rna=utr5_composite[-45:],
                stop_rna=stop_rna,
                utr3_rna=utr3_composite[:30],
                compliance_track=compliance_track,
                eliminate_uuuu=True,
                max_homopolymer=5,
            )
            used_vendor_polisher = True

    cds_tier = _strictest_license_track(
        str(cds_meta.get("license_track", "commercial_clean")),
        "needs_vendor_license" if used_vendor_polisher else "commercial_clean",
    )

    c1 = len(utr5_composite)
    c2 = c1 + len(cds_seq)
    c3 = c2 + len(stop_rna)
    c4 = c3 + len(utr3_composite)
    full_circ_seq = utr5_composite + cds_seq + stop_rna + utr3_composite
    precursor_len = len(up_arm) + len(spacer5_rna) + len(ires_rna) + len(cds_seq) + len(stop_rna) + len(spacer3_rna) + len(dn_arm)

    anns = [
        FeatureAnnotation(
            feature_id="pie_exon2_scar",
            feature_key="misc_feature",
            intervals=[(0, len(e2_scar))],
            qualifiers={
                "pie_scaffold_id": [pie_scaf["scaffold_id"]],
                "ivt_precursor_length_nt": [str(precursor_len)],
            },
        ),
        FeatureAnnotation(
            feature_id="circ_5prime_spacer",
            feature_key="misc_feature",
            intervals=[(len(e2_scar), len(e2_scar) + len(spacer5_rna))],
            qualifiers={"scaffold_id": [sp5_scaf["scaffold_id"]]},
        ),
        FeatureAnnotation(
            feature_id=f"viral_ires_{ires_scaf['ires_short_id']}",
            feature_key="regulatory",
            intervals=[(len(e2_scar) + len(spacer5_rna), c1)],
            qualifiers={
                "regulatory_class": ["ribosome_binding_site"],
                "ires_short_id": [str(ires_scaf["ires_short_id"])],
                "scaffold_id": [ires_scaf["scaffold_id"]],
            },
        ),
        FeatureAnnotation(
            feature_id="circ_3prime_spacer",
            feature_key="misc_feature",
            intervals=[(c3, c3 + len(spacer3_rna))],
            qualifiers={"scaffold_id": [sp3_scaf["scaffold_id"]]},
        ),
        FeatureAnnotation(
            feature_id="pie_exon1_scar",
            feature_key="misc_feature",
            intervals=[(c3 + len(spacer3_rna), c4)],
            qualifiers={"pie_scaffold_id": [pie_scaf["scaffold_id"]]},
        ),
    ]

    mol = RNAMoleculeSpec(
        molecule_id="mol_circrna_primary",
        modality="circrna",
        topology="circular",
        chemistry_track=ChemistryTrackSpec(
            uridine_chemistry="unmodified_u",
            cap_chemistry="unknown",
            source="user_declared",
            custom_modification_notes="Cap-independent covalently closed PIE circRNA (unmodified-U required for ribozyme & IRES).",
        ),
        effective_translation_target=tgt,
        sequence=full_circ_seq,
        segments=[
            SegmentSpan(
                segment_type="five_prime_utr",
                start=0,
                end=c1,
                component_id=ires_scaf["scaffold_id"],
                origin_source=f"modality_scaffolds_v1:{pie_scaf['scaffold_id']}+{sp5_scaf['scaffold_id']}+{ires_scaf['scaffold_id']}",
                license_track=utr5_circ_tier,
                action_taken="selected_from_library",
            ),
            SegmentSpan(
                segment_type="cds",
                start=c1,
                end=c2,
                component_id=None,
                origin_source=cds_origin,
                license_track=cds_tier,
                action_taken="generated_novo",
            ),
            SegmentSpan(
                segment_type="stop_cassette",
                start=c2,
                end=c3,
                component_id=None,
                origin_source="mrnax:standard_stop",
                license_track="commercial_clean",
                action_taken="locked_verbatim",
            ),
            SegmentSpan(
                segment_type="three_prime_utr",
                start=c3,
                end=c4,
                component_id=sp3_scaf["scaffold_id"],
                origin_source=f"modality_scaffolds_v1:{sp3_scaf['scaffold_id']}+{pie_scaf['scaffold_id']}",
                license_track=utr3_circ_tier,
                action_taken="selected_from_library",
            ),
        ],
        annotations=anns,
    )
    return ConstructBundle.create(
        bundle_id=bundle_id,
        parent_bundle_id=None,
        derivation_action="de_novo_design",
        target_stoichiometry={mol.molecule_id: 1.0},
        co_expression_mode="single_monocistronic",
        molecules=[mol],
    )


def evaluate_circrna_bundle(
    bundle: ConstructBundle,
    *,
    compliance_track: ComplianceProfile = "research_max",
    seed_dir: Path | None = None,
) -> dict[str, Any]:
    """Evaluate circRNA PIE junction hairpin insulation, IRES-CDS uncoupling, circular folding, and surrogate_v2 IRES proxy."""
    eff_seed = Path(seed_dir) if seed_dir is not None else resolve_default_seed_dir()
    scaffolds_doc = load_modality_scaffolds(eff_seed)
    mol = bundle.molecules[0]
    seq = mol.sequence
    u5_seq = mol.get_segment_sequence("five_prime_utr")
    cds_seq = mol.get_segment_sequence("cds")
    u3_seq = mol.get_segment_sequence("three_prime_utr")

    ires_seq = extract_circrna_ires_interval(mol)
    ires_ann = next(
        (a for a in mol.annotations if a.feature_id.startswith("viral_ires_")),
        None,
    )
    if ires_ann is not None:
        ires_scaf_id = ires_ann.qualifiers.get("scaffold_id", [""])[0]
        ires_meta = _scaffold_by_id(scaffolds_doc, ires_scaf_id) if ires_scaf_id else {}
    else:
        ires_meta = {}

    max_ires_cds_comp = _max_exact_antisense_kmer(cds_seq, ires_seq[-480:])
    junction_wrap = seq[-40:] + seq[:40]
    allow_vienna = compliance_track != "commercial_clean"
    rna_mod = _try_vienna() if allow_vienna else None
    junc_dg: float | None = None
    cds_dg: float | None = None
    circ_dg: float | None = None
    circ_mfe_scope = "unavailable_without_approved_thermo_backend"
    if rna_mod is not None:
        _, j_e = rna_mod.fold(junction_wrap)
        _, c_e = rna_mod.fold(cds_seq)
        junc_dg = round(float(j_e), 3)
        cds_dg = round(float(c_e), 3)
        if len(seq) <= 3200:
            md = rna_mod.md()
            md.circ = 1
            _, cf_e = rna_mod.fold_compound(seq, md).mfe()
            circ_dg = round(float(cf_e), 3)
            circ_mfe_scope = "exact_circular_mfe"
        else:
            circ_dg = None
            circ_mfe_scope = "unavailable_above_3200nt"

    kozak_open: float | None = (
        compute_5prime_initiation_openness(
            cds_seq, utr5_rna=u5_seq[-35:], upstream_nt=15, downstream_nt=30
        )
        if allow_vienna
        else None
    )
    sp5_ann = next((a for a in mol.annotations if a.feature_id == "circ_5prime_spacer"), None)
    sp3_ann = next((a for a in mol.annotations if a.feature_id == "circ_3prime_spacer"), None)
    sp5_len = (sp5_ann.intervals[0][1] - sp5_ann.intervals[0][0]) if sp5_ann else 0
    sp3_len = (sp3_ann.intervals[0][1] - sp3_ann.intervals[0][0]) if sp3_ann else len(u3_seq)

    ires_v2_pred: float | None = None
    notes: list[str] = []
    if has_surrogate_v2_artifacts(eff_seed):
        jury = load_surrogate_jury(eff_seed, prefer_v2=True)
        if isinstance(jury, CommercialCleanSurrogateV2) and jury._circ_ires_bundle is not None:
            ires_v2_pred = round(float(jury.score_circrna_ires([ires_seq])[0]), 6)
            notes.append(
                "surrogate_v2 OOD disclaimer: circrna_ires_translation_v2 was trained on Chen et al. 2023 Fig 5a "
                "(n=111 across 37 viral IRESs; IRES-grouped 5-fold OOD Spearman=0.347 on Colab / 0.221 on local "
                "LightGBM 4.7, p=0.020). Use as a directional rank-order proxy alongside curated viral IRES "
                "experimental fold-change rankings, not as a calibrated expression fold-change."
            )

    return {
        "modality": "circrna",
        "topology": mol.topology,
        "length_nt": len(seq),
        "aa_exact_match": mol.translate_cds_protein() == mol.effective_translation_target.protein_sequence,
        "chemistry_unmodified_u_ok": mol.chemistry_track.uridine_chemistry == "unmodified_u",
        "no_poly_a_tail_ok": mol.get_segment("poly_a") is None,
        "ires_short_id": ires_meta.get("ires_short_id", "custom"),
        "ires_translation_rank": ires_meta.get("translation_rank_order"),
        "ires_fold_vs_icvb3_geomean": ires_meta.get("quantitative_expression_ranking", {}).get(
            "cross_cell_geometric_mean_vs_icvb3"
        ),
        "surrogate_v2_ires_translation_pred": ires_v2_pred,
        "notes": notes,
        "max_ires_cds_complement_nt": max_ires_cds_comp,
        "ires_cargo_uncoupled_ok": max_ires_cds_comp < 12,
        "spacer_insulation_ok": sp5_len >= 35 and sp3_len >= 35,
        "backsplicing_junction_dg_kcal_mol": junc_dg,
        "cds_mfe_kcal_mol": cds_dg,
        "circ_mfe_kcal_mol": circ_dg,
        "circ_mfe_scope": circ_mfe_scope,
        "cai": compute_human_cai(cds_seq),
        "initiation_unpaired_prob": kozak_open,
    }


# ============================================================================
# 3. Self-Amplifying RNA (`sarna`: Alphavirus Replicon CSE/SGP + GOI CDS)
# ============================================================================


def build_sarna_bundle(
    protein_target: ProteinTargetSpec | str,
    *,
    bundle_id: str = "bnd_v25_sarna",
    replicon_backbone: str = "VEEV_TC83",
    compliance_track: ComplianceProfile = "research_max",
    use_lineardesign: bool = True,
    lambda_val: float = 1.0,
    baseline_v20_mode: bool = False,
    seed: int = 42,
    seed_dir: Path | None = None,
) -> ConstructBundle:
    """Build alphavirus `sarna` subgenomic replicon ConstructBundle with locked CSE & 26S SGP elements."""
    scaffolds_doc = load_modality_scaffolds(seed_dir)
    scaf_key = _SARNA_SCAFFOLD_BY_KEY.get(replicon_backbone.strip().upper(), replicon_backbone)
    rep_scaf = _scaffold_by_id(scaffolds_doc, scaf_key)
    rep_tier = _validate_scaffold_component(
        rep_scaf,
        expected_modality="sarna",
        allowed_categories=("alphavirus_replicon_topology", "alphavirus_cis_replicon", "sarna_replicon"),
        compliance_track=compliance_track,
    )
    pa_sarna_scaf = _scaffold_by_id(scaffolds_doc, "SCAF-LIN-POLYA-HOMO-A100-CLEAN")
    pa_sarna_tier = _validate_scaffold_component(
        pa_sarna_scaf,
        expected_modality="linear_mrna",
        allowed_categories=("poly_a_tail",),
        compliance_track=compliance_track,
    )
    cis = rep_scaf["cis_elements_rna"]

    u5_leader = str(cis["five_prime_utr_leader_rna"]).upper()
    cse_51 = str(cis["nsp1_51nt_cse_rna"]).upper()
    sgp_26s = str(cis["subgenomic_26s_promoter_junction_rna"]).upper()
    kozak_leader = "GCCGCCACC"
    utr5_composite = u5_leader + cse_51 + sgp_26s + kozak_leader

    utr3_full = str(cis["three_prime_utr_full_rna"]).upper()
    cse_19 = str(cis["three_prime_19nt_cse_rna"]).upper()
    polya_seq = str(pa_sarna_scaf["sequence_rna"]).upper()
    stop_rna = "UGA"

    tgt = (
        ProteinTargetSpec(
            protein_sequence=protein_target,
            transl_table=1,
            accession_or_source="sarna_goi_target",
            confirmation_status="confirmed",
        )
        if isinstance(protein_target, str)
        else protein_target
    )

    cds_seq, cds_origin, cds_meta = _optimize_cds_for_modality(
        tgt.protein_sequence,
        utr5_rna=utr5_composite[-45:],
        stop_rna=stop_rna,
        utr3_rna=utr3_full[:30],
        compliance_track=compliance_track,
        chemistry="unmodified_u",
        use_lineardesign=use_lineardesign,
        lambda_val=lambda_val,
        seed=seed,
        baseline_v20_mode=baseline_v20_mode,
    )
    cds_tier = _strictest_license_track(str(cds_meta.get("license_track", "commercial_clean")))

    c1 = len(utr5_composite)
    c2 = c1 + len(cds_seq)
    c3 = c2 + len(stop_rna)
    c4 = c3 + len(utr3_full)
    c5 = c4 + len(polya_seq)
    full_seq = utr5_composite + cds_seq + stop_rna + utr3_full + polya_seq

    nsp_len = int(rep_scaf.get("nsp1_4_replicase_region_descriptor", {}).get("length_nt_incl_stop", 7482))
    anns = [
        FeatureAnnotation(
            feature_id="alphavirus_5prime_au_leader",
            feature_key="regulatory",
            intervals=[(0, len(u5_leader))],
            qualifiers={
                "scaffold_id": [rep_scaf["scaffold_id"]],
                "backbone_strain": [str(rep_scaf["backbone_species_strain"])],
            },
        ),
        FeatureAnnotation(
            feature_id="nsp1_51nt_cse_locked",
            feature_key="regulatory",
            intervals=[(len(u5_leader), len(u5_leader) + len(cse_51))],
            qualifiers={
                "regulatory_class": ["replication_regulatory_region"],
                "nsp1_4_reference_length_nt": [str(nsp_len)],
            },
        ),
        FeatureAnnotation(
            feature_id="subgenomic_26s_promoter",
            feature_key="regulatory",
            intervals=[(len(u5_leader) + len(cse_51), len(u5_leader) + len(cse_51) + len(sgp_26s))],
            qualifiers={"regulatory_class": ["promoter"]},
        ),
        FeatureAnnotation(
            feature_id="three_prime_19nt_cse_locked",
            feature_key="regulatory",
            intervals=[(c4 - len(cse_19), c4)],
            qualifiers={"regulatory_class": ["replication_regulatory_region"]},
        ),
    ]

    mol = RNAMoleculeSpec(
        molecule_id="mol_sarna_replicon",
        modality="sarna",
        topology="linear",
        chemistry_track=ChemistryTrackSpec(
            uridine_chemistry="unmodified_u",
            cap_chemistry="cap1_cleancap_ag",
            source="user_declared",
            custom_modification_notes="CleanCap AU (m7GpppAmU) 5'-AU cap-1 required for alphavirus IFIT1 evasion & RdRp replication.",
        ),
        effective_translation_target=tgt,
        sequence=full_seq,
        segments=[
            SegmentSpan(
                segment_type="five_prime_utr",
                start=0,
                end=c1,
                component_id=rep_scaf["scaffold_id"],
                origin_source=f"modality_scaffolds_v1:{rep_scaf['scaffold_id']}",
                license_track=rep_tier,
                action_taken="locked_verbatim",
            ),
            SegmentSpan(
                segment_type="cds",
                start=c1,
                end=c2,
                component_id=None,
                origin_source=cds_origin,
                license_track=cds_tier,
                action_taken="generated_novo",
            ),
            SegmentSpan(
                segment_type="stop_cassette",
                start=c2,
                end=c3,
                component_id=None,
                origin_source="mrnax:standard_stop",
                license_track="commercial_clean",
                action_taken="locked_verbatim",
            ),
            SegmentSpan(
                segment_type="three_prime_utr",
                start=c3,
                end=c4,
                component_id=rep_scaf["scaffold_id"],
                origin_source=f"modality_scaffolds_v1:{rep_scaf['scaffold_id']}",
                license_track=rep_tier,
                action_taken="locked_verbatim",
            ),
            SegmentSpan(
                segment_type="poly_a",
                start=c4,
                end=c5,
                component_id="SCAF-LIN-POLYA-HOMO-A100-CLEAN",
                origin_source="modality_scaffolds_v1:SCAF-LIN-POLYA-HOMO-A100-CLEAN",
                license_track=pa_sarna_tier,
                action_taken="locked_verbatim",
            ),
        ],
        annotations=anns,
    )
    return ConstructBundle.create(
        bundle_id=bundle_id,
        parent_bundle_id=None,
        derivation_action="de_novo_design",
        target_stoichiometry={mol.molecule_id: 1.0},
        co_expression_mode="single_monocistronic",
        molecules=[mol],
    )


def evaluate_sarna_bundle(
    bundle: ConstructBundle,
    *,
    replicon_backbone: str | None = None,
    compliance_track: ComplianceProfile = "research_max",
    seed_dir: Path | None = None,
) -> dict[str, Any]:
    """Evaluate saRNA replicon topology, CSE/26S-SGP conservation, and innate-sensing metrics."""
    scaffolds_doc = load_modality_scaffolds(seed_dir)
    mol = bundle.molecules[0]
    u5_seq = mol.get_segment_sequence("five_prime_utr")
    cds_seq = mol.get_segment_sequence("cds")
    u3_seq = mol.get_segment_sequence("three_prime_utr")
    pa_seq = mol.get_segment_sequence("poly_a")

    u5_seg = mol.get_segment("five_prime_utr")
    scaf_id = (
        _SARNA_SCAFFOLD_BY_KEY.get(replicon_backbone.strip().upper(), replicon_backbone)
        if replicon_backbone
        else (u5_seg.component_id if u5_seg and u5_seg.component_id else "SCAF-SARNA-VEEV-TC83-CIS-REPLICON")
    )
    rep_scaf = _scaffold_by_id(scaffolds_doc, scaf_id)
    cis = rep_scaf["cis_elements_rna"]
    cse_51 = str(cis["nsp1_51nt_cse_rna"]).upper()
    sgp_26s = str(cis["subgenomic_26s_promoter_junction_rna"]).upper()
    cse_19 = str(cis["three_prime_19nt_cse_rna"]).upper()

    denom = max(1, len(cds_seq) - 1)
    cpg_freq = round(sum(1 for i in range(len(cds_seq) - 1) if cds_seq[i : i + 2] == "CG") / denom, 5)
    upa_freq = round(sum(1 for i in range(len(cds_seq) - 1) if cds_seq[i : i + 2] == "UA") / denom, 5)

    # Max non-overlapping intramolecular perfect dsRNA duplex stem inside GOI CDS
    max_dsrna_bp = 0
    for k in range(8, 28):
        if len(cds_seq) < 2 * k:
            break
        seen_pos: dict[str, int] = {}
        hit_k = False
        for i in range(len(cds_seq) - k + 1):
            sub = cds_seq[i : i + k]
            rc = str(Seq(sub).reverse_complement_rna())
            if rc in seen_pos and (i - seen_pos[rc]) >= k:
                hit_k = True
                break
            seen_pos.setdefault(sub, i)
        if hit_k:
            max_dsrna_bp = k
        else:
            break

    allow_vienna = compliance_track != "commercial_clean"
    rna_mod = _try_vienna() if allow_vienna else None
    goi_dg: float | None = (
        round(float(rna_mod.fold(cds_seq)[1]), 3)
        if (rna_mod is not None and cds_seq)
        else None
    )
    kozak_open: float | None = (
        compute_5prime_initiation_openness(
            cds_seq, utr5_rna=u5_seq[-35:], upstream_nt=15, downstream_nt=30
        )
        if allow_vienna
        else None
    )
    nsp_len = int(rep_scaf.get("nsp1_4_replicase_region_descriptor", {}).get("length_nt_incl_stop", 7482))

    return {
        "modality": "sarna",
        "replicon_scaffold_id": rep_scaf["scaffold_id"],
        "backbone_species_strain": rep_scaf["backbone_species_strain"],
        "aa_exact_match": mol.translate_cds_protein() == mol.effective_translation_target.protein_sequence,
        "starts_with_5prime_au": mol.sequence.startswith("AU"),
        "nsp1_51nt_cse_conserved": cse_51 in u5_seq,
        "subgenomic_26s_sgp_conserved": sgp_26s in u5_seq,
        "three_prime_19nt_cse_conserved": u3_seq.endswith(cse_19) and len(pa_seq) >= 70,
        "chemistry_unmodified_u_ok": mol.chemistry_track.uridine_chemistry == "unmodified_u",
        "innate_sensing_metrics": {
            "goi_cpg_dinuc_freq": cpg_freq,
            "goi_upa_dinuc_freq": upa_freq,
            "max_intramolecular_dsrna_stem_bp": max_dsrna_bp,
            "goi_u_fraction": round(cds_seq.count("U") / max(1, len(cds_seq)), 4),
        },
        "goi_mfe_kcal_mol": goi_dg,
        "goi_cai": compute_human_cai(cds_seq),
        "goi_kozak_unpaired_prob": kozak_open,
        "subgenomic_cassette_length_nt": len(mol.sequence),
        "estimated_full_replicon_length_nt": nsp_len + len(cds_seq) + len(u3_seq) + len(pa_seq) + 100,
    }


__all__ = [
    "build_circrna_bundle",
    "build_linear_mrna_bundle",
    "build_sarna_bundle",
    "evaluate_circrna_bundle",
    "evaluate_linear_mrna_bundle",
    "evaluate_sarna_bundle",
    "load_modality_scaffolds",
    "load_tissue_mirna_atlas",
    "select_atlas_mirna_for_detargeting",
]

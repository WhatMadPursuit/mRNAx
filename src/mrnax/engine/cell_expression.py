"""CellExpressionController: endogenous miRNA tissue de-targeting + RNAup accessibility + CRS boundary."""

from __future__ import annotations

import math
from typing import Any

from mrnax.engine.thermodynamics import MiRNAAccessibilityScore, compute_rnaup_mirna_interaction
from mrnax.schemas.construct_bundle import FeatureAnnotation
from mrnax.schemas.evaluation_report import AdvisoryNote

_TISSUE_TO_DEFAULT_MIRNA: dict[str, str] = {
    "liver": "hsa-miR-122-5p",
    "liver_hepatocyte": "hsa-miR-122-5p",
    "hepatocyte": "hsa-miR-122-5p",
    "myeloid": "hsa-miR-142-3p",
    "myeloid_monocyte_macrophage": "hsa-miR-142-3p",
    "spleen": "hsa-miR-142-3p",
    "endothelial": "hsa-miR-126-3p",
    "skeletal_muscle": "hsa-miR-133a-3p",
    "cardiac_heart": "hsa-miR-133a-3p",
    "t_cell": "hsa-miR-150-5p",
}

# Unstructured CAA-rich spacers free of uAUG, homopolymers >3, BspQI/BsaI sites, and AUUUA AREs
_CLEAN_SPACERS = ("CAAACAAACA", "CUAACUAACU", "CAACCAAUCA")


_CANONICAL_COMPARTMENT_MAP: dict[str, str] = {
    "liver": "liver_hepatocyte",
    "liver_hepatocyte": "liver_hepatocyte",
    "hepatocyte": "liver_hepatocyte",
    "myeloid": "myeloid_monocyte_macrophage",
    "myeloid_monocyte_macrophage": "myeloid_monocyte_macrophage",
    "endothelial": "endothelial",
    "skeletal_muscle": "skeletal_muscle",
    "muscle": "skeletal_muscle",
    "cardiac_heart": "cardiac_heart",
    "heart": "cardiac_heart",
    "t_cell": "t_cell",
    "brain_neural": "brain_neural",
    "brain": "brain_neural",
    "lung_epithelial": "lung_epithelial",
    "lung": "lung_epithelial",
    "kidney_renal": "kidney_renal",
    "kidney": "kidney_renal",
}


class MissingTissueMeasurement(ValueError):
    """The requested compartment has no usable measurement in this atlas."""


def _resolve_measured_cpm(cpm_map: dict[str, Any], tissue: str) -> float:
    name = tissue.strip().lower()
    value = cpm_map.get(_CANONICAL_COMPARTMENT_MAP.get(name, name))
    if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
        raise MissingTissueMeasurement(f"Protected tissue '{tissue}' has no measured FANTOM5 expression coverage")
    return float(value)


def select_detarget_mirna_entry(
    detarget_tissues: list[str],
    protected_tissues: list[str],
    mirna_registry: dict[str, Any],
) -> dict[str, Any]:
    entries = {m["mirna_id"]: m for m in mirna_registry["mirna_entries"]}
    for tissue in detarget_tissues:
        norm = tissue.strip().lower()
        cand_id = _TISSUE_TO_DEFAULT_MIRNA.get(norm)
        canon_det = _CANONICAL_COMPARTMENT_MAP.get(norm, norm)
        candidates = [entries[cand_id]] if (cand_id and cand_id in entries) else []
        extra_cands = sorted(
            [
                m
                for m in mirna_registry["mirna_entries"]
                if m.get("top_enriched_tissue") == canon_det and m["mirna_id"] != cand_id
            ],
            key=lambda x: (-float(x.get("specificity_tau", 0.0)), -float(x.get("top_tissue_cpm", 0.0))),
        )
        candidates.extend(extra_cands)
        for entry in candidates:
            cpm_map = entry.get("fantom5_mean_cpm_by_tissue", {})
            if any(_resolve_measured_cpm(cpm_map, p) > 500.0 for p in protected_tissues):
                continue
            return entry
        if cand_id and cand_id in entries:
            raise ValueError(
                f"Candidate de-targeting miRNA {cand_id} is highly expressed in protected tissue {protected_tissues}"
            )
    raise ValueError(f"No verified de-targeting miRNA configured for {detarget_tissues}")


def assemble_detargeted_3utr(
    *,
    base_utr3: str,
    detarget_tissues: list[str],
    protected_tissues: list[str],
    mirna_registry: dict[str, Any],
    copies: int = 2,
    utr3_global_start: int = 0,
) -> tuple[str, list[FeatureAnnotation], str]:
    """Insert 2..4 copies of 22-nt complementary miRNA sites >=15 nt downstream of stop codon."""
    if copies < 2 or copies > 4:
        raise ValueError("miRNA target cassette copies must be in [2, 4]")
    entry = select_detarget_mirna_entry(detarget_tissues, protected_tissues, mirna_registry)
    site_22 = str(entry["full_complementary_site_22nt"])
    mature = str(entry["mature_sequence"])

    # Remove destabilizing AUUUA ARE from synthetic junction if present in split point
    lead_offset = max(16, min(24, len(base_utr3)))
    prefix = base_utr3[:lead_offset]
    suffix = base_utr3[lead_offset:]

    cursor = len(prefix)
    cassette_parts: list[str] = [prefix]
    annotations: list[FeatureAnnotation] = []
    for idx in range(copies):
        if idx > 0:
            spacer = _CLEAN_SPACERS[idx % len(_CLEAN_SPACERS)]
            cassette_parts.append(spacer)
            cursor += len(spacer)
        site_start = utr3_global_start + cursor
        site_end = site_start + len(site_22)
        cassette_parts.append(site_22)
        annotations.append(
            FeatureAnnotation(
                feature_id=f"mirna_site_{entry['mirna_id'].replace('-', '_')}_{idx + 1}",
                feature_key="misc_binding",
                intervals=[(site_start, site_end)],
                qualifiers={
                    "bound_moiety": [entry["mirna_id"]],
                    "mrnax_origin_source": [f"{mirna_registry['registry_id']}:{entry['mirna_id']}"],
                    "mrnax_license_track": [mirna_registry["license_track"], entry.get("license_track", mirna_registry["license_track"])],
                    "note": ["FANTOM5_CC_BY_4_detarget_cassette"],
                },
            )
        )
        cursor += len(site_22)

    tail_spacer = _CLEAN_SPACERS[0]
    cassette_parts.append(tail_spacer)
    cassette_parts.append(suffix)
    assembled = "".join(cassette_parts)

    # Check protected tissue high-CPM miRNA 8mer seeds are not accidentally introduced
    for m in mirna_registry["mirna_entries"]:
        cpm_map = m.get("fantom5_mean_cpm_by_tissue", {})
        for prot in protected_tissues:
            if _resolve_measured_cpm(cpm_map, prot) >= 1000.0:
                seed8 = str(m["seed_8mer_site"])
                if seed8 in assembled[len(prefix) : len(prefix) + cursor]:
                    raise ValueError(
                        f"Protected tissue ({prot}) miRNA {m['mirna_id']} seed {seed8} collided in cassette"
                    )

    return assembled, annotations, mature


def evaluate_mirna_site_accessibility(
    *,
    cds_tail_rna: str,
    stop_codon: str,
    utr3_rna: str,
    mirna_mature_rna: str,
    site_offset_in_utr3: int | None = None,
    site_len: int = 22,
) -> MiRNAAccessibilityScore:
    from Bio.Seq import Seq

    rc_full = str(Seq(mirna_mature_rna).reverse_complement_rna())
    offset = utr3_rna.find(rc_full) if site_offset_in_utr3 is None else site_offset_in_utr3
    if offset < 0:
        raise ValueError("Target site not found in utr3_rna")
    context = cds_tail_rna[-50:] + stop_codon + utr3_rna[: offset + site_len + 35]
    start_in_ctx = len(cds_tail_rna[-50:]) + len(stop_codon) + offset
    return compute_rnaup_mirna_interaction(context, start_in_ctx, site_len, mirna_mature_rna)


def build_crs_and_tissue_safety_advisory() -> AdvisoryNote:
    return AdvisoryNote(
        advisory_id="adv_crs_three_layer_boundary",
        category="crs_safety_layer_boundary",
        message=(
            "Safety Endpoint Evidence Boundary: (1) RNA sequence layer (miR-122/miR-142 3'UTR sites + "
            "m1Psi + low dsRNA) reduces off-target hepatocyte/myeloid translation and innate RNA sensing only; "
            "(2) Protein engineering layer governs target affinity/CD3 valency; "
            "(3) LNP biodistribution & clinical step-dosing govern systemic CRS. Sequence scores do NOT prove low clinical CRS."
        ),
    )

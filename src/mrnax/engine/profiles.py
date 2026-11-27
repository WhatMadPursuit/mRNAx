"""Clean-Room component profile gatekeeper (`research_max` vs `commercial_clean`)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import re

from mrnax.rules.models import APPROVED_REPLACEMENT_RELEASES, load_replacement_card_yaml, verify_replacement_card_strict
from mrnax.schemas.common import LicenseTrack
from mrnax.schemas.design_intent import DesignIntent

_REPLACEMENT_CARD_BY_COMPONENT: dict[str, str] = {
    "structrfm_rna_foundation": "RC-002-structrfm-v0.0.9.yaml",
    "orthrus_mamba_rna_foundation": "RC-003-orthrus-v1.0.yaml",
    "eternafold_bpp_engine": "RC-004-eternafold-v1.3.1.yaml",
}

_FORBIDDEN_CLEAN_ROOM_LICENSE_RE = re.compile(
    r"("
    r"\b(?:a?gpl|lgpl)(?:[-\s]?v?\d+(?:\.\d+)*)?\b"
    r"|\bvarnaapi\b"
    r"|\bcc[-\s]?by[-\s]?(?:nc|sa)\b"
    r"|\bnon[-\s]?commercial\b"
    r"|\bacademic[-\s]+only\b"
    r"|\bresearch[_\-\s]+only\b"
    r"|\bneeds[_\-\s]+vendor[_\-\s]+license\b"
    r"|\bcommercial[_\-\s]+clearance[_\-\s]+pending\b"
    r"|\bunverified\b"
    r"|\ball\s+rights\s+reserved\b"
    r")",
    re.IGNORECASE,
)

_FORBIDDEN_PYPROJECT_CORE_PACKAGES: frozenset[str] = frozenset(
    {
        "viennarna",
        "rna",
        "varnaapi",
        "helical",
        "utr-lm",
        "ribonn",
        "codonbert",
        "gemorna",
        "aido-rna",
        "lineardesign",
        "rhofold",
    }
)


def _normalize_pkg_name(raw_dep: str) -> str:
    """Extract PEP 508 distribution name (including `@` URLs and extras) and PEP 503 normalize."""
    head = re.split(r"[<>=!~\[\s;@()]", str(raw_dep).strip(), maxsplit=1)[0].strip()
    return re.sub(r"[-_.]+", "-", head).strip("-").lower()


@dataclass(frozen=True)
class ComponentLicenseSpec:
    component_id: str
    display_name: str
    version: str
    role: str
    code_license: str
    weights_license: str
    data_license: str
    license_track: LicenseTrack
    transitive_dependencies: tuple[str, ...] = ()


def _resolve_catalog_dependency(
    raw_dep: str,
    catalog: Mapping[str, ComponentLicenseSpec],
) -> ComponentLicenseSpec | None:
    stripped = raw_dep.strip()
    if stripped in catalog:
        return catalog[stripped]
    head = re.split(r"[<>=!~\[\s;@()]", stripped, maxsplit=1)[0].strip()
    if head in catalog:
        return catalog[head]
    dep_norm = _normalize_pkg_name(stripped)
    if not dep_norm:
        return None
    for cid, cspec in catalog.items():
        if _normalize_pkg_name(cid) == dep_norm or _normalize_pkg_name(cspec.display_name) == dep_norm:
            return cspec
    return None


def validate_three_layer_and_transitive_clean_room(
    spec: ComponentLicenseSpec,
    *,
    catalog: Mapping[str, ComponentLicenseSpec] | None = None,
    _visiting: frozenset[str] = frozenset(),
) -> None:
    """Enforce 3-layer (Code, Weights, Data) + recursive transitive dependency audit on commercial_clean specs."""
    if spec.license_track != "commercial_clean":
        return
    if spec.component_id in _visiting:
        raise ValueError(
            f"Component '{spec.component_id}' has a cyclic transitive_dependencies reference"
        )
    layers = [
        ("code_license", spec.code_license),
        ("weights_license", spec.weights_license),
        ("data_license", spec.data_license),
        *[("transitive_dependency", dep) for dep in spec.transitive_dependencies],
    ]
    for layer_name, text in layers:
        m = _FORBIDDEN_CLEAN_ROOM_LICENSE_RE.search(text)
        if m:
            raise ValueError(
                f"Component '{spec.component_id}' cannot be marked commercial_clean: "
                f"{layer_name} contains restricted/copyleft marker '{m.group(0)}' ({text!r})"
            )
    active_catalog = COMPONENT_CATALOG if catalog is None else catalog
    next_visiting = _visiting | {spec.component_id}
    for dep in spec.transitive_dependencies:
        dep_norm = _normalize_pkg_name(dep)
        if dep_norm in _FORBIDDEN_PYPROJECT_CORE_PACKAGES:
            raise ValueError(
                f"Component '{spec.component_id}' cannot be marked commercial_clean: "
                f"transitive_dependency '{dep}' resolves to restricted package '{dep_norm}'"
            )
        dep_spec = _resolve_catalog_dependency(dep, active_catalog)
        if dep_spec is not None:
            if dep_spec.license_track != "commercial_clean":
                raise ValueError(
                    f"Component '{spec.component_id}' cannot be marked commercial_clean: "
                    f"depends transitively on '{dep_spec.component_id}' ({dep_spec.license_track})"
                )
            validate_three_layer_and_transitive_clean_room(
                dep_spec,
                catalog=active_catalog,
                _visiting=next_visiting,
            )


COMPONENT_CATALOG: dict[str, ComponentLicenseSpec] = {
    "stage0_random": ComponentLicenseSpec(
        component_id="stage0_random",
        display_name="mRNAx Stage0 Synonymous Sampler",
        version="0.1.0",
        role="generator",
        code_license="Proprietary (mRNAx Clean-Room)",
        weights_license="n/a",
        data_license="NCBI Genetic Code Table 1 (Public Domain)",
        license_track="commercial_clean",
    ),
    "dnachisel_cds": ComponentLicenseSpec(
        component_id="dnachisel_cds",
        display_name="DNA Chisel",
        version="3.2.16",
        role="generator",
        code_license="MIT License",
        weights_license="n/a",
        data_license="Public codon frequency tables",
        license_track="commercial_clean",
    ),
    "pareto_nsga2": ComponentLicenseSpec(
        component_id="pareto_nsga2",
        display_name="mRNAx Sliding-Window CDS-UTR Pareto Director",
        version="0.1.0",
        role="generator",
        code_license="Proprietary (mRNAx Clean-Room)",
        weights_license="n/a",
        data_license="miRBase (Public Domain) + FANTOM5 (CC-BY-4.0) + NCBI RefSeq",
        license_track="commercial_clean",
    ),
    "viennarna_thermo": ComponentLicenseSpec(
        component_id="viennarna_thermo",
        display_name="ViennaRNA",
        version="2.8.0",
        role="evaluator",
        code_license="ViennaRNA Custom Disclaimer (contact authors for commercial product; RPM spec.in#L19 conflict per C-B2)",
        weights_license="n/a",
        data_license="Turner 2004 nearest-neighbor parameters",
        license_track="needs_vendor_license",
    ),
    "surrogate_v1_clean": ComponentLicenseSpec(
        component_id="surrogate_v1_clean",
        display_name="mRNAx Commercial-Clean Axis-Routed Surrogate v1",
        version="1.5.0",
        role="evaluator",
        code_license="Proprietary (mRNAx Clean-Room)",
        weights_license="Proprietary Clean-Room LightGBM (zero ViennaRNA labels)",
        data_license="GSE232927 + eLife 45396 Wu2019 + Agarwal&Kelley2022 Saluki (CC-BY-4.0 / Apache-2.0)",
        license_track="commercial_clean",
    ),
    "structrfm_rna_foundation": ComponentLicenseSpec(
        component_id="structrfm_rna_foundation",
        display_name="structRFM Base Foundation Encoder + Chemistry-Specific SHAPE-MaP Probes",
        version=APPROVED_REPLACEMENT_RELEASES["structrfm_rna_foundation"].version,
        role="evaluator",
        code_license="MIT License (heqin-zhu/structRFM PyPI 0.0.9 via Apache-2.0 HuggingFace transformers; zero BPfold)",
        weights_license="MIT License (Zenodo 16754363 v0.0.3 deposit MD5 e572a7781a8099ff45692fcd46f83043)",
        data_license="RNAcentral (upstream license) + Mauger2019 Moderna fLuc SHAPE-MaP (CC-BY-4.0)",
        license_track="commercial_clean",
    ),
    "orthrus_mamba_rna_foundation": ComponentLicenseSpec(
        component_id="orthrus_mamba_rna_foundation",
        display_name="Orthrus 6-Track Mamba Foundation Encoder + Saluki Full-Transcript Half-Life Probe",
        version=APPROVED_REPLACEMENT_RELEASES["orthrus_mamba_rna_foundation"].version,
        role="evaluator",
        code_license="MIT License (bowang-lab/Orthrus + mRNAx portable Triton/PyTorch wrapper; zero mamba_ssm/GenomeKit)",
        weights_license="MIT License (HuggingFace antichronology/orthrus-6-track)",
        data_license="Saluki Agarwal&Kelley2022 13,642-gene consensus half-life (Apache-2.0 / CC-BY-4.0)",
        license_track="commercial_clean",
        transitive_dependencies=("torch (BSD-3-Clause)", "safetensors (Apache-2.0)"),
    ),
    "eternafold_bpp_engine": ComponentLicenseSpec(
        component_id="eternafold_bpp_engine",
        display_name="EternaFold v1.3.1 BSD-3-Clause Base-Pair Probability & Accessibility Engine",
        version=APPROVED_REPLACEMENT_RELEASES["eternafold_bpp_engine"].version,
        role="evaluator",
        code_license="BSD-3-Clause (WaymentSteeleLab/EternaFold v1.3.1; zero libRNA linkage)",
        weights_license="BSD-3-Clause (parameters/EternaFoldParams.v1 inside v1.3.1 release archive)",
        data_license="Stanford RMDB ETERNA_R69_0000 (CC0-1.0) + Mauger2019 GSE139176 (CC-BY-4.0)",
        license_track="commercial_clean",
        transitive_dependencies=("glibc/libstdc++/libm",),
    ),
    # Wave 8 Three-Layer & Transitive Dependency Trap Registry (Blocked in commercial_clean)
    "multistructrna_varnaapi": ComponentLicenseSpec(
        component_id="multistructrna_varnaapi",
        display_name="MultiStructRNA (MIT wrapper pulling GPL-3.0 varnaapi)",
        version="0.1.0",
        role="generator",
        code_license="MIT License (top-level wrapper)",
        weights_license="n/a",
        data_license="Public RNA secondary structure benchmarks",
        license_track="research_only",
        transitive_dependencies=("varnaapi==1.1.1 (GPL-3.0)", "ViennaRNA (needs_vendor_license)"),
    ),
    "helix_mrna_helical": ComponentLicenseSpec(
        component_id="helix_mrna_helical",
        display_name="Helix-mRNA / Helical Bio Framework",
        version="1.0.0",
        role="evaluator",
        code_license="AGPL-3.0 (helical framework)",
        weights_license="CC-BY-NC-SA-4.0 (HuggingFace Helix-mRNA checkpoint)",
        data_license="Mixed mRNA expression corpora",
        license_track="research_only",
        transitive_dependencies=("helical (AGPL-3.0)",),
    ),
    "utr_lm_gpl3": ComponentLicenseSpec(
        component_id="utr_lm_gpl3",
        display_name="UTR-LM",
        version="1.0.0",
        role="evaluator",
        code_license="GPL-3.0",
        weights_license="GPL-3.0",
        data_license="5'UTR MPRA datasets",
        license_track="research_only",
    ),
    "ribonn_noncommercial": ComponentLicenseSpec(
        component_id="ribonn_noncommercial",
        display_name="RiboNN Translation Efficiency Predictor",
        version="1.0.0",
        role="evaluator",
        code_license="Non-Commercial Academic License",
        weights_license="Non-Commercial Academic Weights",
        data_license="Ribosome profiling compilations",
        license_track="research_only",
    ),
    "codonbert_noncommercial": ComponentLicenseSpec(
        component_id="codonbert_noncommercial",
        display_name="CodonBERT",
        version="1.0.0",
        role="evaluator",
        code_license="Apache-2.0 with Non-Commercial Model Clause",
        weights_license="Non-Commercial Research Only",
        data_license="NCBI / RefSeq CDS corpus",
        license_track="research_only",
    ),
    "gemorna_noncommercial": ComponentLicenseSpec(
        component_id="gemorna_noncommercial",
        display_name="GEMORNA Generative mRNA Model",
        version="1.0.0",
        role="generator",
        code_license="Non-Commercial Research License",
        weights_license="Non-Commercial Research Weights",
        data_license="Proprietary/Non-Commercial mRNA corpus",
        license_track="needs_vendor_license",
    ),
    "aido_rna_noncommercial": ComponentLicenseSpec(
        component_id="aido_rna_noncommercial",
        display_name="AIDO.RNA Foundation Model",
        version="1.0.0",
        role="evaluator",
        code_license="GenBio AI Non-Commercial Community License",
        weights_license="GenBio AI Non-Commercial Community License",
        data_license="Multi-species RNA corpus",
        license_track="needs_vendor_license",
    ),
    "lineardesign_academic": ComponentLicenseSpec(
        component_id="lineardesign_academic",
        display_name="LinearDesign Lattice-CFG Optimizer",
        version="1.0.0",
        role="generator",
        code_license="Academic Only / Non-Commercial License (US20220269951A1 family)",
        weights_license="n/a",
        data_license="Turner 2004 parameters + codon usage tables",
        license_track="needs_vendor_license",
    ),
    "ribonanzanet_unverified_weights": ComponentLicenseSpec(
        component_id="ribonanzanet_unverified_weights",
        display_name="RibonanzaNet Chemical Mapping Predictor",
        version="1.0.0",
        role="evaluator",
        code_license="Apache-2.0 (Kaggle competition code)",
        weights_license="Unverified standalone .pt weight distribution license",
        data_license="Stanford Ribonanza competition dataset (CC-BY-4.0 / competition terms)",
        license_track="commercial_clearance_pending",
    ),
    "rhofold_plus_nc_data": ComponentLicenseSpec(
        component_id="rhofold_plus_nc_data",
        display_name="RhoFold+ 3D RNA Structure Predictor",
        version="1.0.0",
        role="evaluator",
        code_license="Apache-2.0",
        weights_license="Trained on non-commercial RNAMSA/PDB-derived corpus",
        data_license="Non-commercial training corpus restriction",
        license_track="research_only",
    ),
}

for _spec in COMPONENT_CATALOG.values():
    validate_three_layer_and_transitive_clean_room(_spec, catalog=COMPONENT_CATALOG)


def validate_pyproject_commercial_dependencies(pyproject_path: Path) -> None:
    """Verify core `[project].dependencies` contains zero copyleft/ViennaRNA/NC packages."""
    import tomllib

    doc = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    deps = doc.get("project", {}).get("dependencies", [])
    for raw_dep in deps:
        pkg = _normalize_pkg_name(str(raw_dep))
        if pkg in _FORBIDDEN_PYPROJECT_CORE_PACKAGES:
            raise ValueError(
                f"Forbidden non-commercial or copyleft dependency '{raw_dep}' found in core [project].dependencies"
            )


@dataclass(frozen=True)
class ProfileReadiness:
    is_runnable: bool
    uncomparable_reason: str | None = None


def _data_dir(name: str, source_path: str) -> Path:
    package = Path(__file__).resolve().parents[1]
    bundled = package / "_bundled_data" / name
    return bundled if bundled.is_dir() else package.parents[1] / source_path


def resolve_default_cards_dir() -> Path:
    return _data_dir("replacement_cards", "docs/replacement_cards")


def resolve_default_seed_dir() -> Path:
    return _data_dir("registry_seed", "registry_seed")


def _audit_component_for_commercial_clean(
    comp_id: str,
    *,
    resolved_cards_dir: Path,
    visiting: frozenset[str] = frozenset(),
) -> str | None:
    """Return None if `comp_id` and all its recursive transitive dependencies pass `commercial_clean`, else reason."""
    if comp_id in visiting:
        return f"{comp_id} (cyclic_transitive_dependency)"
    spec = COMPONENT_CATALOG.get(comp_id)
    if spec is None or spec.license_track != "commercial_clean":
        reason_tag = "unregistered" if spec is None else spec.license_track
        return f"{comp_id} ({reason_tag})"
    try:
        validate_three_layer_and_transitive_clean_room(
            spec,
            catalog=COMPONENT_CATALOG,
            _visiting=visiting,
        )
    except ValueError as exc:
        return f"{comp_id} (three_layer_or_transitive_violation: {exc})"
    card_fname = _REPLACEMENT_CARD_BY_COMPONENT.get(comp_id)
    if card_fname is not None:
        card_path = resolved_cards_dir / card_fname
        if not card_path.is_file():
            return f"{comp_id} (missing_replacement_card:{card_fname})"
        try:
            card = load_replacement_card_yaml(card_path)
            verify_replacement_card_strict(
                card, expected_component_id=comp_id, expected_version=spec.version
            )
        except ValueError as exc:
            return f"{comp_id} (failed_replacement_threshold: {exc})"
    next_visiting = visiting | {comp_id}
    for dep in spec.transitive_dependencies:
        dep_spec = _resolve_catalog_dependency(dep, COMPONENT_CATALOG)
        if dep_spec is not None:
            sub_err = _audit_component_for_commercial_clean(
                dep_spec.component_id,
                resolved_cards_dir=resolved_cards_dir,
                visiting=next_visiting,
            )
            if sub_err is not None:
                return f"{comp_id} -> {sub_err}"
    return None


def check_profile_readiness(intent: DesignIntent, *, cards_dir: Path | None = None) -> ProfileReadiness:
    requested = list(intent.runtime_policy.enabled_generators) + list(
        intent.runtime_policy.enabled_evaluators
    )
    def _uses_viennarna_recursively(cid: str, seen: frozenset[str] = frozenset()) -> bool:
        if cid == "viennarna_thermo":
            return True
        if cid in seen or cid not in COMPONENT_CATALOG:
            return False
        sp = COMPONENT_CATALOG[cid]
        nxt = seen | {cid}
        for dep in sp.transitive_dependencies:
            if _normalize_pkg_name(dep) in {"viennarna", "rna"}:
                return True
            ds = _resolve_catalog_dependency(dep, COMPONENT_CATALOG)
            if ds is not None and _uses_viennarna_recursively(ds.component_id, nxt):
                return True
        return False

    if any(_uses_viennarna_recursively(cid) for cid in requested) and importlib.util.find_spec("RNA") is None:
        return ProfileReadiness(
            is_runnable=False,
            uncomparable_reason=(
                "无法比较：当前纯商用精简环境未安装可选包 ViennaRNA (`pip install mrnax[research]`)，请切换至 commercial_clean + surrogate_v1_clean。"
            ),
        )

    if intent.compliance_track == "research_max":
        return ProfileReadiness(is_runnable=True)

    # In `commercial_clean`, block any requested component (or transitive dependency) that is not strictly `commercial_clean`
    resolved_cards_dir = cards_dir or resolve_default_cards_dir()
    blocked: list[str] = []
    for comp_id in requested:
        err = _audit_component_for_commercial_clean(comp_id, resolved_cards_dir=resolved_cards_dir)
        if err is not None:
            blocked.append(err)

    if blocked:
        return ProfileReadiness(
            is_runnable=False,
            uncomparable_reason=(
                "无法比较：commercial_clean 档阻断未获纯商用产品放行的组件 "
                f"[{', '.join(blocked)}]（详见组件商用替换评估卡 RC-001-viennarna-v2.8.0.yaml）"
            ),
        )
    return ProfileReadiness(is_runnable=True)

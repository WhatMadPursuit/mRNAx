"""Clean-Room component profile gatekeeper (`research_max` vs `commercial_clean`)."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path

from mrnax.rules.models import APPROVED_REPLACEMENT_RELEASES, load_replacement_card_yaml, verify_replacement_card_strict
from mrnax.schemas.common import LicenseTrack
from mrnax.schemas.design_intent import DesignIntent

_REPLACEMENT_CARD_BY_COMPONENT: dict[str, str] = {
    "structrfm_rna_foundation": "RC-002-structrfm-v0.0.9.yaml",
    "orthrus_mamba_rna_foundation": "RC-003-orthrus-v1.0.yaml",
}


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
    ),
}


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


def check_profile_readiness(intent: DesignIntent, *, cards_dir: Path | None = None) -> ProfileReadiness:
    requested = list(intent.runtime_policy.enabled_generators) + list(
        intent.runtime_policy.enabled_evaluators
    )
    if "viennarna_thermo" in requested and importlib.util.find_spec("RNA") is None:
        return ProfileReadiness(
            is_runnable=False,
            uncomparable_reason=(
                "无法比较：当前纯商用精简环境未安装可选包 ViennaRNA (`pip install mrnax[research]`)，请切换至 commercial_clean + surrogate_v1_clean。"
            ),
        )

    if intent.compliance_track == "research_max":
        return ProfileReadiness(is_runnable=True)

    # In `commercial_clean`, block any requested component that is not strictly `commercial_clean`
    resolved_cards_dir = cards_dir or resolve_default_cards_dir()
    blocked: list[str] = []
    for comp_id in requested:
        spec = COMPONENT_CATALOG.get(comp_id)
        if spec is None or spec.license_track != "commercial_clean":
            reason_tag = "unregistered" if spec is None else spec.license_track
            blocked.append(f"{comp_id} ({reason_tag})")
            continue
        card_fname = _REPLACEMENT_CARD_BY_COMPONENT.get(comp_id)
        if card_fname is not None:
            card_path = resolved_cards_dir / card_fname
            if not card_path.is_file():
                blocked.append(f"{comp_id} (missing_replacement_card:{card_fname})")
                continue
            try:
                card = load_replacement_card_yaml(card_path)
                verify_replacement_card_strict(
                    card, expected_component_id=comp_id, expected_version=spec.version
                )
            except ValueError as exc:
                blocked.append(f"{comp_id} (failed_replacement_threshold: {exc})")

    if blocked:
        return ProfileReadiness(
            is_runnable=False,
            uncomparable_reason=(
                "无法比较：commercial_clean 档阻断未获纯商用产品放行的组件 "
                f"[{', '.join(blocked)}]（详见组件商用替换评估卡 RC-001-viennarna-v2.8.0.yaml）"
            ),
        )
    return ProfileReadiness(is_runnable=True)

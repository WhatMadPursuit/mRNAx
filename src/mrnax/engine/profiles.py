"""Clean-Room component profile gatekeeper (`research_max` vs `commercial_clean`)."""

from __future__ import annotations

from dataclasses import dataclass

from mrnax.schemas.common import LicenseTrack
from mrnax.schemas.design_intent import DesignIntent


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
    "structrfm_mit_probe": ComponentLicenseSpec(
        component_id="structrfm_mit_probe",
        display_name="structRFM Base Foundation Encoder + Chemistry-Specific SHAPE-MaP Probes",
        version="0.0.9",
        role="evaluator",
        code_license="MIT License (heqin-zhu/structRFM via Apache-2.0 HuggingFace transformers; zero BPfold)",
        weights_license="MIT License (Zenodo 16754363 MD5 e572a7781a8099ff45692fcd46f83043)",
        data_license="RNAcentral (MIT deposit) + Mauger2019 Moderna SHAPE-MaP (CC-BY-4.0)",
        license_track="commercial_clean",
    ),
}


@dataclass(frozen=True)
class ProfileReadiness:
    is_runnable: bool
    uncomparable_reason: str | None = None


def check_profile_readiness(intent: DesignIntent) -> ProfileReadiness:
    if intent.compliance_track == "research_max":
        return ProfileReadiness(is_runnable=True)

    # In `commercial_clean`, block any requested component that is not strictly `commercial_clean`
    requested = list(intent.runtime_policy.enabled_generators) + list(
        intent.runtime_policy.enabled_evaluators
    )
    blocked: list[str] = []
    for comp_id in requested:
        spec = COMPONENT_CATALOG.get(comp_id)
        if spec is None or spec.license_track != "commercial_clean":
            reason_tag = "unregistered" if spec is None else spec.license_track
            blocked.append(f"{comp_id} ({reason_tag})")

    if blocked:
        return ProfileReadiness(
            is_runnable=False,
            uncomparable_reason=(
                "无法比较：commercial_clean 档阻断未获纯商用产品放行的组件 "
                f"[{', '.join(blocked)}]（详见组件商用替换评估卡 RC-001-viennarna-v2.8.0.yaml）"
            ),
        )
    return ProfileReadiness(is_runnable=True)

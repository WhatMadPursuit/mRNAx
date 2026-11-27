"""Shared strongly-typed literals and enums for mRNAx v2 schemas."""

from __future__ import annotations

from typing import Literal

SegmentScope = Literal[
    "five_prime_utr",
    "cds",
    "stop_cassette",
    "three_prime_utr",
    "poly_a",
]

ConstraintScope = Literal[
    "five_prime_utr",
    "cds",
    "stop_cassette",
    "three_prime_utr",
    "poly_a",
    "full_transcript",
    "inter_chain",
    "three_prime_terminal",
]

LicenseTrack = Literal[
    "commercial_clean",
    "commercial_clearance_pending",
    "needs_vendor_license",
    "research_only",
    "quarantine_reference",
]

ComplianceProfile = Literal["research_max", "commercial_clean"]

UridineChemistry = Literal["unmodified_u", "pseudouridine", "m1psi", "5mou", "custom_mod"]

CapChemistry = Literal[
    "cap1_cleancap_ag",
    "cap1_arca_gg",
    "cap1_enzymatic",
    "unknown",
]

Modality = Literal["linear_mrna", "sarna", "tarna", "circrna"]

Topology = Literal["linear", "circular"]

DerivationAction = Literal[
    "reference_import",
    "computational_reference",
    "de_novo_design",
    "locked_refinement",
    "disambiguation_chimera",
]

CoExpressionMode = Literal[
    "single_monocistronic",
    "lnp_co_encapsulation",
    "bicistronic_2a",
    "bicistronic_ires",
]

RuleStatus = Literal[
    "verified_rule",
    "conditional_rule",
    "disproven_blacklist",
    "insufficient_evidence",
]

ChemistryEvalMode = Literal[
    "endogenous_unmodified_proxy",
    "native_chemistry_validated",
    "cross_chemistry_thermodynamic_approximation",
    "not_evaluated",
]

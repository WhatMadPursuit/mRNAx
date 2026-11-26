"""Schema 1: INSDC Dual-Layer Molecular IR (`ConstructBundle`)."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Literal, Self

from Bio.Seq import Seq
from pydantic import BaseModel, ConfigDict, Field, model_validator

from mrnax.schemas.common import (
    CapChemistry,
    CoExpressionMode,
    DerivationAction,
    LicenseTrack,
    Modality,
    SegmentScope,
    Topology,
    UridineChemistry,
)

_RNA_RE = re.compile(r"^[ACGU]+$")
_STOP_CODONS = frozenset({"UAA", "UAG", "UGA"})
_SEGMENT_ORDER: dict[str, int] = {
    "five_prime_utr": 0,
    "cds": 1,
    "stop_cassette": 2,
    "three_prime_utr": 3,
    "poly_a": 4,
}


class FrozenSchemaModel(BaseModel):
    """Base frozen strict Pydantic model with validated copy helper."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    def model_copy_validated(self, **updates: object) -> Self:
        payload = self.model_dump(mode="python")
        payload.update(updates)
        return type(self).model_validate(payload)


class ChemistryTrackSpec(FrozenSchemaModel):
    uridine_chemistry: UridineChemistry
    cap_chemistry: CapChemistry = "unknown"
    source: Literal["user_declared", "inferred", "unknown"] = "user_declared"
    custom_modification_notes: str | None = None


class ProteinTargetSpec(FrozenSchemaModel):
    protein_sequence: str = Field(min_length=1)
    transl_table: int = Field(default=1, ge=1)
    accession_or_source: str = Field(min_length=1)
    entry_version: str | None = None
    has_signal_peptide: bool = False
    is_secreted_or_membrane: bool = False
    confirmation_status: Literal["confirmed", "needs_protein_confirmation"] = "confirmed"


class SegmentSpan(FrozenSchemaModel):
    segment_type: SegmentScope
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    component_id: str | None = None
    origin_source: str = Field(min_length=1)
    license_track: LicenseTrack
    action_taken: Literal[
        "locked_verbatim",
        "selected_from_library",
        "generated_novo",
        "synonymous_mutated",
        "chimera_spliced",
    ]

    @model_validator(mode="after")
    def _check_span(self) -> Self:
        if self.end <= self.start:
            raise ValueError(f"Segment end ({self.end}) must exceed start ({self.start})")
        return self


class FeatureAnnotation(FrozenSchemaModel):
    """INSDC Feature Table style annotation supporting nested & overlapping intervals."""

    feature_id: str = Field(min_length=1)
    feature_key: str = Field(min_length=1)
    intervals: list[tuple[int, int]] = Field(min_length=1)
    strand: Literal[1, -1] = 1
    qualifiers: dict[str, list[str]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_intervals(self) -> Self:
        for start, end in self.intervals:
            if start < 0 or end <= start:
                raise ValueError(f"Invalid annotation interval ({start}, {end})")
        return self


class RNAMoleculeSpec(FrozenSchemaModel):
    molecule_id: str = Field(min_length=1)
    modality: Modality = "linear_mrna"
    topology: Topology = "linear"
    chemistry_track: ChemistryTrackSpec
    effective_translation_target: ProteinTargetSpec
    sequence: str = Field(min_length=3)
    segments: list[SegmentSpan] = Field(min_length=1)
    annotations: list[FeatureAnnotation] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_insdc_and_translation(self) -> Self:
        if not _RNA_RE.match(self.sequence):
            raise ValueError("RNA sequence must contain only uppercase A, C, G, U characters")

        seq_len = len(self.sequence)
        cursor = 0
        last_order = -1
        cds_spans: list[SegmentSpan] = []
        stop_spans: list[SegmentSpan] = []

        for idx, seg in enumerate(self.segments):
            if seg.start != cursor:
                raise ValueError(
                    f"Segments must be contiguous and non-overlapping: expected start {cursor}, "
                    f"got {seg.start} for {seg.segment_type}"
                )
            seg_rank = _SEGMENT_ORDER[seg.segment_type]
            if seg_rank <= last_order:
                raise ValueError(
                    f"Segments must follow unique canonical 5'->3' order; unexpected '{seg.segment_type}' at index {idx}"
                )
            if seg.segment_type == "poly_a" and idx != len(self.segments) - 1:
                raise ValueError("poly_a segment may only appear as the terminal 3' segment")
            last_order = seg_rank
            cursor = seg.end
            if seg.segment_type == "cds":
                cds_spans.append(seg)
            elif seg.segment_type == "stop_cassette":
                stop_spans.append(seg)

        if cursor != seq_len:
            raise ValueError(
                f"Segments must be contiguous and cover full sequence [0, {seq_len}), ended at {cursor}"
            )

        for ann in self.annotations:
            for start, end in ann.intervals:
                if end > seq_len:
                    raise ValueError(
                        f"Annotation {ann.feature_id} interval ({start}, {end}) exceeds sequence length {seq_len}"
                    )

        if not cds_spans:
            raise ValueError("RNAMoleculeSpec must contain at least one 'cds' segment")

        if self.effective_translation_target.confirmation_status == "confirmed":
            cds_nt = "".join(self.sequence[s.start : s.end] for s in cds_spans)
            if len(cds_nt) % 3 != 0:
                raise ValueError(f"CDS length ({len(cds_nt)}) must be a multiple of 3")
            translated = str(
                Seq(cds_nt).translate(table=self.effective_translation_target.transl_table, to_stop=False)
            )
            if not stop_spans and translated.endswith("*"):
                translated = translated[:-1]
            if "*" in translated:
                raise ValueError("Internal premature stop codon found inside CDS")
            if translated != self.effective_translation_target.protein_sequence:
                raise ValueError(
                    "CDS translation does not match effective_translation_target.protein_sequence"
                )
            for st in stop_spans:
                st_seq = self.sequence[st.start : st.end]
                if len(st_seq) < 3 or st_seq[:3] not in _STOP_CODONS:
                    raise ValueError(
                        f"stop_cassette '{st_seq}' must begin with a canonical stop codon (UAA/UAG/UGA)"
                    )
        return self

    def get_segment(self, segment_type: SegmentScope) -> SegmentSpan | None:
        for seg in self.segments:
            if seg.segment_type == segment_type:
                return seg
        return None

    def get_segment_sequence(self, segment_type: SegmentScope) -> str:
        seg = self.get_segment(segment_type)
        return "" if seg is None else self.sequence[seg.start : seg.end]


class ConstructBundle(FrozenSchemaModel):
    bundle_id: str = Field(min_length=1)
    version_digest: str = Field(min_length=64, max_length=64)
    parent_bundle_id: str | None = None
    derivation_action: DerivationAction
    target_stoichiometry: dict[str, float]
    co_expression_mode: CoExpressionMode = "single_monocistronic"
    molecules: list[RNAMoleculeSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def _verify_digest_integrity(self) -> Self:
        expected = self.compute_digest(
            self.molecules, self.target_stoichiometry, self.co_expression_mode
        )
        if self.version_digest != expected:
            raise ValueError(
                f"ConstructBundle version_digest mismatch: expected {expected}, got {self.version_digest}"
            )
        return self

    @classmethod
    def compute_digest(
        cls,
        molecules: list[RNAMoleculeSpec],
        target_stoichiometry: dict[str, float],
        co_expression_mode: CoExpressionMode,
    ) -> str:
        canonical = json.dumps(
            {
                "co_expression_mode": co_expression_mode,
                "target_stoichiometry": dict(sorted(target_stoichiometry.items())),
                "molecules": [m.model_dump(mode="json") for m in molecules],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @classmethod
    def create(
        cls,
        *,
        bundle_id: str,
        parent_bundle_id: str | None,
        derivation_action: DerivationAction,
        target_stoichiometry: dict[str, float],
        co_expression_mode: CoExpressionMode = "single_monocistronic",
        molecules: list[RNAMoleculeSpec],
    ) -> Self:
        digest = cls.compute_digest(molecules, target_stoichiometry, co_expression_mode)
        return cls(
            bundle_id=bundle_id,
            version_digest=digest,
            parent_bundle_id=parent_bundle_id,
            derivation_action=derivation_action,
            target_stoichiometry=target_stoichiometry,
            co_expression_mode=co_expression_mode,
            molecules=molecules,
        )

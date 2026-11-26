# ADR-0001: Four-Tier Clean-Room Asset Licensing & Patent FTO Guard

- **Status**: Accepted (2026-11-27)
- **Context**: `mRNAx v2` targets ultimate pure commercial deployment (`commercial_clean`) while allowing R&D benchmarking (`research_max`). Cross-audits (`C-B2`, `G-P01`, `G-B1`) identified two critical compliance traps:
  1. Publicly accessible GEO tables (`GSE114002`) lack explicit file-level commercial licenses (`commercial_clearance_pending`), and `ViennaRNA v2.8.0` exhibits a `COPYING` ("contact authors to include in a commercial product") vs `viennarna.spec.in#L19` ("non-commercial") packaging conflict (`needs_vendor_license`).
  2. BioNTech patent family `51220548` (`US10717982B2`, `US12054723B2`, `EP3167059B1`) claims the broad genus `>=20A + non-A linker + >=20A` (total length >=80 nt), not merely the `GCAUAUGACU` example linker.
- **Decision**:
  1. Enforce five explicit `LicenseTrack` tags across data, rules, weights, and software components: `commercial_clean`, `commercial_clearance_pending`, `needs_vendor_license`, `research_only`, and `quarantine_reference`.
  2. Physically separate `rules_commercial_v1` from `rules_research_v1`; block NC/GPL/unapproved components from executing in `commercial_clean` profile, returning explicit `UNCOMPARABLE_MISSING_APPROVED_COMPONENT` when an approved commercial replacement is not yet validated.
  3. Enforce `sequence_fto_guard` across generation, filtering, and export, blocking both patented clinical UTR sequences (`>=80%` identity) and the `51220548` segmented poly(A) genus in `commercial_clean`.

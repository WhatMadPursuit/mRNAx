# mRNAx v2 · Agentic RNA Sequence Design Platform

**Live Web UI (`github.io`)**: [https://whatmadpursuit.github.io/mRNAx/](https://whatmadpursuit.github.io/mRNAx/)

`mRNAx v2` is an agent-driven, multi-objective computational RNA sequence design and verification platform built from scratch around four decoupled infrastructure pillars:

1. **7×24 Control Plane & Web Harness Gateway (`Vultr` Linux VPS)**: Persistent FastAPI + SQLite WAL (`synchronous=FULL`) lease/CAS state machine (`mrnax-gateway.service` under `systemd`) + sliding-window thermodynamic optimization engine (`ViennaRNA v2.8.0` + `DNA Chisel`).
2. **On-Demand Heavy Compute (`Google Colab` CPU / GPU)**: Ephemeral `colab run` worker pool for bulk MPRA evidence audits (`GSE114002`, `GSE232927`) and open-source neural predictors.
3. **Durable Artifact & Checkpoint Vault (`Google Drive REST v3`)**: Zero-FUSE per-attempt `SEALED.json` + `COMMIT_RECEIPT` snapshots for power-loss recovery.
4. **Interactive Web Workbench (`GitHub Pages`)**: Closed-loop **Input → Agent Intent Compilation & Guardrails → Multi-Objective Pareto Output → Single-Item Iterative Refinement → 4-File INSDC Export** (`full_rna.fasta`, `cds_only.fasta`, `construct.gb`, `chemistry_metadata.json`).

---

## Quick Start (Local CLI & Unit Tests)

```bash
uv sync --frozen
uv run pytest tests/unit/ -v

# Deterministic CLI replay from a versioned DesignIntent JSON
uv run mrnax replay \
  --intent-file artifacts/m1_egfp/protein_only_intent_v2.json \
  --out-dir /tmp/mrnax_replay_check
```

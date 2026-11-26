# ADR-0002: Zero-Loss Drive REST v3 Write-Ahead Checkpointing & VPS SQLite WAL CAS

- **Status**: Accepted (2026-11-27)
- **Context**: Local workstations may power down unexpectedly, Colab VMs are preemptible, and Google Drive allows duplicate filenames in the same folder while `drive.mount` FUSE risks silent half-written files on disconnect.
- **Decision**:
  1. **Zero FUSE**: Workers write checkpoints, candidate bundles, and model artifacts exclusively via HTTPS Google Drive REST API v3 into isolated per-attempt folders (`/mRNAx/runs/{execution_id}/attempts/{attempt_id}/`), verifying server `md5Checksum` on every upload before writing `SEALED.json`.
  2. **Drive-First Durability**: Checkpoints and sealed artifacts are persisted to Google Drive *before* notifying the Control Plane, guaranteeing zero lost computation or unsaved weights even if the local laptop powers off or the gateway restarts mid-run.
  3. **VPS-Ready Control Plane + Disaster Recovery**: Control Plane runs FastAPI + SQLite (`journal_mode=WAL`, `synchronous=FULL`) with single-statement `CommitAttempt` CAS (`WHERE execution_id=? AND lease_epoch=? AND status='RUNNING'`), rejecting stale zombie workers with HTTP 409, and supports `reconcile_from_drive_seals()` to reconstruct state from Drive `SEALED.json` manifests if a host restarts.

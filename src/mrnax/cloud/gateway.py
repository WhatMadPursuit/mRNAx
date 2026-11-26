"""VPS-ready FastAPI Control Plane Gateway exposing atomic lease & CommitAttempt CAS endpoints."""

from __future__ import annotations

from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from mrnax.cloud.cas_store import (
    ExecutionCASStore,
    IdempotencyPayloadMismatchError,
    StaleLeaseConflictError,
)


class SubmitExecutionRequest(BaseModel):
    submission_id: str
    request_digest: str
    intent_id: str


class LeaseAcquireRequest(BaseModel):
    worker_id: str
    ttl_seconds: int = 300


class CommitAttemptRequest(BaseModel):
    lease_epoch: int
    sealed_file_id: str
    sealed_sha256: str


def create_control_plane_app(store: ExecutionCASStore) -> FastAPI:
    app = FastAPI(title="mRNAx v2 Control Plane Gateway", version="0.1.0")

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/executions")
    def submit_execution(req: SubmitExecutionRequest) -> dict[str, object]:
        try:
            row = store.submit_execution(req.submission_id, req.request_digest, req.intent_id)
            return asdict(row)
        except IdempotencyPayloadMismatchError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/v1/executions/{execution_id}/leases")
    def acquire_lease(execution_id: str, req: LeaseAcquireRequest) -> dict[str, object]:
        try:
            grant = store.acquire_or_renew_lease(
                execution_id, req.worker_id, ttl_seconds=req.ttl_seconds
            )
            return asdict(grant)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except StaleLeaseConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/v1/executions/{execution_id}/attempts/{attempt_id}:commit")
    def commit_attempt(
        execution_id: str, attempt_id: str, req: CommitAttemptRequest
    ) -> dict[str, object]:
        try:
            receipt = store.commit_attempt_cas(
                execution_id=execution_id,
                attempt_id=attempt_id,
                lease_epoch=req.lease_epoch,
                sealed_file_id=req.sealed_file_id,
                sealed_sha256=req.sealed_sha256,
            )
            return asdict(receipt)
        except StaleLeaseConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/v1/executions/{execution_id}")
    def get_execution(execution_id: str) -> dict[str, object]:
        try:
            return asdict(store.get_execution(execution_id))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    return app

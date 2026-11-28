"""Authenticate checkpoint, config, and tokenizer bytes against Replacement Card approved release digests."""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

from mrnax.rules.models import APPROVED_REPLACEMENT_RELEASES

APPROVED_SURROGATE_MANIFEST_SHA256: dict[str, str] = {
    "surrogate_v1": "4b0c2357699f7f15c6686251ac2bf9c34d53f9604ed0eac37290e569441225a6",
    "surrogate_v2": "b8cc1b651ca419bbe634b7816e5c4ff5cb05d74b4186d4b83c23199a7aef8e6b",
}


def verify_surrogate_manifest_trust_anchor(
    manifest_path: Path,
    *,
    allow_sandbox_override: bool = False,
) -> str:
    """Verify TRAINING_PROVENANCE_MANIFEST.json against pinned release digests unless sandbox override is explicit."""
    actual = _sha256_file(manifest_path)
    if allow_sandbox_override:
        return actual
    dir_name = manifest_path.parent.name
    expected_for_dir = APPROVED_SURROGATE_MANIFEST_SHA256.get(dir_name)
    if expected_for_dir is not None:
        if actual != expected_for_dir:
            raise PermissionError(
                f"Untrusted provenance manifest '{manifest_path}': sha256 {actual} != pinned {dir_name} release digest {expected_for_dir}"
            )
        return actual
    if actual not in APPROVED_SURROGATE_MANIFEST_SHA256.values():
        raise PermissionError(
            f"Untrusted provenance manifest '{manifest_path}': sha256 {actual} is not pinned in APPROVED_SURROGATE_MANIFEST_SHA256"
        )
    return actual


def _sha256_file(path: Path) -> str:
    actual = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            actual.update(chunk)
    return actual.hexdigest()


def verify_checkpoint(
    path: Path,
    expected: str | None,
    setting: str,
    *,
    approved_release_digest: str | None = None,
    allow_sandbox_override: bool = False,
) -> None:
    supplied = expected if expected is not None else (os.environ.get(setting, "").strip() or None)
    if not allow_sandbox_override and approved_release_digest is not None:
        if supplied is not None and supplied != approved_release_digest:
            raise ValueError(
                f"Configured digest for {path.name} ({setting}) disagrees with the approved Replacement Card release digest"
            )
        digest = approved_release_digest
    else:
        digest = supplied or ""
    if not re.fullmatch("[0-9a-f]{64}", digest):
        raise ValueError(f"{path.name} requires an approved lowercase SHA-256 digest ({setting})")
    if _sha256_file(path) != digest:
        raise ValueError(
            f"Checkpoint SHA-256 mismatch for {path.name} against approved lowercase SHA-256 release digest ({setting})"
        )


def verify_release_directory_artifacts(
    checkpoint_dir: Path,
    component_id: str,
    *,
    allow_sandbox_override: bool = False,
    weights_sha256_override: str | None = None,
    weights_env_var: str = "",
) -> None:
    release = APPROVED_REPLACEMENT_RELEASES.get(component_id)
    if release is None or not release.artifact_digests:
        raise ValueError(f"Component '{component_id}' has no pinned approved_artifact_digests")
    if allow_sandbox_override:
        wt_candidates = [
            checkpoint_dir / fn for fn in ("model.safetensors", "pytorch_model.bin") if (checkpoint_dir / fn).is_file()
        ]
        if not wt_candidates:
            raise FileNotFoundError(f"Checkpoint directory '{checkpoint_dir}' is missing model weights")
        verify_checkpoint(
            wt_candidates[0],
            weights_sha256_override,
            weights_env_var,
            allow_sandbox_override=True,
        )
        return

    for rel_name, approved_sha in sorted(release.artifact_digests.items()):
        fp = checkpoint_dir / rel_name
        if not fp.is_file():
            raise ValueError(
                f"Checkpoint directory '{checkpoint_dir}' is missing required file '{rel_name}' from approved lowercase SHA-256 release manifest ({weights_env_var})"
            )
        is_weight = rel_name in {"model.safetensors", "pytorch_model.bin"}
        if is_weight:
            supplied = weights_sha256_override if weights_sha256_override is not None else (
                os.environ.get(weights_env_var, "").strip() or None
            )
            if supplied is not None and supplied != approved_sha:
                raise ValueError(
                    f"Checkpoint SHA-256 mismatch for {rel_name}: configured {weights_env_var} disagrees with approved lowercase SHA-256 release digest"
                )
        verify_checkpoint(
            fp,
            approved_sha,
            weights_env_var if is_weight else f"MRNAX_{component_id.upper()}_{rel_name.upper()}_SHA256",
            approved_release_digest=approved_sha,
            allow_sandbox_override=False,
        )

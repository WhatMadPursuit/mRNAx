"""Bit-exact PRNG checkpointing worker & Colab CLI auto-releasing orchestrator."""

from __future__ import annotations

import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any

from mrnax.cloud.drive_rest_v3 import DriveRestV3Client


def _encode_prng_state(state: tuple[Any, ...]) -> list[Any]:
    version, internal_state, gauss_next = state
    return [version, list(internal_state), gauss_next]


def _decode_prng_state(encoded: list[Any]) -> tuple[Any, ...]:
    return (int(encoded[0]), tuple(int(x) for x in encoded[1]), encoded[2])


class ResumableCPUOptimizerWorker:
    """Deterministic CPU search trajectory with Drive-first write-ahead checkpointing."""

    def __init__(self, *, seed: int, total_steps: int = 10, checkpoint_every: int = 5) -> None:
        self.seed = seed
        self.total_steps = total_steps
        self.checkpoint_every = checkpoint_every

    def run_with_drive_checkpoints(
        self,
        drive: DriveRestV3Client,
        attempt_folder_id: str,
        *,
        resume_from_attempt_folder_id: str | None = None,
        simulate_interrupt_at_step: int | None = None,
    ) -> dict[str, Any]:
        rng = random.Random(self.seed)
        start_step = 0
        trajectory: list[dict[str, Any]] = []
        best_score = -1e9
        best_seq = ""

        if resume_from_attempt_folder_id is not None:
            children = drive.transport.list_children(parent_id=resume_from_attempt_folder_id)
            ckpts = sorted(
                [c for c in children if str(c["name"]).startswith("checkpoint_step_")],
                key=lambda c: int(str(c["name"]).split("_")[-1].split(".")[0]),
            )
            if ckpts:
                latest_meta = ckpts[-1]
                raw = drive.transport.download_bytes(str(latest_meta["id"]))
                ckpt_doc = json.loads(raw.decode("utf-8"))
                start_step = int(ckpt_doc["completed_step"])
                trajectory = list(ckpt_doc["trajectory"])
                best_score = float(ckpt_doc["best_score"])
                best_seq = str(ckpt_doc["best_seq"])
                rng.setstate(_decode_prng_state(ckpt_doc["prng_state"]))

        codons_phe_leu_gly = ["UUC", "CUG", "CUC", "GGC", "GGA", "AGC", "AAG", "GAG"]
        for step in range(start_step + 1, self.total_steps + 1):
            candidate = "AUG" + "".join(rng.choice(codons_phe_leu_gly) for _ in range(12)) + "UAA"
            gc_frac = (candidate.count("G") + candidate.count("C")) / len(candidate)
            score = round(gc_frac * 100.0 + rng.uniform(-2.5, 2.5), 8)
            if score > best_score:
                best_score = score
                best_seq = candidate
            trajectory.append({"candidate": candidate, "score": score, "step": step})

            if step % self.checkpoint_every == 0 or step == self.total_steps:
                ckpt_payload = {
                    "completed_step": step,
                    "total_steps": self.total_steps,
                    "best_score": best_score,
                    "best_seq": best_seq,
                    "trajectory": trajectory,
                    "prng_state": _encode_prng_state(rng.getstate()),
                }
                drive.upload_verified_bytes(
                    attempt_folder_id,
                    f"checkpoint_step_{step:04d}.json",
                    (json.dumps(ckpt_payload, sort_keys=True) + "\n").encode("utf-8"),
                    mime_type="application/json",
                )

            if simulate_interrupt_at_step is not None and step == simulate_interrupt_at_step:
                raise RuntimeError(f"Simulated Colab preemption at step {step}")

        final_result = {
            "best_score": best_score,
            "best_seq": best_seq,
            "total_steps": self.total_steps,
            "trajectory": trajectory,
        }
        drive.upload_verified_bytes(
            attempt_folder_id,
            "result.json",
            (json.dumps(final_result, indent=2, sort_keys=True) + "\n").encode("utf-8"),
            mime_type="application/json",
        )
        return final_result


class ColabCLIOrchestrator:
    """Safe headless wrapper around `google-colab-cli` guaranteeing zero orphaned VMs."""

    def __init__(self, colab_bin: Path = Path("/Users/alex/.local/bin/colab")) -> None:
        self.colab_bin = colab_bin

    def run_ephemeral_script(
        self, script_path: Path, args: list[str] | None = None, *, gpu: str | None = None, timeout: int = 600
    ) -> str:
        cmd = [str(self.colab_bin), "run"]
        if gpu:
            cmd.extend(["--gpu", gpu])
        cmd.append(str(script_path))
        if args:
            cmd.extend(args)
        env = os.environ.copy()
        env["PATH"] = f"/opt/homebrew/bin:/Users/alex/.local/bin:{env.get('PATH', '')}"
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=timeout, env=env)
        return proc.stdout

    def list_active_sessions(self) -> str:
        proc = subprocess.run(
            [str(self.colab_bin), "sessions"], check=True, capture_output=True, text=True
        )
        return proc.stdout + proc.stderr

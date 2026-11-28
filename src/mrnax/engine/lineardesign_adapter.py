"""LinearDesign Lattice-CFG Adapter + Post-Pass 5'-Opening & Constraint Surgical Polisher.

Track Isolation Contract:
- Strictly restricted to `research_max` / `sota_open` profiles (`needs_vendor_license`).
- NEVER executed in `commercial_clean`; any attempt raises `PermissionError` immediately.
"""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import contextvars
from dataclasses import dataclass
import functools
import math
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from typing import Any

from Bio.Seq import Seq

from mrnax.constraints.sequence_gates import RESTRICTION_SITES_RNA
from mrnax.engine.stage0_baseline import (
    AA_TO_SYNONYMOUS_RNA_CODONS,
    generate_random_synonymous_cds,
    propose_non_partitioned_bicodon_cds,
)
from mrnax.engine.surrogates import CODON_W
from mrnax.schemas.common import UridineChemistry
from mrnax.schemas.design_intent import ConstraintSpec

DEFAULT_LAMBDA_SWEEP: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 4.0)
DEFAULT_POLISH_ENZYMES: tuple[str, ...] = ("BsaI", "BspQI", "BsmBI", "SapI", "EcoRI")
DEFAULT_MAX_LINEARDESIGN_AA: int = 900
DEFAULT_LINEARDESIGN_TIMEOUT_SEC: float = 45.0

_WAVE1B_DARWIN_BIN_DIR = Path(
    "/Users/alex/.gemini/antigravity/brain/4db4d24c-55e0-4968-869d-a93a1c9c7b7b/"
    "scratch/wave1_sota_models/bin/darwin_arm64"
)
_WAVE1B_LINUX_BIN_DIR = Path(
    "/Users/alex/.gemini/antigravity/brain/4db4d24c-55e0-4968-869d-a93a1c9c7b7b/"
    "scratch/wave1_sota_models/bin/linux_x86_64"
)

_SEQ_LINE_RE = re.compile(r"(?:^|[\r\n])[ \t]*mRNA sequence:[ \t]*([ACGUacgu]+)[ \t]*(?=\r?\n|$)")
_ENERGY_CAI_RE = re.compile(
    r"mRNA folding free energy:\s*([-+]?\d+(?:\.\d+)?)\s*kcal/mol;\s*mRNA CAI:\s*([-+]?\d+(?:\.\d+)?)"
)
_STANDARD_AA_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")
_CLASS_II_T7_RNA = "AUCUGUU"


def _assert_research_track_allowed(compliance_track: str) -> None:
    if compliance_track not in ("research_max", "sota_open"):
        raise PermissionError(
            f"LinearDesign is strictly isolated from '{compliance_track}'; "
            "only explicit 'research_max' or 'sota_open' tracks are permitted."
        )


def _try_vienna() -> Any | None:
    try:
        import RNA  # type: ignore[import-untyped]

        return RNA
    except Exception:
        return None


def compute_human_cai(cds_rna: str) -> float:
    dna = cds_rna.upper().replace("U", "T")
    n_cod = len(dna) // 3
    logs = [
        math.log(CODON_W[dna[3 * i : 3 * i + 3]])
        for i in range(n_cod)
        if dna[3 * i : 3 * i + 3] in CODON_W
    ]
    return round(math.exp(sum(logs) / len(logs)), 4) if logs else 0.5


def _compute_folding_dg_kcal(seq_rna: str) -> float:
    rna_mod = _try_vienna()
    if rna_mod is None or not seq_rna:
        gc = seq_rna.count("G") + seq_rna.count("C")
        return round(-0.45 * gc, 3)
    n_len = len(seq_rna)
    if n_len <= 2800:
        _, dg_val = rna_mod.fold(seq_rna)
        return round(float(dg_val), 3)
    win = 400
    step = 200
    starts = list(range(0, n_len - win + 1, step))
    last_covered = starts[-1] + win if starts else 0
    weighted_sum = 0.0
    total_weight = 0.0
    for s_idx in starts:
        ch = seq_rna[s_idx : s_idx + win]
        weighted_sum += float(rna_mod.fold(ch)[1])
        total_weight += float(win)
    if last_covered < n_len:
        rem = n_len - last_covered
        tail_ch = seq_rna[n_len - win :]
        weighted_sum += float(rna_mod.fold(tail_ch)[1]) * (rem / win)
        total_weight += float(rem)
    return round(weighted_sum * (n_len / max(1.0, total_weight)), 3)


def compute_5prime_initiation_openness(
    cds_rna: str,
    *,
    utr5_rna: str = "",
    upstream_nt: int = 15,
    downstream_nt: int = 30,
) -> float:
    """Compute mean ensemble unpaired probability across `[-upstream_nt, +downstream_nt]` around AUG."""
    u5_tail = utr5_rna.upper().replace("T", "U")[-35:]
    cds_head = cds_rna.upper().replace("T", "U")[: max(55, downstream_nt + 20)]
    ctx = u5_tail + cds_head
    if len(ctx) < 12:
        return 0.5
    rel_aug = len(u5_tail)
    lo = max(0, rel_aug - upstream_nt)
    hi = min(len(ctx), rel_aug + downstream_nt)
    rna_mod = _try_vienna()
    if rna_mod is not None:
        w_size = min(len(ctx), 70)
        l_span = min(w_size, 50)
        up = rna_mod.pfl_fold_up(ctx, 1, w_size, l_span)
        probs = [float(up[pos + 1][1]) for pos in range(lo, hi)]
        return round(sum(probs) / max(1, len(probs)), 6)
    window_slice = ctx[lo:hi]
    gc_frac = (window_slice.count("G") + window_slice.count("C")) / max(1, len(window_slice))
    return round(max(0.05, min(0.95, 1.0 - 0.85 * gc_frac)), 6)


_LOCAL_HEAVY_BINARY_DISABLED_CTX: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "mrnax_disable_local_heavy_binaries", default=False
)


def local_heavy_binaries_disabled() -> bool:
    """Return True when local external heavy C++ binaries (LinearDesign/EternaFold) are disabled by policy."""
    if _LOCAL_HEAVY_BINARY_DISABLED_CTX.get():
        return True
    raw = os.environ.get("MRNAX_DISABLE_LOCAL_HEAVY_BINARIES", "").strip().lower()
    return raw in ("1", "true", "yes")


@contextmanager
def enforce_local_heavy_binaries_policy(disabled: bool = True) -> Iterator[None]:
    """Temporarily enforce local external heavy-binary policy across gateway calls."""
    token = _LOCAL_HEAVY_BINARY_DISABLED_CTX.set(bool(disabled))
    try:
        yield
    finally:
        _LOCAL_HEAVY_BINARY_DISABLED_CTX.reset(token)


def discover_lineardesign_binary() -> tuple[Path, Path] | None:
    """Discover executable `lineardesign` binary and runtime table directory (`coding_wheel.txt` + CSV)."""
    if local_heavy_binaries_disabled():
        return None
    candidates: list[Path] = []
    env_bin = os.environ.get("MRNAX_LINEARDESIGN_BIN", "").strip()
    if env_bin:
        candidates.append(Path(env_bin).expanduser())
    candidates.append(Path.home() / ".mrnax" / "bin" / "lineardesign")
    candidates.append(_WAVE1B_DARWIN_BIN_DIR / "lineardesign")
    candidates.append(_WAVE1B_LINUX_BIN_DIR / "lineardesign")
    which_hit = shutil.which("lineardesign")
    if which_hit:
        candidates.append(Path(which_hit))

    runtime_dirs = [
        Path.home() / ".mrnax" / "bin",
        _WAVE1B_DARWIN_BIN_DIR,
        _WAVE1B_LINUX_BIN_DIR,
    ]

    for bin_cand in candidates:
        if not bin_cand.is_file() or not os.access(bin_cand, os.X_OK):
            continue
        search_dirs = [bin_cand.parent, *runtime_dirs]
        for rdir in search_dirs:
            if (rdir / "coding_wheel.txt").is_file() and (
                rdir / "codon_usage_freq_table_human.csv"
            ).is_file():
                return bin_cand.resolve(), rdir.resolve()
    return None


def is_lineardesign_available() -> bool:
    return discover_lineardesign_binary() is not None


def _normalize_sense_protein(protein_seq: str) -> str:
    cleaned = protein_seq.strip().upper()
    if cleaned.endswith("*"):
        cleaned = cleaned[:-1]
    if not cleaned or not _STANDARD_AA_RE.fullmatch(cleaned):
        raise ValueError(
            "Protein sequence must be non-empty and contain only canonical 20 standard amino acids."
        )
    return cleaned


def verify_exact_aa_translation(cds_rna: str, expected_protein: str, *, transl_table: int = 1) -> bool:
    rna = cds_rna.strip().upper().replace("T", "U")
    if len(rna) != 3 * len(expected_protein):
        return False
    translated = str(Seq(rna).translate(table=transl_table, to_stop=False))
    return translated == expected_protein


_MAX_LD_STDOUT_BYTES = 262_144
_MAX_LD_STDERR_BYTES = 65_536
_MAX_LD_TIMEOUT_SEC = 120.0
_HARD_MAX_LINEARDESIGN_AA = 5000


def _validate_lambda_val(lambda_val: float) -> float:
    if isinstance(lambda_val, bool) or not isinstance(lambda_val, (int, float)):
        raise ValueError("lambda_val must be a finite numeric value")
    val = float(lambda_val)
    if not math.isfinite(val) or val < 0.0 or val > 100.0:
        raise ValueError(f"lambda_val ({lambda_val!r}) must be finite and within [0.0, 100.0]")
    return val


def _validate_timeout_sec(timeout_sec: float) -> float:
    if isinstance(timeout_sec, bool) or not isinstance(timeout_sec, (int, float)):
        raise ValueError("timeout_sec must be a finite positive number")
    val = float(timeout_sec)
    if not math.isfinite(val) or val <= 0.0:
        raise ValueError(f"timeout_sec ({timeout_sec!r}) must be finite and > 0")
    return min(max(0.1, val), _MAX_LD_TIMEOUT_SEC)


_LD_GATE_COND = threading.Condition()
_LD_ACTIVE_SLOTS = 0
_LD_SEM_STATE = threading.local()


def resolve_lineardesign_max_workers() -> int:
    """Resolve bounded LinearDesign worker concurrency from `MRNAX_LINEARDESIGN_MAX_WORKERS` (clamped `[1, 8]`)."""
    raw = os.environ.get("MRNAX_LINEARDESIGN_MAX_WORKERS", "").strip()
    if raw:
        try:
            return min(8, max(1, int(raw)))
        except ValueError:
            pass
    return min(5, max(1, os.cpu_count() or 1))


@contextmanager
def _lineardesign_concurrency_slot() -> Iterator[None]:
    global _LD_ACTIVE_SLOTS
    depth = int(getattr(_LD_SEM_STATE, "depth", 0))
    if depth > 0:
        _LD_SEM_STATE.depth = depth + 1
        try:
            yield
        finally:
            _LD_SEM_STATE.depth = depth
        return
    with _LD_GATE_COND:
        while _LD_ACTIVE_SLOTS >= resolve_lineardesign_max_workers():
            _LD_GATE_COND.wait()
        _LD_ACTIVE_SLOTS += 1
    _LD_SEM_STATE.depth = 1
    try:
        yield
    finally:
        _LD_SEM_STATE.depth = 0
        with _LD_GATE_COND:
            _LD_ACTIVE_SLOTS = max(0, _LD_ACTIVE_SLOTS - 1)
            _LD_GATE_COND.notify_all()


def _build_lineardesign_subprocess_env(argv: list[str]) -> dict[str, str]:
    """Construct a minimal secret-free environment; prepend binary dir to `LD_LIBRARY_PATH` on Linux."""
    env: dict[str, str] = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "LC_ALL": "C",
        "LANG": "C",
    }
    if sys.platform.startswith("linux") and argv:
        bin_dir = str(Path(argv[0]).resolve().parent)
        existing_ld = os.environ.get("LD_LIBRARY_PATH", "").strip()
        env["LD_LIBRARY_PATH"] = f"{bin_dir}:{existing_ld}" if existing_ld else bin_dir
    return env


def _run_bounded_subprocess(
    argv: list[str],
    *,
    stdin_text: str,
    cwd_str: str,
    timeout_sec: float,
    max_stdout_bytes: int = _MAX_LD_STDOUT_BYTES,
    max_stderr_bytes: int = _MAX_LD_STDERR_BYTES,
) -> str:
    import selectors

    with _lineardesign_concurrency_slot():
        deadline = time.monotonic() + timeout_sec
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd_str,
            env=_build_lineardesign_subprocess_env(argv),
        )
        out_chunks: list[bytes] = []
        err_chunks: list[bytes] = []
        out_len = 0
        err_len = 0
        in_buf = memoryview(stdin_text.encode("ascii"))
        in_written = 0
        try:
            assert proc.stdin is not None and proc.stdout is not None and proc.stderr is not None
            os.set_blocking(proc.stdin.fileno(), False)
            os.set_blocking(proc.stdout.fileno(), False)
            os.set_blocking(proc.stderr.fileno(), False)
            with selectors.DefaultSelector() as sel:
                sel.register(proc.stdin, selectors.EVENT_WRITE, data="in")
                sel.register(proc.stdout, selectors.EVENT_READ, data="out")
                sel.register(proc.stderr, selectors.EVENT_READ, data="err")
                while sel.get_map():
                    rem = deadline - time.monotonic()
                    if rem <= 0.0:
                        raise subprocess.TimeoutExpired(argv, timeout_sec)
                    events = sel.select(timeout=rem)
                    if not events:
                        raise subprocess.TimeoutExpired(argv, timeout_sec)
                    for key, _ in events:
                        stream = key.fileobj
                        if key.data == "in":
                            try:
                                nw = os.write(stream.fileno(), in_buf[in_written : min(len(in_buf), in_written + 16384)])  # type: ignore[union-attr]
                                in_written += nw
                            except BlockingIOError:
                                continue
                            if in_written >= len(in_buf):
                                sel.unregister(stream)
                                proc.stdin.close()
                            continue
                        try:
                            chunk = os.read(stream.fileno(), 16384)  # type: ignore[union-attr]
                        except BlockingIOError:
                            continue
                        if not chunk:
                            sel.unregister(stream)
                            continue
                        if key.data == "out":
                            out_len += len(chunk)
                            if out_len > max_stdout_bytes:
                                raise RuntimeError("LinearDesign stdout exceeded byte cap")
                            out_chunks.append(chunk)
                        else:
                            err_len += len(chunk)
                            if err_len > max_stderr_bytes:
                                raise RuntimeError("LinearDesign stderr exceeded byte cap")
                            err_chunks.append(chunk)
            rem_wait = max(0.05, deadline - time.monotonic())
            rc = proc.wait(timeout=rem_wait)
        except Exception:
            try:
                proc.kill()
                proc.wait(timeout=1.0)
            except Exception:
                pass
            raise
        stdout_text = b"".join(out_chunks).decode("utf-8", errors="replace")
        stderr_text = b"".join(err_chunks).decode("utf-8", errors="replace")
        if rc != 0:
            raise subprocess.CalledProcessError(rc, argv, output=stdout_text, stderr=stderr_text)
        return stdout_text


@functools.lru_cache(maxsize=256)
def _cached_cpp_lineardesign_call(
    bin_path_str: str,
    cwd_str: str,
    sense_aa: str,
    lam_rounded: float,
    timeout_sec: float,
) -> tuple[str, float, float]:
    with _lineardesign_concurrency_slot():
        stdout_text = _run_bounded_subprocess(
            [bin_path_str, "-l", f"{lam_rounded:g}"],
            stdin_text=f"{sense_aa}\n",
            cwd_str=cwd_str,
            timeout_sec=timeout_sec,
        )
    seq_m = _SEQ_LINE_RE.search(stdout_text)
    ec_m = _ENERGY_CAI_RE.search(stdout_text)
    if seq_m is None:
        raise RuntimeError(f"LinearDesign stdout missing valid 'mRNA sequence:' line: {stdout_text[-300:]!r}")
    cds_rna = seq_m.group(1).upper().replace("T", "U")
    if len(cds_rna) != 3 * len(sense_aa):
        raise RuntimeError("LinearDesign stdout sequence length mismatch")
    if ec_m is not None:
        dg_kcal = float(ec_m.group(1))
        cai_val = float(ec_m.group(2))
        if not math.isfinite(dg_kcal) or not math.isfinite(cai_val) or not (0.0 < cai_val <= 1.0):
            raise RuntimeError(f"LinearDesign stdout reported invalid energy/CAI ({dg_kcal}, {cai_val})")
    else:
        dg_kcal = _compute_folding_dg_kcal(cds_rna)
        cai_val = compute_human_cai(cds_rna)
    return cds_rna, dg_kcal, cai_val


def _fallback_guided_cds(
    sense_aa: str,
    *,
    lambda_val: float,
    chemistry: UridineChemistry = "m1psi",
    constraints: list[ConstraintSpec] | None = None,
    seed: int = 42,
) -> tuple[str, float, float]:
    """Deterministic guided bicodon + ViennaRNA window energy fallback when C++ binary is absent."""
    rng = random.Random(seed + int(round(lambda_val * 1000)))
    init_cds = generate_random_synonymous_cds(sense_aa, rng, chemistry=chemistry)
    target_gc = min(0.64, 0.52 + 0.022 * min(lambda_val, 5.0))
    trials: list[str] = []
    cur = init_cds
    for step_idx in range(4):
        cur = propose_non_partitioned_bicodon_cds(
            cur,
            sense_aa,
            rng,
            chemistry=chemistry,
            constraints=constraints,
            gc_weight=0.018 + 0.004 * step_idx,
            temperature=max(0.02, 0.09 - 0.015 * min(lambda_val, 4.0)),
            target_gc=target_gc,
            mutation_prob=0.80,
        )
        trials.append(cur)
    best_cds = min(trials, key=_compute_folding_dg_kcal)
    return best_cds, _compute_folding_dg_kcal(best_cds), compute_human_cai(best_cds)


def _resolve_max_lineardesign_aa(max_aa: int | None = None) -> int:
    """Resolve maximum protein length permitted for C++ LinearDesign, never exceeding deployment cap."""
    env_cap = DEFAULT_MAX_LINEARDESIGN_AA
    raw_env = os.environ.get("MRNAX_MAX_LINEARDESIGN_AA", "").strip()
    if raw_env:
        try:
            env_cap = min(_HARD_MAX_LINEARDESIGN_AA, max(1, int(raw_env)))
        except ValueError:
            env_cap = DEFAULT_MAX_LINEARDESIGN_AA
    if max_aa is not None:
        if isinstance(max_aa, bool) or int(max_aa) < 1:
            raise ValueError(f"max_aa ({max_aa!r}) must be a positive integer")
        return min(env_cap, int(max_aa))
    return env_cap


def run_lineardesign_single(
    protein_seq: str,
    *,
    lambda_val: float = 1.0,
    compliance_track: str = "research_max",
    transl_table: int = 1,
    chemistry: UridineChemistry = "m1psi",
    constraints: list[ConstraintSpec] | None = None,
    timeout_sec: float = DEFAULT_LINEARDESIGN_TIMEOUT_SEC,
    max_aa: int | None = None,
    force_fallback: bool = False,
    seed: int = 42,
) -> tuple[str, float, float, str]:
    """Run LinearDesign for a single `lambda_val` with 100% AA verification, max-AA VPS guard, and clean fallback."""
    _assert_research_track_allowed(compliance_track)
    valid_lam = _validate_lambda_val(lambda_val)
    valid_timeout = _validate_timeout_sec(timeout_sec)
    sense_aa = _normalize_sense_protein(protein_seq)
    exceeds_max_aa = len(sense_aa) > _resolve_max_lineardesign_aa(max_aa)
    discovered = None if (force_fallback or exceeds_max_aa) else discover_lineardesign_binary()
    if discovered is not None:
        bin_path, runtime_cwd = discovered
        try:
            cds_rna, dg_kcal, cai_val = _cached_cpp_lineardesign_call(
                str(bin_path),
                str(runtime_cwd),
                sense_aa,
                round(valid_lam, 4),
                valid_timeout,
            )
            backend = "lineardesign_cpp_binary"
        except (subprocess.SubprocessError, OSError, RuntimeError):
            cds_rna, dg_kcal, cai_val = _fallback_guided_cds(
                sense_aa,
                lambda_val=valid_lam,
                chemistry=chemistry,
                constraints=constraints,
                seed=seed,
            )
            backend = "guided_bicodon_vienna_fallback"
    else:
        cds_rna, dg_kcal, cai_val = _fallback_guided_cds(
            sense_aa,
            lambda_val=valid_lam,
            chemistry=chemistry,
            constraints=constraints,
            seed=seed,
        )
        backend = "guided_bicodon_vienna_fallback"

    if not verify_exact_aa_translation(cds_rna, sense_aa, transl_table=transl_table):
        raise RuntimeError(
            f"LinearDesign candidate failed exact amino-acid verification (lambda={lambda_val}, backend={backend})."
        )
    return cds_rna, dg_kcal, cai_val, backend


def build_forbidden_motif_list(
    *,
    max_homopolymer: int = 5,
    eliminate_uuuu: bool = True,
    restriction_enzymes: tuple[str, ...] | list[str] = DEFAULT_POLISH_ENZYMES,
    extra_forbidden_motifs: tuple[str, ...] | list[str] = (),
    constraints: list[ConstraintSpec] | None = None,
) -> tuple[str, ...]:
    """Assemble deduplicated forbidden RNA substrings (`UUUU`, homopolymers >=6, T7, restriction sites)."""
    motifs: list[str] = [_CLASS_II_T7_RNA]
    if eliminate_uuuu:
        motifs.append("UUUU")
    eff_max_run = max_homopolymer
    enz_set = list(restriction_enzymes)
    for cspec in constraints or []:
        if "cds" not in cspec.target_scopes and "full_transcript" not in cspec.target_scopes:
            continue
        if cspec.constraint_kind == "homopolymer_max":
            eff_max_run = min(eff_max_run, int(cspec.params.get("max_run", eff_max_run)))
        elif cspec.constraint_kind == "type_iis_restriction_free":
            enz_set.extend(cspec.params.get("enzymes", []))
            for raw_m in cspec.params.get("forbidden_motifs", []):
                if raw_m:
                    motifs.append(str(raw_m).upper().replace("T", "U"))
    for base in "ACGU":
        motifs.append(base * (eff_max_run + 1))
    for enz in enz_set:
        if enz in RESTRICTION_SITES_RNA:
            fwd, rev = RESTRICTION_SITES_RNA[enz]
            motifs.append(fwd)
            motifs.append(rev)
    for raw_m in extra_forbidden_motifs:
        if raw_m:
            motifs.append(str(raw_m).upper().replace("T", "U"))
    return tuple(dict.fromkeys(m for m in motifs if m))


def count_forbidden_motif_hits(
    cds_rna: str,
    forbidden_motifs: tuple[str, ...],
    *,
    utr5_tail: str = "",
    downstream_head: str = "",
) -> int:
    """Count forbidden motif occurrences overlapping the CDS (including UTR/stop boundary junctions)."""
    u5 = utr5_tail.upper().replace("T", "U")[-12:]
    dn = downstream_head.upper().replace("T", "U")[:12]
    ctx = u5 + cds_rna + dn
    cds_lo = len(u5)
    cds_hi = cds_lo + len(cds_rna)
    hits = 0
    for pat in forbidden_motifs:
        start = 0
        plen = len(pat)
        while True:
            idx = ctx.find(pat, start)
            if idx < 0:
                break
            if idx < cds_hi and (idx + plen) > cds_lo:
                hits += 1
            start = idx + 1
    return hits


def _find_cds_overlapping_spans(
    codons: list[str],
    forbidden_motifs: tuple[str, ...],
    u5_tail: str,
    dn_head: str,
) -> list[tuple[int, int]]:
    cds_rna = "".join(codons)
    ctx = u5_tail + cds_rna + dn_head
    cds_lo = len(u5_tail)
    cds_hi = cds_lo + len(cds_rna)
    n_cod = len(codons)
    spans: list[tuple[int, int]] = []
    for pat in forbidden_motifs:
        start = 0
        plen = len(pat)
        while True:
            idx = ctx.find(pat, start)
            if idx < 0:
                break
            ov_lo = max(0, idx - cds_lo)
            ov_hi = min(len(cds_rna) - 1, idx + plen - 1 - cds_lo)
            if ov_lo <= ov_hi and idx < cds_hi and (idx + plen) > cds_lo:
                c_lo = max(0, ov_lo // 3)
                c_hi = min(n_cod - 1, ov_hi // 3)
                spans.append((c_lo, c_hi))
            start = idx + 1
    spans.sort()
    return spans


def _eliminate_forbidden_motifs_synonymously(
    codons: list[str],
    sense_aa: str,
    forbidden_motifs: tuple[str, ...],
    u5_tail: str,
    dn_head: str,
) -> list[str]:
    """Eliminate forbidden motifs via minimal synonymous codon edits preserving local folding stability."""
    work = list(codons)
    n_cod = len(work)
    for _pass in range(6):
        spans = _find_cds_overlapping_spans(work, forbidden_motifs, u5_tail, dn_head)
        if not spans:
            break
        changed_any = False
        for c_lo, c_hi in spans:
            win_lo = max(0, c_lo - 4)
            win_hi = min(n_cod, c_hi + 5)
            left_ctx = (u5_tail if win_lo == 0 else "") + "".join(work[win_lo:c_lo])
            right_ctx = "".join(work[c_hi + 1 : win_hi]) + (dn_head if win_hi == n_cod else "")
            cur_mid = "".join(work[c_lo : c_hi + 1])
            cur_sub = left_ctx + cur_mid + right_ctx
            if not any(p in cur_sub for p in forbidden_motifs):
                continue

            # 1. Try all single-codon synonymous edits in [c_lo .. c_hi]
            single_candidates: list[tuple[int, float, float, int, str]] = []
            for pos in range(c_lo, c_hi + 1):
                aa = sense_aa[pos]
                for syn in AA_TO_SYNONYMOUS_RNA_CODONS.get(aa, (work[pos],)):
                    if syn == work[pos]:
                        continue
                    trial_mid = "".join(
                        syn if k == pos else work[k] for k in range(c_lo, c_hi + 1)
                    )
                    trial_sub = left_ctx + trial_mid + right_ctx
                    rem_v = sum(trial_sub.count(p) for p in forbidden_motifs)
                    local_dg = _compute_folding_dg_kcal(trial_sub)
                    cw = CODON_W.get(syn.replace("U", "T"), 0.5)
                    single_candidates.append((rem_v, local_dg, -cw, pos, syn))

            clean_singles = [x for x in single_candidates if x[0] == 0]
            if clean_singles:
                best = min(clean_singles)
                work[best[3]] = best[4]
                changed_any = True
                continue

            # 2. Try 2-codon synonymous pairs across [c_lo .. min(c_hi, c_lo + 3)]
            pair_candidates: list[tuple[int, float, float, int, str, int, str]] = []
            hi_pair = min(c_hi + 1, c_lo + 4)
            for p1 in range(c_lo, hi_pair):
                syns1 = AA_TO_SYNONYMOUS_RNA_CODONS.get(sense_aa[p1], (work[p1],))
                for p2 in range(p1 + 1, hi_pair):
                    syns2 = AA_TO_SYNONYMOUS_RNA_CODONS.get(sense_aa[p2], (work[p2],))
                    for s1 in syns1:
                        for s2 in syns2:
                            if s1 == work[p1] and s2 == work[p2]:
                                continue
                            trial_mid = "".join(
                                s1 if k == p1 else (s2 if k == p2 else work[k])
                                for k in range(c_lo, c_hi + 1)
                            )
                            trial_sub = left_ctx + trial_mid + right_ctx
                            rem_v = sum(trial_sub.count(p) for p in forbidden_motifs)
                            if rem_v == 0:
                                local_dg = _compute_folding_dg_kcal(trial_sub)
                                cw = CODON_W.get(s1.replace("U", "T"), 0.5) + CODON_W.get(
                                    s2.replace("U", "T"), 0.5
                                )
                                pair_candidates.append((0, local_dg, -cw, p1, s1, p2, s2))
            if pair_candidates:
                best_p = min(pair_candidates)
                work[best_p[3]] = best_p[4]
                work[best_p[5]] = best_p[6]
                changed_any = True
                continue

            # 3. Partial reduction fallback if spanning >=3 codons
            if single_candidates:
                best_part = min(single_candidates)
                cur_v = sum(cur_sub.count(p) for p in forbidden_motifs)
                if best_part[0] < cur_v:
                    work[best_part[3]] = best_part[4]
                    changed_any = True
        if not changed_any:
            break
    return work


def _polish_5prime_initiation_openness(
    codons: list[str],
    sense_aa: str,
    *,
    utr5_rna: str,
    forbidden_motifs: tuple[str, ...],
    polish_codons: int = 13,
) -> list[str]:
    """Optimize synonymous codons in first 10-15 codons (`[-15, +30]` nt) to improve initiation openness."""
    work = list(codons)
    n_cod = len(work)
    if n_cod <= 1 or polish_codons <= 0:
        return work

    u5_tail_short = utr5_rna.upper().replace("T", "U")[-12:]
    hi_cod = min(n_cod, polish_codons + 1)
    head_len_codons = min(n_cod, 20)

    def _eval_open(cand_codons: list[str]) -> float:
        head_rna = "".join(cand_codons[:head_len_codons])
        return compute_5prime_initiation_openness(
            head_rna, utr5_rna=utr5_rna, upstream_nt=15, downstream_nt=30
        )

    best_open = _eval_open(work)
    for _sweep in range(2):
        improved = False
        for idx in range(1, hi_cod):
            aa = sense_aa[idx]
            choices = AA_TO_SYNONYMOUS_RNA_CODONS.get(aa, (work[idx],))
            if len(choices) <= 1:
                continue
            viable = [c for c in choices if CODON_W.get(c.replace("U", "T"), 0.5) >= 0.25] or list(
                choices
            )
            best_c = work[idx]
            best_cw = CODON_W.get(best_c.replace("U", "T"), 0.5)
            for cand_c in viable:
                if cand_c == work[idx]:
                    continue
                old_c = work[idx]
                work[idx] = cand_c
                local_check = u5_tail_short + "".join(work[: min(n_cod, hi_cod + 4)])
                if any(pat in local_check for pat in forbidden_motifs):
                    work[idx] = old_c
                    continue
                cand_open = _eval_open(work)
                cand_cw = CODON_W.get(cand_c.replace("U", "T"), 0.5)
                if cand_open > best_open + 1e-4 or (
                    abs(cand_open - best_open) <= 1e-4 and cand_cw > best_cw + 0.05
                ):
                    best_open = cand_open
                    best_c = cand_c
                    best_cw = cand_cw
                    improved = True
                work[idx] = old_c
            work[idx] = best_c
        if not improved:
            break
    return work


def polish_cds_5prime_and_constraints(
    cds_rna: str,
    protein_seq: str,
    *,
    utr5_rna: str = "",
    stop_rna: str = "UGA",
    utr3_rna: str = "",
    compliance_track: str = "research_max",
    transl_table: int = 1,
    max_homopolymer: int = 5,
    eliminate_uuuu: bool = True,
    restriction_enzymes: tuple[str, ...] | list[str] = DEFAULT_POLISH_ENZYMES,
    extra_forbidden_motifs: tuple[str, ...] | list[str] = (),
    constraints: list[ConstraintSpec] | None = None,
    polish_5prime_codons: int = 13,
) -> tuple[str, int]:
    """Apply post-pass 5'-Opening & Restriction/Slippery/Homopolymer Surgical Polisher."""
    _assert_research_track_allowed(compliance_track)
    sense_aa = _normalize_sense_protein(protein_seq)
    norm_cds = cds_rna.strip().upper().replace("T", "U")
    if not verify_exact_aa_translation(norm_cds, sense_aa, transl_table=transl_table):
        raise ValueError("Input CDS does not translate to expected protein_seq before polishing.")

    orig_codons = [norm_cds[3 * i : 3 * i + 3] for i in range(len(sense_aa))]
    forbidden = build_forbidden_motif_list(
        max_homopolymer=max_homopolymer,
        eliminate_uuuu=eliminate_uuuu,
        restriction_enzymes=restriction_enzymes,
        extra_forbidden_motifs=extra_forbidden_motifs,
        constraints=constraints,
    )
    u5_tail = utr5_rna.upper().replace("T", "U")[-12:]
    dn_head = (stop_rna + utr3_rna).upper().replace("T", "U")[:12]

    # Step 1: Remove forbidden motifs across full CDS & boundaries with minimal synonymous edits
    step1_codons = _eliminate_forbidden_motifs_synonymously(
        orig_codons, sense_aa, forbidden, u5_tail, dn_head
    )
    # Step 2: Refine 5' initiation codons [1..polish_5prime_codons] for higher unpaired accessibility
    step2_codons = _polish_5prime_initiation_openness(
        step1_codons,
        sense_aa,
        utr5_rna=utr5_rna,
        forbidden_motifs=forbidden,
        polish_codons=polish_5prime_codons,
    )
    # Step 3: Final verification pass ensuring zero boundary/motif regressions
    final_codons = _eliminate_forbidden_motifs_synonymously(
        step2_codons, sense_aa, forbidden, u5_tail, dn_head
    )
    polished_rna = "".join(final_codons)
    if not verify_exact_aa_translation(polished_rna, sense_aa, transl_table=transl_table):
        raise RuntimeError("Polished CDS failed exact amino-acid translation verification.")

    changed_codons = sum(
        1 for a, b in zip(orig_codons, final_codons, strict=True) if a != b
    )
    return polished_rna, changed_codons


@dataclass(frozen=True)
class LinearDesignCandidate:
    lambda_val: float
    raw_cds_rna: str
    polished_cds_rna: str
    protein_sequence: str
    aa_exact_match: bool
    raw_dg_kcal_mol: float
    polished_dg_kcal_mol: float
    raw_cai: float
    polished_cai: float
    raw_kozak_unpaired_prob: float
    polished_kozak_unpaired_prob: float
    synonymous_codons_changed: int
    eliminated_motifs_before: int
    remaining_forbidden_motifs: int
    backend_used: str
    wall_seconds: float

    @property
    def raw_mfe_kcal_mol(self) -> float:
        return self.raw_dg_kcal_mol

    @property
    def polished_mfe_kcal_mol(self) -> float:
        return self.polished_dg_kcal_mol

    def to_dict(self) -> dict[str, Any]:
        return {
            "lambda": self.lambda_val,
            "raw_cds_rna": self.raw_cds_rna,
            "polished_cds_rna": self.polished_cds_rna,
            "aa_length": len(self.protein_sequence),
            "aa_exact_match": self.aa_exact_match,
            "raw_mfe_kcal_mol": self.raw_dg_kcal_mol,
            "polished_mfe_kcal_mol": self.polished_dg_kcal_mol,
            "raw_cai": self.raw_cai,
            "polished_cai": self.polished_cai,
            "raw_kozak_unpaired_prob": self.raw_kozak_unpaired_prob,
            "polished_kozak_unpaired_prob": self.polished_kozak_unpaired_prob,
            "synonymous_codons_changed": self.synonymous_codons_changed,
            "eliminated_motifs_before": self.eliminated_motifs_before,
            "remaining_forbidden_motifs": self.remaining_forbidden_motifs,
            "backend_used": self.backend_used,
            "wall_seconds": self.wall_seconds,
        }


def sweep_lineardesign_candidates(
    protein_seq: str,
    *,
    lambdas: tuple[float, ...] | list[float] = DEFAULT_LAMBDA_SWEEP,
    utr5_rna: str = "",
    stop_rna: str = "UGA",
    utr3_rna: str = "",
    compliance_track: str = "research_max",
    transl_table: int = 1,
    chemistry: UridineChemistry = "m1psi",
    max_homopolymer: int = 5,
    eliminate_uuuu: bool = True,
    restriction_enzymes: tuple[str, ...] | list[str] = DEFAULT_POLISH_ENZYMES,
    extra_forbidden_motifs: tuple[str, ...] | list[str] = (),
    constraints: list[ConstraintSpec] | None = None,
    polish_5prime_codons: int = 13,
    max_aa: int | None = None,
    force_fallback: bool = False,
    seed: int = 42,
    timeout_sec: float = DEFAULT_LINEARDESIGN_TIMEOUT_SEC,
) -> list[LinearDesignCandidate]:
    """Sweep `lambda` across `{0.0, 0.5, 1.0, 2.0, 4.0}` in parallel and polish each candidate."""
    _assert_research_track_allowed(compliance_track)
    sense_aa = _normalize_sense_protein(protein_seq)
    lam_tuple = tuple(_validate_lambda_val(x) for x in lambdas)
    if not lam_tuple:
        return []

    forbidden = build_forbidden_motif_list(
        max_homopolymer=max_homopolymer,
        eliminate_uuuu=eliminate_uuuu,
        restriction_enzymes=restriction_enzymes,
        extra_forbidden_motifs=extra_forbidden_motifs,
        constraints=constraints,
    )
    dn_head = stop_rna + utr3_rna

    def _run_one(lam: float) -> LinearDesignCandidate:
        t0 = time.perf_counter()
        raw_cds, raw_dg, raw_cai, backend = run_lineardesign_single(
            sense_aa,
            lambda_val=lam,
            compliance_track=compliance_track,
            transl_table=transl_table,
            chemistry=chemistry,
            constraints=constraints,
            timeout_sec=timeout_sec,
            max_aa=max_aa,
            force_fallback=force_fallback,
            seed=seed,
        )
        before_hits = count_forbidden_motif_hits(
            raw_cds, forbidden, utr5_tail=utr5_rna, downstream_head=dn_head
        )
        raw_open = compute_5prime_initiation_openness(
            raw_cds, utr5_rna=utr5_rna, upstream_nt=15, downstream_nt=30
        )
        polished_cds, n_changed = polish_cds_5prime_and_constraints(
            raw_cds,
            sense_aa,
            utr5_rna=utr5_rna,
            stop_rna=stop_rna,
            utr3_rna=utr3_rna,
            compliance_track=compliance_track,
            transl_table=transl_table,
            max_homopolymer=max_homopolymer,
            eliminate_uuuu=eliminate_uuuu,
            restriction_enzymes=restriction_enzymes,
            extra_forbidden_motifs=extra_forbidden_motifs,
            constraints=constraints,
            polish_5prime_codons=polish_5prime_codons,
        )
        after_hits = count_forbidden_motif_hits(
            polished_cds, forbidden, utr5_tail=utr5_rna, downstream_head=dn_head
        )
        pol_dg = _compute_folding_dg_kcal(polished_cds)
        pol_cai = compute_human_cai(polished_cds)
        pol_open = compute_5prime_initiation_openness(
            polished_cds, utr5_rna=utr5_rna, upstream_nt=15, downstream_nt=30
        )
        aa_ok = verify_exact_aa_translation(polished_cds, sense_aa, transl_table=transl_table)
        return LinearDesignCandidate(
            lambda_val=lam,
            raw_cds_rna=raw_cds,
            polished_cds_rna=polished_cds,
            protein_sequence=sense_aa,
            aa_exact_match=aa_ok,
            raw_dg_kcal_mol=round(raw_dg, 3),
            polished_dg_kcal_mol=round(pol_dg, 3),
            raw_cai=round(raw_cai, 4),
            polished_cai=round(pol_cai, 4),
            raw_kozak_unpaired_prob=raw_open,
            polished_kozak_unpaired_prob=pol_open,
            synonymous_codons_changed=n_changed,
            eliminated_motifs_before=before_hits,
            remaining_forbidden_motifs=after_hits,
            backend_used=backend,
            wall_seconds=round(time.perf_counter() - t0, 4),
        )

    max_w = min(resolve_lineardesign_max_workers(), len(lam_tuple))
    if max_w <= 1:
        return [_run_one(lam) for lam in lam_tuple]
    with ThreadPoolExecutor(max_workers=max_w) as pool:
        futs = [pool.submit(contextvars.copy_context().run, _run_one, lam) for lam in lam_tuple]
        return [f.result() for f in futs]


__all__ = [
    "DEFAULT_LAMBDA_SWEEP",
    "DEFAULT_LINEARDESIGN_TIMEOUT_SEC",
    "DEFAULT_MAX_LINEARDESIGN_AA",
    "DEFAULT_POLISH_ENZYMES",
    "LinearDesignCandidate",
    "build_forbidden_motif_list",
    "compute_5prime_initiation_openness",
    "compute_human_cai",
    "count_forbidden_motif_hits",
    "discover_lineardesign_binary",
    "is_lineardesign_available",
    "polish_cds_5prime_and_constraints",
    "resolve_lineardesign_max_workers",
    "run_lineardesign_single",
    "sweep_lineardesign_candidates",
    "verify_exact_aa_translation",
]

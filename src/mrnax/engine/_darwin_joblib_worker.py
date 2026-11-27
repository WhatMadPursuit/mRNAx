"""Process-isolated `.joblib` (LightGBM / scikit-learn) worker for macOS (`darwin`).

Prevents loading multiple incompatible `libomp.dylib` runtimes (Homebrew / scikit-learn / PyTorch)
into the main `mrnax` process while keeping `mrnax/__init__.py` completely free of native preloads
or process-wide `OMP_NUM_THREADS` mutations.
"""

from __future__ import annotations

import atexit
import io
import os
import pickle
import struct
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np


def _write_frame(stream: Any, obj: Any) -> None:
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(struct.pack(">I", len(payload)))
    stream.write(payload)
    stream.flush()


def _read_exact(stream: Any, n: int) -> bytes:
    chunks: list[bytes] = []
    rem = n
    while rem > 0:
        chunk = stream.read(rem)
        if not chunk:
            raise EOFError("Darwin joblib worker closed pipe unexpectedly")
        chunks.append(chunk)
        rem -= len(chunk)
    return b"".join(chunks)


def _read_frame(stream: Any) -> Any:
    hdr = _read_exact(stream, 4)
    (size,) = struct.unpack(">I", hdr)
    return pickle.loads(_read_exact(stream, size))


class _IsolatedTransformProxy:
    def __init__(self, client: _DarwinJoblibClient, key: str) -> None:
        self._client = client
        self._key = key

    def transform(self, X: Any) -> np.ndarray:
        return self._client.call("transform", self._key, np.asarray(X))


class _IsolatedPredictProxy:
    def __init__(self, client: _DarwinJoblibClient, key: str) -> None:
        self._client = client
        self._key = key

    def predict(self, X: Any, **_kwargs: Any) -> np.ndarray:
        arr = np.asarray(X)
        return np.asarray(self._client.call("predict", self._key, arr), dtype=np.float64)


class _DarwinJoblibClient:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proc: subprocess.Popen[bytes] | None = None
        self._loaded_bytes: dict[str, bytes] = {}
        atexit.register(self.close)

    def _ensure_proc(self) -> subprocess.Popen[bytes]:
        if self._proc is not None and self._proc.poll() is None:
            return self._proc
        env = dict(os.environ)
        env["OMP_NUM_THREADS"] = "1"
        src_root = str(Path(__file__).resolve().parents[2])
        prev_py = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{src_root}:{prev_py}" if prev_py else src_root
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "mrnax.engine._darwin_joblib_worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            env=env,
        )
        for key, raw in self._loaded_bytes.items():
            assert self._proc.stdin is not None and self._proc.stdout is not None
            _write_frame(self._proc.stdin, ("load", key, raw))
            status, payload = _read_frame(self._proc.stdout)
            if status != "ok":
                raise RuntimeError(f"Darwin joblib worker reload failed: {payload}")
        return self._proc

    def call(self, op: str, *args: Any) -> Any:
        with self._lock:
            proc = self._ensure_proc()
            assert proc.stdin is not None and proc.stdout is not None
            _write_frame(proc.stdin, (op, *args))
            status, payload = _read_frame(proc.stdout)
            if status != "ok":
                raise RuntimeError(f"Darwin joblib worker error ({op}): {payload}")
            return payload

    def load_verified_bytes(self, cache_key: str, raw_bytes: bytes) -> dict[str, Any]:
        meta: dict[str, Any] = self.call("load", cache_key, raw_bytes)
        with self._lock:
            self._loaded_bytes[cache_key] = raw_bytes
        out = dict(meta["primitives"])
        if meta.get("has_scaler"):
            out["scaler"] = _IsolatedTransformProxy(self, cache_key)
        if meta.get("has_model"):
            out["model"] = _IsolatedPredictProxy(self, cache_key)
        return out

    def close(self) -> None:
        with self._lock:
            if self._proc is not None:
                try:
                    if self._proc.stdin is not None and self._proc.poll() is None:
                        _write_frame(self._proc.stdin, ("shutdown",))
                        self._proc.wait(timeout=2)
                except Exception:
                    self._proc.kill()
                finally:
                    self._proc = None


_SINGLETON_CLIENT: _DarwinJoblibClient | None = None
_SINGLETON_LOCK = threading.Lock()


def get_darwin_joblib_client() -> _DarwinJoblibClient:
    global _SINGLETON_CLIENT
    with _SINGLETON_LOCK:
        if _SINGLETON_CLIENT is None:
            _SINGLETON_CLIENT = _DarwinJoblibClient()
        return _SINGLETON_CLIENT


def _worker_main() -> None:
    import joblib

    bundles: dict[str, dict[str, Any]] = {}
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    while True:
        try:
            msg = _read_frame(stdin)
        except EOFError:
            break
        op = msg[0]
        if op == "shutdown":
            break
        try:
            if op == "load":
                _, key, raw_bytes = msg
                loaded = joblib.load(io.BytesIO(raw_bytes))
                if not isinstance(loaded, dict):
                    raise TypeError(f"Expected dict joblib payload for '{key}', got {type(loaded).__name__}")
                mdl = loaded.get("model")
                if hasattr(mdl, "set_params"):
                    try:
                        mdl.set_params(n_jobs=1)
                    except Exception:
                        pass
                bundles[key] = loaded
                primitives = {
                    k: v
                    for k, v in loaded.items()
                    if k not in {"model", "scaler"} and isinstance(v, (str, int, float, bool, list, tuple, dict, type(None)))
                }
                _write_frame(
                    stdout,
                    (
                        "ok",
                        {
                            "primitives": primitives,
                            "has_model": "model" in loaded,
                            "has_scaler": "scaler" in loaded,
                        },
                    ),
                )
            elif op == "transform":
                _, key, arr = msg
                res = bundles[key]["scaler"].transform(arr)
                _write_frame(stdout, ("ok", res))
            elif op == "predict":
                _, key, arr = msg
                mdl = bundles[key]["model"]
                try:
                    res = np.asarray(mdl.predict(arr, num_threads=1), dtype=np.float64)
                except TypeError:
                    res = np.asarray(mdl.predict(arr), dtype=np.float64)
                _write_frame(stdout, ("ok", res))
            else:
                _write_frame(stdout, ("err", f"Unknown op {op!r}"))
        except Exception as exc:
            _write_frame(stdout, ("err", f"{type(exc).__name__}: {exc}"))


if __name__ == "__main__":
    _worker_main()

"""Process-isolated `.joblib` (LightGBM / scikit-learn) worker for macOS (`darwin`).

Prevents loading multiple incompatible `libomp.dylib` runtimes (Homebrew / scikit-learn / PyTorch)
into the main `mrnax` process while keeping `mrnax/__init__.py` completely free of native preloads
or process-wide `OMP_NUM_THREADS` mutations.

Security & reliability invariants:
- BOTH parent -> worker requests AND worker -> parent replies use a strict data-only wire format
  (JSON metadata + raw bytes / typed `<f4`/`<f8` ndarrays; zero IPC `pickle.loads` in either process).
- Every IPC exchange enforces `MAX_IPC_FRAME_BYTES` (32 MiB) and a shared monotonic wall-clock deadline
  across non-blocking request writes AND reply reads, terminating any stalled child before releasing lock.
"""

from __future__ import annotations

import atexit
import io
import json
import os
import selectors
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

MAX_IPC_FRAME_BYTES = 32 * 1024 * 1024
DEFAULT_WORKER_TIMEOUT_SEC = 30.0

_REPLY_JSON = 1
_REPLY_NDARRAY_F64 = 2
_REPLY_ERROR = 3
_REPLY_NDARRAY_F32 = 4

_REQ_SHUTDOWN = 10
_REQ_LOAD = 11
_REQ_NDARRAY = 12
_REQ_STR_LIST = 13


def _to_json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items() if isinstance(k, (str, int))}
    return str(obj)


def _encode_request_frame(req: tuple[Any, ...]) -> bytes:
    op = str(req[0])
    if op == "shutdown":
        body = b""
        kind = _REQ_SHUTDOWN
    elif op == "load":
        _, key, raw_bytes = req
        hdr_json = json.dumps({"op": "load", "key": str(key)}, separators=(",", ":")).encode("utf-8")
        body = struct.pack(">I", len(hdr_json)) + hdr_json + bytes(raw_bytes)
        kind = _REQ_LOAD
    elif op in ("transform", "predict", "predict_attr"):
        if op == "predict_attr":
            _, key, attr_key, payload = req
        else:
            _, key, payload = req
            attr_key = "model"
        if isinstance(payload, list) and (not payload or isinstance(payload[0], str)):
            hdr_json = json.dumps(
                {"op": op, "key": str(key), "attr": str(attr_key), "strings": [str(s) for s in payload]},
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            body = hdr_json
            kind = _REQ_STR_LIST
        else:
            arr = np.asarray(payload)
            if arr.dtype == np.float32:
                typed = np.ascontiguousarray(arr, dtype="<f4")
                dt_tag = "f4"
            else:
                typed = np.ascontiguousarray(arr, dtype="<f8")
                dt_tag = "f8"
            if typed.ndim > 8:
                raise ValueError(f"Unsupported ndarray rank {typed.ndim}")
            hdr_json = json.dumps(
                {
                    "op": op,
                    "key": str(key),
                    "attr": str(attr_key),
                    "dtype": dt_tag,
                    "shape": [int(d) for d in typed.shape],
                },
                separators=(",", ":"),
            ).encode("utf-8")
            body = struct.pack(">I", len(hdr_json)) + hdr_json + typed.tobytes(order="C")
            kind = _REQ_NDARRAY
    else:
        raise ValueError(f"Unsupported Darwin worker op '{op}'")

    total = 1 + len(body)
    if total > MAX_IPC_FRAME_BYTES:
        raise ValueError(f"Darwin worker request frame ({total} bytes) exceeds {MAX_IPC_FRAME_BYTES} cap")
    return struct.pack(">IB", total, kind) + body


def _read_exact_raw(stream: Any, n: int) -> bytes:
    chunks: list[bytes] = []
    rem = n
    while rem > 0:
        chunk = stream.read(rem)
        if not chunk:
            raise EOFError("Darwin joblib worker closed pipe unexpectedly")
        chunks.append(chunk)
        rem -= len(chunk)
    return b"".join(chunks)


def _read_request_frame_in_worker(stream: Any) -> tuple[Any, ...]:
    hdr = _read_exact_raw(stream, 4)
    (size,) = struct.unpack(">I", hdr)
    if size < 1 or size > MAX_IPC_FRAME_BYTES:
        raise ValueError(f"Invalid IPC request frame size ({size} bytes)")
    frame = _read_exact_raw(stream, size)
    kind = frame[0]
    body = memoryview(frame)[1:]
    if kind == _REQ_SHUTDOWN:
        return ("shutdown",)
    if kind == _REQ_LOAD:
        if len(body) < 4:
            raise ValueError("Truncated load request")
        (hlen,) = struct.unpack(">I", bytes(body[:4]))
        if hlen > len(body) - 4:
            raise ValueError("Malformed load request header")
        meta = json.loads(bytes(body[4 : 4 + hlen]).decode("utf-8"))
        raw_bytes = bytes(body[4 + hlen :])
        return ("load", str(meta["key"]), raw_bytes)
    if kind == _REQ_STR_LIST:
        meta = json.loads(bytes(body).decode("utf-8"))
        op = str(meta["op"])
        if op == "predict_attr":
            return (op, str(meta["key"]), str(meta["attr"]), list(meta["strings"]))
        return (op, str(meta["key"]), list(meta["strings"]))
    if kind == _REQ_NDARRAY:
        if len(body) < 4:
            raise ValueError("Truncated ndarray request")
        (hlen,) = struct.unpack(">I", bytes(body[:4]))
        if hlen > len(body) - 4:
            raise ValueError("Malformed ndarray request header")
        meta = json.loads(bytes(body[4 : 4 + hlen]).decode("utf-8"))
        raw_buf = bytes(body[4 + hlen :])
        dt_tag = str(meta["dtype"])
        wire_dt = "<f4" if dt_tag == "f4" else "<f8"
        out_dt = np.float32 if dt_tag == "f4" else np.float64
        item_sz = 4 if dt_tag == "f4" else 8
        shape = tuple(int(x) for x in meta["shape"])
        elems = 1
        for d in shape:
            elems *= d
            if elems > (MAX_IPC_FRAME_BYTES // item_sz):
                raise ValueError("Oversized ndarray element count")
        if len(raw_buf) != elems * item_sz:
            raise ValueError("Mismatched ndarray request byte count")
        arr = np.frombuffer(raw_buf, dtype=wire_dt).astype(out_dt, copy=True).reshape(shape)
        op = str(meta["op"])
        if op == "predict_attr":
            return (op, str(meta["key"]), str(meta["attr"]), arr)
        return (op, str(meta["key"]), arr)
    raise ValueError(f"Unsupported request frame kind {kind}")


def _write_reply_json(stream: Any, data: dict[str, Any], *, kind: int = _REPLY_JSON) -> None:
    body = json.dumps(_to_json_safe(data), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    total = 1 + len(body)
    if total > MAX_IPC_FRAME_BYTES:
        raise ValueError(f"Oversized JSON reply ({total} bytes)")
    stream.write(struct.pack(">IB", total, kind))
    stream.write(body)
    stream.flush()


def _write_reply_ndarray(stream: Any, arr: np.ndarray) -> None:
    raw_arr = np.asarray(arr)
    if raw_arr.dtype == np.float32:
        typed = np.ascontiguousarray(raw_arr, dtype="<f4")
        kind = _REPLY_NDARRAY_F32
    else:
        typed = np.ascontiguousarray(raw_arr, dtype="<f8")
        kind = _REPLY_NDARRAY_F64
    ndim = typed.ndim
    if ndim > 8:
        raise ValueError(f"Unsupported ndarray rank {ndim}")
    shape_hdr = struct.pack(f">B{ndim}Q", ndim, *typed.shape)
    raw = typed.tobytes(order="C")
    total = 1 + len(shape_hdr) + len(raw)
    if total > MAX_IPC_FRAME_BYTES:
        raise ValueError(f"Oversized ndarray reply ({total} bytes)")
    stream.write(struct.pack(">IB", total, kind))
    stream.write(shape_hdr)
    stream.write(raw)
    stream.flush()


def _write_all_fd_with_deadline(fd: int, data: bytes, deadline: float) -> None:
    view = memoryview(data)
    written = 0
    total = len(data)
    with selectors.DefaultSelector() as sel:
        sel.register(fd, selectors.EVENT_WRITE)
        while written < total:
            rem_sec = deadline - time.monotonic()
            if rem_sec <= 0.0:
                raise TimeoutError("Darwin joblib worker timed out writing request")
            events = sel.select(timeout=rem_sec)
            if not events:
                raise TimeoutError("Darwin joblib worker timed out writing request")
            try:
                n = os.write(fd, view[written : min(total, written + 65536)])
            except BlockingIOError:
                continue
            if n <= 0:
                raise BrokenPipeError("Darwin joblib worker stdin closed unexpectedly")
            written += n


def _read_exact_fd_with_deadline(fd: int, n: int, deadline: float) -> bytes:
    chunks: list[bytes] = []
    rem = n
    with selectors.DefaultSelector() as sel:
        sel.register(fd, selectors.EVENT_READ)
        while rem > 0:
            remaining_sec = deadline - time.monotonic()
            if remaining_sec <= 0.0:
                raise TimeoutError("Darwin joblib worker timed out waiting for reply")
            events = sel.select(timeout=remaining_sec)
            if not events:
                raise TimeoutError("Darwin joblib worker timed out waiting for reply")
            try:
                chunk = os.read(fd, min(rem, 65536))
            except BlockingIOError:
                continue
            if not chunk:
                raise EOFError("Darwin joblib worker closed stdout unexpectedly")
            chunks.append(chunk)
            rem -= len(chunk)
    return b"".join(chunks)


def _read_data_only_reply_in_parent(fd: int, deadline: float) -> tuple[str, Any]:
    hdr = _read_exact_fd_with_deadline(fd, 4, deadline)
    (size,) = struct.unpack(">I", hdr)
    if size < 1 or size > MAX_IPC_FRAME_BYTES:
        raise RuntimeError(f"Invalid Darwin joblib worker frame length ({size} bytes)")
    frame = _read_exact_fd_with_deadline(fd, size, deadline)
    kind = frame[0]
    body = memoryview(frame)[1:]
    if kind == _REPLY_JSON:
        return "ok", json.loads(bytes(body).decode("utf-8"))
    if kind == _REPLY_ERROR:
        err_doc = json.loads(bytes(body).decode("utf-8"))
        return "err", str(err_doc.get("error", "unknown worker error"))
    if kind in (_REPLY_NDARRAY_F64, _REPLY_NDARRAY_F32):
        item_bytes = 4 if kind == _REPLY_NDARRAY_F32 else 8
        wire_dt = "<f4" if kind == _REPLY_NDARRAY_F32 else "<f8"
        out_dt = np.float32 if kind == _REPLY_NDARRAY_F32 else np.float64
        if len(body) < 1:
            raise RuntimeError("Truncated ndarray reply header")
        ndim = int(body[0])
        if ndim > 8 or len(body) < 1 + 8 * ndim:
            raise RuntimeError("Malformed ndarray shape header from worker")
        shape = struct.unpack(f">{ndim}Q", bytes(body[1 : 1 + 8 * ndim]))
        expected_elems = 1
        for dim in shape:
            expected_elems *= int(dim)
            if expected_elems > (MAX_IPC_FRAME_BYTES // item_bytes):
                raise RuntimeError("Oversized ndarray element count from worker")
        raw_buf = bytes(body[1 + 8 * ndim :])
        if len(raw_buf) != expected_elems * item_bytes:
            raise RuntimeError("Mismatched ndarray byte payload length from worker")
        arr = np.frombuffer(raw_buf, dtype=wire_dt).astype(out_dt, copy=True).reshape(shape)
        return "ok", arr
    raise RuntimeError(f"Unsupported Darwin worker reply kind {kind}")


class _IsolatedTransformProxy:
    def __init__(self, client: _DarwinJoblibClient, key: str) -> None:
        self._client = client
        self._key = key

    def transform(self, X: Any) -> np.ndarray:
        return np.asarray(self._client.call("transform", self._key, np.asarray(X)))


class _IsolatedPredictProxy:
    def __init__(self, client: _DarwinJoblibClient, key: str, attr_key: str = "model") -> None:
        self._client = client
        self._key = key
        self._attr_key = attr_key

    def predict(self, X: Any, **_kwargs: Any) -> np.ndarray:
        if isinstance(X, list) and (not X or isinstance(X[0], str)):
            payload: Any = X
        else:
            payload = np.asarray(X)
        if self._attr_key == "model":
            return np.asarray(self._client.call("predict", self._key, payload), dtype=np.float64)
        return np.asarray(
            self._client.call("predict_attr", self._key, self._attr_key, payload),
            dtype=np.float64,
        )


class _DarwinJoblibClient:
    def __init__(self, *, timeout_sec: float = DEFAULT_WORKER_TIMEOUT_SEC) -> None:
        self._lock = threading.Lock()
        self._proc: subprocess.Popen[bytes] | None = None
        self._loaded_bytes: dict[str, bytes] = {}
        self._timeout_sec = max(1.0, float(timeout_sec))
        atexit.register(self.close)

    def _kill_proc_unlocked(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            if proc.stdin is not None:
                try:
                    proc.stdin.close()
                except Exception:
                    pass
            if proc.stdout is not None:
                try:
                    proc.stdout.close()
                except Exception:
                    pass
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=1.5)
        except Exception:
            pass

    def _exchange_unlocked(self, proc: subprocess.Popen[bytes], req: tuple[Any, ...]) -> Any:
        assert proc.stdin is not None and proc.stdout is not None
        deadline = time.monotonic() + self._timeout_sec
        try:
            wire_req = _encode_request_frame(req)
            _write_all_fd_with_deadline(proc.stdin.fileno(), wire_req, deadline)
            status, payload = _read_data_only_reply_in_parent(proc.stdout.fileno(), deadline)
        except Exception as exc:
            self._kill_proc_unlocked()
            raise RuntimeError(f"Darwin joblib worker IPC failure ({req[0]}): {exc}") from exc
        if status != "ok":
            raise RuntimeError(f"Darwin joblib worker error ({req[0]}): {payload}")
        return payload

    def _ensure_proc(self) -> subprocess.Popen[bytes]:
        if self._proc is not None and self._proc.poll() is None:
            return self._proc
        self._kill_proc_unlocked()
        env = dict(os.environ)
        env["OMP_NUM_THREADS"] = "1"
        src_root = str(Path(__file__).resolve().parents[2])
        prev_py = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{src_root}:{prev_py}" if prev_py else src_root
        proc = subprocess.Popen(
            [sys.executable, "-m", "mrnax.engine._darwin_joblib_worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            bufsize=0,
            env=env,
        )
        assert proc.stdin is not None and proc.stdout is not None
        os.set_blocking(proc.stdin.fileno(), False)
        os.set_blocking(proc.stdout.fileno(), False)
        self._proc = proc
        for key, raw in self._loaded_bytes.items():
            self._exchange_unlocked(proc, ("load", key, raw))
        return proc

    def call(self, op: str, *args: Any) -> Any:
        with self._lock:
            proc = self._ensure_proc()
            return self._exchange_unlocked(proc, (op, *args))

    def load_verified_bytes(self, cache_key: str, raw_bytes: bytes) -> dict[str, Any]:
        meta: dict[str, Any] = self.call("load", cache_key, raw_bytes)
        with self._lock:
            self._loaded_bytes[cache_key] = raw_bytes
        out = dict(meta["primitives"])
        if meta.get("has_scaler"):
            out["scaler"] = _IsolatedTransformProxy(self, cache_key)
        if meta.get("has_model"):
            out["model"] = _IsolatedPredictProxy(self, cache_key, attr_key="model")
        for sub_key in meta.get("sub_predictors", ()):
            out[sub_key] = _IsolatedPredictProxy(self, cache_key, attr_key=sub_key)
        return out

    def close(self) -> None:
        with self._lock:
            if self._proc is not None:
                try:
                    if self._proc.stdin is not None and self._proc.poll() is None:
                        wire_stop = _encode_request_frame(("shutdown",))
                        _write_all_fd_with_deadline(self._proc.stdin.fileno(), wire_stop, time.monotonic() + 1.5)
                        self._proc.wait(timeout=1.5)
                except Exception:
                    pass
                finally:
                    self._kill_proc_unlocked()


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
            msg = _read_request_frame_in_worker(stdin)
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
                for est_obj in loaded.values():
                    if hasattr(est_obj, "set_params"):
                        try:
                            est_obj.set_params(n_jobs=1)
                        except Exception:
                            pass
                    for sub_est in getattr(est_obj, "estimators_", ()):
                        if hasattr(sub_est, "set_params"):
                            try:
                                sub_est.set_params(n_jobs=1)
                            except Exception:
                                pass
                bundles[key] = loaded
                sub_predictors = [
                    k
                    for k, v in loaded.items()
                    if k not in {"model", "scaler"} and callable(getattr(v, "predict", None))
                ]
                excluded_keys = {"model", "scaler", *sub_predictors}
                primitives = {
                    k: _to_json_safe(v)
                    for k, v in loaded.items()
                    if k not in excluded_keys
                    and isinstance(v, (str, int, float, bool, list, tuple, dict, type(None)))
                }
                _write_reply_json(
                    stdout,
                    {
                        "primitives": primitives,
                        "has_model": "model" in loaded,
                        "has_scaler": "scaler" in loaded,
                        "sub_predictors": sub_predictors,
                    },
                )
            elif op == "transform":
                _, key, arr = msg
                res = np.asarray(bundles[key]["scaler"].transform(arr))
                _write_reply_ndarray(stdout, res)
            elif op == "predict":
                _, key, arr = msg
                mdl = bundles[key]["model"]
                try:
                    res = np.asarray(mdl.predict(arr, num_threads=1), dtype=np.float64)
                except TypeError:
                    res = np.asarray(mdl.predict(arr), dtype=np.float64)
                _write_reply_ndarray(stdout, res)
            elif op == "predict_attr":
                _, key, attr_key, payload = msg
                est = bundles[key][attr_key]
                try:
                    res = np.asarray(est.predict(payload, num_threads=1), dtype=np.float64)
                except TypeError:
                    res = np.asarray(est.predict(payload), dtype=np.float64)
                _write_reply_ndarray(stdout, res)
            else:
                _write_reply_json(stdout, {"error": f"Unknown op {op!r}"}, kind=_REPLY_ERROR)
        except Exception as exc:
            _write_reply_json(stdout, {"error": f"{type(exc).__name__}: {exc}"}, kind=_REPLY_ERROR)


if __name__ == "__main__":
    _worker_main()

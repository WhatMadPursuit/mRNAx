"""Scheme B: Standard JSON-RPC 2.0 stdio MCP Server exposing `mrnax_core` atomic tools to external Agents."""

from __future__ import annotations

from decimal import Decimal
import json
import sys
from pathlib import Path
from typing import Any

from mrnax.harness.state_machine import ProjectHarnessSession
from mrnax.harness.tools import AtomicToolbox


def _decimals_to_floats(val: Any) -> Any:
    if isinstance(val, Decimal):
        return float(val)
    if isinstance(val, dict):
        return {k: _decimals_to_floats(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_decimals_to_floats(v) for v in val]
    return val


def _normalize_jsonrpc_id(req_id: Any) -> Any:
    if isinstance(req_id, Decimal):
        if req_id.is_finite() and req_id == req_id.to_integral_value():
            return int(req_id)
        return float(req_id)
    return req_id


def handle_mcp_request(toolbox: AtomicToolbox, req: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(req, dict):
        return {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Invalid Request"}}
    method = req.get("method", "")
    req_id = _normalize_jsonrpc_id(req.get("id"))

    if method == "notifications/initialized":
        return None
    if method == "ping":
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "mrnax-v2-mcp", "version": "0.1.0"},
            },
        }
    if method == "tools/list":
        tools = [
            {
                "name": t["function"]["name"],
                "description": t["function"]["description"],
                "inputSchema": t["function"]["parameters"],
            }
            for t in toolbox.openai_tool_schemas()
        ]
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": tools}}
    if method == "tools/call":
        params = req.get("params")
        if params is not None and not isinstance(params, dict):
            return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32602, "message": "Invalid params"}}
        params = params or {}
        name = params.get("name", "")
        raw_args = params.get("arguments")
        args = {} if raw_args is None else raw_args
        if isinstance(args, dict) and name not in ("design_constructs", "refine_construct"):
            args = _decimals_to_floats(args)
        try:
            out = toolbox.dispatch_tool_call(name, args)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(out, ensure_ascii=False, indent=2)}],
                    "isError": False,
                },
            }
        except Exception as exc:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": f"Error: {exc}"}],
                    "isError": True,
                },
            }

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


def handle_jsonrpc_line(toolbox: AtomicToolbox, raw_line: str) -> str | None:
    line = raw_line.strip()
    if not line:
        return None
    try:
        req = json.loads(line, parse_float=Decimal)
    except json.JSONDecodeError as exc:
        err = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": f"Parse error: {exc}"}}
        return json.dumps(err, ensure_ascii=False)
    resp = handle_mcp_request(toolbox, req)
    if resp is None:
        return None
    return json.dumps(resp, ensure_ascii=False)


def serve_mcp_stdio(*, project_id: str | None = None, store_dir: Path, seed_dir: Path) -> None:
    session = ProjectHarnessSession(
        project_id=project_id,
        store_dir=store_dir,
        seed_dir=seed_dir,
        default_project_id="proj_mcp",
    )
    toolbox = AtomicToolbox(session)
    for raw_line in sys.stdin:
        out_line = handle_jsonrpc_line(toolbox, raw_line)
        if out_line is not None:
            sys.stdout.write(out_line + "\n")
            sys.stdout.flush()

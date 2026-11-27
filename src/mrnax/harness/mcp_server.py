"""Scheme B: Standard JSON-RPC 2.0 stdio MCP Server exposing `mrnax_core` atomic tools to external Agents."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from mrnax.harness.state_machine import ProjectHarnessSession
from mrnax.harness.tools import AtomicToolbox


def handle_mcp_request(toolbox: AtomicToolbox, req: dict[str, Any]) -> dict[str, Any] | None:
    method = req.get("method", "")
    req_id = req.get("id")

    if method == "notifications/initialized":
        return None
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
        params = req.get("params") or {}
        name = params.get("name", "")
        args = params.get("arguments") or {}
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


def serve_mcp_stdio(*, project_id: str | None = None, store_dir: Path, seed_dir: Path) -> None:
    session = ProjectHarnessSession(
        project_id=project_id,
        store_dir=store_dir,
        seed_dir=seed_dir,
        default_project_id="proj_mcp",
    )
    toolbox = AtomicToolbox(session)
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        req = json.loads(line)
        resp = handle_mcp_request(toolbox, req)
        if resp is not None:
            sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
            sys.stdout.flush()

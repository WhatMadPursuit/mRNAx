"""Unit test for Scheme B (`mcp_server.py` JSON-RPC 2.0 tool server for fresh external agent sessions)."""

from __future__ import annotations

import json
from pathlib import Path

from mrnax.harness.mcp_server import handle_mcp_request
from mrnax.harness.state_machine import ProjectHarnessSession
from mrnax.harness.tools import AtomicToolbox

SEED_DIR = Path("/Users/alex/.gemini/antigravity/scratch/mrnax-v2/registry_seed")


def test_mcp_jsonrpc_initialize_list_and_call(tmp_path: Path) -> None:
    session = ProjectHarnessSession(project_id="proj_mcp_test", store_dir=tmp_path, seed_dir=SEED_DIR)
    toolbox = AtomicToolbox(session)

    init_resp = handle_mcp_request(toolbox, {"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    assert init_resp is not None
    assert init_resp["result"]["serverInfo"]["name"] == "mrnax-v2-mcp"

    list_resp = handle_mcp_request(toolbox, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    assert list_resp is not None
    tool_names = {t["name"] for t in list_resp["result"]["tools"]}
    assert {"search_protein_assets", "design_constructs", "refine_construct", "export_bundle"} <= tool_names

    call_resp = handle_mcp_request(
        toolbox,
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "search_protein_assets", "arguments": {"query": "eGFP"}},
        },
    )
    assert call_resp is not None
    assert call_resp["result"]["isError"] is False
    payload = json.loads(call_resp["result"]["content"][0]["text"])
    assert payload["matched"] is True
    assert payload["mammalian_reporter_egfp"]["length_aa"] == 239

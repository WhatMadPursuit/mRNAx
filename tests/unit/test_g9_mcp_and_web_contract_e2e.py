"""Gate G9 E2E tests: Live stdio MCP server (`mcp_server.py`) & HTTP Web API <-> MCP contract parity."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from mrnax.cloud.cas_store import ExecutionCASStore
from mrnax.cloud.gateway import create_control_plane_app

ROOT = Path(__file__).resolve().parents[2]
SEED = ROOT / "registry_seed"


def test_mcp_stdio_subprocess_full_commercial_clean_lifecycle_and_web_contract_parity(tmp_path: Path) -> None:
    snap = json.loads((SEED / "egfp_reporter_snapshot.json").read_text(encoding="utf-8"))
    egfp_aa = snap["mammalian_reporter_egfp"]["protein_sequence"]

    mcp_store = tmp_path / "mcp_proj"
    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize"},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 99, "method": "ping"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "design_constructs",
                "arguments": {
                    "user_utterance": "Commercial clean eGFP avoiding BsaI with liver detargeting",
                    "protein_sequence": egfp_aa,
                    "compliance_track": "commercial_clean",
                    "uridine_chemistry": "m1psi",
                    "restriction_enzymes": ["BsaI"],
                    "detarget_liver": True,
                    "random_seed": 42,
                },
            },
        },
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "refine_construct",
                "arguments": {
                    "user_utterance": "Switch restriction enzyme to BspQI while inheriting protein lock",
                    "updated_restriction_enzymes": ["BspQI"],
                },
            },
        },
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {"name": "explain_tradeoffs", "arguments": {}},
        },
    ]
    stdin_payload = "\n".join(json.dumps(r) for r in requests) + "\n"
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{ROOT / 'src'}:{env.get('PYTHONPATH', '')}"
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mrnax.harness.cli",
            "mcp-serve",
            "--project-id",
            "proj_g9_mcp",
            "--store-dir",
            str(mcp_store),
            "--seed-dir",
            str(SEED),
        ],
        input=stdin_payload,
        capture_output=True,
        text=True,
        env=env,
        check=True,
        timeout=60.0,
    )
    lines = [json.loads(line) for line in proc.stdout.strip().splitlines() if line.strip()]
    by_id = {msg["id"]: msg for msg in lines}
    assert by_id[1]["result"]["serverInfo"]["name"] == "mrnax-v2-mcp"
    assert by_id[99]["result"] == {}
    listed_tools = {t["name"]: t for t in by_id[2]["result"]["tools"]}
    assert len(listed_tools) == 9
    assert "random_seed" in listed_tools["design_constructs"]["inputSchema"]["properties"]
    seed_prop = listed_tools["design_constructs"]["inputSchema"]["properties"]["random_seed"]
    assert seed_prop["minimum"] == 0 and seed_prop["maximum"] == 2147483647
    assert listed_tools["design_constructs"]["inputSchema"]["additionalProperties"] is False
    assert listed_tools["refine_construct"]["inputSchema"]["additionalProperties"] is False
    ev_prop = listed_tools["design_constructs"]["inputSchema"]["properties"]["entry_version"]
    assert {"type": "null"} in ev_prop.get("anyOf", [])

    d_res = json.loads(by_id[3]["result"]["content"][0]["text"])
    assert d_res["outcome_status"].startswith("COMPLETED")
    assert d_res["intent_version"] == 1
    assert d_res["qualified_count"] >= 4

    r_res = json.loads(by_id[4]["result"]["content"][0]["text"])
    assert r_res["outcome_status"].startswith("COMPLETED")
    assert r_res["intent_version"] == 2
    assert "protein_sequence" in r_res["inherited_locks"]
    top_bid = r_res["recommended_bundle_ids"][0]

    exp_res = json.loads(by_id[5]["result"]["content"][0]["text"])
    assert exp_res["recommended_bundle_ids"] == r_res["recommended_bundle_ids"]

    # Export top recommended construct via second stdio MCP call on persisted store
    export_req = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {
                "name": "export_bundle",
                "arguments": {"bundle_id": top_bid, "out_dir": str(mcp_store / "exports" / top_bid)},
            },
        }
    )
    proc_exp = subprocess.run(
        [
            sys.executable,
            "-m",
            "mrnax.harness.cli",
            "mcp-serve",
            "--project-id",
            "proj_g9_mcp",
            "--store-dir",
            str(mcp_store),
            "--seed-dir",
            str(SEED),
        ],
        input=export_req + "\n",
        capture_output=True,
        text=True,
        env=env,
        check=True,
        timeout=30.0,
    )
    exp_msg = json.loads(proc_exp.stdout.strip())
    assert exp_msg["result"]["isError"] is False
    exported_files = sorted(p.name for p in (mcp_store / "exports" / top_bid).iterdir() if p.is_file())
    assert exported_files == ["cds_only.fasta", "chemistry_metadata.json", "construct.gb", "full_rna.fasta"]

    # Verify HTTP Web API exposes identical persisted state, recommended_bundle_ids, and 4-file export
    app = create_control_plane_app(
        ExecutionCASStore(tmp_path / "cas.db"),
        projects_root=tmp_path,
        seed_dir=SEED,
        access_token="g9-admin",
    )
    with TestClient(app, headers={"X-MRNAX-Access-Token": "g9-admin"}) as client:
        mcp_store_named = tmp_path / "proj_g9_mcp"
        mcp_store.rename(mcp_store_named)
        st = client.get("/v1/projects/proj_g9_mcp/state")
        assert st.status_code == 200
        st_json = st.json()
        assert st_json["intent_count"] == 2
        assert st_json["latest_report"]["recommended_bundle_ids"] == r_res["recommended_bundle_ids"]
        http_exp = client.get(f"/v1/projects/proj_g9_mcp/bundles/{top_bid}/export")
        assert http_exp.status_code == 200
        assert sorted(http_exp.json()["files"].keys()) == exported_files

        # Independent HTTP write-path parity: fresh HTTP commercial_clean turn (omitting use_commercial_surrogate)
        # must match fresh MCP design_constructs on both ready and clarification-only turns.
        http_ready = client.post(
            "/v1/projects/proj_g9_http_indep/turn",
            json={
                "mode": "initial",
                "user_utterance": "Commercial clean eGFP avoiding BsaI with liver detargeting",
                "protein_sequence": egfp_aa,
                "compliance_track": "commercial_clean",
                "uridine_chemistry": "m1psi",
                "restriction_enzymes": ["BsaI"],
                "detarget_liver": True,
                "random_seed": 42,
            },
        )
        assert http_ready.status_code == 200
        hr_body = http_ready.json()
        assert hr_body["status"] == "COMPLETED"
        assert hr_body["report"]["outcome_status"] == d_res["outcome_status"]
        assert len(hr_body["candidates"]) == d_res["qualified_count"]
        assert len(hr_body["report"]["recommended_bundle_ids"]) == len(d_res["recommended_bundle_ids"])
        mcp_v1_rep = sorted(
            (json.loads(p.read_text(encoding="utf-8")) for p in (mcp_store_named / "reports").glob("rep_*.json")),
            key=lambda r: r["intent_version"],
        )[0]
        assert [c["version_digest"] for c in hr_body["candidates"]] == [
            e["version_digest"] for e in mcp_v1_rep["candidate_evaluations"]
        ]
        mcp_v1_objs = [o["metric_id"] for o in st_json["intents"][0]["objectives"]]
        http_v1_objs = [o["metric_id"] for o in hr_body["intent"]["objectives"]]
        assert http_v1_objs == mcp_v1_objs == [
            "utr5_mrl_pred",
            "cds_stability_pred",
            "consensus_halflife_pred",
            "utr3_stability_pred",
        ]

        # Clarification-only commercial_clean parity across independent MCP and HTTP stores
        mcp_clar_store = tmp_path / "mcp_clar"
        clar_req = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 10,
                "method": "tools/call",
                "params": {
                    "name": "design_constructs",
                    "arguments": {
                        "user_utterance": "Truncated commercial GFP",
                        "protein_sequence": egfp_aa[1:],
                        "compliance_track": "commercial_clean",
                    },
                },
            }
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "mrnax.harness.cli",
                "mcp-serve",
                "--project-id",
                "proj_clar_mcp",
                "--store-dir",
                str(mcp_clar_store),
                "--seed-dir",
                str(SEED),
            ],
            input=clar_req + "\n",
            capture_output=True,
            text=True,
            env=env,
            check=True,
            timeout=30.0,
        )
        mcp_clar_intent = json.loads(
            next((mcp_clar_store / "intents").glob("intent_v*.json")).read_text(encoding="utf-8")
        )
        http_clar = client.post(
            "/v1/projects/proj_clar_http/turn",
            json={
                "mode": "initial",
                "user_utterance": "Truncated commercial GFP",
                "protein_sequence": egfp_aa[1:],
                "compliance_track": "commercial_clean",
            },
        ).json()
        assert http_clar["status"] == "NEEDS_CLARIFICATION"
        assert [o["metric_id"] for o in http_clar["intent"]["objectives"]] == [
            o["metric_id"] for o in mcp_clar_intent["objectives"]
        ]

        # Implicit utterance chemistry ("with unmodified uridine") & implicit tissue ("and detarget liver")
        # + exact version_digest parity across independent MCP and HTTP stores
        for case_tag, payload in (
            (
                "imp_chem",
                {
                    "user_utterance": "Evaluate the existing reporter with unmodified uridine",
                    "protein_sequence": egfp_aa,
                    "compliance_track": "commercial_clean",
                    "restriction_enzymes": ["BsaI"],
                    "random_seed": 43,
                },
            ),
            (
                "imp_tiss",
                {
                    "user_utterance": "Evaluate the existing reporter and detarget liver",
                    "protein_sequence": egfp_aa,
                    "compliance_track": "commercial_clean",
                    "restriction_enzymes": ["BsaI"],
                    "random_seed": 43,
                },
            ),
        ):
            m_dir = tmp_path / f"mcp_{case_tag}"
            rpc_in = json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 20,
                    "method": "tools/call",
                    "params": {"name": "design_constructs", "arguments": payload},
                }
            )
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mrnax.harness.cli",
                    "mcp-serve",
                    "--project-id",
                    f"proj_{case_tag}",
                    "--store-dir",
                    str(m_dir),
                    "--seed-dir",
                    str(SEED),
                ],
                input=rpc_in + "\n",
                capture_output=True,
                text=True,
                env=env,
                check=True,
                timeout=60.0,
            )
            h_res = client.post(f"/v1/projects/proj_{case_tag}_http/turn", json=payload)
            assert h_res.status_code == 200
            h_body = h_res.json()
            m_intent = json.loads(next((m_dir / "intents").glob("intent_v*.json")).read_text(encoding="utf-8"))
            m_rep = json.loads(next((m_dir / "reports").glob("rep_*.json")).read_text(encoding="utf-8"))
            h_intent = h_body["intent"]
            assert m_intent["uridine_chemistry"] == h_intent["uridine_chemistry"]
            assert m_intent["cell_expression_control"] == h_intent["cell_expression_control"]
            assert m_intent["protein_targets"][0]["entry_version"] == h_intent["protein_targets"][0]["entry_version"]
            assert [e["version_digest"] for e in m_rep["candidate_evaluations"]] == [
                c["version_digest"] for c in h_body["candidates"]
            ]

        # Seed bounds & JSON Schema integer encoding parity across MCP and HTTP
        for bad_seed in (-1, 2147483648, 42.5, True, "42"):
            tag = str(bad_seed).replace("-", "neg").replace(".", "p")
            bad_mcp_dir = tmp_path / f"bad_seed_{tag}_{type(bad_seed).__name__}"
            bad_rpc = json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 30,
                    "method": "tools/call",
                    "params": {
                        "name": "design_constructs",
                        "arguments": {
                            "user_utterance": "Seed check",
                            "protein_sequence": "A",
                            "random_seed": bad_seed,
                        },
                    },
                }
            )
            bad_proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mrnax.harness.cli",
                    "mcp-serve",
                    "--project-id",
                    "proj_bad_seed",
                    "--store-dir",
                    str(bad_mcp_dir),
                    "--seed-dir",
                    str(SEED),
                ],
                input=bad_rpc + "\n",
                capture_output=True,
                text=True,
                env=env,
                check=True,
                timeout=30.0,
            )
            bad_out = json.loads(bad_proc.stdout.strip())
            assert bad_out["result"]["isError"] is True
            assert list((bad_mcp_dir / "intents").glob("*.json")) == []

            bad_http_id = f"proj_bad_http_{tag}_{type(bad_seed).__name__}"
            bad_h = client.post(
                f"/v1/projects/{bad_http_id}/turn",
                json={"user_utterance": "Seed check", "protein_sequence": "A", "random_seed": bad_seed},
            )
            assert bad_h.status_code == 422
            assert not (tmp_path / bad_http_id / "intents").exists() or list(
                (tmp_path / bad_http_id / "intents").glob("*.json")
            ) == []

        for idx_s, (ok_seed, expected_int) in enumerate(
            ((0, 0), (2147483647, 2147483647), (0.0, 0), (42.0, 42), (2147483647.0, 2147483647))
        ):
            ok_mcp_dir = tmp_path / f"ok_seed_{idx_s}"
            ok_rpc = json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 31,
                    "method": "tools/call",
                    "params": {
                        "name": "design_constructs",
                        "arguments": {
                            "user_utterance": "Seed boundary ok",
                            "protein_sequence": "A",
                            "random_seed": ok_seed,
                        },
                    },
                }
            )
            ok_proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mrnax.harness.cli",
                    "mcp-serve",
                    "--project-id",
                    "proj_ok_seed",
                    "--store-dir",
                    str(ok_mcp_dir),
                    "--seed-dir",
                    str(SEED),
                ],
                input=ok_rpc + "\n",
                capture_output=True,
                text=True,
                env=env,
                check=True,
                timeout=30.0,
            )
            assert json.loads(ok_proc.stdout.strip())["result"]["isError"] is False
            mcp_saved_seed = json.loads(
                next((ok_mcp_dir / "intents").glob("intent_v*.json")).read_text(encoding="utf-8")
            )["runtime_policy"]["random_seed"]
            assert mcp_saved_seed == expected_int
            ok_h = client.post(
                f"/v1/projects/proj_ok_http_{idx_s}/turn",
                json={"user_utterance": "Seed boundary ok", "protein_sequence": "A", "random_seed": ok_seed},
            )
            assert ok_h.status_code == 200
            assert ok_h.json()["intent"]["runtime_policy"]["random_seed"] == expected_int

        # Exact raw JSON non-integral float tokens (`2147483647.0000001`, `42.000000000000001`) + mode-specific/malformed-alias parity
        for tok_i, raw_tok in enumerate(("2147483647.0000001", "42.000000000000001")):
            raw_json = (
                f'{{"user_utterance":"Exact dec check","protein_sequence":"A","random_seed":{raw_tok}}}'.encode("utf-8")
            )
            h_dec = client.post(
                f"/v1/projects/proj_dec_http_{tok_i}/turn",
                content=raw_json,
                headers={"Content-Type": "application/json"},
            )
            assert h_dec.status_code == 422
            assert not (tmp_path / f"proj_dec_http_{tok_i}" / "intents").exists()

            m_dec_dir = tmp_path / f"proj_dec_mcp_{tok_i}"
            m_wire = (
                b'{"jsonrpc":"2.0","id":41,"method":"tools/call","params":{"name":"design_constructs","arguments":'
                + raw_json
                + b"}}\n"
            )
            m_dec = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mrnax.harness.cli",
                    "mcp-serve",
                    "--project-id",
                    "proj_dec",
                    "--store-dir",
                    str(m_dec_dir),
                    "--seed-dir",
                    str(SEED),
                ],
                input=m_wire.decode("utf-8"),
                capture_output=True,
                text=True,
                env=env,
                check=True,
                timeout=30.0,
            )
            assert json.loads(m_dec.stdout.strip())["result"]["isError"] is True
            assert list((m_dec_dir / "intents").glob("*.json")) == []

        # Malformed alias (`restriction_enzymes: [42]` alongside valid `updated_restriction_enzymes`) & refine `random_seed` rejected on both
        for bad_ref in (
            {"user_utterance": "Bad alias", "restriction_enzymes": [42], "updated_restriction_enzymes": ["BsaI"]},
            {"user_utterance": "Seed on refine", "random_seed": 7},
        ):
            h_br = client.post("/v1/projects/proj_g9_http_indep/turn", json={"mode": "refine", **bad_ref})
            assert h_br.status_code == 422
            assert len(list((tmp_path / "proj_g9_http_indep" / "intents").glob("intent_v*.json"))) == 1

            m_br = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mrnax.harness.cli",
                    "mcp-serve",
                    "--project-id",
                    "proj_g9_mcp",
                    "--store-dir",
                    str(mcp_store_named),
                    "--seed-dir",
                    str(SEED),
                ],
                input=json.dumps(
                    {"jsonrpc": "2.0", "id": 42, "method": "tools/call", "params": {"name": "refine_construct", "arguments": bad_ref}}
                )
                + "\n",
                capture_output=True,
                text=True,
                env=env,
                check=True,
                timeout=30.0,
            )
            assert json.loads(m_br.stdout.strip())["result"]["isError"] is True
            assert len(list((mcp_store_named / "intents").glob("intent_v*.json"))) == 2

        # OpenAPI schema + MCP tools/list + fractional JSON-RPC id + 9-tool strict validation
        openapi_resp = client.get("/openapi.json")
        assert openapi_resp.status_code == 200
        oa_schemas = openapi_resp.json()["components"]["schemas"]
        assert oa_schemas["InitialTurnRequest"]["additionalProperties"] is False
        assert oa_schemas["RefineTurnRequest"]["additionalProperties"] is False

        bad_init_extra = {"user_utterance": "Extra key", "protein_sequence": "A", "custom_reference_rna_sequence": None}
        assert client.post("/v1/projects/proj_extra_http/turn", json={"mode": "initial", **bad_init_extra}).status_code == 422

        from mrnax.harness.mcp_server import handle_jsonrpc_line
        from mrnax.harness.state_machine import ProjectHarnessSession
        from mrnax.harness.tools import AtomicToolbox

        tb_probe = AtomicToolbox(ProjectHarnessSession(project_id="proj_g9_mcp", store_dir=mcp_store_named, seed_dir=SEED))
        ping_frac = json.loads(handle_jsonrpc_line(tb_probe, '{"jsonrpc":"2.0","id":1.5,"method":"ping"}') or "")
        assert ping_frac["id"] == 1.5 and ping_frac["result"] == {}
        parse_err = json.loads(handle_jsonrpc_line(tb_probe, "{bad json") or "")
        assert parse_err["error"]["code"] == -32700

        list_tools = json.loads(handle_jsonrpc_line(tb_probe, '{"jsonrpc":"2.0","id":2,"method":"tools/list"}') or "")["result"]["tools"]
        assert len(list_tools) == 9
        for t_spec in list_tools:
            assert t_spec["inputSchema"].get("additionalProperties") is False
        dc_props = next(t["inputSchema"]["properties"] for t in list_tools if t["name"] == "design_constructs")
        assert {"type": "null"} in dc_props["entry_version"]["anyOf"]

        extra_mcp = json.loads(
            handle_jsonrpc_line(
                tb_probe,
                json.dumps({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "design_constructs", "arguments": bad_init_extra}}),
            )
            or ""
        )
        assert extra_mcp["result"]["isError"] is True
        bad_audit = json.loads(
            handle_jsonrpc_line(
                tb_probe,
                json.dumps({"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "audit_construct", "arguments": {"bundle_id": "b1", "compliance_track": "bogus"}}}),
            )
            or ""
        )
        assert bad_audit["result"]["isError"] is True

    # Permutation/idempotence check on tied objective vectors across persistence & projection
    from mrnax.engine.pareto_director import _rank_and_order_pareto_candidates

    tied_vecs = [
        [0.9, 0.9],
        [0.8, 0.2],
        [0.2, 0.8],
        [0.5, 0.5],
        [0.5, 0.5],
        [0.1, 0.7],
        [0.7, 0.1],
    ]
    _, ord1 = _rank_and_order_pareto_candidates(tied_vecs)
    reordered = [tied_vecs[i] for i in ord1]
    _, ord2 = _rank_and_order_pareto_candidates(reordered)
    assert ord2 == list(range(len(reordered)))

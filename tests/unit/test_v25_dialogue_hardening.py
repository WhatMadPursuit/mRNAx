"""Regression tests for DeepSeek DSML markup sanitization, bounded wrap-up, transient 429 retry, and history cleaning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from mrnax.cloud.cas_store import ExecutionCASStore
from mrnax.cloud.gateway import _load_agent_chat_history, create_control_plane_app
from mrnax.harness.auth_secrets import SecretVault
from mrnax.harness.dialogue import (
    LiveDialogueAgent,
    _build_tool_trace_fallback_reply,
    _post_json_with_429_retry,
    _strip_leaked_tool_markup,
)
from mrnax.harness.state_machine import ProjectHarnessSession
from mrnax.harness.tools import AtomicToolbox

ROOT = Path(__file__).resolve().parents[2]
SEED_DIR = ROOT / "registry_seed"


@pytest.fixture(autouse=True)
def _isolate_vault(tmp_path: Path):
    iso_home = tmp_path / "iso_home"
    iso_home.mkdir(parents=True, exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("HOME", str(iso_home))
        mp.delenv("MRNAX_VAULT_PASSPHRASE", raising=False)
        SecretVault().set_key("deepseek", "sk-TestDeepSeekKey1234567890")
        SecretVault().set_key("kimi", "sk-TestKimiKey1234567890")
        yield


def test_strip_leaked_tool_markup_and_fallback_reply() -> None:
    leaked = (
        "\n\n<｜DSML｜ calls>\n"
        '<｜DSML｜ invoke name="audit_construct">\n'
        '<｜DSML｜ parameter name="bundle_id" string="true">bnd_123</｜DSML｜ parameter>\n'
        "</｜DSML｜ invoke>\n</｜DSML｜ calls>"
    )
    assert _strip_leaked_tool_markup(leaked) == ""

    mixed = "结论：请确认靶点。<|DSML| invoke name='x'>abc</|DSML| invoke><tool_call>foo</tool_call>"
    assert _strip_leaked_tool_markup(mixed) == "结论：请确认靶点。"

    unclosed = "已收到。<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"x\">"
    assert _strip_leaked_tool_markup(unclosed) == "已收到。"

    fb_zh = _build_tool_trace_fallback_reply(
        [
            {"name": "resolve_target_protein", "result": {"gene_symbol": "EPO", "uniprot_accession": "P01588"}},
            {"name": "design_constructs", "result": {"candidates": [{"display_label": "Pareto #1"}]}},
        ],
        "请设计 EPO mRNA",
    )
    assert "EPO (P01588)" in fb_zh
    assert "1 条合格候选" in fb_zh
    assert "Pareto #1" in fb_zh
    assert "snake_case" not in fb_zh

    fb_bq = _build_tool_trace_fallback_reply(
        [{"name": "design_constructs", "result": {"blocking_questions": ["Missing start M"]}}],
        "Design eGFP",
    )
    assert "Missing start M" in fb_bq


def test_max_tool_rounds_wrapup_omits_tools_and_cleans_leaked_dsml(tmp_path: Path) -> None:
    session = ProjectHarnessSession(project_id="proj_wrapup", store_dir=tmp_path / "proj_wrapup", seed_dir=SEED_DIR)
    toolbox = AtomicToolbox(session)
    seen_payloads: list[dict[str, Any]] = []

    def _handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content.decode("utf-8"))
        seen_payloads.append(body)
        if len(seen_payloads) == 1:
            return httpx.Response(
                200,
                headers={"Content-Type": "application/json"},
                json={
                    "choices": [
                        {
                            "finish_reason": "tool_calls",
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "query_design_rules",
                                            "arguments": json.dumps({"modality": "linear_mrna", "query": "kozak"}),
                                        },
                                    }
                                ],
                            },
                        }
                    ]
                },
            )
        # Simulate model stubbornly returning raw DSML even when tools were disabled
        return httpx.Response(
            200,
            headers={"Content-Type": "application/json"},
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "<｜DSML｜ calls><｜DSML｜ invoke name=\"audit_construct\"></｜DSML｜ invoke></｜DSML｜ calls>",
                        },
                    }
                ]
            },
        )

    agent = LiveDialogueAgent(
        toolbox,
        provider="deepseek",
        transport=httpx.MockTransport(_handler),
    )
    reply = agent.send_user_message("请帮我查看 Kozak 设计规则", max_tool_rounds=1)
    assert len(seen_payloads) == 2
    wrapup_payload = seen_payloads[1]
    assert "tools" not in wrapup_payload
    assert "tool_choice" not in wrapup_payload
    assert wrapup_payload["messages"][-1]["role"] == "user"
    assert "不要再调用任何工具" in wrapup_payload["messages"][-1]["content"]
    assert "DSML" not in reply
    assert reply.strip() != ""
    assert "DSML" not in str(agent.messages[-1].get("content") or "")


def test_transient_429_retry_vs_billing_quota_fast_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    from mrnax.harness import dialogue as dlg_mod

    now = [100.0]
    monkeypatch.setattr(dlg_mod._time, "monotonic", lambda: now[0])
    sleeps: list[float] = []

    calls = 0

    def _transient_handler(_req: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 3:
            return httpx.Response(
                429,
                headers={"Content-Type": "application/json"},
                json={"detail": "rate_limit_reached_error"},
            )
        return httpx.Response(200, headers={"Content-Type": "application/json"}, json={"ok": True})

    with httpx.Client(transport=httpx.MockTransport(_transient_handler)) as client:
        resp = _post_json_with_429_retry(
            client,
            "https://example.invalid/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json_body={"ok": True},
            overall_deadline=130.0,
            sleep_fn=lambda d: (sleeps.append(d), now.__setitem__(0, now[0] + d)),
        )
    assert resp.status_code == 200
    assert calls == 3
    assert sleeps == [1.5, 3.0]

    # Billing exhaustion must fail immediately with 0 sleeps
    billing_calls = 0
    billing_sleeps: list[float] = []

    def _billing_handler(_req: httpx.Request) -> httpx.Response:
        nonlocal billing_calls
        billing_calls += 1
        return httpx.Response(
            429,
            headers={"Content-Type": "application/json"},
            json={"error": {"code": "credit_balance_exhausted"}},
        )

    with httpx.Client(transport=httpx.MockTransport(_billing_handler)) as client:
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            _post_json_with_429_retry(
                client,
                "https://example.invalid/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json_body={"ok": True},
                overall_deadline=200.0,
                sleep_fn=billing_sleeps.append,
            )
    assert exc_info.value.response.status_code == 429
    assert billing_calls == 1
    assert billing_sleeps == []


def test_gateway_scrubs_poisoned_dsml_history_and_responses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = ExecutionCASStore(tmp_path / "cas.sqlite3", require_drive_verifier=False)
    projects_root = tmp_path / "projects"
    app = create_control_plane_app(
        store,
        projects_root=projects_root,
        seed_dir=SEED_DIR,
        access_token="tok_gw_dsml",
    )
    pid = "proj_dsml_clean"
    proj_dir = projects_root / pid
    proj_dir.mkdir(parents=True, exist_ok=True)
    (proj_dir / "agent_chat_history.json").write_text(
        json.dumps(
            [
                {
                    "turn_ts_utc": "2026-11-28T14:19:38Z",
                    "client_request_key": "k1",
                    "provider": "deepseek",
                    "model": "deepseek-flash",
                    "user_message": "设计 eGFP",
                    "assistant_reply": "<｜DSML｜ calls><｜DSML｜ invoke name=\"audit_construct\"></｜DSML｜ invoke></｜DSML｜ calls>",
                    "tool_calls": [{"name": "design_constructs", "arguments": {}}],
                }
            ]
        ),
        encoding="utf-8",
    )
    loaded = _load_agent_chat_history(proj_dir)
    assert len(loaded) == 1
    assert "DSML" not in loaded[0]["assistant_reply"]
    assert loaded[0]["assistant_reply"].strip() != ""

    client = TestClient(app)
    r_state = client.get(f"/v1/projects/{pid}/state", headers={"X-MRNAX-Access-Token": "tok_gw_dsml"})
    assert r_state.status_code == 200
    assert "DSML" not in r_state.text

    def _fake_send(self: LiveDialogueAgent, user_text: str, **_kwargs: Any) -> str:
        # Poisoned DSML must never be injected into LiveDialogueAgent history
        assert all("DSML" not in str(m.get("content") or "") for m in self.messages)
        return "干净回复 <｜DSML｜ calls><｜DSML｜ invoke name=\"x\"></｜DSML｜ invoke></｜DSML｜ calls>"

    monkeypatch.setattr(LiveDialogueAgent, "send_user_message", _fake_send)
    r_chat = client.post(
        f"/v1/projects/{pid}/agent-chat",
        headers={"X-MRNAX-Access-Token": "tok_gw_dsml"},
        json={"message": "继续解释权衡", "provider": "deepseek", "model": "deepseek-flash"},
    )
    assert r_chat.status_code == 200
    body = r_chat.json()
    assert body["assistant_reply"] == "干净回复"
    assert "DSML" not in json.dumps(body, ensure_ascii=False)


def test_system_status_honors_mrnax_default_llm_provider(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = ExecutionCASStore(tmp_path / "cas_def.sqlite3", require_drive_verifier=False)
    app = create_control_plane_app(
        store,
        projects_root=tmp_path / "projects_def",
        seed_dir=SEED_DIR,
        access_token="tok_gw_def",
    )
    client = TestClient(app)

    # Unset -> legacy default deepseek
    monkeypatch.delenv("MRNAX_DEFAULT_LLM_PROVIDER", raising=False)
    assert client.get("/v1/system/status").json()["default_llm_provider"] == "deepseek"

    # Configured MRNAX_DEFAULT_LLM_PROVIDER=openai without OpenAI key -> safe fallback to configured deepseek
    monkeypatch.setenv("MRNAX_DEFAULT_LLM_PROVIDER", "openai")
    st_no_openai = client.get("/v1/system/status").json()
    assert st_no_openai["default_llm_provider"] == "deepseek"

    # Configured MRNAX_DEFAULT_LLM_PROVIDER=openai with OpenAI key present -> openai
    SecretVault().set_key("openai", "sk-TestOpenAIKey1234567890")
    st_with_openai = client.get("/v1/system/status").json()
    assert st_with_openai["default_llm_provider"] == "openai"
    assert "openai" in st_with_openai["server_llm_providers_ready"]
    assert st_with_openai["default_llm_models"]["openai"] == "gpt-5.4"

"""TDD tests for DeepSeek Flash/Pro defaults, multi-turn chat persistence, project context isolation, idempotency, and error/secret hygiene."""

from __future__ import annotations

import copy
import hashlib
import json
import stat
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from mrnax.cloud.cas_store import ExecutionCASStore
from mrnax.cloud.gateway import create_control_plane_app
from mrnax.harness.auth_secrets import SecretVault
from mrnax.harness.dialogue import (
    DEFAULT_PROVIDER_MODELS,
    SUPPORTED_PROVIDER_MODELS,
    LiveDialogueAgent,
)
from mrnax.harness.state_machine import ProjectHarnessSession
from mrnax.harness.tools import AtomicToolbox

ROOT = Path(__file__).resolve().parents[2]
SEED_DIR = ROOT / "registry_seed"


@pytest.fixture(autouse=True)
def _isolate_home_and_vault(tmp_path: Path):
    iso_home = tmp_path / "isolated_home"
    iso_home.mkdir(parents=True, exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("HOME", str(iso_home))
        mp.delenv("MRNAX_VAULT_PASSPHRASE", raising=False)
        SecretVault().set_key("deepseek", "sk-IsolatedTestVaultKey1234567890")
        yield


def test_deepseek_defaults_and_supported_models_include_flash_and_v4_pro() -> None:
    assert DEFAULT_PROVIDER_MODELS["deepseek"] == "deepseek-flash"
    assert SUPPORTED_PROVIDER_MODELS["deepseek"] == ["deepseek-flash", "deepseek-v4-pro"]
    assert "openai" in SUPPORTED_PROVIDER_MODELS
    assert "kimi" in SUPPORTED_PROVIDER_MODELS
    assert "qwen" in SUPPORTED_PROVIDER_MODELS
    assert "gemini" in SUPPORTED_PROVIDER_MODELS

    with pytest.raises(ValueError, match="Unsupported dialogue provider"):
        LiveDialogueAgent(None, provider="anthropic")


def test_live_dialogue_agent_isolates_context_from_system_sets_flash_thinking_and_recovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = ProjectHarnessSession(project_id="proj_hist", store_dir=tmp_path / "proj_hist", seed_dir=SEED_DIR)
    toolbox = AtomicToolbox(session)
    captured_payloads: list[dict[str, Any]] = []

    class _FakeResp:
        def __init__(self, payload: dict[str, Any]) -> None:
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return self._payload

    turn_idx = {"n": 0}

    def _fake_post(self: httpx.Client, url: str, headers: dict[str, str], json: dict[str, Any]) -> _FakeResp:
        captured_payloads.append(copy.deepcopy(json))
        turn_idx["n"] += 1
        if turn_idx["n"] == 1:
            return _FakeResp(
                {
                    "choices": [
                        {
                            "finish_reason": "tool_calls",
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "tc_bad_json",
                                        "type": "function",
                                        "function": {"name": "audit_construct", "arguments": "{not-valid-json"},
                                    },
                                    {
                                        "id": "tc_missing_bundle",
                                        "type": "function",
                                        "function": {
                                            "name": "audit_construct",
                                            "arguments": '{"bundle_id": "nonexistent_bundle"}',
                                        },
                                    },
                                ],
                            },
                        }
                    ]
                }
            )
        return _FakeResp(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "Recovered cleanly using sk-FakeSecret1234567890abcdef and pk_DummyCapKey1234567890ab",
                        },
                    }
                ]
            }
        )

    monkeypatch.setattr(httpx.Client, "post", _fake_post)

    agent = LiveDialogueAgent(
        toolbox,
        provider="deepseek",
        history=[
            {"role": "user", "content": "第1轮设计请求 sk-OldSecretToken1234567890abcd"},
            {"role": "assistant", "content": "已完成 v01 设计"},
        ],
        project_context={"project_id": "proj_hist", "active_intent_version": 1, "enzymes": ["BsaI"]},
    )
    reply = agent.send_user_message("请检查不存在的构建", explicit_api_key="sk-TestDeepSeekKey1234567890abcdef")

    assert "[REDACTED_API_KEY]" in reply
    assert "sk-FakeSecret" not in reply
    assert "pk_DummyCapKey" not in reply
    assert len(captured_payloads) == 2

    # System prompt remains strictly constant; project_context is isolated in a non-system data message.
    first_msgs = captured_payloads[0]["messages"]
    assert captured_payloads[0]["model"] == "deepseek-flash"
    assert captured_payloads[0]["thinking"] == {"type": "disabled"}
    assert sum(1 for m in first_msgs if m["role"] == "system") == 1
    assert any(
        m["role"] == "user" and "[READ_ONLY_PROJECT_CONTEXT_DATA]" in m["content"] and "active_intent_version" in m["content"]
        for m in first_msgs
    )
    assert any(m["role"] == "user" and "[REDACTED_API_KEY]" in m["content"] and "sk-OldSecret" not in m["content"] for m in first_msgs)
    assert any(m["role"] == "assistant" and "已完成 v01 设计" in m["content"] for m in first_msgs)

    second_msgs = captured_payloads[1]["messages"]
    tool_msgs = [json.loads(m["content"]) for m in second_msgs if m["role"] == "tool"]
    assert len(tool_msgs) == 2
    assert "error" in tool_msgs[0]
    assert "error" in tool_msgs[1]


def test_gateway_agent_chat_persists_redacted_history_handles_502_and_preserves_clarification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ExecutionCASStore(tmp_path / "cas.sqlite3")
    projects_root = tmp_path / "projects"
    app = create_control_plane_app(
        store,
        projects_root=projects_root,
        seed_dir=SEED_DIR,
        access_token="tok_admin_v2",
    )
    client = TestClient(app, headers={"X-MRNAX-Access-Token": "tok_admin_v2"})

    st = client.get("/v1/system/status").json()
    assert st["default_llm_provider"] == "deepseek"
    assert st["default_llm_models"]["deepseek"] == "deepseek-flash"
    assert st["supported_llm_models"]["deepseek"] == ["deepseek-flash", "deepseek-v4-pro"]

    seen_calls: list[dict[str, Any]] = []

    def _fake_send(self: LiveDialogueAgent, user_text: str, **kwargs: Any) -> str:
        seen_calls.append(
            {
                "provider": self.provider,
                "model": self.model,
                "messages": list(self.messages),
                "user_text": user_text,
                "allow_tools": kwargs.get("allow_tools", True),
            }
        )
        if len(seen_calls) == 1:
            snap = json.loads((SEED_DIR / "egfp_reporter_snapshot.json").read_text(encoding="utf-8"))
            res = self.toolbox.design_constructs(
                user_utterance=user_text,
                protein_sequence=snap["mammalian_reporter_egfp"]["protein_sequence"],
                uridine_chemistry="m1psi",
                restriction_enzymes=["BsaI"],
                detarget_liver=True,
            )
            self.tool_call_trace.append(
                {
                    "tool_call_id": "t1",
                    "name": "design_constructs",
                    "arguments": {"nested_secret": "sk-NestedToolArgSecret1234567890", "token": "tok_admin_v2"},
                    "result": res,
                }
            )
            return "已完成 v01 设计，建议避开 BsaI。"
        if len(seen_calls) == 2:
            res = self.toolbox.refine_construct(
                user_utterance=user_text,
                restriction_enzymes=["BspQI"],
            )
            self.tool_call_trace.append({"tool_call_id": "t2", "name": "refine_construct", "arguments": {}, "result": res})
            return "已切换至 BspQI，保留所有锁定项。"
        if len(seen_calls) == 3:
            self.toolbox.design_constructs(
                user_utterance=user_text,
                protein_sequence="VSKGEELFT",
            )
            return "序列缺少起始 M，请确认。"
        if len(seen_calls) == 4:
            # Informational text-only follow-up while intent is still unresolved must preserve NEEDS_CLARIFICATION
            return "请确认是否补充起始 M。"
        raise httpx.RequestError("upstream timeout with sk-ShouldBeRedacted1234567890")

    monkeypatch.setattr(LiveDialogueAgent, "send_user_message", _fake_send)

    r1 = client.post(
        "/v1/projects/proj_chat_v2/agent-chat",
        json={"message": "帮我设计 eGFP mRNA，密钥 sk-LeakTestKey1234567890abcd", "provider": "deepseek", "model": "deepseek-flash"},
    )
    assert r1.status_code == 200
    d1 = r1.json()
    assert d1["status"] == "COMPLETED"
    assert d1["provider"] == "deepseek"
    assert d1["model"] == "deepseek-flash"
    assert d1["intent"]["intent_version"] == 1
    assert d1["cli_replay_command"].startswith("uv run mrnax replay")
    assert d1["blocking_questions"] == []
    assert "sk-NestedToolArgSecret" not in json.dumps(d1["tool_calls"])
    assert "tok_admin_v2" not in json.dumps(d1["tool_calls"])

    r2 = client.post(
        "/v1/projects/proj_chat_v2/agent-chat",
        json={"message": "把酶改成 BspQI", "provider": "deepseek", "model": "deepseek-v4-pro"},
    )
    assert r2.status_code == 200
    d2 = r2.json()
    assert d2["model"] == "deepseek-v4-pro"
    assert d2["intent"]["intent_version"] == 2
    assert d2["intent_diff"]["v2_enzymes"] == ["BspQI"]

    t2_msgs = seen_calls[1]["messages"]
    assert any(m["role"] == "user" and "[REDACTED_API_KEY]" in m["content"] and "sk-LeakTestKey" not in m["content"] for m in t2_msgs)
    assert any(m["role"] == "assistant" and "已完成 v01 设计" in m["content"] for m in t2_msgs)
    assert any(m["role"] == "user" and "BsaI" in m["content"] for m in t2_msgs)

    st_proj = client.get("/v1/projects/proj_chat_v2/state").json()
    assert len(st_proj["chat_history"]) == 2
    hist_path = projects_root / "proj_chat_v2" / "agent_chat_history.json"
    assert stat.S_IMODE(hist_path.stat().st_mode) == 0o600
    raw_hist = hist_path.read_text(encoding="utf-8")
    assert "sk-LeakTestKey" not in raw_hist
    assert "sk-NestedToolArgSecret" not in raw_hist
    assert "tok_admin_v2" not in raw_hist

    r3 = client.post("/v1/projects/proj_chat_v2/agent-chat", json={"message": "用缺 M 的序列设计"})
    assert r3.status_code == 200
    d3 = r3.json()
    assert d3["status"] == "NEEDS_CLARIFICATION"
    assert len(d3["blocking_questions"]) >= 1

    # Turn 4 text-only reply does NOT clear active unresolved clarification!
    r4 = client.post("/v1/projects/proj_chat_v2/agent-chat", json={"message": "为什么拦住了？"})
    assert r4.status_code == 200
    d4 = r4.json()
    assert d4["status"] == "NEEDS_CLARIFICATION"
    assert len(d4["blocking_questions"]) >= 1

    # Turn 5 upstream network failure maps cleanly to bounded HTTP 502 with zero secret leak
    r5 = client.post("/v1/projects/proj_chat_v2/agent-chat", json={"message": "触发上游超时"})
    assert r5.status_code == 502
    assert "sk-ShouldBeRedacted" not in r5.text
    assert "Upstream LLM network error" in r5.json()["detail"]


def test_gateway_agent_chat_idempotent_retry_and_invalid_envelope_and_legacy_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ExecutionCASStore(tmp_path / "cas.sqlite3")
    projects_root = tmp_path / "projects"

    # 1. Pre-create legacy anonymous project (no owner_token_sha256.txt) with chat history on disk
    legacy_sess = ProjectHarnessSession(project_id="legacy_proj", store_dir=projects_root / "legacy_proj", seed_dir=SEED_DIR)
    legacy_sess.create_turn_intent_canonical(
        user_utterance="legacy",
        protein_sequence="MVSKGEELFT",
    )
    (projects_root / "legacy_proj" / "agent_chat_history.json").write_text(
        json.dumps([{"user_message": "private", "assistant_reply": "secret reply"}]),
        encoding="utf-8",
    )
    anon_app = create_control_plane_app(store, projects_root=projects_root, seed_dir=SEED_DIR, access_token="")
    anon_client = TestClient(anon_app)
    st_legacy = anon_client.get("/v1/projects/legacy_proj/state")
    assert st_legacy.status_code == 200
    assert st_legacy.json()["intent_count"] == 1
    assert st_legacy.json()["chat_history"] == []
    post_legacy = anon_client.post(
        "/v1/projects/legacy_proj/agent-chat",
        headers={"X-MRNAX-LLM-Key": "sk-UserByoKey1234567890abcdef"},
        json={"message": "hello"},
    )
    assert post_legacy.status_code == 403

    # 2. Retry after partial tool commit + 502 recovers WITHOUT creating duplicate intent
    admin_app = create_control_plane_app(
        store, projects_root=projects_root, seed_dir=SEED_DIR, access_token="tok_retry_test"
    )
    client = TestClient(admin_app, headers={"X-MRNAX-Access-Token": "tok_retry_test"})

    attempts = {"n": 0}

    def _flaky_send(self: LiveDialogueAgent, user_text: str, **kwargs: Any) -> str:
        attempts["n"] += 1
        if attempts["n"] == 1:
            assert kwargs.get("allow_tools", True) is True
            res = self.toolbox.design_constructs(
                user_utterance=user_text,
                protein_sequence="VSKGEELFT",
            )
            self.tool_call_trace.append({"tool_call_id": "t_partial", "name": "design_constructs", "arguments": {}, "result": res})
            raise httpx.RequestError("transient drop")
        # Retry with same client_request_id MUST recover with allow_tools=False and not run tool twice!
        assert kwargs.get("allow_tools") is False
        assert "[RECOVERED_TOOL_RESULTS_FROM_PRIOR_ATTEMPT]" in user_text
        return "已恢复并完成总结。"

    monkeypatch.setattr(LiveDialogueAgent, "send_user_message", _flaky_send)

    r_fail = client.post(
        "/v1/projects/proj_retry/agent-chat",
        json={"message": "试一下重试", "client_request_id": "req_once_001"},
    )
    assert r_fail.status_code == 502

    r_ok = client.post(
        "/v1/projects/proj_retry/agent-chat",
        json={"message": "试一下重试", "client_request_id": "req_once_001"},
    )
    assert r_ok.status_code == 200
    assert r_ok.json()["intent"]["intent_version"] == 1
    assert r_ok.json()["status"] == "NEEDS_CLARIFICATION"

    # Third call with same client_request_id hits completed idempotency cache without invoking LLM again
    r_replay = client.post(
        "/v1/projects/proj_retry/agent-chat",
        json={"message": "试一下重试", "client_request_id": "req_once_001"},
    )
    assert r_replay.status_code == 200
    assert attempts["n"] == 2

    st_retry = client.get("/v1/projects/proj_retry/state").json()
    assert st_retry["intent_count"] == 1

    # 3. Malformed provider JSON envelope (HTTP 200 with empty choices) maps to HTTP 502 (not 500)
    monkeypatch.undo()

    class _EmptyChoicesResp:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"choices": []}

    monkeypatch.setattr(httpx.Client, "post", lambda *a, **kw: _EmptyChoicesResp())
    r_bad_env = client.post(
        "/v1/projects/proj_bad_env/agent-chat",
        headers={"X-MRNAX-LLM-Key": "sk-DummyByoKey1234567890abcdef"},
        json={"message": "空响应包测试", "client_request_id": "req_bad_env_1"},
    )
    assert r_bad_env.status_code == 502
    assert "invalid response envelope" in r_bad_env.json()["detail"]


def test_deepseek_pro_preserves_reasoning_content_and_gateway_validates_idempotency_and_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 1. deepseek-v4-pro must NOT inject flash thinking-disabled flag and MUST preserve reasoning_content across tool loops
    session = ProjectHarnessSession(project_id="proj_pro", store_dir=tmp_path / "proj_pro", seed_dir=SEED_DIR)
    toolbox = AtomicToolbox(session)
    pro_payloads: list[dict[str, Any]] = []

    class _ProResp:
        def __init__(self, payload: dict[str, Any]) -> None:
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return self._payload

    idx = {"n": 0}

    def _pro_post(self: httpx.Client, url: str, headers: dict[str, str], json: dict[str, Any]) -> _ProResp:
        pro_payloads.append(copy.deepcopy(json))
        idx["n"] += 1
        if idx["n"] == 1:
            return _ProResp(
                {
                    "choices": [
                        {
                            "finish_reason": "tool_calls",
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "reasoning_content": "First check workspace status before replying.",
                                "tool_calls": [
                                    {
                                        "id": "tc_pro_1",
                                        "type": "function",
                                        "function": {"name": "get_job_status", "arguments": "{}"},
                                    }
                                ],
                            },
                        }
                    ]
                }
            )
        return _ProResp({"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "DeepSeek Pro checked."}}]})

    monkeypatch.setattr(httpx.Client, "post", _pro_post)
    pro_agent = LiveDialogueAgent(toolbox, provider="deepseek", model="deepseek-v4-pro")
    pro_reply = pro_agent.send_user_message("看看当前有哪些版本", explicit_api_key="sk-TestProKey1234567890abcdef")
    assert pro_reply == "DeepSeek Pro checked."
    assert len(pro_payloads) == 2
    assert pro_payloads[0]["model"] == "deepseek-v4-pro"
    assert "thinking" not in pro_payloads[0]
    assert len(pro_payloads[0]["messages"]) == 2
    second_turn_msgs = pro_payloads[1]["messages"]
    assistant_with_tool = [m for m in second_turn_msgs if m.get("role") == "assistant" and m.get("tool_calls")]
    assert len(assistant_with_tool) == 1
    assert assistant_with_tool[0]["reasoning_content"] == "First check workspace status before replying."
    tool_msgs = [json.loads(m["content"]) for m in second_turn_msgs if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["project_id"] == "proj_pro"

    # 2. Gateway rejects secret-like/malformed model IDs (422) without input echo, and conflicting client_request_id (409)
    monkeypatch.undo()
    monkeypatch.setenv("MRNAX_ACCESS_TOKEN", "tok_validate_only")
    monkeypatch.delenv("MRNAX_DEV_ALLOW_SERVER_LLM", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-TestGatewayKey1234567890abcdef")

    def _ok_once(self: Any, user_text: str, explicit_api_key: str | None = None, allow_tools: bool = True) -> str:
        return "已记录本轮。"

    monkeypatch.setattr(LiveDialogueAgent, "send_user_message", _ok_once)
    cas = ExecutionCASStore(tmp_path / "cas_val.sqlite")
    app = create_control_plane_app(
        cas,
        projects_root=tmp_path / "sessions_val",
        seed_dir=SEED_DIR,
        access_token="tok_validate_only",
    )
    client = TestClient(app)

    r_bad_model = client.post(
        "/v1/projects/proj_val/agent-chat",
        headers={"X-MRNAX-Access-Token": "tok_validate_only"},
        json={"message": "测试非法模型名", "provider": "deepseek", "model": "sk-secret123"},
    )
    assert r_bad_model.status_code == 422
    assert "sk-secret123" not in r_bad_model.text

    r_first = client.post(
        "/v1/projects/proj_val/agent-chat",
        headers={"X-MRNAX-Access-Token": "tok_validate_only"},
        json={"message": "第一条消息", "client_request_id": "req_conflict_1"},
    )
    assert r_first.status_code == 200

    r_conflict = client.post(
        "/v1/projects/proj_val/agent-chat",
        headers={"X-MRNAX-Access-Token": "tok_validate_only"},
        json={"message": "改成了另一条消息", "client_request_id": "req_conflict_1"},
    )
    assert r_conflict.status_code == 409
    assert "Idempotency conflict" in r_conflict.json()["detail"]


def test_agent_chat_header_provider_routing_and_mismatch_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen_targets: list[tuple[str, str, str]] = []

    class _OkResp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "Provider routed cleanly."}}]}

    def _capture_post(self: httpx.Client, url: str, headers: dict[str, str], json: dict[str, Any]) -> _OkResp:
        seen_targets.append((url, headers["Authorization"], json["model"]))
        return _OkResp()

    monkeypatch.setattr(httpx.Client, "post", _capture_post)
    app = create_control_plane_app(
        ExecutionCASStore(tmp_path / "cas_hdr.sqlite"),
        projects_root=tmp_path / "proj_hdr",
        seed_dir=SEED_DIR,
        access_token="tok_hdr",
    )
    client = TestClient(app)

    # Disagreement between X-MRNAX-LLM-Provider header and JSON body provider fails closed before upstream call.
    r_mismatch = client.post(
        "/v1/projects/proj_hdr_1/agent-chat",
        headers={"X-MRNAX-LLM-Key": "sk-OpenAiTestKey1234567890", "X-MRNAX-LLM-Provider": "openai"},
        json={"message": "路由冲突检查", "provider": "deepseek"},
    )
    assert r_mismatch.status_code == 400
    assert seen_targets == []

    # Unsupported header provider returns 422 before upstream call.
    r_bad_hdr = client.post(
        "/v1/projects/proj_hdr_1/agent-chat",
        headers={"X-MRNAX-LLM-Key": "sk-OpenAiTestKey1234567890", "X-MRNAX-LLM-Provider": "anthropic"},
        json={"message": "不支持的供应商"},
    )
    assert r_bad_hdr.status_code == 422
    assert seen_targets == []

    # Header provider alone (body provider omitted) routes OpenAI key strictly to api.openai.com, never DeepSeek.
    r_openai = client.post(
        "/v1/projects/proj_hdr_1/agent-chat",
        headers={"X-MRNAX-LLM-Key": "sk-OpenAiTestKey1234567890", "X-MRNAX-LLM-Provider": "openai"},
        json={"message": "只走 OpenAI"},
    )
    assert r_openai.status_code == 200
    assert r_openai.json()["provider"] == "openai"
    assert len(seen_targets) == 1
    assert "api.openai.com" in seen_targets[0][0]
    assert "api.deepseek.com" not in seen_targets[0][0]


def test_secret_redaction_before_llm_tool_dispatch_journal_keys_and_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opaque_token = "opaque_gateway_secret_XYZ_987654"
    seen_outbound: list[dict[str, Any]] = []
    snap = json.loads((SEED_DIR / "egfp_reporter_snapshot.json").read_text(encoding="utf-8"))
    egfp_seq = snap["mammalian_reporter_egfp"]["protein_sequence"]

    class _Resp:
        def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
            self._payload = payload
            self.status_code = status_code

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                req = httpx.Request("POST", "https://api.deepseek.com/v1/chat/completions")
                resp = httpx.Response(self.status_code, json=self._payload, request=req)
                raise httpx.HTTPStatusError("upstream err", request=req, response=resp)

        def json(self) -> dict[str, Any]:
            return self._payload

    turn = {"n": 0}

    def _post(self: httpx.Client, url: str, headers: dict[str, str], json: dict[str, Any]) -> _Resp:
        seen_outbound.append(copy.deepcopy(json))
        turn["n"] += 1
        if turn["n"] == 1:
            return _Resp(
                {
                    "choices": [
                        {
                            "finish_reason": "tool_calls",
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "tc_redact_1",
                                        "type": "function",
                                        "function": {
                                            "name": "design_constructs",
                                            "arguments": (
                                                '{"user_utterance": "design with token ' + opaque_token + '",'
                                                f'"protein_sequence": "{egfp_seq}"}}'
                                            ),
                                        },
                                    }
                                ],
                            },
                        }
                    ]
                }
            )
        if turn["n"] == 2:
            return _Resp(
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": f"Completed cleanly without exposing {opaque_token}",
                            },
                        }
                    ]
                }
            )
        return _Resp({"error": {"code": "sk-LeakInCode1234567890abcdef"}}, status_code=401)

    monkeypatch.setattr(httpx.Client, "post", _post)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-VaultDeepSeekSecret1234567890")
    projects_root = tmp_path / "proj_sec"
    app = create_control_plane_app(
        ExecutionCASStore(tmp_path / "cas_sec.sqlite"),
        projects_root=projects_root,
        seed_dir=SEED_DIR,
        access_token=opaque_token,
    )
    client = TestClient(app, headers={"X-MRNAX-Access-Token": opaque_token, "X-MRNAX-Project-Key": "pk_ProjCap1234567890abcdef"})

    # Credential-shaped client_request_id rejected with 422 and zero echo
    bad_cid = client.post(
        "/v1/projects/proj_sec_1/agent-chat",
        json={"message": "hello", "client_request_id": "sk-SecretReqId1234567890"},
    )
    assert bad_cid.status_code == 422
    assert "sk-SecretReqId1234567890" not in bad_cid.text

    r_ok = client.post(
        "/v1/projects/proj_sec_1/agent-chat",
        json={"message": f"请设计，不小心带了口令 https://example.com/#token={opaque_token}", "client_request_id": "req_safe_sec_1"},
    )
    assert r_ok.status_code == 200
    body = r_ok.json()
    assert opaque_token not in json.dumps(seen_outbound)
    assert opaque_token not in json.dumps(body["intent"])
    assert opaque_token not in body["assistant_reply"]
    assert "[REDACTED_API_KEY]" in json.dumps(body["intent"])

    # Journal keys must be hashed, never storing plaintext client_request_id
    raw_req_journal = (projects_root / "proj_sec_1" / "agent_chat_requests.json").read_text(encoding="utf-8")
    assert "req_safe_sec_1" not in raw_req_journal

    # Readback through /state also redacts opaque configured_token even without access token header
    st_res = client.get(
        "/v1/projects/proj_sec_1/state",
        headers={"X-MRNAX-Project-Key": "pk_ProjCap1234567890abcdef"},
    )
    assert st_res.status_code == 200
    assert opaque_token not in st_res.text

    # Upstream error code shaped like a credential is never copied into 502 detail
    r_err = client.post(
        "/v1/projects/proj_sec_1/agent-chat",
        json={"message": "触发上游错误码清理"},
    )
    assert r_err.status_code == 502
    assert "sk-LeakInCode" not in r_err.text


def test_dialogue_agent_rejects_malformed_assistant_envelopes_before_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    malformed_envelopes = [
        {"choices": [{}]},
        {"choices": [{"message": {"role": "user", "content": "hi"}}]},
        {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": []}}]},
        {"choices": [{"message": {"role": "assistant", "content": ["list", "not", "str"]}}]},
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{"type": "function", "function": {"name": "get_job_status", "arguments": "{}"}}],
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{"id": "tc_noargs", "type": "function", "function": {"name": "get_job_status"}}],
                    }
                }
            ]
        },
    ]

    class _EnvResp:
        def __init__(self, payload: dict[str, Any]) -> None:
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return self._payload

    projects_root = tmp_path / "proj_env"
    app = create_control_plane_app(
        ExecutionCASStore(tmp_path / "cas_env.sqlite"),
        projects_root=projects_root,
        seed_dir=SEED_DIR,
        access_token="tok_env",
    )
    client = TestClient(app, headers={"X-MRNAX-Access-Token": "tok_env"})

    extra_malformed = [
        {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": None, "tool_calls": [{"id": "tc_notype", "function": {"name": "get_project_state", "arguments": "{}"}}]}}]},
        {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": None, "tool_calls": [{"id": "dup_1", "type": "function", "function": {"name": "get_project_state", "arguments": "{}"}}, {"id": "dup_1", "type": "function", "function": {"name": "get_project_state", "arguments": "{}"}}]}}]},
        {"choices": [{"finish_reason": "length", "message": {"role": "assistant", "content": "partial truncated reply"}}]},
        {"choices": [{"finish_reason": None, "message": {"role": "assistant", "content": "null finish reason text"}}]},
        {"choices": [{"finish_reason": None, "message": {"role": "assistant", "content": None, "tool_calls": [{"id": "tc_null_fr", "type": "function", "function": {"name": "get_job_status", "arguments": "{}"}}]}}]},
    ]
    all_malformed = [*malformed_envelopes, *extra_malformed]
    post_calls = 0

    for idx, bad_env in enumerate(all_malformed):
        def _fake_post(*a, _p=bad_env, **kw):
            nonlocal post_calls
            post_calls += 1
            return _EnvResp(_p)

        monkeypatch.setattr(httpx.Client, "post", _fake_post)
        r = client.post(
            "/v1/projects/proj_bad_envs/agent-chat",
            json={"message": f"envelope check {idx}", "client_request_id": f"req_bad_env_{idx}"},
        )
        assert r.status_code == 502, f"failed on envelope {idx}"
    assert post_calls == len(all_malformed)
    assert not (projects_root / "proj_bad_envs" / "agent_chat_history.json").exists()


def test_agent_chat_recovery_readonly_vs_mutating_interleaved_and_crash_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snap = json.loads((SEED_DIR / "egfp_reporter_snapshot.json").read_text(encoding="utf-8"))
    egfp_seq = snap["mammalian_reporter_egfp"]["protein_sequence"]
    projects_root = tmp_path / "proj_recov"
    app = create_control_plane_app(
        ExecutionCASStore(tmp_path / "cas_recov.sqlite"),
        projects_root=projects_root,
        seed_dir=SEED_DIR,
        access_token="tok_recov",
    )
    client = TestClient(app, headers={"X-MRNAX-Access-Token": "tok_recov"})

    # Seed v01 baseline
    t1 = client.post(
        "/v1/projects/proj_interleave/turn",
        json={"mode": "initial", "user_utterance": "seed v1", "protein_sequence": egfp_seq, "restriction_enzymes": ["BsaI"]},
    )
    assert t1.status_code == 200
    assert t1.json()["intent"]["intent_version"] == 1

    # 1. Read-only lookup followed by 502 must keep allow_tools=True on retry so mutation runs
    ro_calls = {"n": 0}

    def _ro_then_mut(self: LiveDialogueAgent, user_text: str, **kwargs: Any) -> str:
        ro_calls["n"] += 1
        if ro_calls["n"] == 1:
            assert kwargs.get("allow_tools", True) is True
            res = self.toolbox.get_job_status()
            self.tool_call_trace.append({"tool_call_id": "tc_ro", "name": "get_job_status", "arguments": {}, "result": res})
            raise httpx.RequestError("network blip after read-only")
        assert kwargs.get("allow_tools") is True
        res = self.toolbox.refine_construct(user_utterance=user_text, restriction_enzymes=["BspQI"])
        self.tool_call_trace.append({"tool_call_id": "tc_mut", "name": "refine_construct", "arguments": {"restriction_enzymes": ["BspQI"]}, "result": res})
        return "已通过只读失败后的重试完成 BspQI 微调。"

    monkeypatch.setattr(LiveDialogueAgent, "send_user_message", _ro_then_mut)
    r_ro_fail = client.post(
        "/v1/projects/proj_interleave/agent-chat",
        json={"message": "查状态后改 BspQI", "client_request_id": "req_ro_retry"},
    )
    assert r_ro_fail.status_code == 502
    r_ro_ok = client.post(
        "/v1/projects/proj_interleave/agent-chat",
        json={"message": "查状态后改 BspQI", "client_request_id": "req_ro_retry"},
    )
    assert r_ro_ok.status_code == 200
    assert r_ro_ok.json()["intent"]["intent_version"] == 2

    # 2. Request A mutates v03 and fails 502; interleaved Request B mutates v04; retrying Request A returns v03 (not v04!)
    inter_calls = {"a": 0}

    def _interleaved_send(self: LiveDialogueAgent, user_text: str, **kwargs: Any) -> str:
        if "请求A" in user_text:
            inter_calls["a"] += 1
            if inter_calls["a"] == 1:
                assert kwargs.get("allow_tools", True) is True
                res = self.toolbox.refine_construct(user_utterance="请求A改SapI", restriction_enzymes=["SapI"])
                self.tool_call_trace.append({"tool_call_id": "tc_a", "name": "refine_construct", "arguments": {"restriction_enzymes": ["SapI"]}, "result": res})
                raise httpx.RequestError("drop after v3")
            assert kwargs.get("allow_tools") is False
            assert any(
                m.get("role") == "user" and '"active_intent_version": 3' in m.get("content", "")
                for m in self.messages
            )
            return "请求A恢复完成（v03）。"
        res_b = self.toolbox.refine_construct(user_utterance="请求B改EcoRI", restriction_enzymes=["EcoRI"])
        self.tool_call_trace.append({"tool_call_id": "tc_b", "name": "refine_construct", "arguments": {}, "result": res_b})
        return "请求B完成（v04）。"

    monkeypatch.setattr(LiveDialogueAgent, "send_user_message", _interleaved_send)
    ra_fail = client.post(
        "/v1/projects/proj_interleave/agent-chat",
        json={"message": "请求A改为SapI", "client_request_id": "req_a_interleave"},
    )
    assert ra_fail.status_code == 502

    rb_ok = client.post(
        "/v1/projects/proj_interleave/agent-chat",
        json={"message": "请求B改为EcoRI", "client_request_id": "req_b_interleave"},
    )
    assert rb_ok.status_code == 200
    assert rb_ok.json()["intent"]["intent_version"] == 4

    ra_retry = client.post(
        "/v1/projects/proj_interleave/agent-chat",
        json={"message": "请求A改为SapI", "client_request_id": "req_a_interleave"},
    )
    assert ra_retry.status_code == 200
    ra_data = ra_retry.json()
    assert ra_data["intent"]["intent_version"] == 3
    assert ra_data["intent_diff"]["v1_version"] == 2
    assert ra_data["intent_diff"]["v2_version"] == 3
    assert ra_data["intent_diff"]["v2_enzymes"] == ["SapI"]

    # 3. Crash Window 1: intent written while journal only shows in_progress -> recovery uses allow_tools=False (no v06)
    cw1 = {"n": 0}

    class _SimulatedWorkerKill(Exception):
        pass

    def _crash_window_1(self: LiveDialogueAgent, user_text: str, **kwargs: Any) -> str:
        cw1["n"] += 1
        if cw1["n"] == 1:
            # Mutate session directly WITHOUT appending to agent.tool_call_trace and abort before journal update
            self.toolbox.refine_construct(user_utterance="cw1_mutation", restriction_enzymes=["BsmBI"])
            raise _SimulatedWorkerKill("simulated worker crash before journal write")
        assert kwargs.get("allow_tools") is False
        assert "[RECOVERED_TOOL_RESULTS_FROM_PRIOR_ATTEMPT]" in user_text
        return "崩溃窗口1已安全恢复。"

    monkeypatch.setattr(LiveDialogueAgent, "send_user_message", _crash_window_1)
    with pytest.raises(_SimulatedWorkerKill):
        client.post(
            "/v1/projects/proj_interleave/agent-chat",
            json={"message": "崩溃窗口1测试", "client_request_id": "req_cw1"},
        )
    req_file = projects_root / "proj_interleave" / "agent_chat_requests.json"
    import hashlib

    cw1_hash = hashlib.sha256(b"req_cw1").hexdigest()
    req_doc_cw1 = json.loads(req_file.read_text(encoding="utf-8"))
    assert req_doc_cw1["schema_version"] == 2
    assert req_doc_cw1["entries"][cw1_hash]["status"] == "in_progress"
    r_cw1_ok = client.post(
        "/v1/projects/proj_interleave/agent-chat",
        json={"message": "崩溃窗口1测试", "client_request_id": "req_cw1"},
    )
    assert r_cw1_ok.status_code == 200
    assert r_cw1_ok.json()["intent"]["intent_version"] == 5

    # 4. Crash Window 2: history contains client_request_key even if requests journal stayed incomplete
    req_doc = json.loads(req_file.read_text(encoding="utf-8"))
    req_doc["entries"][cw1_hash]["status"] = "partial_mutating_committed"
    req_file.write_text(json.dumps(req_doc), encoding="utf-8")

    monkeypatch.setattr(LiveDialogueAgent, "send_user_message", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("LLM must not run")))
    r_cw2 = client.post(
        "/v1/projects/proj_interleave/agent-chat",
        json={"message": "崩溃窗口1测试", "client_request_id": "req_cw1"},
    )
    assert r_cw2.status_code == 200
    assert r_cw2.json()["intent"]["intent_version"] == 5


def test_first_call_byo_502_rollback_or_key_delivery_and_history_only_legacy_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snap = json.loads((SEED_DIR / "egfp_reporter_snapshot.json").read_text(encoding="utf-8"))
    egfp_seq = snap["mammalian_reporter_egfp"]["protein_sequence"]
    projects_root = tmp_path / "proj_byo"
    app = create_control_plane_app(
        ExecutionCASStore(tmp_path / "cas_byo.sqlite"),
        projects_root=projects_root,
        seed_dir=SEED_DIR,
        access_token="",
    )

    # 1. History-only unbound legacy directory (0 intents) MUST block anonymous ownership & /agent-chat
    legacy_dir = projects_root / "legacy_history_only"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    (legacy_dir / "agent_chat_history.json").write_text(
        json.dumps([{"user_message": "hi", "assistant_reply": "confidential"}]),
        encoding="utf-8",
    )
    anon_client = TestClient(app)
    assert anon_client.get("/v1/projects/legacy_history_only/state").json()["chat_history"] == []
    r_leg = anon_client.post(
        "/v1/projects/legacy_history_only/agent-chat",
        headers={"X-MRNAX-LLM-Key": "sk-ByoUserKey1234567890abcdef"},
        json={"message": "read my legacy chat"},
    )
    assert r_leg.status_code == 403

    # 2. Brand-new BYO REST client omitting X-MRNAX-Project-Key: 502 BEFORE mutation rolls back ownership -> clean retry
    call_n = {"n": 0}

    def _fail_then_ok_no_mut(self: LiveDialogueAgent, user_text: str, **kwargs: Any) -> str:
        call_n["n"] += 1
        if call_n["n"] == 1:
            raise httpx.RequestError("first-call upstream fail")
        return "重试成功！"

    monkeypatch.setattr(LiveDialogueAgent, "send_user_message", _fail_then_ok_no_mut)
    c1 = TestClient(app)
    r_pre_fail = c1.post(
        "/v1/projects/byo_brand_new_1/agent-chat",
        headers={"X-MRNAX-LLM-Key": "sk-ByoUserKey1234567890abcdef"},
        json={"message": "hello", "client_request_id": "req_byo_1"},
    )
    assert r_pre_fail.status_code == 502
    assert not (projects_root / "byo_brand_new_1" / "owner_token_sha256.txt").exists()

    # Same client without project key retries cleanly with zero 403 lockout
    r_pre_ok = c1.post(
        "/v1/projects/byo_brand_new_1/agent-chat",
        headers={"X-MRNAX-LLM-Key": "sk-ByoUserKey1234567890abcdef"},
        json={"message": "hello", "client_request_id": "req_byo_1"},
    )
    assert r_pre_ok.status_code == 200
    assert r_pre_ok.json()["project_key"].startswith("pk_")

    # 3. Brand-new BYO REST client omitting X-MRNAX-Project-Key: 502 AFTER mutation delivers minted key on 502 so retry works
    mut_calls = {"n": 0}

    def _fail_after_mut_then_recover(self: LiveDialogueAgent, user_text: str, **kwargs: Any) -> str:
        mut_calls["n"] += 1
        if mut_calls["n"] == 1:
            res = self.toolbox.design_constructs(user_utterance=user_text, protein_sequence=egfp_seq)
            self.tool_call_trace.append({"tool_call_id": "tc_byo_mut", "name": "design_constructs", "arguments": {}, "result": res})
            raise httpx.RequestError("drop after v1 commit")
        assert kwargs.get("allow_tools") is False
        return "凭 502 返回的 project_key 恢复完成！"

    monkeypatch.setattr(LiveDialogueAgent, "send_user_message", _fail_after_mut_then_recover)
    c_header_only = TestClient(app)
    r_mut_fail = c_header_only.post(
        "/v1/projects/byo_brand_new_2/agent-chat",
        headers={"X-MRNAX-LLM-Key": "sk-ByoUserKey1234567890abcdef"},
        json={"message": "设计 eGFP", "client_request_id": "req_byo_mut_1"},
    )
    assert r_mut_fail.status_code == 502
    delivered_pk = r_mut_fail.headers.get("X-MRNAX-Project-Key") or r_mut_fail.json().get("project_key")
    assert delivered_pk and delivered_pk.startswith("pk_")

    r_mut_ok = c_header_only.post(
        "/v1/projects/byo_brand_new_2/agent-chat",
        headers={
            "X-MRNAX-LLM-Key": "sk-ByoUserKey1234567890abcdef",
            "X-MRNAX-Project-Key": delivered_pk,
        },
        json={"message": "设计 eGFP", "client_request_id": "req_byo_mut_1"},
    )
    assert r_mut_ok.status_code == 200
    assert r_mut_ok.json()["intent"]["intent_version"] == 1


def test_r3_model_token_guard_short_secret_and_state_redaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    short_tok = "abCD"
    byo_short = "k9Zx"
    monkeypatch.setenv("MRNAX_DEV_ALLOW_SERVER_LLM", "1")
    projects_root = tmp_path / "proj_sec_r3"
    app = create_control_plane_app(
        ExecutionCASStore(tmp_path / "cas_sec_r3.sqlite"),
        projects_root=projects_root,
        seed_dir=SEED_DIR,
        access_token=short_tok,
    )
    client = TestClient(app, headers={"X-MRNAX-Access-Token": short_tok})

    # Passing configured access token as model must fail closed with 422
    r_tok_model = client.post(
        "/v1/projects/proj_sec_1/agent-chat",
        json={"message": "check model guard", "model": short_tok, "client_request_id": "req_sec_tok_model"},
    )
    assert r_tok_model.status_code == 422
    assert short_tok not in r_tok_model.text

    class _OkResp:
        def __init__(self, content: str) -> None:
            self._content = content

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": self._content}}]}

    snap = json.loads((SEED_DIR / "egfp_reporter_snapshot.json").read_text(encoding="utf-8"))
    egfp_seq = snap["mammalian_reporter_egfp"]["protein_sequence"]

    # Create deterministic turn whose utterance embeds short_tok, then refine embedding cap_secret -> test /state redaction
    r_turn = client.post(
        "/v1/projects/proj_sec_1/turn",
        json={
            "mode": "initial",
            "user_utterance": f"Design eGFP using {short_tok}",
            "protein_sequence": egfp_seq,
        },
    )
    assert r_turn.status_code == 200
    set_cookie_val = r_turn.headers.get("set-cookie", "")
    cap_secret = ""
    if "mrnax_pk_" in set_cookie_val:
        cap_secret = set_cookie_val.split("=", 1)[1].split(";", 1)[0].strip()
    if cap_secret:
        r_ref_sec = client.post(
            "/v1/projects/proj_sec_1/turn",
            json={
                "mode": "refine",
                "user_utterance": f"Switch to BspQI keep {short_tok} {cap_secret}",
                "restriction_enzymes": ["BspQI"],
            },
        )
        assert r_ref_sec.status_code == 200

    def _leak_post(self: httpx.Client, url: str, headers: dict[str, str], json: dict[str, Any]) -> _OkResp:
        return _OkResp(f"boundary safe xabCDy leaked {short_tok} and {byo_short} and {cap_secret}")

    monkeypatch.setattr(httpx.Client, "post", _leak_post)
    r_chat = client.post(
        "/v1/projects/proj_sec_1/agent-chat",
        headers={"X-MRNAX-LLM-Key": byo_short},
        json={"message": f"Say hi {short_tok} {byo_short}", "client_request_id": "req_sec_short_1"},
    )
    assert r_chat.status_code == 200
    reply_txt = r_chat.json()["assistant_reply"]
    assert "xabCDy" in reply_txt
    assert f" {short_tok} " not in reply_txt
    assert byo_short not in reply_txt
    assert byo_short not in r_chat.text
    if cap_secret:
        assert cap_secret not in reply_txt

    r_state = client.get("/v1/projects/proj_sec_1/state")
    assert r_state.status_code == 200
    state_dump = r_state.text
    assert "xabCDy" in state_dump
    assert f" {short_tok}" not in state_dump and f"{short_tok} " not in state_dump
    assert short_tok not in json.dumps(r_state.json()["intents"])
    assert byo_short not in json.dumps(r_state.json()["chat_history"])
    if cap_secret:
        assert cap_secret not in state_dump


def test_r3_readonly_timeout_retry_does_not_claim_concurrent_revision_b(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MRNAX_DEV_ALLOW_SERVER_LLM", "1")
    snap = json.loads((SEED_DIR / "egfp_reporter_snapshot.json").read_text(encoding="utf-8"))
    egfp_seq = snap["mammalian_reporter_egfp"]["protein_sequence"]
    projects_root = tmp_path / "proj_conc_r3"
    app = create_control_plane_app(
        ExecutionCASStore(tmp_path / "cas_conc_r3.sqlite"),
        projects_root=projects_root,
        seed_dir=SEED_DIR,
        access_token="tok_conc",
    )
    client = TestClient(app, headers={"X-MRNAX-Access-Token": "tok_conc"})

    r_init = client.post(
        "/v1/projects/proj_conc/turn",
        json={"mode": "initial", "user_utterance": "seed v1", "protein_sequence": egfp_seq},
    )
    assert r_init.status_code == 200
    assert r_init.json()["intent"]["intent_version"] == 1

    class _OkResp:
        def __init__(self, payload: dict[str, Any]) -> None:
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return self._payload

    captured_payloads: list[dict[str, Any]] = []
    scenario = {"mode": "timeout_a"}

    def _conc_post(self: httpx.Client, url: str, headers: dict[str, str], json: dict[str, Any]) -> _OkResp:
        captured_payloads.append(copy.deepcopy(json))
        if scenario["mode"] == "timeout_a":
            raise httpx.ReadTimeout("Request A timed out before tool calls")
        if scenario["mode"] == "mutate_b":
            scenario["mode"] = "finish_b"
            return _OkResp(
                {
                    "choices": [
                        {
                            "finish_reason": "tool_calls",
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "tc_b_mut",
                                        "type": "function",
                                        "function": {
                                            "name": "refine_construct",
                                            "arguments": '{"user_utterance": "Switch to BspQI", "restriction_enzymes": ["BspQI"]}',
                                        },
                                    }
                                ],
                            },
                        }
                    ]
                }
            )
        if scenario["mode"] == "finish_b":
            return _OkResp({"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "B updated to v02."}}]})
        return _OkResp({"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "A read-only reply."}}]})

    monkeypatch.setattr(httpx.Client, "post", _conc_post)

    # 1. Request A times out before issuing any mutating tool call
    r_a1 = client.post(
        "/v1/projects/proj_conc/agent-chat",
        json={"message": "Explain current design", "client_request_id": "req_A_ro"},
    )
    assert r_a1.status_code in (502, 504)

    # 2. Concurrent Request B mutates project to v02
    scenario["mode"] = "mutate_b"
    r_b = client.post(
        "/v1/projects/proj_conc/agent-chat",
        json={"message": "Switch to BspQI", "client_request_id": "req_B_mut"},
    )
    assert r_b.status_code == 200
    assert r_b.json()["intent"]["intent_version"] == 2

    # 3. Request A retries -> must retain allow_tools=True and NOT claim B's v02 tool call
    captured_payloads.clear()
    scenario["mode"] = "retry_a"
    r_a2 = client.post(
        "/v1/projects/proj_conc/agent-chat",
        json={"message": "Explain current design", "client_request_id": "req_A_ro"},
    )
    assert r_a2.status_code == 200
    assert r_a2.json()["assistant_reply"] == "A read-only reply."
    assert r_a2.json()["intent"]["intent_version"] == 1
    assert r_a2.json().get("tool_calls", []) == []
    assert len(captured_payloads) == 1
    assert "tools" in captured_payloads[0]


def test_r3_interrupted_refinement_and_multi_call_batch_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MRNAX_DEV_ALLOW_SERVER_LLM", "1")
    snap = json.loads((SEED_DIR / "egfp_reporter_snapshot.json").read_text(encoding="utf-8"))
    egfp_seq = snap["mammalian_reporter_egfp"]["protein_sequence"]
    projects_root = tmp_path / "proj_rec_r3"
    app = create_control_plane_app(
        ExecutionCASStore(tmp_path / "cas_rec_r3.sqlite"),
        projects_root=projects_root,
        seed_dir=SEED_DIR,
        access_token="tok_rec",
    )
    client = TestClient(app, headers={"X-MRNAX-Access-Token": "tok_rec"})

    r_init = client.post(
        "/v1/projects/proj_rec_1/turn",
        json={"mode": "initial", "user_utterance": "seed v1", "protein_sequence": egfp_seq},
    )
    assert r_init.status_code == 200

    class _Resp:
        def __init__(self, payload: dict[str, Any]) -> None:
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return self._payload

    class _SimulatedCrash(BaseException):
        pass

    # Case 1: Interrupted refinement after _persist_intent before execute_design completes
    orig_exec = ProjectHarnessSession.execute_design
    orig_refine = AtomicToolbox.refine_construct
    refine_calls = {"n": 0}
    exec_calls = {"n": 0}

    def _counting_refine(self: AtomicToolbox, *a: Any, **kw: Any) -> dict[str, Any]:
        refine_calls["n"] += 1
        return orig_refine(self, *a, **kw)

    def _fail_first_exec(self: ProjectHarnessSession, *a: Any, **kw: Any) -> Any:
        exec_calls["n"] += 1
        if exec_calls["n"] == 1:
            raise _SimulatedCrash("crash before execute_design completes")
        return orig_exec(self, *a, **kw)

    monkeypatch.setattr(AtomicToolbox, "refine_construct", _counting_refine)
    monkeypatch.setattr(ProjectHarnessSession, "execute_design", _fail_first_exec)

    llm_calls = {"n": 0}

    def _llm_c1(self: httpx.Client, url: str, headers: dict[str, str], json: dict[str, Any]) -> _Resp:
        llm_calls["n"] += 1
        if llm_calls["n"] == 1:
            return _Resp(
                {
                    "choices": [
                        {
                            "finish_reason": "tool_calls",
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "tc_ref_c1",
                                        "type": "function",
                                        "function": {
                                            "name": "refine_construct",
                                            "arguments": '{"user_utterance": "Switch to BspQI", "restriction_enzymes": ["BspQI"]}',
                                        },
                                    }
                                ],
                            },
                        }
                    ]
                }
            )
        return _Resp({"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "Finished v2 after recovery."}}]})

    monkeypatch.setattr(httpx.Client, "post", _llm_c1)
    with pytest.raises(_SimulatedCrash):
        client.post(
            "/v1/projects/proj_rec_1/agent-chat",
            json={"message": "Switch to BspQI", "client_request_id": "req_c1_rec"},
        )
    assert refine_calls["n"] == 1

    r_c1_ok = client.post(
        "/v1/projects/proj_rec_1/agent-chat",
        json={"message": "Switch to BspQI", "client_request_id": "req_c1_rec"},
    )
    assert r_c1_ok.status_code == 200
    assert refine_calls["n"] == 1  # Not repeated!
    assert r_c1_ok.json()["intent"]["intent_version"] == 2
    assert r_c1_ok.json()["report"] is not None
    assert r_c1_ok.json()["report"]["intent_version"] == 2

    # Case 2: 2-call batch interrupted after call 1 executes call 2 on recovery without repeating call 1
    monkeypatch.setattr(AtomicToolbox, "refine_construct", orig_refine)
    monkeypatch.setattr(ProjectHarnessSession, "execute_design", orig_exec)

    r_init_2 = client.post(
        "/v1/projects/proj_rec_2/turn",
        json={"mode": "initial", "user_utterance": "seed v1", "protein_sequence": egfp_seq},
    )
    assert r_init_2.status_code == 200

    batch_counts = {"refine": 0, "status": 0}
    orig_status = AtomicToolbox.get_job_status

    def _track_refine(self: AtomicToolbox, *a: Any, **kw: Any) -> dict[str, Any]:
        batch_counts["refine"] += 1
        return orig_refine(self, *a, **kw)

    def _crash_first_status(self: AtomicToolbox, *a: Any, **kw: Any) -> dict[str, Any]:
        batch_counts["status"] += 1
        if batch_counts["status"] == 1:
            raise _SimulatedCrash("crash on call 2 of batch")
        return orig_status(self, *a, **kw)

    monkeypatch.setattr(AtomicToolbox, "refine_construct", _track_refine)
    monkeypatch.setattr(AtomicToolbox, "get_job_status", _crash_first_status)

    batch_llm = {"n": 0}

    def _llm_batch(self: httpx.Client, url: str, headers: dict[str, str], json: dict[str, Any]) -> _Resp:
        batch_llm["n"] += 1
        if batch_llm["n"] == 1:
            return _Resp(
                {
                    "choices": [
                        {
                            "finish_reason": "tool_calls",
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "tc_batch_1",
                                        "type": "function",
                                        "function": {
                                            "name": "refine_construct",
                                            "arguments": '{"user_utterance": "Also clear XbaI", "restriction_enzymes": ["BspQI", "XbaI"]}',
                                        },
                                    },
                                    {
                                        "id": "tc_batch_2",
                                        "type": "function",
                                        "function": {"name": "get_job_status", "arguments": "{}"},
                                    },
                                ],
                            },
                        }
                    ]
                }
            )
        return _Resp({"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "Batch completed cleanly."}}]})

    monkeypatch.setattr(httpx.Client, "post", _llm_batch)
    with pytest.raises(_SimulatedCrash):
        client.post(
            "/v1/projects/proj_rec_2/agent-chat",
            json={"message": "Batch test", "client_request_id": "req_batch_rec_2"},
        )
    assert batch_counts == {"refine": 1, "status": 1}

    r_b_ok = client.post(
        "/v1/projects/proj_rec_2/agent-chat",
        json={"message": "Batch test", "client_request_id": "req_batch_rec_2"},
    )
    assert r_b_ok.status_code == 200
    assert batch_counts == {"refine": 1, "status": 2}


def test_r3_double_timeout_preserves_recovered_tool_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MRNAX_DEV_ALLOW_SERVER_LLM", "1")
    snap = json.loads((SEED_DIR / "egfp_reporter_snapshot.json").read_text(encoding="utf-8"))
    egfp_seq = snap["mammalian_reporter_egfp"]["protein_sequence"]
    app = create_control_plane_app(
        ExecutionCASStore(tmp_path / "cas_dbl.sqlite"),
        projects_root=tmp_path / "proj_dbl",
        seed_dir=SEED_DIR,
        access_token="tok_dbl",
    )
    client = TestClient(app, headers={"X-MRNAX-Access-Token": "tok_dbl"})

    orig_design = AtomicToolbox.design_constructs
    design_count = {"n": 0}

    def _count_design(self: AtomicToolbox, *a: Any, **kw: Any) -> dict[str, Any]:
        design_count["n"] += 1
        return orig_design(self, *a, **kw)

    monkeypatch.setattr(AtomicToolbox, "design_constructs", _count_design)

    class _Resp:
        def __init__(self, payload: dict[str, Any]) -> None:
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return self._payload

    calls: list[dict[str, Any]] = []
    step = {"n": 0}

    def _dbl_post(self: httpx.Client, url: str, headers: dict[str, str], json: dict[str, Any]) -> _Resp:
        calls.append(copy.deepcopy(json))
        step["n"] += 1
        if step["n"] == 1:
            return _Resp(
                {
                    "choices": [
                        {
                            "finish_reason": "tool_calls",
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "tc_dbl_1",
                                        "type": "function",
                                        "function": {
                                            "name": "design_constructs",
                                            "arguments": f'{{"user_utterance": "Design eGFP", "protein_sequence": "{egfp_seq}"}}',
                                        },
                                    }
                                ],
                            },
                        }
                    ]
                }
            )
        if step["n"] in (2, 3):
            raise httpx.ReadTimeout(f"timeout {step['n']}")
        return _Resp({"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "Completed on retry #3."}}]})

    monkeypatch.setattr(httpx.Client, "post", _dbl_post)
    r1 = client.post("/v1/projects/proj_dbl_1/agent-chat", json={"message": "Design eGFP", "client_request_id": "req_dbl_1"})
    assert r1.status_code in (502, 504)
    r2 = client.post("/v1/projects/proj_dbl_1/agent-chat", json={"message": "Design eGFP", "client_request_id": "req_dbl_1"})
    assert r2.status_code in (502, 504)
    assert "tools" not in calls[-1]
    r3 = client.post("/v1/projects/proj_dbl_1/agent-chat", json={"message": "Design eGFP", "client_request_id": "req_dbl_1"})
    assert r3.status_code == 200
    assert "tools" not in calls[-1]
    assert design_count["n"] == 1
    assert any(tc.get("name") == "design_constructs" for tc in r3.json()["tool_calls"])


def test_r3_legacy_plaintext_key_migration_and_unresolved_eviction_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MRNAX_DEV_ALLOW_SERVER_LLM", "1")
    projects_root = tmp_path / "proj_mig"
    proj_dir = projects_root / "proj_mig_1"
    proj_dir.mkdir(parents=True, exist_ok=True)

    app = create_control_plane_app(
        ExecutionCASStore(tmp_path / "cas_mig.sqlite"),
        projects_root=projects_root,
        seed_dir=SEED_DIR,
        access_token="tok_mig",
    )
    client = TestClient(app, headers={"X-MRNAX-Access-Token": "tok_mig"})

    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}]}

    llm_calls = {"n": 0}

    def _ok_post(self: httpx.Client, url: str, headers: dict[str, str], json: dict[str, Any]) -> _Resp:
        llm_calls["n"] += 1
        return _Resp()

    monkeypatch.setattr(httpx.Client, "post", _ok_post)

    # First execute 1 real request so we capture canonical request_hash for legacy plaintext migration
    r_seed = client.post(
        "/v1/projects/proj_mig_1/agent-chat",
        json={"message": "legacy replay msg", "client_request_id": "legacy_plain_req"},
    )
    assert r_seed.status_code == 200
    req_file = proj_dir / "agent_chat_requests.json"
    stored_doc = json.loads(req_file.read_text(encoding="utf-8"))
    assert stored_doc["schema_version"] == 2
    stored = stored_doc["entries"]
    hashed_legacy_id = hashlib.sha256(b"legacy_plain_req").hexdigest()
    assert hashed_legacy_id in stored

    # Rewrite file in legacy v1 flat format with plaintext key "legacy_plain_req", 52 completed entries, + "unresolved_req"
    legacy_record = stored[hashed_legacy_id]
    seeded_entries: dict[str, Any] = {
        "legacy_plain_req": legacy_record,
        "unresolved_req": {"status": "in_progress", "request_digest": "abc", "updated_at": "2026-11-01T00:00:00Z"},
    }
    for idx in range(52):
        seeded_entries[f"seed_comp_{idx}"] = {
            **legacy_record,
            "status": "completed",
            "updated_at": f"2026-11-02T00:{idx:02d}:00Z",
        }
    req_file.write_text(json.dumps(seeded_entries), encoding="utf-8")

    # Replaying legacy_plain_req must hit migrated cache without calling LLM AND persist v2 migration immediately
    llm_calls["n"] = 0
    r_replay = client.post(
        "/v1/projects/proj_mig_1/agent-chat",
        json={"message": "legacy replay msg", "client_request_id": "legacy_plain_req"},
    )
    assert r_replay.status_code == 200
    assert llm_calls["n"] == 0
    post_hit_doc = json.loads(req_file.read_text(encoding="utf-8"))
    assert post_hit_doc["schema_version"] == 2
    assert "legacy_plain_req" not in post_hit_doc["entries"]

    # One new request triggers _save_hashed_requests -> unresolved entry must survive max-50 completed cap
    r_new = client.post(
        "/v1/projects/proj_mig_1/agent-chat",
        json={"message": "fresh request", "client_request_id": "comp_req_new"},
    )
    assert r_new.status_code == 200

    final_doc = json.loads(req_file.read_text(encoding="utf-8"))
    assert final_doc["schema_version"] == 2
    final_store = final_doc["entries"]
    assert "legacy_plain_req" not in final_store
    assert "unresolved_req" not in final_store
    unresolved_hash = hashlib.sha256(b"unresolved_req").hexdigest()
    assert unresolved_hash in final_store
    completed_count = sum(1 for v in final_store.values() if isinstance(v, dict) and v.get("status") == "completed")
    assert completed_count <= 50


def test_r3_first_call_byo_crash_recovery_and_https_secure_cookie(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snap = json.loads((SEED_DIR / "egfp_reporter_snapshot.json").read_text(encoding="utf-8"))
    egfp_seq = snap["mammalian_reporter_egfp"]["protein_sequence"]
    projects_root = tmp_path / "proj_byo_rec"
    app = create_control_plane_app(
        ExecutionCASStore(tmp_path / "cas_byo_rec.sqlite"),
        projects_root=projects_root,
        seed_dir=SEED_DIR,
        access_token="",
    )
    client = TestClient(app)

    step = {"n": 0}

    def _post(self: httpx.Client, url: str, headers: dict[str, str], json: dict[str, Any]) -> Any:
        step["n"] += 1

        class _Ok:
            def __init__(self, payload: dict[str, Any]) -> None:
                self._payload = payload

            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, Any]:
                return self._payload

        if step["n"] == 1:
            return _Ok(
                {
                    "choices": [
                        {
                            "finish_reason": "tool_calls",
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "tc_byo_init",
                                        "type": "function",
                                        "function": {
                                            "name": "design_constructs",
                                            "arguments": f'{{"user_utterance": "Design eGFP", "protein_sequence": "{egfp_seq}"}}',
                                        },
                                    }
                                ],
                            },
                        }
                    ]
                }
            )
        if step["n"] == 2:
            raise httpx.ReadTimeout("crashed after mutation before BYO key delivery")
        return _Ok({"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "BYO recovered."}}]})

    monkeypatch.setattr(httpx.Client, "post", _post)

    # First call binds atomic owner_token_sha256.txt + inline claim proof, keeps claim on crash
    owner_file = projects_root / "byo_first_crash" / "owner_token_sha256.txt"
    r_fail = client.post(
        "/v1/projects/byo_first_crash/agent-chat",
        headers={"X-MRNAX-LLM-Key": "sk-ByoCrashUser1234567890", "X-Forwarded-Proto": "https"},
        json={"message": "Design eGFP", "client_request_id": "req_byo_first_1"},
    )
    assert r_fail.status_code in (502, 504)
    assert owner_file.is_file()
    assert "claim:" in owner_file.read_text(encoding="utf-8")

    # Different client_request_id without delivered project key must be rejected
    r_other = client.post(
        "/v1/projects/byo_first_crash/agent-chat",
        headers={"X-MRNAX-LLM-Key": "sk-ByoCrashUser1234567890"},
        json={"message": "hijack attempt", "client_request_id": "req_other_hijack"},
    )
    assert r_other.status_code in (401, 403)

    # Retrying exact same first call recovers delivery + sets Secure cookie AND retains claim until receipt proof
    r_ok = client.post(
        "/v1/projects/byo_first_crash/agent-chat",
        headers={"X-MRNAX-LLM-Key": "sk-ByoCrashUser1234567890", "X-Forwarded-Proto": "https"},
        json={"message": "Design eGFP", "client_request_id": "req_byo_first_1"},
    )
    assert r_ok.status_code == 200
    assert r_ok.json()["assistant_reply"] == "BYO recovered."
    cookie_hdr = r_ok.headers.get("set-cookie", "").lower()
    assert "secure" in cookie_hdr
    delivered_pk = r_ok.json()["project_key"]
    assert "claim:" in owner_file.read_text(encoding="utf-8")

    # Follow-up using delivered project key proves receipt -> clears claim -> unkeyed replay now rejected 403
    r_followup = client.post(
        "/v1/projects/byo_first_crash/agent-chat",
        headers={"X-MRNAX-LLM-Key": "sk-ByoCrashUser1234567890", "X-MRNAX-Project-Key": delivered_pk},
        json={"message": "Followup", "client_request_id": "req_byo_follow_2"},
    )
    assert r_followup.status_code == 200
    assert "claim:" not in owner_file.read_text(encoding="utf-8")

    r_stale_unkeyed = client.post(
        "/v1/projects/byo_first_crash/agent-chat",
        headers={"X-MRNAX-LLM-Key": "sk-ByoCrashUser1234567890"},
        json={"message": "Design eGFP", "client_request_id": "req_byo_first_1"},
    )
    assert r_stale_unkeyed.status_code == 403


def test_r3_turn_idempotency_key_and_whitespace_provider_header(
    tmp_path: Path,
) -> None:
    snap = json.loads((SEED_DIR / "egfp_reporter_snapshot.json").read_text(encoding="utf-8"))
    egfp_seq = snap["mammalian_reporter_egfp"]["protein_sequence"]
    projects_root = tmp_path / "proj_turn_r3"
    app = create_control_plane_app(
        ExecutionCASStore(tmp_path / "cas_turn_r3.sqlite"),
        projects_root=projects_root,
        seed_dir=SEED_DIR,
        access_token="tok_turn_r3",
    )
    client = TestClient(app, headers={"X-MRNAX-Access-Token": "tok_turn_r3"})

    payload = {"mode": "initial", "user_utterance": "Design eGFP", "protein_sequence": egfp_seq}
    r1 = client.post("/v1/projects/proj_tidem/turn", headers={"X-MRNAX-Idempotency-Key": "idem_turn_1"}, json=payload)
    assert r1.status_code == 200
    assert r1.json()["intent"]["intent_version"] == 1

    # Replay with exact same payload & idempotency key does NOT create v2
    r2 = client.post("/v1/projects/proj_tidem/turn", headers={"X-MRNAX-Idempotency-Key": "idem_turn_1"}, json=payload)
    assert r2.status_code == 200
    assert r2.json()["intent"]["intent_version"] == 1
    turn_req_txt = (projects_root / "proj_tidem" / "turn_requests.json").read_text(encoding="utf-8")
    assert "idem_turn_1" not in turn_req_txt
    assert hashlib.sha256(b"idem_turn_1").hexdigest() in turn_req_txt

    # Conflict with modified payload & same idempotency key -> 409
    r_conflict = client.post(
        "/v1/projects/proj_tidem/turn",
        headers={"X-MRNAX-Idempotency-Key": "idem_turn_1"},
        json={"mode": "initial", "user_utterance": "Changed utterance", "protein_sequence": egfp_seq},
    )
    assert r_conflict.status_code == 409

    # Whitespace-only X-MRNAX-LLM-Provider header fails closed with 422
    r_ws = client.post(
        "/v1/projects/proj_ws_1/agent-chat",
        headers={"X-MRNAX-LLM-Provider": "   "},
        json={"message": "hi"},
    )
    assert r_ws.status_code == 422


def test_r4_concurrency_revision_binding_experiments_and_batch_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MRNAX_DEV_ALLOW_SERVER_LLM", "1")
    snap = json.loads((SEED_DIR / "egfp_reporter_snapshot.json").read_text(encoding="utf-8"))
    egfp_seq = snap["mammalian_reporter_egfp"]["protein_sequence"]
    projects_root = tmp_path / "proj_r4_c1"
    app = create_control_plane_app(
        ExecutionCASStore(tmp_path / "cas_r4_c1.sqlite"),
        projects_root=projects_root,
        seed_dir=SEED_DIR,
        access_token="tok_r4_c1",
    )
    client = TestClient(app, headers={"X-MRNAX-Access-Token": "tok_r4_c1"})

    r_init = client.post(
        "/v1/projects/proj_r4_conc/turn",
        json={"mode": "initial", "user_utterance": "seed v1", "protein_sequence": egfp_seq},
    )
    assert r_init.status_code == 200

    class _OkResp:
        def __init__(self, payload: dict[str, Any]) -> None:
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return self._payload

    captured: list[dict[str, Any]] = []
    mode = {"v": "a_step1"}

    def _conc_post(self: httpx.Client, url: str, headers: dict[str, str], json: dict[str, Any]) -> _OkResp:
        captured.append(copy.deepcopy(json))
        if mode["v"] == "a_step1":
            mode["v"] = "a_step2_timeout"
            return _OkResp(
                {
                    "choices": [
                        {
                            "finish_reason": "tool_calls",
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "tc_a_lu",
                                        "type": "function",
                                        "function": {"name": "get_job_status", "arguments": "{}"},
                                    }
                                ],
                            },
                        }
                    ]
                }
            )
        if mode["v"] == "a_step2_timeout":
            raise httpx.ReadTimeout("A timed out after readonly lookup on v1")
        if mode["v"] == "b_mut":
            mode["v"] = "b_done"
            return _OkResp(
                {
                    "choices": [
                        {
                            "finish_reason": "tool_calls",
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "tc_b_ref",
                                        "type": "function",
                                        "function": {
                                            "name": "refine_construct",
                                            "arguments": '{"user_utterance": "B switches to BspQI", "restriction_enzymes": ["BspQI"]}',
                                        },
                                    }
                                ],
                            },
                        }
                    ]
                }
            )
        return _OkResp({"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "Done."}}]})

    monkeypatch.setattr(httpx.Client, "post", _conc_post)

    # 1. A runs lookup on v1 and times out before final reply
    r_a1 = client.post(
        "/v1/projects/proj_r4_conc/agent-chat",
        json={"message": "Explain v1", "client_request_id": "req_a_lu"},
    )
    assert r_a1.status_code == 502

    # B commits v2 and writes v2 history
    mode["v"] = "b_mut"
    r_b = client.post(
        "/v1/projects/proj_r4_conc/agent-chat",
        json={"message": "B switches to BspQI", "client_request_id": "req_b_v2"},
    )
    assert r_b.status_code == 200 and r_b.json()["intent"]["intent_version"] == 2

    # A retries -> provider must receive v1 context (no B history), and response must return v1 + intent_diff=None
    captured.clear()
    mode["v"] = "a_retry"
    r_a2 = client.post(
        "/v1/projects/proj_r4_conc/agent-chat",
        json={"message": "Explain v1", "client_request_id": "req_a_lu"},
    )
    assert r_a2.status_code == 200
    assert r_a2.json()["intent"]["intent_version"] == 1
    assert r_a2.json()["intent_diff"] is None
    msg_blob_a = "\n".join(str(m.get("content") or "") for m in captured[0]["messages"])
    assert '"active_intent_version": 1' in msg_blob_a
    assert "B switches to BspQI" not in msg_blob_a

    # 2. Stale resumed refinement against concurrently moved head must return 409 Conflict
    class _SimCrash(BaseException):
        pass

    orig_status = AtomicToolbox.get_job_status

    def _c_lu_crash_ref(self: httpx.Client, url: str, headers: dict[str, str], json: dict[str, Any]) -> _OkResp:
        return _OkResp(
            {
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {"id": "tc_c_1", "type": "function", "function": {"name": "get_job_status", "arguments": "{}"}},
                                {
                                    "id": "tc_c_2",
                                    "type": "function",
                                    "function": {
                                        "name": "refine_construct",
                                        "arguments": '{"user_utterance": "C wants EcoRI", "restriction_enzymes": ["EcoRI"]}',
                                    },
                                },
                            ],
                        },
                    }
                ]
            }
        )

    orig_refine = AtomicToolbox.refine_construct
    crash_ref = {"on": True}

    def _crash_on_ref(self: AtomicToolbox, *a: Any, **kw: Any) -> dict[str, Any]:
        if crash_ref["on"]:
            crash_ref["on"] = False
            raise _SimCrash("crash before C refine commits")
        return orig_refine(self, *a, **kw)

    monkeypatch.setattr(httpx.Client, "post", _c_lu_crash_ref)
    monkeypatch.setattr(AtomicToolbox, "refine_construct", _crash_on_ref)
    with pytest.raises(_SimCrash):
        client.post(
            "/v1/projects/proj_r4_conc/agent-chat",
            json={"message": "C lookup then refine", "client_request_id": "req_c_stale"},
        )
    # Concurrent deterministic turn moves head from v2 -> v3
    r_d = client.post(
        "/v1/projects/proj_r4_conc/turn",
        json={"mode": "refine", "user_utterance": "D moves to v3", "restriction_enzymes": ["SapI"]},
    )
    assert r_d.status_code == 200 and r_d.json()["intent"]["intent_version"] == 3

    monkeypatch.setattr(httpx.Client, "post", lambda *a, **kw: _OkResp({"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "unreachable"}}]}))
    r_c_conflict = client.post(
        "/v1/projects/proj_r4_conc/agent-chat",
        json={"message": "C lookup then refine", "client_request_id": "req_c_stale"},
    )
    assert r_c_conflict.status_code == 409

    # 3. Lookup-first batch [get_job_status, refine_construct] preserves tail refinement cleanly on retry
    # And concurrent B ingest_experiment does NOT make E's unexecuted ingest look committed
    batch_seq = {"step": "lu_first"}
    orig_ingest = AtomicToolbox.ingest_experiment

    def _batch_post(self: httpx.Client, url: str, headers: dict[str, str], json: dict[str, Any]) -> _OkResp:
        captured.append(copy.deepcopy(json))
        if batch_seq["step"] == "lu_first":
            batch_seq["step"] = "fin"
            return _OkResp(
                {
                    "choices": [
                        {
                            "finish_reason": "tool_calls",
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {"id": "tc_lu_1", "type": "function", "function": {"name": "get_job_status", "arguments": "{}"}},
                                    {
                                        "id": "tc_ref_2",
                                        "type": "function",
                                        "function": {
                                            "name": "refine_construct",
                                            "arguments": '{"user_utterance": "Add XbaI", "restriction_enzymes": ["SapI", "XbaI"]}',
                                        },
                                    },
                                ],
                            },
                        }
                    ]
                }
            )
        if batch_seq["step"] == "mut_first":
            batch_seq["step"] = "fin"
            return _OkResp(
                {
                    "choices": [
                        {
                            "finish_reason": "tool_calls",
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "tc_m_1",
                                        "type": "function",
                                        "function": {
                                            "name": "refine_construct",
                                            "arguments": '{"user_utterance": "Clear EcoRI", "restriction_enzymes": ["EcoRI"]}',
                                        },
                                    },
                                    {"id": "tc_m_2", "type": "function", "function": {"name": "get_job_status", "arguments": "{}"}},
                                ],
                            },
                        }
                    ]
                }
            )
        return _OkResp({"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "Batch recovered."}}]})

    crash_ref = {"on": True}
    monkeypatch.setattr(httpx.Client, "post", _batch_post)
    with pytest.raises(_SimCrash):
        client.post(
            "/v1/projects/proj_r4_conc/agent-chat",
            json={"message": "Lookup then refine", "client_request_id": "req_lu_then_ref"},
        )
    r_lu_ref_ok = client.post(
        "/v1/projects/proj_r4_conc/agent-chat",
        json={"message": "Lookup then refine", "client_request_id": "req_lu_then_ref"},
    )
    assert r_lu_ref_ok.status_code == 200
    assert r_lu_ref_ok.json()["intent"]["intent_version"] == 4
    assert [tc["name"] for tc in r_lu_ref_ok.json()["tool_calls"]] == ["get_job_status", "refine_construct"]

    # Mutation-first batch: check get_job_status result retains intent_versions in synthesis prompt
    crash_stat = {"on": True}

    def _crash_stat(self: AtomicToolbox, *a: Any, **kw: Any) -> dict[str, Any]:
        if crash_stat["on"]:
            crash_stat["on"] = False
            raise _SimCrash("crash on get_job_status")
        return orig_status(self, *a, **kw)

    monkeypatch.setattr(AtomicToolbox, "get_job_status", _crash_stat)
    batch_seq["step"] = "mut_first"
    with pytest.raises(_SimCrash):
        client.post(
            "/v1/projects/proj_r4_conc/agent-chat",
            json={"message": "Refine then status", "client_request_id": "req_mut_then_lu"},
        )
    captured.clear()
    r_mut_lu_ok = client.post(
        "/v1/projects/proj_r4_conc/agent-chat",
        json={"message": "Refine then status", "client_request_id": "req_mut_then_lu"},
    )
    assert r_mut_lu_ok.status_code == 200
    assert r_mut_lu_ok.json()["intent"]["intent_version"] == 5
    synth_user_msg = captured[-1]["messages"][-1]["content"]
    assert "intent_versions" in synth_user_msg

    # Concurrent B ingest_experiment does NOT mark pending E ingest committed or disable read-only G tools
    monkeypatch.setattr(AtomicToolbox, "get_job_status", orig_status)
    base_exp_fields = {
        "project_id": "proj_r4_conc",
        "batch_id": "batch_r4_1",
        "extraction_state": "draft_extraction",
        "assay_type": "luciferase",
        "readout_name_exact": "RLU",
        "unit": "RLU",
        "cell_or_model_context": "HEK293T",
        "chemistry_track": "m1psi",
        "delivery_vehicle": "LNP",
        "shared_controls": [],
        "observations": [
            {
                "bundle_id": r_init.json()["intent"]["intent_id"],
                "version_digest": "a" * 64,
                "source_row_or_figure": "Fig1",
                "biological_replicates": [{"bio_rep_id": "r1", "technical_replicates": [1.0]}],
            }
        ],
    }
    exp_e_rec = {"experiment_id": "exp_e_pending", **base_exp_fields}
    exp_f_rec = {"experiment_id": "exp_f_conc", **base_exp_fields}
    ingest_calls = {"e": 0}

    def _track_ingest(self: AtomicToolbox, record_json: dict[str, Any], **kw: Any) -> dict[str, Any]:
        if record_json.get("experiment_id") == "exp_e_pending":
            ingest_calls["e"] += 1
            if ingest_calls["e"] == 1:
                raise _SimCrash("crash before E ingest runs")
        return orig_ingest(self, record_json, **kw)

    monkeypatch.setattr(AtomicToolbox, "ingest_experiment", _track_ingest)
    monkeypatch.setattr(
        httpx.Client,
        "post",
        lambda *a, **kw: _OkResp(
            {
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "tc_ing_e",
                                    "type": "function",
                                    "function": {"name": "ingest_experiment", "arguments": json.dumps({"record_json": exp_e_rec})},
                                }
                            ],
                        },
                    }
                ]
            }
        ),
    )
    with pytest.raises(_SimCrash):
        client.post(
            "/v1/projects/proj_r4_conc/agent-chat",
            json={"message": "Ingest E", "client_request_id": "req_e_ing"},
        )
    # Concurrent B ingests exp_f_conc while E is pending
    sess_conc = ProjectHarnessSession(project_id="proj_r4_conc", store_dir=projects_root / "proj_r4_conc", seed_dir=SEED_DIR)
    AtomicToolbox(sess_conc, turn_id_prefix="req_other_b").ingest_experiment(exp_f_rec)

    captured.clear()
    monkeypatch.setattr(httpx.Client, "post", lambda *a, **kw: _OkResp({"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "E ingested."}}]}))
    r_e_ok = client.post(
        "/v1/projects/proj_r4_conc/agent-chat",
        json={"message": "Ingest E", "client_request_id": "req_e_ing"},
    )
    assert r_e_ok.status_code == 200
    assert ingest_calls["e"] == 2
    assert (projects_root / "proj_r4_conc" / "experiments" / "exp_e_pending.json").exists()


def test_r4_ambiguity_reference_order_turn_claim_and_scrubbing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MRNAX_DEV_ALLOW_SERVER_LLM", "1")
    snap = json.loads((SEED_DIR / "egfp_reporter_snapshot.json").read_text(encoding="utf-8"))
    egfp_seq = snap["mammalian_reporter_egfp"]["protein_sequence"]
    projects_root = tmp_path / "proj_r4_2"
    secret_turn = "tok_turn_scrub_99"
    app = create_control_plane_app(
        ExecutionCASStore(tmp_path / "cas_r4_2.sqlite"),
        projects_root=projects_root,
        seed_dir=SEED_DIR,
        access_token=secret_turn,
    )
    client = TestClient(app)

    # 1. Ambiguous legacy report must fail read-only /agent-chat with 409 and ZERO execute_design calls (Imp 5)
    r_t1 = client.post(
        "/v1/projects/proj_ambig/turn",
        headers={"X-MRNAX-Idempotency-Key": "idem_amb_1"},
        json={"mode": "initial", "user_utterance": f"Design eGFP {secret_turn}", "protein_sequence": egfp_seq},
    )
    assert r_t1.status_code == 200
    amb_pk = r_t1.json()["project_key"]
    assert secret_turn not in r_t1.text
    # Second turn refining must ALSO scrub inherited objectives/locks + secret_turn
    r_t2 = client.post(
        "/v1/projects/proj_ambig/turn",
        headers={"X-MRNAX-Project-Key": amb_pk, "X-MRNAX-Idempotency-Key": "idem_amb_2"},
        json={"mode": "refine", "user_utterance": f"Switch to BspQI {secret_turn}", "restriction_enzymes": ["BspQI"]},
    )
    assert r_t2.status_code == 200
    assert secret_turn not in r_t2.text
    for ipath in (projects_root / "proj_ambig" / "intents").glob("intent_v*.json"):
        assert secret_turn not in ipath.read_text(encoding="utf-8")

    # Also verify claim cleared from owner_token_sha256.txt after keyed refine (Imp 7)
    owner_amb = (projects_root / "proj_ambig" / "owner_token_sha256.txt").read_text(encoding="utf-8")
    assert "claim:" not in owner_amb

    # Simulate ambiguous legacy reports on proj_ambig intent v2
    import os
    rep_dir = projects_root / "proj_ambig" / "reports"
    rep_files = list(rep_dir.glob("rep_*.json"))
    v2_rep_path = next(p for p in rep_files if json.loads(p.read_text())["intent_id"] == r_t2.json()["intent"]["intent_id"])
    v2_rep_data = json.loads(v2_rep_path.read_text(encoding="utf-8"))
    v2_rep_data["created_at_ns"] = None
    v2_rep_path.write_text(json.dumps(v2_rep_data), encoding="utf-8")
    dup_rep = dict(v2_rep_data, report_id="rep_dup_ambig")
    dup_path = rep_dir / "rep_dup_ambig.json"
    dup_path.write_text(json.dumps(dup_rep), encoding="utf-8")
    os.utime(v2_rep_path, ns=(1_700_000_000_000_000_000, 1_700_000_000_000_000_000))
    os.utime(dup_path, ns=(1_700_000_000_000_000_000, 1_700_000_000_000_000_000))
    pstate_path = projects_root / "proj_ambig" / "project_state.json"
    pstate = json.loads(pstate_path.read_text(encoding="utf-8"))
    pstate.pop("report_order", None)
    pstate_path.write_text(json.dumps(pstate), encoding="utf-8")

    exec_cnt = {"n": 0}
    orig_exec = ProjectHarnessSession.execute_design

    def _count_exec(self: ProjectHarnessSession, *a: Any, **kw: Any) -> Any:
        exec_cnt["n"] += 1
        return orig_exec(self, *a, **kw)

    monkeypatch.setattr(ProjectHarnessSession, "execute_design", _count_exec)
    monkeypatch.setattr(
        httpx.Client,
        "post",
        lambda *a, **kw: type(
            "_R",
            (),
            {
                "raise_for_status": lambda s: None,
                "json": lambda s: {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "Read-only reply"}}]},
            },
        )(),
    )
    r_amb_chat = client.post(
        "/v1/projects/proj_ambig/agent-chat",
        headers={"X-MRNAX-LLM-Key": "sk-AmbTestUser1234567890", "X-MRNAX-Project-Key": amb_pk},
        json={"message": "Summarize", "client_request_id": "req_amb_ro"},
    )
    assert r_amb_chat.status_code == 409
    assert exec_cnt["n"] == 0

    # 2. Reference bundle registered BEFORE intent persistence + missing reference fails closed 409 (Imp 6)
    monkeypatch.setattr(ProjectHarnessSession, "execute_design", orig_exec)
    sess_ref = ProjectHarnessSession(project_id="proj_ref_ord", store_dir=tmp_path / "proj_ref_ord", seed_dir=SEED_DIR)
    ref_intent = sess_ref.create_turn_intent_canonical(
        user_utterance="With R", protein_sequence=egfp_seq, use_preset_reference_r=True
    )
    assert ref_intent.reference_bundle_id in sess_ref._bundles
    (tmp_path / "proj_ref_ord" / "bundles" / f"{ref_intent.reference_bundle_id}.json").unlink()
    sess_reloaded = ProjectHarnessSession(project_id="proj_ref_ord", store_dir=tmp_path / "proj_ref_ord", seed_dir=SEED_DIR)
    with pytest.raises(ValueError, match="missing from project store"):
        sess_reloaded.execute_design(ref_intent)

    # 3. CLI / LiveDialogueAgent checks model against vault secret (Imp 8)
    SecretVault().set_key("deepseek", "my_secret_model_tok")
    with pytest.raises(ValueError, match="Invalid model identifier"):
        LiveDialogueAgent(AtomicToolbox(sess_reloaded), provider="deepseek", model="my_secret_model_tok")


def test_r4_journal_v2_hex_migration_missing_protein_and_finish_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 1. Secret matching protocol word "completed" must NOT corrupt journal status markers;
    #    64-char hex plaintext ID in legacy flat file must hit cache and persist v2 immediately (Imp 9)
    projects_root = tmp_path / "proj_jrn_v2"
    app = create_control_plane_app(
        ExecutionCASStore(tmp_path / "cas_jrn_v2.sqlite"),
        projects_root=projects_root,
        seed_dir=SEED_DIR,
        access_token="completed",
    )
    client = TestClient(app, headers={"X-MRNAX-Access-Token": "completed"})

    class _OkResp:
        def __init__(self, payload: dict[str, Any]) -> None:
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return self._payload

    llm_hits = {"n": 0}

    def _post_jrn(self: httpx.Client, url: str, headers: dict[str, str], json: dict[str, Any]) -> _OkResp:
        llm_hits["n"] += 1
        return _OkResp({"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "Journal OK"}}]})

    monkeypatch.setattr(httpx.Client, "post", _post_jrn)
    r_word = client.post(
        "/v1/projects/proj_word/agent-chat",
        json={"message": "hi completed", "client_request_id": "req_word_1"},
    )
    assert r_word.status_code == 200
    assert r_word.json()["status"] == "COMPLETED"
    jfile = projects_root / "proj_word" / "agent_chat_requests.json"
    jdoc = json.loads(jfile.read_text(encoding="utf-8"))
    h_word = hashlib.sha256(b"req_word_1").hexdigest()
    assert jdoc["schema_version"] == 2
    assert jdoc["entries"][h_word]["status"] == "completed"
    llm_hits["n"] = 0
    r_word_replay = client.post(
        "/v1/projects/proj_word/agent-chat",
        json={"message": "hi completed", "client_request_id": "req_word_1"},
    )
    assert r_word_replay.status_code == 200 and llm_hits["n"] == 0

    # 64-char hex plaintext ID in legacy flat v1 journal
    hex_id = "b" * 64
    jfile.write_text(json.dumps({hex_id: jdoc["entries"][h_word]}), encoding="utf-8")
    llm_hits["n"] = 0
    r_hex_hit = client.post(
        "/v1/projects/proj_word/agent-chat",
        json={"message": "hi completed", "client_request_id": hex_id},
    )
    assert r_hex_hit.status_code == 200 and llm_hits["n"] == 0
    upgraded_doc = json.loads(jfile.read_text(encoding="utf-8"))
    assert upgraded_doc["schema_version"] == 2
    assert hashlib.sha256(hex_id.encode("utf-8")).hexdigest() in upgraded_doc["entries"]
    assert hex_id not in upgraded_doc["entries"]

    # 2. Missing protein_sequence for non-eGFP target must raise ValueError instead of silent eGFP (Imp 11)
    sess_no_prot = ProjectHarnessSession(project_id="proj_noprot", store_dir=tmp_path / "proj_noprot", seed_dir=SEED_DIR)
    with pytest.raises(ValueError, match="Missing protein_sequence"):
        AtomicToolbox(sess_no_prot).design_constructs(
            user_utterance="Design mCherry mRNA",
            accession_or_source="UniProt:Q9U6Y8",
        )

    # 3. Contradictory finish_reason="stop" with tool_calls -> 502; unknown finish_reason -> 502;
    #    whitespace-padded tool_call.id normalized identically in assistant + tool messages (Imp 13, Minor 2)
    monkeypatch.setattr(
        httpx.Client,
        "post",
        lambda *a, **kw: _OkResp(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {"id": "tc_stop_1", "type": "function", "function": {"name": "get_job_status", "arguments": "{}"}}
                            ],
                        },
                    }
                ]
            }
        ),
    )
    assert client.post("/v1/projects/proj_fr/agent-chat", json={"message": "x", "client_request_id": "fr1"}).status_code == 502

    monkeypatch.setattr(
        httpx.Client,
        "post",
        lambda *a, **kw: _OkResp({"choices": [{"finish_reason": "weird_unknown", "message": {"role": "assistant", "content": "hi"}}]}),
    )
    assert client.post("/v1/projects/proj_fr/agent-chat", json={"message": "x", "client_request_id": "fr2"}).status_code == 502

    pad_payloads: list[dict[str, Any]] = []
    pad_n = {"i": 0}

    def _pad_post(self: httpx.Client, url: str, headers: dict[str, str], json: dict[str, Any]) -> _OkResp:
        pad_payloads.append(copy.deepcopy(json))
        pad_n["i"] += 1
        if pad_n["i"] == 1:
            return _OkResp(
                {
                    "choices": [
                        {
                            "finish_reason": "tool_calls",
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {"id": "  tc_pad_1  ", "type": "function", "function": {"name": "get_job_status", "arguments": "{}"}}
                                ],
                            },
                        }
                    ]
                }
            )
        return _OkResp({"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "Trimmed OK"}}]})

    monkeypatch.setattr(httpx.Client, "post", _pad_post)
    r_pad = client.post("/v1/projects/proj_fr/agent-chat", json={"message": "Check status", "client_request_id": "fr3"})
    assert r_pad.status_code == 200
    second_msgs = pad_payloads[1]["messages"]
    asst_msg = next(m for m in second_msgs if m.get("role") == "assistant" and m.get("tool_calls"))
    tool_msg = next(m for m in second_msgs if m.get("role") == "tool")
    assert asst_msg["tool_calls"][0]["id"] == "tc_pad_1"
    assert tool_msg["tool_call_id"] == "tc_pad_1"


def test_r5_astra_closure_suites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MRNAX_DEV_ALLOW_SERVER_LLM", "1")
    snap = json.loads((SEED_DIR / "egfp_reporter_snapshot.json").read_text(encoding="utf-8"))
    egfp_seq = snap["mammalian_reporter_egfp"]["protein_sequence"]
    projects_root = tmp_path / "proj_r5"
    secret_r5 = "tok_r5_admin_secret_99"
    app = create_control_plane_app(
        ExecutionCASStore(tmp_path / "cas_r5.sqlite"),
        projects_root=projects_root,
        seed_dir=SEED_DIR,
        access_token=secret_r5,
    )
    client = TestClient(app, headers={"X-MRNAX-Access-Token": secret_r5})

    class _OkResp:
        def __init__(self, payload: dict[str, Any]) -> None:
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return self._payload

    # 1. Imp 2: Pre-tool timeout on EMPTY v0 project + concurrent v1 keeps v0 baseline (intent=None, report=None)
    step = {"v": "timeout"}

    def _empty_timeout_post(self: httpx.Client, url: str, headers: dict[str, str], json: dict[str, Any]) -> _OkResp:
        if step["v"] == "timeout":
            raise httpx.ReadTimeout("timeout before first tool on empty project")
        return _OkResp({"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "Empty baseline reply."}}]})

    monkeypatch.setattr(httpx.Client, "post", _empty_timeout_post)
    assert (
        client.post(
            "/v1/projects/proj_r5_empty/agent-chat",
            json={"message": "Explain empty", "client_request_id": "req_r5_empty_a"},
        ).status_code
        == 502
    )
    r_b_empty = client.post(
        "/v1/projects/proj_r5_empty/turn",
        json={"mode": "initial", "user_utterance": "Concurrent seed v1", "protein_sequence": egfp_seq},
    )
    assert r_b_empty.status_code == 200
    step["v"] = "ok"
    r_a_empty_ok = client.post(
        "/v1/projects/proj_r5_empty/agent-chat",
        json={"message": "Explain empty", "client_request_id": "req_r5_empty_a"},
    )
    assert r_a_empty_ok.status_code == 200
    assert r_a_empty_ok.json()["intent"] is None
    assert r_a_empty_ok.json()["report"] is None

    # 2. Imp 2: Interrupted /turn refinement fails 409 when head advances concurrently
    class _SimCrash(BaseException):
        pass

    orig_refine_fn = ProjectHarnessSession.refine_turn_intent_canonical
    crash_flag = {"on": True}

    def _crash_once_refine(self: ProjectHarnessSession, *a: Any, **kw: Any) -> Any:
        if crash_flag["on"]:
            crash_flag["on"] = False
            raise _SimCrash("crash inside /turn refine")
        return orig_refine_fn(self, *a, **kw)

    monkeypatch.setattr(ProjectHarnessSession, "refine_turn_intent_canonical", _crash_once_refine)
    with pytest.raises(_SimCrash):
        client.post(
            "/v1/projects/proj_r5_empty/turn",
            headers={"X-MRNAX-Idempotency-Key": "idem_r5_ref"},
            json={"mode": "refine", "user_utterance": "Ref A", "restriction_enzymes": ["SapI"]},
        )
    assert (
        client.post(
            "/v1/projects/proj_r5_empty/turn",
            headers={"X-MRNAX-Idempotency-Key": "idem_r5_b"},
            json={"mode": "refine", "user_utterance": "Ref B", "restriction_enzymes": ["EcoRI"]},
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/v1/projects/proj_r5_empty/turn",
            headers={"X-MRNAX-Idempotency-Key": "idem_r5_ref"},
            json={"mode": "refine", "user_utterance": "Ref A", "restriction_enzymes": ["SapI"]},
        ).status_code
        == 409
    )

    # 3. Imp 3: Multi-writer _experiment_origins prevents concurrent draft B on same experiment_id from blocking A retry
    orig_ingest = AtomicToolbox.ingest_experiment
    exp_same_id = "exp_shared_draft_1"
    base_exp = {
        "experiment_id": exp_same_id,
        "project_id": "proj_r5_empty",
        "batch_id": "b1",
        "extraction_state": "draft_extraction",
        "assay_type": "luciferase",
        "readout_name_exact": "RLU",
        "unit": "RLU",
        "cell_or_model_context": "HEK293T",
        "chemistry_track": "m1psi",
        "delivery_vehicle": "LNP",
        "shared_controls": [],
        "observations": [
            {
                "bundle_id": r_b_empty.json()["intent"]["intent_id"],
                "version_digest": "c" * 64,
                "source_row_or_figure": "Fig1",
                "biological_replicates": [{"bio_rep_id": "r1", "technical_replicates": [1.0]}],
            }
        ],
    }

    def _crash_after_ingest_a(self: AtomicToolbox, record_json: dict[str, Any], **kw: Any) -> dict[str, Any]:
        out = orig_ingest(self, record_json, **kw)
        if self.turn_id_prefix and "req_exp_a" in str(kw) or True:
            pass
        return out

    # Request A commits exp_same_id, then B overwrites same draft exp_same_id before A finishes tool checkpoint
    orig_sess_ingest = ProjectHarnessSession.ingest_experiment_feedback
    ingest_state = {"step": 0}

    def _hook_ingest(self: ProjectHarnessSession, rec: Any, **kw: Any) -> Any:
        ingest_state["step"] += 1
        res = orig_sess_ingest(self, rec, **kw)
        if ingest_state["step"] == 1:
            # Simulate concurrent writer B updating the same draft experiment_id before A checkpoints tool call
            other = ProjectHarnessSession(project_id="proj_r5_empty", store_dir=projects_root / "proj_r5_empty", seed_dir=SEED_DIR)
            b_rec = rec.model_copy_validated(observations=[rec.observations[0].model_copy_validated(source_row_or_figure="Fig2_B")])
            orig_sess_ingest(other, b_rec, batch_comparable=True, held_out_validation_passed=False, request_turn_id="req_other_writer_1")
            raise _SimCrash("A missed tool checkpoint after B overwrote same experiment")
        return res

    monkeypatch.setattr(ProjectHarnessSession, "ingest_experiment_feedback", _hook_ingest)
    monkeypatch.setattr(
        httpx.Client,
        "post",
        lambda *a, **kw: _OkResp(
            {
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "tc_exp_a",
                                    "type": "function",
                                    "function": {"name": "ingest_experiment", "arguments": json.dumps({"record_json": base_exp})},
                                }
                            ],
                        },
                    }
                ]
            }
        ),
    )
    with pytest.raises(_SimCrash):
        client.post(
            "/v1/projects/proj_r5_empty/agent-chat",
            json={"message": "Save A draft", "client_request_id": "req_exp_a"},
        )
    monkeypatch.setattr(ProjectHarnessSession, "ingest_experiment_feedback", orig_sess_ingest)
    monkeypatch.setattr(
        httpx.Client,
        "post",
        lambda *a, **kw: _OkResp({"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "A recovered."}}]}),
    )
    r_exp_a_ok = client.post(
        "/v1/projects/proj_r5_empty/agent-chat",
        json={"message": "Save A draft", "client_request_id": "req_exp_a"},
    )
    assert r_exp_a_ok.status_code == 200
    assert ingest_state["step"] == 1  # Not re-run!
    disk_exp = json.loads((projects_root / "proj_r5_empty" / "experiments" / f"{exp_same_id}.json").read_text())
    assert disk_exp["observations"][0]["source_row_or_figure"] == "Fig2_B"  # B's edit preserved!

    # 4. Imp 4: Long first batch result (>3500 chars) preserves trailing get_job_status in recovery prompt
    captured_r5: list[dict[str, Any]] = []
    orig_ref_tool = AtomicToolbox.refine_construct
    orig_stat_tool = AtomicToolbox.get_job_status

    def _huge_ref(self: AtomicToolbox, *a: Any, **kw: Any) -> dict[str, Any]:
        return {"huge_blob": "X" * 4500, "status": "ok"}

    stat_once = {"crash": True}

    def _crash_stat_once(self: AtomicToolbox, *a: Any, **kw: Any) -> dict[str, Any]:
        if stat_once["crash"]:
            stat_once["crash"] = False
            raise _SimCrash("crash on status")
        return orig_stat_tool(self, *a, **kw)

    monkeypatch.setattr(AtomicToolbox, "refine_construct", _huge_ref)
    monkeypatch.setattr(AtomicToolbox, "get_job_status", _crash_stat_once)
    monkeypatch.setattr(
        httpx.Client,
        "post",
        lambda *a, **kw: _OkResp(
            {
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {"id": "tc_huge_1", "type": "function", "function": {"name": "refine_construct", "arguments": '{"user_utterance": "x"}'}},
                                {"id": "tc_stat_2", "type": "function", "function": {"name": "get_job_status", "arguments": "{}"}},
                            ],
                        },
                    }
                ]
            }
        ),
    )
    with pytest.raises(_SimCrash):
        client.post(
            "/v1/projects/proj_r5_empty/agent-chat",
            json={"message": "Huge then status", "client_request_id": "req_huge_batch"},
        )
    monkeypatch.setattr(AtomicToolbox, "refine_construct", orig_ref_tool)

    def _capture_final(self: httpx.Client, url: str, headers: dict[str, str], json: dict[str, Any]) -> _OkResp:
        captured_r5.append(copy.deepcopy(json))
        return _OkResp({"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "All kept."}}]})

    monkeypatch.setattr(httpx.Client, "post", _capture_final)
    r_huge_ok = client.post(
        "/v1/projects/proj_r5_empty/agent-chat",
        json={"message": "Huge then status", "client_request_id": "req_huge_batch"},
    )
    assert r_huge_ok.status_code == 200
    assert "intent_versions" in captured_r5[0]["messages"][-1]["content"]

    # 5. Imp 6: Failed _persist_intent leaves active_reference_bundle_id unchanged
    sess_r6 = ProjectHarnessSession(project_id="proj_r6_ref", store_dir=tmp_path / "proj_r6_ref", seed_dir=SEED_DIR)
    orig_persist = ProjectHarnessSession._persist_intent
    monkeypatch.setattr(ProjectHarnessSession, "_persist_intent", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("persist fail")))
    from mrnax.harness.state_machine import build_preset_reference_r
    ref_r = build_preset_reference_r(SEED_DIR)
    with pytest.raises(RuntimeError, match="persist fail"):
        sess_r6.create_initial_intent(
            turn_id="t_fail",
            user_utterance="With R",
            protein_sequence=egfp_seq,
            reference_bundle=ref_r,
        )
    assert sess_r6.active_reference_bundle_id is None
    monkeypatch.setattr(ProjectHarnessSession, "_persist_intent", orig_persist)

    # 6. Imp 7 & 8 & 9 & 11:
    # Admin GET /state preserves claim:; failed 422 /turn rolls back new owner; restriction_enzymes + legacy accession scrubbed;
    # 2-entry 64-hex legacy journal migrates both + malformed journal 409; use_preset_reference_r blocks conflicting accession.
    r_bad_init = TestClient(app).post(
        "/v1/projects/proj_r5_422_owner/turn",
        headers={"X-MRNAX-Idempotency-Key": "idem_422_1"},
        json={"mode": "initial", "user_utterance": "Design mCherry"},
    )
    assert r_bad_init.status_code == 422
    assert not (projects_root / "proj_r5_422_owner" / "owner_token_sha256.txt").exists()

    # restriction_enzymes carrying active secret is scrubbed before persisting intent (test on clarification turn so 200 returns)
    r_enz_scrub = client.post(
        "/v1/projects/proj_r5_enz_scrub/turn",
        headers={"X-MRNAX-Idempotency-Key": "idem_enz_scrub"},
        json={
            "mode": "initial",
            "user_utterance": "Design eGFP conflict",
            "protein_sequence": egfp_seq,
            "conflicting_single_pick": True,
            "restriction_enzymes": [f"BsaI_{secret_r5}"],
        },
    )
    assert r_enz_scrub.status_code == 200
    assert secret_r5 not in r_enz_scrub.text
    enz_ipath = next((projects_root / "proj_r5_enz_scrub" / "intents").glob("intent_v*.json"))
    assert secret_r5 not in enz_ipath.read_text(encoding="utf-8")

    # Corrected /turn binds fresh owner + claim, admin GET /state keeps claim!
    r_good_init = TestClient(app).post(
        "/v1/projects/proj_r5_422_owner/turn",
        headers={"X-MRNAX-Idempotency-Key": "idem_422_1"},
        json={
            "mode": "initial",
            "user_utterance": f"Design eGFP {secret_r5}",
            "protein_sequence": egfp_seq,
            "accession_or_source": f"custom:{secret_r5}",
            "entry_version": f"v_{secret_r5}",
            "restriction_enzymes": ["BsaI"],
        },
    )
    assert r_good_init.status_code == 200
    assert secret_r5 not in r_good_init.text
    owner_txt = (projects_root / "proj_r5_422_owner" / "owner_token_sha256.txt").read_text(encoding="utf-8")
    assert "claim:" in owner_txt
    assert client.get("/v1/projects/proj_r5_422_owner/state").status_code == 200
    assert "claim:" in (projects_root / "proj_r5_422_owner" / "owner_token_sha256.txt").read_text(encoding="utf-8")

    # Refinement also scrubs inherited accession_or_source / entry_version / restriction_enzymes
    pk_422 = r_good_init.json()["project_key"]
    r_ref_scrub = TestClient(app).post(
        "/v1/projects/proj_r5_422_owner/turn",
        headers={"X-MRNAX-Project-Key": pk_422, "X-MRNAX-Idempotency-Key": "idem_422_ref"},
        json={"mode": "refine", "user_utterance": "Switch to BspQI", "restriction_enzymes": ["BspQI"]},
    )
    assert r_ref_scrub.status_code == 200
    assert secret_r5 not in r_ref_scrub.text
    for ipath in (projects_root / "proj_r5_422_owner" / "intents").glob("intent_v*.json"):
        assert secret_r5 not in ipath.read_text(encoding="utf-8")

    # Imp 9: 2-entry 64-hex legacy v1 journal preserves second entry across migration of first, and corrupted journal returns 409
    j2 = projects_root / "proj_r5_j2" / "agent_chat_requests.json"
    j2.parent.mkdir(parents=True, exist_ok=True)
    hex_1 = "1" * 64
    hex_2 = "2" * 64
    sample_resp = {"status": "COMPLETED", "provider": "deepseek", "model": "deepseek-flash", "assistant_reply": "hi", "tool_calls": [], "intent": None, "intent_diff": None, "report": None, "candidates": []}
    j2.write_text(
        json.dumps(
            {
                hex_1: {"status": "completed", "request_digest": hashlib.sha256(json.dumps({"message": "m1", "provider": "deepseek", "model": ""}, sort_keys=True).encode()).hexdigest(), "response": sample_resp},
                hex_2: {"status": "completed", "request_digest": hashlib.sha256(json.dumps({"message": "m2", "provider": "deepseek", "model": ""}, sort_keys=True).encode()).hexdigest(), "response": sample_resp},
            }
        ),
        encoding="utf-8",
    )
    assert client.post("/v1/projects/proj_r5_j2/agent-chat", json={"message": "m1", "client_request_id": hex_1}).status_code == 200
    assert json.loads(j2.read_text())["schema_version"] == 2
    assert client.post("/v1/projects/proj_r5_j2/agent-chat", json={"message": "m2", "client_request_id": hex_2}).status_code == 200
    j2.write_text("{corrupt-json", encoding="utf-8")
    assert client.post("/v1/projects/proj_r5_j2/agent-chat", json={"message": "m3", "client_request_id": "req_corr"}).status_code == 409

    # Imp 11: use_preset_reference_r cannot substitute eGFP when accession conflicts
    sess_conf_r = ProjectHarnessSession(project_id="proj_conf_r", store_dir=tmp_path / "proj_conf_r", seed_dir=SEED_DIR)
    with pytest.raises(ValueError, match="Missing protein_sequence"):
        AtomicToolbox(sess_conf_r).design_constructs(
            user_utterance="With R",
            accession_or_source="UniProt:Q9U6Y8",
            use_preset_reference_r=True,
        )

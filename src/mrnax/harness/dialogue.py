"""Scheme A: Built-in Multi-Provider BYO-LLM Tool-Calling Dialogue Agent (OpenAI / DeepSeek / Kimi / Gemini)."""

from __future__ import annotations

import json
import re
from typing import Any

import httpx

from mrnax.harness.auth_secrets import PROVIDER_ENV_MAP, SecretVault
from mrnax.harness.sandbox import validate_byo_llm_base_url
from mrnax.harness.tools import AtomicToolbox

_REDACT_KEY_RE = re.compile(r"\b(sk-[A-Za-z0-9._-]{16,300}|AIza[0-9A-Za-z_-]{31,45})\b")


def redact_secrets_in_text(text: str) -> str:
    return _REDACT_KEY_RE.sub("[REDACTED_API_KEY]", text)

DEFAULT_PROVIDER_MODELS: dict[str, str] = {
    "deepseek": "deepseek-chat",
    "openai": "gpt-4.1-mini",
    "kimi": "moonshot-v1-32k",
    "qwen": "qwen-plus",
    "gemini": "gemini-2.5-flash",
}

def normalize_dialogue_provider(value: str) -> str:
    provider = value.strip().lower()
    if provider not in DEFAULT_PROVIDER_MODELS:
        raise ValueError(f"Unsupported dialogue provider: {value}")
    return provider


SYSTEM_PROMPT = (
    "You are the mRNAx v2 Design Agent. STRICT SCIENTIFIC & SAFETY RULES:\n"
    "1. NEVER invent, guess, or write nucleotide/protein sequences or numerical scores yourself.\n"
    "2. ALWAYS call `search_protein_assets`, `design_constructs`, `refine_construct`, `explain_tradeoffs`, "
    "`compare_bundles`, `audit_construct`, or `export_bundle` to execute every biological operation.\n"
    "3. When the user modifies a single constraint or lock in a follow-up turn, call `refine_construct` "
    "so all untouched locks and protein targets are deterministically inherited.\n"
    "4. Summarize tool outputs concisely and accurately, distinguishing verified measurements, "
    "thermodynamic approximations (`cross_chemistry_thermodynamic_approximation`), and unverified wet-lab targets."
)


class LiveDialogueAgent:
    """Executes blank-slate multi-turn natural language sessions against real tool-calling LLMs."""

    def __init__(
        self,
        toolbox: AtomicToolbox,
        *,
        provider: str = "deepseek",
        model: str | None = None,
        base_url: str | None = None,
        vault: SecretVault | None = None,
    ) -> None:
        self.toolbox = toolbox
        self.provider = normalize_dialogue_provider(provider)
        self.vault = vault or SecretVault()
        raw_url = base_url or PROVIDER_ENV_MAP[self.provider][1]
        self.base_url = validate_byo_llm_base_url(raw_url, allow_localhost=False)
        self.model = model or DEFAULT_PROVIDER_MODELS[self.provider]
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.tool_call_trace: list[dict[str, Any]] = []

    def send_user_message(
        self,
        user_text: str,
        *,
        max_tool_rounds: int = 5,
        explicit_api_key: str | None = None,
    ) -> str:
        api_key = explicit_api_key or self.vault.get_key(self.provider)
        if not api_key:
            raise RuntimeError(
                f"No API key configured for provider '{self.provider}'; run `uv run mrnax auth set-key --provider {self.provider}`"
            )

        self.messages.append({"role": "user", "content": redact_secrets_in_text(user_text)})
        tools_spec = self.toolbox.openai_tool_schemas()

        with httpx.Client(timeout=90.0) as client:
            for _round in range(max_tool_rounds):
                resp = client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": self.messages,
                        "tools": tools_spec,
                        "tool_choice": "auto",
                        "temperature": 0.0,
                    },
                )
                resp.raise_for_status()
                choice = resp.json()["choices"][0]["message"]
                self.messages.append(choice)

                tool_calls = choice.get("tool_calls") or []
                if not tool_calls:
                    return str(choice.get("content") or "")

                for tc in tool_calls:
                    fn_name = tc["function"]["name"]
                    raw_args = tc["function"].get("arguments") or "{}"
                    parsed_args = json.loads(raw_args)
                    result = self.toolbox.dispatch_tool_call(fn_name, parsed_args)
                    self.tool_call_trace.append(
                        {
                            "tool_call_id": tc["id"],
                            "name": fn_name,
                            "arguments": parsed_args,
                            "result": result,
                        }
                    )
                    self.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": json.dumps(result, ensure_ascii=False),
                        }
                    )

        return str(self.messages[-1].get("content") or "")

    def save_transcript(self, out_path: str) -> None:
        from pathlib import Path

        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        raw_json = json.dumps(
            {
                "provider": self.provider,
                "model": self.model,
                "messages": self.messages,
                "tool_call_trace": self.tool_call_trace,
            },
            indent=2,
            ensure_ascii=False,
        )
        p.write_text(redact_secrets_in_text(raw_json) + "\n", encoding="utf-8")

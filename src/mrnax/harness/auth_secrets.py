"""Zero-leak multi-provider BYO-LLM credential vault (`~/.mrnax/secrets.env`, chmod 0600).

Supports OpenAI, DeepSeek, Kimi (Moonshot), Gemini, and Anthropic keys without ever
exposing raw secrets to stdout, logs, git repositories, or agent transcripts.
"""

from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path
from typing import TypedDict

from dotenv import dotenv_values
from pypdf import PdfReader

PROVIDER_ENV_MAP: dict[str, tuple[str, str]] = {
    "openai": ("OPENAI_API_KEY", "https://api.openai.com/v1"),
    "deepseek": ("DEEPSEEK_API_KEY", "https://api.deepseek.com/v1"),
    "kimi": ("KIMI_API_KEY", "https://api.moonshot.cn/v1"),
    "qwen": ("QWEN_API_KEY", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    "zai": ("ZAI_API_KEY", "https://open.bigmodel.cn/api/paas/v4"),
    "gemini": ("GEMINI_API_KEY", "https://generativelanguage.googleapis.com/v1beta/openai"),
    "anthropic": ("ANTHROPIC_API_KEY", "https://api.anthropic.com/v1"),
}

_GEMINI_KEY_RE = re.compile(r"\bAIza[0-9A-Za-z_-]{31,45}\b")
_SK_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9._-]{16,300}\b")
_ZAI_KEY_RE = re.compile(r"\b[0-9a-f]{32}\.[A-Za-z0-9_-]{12,32}\b")
_CONTINUATION_RE = re.compile(r"^[A-Za-z0-9._-]{4,180}$")
_PROVIDER_TAG_RE = re.compile(
    r"(openai|chatgpt|gpt|deepseek|kimi|moonshot|qwen|zai|zhipu|anthropic|claude|gemini|google)",
    re.IGNORECASE,
)


class ProviderKeyStatus(TypedDict):
    env_var: str
    configured: bool
    masked: str
    length: int
    default_base_url: str


def mask_secret(secret: str) -> str:
    cleaned = secret.strip()
    if len(cleaned) <= 8:
        return "****"
    return f"{cleaned[:4]}****{cleaned[-2:]}"


def _classify_provider_tag(tag: str) -> str | None:
    low = tag.lower()
    if low in {"openai", "chatgpt", "gpt"}:
        return "openai"
    if low == "deepseek":
        return "deepseek"
    if low in {"kimi", "moonshot"}:
        return "kimi"
    if low == "qwen":
        return "qwen"
    if low in {"zai", "zhipu"}:
        return "zai"
    if low in {"anthropic", "claude"}:
        return "anthropic"
    if low in {"gemini", "google"}:
        return "gemini"
    return None


def _normalize_pdf_glyphs(text: str) -> str:
    """Repair PDF font glyph substitutions such as Latin small alpha U+0251 -> ASCII 'a'."""
    repaired = text.replace("\u0251", "a").replace("\u0261", "g")
    return unicodedata.normalize("NFKC", repaired)


def parse_provider_keys_from_text(full_text: str) -> dict[str, str]:
    """Extract provider keys from potentially line-wrapped and glyph-substituted PDF text."""
    normalized = _normalize_pdf_glyphs(full_text)
    discovered: dict[str, str] = {}

    gemini_match = _GEMINI_KEY_RE.search(normalized)
    if gemini_match:
        discovered["gemini"] = gemini_match.group(0)

    zai_match = _ZAI_KEY_RE.search(normalized)
    if zai_match:
        discovered["zai"] = zai_match.group(0)

    raw_lines = [ln.strip() for ln in normalized.splitlines() if ln.strip()]
    stitched_lines: list[str] = []
    idx = 0
    while idx < len(raw_lines):
        line = raw_lines[idx]
        if "sk-" in line:
            while idx + 1 < len(raw_lines):
                nxt = raw_lines[idx + 1]
                if _PROVIDER_TAG_RE.search(nxt) or nxt.startswith("sk-") or ":" in nxt or " " in nxt:
                    break
                if _CONTINUATION_RE.match(nxt):
                    line = line + nxt
                    idx += 1
                else:
                    break
        stitched_lines.append(line)
        idx += 1

    active_provider: str | None = None
    for line in stitched_lines:
        tags = _PROVIDER_TAG_RE.findall(line)
        if tags:
            mapped = _classify_provider_tag(tags[-1])
            if mapped:
                active_provider = mapped

        for match in _SK_KEY_RE.finditer(line):
            token = match.group(0)
            if token.startswith("sk-ant-"):
                discovered.setdefault("anthropic", token)
            elif token.startswith(("sk-proj-", "sk-svcacct-")):
                discovered["openai"] = token
            elif active_provider in {"deepseek", "kimi", "qwen", "openai", "anthropic"}:
                discovered.setdefault(active_provider, token)

    return discovered


class SecretVault:
    """File-backed credential store isolated outside the repository with 0600 permissions."""

    def __init__(self, vault_path: Path | None = None) -> None:
        self.vault_path = vault_path or (Path.home() / ".mrnax" / "secrets.env")

    def _read_all(self) -> dict[str, str]:
        data: dict[str, str] = {}
        if self.vault_path.is_file():
            for k, v in dotenv_values(self.vault_path, interpolate=False).items():
                if v:
                    data[k] = v.strip()
        return data

    def _write_vault(self, entries: dict[str, str]) -> None:
        self.vault_path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(self.vault_path.parent, 0o700)
        lines = [f"{k}={v}" for k, v in sorted(entries.items()) if v]
        tmp_path = self.vault_path.with_suffix(".tmp")
        fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + ("\n" if lines else ""))
        os.replace(tmp_path, self.vault_path)
        os.chmod(self.vault_path, 0o600)

    def set_key(self, provider: str, raw_key: str) -> ProviderKeyStatus:
        norm = "kimi" if provider.strip().lower() == "moonshot" else provider.strip().lower()
        if norm not in PROVIDER_ENV_MAP:
            raise ValueError(
                f"Unsupported provider '{provider}'; expected one of {list(PROVIDER_ENV_MAP)}"
            )
        env_var, base_url = PROVIDER_ENV_MAP[norm]
        cleaned = raw_key.strip()
        if not cleaned:
            raise ValueError("API key must not be empty")

        existing = self._read_all()
        existing[env_var] = cleaned
        self._write_vault(existing)
        return {
            "env_var": env_var,
            "configured": True,
            "masked": mask_secret(cleaned),
            "length": len(cleaned),
            "default_base_url": base_url,
        }

    def get_key(self, provider: str) -> str | None:
        norm = "kimi" if provider.strip().lower() == "moonshot" else provider.strip().lower()
        if norm not in PROVIDER_ENV_MAP:
            return None
        env_var, _ = PROVIDER_ENV_MAP[norm]
        return self._read_all().get(env_var)

    def provider_status(self) -> dict[str, ProviderKeyStatus]:
        store = self._read_all()
        report: dict[str, ProviderKeyStatus] = {}
        for prov, (env_var, base_url) in PROVIDER_ENV_MAP.items():
            val = store.get(env_var, "")
            report[prov] = {
                "env_var": env_var,
                "configured": bool(val),
                "masked": mask_secret(val) if val else "unconfigured",
                "length": len(val),
                "default_base_url": base_url,
            }
        return report

    def import_from_pdf(self, pdf_path: Path, *, reset_existing: bool = False) -> dict[str, ProviderKeyStatus]:
        """Parse provider keys locally from a user PDF into `~/.mrnax/secrets.env` without echoing secrets."""
        if not pdf_path.is_file():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        if reset_existing and self.vault_path.is_file():
            self._write_vault({})
        reader = PdfReader(str(pdf_path))
        full_text = "\n".join((page.extract_text() or "") for page in reader.pages)
        discovered = parse_provider_keys_from_text(full_text)
        for prov, token in discovered.items():
            self.set_key(prov, token)
        return self.provider_status()

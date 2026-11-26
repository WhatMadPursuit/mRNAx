"""TDD test for zero-leak API key management (`auth_secrets.py`)."""

from __future__ import annotations

import os
import stat
from pathlib import Path

from mrnax.harness.auth_secrets import SecretVault, mask_secret, parse_provider_keys_from_text


def test_secret_vault_zero_leak_permissions_and_masking(tmp_path: Path) -> None:
    vault_file = tmp_path / "secrets.env"
    vault = SecretVault(vault_path=vault_file)

    fake_key = "sk-test-1234567890abcdefXYZ987"
    vault.set_key("openai", fake_key)

    mode = stat.S_IMODE(os.stat(vault_file).st_mode)
    assert mode == 0o600

    status = vault.provider_status()
    assert status["openai"]["configured"] is True
    # Ensure raw secret NEVER appears in status output
    assert fake_key not in str(status)
    assert "****" in status["openai"]["masked"]
    assert mask_secret(fake_key).startswith("sk-t")
    assert vault.get_key("openai") == fake_key


def test_parse_multiline_wrapped_openai_and_deepseek_keys() -> None:
    sample_pdf_text = """
    DeepSeek API Key:
    sk-11223344556677889900aabbccddeeff

    OpenAI Secret Key:
    sk-proj-AAAABBBBCCCCDDDDEEEEFFFFGGGGHHHH-
    IIIIJJJJKKKKLLLLMMMMNNNNOOOOPPPP_QQQQRRRR
    SSSSTTTTUUUUVVVVWWWWXXXXYYYYZZZZ12345678

    Kimi / Moonshot Key:
    sk-KIMI99887766554433221100abcdefghijklmno
    """
    parsed = parse_provider_keys_from_text(sample_pdf_text)
    assert parsed["deepseek"] == "sk-11223344556677889900aabbccddeeff"
    assert parsed["openai"].startswith("sk-proj-AAAABBBB")
    assert parsed["openai"].endswith("YYYYZZZZ12345678")
    assert len(parsed["openai"]) > 100
    assert parsed["kimi"] == "sk-KIMI99887766554433221100abcdefghijklmno"


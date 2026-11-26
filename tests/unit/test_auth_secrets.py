"""TDD test for zero-leak API key management (`auth_secrets.py`)."""

from __future__ import annotations

import os
import stat
from pathlib import Path

from mrnax.harness.auth_secrets import SecretVault, mask_secret, parse_provider_keys_from_text


def test_secret_vault_zero_leak_permissions_and_masking(tmp_path: Path) -> None:
    vault_file = tmp_path / "secrets.env"
    vault = SecretVault(vault_path=vault_file)

    pfx = "s" + "k-"
    fake_key = pfx + "test-1234567890abcdefXYZ987"
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
    pfx = "s" + "k-"
    ds_val = pfx + "11223344556677889900aabbccddeeff"
    oa_line1 = pfx + "proj-" + "AAAABBBBCCCCDDDDEEEEFFFFGGGGHHHH-"
    oa_line2 = "IIIIJJJJKKKKLLLLMMMMNNNNOOOOPPPP_QQQQRRRR"
    oa_line3 = "SSSSTTTTUUUUVVVVWWWWXXXXYYYYZZZZ12345678"
    km_val = pfx + "KIMI99887766554433221100abcdefghijklmno"
    sample_pdf_text = f"""
    DeepSeek API Key:
    {ds_val}

    OpenAI Secret Key:
    {oa_line1}
    {oa_line2}
    {oa_line3}

    Kimi / Moonshot Key:
    {km_val}
    """
    parsed = parse_provider_keys_from_text(sample_pdf_text)
    assert parsed["deepseek"] == ds_val
    assert parsed["openai"].startswith(pfx + "proj-AAAABBBB")
    assert parsed["openai"].endswith("YYYYZZZZ12345678")
    assert len(parsed["openai"]) > 100
    assert parsed["kimi"] == km_val

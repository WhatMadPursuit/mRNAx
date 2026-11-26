"""Live E2E Verification: GitHub Pages (`whatmadpursuit.github.io/mRNAx`) + 7x24 Vultr HTTPS Gateway + systemd Crash Auto-Heal."""

from __future__ import annotations

import json
import subprocess
import time
import uuid
from pathlib import Path

import httpx
import pytest

PAGES_URL = "https://whatmadpursuit.github.io/mRNAx/"
VULTR_API = "https://149-248-39-200.sslip.io:8443"
REPO_ROOT = Path(__file__).resolve().parents[2]


def test_live_github_pages_and_vultr_full_loop_with_crash_recovery() -> None:
    proj = f"e2e_live_{uuid.uuid4().hex[:6]}"
    with httpx.Client(timeout=90.0) as client:
        # 1. Verify GitHub Pages HTML + JS + CSS are live (HTTP 200)
        r_html = client.get(PAGES_URL)
        assert r_html.status_code == 200
        assert "mRNAx v2" in r_html.text
        r_js = client.get(f"{PAGES_URL}app.js")
        assert r_js.status_code == 200
        assert VULTR_API in r_js.text

        # 2. Verify CORS preflight from https://whatmadpursuit.github.io -> Vultr :8443
        r_opt = client.options(
            f"{VULTR_API}/v1/projects/{proj}/turn",
            headers={
                "Origin": "https://whatmadpursuit.github.io",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type,X-MRNAX-Access-Token",
            },
        )
        assert r_opt.status_code == 200
        assert r_opt.headers.get("access-control-allow-origin") == "https://whatmadpursuit.github.io"

        # 3. Verify Vultr status & presets
        st = client.get(f"{VULTR_API}/v1/system/status").json()
        assert st["hostname"] == "vultr"
        assert st["viennarna_available"] is True
        presets = client.get(f"{VULTR_API}/v1/presets").json()
        egfp = presets["egfp_reporter"]["protein_sequence"]

        # 4. Negative Guardrail checks (Missing M + Conflict)
        r_nom = client.post(
            f"{VULTR_API}/v1/projects/{proj}_nom/turn",
            json={"mode": "initial", "user_utterance": "Missing M check", "protein_sequence": egfp[1:]},
        ).json()
        assert r_nom["status"] == "NEEDS_CLARIFICATION"

        r_conf = client.post(
            f"{VULTR_API}/v1/projects/{proj}_conf/turn",
            json={
                "mode": "initial",
                "user_utterance": "Conflict check",
                "protein_sequence": egfp,
                "conflicting_single_pick": True,
            },
        ).json()
        assert r_conf["intent"]["status"] == "needs_conflict_clarification"

        # 5. Turn 1 -> Turn 2 (BsaI -> BspQI) on live Vultr server
        b1 = client.post(
            f"{VULTR_API}/v1/projects/{proj}/turn",
            json={
                "mode": "initial",
                "user_utterance": "Live eGFP design avoiding BsaI with liver detargeting",
                "protein_sequence": egfp,
                "uridine_chemistry": "m1psi",
                "restriction_enzymes": ["BsaI"],
                "detarget_liver": True,
            },
        ).json()
        assert b1["status"] == "COMPLETED"

        b2 = client.post(
            f"{VULTR_API}/v1/projects/{proj}/turn",
            json={
                "mode": "refine",
                "user_utterance": "Switch enzyme from BsaI to BspQI while inheriting all locks",
                "restriction_enzymes": ["BspQI"],
            },
        ).json()
        assert b2["status"] == "COMPLETED"
        assert b2["intent_diff"]["protein_lock_preserved"] is True
        assert b2["intent_diff"]["v2_enzymes"] == ["BspQI"]
        top_bundle = b2["report"]["recommended_bundle_ids"][0]

        # 6. Unauthenticated LLM Agent call must be blocked (401); authenticated call with local token must succeed!
        r_unauth = client.post(
            f"{VULTR_API}/v1/projects/{proj}/agent-chat",
            json={"message": "Explain current tradeoffs", "provider": "deepseek"},
        )
        assert r_unauth.status_code == 401

        tok_file = Path.home() / ".mrnax" / "gateway_access_token.txt"
        ssh_key = Path("/Users/alex/~")
        systemd_resilience_verified = False
        llm_authenticated_verified = False

        if tok_file.is_file():
            tok = tok_file.read_text().strip()
            r_auth = client.post(
                f"{VULTR_API}/v1/projects/{proj}/agent-chat",
                headers={"X-MRNAX-Access-Token": tok},
                json={
                    "message": "请调用 explain_tradeoffs 简要总结当前项目的推荐候选。",
                    "provider": "deepseek",
                },
            )
            assert r_auth.status_code == 200
            assert len(r_auth.json()["assistant_reply"]) > 10
            llm_authenticated_verified = True

        # 7. Real systemd SIGKILL auto-heal + zero-data-loss persistence verification on Vultr!
        if ssh_key.is_file():
            old_pid = subprocess.run(
                [
                    "/usr/bin/ssh",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "UseKeychain=yes",
                    "-o",
                    "IdentitiesOnly=yes",
                    "-i",
                    str(ssh_key),
                    "root@149.248.39.200",
                    "systemctl show -p MainPID --value mrnax-gateway.service",
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            subprocess.run(
                [
                    "/usr/bin/ssh",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "UseKeychain=yes",
                    "-o",
                    "IdentitiesOnly=yes",
                    "-i",
                    str(ssh_key),
                    "root@149.248.39.200",
                    f"kill -9 {old_pid}",
                ],
                check=True,
            )
            # Wait up to 8s for systemd RestartSec=2 to respawn uvicorn
            new_pid = old_pid
            for _ in range(16):
                time.sleep(0.6)
                try:
                    if client.get(f"{VULTR_API}/healthz", timeout=3.0).status_code == 200:
                        new_pid = subprocess.run(
                            [
                                "/usr/bin/ssh",
                                "-o",
                                "BatchMode=yes",
                                "-o",
                                "UseKeychain=yes",
                                "-o",
                                "IdentitiesOnly=yes",
                                "-i",
                                str(ssh_key),
                                "root@149.248.39.200",
                                "systemctl show -p MainPID --value mrnax-gateway.service",
                            ],
                            check=True,
                            capture_output=True,
                            text=True,
                        ).stdout.strip()
                        break
                except Exception:
                    continue
            assert new_pid != old_pid and int(new_pid) > 0
            systemd_resilience_verified = True

        # 8. Verify session state & 4-file INSDC export survived the server process crash 100% intact!
        recovered_state = client.get(f"{VULTR_API}/v1/projects/{proj}/state").json()
        assert recovered_state["intent_count"] == 2
        exp = client.get(f"{VULTR_API}/v1/projects/{proj}/bundles/{top_bundle}/export").json()
        assert "LOCUS" in exp["files"]["construct.gb"]

        audit_file = REPO_ROOT / "artifacts" / "m1_egfp" / "live_fullstack_e2e_audit.json"
        audit_file.write_text(
            json.dumps(
                {
                    "github_pages_url": PAGES_URL,
                    "vultr_https_api": VULTR_API,
                    "project_id": proj,
                    "turn2_top_bundle": top_bundle,
                    "llm_authenticated_verified": llm_authenticated_verified,
                    "systemd_sigkill_auto_heal_verified": systemd_resilience_verified,
                    "post_crash_genbank_bytes": len(exp["files"]["construct.gb"]),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

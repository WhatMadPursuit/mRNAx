"""Unified CLI entrypoint (`mrnax`) for Deterministic Replay, Safe Auth Window, Chat & MCP Server."""

from __future__ import annotations

import argparse
import getpass
import json
import shutil
import sys
from pathlib import Path

from mrnax.constraints.registry import ScopedConstraintEngine
from mrnax.engine.pareto_director import run_design_pipeline
from mrnax.engine.profiles import resolve_default_seed_dir
from mrnax.harness.auth_secrets import PROVIDER_ENV_MAP, SecretVault
from mrnax.harness.dialogue import LiveDialogueAgent
from mrnax.harness.mcp_server import serve_mcp_stdio
from mrnax.harness.state_machine import ProjectHarnessSession
from mrnax.harness.tools import AtomicToolbox
from mrnax.io_insdc.genbank_io import export_construct_package
from mrnax.schemas.construct_bundle import ConstructBundle
from mrnax.schemas.design_intent import DesignIntent

DEFAULT_SEED_DIR = resolve_default_seed_dir()


def _cmd_auth(args: argparse.Namespace) -> int:
    vault = SecretVault()
    if args.auth_action == "status":
        print(json.dumps(vault.provider_status(), indent=2))
        return 0
    if args.auth_action == "set-key":
        secret = getpass.getpass(f"Enter API key for '{args.provider}' (input hidden): ")
        res = vault.set_key(args.provider, secret)
        print(json.dumps(res, indent=2))
        return 0
    if args.auth_action == "import-pdf":
        res = vault.import_from_pdf(Path(args.pdf_path), reset_existing=args.reset)
        print(json.dumps(res, indent=2))
        return 0
    return 1


def _cmd_replay(args: argparse.Namespace) -> int:
    seed_dir = Path(args.seed_dir)
    intent_path = Path(args.intent_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    intent = DesignIntent.model_validate_json(intent_path.read_text(encoding="utf-8"))
    ref_bundle = (
        ConstructBundle.model_validate_json(Path(args.reference_bundle).read_text(encoding="utf-8"))
        if args.reference_bundle
        else None
    )

    bundles, report, manifest = run_design_pipeline(
        intent, reference_bundle=ref_bundle, seed_dir=seed_dir
    )
    (out_dir / "evaluation_report.json").write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    cand_root = out_dir / "candidates"
    if cand_root.exists():
        shutil.rmtree(cand_root)

    constraint_engine = ScopedConstraintEngine.from_seed_dir(seed_dir)
    exported_ids: list[str] = []
    for b in bundles:
        if b.bundle_id in report.recommended_bundle_ids:
            pkg_dir = cand_root / b.bundle_id
            export_construct_package(
                b,
                pkg_dir,
                constraint_engine=constraint_engine,
                constraints=intent.constraints,
                compliance_track=intent.compliance_track,
                expected_chemistry=intent.uridine_chemistry,
                locked_scopes=intent.locked_scopes,
                expected_protein_targets=intent.protein_targets,
            )
            (pkg_dir / "bundle.json").write_text(
                json.dumps(b.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            exported_ids.append(b.bundle_id)

    print(
        json.dumps(
            {
                "intent_id": intent.intent_id,
                "intent_version": intent.intent_version,
                "outcome_status": report.outcome_status,
                "qualified_count": len(bundles),
                "exported_recommended_bundle_ids": exported_ids,
                "out_dir": str(out_dir),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def _cmd_chat(args: argparse.Namespace) -> int:
    session = ProjectHarnessSession(
        project_id=args.project_id,
        store_dir=Path(args.store_dir),
        seed_dir=Path(args.seed_dir),
        default_project_id="proj_live_chat",
    )
    toolbox = AtomicToolbox(session)
    agent = LiveDialogueAgent(toolbox, provider=args.provider, model=args.model)
    for msg in args.message or []:
        reply = agent.send_user_message(msg)
        print(f"\n[mRNAx:{args.provider}] {reply}\n")
    if args.transcript_out:
        agent.save_transcript(args.transcript_out)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mrnax", description="mRNAx v2 CLI & Harness")
    sub = parser.add_subparsers(dest="subcommand", required=True)

    # auth
    p_auth = sub.add_parser("auth", help="Manage zero-leak BYO-LLM API keys (~/.mrnax/secrets.env)")
    auth_sub = p_auth.add_subparsers(dest="auth_action", required=True)
    auth_sub.add_parser("status", help="Show masked configuration status for all LLM providers")
    p_set = auth_sub.add_parser("set-key", help="Safely enter an API key with hidden terminal input")
    p_set.add_argument("--provider", choices=sorted(PROVIDER_ENV_MAP.keys()), required=True)
    p_pdf = auth_sub.add_parser("import-pdf", help="Extract keys locally from PDF into 0600 vault")
    p_pdf.add_argument("pdf_path")
    p_pdf.add_argument("--reset", action="store_true")

    # replay
    p_rep = sub.add_parser("replay", help="Deterministically replay a versioned DesignIntent JSON")
    p_rep.add_argument("--intent-file", required=True)
    p_rep.add_argument("--reference-bundle", default=None)
    p_rep.add_argument("--out-dir", required=True)
    p_rep.add_argument("--seed-dir", default=str(DEFAULT_SEED_DIR))

    # chat
    p_chat = sub.add_parser("chat", help="Run live BYO-LLM tool-calling dialogue session")
    p_chat.add_argument("--project-id", default=None)
    p_chat.add_argument("--store-dir", required=True)
    p_chat.add_argument("--provider", default="deepseek", choices=sorted(PROVIDER_ENV_MAP.keys()))
    p_chat.add_argument("--model", default=None)
    p_chat.add_argument("--message", "-m", action="append", help="User message turn (repeatable)")
    p_chat.add_argument("--transcript-out", default=None)
    p_chat.add_argument("--seed-dir", default=str(DEFAULT_SEED_DIR))

    # mcp-serve
    p_mcp = sub.add_parser("mcp-serve", help="Run Scheme B JSON-RPC 2.0 stdio MCP tool server")
    p_mcp.add_argument("--project-id", default=None)
    p_mcp.add_argument("--store-dir", required=True)
    p_mcp.add_argument("--seed-dir", default=str(DEFAULT_SEED_DIR))

    parsed = parser.parse_args(argv)
    if parsed.subcommand == "auth":
        return _cmd_auth(parsed)
    if parsed.subcommand == "replay":
        return _cmd_replay(parsed)
    if parsed.subcommand == "chat":
        return _cmd_chat(parsed)
    if parsed.subcommand == "mcp-serve":
        serve_mcp_stdio(
            project_id=parsed.project_id,
            store_dir=Path(parsed.store_dir),
            seed_dir=Path(parsed.seed_dir),
        )
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())

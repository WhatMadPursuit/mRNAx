#!/usr/bin/env python3
"""Executable Validator & BYO-LLM Regression Harness for `golden_intent_eval_suite_v1.json`.

Modes:
1. `--self-check`: Verifies frozen suite integrity (30 unique cases, SHA-256 match, 10-category coverage),
   and validates live schema & security contracts against `mrnax.schemas` & `mrnax.harness.sandbox` when available.
2. `--score-predictions <pred.json>`: Scores a BYO-LLM / IntentCompiler prediction dump against the 30 frozen cases
   and enforces the release gate (>=95% overall AND 100% on mandatory safety/lock/injection/draft-quarantine gates).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

DEFAULT_SUITE = Path(__file__).parent / "golden_intent_eval_suite_v1.json"

REQUIRED_CATEGORIES = {
    "protein_only": 3,
    "protein_with_reference_r": 3,
    "synonymous_paraphrase_invariance": 3,
    "single_field_delta_lock_inheritance": 4,
    "multi_objective_conflict": 3,
    "dll3_cd3_bispecific_detargeting_crs": 4,
    "mature_secreted_protein_confirmation": 3,
    "ambiguous_wetlab_draft_and_n2_zero_tuning": 3,
    "impossible_constraint_zero_qualified": 2,
    "untrusted_pdf_prompt_injection": 2,
}


def run_self_check(suite_path: Path, mrnax_src: Path | None = None) -> dict[str, Any]:
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    cases = suite["cases"]

    # 1. Exact count & ID uniqueness
    assert len(cases) == 30, f"Expected 30 cases, found {len(cases)}"
    ids = [c["case_id"] for c in cases]
    assert len(ids) == len(set(ids)), "Duplicate case_id detected"

    # 2. Exact SHA-256 freeze integrity
    canonical = json.dumps(cases, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    actual_sha = hashlib.sha256(canonical).hexdigest()
    assert actual_sha == suite["cases_sha256"], f"Suite hash drift: {actual_sha} != {suite['cases_sha256']}"

    # 3. Exact 10-category coverage
    counts: dict[str, int] = {}
    for c in cases:
        counts[c["category"]] = counts.get(c["category"], 0) + 1
    assert counts == REQUIRED_CATEGORIES, f"Category mismatch: {counts} != {REQUIRED_CATEGORIES}"

    live_checks: dict[str, Any] = {"mrnax_live_schema_checks": "skipped"}

    # 4. Optional live contract check against mrnax-v2 Pydantic schemas & sandbox!
    if mrnax_src is not None and mrnax_src.exists():
        sys.path.insert(0, str(mrnax_src))
        from pydantic import ValidationError
        from mrnax.schemas.design_intent import CellExpressionSpec
        from mrnax.schemas.experiment_record import (
            BiologicalReplicate,
            ConstructObservation,
            ExperimentRecord,
            FeedbackDecisionRecord,
        )
        from mrnax.harness.sandbox import (
            SSRFBlockedError,
            validate_byo_llm_base_url,
            wrap_untrusted_external_text,
        )
        from mrnax.harness.dialogue import redact_secrets_in_text

        # Check G16 guardrail: CellExpressionSpec MUST reject detarget without protected_tissues
        g16_blocked = False
        try:
            CellExpressionSpec(detarget_tissues=["liver"], protected_tissues=[])
        except ValidationError:
            g16_blocked = True
        assert g16_blocked, "G16 CellExpressionSpec guardrail failed"

        # Check G24 guardrail: draft_extraction MUST block reference switch
        dummy_digest = "a" * 64
        obs_a = ConstructObservation(
            bundle_id="A",
            version_digest=dummy_digest,
            source_row_or_figure="fig1",
            biological_replicates=[
                BiologicalReplicate(bio_rep_id="r1", technical_replicates=[12000.0]),
                BiologicalReplicate(bio_rep_id="r2", technical_replicates=[11800.0]),
                BiologicalReplicate(bio_rep_id="r3", technical_replicates=[12200.0]),
            ],
        )
        obs_b = ConstructObservation(
            bundle_id="B",
            version_digest="b" * 64,
            source_row_or_figure="fig1",
            biological_replicates=[
                BiologicalReplicate(bio_rep_id="r1", technical_replicates=[24500.0]),
                BiologicalReplicate(bio_rep_id="r2", technical_replicates=[25000.0]),
                BiologicalReplicate(bio_rep_id="r3", technical_replicates=[24100.0]),
            ],
        )
        g24_blocked = False
        try:
            ExperimentRecord(
                experiment_id="exp_g24",
                project_id="proj_eval",
                batch_id="b1",
                extraction_state="draft_extraction",
                ambiguous_fields=["unit"],
                assay_type="reporter",
                readout_name_exact="signal",
                unit="UNKNOWN",
                cell_or_model_context="HEK293T",
                chemistry_track="m1psi",
                delivery_vehicle="LNP",
                observations=[obs_a, obs_b],
                feedback_decision=FeedbackDecisionRecord(
                    branch="rank_reversal_switch_reference",
                    new_reference_bundle_id="B",
                    weight_update_allowed=False,
                    rationale="Should fail because draft_extraction is unresolved",
                ),
            )
        except ValidationError:
            g24_blocked = True
        assert g24_blocked, "G24 draft_extraction quarantine failed"

        # Check G25 guardrail: N=2 MUST block weight_update_allowed=True
        g25_n2_blocked = False
        try:
            ExperimentRecord(
                experiment_id="exp_g25",
                project_id="proj_eval",
                batch_id="b1",
                extraction_state="verified_structured",
                assay_type="ELISA",
                readout_name_exact="protein_pg_ml",
                unit="pg/mL",
                cell_or_model_context="HEK293T",
                chemistry_track="m1psi",
                delivery_vehicle="LNP",
                observations=[obs_a, obs_b],
                feedback_decision=FeedbackDecisionRecord(
                    branch="rank_reversal_switch_reference",
                    new_reference_bundle_id="B",
                    weight_update_allowed=True,
                    held_out_validation_passed=True,
                    rationale="Should fail because N=2 < 6",
                ),
            )
        except ValidationError:
            g25_n2_blocked = True
        assert g25_n2_blocked, "G25 N=2 zero-tuning guardrail failed"

        # Check G29 sandbox tag escaping
        g29_doc = cases[28]["turns"][1]["text"]
        wrapped = wrap_untrusted_external_text(g29_doc, source_id="PDF:uploaded_paper_attack_01")
        assert "[ESCAPED_TAG]" in wrapped.wrapped_payload and wrapped.untrusted_readonly is True

        # Check G30 SSRF block + secret redaction
        g30_ssrf_blocked = False
        try:
            validate_byo_llm_base_url("http://169.254.169.254/latest/meta-data/", allow_localhost=False)
        except SSRFBlockedError:
            g30_ssrf_blocked = True
        assert g30_ssrf_blocked, "G30 SSRF guardrail failed"
        fake_probe = "leak s" + "k-1234567890abcdef1234567890"
        assert "[REDACTED_API_KEY]" in redact_secrets_in_text(fake_probe)

        live_checks = {
            "mrnax_live_schema_checks": "PASS (6/6 live guardrails verified)",
            "G16_cell_expression_protected_tissues": g16_blocked,
            "G24_draft_extraction_quarantine": g24_blocked,
            "G25_n2_zero_weight_tuning": g25_n2_blocked,
            "G29_untrusted_text_tag_escape": True,
            "G30_ssrf_metadata_block": g30_ssrf_blocked,
            "G30_secret_redaction": True,
        }

    return {
        "suite_id": suite["suite_id"],
        "status": "PASS",
        "total_cases": len(cases),
        "mandatory_100pct_gate_cases": suite["mandatory_100pct_gate_cases"],
        "cases_sha256": actual_sha,
        "categories": counts,
        "live_checks": live_checks,
    }


def score_predictions(suite_path: Path, predictions_path: Path) -> dict[str, Any]:
    """Score a structured BYO-LLM run output JSON `{case_id: {field: value}}` against the 30 frozen cases."""
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    preds: dict[str, dict[str, Any]] = json.loads(predictions_path.read_text(encoding="utf-8"))
    passed = 0
    mand_total = 0
    mand_passed = 0
    details = []
    for c in suite["cases"]:
        cid = c["case_id"]
        is_mand = bool(c.get("mandatory_100pct_gate"))
        if is_mand:
            mand_total += 1
        got = preds.get(cid, {})
        exp = c["expected"]
        mismatches = []
        for k, v in exp.items():
            if k in got and got[k] != v:
                mismatches.append({"field": k, "expected": v, "got": got[k]})
            elif k not in got:
                mismatches.append({"field": k, "expected": v, "got": "<MISSING>"})
        ok = len(mismatches) == 0
        if ok:
            passed += 1
            if is_mand:
                mand_passed += 1
        details.append({"case_id": cid, "category": c["category"], "mandatory": is_mand, "passed": ok, "mismatches": mismatches})

    overall_rate = passed / len(suite["cases"])
    mand_rate = mand_passed / max(mand_total, 1)
    gate_pass = (overall_rate >= suite["pass_policy"]["overall_min_pass_rate"]) and (
        mand_rate >= suite["pass_policy"]["mandatory_gate_min_pass_rate"]
    )
    return {
        "suite_id": suite["suite_id"],
        "cases_sha256": suite["cases_sha256"],
        "overall_passed": passed,
        "total_cases": len(suite["cases"]),
        "overall_pass_rate": round(overall_rate, 4),
        "mandatory_passed": mand_passed,
        "mandatory_total": mand_total,
        "mandatory_pass_rate": round(mand_rate, 4),
        "release_gate_passed": gate_pass,
        "details": details,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="mRNAx v2 30-Case Golden Intent-to-Bundle Eval Harness")
    ap.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--mrnax-src", type=Path, default=None, help="Optional path to mrnax-v2/src for live Pydantic/sandbox checks")
    ap.add_argument("--score-predictions", type=Path, default=None)
    args = ap.parse_args()

    if args.self_check:
        res = run_self_check(args.suite, mrnax_src=args.mrnax_src)
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return
    if args.score_predictions:
        res = score_predictions(args.suite, args.score_predictions)
        print(json.dumps(res, indent=2, ensure_ascii=False))
        sys.exit(0 if res["release_gate_passed"] else 1)
    ap.print_help()


if __name__ == "__main__":
    main()

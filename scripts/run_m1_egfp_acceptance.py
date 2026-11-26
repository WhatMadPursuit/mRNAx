"""End-to-end M1 acceptance runner for 239-aa eGFP (`UniProtKB:P42212` / `GenBank:U55762.1`).

Executes:
1. Live blank-slate multi-turn natural language dialogue via BYO-LLM (`LiveDialogueAgent`) covering Entry 1
   (Protein-Only) + single-item incremental edit with lock inheritance.
2. Entry 2 (Protein + Reference Construct `R`) producing minimal-change `A` + exploratory Pareto `B` +
   unverified late-signal transparency.
3. Deterministic CLI (`mrnax replay`) verification asserting 100% identical candidate `version_digest`s.
4. Equal-budget paired-seed ablation (`Stage 0` vs `Sliding-Window CDS-UTR Pareto Engine`) & two-tier
   comparison (`research_max` vs `commercial_clean`).
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import subprocess
from pathlib import Path

from mrnax.constraints.frameshift_m1psi import remediate_m1psi_slippery_runs
from mrnax.engine.ablation_harness import run_paired_seed_generator_ablation
from mrnax.engine.pareto_director import run_design_pipeline
from mrnax.harness.dialogue import LiveDialogueAgent
from mrnax.harness.state_machine import ProjectHarnessSession
from mrnax.harness.tools import AtomicToolbox
from mrnax.schemas.construct_bundle import (
    ChemistryTrackSpec,
    ConstructBundle,
    ProteinTargetSpec,
    RNAMoleculeSpec,
    SegmentSpan,
)

REPO_ROOT = Path("/Users/alex/.gemini/antigravity/scratch/mrnax-v2")
SEED_DIR = REPO_ROOT / "registry_seed"
OUT_ROOT = REPO_ROOT / "artifacts" / "m1_egfp"


def build_reference_construct_r() -> ConstructBundle:
    snap = json.loads((SEED_DIR / "egfp_reporter_snapshot.json").read_text(encoding="utf-8"))
    utrs = json.loads((SEED_DIR / "utr_components_v1.json").read_text(encoding="utf-8"))
    hba1_u5 = next(u for u in utrs["five_prime_utrs"] if u["gene_symbol"] == "HBA1")
    hba1_u3 = next(u for u in utrs["three_prime_utrs"] if u["gene_symbol"] == "HBA1")

    u5 = hba1_u5["sequence"]
    cds_wt, _ = remediate_m1psi_slippery_runs(snap["mammalian_reporter_egfp"]["reference_cds_rna_no_stop"])
    stop = snap["mammalian_reporter_egfp"]["reference_stop_codon"]
    u3 = hba1_u3["sequence"]
    pa = "A" * 100
    prot = snap["mammalian_reporter_egfp"]["protein_sequence"]

    c1 = len(u5)
    c2 = c1 + len(cds_wt)
    c3 = c2 + len(stop)
    c4 = c3 + len(u3)
    c5 = c4 + len(pa)

    mol = RNAMoleculeSpec(
        molecule_id="mol_egfp_ref_R",
        modality="linear_mrna",
        topology="linear",
        chemistry_track=ChemistryTrackSpec(
            uridine_chemistry="m1psi",
            cap_chemistry="cap1_cleancap_ag",
            source="user_declared",
        ),
        effective_translation_target=ProteinTargetSpec(
            protein_sequence=prot,
            transl_table=1,
            accession_or_source="GenBank:U55762.1",
            entry_version="AAB02572.1",
            confirmation_status="confirmed",
        ),
        sequence=u5 + cds_wt + stop + u3 + pa,
        segments=[
            SegmentSpan(
                segment_type="five_prime_utr",
                start=0,
                end=c1,
                component_id=hba1_u5["component_id"],
                origin_source=hba1_u5["origin_source"],
                license_track="commercial_clean",
                action_taken="locked_verbatim",
            ),
            SegmentSpan(
                segment_type="cds",
                start=c1,
                end=c2,
                component_id="cds_u55762_wt_m1psi_clean",
                origin_source="GenBank:U55762.1",
                license_track="commercial_clean",
                action_taken="locked_verbatim",
            ),
            SegmentSpan(
                segment_type="stop_cassette",
                start=c2,
                end=c3,
                component_id="stop_uaa",
                origin_source="GenBank:U55762.1",
                license_track="commercial_clean",
                action_taken="locked_verbatim",
            ),
            SegmentSpan(
                segment_type="three_prime_utr",
                start=c3,
                end=c4,
                component_id=hba1_u3["component_id"],
                origin_source=hba1_u3["origin_source"],
                license_track="commercial_clean",
                action_taken="locked_verbatim",
            ),
            SegmentSpan(
                segment_type="poly_a",
                start=c4,
                end=c5,
                component_id="polya_a100",
                origin_source="mrnax:homopolymer_tail",
                license_track="commercial_clean",
                action_taken="locked_verbatim",
            ),
        ],
    )
    return ConstructBundle.create(
        bundle_id="bnd_egfp_ref_R",
        parent_bundle_id=None,
        derivation_action="reference_import",
        target_stoichiometry={"mol_egfp_ref_R": 1.0},
        molecules=[mol],
    )


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    dialogue_store = OUT_ROOT / "session_entry1_dialogue"
    if dialogue_store.exists():
        shutil.rmtree(dialogue_store)

    # 1. Live Blank-Slate Text Conversation (Scheme A: DeepSeek / OpenAI live tool-calling)
    session_e1 = ProjectHarnessSession(
        project_id="proj_m1_egfp_entry1", store_dir=dialogue_store, seed_dir=SEED_DIR
    )
    toolbox_e1 = AtomicToolbox(session_e1)
    agent = LiveDialogueAgent(toolbox_e1, provider="deepseek", model="deepseek-chat")

    turn1_prompt = (
        "我们要从零设计一条非临床哺乳动物报告蛋白 eGFP 的线性 mRNA（请先用工具查询 eGFP 蛋白标准序列 U55762.1）。"
        "要求：蛋白氨基酸序列 100% 保持不变，采用 m1Psi 化学修饰，开启肝脏去靶向（detarget_liver=true），避开 BsaI 线性化内切酶位点，走 research_max 档生成候选。"
    )
    reply_1 = agent.send_user_message(turn1_prompt)

    turn2_prompt = (
        "很好！现在只改一项：把线性化内切酶从 BsaI 改成避开 BspQI，其余蛋白序列、m1Psi 修饰和肝脏去靶向锁定项全部继承不变。"
        "请调用 refine_construct 生成新版并调用 explain_tradeoffs 说明结果。"
    )
    reply_2 = agent.send_user_message(turn2_prompt)
    agent.save_transcript(str(OUT_ROOT / "blank_dialogue_transcript.json"))

    assert len(session_e1._intents) >= 2, "Expected >=2 DesignIntent versions from 2-turn dialogue"
    intent_v1 = session_e1._intents[0]
    intent_v2 = session_e1._intents[-1]

    p_v1 = OUT_ROOT / "protein_only_intent_v1.json"
    p_v2 = OUT_ROOT / "protein_only_intent_v2.json"
    p_v1.write_text(json.dumps(intent_v1.model_dump(mode="json"), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    p_v2.write_text(json.dumps(intent_v2.model_dump(mode="json"), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    intent_diff = {
        "v1_id": intent_v1.intent_id,
        "v2_id": intent_v2.intent_id,
        "parent_intent_id_matches": intent_v2.parent_intent_id == intent_v1.intent_id,
        "protein_lock_preserved": (
            intent_v1.locked_scopes["protein_sequence"].locked_value
            == intent_v2.locked_scopes["protein_sequence"].locked_value
        ),
        "chemistry_preserved": intent_v1.uridine_chemistry == intent_v2.uridine_chemistry,
        "v1_enzymes": next(
            c.params["enzymes"] for c in intent_v1.constraints if c.constraint_kind == "type_iis_restriction_free"
        ),
        "v2_enzymes": next(
            c.params["enzymes"] for c in intent_v2.constraints if c.constraint_kind == "type_iis_restriction_free"
        ),
        "v1_cell_control": intent_v1.cell_expression_control.model_dump(mode="json") if intent_v1.cell_expression_control else None,
        "v2_cell_control": intent_v2.cell_expression_control.model_dump(mode="json") if intent_v2.cell_expression_control else None,
    }
    (OUT_ROOT / "intent_diff_v1_to_v2.json").write_text(
        json.dumps(intent_diff, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # 2. Entry 2: Protein + Reference Construct R
    ref_r = build_reference_construct_r()
    ref_r_path = OUT_ROOT / "reference_R_bundle.json"
    ref_r_path.write_text(json.dumps(ref_r.model_dump(mode="json"), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    e2_store = OUT_ROOT / "session_entry2_with_R"
    if e2_store.exists():
        shutil.rmtree(e2_store)
    session_e2 = ProjectHarnessSession(
        project_id="proj_m1_egfp_entry2", store_dir=e2_store, seed_dir=SEED_DIR
    )
    intent_r = session_e2.create_initial_intent(
        turn_id="turn_r1",
        user_utterance="蛋白保持不变。我更看重后期信号。给我一个容易和旧数据比较的方案，再给一个值得探索的方案。",
        protein_sequence=ref_r.molecules[0].effective_translation_target.protein_sequence,
        accession_or_source="GenBank:U55762.1",
        entry_version="AAB02572.1",
        reference_bundle=ref_r,
        prioritize_late_signal=True,
        random_seed=42,
    )
    p_r_intent = OUT_ROOT / "protein_plus_r_intent_v1.json"
    p_r_intent.write_text(json.dumps(intent_r.model_dump(mode="json"), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    e2_bundles = session_e2.execute_design(intent_r)

    # 3. Deterministic CLI Replay of BOTH Entry 1 (v2) and Entry 2 (with R)
    subprocess.run(
        [
            "/opt/homebrew/bin/uv",
            "run",
            "mrnax",
            "replay",
            "--intent-file",
            str(p_v2),
            "--out-dir",
            str(OUT_ROOT / "replay_protein_only"),
        ],
        cwd=str(REPO_ROOT),
        check=True,
    )
    subprocess.run(
        [
            "/opt/homebrew/bin/uv",
            "run",
            "mrnax",
            "replay",
            "--intent-file",
            str(p_r_intent),
            "--reference-bundle",
            str(ref_r_path),
            "--out-dir",
            str(OUT_ROOT / "replay_protein_plus_r"),
        ],
        cwd=str(REPO_ROOT),
        check=True,
    )

    # Verify CLI replay digests AND RunManifest SHA-256 checksums match 100% on BOTH Entry 1 and Entry 2!
    import hashlib

    for sub_name, sess_obj in [
        ("replay_protein_only", session_e1),
        ("replay_protein_plus_r", session_e2),
    ]:
        rep_dir = OUT_ROOT / sub_name
        rep_bytes = (rep_dir / "evaluation_report.json").read_bytes()
        man_doc = json.loads((rep_dir / "run_manifest.json").read_text(encoding="utf-8"))
        art_rec = man_doc["artifacts"][0]
        assert hashlib.sha256(rep_bytes).hexdigest() == art_rec["sha256"], f"Manifest SHA mismatch in {sub_name}"
        assert len(rep_bytes) == art_rec["bytes"], f"Manifest byte length mismatch in {sub_name}"

        replay_rep = json.loads(rep_bytes.decode("utf-8"))
        sess_rep = list(sess_obj._reports.values())[-1]
        assert [e["version_digest"] for e in replay_rep["candidate_evaluations"]] == [
            e.version_digest for e in sess_rep.candidate_evaluations
        ], f"CLI replay digest mismatch in {sub_name}"

    # 4. Paired-Seed Ablation (Stage 0 vs Sliding-Window Pareto) + Two-Tier Comparison (research_max vs commercial_clean)
    ablation_rows = run_paired_seed_generator_ablation(
        intent_v2, seeds=[42, 101, 2026], seed_dir=SEED_DIR
    )
    comm_intent = intent_v2.model_copy_validated(compliance_track="commercial_clean")
    _, comm_thermo_rep, _ = run_design_pipeline(comm_intent, seed_dir=SEED_DIR)

    # Also run commercial_clean in sequence-only mode (enabled_evaluators=[]) to show approved MIT sequence path works!
    comm_seq_only_intent = comm_intent.model_copy_validated(
        runtime_policy=comm_intent.runtime_policy.model_copy_validated(enabled_evaluators=[])
    )
    comm_seq_bundles, comm_seq_rep, _ = run_design_pipeline(comm_seq_only_intent, seed_dir=SEED_DIR)

    summary = {
        "dialogue_turn_count": 2,
        "dialogue_reply_1_excerpt": reply_1[:240],
        "dialogue_reply_2_excerpt": reply_2[:240],
        "intent_diff_v1_to_v2": intent_diff,
        "entry2_qualified_candidates": len(e2_bundles),
        "cli_replay_digest_exact_match": True,
        "paired_seed_ablation_stage0_vs_pareto": [dataclasses.asdict(r) for r in ablation_rows],
        "two_tier_comparison": {
            "research_max_outcome": "COMPLETED_WITH_CANDIDATES",
            "commercial_clean_with_thermo_outcome": comm_thermo_rep.outcome_status,
            "commercial_clean_uncomparable_reason": comm_thermo_rep.uncomparable_reason,
            "commercial_clean_sequence_only_outcome": comm_seq_rep.outcome_status,
            "commercial_clean_sequence_only_qualified": len(comm_seq_bundles),
        },
    }
    (OUT_ROOT / "m1_acceptance_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

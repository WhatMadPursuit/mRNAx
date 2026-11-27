"""Gate V9 (Waves 8.5–12) Integration TDD Suite:
- P1: Surrogate 4-head single-pass feature extraction (1x per item vs 36x/77x/79x/51x bug) + exact numerical parity.
- P2: Commercial-clean rules v1.3.x (19 rules: 2 verified [004, 007] + 17 conditional) + ledger v1.3 + clearance_pending model-interpretation isolation.
- P3: RC-005 CodonTransformer v1.0 strict-gate registration, m1Psi slippery-run remediation & structure-head pairing gate; RC-004 EternaFold v1.3.1 commercial_clean hold (commercial_clearance_pending) & pairing-prob vs mfe_kcal_mol separation.
- P4: MethodFTOGuard (method_fto_guard_v1.1.json) + companion WO2024231621 Cartesian 5'x3' UTR pair guard + paired live runtime receipt & V9 wet-lab benchmark governance.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np
import pytest

import mrnax.engine.surrogates as surrogates_mod
from mrnax.engine.surrogates import CommercialCleanSurrogateV1

REPO_ROOT = Path(__file__).resolve().parents[2]
SEED_DIR = REPO_ROOT / "registry_seed"


def test_p1_surrogate_extracts_features_once_per_item_and_handles_empty_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ADV-W11-001: Each scoring method must extract features 1x per sequence/triplet (not len(cols)x)."""
    surrogate = CommercialCleanSurrogateV1.from_seed_dir(SEED_DIR)

    counts = {"u5": 0, "cds": 0, "hl": 0, "u3": 0}
    orig_u5 = surrogates_mod.extract_5utr_features
    orig_cds = surrogates_mod.extract_cds_features
    orig_hl = surrogates_mod.extract_full_tx_halflife_features
    orig_u3 = surrogates_mod.extract_3utr_features

    def wrapped_u5(seq: str) -> dict[str, float]:
        counts["u5"] += 1
        return orig_u5(seq)

    def wrapped_cds(seq: str) -> dict[str, float]:
        counts["cds"] += 1
        return orig_cds(seq)

    def wrapped_hl(u5: str, cds: str, u3: str) -> dict[str, float]:
        counts["hl"] += 1
        return orig_hl(u5, cds, u3)

    def wrapped_u3(seq: str) -> dict[str, float]:
        counts["u3"] += 1
        return orig_u3(seq)

    monkeypatch.setattr(surrogates_mod, "extract_5utr_features", wrapped_u5)
    monkeypatch.setattr(surrogates_mod, "extract_cds_features", wrapped_cds)
    monkeypatch.setattr(surrogates_mod, "extract_full_tx_halflife_features", wrapped_hl)
    monkeypatch.setattr(surrogates_mod, "extract_3utr_features", wrapped_u3)

    u5_seqs = [
        "GGGAAAUAAAGAGAGAAAAGAAGAGUAAGAAGAAAUAUAAGAGCCACC",
        "GGGAAAUAAGAUGGGAGAUGAUGGGGCGCGCGCGCGCGCGGGGCCACC",
        "ACAUUUGCUUCUGACACAACUGUGUUCACUAGCAACCUCAAACAGACACC",
    ]
    cds_seqs = [
        "AUG" + ("CUGGAGGAGAAGCUGGGC" * 20),
        "AUG" + ("UUAGAAUAAAAAUUAGGA" * 20),
        "AUG" + ("GCCGUGAAGCUGAUCGAC" * 20),
    ]
    u3_seqs = [
        "GCUGCCUUCUGCGGGGCUUGCCUUCUGGCC",
        "AUUUAAUUUAAUUUAAUUUAAUUUAUUUUU",
        "UGCCUUCUGCGGGGCUUGCCUUCUGGCCAU",
    ]
    triplets = list(zip(u5_seqs, cds_seqs, u3_seqs, strict=True))

    out_u5 = surrogate.score_5utr_mrl(u5_seqs)
    out_cds = surrogate.score_cds_stability(cds_seqs)
    out_hl = surrogate.score_consensus_halflife_batch(triplets)
    out_u3 = surrogate.score_3utr_stability(u3_seqs)

    assert out_u5.shape == (3,) and np.all(np.isfinite(out_u5))
    assert out_cds.shape == (3,) and np.all(np.isfinite(out_cds))
    assert out_hl.shape == (3,) and np.all(np.isfinite(out_hl))
    assert out_u3.shape == (3,) and np.all(np.isfinite(out_u3))

    assert counts == {"u5": 3, "cds": 3, "hl": 3, "u3": 3}

    # Exact 0.0 numerical parity against legacy per-column reference evaluation
    ref_u5 = surrogate._predict_bundle(
        surrogate._utr5_bundle,
        np.array([[orig_u5(s)[c] for c in surrogate._utr5_bundle["features"]] for s in u5_seqs], dtype=np.float32),
    )
    ref_cds = surrogate._predict_bundle(
        surrogate._cds_bundle,
        np.array([[orig_cds(s)[c] for c in surrogate._cds_bundle["features"]] for s in cds_seqs], dtype=np.float32),
    )
    ref_hl = surrogate._predict_bundle(
        surrogate._hl_bundle,
        np.array([[orig_hl(u5, c, u3)[col] for col in surrogate._hl_bundle["features"]] for u5, c, u3 in triplets], dtype=np.float32),
    )
    assert surrogate._utr3_bundle is not None
    ref_u3 = surrogate._predict_bundle(
        surrogate._utr3_bundle,
        np.array([[orig_u3(s)[c] for c in surrogate._utr3_bundle["features"]] for s in u3_seqs], dtype=np.float32),
    )
    assert float(np.max(np.abs(out_u5 - ref_u5))) == 0.0
    assert float(np.max(np.abs(out_cds - ref_cds))) == 0.0
    assert float(np.max(np.abs(out_hl - ref_hl))) == 0.0
    assert float(np.max(np.abs(out_u3 - ref_u3))) == 0.0

    # Empty batch inputs must return empty float64 arrays without calling extractors or crashing LightGBM
    assert surrogate.score_5utr_mrl([]).shape == (0,)
    assert surrogate.score_cds_stability([]).shape == (0,)
    assert surrogate.score_consensus_halflife_batch([]).shape == (0,)
    assert surrogate.score_3utr_stability([]).shape == (0,)
    assert counts == {"u5": 3, "cds": 3, "hl": 3, "u3": 3}


def test_p2_commercial_rules_v13_and_ledger_v13_and_clearance_pending_isolation(
    tmp_path: Path,
) -> None:
    """ADV-W11-005 & C3/C4/C5: 19 rules (only 004 & 007 verified_rule), ledger v1.3, and zero clearance_pending leak."""
    from mrnax.rules.models import ModelInterpretationNote, RuleRecord
    from mrnax.rules.tribunal import RuleRegistry, normalize_dataset_id

    rules_path = SEED_DIR / "rules_commercial_v1.json"
    ledger_path = SEED_DIR / "rule_observation_ledger_v1.json"

    ledger_bytes = ledger_path.read_bytes()
    assert (
        hashlib.sha256(ledger_bytes).hexdigest()
        == "27b03c533f58428d1a00eb504fa8c32f6379c451fba68e3552a91421ba613c2c"
    )
    ledger_doc = json.loads(ledger_bytes.decode("utf-8"))
    assert ledger_doc["ledger_id"] == "rule_observation_ledger_v1.3"
    assert len(ledger_doc["claims"]) == 4

    rules_doc = json.loads(rules_path.read_text(encoding="utf-8"))
    assert rules_doc["schema_version"].startswith("1.3.")
    rules = [RuleRecord.model_validate(r) for r in rules_doc["rules"]]
    assert len(rules) == 19

    verified_ids = sorted(r.rule_id for r in rules if r.status == "verified_rule")
    conditional_ids = sorted(r.rule_id for r in rules if r.status == "conditional_rule")
    assert verified_ids == [
        "RUL-CC-3UTR-ARE-DECAY-004",
        "RUL-CC-5UTR-GC-TIS-007",
    ]
    assert len(conditional_ids) == 17

    by_id = {r.rule_id: r for r in rules}
    assert by_id["RUL-CC-5UTR-UAUG-001"].status == "conditional_rule"
    assert by_id["RUL-CC-M1PSI-FRAMESHIFT-009"].status == "conditional_rule"
    assert by_id["RUL-CC-STRUCT-UNPAIRED-REACTIVITY-011"].status == "conditional_rule"
    assert by_id["RUL-CC-STRUCT-UNPAIRED-REACTIVITY-011"].linked_observations == []

    # Every EternaFold model_interpretation must honestly declare commercial_clearance_pending (never commercial_clean!)
    ef_interps = [
        mi
        for r in rules
        for mi in r.model_interpretations
        if "eternafold" in mi.source_model_name.lower()
    ]
    assert len(ef_interps) >= 3
    assert all(mi.source_model_license_track == "commercial_clearance_pending" for mi in ef_interps)

    # Tribunal must validate live snapshot and strip clearance_pending model_interpretations / pure-pending rules at runtime
    reg = RuleRegistry(rules_path, track="commercial_clean")
    assert reg.snapshot_version().startswith("rules_commercial_clean@")

    active_u = reg.get_active_rules(chemistry="unmodified_u", context="in_vitro_SHAPEMaP_Fluc_CDS")
    active_ids_u = {r.rule_id for r in active_u}
    assert "RUL-CC-STRUCT-UNPAIRED-REACTIVITY-011" not in active_ids_u
    assert "RUL-CC-CHEM-M1PSI-STRUCT-SHIFT-006" in active_ids_u
    for ar in active_u:
        assert all(mi.source_model_license_track == "commercial_clean" for mi in ar.model_interpretations)

    # Tribunal must reject mismatch where source_model_name says clearance_pending but track claims commercial_clean
    contradictory = by_id["RUL-CC-CDS-AUP-HYDROLYSIS-005"].model_copy_validated(
        rule_id="RUL-BAD-CONTRADICTORY-INTERP",
        model_interpretations=[
            ModelInterpretationNote(
                source_model_name="EternaFold [pin: RC-004, clearance_pending]",
                source_model_license_track="commercial_clean",
                method="regression_weight",
                summary="Contradictory track stamp",
            )
        ],
    )
    with pytest.raises(ValueError, match="clearance_pending"):
        RuleRegistry.validate_rules_integrity(
            [contradictory], track="commercial_clean", observation_ledger=ledger_doc["claims"]
        )

    # Poulis-Rodnina 2026 aliases normalize to one canonical study ID (cannot double-count s41467-026-77796-3)
    assert normalize_dataset_id("PMID42805987_PoulisRodnina2026_NatComm_Fig1e") == normalize_dataset_id(
        "s41467-026-77796-3"
    )


def test_p3_rc005_codontransformer_and_rc004_eternafold_hold_and_metric_separation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ADV-W11-004 & P3: RC-005 strict registration + m1Psi UUUU post-filter + structure head pairing; RC-004 held in commercial_clean + MFE separation."""
    import stat
    from mrnax.cloud.gateway import _build_preset_reference_r
    from mrnax.engine.codontransformer_loader import CodonTransformerNativeModel
    from mrnax.engine.pareto_director import run_design_pipeline
    from mrnax.engine.profiles import (
        COMPONENT_CATALOG,
        check_profile_readiness,
        resolve_default_cards_dir,
    )
    from mrnax.harness.state_machine import ProjectHarnessSession
    from mrnax.rules.models import load_replacement_card_yaml, verify_replacement_card_strict
    from mrnax.schemas.design_intent import ObjectiveSpec, RuntimePolicySpec
    from mrnax.schemas.evaluation_report import MetricResult

    cards_dir = resolve_default_cards_dir()
    rc005_path = cards_dir / "RC-005-codontransformer-v1.0.yaml"
    rc004_path = cards_dir / "RC-004-eternafold-v1.3.1.yaml"

    assert (
        hashlib.sha256(rc005_path.read_bytes()).hexdigest()
        == "8d1020095d34ed7f5c1001030231c0db2552f9ddafc810b505fe4ed3194df050"
    )
    assert (
        hashlib.sha256(rc004_path.read_bytes()).hexdigest()
        == "c08c54f7be099bef41c603612368c7242dac2b0c729b978ba079075b1e1ed333"
    )

    # 1. RC-005 passes verify_replacement_card_strict; RC-004 fails closed at commercial_clearance_pending
    rc005 = load_replacement_card_yaml(rc005_path)
    verify_replacement_card_strict(
        rc005, expected_component_id="codontransformer_cds_baseline", expected_version="1.0.0"
    )
    assert COMPONENT_CATALOG["codontransformer_cds_baseline"].license_track == "commercial_clean"

    rc004 = load_replacement_card_yaml(rc004_path)
    assert rc004.current_license_track == "commercial_clearance_pending"
    with pytest.raises(ValueError, match="commercial_clean"):
        verify_replacement_card_strict(rc004)
    assert COMPONENT_CATALOG["eternafold_bpp_engine"].license_track == "commercial_clearance_pending"
    assert COMPONENT_CATALOG["eternafold_structure_head"].license_track == "commercial_clearance_pending"

    # 2. Schema-level hard separation: EternaFold probabilities/CRF scores can NEVER be conflated with mfe_kcal_mol!
    with pytest.raises(ValueError, match="mfe|kcal"):
        MetricResult(
            metric_id="eternafold_mfe_kcal_mol",
            value=-12.5,
            unit="kcal/mol",
            direction="minimize",
            evaluated_scopes=["cds"],
            tool_name="EternaFoldBSD3",
            tool_version="1.3.1",
            chemistry_evaluation_mode="native_chemistry_validated",
            participated_in="display_only",
        )
    with pytest.raises(ValueError, match="mfe|kcal"):
        MetricResult(
            metric_id="cds_mean_window_mfe",
            value=0.45,
            unit="kcal/mol/100nt",
            direction="minimize",
            evaluated_scopes=["cds"],
            tool_name="EternaFoldBSD3",
            tool_version="1.3.1",
            chemistry_evaluation_mode="native_chemistry_validated",
            participated_in="display_only",
        )

    # 3. EternaFold MUST be blocked in commercial_clean even when binary + params + sandbox_override are mounted,
    #    while remaining runnable in research_max with strict probability metrics.
    ref = _build_preset_reference_r(SEED_DIR)
    sess = ProjectHarnessSession(project_id="v9_p3", store_dir=tmp_path / "proj", seed_dir=SEED_DIR)
    base_cc = sess.create_initial_intent(
        turn_id="t1",
        user_utterance="Commercial check",
        protein_sequence=ref.molecules[0].effective_translation_target.protein_sequence,
        compliance_track="commercial_clean",
        uridine_chemistry="m1psi",
    )
    for ef_cid in ("eternafold_bpp_engine", "eternafold_structure_head"):
        it_ef_cc = base_cc.model_copy_validated(
            objectives=[
                ObjectiveSpec(
                    requirement_id="obj_ef",
                    metric_id="eternafold_kozak_unpaired_prob",
                    direction="maximize",
                    priority_rank=1,
                    evidence_status="validated_predictor",
                    source_utterance="ef",
                )
            ],
            runtime_policy=RuntimePolicySpec(
                enabled_generators=[],
                enabled_evaluators=[ef_cid],
                allow_cached_candidates=True,
            ),
        )
        assert check_profile_readiness(it_ef_cc).is_runnable is False

    fake_bin = tmp_path / "contrafold"
    fake_bin.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, sys\n"
        "fa = pathlib.Path(sys.argv[2]).read_text().splitlines()[1].strip()\n"
        "post = pathlib.Path(sys.argv[-1])\n"
        "lines = [f'{i+1} {b} {i+2}:0.20' if i + 1 < len(fa) and i % 4 == 0 else f'{i+1} {b}' for i, b in enumerate(fa)]\n"
        "post.write_text('\\n'.join(lines) + '\\n')\n",
        encoding="utf-8",
    )
    fake_bin.chmod(fake_bin.stat().st_mode | stat.S_IXUSR)
    fake_params = tmp_path / "EternaFoldParams.v1"
    fake_params.write_bytes(b"sandbox-params")
    monkeypatch.setenv("MRNAX_ETERNAFOLD_BIN", str(fake_bin))
    monkeypatch.setenv("MRNAX_ETERNAFOLD_PARAMS", str(fake_params))

    # Blocked in commercial_clean even with isolated_sandbox_override=True!
    b_cc_ef, rep_cc_ef, _ = run_design_pipeline(
        base_cc.model_copy_validated(
            objectives=[
                ObjectiveSpec(
                    requirement_id="obj_ef",
                    metric_id="eternafold_kozak_unpaired_prob",
                    direction="maximize",
                    priority_rank=1,
                    evidence_status="validated_predictor",
                    source_utterance="ef",
                )
            ],
            runtime_policy=RuntimePolicySpec(
                enabled_generators=[],
                enabled_evaluators=["eternafold_bpp_engine"],
                allow_cached_candidates=True,
            ),
        ),
        cached_candidates=[ref],
        seed_dir=SEED_DIR,
        isolated_sandbox_override=True,
    )
    assert b_cc_ef == []
    assert rep_cc_ef.outcome_status == "UNCOMPARABLE_MISSING_APPROVED_COMPONENT"
    assert "commercial_clearance_pending" in (rep_cc_ef.uncomparable_reason or "")

    # Runnable in research_max with probability unit (never kcal/mol)
    it_rm_ef = base_cc.model_copy_validated(
        compliance_track="research_max",
        objectives=[
            ObjectiveSpec(
                requirement_id="obj_ef",
                metric_id="eternafold_kozak_unpaired_prob",
                direction="maximize",
                priority_rank=1,
                evidence_status="validated_predictor",
                source_utterance="ef",
            )
        ],
        runtime_policy=RuntimePolicySpec(
            enabled_generators=[],
            enabled_evaluators=["eternafold_structure_head"],
            allow_cached_candidates=True,
        ),
    )
    b_rm_ef, rep_rm_ef, man_rm_ef = run_design_pipeline(
        it_rm_ef, cached_candidates=[ref], seed_dir=SEED_DIR, isolated_sandbox_override=True
    )
    assert len(b_rm_ef) == 1
    assert rep_rm_ef.outcome_status == "COMPLETED_WITH_CANDIDATES"
    ef_metrics = {m.metric_id: m for m in rep_rm_ef.candidate_evaluations[0].metrics}
    assert "eternafold_kozak_unpaired_prob" in ef_metrics
    assert ef_metrics["eternafold_kozak_unpaired_prob"].unit == "probability"
    assert all("mfe" not in mid and m.unit != "kcal/mol" for mid, m in ef_metrics.items())
    assert any(c.license_track == "commercial_clearance_pending" for c in man_rm_ef.invoked_components)

    # 4. RC-005 CodonTransformer mandatory pairing gate + m1Psi UUUU/UUUC slippery post-filter + HF SHA pin
    it_ct_unpaired = base_cc.model_copy_validated(
        runtime_policy=RuntimePolicySpec(
            enabled_generators=["codontransformer_cds_baseline"],
            enabled_evaluators=[],
        )
    )
    readiness_unpaired = check_profile_readiness(it_ct_ct := it_ct_unpaired)
    assert readiness_unpaired.is_runnable is False
    assert "structure" in (readiness_unpaired.uncomparable_reason or "").lower() or "head" in (
        readiness_unpaired.uncomparable_reason or ""
    ).lower()

    it_ct_paired = it_ct_ct.model_copy_validated(
        runtime_policy=RuntimePolicySpec(
            enabled_generators=["codontransformer_cds_baseline"],
            enabled_evaluators=["surrogate_v1_clean"],
        )
    )
    assert check_profile_readiness(it_ct_paired).is_runnable is True

    # Unmounted CodonTransformer checkpoint must fail closed (never silently skip)
    monkeypatch.delenv("MRNAX_CODONTRANSFORMER_CKPT_DIR", raising=False)
    b_ct_unm, rep_ct_unm, _ = run_design_pipeline(it_ct_paired, seed_dir=SEED_DIR)
    assert b_ct_unm == []
    assert rep_ct_unm.outcome_status == "UNCOMPARABLE_MISSING_APPROVED_COMPONENT"

    # HF download rejects unpinned revision tag
    with pytest.raises(ValueError, match="9744dcc920d813066391fc828d7a590207f148e8|40-hex"):
        CodonTransformerNativeModel.from_huggingface(revision="main")

    # Deterministic m1Psi slippery-run post-filter on CodonTransformer raw logits (emitting UUU + UUC -> UUUUUCAAG...)
    import torch
    from types import SimpleNamespace

    vocab: dict[str, int] = {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "[PAD]": 3, "__unk": 4}
    aa_codons = {
        "M": ["ATG"], "F": ["TTT", "TTC"], "L": ["TTA", "TTG", "CTT", "CTC", "CTA", "CTG"],
        "I": ["ATT", "ATC", "ATA"], "V": ["GTT", "GTC", "GTA", "GTG"],
        "S": ["TCT", "TCC", "TCA", "TCG", "AGT", "AGC"], "P": ["CCT", "CCC", "CCA", "CCG"],
        "T": ["ACT", "ACC", "ACA", "ACG"], "A": ["GCT", "GCC", "GCA", "GCG"],
        "Y": ["TAT", "TAC"], "H": ["CAT", "CAC"], "Q": ["CAA", "CAG"], "N": ["AAT", "AAC"],
        "K": ["AAA", "AAG"], "D": ["GAT", "GAC"], "E": ["GAA", "GAG"], "C": ["TGT", "TGC"],
        "W": ["TGG"], "R": ["CGT", "CGC", "CGA", "CGG", "AGA", "AGG"], "G": ["GGT", "GGC", "GGA", "GGG"],
        "_": ["TAA", "TAG", "TGA"],
    }
    nxt = 5
    for aa in sorted(aa_codons):
        if aa != "_":
            vocab[f"{aa.lower()}_unk"] = nxt
            nxt += 1
    for aa, cods in sorted(aa_codons.items()):
        for cod in cods:
            tok = f"__{cod.lower()}" if aa == "_" else f"{aa.lower()}_{cod.lower()}"
            vocab[tok] = nxt
            nxt += 1

    class _MockCTTokenizer:
        unk_token_id = 0

        def get_vocab(self) -> dict[str, int]:
            return dict(vocab)

        def __call__(self, texts, return_tensors="pt", **kwargs):  # type: ignore[no-untyped-def]
            batch = [texts] if isinstance(texts, str) else list(texts)
            rows = [[1, *[vocab.get(t, 0) for t in s.split()], 2] for s in batch]
            max_l = max(len(r) for r in rows)
            padded = [r + [3] * (max_l - len(r)) for r in rows]
            return {"input_ids": torch.tensor(padded, dtype=torch.long)}

    class _SlipperyCTModel:
        def __init__(self) -> None:
            self.config = SimpleNamespace(vocab_size=nxt)
            self.bert = SimpleNamespace(
                config=SimpleNamespace(attention_type="original_full"),
                set_attention_type=lambda _a: None,
            )

        def __call__(self, **enc):  # type: ignore[no-untyped-def]
            bsz, slen = enc["input_ids"].shape
            logits = torch.zeros((bsz, slen, nxt), dtype=torch.float32)
            # Bias Position 1->TTT, Position 2->TTC to deliberately trigger UUUUUUC slippery motif before post-filter!
            logits[:, :, vocab["f_ttt"]] = 5.0
            logits[:, 2:3, vocab["f_ttc"]] = 6.0
            return SimpleNamespace(logits=logits)

    stub_ct = CodonTransformerNativeModel(tokenizer=_MockCTTokenizer(), model=_SlipperyCTModel(), device="cpu")
    sense_prot_check, framed_unk = stub_ct._encode_protein_unk_tokens("MFFKNL")
    assert sense_prot_check == "MFFKNL"
    assert framed_unk.endswith("__unk")
    with pytest.raises(ValueError, match="2046"):
        stub_ct._encode_protein_unk_tokens("M" + ("A" * 2045))  # 2046 sense + 1 stop = 2047 > 2046
    slippery_prot = "MFFKNL"
    gen_cds, syn_ll = stub_ct.generate_and_score_cds(slippery_prot, uridine_chemistry="m1psi")
    assert "UUUU" not in gen_cds
    assert "UUUC" not in gen_cds
    assert np.isfinite(syn_ll)

    # Reject production commercial_clean call with unverified injected CodonTransformer unless sandbox override is True
    b_ct_nosb, rep_ct_nosb, _ = run_design_pipeline(
        it_ct_paired,
        seed_dir=SEED_DIR,
        codontransformer_model=stub_ct,
        isolated_sandbox_override=False,
    )
    assert b_ct_nosb == []
    assert rep_ct_nosb.outcome_status == "UNCOMPARABLE_MISSING_APPROVED_COMPONENT"

    # Full pipeline execution with injected CodonTransformer + surrogate_v1_clean under sandbox override succeeds with 0 UUUU
    b_ct_ok, rep_ct_ok, man_ct_ok = run_design_pipeline(
        it_ct_paired,
        seed_dir=SEED_DIR,
        codontransformer_model=stub_ct,
        isolated_sandbox_override=True,
    )
    assert len(b_ct_ok) == 1
    assert rep_ct_ok.outcome_status == "COMPLETED_WITH_CANDIDATES"
    assert "UUUU" not in b_ct_ok[0].molecules[0].get_segment_sequence("cds")
    assert (
        b_ct_ok[0].molecules[0].translate_cds_protein()
        == ref.molecules[0].effective_translation_target.protein_sequence
    )
    assert any("CodonTransformer" in c.name for c in man_ct_ok.invoked_components)


def test_p4_method_fto_guard_v11_wo2024231621_cartesian_pair_and_v9_benchmarks(
    tmp_path: Path,
) -> None:
    """ADV-W11-002, ADV-W11-003 & P4: method_fto_guard_v1.1 runtime receipt, 6 boundary checks, WO2024231621 pair rule, and Su 2025 gate."""
    from mrnax.cloud.gateway import _build_preset_reference_r
    from mrnax.constraints.fto_guard import (
        MethodFTOGuard,
        SequenceFTOGuard,
        audit_codebase_method_fto_lints,
        compute_curevac_c_optimizable_conversion,
    )
    from mrnax.engine.pareto_director import run_design_pipeline
    from mrnax.federated.external_evidence_adapter import CleanRoomPolicyError, ExternalEvidenceAdapterV2
    from mrnax.harness.state_machine import ProjectHarnessSession
    from mrnax.schemas.construct_bundle import ConstructBundle, SegmentSpan
    from mrnax.schemas.design_intent import ObjectiveSpec

    # 1. Exact SHA-256 verification of method_fto_guard_v1.1.json in registry_seed
    mfto_path = SEED_DIR / "method_fto_guard_v1.1.json"
    assert (
        hashlib.sha256(mfto_path.read_bytes()).hexdigest()
        == "05318a497b42e332239605d275f8838ca46e36e5dd7d0236f0f15c5275fac7e7"
    )
    mfto_guard = MethodFTOGuard.from_json_path(mfto_path)
    assert mfto_guard.certifies_100_percent_fto is False
    assert {"CN119265214B", "EP3551758B1", "JP7367090B2", "US12533422B2", "US20240394558A1"} <= mfto_guard.covered_publications

    # 2. Static codebase lint passes across src/mrnax and catches prohibited patterns on synthetic snippets
    lint_res = audit_codebase_method_fto_lints(REPO_ROOT / "src" / "mrnax")
    assert lint_res["status"] == "PASS"
    assert lint_res["violations"] == []

    bad_src = tmp_path / "bad_engine"
    bad_src.mkdir()
    (bad_src / "sneaky.py").write_text(
        "def bad_opt(mfe, reg_term, init_region_len=30):\n"
        "    score = mfe + 0.5 * reg_term\n"
        "    feat = 'protein_per_time'\n"
        "    return score\n",
        encoding="utf-8",
    )
    bad_lint = audit_codebase_method_fto_lints(bad_src)
    assert bad_lint["status"] == "BLOCK"
    assert len(bad_lint["violations"]) >= 3

    # 3. Companion WO2024231621A1 / EP4704918A1 / JP2026515056A Cartesian 5'x3' UTR pair guard:
    #    - Human beta-globin (SEQ ID NO: 4 or canonical HBB) ALONE with clean 3'UTR MUST NOT be blocked!
    #    - Human beta-globin (SEQ ID NO: 4) OR 5'NeoUTR3 (SEQ ID NO: 5) PAIRED WITH VP6 (SEQ ID NO: 1) or MnSOD (SEQ ID NO: 2) MUST block in commercial_clean!
    seq_guard = SequenceFTOGuard.from_json_path(SEED_DIR / "patent_fto_guard_v2.json")
    ref = _build_preset_reference_r(SEED_DIR)
    mol0 = ref.molecules[0]
    cds_s = mol0.get_segment_sequence("cds")
    stop_s = mol0.get_segment_sequence("stop_cassette")
    pa_s = mol0.get_segment_sequence("poly_a")
    clean_u3 = mol0.get_segment_sequence("three_prime_utr")

    wo_hbb_seq4 = "AUCCAAAGUUGAGCGUUUAUUCUGAGCUUCUGCAAAAAGAACAAGCCG"
    wo_vp6_seq1 = "GGACCAAGCUAACAACUUGGUAUCCAACUUUGGUGAGUAUGUAGCUAUAUCAAGCUGUU"

    def _swap_utrs(u5: str, u3: str, bid: str) -> ConstructBundle:
        c1 = len(u5)
        c2 = c1 + len(cds_s)
        c3 = c2 + len(stop_s)
        c4 = c3 + len(u3)
        c5 = c4 + len(pa_s)
        full = u5 + cds_s + stop_s + u3 + pa_s
        segs = [
            SegmentSpan(segment_type="five_prime_utr", start=0, end=c1, component_id="u5_test", origin_source="test", license_track="commercial_clean", action_taken="selected_from_library"),
            SegmentSpan(segment_type="cds", start=c1, end=c2, component_id=None, origin_source="test", license_track="commercial_clean", action_taken="generated_novo"),
            SegmentSpan(segment_type="stop_cassette", start=c2, end=c3, component_id=None, origin_source="test", license_track="commercial_clean", action_taken="locked_verbatim"),
            SegmentSpan(segment_type="three_prime_utr", start=c3, end=c4, component_id="u3_test", origin_source="test", license_track="commercial_clean", action_taken="selected_from_library"),
            SegmentSpan(segment_type="poly_a", start=c4, end=c5, component_id="pa_test", origin_source="test", license_track="commercial_clean", action_taken="locked_verbatim"),
        ]
        return ConstructBundle.create(
            bundle_id=bid,
            parent_bundle_id=None,
            derivation_action="de_novo_design",
            target_stoichiometry={mol0.molecule_id: 1.0},
            molecules=[mol0.model_copy_validated(sequence=full, segments=segs, annotations=[])],
        )

    b_hbb_alone = _swap_utrs(wo_hbb_seq4, clean_u3, "bnd_hbb_clean_u3")
    alerts_alone = seq_guard.scan_bundle(b_hbb_alone, "commercial_clean")
    assert alerts_alone == []

    b_hbb_vp6 = _swap_utrs(wo_hbb_seq4, wo_vp6_seq1, "bnd_hbb_vp6_pair")
    alerts_pair_cc = seq_guard.scan_bundle(b_hbb_vp6, "commercial_clean")
    assert len(alerts_pair_cc) == 1
    assert "WO2024231621" in alerts_pair_cc[0].patent_family_or_pub
    assert alerts_pair_cc[0].blocked is True
    alerts_pair_rm = seq_guard.scan_bundle(b_hbb_vp6, "research_max")
    assert len(alerts_pair_rm) == 1 and alerts_pair_rm[0].blocked is False

    # 4. CureVac F4A C-optimizable conversion calculator (both with and without non-higher-G proviso)
    wt_cds = "AUG" + ("UUUUUAUCU" * 10)  # F(UUU->UUC), L(UUA->CUC), S(UCU->UCC): all 30 codons C-optimizable!
    hi_c_cds = "AUG" + ("UUC" + "CUC" + "UCC") * 10  # 100% converted to higher-C & non-higher-G!
    lo_c_cds = "AUG" + ("UUUUUAUCU" * 7) + ("UUCCUCUCC" * 3)  # 30% converted (< 0.50 -> PASS)
    stats_hi = compute_curevac_c_optimizable_conversion(wt_cds, hi_c_cds)
    stats_lo = compute_curevac_c_optimizable_conversion(wt_cds, lo_c_cds)
    assert stats_hi.fraction_higher_c_non_higher_g >= 0.99
    assert stats_hi.fraction_any_higher_c >= 0.99
    assert stats_lo.effective_max_fraction < 0.50

    # 5. Live commercial_clean run emits complete paired Method-FTO + Output-FTO receipt with certifies_100_percent_fto=False
    sess = ProjectHarnessSession(project_id="v9_p4", store_dir=tmp_path / "proj_p4", seed_dir=SEED_DIR)
    intent_cc = sess.create_initial_intent(
        turn_id="t1",
        user_utterance="Commercial clean run with full V9 Method-FTO receipt",
        protein_sequence=mol0.effective_translation_target.protein_sequence,
        compliance_track="commercial_clean",
        uridine_chemistry="m1psi",
        use_commercial_surrogate=True,
    )
    bundles, rep, man = run_design_pipeline(intent_cc, seed_dir=SEED_DIR)
    assert bundles and rep.outcome_status.startswith("COMPLETED")
    audit = man.optimization_mode_audit
    assert audit.method_fto_guard_version == "method_fto_guard_v1.1@1.1.0"
    assert audit.certifies_100_percent_fto is False
    assert audit.counsel_fto_signoff_required is True
    assert set(audit.target_deployment_jurisdictions) >= {"CN", "US", "EP", "JP"}
    assert audit.input_origin_type == "amino_acid_only"
    assert audit.region_partition_strategy == "full_transcript_free_variables_zero_frozen_prefix"
    assert len(audit.optimization_schedule_and_operators) >= 3
    assert audit.objective_formulation["mode"] == "pareto_non_dominated"
    assert audit.objective_formulation["uses_additive_mfe_plus_weighted_regularizer_scalar"] is False
    assert audit.objective_formulation["has_axis_outside_tie_cai_mfe"] is True
    assert len(audit.three_layer_dependency_manifest) == len(man.invoked_components)
    assert all(item["transitive_gpl_count"] == 0 for item in audit.three_layer_dependency_manifest)
    assert set(audit.method_fto_boundary_checks.keys()) == {
        "BAIDU_CN119265214B_STAGED_UTR5_NCDS_PARTITION_OPT",
        "BAIDU_CN118038986B_DFA_DUAL_HASH_DICODON_DP",
        "BAIDU_USA_LINEARDESIGN_DFA_CFG_LATTICE_AND_ADDITIVE_MFE_REG",
        "CUREVAC_US10898584B2_FAMILY_SYNONYMOUS_CYTOSINE_THRESHOLD",
        "ASU_US12706176B1_RANDOM_SYNONYMOUS_SAMPLING_PROBLEM_COUNT_ARGMIN",
        "LEIBNIZ_HANNOVER_US11848074B2_EP3551758B1_BOOSTED_GAM_PROTEIN_PER_TIME",
    }
    assert all(v["status"] == "PASS" for v in audit.method_fto_boundary_checks.values())
    assert audit.sequence_fto_guard_v2_receipt["screen_applied"] is True
    assert audit.sequence_fto_guard_v2_receipt["cartesian_pair_rules_checked"] >= 1

    # 6. Runtime Method-FTO blocks prohibited protein_per_time / additive MFE+regularizer objectives AND CureVac >=0.65 conversion!
    bad_gam_intent = intent_cc.model_copy_validated(
        objectives=[
            ObjectiveSpec(
                requirement_id="obj_ppt",
                metric_id="protein_per_time",
                direction="maximize",
                priority_rank=1,
                evidence_status="validated_predictor",
                source_utterance="Maximize protein_per_time",
            )
        ]
    )
    b_gam, rep_gam, man_gam = run_design_pipeline(bad_gam_intent, seed_dir=SEED_DIR)
    assert b_gam == []
    assert rep_gam.outcome_status == "UNCOMPARABLE_MISSING_APPROVED_COMPONENT"
    assert "EP3551758B1" in (rep_gam.uncomparable_reason or "")
    assert (
        man_gam.optimization_mode_audit.method_fto_boundary_checks[
            "LEIBNIZ_HANNOVER_US11848074B2_EP3551758B1_BOOSTED_GAM_PROTEIN_PER_TIME"
        ]["status"]
        == "BLOCK"
    )

    # CureVac >=0.65 C-optimizable conversion against wildtype_cds_reference blocks offending candidate in commercial_clean
    short_prot = "M" + ("FLS" * 10)
    it_short = sess.create_initial_intent(
        turn_id="t2",
        user_utterance="Check CureVac C-conversion ceiling",
        protein_sequence=short_prot,
        compliance_track="commercial_clean",
        uridine_chemistry="unmodified_u",
    )
    hi_c_cand = ConstructBundle.create(
        bundle_id="bnd_curevac_high_c",
        parent_bundle_id=None,
        derivation_action="de_novo_design",
        target_stoichiometry={"mol_primary": 1.0},
        molecules=[
            mol0.model_copy_validated(
                effective_translation_target=it_short.protein_targets[0],
                sequence=mol0.get_segment_sequence("five_prime_utr") + hi_c_cds + stop_s + clean_u3 + pa_s,
                segments=[
                    SegmentSpan(segment_type="five_prime_utr", start=0, end=len(mol0.get_segment_sequence("five_prime_utr")), component_id="u5", origin_source="clean", license_track="commercial_clean", action_taken="selected_from_library"),
                    SegmentSpan(segment_type="cds", start=len(mol0.get_segment_sequence("five_prime_utr")), end=len(mol0.get_segment_sequence("five_prime_utr")) + len(hi_c_cds), component_id=None, origin_source="clean", license_track="commercial_clean", action_taken="generated_novo"),
                    SegmentSpan(segment_type="stop_cassette", start=len(mol0.get_segment_sequence("five_prime_utr")) + len(hi_c_cds), end=len(mol0.get_segment_sequence("five_prime_utr")) + len(hi_c_cds) + len(stop_s), component_id=None, origin_source="clean", license_track="commercial_clean", action_taken="locked_verbatim"),
                    SegmentSpan(segment_type="three_prime_utr", start=len(mol0.get_segment_sequence("five_prime_utr")) + len(hi_c_cds) + len(stop_s), end=len(mol0.get_segment_sequence("five_prime_utr")) + len(hi_c_cds) + len(stop_s) + len(clean_u3), component_id="u3", origin_source="clean", license_track="commercial_clean", action_taken="selected_from_library"),
                    SegmentSpan(segment_type="poly_a", start=len(mol0.get_segment_sequence("five_prime_utr")) + len(hi_c_cds) + len(stop_s) + len(clean_u3), end=len(mol0.get_segment_sequence("five_prime_utr")) + len(hi_c_cds) + len(stop_s) + len(clean_u3) + len(pa_s), component_id="pa", origin_source="clean", license_track="commercial_clean", action_taken="locked_verbatim"),
                ],
                annotations=[],
            )
        ],
    )
    b_cv, rep_cv, man_cv = run_design_pipeline(
        it_short.model_copy_validated(
            runtime_policy=it_short.runtime_policy.model_copy_validated(
                allow_cached_candidates=True, enabled_generators=[], enabled_evaluators=["surrogate_v1_clean"]
            )
        ),
        cached_candidates=[hi_c_cand],
        wildtype_cds_reference=wt_cds,
        seed_dir=SEED_DIR,
    )
    assert b_cv == []
    assert rep_cv.outcome_status == "COMPLETED_ZERO_QUALIFIED"
    assert man_cv.optimization_mode_audit.input_origin_type == "wildtype_mrna_reference"
    assert (man_cv.optimization_mode_audit.wildtype_c_optimizable_replacement_fraction or 0.0) >= 0.99
    assert (
        man_cv.optimization_mode_audit.method_fto_boundary_checks[
            "CUREVAC_US10898584B2_FAMILY_SYNONYMOUS_CYTOSINE_THRESHOLD"
        ]["status"]
        == "BLOCK"
    )

    # 7. V9 13-Parquet wet-lab inventory & Su 2025 internal-evaluation-only export block
    inv_doc = json.loads((SEED_DIR / "v9_wetlab_benchmarks_inventory.json").read_text(encoding="utf-8"))
    assert len(inv_doc["benchmarks"]) == 13
    su_entry = next(b for b in inv_doc["benchmarks"] if b["filename"] == "su2025_elife97682_utr_variants_v1.parquet")
    assert su_entry["external_redistribution_allowed"] is False
    assert su_entry["governance_scope"] == "internal_evaluation_only"

    adapter = ExternalEvidenceAdapterV2(license_track="commercial_clean")
    with pytest.raises(CleanRoomPolicyError, match="su2025|HGMD|internal"):
        adapter.guard_export_to_commercial_bundle(
            {
                "source_name": "su2025_elife97682_utr_variants_v1.parquet",
                "license_spdx": "CC-BY-4.0",
                "license_tier": "commercial_clean",
                "request_url": "https://elifesciences.org/articles/97682",
                "payload_sha256": su_entry["sha256"],
            },
            purpose="commercial_redistribution_export",
        )



"""M1.5 TDD Contract Tests:
1. 7 commercial-clean rules (`rules_commercial_v1.json`) + `RC-002-structrfm-v0.0.9.yaml` + sandbox counter-evidence replay.
2. Axis-routed `CommercialCleanSurrogateV1` (5'UTR MRL + CDS Stability + Saluki Consensus Half-Life, zero ViennaRNA, zero 50/50 dilution).
3. `StructRFMNativeScorer` 4 engineering gotchas (base checkpoint guard, `USE_SPACED=False` assertion, >510 nt sliding-window stitch, chemistry-specific `U/moU/mpU` probes).
4. Pure-rerank 3-arm anti-cheat replay (`"本次未改善生成器"`) + equal-budget paired-seed inner-loop switch ablation in `commercial_clean`.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from mrnax.constraints.registry import default_m1_constraints
from mrnax.engine.ablation_harness import (
    run_fixed_pool_scorer_replay,
    run_paired_seed_surrogate_switch_ablation,
)
from mrnax.engine.pareto_director import run_design_pipeline
from mrnax.engine.surrogates import (
    CommercialCleanSurrogateV1,
    StructRFMNativeScorer,
    prep_structrfm_unspaced_rna,
    stitch_sliding_window_embeddings,
    validate_structrfm_checkpoint_path,
    validate_structrfm_tokenizer,
)
from mrnax.rules.models import RuleRecord, load_replacement_card_yaml
from mrnax.rules.tribunal import RuleRegistry
from mrnax.schemas.construct_bundle import ProteinTargetSpec
from mrnax.schemas.design_intent import DesignIntent, LockedRegionSpec, ObjectiveSpec, RuntimePolicySpec

REPO_ROOT = Path(__file__).resolve().parents[2]
SEED_DIR = REPO_ROOT / "registry_seed"
RC_DIR = REPO_ROOT / "docs" / "replacement_cards"


def test_m15_commercial_rules_v1_and_rc002_and_sandbox_replay(tmp_path: Path) -> None:
    comm_rules_path = SEED_DIR / "rules_commercial_v1.json"
    orig_bytes = comm_rules_path.read_bytes()
    doc = json.loads(orig_bytes.decode("utf-8"))
    assert doc["registry_track"] == "commercial_clean"
    rules = [RuleRecord.model_validate(r) for r in doc["rules"]]
    assert len(rules) >= 7

    by_id = {r.rule_id: r for r in rules}
    assert by_id["RUL-CC-5UTR-GC-TIS-007"].status == "verified_rule"
    assert len({o.underlying_dataset_id for o in by_id["RUL-CC-5UTR-GC-TIS-007"].linked_observations}) == 2

    # Simulated counter-evidence in isolated sandbox MUST NOT touch production file on disk!
    prod_reg = RuleRegistry(comm_rules_path, track="commercial_clean")
    sandbox_reg = prod_reg.fork_isolated_copy(tmp_path / "sandbox_rules_commercial.json")
    downgraded = sandbox_reg.apply_counter_evidence(
        rule_id="RUL-CC-5UTR-GC-TIS-007",
        reason="Simulated contradictory third-lab replication for isolated replay test",
        new_status="disproven_blacklist",
        is_simulated=True,
    )
    assert downgraded.status == "disproven_blacklist"
    assert comm_rules_path.read_bytes() == orig_bytes

    # Validate RC-002 replacement card
    rc002 = load_replacement_card_yaml(RC_DIR / "RC-002-structrfm-v0.0.9.yaml")
    assert rc002.card_id == "RC-002-STRUCTRFM-0.0.9-MIT"
    assert rc002.current_license_track == "commercial_clean"
    assert len(rc002.acceptance_metrics) == 5


def test_m15_axis_routed_commercial_surrogate_predictions() -> None:
    surrogate = CommercialCleanSurrogateV1.from_seed_dir(SEED_DIR)

    # Axis 1: Clean Kozak/low-tail-GC 5'UTR vs out-of-frame uAUG + high-GC-tail 5'UTR
    clean_u5 = "GGGAAAUAAAGAGAGAAAAGAAGAGUAAGAAGAAAUAUAAGAGCCACC"
    bad_u5 = "GGGAAAUAAGAUGGGAGAUGAUGGGGCGCGCGCGCGCGCGGGGCCACC"
    mrl_scores = surrogate.score_5utr_mrl([clean_u5, bad_u5])
    assert float(mrl_scores[0]) > float(mrl_scores[1])

    # Axis 2 & Axis 3: Optimal human CDS + low-ARE 3'UTR vs rare-codon AU-rich CDS + high-ARE 3'UTR
    good_cds = "AUG" + ("CUGGAGGAGAAGCUGGGC" * 20)
    poor_cds = "AUG" + ("UUAGAAUAAAAAUUAGGA" * 20)
    stab_scores = surrogate.score_cds_stability([good_cds, poor_cds])
    assert float(stab_scores[0]) > float(stab_scores[1])

    hl_good = surrogate.score_consensus_halflife(clean_u5, good_cds, "GCUGCCUUCUGCGGGGCUUGCCUUCUGGCC")
    hl_poor = surrogate.score_consensus_halflife(bad_u5, poor_cds, "AUUUAAUUUAAUUUAAUUUAAUUUAUUUUU")
    assert hl_good > hl_poor


def test_m15_structrfm_four_gotcha_contracts() -> None:
    # Gotcha 1: Reject wrong splice-site checkpoint directory
    with pytest.raises(ValueError, match="structRFM_checkpoint"):
        validate_structrfm_checkpoint_path(Path("/tmp/checkpoints/splice_site_prediction/structRFM_for_SPL"))
    validate_structrfm_checkpoint_path(Path("/tmp/checkpoints/structRFM_checkpoint"))

    # Gotcha 2: Unspaced RNA string + tokenizer [UNK] / length trap guard
    assert prep_structrfm_unspaced_rna("atgc") == "AUGC"
    assert " " not in prep_structrfm_unspaced_rna("AUGCAUGC")

    class _BadSpacedTokenizer:
        unk_token_id = 0

        def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
            return [0] if " " not in text else [5, 6, 8, 7]

    class _GoodUnspacedTokenizer:
        unk_token_id = 0

        def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
            if " " in text:
                return [5, 0, 6, 0, 8, 0, 7]
            mp = {"A": 5, "U": 6, "C": 7, "G": 8}
            return [mp.get(ch, 0) for ch in text]

    with pytest.raises(RuntimeError, match="USE_SPACED"):
        validate_structrfm_tokenizer(_BadSpacedTokenizer())
    validate_structrfm_tokenizer(_GoodUnspacedTokenizer())

    # Gotcha 3: >510 nt long-transcript sliding-window stitching (win=510, stride=384)
    seq_720 = "AUG" + ("CUGGGC" * 119) + "UAA"  # 720 nt > 510
    assert len(seq_720) == 720
    calls: list[int] = []

    def _mock_window_encoder(subseq: str) -> np.ndarray:
        assert len(subseq) <= 510
        assert " " not in subseq
        calls.append(len(subseq))
        return np.ones((len(subseq), 768), dtype=np.float32) * 0.25

    stitched = stitch_sliding_window_embeddings(seq_720, _mock_window_encoder, win=510, stride=384)
    assert stitched.shape == (720, 768)
    assert len(calls) >= 2
    assert np.allclose(stitched, 0.25)

    # Gotcha 4: Chemistry-specific SHAPE-MaP probe routing (U vs moU vs mpU)
    probe_scorer = StructRFMNativeScorer.probes_only_from_seed_dir(SEED_DIR)
    rng = np.random.default_rng(123)
    fake_hidden = rng.normal(0.0, 0.5, size=(120, 768)).astype(np.float32)
    pred_u = probe_scorer.predict_shapemap_from_hidden(fake_hidden, uridine_chemistry="unmodified_u")
    pred_m1psi = probe_scorer.predict_shapemap_from_hidden(fake_hidden, uridine_chemistry="m1psi")
    pred_mou = probe_scorer.predict_shapemap_from_hidden(fake_hidden, uridine_chemistry="5mou")
    assert pred_u.shape == (120,)
    assert pred_m1psi.shape == (120,)
    assert pred_mou.shape == (120,)
    # Must NOT use identical weights across U and m1Psi, and must fail closed on unsupported chemistry!
    assert not np.allclose(pred_u, pred_m1psi)
    with pytest.raises(ValueError, match="chemistry"):
        probe_scorer.predict_shapemap_from_hidden(fake_hidden, uridine_chemistry="pseudouridine")


def test_m15_commercial_clean_pipeline_3arm_replay_and_paired_switch() -> None:
    snap = json.loads((SEED_DIR / "egfp_reporter_snapshot.json").read_text(encoding="utf-8"))
    egfp_prot = snap["mammalian_reporter_egfp"]["protein_sequence"]

    intent_cc = DesignIntent(
        intent_id="int_m15_cc",
        project_id="proj_m15_cc",
        intent_version=1,
        created_from_turn_ids=["t1"],
        entry_mode="protein_only",
        compliance_track="commercial_clean",
        uridine_chemistry="m1psi",
        status="ready",
        protein_targets=[
            ProteinTargetSpec(
                protein_sequence=egfp_prot,
                transl_table=1,
                accession_or_source="GenBank:U55762.1",
                confirmation_status="confirmed",
            )
        ],
        locked_scopes={
            "protein_sequence": LockedRegionSpec(
                scope="protein_sequence",
                locked_value=egfp_prot,
                reason="100% AA lock",
                locked_at_turn_id="t1",
            )
        },
        mutable_scopes=["five_prime_utr", "cds", "three_prime_utr"],
        constraints=default_m1_constraints(restriction_enzymes=["BspQI"]),
        objectives=[
            ObjectiveSpec(
                requirement_id="obj_utr5_mrl_cc",
                metric_id="utr5_mrl_pred",
                direction="maximize",
                priority_rank=1,
                evidence_status="validated_predictor",
                source_utterance="Maximize 5'UTR MRL via commercial_clean surrogate",
            ),
            ObjectiveSpec(
                requirement_id="obj_cds_stab_cc",
                metric_id="cds_stability_pred",
                direction="maximize",
                priority_rank=2,
                evidence_status="validated_predictor",
                source_utterance="Maximize CDS codon stability",
            ),
            ObjectiveSpec(
                requirement_id="obj_tx_hl_cc",
                metric_id="consensus_halflife_pred",
                direction="maximize",
                priority_rank=2,
                evidence_status="validated_predictor",
                source_utterance="Maximize full-transcript consensus half-life",
            ),
        ],
        runtime_policy=RuntimePolicySpec(
            enabled_generators=["stage0_random", "dnachisel_cds", "pareto_nsga2"],
            enabled_evaluators=["surrogate_v1_clean"],
            random_seed=42,
        ),
    )

    with patch(
        "mrnax.engine.pareto_director.evaluate_molecule_thermodynamics",
        side_effect=AssertionError("ViennaRNA must NEVER be called in commercial_clean!"),
    ), patch(
        "mrnax.engine.pareto_director.compute_cds_sliding_window_mfe",
        side_effect=AssertionError("ViennaRNA must NEVER be called in commercial_clean!"),
    ):
        bundles, rep, manifest = run_design_pipeline(intent_cc, seed_dir=SEED_DIR)
    assert rep.outcome_status == "COMPLETED_WITH_CANDIDATES"
    assert len(bundles) >= 4
    assert all(c.license_track == "commercial_clean" for c in manifest.invoked_components)

    # Pure-rerank 3-group anti-cheat replay on unguided pool
    surrogate = CommercialCleanSurrogateV1.from_seed_dir(SEED_DIR)
    unguided_intent = intent_cc.model_copy_validated(
        objectives=[],
        runtime_policy=intent_cc.runtime_policy.model_copy_validated(enabled_evaluators=[]),
    )
    pool, base_rep, _ = run_design_pipeline(unguided_intent, seed_dir=SEED_DIR)
    assert len(pool) >= 4
    replay = run_fixed_pool_scorer_replay(
        candidate_pool=pool,
        base_evaluations=base_rep.candidate_evaluations,
        scorer_name="surrogate_v1_composite",
        scorer_fn=surrogate.score_bundle_composite,
    )
    assert replay.baseline_ordered_ids == replay.display_only_ordered_ids
    assert replay.honesty_banner == "本次未改善生成器"

    # Equal-budget paired-seed inner-loop switch ablation (surrogate OFF vs surrogate ON)
    switch_rows = run_paired_seed_surrogate_switch_ablation(
        intent_cc, seeds=[42, 101], seed_dir=SEED_DIR
    )
    assert len(switch_rows) == 2
    for row in switch_rows:
        assert row.guided_best_utr5_mrl >= row.unguided_best_utr5_mrl
        assert row.guided_best_halflife >= row.unguided_best_halflife

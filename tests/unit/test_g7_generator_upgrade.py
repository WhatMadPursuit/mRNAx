"""Gate G7 unit tests: True inner-loop commercial_clean generator upgrade & equal-budget paired attribution."""

from __future__ import annotations

import json
import random
from pathlib import Path
import pytest

from mrnax.cloud.gateway import _build_preset_reference_r
from mrnax.engine import pareto_director
from mrnax.engine.ablation_harness import run_paired_seed_surrogate_switch_ablation
from mrnax.engine.pareto_director import run_design_pipeline
from mrnax.engine.stage0_baseline import (
    _load_bicodon_atlas,
    generate_random_synonymous_cds,
    propose_non_partitioned_bicodon_cds,
)
from mrnax.harness.state_machine import ProjectHarnessSession
from mrnax.schemas.design_intent import ConstraintSpec, DesignIntent

ROOT = Path(__file__).resolve().parents[2]
SEED = ROOT / "registry_seed"


def test_non_partitioned_bicodon_operator_improves_atlas_score_and_preserves_protein_and_gates() -> None:
    b2 = json.loads((ROOT / "tests/golden/m2_benchmark2_dll3_cd3_acceptance_fixture_v1.json").read_text())
    raw_intent = DesignIntent.model_validate(b2["design_intent"])
    assert raw_intent.cell_expression_control is not None
    intent = raw_intent.model_copy_validated(
        cell_expression_control=raw_intent.cell_expression_control.model_copy_validated(
            protected_tissues=["t_cell", "myeloid_monocyte_macrophage"]
        )
    )
    prot = intent.protein_targets[0].protein_sequence
    rng = random.Random(42)
    raw_cds = generate_random_synonymous_cds(prot, rng, chemistry="m1psi")
    upgraded = propose_non_partitioned_bicodon_cds(
        raw_cds,
        prot,
        random.Random(42),
        chemistry="m1psi",
        constraints=[
            ConstraintSpec(
                requirement_id="enz",
                constraint_kind="type_iis_restriction_free",
                target_scopes=["full_transcript"],
                params={"enzymes": ["BspQI", "BsaI"]},
                source_type="user_explicit",
                source_locator="turn:t1",
            )
        ],
        utr5_tail="GGGAAAUUAAGAGAGAAAAGAAGAGUAAGAAGAAAUAUAAGAGCCACC",
        stop_head="UGAUAAUAG",
    )
    assert len(upgraded) == len(raw_cds)
    assert "UUUU" not in upgraded
    assert "GCUCUUC" not in upgraded and "GAAGAGC" not in upgraded
    assert "GGUCUC" not in upgraded and "GAGACC" not in upgraded

    csc_map, bic_map = _load_bicodon_atlas()
    raw_codons = [raw_cds[i : i + 3] for i in range(0, len(raw_cds), 3)]
    up_codons = [upgraded[i : i + 3] for i in range(0, len(upgraded), 3)]
    raw_bic = sum(bic_map.get(f"{raw_codons[i]}-{raw_codons[i + 1]}", 0.0) for i in range(len(raw_codons) - 1))
    up_bic = sum(bic_map.get(f"{up_codons[i]}-{up_codons[i + 1]}", 0.0) for i in range(len(up_codons) - 1))
    assert up_bic > raw_bic


def test_two_chain_guided_generator_beats_pure_rerank_under_equal_budget_and_report_valid() -> None:
    b2 = json.loads((ROOT / "tests/golden/m2_benchmark2_dll3_cd3_acceptance_fixture_v1.json").read_text())
    raw_intent = DesignIntent.model_validate(b2["design_intent"])
    assert raw_intent.cell_expression_control is not None
    intent = raw_intent.model_copy_validated(
        cell_expression_control=raw_intent.cell_expression_control.model_copy_validated(
            protected_tissues=["t_cell", "myeloid_monocyte_macrophage"]
        )
    )
    pol = intent.runtime_policy.model_copy_validated(
        random_seed=42,
        enabled_generators=["stage0_random", "dnachisel_cds", "pareto_nsga2"],
        enabled_evaluators=["surrogate_v1_clean"],
        max_evaluations=8,
    )
    run_intent = intent.model_copy_validated(runtime_policy=pol)

    b_rerank, rep_rerank, man_rerank = run_design_pipeline(
        run_intent, seed_dir=SEED, legacy_pure_rerank_mode=True
    )
    b_guided, rep_guided, man_guided = run_design_pipeline(
        run_intent, seed_dir=SEED, legacy_pure_rerank_mode=False
    )

    assert man_rerank.honesty_banner == "本次未改善生成器"
    assert man_guided.honesty_banner is None
    assert sum(c.attempt_count for c in man_rerank.invoked_components if c.role == "generator") == sum(
        c.attempt_count for c in man_guided.invoked_components if c.role == "generator"
    )
    assert len(b_guided[0].molecules) == 2

    # Equal-work switch harness continues to hold on 2-chain intents
    switch_rows = run_paired_seed_surrogate_switch_ablation(run_intent, [42, 101], seed_dir=SEED)
    assert all(r.unguided_trial_count == r.guided_trial_count for r in switch_rows)

    report_path = ROOT / "artifacts" / "m2_generator_upgrade" / "paired_ablation_report.json"
    assert report_path.is_file()
    doc = json.loads(report_path.read_text(encoding="utf-8"))
    for key in ("single_chain_egfp", "two_chain_dll3_x_cd3"):
        summary = doc["benchmarks"][key]["summary_means"]
        assert doc["benchmarks"][key]["n_seeds"] >= 5
        assert summary["mean_paired_delta_halflife_arm3_vs_arm2"] > 0.0
        assert summary["mean_paired_delta_cds_stability_arm3_vs_arm2"] > 0.0
        assert summary["mean_paired_delta_bicodon_index_arm3_vs_arm2"] > 0.0


def test_scoped_restriction_on_locked_utr3_and_direct_cds_operator_call_equality(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ref = _build_preset_reference_r(SEED)
    mol0 = ref.molecules[0]
    u1_rows = json.loads((SEED / "utr_components_v1.json").read_text(encoding="utf-8"))["three_prime_utrs"]
    actb_row = next(r for r in u1_rows if r["component_id"] == "utr3_gapdh_NM_002046_7")
    u3_bsai = actb_row["sequence"]
    u5_s = mol0.get_segment_sequence("five_prime_utr")
    cds_s = mol0.get_segment_sequence("cds")
    stop_s = mol0.get_segment_sequence("stop_cassette")
    pa_s = mol0.get_segment_sequence("poly_a")
    new_seq = u5_s + cds_s + stop_s + u3_bsai + pa_s
    c3_start = len(u5_s) + len(cds_s) + len(stop_s)
    c3_end = c3_start + len(u3_bsai)
    new_segs = []
    for seg in mol0.segments:
        if seg.segment_type == "three_prime_utr":
            new_segs.append(
                seg.model_copy_validated(
                    start=c3_start,
                    end=c3_end,
                    component_id=actb_row["component_id"],
                    origin_source=actb_row["origin_source"],
                    license_track="commercial_clean",
                )
            )
        elif seg.segment_type == "poly_a":
            new_segs.append(seg.model_copy_validated(start=c3_end, end=len(new_seq)))
        else:
            new_segs.append(seg)
    from mrnax.schemas.construct_bundle import ConstructBundle

    ref_bsai = ConstructBundle.create(
        bundle_id="bnd_ref_actb_bsai",
        parent_bundle_id=None,
        derivation_action="reference_import",
        target_stoichiometry={mol0.molecule_id: 1.0},
        molecules=[mol0.model_copy_validated(sequence=new_seq, segments=new_segs)],
    )
    sess = ProjectHarnessSession(project_id="g7_scoped", store_dir=tmp_path / "scoped", seed_dir=SEED)
    base_it = sess.create_initial_intent(
        turn_id="t1",
        user_utterance="EGFP with locked ACTB 3'UTR containing BsaI",
        protein_sequence=mol0.effective_translation_target.protein_sequence,
        reference_bundle=ref_bsai,
        compliance_track="commercial_clean",
        uridine_chemistry="m1psi",
        locked_regions={"three_prime_utr": u3_bsai},
        restriction_enzymes=["BspQI"],
    )
    # 1. CDS-only BsaI constraint must NOT crash or reject locked 3'UTR containing BsaI
    cds_only_bsai = [
        c.model_copy_validated(target_scopes=frozenset({"cds"}), params={"enzymes": ["BsaI"], "forbidden_motifs": []})
        if c.constraint_kind == "type_iis_restriction_free"
        else c
        for c in base_it.constraints
    ]
    cands_cds, rep_cds, _ = run_design_pipeline(
        base_it.model_copy_validated(constraints=cds_only_bsai), reference_bundle=ref_bsai, seed_dir=SEED
    )
    assert rep_cds.outcome_status == "COMPLETED_WITH_CANDIDATES"
    assert len(cands_cds) >= 8

    # 2. Full-transcript BsaI constraint on locked 3'UTR containing BsaI must return COMPLETED_ZERO_QUALIFIED cleanly
    full_tx_bsai = [
        c.model_copy_validated(
            target_scopes=frozenset({"full_transcript"}), params={"enzymes": ["BsaI"], "forbidden_motifs": []}
        )
        if c.constraint_kind == "type_iis_restriction_free"
        else c
        for c in base_it.constraints
    ]
    cands_ft, rep_ft, _ = run_design_pipeline(
        base_it.model_copy_validated(constraints=full_tx_bsai), reference_bundle=ref_bsai, seed_dir=SEED
    )
    assert cands_ft == []
    assert rep_ft.outcome_status == "COMPLETED_ZERO_QUALIFIED"
    assert len(rep_ft.failed_diagnostics) >= 1

    # 3. Direct CDS operator call instrumentation: manifest_attempts == direct_calls AND legacy == guided AND OFF == ON
    orig_fns = {
        name: getattr(pareto_director, name)
        for name in (
            "generate_random_synonymous_cds",
            "generate_dnachisel_cds",
            "mutate_cds_region_synonymously",
            "propose_non_partitioned_bicodon_cds",
        )
    }
    counts: dict[str, int] = {}
    for fn_name, orig_fn in orig_fns.items():
        def _wrap(fn: object = orig_fn, k: str = fn_name):  # type: ignore[no-untyped-def]
            def _inner(*a, **kw):  # type: ignore[no-untyped-def]
                counts[k] = counts.get(k, 0) + 1
                return fn(*a, **kw)  # type: ignore[misc]
            return _inner
        monkeypatch.setattr(pareto_director, fn_name, _wrap())

    def _measure(it: DesignIntent, *, legacy: bool = False) -> tuple[int, int, int]:
        counts.clear()
        _, _, man = run_design_pipeline(it, seed_dir=SEED, legacy_pure_rerank_mode=legacy)
        rep_att = sum(c.attempt_count for c in man.invoked_components if c.role == "generator")
        direct = sum(counts.values())
        return rep_att, direct, counts.get("generate_dnachisel_cds", 0)

    # Single-chain legacy vs guided
    it_1 = sess.create_initial_intent(
        turn_id="t2",
        user_utterance="1-chain parity",
        protein_sequence=mol0.effective_translation_target.protein_sequence,
        compliance_track="commercial_clean",
        uridine_chemistry="m1psi",
    )
    m_leg1, d_leg1, _ = _measure(it_1, legacy=True)
    m_gui1, d_gui1, _ = _measure(it_1, legacy=False)
    assert m_leg1 == d_leg1 == m_gui1 == d_gui1 == 38

    # Two-chain stage0, legacy, guided, and surrogate OFF vs ON
    b2 = json.loads((ROOT / "tests/golden/m2_benchmark2_dll3_cd3_acceptance_fixture_v1.json").read_text())
    it_2_raw = DesignIntent.model_validate(b2["design_intent"])
    assert it_2_raw.cell_expression_control is not None
    it_2 = it_2_raw.model_copy_validated(
        cell_expression_control=it_2_raw.cell_expression_control.model_copy_validated(
            protected_tissues=["t_cell", "myeloid_monocyte_macrophage"]
        ),
        runtime_policy=it_2_raw.runtime_policy.model_copy_validated(
            random_seed=42,
            enabled_generators=["stage0_random", "dnachisel_cds", "pareto_nsga2"],
            enabled_evaluators=["surrogate_v1_clean"],
            max_evaluations=10,
        ),
    )
    it_2_s0 = it_2.model_copy_validated(
        runtime_policy=it_2.runtime_policy.model_copy_validated(enabled_generators=["stage0_random", "dnachisel_cds"])
    )
    m_s0, d_s0, _ = _measure(it_2_s0)
    m_leg2, d_leg2, _ = _measure(it_2, legacy=True)
    m_gui2, d_gui2, dc_on = _measure(it_2, legacy=False)
    assert m_s0 == d_s0 == 4
    assert m_leg2 == d_leg2 == m_gui2 == d_gui2 == 72

    it_2_off = it_2.model_copy_validated(
        runtime_policy=it_2.runtime_policy.model_copy_validated(enabled_evaluators=[])
    )
    m_off2, d_off2, dc_off = _measure(it_2_off, legacy=False)
    assert m_off2 == d_off2 == m_gui2 == d_gui2 == 72
    assert dc_off == dc_on == 2

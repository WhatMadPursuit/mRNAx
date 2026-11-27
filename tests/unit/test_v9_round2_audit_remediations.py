"""TDD Regression Suite for V9 Round-1 `gpt-6.2-astra` (`ultra`) Adversarial Audit Findings:
- Critical 1..5, Important 1..7, Minor 1..2.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SEED_DIR = REPO_ROOT / "registry_seed"


# ============================================================================
# Batch A: P1 Minor 1 + Important 4 (Darwin OpenMP isolation) + Important 5 + Critical 5
# ============================================================================


def test_minor1_surrogate_empty_3utr_when_head_absent_and_exact_numerical_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Minor 1: score_3utr_stability([]) returns empty float64 even if _utr3_bundle is None, and all 4 heads match reference 0.0 diff."""
    from mrnax.engine.surrogates import (
        CommercialCleanSurrogateV1,
        extract_3utr_features,
        extract_5utr_features,
        extract_cds_features,
        extract_full_tx_halflife_features,
    )

    surrogate = CommercialCleanSurrogateV1.from_seed_dir(SEED_DIR)
    u5_seqs = [
        "GGGAAAUAAAGAGAGAAAAGAAGAGUAAGAAGAAAUAUAAGAGCCACC",
        "GGGAAAUAAGAUGGGAGAUGAUGGGGCGCGCGCGCGCGCGGGGCCACC",
    ]
    cds_seqs = [
        "AUG" + ("CUGGAGGAGAAGCUGGGC" * 18),
        "AUG" + ("GCCGUGAAGCUGAUCGAC" * 18),
    ]
    u3_seqs = [
        "GCUGCCUUCUGCGGGGCUUGCCUUCUGGCC",
        "UGCCUUCUGCGGGGCUUGCCUUCUGGCCAU",
    ]
    triplets = list(zip(u5_seqs, cds_seqs, u3_seqs, strict=True))

    # Exact parity vs per-column reference feature matrix construction
    cols_u5 = surrogate._utr5_bundle["features"]
    ref_X_u5 = np.array([[extract_5utr_features(s)[c] for c in cols_u5] for s in u5_seqs], dtype=np.float32)
    assert float(np.max(np.abs(surrogate.score_5utr_mrl(u5_seqs) - surrogate._predict_bundle(surrogate._utr5_bundle, ref_X_u5)))) == 0.0

    cols_cds = surrogate._cds_bundle["features"]
    ref_X_cds = np.array([[extract_cds_features(s)[c] for c in cols_cds] for s in cds_seqs], dtype=np.float32)
    assert float(np.max(np.abs(surrogate.score_cds_stability(cds_seqs) - surrogate._predict_bundle(surrogate._cds_bundle, ref_X_cds)))) == 0.0

    cols_hl = surrogate._hl_bundle["features"]
    ref_X_hl = np.array(
        [[extract_full_tx_halflife_features(u5, c, u3)[col] for col in cols_hl] for u5, c, u3 in triplets],
        dtype=np.float32,
    )
    assert float(np.max(np.abs(surrogate.score_consensus_halflife_batch(triplets) - surrogate._predict_bundle(surrogate._hl_bundle, ref_X_hl)))) == 0.0

    assert surrogate._utr3_bundle is not None
    cols_u3 = surrogate._utr3_bundle["features"]
    ref_X_u3 = np.array([[extract_3utr_features(s)[c] for c in cols_u3] for s in u3_seqs], dtype=np.float32)
    assert float(np.max(np.abs(surrogate.score_3utr_stability(u3_seqs) - surrogate._predict_bundle(surrogate._utr3_bundle, ref_X_u3)))) == 0.0

    # When _utr3_bundle is absent, empty batch [] must return shape (0,) float64 without raising; non-empty must still raise
    monkeypatch.setattr(surrogate, "_utr3_bundle", None)
    empty_res = surrogate.score_3utr_stability([])
    assert empty_res.shape == (0,) and empty_res.dtype == np.float64
    with pytest.raises(RuntimeError, match="utr3_stability_pure_seq_lgbm"):
        surrogate.score_3utr_stability(["GCUGCCUUCUGCGGGGCUUGCCUUCUGGCC"])


@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin dyld libomp isolation test")
def test_important4_darwin_zero_init_preload_and_single_libomp_both_import_orders() -> None:
    """Important 4: `import mrnax` must not set OMP_NUM_THREADS or load libomp; both mrnax->torch & torch->mrnax keep <=1 libomp."""
    probe_code_init = (
        "import ctypes, os, mrnax\n"
        "assert 'OMP_NUM_THREADS' not in os.environ, os.environ.get('OMP_NUM_THREADS')\n"
        "assert 'lightgbm' not in __import__('sys').modules\n"
        "dyld = ctypes.CDLL('/usr/lib/system/libdyld.dylib')\n"
        "dyld._dyld_get_image_name.restype = ctypes.c_char_p\n"
        "omps = [dyld._dyld_get_image_name(i).decode() for i in range(dyld._dyld_image_count()) if 'libomp' in dyld._dyld_get_image_name(i).decode().lower()]\n"
        "assert omps == [], omps\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "OMP_NUM_THREADS"}
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    subprocess.run([sys.executable, "-c", probe_code_init], check=True, env=env)

    for order_stmt in (
        "import mrnax; from mrnax.engine.surrogates import CommercialCleanSurrogateV1; import torch",
        "import torch; import mrnax; from mrnax.engine.surrogates import CommercialCleanSurrogateV1",
    ):
        probe_both = (
            f"{order_stmt}\n"
            "import ctypes, pathlib\n"
            f"s = CommercialCleanSurrogateV1.from_seed_dir(pathlib.Path({str(SEED_DIR)!r}))\n"
            "out = s.score_5utr_mrl(['GGGAAAUAAAGAGAGAAAAGAAGAGUAAGAAGAAAUAUAAGAGCCACC'])\n"
            "assert out.shape == (1,)\n"
            "dyld = ctypes.CDLL('/usr/lib/system/libdyld.dylib')\n"
            "dyld._dyld_get_image_name.restype = ctypes.c_char_p\n"
            "omps = [dyld._dyld_get_image_name(i).decode() for i in range(dyld._dyld_image_count()) if 'libomp' in dyld._dyld_get_image_name(i).decode().lower()]\n"
            "assert len(omps) == 1, omps\n"
        )
        subprocess.run([sys.executable, "-c", probe_both], check=True, env=env)


def test_important5_rules_012_013_prose_and_component_id_track_resolution() -> None:
    """Important 5: Active rule 012 must not recommend held EternaFold; rule 013 must not claim equivalent/calibrated; relabeled RC-004 fails."""
    from mrnax.rules.models import ModelInterpretationNote
    from mrnax.rules.tribunal import RuleRegistry

    rules_path = SEED_DIR / "rules_commercial_v1.json"
    ledger_doc = json.loads((SEED_DIR / "rule_observation_ledger_v1.json").read_text(encoding="utf-8"))
    reg = RuleRegistry(rules_path, track="commercial_clean")

    active_m1psi = {
        r.rule_id: r
        for r in reg.get_active_rules(chemistry="m1psi", context="m1Psi_IVT_mRNA_design")
    }
    r012 = active_m1psi["RUL-CC-CDS-CODONTRANSFORMER-M1PSI-FILTER-012"]
    assert "eternafold" not in r012.statement.lower() or "commercial_clearance_pending" in r012.statement.lower()
    assert "pair codontransformer proposals with deterministic" in r012.statement.lower()
    assert "(eternafold" not in r012.statement.lower()

    active_hela = {
        r.rule_id: r
        for r in reg.get_active_rules(chemistry="unmodified_u", context="HeLa")
    }
    r013 = active_hela["RUL-CC-UTR-JOINT-SURROGATE-NOISEFLOOR-013"]
    interp013 = r013.model_interpretations[0].summary.lower()
    assert "equivalent pairs" not in interp013
    assert "calibrated false-positive" not in interp013

    # Relabeled RC-004 / EternaFold WITHOUT 'clearance_pending' in source_model_name MUST STILL be rejected!
    relabeled_rc004 = r012.model_copy_validated(
        rule_id="RUL-BAD-RELABELED-RC004",
        model_interpretations=[
            ModelInterpretationNote(
                source_component_id="eternafold_structure_head",
                source_model_name="BSD3_Structure_Scorer_v1.3.1",
                source_model_license_track="commercial_clean",
                method="regression_weight",
                summary="Relabeled held component without clearance_pending substring",
            )
        ],
    )
    with pytest.raises(ValueError, match="eternafold|commercial_clearance_pending"):
        RuleRegistry.validate_rules_integrity(
            [relabeled_rc004], track="commercial_clean", observation_ledger=ledger_doc["claims"]
        )

    relabeled_pin = r012.model_copy_validated(
        rule_id="RUL-BAD-PIN-RC004",
        model_interpretations=[
            ModelInterpretationNote(
                source_model_name="StructureModel [pin: RC-004-eternafold-v1.3.1.yaml]",
                source_model_license_track="commercial_clean",
                method="regression_weight",
                summary="Pin without clearance_pending substring",
            )
        ],
    )
    with pytest.raises(ValueError, match="RC-004|eternafold|commercial_clearance_pending"):
        RuleRegistry.validate_rules_integrity(
            [relabeled_pin], track="commercial_clean", observation_ledger=ledger_doc["claims"]
        )


def test_critical5_su2025_nested_provenance_aliases_and_publication_metadata_separation() -> None:
    """Critical 5: Nested dataset_id/digest/DOI/PMID/boolean restrictions block in filter_and_export_commercial_records; pure article citation metadata remains allowed."""
    from mrnax.federated.external_evidence_adapter import (
        CleanRoomPolicyError,
        ExternalEvidenceAdapterV2,
        sha256_canonical,
    )

    adapter = ExternalEvidenceAdapterV2(license_track="commercial_clean")
    inv_doc = json.loads((SEED_DIR / "v9_wetlab_benchmarks_inventory.json").read_text(encoding="utf-8"))
    su_row = next(b for b in inv_doc["benchmarks"] if b["dataset_id"] == "su2025_elife97682")

    def _epmc_record(summary: dict, query_str: str = "PMCID:PMC8728028") -> dict:
        full_sum = {
            "pmcid": "PMC8728028",
            "is_open_access": True,
            "commercial_clean_article_eligible": True,
            "license": "cc by",
            **summary,
        }
        return {
            "source_name": "EuropePMC",
            "license_spdx": "CC0-1.0_Metadata",
            "license_tier": "commercial_clean",
            "request_url": f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query={query_str}&format=json",
            "summary": full_sum,
            "payload_sha256": sha256_canonical(full_sum),
        }

    # 1. Nested restricted dataset_id, external_redistribution_allowed=False, or internal_evaluation_only MUST block!
    bad_nested_ds = _epmc_record(
        {
            "nested_provenance": {
                "dataset_id": "su2025_elife97682",
                "external_redistribution_allowed": False,
                "governance_scope": "internal_evaluation_only",
            }
        }
    )
    with pytest.raises(CleanRoomPolicyError):
        adapter.filter_and_export_commercial_records([bad_nested_ds])

    # 2. Nested restricted Parquet SHA-256 or Drive ID MUST block!
    bad_nested_sha = _epmc_record({"upstream_artifact_sha256": su_row["sha256"]})
    with pytest.raises(CleanRoomPolicyError):
        adapter.filter_and_export_commercial_records([bad_nested_sha])

    # 3. Variant/coordinate record citing Su 2025 DOI (10.7554/eLife.97682) or PMID (39964837) MUST block!
    bad_doi_variants = _epmc_record(
        {
            "pmcid": "PMC11835390",
            "doi": "10.7554/eLife.97682",
            "pmid": "39964837",
            "utr_variant_coordinates_hg38": ["chr17:7676594:C>T"],
        },
        query_str="PMCID:PMC11835390",
    )
    with pytest.raises(CleanRoomPolicyError):
        adapter.filter_and_export_commercial_records([bad_doi_variants])

    bad_pmid_dataset = _epmc_record(
        {
            "source_pmid": "39964837",
            "benchmark_table": "5072_utr_variant_pairs",
        }
    )
    with pytest.raises(CleanRoomPolicyError):
        adapter.filter_and_export_commercial_records([bad_pmid_dataset])

    # 4. Pure bibliographic open-access publication metadata (no variant coordinates / restricted dataset) is preserved!
    pure_biblio = _epmc_record(
        {
            "pmcid": "PMC11835390",
            "pmid": "39964837",
            "doi": "10.7554/eLife.97682",
            "title": "Massively parallel interrogation of human 5' and 3' UTR variants",
        },
        query_str="PMCID:PMC11835390",
    )
    exported = adapter.filter_and_export_commercial_records([pure_biblio])
    assert len(exported) == 1


# ============================================================================
# Batch B: Critical 3, Critical 4, Important 3, Important 6, Important 7
# ============================================================================


def test_critical3_stored_commercial_report_rejects_stale_snapshot_and_held_rc004_through_export(
    tmp_path: Path,
) -> None:
    """Critical 3: Missing/zero/stale snapshot or held EternaFold ranking metric strips commercial recommendation & blocks export."""
    from mrnax.cloud.gateway import _build_preset_reference_r
    from mrnax.engine.pareto_director import project_report_for_intent
    from mrnax.harness.state_machine import ProjectHarnessSession
    from mrnax.harness.tools import AtomicToolbox
    from mrnax.rules.tribunal import RuleRegistry
    from mrnax.schemas.design_intent import ObjectiveSpec
    from mrnax.schemas.evaluation_report import BundleEvaluation, EvaluationReport, MetricResult

    ref = _build_preset_reference_r(SEED_DIR)
    sess = ProjectHarnessSession(project_id="c3_hist", store_dir=tmp_path / "proj_c3", seed_dir=SEED_DIR)
    intent_cc = sess.create_initial_intent(
        turn_id="t1",
        user_utterance="Commercial historical projection check",
        protein_sequence=ref.molecules[0].effective_translation_target.protein_sequence,
        compliance_track="commercial_clean",
        uridine_chemistry="m1psi",
        use_commercial_surrogate=True,
    )
    sess.register_bundle(ref, originating_intent_id=intent_cc.intent_id)
    current_snap = RuleRegistry(SEED_DIR / "rules_commercial_v1.json", track="commercial_clean").snapshot_version()

    clean_metric = MetricResult(
        metric_id="utr5_mrl_pred",
        value=6.2,
        unit="ribosome_load",
        direction="maximize",
        evaluated_scopes=["five_prime_utr"],
        tool_name="CommercialCleanSurrogateV1",
        tool_version="1.0.0",
        chemistry_evaluation_mode="endogenous_unmodified_proxy",
        participated_in="pareto_selection_or_ranking",
    )

    for bad_snap in (None, "rules_commercial_clean@missing", "rules_commercial_clean@" + ("0" * 64)):
        bad_rep = EvaluationReport(
            report_id="rep_bad_snap",
            execution_id="ex_bad",
            intent_id=intent_cc.intent_id,
            intent_version=intent_cc.intent_version,
            compliance_track="commercial_clean",
            outcome_status="COMPLETED_WITH_CANDIDATES",
            recommended_bundle_ids=[ref.bundle_id],
            candidate_evaluations=[
                BundleEvaluation(
                    bundle_id=ref.bundle_id,
                    version_digest=ref.version_digest,
                    passed_all_hard_gates=True,
                    pareto_rank=1,
                    metrics=[clean_metric],
                )
            ],
            failed_diagnostics=[],
            fto_alerts=[],
            pre_synthesis_advisories=[],
            action_transparency=[],
            rule_snapshot_version=bad_snap,
        )
        proj = project_report_for_intent(
            bad_rep, {ref.bundle_id: ref}, intent_cc, seed_dir=SEED_DIR, disallow_sandbox_override=True
        )
        assert proj.outcome_status == "UNCOMPARABLE_MISSING_APPROVED_COMPONENT"
        assert proj.recommended_bundle_ids == []

    # Stored report with VALID current rule digest, but ranked via held EternaFold (RC-004)!
    ef_intent = intent_cc.model_copy_validated(
        objectives=[
            ObjectiveSpec(
                requirement_id="obj_ef",
                metric_id="eternafold_kozak_unpaired_prob",
                direction="maximize",
                priority_rank=1,
                evidence_status="validated_predictor",
                source_utterance="ef",
            )
        ]
    )
    sess._intents[-1] = ef_intent
    ef_rep = EvaluationReport(
        report_id="rep_ef_held",
        execution_id="ex_ef",
        intent_id=ef_intent.intent_id,
        intent_version=ef_intent.intent_version,
        compliance_track="commercial_clean",
        outcome_status="COMPLETED_WITH_CANDIDATES",
        recommended_bundle_ids=[ref.bundle_id],
        candidate_evaluations=[
            BundleEvaluation(
                bundle_id=ref.bundle_id,
                version_digest=ref.version_digest,
                passed_all_hard_gates=True,
                pareto_rank=1,
                metrics=[
                    MetricResult(
                        metric_id="eternafold_kozak_unpaired_prob",
                        value=0.72,
                        unit="probability",
                        direction="maximize",
                        evaluated_scopes=["five_prime_utr", "cds"],
                        tool_name="EternaFoldBSD3",
                        tool_version="1.3.1",
                        chemistry_evaluation_mode="endogenous_unmodified_proxy",
                        participated_in="pareto_selection_or_ranking",
                    )
                ],
            )
        ],
        failed_diagnostics=[],
        fto_alerts=[],
        pre_synthesis_advisories=[],
        action_transparency=[],
        rule_snapshot_version=current_snap,
    )
    rep_disk = sess.store_dir / "reports" / "rep_ef_held.json"
    raw_bytes = (json.dumps(ef_rep.model_dump(mode="json"), indent=2, sort_keys=True) + "\n").encode("utf-8")
    rep_disk.write_bytes(raw_bytes)
    sess._reports["rep_ef_held"] = ef_rep

    proj_ef = project_report_for_intent(
        ef_rep, {ref.bundle_id: ref}, ef_intent, seed_dir=SEED_DIR, disallow_sandbox_override=True
    )
    assert proj_ef.outcome_status == "UNCOMPARABLE_MISSING_APPROVED_COMPONENT"
    assert proj_ef.recommended_bundle_ids == []

    toolbox = AtomicToolbox(sess)
    with pytest.raises(ValueError, match="blocked|uncomparable|approved|EternaFold"):
        toolbox.export_bundle(ref.bundle_id)
    # Historical report on disk remains untouched!
    assert rep_disk.read_bytes() == raw_bytes


def test_critical4_rc005_spoofed_identity_empty_partner_and_research_max_partner_gate(
    tmp_path: Path,
) -> None:
    """Critical 4: Spoofed verified_release_digests=True rejected; empty partner metrics rejected; research_max requires cleared commercial partner."""
    from mrnax.cloud.gateway import _build_preset_reference_r
    from mrnax.engine.pareto_director import run_design_pipeline
    from mrnax.engine.profiles import check_profile_readiness
    from mrnax.harness.state_machine import ProjectHarnessSession
    from mrnax.schemas.design_intent import ObjectiveSpec, RuntimePolicySpec

    ref = _build_preset_reference_r(SEED_DIR)
    sess = ProjectHarnessSession(project_id="c4_ct", store_dir=tmp_path / "proj_c4", seed_dir=SEED_DIR)
    base_cc = sess.create_initial_intent(
        turn_id="t1",
        user_utterance="Check RC-005 gates",
        protein_sequence=ref.molecules[0].effective_translation_target.protein_sequence,
        compliance_track="commercial_clean",
        uridine_chemistry="m1psi",
    )

    # 1. research_max MUST NOT accept RC-004 or ViennaRNA as the SOLE partner for RC-005!
    for non_cleared_partner in ("eternafold_structure_head", "viennarna_thermo"):
        it_rm_bad_partner = base_cc.model_copy_validated(
            compliance_track="research_max",
            runtime_policy=RuntimePolicySpec(
                enabled_generators=["codontransformer_cds_baseline"],
                enabled_evaluators=[non_cleared_partner],
            ),
        )
        assert check_profile_readiness(it_rm_bad_partner).is_runnable is False

    # 2. Spoofed placeholder claiming `verified_release_digests = True` MUST be rejected in production mode!
    class _SpoofedCT:
        verified_release_digests = True

        def generate_and_score_cds(self, prot: str, **_kw: object) -> tuple[str, float]:
            return ref.molecules[0].get_segment_sequence("cds"), -0.35

        def score_cds_synonymous_loglik(self, seqs: list[str]) -> np.ndarray:
            return np.full(len(seqs), -0.35, dtype=np.float64)

    it_cc_ct = base_cc.model_copy_validated(
        objectives=[
            ObjectiveSpec(
                requirement_id="obj_ct",
                metric_id="codontransformer_syn_loglik",
                direction="maximize",
                priority_rank=1,
                evidence_status="validated_predictor",
                source_utterance="ct",
            )
        ],
        runtime_policy=RuntimePolicySpec(
            enabled_generators=["codontransformer_cds_baseline"],
            enabled_evaluators=["structrfm_rna_foundation"],
        ),
    )

    class _EmptyPartnerScorer:
        def evaluate_molecule(self, _mol: object) -> tuple[list, list]:
            return [], []

    b_spoof, rep_spoof, _ = run_design_pipeline(
        it_cc_ct,
        seed_dir=SEED_DIR,
        codontransformer_model=_SpoofedCT(),
        structrfm_scorer=_EmptyPartnerScorer(),  # type: ignore[arg-type]
        isolated_sandbox_override=False,
    )
    assert b_spoof == []
    assert rep_spoof.outcome_status == "UNCOMPARABLE_MISSING_APPROVED_COMPONENT"

    # 3. Even under isolated_sandbox_override=True, if the partner evaluator returns ZERO metrics, RC-005 MUST NOT recommend!
    b_empty_p, rep_empty_p, man_empty_p = run_design_pipeline(
        it_cc_ct,
        seed_dir=SEED_DIR,
        codontransformer_model=_SpoofedCT(),
        structrfm_scorer=_EmptyPartnerScorer(),  # type: ignore[arg-type]
        isolated_sandbox_override=True,
    )
    assert b_empty_p == []
    assert rep_empty_p.outcome_status == "UNCOMPARABLE_MISSING_APPROVED_COMPONENT"
    assert not any(
        "structRFM" in c.name and c.failure_count == 0 and c.attempt_count > 0
        for c in man_empty_p.invoked_components
    )


def test_important3_and_7_rc005_standalone_uuuc_two_chain_unmodified_framing_and_bigbird_attn(
    tmp_path: Path,
) -> None:
    """Important 3 & 7: standalone UUUC remediation & 2-chain admission gate, unmodified control receipt, 2045 framing, multi-stop, & BigBird attention toggle."""
    from types import SimpleNamespace
    import torch
    from transformers import BigBirdConfig, BigBirdForMaskedLM
    from mrnax.constraints.frameshift_m1psi import find_m1psi_slippery_spans, remediate_m1psi_slippery_runs
    from mrnax.engine.codontransformer_loader import CodonTransformerNativeModel
    from mrnax.engine.pareto_director import run_design_pipeline
    from mrnax.schemas.construct_bundle import ChemistryTrackSpec, ConstructBundle, RNAMoleculeSpec, SegmentSpan
    from mrnax.schemas.design_intent import DesignIntent, ObjectiveSpec, ProteinTargetSpec, RuntimePolicySpec
    from mrnax.constraints.registry import default_m1_constraints

    # 1. Standalone UUUC (with ZERO UUUU!) detected and synonymously remediated to UUCC!
    standalone_uuuc = "AUGUUUCUGGAA"  # M-F(UUU)-L(CUG)-E(GAA) -> contains UUUC at [3..7), zero U{4,}!
    assert "UUUU" not in standalone_uuuc
    assert find_m1psi_slippery_spans(standalone_uuuc, include_uuuc=True) == [(3, 7, "UUUC")]
    rem_uuuc, n_ed = remediate_m1psi_slippery_runs(standalone_uuuc, include_uuuc=True)
    assert n_ed >= 1
    assert "UUUC" not in rem_uuuc and "UUUU" not in rem_uuuc
    assert rem_uuuc == "AUGUUCCUGGAA"

    # Build real tiny BigBirdForMaskedLM (default block_sparse!) to test short->long->short attention switching + framing
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

    class _Tok:
        unk_token_id = 0

        def get_vocab(self) -> dict[str, int]:
            return dict(vocab)

        def __call__(self, texts, return_tensors="pt", pad_to_multiple_of=None, **_kw):  # type: ignore[no-untyped-def]
            batch = [texts] if isinstance(texts, str) else list(texts)
            rows = [[1, *[vocab.get(t, 0) for t in s.split()], 2] for s in batch]
            max_l = max(len(r) for r in rows)
            if pad_to_multiple_of and max_l % pad_to_multiple_of != 0:
                max_l += pad_to_multiple_of - (max_l % pad_to_multiple_of)
            padded = [r + [3] * (max_l - len(r)) for r in rows]
            return {"input_ids": torch.tensor(padded, dtype=torch.long)}

    cfg = BigBirdConfig(
        vocab_size=nxt,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=2048,
        type_vocab_size=165,
        attention_type="block_sparse",
        block_size=16,
        num_random_blocks=1,
    )
    bb_model = BigBirdForMaskedLM(cfg).eval()
    ct = CodonTransformerNativeModel(tokenizer=_Tok(), model=bb_model, device="cpu")

    # Unified 2,045 sense-codon limit across BOTH generate_and_score_cds AND score_cds_synonymous_loglik!
    with pytest.raises(ValueError, match="2045|2046"):
        ct.generate_and_score_cds("M" + ("A" * 2045))
    with pytest.raises(ValueError, match="2045|2046"):
        ct.score_cds_synonymous_loglik(["AUG" + ("GCU" * 2045)])

    # Multi-stop cassette (`UGAUAAUAG`) MUST NOT suppress score!
    s_single_stop = ct.score_cds_synonymous_loglik(["AUGUUCCUGGAAUGA"])[0]
    s_multi_stop = ct.score_cds_synonymous_loglik(["AUGUUCCUGGAAUGAUAAUAG"])[0]
    assert np.isfinite(s_single_stop) and np.isfinite(s_multi_stop)
    assert abs(s_single_stop - s_multi_stop) < 1e-6

    clean, spaced, ambiguous, punctuation = ct.score_cds_synonymous_loglik(
        ["AUGGCU", "atg\n gct", "AUGNNNGCU", "AUG---GCU"]
    )
    assert np.isfinite(clean) and clean == spaced
    assert np.isnan(ambiguous) and np.isnan(punctuation)

    # Short -> Long (>1020) -> Short on the SAME BigBirdForMaskedLM instance switches attention_type properly!
    _ = ct.score_cds_synonymous_loglik(["AUG" + ("GCU" * 10)])
    assert bb_model.bert.attention_type == "original_full"
    _ = ct.score_cds_synonymous_loglik(["AUG" + ("GCU" * 1025)])
    assert bb_model.bert.attention_type == "block_sparse"
    _ = ct.score_cds_synonymous_loglik(["AUG" + ("GCU" * 10)])
    assert bb_model.bert.attention_type == "original_full"

    # Pipeline check: cached 2-chain m1psi bundle where ONLY chain 2 has standalone `UUUC` (or `UUUU`) is rejected under RC-005,
    # AND unmodified_u control does NOT claim m1psi remediation in manifest!
    p1 = ProteinTargetSpec(protein_sequence="MAKL", accession_or_source="t1")
    p2 = ProteinTargetSpec(protein_sequence="MFLE", accession_or_source="t2")
    u5 = "GGGAAAUAAAGAGAGAAAAGAAGAGUAAGAAGAAAUAUAAGAGCCACC"
    u3 = "GCUGCCUUCUGCGGGGCUUGCCUUCUGGCC"
    pa = "A" * 60
    st = "UGAUAA"

    def _mk_mol(mid: str, pt: ProteinTargetSpec, cds: str, chem: str = "m1psi") -> RNAMoleculeSpec:
        full = u5 + cds + st + u3 + pa
        c1, c2, c3, c4, c5 = len(u5), len(u5) + len(cds), len(u5) + len(cds) + len(st), len(u5) + len(cds) + len(st) + len(u3), len(full)
        return RNAMoleculeSpec(
            molecule_id=mid,
            modality="linear_mrna",
            topology="linear",
            chemistry_track=ChemistryTrackSpec(uridine_chemistry=chem, cap_chemistry="cap1_cleancap_ag", source="user_declared"),  # type: ignore[arg-type]
            effective_translation_target=pt,
            sequence=full,
            segments=[
                SegmentSpan(segment_type="five_prime_utr", start=0, end=c1, component_id="u5", origin_source="c", license_track="commercial_clean", action_taken="selected_from_library"),
                SegmentSpan(segment_type="cds", start=c1, end=c2, component_id=None, origin_source="c", license_track="commercial_clean", action_taken="generated_novo"),
                SegmentSpan(segment_type="stop_cassette", start=c2, end=c3, component_id=None, origin_source="c", license_track="commercial_clean", action_taken="locked_verbatim"),
                SegmentSpan(segment_type="three_prime_utr", start=c3, end=c4, component_id="u3", origin_source="c", license_track="commercial_clean", action_taken="selected_from_library"),
                SegmentSpan(segment_type="poly_a", start=c4, end=c5, component_id="pa", origin_source="c", license_track="commercial_clean", action_taken="locked_verbatim"),
            ],
        )

    bad_ch2_bundle = ConstructBundle.create(
        bundle_id="bnd_2ch_ch2_uuuc",
        parent_bundle_id=None,
        derivation_action="de_novo_design",
        co_expression_mode="lnp_co_encapsulation",
        target_stoichiometry={"ch1": 0.5, "ch2": 0.5},
        molecules=[
            _mk_mol("ch1", p1, "AUGGCUAAGCUG"),
            _mk_mol("ch2", p2, "AUGUUUCUGGAA"),  # standalone UUUC on chain 2!
        ],
    )
    from mrnax.harness.state_machine import ProjectHarnessSession

    sess_2ch = ProjectHarnessSession(project_id="i3_2ch", store_dir=tmp_path / "proj_i3", seed_dir=SEED_DIR)
    it_2ch_ct = sess_2ch.create_initial_intent(
        turn_id="t1",
        user_utterance="2-chain RC-005 UUUC check",
        protein_sequence="MAKL",
        compliance_track="commercial_clean",
        uridine_chemistry="m1psi",
    ).model_copy_validated(
        protein_targets=[p1, p2],
        locked_scopes={},
        constraints=default_m1_constraints(restriction_enzymes=["BspQI"]),
        objectives=[
            ObjectiveSpec(
                requirement_id="o1",
                metric_id="cds_stability_pred",
                direction="maximize",
                priority_rank=1,
                evidence_status="validated_predictor",
                source_utterance="stab",
            )
        ],
        runtime_policy=RuntimePolicySpec(
            enabled_generators=["codontransformer_cds_baseline"],
            enabled_evaluators=["surrogate_v1_clean"],
            allow_cached_candidates=True,
        ),
    )
    # Remove generated candidate by locking? No, check that bad_ch2_bundle is eliminated by final RC-005 admission gate!
    b_out, rep_out, _ = run_design_pipeline(
        it_2ch_ct,
        cached_candidates=[bad_ch2_bundle],
        seed_dir=SEED_DIR,
        codontransformer_model=ct,
        isolated_sandbox_override=True,
    )
    assert all(b.bundle_id != "bnd_2ch_ch2_uuuc" for b in b_out)
    assert any("UUUC" in d.reason for d in rep_out.failed_diagnostics if d.candidate_ref == "bnd_2ch_ch2_uuuc")

    # Unmodified-U control run MUST NOT claim m1psi remediation in optimization_schedule_and_operators!
    it_unmod_ct = it_2ch_ct.model_copy_validated(
        uridine_chemistry="unmodified_u",
        runtime_policy=RuntimePolicySpec(
            enabled_generators=["codontransformer_cds_baseline"],
            enabled_evaluators=["surrogate_v1_clean"],
            allow_cached_candidates=False,
        ),
    )
    b_unmod, rep_unmod, man_unmod = run_design_pipeline(
        it_unmod_ct,
        seed_dir=SEED_DIR,
        codontransformer_model=ct,
        isolated_sandbox_override=True,
    )
    assert len(b_unmod) == 1
    ops_joined = " ".join(man_unmod.optimization_mode_audit.optimization_schedule_and_operators)
    assert "m1psi" not in ops_joined.lower()
    assert "unmodified_u" in ops_joined.lower()


def test_important6_eternafold_empty_posterior_and_unverified_binary_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Important 6: Empty/truncated EternaFold posterior file must NOT emit 1.0 probability, and production mode verifies binary digest."""
    import hashlib
    import stat
    from mrnax.cloud.gateway import _build_preset_reference_r
    from mrnax.engine.pareto_director import run_design_pipeline
    from mrnax.harness.state_machine import ProjectHarnessSession
    from mrnax.rules.models import APPROVED_REPLACEMENT_RELEASES
    from mrnax.schemas.design_intent import ObjectiveSpec, RuntimePolicySpec

    ref = _build_preset_reference_r(SEED_DIR)
    sess = ProjectHarnessSession(project_id="i6_ef", store_dir=tmp_path / "proj_i6", seed_dir=SEED_DIR)
    it_rm = sess.create_initial_intent(
        turn_id="t1",
        user_utterance="Research EternaFold empty posterior check",
        protein_sequence=ref.molecules[0].effective_translation_target.protein_sequence,
        compliance_track="research_max",
        uridine_chemistry="m1psi",
    ).model_copy_validated(
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

    empty_bin = tmp_path / "contrafold_empty"
    empty_bin.write_text(
        "#!/usr/bin/env python3\nimport pathlib, sys\npathlib.Path(sys.argv[-1]).write_text('')\n",
        encoding="utf-8",
    )
    empty_bin.chmod(empty_bin.stat().st_mode | stat.S_IXUSR)
    fake_params = tmp_path / "EternaFoldParams.v1"
    fake_params.write_bytes(b"sandbox")
    monkeypatch.setenv("MRNAX_ETERNAFOLD_BIN", str(empty_bin))
    monkeypatch.setenv("MRNAX_ETERNAFOLD_PARAMS", str(fake_params))

    b_emp, rep_emp, _ = run_design_pipeline(
        it_rm, cached_candidates=[ref], seed_dir=SEED_DIR, isolated_sandbox_override=True
    )
    assert b_emp == []
    assert rep_emp.outcome_status == "UNCOMPARABLE_MISSING_APPROVED_COMPONENT"

    # Production mode (`isolated_sandbox_override=False`) without MRNAX_ETERNAFOLD_BIN_SHA256 must fail closed even if params SHA matches!
    monkeypatch.delenv("MRNAX_ETERNAFOLD_BIN_SHA256", raising=False)
    monkeypatch.setitem(
        APPROVED_REPLACEMENT_RELEASES,
        "eternafold_structure_head",
        APPROVED_REPLACEMENT_RELEASES["eternafold_structure_head"]._replace(
            artifact_digests={"parameters/EternaFoldParams.v1": hashlib.sha256(b"sandbox").hexdigest()}
        ),
    )
    b_nobin_sha, rep_nobin_sha, _ = run_design_pipeline(
        it_rm, cached_candidates=[ref], seed_dir=SEED_DIR, isolated_sandbox_override=False
    )
    assert b_nobin_sha == []
    assert rep_nobin_sha.outcome_status == "UNCOMPARABLE_MISSING_APPROVED_COMPONENT"


# ============================================================================
# Batch C: Critical 1, Critical 2, Important 1, Important 2, Minor 2
# ============================================================================


def test_critical1_and_important1_method_fto_deterministic_repair_typed_events_and_policy_lint_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Critical 1 & Important 1: zero weighted-random repair on must_fix, accurate locked-CDS operators, all 6 boundaries reactive, incomplete registries/empty lints rejected."""
    import random
    from mrnax.cloud.gateway import _build_preset_reference_r
    from mrnax.constraints.fto_guard import MethodFTOGuard, SequenceFTOGuard, audit_codebase_method_fto_lints
    from mrnax.engine.pareto_director import run_design_pipeline
    from mrnax.engine.stage0_baseline import propose_non_partitioned_bicodon_cds
    from mrnax.harness.state_machine import ProjectHarnessSession
    from mrnax.schemas.design_intent import ConstraintSpec

    # 1. Incomplete policy & sequence registries MUST raise ValueError on load!
    with pytest.raises(ValueError):
        MethodFTOGuard({"certifies_100_percent_fto": False})
    with pytest.raises(ValueError):
        SequenceFTOGuard({})

    # 2. Static lint on nonexistent or empty directory MUST return BLOCK (never PASS on 0 files),
    #    and catch config dict keys, dict-subscript MFE+regularizer, F2 dual_hash, and F4C function names!
    nonexist_lint = audit_codebase_method_fto_lints(tmp_path / "does_not_exist")
    assert nonexist_lint["status"] == "BLOCK"
    empty_dir = tmp_path / "empty_src"
    empty_dir.mkdir()
    assert audit_codebase_method_fto_lints(empty_dir)["status"] == "BLOCK"

    adv_bad = tmp_path / "adv_bad_src"
    adv_bad.mkdir()
    (adv_bad / "bad_cases.py").write_text(
        "CFG = {'init_region_len': 30}\n"
        "def compute_protein_per_time(row, w):\n"
        "    dual_hash_dfa = True\n"
        "    return row['mfe'] + w * row['cai']\n",
        encoding="utf-8",
    )
    res_adv = audit_codebase_method_fto_lints(adv_bad)
    assert res_adv["status"] == "BLOCK"
    hit_b = {v["boundary_id"] for v in res_adv["violations"]}
    assert "BAIDU_CN119265214B_STAGED_UTR5_NCDS_PARTITION_OPT" in hit_b
    assert "BAIDU_CN118038986B_DFA_DUAL_HASH_DICODON_DP" in hit_b
    assert "BAIDU_USA_LINEARDESIGN_DFA_CFG_LATTICE_AND_ADDITIVE_MFE_REG" in hit_b
    assert "LEIBNIZ_HANNOVER_US11848074B2_EP3551758B1_BOOSTED_GAM_PROTEIN_PER_TIME" in hit_b

    # Real codebase src/mrnax MUST pass static lint with >=40 files scanned!
    real_lint = audit_codebase_method_fto_lints(REPO_ROOT / "src" / "mrnax")
    assert real_lint["status"] == "PASS"
    assert real_lint["files_scanned"] >= 40

    # 3. `propose_non_partitioned_bicodon_cds` MUST NOT invoke weighted `rng.choices` on mandatory motif repair (`must_fix`)!
    class _SpyRandom(random.Random):
        def __init__(self, seed: int) -> None:
            super().__init__(seed)
            self.weighted_choices_calls = 0

        def choices(self, population, weights=None, *, cum_weights=None, k=1):  # type: ignore[no-untyped-def]
            if weights is not None or cum_weights is not None:
                self.weighted_choices_calls += 1
            return super().choices(population, weights=weights, cum_weights=cum_weights, k=k)

    # Set mutation_prob=0.0 so ONLY mandatory `must_fix` codons execute repair!
    spy_rng = _SpyRandom(42)
    dirty_cds = "AUG" + "UUUUUU" + "GCUCUUCAA"  # M-F-F-A-L-Q containing UUUUUU + BspQI GCUCUUC!
    repaired = propose_non_partitioned_bicodon_cds(
        dirty_cds,
        "MFFALQ",
        spy_rng,
        chemistry="m1psi",
        mutation_prob=0.0,
        temperature=0.2,
        constraints=[
            ConstraintSpec(
                requirement_id="enz",
                constraint_kind="type_iis_restriction_free",
                target_scopes=["full_transcript"],
                params={"enzymes": ["BspQI"]},
                source_type="user_explicit",
                source_locator="t1",
            )
        ],
    )
    assert spy_rng.weighted_choices_calls == 0
    assert "UUUU" not in repaired and "GCUCUUC" not in repaired

    # 4. Locked-CDS run MUST NOT list Stage0 generation among executed_operators when Stage0 made 0 attempts,
    #    and zero-objective run MUST report `has_axis_outside_tie_cai_mfe=False`!
    ref = _build_preset_reference_r(SEED_DIR)
    sess = ProjectHarnessSession(project_id="c1_lock", store_dir=tmp_path / "proj_c1", seed_dir=SEED_DIR)
    it_locked = sess.create_initial_intent(
        turn_id="t1",
        user_utterance="Locked CDS zero objective check",
        protein_sequence=ref.molecules[0].effective_translation_target.protein_sequence,
        reference_bundle=ref,
        compliance_track="commercial_clean",
        uridine_chemistry="m1psi",
        locked_regions={"cds": ref.molecules[0].get_segment_sequence("cds")},
    ).model_copy_validated(objectives=[])
    _, _, man_locked = run_design_pipeline(it_locked, reference_bundle=ref, seed_dir=SEED_DIR)
    ops_locked = man_locked.optimization_mode_audit.optimization_schedule_and_operators
    assert "stage0_random_synonymous_cds_constructive" not in ops_locked
    assert man_locked.optimization_mode_audit.objective_formulation["has_axis_outside_tie_cai_mfe"] is False
    assert man_locked.optimization_mode_audit.timestamp_utc
    assert all("sha256" in d for d in man_locked.optimization_mode_audit.three_layer_dependency_manifest)

    # 5. F2, F3, F4B, and F4C runtime decisions MUST flip to BLOCK when prohibited operator evidence is supplied!
    mfto = MethodFTOGuard.from_json_path(SEED_DIR / "method_fto_guard_v1.1.json")
    seq_rcpt = {"screen_applied": True, "cartesian_pair_rules_checked": 1}
    for op_tok, expected_bid, extra_assert in (
        ("dfa_dual_hash_backpointer_dp", "BAIDU_CN118038986B_DFA_DUAL_HASH_DICODON_DP", None),
        ("dfa_cfg_lattice_intersection", "BAIDU_USA_LINEARDESIGN_DFA_CFG_LATTICE_AND_ADDITIVE_MFE_REG", "uses_dfa_cfg_intersection"),
        ("weighted_random_replacement_repair", "ASU_US12706176B1_RANDOM_SYNONYMOUS_SAMPLING_PROBLEM_COUNT_ARGMIN", None),
        ("boosted_gam_mboost_scorer", "LEIBNIZ_HANNOVER_US11848074B2_EP3551758B1_BOOSTED_GAM_PROTEIN_PER_TIME", None),
    ):
        rec_b = mfto.build_runtime_audit_record(
            intent=it_locked,
            opt_mode="full_transcript_non_partitioned_pareto",
            used_two_zone_partition=False,
            executed_operators=[op_tok],
            invoked_components=[],
            seq_fto_receipt=seq_rcpt,
            input_origin_type="amino_acid_only",
        )
        assert rec_b.method_fto_boundary_checks[expected_bid]["status"] == "BLOCK"
        if extra_assert == "uses_dfa_cfg_intersection":
            assert rec_b.uses_dfa_cfg_intersection is True


def test_critical2_curevac_two_chain_invalid_reference_wild_hyphen_type_and_export_block(
    tmp_path: Path,
) -> None:
    """Critical 2: 2-chain second-chain >=0.65 blocked; invalid/mismatched/non-synonymous WT fails closed; 'wild-type' matched; direct export blocked with 0 files written."""
    from mrnax.constraints.fto_guard import compute_curevac_c_optimizable_conversion
    from mrnax.constraints.registry import ScopedConstraintEngine, default_m1_constraints
    from mrnax.engine.pareto_director import run_design_pipeline
    from mrnax.harness.state_machine import ProjectHarnessSession
    from mrnax.io_insdc.genbank_io import ExportConstraintViolationError, export_construct_package
    from mrnax.schemas.construct_bundle import ChemistryTrackSpec, ConstructBundle, RNAMoleculeSpec, SegmentSpan
    from mrnax.schemas.design_intent import ObjectiveSpec, ProteinTargetSpec, RuntimePolicySpec

    # 1. Invalid, empty, length-mismatched, or non-synonymous comparisons MUST raise ValueError (never return 0.0)!
    with pytest.raises(ValueError):
        compute_curevac_c_optimizable_conversion("", "")
    with pytest.raises(ValueError):
        compute_curevac_c_optimizable_conversion("AUGUUUUUA", "AUGUUUUUAUCU")
    with pytest.raises(ValueError):
        compute_curevac_c_optimizable_conversion("AUGUUUUUA", "AUGGGGUUA")  # F(UUU) vs G(GGG) non-synonymous!

    wt1 = "AUG" + ("GCCGUGAAG" * 8)
    wt2 = "AUG" + ("UUUUUAUCU" * 8)  # 24 C-optimizable codons on chain 2
    hi_c_ch2 = "AUG" + ("UUCCUCUCC" * 8)  # 100% C-converted on chain 2!
    prot1 = "M" + ("AVK" * 8)
    prot2 = "M" + ("FLS" * 8)
    u5 = "GGGAAAUAAAGAGAGAAAAGAAGAGUAAGAAGAAAUAUAAGAGCCACC"
    u3 = "GCUGCCUUCUGCGGGGCUUGCCUUCUGGCC"
    st = "UGAUAA"
    pa = "A" * 60

    def _mol(mid: str, pt: ProteinTargetSpec, cds: str, origin: str = "clean") -> RNAMoleculeSpec:
        full = u5 + cds + st + u3 + pa
        c1, c2, c3, c4, c5 = len(u5), len(u5) + len(cds), len(u5) + len(cds) + len(st), len(u5) + len(cds) + len(st) + len(u3), len(full)
        return RNAMoleculeSpec(
            molecule_id=mid,
            modality="linear_mrna",
            topology="linear",
            chemistry_track=ChemistryTrackSpec(uridine_chemistry="unmodified_u", cap_chemistry="cap1_cleancap_ag", source="user_declared"),
            effective_translation_target=pt,
            sequence=full,
            segments=[
                SegmentSpan(segment_type="five_prime_utr", start=0, end=c1, component_id="u5", origin_source="c", license_track="commercial_clean", action_taken="selected_from_library"),
                SegmentSpan(segment_type="cds", start=c1, end=c2, component_id=None, origin_source=origin, license_track="commercial_clean", action_taken="generated_novo"),
                SegmentSpan(segment_type="stop_cassette", start=c2, end=c3, component_id=None, origin_source="c", license_track="commercial_clean", action_taken="locked_verbatim"),
                SegmentSpan(segment_type="three_prime_utr", start=c3, end=c4, component_id="u3", origin_source="c", license_track="commercial_clean", action_taken="selected_from_library"),
                SegmentSpan(segment_type="poly_a", start=c4, end=c5, component_id="pa", origin_source="c", license_track="commercial_clean", action_taken="locked_verbatim"),
            ],
        )

    pt1 = ProteinTargetSpec(protein_sequence=prot1, accession_or_source="c1", wildtype_cds_rna=wt1)
    pt2 = ProteinTargetSpec(protein_sequence=prot2, accession_or_source="c2", wildtype_cds_rna=wt2)
    cand_2ch = ConstructBundle.create(
        bundle_id="bnd_2ch_curevac_ch2_high",
        parent_bundle_id=None,
        derivation_action="de_novo_design",
        co_expression_mode="lnp_co_encapsulation",
        target_stoichiometry={"ch1": 0.5, "ch2": 0.5},
        molecules=[
            _mol("ch1", pt1, wt1),       # Chain 1: 0.0 conversion!
            _mol("ch2", pt2, hi_c_ch2),  # Chain 2: 1.0 conversion -> MUST BLOCK!
        ],
    )
    sess = ProjectHarnessSession(project_id="c2_cv", store_dir=tmp_path / "proj_c2", seed_dir=SEED_DIR)
    it_2ch = sess.create_initial_intent(
        turn_id="t1",
        user_utterance="2-chain CureVac check",
        protein_sequence=prot1,
        compliance_track="commercial_clean",
        uridine_chemistry="unmodified_u",
    ).model_copy_validated(
        protein_targets=[pt1, pt2],
        locked_scopes={},
        constraints=default_m1_constraints(restriction_enzymes=["BspQI"]),
        objectives=[
            ObjectiveSpec(
                requirement_id="o1",
                metric_id="cds_stability_pred",
                direction="maximize",
                priority_rank=1,
                evidence_status="validated_predictor",
                source_utterance="stab",
            )
        ],
        runtime_policy=RuntimePolicySpec(
            enabled_generators=[],
            enabled_evaluators=["surrogate_v1_clean"],
            allow_cached_candidates=True,
        ),
    )

    # 2. Chain 1 = 0.0, Chain 2 = 1.0 MUST be blocked and report >=0.99 / BLOCK (NOT 0.0 / PASS)!
    b_2ch, rep_2ch, man_2ch = run_design_pipeline(
        it_2ch, cached_candidates=[cand_2ch], seed_dir=SEED_DIR
    )
    assert b_2ch == []
    assert rep_2ch.outcome_status == "COMPLETED_ZERO_QUALIFIED"
    assert (man_2ch.optimization_mode_audit.wildtype_c_optimizable_replacement_fraction or 0.0) >= 0.99
    assert (
        man_2ch.optimization_mode_audit.method_fto_boundary_checks[
            "CUREVAC_US10898584B2_FAMILY_SYNONYMOUS_CYTOSINE_THRESHOLD"
        ]["status"]
        == "BLOCK"
    )

    # 3. Direct commercial export of blocked candidate MUST fail BEFORE creating out_dir / writing any file!
    exp_dir = tmp_path / "should_not_be_written"
    ce = ScopedConstraintEngine.from_seed_dir(SEED_DIR)
    with pytest.raises(ExportConstraintViolationError, match="CUREVAC|0.65"):
        export_construct_package(
            cand_2ch,
            exp_dir,
            constraint_engine=ce,
            constraints=it_2ch.constraints,
            compliance_track="commercial_clean",
            expected_chemistry="unmodified_u",
            expected_protein_targets=it_2ch.protein_targets,
        )
    assert not exp_dir.exists()

    # 4. Invalid supplied wildtype_cds_reference MUST fail closed (UNCOMPARABLE / BLOCK), never silently pass!
    b_bad_wt, rep_bad_wt, man_bad_wt = run_design_pipeline(
        it_2ch,
        cached_candidates=[cand_2ch],
        wildtype_cds_reference="AUGGGGGGG",
        seed_dir=SEED_DIR,
    )
    assert b_bad_wt == []
    assert rep_bad_wt.outcome_status == "UNCOMPARABLE_MISSING_APPROVED_COMPONENT"
    assert (
        man_bad_wt.optimization_mode_audit.method_fto_boundary_checks[
            "CUREVAC_US10898584B2_FAMILY_SYNONYMOUS_CYTOSINE_THRESHOLD"
        ]["status"]
        == "BLOCK"
    )

    # 5. Provenance text 'wild-type' (with hyphen!) on reference_bundle MUST be recognized identically to 'wildtype'!
    pt2_plain = ProteinTargetSpec(protein_sequence=prot2, accession_or_source="c2")
    ref_hyphen = ConstructBundle.create(
        bundle_id="bnd_ref_wild_hyphen_type",
        parent_bundle_id=None,
        derivation_action="reference_import",
        target_stoichiometry={"m1": 1.0},
        molecules=[_mol("m1", pt2_plain, wt2, origin="Endogenous wild-type mRNA")],
    )
    cand_1ch_hi = ConstructBundle.create(
        bundle_id="bnd_1ch_hi",
        parent_bundle_id="bnd_ref_wild_hyphen_type",
        derivation_action="de_novo_design",
        target_stoichiometry={"m1": 1.0},
        molecules=[_mol("m1", pt2_plain, hi_c_ch2, origin="generated")],
    )
    it_1ch = it_2ch.model_copy_validated(protein_targets=[pt2_plain])
    b_hyp, rep_hyp, man_hyp = run_design_pipeline(
        it_1ch, reference_bundle=ref_hyphen, cached_candidates=[cand_1ch_hi], seed_dir=SEED_DIR
    )
    assert b_hyp == []
    assert rep_hyp.outcome_status == "COMPLETED_ZERO_QUALIFIED"
    assert man_hyp.optimization_mode_audit.input_origin_type == "wildtype_mrna_reference"


def test_important2_and_minor2_cartesian_pair_95pct_alignment_intervals_molecule_ids_and_rat_mnsod() -> None:
    """Important 2 & Minor 2: 19/20 (95%) gapped alignment matches, exact paired intervals & molecule_ids recorded, performed counts accurate, rat MnSOD label."""
    from mrnax.constraints.fto_guard import SequenceFTOGuard
    from mrnax.schemas.construct_bundle import ChemistryTrackSpec, ConstructBundle, RNAMoleculeSpec, SegmentSpan
    from mrnax.schemas.design_intent import ProteinTargetSpec

    fto_doc = json.loads((SEED_DIR / "patent_fto_guard_v2.json").read_text(encoding="utf-8"))
    pair_rule = fto_doc["cartesian_utr_pair_rules"][0]
    c3_ids = {c["id"] for c in pair_rule["three_prime_utr_candidates"]}
    assert "SEQ_ID_2_rat_MnSOD_3utr_tableau1" in c3_ids
    assert "SEQ_ID_2_human_MnSOD_3utr" not in c3_ids
    alias_row = next(c for c in pair_rule["five_prime_utr_candidates"] if "alias" in c["id"])
    assert alias_row["source_kind"] == "internal_screening_alias_not_in_tableau1"

    guard = SequenceFTOGuard(fto_doc)
    # Symbolic 20-nt alignment check: 19 identical bases + 1 internal deletion = 19/20 = 0.950 identity!
    ref20 = "ACGUACGUACGUACGUACGU"
    sub_with_flank = "GG" + "ACGUACGUAC" + "GUACGUACGU" + "CC"  # deleted index 10 ('G')? Wait: 19 bases matching out of 20!
    sub_19_of_20 = "GG" + ref20[:10] + ref20[11:] + "CC"  # 19/20 exact columns with 1 gap!
    ok, pat, span = guard._match_identity(sub_19_of_20, ref20, 0.95)
    assert ok is True
    assert span == (2, 21)

    # Two affected molecules produce distinct alerts carrying exact paired_intervals and molecule_id!
    u5_hit = "GG" + pair_rule["five_prime_utr_candidates"][0]["sequence_rna"] + "CC"
    vp6_ref = pair_rule["three_prime_utr_candidates"][0]["sequence_rna"]  # 60 nt -> drop 1 base = 59/60 = 0.983 >= 0.95
    u3_hit = "AA" + vp6_ref[:30] + vp6_ref[31:] + "GG"
    cds = "AUGGCUAAGCUG"
    st = "UGAUAA"
    pa = "A" * 40
    pt = ProteinTargetSpec(protein_sequence="MAKL", accession_or_source="t")

    def _m(mid: str) -> RNAMoleculeSpec:
        full = u5_hit + cds + st + u3_hit + pa
        c1, c2, c3, c4, c5 = len(u5_hit), len(u5_hit) + len(cds), len(u5_hit) + len(cds) + len(st), len(u5_hit) + len(cds) + len(st) + len(u3_hit), len(full)
        return RNAMoleculeSpec(
            molecule_id=mid,
            modality="linear_mrna",
            topology="linear",
            chemistry_track=ChemistryTrackSpec(uridine_chemistry="m1psi", cap_chemistry="cap1_cleancap_ag", source="user_declared"),
            effective_translation_target=pt,
            sequence=full,
            segments=[
                SegmentSpan(segment_type="five_prime_utr", start=0, end=c1, component_id="u5", origin_source="c", license_track="commercial_clean", action_taken="selected_from_library"),
                SegmentSpan(segment_type="cds", start=c1, end=c2, component_id=None, origin_source="c", license_track="commercial_clean", action_taken="generated_novo"),
                SegmentSpan(segment_type="stop_cassette", start=c2, end=c3, component_id=None, origin_source="c", license_track="commercial_clean", action_taken="locked_verbatim"),
                SegmentSpan(segment_type="three_prime_utr", start=c3, end=c4, component_id="u3", origin_source="c", license_track="commercial_clean", action_taken="selected_from_library"),
                SegmentSpan(segment_type="poly_a", start=c4, end=c5, component_id="pa", origin_source="c", license_track="commercial_clean", action_taken="locked_verbatim"),
            ],
        )

    two_mol_bundle = ConstructBundle.create(
        bundle_id="bnd_2mol_pairs",
        parent_bundle_id=None,
        derivation_action="de_novo_design",
        co_expression_mode="lnp_co_encapsulation",
        target_stoichiometry={"mol_A": 0.5, "mol_B": 0.5},
        molecules=[_m("mol_A"), _m("mol_B")],
    )
    alerts = guard.scan_bundle(two_mol_bundle, "commercial_clean")
    pair_alerts = [a for a in alerts if a.rule_id == "WO2024231621_CNRS_DUAL_UTR_CARTESIAN_PAIR"]
    assert len(pair_alerts) == 2
    assert {a.molecule_id for a in pair_alerts} == {"mol_A", "mol_B"}
    assert pair_alerts[0].identity_or_pattern != pair_alerts[1].identity_or_pattern
    for pa_al in pair_alerts:
        assert pa_al.paired_intervals is not None and len(pa_al.paired_intervals) == 2
        assert pa_al.paired_intervals[0] == (2, 2 + len(pair_rule["five_prime_utr_candidates"][0]["sequence_rna"]))
        u3_seg_start = len(u5_hit) + len(cds) + len(st)
        assert pa_al.paired_intervals[1] == (u3_seg_start + 2, u3_seg_start + 2 + len(vp6_ref) - 1)


    first_receipt = guard.build_receipt(alerts, screen_applied=True)
    only_cds = []
    for molecule in two_mol_bundle.molecules:
        cds = molecule.get_segment_sequence("cds")
        span = molecule.get_segment("cds").model_copy_validated(start=0, end=len(cds))
        only_cds.append(molecule.model_copy_validated(sequence=cds, segments=[span]))
    sparse = ConstructBundle.create(
        **two_mol_bundle.model_dump(exclude={"version_digest", "molecules"}), molecules=only_cds,
    )
    second = guard.scan_bundle(sparse, "commercial_clean")
    fields = ("genus_rules_checked", "collision_rules_checked", "cartesian_pair_rules_checked")
    second_receipt = guard.build_receipt(second, screen_applied=True)
    assert [second_receipt[key] for key in fields] == [0, 0, 0]
    assert not second_receipt["screen_applied"]
    assert [first_receipt[key] for key in fields] == [4, 12, 2]
    assert first_receipt["screen_applied"] and first_receipt["blocked_alerts"] == 2
    assert guard.build_receipt(alerts, screen_applied=True) == first_receipt
    assert guard.build_receipt(alerts) == first_receipt
    aggregate = guard.build_receipt(
        alerts, screen_applied=True, performed_counts={"genus": 8, "collision": 24, "cartesian_pair": 4},
    )
    assert [aggregate[key] for key in fields] == [8, 24, 4]

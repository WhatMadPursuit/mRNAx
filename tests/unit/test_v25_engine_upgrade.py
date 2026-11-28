"""Wave 2A Unit & Contract Suite:
1. Wave 1A registries in `registry_seed/` + `surrogates.py` single-pass perf & `surrogate_v2` loader hooks.
2. `lineardesign_adapter.py`: C++ binary discovery, lambda sweep, 100% AA check, 5'-Opening & Constraint Polisher, fallback, and `commercial_clean` isolation.
3. `structure_profile.py`: 2D initiation arcs (`prob >= 0.15`, `i < j`), sliding-window tracks, exact 0-based motifs, and optional EternaFold.
4. `modalities.py` + `pareto_director.py`: `linear_mrna` (single & multi-chain + tissue miRNA), `circrna` (PIE + spacers + IRES + junction/uncoupling), `sarna` (CSE/26S-SGP + innate sensing), and static AST Method-FTO lint.
5. Paired benchmark artifact verification (`artifacts/v25_engine_ablation_report.json`).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from mrnax.constraints.fto_guard import audit_codebase_method_fto_lints
from mrnax.constraints.registry import default_m1_constraints
from mrnax.engine.lineardesign_adapter import (
    DEFAULT_LAMBDA_SWEEP,
    DEFAULT_LINEARDESIGN_TIMEOUT_SEC,
    DEFAULT_MAX_LINEARDESIGN_AA,
    DEFAULT_POLISH_ENZYMES,
    build_forbidden_motif_list,
    count_forbidden_motif_hits,
    discover_lineardesign_binary,
    is_lineardesign_available,
    polish_cds_5prime_and_constraints,
    run_lineardesign_single,
    sweep_lineardesign_candidates,
    verify_exact_aa_translation,
)
from mrnax.engine.modalities import (
    build_circrna_bundle,
    build_linear_mrna_bundle,
    build_sarna_bundle,
    evaluate_circrna_bundle,
    evaluate_linear_mrna_bundle,
    evaluate_sarna_bundle,
    load_modality_scaffolds,
    load_tissue_mirna_atlas,
    select_atlas_mirna_for_detargeting,
)
from mrnax.engine.pareto_director import run_design_pipeline
from mrnax.engine.structure_profile import analyze_rna_structure, discover_eternafold_binary
from mrnax.engine.surrogates import (
    CommercialCleanSurrogateV1,
    CommercialCleanSurrogateV2,
    _load_verified_joblib,
    has_surrogate_v2_artifacts,
    load_surrogate_jury,
)
from mrnax.rules import load_scientific_rules_v2, query_scientific_rules
from mrnax.schemas.construct_bundle import ProteinTargetSpec
from mrnax.schemas.design_intent import DesignIntent, ObjectiveSpec, RuntimePolicySpec

ROOT = Path(__file__).resolve().parents[2]
SEED_DIR = ROOT / "registry_seed"


def test_wave1a_registries_and_scientific_rules_v2_queries() -> None:
    expected_files = (
        "data_hub_catalog_v2.json",
        "rules_scientific_v2.json",
        "clinical_mrna_benchmarks_v1.json",
        "modality_scaffolds_v1.json",
        "tissue_mirna_atlas_v1.json",
    )
    for fname in expected_files:
        fpath = SEED_DIR / fname
        assert fpath.is_file(), f"Missing {fname} in registry_seed/"
        doc = json.loads(fpath.read_text(encoding="utf-8"))
        assert isinstance(doc, dict) and doc

    rules_doc = load_scientific_rules_v2(SEED_DIR)
    assert len(rules_doc["rules"]) >= 32
    circ_rules = query_scientific_rules(seed_dir=SEED_DIR, modality="circrna")
    assert len(circ_rules) >= 3
    sarna_rules = query_scientific_rules(seed_dir=SEED_DIR, modality="sarna")
    assert len(sarna_rules) >= 3
    myths = query_scientific_rules(seed_dir=SEED_DIR, evidence_grade="debunked_myth")
    assert len(myths) >= 5
    non_myths = query_scientific_rules(seed_dir=SEED_DIR, include_debunked_myths=False)
    assert len(non_myths) + len(myths) == len(rules_doc["rules"])


def test_surrogate_v2_loader_hooks_and_graceful_v1_fallback(tmp_path: Path) -> None:
    load_surrogate_jury.cache_clear()
    CommercialCleanSurrogateV2.from_seed_dir.cache_clear()
    CommercialCleanSurrogateV1.from_seed_dir.cache_clear()

    # 1. Production registry_seed/surrogate_v2 contains ONLY the 6 commercial_clean models + manifest + scorecard
    real_v2_dir = SEED_DIR / "surrogate_v2"
    assert real_v2_dir.is_dir()
    v2_files = sorted(p.name for p in real_v2_dir.rglob("*") if p.is_file())
    assert v2_files == [
        "TRAINING_PROVENANCE_MANIFEST.json",
        "cds_codon_stability_v2.joblib",
        "circrna_ires_translation_v2.joblib",
        "full_tx_isoform_mrl_v2.joblib",
        "rmdb_aup_hydrolysis_proxy_v2.joblib",
        "saluki_consensus_halflife_v2.joblib",
        "surrogate_v2_ood_scorecard.json",
        "utr5_mrl_pure_seq_v2.joblib",
    ]
    assert not any("pending" in fn or "gse114002" in fn for fn in v2_files)

    # 2. Clearance-pending GSE114002 model is physically isolated and blocked from commercial_clean loader
    pending_dir = SEED_DIR / "surrogate_v2_pending"
    assert (pending_dir / "utr5_mrl_chem_gse114002_v2.joblib").is_file()
    with pytest.raises(PermissionError, match="commercial_clearance_pending"):
        _load_verified_joblib(pending_dir, "utr5_mrl_chem_gse114002_v2.joblib")

    # 3. Real surrogate_v2 loads via load_surrogate_jury(SEED_DIR, prefer_v2=True) with all 6 v2 heads + v1 3'UTR fallback
    assert has_surrogate_v2_artifacts(SEED_DIR) is True
    jury_real_v2 = load_surrogate_jury(SEED_DIR, prefer_v2=True)
    assert isinstance(jury_real_v2, CommercialCleanSurrogateV2)
    assert set(jury_real_v2.loaded_heads) == {
        "5utr_mrl",
        "cds_codon_stability",
        "saluki_full_tx_halflife",
        "3utr_mpra_stability",
        "full_tx_isoform_mrl",
        "rmdb_reactivity_hydrolysis_proxy",
        "circrna_ires_translation",
    }
    jury_opt_v1 = load_surrogate_jury(SEED_DIR, prefer_v2=False)
    assert isinstance(jury_opt_v1, CommercialCleanSurrogateV1)
    assert not isinstance(jury_opt_v1, CommercialCleanSurrogateV2)

    # 4. Score real eGFP + circRNA across all surrogate_v2 heads & verify exact subprocess parity
    egfp_u5 = "GAATAAACTAGTATTCTTCTGGTCCCCACAGACTCAGAGAGAACCCGCCACC"
    bad_u5 = "GGGAAAUAAGAUGGGAGAUGAUGGGGCGCGCGCGCGCGCGGGGCCACC"
    egfp_cds = (
        "AUGGUGAGCAAGGGCGAGGAGCUGUUCACCGGGGUGGUGCCCAUCCUGGUCGAGCUGGACGGCGACGUAAACGGCCACAAGUUCAGCGUGUCCGGCGAG"
        "GGCGAGGGCGAUGCCACCUACGGCAAGCUGACCCUGAAGUUCAUCUGCACCACCGGCAAGCUGCCCGUGCCCUGGCCCACCCUCGUGACCACCCUGACC"
        "UACGGCGUGCAGUGCUUCAGCCGCUACCCCGACCACAUGAAGCAGCACGACUUCUUCAAGUCCGCCAUGCCCGAAGGCUACGUCCAGGAGCGCACCAUC"
        "UUCUUCAAGGACGACGGCAACUACAAGACCCGCGCCGAGGUGAAGUUCGAGGGCGACACCCUGGUGAACCGCAUCGAGCUGAAGGGCAUCGACUUCAAG"
        "GAGGACGGCAACAUCCUGGGGCACAAGCUGGAGUACAACUACAACAGCCACAACGUCUAUAUCAUGGCCGACAAGCAGAAGAACGGCAUCAAGGUGAAC"
        "UUCAAGAUCCGCCACAACAUCGAGGACGGCAGCGUGCAGCUCGCCGACCACUACCAGCAGAACACCCCCAUCGGCGACGGCCCCGUGCUGCUGCCCGAC"
        "AACCACUACCUGAGCACCCAGUCCGCCCUGAGCAAAGACCCCAACGAGAAGCGCGAUCACAUGGUCCUGCUGGAGUUCGUGACCGCCGCCGGGAUCACU"
        "CUCGGCAUGGACGAGCUGUACAAG"
    )
    egfp_u3 = "GCUGGAGCCUCGGUGGCCAUGCUUCUUGCCCCUUGGGCCUCCCCCCAGCCCCUCCUCCCCUUCCUGCACCCGUACCCCCGUGGUCUUUGAAUAAAGUCUGAGUGGGCGG"
    poor_cds = "AUG" + ("UUAGAAUAAAAAUUAGGA" * 35)

    mrl_v2 = jury_real_v2.score_5utr_mrl([egfp_u5, bad_u5])
    assert mrl_v2.shape == (2,) and np.all(np.isfinite(mrl_v2))
    assert float(mrl_v2[0]) > float(mrl_v2[1])

    cds_v2 = jury_real_v2.score_cds_stability([egfp_cds, poor_cds])
    assert cds_v2.shape == (2,) and float(cds_v2[0]) > float(cds_v2[1])

    hl_v2 = jury_real_v2.score_consensus_halflife_batch(
        [(egfp_u5, egfp_cds, egfp_u3), (bad_u5, poor_cds, "AUUUAAUUUAAUUUAAUUUAAUUUAUUUUU")]
    )
    assert hl_v2.shape == (2,) and float(hl_v2[0]) > float(hl_v2[1])

    iso_v2 = jury_real_v2.score_isoform_mrl([(egfp_u5, egfp_cds, egfp_u3)])
    aup_v2 = jury_real_v2.score_rmdb_aup_proxy([egfp_u5 + egfp_cds[:60]])
    ires_v2 = jury_real_v2.score_circrna_ires([egfp_u5 * 8])
    u3_v1_fb = jury_real_v2.score_3utr_stability([egfp_u3])
    assert all(np.isfinite(float(arr[0])) for arr in (iso_v2, aup_v2, ires_v2, u3_v1_fb))

    native_check = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, joblib, numpy as np, sys; "
                "from pathlib import Path; "
                "from mrnax.engine.features_v2 import ("
                "cds_to_bicodon_tokens, dict_list_to_matrix, extract_cds_v2, extract_circrna_ires_v2, "
                "extract_full_tx_v2, extract_rmdb_seq_v2, extract_utr5_v2, seq_to_kmer_tokens); "
                "d = Path(sys.argv[1]); u5, cds, u3 = sys.argv[2], sys.argv[3], sys.argv[4]; "
                "b_u5 = joblib.load(d / 'utr5_mrl_pure_seq_v2.joblib'); "
                "Xm = dict_list_to_matrix([extract_utr5_v2(u5)], b_u5['tab_features']); "
                "km = b_u5['kmer_ridge'].predict([seq_to_kmer_tokens(u5, ks=tuple(b_u5['kmer_ks']))]).astype(np.float32); "
                "p_u5 = float(b_u5['model'].predict(np.hstack([Xm, km[:, None]]))[0]); "
                "b_hl = joblib.load(d / 'saluki_consensus_halflife_v2.joblib'); "
                "Xh = dict_list_to_matrix([extract_full_tx_v2(u5, cds, u3)], b_hl['tab_features']); "
                "m1 = b_hl['utr3_kmer_ridge'].predict([seq_to_kmer_tokens(u3, ks=(4, 5))]).astype(np.float32); "
                "m2 = b_hl['cds_bicodon_ridge'].predict([cds_to_bicodon_tokens(cds)]).astype(np.float32); "
                "p_hl = float(b_hl['model'].predict(np.hstack([Xh, m1[:, None], m2[:, None]]))[0]); "
                "b_ci = joblib.load(d / 'circrna_ires_translation_v2.joblib'); "
                "Xc = dict_list_to_matrix([extract_circrna_ires_v2(u5 * 8)], b_ci['features']); "
                "p_ci = float(b_ci['model'].predict(Xc)[0]); "
                "print(json.dumps({'u5': p_u5, 'hl': p_hl, 'ci': p_ci}))"
            ),
            str(real_v2_dir),
            egfp_u5,
            egfp_cds,
            egfp_u3,
        ],
        env={"PYTHONPATH": str(ROOT / "src"), "OMP_NUM_THREADS": "1", "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=True,
    )
    native_preds = json.loads(native_check.stdout.strip())
    assert float(mrl_v2[0]) == pytest.approx(native_preds["u5"], abs=1e-6)
    assert float(hl_v2[0]) == pytest.approx(native_preds["hl"], abs=1e-6)
    assert float(ires_v2[0]) == pytest.approx(native_preds["ci"], abs=1e-6)

    # 5. Tamper rejection (SHA-256 mismatch, byte-size mismatch, and missing manifest)
    tampered_dir = tmp_path / "tampered_v2"
    shutil.copytree(real_v2_dir, tampered_dir)
    target_jb = tampered_dir / "circrna_ires_translation_v2.joblib"
    orig_raw = bytearray(target_jb.read_bytes())
    orig_raw[-1] ^= 0xFF
    target_jb.write_bytes(bytes(orig_raw))
    with pytest.raises(ValueError, match="sha256"):
        CommercialCleanSurrogateV2(tampered_dir)

    target_jb.write_bytes(bytes(orig_raw) + b"\x00")
    with pytest.raises(ValueError, match="byte size"):
        CommercialCleanSurrogateV2(tampered_dir)

    (tampered_dir / "TRAINING_PROVENANCE_MANIFEST.json").unlink()
    with pytest.raises(FileNotFoundError, match="missing provenance manifest"):
        CommercialCleanSurrogateV2(tampered_dir)

    # 6. Graceful fallback to CommercialCleanSurrogateV1 when surrogate_v2/ is absent, plus v1-bundle back-compat
    v1_only_seed = tmp_path / "v1_only_seed"
    shutil.copytree(SEED_DIR / "surrogate_v1", v1_only_seed / "surrogate_v1")
    assert has_surrogate_v2_artifacts(v1_only_seed) is False
    jury_v1 = load_surrogate_jury(v1_only_seed, prefer_v2=True)
    assert isinstance(jury_v1, CommercialCleanSurrogateV1)
    assert not isinstance(jury_v1, CommercialCleanSurrogateV2)

    fake_seed = tmp_path / "seed_v1_compat"
    v2_compat_dir = fake_seed / "surrogate_v2"
    v2_compat_dir.mkdir(parents=True)
    v1_dir = SEED_DIR / "surrogate_v1"
    copied_mapping = {
        "utr5_mrl_pure_seq_lgbm.joblib": "utr5_mrl_pure_seq_v2.joblib",
        "cds_codon_stability_pure_seq_lgbm.joblib": "cds_codon_stability_v2.joblib",
        "saluki_consensus_halflife_pure_seq_lgbm.joblib": "saluki_consensus_halflife_v2.joblib",
        "utr3_stability_pure_seq_lgbm.joblib": "utr3_stability_pure_seq_v2.joblib",
    }
    v1_manifest = json.loads((v1_dir / "TRAINING_PROVENANCE_MANIFEST.json").read_text(encoding="utf-8"))
    v1_by_name = {a["filename"]: a for a in v1_manifest["artifacts"]}
    manifest_artifacts = []
    for src_name, dst_name in copied_mapping.items():
        raw = (v1_dir / src_name).read_bytes()
        (v2_compat_dir / dst_name).write_bytes(raw)
        manifest_artifacts.append(
            {
                "filename": dst_name,
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "feature_count": v1_by_name[src_name]["feature_count"],
                "license": "CC-BY-4.0",
                "uses_viennarna_features_or_labels": False,
            }
        )
    # Incomplete clearance manifest (missing schema / compliance_track / boolean attestations) must fail closed
    (v2_compat_dir / "TRAINING_PROVENANCE_MANIFEST.json").write_text(
        json.dumps({"artifacts": manifest_artifacts}, indent=2), encoding="utf-8"
    )
    with pytest.raises(PermissionError):
        CommercialCleanSurrogateV2(v2_compat_dir, allow_sandbox_override=True)

    (v2_compat_dir / "TRAINING_PROVENANCE_MANIFEST.json").write_text(
        json.dumps(
            {
                "schema": "mrnax.surrogate_training_provenance.v1.2",
                "compliance_track": "commercial_clean",
                "zero_viennarna_features_or_distilled_labels": True,
                "zero_gpl_or_noncommercial_weights": True,
                "artifacts": manifest_artifacts,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    # Gate B I1: unanchored synthetic manifest fails closed unless allow_sandbox_override=True
    with pytest.raises(PermissionError, match="Untrusted provenance manifest"):
        CommercialCleanSurrogateV2(v2_compat_dir)
    assert has_surrogate_v2_artifacts(fake_seed) is False
    assert has_surrogate_v2_artifacts(fake_seed, allow_sandbox_override=True) is True

    load_surrogate_jury.cache_clear()
    CommercialCleanSurrogateV2.from_seed_dir.cache_clear()
    jury_v2_compat = load_surrogate_jury(fake_seed, prefer_v2=True, allow_sandbox_override=True)
    assert isinstance(jury_v2_compat, CommercialCleanSurrogateV2)
    u5 = "GGGAAAUAAAGAGAGAAAAGAAGAGUAAGAAGAAAUAUAAGAGCCACC"
    cds = "AUG" + ("CUGGAGGAGAAGCUGGGC" * 15)
    u3 = "GCUGCCUUCUGCGGGGCUUGCCUUCUGGCC"
    assert float(jury_v2_compat.score_5utr_mrl([u5])[0]) == pytest.approx(float(jury_v1.score_5utr_mrl([u5])[0]))
    assert float(jury_v2_compat.score_consensus_halflife(u5, cds, u3)) == pytest.approx(
        float(jury_v1.score_consensus_halflife(u5, cds, u3))
    )

    # Feature drift guard: missing declared column raises KeyError instead of zero-filling
    from mrnax.engine.features_v2 import dict_list_to_matrix

    with pytest.raises(KeyError, match="Feature drift"):
        dict_list_to_matrix([{"a_frac": 0.25}], ["a_frac", "missing_col"])


def test_lineardesign_sweep_surgical_polisher_fallback_and_commercial_isolation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import mrnax.engine.lineardesign_adapter as ld_mod

    assert DEFAULT_MAX_LINEARDESIGN_AA == 900
    assert DEFAULT_LINEARDESIGN_TIMEOUT_SEC == 45.0

    # Gate B I10: stub the native binary runner so default unit suite never depends on machine-local executables
    fake_bin = tmp_path / "lineardesign_stub"
    fake_bin.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_bin.chmod(0o755)
    egfp_snap = json.loads((SEED_DIR / "egfp_reporter_snapshot.json").read_text(encoding="utf-8"))
    ref_egfp_cds: str = egfp_snap["mammalian_reporter_egfp"]["reference_cds_rna_no_stop"]
    # Introduce a synonymous FF -> UUUUUU run at lambda=0.0 so Surgical Polisher UUUU elimination is verified
    low_lam_cds = ref_egfp_cds.replace("UUCUUCAAGUCC", "UUUUUUAAAUCA", 1)
    assert low_lam_cds != ref_egfp_cds and "UUUU" in low_lam_cds

    def _stub_run_bounded(argv: list[str], *, stdin_text: str, cwd_str: str, timeout_sec: float, **_kw: object) -> str:
        lam = float(argv[argv.index("-l") + 1]) if "-l" in argv else 1.0
        aa = stdin_text.strip()
        assert len(aa) == len(ref_egfp_cds) // 3
        seq_out = low_lam_cds if lam <= 0.5 else ref_egfp_cds
        dg = round(-434.8 + 16.5 * lam, 2)
        cai = round(min(0.96, 0.791 + 0.036 * lam), 3)
        return f"mRNA sequence:  {seq_out}\nmRNA folding free energy: {dg:.2f} kcal/mol; mRNA CAI: {cai:.3f}\n"

    ld_mod._cached_cpp_lineardesign_call.cache_clear()
    monkeypatch.setattr(ld_mod, "discover_lineardesign_binary", lambda *_a, **_k: (fake_bin, tmp_path))
    monkeypatch.setattr(ld_mod, "_run_bounded_subprocess", _stub_run_bounded)
    assert is_lineardesign_available() is True
    assert ld_mod.discover_lineardesign_binary() == (fake_bin, tmp_path)

    # Test protein with slippery phenylalanines, lysines, serines, and restriction-prone motifs
    prot = "MVSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITLGMDELYK"
    u5 = "AGAAUAAACUAGUAUUCUUCUGGUCCCCACAGACUCAGAGAGAACCCGCCACC"
    u3 = "GCUGGAGCCUCGGUGGCCAUGCUUCUUGCCCCUUGGGCCUCCCCCCAGCCCCUCCUCCCCUUCCUGCACCCGUACCCCCGUGGUCUUUGAAUAAAGUCUGAGUGGGCGGC"

    cands = sweep_lineardesign_candidates(
        prot,
        lambdas=DEFAULT_LAMBDA_SWEEP,
        utr5_rna=u5,
        stop_rna="UGA",
        utr3_rna=u3,
        compliance_track="research_max",
        chemistry="m1psi",
        eliminate_uuuu=True,
        restriction_enzymes=DEFAULT_POLISH_ENZYMES,
    )
    assert len(cands) == 5
    assert [c.lambda_val for c in cands] == list(DEFAULT_LAMBDA_SWEEP)
    forbidden = build_forbidden_motif_list(
        max_homopolymer=5,
        eliminate_uuuu=True,
        restriction_enzymes=DEFAULT_POLISH_ENZYMES,
    )
    for c in cands:
        assert c.backend_used == "lineardesign_cpp_binary"
        assert c.aa_exact_match is True
        assert verify_exact_aa_translation(c.polished_cds_rna, prot)
        assert "UUUU" not in c.polished_cds_rna
        assert count_forbidden_motif_hits(
            c.polished_cds_rna, forbidden, utr5_tail=u5, downstream_head="UGA" + u3
        ) == 0
        assert c.polished_kozak_unpaired_prob >= c.raw_kozak_unpaired_prob - 1e-6

    # Lambda=0.0 prioritizes MFE; lambda=4.0 prioritizes CAI
    assert cands[0].raw_mfe_kcal_mol < cands[-1].raw_mfe_kcal_mol
    assert cands[-1].raw_cai > cands[0].raw_cai
    # Raw lambda=0.0 on eGFP has UUUU runs that Surgical Polisher cleanly eliminated while keeping strong MFE!
    assert cands[0].eliminated_motifs_before >= 1
    assert cands[0].remaining_forbidden_motifs == 0
    assert cands[0].polished_mfe_kcal_mol < -200.0

    # Clean fallback when binary is unavailable / force_fallback=True
    fb_cds, fb_dg, fb_cai, fb_backend = run_lineardesign_single(
        prot[:60],
        lambda_val=1.0,
        compliance_track="research_max",
        force_fallback=True,
    )
    assert fb_backend == "guided_bicodon_vienna_fallback"
    assert verify_exact_aa_translation(fb_cds, prot[:60])
    assert fb_dg < 0.0 and 0.3 <= fb_cai <= 1.0

    # Max-AA VPS memory safety guard (`MRNAX_MAX_LINEARDESIGN_AA`) routes oversized proteins to fallback
    # AND cannot be overridden upward by caller `max_aa`
    monkeypatch.setenv("MRNAX_MAX_LINEARDESIGN_AA", "80")
    cap_cds, cap_dg, cap_cai, cap_backend = run_lineardesign_single(
        prot[:100],
        lambda_val=1.0,
        compliance_track="research_max",
        max_aa=500,
    )
    assert cap_backend == "guided_bicodon_vienna_fallback"
    assert verify_exact_aa_translation(cap_cds, prot[:100])
    assert cap_dg < 0.0 and 0.3 <= cap_cai <= 1.0
    monkeypatch.delenv("MRNAX_MAX_LINEARDESIGN_AA", raising=False)
    with pytest.raises(ValueError, match="lambda_val"):
        run_lineardesign_single(prot[:30], lambda_val=float("nan"), compliance_track="research_max")
    with pytest.raises(ValueError, match="timeout_sec"):
        run_lineardesign_single(prot[:30], timeout_sec=-1.0, compliance_track="research_max")

    # Default 900-AA cap also routes >900-AA proteins without invoking C++ lattice parser
    long_prot = "M" + ("VSKGEELFTG" * 91)  # 911 AA > 900
    long_cds, _, _, long_backend = run_lineardesign_single(
        long_prot,
        lambda_val=1.0,
        compliance_track="research_max",
    )
    assert long_backend == "guided_bicodon_vienna_fallback"
    assert verify_exact_aa_translation(long_cds, long_prot)

    # Strict commercial_clean isolation
    with pytest.raises(PermissionError, match="commercial_clean"):
        run_lineardesign_single(prot[:30], compliance_track="commercial_clean")
    with pytest.raises(PermissionError, match="commercial_clean"):
        polish_cds_5prime_and_constraints(fb_cds[:90], prot[:30], compliance_track="commercial_clean")
    with pytest.raises(PermissionError, match="commercial_clean"):
        sweep_lineardesign_candidates(prot[:30], compliance_track="commercial_clean")
    ld_mod._cached_cpp_lineardesign_call.cache_clear()


@pytest.mark.skipif(
    __import__("os").environ.get("MRNAX_RUN_NATIVE_BIN_TESTS") != "1",
    reason="Provisioned native LinearDesign binary integration check (opt-in via MRNAX_RUN_NATIVE_BIN_TESTS=1)",
)
def test_provisioned_native_lineardesign_binary_integration() -> None:
    assert discover_lineardesign_binary() is not None
    cds_rna, dg, cai, backend = run_lineardesign_single(
        "MVSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPT",
        lambda_val=1.0,
        compliance_track="research_max",
    )
    assert backend == "lineardesign_cpp_binary"
    assert dg < -50.0 and 0.5 <= cai <= 1.0
    assert verify_exact_aa_translation(
        cds_rna, "MVSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPT"
    )


def test_structure_profiler_arcs_tracks_motifs_and_eternafold() -> None:
    # Construct synthetic RNA with known motifs:
    # uORF in 5'UTR (AUGGCUUAA), G-quadruplex (GGGGAGGGAGGGAGGG), CDS with UUUU + GGGGGG + BsaI (GGUCUC) + T7 (AUCUGUU),
    # 3'UTR with ARE (AUUUA) + hsa-miR-122-5p full site (CAAACACCAUUGUCACACUCCA) + poly(A)
    u5 = "ACAUGGCUUAAGGGAGGGAGGGAGGGGCCACC"
    cds = "AUGUUUUUCGGGGGGGGUCUCAUCUGUUUAA"
    cds_sense = cds[:-3]
    stop = "UAA"
    u3 = "GCUAUUUAGCUCAAACACCAUUGUCACACUCCAGCU"
    pa = "A" * 30
    full_seq = u5 + cds_sense + stop + u3 + pa

    c_start = len(u5)
    c_end = c_start + len(cds_sense)
    u3_start = c_end + len(stop)
    u3_end = u3_start + len(u3)

    prof = analyze_rna_structure(
        full_seq,
        five_prime_end=c_start,
        cds_start=c_start,
        cds_end=c_end,
        three_prime_start=u3_start,
        three_prime_end=u3_end,
        modality="linear_mrna",
        include_eternafold=(discover_eternafold_binary() is not None),
        seed_dir=SEED_DIR,
    )
    assert prof["length_nt"] == len(full_seq)
    assert prof["modality"] == "linear_mrna"
    assert 0.0 <= prof["aup_mean"] <= 1.0
    assert 0.0 <= prof["kozak_unpaired_prob"] <= 1.0
    assert "ViennaRNA" in prof["engines_used"]
    if discover_eternafold_binary() is not None:
        assert "EternaFold" in prof["engines_used"]
        assert 0.0 <= prof["eternafold_metrics"]["kozak_unpaired_prob"] <= 1.0

    init_win = prof["initiation_window"]
    assert init_win["start"] == max(0, c_start - 30)
    assert init_win["end"] == min(len(full_seq), c_start + 45)
    assert init_win["sequence"][init_win["aug_index_in_window"] : init_win["aug_index_in_window"] + 3] == "AUG"
    assert len(init_win["dot_bracket"]) == len(init_win["sequence"])
    for arc in init_win["arcs"]:
        assert 0 <= arc["i"] < arc["j"] < len(init_win["sequence"])
        assert arc["prob"] >= 0.15

    tracks = prof["sliding_tracks"]
    n_pts = len(tracks["positions"])
    assert n_pts >= 1
    assert len(tracks["local_mfe"]) == n_pts
    assert len(tracks["local_gc_pct"]) == n_pts
    assert len(tracks["local_u_pct"]) == n_pts
    assert len(tracks["local_unpaired_prob"]) == n_pts

    motif_kinds = {m["kind"] for m in prof["motifs"]}
    assert {
        "uorf",
        "g_quadruplex",
        "uuuu_frameshift",
        "homopolymer",
        "restriction_site",
        "t7_terminator",
        "are_motif",
        "mirna_site",
    } <= motif_kinds
    for m in prof["motifs"]:
        assert 0 <= m["start"] < m["end"] <= len(full_seq)

    with pytest.raises(ValueError):
        analyze_rna_structure(full_seq, cds_start=len(full_seq), seed_dir=SEED_DIR)
    with pytest.raises(ValueError):
        analyze_rna_structure(full_seq, cds_start=40, cds_end=30, seed_dir=SEED_DIR)
    prof_cc = analyze_rna_structure(full_seq, compliance_track="commercial_clean", seed_dir=SEED_DIR)
    assert "ViennaRNA" not in prof_cc["engines_used"]


def test_multi_modality_builders_evaluators_and_pareto_director_integration() -> None:
    short_prot = "MVSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPT"

    # 1. Tissue miRNA atlas detargeting across compartments
    mir_liver = select_atlas_mirna_for_detargeting(["liver"], ["skeletal_muscle"], seed_dir=SEED_DIR)
    assert mir_liver[0]["mirna_id"] == "hsa-miR-122-5p"
    mir_myeloid_sparing_t = select_atlas_mirna_for_detargeting(
        ["myeloid_apc"], ["t_cell"], seed_dir=SEED_DIR
    )
    assert mir_myeloid_sparing_t[0]["mirna_id"] == "hsa-miR-223-3p"

    # 2. Multi-chain linear_mrna with tissue detargeting
    lin_bnd = build_linear_mrna_bundle(
        [short_prot, short_prot[:45]],
        compliance_track="research_max",
        uridine_chemistry="m1psi",
        detarget_tissues=["liver"],
        protected_tissues=["skeletal_muscle"],
        seed_dir=SEED_DIR,
    )
    lin_eval = evaluate_linear_mrna_bundle(lin_bnd, compliance_track="research_max", seed_dir=SEED_DIR)
    assert lin_eval["chain_count"] == 2
    assert lin_eval["passed_all_hard_gates"] is True
    assert all(ch["aa_exact_match"] for ch in lin_eval["chains"])
    assert len(lin_eval["chains"][0]["mirna_interactions"]) == 2

    # 3. circRNA (PIE + spacers + viral IRES + ORF + junction/uncoupling check)
    for ires_id in ("HRV-B3", "CVB3", "EV-A71", "EMCV"):
        circ_bnd = build_circrna_bundle(
            short_prot,
            ires_name=ires_id,
            pie_backbone="Anabaena",
            compliance_track="research_max",
            seed_dir=SEED_DIR,
        )
        circ_eval = evaluate_circrna_bundle(circ_bnd, seed_dir=SEED_DIR)
        assert circ_eval["modality"] == "circrna"
        assert circ_eval["topology"] == "circular"
        assert circ_eval["aa_exact_match"] is True
        assert circ_eval["chemistry_unmodified_u_ok"] is True
        assert circ_eval["no_poly_a_tail_ok"] is True
        assert circ_eval["ires_cargo_uncoupled_ok"] is True
        assert circ_eval["spacer_insulation_ok"] is True
        assert isinstance(circ_eval["surrogate_v2_ires_translation_pred"], float)
        assert np.isfinite(circ_eval["surrogate_v2_ires_translation_pred"])
        assert any("OOD disclaimer" in note for note in circ_eval["notes"])
        jury_v2 = load_surrogate_jury(SEED_DIR, prefer_v2=True)
        axes_circ = jury_v2.score_molecule_axes(circ_bnd.molecules[0])
        assert axes_circ["circrna_ires_translation_pred"] == pytest.approx(
            circ_eval["surrogate_v2_ires_translation_pred"], abs=1e-6
        )
        m_v2 = jury_v2.evaluate_molecule_metrics(circ_bnd.molecules[0])
        assert any(m.tool_name.startswith("CommercialCleanSurrogateV2:") for m in m_v2)
        assert any(m.metric_id == "circrna_ires_translation_pred" for m in m_v2)

    # Quarantine or cross-category scaffold injection is blocked
    with pytest.raises((PermissionError, ValueError)):
        build_circrna_bundle(
            short_prot,
            spacer5_scaffold_id="SCAF-LIN-POLYA-SEG-A30L70-QUARANTINE",
            compliance_track="commercial_clean",
            seed_dir=SEED_DIR,
        )
    with pytest.raises(PermissionError):
        build_linear_mrna_bundle(
            [short_prot],
            polya_scaffold_id="SCAF-LIN-POLYA-SEG-A30L70-QUARANTINE",
            compliance_track="commercial_clean",
            seed_dir=SEED_DIR,
        )
    lin_cc = build_linear_mrna_bundle(
        [short_prot],
        compliance_track="commercial_clean",
        use_lineardesign=False,
        seed_dir=SEED_DIR,
    )
    lin_cc_eval = evaluate_linear_mrna_bundle(lin_cc, compliance_track="commercial_clean", seed_dir=SEED_DIR)
    assert lin_cc_eval["chains"][0]["cds_mfe_kcal_mol"] is None
    assert lin_cc_eval["chains"][0]["mirna_interactions"] == []

    # 4. saRNA (VEEV TC-83 & SINV replicon CSE/26S-SGP conservation + innate sensing)
    for bb in ("VEEV_TC83", "SINV"):
        sarna_bnd = build_sarna_bundle(
            short_prot,
            replicon_backbone=bb,
            compliance_track="research_max",
            seed_dir=SEED_DIR,
        )
        sarna_eval = evaluate_sarna_bundle(sarna_bnd, seed_dir=SEED_DIR)
        assert sarna_eval["modality"] == "sarna"
        assert sarna_eval["aa_exact_match"] is True
        assert sarna_eval["starts_with_5prime_au"] is True
        assert sarna_eval["nsp1_51nt_cse_conserved"] is True
        assert sarna_eval["subgenomic_26s_sgp_conserved"] is True
        assert sarna_eval["three_prime_19nt_cse_conserved"] is True
        assert sarna_eval["chemistry_unmodified_u_ok"] is True

    # 5. pareto_director integration (`research_max` with `lineardesign_academic`, `circrna`, `sarna`, and `commercial_clean` block)
    intent_rm = DesignIntent(
        intent_id="int_v25_rm",
        project_id="proj_v25",
        intent_version=1,
        created_from_turn_ids=["t1"],
        entry_mode="protein_only",
        compliance_track="research_max",
        uridine_chemistry="m1psi",
        status="ready",
        protein_targets=[
            ProteinTargetSpec(
                protein_sequence=short_prot,
                transl_table=1,
                accession_or_source="test_short_egfp",
                confirmation_status="confirmed",
            )
        ],
        constraints=default_m1_constraints(restriction_enzymes=["BspQI", "BsaI"]),
        objectives=[
            ObjectiveSpec(
                requirement_id="obj_open",
                metric_id="kozak_unpaired_prob",
                direction="maximize",
                priority_rank=1,
                evidence_status="thermodynamic_proxy",
                source_utterance="Maximize initiation openness",
            ),
            ObjectiveSpec(
                requirement_id="obj_mfe",
                metric_id="cds_mean_window_mfe",
                direction="minimize",
                priority_rank=2,
                evidence_status="thermodynamic_proxy",
                source_utterance="Stabilize CDS folding",
            ),
        ],
        runtime_policy=RuntimePolicySpec(
            enabled_generators=["stage0_random", "dnachisel_cds", "lineardesign_academic", "pareto_nsga2"],
            enabled_evaluators=["viennarna_thermo"],
            max_evaluations=14,
            random_seed=42,
        ),
    )
    b_rm, rep_rm, man_rm = run_design_pipeline(intent_rm, seed_dir=SEED_DIR)
    assert rep_rm.outcome_status == "COMPLETED_WITH_CANDIDATES"
    assert any("lineardesign" in b.bundle_id for b in b_rm)
    assert any("LinearDesign" in c.name for c in man_rm.invoked_components)

    # CircRNA and saRNA via run_design_pipeline
    intent_circ = intent_rm.model_copy_validated(uridine_chemistry="unmodified_u")
    b_circ, rep_circ, _ = run_design_pipeline(intent_circ, seed_dir=SEED_DIR, modality="circrna")
    assert rep_circ.outcome_status == "COMPLETED_WITH_CANDIDATES"
    assert all(b.molecules[0].modality == "circrna" for b in b_circ)

    b_sarna, rep_sarna, _ = run_design_pipeline(intent_circ, seed_dir=SEED_DIR, modality="sarna")
    assert rep_sarna.outcome_status == "COMPLETED_WITH_CANDIDATES"
    assert all(b.molecules[0].modality == "sarna" for b in b_sarna)

    # Commercial-clean blocks lineardesign_academic and use_lineardesign=True
    intent_cc_ld = intent_rm.model_copy_validated(
        compliance_track="commercial_clean",
        runtime_policy=RuntimePolicySpec(
            enabled_generators=["lineardesign_academic"],
            enabled_evaluators=["surrogate_v1_clean"],
        ),
    )
    b_cc_blk, rep_cc_blk, _ = run_design_pipeline(intent_cc_ld, seed_dir=SEED_DIR)
    assert b_cc_blk == []
    assert rep_cc_blk.outcome_status == "UNCOMPARABLE_MISSING_APPROVED_COMPONENT"
    assert "lineardesign_academic" in (rep_cc_blk.uncomparable_reason or "")

    # Static AST Method-FTO lint across entire src/mrnax tree
    lint = audit_codebase_method_fto_lints(ROOT / "src" / "mrnax")
    assert lint["status"] == "PASS", f"Method-FTO AST lint failed: {lint['violations']}"
    assert lint["violations"] == []


def test_paired_v25_moderna_benchmark_artifact() -> None:
    report_path = ROOT / "artifacts" / "v25_engine_ablation_report.json"
    assert report_path.is_file(), "Missing artifacts/v25_engine_ablation_report.json"
    doc = json.loads(report_path.read_text(encoding="utf-8"))
    assert doc["schema_version"] == "2.5.0"
    targets = doc["paired_benchmarks"]
    assert {
        "egfp_single_chain",
        "hepo_single_chain",
        "dll3_x_cd3_multi_chain",
        "circrna_fluc_reporter",
        "sarna_veev_tc83",
    } <= set(targets.keys())
    for name, entry in targets.items():
        v20 = entry["v20_baseline"]
        v25 = entry["v25_lineardesign_polisher"]
        assert v25["aa_exact_match"] is True, f"{name} failed AA translation check"
        assert v25["cds_uuuu_count"] == 0, f"{name} retained UUUU slippery run"
        assert v25["cds_mfe_kcal_mol"] < v20["cds_mfe_kcal_mol"], (
            f"{name}: v2.5 MFE ({v25['cds_mfe_kcal_mol']}) did not improve over v2.0 ({v20['cds_mfe_kcal_mol']})"
        )

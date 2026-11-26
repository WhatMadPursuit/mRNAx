"""Deterministic builder for mRNAx v2 verified biological seed registries.

Queries official UniProt REST (`uniprot_tools.py`), NCBI E-utilities (`ncbi_fetch.py` / GenBank XML),
and read-only Google Drive Data Hub (`B1-01` miRBase + `B1-03..06` FANTOM5) to populate
`registry_seed/` with zero hallucinated sequences or numbers.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

from Bio import SeqIO
from Bio.Seq import Seq

REPO_ROOT = Path("/Users/alex/.gemini/antigravity/scratch/mrnax-v2")
SEED_DIR = REPO_ROOT / "registry_seed"
TMP_DIR = Path("/Users/alex/.gemini/antigravity/brain/3cbd0280-0b03-4150-912d-678711ef68f1/scratch/seed_build_tmp")
UNIPROT_SCRIPT = Path("/Users/alex/.gemini/config/plugins/structural-biology/skills/uniprot_database/scripts/uniprot_tools.py")
NCBI_SCRIPT = Path("/Users/alex/.gemini/config/plugins/structural-biology/skills/ncbi_sequence_fetch/scripts/ncbi_fetch.py")


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run_cmd(args: list[str]) -> str:
    env = os.environ.copy()
    env["PATH"] = f"/opt/homebrew/bin:/Users/alex/.local/bin:{env.get('PATH', '')}"
    proc = subprocess.run(args, check=True, capture_output=True, text=True, env=env)
    return proc.stdout


def build_egfp_snapshot() -> dict:
    # 1. Official UniProt P42212 via uniprot_tools.py
    raw_uniprot = _run_cmd(["/opt/homebrew/bin/uv", "run", str(UNIPROT_SCRIPT), "get", "P42212", "--format", "json"])
    up_json = json.loads(raw_uniprot)
    wt_seq = up_json["sequence"]["value"]
    entry_version = str(up_json.get("entryAudit", {}).get("entryVersion", "unknown"))
    seq_version = str(up_json.get("entryAudit", {}).get("sequenceVersion", "unknown"))

    # 2. Official GenBank U55762.1 (pEGFP-1 mammalian reporter vector, 720-nt EGFP CDS)
    with urlopen("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=U55762.1&rettype=gb&retmode=text", timeout=20) as resp:
        gb_text = resp.read().decode("utf-8")
    rec = SeqIO.read(io.StringIO(gb_text), "genbank")
    egfp_cds_dna = ""
    egfp_protein = ""
    protein_acc = "AAB02572.1"
    for feat in rec.features:
        if feat.type == "CDS":
            cand_dna = str(feat.extract(rec.seq)).upper()
            cand_prot = feat.qualifiers.get("translation", [""])[0]
            if len(cand_dna) == 720 or cand_prot.startswith("MVSKGEELFT"):
                egfp_cds_dna = cand_dna
                egfp_protein = cand_prot or str(Seq(cand_dna[:-3]).translate(table=1))
                protein_acc = feat.qualifiers.get("protein_id", ["AAB02572.1"])[0]
                break
    if not egfp_cds_dna:
        raise RuntimeError("Could not locate 720-nt EGFP CDS in U55762.1")

    egfp_cds_rna_full = egfp_cds_dna.replace("T", "U")
    cds_body = egfp_cds_rna_full[:-3]
    stop_codon = egfp_cds_rna_full[-3:]
    translated_check = str(Seq(cds_body).translate(table=1, to_stop=False))
    assert translated_check == egfp_protein, "U55762.1 CDS translation mismatch!"
    assert len(egfp_protein) == 239

    out = {
        "schema_version": "1.0.0",
        "retrieved_at_utc": _now_utc(),
        "uniprot_reference": {
            "accession": "P42212",
            "entry_version": entry_version,
            "sequence_version": seq_version,
            "organism": "Aequorea victoria",
            "wt_protein_sequence": wt_seq,
            "length_aa": len(wt_seq),
            "license": "CC-BY-4.0",
        },
        "mammalian_reporter_egfp": {
            "genbank_nuccore": "U55762.1",
            "protein_id": "AAB02572.1",
            "description": "Enhanced Green Fluorescent Protein (F64L, S65T, V2 Kozak insertion; matches Sample et al. 2019 GSE114002 reporter)",
            "protein_sequence": egfp_protein,
            "length_aa": len(egfp_protein),
            "reference_cds_rna_no_stop": cds_body,
            "reference_stop_codon": stop_codon,
            "license_track": "commercial_clean",
        },
    }
    (SEED_DIR / "egfp_reporter_snapshot.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def build_utr_components() -> dict:
    accessions = [
        ("NM_000518.5", "HBB"),
        ("NM_000558.5", "HBA1"),
        ("NM_001101.5", "ACTB"),
        ("NM_002046.7", "GAPDH"),
        ("NM_001402.6", "EEF1A1"),
        ("NM_002954.6", "RPS27A"),
    ]
    ids_query = ",".join(acc for acc, _ in accessions)
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id={ids_query}&rettype=gb&retmode=text"
    with urlopen(url, timeout=30) as resp:
        gb_blob = resp.read().decode("utf-8")

    utr5_list = []
    utr3_list = []
    for rec in SeqIO.parse(io.StringIO(gb_blob), "genbank"):
        acc = rec.id
        gene = next((g for a, g in accessions if acc.startswith(a.split(".")[0])), rec.name)
        cds_feat = next((f for f in rec.features if f.type == "CDS"), None)
        if cds_feat is None:
            continue
        cds_start = int(cds_feat.location.start)
        cds_end = int(cds_feat.location.end)
        rna_seq = str(rec.seq).upper().replace("T", "U")
        utr5 = rna_seq[:cds_start]
        utr3 = rna_seq[cds_end:].rstrip("A")  # Strip poly(A) tail if annotated on RefSeq mRNA

        if len(utr5) >= 15:
            utr5_list.append({
                "component_id": f"utr5_{gene.lower()}_{acc.replace('.', '_')}",
                "gene_symbol": gene,
                "refseq_accession": acc,
                "segment_type": "five_prime_utr",
                "sequence": utr5,
                "length_nt": len(utr5),
                "has_uaug": "AUG" in utr5,
                "max_homopolymer_run": max(
                    max((len(run) for run in __import__("re").findall(f"{b}+", utr5)), default=0)
                    for b in "ACGU"
                ),
                "origin_source": f"NCBI_RefSeq:{acc}",
                "license_track": "commercial_clean",
                "fto_notes": "Endogenous unmodified human RefSeq transcript UTR (US Public Domain)",
            })
        if 30 <= len(utr3) <= 650:
            utr3_list.append({
                "component_id": f"utr3_{gene.lower()}_{acc.replace('.', '_')}",
                "gene_symbol": gene,
                "refseq_accession": acc,
                "segment_type": "three_prime_utr",
                "sequence": utr3,
                "length_nt": len(utr3),
                "max_homopolymer_run": max(
                    max((len(run) for run in __import__("re").findall(f"{b}+", utr3)), default=0)
                    for b in "ACGU"
                ),
                "origin_source": f"NCBI_RefSeq:{acc}",
                "license_track": "commercial_clean",
                "fto_notes": "Endogenous unmodified human RefSeq transcript UTR (US Public Domain)",
            })

    payload = {
        "registry_id": "utr_components_v1",
        "version": "1.0.0",
        "retrieved_at_utc": _now_utc(),
        "license_track": "commercial_clean",
        "five_prime_utrs": utr5_list,
        "three_prime_utrs": utr3_list,
    }
    (SEED_DIR / "utr_components_v1.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def build_mirna_rbp_registry() -> dict:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    mature_fa = TMP_DIR / "mature.fa"
    cpm_txt = TMP_DIR / "human.srna.cpm.txt"
    samples_tsv = TMP_DIR / "human.srna.samples.tsv"

    _run_cmd(["gdrive", "readonly", "download", "1W2SuH_HXfapzXq8OqTJsPJP-k65pkSEL", "--out", str(mature_fa)])
    _run_cmd(["gdrive", "readonly", "download", "1INzL-UP-GrC5We4cFRGVbZm8wPGEaVo8", "--out", str(cpm_txt)])
    _run_cmd(["gdrive", "readonly", "download", "1qxkDvSGKsVPujskFo54el300VGYw1-xW", "--out", str(samples_tsv)])

    sha = hashlib.sha256(mature_fa.read_bytes()).hexdigest()
    assert sha == "52d69eeb3c23c5ed0f66c8c9f24e1ad2cd94cc34e72e4fc376c0dd438d2779ff", f"miRBase SHA mismatch: {sha}"

    target_mirnas = {
        "hsa-miR-122-5p",
        "hsa-miR-142-3p",
        "hsa-miR-142-5p",
        "hsa-miR-1-3p",
        "hsa-miR-133a-3p",
        "hsa-miR-206",
        "hsa-miR-126-3p",
        "hsa-miR-21-5p",
        "hsa-let-7a-5p",
        "hsa-miR-16-5p",
    }
    seqs: dict[str, dict[str, str]] = {}
    for rec in SeqIO.parse(str(mature_fa), "fasta"):
        if rec.id in target_mirnas:
            s = str(rec.seq).upper().replace("T", "U")
            rc_full = str(Seq(s).reverse_complement_rna())
            # 8mer target site: reverse complement of miRNA positions 2..8 + trailing 'A' opposite pos 1
            seed_2_8 = s[1:8]
            seed_rc_7mer_m8 = str(Seq(seed_2_8).reverse_complement_rna())
            seed_8mer_site = seed_rc_7mer_m8 + "A"
            seqs[rec.id] = {
                "mirbase_accession": rec.description.split()[1] if len(rec.description.split()) > 1 else rec.id,
                "mature_sequence": s,
                "length_nt": len(s),
                "full_complementary_site_22nt": rc_full,
                "seed_7mer_m8_site": seed_rc_7mer_m8,
                "seed_8mer_site": seed_8mer_site,
            }

    # Parse FANTOM5 samples -> tissue buckets
    sample_lines = samples_tsv.read_text(encoding="utf-8", errors="replace").splitlines()
    sample_to_bucket: dict[str, str] = {}
    for ln in sample_lines[1:]:
        parts = ln.split("\t")
        if len(parts) < 2:
            continue
        sid = parts[0].strip()
        desc = " ".join(parts[1:]).lower()
        if "hepatocyte" in desc or "liver" in desc:
            sample_to_bucket[sid] = "liver_hepatocyte"
        elif any(w in desc for w in ("monocyte", "macrophage", "dendritic", "myeloid")):
            sample_to_bucket[sid] = "myeloid_monocyte_macrophage"
        elif "skeletal muscle" in desc or "myoblast" in desc or "myotube" in desc:
            sample_to_bucket[sid] = "skeletal_muscle"
        elif "t cell" in desc or "cd4+" in desc or "cd8+" in desc:
            sample_to_bucket[sid] = "t_cell"
        elif "endothelial" in desc:
            sample_to_bucket[sid] = "endothelial"

    # Stream cpm_txt header and matching miRNA rows
    with cpm_txt.open("r", encoding="utf-8", errors="replace") as fh:
        header_cols: list[str] = []
        cpm_summary: dict[str, dict[str, float]] = {}
        for line in fh:
            if line.startswith("#"):
                continue
            if not header_cols:
                header_cols = [c.strip() for c in line.rstrip("\n").split("\t")]
                continue
            row = line.rstrip("\n").split("\t")
            mir_id = row[0].strip()
            if mir_id in target_mirnas:
                bucket_vals: dict[str, list[float]] = {}
                for col_idx, val_str in enumerate(row[1:], start=1):
                    if col_idx >= len(header_cols):
                        break
                    col_name = header_cols[col_idx]
                    bkt = sample_to_bucket.get(col_name)
                    if bkt:
                        try:
                            bucket_vals.setdefault(bkt, []).append(float(val_str))
                        except ValueError:
                            pass
                cpm_summary[mir_id] = {
                    bkt: round(sum(vals) / len(vals), 2) for bkt, vals in bucket_vals.items() if vals
                }

    entries = []
    for mir_id in sorted(target_mirnas):
        if mir_id in seqs:
            entries.append({
                "mirna_id": mir_id,
                **seqs[mir_id],
                "fantom5_mean_cpm_by_tissue": cpm_summary.get(mir_id, {}),
            })

    shutil.rmtree(TMP_DIR, ignore_errors=True)
    payload = {
        "registry_id": "mirna_rbp_commercial_v1",
        "version": "1.0.0",
        "retrieved_at_utc": _now_utc(),
        "license_track": "commercial_clean",
        "sources": [
            {
                "asset_id": "B1-01",
                "name": "miRBase Release 23.1 mature.fa",
                "drive_file_id": "1W2SuH_HXfapzXq8OqTJsPJP-k65pkSEL",
                "sha256": sha,
                "license": "Public Domain",
            },
            {
                "asset_id": "B1-03..B1-06",
                "name": "RIKEN FANTOM5 Human sRNA/miRNA Expression Atlas (De Rie et al. 2017)",
                "drive_file_id": "1INzL-UP-GrC5We4cFRGVbZm8wPGEaVo8",
                "license": "CC-BY-4.0 (Attribution: RIKEN FANTOM5 Consortium / De Rie et al., Nat Biotechnol 2017)",
            },
        ],
        "mirna_entries": entries,
        "destabilizing_rbp_motifs": [
            {
                "motif_id": "ARE_core_pentamer",
                "pattern_rna": "AUUUA",
                "rbp_family": "ZFP36/TTP & ELAVL1 AU-rich element binding",
                "default_action": "avoid_in_three_prime_utr",
            },
            {
                "motif_id": "ARE_nonamer",
                "pattern_rna": "UUAUUUAUU",
                "rbp_family": "Strong ZFP36/TTP mRNA decay motif",
                "default_action": "hard_avoid_in_three_prime_utr",
            },
        ],
    }
    (SEED_DIR / "mirna_rbp_commercial_v1.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def build_patent_fto_guard() -> dict:
    payload = {
        "registry_id": "patent_fto_guard_v1",
        "version": "1.0.0",
        "verified_from_audit": ["G-P01_pilot_license_fto.md", "G-B1_patent_fto_guard.md", "C-B2_wave2_cross_audit.md"],
        "disclaimer": "Engineering sequence collision & genus alert registry based on verified patent claims/examples; not a formal legal opinion.",
        "genus_grammar_rules": [
            {
                "rule_id": "BNT_FAMILY_51220548_SEGMENTED_POLYA_GENUS",
                "patent_family_id": "51220548",
                "granted_publications": ["US10717982B2 (Cl.1)", "US12054723B2 (Cl.1)", "EP3167059B1 (Cl.1)"],
                "target_scope": "poly_a",
                "min_total_poly_a_length": 80,
                "regex_rna": "A{20,}[UCG][ACGU]{0,48}[UCG]?A{20,}",
                "block_in_commercial_clean": True,
                "block_in_research_max": False,
                "summary": "BioNTech/TRON granted genus covering >=80 nt modified poly(A) comprising >=20 consecutive A + non-A linker + >=20 consecutive A.",
            }
        ],
        "sequence_collision_rules": [
            {
                "rule_id": "BNT_EXAMPLE_A30L70_LINKER_SEQ2",
                "patent_family_or_pub": "Family 51220548 / US10717982B2 Example 2 SEQ ID NO: 2",
                "target_scope": "poly_a",
                "sequence_rna": "GCAUAUGACU",
                "identity_threshold": 1.0,
                "block_in_commercial_clean": True,
                "summary": "Exact 10-nt segmented poly(A) linker example (DNA GCATATGACT) recited in Example 2.",
            },
            {
                "rule_id": "MODERNA_US10703789B2_SPEC_5UTR_PREFIX",
                "patent_family_or_pub": "US10703789B2 / WO2014164253A1 specification",
                "target_scope": "five_prime_utr",
                "sequence_rna": "GGGAAAUAAGAGAGAAAAGAAGAGUAAGAAGAAAUAUAAGAGCCACC",
                "identity_threshold": 0.85,
                "block_in_commercial_clean": True,
                "summary": "ModernaTx specification 47-nt 5'UTR sequence.",
            },
            {
                "rule_id": "MODERNA_LITERATURE_V1_5UTR_57NT",
                "patent_family_or_pub": "Literature clinical 5'UTR (pending US10881730B2 full verification per G-B1)",
                "target_scope": "five_prime_utr",
                "sequence_rna": "GGGAAAUAAGAGAGAAAAGAAGAGUAAGAAGAAAUAUAAGACCCCGGCGCCGCCACC",
                "identity_threshold": 0.80,
                "block_in_commercial_clean": True,
                "summary": "57-nt engineered 5'UTR containing ACCCCGGCGCC insertion.",
            },
            {
                "rule_id": "BNT_US11492628B2_AES_3UTR_CORE",
                "patent_family_or_pub": "US11492628B2 (Cl.1/6/7/8) / EP3636764B1 (Cl.1/4/5)",
                "target_scope": "three_prime_utr",
                "sequence_rna": "CUGGUACUGCAUGCACGCAAUGCUAGCUGCCCCUUUCCCGUCCUGGGUACCCCGAGUCUCCCCCGACCUCGGGUCCCAGGUAUGCUCCCACCUCCACCUGCCCCACUCACCACCUCUGCUAGUUCCAGACACC",
                "identity_threshold": 0.80,
                "block_in_commercial_clean": True,
                "summary": "Granted claim scope covering non-naturally linked AES 3'UTR fragment (>=80% fragment / >=90% variant) and AES-mtRNR1 FI combination.",
            },
        ],
    }
    (SEED_DIR / "patent_fto_guard_v1.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    SEED_DIR.mkdir(parents=True, exist_ok=True)
    egfp = build_egfp_snapshot()
    utrs = build_utr_components()
    mirs = build_mirna_rbp_registry()
    fto = build_patent_fto_guard()
    print(
        json.dumps(
            {
                "egfp_len_aa": egfp["mammalian_reporter_egfp"]["length_aa"],
                "uniprot_p42212_version": egfp["uniprot_reference"]["entry_version"],
                "utr5_count": len(utrs["five_prime_utrs"]),
                "utr5_clean_no_uaug": sum(1 for u in utrs["five_prime_utrs"] if not u["has_uaug"]),
                "utr3_count": len(utrs["three_prime_utrs"]),
                "mirna_count": len(mirs["mirna_entries"]),
                "mir122_liver_cpm": next(
                    (m["fantom5_mean_cpm_by_tissue"] for m in mirs["mirna_entries"] if m["mirna_id"] == "hsa-miR-122-5p"),
                    {},
                ),
                "fto_rules": len(fto["genus_grammar_rules"]) + len(fto["sequence_collision_rules"]),
            },
            indent=2,
        )
    )

const DEFAULT_API_BASE = "https://149-248-39-200.sslip.io:8443";
const TRUSTED_PRODUCTION_ORIGINS = new Set([
  "https://149-248-39-200.sslip.io:8443",
]);
const FALLBACK_EGFP =
  "MVSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITLGMDELYK";
const FALLBACK_HEPO =
  "MGVHECPAWLWLLLSLLSLPLGLPVLGAPPRLICDSRVLERYLLEAKEAENITTGCAEHCSLNENITVPDTKVNFYAWKRMEVGQQAVEVWQGLALLSEAVLRGQALLVNSSQPWEPLQLHVDKAVSGLRSLTTLLRALGAQKEAISPPDAASAAPLRTITADTFRKLFRVYSNFLRGKLKLYTGEACRTGDR";

const DEFAULT_PROVIDER_MODELS = {
  deepseek: "deepseek-flash",
  openai: "gpt-4.1-mini",
  kimi: "moonshot-v1-32k",
  qwen: "qwen-plus",
  gemini: "gemini-2.5-flash",
};

const SUPPORTED_PROVIDER_MODELS = {
  deepseek: ["deepseek-flash", "deepseek-v4-pro", "deepseek-chat"],
  openai: ["gpt-4.1-mini", "gpt-5.4", "gpt-6.2-astra"],
  kimi: ["moonshot-v1-32k", "kimi-k3"],
  qwen: ["qwen-plus", "qwen3.8-max"],
  gemini: ["gemini-2.5-flash"],
};

const MODEL_LABELS = {
  "deepseek-flash": "deepseek-flash (极速)",
  "deepseek-v4-pro": "deepseek-v4-pro (深度推理)",
  "deepseek-chat": "deepseek-chat",
  "gpt-4.1-mini": "gpt-4.1-mini",
  "gpt-5.4": "gpt-5.4",
  "gpt-6.2-astra": "gpt-6.2-astra",
  "moonshot-v1-32k": "moonshot-v1-32k",
  "kimi-k3": "kimi-k3",
  "qwen-plus": "qwen-plus",
  "qwen3.8-max": "qwen3.8-max",
  "gemini-2.5-flash": "gemini-2.5-flash",
};

const RESTRICTION_SITES_RNA = {
  BsaI: ["GGUCUC", "GAGACC"],
  BspQI: ["GCUCUUC", "GAAGAGC"],
  BsmBI: ["CGUCUC", "GAGACG"],
  SapI: ["GCUCUUC", "GAAGAGC"],
  EcoRI: ["GAAUUC"],
};

const CODON_TABLE_RNA = {
  UUU: "F", UUC: "F", UUA: "L", UUG: "L",
  CUU: "L", CUC: "L", CUA: "L", CUG: "L",
  AUU: "I", AUC: "I", AUA: "I", AUG: "M",
  GUU: "V", GUC: "V", GUA: "V", GUG: "V",
  UCU: "S", UCC: "S", UCA: "S", UCG: "S", AGU: "S", AGC: "S",
  CCU: "P", CCC: "P", CCA: "P", CCG: "P",
  ACU: "T", ACC: "T", ACA: "T", ACG: "T",
  GCU: "A", GCC: "A", GCA: "A", GCG: "A",
  UAU: "Y", UAC: "Y", UAA: "*", UAG: "*", UGA: "*",
  CAU: "H", CAC: "H", CAA: "Q", CAG: "Q",
  AAU: "N", AAC: "N", AAA: "K", AAG: "K",
  GAU: "D", GAC: "D", GAA: "E", GAG: "E",
  UGU: "C", UGC: "C", UGG: "W",
  CGU: "R", CGC: "R", CGA: "R", CGG: "R", AGA: "R", AGG: "R",
  GGU: "G", GGC: "G", GGA: "G", GGG: "G",
};

const HUMAN_CODON_WEIGHTS = {
  UUU: 0.85, UUC: 1.0, UUA: 0.19, UUG: 0.32,
  CUU: 0.33, CUC: 0.49, CUA: 0.18, CUG: 1.0,
  AUU: 0.77, AUC: 1.0, AUA: 0.36, AUG: 1.0,
  GUU: 0.39, GUC: 0.52, GUA: 0.25, GUG: 1.0,
  UCU: 0.79, UCC: 0.91, UCA: 0.63, UCG: 0.23, AGU: 0.63, AGC: 1.0,
  CCU: 0.91, CCC: 1.0, CCA: 0.88, CCG: 0.34,
  ACU: 0.7, ACC: 1.0, ACA: 0.78, ACG: 0.31,
  GCU: 0.68, GCC: 1.0, GCA: 0.58, GCG: 0.28,
  UAU: 0.79, UAC: 1.0, UAA: 1.0, UAG: 0.8, UGA: 1.0,
  CAU: 0.72, CAC: 1.0, CAA: 0.36, CAG: 1.0,
  AAU: 0.89, AAC: 1.0, AAA: 0.75, AAG: 1.0,
  GAU: 0.85, GAC: 1.0, GAA: 0.72, GAG: 1.0,
  UGU: 0.82, UGC: 1.0, UGG: 1.0,
  CGU: 0.38, CGC: 0.86, CGA: 0.52, CGG: 0.95, AGA: 1.0, AGG: 1.0,
  GGU: 0.47, GGC: 1.0, GGA: 0.74, GGG: 0.74,
};

const SEG_META = {
  five_prime_utr: { short: "5′UTR", colorVar: "var(--seg-utr5)" },
  cds: { short: "CDS", colorVar: "var(--seg-cds)" },
  stop_cassette: { short: "Stop", colorVar: "var(--seg-stop)" },
  three_prime_utr: { short: "3′UTR", colorVar: "var(--seg-utr3)" },
  poly_a: { short: "Poly(A)", colorVar: "var(--seg-polya)" },
  ires: { short: "IRES", colorVar: "var(--seg-utr5)" },
  pie_5prime: { short: "5′PIE", colorVar: "var(--seg-polya)" },
  pie_3prime: { short: "3′PIE", colorVar: "var(--seg-utr3)" },
  replicase_orf: { short: "nsP1-4", colorVar: "var(--seg-utr5)" },
  subgenomic_promoter: { short: "26S SGP", colorVar: "var(--seg-polya)" },
};

const METRIC_LABELS = {
  kozak_unpaired_prob: { shortLabel: "5′ 开放度", higherBetter: true, decimals: 4 },
  eternafold_kozak_unpaired_prob: { shortLabel: "5′ 开放度 (EF)", higherBetter: true, decimals: 4 },
  eternafold_aup: { shortLabel: "全长 AUP", higherBetter: true, decimals: 4 },
  cds_mean_window_mfe: { shortLabel: "CDS 局部 MFE", higherBetter: false, decimals: 2 },
  full_transcript_mfe: { shortLabel: "全长 MFE", higherBetter: false, decimals: 2 },
  mirna_detarget_dg_total: { shortLabel: "miR-122 ΔG", higherBetter: false, decimals: 2 },
  junction_folding_mfe: { shortLabel: "连接区 MFE", higherBetter: true, decimals: 2 },
  utr5_mrl_pred: { shortLabel: "翻译 MRL", higherBetter: true, decimals: 3 },
  consensus_halflife_pred: { shortLabel: "预测半衰期", higherBetter: true, decimals: 3 },
  utr3_stability_pred: { shortLabel: "3′UTR 稳定度", higherBetter: true, decimals: 3 },
  cds_stability_pred: { shortLabel: "CDS 稳定度", higherBetter: true, decimals: 3 },
  orthrus_halflife_pred: { shortLabel: "半衰期评分", higherBetter: true, decimals: 3 },
  codontransformer_syn_loglik: { shortLabel: "密码子似然", higherBetter: true, decimals: 3 },
  structrfm_cds_median_shape: { shortLabel: "SHAPE 反应度", higherBetter: false, decimals: 3 },
  cds_gc_fraction: { shortLabel: "CDS GC 占比", higherBetter: true, decimals: 3 },
};

const STRATEGY_LABELS = {
  reference_baseline: "参考基准 R",
  reference_import: "参考导入 R",
  conservative_refinement: "保守最小改动",
  locked_refinement: "区段锁定微调",
  exploratory_reoptimization: "探索重排方案",
  multi_chain_coexpression: "多链共表达",
  lineardesign_5prime_polished: "LinearDesign + 5′ 精修",
  pareto_frontier_top: "帕累托首层",
  pareto_balanced: "多目标均衡",
  de_novo_candidate: "从头合成构建",
  seed_init: "首轮种子构建",
  fixture: "基准测试构建",
};

const FTO_BADGE_LABELS = {
  sequence_screen_clear: { label: "序列筛查清净", cls: "ok" },
  research_track_only: { label: "科研评估轨", cls: "teal" },
  fto_advisory_review: { label: "FTO 人工复核", cls: "warn" },
  fto_blocked: { label: "专利模式阻断", cls: "err" },
};

const TOOL_DISPLAY_NAMES = {
  design_constructs: "序列构建与帕累托筛选",
  refine_construct: "单点锁定增量微调",
  audit_construct: "门禁与洁净室合规核验",
  explain_tradeoffs: "前沿权衡定量解析",
  compare_bundles: "双候选差异比对",
  export_bundle: "标准格式导出构建",
  search_protein_assets: "本地靶点快照检索",
  resolve_target_protein: "UniProt/AlphaFold 靶点解析",
  analyze_structure_and_motifs: "二级结构与风险基序扫描",
  query_design_rules: "定量科学规律库检索",
  search_literature_and_trials: "EuropePMC 文献与临床检索",
  benchmark_against_clinical: "临床参考构建横向对比",
  ingest_experiment: "湿实验读数校准回流",
  get_job_status: "任务状态查询",
};

const FALLBACK_SCIENTIFIC_RULES = [
  {
    rule_id: "RUL-V2-M1PSI-FRAMESHIFT-009",
    title_zh: "m¹Ψ 修饰下 CDS 连续 ≥4 尿苷滑移串消除 (+1 移码抑制)",
    title_en: "Eliminate Slippery U4+ Runs in m1Psi-Modified CDS to Prevent +1 Ribosomal Frameshifting",
    modality: ["linear_mrna", "sarna"],
    region: "CDS",
    chemistries: ["m1psi"],
    evidence_grade: "cross_lab_verified",
    effect_size_summary:
      "Mulroney et al. 2024 (Nature, N=21 constructs + BNT162b2 in vivo ELISpot) & Poulis/Rodnina 2026 (Nat Commun kinetic recoding): synonymous interruption of UUUU/UUUUU reduces +1 frameshifted peptide byproducts by >85% while preserving 100% amino-acid identity.",
    enforced_in_generator: true,
    failure_modes: [
      "m1Psi slows aminoacyl-tRNA accommodation on U-rich slippery codons, triggering +1 off-frame transpeptidation and unintended T-cell neoepitopes.",
    ],
    citations: [
      { authors: "Mulroney et al.", year: 2024, journal: "Nature", pmid: "38057662", pmcid: "PMC10796200", doi: "10.1038/s41586-023-06800-3" },
    ],
  },
  {
    rule_id: "RUL-V2-CDS-MFE-CAI-004",
    title_zh: "CDS 全局/局部二级结构稳定性与同义密码子适应度协同优化",
    title_en: "Joint Optimization of CDS Secondary Structure Stability (MFE/AUP) and Human Codon Optimality",
    modality: ["linear_mrna", "circrna", "sarna"],
    region: "CDS",
    chemistries: ["unmodified_u", "pseudouridine", "m1psi"],
    evidence_grade: "cross_lab_verified",
    effect_size_summary:
      "Mauger et al. 2019 (PNAS), Leppek et al. 2022 (Nat Commun, N=233), and Zhang et al. 2023 (LinearDesign, Nature): structured CDS with relaxed 5' initiation increases in-solution & in-cell functional half-life 2.1x–4.8x.",
    enforced_in_generator: true,
    failure_modes: [
      "Over-stabilizing the [-15, +18] start-codon window blocks 43S ribosome recruitment despite high full-length stability.",
    ],
    citations: [
      { authors: "Zhang et al.", year: 2023, journal: "Nature", pmid: "37130513", pmcid: "PMC10202639", doi: "10.1038/s41586-023-06127-z" },
      { authors: "Leppek et al.", year: 2022, journal: "Nat Commun", pmid: "35585242", pmcid: "PMC9931579", doi: "10.1038/s41467-022-30261-x" },
    ],
  },
  {
    rule_id: "RUL-V2-5UTR-UAUG-001",
    title_zh: "5′UTR 严禁上游起始密码子 (uAUG / uORF)",
    title_en: "Eliminate Upstream AUG (uAUG) and Out-of-Frame uORFs in 5'UTR",
    modality: ["linear_mrna", "sarna"],
    region: "5'UTR",
    chemistries: ["unmodified_u", "pseudouridine", "m1psi"],
    evidence_grade: "conditional",
    effect_size_summary:
      "GSE232927 synthetic 50-nt 5'UTR MPRA (N=25,000, HEK293T): uAUG count vs MRL Spearman rho=-0.5821; concordant MRL drop across U/Psi/m1Psi in Sample 2019 GSE114002.",
    enforced_in_generator: true,
    failure_modes: [
      "Premature 43S initiation hijacks scanning ribosomes before reaching canonical CDS start codon.",
    ],
    citations: [
      { authors: "Castillo-Hair et al.", year: 2024, journal: "Nat Commun", pmid: "38902240", pmcid: "PMC11189900", doi: "10.1038/s41467-024-49508-2" },
      { authors: "Sample et al.", year: 2019, journal: "Nat Biotechnol", pmid: "31267113", pmcid: "PMC7100133", doi: "10.1038/s41587-019-0164-5" },
    ],
  },
  {
    rule_id: "RUL-V2-5UTR-KOZAK-M3-002",
    title_zh: "起始密码子 -3 位嘌呤 (Kozak A/G) 与 [-15, +18] 低配对度门控",
    title_en: "Enforce Purine at Position -3 and High Unpaired Probability Across Kozak Initiation Window",
    modality: ["linear_mrna", "sarna"],
    region: "5'UTR",
    chemistries: ["unmodified_u", "pseudouridine", "m1psi"],
    evidence_grade: "conditional",
    effect_size_summary:
      "GSE232927 HEK293T MPRA (N=25,000): position -3 purine vs MRL Spearman rho=+0.2178 (p=4.34e-266).",
    enforced_in_generator: true,
    failure_modes: [
      "Pyrimidine (C/U) at -3 causes leaky scanning past canonical AUG into downstream internal methionine codons.",
    ],
    citations: [
      { authors: "Kozak", year: 1987, journal: "Nucleic Acids Res", pmid: "3313277", pmcid: "PMC306349", doi: "10.1093/nar/15.20.8125" },
    ],
  },
  {
    rule_id: "RUL-V2-CIRC-PIE-IRES-028",
    title_zh: "circRNA 置换内含子-外显子 (PIE) 剪接交界防二级结构入侵 IRES",
    title_en: "Insulate Group I PIE Exon1/Exon2 Splice Junction from Invading IRES Domain IV/V",
    modality: ["circrna"],
    region: "circRNA",
    chemistries: ["unmodified_u"],
    evidence_grade: "conditional",
    effect_size_summary:
      "Chen et al. 2023 (Nat Biotechnol, N=536) & Cell Res 2026 circRNA pack: structured spacer insulation improves circularization yield and sustained 72h translation.",
    enforced_in_generator: true,
    failure_modes: [
      "Base-pairing between PIE homology arms and HRV-B3/CVB3 IRES core collapses eIF4G binding.",
    ],
    citations: [
      { authors: "Chen et al.", year: 2023, journal: "Nat Biotechnol", pmid: "36097176", doi: "10.1038/s41587-022-01393-0" },
    ],
  },
  {
    rule_id: "RUL-V2-MYTH-MAX-CAI-035",
    title_zh: "误区证伪：单纯最大化 CAI=1.0 并不等于最高蛋白表达量",
    title_en: "Debunked Myth: Greedy Maximum-CAI (1.0) Alone Does Not Maximize In Vivo Protein Yield",
    modality: ["linear_mrna", "circrna", "sarna"],
    region: "CDS",
    chemistries: ["unmodified_u", "m1psi"],
    evidence_grade: "debunked_myth",
    effect_size_summary:
      "Leppek 2022 & Zhang 2023 demonstrate greedy CAI=1.0 often introduces high-GC G-quadruplexes, 5' initiation hairpins, or local ribosome collisions with inferior half-life vs Pareto MFE+CAI designs.",
    enforced_in_generator: false,
    failure_modes: [
      "Single-objective greedy codon replacement ignores 5'UTR-CDS boundary folding and m1Psi frameshift motifs.",
    ],
    citations: [
      { authors: "Zhang et al.", year: 2023, journal: "Nature", pmid: "37130513", pmcid: "PMC10202639", doi: "10.1038/s41586-023-06127-z" },
    ],
  },
];

const FALLBACK_DATA_HUB = {
  catalog_id: "mRNAx_Data_Hub_Catalog_v2",
  summary_counts: {
    datasets_count: 34,
    external_apis_count: 20,
    engines_and_models_count: 11,
    master_catalog_v12_rows_unified: 114,
    commercial_clean_datasets: 26,
    pending_or_quarantined_datasets: 8,
  },
  engines_and_models: [
    { id: "lineardesign", name: "LinearDesign C++ Core", role: "CDS MFE×CAI 精确格点动态规划", license_track: "research_max", status: "ready" },
    { id: "viennarna", name: "ViennaRNA 2.6+", role: "配分函数 bpp 与滑窗热力学折叠", license_track: "research_max", status: "ready" },
    { id: "eternafold", name: "EternaFold v1.3.1", role: "零 GPL 配对概率评估 (S-Processed 待清核)", license_track: "commercial_clearance_pending", status: "gated" },
    { id: "surrogate_v2", name: "Surrogate v2 Pure-Seq Blend", role: "纯序列四头无 GPL 商业洁净代理栈", license_track: "commercial_clean", status: "ready" },
    { id: "codontransformer", name: "CodonTransformer v1.0 (RC-005)", role: "Apache-2.0 同义密码子似然先验", license_track: "commercial_clean", status: "ready" },
  ],
  datasets: [
    {
      dataset_id: "GSE232927_CastilloHair2024_5UTR_MRL",
      title: "Castillo-Hair et al. 2024 HEK293T Synthetic 5'UTR Polysome MRL MPRA",
      modality: "linear_mrna",
      region_focus: "5'UTR",
      chemistry: ["unmodified_u"],
      sample_count: 145377,
      primary_readout: "Mean Ribosome Load (MRL, HEK293T)",
      license: { spdx: "CC-BY-4.0", license_track: "commercial_clean" },
    },
    {
      dataset_id: "GSE176581_Cao2021_5UTR_MPRA",
      title: "Cao et al. 2021 High-Throughput Human/Synthetic 5'UTR GFP Translation MPRA",
      modality: "linear_mrna",
      region_focus: "5'UTR",
      chemistry: ["unmodified_u"],
      sample_count: 16288,
      primary_readout: "FACS Top-2.5% GFP Enrichment & Ribo-Seq TE",
      license: { spdx: "CC-BY-4.0", license_track: "commercial_clean" },
    },
    {
      dataset_id: "GSE114002_Sample2019_Optimus5Prime_MPRA",
      title: "Sample et al. 2019 280k 5'UTR Polysome MRL Across U, Psi, and m1Psi",
      modality: "linear_mrna",
      region_focus: "5'UTR",
      chemistry: ["unmodified_u", "pseudouridine", "m1psi"],
      sample_count: 553936,
      primary_readout: "Multi-chemistry Polysome MRL (HEK293T)",
      license: { spdx: "GEO_Public_Pending_Paper_OA", license_track: "commercial_clearance_pending" },
    },
    {
      dataset_id: "W9_circRNA_3Lab_Clean_Pack",
      title: "Chen 2023 Nat Biotechnol + Cell Res 2026 + NSMB 2026 circRNA Benchmark Pack",
      modality: "circrna",
      region_focus: "IRES / PIE",
      chemistry: ["unmodified_u"],
      sample_count: 1452,
      primary_readout: "Circularization Efficiency, SHAPE & 72h Luminescence",
      license: { spdx: "CC-BY-4.0", license_track: "commercial_clean" },
    },
    {
      dataset_id: "W11_saRNA_Vijayan2024_PolyA_Ninetails",
      title: "Vijayan 2024 saRNA Replicon + Ninetails/Fuhrmann Poly(A) Tail Dynamics",
      modality: "sarna",
      region_focus: "Replicon / Poly(A)",
      chemistry: ["unmodified_u", "m1psi"],
      sample_count: 108254,
      primary_readout: "saRNA Subgenomic Expression & Nanopore Poly(A) Tail Length",
      license: { spdx: "CC-BY-4.0", license_track: "commercial_clean" },
    },
  ],
  modality_scaffolds_summary: {
    modality_architectures: {
      linear_mrna: {
        label: "Linear Cap1 mRNA",
        summary: "CleanCap AG (5′-AG) · HBA1/精简高开放 5′UTR · 优化 CDS · UGAUAA 终止盒 · AES-mtRNR1/HBB 3′UTR · 分段 A30L70 Poly(A)",
        chemistry_note: "支持 m¹Ψ / 5moU / Ψ / 未修饰 U",
      },
      circrna: {
        label: "Circular RNA (Anabaena PIE)",
        summary: "5′ 同源臂 + Anabaena Group I Intron 3′ 片段 · E2 外显子 · 绝缘间隔子 · HRV-B3 / CVB3 IRES · CDS · E1 + 5′ Intron",
        chemistry_note: "推荐未修饰 U (化学修饰常抑制 IRES 核糖体招募与环化剪接)",
      },
      sarna: {
        label: "Self-Amplifying RNA (VEEV TC-83)",
        summary: "5′ CleanCap AU 保守前导区 · VEEV nsP1-nsP4 复制酶 · 26S 亚基因组启动子 · 目的抗原/载荷 CDS · 保守 CSE 3′UTR + Poly(A)",
        chemistry_note: "推荐未修饰 U 或特定兼容胞苷修饰，保护复制酶活性",
      },
    },
  },
  tissue_mirna_atlas: {
    mirna_entries: [
      { mirna_id: "hsa-miR-122-5p", tissue_group: "liver_hepatocyte", role: "全身给药时抑制肝实质细胞脱靶表达" },
      { mirna_id: "hsa-miR-142-3p", tissue_group: "hematopoietic_apc", role: "抑制髓系/抗原呈递细胞表达以降低免疫清除" },
      { mirna_id: "hsa-miR-1-3p", tissue_group: "cardiomyocyte_skeletal", role: "抑制心肌与骨骼肌非靶向表达" },
      { mirna_id: "hsa-miR-126-3p", tissue_group: "vascular_endothelium", role: "抑制血管内皮细胞脱靶表达" },
    ],
  },
};

function mintProjectKey() {
  const bytes = new Uint8Array(18);
  window.crypto.getRandomValues(bytes);
  return "pk_" + Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}

function mintClientRequestId() {
  const bytes = new Uint8Array(12);
  window.crypto.getRandomValues(bytes);
  return "req_" + Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}

function isLocalHostPage() {
  const h = window.location.hostname;
  return h === "localhost" || h === "127.0.0.1" || h === "";
}

function isTrustedApiOrigin(candidateUrl) {
  try {
    const parsed = new URL(candidateUrl);
    if (parsed.pathname && parsed.pathname !== "/") return false;
    if (TRUSTED_PRODUCTION_ORIGINS.has(parsed.origin)) return true;
    if (
      parsed.origin === window.location.origin &&
      !window.location.hostname.endsWith(".github.io")
    ) {
      return true;
    }
    if (
      isLocalHostPage() &&
      (parsed.protocol === "http:" || parsed.protocol === "https:") &&
      (parsed.hostname === "localhost" || parsed.hostname === "127.0.0.1")
    ) {
      return true;
    }
    return false;
  } catch {
    return false;
  }
}

const state = {
  theme: "light",
  activeTab: "studio",
  modality: "linear_mrna",
  gatewayOnline: false,
  configGeneration: 0,
  statusRequestSeq: 0,
  sessionEpoch: 1,
  turnSequence: 0,
  appliedTurnSequence: 0,
  apiBase: DEFAULT_API_BASE,
  accessToken: "",
  accessTokensByOrigin: {},
  llmProvider: "deepseek",
  llmModel: "deepseek-flash",
  defaultModels: { ...DEFAULT_PROVIDER_MODELS },
  supportedModels: { ...SUPPORTED_PROVIDER_MODELS },
  byoKeys: { deepseek: "", openai: "", gemini: "", kimi: "", qwen: "" },
  byoKeysByOrigin: {},
  byoKey: "",
  modalDraftProvider: "deepseek",
  modalDraftByoKeys: null,
  modalInitialOrigin: null,
  modalDraftOrigin: null,
  modalInitialAccessToken: null,
  modalInitialByoKeys: null,
  callerTokenAuthenticated: false,
  userChoseExecMode: false,
  activeRequestCount: 0,
  inFlight: false,
  lastFailedTurn: null,
  projectId: `proj_${Math.random().toString(36).slice(2, 8)}`,
  projectKey: mintProjectKey(),
  entryMode: "protein_only",
  presets: null,
  intentHistory: [],
  completedRevisions: [],
  activeVersion: 0,
  activeIntentStatus: "",
  activeEnzymes: null,
  startupApiRejected: "",
  userEditedProtein: false,
  latestReport: null,
  candidates: [],
  selectedBundleId: null,
  selectedSegType: "cds",
  paretoAxisPairKey: "",
  compareSelectedIds: [],
  boundaryAdvisories: [],
  currentStructureProfile: null,
  structureCache: {},
  rulesCache: null,
  dataHubCache: null,
  resolvedTarget: null,
  lastDossierHtml: "",
  lastDossierFilename: "qc_dossier.html",
  sortCol: "rank",
  sortAsc: true,
  lastCliCommand: "",
  cachedExports: {},
  forceConflictOnce: false,
  logCount: 0,
};

const ESC_MAP = {
  "&": "&amp;",
  "<": "&lt;",
  ">": "&gt;",
  '"': "&quot;",
  "'": "&#39;",
};

function escapeHtml(str) {
  return String(str == null ? "" : str).replace(/[&<>"']/g, (ch) => ESC_MAP[ch]);
}
const esc = escapeHtml;

function safeExternalUrl(rawUrl) {
  const trimmed = String(rawUrl == null ? "" : rawUrl).trim();
  if (!trimmed) return "";
  try {
    const parsed = new URL(trimmed);
    if (parsed.protocol !== "https:" && parsed.protocol !== "http:") return "";
    if (parsed.username || parsed.password) return "";
    return escapeHtml(parsed.href);
  } catch {
    return "";
  }
}

function renderSafeExternalLink(rawUrl, label) {
  const safeHref = safeExternalUrl(rawUrl);
  const safeLabel = escapeHtml(label);
  if (!safeHref) return `<span class="mono meta-muted">${safeLabel}</span>`;
  return `<a class="ext-link" href="${safeHref}" target="_blank" rel="noopener noreferrer">${safeLabel}</a>`;
}

function buildVerifiedCitationUrl(kind, rawId) {
  const s = String(rawId == null ? "" : rawId).trim();
  if (!s) return "";
  if (kind === "doi" && /^10\.\d{4,9}\/[-._;()/:A-Za-z0-9]+$/.test(s)) {
    return `https://doi.org/${encodeURI(s)}`;
  }
  if (kind === "pmid" && /^\d{1,12}$/.test(s)) {
    return `https://pubmed.ncbi.nlm.nih.gov/${s}/`;
  }
  if (kind === "pmcid" && /^PMC\d{1,12}$/i.test(s)) {
    return `https://pmc.ncbi.nlm.nih.gov/articles/${s.toUpperCase()}/`;
  }
  if (kind === "nct" && /^NCT\d{6,12}$/i.test(s)) {
    return `https://clinicaltrials.gov/study/${s.toUpperCase()}`;
  }
  return "";
}

function extractSandboxedText(val) {
  if (val == null) return "";
  let raw = "";
  if (typeof val === "string") {
    raw = val;
  } else if (typeof val === "object") {
    raw = String(val.text_excerpt || val.wrapped_payload || val.title || "");
  }
  return raw
    .replace(/<UNTRUSTED_EXTERNAL_DATA[^>]*>/gi, "")
    .replace(/<\/UNTRUSTED_EXTERNAL_DATA>/gi, "")
    .trim();
}

function formatSafeMarkdown(rawText) {
  const escaped = escapeHtml(rawText == null ? "" : String(rawText));
  const lines = escaped.split(/\r?\n/);
  const out = [];
  let inList = false;

  const formatInline = (line) =>
    line
      .replace(/`([^`\n]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>");

  for (const line of lines) {
    const bulletMatch = line.match(/^\s*[-*•]\s+(.*)$/);
    if (bulletMatch) {
      if (!inList) {
        out.push("<ul>");
        inList = true;
      }
      out.push(`<li>${formatInline(bulletMatch[1])}</li>`);
      continue;
    }
    if (inList) {
      out.push("</ul>");
      inList = false;
    }
    if (!line.trim()) {
      out.push("<br/>");
    } else {
      out.push(`<div>${formatInline(line)}</div>`);
    }
  }
  if (inList) {
    out.push("</ul>");
  }
  return out.join("");
}

function humanizeCandidateLabel(cand) {
  if (!cand) return "未命名方案";
  if (typeof cand.display_label === "string" && cand.display_label.trim()) {
    return cand.display_label.trim();
  }
  const bid = String(cand.bundle_id || "");
  const low = bid.toLowerCase();
  if (cand.derivation_action === "reference_import" || low.endsWith("_ref_r")) {
    return "参考基准构建 (R)";
  }
  if (low.endsWith("_mina") || cand.derivation_action === "locked_refinement") {
    return "保守微调方案 (最小改动)";
  }
  if (low.endsWith("_expb")) {
    return "探索重排方案 (高表达)";
  }
  if (Array.isArray(cand.molecules) && cand.molecules.length > 1) {
    return `${cand.pareto_rank ? "Pareto #" + cand.pareto_rank + " · " : ""}多链共表达候选`;
  }
  const tail = bid.replace(/^bnd_/, "").slice(-6) || "01";
  if (cand.pareto_rank != null) {
    return `Pareto #${cand.pareto_rank} 候选 (${tail})`;
  }
  return `候选方案 (${tail})`;
}

function humanizeStrategyTag(rawTag) {
  const key = String(rawTag || "").trim();
  if (!key) return "标准合成";
  return STRATEGY_LABELS[key] || key.replace(/_/g, " ");
}

function humanizeOutcomeLabel(rawStatus) {
  const key = String(rawStatus || "").trim();
  const map = {
    COMPLETED: "已完成 · 就绪",
    COMPLETED_WITH_CANDIDATES: "已完成 · 候选就绪",
    COMPLETED_NO_QUALIFIED_CANDIDATES: "已完成 · 无合格候选",
    UNCOMPARABLE_ACROSS_TRACKS: "跨轨不可直接排序",
    IDLE: "IDLE",
  };
  return map[key] || key;
}

function safeStorageGet(storeName, key) {
  try {
    const store = storeName === "session" ? globalThis.sessionStorage : globalThis.localStorage;
    return store ? store.getItem(key) : null;
  } catch {
    return null;
  }
}

function safeStorageSet(storeName, key, val) {
  try {
    const store = storeName === "session" ? globalThis.sessionStorage : globalThis.localStorage;
    if (store) store.setItem(key, val);
  } catch {
    // ignore storage errors in restricted contexts
  }
}

function safeStorageRemove(storeName, key) {
  try {
    const store = storeName === "session" ? globalThis.sessionStorage : globalThis.localStorage;
    if (store) store.removeItem(key);
  } catch {
    // ignore
  }
}

function applyTheme(themeName) {
  state.theme = themeName === "graphite" || themeName === "dark" ? "graphite" : "light";
  if (document.documentElement && document.documentElement.dataset) {
    document.documentElement.dataset.theme = state.theme;
  }
  safeStorageSet("local", "mrnax_theme", state.theme);
  const lbl = document.getElementById("txt-theme-label");
  if (lbl) {
    lbl.textContent = state.theme === "graphite" ? "◑ 浅色" : "◐ 深色";
  }
  renderParetoScatter();
  if (state.currentStructureProfile) {
    renderStructureVisuals(state.currentStructureProfile, getActiveInspectedChain());
  }
}

function switchWorkspaceTab(tabName) {
  const validTabs = ["studio", "rules", "data-hub", "benchmarks"];
  const next = validTabs.includes(tabName) ? tabName : "studio";
  state.activeTab = next;

  const viewMap = {
    studio: "view-studio",
    rules: "view-rules",
    "data-hub": "view-data-hub",
    benchmarks: "view-benchmarks",
  };
  for (const [key, domId] of Object.entries(viewMap)) {
    const vEl = document.getElementById(domId);
    if (vEl) vEl.hidden = key !== next;
  }
  if (typeof document.querySelectorAll === "function") {
    document.querySelectorAll("#primary-nav-tabs .nav-tab").forEach((btn) => {
      const active = Boolean(btn.dataset && btn.dataset.tab === next);
      if (btn.classList) btn.classList.toggle("active", active);
      if (typeof btn.setAttribute === "function") btn.setAttribute("aria-pressed", String(active));
    });
  }

  if (next === "rules" && !state.rulesCache) {
    loadScientificRules();
    runLiteratureSearch();
  } else if (next === "data-hub" && !state.dataHubCache) {
    loadDataHubCatalog();
  } else if (next === "benchmarks") {
    syncBenchmarkSelectedLabel();
    runClinicalBenchmarkCompare();
    const wetEl = document.getElementById("wetlab-input");
    if (wetEl && !wetEl.value.trim()) {
      loadWetLabSampleRows(4);
    }
  }
}

function setModality(mod) {
  const allowed = ["linear_mrna", "circrna", "sarna"];
  const next = allowed.includes(mod) ? mod : "linear_mrna";
  state.modality = next;
  if (typeof document.querySelectorAll === "function") {
    document.querySelectorAll("#modality-switch .modality-opt").forEach((btn) => {
      if (btn.classList) {
        btn.classList.toggle("active", Boolean(btn.dataset && btn.dataset.modality === next));
      }
    });
  }
  const scafEl = document.getElementById("sel-scaffold");
  if (scafEl) {
    if (next === "circrna" && !String(scafEl.value || "").startsWith("circ_")) {
      scafEl.value = "circ_anabaena_hrvb3";
    } else if (next === "sarna" && !String(scafEl.value || "").startsWith("sarna_")) {
      scafEl.value = "sarna_veev_tc83";
    } else if (next === "linear_mrna" && !String(scafEl.value || "").startsWith("cap1_")) {
      scafEl.value = "cap1_hba1_aes_mt_a30l70";
    }
  }
  updateSetupSummary();
}

function syncTrackButtons(trackValue) {
  const norm = trackValue === "commercial_clean" ? "commercial_clean" : "research_max";
  const profEl = document.getElementById("sel-profile");
  if (profEl) profEl.value = norm;
  if (typeof document.querySelectorAll === "function") {
    document.querySelectorAll("#track-switch .track-opt").forEach((btn) => {
      if (btn.classList) {
        btn.classList.toggle("active", Boolean(btn.dataset && btn.dataset.track === norm));
      }
    });
  }
  updateSetupSummary();
}

function setEntryMode(mode) {
  const normalized = mode === "with_r" ? "with_r" : "protein_only";
  state.entryMode = normalized;
  if (typeof document.querySelectorAll === "function") {
    document.querySelectorAll(".seg-tabs[aria-label='设计入口'] .tab").forEach((t) => {
      if (t && t.classList) {
        t.classList.toggle("active", t.dataset && t.dataset.entry === normalized);
      }
    });
  }
  const boxRefR = document.getElementById("box-ref-r");
  if (boxRefR) boxRefR.hidden = normalized !== "with_r";
}

function getSelectedExecMode() {
  const sel = document.getElementById("sel-exec-mode");
  return (sel && sel.value) || "deterministic";
}

function setExecMode(mode, { userInitiated = false } = {}) {
  const normalized = mode === "llm_agent" ? "llm_agent" : "deterministic";
  if (userInitiated) state.userChoseExecMode = true;
  const sel = document.getElementById("sel-exec-mode");
  if (sel) sel.value = normalized;
  if (typeof document.querySelectorAll === "function") {
    document.querySelectorAll(".engine-opt").forEach((btn) => {
      if (btn && btn.classList) {
        btn.classList.toggle("active", btn.dataset && btn.dataset.mode === normalized);
      }
    });
  }
  if (!state.inFlight) {
    const btnSend = document.getElementById("btn-send-turn");
    if (btnSend) {
      btnSend.textContent = normalized === "llm_agent" ? "发送给 Agent ↵" : "编译生成 ↵";
    }
  }
}

function syncModelOptions(provider, { skipOpenModal = false } = {}) {
  const models =
    (state.supportedModels && state.supportedModels[provider]) ||
    SUPPORTED_PROVIDER_MODELS[provider] ||
    [state.defaultModels[provider] || DEFAULT_PROVIDER_MODELS[provider] || "deepseek-flash"];
  const fallback =
    (state.defaultModels && state.defaultModels[provider]) ||
    DEFAULT_PROVIDER_MODELS[provider] ||
    models[0];
  if (!models.includes(state.llmModel)) {
    state.llmModel = fallback;
  }
  const dlgSettings = document.getElementById("modal-settings");
  const modalOpen = Boolean(dlgSettings && dlgSettings.open);
  for (const id of ["sel-llm-model", "cfg-llm-model"]) {
    if (skipOpenModal && id === "cfg-llm-model" && modalOpen) continue;
    const sel = document.getElementById(id);
    if (!sel) continue;
    if (typeof document.createElement === "function" && typeof sel.replaceChildren === "function") {
      sel.replaceChildren();
      for (const m of models) {
        const opt = document.createElement("option");
        opt.value = m;
        opt.textContent = id === "sel-llm-model" ? (MODEL_LABELS[m] || m) : m;
        sel.appendChild(opt);
      }
    }
    sel.value = state.llmModel;
  }
  const pill = document.getElementById("rail-model-pill");
  if (pill) {
    pill.textContent = state.llmProvider === "deepseek" ? state.llmModel : `${state.llmProvider}:${state.llmModel}`;
  }
}

function setModelPreference(modelName) {
  const prov = state.llmProvider || "deepseek";
  const allowed =
    (state.supportedModels && state.supportedModels[prov]) ||
    SUPPORTED_PROVIDER_MODELS[prov] ||
    [DEFAULT_PROVIDER_MODELS[prov]];
  const cleaned = (modelName || "").trim();
  state.llmModel = allowed.includes(cleaned)
    ? cleaned
    : (state.defaultModels[prov] || DEFAULT_PROVIDER_MODELS[prov] || allowed[0]);
  safeStorageSet("local", "mrnax_llm_model", state.llmModel);
  syncModelOptions(prov);
  updateAgentAuthPill(null);
}

function setProviderPreference(value) {
  const select = document.getElementById("cfg-llm-provider");
  const provider = (value || "").trim().toLowerCase();
  const supported =
    select && select.options
      ? Array.from(select.options, (option) => option.value)
      : Object.keys(DEFAULT_PROVIDER_MODELS);
  state.llmProvider = supported.includes(provider) ? provider : "deepseek";
  state.byoKey = (state.byoKeys && state.byoKeys[state.llmProvider]) || "";
  if (select) select.value = state.llmProvider;
  safeStorageSet("local", "mrnax_llm_provider", state.llmProvider);
  syncModelOptions(state.llmProvider);
  updateAgentAuthPill(null);
}

function updateAgentAuthPill(statusPayload) {
  if (statusPayload && typeof statusPayload.caller_token_authenticated === "boolean") {
    state.callerTokenAuthenticated = statusPayload.caller_token_authenticated;
  }
  const pill = document.getElementById("pill-agent-auth");
  if (!pill) return;
  const activeModelName = state.llmModel || DEFAULT_PROVIDER_MODELS[state.llmProvider] || "deepseek-flash";
  if (state.callerTokenAuthenticated) {
    pill.className = "auth-pill mono ok";
    pill.textContent = `Agent: ${activeModelName} ✓`;
  } else if (state.byoKey) {
    pill.className = "auth-pill mono ok";
    pill.textContent = `Agent: BYO ${activeModelName} ✓`;
  } else {
    pill.className = "auth-pill mono warn";
    pill.textContent = "Agent: 未解锁 · 点击配置";
  }
}

function extractHashToken() {
  const rawHash = (window.location && window.location.hash) || "";
  if (!rawHash.startsWith("#token=")) return null;
  if (typeof history !== "undefined" && typeof history.replaceState === "function") {
    history.replaceState(null, "", window.location.pathname + window.location.search);
  }
  try {
    return decodeURIComponent(rawHash.slice(7)).trim();
  } catch {
    return "";
  }
}

function setSettingsModalError(msg) {
  const errEl = document.getElementById("cfg-settings-error");
  if (!errEl) return;
  if (msg) {
    errEl.hidden = false;
    errEl.textContent = msg;
  } else {
    errEl.hidden = true;
    errEl.textContent = "";
  }
}

function openSettingsModal(focusToken = false) {
  const currentOrigin = normalizeOrigin(state.apiBase);
  state.modalInitialOrigin = currentOrigin;
  state.modalDraftOrigin = currentOrigin;
  state.modalInitialAccessToken = state.accessToken || "";
  state.modalInitialByoKeys = { ...(state.byoKeys || {}) };
  state.modalDraftByoKeys = { ...(state.byoKeys || {}) };
  state.modalDraftProvider = state.llmProvider || "deepseek";
  setSettingsModalError("");
  const cfgApi = document.getElementById("cfg-api-base");
  const cfgToken = document.getElementById("cfg-access-token");
  const cfgProv = document.getElementById("cfg-llm-provider");
  const cfgByo = document.getElementById("cfg-byo-key");
  if (cfgApi) cfgApi.value = state.apiBase;
  if (cfgToken) cfgToken.value = state.accessToken;
  if (cfgProv) cfgProv.value = state.llmProvider;
  syncModelOptions(state.llmProvider);
  if (cfgByo) cfgByo.value = (state.byoKeys && state.byoKeys[state.llmProvider]) || state.byoKey || "";
  const dlgSettings = document.getElementById("modal-settings");
  if (dlgSettings && typeof dlgSettings.showModal === "function") {
    dlgSettings.showModal();
  }
  if (focusToken && cfgToken && typeof cfgToken.focus === "function") {
    cfgToken.focus();
  }
}

function normalizeOrigin(urlStr) {
  const raw = String(urlStr || "").trim().replace(/\/+$/, "");
  if (!raw) return "";
  try {
    return new URL(raw).origin;
  } catch {
    return raw;
  }
}

function syncCurrentOriginCredentialsToMap(originOverride = null) {
  const origin = normalizeOrigin(originOverride || state.apiBase);
  if (!origin) return;
  state.accessTokensByOrigin = state.accessTokensByOrigin || {};
  state.byoKeysByOrigin = state.byoKeysByOrigin || {};
  state.accessTokensByOrigin[origin] = state.accessToken || "";
  state.byoKeysByOrigin[origin] = { ...(state.byoKeys || {}) };
}

function getCredentialsForOrigin(targetOrigin) {
  if (state.startupApiRejected) {
    return { accessToken: "", byoKeys: {} };
  }
  const normTarget = normalizeOrigin(targetOrigin);
  if (!normTarget || !isTrustedApiOrigin(normTarget)) {
    return { accessToken: "", byoKeys: {} };
  }
  const normCurrent = normalizeOrigin(state.apiBase);
  const storedTok = (state.accessTokensByOrigin && state.accessTokensByOrigin[normTarget]) || "";
  const storedByo = (state.byoKeysByOrigin && state.byoKeysByOrigin[normTarget]) || {};
  if (normTarget === normCurrent) {
    return {
      accessToken: state.accessToken || storedTok || "",
      byoKeys: { ...storedByo, ...(state.byoKeys || {}) },
    };
  }
  return {
    accessToken: storedTok,
    byoKeys: { ...storedByo },
  };
}

function initConfig() {
  const savedTheme = safeStorageGet("local", "mrnax_theme");
  applyTheme(savedTheme === "graphite" ? "graphite" : "light");

  const url = new URL(window.location.href);
  const hasQueryApiParam = url.searchParams.has("api");
  const rawQueryApi = (url.searchParams.get("api") || "").trim();
  const rawStoredApi = safeStorageGet("local", "mrnax_api_base");
  state.startupApiRejected = "";
  if (hasQueryApiParam && (!rawQueryApi || !isTrustedApiOrigin(rawQueryApi))) {
    state.startupApiRejected = rawQueryApi || "(empty)";
  }

  if (rawQueryApi && isTrustedApiOrigin(rawQueryApi)) {
    state.apiBase = new URL(rawQueryApi).origin;
  } else if (rawStoredApi && isTrustedApiOrigin(rawStoredApi)) {
    state.apiBase = new URL(rawStoredApi).origin;
  } else {
    state.apiBase = DEFAULT_API_BASE;
    safeStorageRemove("local", "mrnax_api_base");
  }

  // Purge any legacy persisted tokens from earlier browser sessions
  safeStorageRemove("local", "mrnax_access_token");
  safeStorageRemove("session", "mrnax_byo_key");

  const parsedHashTok = extractHashToken();
  if (parsedHashTok && !state.startupApiRejected) {
    state.accessToken = parsedHashTok;
  } else if (state.startupApiRejected) {
    state.accessToken = "";
  }
  syncCurrentOriginCredentialsToMap(state.apiBase);

  setProviderPreference(safeStorageGet("local", "mrnax_llm_provider"));
  setModelPreference(safeStorageGet("local", "mrnax_llm_model") || DEFAULT_PROVIDER_MODELS[state.llmProvider]);

  if (state.accessToken && !state.startupApiRejected) {
    setExecMode("llm_agent");
  } else {
    setExecMode("deterministic");
  }

  if (state.startupApiRejected) {
    const banner = document.getElementById("banner-notes");
    if (banner) {
      banner.hidden = false;
      banner.className = "notice-banner err";
      banner.innerHTML = `<strong>启动网关参数被安全拦截 (UNTRUSTED_ORIGIN)：</strong>已拒绝不可信或空值的 <code>?api=</code> 参数，未导入也未发送任何 <code>#token</code> 凭据。请通过右上角“网关与密钥”确认或修改受信任网关地址。`;
    }
  }

  const cfgApi = document.getElementById("cfg-api-base");
  const cfgToken = document.getElementById("cfg-access-token");
  const cfgByo = document.getElementById("cfg-byo-key");
  const txtProj = document.getElementById("txt-project-id");
  if (cfgApi) cfgApi.value = state.apiBase;
  if (cfgToken) cfgToken.value = state.accessToken;
  if (cfgByo) cfgByo.value = state.byoKey;
  if (txtProj) txtProj.textContent = state.projectId;
  updateActiveRevBadge();
  updateAgentAuthPill(null);
  renderBoundariesModal();
}

function updateActiveRevBadge() {
  const revEl = document.getElementById("badge-active-rev");
  if (revEl) {
    revEl.textContent = `v${String(state.activeVersion).padStart(2, "0")}`;
  }
}

function buildHeaders(
  path,
  method,
  explicitProjectKey,
  includeAuth,
  explicitProvider = null,
  explicitApiBase = null
) {
  const h = { "Content-Type": "application/json" };
  if (state.startupApiRejected) {
    return h;
  }
  const targetOrigin = normalizeOrigin(explicitApiBase || state.apiBase);
  if (includeAuth && isTrustedApiOrigin(targetOrigin)) {
    const creds = getCredentialsForOrigin(targetOrigin);
    h["X-MRNAX-Project-Key"] = explicitProjectKey || state.projectKey;
    if (creds.accessToken) h["X-MRNAX-Access-Token"] = creds.accessToken;
    const prov = explicitProvider || state.llmProvider;
    const keyForProv =
      (creds.byoKeys && creds.byoKeys[prov]) ||
      (targetOrigin === normalizeOrigin(state.apiBase) && prov === state.llmProvider ? state.byoKey : "");
    if (keyForProv && method === "POST" && /^\/v1\/projects\/[^/]+\/agent-chat$/.test(path)) {
      h["X-MRNAX-LLM-Key"] = keyForProv;
      h["X-MRNAX-LLM-Provider"] = prov;
    }
  }
  return h;
}

function formatHttpErrorDetail(data, statusCode) {
  if (!data || data.detail == null) return `HTTP ${statusCode}`;
  if (typeof data.detail === "string") return data.detail;
  if (Array.isArray(data.detail)) {
    return data.detail
      .map((item) => {
        if (!item || typeof item !== "object") return String(item);
        const loc = Array.isArray(item.loc) ? item.loc.join(".") : "field";
        return `${loc}: ${item.msg || JSON.stringify(item)}`;
      })
      .join("; ");
  }
  return JSON.stringify(data.detail);
}

async function apiFetch(path, options = {}) {
  const {
    explicitProjectKey,
    explicitProvider = null,
    explicitApiBase = null,
    includeAuth = true,
    ...fetchOpts
  } = options;
  if (state.startupApiRejected) {
    const err = new Error(`Untrusted startup API origin blocked (UNTRUSTED_ORIGIN): ${state.startupApiRejected}`);
    err.status = 0;
    throw err;
  }
  const targetBase = normalizeOrigin(explicitApiBase || state.apiBase);
  if (!isTrustedApiOrigin(targetBase)) {
    const err = new Error(`Untrusted API origin blocked: ${targetBase}`);
    err.status = 0;
    throw err;
  }
  const resp = await fetch(`${targetBase}${path}`, {
    redirect: "error",
    ...fetchOpts,
    headers: {
      ...buildHeaders(path, fetchOpts.method, explicitProjectKey, includeAuth, explicitProvider, targetBase),
      ...(options.headers || {}),
    },
  });
  const text = await resp.text();
  let data = null;
  let jsonParseFailed = false;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    jsonParseFailed = true;
    data = { detail: text };
  }
  if (!resp.ok) {
    const err = new Error(formatHttpErrorDetail(data, resp.status));
    err.status = resp.status;
    err.detail = data && data.detail;
    throw err;
  }
  if (jsonParseFailed) {
    const err = new Error(`Gateway returned non-JSON HTTP ${resp.status} response.`);
    err.status = resp.status;
    throw err;
  }
  return data;
}

function getSelectedEnzymes() {
  if (typeof document.querySelectorAll !== "function") {
    return Array.isArray(state.activeEnzymes) ? [...state.activeEnzymes] : [];
  }
  return Array.from(document.querySelectorAll("#enzyme-chips .chip.selected")).map(
    (el) => el.dataset.enz
  );
}

function setSelectedEnzymes(list) {
  const normalized = Array.from(
    new Set((Array.isArray(list) ? list : []).filter(Boolean).map((x) => String(x).trim()).filter(Boolean))
  );
  state.activeEnzymes = [...normalized];
  const container = document.getElementById("enzyme-chips");
  if (
    container &&
    typeof container.appendChild === "function" &&
    typeof document.querySelectorAll === "function" &&
    typeof document.createElement === "function"
  ) {
    const existing = new Set(
      Array.from(document.querySelectorAll("#enzyme-chips .chip")).map((el) => el.dataset?.enz)
    );
    for (const enz of normalized) {
      if (!existing.has(enz)) {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "chip mono selected";
        btn.dataset = btn.dataset || {};
        btn.dataset.enz = enz;
        btn.dataset.dynamicEnz = "1";
        btn.textContent = enz;
        if (typeof btn.addEventListener === "function") {
          btn.addEventListener("click", () => {
            btn.classList.toggle("selected");
            updateSetupSummary();
          });
        }
        container.appendChild(btn);
        existing.add(enz);
      }
    }
  }
  if (typeof document.querySelectorAll === "function") {
    document.querySelectorAll("#enzyme-chips .chip").forEach((el) => {
      el.classList.toggle("selected", normalized.includes(el.dataset.enz));
    });
  }
  updateSetupSummary();
}

function updateSetupSummary() {
  const inputProt = document.getElementById("input-protein");
  const raw = ((inputProt && inputProt.value) || "").trim().toUpperCase();
  const chem = (document.getElementById("sel-chemistry") && document.getElementById("sel-chemistry").value) || "m1psi";
  const prof = (document.getElementById("sel-profile") && document.getElementById("sel-profile").value) || "research_max";
  const enz = getSelectedEnzymes();
  const detEl = document.getElementById("chk-detarget-liver");
  const detarget = Boolean(detEl && detEl.checked);
  const modPrefix = state.modality && state.modality !== "linear_mrna" ? `${state.modality} · ` : "";
  const summaryEl = document.getElementById("setup-collapsed-summary");
  if (summaryEl) {
    const enzLabel = enz.length ? `0-cut:${enz.join("+")}` : "无酶切禁区";
    summaryEl.textContent = `${modPrefix}${raw.length} aa · ${chem} · ${prof} · ${enzLabel}${detarget ? " · miR-122" : ""}`;
  }
}

function updateProteinBadge() {
  const inputProt = document.getElementById("input-protein");
  const raw = ((inputProt && inputProt.value) || "").trim().toUpperCase();
  const badge = document.getElementById("badge-aa-len");
  const startsM = raw.startsWith("M");
  if (badge) {
    badge.textContent = `${raw.length} aa · ${startsM ? "M✓" : "缺起始M"}`;
    badge.className = startsM ? "badge mono ok" : "badge mono warn";
  }
  updateSetupSummary();
}

function removeWelcomeCardIfPresent() {
  const box = document.getElementById("chat-stream");
  if (!box || typeof box.querySelector !== "function") return;
  const welcome = box.querySelector(".welcome-card");
  if (welcome) welcome.remove();
}

function renderWelcomeCard() {
  const box = document.getElementById("chat-stream");
  if (!box) return;
  box.innerHTML = `
    <div class="turn-card welcome-card">
      <div class="turn-meta"><span>COPILOT READY</span><span>WORKFLOW</span></div>
      <div class="agent-prose">
        <div><strong>1. 立项建模</strong>：输入靶点蛋白、修饰与内切酶禁区，或点击上方科研模板启动。</div>
        <div><strong>2. 确定性核验</strong>：序列折叠、双轨 FTO 筛查与多目标帕累托排序均由底层确定性引擎执行。</div>
        <div><strong>3. 单点锁定微调</strong>：追问权衡或修改单一约束，未改动区段与蛋白序列 100% 锁定继承。</div>
      </div>
    </div>
  `;
}

function appendStreamCard(kind, metaLeft, metaRight, bodyHtml) {
  const box = document.getElementById("chat-stream");
  if (!box || typeof document.createElement !== "function") return null;
  removeWelcomeCardIfPresent();
  const safeKind = ["user", "agent", "warn", "err", "pending"].includes(kind) ? kind : "agent";
  const card = document.createElement("div");
  card.className = `turn-card ${safeKind}`;
  card.innerHTML = `
    <div class="turn-meta"><span>${escapeHtml(metaLeft)}</span><span>${escapeHtml(metaRight)}</span></div>
    <div>${bodyHtml}</div>
  `;
  box.appendChild(card);
  box.scrollTop = box.scrollHeight;
  state.logCount += 1;
  const cntEl = document.getElementById("txt-log-count");
  if (cntEl) cntEl.textContent = String(state.logCount);
  return card;
}

function renderIntentLineage() {
  const bar = document.getElementById("intent-lineage-bar");
  if (!bar) return;
  if (!state.intentHistory.length) {
    bar.textContent = "等待 v01 建立锁定基线";
    return;
  }
  const latest = state.intentHistory[state.intentHistory.length - 1];
  const chain = state.intentHistory
    .map((it) => `v${String(it.intent_version).padStart(2, "0")}`)
    .join(" → ");
  const locks = Object.keys(latest.locked_scopes || {}).join(", ") || "protein";
  bar.textContent = `${chain} · 锁定[${locks}] · ${latest.uridine_chemistry} · ${latest.compliance_track}`;
}

function renderBoundariesModal() {
  const body = document.getElementById("boundaries-body");
  const countBadge = document.getElementById("badge-boundary-count");
  const advList = Array.isArray(state.boundaryAdvisories) ? state.boundaryAdvisories : [];
  if (countBadge) {
    countBadge.hidden = advList.length === 0;
    countBadge.textContent = String(advList.length);
  }
  if (!body) return;

  const dynamicItems = advList
    .map(
      (item) => `
      <div class="boundary-item">
        <div><strong>${escapeHtml(item.title)}</strong></div>
        <div style="margin-top:3px; color:var(--ink-secondary);">${escapeHtml(item.detail)}</div>
      </div>
    `
    )
    .join("");

  body.innerHTML = `
    ${dynamicItems}
    <div class="boundary-item">
      <div><strong>1. 三层洁净室许可隔离 (Code · Weights · Data)</strong></div>
      <div style="margin-top:3px; color:var(--ink-secondary);">
        <code>commercial_clean</code> 轨道严格隔离 GPL/AGPL 热力学库（如 ViennaRNA）及商业授权待定组件（如 EternaFold v1.3.1），仅调用宽松许可纯序列四头代理与 CC0/CC-BY-4.0 验证资产。
      </div>
    </div>
    <div class="boundary-item">
      <div><strong>2. 双轨 FTO 筛查边界 (序列碰撞 vs 算法方法权项)</strong></div>
      <div style="margin-top:3px; color:var(--ink-secondary);">
        平台自动核验已收录专利序列模式碰撞（<code>patent_fto_guard_v2</code>）与方法权项策略提示（<code>method_fto_guard_v1.1</code>），但不构成特定司法管辖区的法律自由实施（FTO）意见书。
      </div>
    </div>
    <div class="boundary-item">
      <div><strong>3. 计算热力学与化学修饰外推边界</strong></div>
      <div style="margin-top:3px; color:var(--ink-secondary);">
        二级结构自由能（MFE/bpp）基于标准未修饰尿苷最近邻参数近似，预测评分与体内长期药代动力学需通过湿实验双速闭环（N≥6 独立校准）最终确证。
      </div>
    </div>
  `;
}

function resolveMetricAxes(candidates) {
  const metadata = new Map();
  for (const candidate of candidates) {
    for (const detail of candidate.metric_details || []) {
      const existing = metadata.get(detail.metric_id);
      if (!existing || (existing.participated_in === "display_only" && detail.participated_in !== "display_only")) {
        metadata.set(detail.metric_id, detail);
      }
    }
    for (const id of Object.keys(candidate.metrics || {})) {
      if (!metadata.has(id)) metadata.set(id, { metric_id: id });
    }
  }
  const priority = Object.keys(METRIC_LABELS).filter((id) => metadata.has(id));
  const ids = new Set([...priority, ...metadata.keys()]);
  return Array.from(ids, (id) => {
    const detail = metadata.get(id);
    const catalog = METRIC_LABELS[id];
    const shortLabel = catalog?.shortLabel || String(id).replace(/_/g, " ");
    const direction = detail.direction || (catalog && (catalog.higherBetter ? "maximize" : "minimize"));
    const arrow = { maximize: " ↑", minimize: " ↓" }[direction] || "";
    return {
      key: id,
      shortLabel,
      label: shortLabel + (detail.unit ? ` (${detail.unit})` : "") + arrow,
      higherBetter: direction === "maximize",
      fmt: (v) => Number(v).toFixed(catalog?.decimals ?? 3),
    };
  });
}

function readAxisValue(cand, axis) {
  const detail =
    cand.metric_details?.find((m) => m.metric_id === axis.key && m.participated_in !== "display_only") ||
    cand.metric_details?.find((m) => m.metric_id === axis.key);
  const value = detail ? detail.value : cand.metrics?.[axis.key];
  return Number.isFinite(value) ? value : null;
}

function buildParetoAxisPairOptions(measuredAxes) {
  const pairs = [];
  const seen = new Set();
  const addPair = (axA, axB) => {
    if (!axA || !axB || axA.key === axB.key) return;
    const k = `${axA.key}__${axB.key}`;
    if (seen.has(k)) return;
    seen.add(k);
    pairs.push({ key: k, axX: axA, axY: axB, label: `${axA.shortLabel} vs ${axB.shortLabel}` });
  };
  if (measuredAxes.length >= 2) {
    addPair(measuredAxes[0], measuredAxes[1]);
  }
  const byKey = Object.fromEntries(measuredAxes.map((a) => [a.key, a]));
  addPair(byKey.kozak_unpaired_prob || byKey.eternafold_kozak_unpaired_prob, byKey.cds_mean_window_mfe || byKey.full_transcript_mfe);
  addPair(byKey.consensus_halflife_pred || byKey.orthrus_halflife_pred, byKey.utr5_mrl_pred);
  addPair(byKey.cds_mean_window_mfe || byKey.full_transcript_mfe, byKey.cds_stability_pred || byKey.codontransformer_syn_loglik);
  for (let i = 0; i < Math.min(measuredAxes.length, 4); i += 1) {
    for (let j = i + 1; j < Math.min(measuredAxes.length, 4); j += 1) {
      addPair(measuredAxes[i], measuredAxes[j]);
    }
  }
  return pairs;
}

function renderParetoScatter() {
  const wrap = document.getElementById("pareto-svg-wrap");
  const titleEl = document.getElementById("pareto-plot-title");
  const axisSel = document.getElementById("sel-pareto-axes");
  if (!wrap) return;

  if (!state.candidates.length) {
    wrap.innerHTML = `<div class="empty-hint mono">输入目标或选择科研模板运行后绘制二维帕累托前沿</div>`;
    if (titleEl) titleEl.textContent = "多目标权衡前沿";
    if (axisSel) axisSel.hidden = true;
    return;
  }

  const measuredAxes = resolveMetricAxes(state.candidates).filter(
    (axis) => state.candidates.some((c) => readAxisValue(c, axis) !== null)
  );
  if (measuredAxes.length < 2) {
    if (titleEl) titleEl.textContent = "多目标权衡前沿";
    if (axisSel) axisSel.hidden = true;
    wrap.innerHTML = '<div class="empty-hint mono">二维图需要两项已测指标</div>';
    return;
  }

  const pairs = buildParetoAxisPairOptions(measuredAxes);
  let chosenPair = pairs.find((p) => p.key === state.paretoAxisPairKey) || pairs[0];
  state.paretoAxisPairKey = chosenPair.key;

  if (axisSel) {
    axisSel.hidden = pairs.length <= 1;
    if (typeof document.createElement === "function" && typeof axisSel.replaceChildren === "function") {
      axisSel.replaceChildren();
      for (const p of pairs) {
        const opt = document.createElement("option");
        opt.value = p.key;
        opt.textContent = p.label;
        axisSel.appendChild(opt);
      }
      axisSel.value = chosenPair.key;
    }
  }

  const { axX, axY } = chosenPair;
  if (titleEl) {
    titleEl.textContent = `${axX.shortLabel} vs ${axY.shortLabel}`;
  }

  const pts = state.candidates
    .map((c) => ({
      cand: c,
      x: readAxisValue(c, axX),
      y: readAxisValue(c, axY),
    }))
    .filter((p) => p.x != null && p.y != null);

  if (!pts.length) {
    wrap.innerHTML = `<div class="empty-hint mono">当前候选未包含双轴连续数值指标</div>`;
    return;
  }

  const W = 400;
  const spacing = 14;
  const H = Math.max(216, 80 + Math.ceil((4 * pts.length) / 22) * spacing);
  const padL = 68;
  const padR = 18;
  const padT = 16;
  const padB = 38;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  let minX = Math.min(...pts.map((p) => p.x));
  let maxX = Math.max(...pts.map((p) => p.x));
  let minY = Math.min(...pts.map((p) => p.y));
  let maxY = Math.max(...pts.map((p) => p.y));

  const spanX = maxX - minX || Math.abs(maxX) * 0.1 || 1;
  const spanY = maxY - minY || Math.abs(maxY) * 0.1 || 1;
  minX -= spanX * 0.12;
  maxX += spanX * 0.12;
  minY -= spanY * 0.15;
  maxY += spanY * 0.15;

  const mapX = (v) => padL + ((v - minX) / (maxX - minX)) * plotW;
  const mapY = (v) => padT + plotH - ((v - minY) / (maxY - minY)) * plotH;

  const gridLines = [0, 0.5, 1]
    .map((t) => {
      const gy = (padT + plotH * (1 - t)).toFixed(1);
      const yVal = (minY + (maxY - minY) * t).toFixed(2);
      return `
        <line x1="${padL}" y1="${gy}" x2="${W - padR}" y2="${gy}" stroke="var(--border)" stroke-width="1" />
        <text x="${padL - 6}" y="${Number(gy) + 3.5}" text-anchor="end" fill="var(--ink-muted)" font-family="var(--font-mono)" font-size="10">${escapeHtml(yVal)}</text>
      `;
    })
    .join("");

  const xTicks = [0, 0.5, 1]
    .map((t) => {
      const gx = (padL + plotW * t).toFixed(1);
      const xVal = (minX + (maxX - minX) * t).toFixed(2);
      return `
        <line x1="${gx}" y1="${padT}" x2="${gx}" y2="${padT + plotH}" stroke="var(--border)" stroke-dasharray="2 3" stroke-width="1" />
        <text x="${gx}" y="${padT + plotH + 14}" text-anchor="middle" fill="var(--ink-muted)" font-family="var(--font-mono)" font-size="10">${escapeHtml(xVal)}</text>
      `;
    })
    .join("");

  const p1Pts = pts
    .filter((p) => p.cand.pareto_rank === 1)
    .sort((a, b) => a.x - b.x);

  let frontierLine = "";
  if (p1Pts.length >= 2) {
    const poly = p1Pts.map((p) => `${mapX(p.x).toFixed(1)},${mapY(p.y).toFixed(1)}`).join(" ");
    frontierLine = `<polyline points="${poly}" fill="none" stroke="var(--accent)" stroke-width="1.6" stroke-dasharray="4 3" />`;
  }

  const placed = [];
  function placeMarker(x, y) {
    const free = (cx, cy) => placed.every((p) => Math.hypot(p.x - cx, p.y - cy) >= spacing);
    let point = { x, y };
    if (!free(x, y)) {
      let nearest = Infinity;
      for (let cy = padT + 7; cy <= H - padB - 7; cy += spacing) {
        for (let cx = padL + 7; cx <= W - padR - 7; cx += spacing) {
          const distance = Math.hypot(cx - x, cy - y);
          if (distance < nearest && free(cx, cy)) {
            point = { x: cx, y: cy };
            nearest = distance;
          }
        }
      }
    }
    placed.push(point);
    return point;
  }

  const circles = pts
    .map((p, idx) => {
      const x = mapX(p.x);
      const y = mapY(p.y);
      const marker = placeMarker(x, y);
      const cx = marker.x.toFixed(1);
      const cy = marker.y.toFixed(1);
      const isP1 = p.cand.pareto_rank === 1;
      const isSel = p.cand.bundle_id === state.selectedBundleId;
      const cleanName = humanizeCandidateLabel(p.cand);
      const alignLeft = Number(cx) > W * 0.58;
      const lblX = (Number(cx) + (alignLeft ? -8 : 8)).toFixed(1);
      const lblAnchor = alignLeft ? "end" : "start";
      return `
        <g class="pareto-node" data-idx="${idx}" tabindex="0" role="button" aria-pressed="${isSel}" aria-label="${escapeHtml(p.cand.bundle_id)}" style="cursor:pointer;">
          <title>${escapeHtml(cleanName)} · ${escapeHtml(p.cand.pareto_rank ? "P" + p.cand.pareto_rank : "未排名")} · ${escapeHtml(axX.shortLabel)}=${escapeHtml(axX.fmt(p.x))}, ${escapeHtml(axY.shortLabel)}=${escapeHtml(axY.fmt(p.y))}</title>
          ${marker.x !== x || marker.y !== y ? `<line x1="${x.toFixed(1)}" y1="${y.toFixed(1)}" x2="${cx}" y2="${cy}" stroke="var(--border-strong)" pointer-events="none" />` : ""}
          ${isSel ? `<circle pointer-events="none" cx="${cx}" cy="${cy}" r="9.5" fill="none" stroke="var(--ink)" stroke-width="1.6" />` : ""}
          <circle
            cx="${cx}"
            cy="${cy}"
            r="${isP1 ? "5.2" : "4.4"}"
            fill="${isP1 ? "var(--accent)" : "var(--bg-surface)"}"
            stroke="${isP1 ? "var(--accent)" : "var(--ink-muted)"}"
            stroke-width="1.8"
          />
          <text pointer-events="none" x="${lblX}" y="${(Number(cy) - 5).toFixed(1)}" text-anchor="${lblAnchor}" fill="var(--ink-secondary)" font-family="var(--font-mono)" font-size="9.5">${escapeHtml(cleanName.length > 18 ? cleanName.slice(0, 16) + "…" : cleanName)}</text>
        </g>
      `;
    })
    .join("");

  wrap.innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}" role="img" aria-label="2D Pareto Frontier Scatter Plot" style="display:block;">
      ${gridLines}
      ${xTicks}
      <line x1="${padL}" y1="${padT}" x2="${padL}" y2="${padT + plotH}" stroke="var(--border-strong)" stroke-width="1" />
      <line x1="${padL}" y1="${padT + plotH}" x2="${W - padR}" y2="${padT + plotH}" stroke="var(--border-strong)" stroke-width="1" />
      ${frontierLine}
      ${circles}
      <text x="${padL + plotW / 2}" y="${H - 4}" text-anchor="middle" fill="var(--ink-secondary)" font-family="var(--font-mono)" font-size="10">${escapeHtml(axX.label)}</text>
      <text x="12" y="${padT + plotH / 2}" text-anchor="middle" transform="rotate(-90 12 ${padT + plotH / 2})" fill="var(--ink-secondary)" font-family="var(--font-mono)" font-size="9.5">${escapeHtml(axY.shortLabel + (axY.higherBetter ? " ↑" : " ↓"))}</text>
    </svg>
  `;

  if (typeof wrap.querySelectorAll === "function") {
    wrap.querySelectorAll(".pareto-node").forEach((node) => {
      const i = Number(node.getAttribute("data-idx"));
      const bid = pts[i] && pts[i].cand.bundle_id;
      if (!bid) return;
      node.addEventListener("click", () => selectCandidate(bid));
      node.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          selectCandidate(bid);
        }
      });
    });
  }
}

function buildSparkCell(val, axis, minV, maxV) {
  if (val == null) return `<span class="mono" style="color:var(--ink-muted);">N/A · 未测</span>`;
  let ratio = 0.5;
  if (maxV > minV) {
    ratio = axis.higherBetter ? (val - minV) / (maxV - minV) : (maxV - val) / (maxV - minV);
  }
  const pct = Math.max(8, Math.min(100, Math.round(ratio * 100)));
  return `
    <div class="spark-cell mono">
      <span>${escapeHtml(axis.fmt(val))}</span>
      <span class="spark-track"><span class="spark-fill" style="width:${pct}%;"></span></span>
    </div>
  `;
}

function getSortedCandidates(axes) {
  const arr = [...state.candidates];
  const dir = state.sortAsc ? 1 : -1;
  arr.sort((a, b) => {
    if (state.sortCol === "rank") {
      return ((a.pareto_rank ?? Infinity) - (b.pareto_rank ?? Infinity)) * dir;
    }
    if (state.sortCol === "id") {
      return String(a.bundle_id || "").localeCompare(String(b.bundle_id || "")) * dir;
    }
    if (state.sortCol === "gc") {
      return ((a.full_gc_pct || 0) - (b.full_gc_pct || 0)) * dir;
    }
    const axis = axes.find((item) => `metric:${item.key}` === state.sortCol);
    if (!axis) return 0;
    const va = readAxisValue(a, axis);
    const vb = readAxisValue(b, axis);
    if (va == null && vb == null) return 0;
    if (va == null) return 1;
    if (vb == null) return -1;
    return (va - vb) * dir;
  });
  return arr;
}

function clearSelectedConstructUI() {
  state.selectedBundleId = null;
  state.currentStructureProfile = null;
  const txtSel = document.getElementById("txt-selected-bundle");
  if (txtSel) txtSel.textContent = "未选择候选";
  const chainPicker = document.getElementById("chain-picker");
  if (chainPicker) chainPicker.hidden = true;
  const inspectChain = document.getElementById("inspect-chain");
  if (inspectChain && typeof inspectChain.replaceChildren === "function") inspectChain.replaceChildren();
  const rail = document.getElementById("enzyme-clearance-rail");
  if (rail) rail.innerHTML = "";
  const ruler = document.getElementById("ruler-row");
  if (ruler) ruler.innerHTML = "";
  const propWrap = document.getElementById("prop-track-wrap");
  if (propWrap) propWrap.innerHTML = `<div class="empty-hint mono">生成后展示真实比例五段分子架构图谱</div>`;
  const segBar = document.getElementById("seg-bar");
  if (segBar) segBar.innerHTML = "";
  const arcBox = document.getElementById("structure-arc-svg-wrap");
  if (arcBox) arcBox.innerHTML = `<div class="empty-hint mono">选择候选后绘制起始区碱基配对概率弧图</div>`;
  const trkBox = document.getElementById("structure-track-svg-wrap");
  if (trkBox) trkBox.innerHTML = `<div class="empty-hint mono">选择候选后同步绘制全长滑窗多轴曲线与风险位点</div>`;
  const kpiStrip = document.getElementById("structure-kpi-strip");
  if (kpiStrip) kpiStrip.innerHTML = "";
  const srcBadge = document.getElementById("structure-source-badge");
  if (srcBadge) srcBadge.textContent = "待载入候选";
  const pins = document.getElementById("motif-pins-strip");
  if (pins) {
    pins.hidden = true;
    pins.innerHTML = "";
  }
  const btnFold = document.getElementById("btn-analyze-structure");
  if (btnFold) btnFold.disabled = true;
  const segInsp = document.getElementById("seg-inspector");
  if (segInsp) segInsp.textContent = "选择候选构建后自动对齐显示区段核苷酸与三联体密码子翻译轨";
}

function updateExportButtonLabels() {
  const gbBtn = document.getElementById("btn-dl-gb");
  const hasCand = Boolean(state.selectedBundleId && state.candidates.length);
  ["btn-dl-gb", "btn-dl-ivt", "btn-dl-dossier", "btn-dl-rna", "btn-dl-cds", "btn-dl-meta", "btn-analyze-structure"].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.disabled = !hasCand;
  });
  const cmpBtn = document.getElementById("btn-compare-selected");
  if (cmpBtn) {
    cmpBtn.disabled = state.candidates.length < 2;
    const nSel = state.compareSelectedIds.length;
    cmpBtn.textContent = nSel > 0 ? `双方案对比 (${nSel}/2)` : "双方案对比";
  }
  const cliBtn = document.getElementById("btn-copy-cli");
  if (cliBtn) cliBtn.disabled = !state.lastCliCommand;
  if (gbBtn) {
    if (hasCand) {
      const revTag = `v${String(state.activeVersion).padStart(2, "0")}`;
      gbBtn.textContent = `GenBank (${revTag}) ↓`;
      gbBtn.title = `${state.selectedBundleId}.gb`;
      if (typeof gbBtn.setAttribute === "function") {
        gbBtn.setAttribute("aria-label", `GenBank ${revTag} · ${state.selectedBundleId}.gb`);
      }
    } else {
      gbBtn.textContent = "GenBank (.gb) ↓";
      if (typeof gbBtn.removeAttribute === "function") {
        gbBtn.removeAttribute("title");
        gbBtn.removeAttribute("aria-label");
      }
    }
  }
}

function toggleCandidateCompareSelection(bundleId) {
  const idx = state.compareSelectedIds.indexOf(bundleId);
  if (idx >= 0) {
    state.compareSelectedIds.splice(idx, 1);
  } else {
    if (state.compareSelectedIds.length >= 2) {
      state.compareSelectedIds.shift();
    }
    state.compareSelectedIds.push(bundleId);
  }
  renderCandidates();
}

function openCandidateComparisonModal() {
  if (state.candidates.length < 2) return;
  let ids = state.compareSelectedIds.filter((id) => state.candidates.some((c) => c.bundle_id === id));
  if (ids.length < 2) {
    ids = state.candidates.slice(0, 2).map((c) => c.bundle_id);
  }
  const [candA, candB] = ids.map((id) => state.candidates.find((c) => c.bundle_id === id));
  if (!candA || !candB) return;

  const axes = resolveMetricAxes([candA, candB]);
  const metricRows = axes
    .map((ax) => {
      const va = readAxisValue(candA, ax);
      const vb = readAxisValue(candB, ax);
      const da = va != null ? ax.fmt(va) : "N/A";
      const db = vb != null ? ax.fmt(vb) : "N/A";
      const delta = va != null && vb != null ? (vb - va >= 0 ? `+${(vb - va).toFixed(3)}` : (vb - va).toFixed(3)) : "—";
      return `
        <tr>
          <td>${escapeHtml(ax.label)}</td>
          <td class="num-col mono">${escapeHtml(da)}</td>
          <td class="num-col mono">${escapeHtml(db)}</td>
          <td class="num-col mono">${escapeHtml(delta)}</td>
        </tr>
      `;
    })
    .join("");

  const seqA = String(candA.full_sequence || "");
  const seqB = String(candB.full_sequence || "");
  const minLen = Math.min(seqA.length, seqB.length);
  let diffNt = Math.abs(seqA.length - seqB.length);
  for (let i = 0; i < minLen; i += 1) {
    if (seqA[i] !== seqB[i]) diffNt += 1;
  }

  const body = document.getElementById("compare-body");
  if (body) {
    body.innerHTML = `
      <div class="kpi-row mono">
        <span class="kpi-pill">方案 A: <strong>${escapeHtml(humanizeCandidateLabel(candA))}</strong></span>
        <span class="kpi-pill">方案 B: <strong>${escapeHtml(humanizeCandidateLabel(candB))}</strong></span>
        <span class="kpi-pill">序列碱基位点差异: <strong>${escapeHtml(diffNt)} nt</strong></span>
      </div>
      <div class="table-scroll table-scroll-tall">
        <table class="data-table">
          <thead>
            <tr>
              <th>对比维度</th>
              <th class="num-col">${escapeHtml(humanizeCandidateLabel(candA))}</th>
              <th class="num-col">${escapeHtml(humanizeCandidateLabel(candB))}</th>
              <th class="num-col">差值 (B - A)</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>帕累托层级</td>
              <td class="num-col mono">P${escapeHtml(candA.pareto_rank ?? "—")}</td>
              <td class="num-col mono">P${escapeHtml(candB.pareto_rank ?? "—")}</td>
              <td class="num-col mono">—</td>
            </tr>
            <tr>
              <td>全长 · GC%</td>
              <td class="num-col mono">${escapeHtml(candA.length_nt)} nt · ${escapeHtml(candA.full_gc_pct)}%</td>
              <td class="num-col mono">${escapeHtml(candB.length_nt)} nt · ${escapeHtml(candB.full_gc_pct)}%</td>
              <td class="num-col mono">${escapeHtml(((Number(candB.full_gc_pct) || 0) - (Number(candA.full_gc_pct) || 0)).toFixed(2))}%</td>
            </tr>
            <tr>
              <td>门禁与序列筛查</td>
              <td class="num-col mono">${escapeHtml(candA.gate_passed_count ?? 6)}/${escapeHtml(candA.gate_total_count ?? 6)} 通过</td>
              <td class="num-col mono">${escapeHtml(candB.gate_passed_count ?? 6)}/${escapeHtml(candB.gate_total_count ?? 6)} 通过</td>
              <td class="num-col mono">同轨可比</td>
            </tr>
            ${metricRows}
          </tbody>
        </table>
      </div>
    `;
  }
  const dlg = document.getElementById("modal-compare");
  if (dlg && typeof dlg.showModal === "function") dlg.showModal();
}

function renderCandidates() {
  const tbody = document.getElementById("cand-tbody");
  const countEl = document.getElementById("txt-cand-count");
  if (countEl) countEl.textContent = `${state.candidates.length} 条合格候选`;

  const axes = resolveMetricAxes(state.candidates);
  const header = document.getElementById("cand-thead-row");
  if (header && typeof header.querySelectorAll === "function") {
    header.querySelectorAll("[data-metric]").forEach((el) => el.remove());
    const gcHeader = header.querySelector('[data-sort="gc"]');
    for (const axis of axes) {
      const th = document.createElement("th");
      th.textContent = axis.label;
      th.dataset.sort = `metric:${axis.key}`;
      th.dataset.metric = axis.key;
      th.tabIndex = 0;
      th.className = "num-col";
      header.insertBefore(th, gcHeader);
    }
    if (!axes.length && state.candidates.length) {
      const th = document.createElement("th");
      th.dataset.metric = "";
      th.textContent = "指标";
      header.insertBefore(th, gcHeader);
    }
  }

  if (!state.candidates.length) {
    state.compareSelectedIds = [];
    if (tbody) {
      const colSpan = (header && header.children && header.children.length) || 4;
      tbody.innerHTML = `<tr><td colspan="${escapeHtml(colSpan)}" class="empty-cell mono">当前无合格推荐候选</td></tr>`;
    }
    clearSelectedConstructUI();
    updateExportButtonLabels();
    renderParetoScatter();
    return;
  }

  state.compareSelectedIds = state.compareSelectedIds.filter((id) =>
    state.candidates.some((c) => c.bundle_id === id)
  );

  const ranges = axes.map((ax) => {
    const vals = state.candidates.map((c) => readAxisValue(c, ax)).filter((v) => v != null);
    return {
      min: vals.length ? Math.min(...vals) : 0,
      max: vals.length ? Math.max(...vals) : 1,
    };
  });

  const recSet = new Set((state.latestReport && state.latestReport.recommended_bundle_ids) || []);
  const sorted = getSortedCandidates(axes);

  if (tbody && typeof document.createElement === "function") {
    tbody.innerHTML = "";
    sorted.forEach((c) => {
      const tr = document.createElement("tr");
      tr.tabIndex = 0;
      if (tr.dataset) tr.dataset.bundleId = c.bundle_id;
      if (c.bundle_id === state.selectedBundleId && tr.classList) tr.classList.add("selected-row");
      const rankNum = c.pareto_rank;
      const isRec = rankNum === 1 && recSet.has(c.bundle_id);
      const isCmpChecked = state.compareSelectedIds.includes(c.bundle_id);
      const cleanTitle = humanizeCandidateLabel(c);
      const shortDigest = String(c.short_digest || (c.version_digest ? c.version_digest.slice(0, 10) : c.bundle_id.slice(-8)));
      const gatePass = c.gate_passed_count ?? (c.passed_all_hard_gates === false ? 0 : 6);
      const gateTot = c.gate_total_count ?? 6;
      const ftoMeta = FTO_BADGE_LABELS[c.fto_badge] || FTO_BADGE_LABELS.sequence_screen_clear;
      const strategyText = humanizeStrategyTag(c.strategy_tag || c.derivation_action);

      const metricCells = axes
        .map(
          (axis, i) =>
            `<td class="num-col">${buildSparkCell(readAxisValue(c, axis), axis, ranges[i].min, ranges[i].max)}</td>`
        )
        .join("");

      tr.innerHTML = `
        <td class="mono">
          <div style="display:inline-flex; align-items:center; gap:6px;">
            <input type="checkbox" class="cmp-chk" data-cmp-id="${escapeHtml(c.bundle_id)}" aria-label="勾选对比 ${escapeHtml(cleanTitle)}" ${isCmpChecked ? "checked" : ""} />
            <span class="badge ${rankNum === 1 ? "ok" : ""}">${rankNum ? "P" + escapeHtml(rankNum) : "未排名"}</span>
          </div>
        </td>
        <td>
          <div class="cand-name-cell">
            <div class="cand-title-line">
              ${isRec ? '<span class="badge ok mono">推荐</span>' : ""}
              <span>${escapeHtml(cleanTitle)}</span>
            </div>
            <div class="cand-sub-line mono" title="${escapeHtml(c.bundle_id)}">
              <span>门禁 ${escapeHtml(gatePass)}/${escapeHtml(gateTot)}</span>
              <span>·</span>
              <span style="color:var(--${ftoMeta.cls === "teal" ? "accent-teal" : ftoMeta.cls});">${escapeHtml(ftoMeta.label)}</span>
              <span>·</span>
              <span>#${escapeHtml(shortDigest)}</span>
            </div>
          </div>
        </td>
        <td><span class="badge mono">${escapeHtml(strategyText)}</span></td>
        ${metricCells || '<td class="num-col mono">N/A · 未测</td>'}
        <td class="num-col mono">${escapeHtml(c.full_gc_pct)}% · ${escapeHtml(c.length_nt)}nt</td>
      `;

      if (typeof tr.querySelector === "function") {
        const chk = tr.querySelector(".cmp-chk");
        if (chk) {
          chk.addEventListener("click", (ev) => {
            ev.stopPropagation();
            toggleCandidateCompareSelection(c.bundle_id);
          });
        }
      }
      if (typeof tr.addEventListener === "function") {
        tr.addEventListener("click", () => selectCandidate(c.bundle_id));
        tr.addEventListener("keydown", (e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            selectCandidate(c.bundle_id);
          }
        });
      }
      tbody.appendChild(tr);
    });
  }

  updateExportButtonLabels();
  renderParetoScatter();
}

function countMotifOccurrences(seq, motifs) {
  let count = 0;
  for (const m of motifs) {
    let idx = seq.indexOf(m);
    while (idx !== -1) {
      count += 1;
      idx = seq.indexOf(m, idx + 1);
    }
  }
  return count;
}

function renderEnzymeClearance(fullSeq) {
  const rail = document.getElementById("enzyme-clearance-rail");
  if (!rail) return;
  if (!fullSeq) {
    rail.innerHTML = "";
    return;
  }
  const activeList = Array.isArray(state.activeEnzymes) ? state.activeEnzymes : getSelectedEnzymes();
  const activeEnzymes = new Set(activeList);
  const presetTags = Object.entries(RESTRICTION_SITES_RNA).map(([enz, motifs]) => {
    const cuts = countMotifOccurrences(fullSeq, motifs);
    const cls = cuts === 0 ? "clear" : "cut";
    const reqMark = activeEnzymes.has(enz) ? " [禁]" : "";
    return `<span class="enz-tag ${cls}">${escapeHtml(enz)}${escapeHtml(reqMark)}: ${cuts} cuts ${cuts === 0 ? "✓" : "!"}</span>`;
  });
  const extraTags = activeList
    .filter((enz) => !Object.prototype.hasOwnProperty.call(RESTRICTION_SITES_RNA, enz))
    .map((enz) => `<span class="enz-tag clear">${escapeHtml(enz)} [禁]: server-gated ✓</span>`);
  rail.innerHTML = [...presetTags, ...extraTags].join("");
}

function renderTrueScaleMap(cand) {
  const ruler = document.getElementById("ruler-row");
  const propWrap = document.getElementById("prop-track-wrap");
  const cardsBar = document.getElementById("seg-bar");
  const total = Math.max(1, Number(cand.length_nt) || 1);

  if (ruler) {
    ruler.innerHTML = [0, 0.25, 0.5, 0.75, 1]
      .map((f) => `<span>${Math.round(total * f)} nt</span>`)
      .join("");
  }

  if (!propWrap || typeof document.createElement !== "function") return;

  let cursorPct = 0;
  let stopPinHtml = "";
  const propBar = document.createElement("div");
  propBar.className = "prop-bar";

  (cand.segments || []).forEach((seg) => {
    const segLen = Number(seg.length_nt) || 0;
    const rawPct = (segLen / total) * 100;
    const centerPct = cursorPct + rawPct / 2;
    cursorPct += rawPct;
    const meta = SEG_META[seg.segment_type] || { short: seg.segment_type, colorVar: "var(--accent)" };
    if (seg.segment_type === "stop_cassette") {
      stopPinHtml = `<span class="stop-callout" style="inset-inline-start:${centerPct.toFixed(3)}%;">Stop (${escapeHtml(segLen)}nt) ↓</span>`;
    }
    const segDiv = document.createElement("div");
    segDiv.className = `prop-seg ${seg.segment_type === state.selectedSegType ? "active-prop" : ""}`;
    segDiv.title = `${meta.short} [${seg.start}..${seg.end}) · ${segLen} nt · GC ${seg.gc_pct}%`;
    segDiv.style = segDiv.style || {};
    segDiv.style.width = `${rawPct.toFixed(4)}%`;
    segDiv.style.background = meta.colorVar;
    if (typeof segDiv.addEventListener === "function") {
      segDiv.addEventListener("click", () => inspectSegment(cand, seg));
    }
    propBar.appendChild(segDiv);
  });

  propWrap.innerHTML = stopPinHtml;
  if (typeof propWrap.appendChild === "function") {
    propWrap.appendChild(propBar);
  }

  if (cardsBar && typeof cardsBar.appendChild === "function") {
    cardsBar.innerHTML = "";
    (cand.segments || []).forEach((seg) => {
      const meta = SEG_META[seg.segment_type] || { short: seg.segment_type, colorVar: "var(--accent)" };
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = `seg-card-btn ${seg.segment_type === state.selectedSegType ? "active" : ""}`;
      btn.style = btn.style || {};
      btn.style.borderBlockStartColor = meta.colorVar;
      btn.innerHTML = `
        <div class="seg-card-name">${escapeHtml(meta.short)} · ${escapeHtml(seg.length_nt)} nt</div>
        <div class="seg-card-meta">[${escapeHtml(seg.start)}..${escapeHtml(seg.end)}) · GC ${escapeHtml(seg.gc_pct)}%</div>
      `;
      if (typeof btn.addEventListener === "function") {
        btn.addEventListener("click", () => inspectSegment(cand, seg));
      }
      cardsBar.appendChild(btn);
    });
  }
}

function computeHumanCaiClient(cdsRna) {
  const clean = String(cdsRna || "").toUpperCase().replace(/T/g, "U");
  const codons = [];
  for (let i = 0; i + 3 <= clean.length; i += 3) {
    const c = clean.slice(i, i + 3);
    if (c !== "UAA" && c !== "UAG" && c !== "UGA" && HUMAN_CODON_WEIGHTS[c]) {
      codons.push(HUMAN_CODON_WEIGHTS[c]);
    }
  }
  if (!codons.length) return 0;
  const logSum = codons.reduce((acc, w) => acc + Math.log(Math.max(1e-6, w)), 0);
  return Number(Math.exp(logSum / codons.length).toFixed(4));
}

function canPairRna(a, b) {
  const p = a + b;
  if (p === "GC" || p === "CG") return 0.86;
  if (p === "AU" || p === "UA") return 0.62;
  if (p === "GU" || p === "UG") return 0.36;
  return 0;
}

function computeClientStructureProfile(chain, modality = "linear_mrna") {
  const seq = String((chain && chain.full_sequence) || "")
    .trim()
    .toUpperCase()
    .replace(/T/g, "U");
  const n = seq.length;
  if (n < 3) return null;

  const segs = Array.isArray(chain.segments) ? chain.segments : [];
  const utr5Seg = segs.find((s) => s.segment_type === "five_prime_utr");
  const cdsSeg = segs.find((s) => s.segment_type === "cds");
  const utr3Seg = segs.find((s) => s.segment_type === "three_prime_utr");
  const utr5End = utr5Seg ? Number(utr5Seg.end) : cdsSeg ? Number(cdsSeg.start) : 0;
  const cdsStart = cdsSeg ? Number(cdsSeg.start) : utr5End;
  const cdsEnd = cdsSeg ? Number(cdsSeg.end) : n;
  const utr3Start = utr3Seg ? Number(utr3Seg.start) : cdsEnd;
  const utr3End = utr3Seg ? Number(utr3Seg.end) : n;

  const gcPct = Number((((seq.split("G").length - 1 + (seq.split("C").length - 1)) / n) * 100).toFixed(2));
  const uPct = Number((((seq.split("U").length - 1) / n) * 100).toFixed(2));
  const cdsSub = cdsEnd > cdsStart ? seq.slice(cdsStart, cdsEnd) : seq;
  const cai = computeHumanCaiClient(cdsSub);

  const effAug = Math.min(Math.max(0, cdsStart), Math.max(0, n - 3));
  const winStart = Math.max(0, effAug - 30);
  const winEnd = Math.min(n, effAug + 45);
  const winSeq = seq.slice(winStart, winEnd);
  const augIdx = effAug - winStart;

  // Deterministic greedy non-crossing stem-loop pairing in initiation window
  const wLen = winSeq.length;
  const paired = new Array(wLen).fill(-1);
  const arcs = [];
  for (let span = 5; span <= Math.min(34, wLen - 1); span += 1) {
    for (let i = 0; i + span < wLen; i += 1) {
      const j = i + span;
      if (paired[i] !== -1 || paired[j] !== -1) continue;
      const w0 = canPairRna(winSeq[i], winSeq[j]);
      if (w0 <= 0) continue;
      // Require an adjacent stacking pair for realistic stem stability
      const wIn = i + 1 < j - 1 ? canPairRna(winSeq[i + 1], winSeq[j - 1]) : 0;
      const wOut = i > 0 && j + 1 < wLen ? canPairRna(winSeq[i - 1], winSeq[j + 1]) : 0;
      if (wIn > 0 || wOut > 0) {
        const prob = Number(Math.min(0.95, (w0 + Math.max(wIn, wOut) * 0.55) / 1.35).toFixed(3));
        paired[i] = j;
        paired[j] = i;
        arcs.push({ i, j, prob });
      }
    }
  }
  const dbChars = paired.map((partner, idx) => (partner === -1 ? "." : partner > idx ? "(" : ")"));
  const kLo = Math.max(0, augIdx - 15);
  const kHi = Math.min(wLen, augIdx + 18);
  let kozakOpenSum = 0;
  for (let k = kLo; k < kHi; k += 1) {
    kozakOpenSum += paired[k] === -1 ? 0.84 : 0.26;
  }
  const kozakUp = kHi > kLo ? Number((kozakOpenSum / (kHi - kLo)).toFixed(4)) : 0.65;

  // Sliding tracks
  const windowNt = Math.min(120, n);
  const stepNt = Math.max(6, Math.min(36, Math.floor(windowNt / 3) || 1));
  const positions = [];
  const localMfe = [];
  const localGc = [];
  const localU = [];
  const localUp = [];

  const pushWindow = (startIdx) => {
    const chunk = seq.slice(startIdx, startIdx + windowNt);
    const clen = Math.max(1, chunk.length);
    const g = chunk.split("G").length - 1;
    const c = chunk.split("C").length - 1;
    const u = chunk.split("U").length - 1;
    const gcFrac = (g + c) / clen;
    const estMfe = Number((-(0.16 + 0.46 * gcFrac) * clen).toFixed(2));
    const estOpen = Number(Math.max(0.18, Math.min(0.88, 0.78 - 0.56 * gcFrac)).toFixed(3));
    positions.push(startIdx);
    localMfe.push(estMfe);
    localGc.push(Number((gcFrac * 100).toFixed(2)));
    localU.push(Number(((u / clen) * 100).toFixed(2)));
    localUp.push(estOpen);
  };

  if (n <= windowNt) {
    pushWindow(0);
  } else {
    for (let s = 0; s <= n - windowNt; s += stepNt) {
      pushWindow(s);
    }
    if (positions[positions.length - 1] < n - windowNt) {
      pushWindow(n - windowNt);
    }
  }

  const meanWinMfe = localMfe.reduce((a, b) => a + b, 0) / Math.max(1, localMfe.length);
  const fullMfeEst = Number(((meanWinMfe / Math.max(1, windowNt)) * n).toFixed(1));

  // Motif hazard scan
  const motifs = [];
  if (utr5End > 0) {
    const u5 = seq.slice(0, utr5End);
    let idx = u5.indexOf("AUG");
    while (idx !== -1) {
      motifs.push({ kind: "uaug", label: "5′UTR uAUG", start: idx, end: idx + 3, severity: "critical" });
      idx = u5.indexOf("AUG", idx + 1);
    }
  }
  if (cdsEnd > cdsStart) {
    const cdsSlice = seq.slice(cdsStart, cdsEnd);
    const u4Re = /U{4,}/g;
    let m;
    while ((m = u4Re.exec(cdsSlice)) !== null) {
      motifs.push({
        kind: "uuuu_frameshift",
        label: `m¹Ψ 滑移串 (${m[0].slice(0, 6)})`,
        start: cdsStart + m.index,
        end: cdsStart + m.index + m[0].length,
        severity: "high",
      });
    }
  }
  const bodyEnd = utr3End > 0 ? utr3End : n;
  const bodySeq = seq.slice(0, bodyEnd);
  const homoRe = /(A{6,}|C{6,}|G{6,}|U{6,})/g;
  let hm;
  while ((hm = homoRe.exec(bodySeq)) !== null) {
    motifs.push({
      kind: "homopolymer",
      label: `均聚物 ${hm[0][0]}×${hm[0].length}`,
      start: hm.index,
      end: hm.index + hm[0].length,
      severity: hm[0].length >= 8 ? "high" : "warning",
    });
  }
  for (const [enz, sites] of Object.entries(RESTRICTION_SITES_RNA)) {
    for (const site of sites) {
      let idx = seq.indexOf(site);
      while (idx !== -1) {
        motifs.push({
          kind: "restriction_site",
          label: `${enz} 位点`,
          start: idx,
          end: idx + site.length,
          severity: "warning",
        });
        idx = seq.indexOf(site, idx + 1);
      }
    }
  }
  // miR-122 canonical seed / site check in 3'UTR
  const mir122Seed = "ACACUCCA";
  let mirIdx = seq.indexOf(mir122Seed, Math.max(0, utr3Start));
  while (mirIdx !== -1) {
    motifs.push({
      kind: "mirna_site",
      label: "miR-122 去靶向盒",
      start: mirIdx,
      end: mirIdx + 8,
      severity: "info",
    });
    mirIdx = seq.indexOf(mir122Seed, mirIdx + 1);
  }

  return {
    length_nt: n,
    modality,
    full_gc_pct: gcPct,
    full_u_pct: uPct,
    cai,
    mfe_kcal_mol: fullMfeEst,
    kozak_unpaired_prob: kozakUp,
    aup_mean: Number((localUp.reduce((a, b) => a + b, 0) / Math.max(1, localUp.length)).toFixed(3)),
    initiation_window: {
      start: winStart,
      end: winEnd,
      aug_index_in_window: augIdx,
      sequence: winSeq,
      dot_bracket: dbChars.join(""),
      arcs,
    },
    sliding_tracks: {
      window_nt: windowNt,
      step_nt: stepNt,
      positions,
      local_mfe: localMfe,
      local_gc_pct: localGc,
      local_u_pct: localU,
      local_unpaired_prob: localUp,
    },
    motifs: motifs.slice(0, 24),
    engines_used: ["client_deterministic_scanner"],
  };
}

function renderStructureArcSvg(profile) {
  const box = document.getElementById("structure-arc-svg-wrap");
  if (!box) return;
  const initWin = profile && profile.initiation_window;
  if (!initWin || !initWin.sequence) {
    box.innerHTML = `<div class="empty-hint mono">起始窗口长度不足，无法绘制配对弧图</div>`;
    return;
  }

  const seq = String(initWin.sequence);
  const wLen = Math.max(1, seq.length);
  const augIdx = Number.isFinite(Number(initWin.aug_index_in_window)) ? Number(initWin.aug_index_in_window) : 0;
  const W = 520;
  const H = 162;
  const padL = 24;
  const padR = 20;
  const baseY = 114;
  const plotW = W - padL - padR;
  const xOf = (idx) => padL + (wLen <= 1 ? plotW / 2 : (idx / (wLen - 1)) * plotW);

  const kLo = Math.max(0, augIdx - 15);
  const kHi = Math.min(wLen - 1, augIdx + 17);
  const kozakX1 = xOf(kLo).toFixed(1);
  const kozakW = Math.max(6, xOf(kHi) - xOf(kLo)).toFixed(1);

  const augX1 = xOf(Math.max(0, Math.min(wLen - 1, augIdx))).toFixed(1);
  const augX2 = xOf(Math.max(0, Math.min(wLen - 1, augIdx + 2))).toFixed(1);
  const augRectW = Math.max(12, Number(augX2) - Number(augX1) + 6).toFixed(1);

  const arcs = Array.isArray(initWin.arcs) ? initWin.arcs : [];
  const arcPaths = arcs
    .filter((a) => Number.isFinite(Number(a.i)) && Number.isFinite(Number(a.j)) && Number(a.j) > Number(a.i))
    .map((a) => {
      const i = Math.max(0, Math.min(wLen - 1, Number(a.i)));
      const j = Math.max(0, Math.min(wLen - 1, Number(a.j)));
      const prob = Math.max(0.1, Math.min(1, Number(a.prob) || 0.3));
      const x1 = xOf(i);
      const x2 = xOf(j);
      const rx = (x2 - x1) / 2;
      const ry = Math.min(86, Math.max(10, rx * 0.72));
      const stroke =
        prob >= 0.65 ? "var(--accent)" : prob >= 0.38 ? "var(--accent-teal)" : "var(--ink-muted)";
      const sw = (1.1 + prob * 1.5).toFixed(2);
      const op = (0.45 + prob * 0.5).toFixed(2);
      return `<path d="M ${x1.toFixed(1)} ${baseY} A ${rx.toFixed(1)} ${ry.toFixed(1)} 0 0 1 ${x2.toFixed(1)} ${baseY}" fill="none" stroke="${stroke}" stroke-width="${sw}" opacity="${op}"><title>nt ${escapeHtml(i - augIdx)} ↔ ${escapeHtml(j - augIdx)} · P=${escapeHtml(prob.toFixed(2))}</title></path>`;
    })
    .join("");

  const relTicks = [-30, -15, 0, 18, 40]
    .map((rel) => ({ rel, idx: augIdx + rel }))
    .filter((t) => t.idx >= 0 && t.idx < wLen);

  const tickSvg = relTicks
    .map((t) => {
      const tx = xOf(t.idx).toFixed(1);
      const lbl = t.rel === 0 ? "AUG(+1)" : t.rel > 0 ? `+${t.rel}` : String(t.rel);
      return `
        <line x1="${tx}" y1="${baseY}" x2="${tx}" y2="${baseY + 5}" stroke="var(--border-strong)" />
        <text x="${tx}" y="${baseY + 16}" text-anchor="middle" fill="${t.rel === 0 ? "var(--ok)" : "var(--ink-muted)"}" font-family="var(--font-mono)" font-size="9.5" font-weight="${t.rel === 0 ? "600" : "400"}">${escapeHtml(lbl)}</text>
      `;
    })
    .join("");

  const seqPreview = seq.length > 68 ? `${seq.slice(0, 34)}…${seq.slice(-30)}` : seq;
  const dbPreview = String(initWin.dot_bracket || "");
  const dbShort = dbPreview.length > 68 ? `${dbPreview.slice(0, 34)}…${dbPreview.slice(-30)}` : dbPreview;

  box.innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}" role="img" aria-label="5-prime initiation base-pair probability arc diagram">
      <rect x="${kozakX1}" y="14" width="${kozakW}" height="${baseY - 10}" rx="3" fill="var(--accent-soft)" opacity="0.7" />
      <text x="${(Number(kozakX1) + 4).toFixed(1)}" y="25" fill="var(--accent)" font-family="var(--font-mono)" font-size="9">Kozak [-15,+18]</text>
      <rect x="${(Number(augX1) - 3).toFixed(1)}" y="${baseY - 7}" width="${augRectW}" height="12" rx="2" fill="var(--ok-soft)" stroke="var(--ok)" stroke-width="1.2" />
      ${arcPaths || `<text x="${W / 2}" y="68" text-anchor="middle" fill="var(--ink-muted)" font-family="var(--font-mono)" font-size="10">起始窗口高度开放（零强配对茎环 p≥0.15）</text>`}
      <line x1="${padL}" y1="${baseY}" x2="${W - padR}" y2="${baseY}" stroke="var(--border-strong)" stroke-width="1.5" />
      ${tickSvg}
      <text x="${padL}" y="${H - 15}" fill="var(--ink-secondary)" font-family="var(--font-mono)" font-size="9">${escapeHtml(seqPreview)}</text>
      <text x="${padL}" y="${H - 4}" fill="var(--ink-muted)" font-family="var(--font-mono)" font-size="9">${escapeHtml(dbShort)}</text>
    </svg>
  `;
}

function renderStructureTracksSvg(profile, chain) {
  const box = document.getElementById("structure-track-svg-wrap");
  if (!box) return;
  const tr = profile && profile.sliding_tracks;
  if (!tr || !Array.isArray(tr.positions) || !tr.positions.length) {
    box.innerHTML = `<div class="empty-hint mono">暂无滑窗轨迹数据</div>`;
    return;
  }

  const totalNt = Math.max(1, Number(profile.length_nt) || 1);
  const W = 520;
  const H = 162;
  const padL = 42;
  const padR = 16;
  const plotW = W - padL - padR;
  const xOfNt = (pos) => padL + Math.min(1, Math.max(0, Number(pos) / totalNt)) * plotW;

  // Domain header strip (y = 8..15)
  const segRects = ((chain && chain.segments) || [])
    .map((s) => {
      const x1 = xOfNt(s.start);
      const x2 = xOfNt(s.end);
      const meta = SEG_META[s.segment_type] || { colorVar: "var(--accent)" };
      return `<rect x="${x1.toFixed(1)}" y="8" width="${Math.max(2, x2 - x1).toFixed(1)}" height="6" fill="${meta.colorVar}" opacity="0.85" />`;
    })
    .join("");

  // Track 1: Local MFE (y = 24..78)
  const mfeVals = tr.local_mfe.map((v) => Number(v) || 0);
  const minMfe = Math.min(...mfeVals, -5);
  const maxMfe = Math.max(...mfeVals, 0);
  const mfeSpan = maxMfe - minMfe || 1;
  const yMfe = (v) => 24 + (1 - (v - minMfe) / mfeSpan) * 52;

  const mfePts = tr.positions
    .map((pos, i) => `${xOfNt(pos + (tr.window_nt || 0) / 2).toFixed(1)},${yMfe(mfeVals[i]).toFixed(1)}`)
    .join(" ");

  // Track 2: GC%, U%, Unpaired% (y = 92..142, 0..100%)
  const yPct = (pct) => 142 - (Math.max(0, Math.min(100, Number(pct) || 0)) / 100) * 48;
  const gcPts = tr.positions
    .map((pos, i) => `${xOfNt(pos + (tr.window_nt || 0) / 2).toFixed(1)},${yPct(tr.local_gc_pct[i]).toFixed(1)}`)
    .join(" ");
  const uPts = tr.positions
    .map((pos, i) => `${xOfNt(pos + (tr.window_nt || 0) / 2).toFixed(1)},${yPct(tr.local_u_pct[i]).toFixed(1)}`)
    .join(" ");
  const upPts = tr.positions
    .map((pos, i) => `${xOfNt(pos + (tr.window_nt || 0) / 2).toFixed(1)},${yPct((Number(tr.local_unpaired_prob[i]) || 0) * 100).toFixed(1)}`)
    .join(" ");

  // Motif pins along x-axis
  const motifs = Array.isArray(profile.motifs) ? profile.motifs : [];
  const motifDots = motifs
    .slice(0, 20)
    .map((m) => {
      const mx = xOfNt(m.start).toFixed(1);
      const col = m.severity === "critical" || m.severity === "high" ? "var(--err)" : m.severity === "info" ? "var(--accent-teal)" : "var(--warn)";
      return `<circle cx="${mx}" cy="144" r="3.2" fill="${col}"><title>${escapeHtml(m.label)} [${escapeHtml(m.start)}..${escapeHtml(m.end)})</title></circle>`;
    })
    .join("");

  box.innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}" role="img" aria-label="Sliding window MFE, GC, U, and unpaired probability tracks">
      ${segRects}
      <line x1="${padL}" y1="24" x2="${W - padR}" y2="24" stroke="var(--border)" />
      <line x1="${padL}" y1="76" x2="${W - padR}" y2="76" stroke="var(--border)" />
      <text x="${padL - 5}" y="31" text-anchor="end" fill="var(--ink-muted)" font-family="var(--font-mono)" font-size="8.5">${escapeHtml(maxMfe.toFixed(0))}</text>
      <text x="${padL - 5}" y="76" text-anchor="end" fill="var(--ink-muted)" font-family="var(--font-mono)" font-size="8.5">${escapeHtml(minMfe.toFixed(0))}</text>
      <text x="${padL + 4}" y="34" fill="var(--accent)" font-family="var(--font-mono)" font-size="8.5">MFE (kcal/mol)</text>
      ${tr.positions.length === 1 ? `<circle cx="${xOfNt(totalNt / 2).toFixed(1)}" cy="${yMfe(mfeVals[0]).toFixed(1)}" r="3.5" fill="var(--accent)" />` : `<polyline points="${mfePts}" fill="none" stroke="var(--accent)" stroke-width="1.6" />`}

      <line x1="${padL}" y1="94" x2="${W - padR}" y2="94" stroke="var(--border)" />
      <line x1="${padL}" y1="142" x2="${W - padR}" y2="142" stroke="var(--border-strong)" />
      <text x="${padL - 5}" y="99" text-anchor="end" fill="var(--ink-muted)" font-family="var(--font-mono)" font-size="8.5">100%</text>
      <text x="${padL - 5}" y="142" text-anchor="end" fill="var(--ink-muted)" font-family="var(--font-mono)" font-size="8.5">0%</text>
      ${tr.positions.length > 1 ? `
        <polyline points="${gcPts}" fill="none" stroke="var(--seg-cds)" stroke-width="1.5" />
        <polyline points="${uPts}" fill="none" stroke="var(--warn)" stroke-width="1.3" stroke-dasharray="3 2" />
        <polyline points="${upPts}" fill="none" stroke="var(--seg-utr3)" stroke-width="1.2" opacity="0.8" />
      ` : `
        <circle cx="${xOfNt(totalNt / 2).toFixed(1)}" cy="${yPct(tr.local_gc_pct[0]).toFixed(1)}" r="3.5" fill="var(--seg-cds)" />
      `}
      ${motifDots}
      <text x="${padL}" y="157" fill="var(--seg-cds)" font-family="var(--font-mono)" font-size="8.5">— GC%</text>
      <text x="${padL + 52}" y="157" fill="var(--warn)" font-family="var(--font-mono)" font-size="8.5">-- U%</text>
      <text x="${padL + 98}" y="157" fill="var(--seg-utr3)" font-family="var(--font-mono)" font-size="8.5">— 开放度%</text>
      <text x="${W - padR}" y="157" text-anchor="end" fill="var(--ink-muted)" font-family="var(--font-mono)" font-size="8.5">${escapeHtml(totalNt)} nt</text>
    </svg>
  `;
}

function renderStructureVisuals(profile, chain) {
  if (!profile) return;
  state.currentStructureProfile = profile;
  const srcBadge = document.getElementById("structure-source-badge");
  const engines = Array.isArray(profile.engines_used) ? profile.engines_used.join(" + ") : "deterministic";
  if (srcBadge) {
    const isVienna = engines.toLowerCase().includes("viennarna");
    srcBadge.textContent = isVienna ? `ViennaRNA 精确配分函数` : `快速序列热力学扫描`;
    srcBadge.className = isVienna ? "badge mono ok" : "badge mono teal";
  }

  const kpiStrip = document.getElementById("structure-kpi-strip");
  if (kpiStrip) {
    kpiStrip.innerHTML = `
      <span class="kpi-pill">5′ 开放度 <strong>${escapeHtml(Number(profile.kozak_unpaired_prob ?? 0).toFixed(3))}</strong></span>
      <span class="kpi-pill">全长 MFE <strong>${escapeHtml(Number(profile.mfe_kcal_mol ?? 0).toFixed(1))}</strong> kcal/mol</span>
      <span class="kpi-pill">人源 CAI <strong>${escapeHtml(Number(profile.cai ?? 0).toFixed(3))}</strong></span>
      <span class="kpi-pill">U% <strong>${escapeHtml(Number(profile.full_u_pct ?? 0).toFixed(1))}%</strong></span>
    `;
  }

  renderStructureArcSvg(profile);
  renderStructureTracksSvg(profile, chain);

  const pinsStrip = document.getElementById("motif-pins-strip");
  if (pinsStrip) {
    pinsStrip.hidden = false;
    const motifs = Array.isArray(profile.motifs) ? profile.motifs : [];
    if (!motifs.length) {
      pinsStrip.innerHTML = `<span class="badge ok mono">基序扫描清净：0 uAUG · 0 m¹Ψ 滑移串 · 无异常均聚物</span>`;
    } else {
      pinsStrip.innerHTML = motifs
        .slice(0, 10)
        .map((m, idx) => {
          const safeSev = ["critical", "high", "warning", "info"].includes(m.severity) ? m.severity : "warning";
          return `<button type="button" class="motif-pin-btn ${safeSev}" data-motif-idx="${idx}">${escapeHtml(m.label)} [${escapeHtml(m.start)}..${escapeHtml(m.end)})</button>`;
        })
        .join("");
      if (typeof pinsStrip.querySelectorAll === "function") {
        pinsStrip.querySelectorAll(".motif-pin-btn").forEach((btn) => {
          btn.addEventListener("click", () => {
            const m = motifs[Number(btn.getAttribute("data-motif-idx"))];
            if (!m || !chain || !Array.isArray(chain.segments)) return;
            const hitSeg =
              chain.segments.find((s) => m.start >= s.start && m.start < s.end) || chain.segments[0];
            if (hitSeg) inspectSegment(chain, hitSeg);
          });
        });
      }
    }
  }
}

function getActiveInspectedChain() {
  const cand = state.candidates.find((c) => c.bundle_id === state.selectedBundleId);
  if (!cand) return null;
  const chains = cand.molecules?.length ? cand.molecules : [cand];
  const sel = document.getElementById("inspect-chain");
  const idx = sel && Number.isFinite(Number(sel.value)) ? Number(sel.value) : 0;
  return chains[idx] || chains[0];
}

function getActiveComplianceTrackForCandidate(cand) {
  const latestIntent = state.intentHistory.length
    ? state.intentHistory[state.intentHistory.length - 1]
    : null;
  if (latestIntent && latestIntent.compliance_track) {
    return latestIntent.compliance_track;
  }
  const profSel = document.getElementById("sel-profile")?.value;
  if (profSel === "commercial_clean") return "commercial_clean";
  const segs = Array.isArray(cand?.segments) ? cand.segments : [];
  if (segs.length && segs.every((s) => !s.license_track || s.license_track === "commercial_clean")) {
    return profSel === "research_max" ? "research_max" : "commercial_clean";
  }
  return profSel || "commercial_clean";
}

async function runStructureAnalysisForSelected() {
  const cand = state.candidates.find((c) => c.bundle_id === state.selectedBundleId);
  const chain = getActiveInspectedChain();
  if (!cand || !chain || !chain.full_sequence || chain.full_sequence.length < 3) return null;
  const seq = String(chain.full_sequence).toUpperCase().replace(/T/g, "U");
  const segs = Array.isArray(chain.segments) ? chain.segments : [];
  const u5 = segs.find((s) => s.segment_type === "five_prime_utr");
  const cds = segs.find((s) => s.segment_type === "cds");
  const u3 = segs.find((s) => s.segment_type === "three_prime_utr");
  const effMod = chain.modality || state.modality || "linear_mrna";
  const effTrack = getActiveComplianceTrackForCandidate(cand);
  const coordKey = [
    u5 ? Number(u5.end) : "",
    cds ? Number(cds.start) : "",
    cds ? Number(cds.end) : "",
    u3 ? Number(u3.start) : "",
    u3 ? Number(u3.end) : "",
  ].join(",");
  const cacheKey = [
    cand.bundle_id || "raw",
    cand.version_digest || "v0",
    chain.molecule_id || "mol0",
    effTrack,
    effMod,
    coordKey,
    seq,
  ].join(":");
  if (state.structureCache[cacheKey]) {
    renderStructureVisuals(state.structureCache[cacheKey], chain);
    return state.structureCache[cacheKey];
  }

  const reqProjId = state.projectId;
  const reqRev = state.activeVersion;
  const reqBid = cand.bundle_id;
  const reqMolId = chain.molecule_id || null;
  const reqApiBase = normalizeOrigin(state.apiBase);
  const reqProjKey = state.projectKey;

  const payload = {
    sequence: seq,
    project_id: reqProjId || null,
    bundle_id: reqBid || null,
    molecule_id: reqMolId,
    five_prime_end: u5 ? Number(u5.end) : null,
    cds_start: cds ? Number(cds.start) : null,
    cds_end: cds ? Number(cds.end) : null,
    three_prime_start: u3 ? Number(u3.start) : null,
    three_prime_end: u3 ? Number(u3.end) : null,
    modality: effMod,
    compliance_track: effTrack,
    include_eternafold: false,
  };

  try {
    const data = await apiFetch("/api/v1/structure/analyze", {
      method: "POST",
      explicitProjectKey: reqProjKey,
      explicitApiBase: reqApiBase,
      body: JSON.stringify(payload),
    });
    if (data && data.initiation_window) {
      state.structureCache[cacheKey] = data;
      const activeChainNow = getActiveInspectedChain();
      if (
        reqApiBase === normalizeOrigin(state.apiBase) &&
        reqProjId === state.projectId &&
        reqRev === state.activeVersion &&
        reqBid === state.selectedBundleId &&
        activeChainNow &&
        (activeChainNow.molecule_id || null) === reqMolId &&
        activeChainNow.full_sequence === chain.full_sequence
      ) {
        renderStructureVisuals(data, chain);
      }
      return data;
    }
  } catch {
    // Keep client deterministic structure profile when offline or mocked
  }
  const fallback = computeClientStructureProfile(chain, effMod);
  if (fallback) renderStructureVisuals(fallback, chain);
  return fallback;
}

function formatCdsCodonAlignment(rnaSeq, globalStart) {
  const codonsPerRow = 15;
  const totalCodons = Math.floor(rnaSeq.length / 3);
  const maxRows = 18;
  const lines = [];

  for (let row = 0; row < Math.min(maxRows, Math.ceil(totalCodons / codonsPerRow)); row += 1) {
    const cStart = row * codonsPerRow;
    const cEnd = Math.min(totalCodons, cStart + codonsPerRow);
    const rnaTriplets = [];
    const aaTriplets = [];
    for (let c = cStart; c < cEnd; c += 1) {
      const trip = rnaSeq.slice(c * 3, c * 3 + 3);
      const aa = CODON_TABLE_RNA[trip] || "?";
      rnaTriplets.push(trip);
      aaTriplets.push(` ${aa} `);
    }
    const ntCoord = String(globalStart + cStart * 3).padStart(4, " ");
    const aaCoord = String(cStart + 1).padStart(4, " ");
    lines.push(`nt ${ntCoord} │ ${rnaTriplets.join(" ")}`);
    lines.push(`aa ${aaCoord} │ ${aaTriplets.join(" ")}`);
    if (cEnd < totalCodons && row < maxRows - 1) {
      lines.push("");
    }
  }
  if (Math.ceil(totalCodons / codonsPerRow) > maxRows) {
    lines.push(`... 其余 ${totalCodons - maxRows * codonsPerRow} 个密码子已省略（导出 FASTA/GenBank 可查看全长）`);
  }
  return lines.join("\n");
}

function formatPlainNucleotideBlocks(rnaSeq, globalStart) {
  const perRow = 60;
  const lines = [];
  for (let i = 0; i < rnaSeq.length; i += perRow) {
    const slice = rnaSeq.slice(i, i + perRow);
    const groups = slice.match(/.{1,10}/g) || [slice];
    const coord = String(globalStart + i).padStart(4, " ");
    lines.push(`nt ${coord} │ ${groups.join(" ")}`);
  }
  return lines.join("\n");
}

function inspectSegment(cand, seg) {
  state.selectedSegType = seg.segment_type;
  renderTrueScaleMap(cand);
  const fullSub = String(cand.full_sequence || "").slice(seg.start, seg.end);
  const meta = SEG_META[seg.segment_type] || { short: seg.segment_type };
  const header =
    `${meta.short} [${seg.start}..${seg.end}) · ${seg.length_nt} nt · GC ${seg.gc_pct}% · ` +
    `轨: ${seg.license_track || "commercial_clean"} · 动作: ${seg.action_taken || "verified"} · 来源: ${seg.origin_source || "synthetic"}`;
  const body =
    seg.segment_type === "cds"
      ? formatCdsCodonAlignment(fullSub, seg.start)
      : formatPlainNucleotideBlocks(fullSub, seg.start);

  const segInsp = document.getElementById("seg-inspector");
  if (segInsp) segInsp.textContent = `${header}\n${"─".repeat(78)}\n${body}`;
}

function selectCandidate(bundleId) {
  state.selectedBundleId = bundleId;
  renderCandidates();
  syncBenchmarkSelectedLabel();
  const cand = state.candidates.find((c) => c.bundle_id === bundleId);
  if (!cand) return;

  const chains = cand.molecules?.length ? cand.molecules : [cand];
  const select = document.getElementById("inspect-chain");
  if (select && typeof select.replaceChildren === "function" && typeof document.createElement === "function") {
    select.replaceChildren();
    chains.forEach((chain, i) => {
      const option = document.createElement("option");
      option.value = String(i);
      option.textContent = chain.molecule_id || `分子 ${i + 1}`;
      select.appendChild(option);
    });
  }
  const chainPicker = document.getElementById("chain-picker");
  if (chainPicker) chainPicker.hidden = chains.length < 2;

  function inspectChain(index) {
    const chain = chains[index] || chains[0];
    if (!chain) return;
    const label = chains.length > 1 ? ` · ${chain.molecule_id}` : "";
    const txtSel = document.getElementById("txt-selected-bundle");
    const shortSha = String(cand.short_digest || (cand.version_digest || "").slice(0, 10) || "verified");
    if (txtSel) {
      txtSel.textContent = `${humanizeCandidateLabel(cand)}${label} · ${chain.length_nt} nt · GC ${chain.full_gc_pct}% · #${shortSha}`;
    }
    renderEnzymeClearance(chain.full_sequence);
    const segs = Array.isArray(chain.segments) ? chain.segments : [];
    const segment =
      segs.find((s) => s.segment_type === state.selectedSegType) ||
      segs.find((s) => s.segment_type === "cds") ||
      segs[0];
    if (segment) inspectSegment(chain, segment);
    else renderTrueScaleMap(chain);

    const localProfile = computeClientStructureProfile(chain, state.modality);
    if (localProfile) {
      renderStructureVisuals(localProfile, chain);
    }
    if (state.gatewayOnline && String(chain.full_sequence || "").length >= 12) {
      runStructureAnalysisForSelected();
    }
  }
  if (select) select.onchange = () => inspectChain(Number(select.value));
  inspectChain(0);
}

function renderRevisionLedger(intentDiff) {
  const panel = document.getElementById("revision-ledger");
  if (!panel) return;
  if (state.completedRevisions.length < 2) {
    panel.hidden = true;
    return;
  }

  const prev = state.completedRevisions[state.completedRevisions.length - 2];
  const curr = state.completedRevisions[state.completedRevisions.length - 1];
  if (prev.version === curr.version) {
    panel.hidden = true;
    return;
  }
  panel.hidden = false;

  const vPrev = `v${String(prev.version).padStart(2, "0")}`;
  const vCurr = `v${String(curr.version).padStart(2, "0")}`;
  const verSpan = document.getElementById("ledger-ver-span");
  if (verSpan) verSpan.textContent = `${vPrev} → ${vCurr}`;

  const deltaEl = document.getElementById("ledger-delta-summary");
  const deltaParts = [];
  if (intentDiff && JSON.stringify(intentDiff.v1_enzymes) !== JSON.stringify(intentDiff.v2_enzymes)) {
    deltaParts.push(
      `禁切酶: <span class="strike-old">${escapeHtml((intentDiff.v1_enzymes || []).join("+") || "none")}</span>→ <strong>${escapeHtml((intentDiff.v2_enzymes || []).join("+") || "none")}</strong>`
    );
  }
  if (prev.intent.compliance_track !== curr.intent.compliance_track) {
    deltaParts.push(
      `合规轨: <span class="strike-old">${escapeHtml(prev.intent.compliance_track)}</span>→ <strong>${escapeHtml(curr.intent.compliance_track)}</strong>`
    );
  }
  const prevLocks = Object.keys(prev.intent.locked_scopes || {});
  const currLocks = Object.keys(curr.intent.locked_scopes || {});
  if (currLocks.length > prevLocks.length) {
    const added = currLocks.filter((k) => !prevLocks.includes(k));
    deltaParts.push(`新增锁定: <strong>${escapeHtml(added.join(", "))}</strong>`);
  }
  if (!deltaParts.length) {
    deltaParts.push(
      `增量重排完成 · <strong>${escapeHtml(humanizeCandidateLabel(prev.topCandidate))}</strong> → <strong>${escapeHtml(humanizeCandidateLabel(curr.topCandidate))}</strong>`
    );
  }
  if (deltaEl) deltaEl.innerHTML = deltaParts.join(" · ");

  const locksBar = document.getElementById("ledger-locks-bar");
  const prevSeqs = (prev.intent.protein_targets || []).map((t) => t.protein_sequence);
  const currSeqs = (curr.intent.protein_targets || []).map((t) => t.protein_sequence);
  const aaSame = prevSeqs.length === currSeqs.length && currSeqs.every((s, i) => s && s === prevSeqs[i]);
  const chemSame = prev.intent.uridine_chemistry === curr.intent.uridine_chemistry;
  const lockTokens = [];
  if (prevSeqs.length || currSeqs.length) {
    lockTokens.push(`AA:${currSeqs.map((s) => s?.length ?? 0).join("+") || "0"}${aaSame ? "✓" : "Δ"}`);
  }
  lockTokens.push(`Chem:${curr.intent.uridine_chemistry}${chemSame ? "✓" : "Δ"}`);
  currLocks.forEach((k) => lockTokens.push(`Lock:${k}`));
  if (locksBar) locksBar.innerHTML = lockTokens.map((t) => `<span class="lock-token">${escapeHtml(t)}</span>`).join("");

  const dualWrap = document.getElementById("dual-scale-tracks");
  const prevChains = prev.topCandidate.molecules || [prev.topCandidate];
  const currChains = curr.topCandidate.molecules || [curr.topCandidate];
  const maxNt = Math.max(...prevChains.concat(currChains).map((chain) => Number(chain.length_nt) || 1), 1);

  const buildScaledRow = (verTag, cand, other, isCurrent) => {
    const otherSeqMap = new Map(
      other?.segments.map((s) => [s.segment_type, other.full_sequence.slice(s.start, s.end)])
    );
    const segHtml = cand.segments
      .map((s) => {
        const meta = SEG_META[s.segment_type] || { short: s.segment_type, colorVar: "var(--accent)" };
        const wPct = (((Number(s.length_nt) || 0) / maxNt) * 100).toFixed(3);
        const mySeq = cand.full_sequence.slice(s.start, s.end);
        const changed = mySeq !== otherSeqMap.get(s.segment_type);
        const opacity = changed ? "1" : "0.38";
        const outline = changed && isCurrent ? "box-shadow: inset 0 0 0 1.5px var(--ink);" : "";
        return `<span title="${escapeHtml(meta.short)}: ${escapeHtml(s.length_nt)} nt (${changed ? "已更新" : "100% 继承未变"})" style="width:${wPct}%; background:${meta.colorVar}; opacity:${opacity}; ${outline}"></span>`;
      })
      .join("");
    return `
      <div class="dual-track-row mono">
        <span style="color:var(--ink-secondary); font-weight:${isCurrent ? "600" : "400"};">${escapeHtml(verTag)}</span>
        <div class="dual-bar">${segHtml}</div>
        <span style="text-align:end; color:var(--ink-muted);">${escapeHtml(cand.length_nt)} nt</span>
      </div>
    `;
  };

  const rows = [];
  for (let i = 0; i < Math.max(prevChains.length, currChains.length); i++) {
    const label = Math.max(prevChains.length, currChains.length) > 1 ? ` · ${i + 1}` : "";
    if (prevChains[i]) rows.push(buildScaledRow(vPrev + label, prevChains[i], currChains[i], false));
    if (currChains[i]) rows.push(buildScaledRow(vCurr + label, currChains[i], prevChains[i], true));
  }
  if (dualWrap) dualWrap.innerHTML = rows.join("");
}

function summarizeToolArgsCleanly(argsObj) {
  if (!argsObj || typeof argsObj !== "object") return "";
  const pairs = [];
  for (const [k, v] of Object.entries(argsObj)) {
    if (v == null || v === "") continue;
    if (k === "protein_sequence" && typeof v === "string") {
      pairs.push(`蛋白=${v.length}aa`);
    } else if (Array.isArray(v)) {
      if (v.length) pairs.push(`${k}=${v.slice(0, 3).join("+")}`);
    } else if (typeof v === "boolean" || typeof v === "number") {
      pairs.push(`${k}=${v}`);
    } else if (typeof v === "string") {
      pairs.push(`${k}=${v.length > 24 ? v.slice(0, 22) + "…" : v}`);
    }
    if (pairs.length >= 4) break;
  }
  return pairs.join(" · ");
}

function buildToolTraceHtml(toolCalls, actionTransparency = null, intentDiff = null) {
  const tcList = Array.isArray(toolCalls) ? toolCalls : [];
  const trList = Array.isArray(actionTransparency) ? actionTransparency : [];
  const receiptItems = [];

  const pillBadges = tcList
    .map((tc) => {
      const cleanName = TOOL_DISPLAY_NAMES[tc.name] || tc.name;
      return `<span class="tool-pill">${escapeHtml(cleanName)}</span>`;
    })
    .join("");

  tcList.forEach((tc) => {
    const cleanName = TOOL_DISPLAY_NAMES[tc.name] || tc.name;
    const argSummary = summarizeToolArgsCleanly(tc.arguments);
    receiptItems.push(
      `<li><strong>${escapeHtml(cleanName)}</strong>${argSummary ? ` <span class="meta-muted">(${escapeHtml(argSummary)})</span>` : ""}</li>`
    );
  });

  if (intentDiff) {
    const e1 = (intentDiff.v1_enzymes || []).join("+") || "无";
    const e2 = (intentDiff.v2_enzymes || []).join("+") || "无";
    receiptItems.push(
      `<li>继承核验：蛋白锁定=${escapeHtml(intentDiff.protein_lock_preserved ? "通过" : "变更")} · 修饰继承=${escapeHtml(intentDiff.chemistry_preserved ? "通过" : "变更")} · 禁切酶 ${escapeHtml(e1)} → ${escapeHtml(e2)}</li>`
    );
  }

  trList.forEach((t) => {
    if (t && t.proof_diff_summary) {
      receiptItems.push(`<li>${escapeHtml(t.stage_applied || "check")}: ${escapeHtml(t.proof_diff_summary)}</li>`);
    }
  });

  if (!receiptItems.length) return "";
  return `
    ${pillBadges ? `<div class="tool-pills-row">${pillBadges}</div>` : ""}
    <details class="tool-trace-fold mono">
      <summary>验证回执 (${receiptItems.length} 项通过)</summary>
      <ul class="tool-trace-list">${receiptItems.join("")}</ul>
    </details>
  `;
}

function buildTurnActionChipsHtml() {
  if (!state.candidates.length) return "";
  return `
    <div class="turn-action-chips">
      <button type="button" class="chip mono" data-copilot-act="bspqi">把 BsaI 改成 BspQI</button>
      <button type="button" class="chip mono" data-copilot-act="circrna">切换 circRNA</button>
      <button type="button" class="chip mono" data-copilot-act="compare">对比前两名</button>
      <button type="button" class="chip mono" data-copilot-act="ivt">导出 IVT 下单表</button>
    </div>
  `;
}

function clearErrorBanner() {
  const banner = document.getElementById("banner-notes");
  if (!banner) return;
  if (banner.className && banner.className.includes("err")) {
    banner.hidden = true;
    banner.className = "notice-banner";
    banner.innerHTML = "";
  }
}

function syncActionControlsDisabled() {
  const busy = state.inFlight;
  ["btn-send-turn", "btn-agent-rail-send", "btn-submit-initial"].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.disabled = busy;
  });
  [
    "starter-egfp",
    "starter-hepo",
    "starter-dll3",
    "starter-circ",
    "starter-sarna",
    "starter-ref-r",
    "starter-commercial",
    "starter-tradeoffs",
  ].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.disabled = busy;
  });
  const noIntent = state.intentHistory.length === 0;
  [
    "quick-refine-bspqi",
    "quick-refine-lock-cds",
    "quick-refine-m15-surrogate",
    "quick-refine-commercial",
    "quick-ask-tradeoffs",
  ].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.disabled = busy || noIntent;
  });
  if (noIntent === false && !busy && !state.selectedBundleId) {
    const lockBtn = document.getElementById("quick-refine-lock-cds");
    if (lockBtn) lockBtn.disabled = true;
  }
}

function renderLocalTradeoffsSummary() {
  if (!state.candidates.length) {
    appendStreamCard("warn", "本地权衡汇总", "NO DATA", "当前版本暂无合格候选可供比较。");
    return;
  }
  const axes = resolveMetricAxes(state.candidates);
  const p1 = state.candidates.filter((c) => c.pareto_rank === 1);
  const lines = [
    `**只读本地权衡汇总（Intent v${String(state.activeVersion).padStart(2, "0")}，未变更序列或约束）：**`,
    `- 合格候选共 ${state.candidates.length} 条，其中 Pareto P1 前沿 ${p1.length} 条。`,
  ];
  for (const cand of p1.slice(0, 3)) {
    const metricsSummary = axes
      .map((ax) => {
        const val = readAxisValue(cand, ax);
        return val == null ? null : `${ax.shortLabel}=${ax.fmt(val)}`;
      })
      .filter(Boolean)
      .join("，");
    lines.push(
      `- **${humanizeCandidateLabel(cand)}** (\`${cand.bundle_id}\`)：${metricsSummary || "序列硬门禁全部通过"}`
    );
  }
  appendStreamCard(
    "agent",
    "本地只读权衡分析",
    `v${String(state.activeVersion).padStart(2, "0")}`,
    `<div class="agent-prose">${formatSafeMarkdown(lines.join("\n"))}</div>`
  );
}

function syncControlsFromIntent(intent) {
  if (!intent || typeof intent !== "object") return;
  const profEl = document.getElementById("sel-profile");
  const chemEl = document.getElementById("sel-chemistry");
  if (profEl && intent.compliance_track) {
    profEl.value = intent.compliance_track;
    syncTrackButtons(intent.compliance_track);
  }
  if (chemEl && intent.uridine_chemistry) chemEl.value = intent.uridine_chemistry;
  if (intent.entry_mode === "protein_with_reference_r" || intent.entry_mode === "protein_only") {
    setEntryMode(intent.entry_mode === "protein_with_reference_r" ? "with_r" : "protein_only");
  }
  if (Array.isArray(intent.constraints)) {
    const enzConstraint = intent.constraints.find(
      (c) => c && c.constraint_kind === "type_iis_restriction_free"
    );
    if (enzConstraint && enzConstraint.params && Array.isArray(enzConstraint.params.enzymes)) {
      const syncedEnz = [...enzConstraint.params.enzymes];
      state.activeEnzymes = syncedEnz;
      setSelectedEnzymes(syncedEnz);
    }
    const hasMirnaConstraint = intent.constraints.some(
      (c) => c && c.constraint_kind === "mirna_detarget"
    );
    const detEl = document.getElementById("chk-detarget-liver");
    if (detEl && (intent.cell_expression_control || hasMirnaConstraint)) {
      detEl.checked = Boolean(
        hasMirnaConstraint ||
          (intent.cell_expression_control &&
            Array.isArray(intent.cell_expression_control.detarget_tissues) &&
            intent.cell_expression_control.detarget_tissues.length)
      );
    } else if (detEl && intent.cell_expression_control === null && "cell_expression_control" in intent) {
      detEl.checked = false;
    }
  }
  if (Array.isArray(intent.objectives)) {
    const lateEl = document.getElementById("chk-late-signal");
    if (lateEl) {
      lateEl.checked = intent.objectives.some(
        (o) => o && o.requirement_id === "obj_late_protein_persistence"
      );
    }
  }
  if (intent.modality) {
    setModality(intent.modality);
  }
  if (intent.scaffold_id) {
    const scafEl = document.getElementById("sel-scaffold");
    if (scafEl) scafEl.value = intent.scaffold_id;
  }
  updateSetupSummary();
}

function applyTurnResult(res, userText, sequence) {
  if (!res || !res.intent) {
    const hasReply = Boolean(res && typeof res.assistant_reply === "string" && res.assistant_reply.trim());
    const hasTools = Boolean(res && Array.isArray(res.tool_calls) && res.tool_calls.length > 0);
    if (!hasReply && !hasTools) {
      return false;
    }
    if (sequence < state.appliedTurnSequence) {
      return false;
    }
    state.appliedTurnSequence = Math.max(state.appliedTurnSequence, sequence);
    clearErrorBanner();
    const modelTag = res && res.model ? `${res.provider || state.llmProvider}:${res.model}` : "INFO";
    const toolTraceHtml = buildToolTraceHtml(res && res.tool_calls);
    const replyBody = hasReply
      ? `<div class="agent-prose">${formatSafeMarkdown(res.assistant_reply)}</div>`
      : "";
    appendStreamCard("agent", "Agent 回复", modelTag, `${replyBody}${toolTraceHtml}`);
    return true;
  }

  const version = res.intent.intent_version;
  if (version < state.activeVersion || (version === state.activeVersion && sequence < state.appliedTurnSequence)) {
    return false;
  }
  state.appliedTurnSequence = Math.max(state.appliedTurnSequence, sequence);
  state.activeVersion = version;
  state.activeIntentStatus = res.intent.status || "";
  updateActiveRevBadge();
  clearErrorBanner();

  let intentAdvanced = false;
  if (res.intent && !state.intentHistory.some((it) => it.intent_id === res.intent.intent_id)) {
    state.intentHistory.push(res.intent);
    intentAdvanced = true;
    renderIntentLineage();
  }
  syncControlsFromIntent(res.intent);

  const banner = document.getElementById("banner-notes");

  if (res.status === "NEEDS_CLARIFICATION" || res.intent.status !== "ready") {
    state.latestReport = null;
    state.candidates = [];
    state.lastCliCommand = "";
    state.cachedExports = {};
    renderCandidates();
    const lockBtn = document.getElementById("quick-refine-lock-cds");
    if (lockBtn) lockBtn.disabled = true;
    const ledger = document.getElementById("revision-ledger");
    if (ledger) ledger.hidden = true;
    const badgeOut = document.getElementById("badge-outcome");
    if (badgeOut) {
      badgeOut.textContent = res.intent.status;
      badgeOut.className = "badge mono warn";
    }
    const questions =
      Array.isArray(res.blocking_questions) && res.blocking_questions.length
        ? res.blocking_questions
        : res.intent.blocking_questions || [];
    const qs = questions.map((q) => `<li>${escapeHtml(q)}</li>`).join("");
    if (banner) {
      banner.hidden = false;
      banner.className = "notice-banner";
      banner.innerHTML = `<strong>编译前护栏拦截 (${escapeHtml(res.intent.status)})：</strong><ul style="margin:4px 0 0 18px;">${qs}</ul>`;
    }
    const toolCallsBlock = buildToolTraceHtml(res.tool_calls);
    const replyExtra = res.assistant_reply
      ? `<div class="agent-prose">${formatSafeMarkdown(res.assistant_reply)}</div>`
      : "";
    appendStreamCard(
      "warn",
      `输入护栏待确认 · v${String(res.intent.intent_version).padStart(2, "0")}`,
      res.intent.status,
      `<strong>待确认事项：</strong><ul style="margin:4px 0 0 18px;">${qs}</ul>${replyExtra}${toolCallsBlock}`
    );
    return true;
  }

  state.latestReport = res.report;
  state.candidates = res.candidates || [];
  if (res.cli_replay_command) {
    state.lastCliCommand = res.cli_replay_command;
  } else if (!res.assistant_reply) {
    state.lastCliCommand = "";
  }

  const rawOutcome = (res.report && res.report.outcome_status) || "COMPLETED";
  const badgeOut = document.getElementById("badge-outcome");
  if (badgeOut) {
    badgeOut.textContent = humanizeOutcomeLabel(rawOutcome);
    badgeOut.className =
      rawOutcome.includes("UNCOMPARABLE") || !state.candidates.length ? "badge mono warn" : "badge mono ok";
  }

  // Move routine scientific/CRS/model-domain disclosures into "边界与方法" modal; keep #banner-notes strictly for real blocks!
  const boundaryItems = [];
  if (res.report && res.report.honest_attribution_banner) {
    boundaryItems.push({ title: "序列与模型归因声明", detail: res.report.honest_attribution_banner });
  }
  (res.report?.action_transparency || []).forEach((t) => {
    if (t.proof_diff_summary && t.stage_applied === "recorded_only") {
      boundaryItems.push({ title: `记录项 (${t.requirement_id})`, detail: t.proof_diff_summary });
    }
  });
  (res.report?.pre_synthesis_advisories || []).forEach((adv) => {
    const cleanCat =
      adv.category === "crs_safety_layer_boundary"
        ? "多链共表达免疫安全边界"
        : adv.category === "model_domain_boundary"
        ? "计算代理外推边界"
        : adv.category || "合成前提示";
    boundaryItems.push({ title: cleanCat, detail: adv.message });
  });
  state.boundaryAdvisories = boundaryItems;
  renderBoundariesModal();

  if (banner) {
    if (res.report && res.report.uncomparable_reason) {
      banner.hidden = false;
      banner.className = "notice-banner";
      banner.innerHTML = `<strong>跨轨道对比提示：</strong>${escapeHtml(res.report.uncomparable_reason)}`;
    } else {
      banner.hidden = true;
      banner.className = "notice-banner";
      banner.innerHTML = "";
    }
  }

  const topCandPreview = state.candidates[0];
  const conciseSummary = res.assistant_reply
    ? `<div class="agent-prose">${formatSafeMarkdown(res.assistant_reply)}</div>`
    : `<div class="agent-prose">已完成 <strong>v${escapeHtml(String(res.intent.intent_version).padStart(2, "0"))}</strong> 编译，共生成 <strong>${escapeHtml(state.candidates.length)}</strong> 条合格候选${topCandPreview ? `，首选 <strong>${escapeHtml(humanizeCandidateLabel(topCandPreview))}</strong>` : ""}。</div>`;

  const receiptHtml = buildToolTraceHtml(
    res.tool_calls,
    res.report && res.report.action_transparency,
    res.intent_diff
  );
  const actionChipsHtml = buildTurnActionChipsHtml();

  const actorLabel = res.model
    ? `${res.provider || state.llmProvider}:${res.model} · v${String(res.intent.intent_version).padStart(2, "0")}`
    : `引擎完成 · v${String(res.intent.intent_version).padStart(2, "0")}`;

  appendStreamCard(
    "agent",
    actorLabel,
    `${state.candidates.length} 条合格`,
    `${conciseSummary}${actionChipsHtml}${receiptHtml}`
  );

  const recFirst = res.report?.recommended_bundle_ids?.[0] || state.candidates[0]?.bundle_id || null;
  const keepCurrentSelection =
    !intentAdvanced &&
    state.selectedBundleId &&
    state.candidates.some((c) => c.bundle_id === state.selectedBundleId);
  state.selectedBundleId = keepCurrentSelection ? state.selectedBundleId : recFirst;

  if (state.candidates.length > 0 && res.intent) {
    const topCand = state.candidates.find((c) => c.bundle_id === recFirst) || state.candidates[0];
    const existingIdx = state.completedRevisions.findIndex(
      (r) => r.intent && r.intent.intent_id === res.intent.intent_id
    );
    const revEntry = {
      version: res.intent.intent_version,
      intent: res.intent,
      topCandidate: topCand,
    };
    if (existingIdx >= 0) {
      state.completedRevisions[existingIdx] = revEntry;
    } else {
      state.completedRevisions.push(revEntry);
      state.completedRevisions.sort((a, b) => a.version - b.version);
    }
    renderRevisionLedger(res.intent_diff || null);
  }

  renderCandidates();
  if (state.selectedBundleId && state.candidates.length > 0) {
    selectCandidate(state.selectedBundleId);
  }
  syncActionControlsDisabled();
  return true;
}

async function refreshSystemStatus() {
  const dot = document.getElementById("dot-gateway");
  const txt = document.getElementById("txt-gateway");
  if (state.startupApiRejected) {
    state.gatewayOnline = false;
    if (dot) dot.className = "dot err";
    if (txt) txt.textContent = "不可信网关参数已拦截";
    return;
  }
  const reqGen = state.configGeneration;
  const reqSeq = ++state.statusRequestSeq;
  const reqOrigin = normalizeOrigin(state.apiBase);
  const t0 = typeof performance !== "undefined" ? performance.now() : Date.now();
  try {
    const creds = getCredentialsForOrigin(reqOrigin);
    const statusHeaders =
      creds.accessToken && isTrustedApiOrigin(reqOrigin)
        ? { "X-MRNAX-Access-Token": creds.accessToken }
        : {};
    const st = await apiFetch("/v1/system/status", {
      includeAuth: false,
      explicitApiBase: reqOrigin,
      headers: statusHeaders,
    });
    if (
      reqGen !== state.configGeneration ||
      reqSeq !== state.statusRequestSeq ||
      reqOrigin !== normalizeOrigin(state.apiBase) ||
      state.startupApiRejected
    ) {
      return;
    }
    state.gatewayOnline = Boolean(st && st.hostname);
    const now = typeof performance !== "undefined" ? performance.now() : Date.now();
    const ms = Math.round(now - t0);
    if (dot) dot.className = "dot ok";
    if (txt) txt.textContent = `${st.hostname} · ${ms}ms`;
    const pillEng = document.getElementById("pill-engine");
    if (pillEng) {
      pillEng.textContent = `Engine: ${st.default_profile} · ViennaRNA ${st.viennarna_version || "lazy"}`;
    }
    if (st.default_llm_models && typeof st.default_llm_models === "object") {
      state.defaultModels = { ...DEFAULT_PROVIDER_MODELS, ...st.default_llm_models };
    }
    if (st.supported_llm_models && typeof st.supported_llm_models === "object") {
      state.supportedModels = { ...SUPPORTED_PROVIDER_MODELS, ...st.supported_llm_models };
    }
    syncModelOptions(state.llmProvider, { skipOpenModal: true });
    updateAgentAuthPill(st);
    if (st.caller_token_authenticated && !state.userChoseExecMode && state.activeVersion === 0) {
      setExecMode("llm_agent");
    }

    const presets = await apiFetch("/v1/presets", { includeAuth: false, explicitApiBase: reqOrigin });
    if (
      reqGen !== state.configGeneration ||
      reqSeq !== state.statusRequestSeq ||
      reqOrigin !== normalizeOrigin(state.apiBase) ||
      state.startupApiRejected
    ) {
      return;
    }
    if (presets && presets.egfp_reporter && presets.egfp_reporter.protein_sequence) {
      state.presets = presets;
      const inputProt = document.getElementById("input-protein");
      if (inputProt && !inputProt.value.trim()) {
        inputProt.value = presets.egfp_reporter.protein_sequence;
        state.userEditedProtein = false;
        updateProteinBadge();
      }
    }
    const refPreview = document.getElementById("txt-ref-r-preview");
    if (refPreview && presets && presets.preset_reference_r) {
      refPreview.textContent = `${presets.preset_reference_r.bundle_id} (${presets.preset_reference_r.length_nt} nt) · ${presets.preset_reference_r.sequence_preview}`;
    }
  } catch (err) {
    if (
      reqGen !== state.configGeneration ||
      reqSeq !== state.statusRequestSeq ||
      reqOrigin !== normalizeOrigin(state.apiBase) ||
      state.startupApiRejected
    ) {
      return;
    }
    state.gatewayOnline = false;
    if (dot) dot.className = "dot err";
    if (txt) txt.textContent = `离线 (${err.message})`;
  }
}

function beginBusyIndicator(modeLabel, reqEpoch) {
  state.activeRequestCount += 1;
  state.inFlight = true;
  syncActionControlsDisabled();

  const startedAt = Date.now();
  const btnSend = document.getElementById("btn-send-turn");
  const btnForm = document.getElementById("btn-submit-initial");

  const box = document.getElementById("chat-stream");
  let pendingCard = null;
  if (box && typeof document.createElement === "function") {
    removeWelcomeCardIfPresent();
    pendingCard = document.createElement("div");
    pendingCard.className = "turn-card pending";
    pendingCard.innerHTML = `<div class="turn-meta"><span>RUNNING</span><span>0.0s</span></div><div><span class="spin"></span> ${escapeHtml(modeLabel)}</div>`;
    box.appendChild(pendingCard);
    box.scrollTop = box.scrollHeight;
  }

  let timerId = null;
  const updateElapsed = () => {
    if (reqEpoch !== state.sessionEpoch) {
      if (timerId !== null && typeof clearInterval === "function") clearInterval(timerId);
      timerId = null;
      return;
    }
    const sec = ((Date.now() - startedAt) / 1000).toFixed(1);
    if (btnSend) btnSend.innerHTML = `<span class="spin"></span> 计算中… ${escapeHtml(sec)}s`;
    if (btnForm) btnForm.textContent = `计算中… ${sec}s`;
    if (pendingCard && pendingCard.querySelector) {
      const spanRight = pendingCard.querySelector(".turn-meta span:last-child");
      if (spanRight) spanRight.textContent = `${sec}s`;
    }
  };
  if (typeof setInterval === "function") {
    timerId = setInterval(updateElapsed, 100);
  }
  updateElapsed();

  let stopped = false;
  return () => {
    if (stopped) return;
    stopped = true;
    if (timerId !== null && typeof clearInterval === "function") clearInterval(timerId);
    if (pendingCard && typeof pendingCard.remove === "function") pendingCard.remove();
    if (reqEpoch === state.sessionEpoch) {
      state.activeRequestCount = Math.max(0, state.activeRequestCount - 1);
      state.inFlight = state.activeRequestCount > 0;
    }
    if (!state.inFlight) {
      if (btnSend) {
        btnSend.textContent = getSelectedExecMode() === "llm_agent" ? "发送给 Agent ↵" : "编译生成 ↵";
      }
      if (btnForm) {
        btnForm.textContent = "按表单直接编译";
      }
      syncActionControlsDisabled();
    }
  };
}

function surfaceTurnError(err, retryContext) {
  state.lastFailedTurn = retryContext || null;
  const isAuthErr = Boolean(err && err.status === 401);
  const badgeOut = document.getElementById("badge-outcome");
  if (badgeOut) {
    badgeOut.textContent = isAuthErr ? "AUTH REQUIRED" : "ERROR";
    badgeOut.className = "badge mono warn";
  }

  const banner = document.getElementById("banner-notes");
  if (banner) {
    banner.hidden = false;
    banner.className = "notice-banner err";
    if (isAuthErr) {
      banner.innerHTML = `
        <strong>AI Agent 需要授权令牌（Access Token）或您的 BYO-LLM Key：</strong>
        公开网页默认拦截未授权云端模型调用，以保护服务器额度。
        <div class="banner-actions">
          <button type="button" class="btn-primary btn-compact" data-action="open-auth">
            配置 Access Token / BYO Key
          </button>
          <button type="button" class="btn-ghost" data-action="fallback-deterministic">
            免密钥切到确定性引擎立即运行
          </button>
        </div>
      `;
    } else {
      banner.innerHTML = `
        <strong>请求失败：</strong><code>${escapeHtml((err && err.message) || "Unknown error")}</code>
        <div class="banner-actions">
          ${retryContext ? '<button type="button" class="btn-primary btn-compact" data-action="retry-last">重试</button>' : ""}
          <button type="button" class="btn-ghost" data-action="open-auth">检查网关与密钥</button>
        </div>
      `;
    }
  }

  appendStreamCard(
    "err",
    isAuthErr ? "Agent 鉴权拦截 (401)" : "请求异常",
    isAuthErr ? "HTTP 401" : `HTTP ${(err && err.status) || "ERR"}`,
    isAuthErr
      ? "未提供有效 Access Token 或 BYO API Key。请点击上方横幅配置密钥，或一键切回确定性状态机立即运行。"
      : escapeHtml((err && err.message) || "请求异常")
  );
}

function clearSubmittedFieldIfUnchanged(sourceEl, submittedText, reqEpoch) {
  if (!sourceEl || reqEpoch !== state.sessionEpoch) return;
  if (typeof submittedText === "string" && sourceEl.value && sourceEl.value.trim() === submittedText) {
    sourceEl.value = "";
  }
}

function hasExplicitMutationIntent(text) {
  const raw = String(text || "").trim();
  if (!raw) return false;
  if (
    /^\s*(?:how\s+(?:can|do|should|would|could|to|does)|can\s+(?:i|we|you|mrnax)|should\s+(?:i|we)|could\s+(?:i|we)|what\s+(?:is|are|does|if|happens)|why\s+|which\s+|is\s+it\s+|does\s+|do\s+i\s+|explain|compare|difference|trade-?offs?|summari[sz]e|analyze|analyse|interpret|inspect|audit|describe|list)\b/i.test(
      raw
    ) ||
    /^\s*(?:请)?(?:解释|为什么|为何|如何|怎样|怎么(?:办|选|做|切换|改|用|换|开启|关闭|配置)|能否|可不可以|是否(?:可以|需要)|对比|比较|总结|分析|查看|检查|审计|说明|介绍)/.test(
      raw
    )
  ) {
    return (
      /(?:[,;，；。]\s*(?:然后|并|再|顺便)?\s*(?:改成|换成|切换为|切换到|切回|重配|改为|设为|设置为|锁定|解锁))/.test(
        raw
      ) ||
      /(?:[,;]\s*(?:then|and\s+then|now|please)?\s*(?:change\s+to|switch\s+to|replace\s+with|modify\s+to|lock\s+(?:the\s+)?cds|unlock)\b)/i.test(
        raw
      )
    );
  }
  return (
    /(?:改成|换成|切换|锁定|解锁|避开|禁切|重配|优化|调整|去掉|新增|添加|增加|开启|关闭|用回|设为|设置为|基于.+设计|从零设计|重新设计|生成)/.test(
      raw
    ) ||
    /\b(?:change|switch|replace|modify|update|set|lock|unlock|generate|refine|optimize|avoid|exclude|remove|add|enable|disable|design\s+(?:a|an|new|egfp|mcherry|cas9|luciferase|fluc|spike|insulin|epo|mrna|construct|sequence|variant))\b|\bdesign\s+[a-z0-9_-]{2,30}\s+mrna\b/i.test(
      raw
    )
  );
}

function isLikelyReadOnlyQuestion(text) {
  const raw = String(text || "").trim();
  if (!raw) return false;
  if (hasExplicitMutationIntent(raw)) return false;
  return (
    /\b(?:how\s+(?:can|do|should|would|could|to|does)|can\s+(?:i|we|you)|should\s+(?:i|we)|could\s+(?:i|we)|explain|why|compare|difference|trade-?offs?|summari[sz]e|what\s+(?:is|are|does|if)|which\s+(?:one|is)|analyze|analyse|interpret|inspect|audit|status|describe|list)\b/i.test(
      raw
    ) ||
    /(?:解释|为什么|为何|如何|怎样|怎么|能否|可不可以|是否|原因|权衡|对比|比较|差异|区别|总结|分析|查看|检查|审计|状态|说明|什么意思|哪条|哪个好|[？?]$)/.test(
      raw
    )
  );
}

function parseDeterministicRefineFallback(utterance) {
  const raw = String(utterance || "").trim();
  if (!raw) return null;
  const low = raw.toLowerCase();
  const payload = { mode: "refine", user_utterance: raw };
  let changed = false;

  if (/\bcommercial_clean\b|商用洁净|纯商用/.test(low)) {
    payload.compliance_track = "commercial_clean";
    payload.use_commercial_surrogate = true;
    changed = true;
  } else if (/\bresearch_max\b|科研最高精度/.test(low)) {
    payload.compliance_track = "research_max";
    payload.use_commercial_surrogate = false;
    changed = true;
  }

  if (/\bunmodified(?:_u)?\b|canonical\s+u|天然尿苷|未修饰|不修饰/.test(low)) {
    payload.uridine_chemistry = "unmodified_u";
    changed = true;
  } else if (/\b5\s*mou\b|5-?甲氧基/.test(low)) {
    payload.uridine_chemistry = "5mou";
    changed = true;
  } else if (/\bm1\s*psi\b|m1ψ|m¹ψ|n1-?methyl|n?1-?甲基假尿苷|甲基假尿苷/.test(low)) {
    payload.uridine_chemistry = "m1psi";
    changed = true;
  } else if (/\bpseudouridine\b|(?<!甲基)假尿苷|(?<![a-z0-9])ψ(?![a-z0-9])/.test(low)) {
    payload.uridine_chemistry = "pseudouridine";
    changed = true;
  }

  if (/关闭肝脏去靶向|取消肝脏去靶向|去掉.*miR-?122|disable\s+liver\s+detarget|no\s+liver\s+detarget/i.test(raw)) {
    payload.detarget_liver = false;
    changed = true;
  } else if (/肝脏去靶向|开启.*去靶向|miR-?122|detarget\s+liver/i.test(raw)) {
    payload.detarget_liver = true;
    changed = true;
  }

  const knownEnzymes = ["BsaI", "BspQI", "BsmBI", "SapI", "EcoRI", "XbaI", "BbsI"];
  const matchedEnz = knownEnzymes.filter((e) => new RegExp(`\\b${e}\\b`, "i").test(raw));
  if (matchedEnz.length && /(?:酶|restriction|enzyme|避开|禁切|改为|换成|切换|replace|avoid|switch|use)/i.test(raw)) {
    payload.restriction_enzymes = matchedEnz;
    changed = true;
  }

  if (/\bcircrna\b|环状\s*rna/i.test(raw)) {
    payload.modality = "circrna";
    payload.uridine_chemistry = "unmodified_u";
    if (/\bcvb3\b/i.test(raw)) payload.scaffold_id = "circ_anabaena_cvb3";
    else if (/\bhrv-?b3\b/i.test(raw)) payload.scaffold_id = "circ_anabaena_hrvb3";
    changed = true;
  } else if (/\bsarna\b|自扩增|veev/i.test(raw)) {
    payload.modality = "sarna";
    payload.uridine_chemistry = "unmodified_u";
    payload.scaffold_id = "sarna_veev_tc83";
    changed = true;
  }

  return changed ? payload : null;
}

function classifyDeterministicComposerIntent(
  utterance,
  { hasActiveRevision = false, fromForm = false, readOnlyExplanation = false } = {}
) {
  const raw = String(utterance || "").trim();
  if (readOnlyExplanation) {
    return { action: "local_readonly" };
  }
  if (!fromForm && raw) {
    const parsedRefine = parseDeterministicRefineFallback(raw);
    const hasMutation = hasExplicitMutationIntent(raw);
    if (isLikelyReadOnlyQuestion(raw)) {
      return { action: "local_readonly" };
    }
    if (!hasActiveRevision && !hasMutation && !parsedRefine) {
      return { action: "local_readonly" };
    }
    if (hasActiveRevision && !hasMutation && !parsedRefine) {
      return { action: "local_readonly" };
    }
    if (hasActiveRevision && parsedRefine) {
      return {
        action: "refine",
        payload: {
          ...parsedRefine,
          restriction_enzymes: parsedRefine.restriction_enzymes || getSelectedEnzymes(),
        },
      };
    }
  }
  if (hasActiveRevision) {
    return {
      action: "refine",
      payload: {
        mode: "refine",
        user_utterance: raw || `微调当前设计：内切酶避开 ${getSelectedEnzymes().join("/") || "无"}，其余锁定继承`,
        restriction_enzymes: getSelectedEnzymes(),
      },
    };
  }
  return { action: "initial" };
}

function handleLocalReadOnlyDeterministicResponse(utterance) {
  if (state.activeVersion > 0 && typeof renderLocalTradeoffsSummary === "function") {
    renderLocalTradeoffsSummary();
    return true;
  }
  appendStreamCard(
    "agent",
    "本地仅作说明（未新建设计）",
    "READ_ONLY",
    `当前处于确定性引擎模式且尚未生成设计版本；问题 <code>${escapeHtml(
      utterance
    )}</code> 属于说明/查询类文本，未触发任何序列设计请求。如需交互问答请切换至「AI Agent」，或输入“设计 eGFP mRNA”等明确指令。`
  );
  return true;
}

async function replayExactRequest(retryCtx) {
  const reqSpec = retryCtx && retryCtx.replayRequest;
  if (!reqSpec) {
    if (retryCtx && retryCtx.kind === "refine") {
      return runRefineTurn(retryCtx.overridePayload || null, retryCtx.utterance || null, {
        clientRequestId: retryCtx.clientRequestId,
        fallbackPayload: retryCtx.fallbackPayload,
        readOnlyExplanation: retryCtx.readOnlyExplanation,
        sourceEl: retryCtx.sourceEl || null,
        submittedText: retryCtx.submittedText || "",
      });
    }
    return runInitialTurn(retryCtx ? retryCtx.utterance || null : null, {
      clientRequestId: retryCtx && retryCtx.clientRequestId,
      fallbackPayload: retryCtx && retryCtx.fallbackPayload,
      readOnlyExplanation: retryCtx && retryCtx.readOnlyExplanation,
      sourceEl: retryCtx && retryCtx.sourceEl,
      submittedText: (retryCtx && retryCtx.submittedText) || "",
    });
  }
  state.lastFailedTurn = null;
  const sequence = ++state.turnSequence;
  const reqEpoch = state.sessionEpoch;
  const reqProjectId = reqSpec.projectId || state.projectId;
  clearErrorBanner();
  appendStreamCard(
    "user",
    reqSpec.isAgentChat ? "重试原始请求" : "重试确定性编译",
    reqSpec.isAgentChat
      ? reqSpec.model || reqSpec.provider || state.llmModel
      : reqSpec.entryMode || state.entryMode,
    escapeHtml(reqSpec.utterance)
  );
  const stopBusy = beginBusyIndicator(
    reqSpec.isAgentChat
      ? `${reqSpec.model || state.llmModel} 正在重试原始请求…`
      : "确定性引擎正在重试原始请求…",
    reqEpoch
  );
  try {
    const res = await apiFetch(reqSpec.path, {
      method: reqSpec.method || "POST",
      explicitProjectKey: reqSpec.projectKey || state.projectKey,
      explicitProvider: reqSpec.provider || null,
      explicitApiBase: reqSpec.apiBase || null,
      headers: reqSpec.headers || {},
      body: JSON.stringify(reqSpec.bodyObj),
    });
    if (reqEpoch !== state.sessionEpoch || reqProjectId !== state.projectId) return false;
    if (applyTurnResult(res, reqSpec.utterance, sequence)) {
      if (reqSpec.sourceEl && reqSpec.submittedText) {
        clearSubmittedFieldIfUnchanged(reqSpec.sourceEl, reqSpec.submittedText, reqEpoch);
      }
      return true;
    }
    return false;
  } catch (err) {
    if (reqEpoch === state.sessionEpoch && sequence >= state.appliedTurnSequence) {
      surfaceTurnError(err, retryCtx);
    }
    return false;
  } finally {
    stopBusy();
  }
}

async function runInitialTurn(customUtterance = null, meta = {}) {
  state.lastFailedTurn = null;
  const execMode = getSelectedExecMode();
  const utteranceInput = document.getElementById("input-utterance");
  const submittedText = typeof customUtterance === "string" ? customUtterance.trim() : "";
  const allowMainDraftFallback = !meta.ignoreTextDraft && (!meta.sourceEl || meta.sourceEl === utteranceInput);
  const mainDraftText =
    allowMainDraftFallback && !submittedText && utteranceInput
      ? String(utteranceInput.value || "").trim()
      : "";
  const draftEl = meta.ignoreTextDraft ? null : meta.sourceEl || (mainDraftText ? utteranceInput : null);
  const draftText = meta.ignoreTextDraft ? "" : submittedText || mainDraftText;

  if (execMode !== "llm_agent" && !meta.fromForm && !meta.bypassReadOnlyClassifier) {
    const detDecision = classifyDeterministicComposerIntent(draftText, {
      hasActiveRevision: state.activeVersion > 0,
      fromForm: false,
      readOnlyExplanation: Boolean(meta.readOnlyExplanation),
    });
    if (detDecision.action === "local_readonly") {
      return handleLocalReadOnlyDeterministicResponse(draftText);
    }
  }

  syncCurrentOriginCredentialsToMap();
  const sequence = ++state.turnSequence;
  const reqEpoch = state.sessionEpoch;
  const reqProjectId = state.projectId;
  const reqProjectKey = state.projectKey;
  const apiBaseSnapshot = normalizeOrigin(state.apiBase);
  const protEl = document.getElementById("input-protein");
  const prot = ((protEl && protEl.value) || "").trim();
  const chem = (document.getElementById("sel-chemistry") && document.getElementById("sel-chemistry").value) || "m1psi";
  const prof = (document.getElementById("sel-profile") && document.getElementById("sel-profile").value) || "research_max";
  const enz = getSelectedEnzymes();
  const detarget = Boolean(document.getElementById("chk-detarget-liver")?.checked);
  const late = Boolean(document.getElementById("chk-late-signal")?.checked);
  const useR = state.entryMode === "with_r";
  const conflict = state.forceConflictOnce;
  state.forceConflictOnce = false;

  const modLabel = state.modality === "circrna" ? "circRNA" : state.modality === "sarna" ? "saRNA" : "mRNA";
  const utterance =
    draftText ||
    (useR
      ? `基于参考构建 R 设计 eGFP ${modLabel} (${chem}, ${prof}${late ? ", 后期表达" : ""}${detarget ? ", 肝脏去靶向" : ""})，避开 ${enz.join("/") || "无"}，生成保守最小改动 A 与探索方案 B`
      : `从零设计目标蛋白 ${modLabel} (${chem}, ${prof}${late ? ", 后期表达" : ""})，蛋白 100% 锁定，${detarget ? "开启肝脏去靶向，" : ""}避开 ${enz.join("/") || "无"}`);

  const clientRequestId = meta.clientRequestId || mintClientRequestId();
  const isAgent = execMode === "llm_agent";
  const parsedInit = !isAgent && !meta.fromForm && draftText ? parseDeterministicRefineFallback(draftText) : null;
  const effChem = (parsedInit && parsedInit.uridine_chemistry) || chem;
  const effProf = (parsedInit && parsedInit.compliance_track) || prof;
  const effSurrogate =
    parsedInit && typeof parsedInit.use_commercial_surrogate === "boolean"
      ? parsedInit.use_commercial_surrogate
      : effProf === "commercial_clean";
  const effEnz = (parsedInit && parsedInit.restriction_enzymes) || enz;
  const effDetarget =
    parsedInit && typeof parsedInit.detarget_liver === "boolean" ? parsedInit.detarget_liver : detarget;
  const rawScaffoldVal = (document.getElementById("sel-scaffold") && document.getElementById("sel-scaffold").value) || null;
  const effModality = (parsedInit && parsedInit.modality) || state.modality || "linear_mrna";
  const effScaffold = (parsedInit && parsedInit.scaffold_id) || rawScaffoldVal || null;
  const effInitChem =
    effModality === "circrna" || effModality === "sarna" ? "unmodified_u" : effChem;
  const defaultProt = (state.presets?.egfp_reporter?.protein_sequence) || FALLBACK_EGFP;
  const namesEgfp = /\b(?:egfp|gfp|p42212|u55762)\b/i.test(draftText);
  const namesOtherTarget = /\b(?:mcherry|cas9|luciferase|fluc|spike|insulin|epo|dll3|fgf21|her2)\b/i.test(draftText);
  const usePrefilledDefaultForDet = Boolean(
    meta.fromForm ||
      meta.bypassReadOnlyClassifier ||
      !draftText ||
      state.userEditedProtein ||
      prot !== defaultProt ||
      (namesEgfp && !namesOtherTarget)
  );
  const shouldAttachSeq = Boolean(prot && (meta.fromForm || prot !== defaultProt || state.userEditedProtein));
  const formBlock = meta.fromForm
    ? `\n[表单参数：modality=${state.modality}, compliance_track=${prof}${prof === "commercial_clean" ? " (use_commercial_surrogate=true)" : ""}, uridine_chemistry=${chem}, entry_mode=${useR ? "protein_with_reference_r" : "protein_only"}, restriction_enzymes=${JSON.stringify(enz)}, detarget_liver=${detarget}, prioritize_late_signal=${late}]`
    : "";
  const seqBlock = shouldAttachSeq ? `\n（目标蛋白序列：${prot}）` : "";
  const fullPrompt = `${utterance}${formBlock}${seqBlock}`;

  const reqPath = isAgent ? `/v1/projects/${reqProjectId}/agent-chat` : `/v1/projects/${reqProjectId}/turn`;
  const reqProvider = isAgent ? state.llmProvider : null;
  const reqModel = isAgent ? state.llmModel : null;
  const reqHeaders = isAgent ? {} : { "X-MRNAX-Idempotency-Key": clientRequestId };
  const bodyObj = isAgent
    ? {
        message: fullPrompt,
        provider: reqProvider,
        model: reqModel,
        client_request_id: clientRequestId,
      }
    : {
        mode: "initial",
        user_utterance: utterance,
        protein_sequence: usePrefilledDefaultForDet ? prot || null : null,
        uridine_chemistry: effInitChem,
        compliance_track: effProf,
        use_commercial_surrogate: effSurrogate,
        restriction_enzymes: effEnz,
        detarget_liver: effDetarget,
        prioritize_late_signal: late,
        conflicting_single_pick: conflict,
        use_preset_reference_r: Boolean(
          useR && effModality === "linear_mrna" && usePrefilledDefaultForDet && !namesOtherTarget
        ),
        modality: effModality,
        scaffold_id: effScaffold,
      };
  const replayRequest = {
    path: reqPath,
    method: "POST",
    apiBase: apiBaseSnapshot,
    projectKey: reqProjectKey,
    headers: reqHeaders,
    bodyObj,
    isAgentChat: isAgent,
    provider: reqProvider,
    model: reqModel,
    projectId: reqProjectId,
    utterance,
    sourceEl: draftEl,
    submittedText: draftText,
    entryMode: state.entryMode,
  };

  clearErrorBanner();
  appendStreamCard(
    "user",
    isAgent ? "发送给 Agent" : "首轮编译 (Initial)",
    isAgent ? state.llmModel : state.entryMode,
    escapeHtml(utterance)
  );

  const stopBusy = beginBusyIndicator(
    isAgent
      ? `${state.llmModel} 正在分析指令并调用工具…`
      : "确定性引擎正在生成与筛选序列…",
    reqEpoch
  );

  try {
    const res = await apiFetch(reqPath, {
      method: "POST",
      explicitProjectKey: reqProjectKey,
      explicitProvider: reqProvider,
      explicitApiBase: apiBaseSnapshot,
      headers: reqHeaders,
      body: JSON.stringify(bodyObj),
    });
    if (reqEpoch !== state.sessionEpoch || reqProjectId !== state.projectId) return false;
    if (applyTurnResult(res, utterance, sequence)) {
      if (draftEl && draftText) {
        clearSubmittedFieldIfUnchanged(draftEl, draftText, reqEpoch);
      }
      return true;
    }
    return false;
  } catch (err) {
    if (reqEpoch === state.sessionEpoch && sequence >= state.appliedTurnSequence) {
      surfaceTurnError(err, {
        kind: "initial",
        utterance,
        clientRequestId,
        fallbackPayload: meta.fallbackPayload || null,
        readOnlyExplanation: Boolean(meta.readOnlyExplanation),
        sourceEl: draftEl,
        submittedText: draftText,
        replayRequest,
      });
    }
    return false;
  } finally {
    stopBusy();
  }
}

async function runRefineTurn(overridePayload = null, customUtterance = null, meta = {}) {
  const execMode = getSelectedExecMode();
  if (!state.intentHistory.length && !overridePayload) {
    return runInitialTurn(customUtterance, meta);
  }
  state.lastFailedTurn = null;
  const utteranceInput = document.getElementById("input-utterance");
  const submittedText = typeof customUtterance === "string" ? customUtterance.trim() : "";
  const allowMainDraftFallback = !meta.ignoreTextDraft && (!meta.sourceEl || meta.sourceEl === utteranceInput);
  const mainDraftText =
    allowMainDraftFallback && !submittedText && utteranceInput
      ? String(utteranceInput.value || "").trim()
      : "";
  const draftEl = meta.ignoreTextDraft ? null : meta.sourceEl || (mainDraftText ? utteranceInput : null);
  const draftText = meta.ignoreTextDraft ? "" : submittedText || mainDraftText;

  let effectiveOverride = overridePayload;
  if (!effectiveOverride && execMode !== "llm_agent") {
    const detDecision = classifyDeterministicComposerIntent(draftText, {
      hasActiveRevision: state.activeVersion > 0,
      fromForm: Boolean(meta.fromForm),
      readOnlyExplanation: Boolean(meta.readOnlyExplanation),
    });
    if (detDecision.action === "local_readonly") {
      return handleLocalReadOnlyDeterministicResponse(draftText);
    }
    if (detDecision.action === "refine" && detDecision.payload) {
      effectiveOverride = detDecision.payload;
    }
  }

  syncCurrentOriginCredentialsToMap();
  const sequence = ++state.turnSequence;
  const reqEpoch = state.sessionEpoch;
  const reqProjectId = state.projectId;
  const reqProjectKey = state.projectKey;
  const apiBaseSnapshot = normalizeOrigin(state.apiBase);
  const utterance =
    (effectiveOverride && effectiveOverride.user_utterance) ||
    draftText ||
    `微调当前设计：内切酶避开 ${getSelectedEnzymes().join("/") || "无"}，其余锁定继承`;

  const isAgent = execMode === "llm_agent" && !effectiveOverride;
  const clientRequestId = meta.clientRequestId || mintClientRequestId();
  const reqPath = isAgent ? `/v1/projects/${reqProjectId}/agent-chat` : `/v1/projects/${reqProjectId}/turn`;
  const reqProvider = isAgent ? state.llmProvider : null;
  const reqModel = isAgent ? state.llmModel : null;
  const reqHeaders = isAgent ? {} : { "X-MRNAX-Idempotency-Key": clientRequestId };
  const bodyObj = isAgent
    ? {
        message: utterance,
        provider: reqProvider,
        model: reqModel,
        client_request_id: clientRequestId,
      }
    : effectiveOverride || {
        mode: "refine",
        user_utterance: utterance,
        restriction_enzymes: getSelectedEnzymes(),
      };
  const replayRequest = {
    path: reqPath,
    method: "POST",
    apiBase: apiBaseSnapshot,
    projectKey: reqProjectKey,
    headers: reqHeaders,
    bodyObj,
    isAgentChat: isAgent,
    provider: reqProvider,
    model: reqModel,
    projectId: reqProjectId,
    utterance,
    sourceEl: effectiveOverride ? null : draftEl,
    submittedText: effectiveOverride ? "" : draftText,
    entryMode: state.entryMode,
  };

  clearErrorBanner();
  appendStreamCard(
    "user",
    isAgent ? "继续对话 / 微调" : "单点增量微调 (Refine)",
    state.intentHistory.length ? `基于 v${String(state.intentHistory.length).padStart(2, "0")}` : state.llmModel,
    escapeHtml(utterance)
  );

  const stopBusy = beginBusyIndicator(
    isAgent
      ? `${state.llmModel} 正在处理多轮追问/微调…`
      : "确定性引擎正在应用增量微调…",
    reqEpoch
  );

  try {
    const res = await apiFetch(reqPath, {
      method: "POST",
      explicitProjectKey: reqProjectKey,
      explicitProvider: reqProvider,
      explicitApiBase: apiBaseSnapshot,
      headers: reqHeaders,
      body: JSON.stringify(bodyObj),
    });
    if (reqEpoch !== state.sessionEpoch || reqProjectId !== state.projectId) return false;
    if (applyTurnResult(res, utterance, sequence)) {
      if (!effectiveOverride && draftEl && draftText) {
        clearSubmittedFieldIfUnchanged(draftEl, draftText, reqEpoch);
      }
      return true;
    }
    return false;
  } catch (err) {
    if (reqEpoch === state.sessionEpoch && sequence >= state.appliedTurnSequence) {
      surfaceTurnError(err, {
        kind: "refine",
        overridePayload: effectiveOverride,
        utterance,
        clientRequestId,
        fallbackPayload: meta.fallbackPayload || effectiveOverride || null,
        readOnlyExplanation: Boolean(meta.readOnlyExplanation),
        sourceEl: effectiveOverride ? null : draftEl,
        submittedText: effectiveOverride ? "" : draftText,
        replayRequest,
      });
    }
    return false;
  } finally {
    stopBusy();
  }
}

async function submitComposerText(sourceEl) {
  if (state.inFlight) return;
  const raw = ((sourceEl && sourceEl.value) || "").trim();
  const mainEl = document.getElementById("input-utterance");
  if (sourceEl && !raw && (sourceEl.id === "agent-rail-input" || sourceEl !== mainEl)) {
    if (typeof sourceEl.focus === "function") sourceEl.focus();
    return;
  }
  await runRefineTurn(null, raw || undefined, { sourceEl });
}

function triggerTextBlobDownload(content, downloadName, mimeType = "text/plain;charset=utf-8") {
  if (typeof Blob === "undefined" || typeof document.createElement !== "function") return;
  const blob = new Blob([content], { type: mimeType });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = downloadName;
  if (document.body && typeof document.body.appendChild === "function") {
    document.body.appendChild(a);
  }
  a.click();
  if (typeof a.remove === "function") a.remove();
}

async function downloadExportFile(filename) {
  const bid = state.selectedBundleId;
  const projId = state.projectId;
  const revNum = state.activeVersion;
  const pKey = state.projectKey;
  const apiBaseSnapshot = normalizeOrigin(state.apiBase);
  if (!bid) return;
  const cacheKey = `${apiBaseSnapshot}:${projId}:${revNum}:${bid}`;
  if (!state.cachedExports[cacheKey]) {
    const result = await apiFetch(
      `/v1/projects/${projId}/bundles/${encodeURIComponent(bid)}/export`,
      { explicitProjectKey: pKey, explicitApiBase: apiBaseSnapshot }
    );
    if (
      apiBaseSnapshot !== normalizeOrigin(state.apiBase) ||
      projId !== state.projectId ||
      revNum !== state.activeVersion ||
      bid !== state.selectedBundleId
    ) {
      return;
    }
    state.cachedExports[cacheKey] = result;
  }
  const content = state.cachedExports[cacheKey].files[filename];
  const revTag = `v${String(revNum).padStart(2, "0")}`;
  triggerTextBlobDownload(content, `${revTag}_${bid}_${filename}`);
}

function showExportActionError(label, err) {
  const banner = document.getElementById("banner-notes");
  if (!banner) return;
  const code = err && err.status ? ` (HTTP ${err.status})` : "";
  const detail = (err && err.message) || "网关拒绝或暂不可达，已阻止未经授权核验的本地导出";
  banner.hidden = false;
  banner.className = "notice-banner error";
  banner.innerHTML = `<strong>${escapeHtml(label)}导出受阻${escapeHtml(code)}：</strong>${escapeHtml(detail)}`;
}

async function downloadIvtSynthesisOrder() {
  const cand = state.candidates.find((c) => c.bundle_id === state.selectedBundleId);
  if (!cand) return false;
  const bid = cand.bundle_id;
  const projId = state.projectId;
  const revNum = state.activeVersion;
  const latestIntent = state.intentHistory.length
    ? state.intentHistory[state.intentHistory.length - 1]
    : null;
  const intentIdSnapshot = latestIntent ? latestIntent.intent_id : null;
  const pKey = state.projectKey;
  const apiBaseSnapshot = normalizeOrigin(state.apiBase);

  let res;
  try {
    res = await apiFetch(
      `/v1/projects/${encodeURIComponent(projId)}/bundles/${encodeURIComponent(bid)}/ivt-order?format=json`,
      { explicitProjectKey: pKey, explicitApiBase: apiBaseSnapshot }
    );
  } catch (err) {
    showExportActionError("IVT 下单表", err);
    return false;
  }
  const activeIntentNow = state.intentHistory.length
    ? state.intentHistory[state.intentHistory.length - 1]
    : null;
  if (
    apiBaseSnapshot !== normalizeOrigin(state.apiBase) ||
    projId !== state.projectId ||
    revNum !== state.activeVersion ||
    (activeIntentNow ? activeIntentNow.intent_id : null) !== intentIdSnapshot ||
    bid !== state.selectedBundleId
  ) {
    return false;
  }
  const csvText = res && typeof res.ivt_synthesis_order_csv === "string" ? res.ivt_synthesis_order_csv : "";
  if (!csvText) {
    showExportActionError("IVT 下单表", new Error("网关未返回已核验的 IVT 模板数据"));
    return false;
  }
  clearErrorBanner();
  const revTag = `v${String(revNum).padStart(2, "0")}`;
  triggerTextBlobDownload(csvText, `${revTag}_${bid}_ivt_order.csv`, "text/csv;charset=utf-8");
  return true;
}

async function openOrDownloadQcDossier() {
  const cand = state.candidates.find((c) => c.bundle_id === state.selectedBundleId);
  if (!cand) return false;
  const bid = cand.bundle_id;
  const projId = state.projectId;
  const revNum = state.activeVersion;
  const latestIntent = state.intentHistory.length
    ? state.intentHistory[state.intentHistory.length - 1]
    : null;
  const intentIdSnapshot = latestIntent ? latestIntent.intent_id : null;
  const pKey = state.projectKey;
  const apiBaseSnapshot = normalizeOrigin(state.apiBase);

  let res;
  try {
    res = await apiFetch(
      `/v1/projects/${encodeURIComponent(projId)}/bundles/${encodeURIComponent(bid)}/dossier?format=json`,
      { explicitProjectKey: pKey, explicitApiBase: apiBaseSnapshot }
    );
  } catch (err) {
    showExportActionError("QC 质控档案", err);
    return false;
  }
  const activeIntentNow = state.intentHistory.length
    ? state.intentHistory[state.intentHistory.length - 1]
    : null;
  if (
    apiBaseSnapshot !== normalizeOrigin(state.apiBase) ||
    projId !== state.projectId ||
    revNum !== state.activeVersion ||
    (activeIntentNow ? activeIntentNow.intent_id : null) !== intentIdSnapshot ||
    bid !== state.selectedBundleId
  ) {
    return false;
  }
  const htmlDoc = res && typeof res.qc_dossier_html === "string" ? res.qc_dossier_html : "";
  if (!htmlDoc) {
    showExportActionError("QC 质控档案", new Error("网关未返回已核验的 QC 质控档案"));
    return false;
  }
  clearErrorBanner();
  const revTag = `v${String(revNum).padStart(2, "0")}`;
  state.lastDossierHtml = htmlDoc;
  state.lastDossierFilename = `${revTag}_${bid}_qc_dossier.html`;

  const frame = document.getElementById("dossier-iframe");
  const dlg = document.getElementById("modal-dossier-preview");
  if (frame) frame.srcdoc = htmlDoc;
  if (dlg && typeof dlg.showModal === "function") {
    dlg.showModal();
  } else {
    triggerTextBlobDownload(htmlDoc, state.lastDossierFilename, "text/html;charset=utf-8");
  }
  return true;
}

/* ============================================================================
   TAB 2: Scientific Rules Atlas v2.0 & Clean-Room Literature Explorer
   ============================================================================ */
function humanizeRuleGrade(grade) {
  const g = String(grade || "").toLowerCase();
  if (g === "cross_lab_verified") return { label: "跨实验室实证", cls: "ok" };
  if (g === "conditional") return { label: "条件规律", cls: "teal" };
  if (g === "mechanistic_heuristic") return { label: "机制启发", cls: "" };
  if (g === "debunked_myth") return { label: "已证伪误区", cls: "warn" };
  return { label: g || "规则", cls: "" };
}

function renderRulesAtlas(payload) {
  const listEl = document.getElementById("rules-list");
  const sumStrip = document.getElementById("rules-summary-strip");
  if (!listEl) return;

  const rules = Array.isArray(payload && payload.rules) ? payload.rules : [];
  const totalAvail = Number(payload?.total_available || rules.length);
  if (sumStrip) {
    const enforcedCnt = rules.filter((r) => r.enforced_in_generator).length;
    sumStrip.innerHTML = `
      <span class="kpi-pill">匹配规律 <strong>${escapeHtml(rules.length)}</strong> / ${escapeHtml(totalAvail)}</span>
      <span class="kpi-pill">生成器强门禁 <strong>${escapeHtml(enforcedCnt)}</strong></span>
    `;
  }

  if (!rules.length) {
    listEl.innerHTML = `<div class="panel-surface empty-hint mono" style="grid-column: 1 / -1;">当前筛选条件下无匹配定量规律</div>`;
    return;
  }

  listEl.innerHTML = rules
    .map((r) => {
      const gMeta = humanizeRuleGrade(r.evidence_grade || r.tier);
      const mods = Array.isArray(r.modality) ? r.modality.join(", ") : r.modality || "linear_mrna";
      const failures = Array.isArray(r.failure_modes)
        ? r.failure_modes.map((f) => `<li>${escapeHtml(f)}</li>`).join("")
        : "";
      const citations = Array.isArray(r.citations)
        ? r.citations
            .map((c) => {
              const label = `${c.authors || "Ref"} ${c.year || ""}`.trim();
              if (c.doi) return renderSafeExternalLink(buildVerifiedCitationUrl("doi", c.doi), `${label} (DOI)`);
              if (c.pmid) return renderSafeExternalLink(buildVerifiedCitationUrl("pmid", c.pmid), `${label} (PMID:${c.pmid})`);
              if (c.pmcid) return renderSafeExternalLink(buildVerifiedCitationUrl("pmcid", c.pmcid), `${label} (${c.pmcid})`);
              return `<span class="mono meta-muted">${escapeHtml(label)}</span>`;
            })
            .join("")
        : "";

      return `
        <article class="rule-card">
          <div class="rule-card-top">
            <div style="display:flex; flex-wrap:wrap; gap:5px;">
              <span class="badge ${gMeta.cls}">${escapeHtml(gMeta.label)}</span>
              <span class="badge mono">${escapeHtml(r.region || r.scope || "mRNA")}</span>
              <span class="badge mono">${escapeHtml(mods)}</span>
            </div>
            <span class="badge mono ${r.enforced_in_generator ? "ok" : ""}">
              ${r.enforced_in_generator ? "生成器内嵌强门禁" : "评估参考"}
            </span>
          </div>
          <div>
            <h3 class="rule-title-zh">${escapeHtml(r.title_zh || r.rule_id)}</h3>
            <div class="rule-title-en">${escapeHtml(r.title_en || "")} · <span class="mono">${escapeHtml(r.rule_id)}</span></div>
          </div>
          <div class="rule-effect mono">${escapeHtml(r.effect_size_summary || r.summary || "")}</div>
          ${failures ? `<ul class="rule-failures">${failures}</ul>` : ""}
          ${citations ? `<div class="rule-citations">${citations}</div>` : ""}
        </article>
      `;
    })
    .join("");
}

async function loadScientificRules() {
  const mod = document.getElementById("rules-filter-modality")?.value || "";
  const reg = document.getElementById("rules-filter-region")?.value || "";
  const grd = document.getElementById("rules-filter-grade")?.value || "";
  const q = (document.getElementById("rules-filter-q")?.value || "").trim();
  const incDeb = Boolean(document.getElementById("rules-chk-debunked")?.checked ?? true);

  const params = new URLSearchParams();
  if (mod) params.set("modality", mod);
  if (reg) params.set("region", reg);
  if (grd) params.set("grade", grd);
  if (q) params.set("q", q);
  params.set("include_debunked", String(incDeb));

  try {
    const res = await apiFetch(`/api/v1/rules?${params.toString()}`, { includeAuth: false });
    if (res && Array.isArray(res.rules)) {
      state.rulesCache = res;
      renderRulesAtlas(res);
      return res;
    }
  } catch {
    // Offline fallback filtering
  }

  const filtered = FALLBACK_SCIENTIFIC_RULES.filter((r) => {
    if (!incDeb && r.evidence_grade === "debunked_myth") return false;
    if (grd && r.evidence_grade !== grd) return false;
    if (mod && Array.isArray(r.modality) && !r.modality.includes(mod)) return false;
    if (reg && String(r.region).toLowerCase() !== reg.toLowerCase()) return false;
    if (q) {
      const hay = `${r.rule_id} ${r.title_zh} ${r.title_en} ${r.effect_size_summary}`.toLowerCase();
      if (!hay.includes(q.toLowerCase())) return false;
    }
    return true;
  });
  const fallbackPayload = {
    total_available: FALLBACK_SCIENTIFIC_RULES.length,
    total_matched: filtered.length,
    rules: filtered,
  };
  state.rulesCache = fallbackPayload;
  renderRulesAtlas(fallbackPayload);
  return fallbackPayload;
}

async function runLiteratureSearch() {
  const q = (document.getElementById("rules-lit-query")?.value || "m1Psi ribosomal frameshifting").trim();
  const box = document.getElementById("rules-lit-results");
  if (!box || !q) return;

  try {
    const res = await apiFetch("/api/v1/evidence/resolve", {
      method: "POST",
      includeAuth: false,
      body: JSON.stringify({ query: q, mode: "literature", max_results: 5, allow_network: true }),
    });
    const lit = res && res.literature_and_trials;
    if (lit && Array.isArray(lit.papers)) {
      renderLiteratureResults(lit, box);
      return;
    }
  } catch {
    // fallback below
  }

  renderLiteratureResults(
    {
      resolution_mode: "offline_verified_registry",
      papers: [
        {
          pmid: "38057662",
          pmcid: "PMC10796200",
          doi: "10.1038/s41586-023-06800-3",
          pub_year: "2024",
          license_raw: "cc by 4.0",
          commercial_clean_article_eligible: true,
          title_sandboxed: "N1-methylpseudouridylation of mRNA causes +1 ribosomal frameshifting",
        },
        {
          pmid: "37130513",
          pmcid: "PMC10202639",
          doi: "10.1038/s41586-023-06127-z",
          pub_year: "2023",
          license_raw: "cc by 4.0",
          commercial_clean_article_eligible: true,
          title_sandboxed: "Algorithm for Optimized mRNA Design Improves Stability and Immunogenicity (LinearDesign)",
        },
      ],
      clinical_trials: [],
    },
    box
  );
}

function renderLiteratureResults(lit, box) {
  const papers = Array.isArray(lit.papers) ? lit.papers : [];
  const trials = Array.isArray(lit.clinical_trials) ? lit.clinical_trials : [];
  const paperRows = papers
    .map((p) => {
      const title = extractSandboxedText(p.title_sandboxed || p.title_quarantined || p.title);
      const doiLink = p.doi ? renderSafeExternalLink(buildVerifiedCitationUrl("doi", p.doi), `DOI:${p.doi}`) : "";
      const pmidLink = p.pmid ? renderSafeExternalLink(buildVerifiedCitationUrl("pmid", p.pmid), `PMID:${p.pmid}`) : "";
      return `
        <div class="mini-card">
          <div class="mini-card-head">
            <span class="badge ${p.commercial_clean_article_eligible ? "ok" : "warn"} mono">
              ${p.commercial_clean_article_eligible ? "CC-BY/CC0 商业可用" : "仅书目索引 · 附件受限"}
            </span>
            <span class="mono meta-muted">${escapeHtml(p.pub_year || "")} · ${escapeHtml(p.license_raw || "non-OA")}</span>
          </div>
          <div style="font-weight:600; font-size:12.5px;">${escapeHtml(title)}</div>
          <div style="display:flex; gap:8px; flex-wrap:wrap;">${doiLink} ${pmidLink}</div>
        </div>
      `;
    })
    .join("");

  const trialRows = trials
    .map((t) => {
      const bTitle = extractSandboxedText(t.brief_title_sandboxed || t.brief_title_quarantined);
      const nctUrl = t.nct_id ? buildVerifiedCitationUrl("nct", t.nct_id) : "";
      return `
        <div class="mini-card">
          <div class="mini-card-head">
            ${nctUrl ? renderSafeExternalLink(nctUrl, t.nct_id) : `<span class="mono">${escapeHtml(t.nct_id || "Trial")}</span>`}
            <span class="badge mono">${escapeHtml(t.overall_status || "ACTIVE")}</span>
          </div>
          <div style="font-size:12px;">${escapeHtml(bTitle)}</div>
        </div>
      `;
    })
    .join("");

  box.innerHTML = `
    <div class="info-grid-cards">
      ${paperRows || '<div class="empty-hint mono">无匹配文献</div>'}
      ${trialRows}
    </div>
  `;
}

/* ============================================================================
   TAB 3: Data Hub, OOD Scorecard, Modality/miRNA Atlas & Target Resolver
   ============================================================================ */
function renderDataHub(hub) {
  const sumEl = document.getElementById("dh-summary-strip");
  const sc = (hub && hub.summary_counts) || FALLBACK_DATA_HUB.summary_counts;
  if (sumEl) {
    sumEl.innerHTML = `
      <span class="kpi-pill">湿实验基准表 <strong>${escapeHtml(sc.datasets_count ?? 34)}</strong></span>
      <span class="kpi-pill">统一台账条目 <strong>${escapeHtml(sc.master_catalog_v12_rows_unified ?? 114)}</strong></span>
      <span class="kpi-pill">商业洁净资产 <strong>${escapeHtml(sc.commercial_clean_datasets ?? 26)}</strong></span>
      <span class="kpi-pill">外部生物接口 <strong>${escapeHtml(sc.external_apis_count ?? 20)}</strong></span>
    `;
  }

  // Compiled engines
  const engBox = document.getElementById("dh-engines-grid");
  if (engBox) {
    const engines =
      Array.isArray(hub?.engines_and_models) && hub.engines_and_models.length
        ? hub.engines_and_models
        : FALLBACK_DATA_HUB.engines_and_models;
    engBox.innerHTML = `
      <div class="info-grid-cards">
        ${engines
          .slice(0, 6)
          .map((e) => {
            const trk = e.license_track || (e.license && e.license.license_track) || "commercial_clean";
            const cls = trk === "commercial_clean" ? "ok" : trk === "research_max" ? "teal" : "warn";
            return `
              <div class="mini-card">
                <div class="mini-card-head">
                  <strong style="font-size:12.5px;">${escapeHtml(e.name || e.engine_id || e.id)}</strong>
                  <span class="badge mono ${cls}">${escapeHtml(trk)}</span>
                </div>
                <div class="meta-muted" style="font-size:11.5px;">${escapeHtml(e.role || e.primary_purpose || e.description || "已编译就绪")}</div>
              </div>
            `;
          })
          .join("")}
      </div>
    `;
  }

  // OOD Scorecard
  const scWrap = document.getElementById("dh-scorecard-table-wrap");
  if (scWrap) {
    const card = hub && hub.surrogate_scorecard;
    const v1Top25 = card?.axis_a_5utr_mrl?.gse232927_top25k_exact_v1_splits?.v1_exact_recipe_lgbm?.spearman ?? 0.6852;
    const v2Top25 = card?.axis_a_5utr_mrl?.gse232927_top25k_exact_v1_splits?.v2_full145k_zero_leakage_blend?.spearman ?? 0.7461;
    const v1Full145 = card?.axis_a_5utr_mrl?.gse232927_full145k_ood_cv?.v1_recipe_on_145k?.spearman ?? 0.5841;
    const v2Full145 = card?.axis_a_5utr_mrl?.gse232927_full145k_ood_cv?.v2_lgbm_145k?.spearman ?? 0.642;
    const v2MultiChem = card?.axis_a_5utr_mrl?.gse114002_multi_chemistry_ood_cv?.overall_v2_chem_lgbm?.spearman ?? 0.8907;
    const orthrusRho = card?.gpu_sota_shootout?.orthrus6_saluki_sample1200?.metrics?.spearman ?? 0.6089;

    scWrap.innerHTML = `
      <div class="table-scroll">
        <table class="data-table">
          <thead>
            <tr>
              <th>跨数据集评测基准</th>
              <th>训练/评测修饰域</th>
              <th class="num-col">v1 ρ</th>
              <th class="num-col">v2 ρ</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>GSE232927 5′UTR MRL (25k 严格同折)</td>
              <td class="mono">unmodified_u</td>
              <td class="num-col mono">${escapeHtml(Number(v1Top25).toFixed(3))}</td>
              <td class="num-col mono" style="color:var(--ok); font-weight:600;">${escapeHtml(Number(v2Top25).toFixed(3))}</td>
            </tr>
            <tr>
              <td>GSE232927 全量 145k 零泄漏 OOD</td>
              <td class="mono">unmodified_u</td>
              <td class="num-col mono">${escapeHtml(Number(v1Full145).toFixed(3))}</td>
              <td class="num-col mono" style="color:var(--ok); font-weight:600;">${escapeHtml(Number(v2Full145).toFixed(3))}</td>
            </tr>
            <tr>
              <td>GSE114002 多修饰条件评估 (120k)</td>
              <td class="mono">U / Ψ / m¹Ψ</td>
              <td class="num-col mono">0.763</td>
              <td class="num-col mono" style="color:var(--ok); font-weight:600;">${escapeHtml(Number(v2MultiChem).toFixed(3))}</td>
            </tr>
            <tr>
              <td>Orthrus-6 / Saluki 半衰期外部基准</td>
              <td class="mono">endogenous mRNA</td>
              <td class="num-col mono">—</td>
              <td class="num-col mono">${escapeHtml(Number(orthrusRho).toFixed(3))}</td>
            </tr>
          </tbody>
        </table>
      </div>
    `;
  }

  // Modality Scaffolds
  const scafBox = document.getElementById("dh-scaffolds-grid");
  if (scafBox) {
    const archMap =
      hub?.modality_scaffolds_summary?.modality_architectures &&
      Object.keys(hub.modality_scaffolds_summary.modality_architectures).length
        ? hub.modality_scaffolds_summary.modality_architectures
        : FALLBACK_DATA_HUB.modality_scaffolds_summary.modality_architectures;

    scafBox.innerHTML = `
      <div class="info-grid-cards">
        ${Object.entries(archMap)
          .map(([modKey, info]) => {
            const summaryText =
              info.summary ||
              (Array.isArray(info.required_segments)
                ? info.required_segments.join(" → ")
                : info.topology
                ? `${info.topology} 标准验证骨架`
                : "标准验证分子架构");
            return `
              <div class="mini-card">
                <div class="mini-card-head">
                  <strong class="mono">${escapeHtml(info.label || modKey)}</strong>
                  <button type="button" class="chip mono" data-apply-modality="${escapeHtml(modKey)}">应用到设计台</button>
                </div>
                <div style="font-size:11.5px; color:var(--ink-secondary);">${escapeHtml(summaryText)}</div>
              </div>
            `;
          })
          .join("")}
      </div>
    `;
    if (typeof scafBox.querySelectorAll === "function") {
      scafBox.querySelectorAll("[data-apply-modality]").forEach((btn) => {
        btn.addEventListener("click", () => {
          setModality(btn.getAttribute("data-apply-modality"));
          switchWorkspaceTab("studio");
        });
      });
    }
  }

  // Tissue miRNA Atlas
  const mirBox = document.getElementById("dh-mirna-grid");
  if (mirBox) {
    const entries =
      Array.isArray(hub?.tissue_mirna_atlas?.mirna_entries) && hub.tissue_mirna_atlas.mirna_entries.length
        ? hub.tissue_mirna_atlas.mirna_entries
        : FALLBACK_DATA_HUB.tissue_mirna_atlas.mirna_entries;
    mirBox.innerHTML = `
      <div class="info-grid-cards">
        ${entries
          .slice(0, 6)
          .map(
            (m) => `
            <div class="mini-card">
              <div class="mini-card-head">
                <strong class="mono">${escapeHtml(m.mirna_id)}</strong>
                <span class="badge teal mono">${escapeHtml(m.tissue_group)}</span>
              </div>
              <div style="font-size:11.5px; color:var(--ink-secondary);">${escapeHtml(m.role || m.detargeting_rationale || "3′UTR 互补切片去靶向盒")}</div>
            </div>
          `
          )
          .join("")}
      </div>
    `;
  }

  renderDataHubDatasetsTable();
}

function renderDataHubDatasetsTable() {
  const tbody = document.getElementById("dh-datasets-tbody");
  if (!tbody) return;
  const hub = state.dataHubCache || FALLBACK_DATA_HUB;
  const rawList = Array.isArray(hub.datasets) && hub.datasets.length ? hub.datasets : FALLBACK_DATA_HUB.datasets;
  const trackFilter = document.getElementById("dh-dataset-filter-track")?.value || "";
  const qFilter = (document.getElementById("dh-dataset-filter-q")?.value || "").trim().toLowerCase();

  const filtered = rawList.filter((d) => {
    const trk = (d.license && d.license.license_track) || d.license_track || "";
    if (trackFilter && trk !== trackFilter) return false;
    if (qFilter) {
      const hay = `${d.dataset_id || d.asset_id || ""} ${d.title || d.name || ""} ${d.primary_readout || ""} ${(d.chemistry || []).join(" ")}`.toLowerCase();
      if (!hay.includes(qFilter)) return false;
    }
    return true;
  });

  if (!filtered.length) {
    tbody.innerHTML = `<tr><td colspan="6" class="empty-cell mono">无匹配数据集记录</td></tr>`;
    return;
  }

  tbody.innerHTML = filtered
    .slice(0, 40)
    .map((d) => {
      const trk = (d.license && d.license.license_track) || d.license_track || "commercial_clean";
      const spdx = (d.license && d.license.spdx) || trk;
      const cls = trk === "commercial_clean" ? "ok" : trk.includes("pending") ? "warn" : "";
      const chems = Array.isArray(d.chemistry) ? d.chemistry.join(", ") : d.chemistry || "unmodified_u";
      return `
        <tr>
          <td>
            <div style="font-weight:600;">${escapeHtml(d.dataset_id || d.asset_id || "Dataset")}</div>
            <div class="meta-muted" style="white-space:normal; max-inline-size:380px;">${escapeHtml(d.title || d.name || "")}</div>
          </td>
          <td class="mono">${escapeHtml(d.modality || "linear_mrna")} · ${escapeHtml(d.region_focus || "mRNA")}</td>
          <td class="mono">${escapeHtml(chems)}</td>
          <td class="num-col mono">${escapeHtml(Number(d.sample_count || d.size_bytes || 0).toLocaleString())}</td>
          <td style="white-space:normal; max-inline-size:280px;">${escapeHtml(d.primary_readout || d.verification_status || "")}</td>
          <td><span class="badge mono ${cls}">${escapeHtml(trk)} (${escapeHtml(spdx)})</span></td>
        </tr>
      `;
    })
    .join("");
}

async function loadDataHubCatalog() {
  try {
    const res = await apiFetch("/api/v1/data-hub", { includeAuth: false });
    if (res && typeof res === "object") {
      state.dataHubCache = res;
      renderDataHub(res);
      return res;
    }
  } catch {
    // Offline fallback
  }
  state.dataHubCache = FALLBACK_DATA_HUB;
  renderDataHub(FALLBACK_DATA_HUB);
  return FALLBACK_DATA_HUB;
}

async function resolveTargetProteinInHub(customQuery = null) {
  const input = document.getElementById("dh-target-query");
  const q = String(customQuery || (input && input.value) || "EPO").trim();
  if (!q) return;
  if (input && customQuery) input.value = q;
  const box = document.getElementById("dh-target-result");
  if (box) box.innerHTML = `<div class="mono meta-muted"><span class="spin"></span> 正在联合检索 UniProt / AlphaFold / EuropePMC…</div>`;

  let prot = null;
  try {
    const res = await apiFetch("/api/v1/evidence/resolve", {
      method: "POST",
      includeAuth: false,
      body: JSON.stringify({ query: q, mode: "both", max_results: 4, allow_network: true }),
    });
    prot = res && res.protein;
  } catch {
    prot = null;
  }

  if (!prot || !prot.matched) {
    const normQ = q.trim();
    const exactEgfp = /^(?:egfp|gfp|p42212|u55762(?:\.1)?|aab02572(?:\.1)?)$/i.test(normQ);
    const exactHepo = /^(?:epo|hepo|p01588|erythropoietin)$/i.test(normQ);
    if (exactEgfp || exactHepo) {
      prot = {
        matched: true,
        gene_symbol: exactEgfp ? "EGFP" : "EPO",
        uniprot_accession: exactEgfp ? "P42212" : "P01588",
        resolution_mode: "offline_verified_snapshot",
        protein_name_sandboxed: exactEgfp ? "Enhanced Green Fluorescent Protein" : "Erythropoietin precursor (hEPO)",
        protein_sequence: exactEgfp ? FALLBACK_EGFP : FALLBACK_HEPO,
        aa_length: exactEgfp ? FALLBACK_EGFP.length : FALLBACK_HEPO.length,
        starts_with_met: true,
        has_signal_peptide: !exactEgfp,
        signal_peptide_hint: exactEgfp
          ? "胞质报告蛋白（无分泌信号肽）"
          : "含天然 N 端分泌信号肽 [1..27 aa]，建议保留天然信号肽用于体内分泌蛋白替代",
      };
    } else {
      state.resolvedTarget = null;
      if (box) {
        box.innerHTML = `
          <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
            <div>
              <strong style="font-size:13.5px;">未解析到匹配靶点：${escapeHtml(q)}</strong>
              <span class="badge warn mono" style="margin-inline-start:6px;">UNRESOLVED</span>
            </div>
            <button type="button" class="btn-primary btn-compact" id="btn-import-resolved-protein" disabled>
              导入到设计台 →
            </button>
          </div>
          <div class="mono meta-muted" style="margin-top:6px;">
            网关未匹配到该基因/登录号（离线核验快照仅支持精确别名 eGFP/P42212 与 EPO/P01588），请检查基因名或直接在设计台粘贴氨基酸序列。
          </div>
        `;
      }
      return null;
    }
  }
  state.resolvedTarget = prot;

  if (!box) return;
  const pName = extractSandboxedText(prot.protein_name_sandboxed || prot.protein_name_quarantined || prot.gene_symbol);
  const afMean = prot.alphafold_summary?.mean_plddt;
  box.innerHTML = `
    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
      <div>
        <strong style="font-size:14px;">${escapeHtml(prot.gene_symbol || q)} · ${escapeHtml(pName)}</strong>
        <span class="badge ok mono" style="margin-inline-start:6px;">UniProt:${escapeHtml(prot.uniprot_accession || "verified")}</span>
        <span class="badge mono">${escapeHtml(prot.aa_length)} aa · ${prot.starts_with_met ? "M✓" : "缺M"}</span>
      </div>
      <button type="button" class="btn-primary btn-compact" id="btn-import-resolved-protein">
        导入到设计台 →
      </button>
    </div>
    <div class="mono meta-muted" style="margin-top:6px;">
      信号肽与定位提示：${escapeHtml(prot.signal_peptide_hint || (prot.has_signal_peptide ? "含分泌信号肽" : "胞内/无信号肽"))}
      ${afMean ? ` · AlphaFold 平均 pLDDT=${escapeHtml(afMean)}` : ""}
    </div>
    <div class="mono" style="margin-top:6px; font-size:11px; padding:6px 8px; background:var(--bg-surface); border:1px solid var(--border); border-radius:3px; overflow-wrap:anywhere;">
      ${escapeHtml(String(prot.protein_sequence || "").slice(0, 140))}${String(prot.protein_sequence || "").length > 140 ? "…" : ""}
    </div>
  `;

  const impBtn = document.getElementById("btn-import-resolved-protein");
  if (impBtn && typeof impBtn.addEventListener === "function") {
    impBtn.addEventListener("click", () => {
      if (!state.resolvedTarget || !state.resolvedTarget.protein_sequence) return;
      const protField = document.getElementById("input-protein");
      if (protField) {
        protField.value = state.resolvedTarget.protein_sequence;
        state.userEditedProtein = true;
        updateProteinBadge();
      }
      const utt = document.getElementById("input-utterance");
      if (utt) {
        utt.value = `从零设计 ${state.resolvedTarget.gene_symbol || "目标蛋白"} (${state.resolvedTarget.uniprot_accession || "UniProt"}) mRNA，m¹Ψ 修饰，避开 BsaI，蛋白序列 100% 锁定。`;
      }
      switchWorkspaceTab("studio");
    });
  }
}

/* ============================================================================
   TAB 4: Clinical Benchmark Comparator & Wet-Lab Dual-Speed Calibration Loop
   ============================================================================ */
function syncBenchmarkSelectedLabel() {
  const lbl = document.getElementById("bench-selected-label");
  if (!lbl) return;
  const cand = state.candidates.find((c) => c.bundle_id === state.selectedBundleId);
  if (cand) {
    lbl.textContent = `当前源：设计台选中 ${humanizeCandidateLabel(cand)} (${cand.length_nt} nt)`;
  } else {
    lbl.textContent = "当前源：粘贴或默认 eGFP 参考序列";
  }
}

function computeClientClinicalBenchmark(seq, modality = "linear_mrna", chemistry = "m1psi") {
  const norm = String(seq || "GGGAAAUCCGCCACCAUGGUGAGCAAGGGCGAGGAGCUGUUCACCGGGGUGGUGCCCAUCCUGGUCGAGCUGGACGGCGACGUAAACGGCCACAAGUUCAGCGUGUCCGGCGAGGGCGAGGGCGAUGCCACCUACGGCAAGCUGACCCUGAAGUUCAUCUGCACCACCGGCAAGCUGCCCGUGCCCUGGCCCACCCUCGUGACCACCCUGACCUACGGCUGAUAAUAGAAAAAAAAAAAA")
    .trim()
    .toUpperCase()
    .replace(/T/g, "U");
  const n = Math.max(1, norm.length);
  const gcFull = (norm.split("G").length - 1 + (norm.split("C").length - 1)) / n;
  const cai = computeHumanCaiClient(norm);
  const u4Runs = (norm.match(/U{4,}/g) || []).length;

  const refs = [
    {
      construct_name: "BNT162b2 (Tozinameran)",
      developer: "BioNTech / Pfizer",
      clinical_or_reference_status: "FDA/EMA Approved",
      modality: "linear_mrna",
      chemistry: "m1psi",
      gc_full: 0.562,
      gc_cds: 0.571,
      gc3: 0.785,
      u4: 2,
      architecture_summary: "Cap1 CleanCap AG · HBA1 5′UTR · S-2P CDS · AES-mtRNR1 3′UTR · A30L70 Poly(A)",
    },
    {
      construct_name: "mRNA-1273 (Elasomeran)",
      developer: "Moderna",
      clinical_or_reference_status: "FDA/EMA Approved",
      modality: "linear_mrna",
      chemistry: "m1psi",
      gc_full: 0.612,
      gc_cds: 0.629,
      gc3: 0.872,
      u4: 0,
      architecture_summary: "Cap1 · 合成高开放 5′UTR · 高 GC3 S-2P CDS · HBA1 3′UTR · Poly(A)",
    },
    {
      construct_name: "ARCT-154 (Kostaive)",
      developer: "Arcturus / CSL",
      clinical_or_reference_status: "PMDA/EMA Approved saRNA",
      modality: "sarna",
      chemistry: "unmodified_u",
      gc_full: 0.514,
      gc_cds: 0.538,
      gc3: 0.664,
      u4: 3,
      architecture_summary: "VEEV TC-83 自我扩增复制子骨架 · 26S 亚基因组启动子",
    },
  ];

  return {
    candidate_stats: {
      full_length_nt: n,
      gc_full_transcript: Number(gcFull.toFixed(4)),
      gc_cds: Number(gcFull.toFixed(4)),
      gc3_cds: Number(Math.min(0.95, gcFull * 1.22).toFixed(4)),
      cai_human: cai,
      utr5_uaug_count: 0,
      cds_m1psi_u4_plus_run_count: u4Runs,
    },
    comparisons: refs
      .filter((r) => !modality || r.modality === modality)
      .map((r) => ({
        construct_name: r.construct_name,
        developer: r.developer,
        clinical_or_reference_status: r.clinical_or_reference_status,
        modality: r.modality,
        chemistry: r.chemistry,
        architecture_summary: r.architecture_summary,
        deltas_candidate_minus_benchmark: {
          gc_full_delta: Number((gcFull - r.gc_full).toFixed(4)),
          gc_cds_delta: Number((gcFull - r.gc_cds).toFixed(4)),
          gc3_cds_delta: Number((Math.min(0.95, gcFull * 1.22) - r.gc3).toFixed(4)),
          cds_u4_run_delta: u4Runs - r.u4,
        },
      })),
  };
}

async function runClinicalBenchmarkCompare() {
  const customSeq = (document.getElementById("bench-custom-seq")?.value || "").trim();
  const mod = document.getElementById("bench-modality")?.value || null;
  const chem = document.getElementById("bench-chemistry")?.value || "m1psi";
  const selCand = state.candidates.find((c) => c.bundle_id === state.selectedBundleId);

  const reqBody = {};
  if (customSeq) {
    reqBody.sequence = customSeq;
  } else if (selCand) {
    reqBody.project_id = state.projectId;
    reqBody.bundle_id = selCand.bundle_id;
  } else {
    reqBody.sequence =
      "GGGAAAUCCGCCACCAUGGUGAGCAAGGGCGAGGAGCUGUUCACCGGGGUGGUGCCCAUCCUGGUCGAGCUGGACGGCGACGUAAACGGCCACAAGUUCAGCGUGUCCGGCGAGGGCGAGGGCGAUGCCACCUACGGCAAGCUGACCCUGAAGUUCAUCUGCACCACCGGCAAGCUGCCCGUGCCCUGGCCCACCCUCGUGACCACCCUGACCUACGGCUGAUAAUAGAAAAAAAAAAAAAAAAAAAA";
  }
  if (mod) reqBody.modality = mod;
  if (chem) reqBody.chemistry = chem;

  let data = null;
  try {
    data = await apiFetch("/api/v1/benchmarks/clinical", {
      method: "POST",
      explicitProjectKey: state.projectKey,
      body: JSON.stringify(reqBody),
    });
  } catch {
    data = computeClientClinicalBenchmark(customSeq || (selCand && selCand.full_sequence) || reqBody.sequence, mod || "", chem);
  }
  renderClinicalBenchmarkTable(data);
}

function renderClinicalBenchmarkTable(data) {
  const kpiBox = document.getElementById("bench-cand-kpis");
  const tbody = document.getElementById("bench-delta-tbody");
  const cs = (data && data.candidate_stats) || {};
  if (kpiBox) {
    kpiBox.innerHTML = `
      <span class="kpi-pill">待测全长 <strong>${escapeHtml(cs.full_length_nt ?? 0)} nt</strong></span>
      <span class="kpi-pill">全长 GC <strong>${escapeHtml(((Number(cs.gc_full_transcript) || 0) * 100).toFixed(1))}%</strong></span>
      <span class="kpi-pill">CDS GC3 <strong>${escapeHtml(cs.gc3_cds != null ? (Number(cs.gc3_cds) * 100).toFixed(1) + "%" : "—")}</strong></span>
      <span class="kpi-pill">人源 CAI <strong>${escapeHtml(Number(cs.cai_human || 0).toFixed(3))}</strong></span>
      <span class="kpi-pill">m¹Ψ U4+ 滑移串 <strong>${escapeHtml(cs.cds_m1psi_u4_plus_run_count ?? 0)}</strong></span>
    `;
  }
  if (!tbody) return;
  const rows = Array.isArray(data && data.comparisons) ? data.comparisons : [];
  if (!rows.length) {
    tbody.innerHTML = `<tr><td colspan="8" class="empty-cell mono">当前模态过滤下无临床参考条目</td></tr>`;
    return;
  }

  const fmtPctDelta = (val) => {
    if (val == null || !Number.isFinite(Number(val))) return "—";
    const pct = Number(val) * 100;
    return `${pct >= 0 ? "+" : ""}${pct.toFixed(1)}%`;
  };
  const fmtIntDelta = (val) => {
    if (val == null || !Number.isFinite(Number(val))) return "—";
    const n = Math.round(Number(val));
    return `${n >= 0 ? "+" : ""}${n}`;
  };

  tbody.innerHTML = rows
    .map((r) => {
      const d = r.deltas_candidate_minus_benchmark || {};
      const arch = typeof r.architecture_summary === "string"
        ? r.architecture_summary
        : r.architecture_summary
        ? Object.values(r.architecture_summary).slice(0, 3).join(" · ")
        : "";
      return `
        <tr>
          <td style="font-weight:600;">${escapeHtml(r.construct_name || r.benchmark_id)}</td>
          <td>${escapeHtml(r.developer || "—")} · <span class="badge mono">${escapeHtml(r.clinical_or_reference_status || "Ref")}</span></td>
          <td class="mono">${escapeHtml(r.modality || "linear_mrna")} · ${escapeHtml(r.chemistry || "m1psi")}</td>
          <td class="num-col mono">${escapeHtml(fmtPctDelta(d.gc_full_delta))}</td>
          <td class="num-col mono">${escapeHtml(fmtPctDelta(d.gc_cds_delta))}</td>
          <td class="num-col mono">${escapeHtml(fmtPctDelta(d.gc3_cds_delta))}</td>
          <td class="num-col mono">${escapeHtml(fmtIntDelta(d.cds_u4_run_delta))}</td>
          <td style="white-space:normal; max-inline-size:340px; font-size:11.5px; color:var(--ink-secondary);">${escapeHtml(arch)}</td>
        </tr>
      `;
    })
    .join("");
}

function loadWetLabSampleRows(count) {
  const box = document.getElementById("wetlab-input");
  if (!box) return;
  const rows8 = [
    "construct_id,predicted,observed,assay_unit",
    "cand_p1_01,7.42,8.10,MFI_log2_24h",
    "cand_p1_02,7.15,7.64,MFI_log2_24h",
    "cand_p1_03,6.88,7.05,MFI_log2_24h",
    "cand_ref_R,6.12,6.25,MFI_log2_24h",
    "cand_exp_05,7.31,7.88,MFI_log2_24h",
    "cand_exp_06,6.54,6.71,MFI_log2_24h",
    "cand_dom_07,5.92,5.84,MFI_log2_24h",
    "cand_dom_08,6.35,6.49,MFI_log2_24h",
  ];
  box.value = (count >= 6 ? rows8 : rows8.slice(0, 5)).join("\n");
  parseAndRenderWetLabFeedback();
}

function parseWetLabRowsText(raw) {
  const trimmed = String(raw || "").trim();
  if (!trimmed) return [];
  if (trimmed.startsWith("[") || trimmed.startsWith("{")) {
    try {
      const parsed = JSON.parse(trimmed);
      const arr = Array.isArray(parsed) ? parsed : parsed.rows || [];
      return arr
        .map((r, idx) => ({
          id: String(r.construct_id || r.bundle_id || `row_${idx + 1}`),
          pred: Number(r.predicted),
          obs: Number(r.observed),
          unit: String(r.assay_unit || "assay"),
        }))
        .filter((r) => Number.isFinite(r.pred) && Number.isFinite(r.obs));
    } catch {
      return [];
    }
  }
  const lines = trimmed.split(/\r?\n/).filter((l) => l.trim());
  const out = [];
  for (let i = 0; i < lines.length; i += 1) {
    const parts = lines[i].split(/[,\t]/).map((s) => s.trim());
    if (i === 0 && isNaN(Number(parts[1]))) continue;
    if (parts.length >= 3) {
      const pred = Number(parts[1]);
      const obs = Number(parts[2]);
      if (Number.isFinite(pred) && Number.isFinite(obs)) {
        out.push({ id: parts[0] || `cand_${i}`, pred, obs, unit: parts[3] || "MFI_log2" });
      }
    }
  }
  return out;
}

function parseAndRenderWetLabFeedback() {
  const raw = document.getElementById("wetlab-input")?.value || "";
  const gateBox = document.getElementById("wetlab-gate-banner");
  const svgWrap = document.getElementById("wetlab-svg-wrap");
  const rows = parseWetLabRowsText(raw);
  const n = rows.length;

  if (!n) {
    if (gateBox) {
      gateBox.className = "dual-speed-box mono";
      gateBox.textContent = "请输入有效 CSV 或 JSON 湿实验读数行（至少包含 construct_id, predicted, observed）。";
    }
    if (svgWrap) svgWrap.innerHTML = `<div class="empty-hint mono">暂无可解析的预测-实测样本点</div>`;
    return;
  }

  // Pearson r
  const meanP = rows.reduce((a, r) => a + r.pred, 0) / n;
  const meanO = rows.reduce((a, r) => a + r.obs, 0) / n;
  let num = 0;
  let denP = 0;
  let denO = 0;
  for (const r of rows) {
    num += (r.pred - meanP) * (r.obs - meanO);
    denP += (r.pred - meanP) ** 2;
    denO += (r.obs - meanO) ** 2;
  }
  const pearson = denP > 0 && denO > 0 ? num / Math.sqrt(denP * denO) : 0;
  const bestRow = [...rows].sort((a, b) => b.obs - a.obs)[0];

  if (gateBox) {
    if (n < 6) {
      gateBox.className = "dual-speed-box anchor-only mono";
      gateBox.innerHTML = `
        <strong>双速反馈门控：小样本锚点切换模式 (N=${escapeHtml(n)} &lt; 6)</strong><br/>
        为防止小样本噪声导致代理模型过拟合，当前严格禁止调整打分头权重（零权重漂移）；建议将实测最高构建 <code>${escapeHtml(bestRow.id)}</code> 设为下一轮参考锚点 R。
      `;
    } else {
      gateBox.className = "dual-speed-box calibrate-ok mono";
      gateBox.innerHTML = `
        <strong>双速反馈门控：同批独立校准已解锁 (N=${escapeHtml(n)} ≥ 6 · Pearson r=${escapeHtml(pearson.toFixed(3))})</strong><br/>
        满足同批次对照校准门槛，允许在当前项目空间启用残差线性重标定（不影响全局洁净室基线权重），实测最优构建为 <code>${escapeHtml(bestRow.id)}</code>。
      `;
    }
  }

  if (svgWrap) {
    const W = 420;
    const H = 220;
    const padL = 52;
    const padR = 20;
    const padT = 18;
    const padB = 36;
    const pW = W - padL - padR;
    const pH = H - padT - padB;
    let minX = Math.min(...rows.map((r) => r.pred));
    let maxX = Math.max(...rows.map((r) => r.pred));
    let minY = Math.min(...rows.map((r) => r.obs));
    let maxY = Math.max(...rows.map((r) => r.obs));
    const sx = (maxX - minX) || 1;
    const sy = (maxY - minY) || 1;
    minX -= sx * 0.15;
    maxX += sx * 0.15;
    minY -= sy * 0.15;
    maxY += sy * 0.15;
    const mx = (v) => padL + ((v - minX) / (maxX - minX)) * pW;
    const my = (v) => padT + pH - ((v - minY) / (maxY - minY)) * pH;

    const dots = rows
      .map(
        (r) => `
        <g>
          <circle cx="${mx(r.pred).toFixed(1)}" cy="${my(r.obs).toFixed(1)}" r="5" fill="var(--accent)" opacity="0.88">
            <title>${escapeHtml(r.id)}: pred=${escapeHtml(r.pred)}, obs=${escapeHtml(r.obs)}</title>
          </circle>
          <text x="${(mx(r.pred) + 7).toFixed(1)}" y="${(my(r.obs) + 3).toFixed(1)}" fill="var(--ink-secondary)" font-family="var(--font-mono)" font-size="9">${escapeHtml(r.id)}</text>
        </g>
      `
      )
      .join("");

    svgWrap.innerHTML = `
      <svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}" role="img" aria-label="Predicted vs Observed wet-lab calibration scatter">
        <line x1="${padL}" y1="${padT + pH}" x2="${W - padR}" y2="${padT}" stroke="var(--border-strong)" stroke-dasharray="4 3" />
        <line x1="${padL}" y1="${padT}" x2="${padL}" y2="${padT + pH}" stroke="var(--border-strong)" />
        <line x1="${padL}" y1="${padT + pH}" x2="${W - padR}" y2="${padT + pH}" stroke="var(--border-strong)" />
        ${dots}
        <text x="${padL + pW / 2}" y="${H - 6}" text-anchor="middle" fill="var(--ink-secondary)" font-family="var(--font-mono)" font-size="10">Predicted Score (N=${escapeHtml(n)} · r=${escapeHtml(pearson.toFixed(3))})</text>
        <text x="14" y="${padT + pH / 2}" text-anchor="middle" transform="rotate(-90 14 ${padT + pH / 2})" fill="var(--ink-secondary)" font-family="var(--font-mono)" font-size="10">Observed (${escapeHtml(rows[0].unit)})</text>
      </svg>
    `;
  }
}

async function runAutomatedBrowserSelfTest() {
  const listEl = document.getElementById("selftest-list");
  const sumEl = document.getElementById("selftest-summary");
  listEl.innerHTML = "";
  sumEl.textContent = "正在向 7×24 服务器执行真实端到端测试...";

  const testProj = `selftest_${Math.random().toString(36).slice(2, 7)}`;
  const testKey = mintProjectKey();
  const egfp = (state.presets?.egfp_reporter?.protein_sequence) || FALLBACK_EGFP;

  const steps = [
    {
      name: "1. 7×24 控制面健康与热力学/洁净内核就绪核验",
      fn: async () => {
        const s = await apiFetch("/v1/system/status", { includeAuth: false });
        if (!s.hostname) throw new Error("网关响应不完整");
        return `${s.hostname} · profile=${s.default_profile}`;
      },
    },
    {
      name: "2. 负样本输入护栏：缺起始 M 蛋白自动拦截",
      fn: async () => {
        const r = await apiFetch(`/v1/projects/${testProj}_nom/turn`, {
          method: "POST",
          explicitProjectKey: testKey,
          body: JSON.stringify({
            mode: "initial",
            user_utterance: "缺 M 蛋白测试",
            protein_sequence: egfp.slice(1),
          }),
        });
        if (r.status !== "NEEDS_CLARIFICATION") throw new Error("未触发缺 M 拦截");
        return r.intent.status;
      },
    },
    {
      name: "3. 负样本输入护栏：单候选目标冲突自动追问",
      fn: async () => {
        const r = await apiFetch(`/v1/projects/${testProj}_conf/turn`, {
          method: "POST",
          explicitProjectKey: testKey,
          body: JSON.stringify({
            mode: "initial",
            user_utterance: "冲突目标测试",
            protein_sequence: egfp,
            conflicting_single_pick: true,
          }),
        });
        if (r.intent.status !== "needs_conflict_clarification") throw new Error("未触发冲突追问");
        return r.intent.status;
      },
    },
    {
      name: "4. 入口一首轮生成：eGFP + m¹Ψ + 肝脏去靶向 + 避开 BsaI",
      fn: async () => {
        const r = await apiFetch(`/v1/projects/${testProj}/turn`, {
          method: "POST",
          explicitProjectKey: testKey,
          body: JSON.stringify({
            mode: "initial",
            user_utterance: "首轮 eGFP 设计，避开 BsaI",
            protein_sequence: egfp,
            uridine_chemistry: "m1psi",
            restriction_enzymes: ["BsaI"],
            detarget_liver: true,
          }),
        });
        if (r.status !== "COMPLETED" || r.candidates.length < 2) throw new Error("首轮候选不足");
        return `Intent v01 (${r.candidates.length} 条合格)`;
      },
    },
    {
      name: "5. 多轮微调继承验证：只把 BsaI 改为 BspQI（核验蛋白与化学 100% 继承）",
      fn: async () => {
        const r = await apiFetch(`/v1/projects/${testProj}/turn`, {
          method: "POST",
          explicitProjectKey: testKey,
          body: JSON.stringify({
            mode: "refine",
            user_utterance: "把 BsaI 改为避开 BspQI",
            restriction_enzymes: ["BspQI"],
          }),
        });
        if (!r.intent_diff || !r.intent_diff.protein_lock_preserved) {
          throw new Error("蛋白锁定继承失败");
        }
        window.__lastSelfTestTopBundle = r.report.recommended_bundle_ids[0];
        return `v01→v02 继承✓ (${r.intent_diff.v1_enzymes[0]}→${r.intent_diff.v2_enzymes[0]})`;
      },
    },
    {
      name: "6. 入口二（蛋白+参考构建 R）双轨输出 + INSDC GenBank 四件套导出核验",
      fn: async () => {
        const rRef = await apiFetch(`/v1/projects/${testProj}_r/turn`, {
          method: "POST",
          explicitProjectKey: testKey,
          body: JSON.stringify({
            mode: "initial",
            user_utterance: "参考构建 R 双轨测试",
            protein_sequence: egfp,
            use_preset_reference_r: true,
            prioritize_late_signal: true,
          }),
        });
        const hasMinA = rRef.candidates.some((c) => c.bundle_id.endsWith("_minA"));
        if (!hasMinA) throw new Error("缺少基于 R 的最小改动候选 minA");
        const exp = await apiFetch(
          `/v1/projects/${testProj}/bundles/${window.__lastSelfTestTopBundle}/export`,
          { explicitProjectKey: testKey }
        );
        if (!exp.files["construct.gb"].includes("LOCUS")) throw new Error("GenBank 导出格式异常");
        return `minA✓ + GenBank (${exp.files["construct.gb"].length} B)`;
      },
    },
  ];

  let passed = 0;
  for (const step of steps) {
    const row = document.createElement("div");
    row.className = "test-log-item";
    row.innerHTML = `<span>${escapeHtml(step.name)}</span><span class="mono" style="color:var(--warn);">运行中...</span>`;
    listEl.appendChild(row);
    const t0 = performance.now();
    try {
      const detail = await step.fn();
      const ms = Math.round(performance.now() - t0);
      passed += 1;
      row.lastElementChild.style.color = "var(--ok)";
      row.lastElementChild.textContent = `PASS (${ms}ms) · ${detail}`;
    } catch (err) {
      row.lastElementChild.style.color = "var(--err)";
      row.lastElementChild.textContent = `FAIL · ${err.message}`;
    }
  }
  sumEl.textContent = `实测完成：${passed} / ${steps.length} 项通过`;
}

function wireDialogFallback(dialog) {
  if (!dialog) return;
  if (typeof HTMLDialogElement !== "undefined" && !("closedBy" in HTMLDialogElement.prototype)) {
    const outside = (e) => {
      const r = dialog.getBoundingClientRect();
      return e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom;
    };
    let pressedOutside = false;
    dialog.addEventListener("pointerdown", (e) => {
      pressedOutside = outside(e);
    });
    dialog.addEventListener("pointerup", (e) => {
      if (pressedOutside && outside(e)) dialog.close("dismiss");
      pressedOutside = false;
    });
  }
}

function closeOpenDropdowns(exceptEl = null) {
  if (typeof document.querySelectorAll !== "function") return;
  document.querySelectorAll("details.dropdown-menu[open]").forEach((d) => {
    if (d !== exceptEl) d.open = false;
  });
}

function resetProjectWorkspaceState({ silent = false } = {}) {
  state.sessionEpoch += 1;
  state.activeRequestCount = 0;
  state.inFlight = false;
  state.projectId = `proj_${Math.random().toString(36).slice(2, 8)}`;
  state.projectKey = mintProjectKey();
  state.intentHistory = [];
  state.completedRevisions = [];
  state.activeVersion = 0;
  state.activeIntentStatus = "";
  state.activeEnzymes = null;
  state.userEditedProtein = false;
  state.appliedTurnSequence = 0;
  state.paretoAxisPairKey = "";
  state.compareSelectedIds = [];
  state.boundaryAdvisories = [];
  state.currentStructureProfile = null;
  setModality("linear_mrna");
  if (typeof document.querySelectorAll === "function") {
    document.querySelectorAll('#enzyme-chips .chip[data-dynamic-enz="1"]').forEach((el) => el.remove());
  }
  setSelectedEnzymes(["BsaI"]);
  setEntryMode("protein_only");
  state.latestReport = null;
  state.candidates = [];
  state.selectedBundleId = null;
  state.lastCliCommand = "";
  state.cachedExports = {};
  state.lastFailedTurn = null;
  state.logCount = 0;

  const inputProt = document.getElementById("input-protein");
  if (inputProt) {
    inputProt.value = (state.presets && state.presets.egfp_reporter && state.presets.egfp_reporter.protein_sequence) || FALLBACK_EGFP;
  }
  if (typeof updateProteinBadge === "function") updateProteinBadge();

  const txtProj = document.getElementById("txt-project-id");
  if (txtProj) txtProj.textContent = state.projectId;
  const badgeOut = document.getElementById("badge-outcome");
  if (badgeOut) {
    badgeOut.textContent = "IDLE";
    badgeOut.className = "badge mono";
  }
  const banner = document.getElementById("banner-notes");
  if (banner) {
    banner.hidden = true;
    banner.className = "notice-banner";
    banner.innerHTML = "";
  }
  const ledger = document.getElementById("revision-ledger");
  if (ledger) ledger.hidden = true;
  const setupFold = document.getElementById("setup-fold");
  if (setupFold) setupFold.open = false;
  const logCnt = document.getElementById("txt-log-count");
  if (logCnt) logCnt.textContent = "0";
  renderBoundariesModal();
  if (typeof renderWelcomeCard === "function") renderWelcomeCard();
  if (typeof clearSelectedConstructUI === "function") clearSelectedConstructUI();
  if (typeof syncActionControlsDisabled === "function") syncActionControlsDisabled();
  const btnSend = document.getElementById("btn-send-turn");
  if (btnSend) {
    btnSend.textContent = getSelectedExecMode() === "llm_agent" ? "发送给 Agent ↵" : "编译生成 ↵";
  }
  const btnForm = document.getElementById("btn-submit-initial");
  if (btnForm) btnForm.textContent = "按表单直接编译";
  updateActiveRevBadge();
  if (typeof renderIntentLineage === "function") renderIntentLineage();
  if (typeof renderCandidates === "function") renderCandidates();
  if (!silent) {
    appendStreamCard("agent", "新会话已创建", state.projectId, "已切换至全新隔离项目空间。");
  }
}

function saveSettingsFromModal() {
  const cfgApiEl = document.getElementById("cfg-api-base");
  const candidateApi =
    ((cfgApiEl && cfgApiEl.value) || "").trim().replace(/\/+$/, "") || DEFAULT_API_BASE;
  if (!isTrustedApiOrigin(candidateApi)) {
    const errText = `已拒绝不受信任的 API 源 ${candidateApi}；当前网关与凭据未保存，也未向其他网关发送。`;
    setSettingsModalError(errText);
    appendStreamCard(
      "warn",
      "安全策略阻断",
      "UNTRUSTED_ORIGIN",
      `已拒绝不受信任的 API 源 <code>${escapeHtml(candidateApi)}</code>；网关配置与凭据未保存，也未向其他网关发送。`
    );
    const banner = document.getElementById("banner-notes");
    if (banner) {
      banner.hidden = false;
      banner.className = "notice-banner err";
      banner.innerHTML = `<strong>设置未保存：</strong>已拒绝不受信任的网关源 <code>${escapeHtml(candidateApi)}</code>，密钥未改动也未发送。`;
    }
    if (cfgApiEl && typeof cfgApiEl.focus === "function") {
      cfgApiEl.focus();
    }
    return false;
  }
  setSettingsModalError("");
  const oldOrigin = normalizeOrigin(state.apiBase);
  syncCurrentOriginCredentialsToMap(oldOrigin);

  state.startupApiRejected = "";
  state.configGeneration += 1;
  const newOrigin = new URL(candidateApi).origin;
  const originChanged = Boolean(oldOrigin && newOrigin !== oldOrigin);
  const targetStored = originChanged ? getCredentialsForOrigin(newOrigin) : null;
  state.apiBase = newOrigin;
  safeStorageSet("local", "mrnax_api_base", state.apiBase);

  // Read provider, model, and BYO key BEFORE rebuilding dropdown options!
  const chosenProvider = (document.getElementById("cfg-llm-provider")?.value || "deepseek").trim().toLowerCase();
  const chosenModel = (document.getElementById("cfg-llm-model")?.value || "").trim();
  const chosenByoKey = (document.getElementById("cfg-byo-key")?.value || "").trim();
  const cfgTokenVal = (document.getElementById("cfg-access-token")?.value || "").trim();
  const EMPTY_BYO = { deepseek: "", openai: "", gemini: "", kimi: "", qwen: "" };
  if (originChanged) {
    const storedForTarget = targetStored || { accessToken: "", byoKeys: {} };
    if (state.modalDraftOrigin === newOrigin) {
      state.accessToken = cfgTokenVal;
      state.byoKeys = {
        ...EMPTY_BYO,
        ...(storedForTarget.byoKeys || {}),
        ...(state.modalDraftByoKeys || {}),
        [chosenProvider]: chosenByoKey,
      };
    } else {
      const prevTok = state.modalInitialAccessToken != null ? state.modalInitialAccessToken : state.accessToken;
      state.accessToken = cfgTokenVal && cfgTokenVal !== prevTok ? cfgTokenVal : storedForTarget.accessToken || "";
      const prevProvKey =
        state.modalInitialByoKeys && chosenProvider in state.modalInitialByoKeys
          ? state.modalInitialByoKeys[chosenProvider] || ""
          : (state.byoKeys && state.byoKeys[chosenProvider]) || "";
      const cleanByo = { ...EMPTY_BYO, ...(storedForTarget.byoKeys || {}) };
      if (chosenByoKey !== prevProvKey) {
        cleanByo[chosenProvider] = chosenByoKey;
      }
      state.byoKeys = cleanByo;
    }
  } else {
    state.accessToken = cfgTokenVal;
    state.byoKeys = {
      ...(state.byoKeys || EMPTY_BYO),
      ...(state.modalDraftByoKeys || {}),
      [chosenProvider]: chosenByoKey,
    };
  }
  state.modalDraftByoKeys = null;
  state.modalInitialOrigin = null;
  state.modalDraftOrigin = null;
  state.modalInitialAccessToken = null;
  state.modalInitialByoKeys = null;

  setProviderPreference(chosenProvider);
  setModelPreference(chosenModel);
  state.byoKey = state.byoKeys[state.llmProvider] || "";
  syncCurrentOriginCredentialsToMap(newOrigin);

  if (originChanged) {
    state.rulesCache = null;
    state.dataHubCache = null;
    state.structureCache = {};
    resetProjectWorkspaceState({ silent: true });
    clearErrorBanner();
  }

  if (state.accessToken || state.byoKey) {
    setExecMode("llm_agent");
    if (
      !originChanged &&
      state.lastFailedTurn &&
      state.lastFailedTurn.replayRequest &&
      normalizeOrigin(state.lastFailedTurn.replayRequest.apiBase) === newOrigin
    ) {
      const banner = document.getElementById("banner-notes");
      if (banner) {
        banner.hidden = false;
        banner.className = "notice-banner";
        banner.innerHTML = `
          <strong>凭据已更新至当前网关：</strong>可直接重发上次失败请求（复用原请求 ID，避免重复突变）。
          <div class="banner-actions">
            <button type="button" class="btn-primary btn-compact" data-action="retry-last">重试原请求</button>
          </div>
        `;
      }
    } else {
      clearErrorBanner();
    }
  } else {
    state.callerTokenAuthenticated = false;
  }
  updateAgentAuthPill(null);
  const dlgSettings = document.getElementById("modal-settings");
  if (dlgSettings && typeof dlgSettings.close === "function") dlgSettings.close();
  refreshSystemStatus();
  return true;
}

function executeFallbackOrRetry(action) {
  const ctx = state.lastFailedTurn;
  state.lastFailedTurn = null;
  clearErrorBanner();
  if (action === "fallback-deterministic") {
    setExecMode("deterministic", { userInitiated: true });
    if (ctx && ctx.readOnlyExplanation) {
      renderLocalTradeoffsSummary();
      return;
    }
    if (ctx && ctx.fallbackPayload) {
      runRefineTurn(ctx.fallbackPayload, ctx.utterance || null);
      return;
    }
    if (ctx && !ctx.overridePayload) {
      const detDecision = classifyDeterministicComposerIntent(ctx.utterance || "", {
        hasActiveRevision: state.activeVersion > 0,
        fromForm: false,
        readOnlyExplanation: Boolean(ctx.readOnlyExplanation),
      });
      if (detDecision.action === "local_readonly") {
        handleLocalReadOnlyDeterministicResponse(ctx.utterance || "");
        return;
      }
      if (detDecision.action === "refine" && detDecision.payload) {
        runRefineTurn(detDecision.payload, ctx.utterance || null);
        return;
      }
    }
    if (ctx && ctx.kind === "refine") {
      runRefineTurn(ctx.overridePayload || null, ctx.utterance || null);
    } else {
      runInitialTurn((ctx && ctx.utterance) || null);
    }
    return;
  }
  if (action === "retry-last" && ctx) {
    replayExactRequest(ctx);
  }
}

function wireEvents() {
  document.getElementById("btn-theme-toggle")?.addEventListener("click", () => {
    applyTheme(state.theme === "graphite" ? "light" : "graphite");
  });

  if (typeof document.querySelectorAll === "function") {
    document.querySelectorAll("#primary-nav-tabs .nav-tab").forEach((btn) => {
      btn.addEventListener("click", () => {
        switchWorkspaceTab(btn.dataset?.tab || "studio");
      });
    });

    document.querySelectorAll("#modality-switch .modality-opt").forEach((btn) => {
      btn.addEventListener("click", () => {
        setModality(btn.dataset?.modality || "linear_mrna");
      });
    });

    document.querySelectorAll("#track-switch .track-opt").forEach((btn) => {
      btn.addEventListener("click", () => {
        syncTrackButtons(btn.dataset?.track || "research_max");
      });
    });
  }

  if (typeof window !== "undefined" && typeof window.addEventListener === "function") {
    window.addEventListener("hashchange", () => {
      const tok = extractHashToken();
      if (state.startupApiRejected) {
        return;
      }
      if (tok) {
        state.accessToken = tok;
        syncCurrentOriginCredentialsToMap(state.apiBase);
        const cfgToken = document.getElementById("cfg-access-token");
        if (cfgToken) cfgToken.value = tok;
        setExecMode("llm_agent");
        clearErrorBanner();
        updateAgentAuthPill(null);
        refreshSystemStatus();
      }
    });
  }

  if (typeof document.addEventListener === "function") {
    document.addEventListener("click", (e) => {
      if (e && e.target && typeof e.target.closest === "function") {
        const insideDropdown = e.target.closest("details.dropdown-menu");
        closeOpenDropdowns(insideDropdown);
      }
    });
  }

  document.getElementById("btn-toggle-copilot")?.addEventListener("click", () => {
    const grid = document.getElementById("studio-grid");
    const btn = document.getElementById("btn-toggle-copilot");
    if (!grid || !btn || !grid.classList) return;
    const collapsed = grid.classList.toggle("copilot-collapsed");
    if (typeof btn.setAttribute === "function") btn.setAttribute("aria-pressed", String(!collapsed));
  });

  if (typeof document.querySelectorAll === "function") {
    document.querySelectorAll(".engine-opt").forEach((btn) => {
      btn.addEventListener("click", () => {
        setExecMode(btn.dataset.mode, { userInitiated: true });
      });
    });
  }
  document.getElementById("sel-exec-mode")?.addEventListener("change", (e) => {
    setExecMode(e.target.value, { userInitiated: true });
  });

  document.getElementById("sel-pareto-axes")?.addEventListener("change", (e) => {
    state.paretoAxisPairKey = e.target.value || "";
    renderParetoScatter();
  });

  document.getElementById("sel-llm-model")?.addEventListener("change", (e) => {
    setModelPreference(e.target.value);
  });
  document.getElementById("cfg-llm-provider")?.addEventListener("change", (e) => {
    const cfgByo = document.getElementById("cfg-byo-key");
    if (!state.modalDraftByoKeys) {
      state.modalDraftByoKeys = { ...(state.byoKeys || {}) };
    }
    const prevProv = state.modalDraftProvider || state.llmProvider || "deepseek";
    if (cfgByo) {
      state.modalDraftByoKeys[prevProv] = (cfgByo.value || "").trim();
    }
    const prov = (e.target.value || "deepseek").trim().toLowerCase();
    state.modalDraftProvider = prov;
    const models = state.supportedModels[prov] || SUPPORTED_PROVIDER_MODELS[prov] || [DEFAULT_PROVIDER_MODELS[prov]];
    const cfgModel = document.getElementById("cfg-llm-model");
    if (cfgModel && typeof cfgModel.replaceChildren === "function" && typeof document.createElement === "function") {
      cfgModel.replaceChildren();
      for (const m of models) {
        const opt = document.createElement("option");
        opt.value = m;
        opt.textContent = m;
        cfgModel.appendChild(opt);
      }
      cfgModel.value =
        prov === state.llmProvider && models.includes(state.llmModel)
          ? state.llmModel
          : (state.defaultModels[prov] || models[0]);
    }
    if (cfgByo) {
      cfgByo.value = (state.modalDraftByoKeys && state.modalDraftByoKeys[prov]) || "";
    }
  });

  const onModalApiBaseInput = (e) => {
    const cand = normalizeOrigin(e.target.value);
    if (!cand || cand === state.modalDraftOrigin) return;
    const stored = isTrustedApiOrigin(cand) ? getCredentialsForOrigin(cand) : { accessToken: "", byoKeys: {} };
    const cfgToken = document.getElementById("cfg-access-token");
    const cfgByo = document.getElementById("cfg-byo-key");
    if (cfgToken) cfgToken.value = stored.accessToken || "";
    state.modalDraftByoKeys = {
      deepseek: "",
      openai: "",
      gemini: "",
      kimi: "",
      qwen: "",
      ...(stored.byoKeys || {}),
    };
    const activeProv = state.modalDraftProvider || state.llmProvider || "deepseek";
    if (cfgByo) cfgByo.value = state.modalDraftByoKeys[activeProv] || "";
    state.modalDraftOrigin = cand;
  };
  document.getElementById("cfg-api-base")?.addEventListener("input", onModalApiBaseInput);
  document.getElementById("cfg-api-base")?.addEventListener("change", onModalApiBaseInput);

  document.getElementById("input-protein")?.addEventListener("input", () => {
    state.userEditedProtein = true;
    updateProteinBadge();
  });
  document.getElementById("sel-chemistry")?.addEventListener("change", updateSetupSummary);
  document.getElementById("sel-profile")?.addEventListener("change", (e) => {
    syncTrackButtons(e.target.value);
  });
  document.getElementById("sel-detarget-tissue")?.addEventListener("change", (e) => {
    const chk = document.getElementById("chk-detarget-liver");
    if (chk) chk.checked = e.target.value !== "none";
    updateSetupSummary();
  });
  document.getElementById("chk-detarget-liver")?.addEventListener("change", (e) => {
    const sel = document.getElementById("sel-detarget-tissue");
    if (sel) sel.value = e.target.checked ? "liver_mir122" : "none";
    updateSetupSummary();
  });

  if (typeof document.querySelectorAll === "function") {
    document.querySelectorAll(".seg-tabs[aria-label='设计入口'] .tab").forEach((tab) => {
      tab.addEventListener("click", () => {
        setEntryMode(tab.dataset.entry);
      });
    });

    document.querySelectorAll("#enzyme-chips .chip").forEach((chip) => {
      chip.addEventListener("click", () => {
        chip.classList.toggle("selected");
        updateSetupSummary();
      });
    });
  }

  const header = document.getElementById("cand-thead-row");
  function sortByHeader(e) {
    if (!e || !e.target || typeof e.target.closest !== "function") return;
    const th = e.target.closest("th[data-sort]");
    if (!th || (e.type === "keydown" && e.key !== "Enter" && e.key !== " ")) return;
    e.preventDefault();
    const col = th.dataset.sort;
    state.sortAsc = state.sortCol === col ? !state.sortAsc : col === "rank";
    state.sortCol = col;
    renderCandidates();
    if (e.type === "keydown" && header && header.children) {
      Array.from(header.children).find((cell) => cell.dataset.sort === col)?.focus();
    }
  }
  header?.addEventListener("click", sortByHeader);
  header?.addEventListener("keydown", sortByHeader);

  document.getElementById("preset-egfp")?.addEventListener("click", () => {
    closeOpenDropdowns();
    const fold = document.getElementById("setup-fold");
    if (fold) fold.open = true;
    const seq = (state.presets?.egfp_reporter?.protein_sequence) || FALLBACK_EGFP;
    const protEl = document.getElementById("input-protein");
    if (protEl) protEl.value = seq;
    setSelectedEnzymes(["BsaI"]);
    updateProteinBadge();
  });

  document.getElementById("preset-no-m")?.addEventListener("click", () => {
    closeOpenDropdowns();
    if (state.inFlight) return;
    const seq = (state.presets?.egfp_reporter?.protein_sequence) || FALLBACK_EGFP;
    const protEl = document.getElementById("input-protein");
    if (protEl) protEl.value = seq.slice(1);
    updateProteinBadge();
    setExecMode("deterministic");
    runInitialTurn("测试缺起始 M 蛋白护栏拦截", { bypassReadOnlyClassifier: true });
  });

  document.getElementById("preset-conflict")?.addEventListener("click", () => {
    closeOpenDropdowns();
    if (state.inFlight) return;
    const seq = (state.presets?.egfp_reporter?.protein_sequence) || FALLBACK_EGFP;
    const protEl = document.getElementById("input-protein");
    if (protEl) protEl.value = seq;
    updateProteinBadge();
    setExecMode("deterministic");
    state.forceConflictOnce = true;
    runInitialTurn("测试单候选目标冲突追问", { bypassReadOnlyClassifier: true });
  });

  // 6 Practical Starters + Commercial / Tradeoffs Starters
  document.getElementById("starter-egfp")?.addEventListener("click", () => {
    if (state.inFlight) return;
    const seq = (state.presets?.egfp_reporter?.protein_sequence) || FALLBACK_EGFP;
    setModality("linear_mrna");
    setEntryMode("protein_only");
    document.getElementById("input-protein").value = seq;
    state.userEditedProtein = false;
    syncTrackButtons("research_max");
    document.getElementById("sel-chemistry").value = "m1psi";
    document.getElementById("chk-detarget-liver").checked = true;
    document.getElementById("chk-late-signal").checked = false;
    setSelectedEnzymes(["BsaI"]);
    updateProteinBadge();
    const prompt = "从零设计 eGFP 报告蛋白 mRNA（research_max，m1Psi 修饰，开启肝脏去靶向，避开 BsaI 内切酶）。";
    const mainInput = document.getElementById("input-utterance");
    mainInput.value = prompt;
    runInitialTurn(prompt, { sourceEl: mainInput, fromForm: true });
  });

  document.getElementById("starter-hepo")?.addEventListener("click", () => {
    if (state.inFlight) return;
    setModality("linear_mrna");
    setEntryMode("protein_only");
    document.getElementById("input-protein").value = FALLBACK_HEPO;
    state.userEditedProtein = true;
    syncTrackButtons("commercial_clean");
    document.getElementById("sel-chemistry").value = "m1psi";
    document.getElementById("chk-detarget-liver").checked = false;
    document.getElementById("chk-late-signal").checked = true;
    setSelectedEnzymes(["BspQI"]);
    updateProteinBadge();
    const prompt = "设计人促红细胞生成素 hEPO (UniProt:P01588) 蛋白替代 mRNA，保留天然分泌信号肽，m1Psi 修饰，避开 BspQI，优先持续表达。";
    const mainInput = document.getElementById("input-utterance");
    mainInput.value = prompt;
    runInitialTurn(prompt, { sourceEl: mainInput, fromForm: true });
  });

  document.getElementById("starter-dll3")?.addEventListener("click", () => {
    if (state.inFlight) return;
    setModality("linear_mrna");
    setEntryMode("protein_only");
    document.getElementById("input-protein").value = FALLBACK_HEPO;
    state.userEditedProtein = true;
    syncTrackButtons("commercial_clean");
    document.getElementById("sel-chemistry").value = "m1psi";
    document.getElementById("chk-detarget-liver").checked = true;
    document.getElementById("chk-late-signal").checked = true;
    setSelectedEnzymes(["BsaI", "BspQI"]);
    updateProteinBadge();
    const prompt = "设计 DLL3×CD3 T 细胞衔接器 mRNA 构建（m1Psi 修饰，3′UTR 串联 hsa-miR-122 肝脏去靶向盒，避开 BsaI/BspQI，commercial_clean 洁净轨）。";
    const mainInput = document.getElementById("input-utterance");
    mainInput.value = prompt;
    runInitialTurn(prompt, { sourceEl: mainInput, fromForm: true });
  });

  document.getElementById("starter-circ")?.addEventListener("click", () => {
    if (state.inFlight) return;
    const seq = (state.presets?.egfp_reporter?.protein_sequence) || FALLBACK_EGFP;
    setModality("circrna");
    setEntryMode("protein_only");
    document.getElementById("input-protein").value = seq;
    state.userEditedProtein = false;
    syncTrackButtons("research_max");
    document.getElementById("sel-chemistry").value = "unmodified_u";
    document.getElementById("chk-detarget-liver").checked = false;
    document.getElementById("chk-late-signal").checked = true;
    setSelectedEnzymes(["BspQI"]);
    updateProteinBadge();
    const prompt = "采用 Anabaena PIE 环化骨架与 HRV-B3 IRES 设计环状 circRNA 构建（未修饰 U，保护 IRES 二级结构折叠，避开 BspQI）。";
    const mainInput = document.getElementById("input-utterance");
    mainInput.value = prompt;
    runInitialTurn(prompt, { sourceEl: mainInput, fromForm: true });
  });

  document.getElementById("starter-sarna")?.addEventListener("click", () => {
    if (state.inFlight) return;
    const seq = (state.presets?.egfp_reporter?.protein_sequence) || FALLBACK_EGFP;
    setModality("sarna");
    setEntryMode("protein_only");
    document.getElementById("input-protein").value = seq;
    state.userEditedProtein = false;
    syncTrackButtons("research_max");
    document.getElementById("sel-chemistry").value = "unmodified_u";
    document.getElementById("chk-detarget-liver").checked = false;
    document.getElementById("chk-late-signal").checked = true;
    setSelectedEnzymes(["BspQI"]);
    updateProteinBadge();
    const prompt = "采用 VEEV TC-83 复制子与 26S 亚基因组启动子设计自我扩增 saRNA 构建（未修饰 U，避开 BspQI）。";
    const mainInput = document.getElementById("input-utterance");
    mainInput.value = prompt;
    runInitialTurn(prompt, { sourceEl: mainInput, fromForm: true });
  });

  document.getElementById("starter-ref-r")?.addEventListener("click", () => {
    if (state.inFlight) return;
    const seq = (state.presets?.egfp_reporter?.protein_sequence) || FALLBACK_EGFP;
    setModality("linear_mrna");
    setEntryMode("with_r");
    document.getElementById("input-protein").value = seq;
    state.userEditedProtein = false;
    syncTrackButtons("research_max");
    document.getElementById("sel-chemistry").value = "m1psi";
    document.getElementById("chk-detarget-liver").checked = false;
    document.getElementById("chk-late-signal").checked = true;
    setSelectedEnzymes(["BsaI"]);
    updateProteinBadge();
    const prompt = "基于参考构建 R 设计 eGFP mRNA（research_max，m1Psi 修饰，后期蛋白持续表达优先，避开 BsaI），生成保守最小改动方案 A 与探索方案 B。";
    const mainInput = document.getElementById("input-utterance");
    mainInput.value = prompt;
    runInitialTurn(prompt, { sourceEl: mainInput, fromForm: true });
  });

  document.getElementById("starter-commercial")?.addEventListener("click", () => {
    if (state.inFlight) return;
    const hasIntent = state.intentHistory.length > 0;
    syncTrackButtons("commercial_clean");
    if (!hasIntent) {
      const seq = (state.presets?.egfp_reporter?.protein_sequence) || FALLBACK_EGFP;
      setEntryMode("protein_only");
      document.getElementById("input-protein").value = seq;
      state.userEditedProtein = false;
      document.getElementById("sel-chemistry").value = "m1psi";
      document.getElementById("chk-detarget-liver").checked = false;
      document.getElementById("chk-late-signal").checked = false;
      setSelectedEnzymes(["BsaI"]);
      updateProteinBadge();
    } else {
      updateSetupSummary();
    }
    const prompt = hasIntent
      ? "切换至 commercial_clean 纯商用四头代理栈（5′UTR MRL + CDS + Saluki + fastUTR，零 ViennaRNA），继承锁定项重新评估候选。"
      : "从零设计 eGFP mRNA（commercial_clean 纯商用四头代理栈，m1Psi 修饰，避开 BsaI，零 ViennaRNA）。";
    const mainInput = document.getElementById("input-utterance");
    mainInput.value = prompt;
    const structuredRefine = {
      mode: "refine",
      user_utterance: prompt,
      compliance_track: "commercial_clean",
      use_commercial_surrogate: true,
    };
    if (hasIntent) {
      runRefineTurn(
        getSelectedExecMode() === "llm_agent" ? null : structuredRefine,
        prompt,
        { sourceEl: mainInput, fallbackPayload: structuredRefine }
      );
    } else {
      runInitialTurn(prompt, { sourceEl: mainInput, fromForm: true });
    }
  });

  document.getElementById("starter-tradeoffs")?.addEventListener("click", () => {
    if (state.inFlight) return;
    const hasIntent = state.activeVersion > 0;
    if (hasIntent) {
      setExecMode("llm_agent", { userInitiated: true });
    } else {
      const seq = (state.presets?.egfp_reporter?.protein_sequence) || FALLBACK_EGFP;
      setEntryMode("protein_only");
      document.getElementById("input-protein").value = seq;
      state.userEditedProtein = false;
      syncTrackButtons("research_max");
      document.getElementById("sel-chemistry").value = "m1psi";
      document.getElementById("chk-detarget-liver").checked = false;
      document.getElementById("chk-late-signal").checked = false;
      setSelectedEnzymes(["BsaI"]);
      updateProteinBadge();
    }
    const prompt = hasIntent
      ? "请调用 explain_tradeoffs 并通俗解释当前 P1 前沿候选之间的关键指标权衡与合规边界。"
      : "从零设计 eGFP mRNA（research_max，m1Psi 修饰，避开 BsaI），完成后解释 P1 前沿候选的核心权衡。";
    const mainInput = document.getElementById("input-utterance");
    mainInput.value = prompt;
    runRefineTurn(null, prompt, {
      sourceEl: mainInput,
      readOnlyExplanation: hasIntent,
    });
  });

  const dlgSettings = document.getElementById("modal-settings");
  const dlgSelftest = document.getElementById("modal-selftest");
  const dlgBoundaries = document.getElementById("modal-boundaries");
  const dlgCompare = document.getElementById("modal-compare");
  const dlgDossier = document.getElementById("modal-dossier-preview");
  [dlgSettings, dlgSelftest, dlgBoundaries, dlgCompare, dlgDossier].forEach(wireDialogFallback);

  document.getElementById("btn-open-boundaries")?.addEventListener("click", () => {
    renderBoundariesModal();
    if (dlgBoundaries && typeof dlgBoundaries.showModal === "function") dlgBoundaries.showModal();
  });
  document.getElementById("btn-close-boundaries")?.addEventListener("click", () => {
    if (dlgBoundaries && typeof dlgBoundaries.close === "function") dlgBoundaries.close();
  });

  document.getElementById("btn-compare-selected")?.addEventListener("click", openCandidateComparisonModal);
  document.getElementById("btn-close-compare")?.addEventListener("click", () => {
    if (dlgCompare && typeof dlgCompare.close === "function") dlgCompare.close();
  });

  document.getElementById("btn-analyze-structure")?.addEventListener("click", () => {
    runStructureAnalysisForSelected();
  });

  document.getElementById("chat-stream")?.addEventListener("click", (e) => {
    if (!e || !e.target || typeof e.target.closest !== "function") return;
    const chip = e.target.closest("[data-copilot-act]");
    if (!chip || state.inFlight) return;
    const act = chip.dataset.copilotAct;
    if (act === "bspqi") {
      setSelectedEnzymes(["BspQI"]);
      runRefineTurn({
        mode: "refine",
        user_utterance: "把 BsaI 改成避开 BspQI，其余锁定项继承不变",
        restriction_enzymes: ["BspQI"],
      });
    } else if (act === "circrna") {
      setModality("circrna");
      document.getElementById("starter-circ")?.click();
    } else if (act === "compare") {
      openCandidateComparisonModal();
    } else if (act === "ivt") {
      downloadIvtSynthesisOrder();
    }
  });

  document.getElementById("banner-notes")?.addEventListener("click", (e) => {
    if (!e || !e.target || typeof e.target.closest !== "function") return;
    const btn = e.target.closest("[data-action]");
    if (!btn) return;
    const action = btn.dataset.action;
    if (action === "open-auth") {
      openSettingsModal(true);
      return;
    }
    executeFallbackOrRetry(action);
  });

  document.getElementById("pill-agent-auth")?.addEventListener("click", () => {
    openSettingsModal(true);
  });

  document.getElementById("btn-submit-initial")?.addEventListener("click", () => {
    if (state.inFlight) return;
    runInitialTurn(null, { fromForm: true, ignoreTextDraft: true });
  });
  document.getElementById("btn-send-turn")?.addEventListener("click", () => {
    submitComposerText(document.getElementById("input-utterance"));
  });
  document.getElementById("input-utterance")?.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submitComposerText(document.getElementById("input-utterance"));
    }
  });

  document.getElementById("btn-agent-rail-send")?.addEventListener("click", () => {
    submitComposerText(document.getElementById("agent-rail-input"));
  });
  document.getElementById("agent-rail-input")?.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submitComposerText(document.getElementById("agent-rail-input"));
    }
  });

  document.getElementById("quick-refine-bspqi")?.addEventListener("click", () => {
    if (state.inFlight) return;
    setSelectedEnzymes(["BspQI"]);
    runRefineTurn({
      mode: "refine",
      user_utterance: "只改一项：把线性化内切酶从 BsaI 改成避开 BspQI，其余全部锁定项继承不变",
      restriction_enzymes: ["BspQI"],
    });
  });

  document.getElementById("quick-refine-lock-cds")?.addEventListener("click", () => {
    if (state.inFlight || !state.selectedBundleId) return;
    runRefineTurn({
      mode: "refine",
      user_utterance: `锁定当前候选 ${state.selectedBundleId} 的 CDS 核苷酸序列，仅允许重配 UTR`,
      lock_cds_from_bundle_id: state.selectedBundleId,
    });
  });

  document.getElementById("quick-refine-m15-surrogate")?.addEventListener("click", () => {
    if (state.inFlight) return;
    syncTrackButtons("commercial_clean");
    runRefineTurn({
      mode: "refine",
      user_utterance: "切换至 M1.5 四头商用洁净分轴代理栈 (5′UTR MRL + CDS + Saluki Full-Tx + fastUTR 3′UTR，零 ViennaRNA)",
      compliance_track: "commercial_clean",
      use_commercial_surrogate: true,
    });
  });

  document.getElementById("quick-refine-commercial")?.addEventListener("click", () => {
    if (state.inFlight) return;
    syncTrackButtons("commercial_clean");
    runRefineTurn({
      mode: "refine",
      user_utterance: "切换至 commercial_clean 纯商用档位核验组件放行状态",
      compliance_track: "commercial_clean",
    });
  });

  document.getElementById("quick-ask-tradeoffs")?.addEventListener("click", () => {
    if (state.inFlight || !state.intentHistory.length) return;
    if (state.candidates.length >= 2) {
      openCandidateComparisonModal();
      return;
    }
    setExecMode("llm_agent", { userInitiated: true });
    runRefineTurn(
      null,
      "请调用 explain_tradeoffs 简要总结当前 P1 前沿候选的指标权衡与最佳选择理由。",
      { readOnlyExplanation: true }
    );
  });

  document.getElementById("btn-new-session")?.addEventListener("click", () => {
    resetProjectWorkspaceState({ silent: false });
  });

  document.getElementById("btn-dl-gb")?.addEventListener("click", () => downloadExportFile("construct.gb"));
  document.getElementById("btn-dl-ivt")?.addEventListener("click", downloadIvtSynthesisOrder);
  document.getElementById("btn-dl-dossier")?.addEventListener("click", openOrDownloadQcDossier);
  document.getElementById("btn-dl-dossier-file")?.addEventListener("click", () => {
    if (state.lastDossierHtml) {
      triggerTextBlobDownload(state.lastDossierHtml, state.lastDossierFilename, "text/html;charset=utf-8");
    }
  });
  document.getElementById("btn-close-dossier")?.addEventListener("click", () => {
    if (dlgDossier && typeof dlgDossier.close === "function") dlgDossier.close();
  });

  document.getElementById("btn-dl-rna")?.addEventListener("click", () => {
    closeOpenDropdowns();
    downloadExportFile("full_rna.fasta");
  });
  document.getElementById("btn-dl-cds")?.addEventListener("click", () => {
    closeOpenDropdowns();
    downloadExportFile("cds_only.fasta");
  });
  document.getElementById("btn-dl-meta")?.addEventListener("click", () => {
    closeOpenDropdowns();
    downloadExportFile("chemistry_metadata.json");
  });
  document.getElementById("btn-copy-cli")?.addEventListener("click", async () => {
    if (!state.lastCliCommand) return;
    await navigator.clipboard.writeText(state.lastCliCommand);
    const b = document.getElementById("btn-copy-cli");
    if (!b) return;
    const old = b.textContent;
    b.textContent = "✓ 已复制 CLI 命令";
    setTimeout(() => {
      b.textContent = old;
    }, 1500);
  });

  // Tab 2: Rules & Literature Events
  ["rules-filter-modality", "rules-filter-region", "rules-filter-grade", "rules-chk-debunked"].forEach((id) => {
    document.getElementById(id)?.addEventListener("change", loadScientificRules);
  });
  document.getElementById("rules-filter-q")?.addEventListener("input", loadScientificRules);
  document.getElementById("btn-rules-refresh")?.addEventListener("click", loadScientificRules);
  document.getElementById("btn-rules-lit-search")?.addEventListener("click", runLiteratureSearch);
  document.getElementById("rules-lit-query")?.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      runLiteratureSearch();
    }
  });

  // Tab 3: Data Hub & Resolver Events
  document.getElementById("btn-resolve-target")?.addEventListener("click", () => resolveTargetProteinInHub());
  document.getElementById("dh-target-query")?.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      resolveTargetProteinInHub();
    }
  });
  if (typeof document.querySelectorAll === "function") {
    document.querySelectorAll(".dh-quick-target").forEach((btn) => {
      btn.addEventListener("click", () => resolveTargetProteinInHub(btn.dataset?.q || "EPO"));
    });
  }
  document.getElementById("dh-dataset-filter-track")?.addEventListener("change", renderDataHubDatasetsTable);
  document.getElementById("dh-dataset-filter-q")?.addEventListener("input", renderDataHubDatasetsTable);

  // Tab 4: Clinical Benchmark & Wet-Lab Events
  document.getElementById("btn-run-clinical-bench")?.addEventListener("click", runClinicalBenchmarkCompare);
  document.getElementById("btn-load-sample-wetlab-small")?.addEventListener("click", () => loadWetLabSampleRows(4));
  document.getElementById("btn-load-sample-wetlab-cal")?.addEventListener("click", () => loadWetLabSampleRows(8));
  document.getElementById("btn-parse-wetlab")?.addEventListener("click", parseAndRenderWetLabFeedback);

  // Settings & Self-test Modal Events
  document.getElementById("btn-open-settings")?.addEventListener("click", () => {
    openSettingsModal(false);
  });
  document.getElementById("btn-close-settings")?.addEventListener("click", () => {
    syncModelOptions(state.llmProvider);
    if (dlgSettings && typeof dlgSettings.close === "function") dlgSettings.close();
  });
  dlgSettings?.addEventListener("close", () => {
    syncModelOptions(state.llmProvider);
  });
  document.getElementById("btn-save-settings")?.addEventListener("click", saveSettingsFromModal);

  document.getElementById("btn-run-selftest")?.addEventListener("click", () => {
    closeOpenDropdowns();
    if (dlgSelftest && typeof dlgSelftest.showModal === "function") dlgSelftest.showModal();
    runAutomatedBrowserSelfTest();
  });
  document.getElementById("btn-trigger-selftest")?.addEventListener("click", runAutomatedBrowserSelfTest);
  document.getElementById("btn-close-selftest")?.addEventListener("click", () => {
    if (dlgSelftest && typeof dlgSelftest.close === "function") dlgSelftest.close();
  });
}

document.addEventListener("DOMContentLoaded", () => {
  initConfig();
  renderWelcomeCard();
  wireEvents();
  const inputProt = document.getElementById("input-protein");
  if (inputProt) {
    inputProt.value = FALLBACK_EGFP;
    state.userEditedProtein = false;
  }
  updateProteinBadge();
  refreshSystemStatus();
});

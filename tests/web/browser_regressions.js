async function runBrowserRegressions() {
  const el = (id) => document.getElementById(id);
  const results = [];
  const check = (name, condition) => {
    if (!condition) throw new Error(name);
    results.push(name);
  };
  const response = (body) => new Response(JSON.stringify(body));
  const molecules = ["A", "B"].map((id, i) => ({
    molecule_id: id, length_nt: 3, full_gc_pct: i, full_sequence: i ? "GGG" : "AAA",
    segments: [{ segment_type: "cds", start: 0, end: 3, length_nt: 3, gc_pct: i, origin_source: id }],
  }));
  const measurement = (id, value, unit) => ({
    metric_id: id, value, unit, direction: "maximize",
  });
  const report = (version, secondTarget = "BB") => {
    const candidates = Array.from({ length: 10 }, (_, i) => ({
      bundle_id: "bnd_" + (version * 100 + i).toString(16).padStart(32, "0"),
      version_digest: "0".repeat(64), derivation_action: "fixture", pareto_rank: 1,
      molecules, ...molecules[0],
      metric_details: [measurement("orthrus_halflife_pred", 0, "saluki_pc1_z")],
    }));
    return {
      status: "COMPLETED", candidates,
      intent: {
        intent_id: "v" + version, intent_version: version, status: "ready",
        protein_targets: [{ protein_sequence: "AA" }, { protein_sequence: secondTarget }],
        compliance_track: "commercial_clean", uridine_chemistry: "unmodified",
      },
      report: { recommended_bundle_ids: [candidates[0].bundle_id] },
      cli_replay_command: "fixture",
    };
  };

  localStorage.setItem("mrnax_llm_provider", "anthropic");
  initConfig();
  check("legacy_provider_normalized", el("cfg-llm-provider").value === "deepseek"
    && localStorage.getItem("mrnax_llm_provider") === "deepseek");
  state.byoKey = "fixture-only";
  const requests = [];
  window.fetch = async (url, options) => {
    requests.push({ path: new URL(url).pathname, ...options });
    if (url.endsWith("/export")) return response({ files: { "construct.gb": "fixture" } });
    return response(report(1));
  };
  await runInitialTurn();
  const untouchedClick = HTMLAnchorElement.prototype.click;
  HTMLAnchorElement.prototype.click = () => {};
  try { await downloadExportFile("construct.gb"); }
  finally { HTMLAnchorElement.prototype.click = untouchedClick; }
  await apiFetch("/v1/system/status", { includeAuth: false });
  el("sel-exec-mode").value = "llm_agent";
  await runRefineTurn();
  const byEndpoint = Object.fromEntries(requests.map((req) => [req.path.split("/").pop(), req]));
  check("llm_credentials_scoped_to_agent_chat",
    ["turn", "export", "status"].every((op) => !("X-MRNAX-LLM-Key" in byEndpoint[op].headers)
      && !("X-MRNAX-LLM-Provider" in byEndpoint[op].headers))
    && byEndpoint["agent-chat"].headers["X-MRNAX-LLM-Key"] === "fixture-only"
    && JSON.parse(byEndpoint["agent-chat"].body).provider === "deepseek");

  check("orthrus_only_value_and_unit", el("cand-thead-row").textContent.includes("saluki_pc1_z")
    && el("cand-tbody").rows[0].textContent.includes("0.000"));
  check("all_chains_selectable", !el("chain-picker").hidden && el("inspect-chain").options.length === 2);
  el("inspect-chain").value = "1";
  el("inspect-chain").dispatchEvent(new Event("change"));
  check("chain_specific_inspector", el("seg-inspector").textContent.includes("GGG")
    && el("txt-selected-bundle").textContent.includes(" · B · "));

  const pending = [];
  window.fetch = () => new Promise((resolve) => pending.push(resolve));
  const earlier = runRefineTurn({ mode: "refine" });
  const later = runRefineTurn({ mode: "refine" });
  const newest = report(3, "BC");
  for (const c of newest.candidates) {
    c.metric_details.push(measurement("metric_two", 4, "units"), measurement("missing", null, "counts"));
  }
  pending[1](response(newest));
  await later;
  pending[0](response(report(2)));
  await earlier;
  check("reversed_turns_keep_v03", el("badge-active-rev").textContent === "v03"
    && state.candidates[0].bundle_id === newest.candidates[0].bundle_id);
  check("all_targets_in_lock_ledger", el("ledger-locks-bar").textContent.includes("AA:2+2Δ")
    && el("dual-scale-tracks").children.length === 4);
  check("explicit_missing_metric", el("cand-tbody").rows[0].textContent.includes("N/A"));
  check("mobile_export_toolbar_390", document.documentElement.clientWidth === 390
    && document.documentElement.scrollWidth === 390
    && el("btn-dl-gb").title.includes(newest.candidates[0].bundle_id));

  for (const c of newest.candidates) {
    const node = document.querySelector(`.pareto-node[aria-label="${c.bundle_id}"]`);
    const mark = node.querySelector("circle[fill]:not([fill=none])");
    mark.scrollIntoView({block: "center"});
    const rect = mark.getBoundingClientRect();
    const hit = document.elementFromPoint(rect.x + rect.width / 2, rect.y + rect.height / 2);
    if (hit?.closest(".pareto-node") !== node) throw new Error("marker center missed " + c.bundle_id);
    hit.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    if (state.selectedBundleId !== c.bundle_id) throw new Error("wrong candidate selected");
  }
  check("ten_coincident_centers_select_exact_candidates", true);

  const clarification = report(4);
  clarification.status = "NEEDS_CLARIFICATION";
  delete clarification.candidates;
  delete clarification.report;
  window.fetch = async () => response(clarification);
  await runRefineTurn({ mode: "refine" });
  check("clarification_invalidates_export", !state.candidates.length && !state.selectedBundleId
    && el("btn-dl-gb").disabled && el("btn-copy-cli").disabled && el("chain-picker").hidden);

  const completedSameRev4 = report(4);
  window.fetch = async () => response(completedSameRev4);
  await runRefineTurn({ mode: "refine" });
  check("first_seen_empty_revision_later_completed_enters_ledger",
    state.completedRevisions.some((r) => r.version === 4) && !el("revision-ledger").hidden);

  // I6 reopened: delayed no-intent /agent-chat reply after active revision must preserve candidates & selection
  const savedCount = state.candidates.length;
  const savedSel = state.selectedBundleId;
  window.fetch = async () => response({ status: "COMPLETED", assistant_reply: "informational", intent: null, candidates: [] });
  el("sel-exec-mode").value = "llm_agent";
  await runRefineTurn();
  check("delayed_no_intent_chat_preserves_candidates_and_selection",
    state.activeVersion === 4 && state.candidates.length === savedCount && state.selectedBundleId === savedSel);

  // Astra Minor 1 + Important 13: 96-char unbroken error detail & collapsed desktop rail at 390px
  el("studio-grid")?.classList.add("copilot-collapsed");
  const err96 = new Error("Upstream_" + "X".repeat(96));
  err96.status = 502;
  surfaceTurnError(err96, { kind: "refine", utterance: "retry" });
  const railVisibleAt390 = el("agent-rail") && getComputedStyle(el("agent-rail")).display !== "none";
  check("long_error_banner_wraps_and_rail_visible_at_390",
    document.documentElement.scrollWidth === 390 && railVisibleAt390);
  el("studio-grid")?.classList.remove("copilot-collapsed");
  clearErrorBanner();

  // Settings model persistence (deepseek-v4-pro survives Save) & provider-key isolation in dialog
  window.fetch = async () => response({ providers: { deepseek: { server_key_configured: true } } });
  el("btn-open-settings").click();
  el("cfg-llm-provider").value = "deepseek";
  el("cfg-llm-provider").dispatchEvent(new Event("change"));
  el("cfg-byo-key").value = "sk-ds-draft";
  el("cfg-llm-provider").value = "openai";
  el("cfg-llm-provider").dispatchEvent(new Event("change"));
  const openaiCleared = el("cfg-byo-key").value === "";
  el("cfg-byo-key").value = "sk-openai-draft";
  el("cfg-llm-provider").value = "deepseek";
  el("cfg-llm-provider").dispatchEvent(new Event("change"));
  const deepseekRestored = el("cfg-byo-key").value === "sk-ds-draft";
  el("cfg-llm-model").value = "deepseek-v4-pro";
  el("cfg-llm-model").dispatchEvent(new Event("change"));
  el("btn-save-settings").click();
  el("btn-open-settings").click();
  check("settings_model_persistence_and_provider_key_isolation",
    openaiCleared && deepseekRestored
    && state.llmProvider === "deepseek"
    && state.llmModel === "deepseek-v4-pro"
    && localStorage.getItem("mrnax_llm_model") === "deepseek-v4-pro"
    && el("sel-llm-model").value === "deepseek-v4-pro"
    && el("cfg-llm-model").value === "deepseek-v4-pro"
    && state.byoKeys.deepseek === "sk-ds-draft"
    && state.byoKeys.openai === "sk-openai-draft"
    && state.byoKey === "sk-ds-draft");
  el("btn-close-settings").click();

  // Settings Cancel must NOT leave #sel-llm-model out of sync with saved model
  el("btn-open-settings").click();
  el("cfg-llm-model").value = "deepseek-flash";
  el("cfg-llm-model").dispatchEvent(new Event("change"));
  el("btn-close-settings").click();
  check("settings_cancel_keeps_main_model_selector_synced",
    state.llmModel === "deepseek-v4-pro" && el("sel-llm-model").value === "deepseek-v4-pro");

  // Agent constraint sync: intent constraints update enzyme chips, summary & inspector markers
  const syncResp = report(5);
  syncResp.intent.constraints = [
    { constraint_kind: "type_iis_restriction_free", params: { enzymes: ["BspQI"] } },
    { constraint_kind: "mirna_detarget", params: { sites: ["miR-122"] } },
  ];
  syncResp.intent.entry_mode = "protein_only";
  window.fetch = async () => response(syncResp);
  el("sel-exec-mode").value = "llm_agent";
  await runRefineTurn(null, "改为 BspQI");
  const nextTurnPayloads = [];
  window.fetch = async (_u, opts) => {
    nextTurnPayloads.push(JSON.parse(opts.body));
    return response(syncResp);
  };
  el("sel-exec-mode").value = "deterministic";
  await runRefineTurn();
  check("agent_intent_syncs_enzyme_chips_and_inspector",
    el("setup-collapsed-summary").textContent.includes("BspQI")
    && el("enzyme-clearance-rail").textContent.includes("BspQI [禁]")
    && JSON.stringify(nextTurnPayloads[0].restriction_enzymes) === '["BspQI"]');

  // Starter chips reset stale entry_mode & direct form compile forwards Agent Bio Spec constraints
  const starterReqs = [];
  window.fetch = async (_u, opts) => {
    starterReqs.push(JSON.parse(opts.body));
    return response(report(6));
  };
  state.intentHistory = [];
  state.activeVersion = 0;
  el("sel-exec-mode").value = "deterministic";
  el("starter-ref-r").click();
  await new Promise((r) => setTimeout(r, 15));
  el("starter-egfp").click();
  await new Promise((r) => setTimeout(r, 15));
  el("sel-exec-mode").value = "llm_agent";
  el("sel-profile").value = "commercial_clean";
  el("chk-late-signal").checked = true;
  el("btn-submit-initial").click();
  await new Promise((r) => setTimeout(r, 15));
  check("starters_reset_stale_controls_and_form_compile_forwards_settings_to_agent",
    starterReqs.length >= 3
    && starterReqs[0].use_preset_reference_r === true
    && starterReqs[1].use_preset_reference_r === false
    && starterReqs[2].message.includes("compliance_track=commercial_clean")
    && starterReqs[2].message.includes("prioritize_late_signal=true"));

  const _prevFetchR3 = window.fetch;
  const r3Reqs = [];
  let r3IntentVersion = Math.max(10, (Number(state.activeVersion) || 0) + 1);
  try {
    window.fetch = async (url, opts = {}) => {
      const urlStr = String(url || "");
      let parsedBody = null;
      if (opts && opts.body) {
        try { parsedBody = JSON.parse(opts.body); } catch (_) {}
      }
      r3Reqs.push({ url: urlStr, method: opts.method || "GET", body: parsedBody });
      if (urlStr.endsWith("/agent-chat")) {
        const msg = parsedBody?.message || "";
        const avoidsXba = msg.includes("XbaI");
        r3IntentVersion = Math.max(r3IntentVersion + 1, (Number(state.activeVersion) || 0) + 1);
        return {
          ok: true,
          status: 200,
          async text() {
            return JSON.stringify({
              status: "COMPLETED",
              assistant_reply: avoidsXba ? "Added XbaI clearance." : "Handled chat.",
              intent: {
                intent_id: `i_r3_${r3IntentVersion}`,
                intent_version: r3IntentVersion,
                status: "ready",
                compliance_track: "research_max",
                uridine_chemistry: "m1psi",
                constraints: [
                  {
                    constraint_kind: "type_iis_restriction_free",
                    params: { enzymes: avoidsXba ? ["XbaI"] : ["BsaI", "BsmBI", "SapI"] },
                  },
                ],
                restriction_enzymes: avoidsXba ? ["XbaI"] : ["BsaI", "BsmBI", "SapI"],
                locked_scopes: {},
              },
              report: { outcome_status: "COMPLETED", intent_version: r3IntentVersion },
              candidates: [],
            });
          },
        };
      }
      if (urlStr.endsWith("/turn")) {
        r3IntentVersion = Math.max(r3IntentVersion + 1, (Number(state.activeVersion) || 0) + 1);
        return {
          ok: true,
          status: 200,
          async text() {
            return JSON.stringify({
              outcome_status: "COMPLETED",
              intent: {
                intent_id: `i_r3_turn_${r3IntentVersion}`,
                intent_version: r3IntentVersion,
                status: "ready",
                compliance_track: parsedBody?.compliance_track || "research_max",
                uridine_chemistry: "m1psi",
                constraints: [
                  {
                    constraint_kind: "type_iis_restriction_free",
                    params: { enzymes: parsedBody?.restriction_enzymes || ["XbaI"] },
                  },
                ],
                restriction_enzymes: parsedBody?.restriction_enzymes || ["XbaI"],
                locked_scopes: {},
              },
              report: { outcome_status: "COMPLETED", intent_version: r3IntentVersion },
              candidates: [],
            });
          },
        };
      }
      return { ok: true, status: 200, async text() { return "{}"; } };
    };

    // 1. btn-submit-initial keeps unsent #input-utterance draft untouched and does not consume it
    const uttEl = document.getElementById("input-utterance");
    const execModeEl = document.getElementById("sel-exec-mode");
    const submitInitEl = document.getElementById("btn-submit-initial");
    const protEl = document.getElementById("input-protein");
    uttEl.value = "Unsent draft note for agent";
    execModeEl.value = "deterministic";
    protEl.value = "MVSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITLGMDELYK";
    submitInitEl.click();
    await new Promise((r) => setTimeout(r, 25));
    const initTurnReq = r3Reqs.find((r) => r.url.endsWith("/turn"));
    check("btn_submit_initial_preserves_unsent_utterance_draft",
      uttEl.value === "Unsent draft note for agent"
      && Boolean(initTurnReq)
      && initTurnReq.body.user_utterance !== "Unsent draft note for agent");

    // 2. Fresh project (btn-new-session) untouched prefilled eGFP is not attached, and custom protein is reset on New Session
    document.getElementById("btn-new-session").click();
    await new Promise((r) => setTimeout(r, 15));
    protEl.value = "MCUSTOMTARGETFROMPROJECTONE";
    protEl.dispatchEvent(new Event("input"));
    document.getElementById("btn-new-session").click();
    await new Promise((r) => setTimeout(r, 15));
    const resetClearedCustom = !protEl.value.includes("MCUSTOMTARGETFROMPROJECTONE");
    const wasFreshBeforeSend = state.activeVersion === 0;
    execModeEl.value = "llm_agent";
    uttEl.value = "Design mCherry mRNA";
    r3Reqs.length = 0;
    document.getElementById("btn-send-turn").click();
    await new Promise((r) => setTimeout(r, 25));
    const mcherryReq = r3Reqs.find((r) => r.url.endsWith("/agent-chat"));
    check("untouched_prefilled_egfp_not_attached_to_unrelated_prompt",
      resetClearedCustom
      && wasFreshBeforeSend
      && Boolean(mcherryReq)
      && !mcherryReq.body.protein_sequence
      && !mcherryReq.body.message.includes("MCUSTOMTARGETFROMPROJECTONE")
      && !mcherryReq.body.message.includes("MVSKGEELFT"));

    // 3. Non-preset enzyme XbaI intent sync creates chip, shows in summary/rail, and forwards on next turn
    r3Reqs.length = 0;
    uttEl.value = "Avoid XbaI";
    document.getElementById("btn-send-turn").click();
    await new Promise((r) => setTimeout(r, 25));
    const chipsWrap = document.getElementById("enzyme-chips");
    const xbaChip = chipsWrap ? chipsWrap.querySelector('[data-enz="XbaI"]') : null;
    renderEnzymeClearance("AUGCCCUAA");
    const setupSummaryText = document.getElementById("setup-collapsed-summary")?.textContent || "";
    const railText = document.getElementById("enzyme-clearance-rail")?.textContent || "";
    execModeEl.value = "deterministic";
    r3Reqs.length = 0;
    uttEl.value = "Switch to commercial_clean";
    document.getElementById("btn-send-turn").click();
    await new Promise((r) => setTimeout(r, 25));
    const nextTurnReq = r3Reqs.find((r) => r.url.endsWith("/turn"));
    check("dynamic_xbai_enzyme_chip_sync_summary_and_forwarding",
      Boolean(xbaChip)
      && setupSummaryText.includes("XbaI")
      && railText.includes("XbaI")
      && railText.includes("server-gated")
      && Boolean(nextTurnReq)
      && Array.isArray(nextTurnReq.body.restriction_enzymes)
      && nextTurnReq.body.restriction_enzymes.includes("XbaI"));

    // 4. Untrusted gateway URL displays inline #cfg-settings-error inside modal
    document.getElementById("btn-open-settings").click();
    document.getElementById("cfg-api-base").value = "https://untrusted-evil.example";
    document.getElementById("btn-save-settings").click();
    const errEl = document.getElementById("cfg-settings-error");
    const modalEl = document.getElementById("modal-settings");
    check("settings_modal_shows_inline_error_on_untrusted_url",
      Boolean(errEl) && !errEl.hidden && errEl.textContent.trim().length > 0 && Boolean(modalEl?.open));
    document.getElementById("btn-close-settings").click();
    clearErrorBanner();

    // 5. Wave 3 De-AI-ification, 2D Structure Arc & Sliding Tracks SVG, Side-by-Side Compare, IVT & QC Dossier
    const wave3Seq =
      "GGGAAAUCCGCCACCAUGGUGAGCAAGGGCGAGGAGCUGUUCACCGGGGUGGUGCCCAUCCUGGUCGAGCUGGACGGCGACGUAAACGGCCACAAGUUCAGCGUGUCCGGCGAGGGCGAGGGCGAUGCCACCUACGGCAAGCUGACCCUGAAGUUCAUCUGCACCACCGGCAAGCUGCCCGUGCCCUGGCCCUGAUAAUAGACACUCCAAAAAAAAAAAAA";
    const wave3Ver = Math.max(20, (Number(state.activeVersion) || 0) + 1);
    const wave3Candidates = [0, 1].map((idx) => ({
      bundle_id: `bnd_wave3_${idx}_` + "a".repeat(24),
      version_digest: (idx ? "b" : "c").repeat(64),
      display_label: idx === 0 ? "Pareto #1 · LinearDesign + 5' Polished Candidate" : "Conservative Refinement (Minimal Edit)",
      strategy_tag: idx === 0 ? "lineardesign_5prime_polished" : "conservative_refinement",
      short_digest: (idx ? "b" : "c").repeat(12),
      gate_passed_count: 6,
      gate_total_count: 6,
      fto_badge: "sequence_screen_clear",
      derivation_action: idx === 0 ? "seed_init" : "locked_refinement",
      pareto_rank: 1,
      length_nt: wave3Seq.length,
      full_gc_pct: 58.2 + idx,
      uridine_chemistry: "m1psi",
      full_sequence: wave3Seq,
      segments: [
        { segment_type: "five_prime_utr", start: 0, end: 15, length_nt: 15, gc_pct: 53.3, license_track: "commercial_clean", action_taken: "selected" },
        { segment_type: "cds", start: 15, end: 192, length_nt: 177, gc_pct: 61.0, license_track: "commercial_clean", action_taken: "optimized" },
        { segment_type: "stop_cassette", start: 192, end: 201, length_nt: 9, gc_pct: 11.1, license_track: "commercial_clean", action_taken: "verified" },
        { segment_type: "three_prime_utr", start: 201, end: 209, length_nt: 8, gc_pct: 50.0, license_track: "commercial_clean", action_taken: "detargeted" },
        { segment_type: "poly_a", start: 209, end: wave3Seq.length, length_nt: wave3Seq.length - 209, gc_pct: 0.0, license_track: "commercial_clean", action_taken: "verified" },
      ],
      metric_details: [
        { metric_id: "kozak_unpaired_prob", value: 0.78 - idx * 0.05, unit: "prob", direction: "maximize" },
        { metric_id: "cds_mean_window_mfe", value: -42.5 + idx * 3.1, unit: "kcal/mol", direction: "minimize" },
      ],
    }));

    window.fetch = async (url, _opts = {}) => {
      const u = String(url);
      if (u.includes("/api/v1/structure/analyze")) {
        return response({
          length_nt: wave3Seq.length,
          modality: "linear_mrna",
          full_gc_pct: 58.2,
          full_u_pct: 16.4,
          cai: 0.892,
          mfe_kcal_mol: -84.6,
          kozak_unpaired_prob: 0.812,
          aup_mean: 0.44,
          initiation_window: {
            start: 0,
            end: 60,
            aug_index_in_window: 15,
            sequence: wave3Seq.slice(0, 60),
            dot_bracket: "...............(((....)))...................................",
            arcs: [{ i: 15, j: 24, prob: 0.72 }, { i: 16, j: 23, prob: 0.54 }],
          },
          sliding_tracks: {
            window_nt: 60,
            step_nt: 20,
            positions: [0, 20, 40, 60, 80, 100],
            local_mfe: [-18.2, -24.5, -27.1, -25.0, -21.4, -12.0],
            local_gc_pct: [52, 61, 64, 60, 55, 32],
            local_u_pct: [18, 14, 12, 15, 19, 10],
            local_unpaired_prob: [0.76, 0.42, 0.38, 0.44, 0.51, 0.82],
          },
          motifs: [{ kind: "mirna_site", label: "hsa-miR-122-5p site", start: 201, end: 209, severity: "info" }],
          engines_used: ["ViennaRNA"],
        });
      }
      if (u.includes("/ivt-order")) {
        return response({ ivt_synthesis_order_csv: "bundle_id,template_dna_5to3\nbnd_wave3_0,ATG\n" });
      }
      if (u.includes("/dossier")) {
        return response({ qc_dossier_html: "<!doctype html><html><body>QC PASS</body></html>" });
      }
      if (u.includes("/api/v1/rules")) {
        return response({
          total_available: 2,
          total_matched: 2,
          rules: FALLBACK_SCIENTIFIC_RULES.slice(0, 2),
        });
      }
      if (u.includes("/api/v1/data-hub")) {
        return response(FALLBACK_DATA_HUB);
      }
      if (u.includes("/api/v1/evidence/resolve")) {
        return response({
          protein: {
            matched: true,
            gene_symbol: "EPO",
            uniprot_accession: "P01588",
            protein_name_sandboxed: "Erythropoietin",
            protein_sequence: FALLBACK_HEPO,
            aa_length: FALLBACK_HEPO.length,
            starts_with_met: true,
            has_signal_peptide: true,
            signal_peptide_hint: "Signal peptide [1..27]",
          },
          literature_and_trials: { papers: [], clinical_trials: [] },
        });
      }
      if (u.includes("/api/v1/benchmarks/clinical")) {
        return response(computeClientClinicalBenchmark(wave3Seq, "linear_mrna", "m1psi"));
      }
      return response({});
    };

    applyTurnResult(
      {
        status: "COMPLETED",
        assistant_reply: "已完成 **eGFP** 多目标帕累托设计。",
        tool_calls: [{ name: "design_constructs", arguments: { uridine_chemistry: "m1psi" } }],
        intent: {
          intent_id: `i_w3_${wave3Ver}`,
          intent_version: wave3Ver,
          status: "ready",
          compliance_track: "research_max",
          uridine_chemistry: "m1psi",
          locked_scopes: {},
        },
        report: {
          outcome_status: "COMPLETED_WITH_CANDIDATES",
          recommended_bundle_ids: [wave3Candidates[0].bundle_id],
          pre_synthesis_advisories: [{ category: "crs_safety_layer_boundary", message: "CRS note" }],
        },
        candidates: wave3Candidates,
      },
      "设计 eGFP",
      ++state.turnSequence
    );

    await runStructureAnalysisForSelected();
    check("wave3_de_ai_studio_and_2d_structure_svgs",
      el("banner-notes").hidden === true
      && !el("cand-tbody").rows[0].textContent.includes("aaaaaaaaaaaa")
      && el("cand-tbody").rows[0].textContent.includes("LinearDesign")
      && Boolean(document.querySelector("#structure-arc-svg-wrap svg path"))
      && Boolean(document.querySelector("#structure-track-svg-wrap svg polyline"))
      && el("structure-source-badge").textContent.includes("ViennaRNA"));

    // Side-by-side compare modal + IVT & QC Dossier actions
    el("btn-compare-selected").click();
    const cmpOpen = Boolean(el("modal-compare")?.open) && el("compare-body").textContent.includes("序列碱基位点差异");
    el("btn-close-compare").click();
    const origAnchorClick = HTMLAnchorElement.prototype.click;
    HTMLAnchorElement.prototype.click = () => {};
    try {
      await downloadIvtSynthesisOrder();
      await openOrDownloadQcDossier();
    } finally {
      HTMLAnchorElement.prototype.click = origAnchorClick;
    }
    const dossierLoaded = Boolean(el("modal-dossier-preview")?.open) && el("dossier-iframe").srcdoc.includes("QC PASS");
    el("btn-close-dossier").click();
    check("wave3_compare_ivt_and_qc_dossier", cmpOpen && dossierLoaded);

    // 6. All 4 Tabs (Rules, Data Hub + Import to Studio, Benchmarks + Wet-Lab) & 390px zero horizontal overflow
    el("tab-btn-rules").click();
    await new Promise((r) => setTimeout(r, 20));
    const rulesWidthOk = document.documentElement.scrollWidth === 390;
    const rulesRendered = el("rules-list").children.length >= 2
      && Array.from(el("rules-list").querySelectorAll("a.ext-link")).every((a) => a.rel.includes("noopener"));

    el("tab-btn-data").click();
    await new Promise((r) => setTimeout(r, 20));
    await resolveTargetProteinInHub("EPO");
    const dataWidthOk = document.documentElement.scrollWidth === 390;
    const importBtn = document.getElementById("btn-import-resolved-protein");
    importBtn.click();
    const importedToStudio = !el("view-studio").hidden && el("input-protein").value === FALLBACK_HEPO;

    el("tab-btn-benchmarks").click();
    await new Promise((r) => setTimeout(r, 20));
    el("btn-load-sample-wetlab-cal").click();
    const benchWidthOk = document.documentElement.scrollWidth === 390;
    const benchRendered = el("bench-delta-tbody").rows.length >= 1
      && Boolean(document.querySelector("#wetlab-svg-wrap svg"));

    el("tab-btn-studio").click();
    const studioWidthOk = document.documentElement.scrollWidth === 390;
    check("wave3_four_tabs_and_390_zero_overflow",
      rulesRendered && importedToStudio && benchRendered
      && rulesWidthOk && dataWidthOk && benchWidthOk && studioWidthOk);
  } finally {
    window.fetch = _prevFetchR3;
  }

  return results;
}

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
  } finally {
    window.fetch = _prevFetchR3;
  }

  return results;
}

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
  return results;
}

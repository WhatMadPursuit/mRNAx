const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const source = fs.readFileSync(path.resolve(__dirname, "../../web/app.js"), "utf8");

function app() {
  const fields = new Map();
  const document = {
    addEventListener() {},
    getElementById(id) {
      if (!fields.has(id)) fields.set(id, {});
      return fields.get(id);
    },
  };
  const location = new URL("http://localhost");
  const window = { location, crypto: require("node:crypto").webcrypto };
  const run = (code) => vm.runInContext(code, context);
  const context = vm.createContext({ window, document, URL });
  run(source);
  return { run, document };
}

test("deterministic and export requests omit provider headers", () => {
  const { run } = app();
  run('state.byoKey = "fixture"; state.apiBase = "http://localhost"');
  for (const [method, path] of [["GET", "bundles/id/export"], ["POST", "turn"], ["GET", "agent-chat"]]) {
    const headers = run(`buildHeaders("/v1/projects/p/${path}", "${method}", null, true)`);
    assert.ok(headers["X-MRNAX-Project-Key"]);
    assert.equal(headers["X-MRNAX-LLM-Key"], undefined);
    assert.equal(headers["X-MRNAX-LLM-Provider"], undefined);
  }
  assert.equal(run('buildHeaders("/v1/projects/p/agent-chat", "POST", null, true)["X-MRNAX-LLM-Key"]'), "fixture");
});

test("arbitrary measured axes retain zero, direction, units and explicit missing state", () => {
  const { run } = app();
  run(`const candidate = {metric_details:[
    {metric_id:"orthrus_halflife_pred", value:0, unit:"saluki_pc1_z", direction:"maximize"},
    {metric_id:"extra", value:null, unit:"seconds", direction:"minimize"}
  ]}; const axes = resolveMetricAxes([candidate]);`);
  assert.equal(run("axes.length"), 2);
  assert.match(run("axes[0].label"), /saluki_pc1_z/);
  assert.equal(run("readAxisValue(candidate, axes[0])"), 0);
  assert.equal(run("readAxisValue(candidate, axes[1])"), null);
  assert.match(run("buildSparkCell(null, axes[1], 0, 1)"), /N\/A/);
});

test("older revisions and earlier replies for the same revision cannot replace current state", () => {
  const { run } = app();
  run("state.activeVersion = 3; state.appliedTurnSequence = 5");
  assert.equal(run('applyTurnResult({intent:{intent_version:2}}, "", 7)'), false);
  assert.equal(run('applyTurnResult({intent:{intent_version:3}}, "", 4)'), false);
  assert.equal(run("state.activeVersion"), 3);
  assert.equal(run("state.appliedTurnSequence"), 5);
});

test("formatSafeMarkdown escapes XSS vectors while formatting bold, code, and bullets", () => {
  const { run } = app();
  const html = run(`formatSafeMarkdown("推荐 **v02** 使用 \`BspQI\`\\n- <script>alert(1)</script>\\n- <img src=x onerror=alert(2)>")`);
  assert.doesNotMatch(html, /<script|<img/i);
  assert.match(html, /&lt;script&gt;alert\(1\)&lt;\/script&gt;/);
  assert.match(html, /<strong>v02<\/strong>/);
  assert.match(html, /<code>BspQI<\/code>/);
  assert.match(html, /<ul><li>.*<\/li><li>.*<\/li><\/ul>/);
});

test("hash #token=... unlocks agent mode and strips token from address bar without storing in localStorage", () => {
  const fields = new Map();
  const document = {
    addEventListener() {},
    documentElement: { dataset: {} },
    querySelectorAll() { return []; },
    getElementById(id) {
      if (!fields.has(id)) fields.set(id, {});
      return fields.get(id);
    },
  };
  let replacedUrl = null;
  const history = { replaceState(_s, _t, u) { replacedUrl = u; } };
  const location = new URL("https://whatmadpursuit.github.io/mRNAx/#token=tok_hash_secret_123");
  const window = { location, crypto: require("node:crypto").webcrypto };
  const context = vm.createContext({ window, document, URL, history });
  vm.runInContext(source, context);
  vm.runInContext("initConfig()", context);

  assert.equal(vm.runInContext("state.accessToken", context), "tok_hash_secret_123");
  assert.equal(vm.runInContext("state.llmModel", context), "deepseek-flash");
  assert.equal(fields.get("sel-exec-mode").value, "llm_agent");
  assert.equal(replacedUrl, "/mRNAx/");
  const chatHeaders = vm.runInContext('buildHeaders("/v1/projects/p/agent-chat", "POST", null, true)', context);
  assert.equal(chatHeaders["X-MRNAX-Access-Token"], "tok_hash_secret_123");
});

test("surfaceTurnError unhides banner-notes with actionable 401 controls", () => {
  const { run, document } = app();
  run(`
    const err = new Error("Privileged LLM Agent mode requires either X-MRNAX-Access-Token or your own X-MRNAX-LLM-Key.");
    err.status = 401;
    surfaceTurnError(err, { kind: "initial", utterance: "design eGFP" });
  `);
  const banner = document.getElementById("banner-notes");
  assert.equal(banner.hidden, false);
  assert.match(banner.className, /\berr\b/);
  assert.match(banner.innerHTML, /data-action="open-auth"/);
  assert.match(banner.innerHTML, /data-action="fallback-deterministic"/);
  assert.equal(document.getElementById("badge-outcome").textContent, "AUTH REQUIRED");

  // Gateway 502 mentioning upstream 401 must NOT trigger local 401 Access Token banner
  run(`
    const err502 = new Error("Upstream LLM HTTP 401");
    err502.status = 502;
    surfaceTurnError(err502, { kind: "initial", utterance: "design eGFP" });
  `);
  assert.equal(document.getElementById("badge-outcome").textContent, "ERROR");
  assert.doesNotMatch(banner.innerHTML, /fallback-deterministic/);
});

test("provider switch isolates per-provider BYO keys and apiFetch rejects redirects", async () => {
  const fields = new Map();
  const document = {
    addEventListener() {},
    documentElement: { dataset: {} },
    querySelectorAll() { return []; },
    getElementById(id) {
      if (!fields.has(id)) fields.set(id, {});
      return fields.get(id);
    },
  };
  let seenFetchOpts = null;
  const fetchMock = async (_url, opts) => {
    seenFetchOpts = opts;
    return { ok: true, status: 200, async text() { return '{"ok":true}'; } };
  };
  for (const badHash of ["#token=%ZZbad", "#token=%E0%A4%A"]) {
    const badLoc = new URL(`https://whatmadpursuit.github.io/mRNAx/${badHash}`);
    let badReplaced = null;
    const badHist = { replaceState(_s, _t, u) { badReplaced = u; } };
    const badCtx = vm.createContext({
      window: { location: badLoc, crypto: require("node:crypto").webcrypto },
      document, URL, history: badHist, fetch: fetchMock,
    });
    vm.runInContext(source, badCtx);
    // Malformed percent-encoding must strip fragment first without throwing URIError
    vm.runInContext("initConfig()", badCtx);
    assert.equal(badReplaced, "/mRNAx/");
    assert.equal(vm.runInContext("state.accessToken", badCtx), "");
  }

  const location = new URL("https://whatmadpursuit.github.io/mRNAx/#token=tok_check");
  const history = { replaceState() {} };
  const window = { location, crypto: require("node:crypto").webcrypto };
  const context = vm.createContext({ window, document, URL, history, fetch: fetchMock });
  vm.runInContext(source, context);
  vm.runInContext("initConfig()", context);

  vm.runInContext('setProviderPreference("deepseek"); state.byoKeys.deepseek = "sk-deepseek-only"; state.byoKey = "sk-deepseek-only";', context);
  vm.runInContext('setProviderPreference("openai");', context);
  assert.equal(vm.runInContext("state.byoKey", context), "");
  assert.equal(vm.runInContext('buildHeaders("/v1/projects/p/agent-chat", "POST", null, true)["X-MRNAX-LLM-Key"]', context), undefined);

  await vm.runInContext('apiFetch("/v1/system/status", { includeAuth: true })', context);
  assert.equal(seenFetchOpts.redirect, "error");
});

test("appliedTurnSequence stays monotonic and non-ready intent preserves clarification state", () => {
  const { run, document } = app();
  run("state.appliedTurnSequence = 5; state.activeVersion = 0;");
  assert.equal(run('applyTurnResult({intent:{intent_version:2, status:"ready", compliance_track:"commercial_clean", uridine_chemistry:"m1psi", locked_scopes:{}}, status:"COMPLETED", candidates:[]}, "v2", 2)'), true);
  assert.equal(run("state.appliedTurnSequence"), 5);
  assert.equal(run('applyTurnResult({intent:null, status:"COMPLETED", assistant_reply:"late"}, "late", 3)'), false);

  assert.equal(run(`applyTurnResult({
    intent: {
      intent_id: "intent_v3",
      intent_version: 3,
      status: "needs_conflict_clarification",
      compliance_track: "commercial_clean",
      uridine_chemistry: "m1psi",
      locked_scopes: {},
      blocking_questions: ["Keep only one constraint?"]
    },
    status: "COMPLETED",
    candidates: [{ bundle_id: "bnd_ignored" }]
  }, "clarify", 6)`), true);
  assert.equal(run("state.activeIntentStatus"), "needs_conflict_clarification");
  assert.equal(run("state.candidates.length"), 0);
  assert.equal(document.getElementById("badge-outcome").textContent, "needs_conflict_clarification");
});

test("saveSettingsFromModal preserves chosen deepseek-v4-pro model and openSettingsModal rehydrates it", () => {
  const fields = new Map();
  const makeSelect = (opts) => ({
    options: opts.map((value) => ({ value })),
    value: opts[0],
    children: [],
    replaceChildren() { this.children = []; },
    appendChild(child) {
      this.children.push(child);
      this.options = this.children.map((c) => ({ value: c.value }));
    },
  });
  fields.set("cfg-llm-provider", makeSelect(["deepseek", "openai", "kimi", "qwen", "gemini"]));
  fields.set("cfg-llm-model", makeSelect(["deepseek-flash", "deepseek-v4-pro"]));
  fields.set("sel-llm-model", makeSelect(["deepseek-flash", "deepseek-v4-pro"]));

  const document = {
    addEventListener() {},
    documentElement: { dataset: {} },
    querySelectorAll() { return []; },
    createElement(tag) { return { tagName: tag.toUpperCase(), value: "", textContent: "" }; },
    getElementById(id) {
      if (!fields.has(id)) fields.set(id, {});
      return fields.get(id);
    },
  };
  const fetchMock = async () => ({ ok: false, status: 503, async text() { return "offline"; } });
  const context = vm.createContext({
    window: { location: new URL("http://localhost"), crypto: require("node:crypto").webcrypto },
    document,
    URL,
    fetch: fetchMock,
  });
  vm.runInContext(source, context);

  document.getElementById("cfg-api-base").value = "https://149-248-39-200.sslip.io:8443";
  document.getElementById("cfg-llm-provider").value = "deepseek";
  document.getElementById("cfg-llm-model").value = "deepseek-v4-pro";
  document.getElementById("cfg-access-token").value = "tok_pro_save";
  document.getElementById("cfg-byo-key").value = "sk-deepseek-saved-fixture";

  vm.runInContext("saveSettingsFromModal()", context);
  assert.equal(vm.runInContext("state.llmProvider", context), "deepseek");
  assert.equal(vm.runInContext("state.llmModel", context), "deepseek-v4-pro");
  assert.equal(vm.runInContext("state.byoKeys.deepseek", context), "sk-deepseek-saved-fixture");
  assert.equal(fields.get("sel-llm-model").value, "deepseek-v4-pro");
  assert.equal(fields.get("cfg-llm-model").value, "deepseek-v4-pro");

  // Re-opening modal populates matching provider, model, token, and per-provider BYO key
  fields.get("cfg-llm-model").value = "deepseek-flash";
  fields.get("cfg-byo-key").value = "";
  vm.runInContext("openSettingsModal()", context);
  assert.equal(fields.get("cfg-llm-provider").value, "deepseek");
  assert.equal(fields.get("cfg-llm-model").value, "deepseek-v4-pro");
  assert.equal(fields.get("cfg-access-token").value, "tok_pro_save");
  assert.equal(fields.get("cfg-byo-key").value, "sk-deepseek-saved-fixture");
});

test("executeFallbackOrRetry handles readOnlyExplanation without mutating state and clearSubmittedFieldIfUnchanged checks epoch", () => {
  const fields = new Map();
  const appended = [];
  const chatStream = {
    children: [],
    scrollTop: 0,
    scrollHeight: 120,
    querySelector() { return null; },
    appendChild(el) { appended.push(el); this.children.push(el); },
  };
  fields.set("chat-stream", chatStream);

  const document = {
    addEventListener() {},
    documentElement: { dataset: {} },
    querySelectorAll() { return []; },
    createElement(tag) { return { tagName: tag.toUpperCase(), className: "", innerHTML: "" }; },
    getElementById(id) {
      if (!fields.has(id)) fields.set(id, {});
      return fields.get(id);
    },
  };
  let fetchCalled = false;
  const fetchMock = async () => {
    fetchCalled = true;
    return { ok: true, status: 200, async text() { return "{}"; } };
  };
  const context = vm.createContext({
    window: { location: new URL("http://localhost"), crypto: require("node:crypto").webcrypto },
    document,
    URL,
    fetch: fetchMock,
  });
  vm.runInContext(source, context);

  vm.runInContext(`
    state.activeVersion = 2;
    state.candidates = [{
      bundle_id: "bnd_p1_01",
      derivation_action: "seed_init",
      pareto_rank: 1,
      metrics: { mfe_dg: -180.5 },
      metric_details: [{ metric_id: "mfe_dg", value: -180.5, unit: "kcal/mol", direction: "minimize" }],
    }];
    state.lastFailedTurn = {
      kind: "initial",
      utterance: "比较 v02 前沿候选之间的权衡",
      readOnlyExplanation: true,
      fallbackPayload: null,
    };
    executeFallbackOrRetry("fallback-deterministic");
  `, context);

  assert.equal(fetchCalled, false);
  assert.equal(fields.get("sel-exec-mode").value, "deterministic");
  assert.equal(appended.length, 1);
  assert.match(appended[0].innerHTML, /bnd_p1_01/);
  assert.match(appended[0].innerHTML, /本地只读权衡分析/);

  // clearSubmittedFieldIfUnchanged only clears when value matches and epoch unchanged
  vm.runInContext('state.sessionEpoch = 4;', context);
  fields.set("hero-box", { value: "设计 eGFP mRNA" });
  vm.runInContext('clearSubmittedFieldIfUnchanged(document.getElementById("hero-box"), "设计 eGFP mRNA", 3);', context);
  assert.equal(fields.get("hero-box").value, "设计 eGFP mRNA");
  vm.runInContext('clearSubmittedFieldIfUnchanged(document.getElementById("hero-box"), "已改字", 4);', context);
  assert.equal(fields.get("hero-box").value, "设计 eGFP mRNA");
  vm.runInContext('clearSubmittedFieldIfUnchanged(document.getElementById("hero-box"), "设计 eGFP mRNA", 4);', context);
  assert.equal(fields.get("hero-box").value, "");
});

test("untrusted gateway URL aborts save without touching saved keys or fetching status", () => {
  const fields = new Map();
  const makeSelect = (opts) => ({
    options: opts.map((value) => ({ value })),
    value: opts[0],
    children: [],
    replaceChildren() { this.children = []; },
    appendChild(child) {
      this.children.push(child);
      this.options = this.children.map((c) => ({ value: c.value }));
    },
  });
  fields.set("cfg-llm-provider", makeSelect(["deepseek", "openai"]));
  fields.set("cfg-llm-model", makeSelect(["deepseek-flash", "deepseek-v4-pro"]));
  fields.set("sel-llm-model", makeSelect(["deepseek-flash", "deepseek-v4-pro"]));
  fields.set("chat-stream", {
    children: [], scrollTop: 0, scrollHeight: 0, querySelector() { return null; },
    appendChild(el) { this.children.push(el); },
  });

  let fetchCount = 0;
  let closed = false;
  fields.set("modal-settings", { close() { closed = true; } });
  const document = {
    addEventListener() {},
    documentElement: { dataset: {} },
    querySelectorAll() { return []; },
    createElement(tag) { return { tagName: tag.toUpperCase(), className: "", innerHTML: "", value: "", textContent: "" }; },
    getElementById(id) {
      if (!fields.has(id)) fields.set(id, {});
      return fields.get(id);
    },
  };
  const context = vm.createContext({
    window: { location: new URL("http://localhost"), crypto: require("node:crypto").webcrypto },
    document,
    URL,
    fetch: async () => { fetchCount += 1; return { ok: true, status: 200, async text() { return "{}"; } }; },
  });
  vm.runInContext(source, context);
  vm.runInContext('state.apiBase = "https://149-248-39-200.sslip.io:8443"; state.accessToken = "tok_old"; state.llmModel = "deepseek-flash";', context);

  document.getElementById("cfg-api-base").value = "https://evil-gateway.example";
  document.getElementById("cfg-access-token").value = "tok_for_evil_only";
  document.getElementById("cfg-llm-provider").value = "openai";
  document.getElementById("cfg-byo-key").value = "sk-evil-only";

  const saved = vm.runInContext("saveSettingsFromModal()", context);
  assert.equal(saved, false);
  assert.equal(fetchCount, 0);
  assert.equal(closed, false);
  assert.equal(vm.runInContext("state.apiBase", context), "https://149-248-39-200.sslip.io:8443");
  assert.equal(vm.runInContext("state.accessToken", context), "tok_old");
  assert.equal(vm.runInContext("state.byoKeys.openai", context), "");
  assert.equal(document.getElementById("banner-notes").hidden, false);
});

test("exact retry preserves original endpoint/provider/model/id, read-only 401 fallback does not mutate, and empty rail preserves main draft", async () => {
  const fields = new Map();
  const chatStream = {
    children: [], scrollTop: 0, scrollHeight: 0, querySelector() { return null; },
    appendChild(el) { this.children.push(el); },
  };
  fields.set("chat-stream", chatStream);
  const requests = [];
  let failOnce = true;
  const fetchMock = async (url, opts) => {
    requests.push({ url, method: opts.method, headers: opts.headers, body: opts.body ? JSON.parse(opts.body) : null });
    if (failOnce) {
      failOnce = false;
      return { ok: false, status: 502, async text() { return '{"detail":"transient upstream failure"}'; } };
    }
    return {
      ok: true,
      status: 200,
      async text() {
        return JSON.stringify({
          status: "COMPLETED",
          assistant_reply: "Recovered.",
          intent: { intent_id: "i1", intent_version: 1, status: "ready", compliance_track: "research_max", uridine_chemistry: "m1psi", locked_scopes: {} },
          report: { outcome_status: "COMPLETED" },
          candidates: [],
        });
      },
    };
  };
  const document = {
    addEventListener() {},
    documentElement: { dataset: {} },
    querySelectorAll() { return []; },
    createElement(tag) { return { tagName: tag.toUpperCase(), className: "", innerHTML: "" }; },
    getElementById(id) {
      if (!fields.has(id)) fields.set(id, {});
      return fields.get(id);
    },
  };
  const context = vm.createContext({
    window: { location: new URL("http://localhost"), crypto: require("node:crypto").webcrypto },
    document,
    URL,
    fetch: fetchMock,
  });
  vm.runInContext(source, context);

  // Start in Agent mode on OpenAI deepseek/openai isolation
  vm.runInContext(`
    state.apiBase = "https://149-248-39-200.sslip.io:8443";
    state.llmProvider = "openai";
    state.llmModel = "gpt-4.1-mini";
    state.byoKeys.openai = "sk-openai-retry";
    state.byoKeys.deepseek = "sk-deepseek-other";
    state.byoKey = "sk-openai-retry";
  `, context);
  document.getElementById("sel-exec-mode").value = "llm_agent";
  await vm.runInContext('runInitialTurn("Design eGFP via OpenAI")', context);
  assert.equal(requests.length, 1);
  const originalReqId = requests[0].body.client_request_id;
  assert.ok(originalReqId);

  // User flips controls to deterministic & deepseek-v4-pro BEFORE clicking Retry -> Retry must send exact original request!
  document.getElementById("sel-exec-mode").value = "deterministic";
  vm.runInContext('state.llmProvider = "deepseek"; state.llmModel = "deepseek-v4-pro"; state.byoKey = "sk-deepseek-other";', context);
  vm.runInContext('executeFallbackOrRetry("retry-last")', context);
  await new Promise((r) => setTimeout(r, 20));
  assert.equal(requests.length, 2);
  assert.match(requests[1].url, /\/agent-chat$/);
  assert.equal(requests[1].body.provider, "openai");
  assert.equal(requests[1].body.model, "gpt-4.1-mini");
  assert.equal(requests[1].body.client_request_id, originalReqId);
  assert.equal(requests[1].headers["X-MRNAX-LLM-Provider"], "openai");
  assert.equal(requests[1].headers["X-MRNAX-LLM-Key"], "sk-openai-retry");

  // Typed English read-only question on existing project must NOT POST /turn on 401 deterministic fallback
  const reqCountBefore = requests.length;
  vm.runInContext(`
    state.activeVersion = 2;
    state.candidates = [{ bundle_id: "bnd_ro", derivation_action: "fixture", pareto_rank: 1, metrics: { mfe_dg: -100 }, metric_details: [{ metric_id: "mfe_dg", value: -100, unit: "kcal/mol" }] }];
    state.lastFailedTurn = { kind: "refine", utterance: "Explain the current results", fallbackPayload: null, readOnlyExplanation: false };
    executeFallbackOrRetry("fallback-deterministic");
  `, context);
  assert.equal(requests.length, reqCountBefore);
  assert.match(chatStream.children[chatStream.children.length - 1].innerHTML, /bnd_ro/);

  // Empty rail submit leaves main #input-utterance draft untouched and makes zero requests
  document.getElementById("input-utterance").value = "unsent main draft";
  document.getElementById("agent-rail-input").value = "   ";
  await vm.runInContext('submitComposerText(document.getElementById("agent-rail-input"))', context);
  assert.equal(requests.length, reqCountBefore);
  assert.equal(document.getElementById("input-utterance").value, "unsent main draft");
});

test("HTTP 200 HTML response throws and empty no-intent reply does not advance appliedTurnSequence", async () => {
  const fields = new Map();
  const document = {
    addEventListener() {},
    documentElement: { dataset: {} },
    querySelectorAll() { return []; },
    getElementById(id) {
      if (!fields.has(id)) fields.set(id, {});
      return fields.get(id);
    },
  };
  const context = vm.createContext({
    window: { location: new URL("http://localhost"), crypto: require("node:crypto").webcrypto },
    document,
    URL,
    fetch: async () => ({ ok: true, status: 200, async text() { return "<html>proxy login</html>"; } }),
  });
  vm.runInContext(source, context);
  await assert.rejects(
    vm.runInContext('apiFetch("/v1/projects/p/agent-chat", { method: "POST", body: "{}" })', context),
    /non-JSON HTTP 200/
  );
  vm.runInContext("state.appliedTurnSequence = 2;", context);
  assert.equal(vm.runInContext('applyTurnResult({ detail: "<html></html>" }, "hi", 5)', context), false);
  assert.equal(vm.runInContext("state.appliedTurnSequence", context), 2);
});

test("startup untrusted ?api= rejects #token= import, strips fragment, shows banner, and skips status fetch", async () => {
  const fields = new Map();
  let fetchCount = 0;
  let replacedUrl = "";
  const document = {
    addEventListener() {},
    documentElement: { dataset: {} },
    querySelectorAll() { return []; },
    createElement(tag) { return { tagName: tag.toUpperCase(), className: "", innerHTML: "", value: "", textContent: "" }; },
    getElementById(id) {
      if (!fields.has(id)) fields.set(id, { hidden: true, textContent: "", value: "", children: [], appendChild(c) { this.children.push(c); }, replaceChildren() { this.children = []; } });
      return fields.get(id);
    },
  };
  const winLoc = new URL("https://whatmadpursuit.github.io/mRNAx/?api=https://evil.example#token=tok_secret");
  const historyObj = {
    replaceState(_s, _t, u) {
      replacedUrl = String(u);
      const parsed = new URL(replacedUrl, winLoc.href);
      winLoc.href = parsed.href;
      winLoc.search = parsed.search;
      winLoc.hash = parsed.hash;
    },
  };
  const context = vm.createContext({
    window: {
      location: winLoc,
      history: historyObj,
      crypto: require("node:crypto").webcrypto,
    },
    history: historyObj,
    document,
    URL,
    fetch: async () => { fetchCount += 1; return { ok: true, status: 200, async text() { return "{}"; } }; },
  });
  vm.runInContext(source, context);
  vm.runInContext("initConfig()", context);
  await vm.runInContext("refreshSystemStatus()", context);

  assert.equal(vm.runInContext("state.startupApiRejected", context), "https://evil.example");
  assert.equal(vm.runInContext("state.accessToken", context), "");
  assert.equal(fetchCount, 0);
  assert.equal(replacedUrl.includes("tok_secret"), false);
  assert.equal(winLoc.hash, "");
  assert.equal(document.getElementById("banner-notes").hidden, false);
});

test("turn retry pins original apiBase and reuses X-MRNAX-Idempotency-Key across gateway switch", async () => {
  const fields = new Map();
  fields.set("chat-stream", { children: [], scrollTop: 0, scrollHeight: 0, querySelector() { return null; }, appendChild(c) { this.children.push(c); } });
  const requests = [];
  let failFirst = true;
  const fetchMock = async (url, opts) => {
    requests.push({ url, method: opts.method, headers: opts.headers, body: opts.body ? JSON.parse(opts.body) : null });
    if (failFirst) {
      failFirst = false;
      return { ok: false, status: 502, async text() { return '{"detail":"transient gateway error"}'; } };
    }
    return {
      ok: true,
      status: 200,
      async text() {
        return JSON.stringify({
          outcome_status: "COMPLETED",
          intent: { intent_id: "i1", intent_version: 1, status: "ready", compliance_track: "research_max", uridine_chemistry: "m1psi", locked_scopes: {} },
          report: { outcome_status: "COMPLETED" },
          candidates: [],
        });
      },
    };
  };
  const document = {
    addEventListener() {},
    documentElement: { dataset: {} },
    querySelectorAll() { return []; },
    createElement(tag) { return { tagName: tag.toUpperCase(), className: "", innerHTML: "", value: "", textContent: "" }; },
    getElementById(id) {
      if (!fields.has(id)) fields.set(id, { hidden: true, textContent: "", value: "" });
      return fields.get(id);
    },
  };
  const context = vm.createContext({
    window: { location: new URL("http://localhost"), crypto: require("node:crypto").webcrypto },
    document,
    URL,
    fetch: fetchMock,
  });
  vm.runInContext(source, context);
  vm.runInContext('state.apiBase = "https://149-248-39-200.sslip.io:8443"; state.accessToken = "tok_gw_a";', context);
  document.getElementById("sel-exec-mode").value = "deterministic";
  await vm.runInContext('runInitialTurn("Design eGFP")', context);

  assert.equal(requests.length, 1);
  assert.ok(requests[0].url.startsWith("https://149-248-39-200.sslip.io:8443/"));
  const idemKey = requests[0].headers["X-MRNAX-Idempotency-Key"];
  assert.ok(idemKey && idemKey.length > 0);

  // Switch gateway to localhost before clicking Retry
  vm.runInContext('state.apiBase = "http://localhost:8443";', context);
  vm.runInContext('executeFallbackOrRetry("retry-last")', context);
  await new Promise((r) => setTimeout(r, 25));

  assert.equal(requests.length, 2);
  assert.ok(requests[1].url.startsWith("https://149-248-39-200.sslip.io:8443/"));
  assert.equal(requests[1].headers["X-MRNAX-Idempotency-Key"], idemKey);
});

test("read-only prompt stays non-mutating and deterministic refine parses commercial_clean + preserves open modal model", async () => {
  const fields = new Map();
  const makeSelect = (opts) => ({
    options: opts.map((value) => ({ value })),
    value: opts[0],
    children: [],
    replaceChildren() { this.children = []; },
    appendChild(child) {
      this.children.push(child);
      this.options = this.children.map((c) => ({ value: c.value }));
    },
  });
  fields.set("cfg-llm-provider", makeSelect(["deepseek", "openai"]));
  fields.set("cfg-llm-model", makeSelect(["deepseek-flash", "deepseek-v4-pro"]));
  fields.set("sel-llm-model", makeSelect(["deepseek-flash", "deepseek-v4-pro"]));
  fields.set("modal-settings", { open: true });

  const document = {
    addEventListener() {},
    documentElement: { dataset: {} },
    querySelectorAll() { return []; },
    createElement(tag) { return { tagName: tag.toUpperCase(), className: "", innerHTML: "", value: "", textContent: "" }; },
    getElementById(id) {
      if (!fields.has(id)) fields.set(id, { hidden: true, textContent: "", value: "" });
      return fields.get(id);
    },
  };
  const context = vm.createContext({
    window: { location: new URL("http://localhost"), crypto: require("node:crypto").webcrypto },
    document,
    URL,
    fetch: async () => ({
      ok: true,
      status: 200,
      async text() {
        return JSON.stringify({
          presets: {},
          agent_providers: {
            deepseek: { default_model: "deepseek-flash", supported_models: ["deepseek-flash", "deepseek-v4-pro"] },
          },
        });
      },
    }),
  });
  vm.runInContext(source, context);

  assert.equal(vm.runInContext('hasExplicitMutationIntent("Explain the current design.")', context), false);
  const parsed = vm.runInContext('JSON.stringify(parseDeterministicRefineFallback("Switch to commercial_clean"))', context);
  const parsedObj = JSON.parse(parsed);
  assert.equal(parsedObj.mode, "refine");
  assert.equal(parsedObj.compliance_track, "commercial_clean");
  assert.equal(parsedObj.use_commercial_surrogate, true);

  // Delayed refreshSystemStatus while settings modal is open must not overwrite user's cfg-llm-model choice
  document.getElementById("cfg-llm-model").value = "deepseek-v4-pro";
  vm.runInContext('state.apiBase = "http://localhost:8443";', context);
  await vm.runInContext("refreshSystemStatus()", context);
  assert.equal(document.getElementById("cfg-llm-model").value, "deepseek-v4-pro");
});

test("origin-scoped retry never leaks Gateway B creds to A, same-gateway repair keeps retry, startup ?api= blocks hashchange, and status checks origin", async () => {
  const fields = new Map();
  fields.set("chat-stream", { children: [], scrollTop: 0, scrollHeight: 0, querySelector() { return null; }, appendChild(c) { this.children.push(c); }, addEventListener() {} });
  const makeSelect = (opts) => ({
    options: opts.map((value) => ({ value })),
    value: opts[0],
    children: [],
    addEventListener() {},
    replaceChildren() { this.children = []; },
    appendChild(c) { this.children.push(c); this.options = this.children.map((x) => ({ value: x.value })); },
  });
  fields.set("cfg-llm-provider", makeSelect(["deepseek", "openai"]));
  fields.set("cfg-llm-model", makeSelect(["deepseek-flash", "deepseek-v4-pro", "gpt-4.1-mini"]));
  fields.set("sel-llm-model", makeSelect(["deepseek-flash", "deepseek-v4-pro", "gpt-4.1-mini"]));
  fields.set("sel-exec-mode", { value: "llm_agent", addEventListener() {} });
  fields.set("modal-settings", { open: false, close() { this.open = false; }, addEventListener() {} });

  const listeners = new Map();
  const requests = [];
  let failOnce = true;
  let statusGateResolve = null;

  const fetchMock = async (url, opts = {}) => {
    const urlStr = String(url);
    requests.push({ url: urlStr, headers: { ...(opts.headers || {}) }, body: opts.body ? JSON.parse(opts.body) : null });
    if (urlStr.endsWith("/v1/system/status")) {
      if (urlStr.startsWith("https://149-248-39-200.sslip.io:8443/")) {
        return await new Promise((resolve) => { statusGateResolve = resolve; });
      }
      return {
        ok: true,
        status: 200,
        async text() {
          return JSON.stringify({ hostname: "local-gw-B", default_profile: "research_max", caller_token_authenticated: true });
        },
      };
    }
    if (urlStr.endsWith("/v1/presets")) {
      return { ok: true, status: 200, async text() { return "{}"; } };
    }
    if (failOnce) {
      failOnce = false;
      return { ok: false, status: 502, async text() { return '{"detail":"transient A error"}'; } };
    }
    return {
      ok: true,
      status: 200,
      async text() {
        return JSON.stringify({ status: "COMPLETED", assistant_reply: "Retried cleanly.", intent: null, report: null, candidates: [] });
      },
    };
  };

  const document = {
    addEventListener() {},
    documentElement: { dataset: {} },
    querySelectorAll() { return []; },
    createElement(tag) { return { tagName: tag.toUpperCase(), className: "", innerHTML: "", value: "", textContent: "", dataset: {}, children: [] }; },
    getElementById(id) {
      if (!fields.has(id)) {
        fields.set(id, {
          hidden: true,
          textContent: "",
          innerHTML: "",
          value: "",
          className: "",
          children: [],
          listeners: {},
          addEventListener(ev, fn) { this.listeners[ev] = fn; },
          appendChild(c) { this.children.push(c); },
          replaceChildren() { this.children = []; },
          focus() {},
        });
      }
      return fields.get(id);
    },
  };
  const winLoc = new URL("http://localhost/");
  const historyObj = {
    replaceState(_s, _t, u) {
      const parsed = new URL(String(u), winLoc.href);
      winLoc.href = parsed.href;
      winLoc.search = parsed.search;
      winLoc.hash = parsed.hash;
    },
  };
  const context = vm.createContext({
    window: {
      location: winLoc,
      history: historyObj,
      addEventListener(ev, fn) { listeners.set(ev, fn); },
      crypto: require("node:crypto").webcrypto,
    },
    history: historyObj,
    document,
    URL,
    fetch: fetchMock,
  });
  vm.runInContext(source, context);
  vm.runInContext("wireEvents();", context);
  vm.runInContext(`
    state.apiBase = "https://149-248-39-200.sslip.io:8443";
    state.accessToken = "tok_A_only";
    state.llmProvider = "deepseek";
    state.llmModel = "deepseek-flash";
    state.byoKeys.deepseek = "sk-key-A-only";
    state.byoKeys.openai = "sk-openai-A-inactive";
    state.byoKey = "sk-key-A-only";
  `, context);

  // Open Settings modal on Gateway A, then change #cfg-api-base to fresh Gateway B via input event -> clears draft A creds!
  vm.runInContext("openSettingsModal(false)", context);
  assert.equal(document.getElementById("cfg-access-token").value, "tok_A_only");
  document.getElementById("cfg-api-base").value = "http://localhost:8443";
  if (document.getElementById("cfg-api-base").listeners?.input) {
    document.getElementById("cfg-api-base").listeners.input({ target: document.getElementById("cfg-api-base") });
  }
  assert.equal(document.getElementById("cfg-access-token").value, "");
  assert.equal(document.getElementById("cfg-byo-key").value, "");

  // Programmatic switch without input event AND replacing visible fields STILL must NOT carry inactive openai key from A to B!
  requests.length = 0;
  vm.runInContext(`
    state.apiBase = "https://149-248-39-200.sslip.io:8443";
    state.accessToken = "tok_A_only";
    state.byoKeys.deepseek = "sk-key-A-only";
    state.byoKeys.openai = "sk-openai-A-inactive";
    state.byoKey = "sk-key-A-only";
    openSettingsModal(false);
  `, context);
  document.getElementById("cfg-api-base").value = "http://localhost:8443";
  // Do NOT clear visible inputs first; save immediately without input event -> inherited A token MUST NOT be sent to B!
  vm.runInContext("saveSettingsFromModal()", context);
  assert.equal(vm.runInContext("state.accessToken", context), "");
  assert.equal(vm.runInContext("state.byoKeys.deepseek", context), "");
  assert.equal(vm.runInContext("state.byoKeys.openai", context), "");
  const bStatusReq = requests.find((r) => r.url.startsWith("http://localhost:8443/") && r.url.endsWith("/v1/system/status"));
  assert.ok(bStatusReq);
  assert.equal(bStatusReq.headers["X-MRNAX-Access-Token"], undefined);

  // Now switch back to A, reopen modal, change #cfg-api-base to B, replace visible fields ONLY -> inactive openai stays empty!
  vm.runInContext(`
    state.apiBase = "https://149-248-39-200.sslip.io:8443";
    state.accessToken = "tok_A_only";
    state.byoKeys.deepseek = "sk-key-A-only";
    state.byoKeys.openai = "sk-openai-A-inactive";
    state.byoKey = "sk-key-A-only";
    openSettingsModal(false);
  `, context);
  document.getElementById("cfg-api-base").value = "http://localhost:8443";
  document.getElementById("cfg-access-token").value = "tok_B_explicit";
  document.getElementById("cfg-byo-key").value = "sk-B-explicit";
  vm.runInContext("saveSettingsFromModal()", context);
  assert.equal(vm.runInContext("state.accessToken", context), "tok_B_explicit");
  assert.equal(vm.runInContext("state.byoKeys.deepseek", context), "sk-B-explicit");
  assert.equal(vm.runInContext("state.byoKeys.openai", context), "");

  // Restore Gateway A cleanly for retry-preservation and delayed-status race test
  vm.runInContext(`
    state.apiBase = "https://149-248-39-200.sslip.io:8443";
    state.accessToken = "tok_A_only";
    state.llmProvider = "deepseek";
    state.llmModel = "deepseek-flash";
    state.byoKeys.deepseek = "sk-key-A-only";
    state.byoKey = "sk-key-A-only";
    syncCurrentOriginCredentialsToMap("https://149-248-39-200.sslip.io:8443");
  `, context);
  requests.length = 0;

  await vm.runInContext('runInitialTurn("Design eGFP")', context);
  const failedRetryCtx = vm.runInContext("state.lastFailedTurn", context);
  assert.ok(failedRetryCtx);
  const origReqId = failedRetryCtx.clientRequestId;

  // Same-gateway credential repair preserves Retry banner and clientRequestId!
  document.getElementById("cfg-api-base").value = "https://149-248-39-200.sslip.io:8443";
  document.getElementById("cfg-access-token").value = "tok_A_repaired";
  document.getElementById("cfg-llm-provider").value = "deepseek";
  document.getElementById("cfg-llm-model").value = "deepseek-flash";
  document.getElementById("cfg-byo-key").value = "sk-key-A-repaired";
  vm.runInContext("saveSettingsFromModal()", context);
  assert.equal(document.getElementById("banner-notes").hidden, false);
  assert.ok(document.getElementById("banner-notes").innerHTML.includes('data-action="retry-last"'));
  assert.equal(vm.runInContext("state.lastFailedTurn.clientRequestId", context), origReqId);

  // Save Gateway B with B's credentials + mark export cache -> Gateway change resets workspace & isolates creds!
  vm.runInContext('state.cachedExports["https://149-248-39-200.sslip.io:8443:p1:1:b1"] = { files: {} };', context);
  // Start a status request to A (stays pending on statusGateResolve)
  const pendingStatusA = vm.runInContext('state.apiBase = "https://149-248-39-200.sslip.io:8443"; refreshSystemStatus()', context);
  document.getElementById("cfg-api-base").value = "http://localhost:8443";
  document.getElementById("cfg-access-token").value = "tok_B_secret";
  document.getElementById("cfg-byo-key").value = "sk-key-B-secret";
  vm.runInContext("saveSettingsFromModal()", context);
  assert.equal(Object.keys(vm.runInContext("state.cachedExports", context)).length, 0);
  assert.equal(vm.runInContext("state.lastFailedTurn", context), null);

  // Resolve delayed status from Gateway A after Gateway B succeeded -> Gateway B hostname must remain!
  if (statusGateResolve) {
    statusGateResolve({
      ok: true,
      status: 200,
      async text() {
        return JSON.stringify({ hostname: "stale-gw-A", default_profile: "commercial_clean", caller_token_authenticated: false });
      },
    });
  }
  await pendingStatusA;
  await new Promise((r) => setTimeout(r, 15));
  assert.ok(document.getElementById("txt-gateway").textContent.includes("local-gw-B"));

  // Directly replaying A's snapshot MUST NOT carry tok_B_secret or sk-key-B-secret!
  requests.length = 0;
  vm.runInContext("globalThis.__tmpRetry = null;", context);
  context.__tmpRetry = failedRetryCtx;
  await vm.runInContext("replayExactRequest(globalThis.__tmpRetry)", context);
  const replayReq = requests.find((r) => r.url.startsWith("https://149-248-39-200.sslip.io:8443/") && r.url.endsWith("/agent-chat"));
  assert.ok(replayReq);
  assert.notEqual(replayReq.headers["X-MRNAX-Access-Token"], "tok_B_secret");
  assert.notEqual(replayReq.headers["X-MRNAX-LLM-Key"], "sk-key-B-secret");

  // Empty ?api= triggers startup rejection and blocks subsequent hashchange #token= & apiFetch
  winLoc.href = "https://whatmadpursuit.github.io/mRNAx/?api=";
  vm.runInContext("wireEvents(); initConfig();", context);
  assert.equal(Boolean(vm.runInContext("state.startupApiRejected", context)), true);
  winLoc.hash = "#token=tok_hashchange_evil";
  listeners.get("hashchange")();
  assert.equal(vm.runInContext("state.accessToken", context), "");
  assert.equal(winLoc.hash, "");
  await assert.rejects(
    vm.runInContext('apiFetch("/v1/projects/p1/turn", { method: "POST", body: "{}" })', context),
    /UNTRUSTED_ORIGIN/
  );
});

test("deterministic direct submit & fallback share readonly classification, commercial_clean struct, and unmodified_u", async () => {
  const fields = new Map();
  fields.set("chat-stream", { children: [], scrollTop: 0, scrollHeight: 0, querySelector() { return null; }, appendChild(c) { this.children.push(c); } });
  const reqs = [];
  const fetchMock = async (url, opts = {}) => {
    reqs.push({ url: String(url), body: opts.body ? JSON.parse(opts.body) : null });
    return {
      ok: true,
      status: 200,
      async text() {
        return JSON.stringify({
          status: "COMPLETED",
          intent: { intent_id: "i1", intent_version: 1, status: "ready", compliance_track: "commercial_clean", uridine_chemistry: "unmodified_u", locked_scopes: {} },
          report: { outcome_status: "COMPLETED" },
          candidates: [],
        });
      },
    };
  };
  const document = {
    addEventListener() {},
    documentElement: { dataset: {} },
    querySelectorAll() { return []; },
    createElement(tag) { return { tagName: tag.toUpperCase(), className: "", innerHTML: "", value: "", textContent: "" }; },
    getElementById(id) {
      if (!fields.has(id)) fields.set(id, { hidden: true, textContent: "", value: "" });
      return fields.get(id);
    },
  };
  const context = vm.createContext({
    window: { location: new URL("http://localhost"), crypto: require("node:crypto").webcrypto },
    document,
    URL,
    fetch: fetchMock,
  });
  vm.runInContext(source, context);
  document.getElementById("sel-exec-mode").value = "deterministic";
  document.getElementById("sel-profile").value = "research_max";

  // 1. Both v0 and v1 direct deterministic "Explain the current design." & "How can I switch to commercial_clean?" emit ZERO /turn calls!
  await vm.runInContext('runInitialTurn("Explain the current design.")', context);
  await vm.runInContext('runInitialTurn("How can I switch to commercial_clean?")', context);
  assert.equal(reqs.length, 0);
  vm.runInContext('state.activeVersion = 1; state.intentHistory = [{ intent_id: "i1", intent_version: 1 }];', context);
  await vm.runInContext('runRefineTurn(null, "Explain the current design.")', context);
  await vm.runInContext('runRefineTurn(null, "How can I switch to commercial_clean?")', context);
  await vm.runInContext('runRefineTurn(null, "怎么切换到 commercial_clean？")', context);
  assert.equal(reqs.length, 0);

  // 2. Direct deterministic "Switch to commercial_clean" sends structured compliance_track + use_commercial_surrogate
  await vm.runInContext('runRefineTurn(null, "Switch to commercial_clean")', context);
  assert.equal(reqs.length, 1);
  assert.equal(reqs[0].body.compliance_track, "commercial_clean");
  assert.equal(reqs[0].body.use_commercial_surrogate, true);

  // 3. Chemistry parser maps "假尿苷" -> "pseudouridine", "1-甲基假尿苷" -> "m1psi", and "未修饰"/"unmodified" -> "unmodified_u"
  const zhChem = JSON.parse(vm.runInContext('JSON.stringify(parseDeterministicRefineFallback("改成未修饰尿苷"))', context));
  const enChem = JSON.parse(vm.runInContext('JSON.stringify(parseDeterministicRefineFallback("switch to unmodified uridine"))', context));
  const psiChem = JSON.parse(vm.runInContext('JSON.stringify(parseDeterministicRefineFallback("改为假尿苷"))', context));
  const m1PsiChem = JSON.parse(vm.runInContext('JSON.stringify(parseDeterministicRefineFallback("改为1-甲基假尿苷"))', context));
  assert.equal(zhChem.uridine_chemistry, "unmodified_u");
  assert.equal(enChem.uridine_chemistry, "unmodified_u");
  assert.equal(psiChem.uridine_chemistry, "pseudouridine");
  assert.equal(m1PsiChem.uridine_chemistry, "m1psi");

  // 4. Fresh deterministic initial turn parses prompt chemistry/compliance AND omits prefilled eGFP when naming mCherry (direct & 401 fallback)
  reqs.length = 0;
  vm.runInContext(`
    state.activeVersion = 0;
    state.intentHistory = [];
    state.entryMode = "with_r";
    state.userEditedProtein = false;
  `, context);
  document.getElementById("input-protein").value = vm.runInContext("FALLBACK_EGFP", context);
  document.getElementById("sel-chemistry").value = "m1psi";
  document.getElementById("sel-profile").value = "research_max";
  await vm.runInContext('runInitialTurn("Design mCherry with unmodified uridine in commercial_clean")', context);
  assert.equal(reqs.length, 1);
  assert.equal(reqs[0].body.protein_sequence, null);
  assert.equal(reqs[0].body.use_preset_reference_r, false);
  assert.equal(reqs[0].body.uridine_chemistry, "unmodified_u");
  assert.equal(reqs[0].body.compliance_track, "commercial_clean");
  assert.equal(reqs[0].body.use_commercial_surrogate, true);

  reqs.length = 0;
  vm.runInContext(`
    state.activeVersion = 0;
    state.intentHistory = [];
    state.lastFailedTurn = { kind: "initial", utterance: "Design mCherry mRNA" };
    executeFallbackOrRetry("fallback-deterministic");
  `, context);
  await new Promise((r) => setTimeout(r, 15));
  assert.equal(reqs.length, 1);
  assert.equal(reqs[0].body.protein_sequence, null);
  assert.equal(reqs[0].body.use_preset_reference_r, false);
});

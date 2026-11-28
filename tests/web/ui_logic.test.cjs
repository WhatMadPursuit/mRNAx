const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const source = fs.readFileSync(path.resolve(__dirname, "../../web/app.js"), "utf8");

function response(status, body) {
  const textBody = typeof body === "string" ? body : JSON.stringify(body);
  return {
    ok: status >= 200 && status < 300,
    status,
    headers: { get: (k) => (String(k).toLowerCase() === "content-type" ? "application/json" : null) },
    text: async () => textBody,
    json: async () => JSON.parse(textBody),
  };
}

function app(opts = {}) {
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
  if (opts.fetch) {
    window.fetch = opts.fetch;
  }
  const run = (code) => vm.runInContext(code, context);
  const context = vm.createContext({
    window,
    document,
    URL,
    ...(opts.fetch ? { fetch: opts.fetch } : {}),
  });
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

test("Wave 3 Gate B security: safeExternalUrl, renderSafeExternalLink, citation allowlist, and XSS escaping", () => {
  const { run, document } = app();
  assert.equal(run('safeExternalUrl("javascript:alert(1)")'), "");
  assert.equal(run('safeExternalUrl("data:text/html,<script>alert(1)</script>")'), "");
  assert.equal(run('safeExternalUrl("vbscript:msgbox(1)")'), "");
  assert.equal(run('safeExternalUrl("https://user:secret@example.com/path")'), "");
  assert.equal(run('safeExternalUrl("https://doi.org/10.1038/s41586-023-06800-3")'), "https://doi.org/10.1038/s41586-023-06800-3");

  const safeLink = run('renderSafeExternalLink("https://pubmed.ncbi.nlm.nih.gov/38057662/", "Mulroney <script>1</script>")');
  assert.match(safeLink, /rel="noopener noreferrer"/);
  assert.match(safeLink, /target="_blank"/);
  assert.doesNotMatch(safeLink, /<script>/i);

  const blockedLink = run('renderSafeExternalLink("javascript:evil()", "Click <img onerror=1>")');
  assert.doesNotMatch(blockedLink, /<a\b|javascript:|<img/i);

  assert.equal(run('buildVerifiedCitationUrl("doi", "10.1038/s41586-023-06800-3")'), "https://doi.org/10.1038/s41586-023-06800-3");
  assert.equal(run('buildVerifiedCitationUrl("doi", "https://evil.example/x")'), "");
  assert.equal(run('buildVerifiedCitationUrl("pmid", "38057662")'), "https://pubmed.ncbi.nlm.nih.gov/38057662/");
  assert.equal(run('buildVerifiedCitationUrl("pmid", "38057662/../../evil")'), "");

  const unwrapped = run('extractSandboxedText({ text_excerpt: "<UNTRUSTED_EXTERNAL_DATA source=\\"x\\">Safe <b>Title</b></UNTRUSTED_EXTERNAL_DATA>" })');
  assert.equal(unwrapped, "Safe <b>Title</b>");
  assert.doesNotMatch(run(`escapeHtml(${JSON.stringify(unwrapped)})`), /<b>/);

  // Rules renderer escapes malicious rule payloads and citations
  run(`
    renderRulesAtlas({
      total_available: 1,
      rules: [{
        rule_id: "RUL-XSS-<script>1</script>",
        title_zh: "<img src=x onerror=alert(9)>",
        title_en: "Test \\"quote\\"",
        evidence_grade: "cross_lab_verified",
        region: "5'UTR",
        modality: ["linear_mrna"],
        effect_size_summary: "rho=-0.58 <svg onload=1>",
        enforced_in_generator: true,
        failure_modes: ["<script>boom()</script>"],
        citations: [{ authors: "Author<x>", year: 2024, doi: "10.1038/s41586-023-06800-3" }]
      }]
    });
  `);
  const rulesHtml = document.getElementById("rules-list").innerHTML;
  assert.doesNotMatch(rulesHtml, /<script|<img|<svg onload/i);
  assert.match(rulesHtml, /跨实验室实证/);
  assert.match(rulesHtml, /rel="noopener noreferrer"/);
});

test("Wave 3 de-AI-ification, tool-call pills, client structure profiler, and dual-speed wet-lab calibration", () => {
  const { run, document } = app();

  // 1. Routine COMPLETED turn keeps #banner-notes hidden and moves CRS/domain advisories to boundaries modal
  run(`
    applyTurnResult({
      status: "COMPLETED",
      intent: { intent_id: "i_v1", intent_version: 1, status: "ready", compliance_track: "research_max", uridine_chemistry: "m1psi", locked_scopes: {} },
      report: {
        outcome_status: "COMPLETED_WITH_CANDIDATES",
        honest_attribution_banner: "Generator attribution note",
        pre_synthesis_advisories: [
          { category: "crs_safety_layer_boundary", message: "CRS note" },
          { category: "model_domain_boundary", message: "Domain note" }
        ]
      },
      candidates: []
    }, "design eGFP", 1);
  `);
  assert.equal(document.getElementById("banner-notes").hidden, true);
  assert.equal(run("state.boundaryAdvisories.length"), 3);
  assert.doesNotMatch(document.getElementById("badge-outcome").textContent, /COMPLETED_WITH_CANDIDATES/);

  // 2. buildToolTraceHtml renders clean pills & collapsed receipt without raw JSON dumps
  const traceHtml = run(`buildToolTraceHtml([{ name: "design_constructs", arguments: { uridine_chemistry: "m1psi", restriction_enzymes: ["BsaI"] } }])`);
  assert.match(traceHtml, /tool-pill/);
  assert.match(traceHtml, /tool-trace-fold/);
  assert.match(traceHtml, /验证回执/);
  assert.doesNotMatch(traceHtml, /\{"uridine_chemistry"/);

  // 3. Client deterministic structure profiler builds initiation arcs, sliding tracks, and scans uAUG / UUUU
  const profJson = run(`JSON.stringify(computeClientStructureProfile({
    full_sequence: "GGGAUGCCCGCCACCAUGUUUUAAAACCCCGGGGUAAUAGAAAAAAAAAAAA",
    segments: [
      { segment_type: "five_prime_utr", start: 0, end: 15, length_nt: 15, gc_pct: 60 },
      { segment_type: "cds", start: 15, end: 36, length_nt: 21, gc_pct: 48 },
      { segment_type: "three_prime_utr", start: 36, end: 40, length_nt: 4, gc_pct: 25 }
    ]
  }, "linear_mrna"))`);
  const prof = JSON.parse(profJson);
  assert.ok(prof.initiation_window && prof.initiation_window.sequence.length > 10);
  assert.ok(prof.sliding_tracks.positions.length >= 1);
  assert.ok(prof.motifs.some((m) => m.kind === "uaug"));
  assert.ok(prof.motifs.some((m) => m.kind === "uuuu_frameshift"));

  // 4. Dual-speed wet-lab calibration rule (N=4 anchor-only vs N=8 calibration unlocked)
  document.getElementById("wetlab-input").value = "id,pred,obs\nc1,1.0,1.2\nc2,2.0,2.1\nc3,3.0,3.4\nc4,4.0,3.9";
  run("parseAndRenderWetLabFeedback()");
  assert.match(document.getElementById("wetlab-gate-banner").className, /anchor-only/);
  assert.match(document.getElementById("wetlab-svg-wrap").innerHTML, /<svg/);

  document.getElementById("wetlab-input").value = "id,pred,obs\nc1,1,1.1\nc2,2,2.1\nc3,3,2.9\nc4,4,4.2\nc5,5,4.8\nc6,6,6.1";
  run("parseAndRenderWetLabFeedback()");
  assert.match(document.getElementById("wetlab-gate-banner").className, /calibrate-ok/);
});

test("Gate B C1/C2/I6/I7: fail-closed authorized exports, track-safe structure cache, strict resolver alias, and modality/scaffold forwarding", async () => {
  const calls = [];
  let rejectExports = true;
  const { run, document } = app({
    fetch: async (url, init) => {
      calls.push({ url, init: init || {} });
      if (url.includes("/ivt-order") || url.includes("/dossier")) {
        if (rejectExports) {
          return response(403, { detail: "Forbidden export" });
        }
        if (url.includes("/ivt-order")) {
          return response(200, { ivt_synthesis_order_csv: "bundle_id,template_dna_5to3\nbnd_cc_1,ATG\n" });
        }
        return response(200, { qc_dossier_html: "<!DOCTYPE html><html><body>AUTHORIZED QC</body></html>" });
      }
      if (url.includes("/api/v1/structure/analyze")) {
        return response(200, {
          initiation_window: { start: 0, end: 20, aug_index_in_window: 10, sequence: "GGGAUGCCCGCCACCAUGUU", dot_bracket: "....................", arcs: [] },
          sliding_tracks: { positions: [0], local_mfe: [-5], local_gc_pct: [50], local_u_pct: [20], local_unpaired_prob: [0.6] },
          motifs: [],
          engines_used: ["CleanRoomSeqScan"],
        });
      }
      if (url.includes("/api/v1/evidence/resolve")) {
        return response(200, { protein: { matched: false } });
      }
      if (url.endsWith("/turn")) {
        const parsed = JSON.parse(init.body);
        return response(200, {
          status: "COMPLETED",
          intent: {
            intent_id: "i_circ_1",
            intent_version: 1,
            status: "ready",
            compliance_track: "commercial_clean",
            uridine_chemistry: "unmodified_u",
            modality: parsed.modality || "linear_mrna",
            scaffold_id: parsed.scaffold_id || null,
            locked_scopes: {},
          },
          report: { outcome_status: "COMPLETED_WITH_CANDIDATES", recommended_bundle_ids: ["bnd_cc_1"] },
          candidates: [{
            bundle_id: "bnd_cc_1",
            version_digest: "a".repeat(64),
            pareto_rank: 1,
            length_nt: 36,
            full_gc_pct: 50,
            full_sequence: "GGGAUGCCCGCCACCAUGGUGAGCAAGGGCGAGUAA",
            molecules: [{
              molecule_id: "mol_primary",
              modality: parsed.modality || "linear_mrna",
              length_nt: 36,
              full_gc_pct: 50,
              full_sequence: "GGGAUGCCCGCCACCAUGGUGAGCAAGGGCGAGUAA",
              segments: [
                { segment_type: "five_prime_utr", start: 0, end: 15, length_nt: 15, gc_pct: 60, license_track: "commercial_clean" },
                { segment_type: "cds", start: 15, end: 33, length_nt: 18, gc_pct: 50, license_track: "commercial_clean" },
                { segment_type: "three_prime_utr", start: 33, end: 36, length_nt: 3, gc_pct: 0, license_track: "commercial_clean" },
              ],
            }],
            segments: [
              { segment_type: "five_prime_utr", start: 0, end: 15, length_nt: 15, gc_pct: 60, license_track: "commercial_clean" },
              { segment_type: "cds", start: 15, end: 33, length_nt: 18, gc_pct: 50, license_track: "commercial_clean" },
              { segment_type: "three_prime_utr", start: 33, end: 36, length_nt: 3, gc_pct: 0, license_track: "commercial_clean" },
            ],
            metric_details: [],
          }],
        });
      }
      return response(200, {});
    },
  });

  // I7: deterministic form forwards selected circRNA modality and CVB3 scaffold
  run('setModality("circrna")');
  document.getElementById("sel-scaffold").value = "circ_anabaena_cvb3";
  document.getElementById("sel-profile").value = "commercial_clean";
  await run('runInitialTurn(null, { fromForm: true })');
  const turnCall = calls.find((c) => c.url.endsWith("/turn"));
  const turnBody = JSON.parse(turnCall.init.body);
  assert.equal(turnBody.modality, "circrna");
  assert.equal(turnBody.scaffold_id, "circ_anabaena_cvb3");
  assert.equal(turnBody.uridine_chemistry, "unmodified_u");

  // C2: structure analysis forwards project_id, bundle_id, molecule_id, and commercial_clean track
  await run("runStructureAnalysisForSelected()");
  const structCall = calls.find((c) => c.url.includes("/api/v1/structure/analyze"));
  const structBody = JSON.parse(structCall.init.body);
  assert.equal(structBody.compliance_track, "commercial_clean");
  assert.equal(structBody.bundle_id, "bnd_cc_1");
  assert.equal(structBody.molecule_id, "mol_primary");
  const cacheKeys = run("Object.keys(state.structureCache)");
  assert.ok(cacheKeys[0].includes("bnd_cc_1:") && cacheKeys[0].includes(":commercial_clean:"));

  // C1: 403 on /ivt-order and /dossier blocks local fallback and surfaces error banner
  const ivtBlocked = await run("downloadIvtSynthesisOrder()");
  assert.equal(ivtBlocked, false);
  assert.equal(document.getElementById("banner-notes").hidden, false);
  assert.match(document.getElementById("banner-notes").innerHTML, /导出受阻/);

  const dosBlocked = await run("openOrDownloadQcDossier()");
  assert.equal(dosBlocked, false);
  assert.equal(Boolean(document.getElementById("modal-dossier-preview").open), false);

  rejectExports = false;
  const dosOk = await run("openOrDownloadQcDossier()");
  assert.equal(dosOk, true);
  assert.match(document.getElementById("dossier-iframe").srcdoc, /AUTHORIZED QC/);

  // I6: unknown target resolver query stays unresolved with disabled import button
  await run('resolveTargetProteinInHub("UNKNOWN_TARGET_999")');
  assert.equal(run("state.resolvedTarget"), null);
  assert.match(document.getElementById("dh-target-result").innerHTML, /id="btn-import-resolved-protein" disabled/);

  await run('resolveTargetProteinInHub("P01588")');
  assert.equal(run("state.resolvedTarget.gene_symbol"), "EPO");
  assert.doesNotMatch(document.getElementById("dh-target-result").innerHTML, /id="btn-import-resolved-protein" disabled/);
});

test("v2.5.2 WebUI upgrades: dual-chain DLL3, circRNA T4 td default, unified optgroup model selector, calm auth pill, and clean Pareto/candidate labels", async () => {
  const fields = new Map();
  const makeNode = (tag) => ({
    tagName: String(tag || "DIV").toUpperCase(),
    value: "",
    label: "",
    textContent: "",
    innerHTML: "",
    className: "",
    hidden: false,
    open: false,
    checked: false,
    disabled: false,
    dataset: {},
    style: {},
    children: [],
    options: [],
    listeners: {},
    classList: {
      _set: new Set(),
      toggle(cls, force) {
        if (force === undefined) {
          if (this._set.has(cls)) this._set.delete(cls);
          else this._set.add(cls);
        } else if (force) this._set.add(cls);
        else this._set.delete(cls);
      },
      add(cls) { this._set.add(cls); },
      remove(cls) { this._set.delete(cls); },
      contains(cls) { return this._set.has(cls); },
    },
    setAttribute(k, v) { this[k] = v; },
    getAttribute(k) { return this[k]; },
    addEventListener(ev, fn) { this.listeners[ev] = fn; },
    querySelector() { return null; },
    querySelectorAll() { return []; },
    remove() {},
    insertBefore(child, _ref) {
      this.children.push(child);
    },
    replaceChildren() {
      this.children = [];
      if (this.tagName === "SELECT") this.options = [];
    },
    appendChild(child) {
      this.children.push(child);
      if (this.tagName === "SELECT") {
        if (child.tagName === "OPTGROUP") {
          for (const opt of child.children || []) this.options.push(opt);
        } else {
          this.options.push(child);
        }
      }
    },
    focus() {},
    close() { this.open = false; },
  });

  for (const id of ["sel-llm-model", "cfg-llm-model", "cfg-llm-provider", "sel-scaffold", "inspect-chain", "sel-chemistry", "sel-profile"]) {
    fields.set(id, makeNode("select"));
  }
  fields.get("cfg-llm-provider").options = ["deepseek", "openai", "kimi", "qwen", "gemini"].map((value) => ({ value }));

  const requests = [];
  const fetchMock = async (url, init = {}) => {
    requests.push({ url: String(url), body: init.body ? JSON.parse(init.body) : null });
    return response(200, {
      status: "COMPLETED",
      intent: {
        intent_id: "i_dll3_1",
        intent_version: 1,
        status: "ready",
        compliance_track: "commercial_clean",
        uridine_chemistry: "m1psi",
        locked_scopes: {},
      },
      report: { outcome_status: "COMPLETED_WITH_CANDIDATES", recommended_bundle_ids: ["bnd_pareto_06"] },
      candidates: [],
    });
  };

  const document = {
    addEventListener() {},
    documentElement: { dataset: {} },
    querySelectorAll() { return []; },
    createElement(tag) { return makeNode(tag); },
    getElementById(id) {
      if (!fields.has(id)) fields.set(id, makeNode("div"));
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
  const run = (code) => vm.runInContext(code, context);

  // 1. Unified optgroup model selector & calm auth pill
  run("wireEvents(); initConfig();");
  const authPill = document.getElementById("pill-agent-auth");
  assert.equal(authPill.textContent, "Agent: 配置密钥");
  assert.match(authPill.className, /\bmuted\b/);
  assert.doesNotMatch(authPill.className, /\bwarn\b/);

  const heroSel = document.getElementById("sel-llm-model");
  assert.equal(heroSel.children.length, 5);
  assert.deepEqual(heroSel.children.map((g) => g.label), ["DeepSeek", "OpenAI", "Kimi", "Qwen", "Gemini"]);
  assert.equal(heroSel.value, "deepseek::deepseek-flash");

  run('state.byoKeys.openai = "sk-openai-123";');
  heroSel.value = "openai::gpt-5.4";
  heroSel.listeners.change({ target: heroSel });
  assert.equal(run("state.llmProvider"), "openai");
  assert.equal(run("state.llmModel"), "gpt-5.4");
  assert.equal(document.getElementById("cfg-llm-provider").value, "openai");
  assert.equal(document.getElementById("cfg-llm-model").value, "gpt-5.4");
  assert.equal(document.getElementById("rail-model-pill").textContent, "openai:gpt-5.4");
  assert.equal(authPill.textContent, "Agent: BYO gpt-5.4 ✓");
  assert.match(authPill.className, /\bok\b/);

  // 2. circRNA default & NL scaffold parser
  run('setModality("circrna")');
  assert.equal(document.getElementById("sel-scaffold").value, "circ_t4td_hrvb3");
  assert.equal(JSON.parse(run('JSON.stringify(parseDeterministicRefineFallback("Switch to circRNA"))')).scaffold_id, "circ_t4td_hrvb3");
  assert.equal(JSON.parse(run('JSON.stringify(parseDeterministicRefineFallback("Use T4 td PIE with CVB3"))')).scaffold_id, "circ_t4td_cvb3");
  assert.equal(JSON.parse(run('JSON.stringify(parseDeterministicRefineFallback("Switch to Anabaena HRV-B3"))')).scaffold_id, "circ_anabaena_hrvb3");
  assert.equal(JSON.parse(run('JSON.stringify(parseDeterministicRefineFallback("Switch to Anabaena CVB3"))')).scaffold_id, "circ_anabaena_cvb3");
  assert.equal(JSON.parse(run('JSON.stringify(parseDeterministicRefineFallback("Use SINV saRNA"))')).scaffold_id, "sarna_sinv");

  // 3. Real dual-chain starter-dll3, payload forwarding, agent prompt, chain picker, and resets
  document.getElementById("sel-exec-mode").value = "deterministic";
  document.getElementById("starter-dll3").listeners.click();
  await new Promise((r) => setTimeout(r, 10));
  const vhExpected = run("FALLBACK_DLL3_CD3.primary_protein_sequence");
  const vlExpected = run("FALLBACK_DLL3_CD3.additional_protein_sequences[0]");
  assert.equal(document.getElementById("input-protein").value, vhExpected);
  assert.equal(document.getElementById("badge-aa-len").textContent, `2 chains · ${vhExpected.length} + ${vlExpected.length} aa`);
  assert.ok(document.getElementById("setup-collapsed-summary").textContent.includes(`2 chains · ${vhExpected.length} + ${vlExpected.length} aa`));
  const dll3Turn = requests.find((r) => r.url.endsWith("/turn"));
  assert.equal(dll3Turn.body.protein_sequence, vhExpected);
  assert.deepEqual(dll3Turn.body.additional_protein_sequences, [vlExpected]);
  assert.equal(dll3Turn.body.accession_or_source, "SYNTHETIC_NON_PROPRIETARY_DLL3_CD3_DUAL_CHAIN_DEMO_V1");

  // llm_agent mode mentions both chains and forwards structured context fields
  document.getElementById("sel-exec-mode").value = "llm_agent";
  await run('runInitialTurn("Design DLL3xCD3", { fromForm: true })');
  const dll3Agent = requests.find((r) => r.url.endsWith("/agent-chat"));
  assert.ok(dll3Agent.body.message.includes(vhExpected));
  assert.ok(dll3Agent.body.message.includes(vlExpected));
  assert.equal(dll3Agent.body.compliance_track, "commercial_clean");
  assert.equal(dll3Agent.body.accession_or_source, "SYNTHETIC_NON_PROPRIETARY_DLL3_CD3_DUAL_CHAIN_DEMO_V1");
  assert.deepEqual(dll3Agent.body.additional_protein_sequences, [vlExpected]);

  // Huge multi-chain inputs (> 4000 chars) keep agent message <= 4000 chars and forward full chains structurally
  const longA = "M" + "A".repeat(2499);
  const longB = "M" + "C".repeat(2499);
  document.getElementById("input-protein").value = longA;
  run(`state.additionalProteinSequences = ["${longB}"]; state.userEditedProtein = true;`);
  await run('runInitialTurn("Design large dual chain", { fromForm: true })');
  const largeAgent = requests[requests.length - 1];
  assert.ok(largeAgent.body.message.length <= 4000);
  assert.equal(largeAgent.body.protein_sequence, longA);
  assert.deepEqual(largeAgent.body.additional_protein_sequences, [longB]);

  // Chain picker switches between chain_01 and chain_02
  run(`
    state.candidates = [{
      bundle_id: "bnd_pareto_06",
      version_digest: "d".repeat(64),
      pareto_rank: 1,
      length_nt: 36,
      full_gc_pct: 55,
      full_sequence: "GGGAAAUCCGCCACCAUGGUGAGCAAGGGCGAGUAA",
      molecules: [
        {
          molecule_id: "chain_01",
          length_nt: 18,
          full_gc_pct: 50,
          full_sequence: "GCCACCAUGGUGAGCUAA",
          segments: [{ segment_type: "cds", start: 6, end: 18, length_nt: 12, gc_pct: 50 }],
        },
        {
          molecule_id: "chain_02",
          length_nt: 18,
          full_gc_pct: 61,
          full_sequence: "GCCACCAUGGACAUCUAA",
          segments: [{ segment_type: "cds", start: 6, end: 18, length_nt: 12, gc_pct: 61 }],
        },
      ],
      metric_details: [
        { metric_id: "kozak_unpaired_prob", value: 0.72, unit: "prob", direction: "maximize" },
        { metric_id: "cds_mean_window_mfe", value: -34.2, unit: "kcal/mol", direction: "minimize" },
      ],
    }];
    selectCandidate("bnd_pareto_06");
  `);
  assert.equal(document.getElementById("chain-picker").hidden, false);
  assert.deepEqual(document.getElementById("inspect-chain").options.map((o) => o.textContent), ["chain_01", "chain_02"]);
  assert.match(document.getElementById("txt-selected-bundle").textContent, / · chain_01 · /);
  document.getElementById("inspect-chain").value = "1";
  document.getElementById("inspect-chain").onchange();
  assert.match(document.getElementById("txt-selected-bundle").textContent, / · chain_02 · /);
  assert.match(document.getElementById("seg-inspector").textContent, /AUG GAC AUC UAA/);

  // Manual single-chain edit resets additionalProteinSequences
  document.getElementById("input-protein").value = "MCUSTOMSINGLECHAINSEQUENCE";
  document.getElementById("input-protein").listeners.input({ target: document.getElementById("input-protein") });
  assert.equal(run("JSON.stringify(state.additionalProteinSequences)"), "[]");

  // 4. humanizeCandidateLabel strips eto_xx & renderParetoScatter emits compact #1 tags
  assert.doesNotMatch(run('humanizeCandidateLabel({ bundle_id: "bnd_pareto_06", pareto_rank: 2 })'), /eto_06|bnd_/i);
  assert.doesNotMatch(run('humanizeCandidateLabel({ bundle_id: "bnd_pareto_06", display_label: "Pareto #2 Candidate (eto_06)" })'), /eto_06/i);

  run("renderParetoScatter()");
  const scatterHtml = document.getElementById("pareto-svg-wrap").innerHTML;
  assert.match(scatterHtml, /data-bundle-id="bnd_pareto_06"/);
  assert.match(scatterHtml, />#1<\/text>/);
  assert.doesNotMatch(scatterHtml, /Line…/);
  assert.match(scatterHtml, /<title>.*0\.7200.*-34\.20<\/title>/);

  // 5. Data Hub compute_tier cards (live status + omitted fallback) and human-readable engine titles
  run(`
    state.gatewayOnline = true;
    state.systemStatus = {
      hostname: "vultr-relay-01",
      colab_bridge_present: false,
      drive_bridge_present: true,
      compute_tier: {
        gateway_role: "lightweight_relay_and_keepalive_control_plane",
        heavy_compute_backend: "google_colab_and_drive",
        vps_heavy_binaries_disabled_by_policy: true,
        lease_keepalive_endpoints: ["/v1/executions", "/v1/executions/{execution_id}/leases", "/v1/executions/{execution_id}/attempts/{attempt_id}:commit"],
        colab_drive_sota_assets: {
          lineardesign_linux_x86_64: { drive_file_id: "1ZHp", sha256: "9498", license_track: "research_max" },
          eternafold_v131_linux_x86_64: { drive_file_id: "1Aio", sha256: "84fe", license_track: "commercial_clearance_pending" },
          surrogate_v2_folder: { drive_folder_id: "1rIE", license_track: "commercial_clean" },
          codontransformer_rc005: { hf_repo: "adibvafa/CodonTransformer", license_track: "commercial_clean" }
        }
      }
    };
    renderDataHub({
      summary_counts: { datasets_count: 34 },
      engines_and_models: [
        { engine_id: "stage0_random_and_bicodon_proposer", role: "cds_generator", license_track: "commercial_clean" },
        { engine_id: "dnachisel_cds", role: "cds_generator", license_track: "commercial_clean" },
        { engine_id: "pareto_nsga2", role: "joint_construct_optimizer", license_track: "commercial_clean" },
        { engine_id: "codontransformer_cds_baseline", role: "cds_generator_and_scorer", license_track: "commercial_clean" },
        { engine_id: "surrogate_v2", role: "evaluator", license_track: "commercial_clean" }
      ],
      datasets: []
    });
  `);
  const ctHtml = document.getElementById("dh-compute-tier-grid").innerHTML;
  assert.match(ctHtml, /Vultr 7×24 轻量控制面与保活中转/);
  assert.match(ctHtml, /Colab \+ Drive 重算引擎池/);
  assert.match(ctHtml, /vultr-relay-01/);
  assert.doesNotMatch(ctHtml, /gateway_role|vps_heavy_binaries_disabled_by_policy|lineardesign_linux_x86_64/);
  const engHtml = document.getElementById("dh-engines-grid").innerHTML;
  assert.match(engHtml, /Stage-0 同义基线/);
  assert.match(engHtml, /DNA Chisel 约束编译/);
  assert.match(engHtml, /Pareto NSGA-II 协同优化/);
  assert.match(engHtml, /CodonTransformer \(RC-005\)/);
  assert.match(engHtml, /Surrogate v2 \(商用洁净代理\)/);
  assert.doesNotMatch(engHtml, /stage0_random_and_bicodon_proposer|cds_generator/);

  // Fallback when st.compute_tier is omitted or gateway goes offline
  run("state.gatewayOnline = false; state.systemStatus = null; renderComputeTierCards(null);");
  const fallbackCtHtml = document.getElementById("dh-compute-tier-grid").innerHTML;
  assert.match(fallbackCtHtml, /Vultr 7×24 轻量控制面与保活中转/);
  assert.match(fallbackCtHtml, /Colab \+ Drive 重算引擎池/);
  assert.equal(document.getElementById("dh-compute-tier-badge").textContent, "离线参考拓扑");

  // Gate B-R2: commercial_clean UI track wins over older research_max intent; missing license_track never becomes commercial_clean; colliding fallback labels disambiguate
  run('state.intentHistory = [{ compliance_track: "research_max" }];');
  document.getElementById("sel-profile").value = "commercial_clean";
  assert.equal(run('getActiveComplianceTrackForCandidate({ segments: [{ license_track: "commercial_clean" }] })'), "commercial_clean");
  run(`
    state.candidates = [
      { bundle_id: "b1", pareto_rank: 1, molecules: [{ molecule_id: "c1" }, { molecule_id: "c2" }] },
      { bundle_id: "b2", pareto_rank: 1, molecules: [{ molecule_id: "c1" }, { molecule_id: "c2" }] },
    ];
  `);
  assert.notEqual(run("humanizeCandidateLabel(state.candidates[0])"), run("humanizeCandidateLabel(state.candidates[1])"));
  run(`
    state.dataHubCache = { datasets: [{ dataset_id: "su2025", external_redistribution_allowed: false, governance_scope: "internal_evaluation_only" }] };
    renderDataHubDatasetsTable();
  `);
  assert.doesNotMatch(document.getElementById("dh-datasets-tbody").innerHTML, /commercial_clean/);
  assert.match(document.getElementById("dh-datasets-tbody").innerHTML, /internal_evaluation_only/);
});

test("Gate B-R2 Round 2 (#10a, #10c, #12): agent structured track/modality forwarding, reset accession, and stale structure race guard", async () => {
  const fetchCalls = [];
  let resolveSlowStruct;
  const { run } = app({
    fetch: async (url, init) => {
      fetchCalls.push({ url, init: init || {} });
      if (url.includes("/agent-chat")) {
        return response(200, {
          status: "COMPLETED",
          assistant_text: "Updated construct",
          tool_calls: [],
          turn: {
            status: "COMPLETED",
            intent: { intent_id: "i_ag", intent_version: 2, status: "ready", compliance_track: "commercial_clean", modality: "circrna", scaffold_id: "circ_t4td_hrvb3", uridine_chemistry: "unmodified", locked_scopes: {} },
            candidates: [],
          },
        });
      }
      if (url.includes("/api/v1/structure/analyze")) {
        return new Promise((resolve) => {
          resolveSlowStruct = () => resolve(response(200, {
            initiation_window: { start: 0, end: 10, aug_index_in_window: 0, sequence: "AUGGCCUUAA", dot_bracket: "..........", arcs: [] },
            sliding_tracks: { positions: [0], local_mfe: [-2], local_gc_pct: [50], local_u_pct: [20], local_unpaired_prob: [0.7] },
            motifs: [],
          }));
        });
      }
      if (url.includes("/turn")) {
        return response(200, {
          status: "COMPLETED",
          intent: {
            intent_id: "i_p",
            intent_version: 1,
            status: "ready",
            compliance_track: "commercial_clean",
            modality: "circrna",
            scaffold_id: "circ_t4td_hrvb3",
            locked_scopes: {},
          },
          candidates: [],
        });
      }
      return response(404, {});
    },
  });

  // #10c: resetProjectWorkspaceState restores state.proteinAccession to eGFP default
  run(`state.proteinAccession = "UniProt:P01588"; resetProjectWorkspaceState();`);
  assert.equal(run("state.proteinAccession"), "GenBank:U55762.1");

  // #10a: syncTrackButtons("commercial_clean") updates sel-profile for runInitialTurn, and runRefineTurn in llm_agent mode forwards compliance_track/modality/scaffold_id
  run(`
    state.apiBase = "http://localhost";
    document.getElementById("sel-exec-mode").value = "deterministic";
    document.getElementById("sel-profile").value = "research_max";
    document.getElementById("input-protein").value = "MVSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTT";
    syncTrackButtons("commercial_clean");
  `);
  assert.equal(run(`document.getElementById("sel-profile").value`), "commercial_clean");
  await run(`runInitialTurn("Commercial eGFP mRNA", { fromForm: true })`);
  const turnPresetCall = fetchCalls.find((c) => c.url.includes("/turn"));
  assert.ok(turnPresetCall);
  assert.equal(JSON.parse(turnPresetCall.init.body).compliance_track, "commercial_clean");

  run(`
    document.getElementById("sel-exec-mode").value = "llm_agent";
    document.getElementById("cfg-llm-provider").value = "deepseek";
    document.getElementById("sel-profile").value = "research_max";
    document.getElementById("sel-modality").value = "circrna";
    document.getElementById("sel-scaffold").value = "circ_t4td_hrvb3";
    state.byoKey = "sk-test-agent-123456";
    state.modality = "circrna";
    state.scaffoldId = "circ_t4td_hrvb3";
    document.getElementById("inp-refine").value = "Optimize Kozak";
  `);
  await run(`runRefineTurn(null, "Optimize Kozak", { fromForm: true, complianceTrack: "commercial_clean" })`);
  const agentCall = fetchCalls.find((c) => c.url.includes("/agent-chat") && c.init.method === "POST");
  assert.ok(agentCall);
  const agentBody = JSON.parse(agentCall.init.body);
  assert.equal(agentBody.compliance_track, "commercial_clean");
  assert.equal(agentBody.modality, "circrna");
  assert.equal(agentBody.scaffold_id, "circ_t4td_hrvb3");

  // #12: Stale in-flight structure response suppresses render when candidate track or modality changes mid-flight
  run(`
    document.getElementById("sel-profile").value = "research_max";
    state.structureCache = {};
    state.candidates = [{
      bundle_id: "bnd_race_1",
      version_digest: "deadbeef1234",
      compliance_track: "research_max",
      molecules: [{ molecule_id: "ch1", modality: "linear_mrna", full_sequence: "AUGGCCUAA", segments: [{ segment_type: "cds", start: 0, end: 9, length_nt: 9, gc_pct: 55 }] }],
      chains: [{ chain_id: "ch1", modality: "linear_mrna", full_sequence: "AUGGCCUAA", segments: [{ segment_type: "cds", start: 0, end: 9, length_nt: 9, gc_pct: 55 }] }]
    }];
    state.selectedBundleId = "bnd_race_1";
    state.selectedChainIndex = 0;
    globalThis.__structRenderCalls = 0;
    const __origRenderStruct = renderStructureVisuals;
    renderStructureVisuals = (...args) => { globalThis.__structRenderCalls += 1; return __origRenderStruct(...args); };
  `);
  const pendingStruct = run(`runStructureAnalysisForSelected()`);
  assert.ok(resolveSlowStruct);
  run(`state.candidates[0].compliance_track = "commercial_clean";`);
  resolveSlowStruct();
  await pendingStruct;
  assert.equal(run(`globalThis.__structRenderCalls`), 0);
  const cacheKeys = run(`Object.keys(state.structureCache)`);
  assert.equal(cacheKeys.length, 1);
  assert.match(cacheKeys[0], /:research_max:/);
  assert.doesNotMatch(cacheKeys[0], /:commercial_clean:/);
});

test("Gate B-R2 Round 3 (#10): syncControlsFromIntent resets linear modality/scaffold after circular turn and starter-commercial clears imported accession", async () => {
  const calls = [];
  const { run, document } = app({
    fetch: async (url, init) => {
      calls.push({ url, init: init || {} });
      return response(200, {
        status: "COMPLETED",
        intent: { intent_id: "i_r3", intent_version: 2, status: "ready", compliance_track: "commercial_clean", locked_scopes: {} },
        candidates: [],
      });
    },
  });
  run(`
    state.apiBase = "http://localhost";
    syncControlsFromIntent({ intent_id: "i_c1", intent_version: 1, status: "ready", compliance_track: "commercial_clean", modality: "circrna", scaffold_id: "circ_t4td_hrvb3", locked_scopes: {} });
    assert_mod = state.modality;
    assert_scaf = document.getElementById("sel-scaffold").value;
  `);
  assert.equal(run("assert_mod"), "circrna");
  assert.equal(run("assert_scaf"), "circ_t4td_hrvb3");

  // Linear intent omits default modality/scaffold_id; syncControlsFromIntent must reset controls back to linear_mrna
  run(`
    state.intentHistory.push({ intent_id: "i_c1", intent_version: 1, status: "ready", compliance_track: "commercial_clean", modality: "circrna", scaffold_id: "circ_t4td_hrvb3" });
    syncControlsFromIntent({ intent_id: "i_lin2", intent_version: 2, status: "ready", compliance_track: "commercial_clean", locked_scopes: {} });
  `);
  assert.equal(run("state.modality"), "linear_mrna");
  assert.equal(run('document.getElementById("sel-scaffold").value'), "cap1_hba1_aes_mt_a30l70");

  // Subsequent agent refine uses linear_mrna, not stale circrna
  run(`
    document.getElementById("sel-exec-mode").value = "llm_agent";
    document.getElementById("cfg-llm-provider").value = "deepseek";
    state.byoKey = "sk-r3-agent-123456";
    document.getElementById("inp-refine").value = "Refine UTR";
  `);
  await run(`runRefineTurn(null, "Refine UTR", { fromForm: true })`);
  const agentCall = calls.find((c) => c.url.includes("/agent-chat"));
  assert.ok(agentCall);
  const agBody = JSON.parse(agentCall.init.body);
  assert.equal(agBody.modality, "linear_mrna");
  assert.equal(agBody.scaffold_id, "cap1_hba1_aes_mt_a30l70");

  // Verify empty-project starter-commercial handler resets imported accession before running eGFP turn
  const listenerMap = new Map();
  run(`
    state.intentHistory = [];
    state.proteinAccession = "UniProt:P01588";
    document.getElementById("sel-exec-mode").value = "deterministic";
    document.getElementById("sel-track-mock");
  `);
  const origGet = document.getElementById;
  const noopEl = {
    addEventListener() {},
    removeEventListener() {},
    classList: { toggle() {}, add() {}, remove() {}, contains: () => false },
    dataset: {},
    style: {},
    options: [],
    appendChild() {},
    querySelectorAll: () => [],
  };
  document.querySelectorAll = () => [];
  document.querySelector = () => null;
  document.addEventListener = () => {};
  document.getElementById = (id) => {
    const el = origGet.call(document, id);
    if (!el.addEventListener) {
      el.addEventListener = (ev, fn) => {
        if (!listenerMap.has(id)) listenerMap.set(id, {});
        listenerMap.get(id)[ev] = fn;
      };
      el.classList = { toggle() {}, add() {}, remove() {}, contains: () => false };
      el.dataset = el.dataset || {};
      el.style = el.style || {};
      el.querySelectorAll = () => [];
    }
    return el;
  };
  run("wireEvents()");
  calls.length = 0;
  listenerMap.get("starter-commercial").click();
  await new Promise((r) => setTimeout(r, 20));
  assert.equal(run("state.proteinAccession"), "GenBank:U55762.1");
  const starterTurn = calls.find((c) => c.url.includes("/turn"));
  assert.ok(starterTurn);
  assert.equal(JSON.parse(starterTurn.init.body).accession_or_source, "GenBank:U55762.1");

  // Also verify preset-egfp menu item clears prior imported accession
  run(`state.proteinAccession = "UniProt:P01588";`);
  listenerMap.get("preset-egfp").click();
  assert.equal(run("state.proteinAccession"), "GenBank:U55762.1");

  // Round-5 Finding A: server-returned multi-chain intent updates state.proteinAccession so GenBank:U55762.1 never leaks
  run(`
    state.proteinAccession = "GenBank:U55762.1";
    syncControlsFromIntent({
      intent_id: "i_mc_r5",
      intent_version: 3,
      status: "ready",
      compliance_track: "commercial_clean",
      protein_targets: [
        { protein_sequence: "MVLSPADKTNVKAAWGKV", accession_or_source: "user_supplied" },
        { protein_sequence: "MHLTPEEKSAVTALWGKV", accession_or_source: "user_supplied" }
      ],
      locked_scopes: {}
    });
  `);
  assert.equal(run("state.proteinAccession"), "user_supplied");
});

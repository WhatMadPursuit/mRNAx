const DEFAULT_API_BASE = "https://149-248-39-200.sslip.io:8443";
const TRUSTED_PRODUCTION_ORIGINS = new Set([
  "https://149-248-39-200.sslip.io:8443",
]);
const FALLBACK_EGFP =
  "MVSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITLGMDELYK";

const DEFAULT_PROVIDER_MODELS = {
  deepseek: "deepseek-flash",
  openai: "gpt-4.1-mini",
  kimi: "moonshot-v1-32k",
  qwen: "qwen-plus",
  gemini: "gemini-2.5-flash",
};

const SUPPORTED_PROVIDER_MODELS = {
  deepseek: ["deepseek-flash", "deepseek-v4-pro"],
  openai: ["gpt-4.1-mini"],
  kimi: ["moonshot-v1-32k"],
  qwen: ["qwen-plus"],
  gemini: ["gemini-2.5-flash"],
};

const MODEL_LABELS = {
  "deepseek-flash": "deepseek-flash (极速)",
  "deepseek-v4-pro": "deepseek-v4-pro (深度推理)",
  "gpt-4.1-mini": "gpt-4.1-mini",
  "moonshot-v1-32k": "moonshot-v1-32k",
  "qwen-plus": "qwen-plus",
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

const SEG_META = {
  five_prime_utr: { short: "5′UTR", colorVar: "var(--seg-utr5)" },
  cds: { short: "CDS", colorVar: "var(--seg-cds)" },
  stop_cassette: { short: "Stop", colorVar: "var(--seg-stop)" },
  three_prime_utr: { short: "3′UTR", colorVar: "var(--seg-utr3)" },
  poly_a: { short: "Poly(A)", colorVar: "var(--seg-polya)" },
};

const METRIC_LABELS = {
  kozak_unpaired_prob: { shortLabel: "Kozak P(open)", higherBetter: true, decimals: 4 },
  cds_mean_window_mfe: { shortLabel: "CDS Win MFE", higherBetter: false, decimals: 2 },
  mirna_detarget_dg_total: { shortLabel: "miR-122 ΔG", higherBetter: false, decimals: 2 },
  junction_folding_mfe: { shortLabel: "Junction MFE", higherBetter: true, decimals: 2 },
  utr5_mrl_pred: { shortLabel: "5′UTR MRL", higherBetter: true, decimals: 3 },
  consensus_halflife_pred: { shortLabel: "Tx Half-Life", higherBetter: true, decimals: 3 },
  utr3_stability_pred: { shortLabel: "3′UTR MPRA", higherBetter: true, decimals: 3 },
  cds_stability_pred: { shortLabel: "CDS Stability", higherBetter: true, decimals: 3 },
  orthrus_halflife_pred: { shortLabel: "Orthrus Half-Life", higherBetter: true, decimals: 3 },
  structrfm_cds_median_shape: { shortLabel: "structRFM SHAPE", higherBetter: false, decimals: 3 },
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

function esc(str) {
  return String(str == null ? "" : str).replace(/[&<>"']/g, (ch) => ESC_MAP[ch]);
}

function formatSafeMarkdown(rawText) {
  const escaped = esc(rawText == null ? "" : String(rawText));
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
  state.theme = themeName === "graphite" ? "graphite" : "light";
  if (document.documentElement && document.documentElement.dataset) {
    document.documentElement.dataset.theme = state.theme;
  }
  safeStorageSet("local", "mrnax_theme", state.theme);
  const lbl = document.getElementById("txt-theme-label");
  if (lbl) {
    lbl.textContent = state.theme === "graphite" ? "☀ 清爽蓝白" : "◐ 石墨深色";
  }
  renderParetoScatter();
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
  const summaryEl = document.getElementById("setup-collapsed-summary");
  if (summaryEl) {
    const enzLabel = enz.length ? `0-cut:${enz.join("+")}` : "无酶切禁区";
    summaryEl.textContent = `${raw.length} aa · ${chem} · ${prof} · ${enzLabel}${detarget ? " · miR-122" : ""}`;
  }
}

function updateProteinBadge() {
  const inputProt = document.getElementById("input-protein");
  const raw = ((inputProt && inputProt.value) || "").trim().toUpperCase();
  const badge = document.getElementById("badge-aa-len");
  const startsM = raw.startsWith("M");
  if (badge) {
    badge.textContent = `${raw.length} aa · ${startsM ? "M✓" : "缺起始M⚠"}`;
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
      <div class="turn-meta"><span>COPILOT READY</span><span>3-STEP WORKFLOW</span></div>
      <div class="agent-prose">
        <div><strong>1. 自然语言立项</strong>：直接输入目标蛋白、修饰与禁切酶，或点击上方模板一键启动。</div>
        <div><strong>2. 确定性工具执行</strong>：所有序列计算、FTO 专利守卫与帕累托排序均由确定性引擎执行，绝不虚构。</div>
        <div><strong>3. 单点锁定迭代</strong>：随时追问权衡对比，或仅修改单一约束（蛋白与未改动项 100% 继承）。</div>
      </div>
    </div>
  `;
}

function appendStreamCard(kind, metaLeft, metaRight, bodyHtml) {
  const box = document.getElementById("chat-stream");
  if (!box || typeof document.createElement !== "function") return null;
  removeWelcomeCardIfPresent();
  const card = document.createElement("div");
  card.className = `turn-card ${kind}`;
  card.innerHTML = `
    <div class="turn-meta"><span>${esc(metaLeft)}</span><span>${esc(metaRight)}</span></div>
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
    const shortLabel = catalog?.shortLabel || id;
    const direction = detail.direction || (catalog && (catalog.higherBetter ? "maximize" : "minimize"));
    const arrow = { maximize: " ↑", minimize: " ↓" }[direction] || "";
    return {
      key: id,
      shortLabel,
      label: shortLabel + (detail.unit ? ` (${detail.unit})` : "") + arrow,
      higherBetter: direction === "maximize",
      fmt: (v) => v.toFixed(catalog?.decimals ?? 3),
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

function renderParetoScatter() {
  const wrap = document.getElementById("pareto-svg-wrap");
  const titleEl = document.getElementById("pareto-plot-title");
  if (!wrap) return;

  if (!state.candidates.length) {
    wrap.innerHTML = `<div class="empty-hint mono">输入需求或点击模板运行后绘制多目标帕累托前沿</div>`;
    if (titleEl) titleEl.textContent = "多目标权衡前沿";
    return;
  }

  const measuredAxes = resolveMetricAxes(state.candidates).filter(
    (axis) => state.candidates.some((c) => readAxisValue(c, axis) !== null)
  );
  const [axX, axY] = measuredAxes;
  if (!axY) {
    if (titleEl) titleEl.textContent = "多目标权衡前沿";
    wrap.innerHTML = '<div class="empty-hint mono">二维图需要两项已测指标</div>';
    return;
  }
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
        <text x="${padL - 6}" y="${Number(gy) + 3.5}" text-anchor="end" fill="var(--ink-muted)" font-family="var(--font-mono)" font-size="10">${esc(yVal)}</text>
      `;
    })
    .join("");

  const xTicks = [0, 0.5, 1]
    .map((t) => {
      const gx = (padL + plotW * t).toFixed(1);
      const xVal = (minX + (maxX - minX) * t).toFixed(2);
      return `
        <line x1="${gx}" y1="${padT}" x2="${gx}" y2="${padT + plotH}" stroke="var(--border)" stroke-dasharray="2 3" stroke-width="1" />
        <text x="${gx}" y="${padT + plotH + 14}" text-anchor="middle" fill="var(--ink-muted)" font-family="var(--font-mono)" font-size="10">${esc(xVal)}</text>
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
      const x = mapX(p.x), y = mapY(p.y);
      const marker = placeMarker(x, y);
      const cx = marker.x.toFixed(1), cy = marker.y.toFixed(1);
      const isP1 = p.cand.pareto_rank === 1;
      const isSel = p.cand.bundle_id === state.selectedBundleId;
      const shortId = p.cand.bundle_id.replace(/^bnd_[a-z0-9]+_/, "");
      return `
        <g class="pareto-node" data-idx="${idx}" tabindex="0" role="button" aria-pressed="${isSel}" aria-label="${esc(p.cand.bundle_id)}" style="cursor:pointer;">
          <title>${esc(p.cand.bundle_id)} · ${esc(p.cand.pareto_rank ? "P" + p.cand.pareto_rank : "未排名")} · ${esc(axX.shortLabel)}=${esc(axX.fmt(p.x))}, ${esc(axY.shortLabel)}=${esc(axY.fmt(p.y))}</title>
          ${marker.x !== x || marker.y !== y ? `<line x1="${x}" y1="${y}" x2="${cx}" y2="${cy}" stroke="var(--border-strong)" pointer-events="none" />` : ""}
          ${isSel ? `<circle pointer-events="none" cx="${cx}" cy="${cy}" r="9.5" fill="none" stroke="var(--ink)" stroke-width="1.6" />` : ""}
          <circle
            cx="${cx}"
            cy="${cy}"
            r="${isP1 ? "5.2" : "4.4"}"
            fill="${isP1 ? "var(--accent)" : "var(--bg-surface)"}"
            stroke="${isP1 ? "var(--accent)" : "var(--ink-muted)"}"
            stroke-width="1.8"
          />
          <text pointer-events="none" x="${(Number(cx) + 7).toFixed(1)}" y="${(Number(cy) - 5).toFixed(1)}" fill="var(--ink-secondary)" font-family="var(--font-mono)" font-size="9.5">${esc(shortId.length > 16 ? "…" + shortId.slice(-10) : shortId)}</text>
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
      <text x="${padL + plotW / 2}" y="${H - 4}" text-anchor="middle" fill="var(--ink-secondary)" font-family="var(--font-mono)" font-size="10">${esc(axX.label)}</text>
      <text x="12" y="${padT + plotH / 2}" text-anchor="middle" transform="rotate(-90 12 ${padT + plotH / 2})" fill="var(--ink-secondary)" font-family="var(--font-mono)" font-size="9.5">${esc(axY.shortLabel + (axY.higherBetter ? " ↑" : " ↓"))}</text>
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
      <span>${esc(axis.fmt(val))}</span>
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
      return a.bundle_id.localeCompare(b.bundle_id) * dir;
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
  if (propWrap) propWrap.innerHTML = `<div class="empty-hint mono">生成后展示五段真实比例分子图谱</div>`;
  const segBar = document.getElementById("seg-bar");
  if (segBar) segBar.innerHTML = "";
  const segInsp = document.getElementById("seg-inspector");
  if (segInsp) segInsp.textContent = "选择候选构建后自动对齐显示区段核苷酸与三联体密码子翻译轨";
}

function updateExportButtonLabels() {
  const gbBtn = document.getElementById("btn-dl-gb");
  const hasCand = Boolean(state.selectedBundleId && state.candidates.length);
  ["btn-dl-gb", "btn-dl-rna", "btn-dl-cds", "btn-dl-meta"].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.disabled = !hasCand;
  });
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
    if (tbody) {
      const colSpan = (header && header.children && header.children.length) || 4;
      tbody.innerHTML = `<tr><td colspan="${colSpan}" class="empty-cell mono">当前无合格推荐候选（详见上方状态横幅）</td></tr>`;
    }
    clearSelectedConstructUI();
    updateExportButtonLabels();
    renderParetoScatter();
    return;
  }

  const ranges = axes.map((ax) => {
    const vals = state.candidates.map((c) => readAxisValue(c, ax)).filter((v) => v != null);
    return {
      min: vals.length ? Math.min(...vals) : 0,
      max: vals.length ? Math.max(...vals) : 1,
    };
  });

  const recSet = new Set((state.latestReport && state.latestReport.recommended_bundle_ids) || []);
  const sorted = getSortedCandidates(axes);

  if (tbody) {
    tbody.innerHTML = "";
    sorted.forEach((c) => {
      const tr = document.createElement("tr");
      tr.tabIndex = 0;
      if (c.bundle_id === state.selectedBundleId) tr.classList.add("selected-row");
      const rankNum = c.pareto_rank;
      const isRec = rankNum === 1 && recSet.has(c.bundle_id);
      const metricCells = axes
        .map(
          (axis, i) =>
            `<td class="num-col">${buildSparkCell(readAxisValue(c, axis), axis, ranges[i].min, ranges[i].max)}</td>`
        )
        .join("");

      tr.innerHTML = `
        <td class="mono"><span class="badge ${rankNum === 1 ? "ok" : ""}">${rankNum ? "P" + esc(rankNum) : "未排名"}</span></td>
        <td class="mono">${isRec ? "★ " : ""}${esc(c.bundle_id)}</td>
        <td class="mono" style="color:var(--ink-secondary);">${esc(c.derivation_action)}</td>
        ${metricCells || '<td class="num-col">N/A · 未测</td>'}
        <td class="num-col mono">${esc(c.full_gc_pct)}% · ${esc(c.length_nt)}nt</td>
      `;
      tr.addEventListener("click", () => selectCandidate(c.bundle_id));
      tr.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          selectCandidate(c.bundle_id);
        }
      });
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
    return `<span class="enz-tag ${cls}">${esc(enz)}${esc(reqMark)}: ${cuts} cuts ${cuts === 0 ? "✓" : "⚠"}</span>`;
  });
  const extraTags = activeList
    .filter((enz) => !Object.prototype.hasOwnProperty.call(RESTRICTION_SITES_RNA, enz))
    .map((enz) => `<span class="enz-tag clear">${esc(enz)} [禁]: server-gated ✓</span>`);
  rail.innerHTML = [...presetTags, ...extraTags].join("");
}

function renderTrueScaleMap(cand) {
  const ruler = document.getElementById("ruler-row");
  const propWrap = document.getElementById("prop-track-wrap");
  const cardsBar = document.getElementById("seg-bar");
  const total = Math.max(1, cand.length_nt);

  if (ruler) {
    ruler.innerHTML = [0, 0.25, 0.5, 0.75, 1]
      .map((f) => `<span>${Math.round(total * f)} nt</span>`)
      .join("");
  }

  let cursorPct = 0;
  let stopPinHtml = "";
  const propBar = document.createElement("div");
  propBar.className = "prop-bar";

  cand.segments.forEach((seg) => {
    const rawPct = (seg.length_nt / total) * 100;
    const centerPct = cursorPct + rawPct / 2;
    cursorPct += rawPct;
    const meta = SEG_META[seg.segment_type] || { short: seg.segment_type, colorVar: "var(--accent)" };
    if (seg.segment_type === "stop_cassette") {
      stopPinHtml = `<span class="stop-callout" style="inset-inline-start:${centerPct.toFixed(3)}%;">Stop (${esc(seg.length_nt)}nt) ↓</span>`;
    }
    const segDiv = document.createElement("div");
    segDiv.className = `prop-seg ${seg.segment_type === state.selectedSegType ? "active-prop" : ""}`;
    segDiv.title = `${meta.short} [${seg.start}..${seg.end}) · ${seg.length_nt} nt · GC ${seg.gc_pct}%`;
    segDiv.style.width = `${rawPct.toFixed(4)}%`;
    segDiv.style.background = meta.colorVar;
    segDiv.addEventListener("click", () => inspectSegment(cand, seg));
    propBar.appendChild(segDiv);
  });

  if (propWrap) {
    propWrap.innerHTML = stopPinHtml;
    propWrap.appendChild(propBar);
  }

  if (cardsBar) {
    cardsBar.innerHTML = "";
    cand.segments.forEach((seg) => {
      const meta = SEG_META[seg.segment_type] || { short: seg.segment_type, colorVar: "var(--accent)" };
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = `seg-card-btn ${seg.segment_type === state.selectedSegType ? "active" : ""}`;
      btn.style.borderBlockStartColor = meta.colorVar;
      btn.innerHTML = `
        <div class="seg-card-name">${esc(meta.short)} · ${esc(seg.length_nt)} nt</div>
        <div class="seg-card-meta">[${esc(seg.start)}..${esc(seg.end)}) · GC ${esc(seg.gc_pct)}%</div>
      `;
      btn.addEventListener("click", () => inspectSegment(cand, seg));
      cardsBar.appendChild(btn);
    });
  }
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
  const fullSub = cand.full_sequence.slice(seg.start, seg.end);
  const meta = SEG_META[seg.segment_type] || { short: seg.segment_type };
  const header =
    `${meta.short} [${seg.start}..${seg.end}) · ${seg.length_nt} nt · GC ${seg.gc_pct}% · ` +
    `轨: ${seg.license_track} · 动作: ${seg.action_taken} · 来源: ${seg.origin_source || "synthetic"}`;
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
  const cand = state.candidates.find((c) => c.bundle_id === bundleId);
  if (!cand) return;

  const chains = cand.molecules?.length ? cand.molecules : [cand];
  const select = document.getElementById("inspect-chain");
  if (select && typeof select.replaceChildren === "function") {
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
    const chain = chains[index];
    const label = chains.length > 1 ? ` · ${chain.molecule_id}` : "";
    const txtSel = document.getElementById("txt-selected-bundle");
    if (txtSel) {
      txtSel.textContent =
        `${cand.bundle_id}${label} · ${chain.length_nt} nt · GC ${chain.full_gc_pct}% · sha256:${cand.version_digest.slice(0, 10)}`;
    }
    renderEnzymeClearance(chain.full_sequence);
    const segment =
      chain.segments.find((s) => s.segment_type === state.selectedSegType) ||
      chain.segments.find((s) => s.segment_type === "cds") ||
      chain.segments[0];
    if (segment) inspectSegment(chain, segment);
    else renderTrueScaleMap(chain);
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
      `禁切酶: <span class="strike-old">${esc((intentDiff.v1_enzymes || []).join("+") || "none")}</span>→ <strong>${esc((intentDiff.v2_enzymes || []).join("+") || "none")}</strong>`
    );
  }
  if (prev.intent.compliance_track !== curr.intent.compliance_track) {
    deltaParts.push(
      `合规轨: <span class="strike-old">${esc(prev.intent.compliance_track)}</span>→ <strong>${esc(curr.intent.compliance_track)}</strong>`
    );
  }
  const prevLocks = Object.keys(prev.intent.locked_scopes || {});
  const currLocks = Object.keys(curr.intent.locked_scopes || {});
  if (currLocks.length > prevLocks.length) {
    const added = currLocks.filter((k) => !prevLocks.includes(k));
    deltaParts.push(`新增区段锁定: <strong>${esc(added.join(", "))}</strong>`);
  }
  if (!deltaParts.length) {
    deltaParts.push(`增量重排完成 · 候选 <strong>${esc(prev.topCandidate.bundle_id)}</strong> → <strong>${esc(curr.topCandidate.bundle_id)}</strong>`);
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
  if (locksBar) locksBar.innerHTML = lockTokens.map((t) => `<span class="lock-token">${esc(t)}</span>`).join("");

  const dualWrap = document.getElementById("dual-scale-tracks");
  const prevChains = prev.topCandidate.molecules || [prev.topCandidate];
  const currChains = curr.topCandidate.molecules || [curr.topCandidate];
  const maxNt = Math.max(...prevChains.concat(currChains).map((chain) => chain.length_nt), 1);

  const buildScaledRow = (verTag, cand, other, isCurrent) => {
    const otherSeqMap = new Map(
      other?.segments.map((s) => [s.segment_type, other.full_sequence.slice(s.start, s.end)])
    );
    const segHtml = cand.segments
      .map((s) => {
        const meta = SEG_META[s.segment_type] || { short: s.segment_type, colorVar: "var(--accent)" };
        const wPct = ((s.length_nt / maxNt) * 100).toFixed(3);
        const mySeq = cand.full_sequence.slice(s.start, s.end);
        const changed = mySeq !== otherSeqMap.get(s.segment_type);
        const opacity = changed ? "1" : "0.38";
        const outline = changed && isCurrent ? "box-shadow: inset 0 0 0 1.5px var(--ink);" : "";
        return `<span title="${esc(meta.short)}: ${esc(s.length_nt)} nt (${changed ? "已更新" : "100% 继承未变"})" style="width:${wPct}%; background:${meta.colorVar}; opacity:${opacity}; ${outline}"></span>`;
      })
      .join("");
    return `
      <div class="dual-track-row mono">
        <span style="color:var(--ink-secondary); font-weight:${isCurrent ? "600" : "400"};">${esc(verTag)}</span>
        <div class="dual-bar">${segHtml}</div>
        <span style="text-align:end; color:var(--ink-muted);">${esc(cand.length_nt)} nt</span>
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

function buildToolTraceHtml(toolCalls) {
  if (!Array.isArray(toolCalls) || !toolCalls.length) return "";
  const names = toolCalls.map((tc) => tc.name).join(" → ");
  const items = toolCalls
    .map((tc) => {
      const argStr = JSON.stringify(tc.arguments || {});
      return `<li><code>${esc(tc.name)}</code>(${esc(argStr.length > 140 ? argStr.slice(0, 140) + "…" : argStr)})</li>`;
    })
    .join("");
  return `
    <details class="tool-trace-strip mono">
      <summary>调用 ${toolCalls.length} 个确定性工具：${esc(names)}</summary>
      <ul class="tool-trace-list">${items}</ul>
    </details>
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
  ["starter-egfp", "starter-ref-r", "starter-commercial", "starter-tradeoffs"].forEach((id) => {
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
    lines.push(`- \`${cand.bundle_id}\` (${cand.derivation_action})：${metricsSummary || "仅序列硬门通过"}`);
  }
  appendStreamCard("agent", "本地只读权衡分析", `v${String(state.activeVersion).padStart(2, "0")}`, `<div class="agent-prose">${formatSafeMarkdown(lines.join("\n"))}</div>`);
}

function syncControlsFromIntent(intent) {
  if (!intent || typeof intent !== "object") return;
  const profEl = document.getElementById("sel-profile");
  const chemEl = document.getElementById("sel-chemistry");
  if (profEl && intent.compliance_track) profEl.value = intent.compliance_track;
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
    appendStreamCard("agent", "Agent 回复", modelTag, `${toolTraceHtml}${replyBody}`);
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
    const qs = questions.map((q) => `<li>${esc(q)}</li>`).join("");
    if (banner) {
      banner.hidden = false;
      banner.className = "notice-banner";
      banner.innerHTML = `<strong>编译前护栏拦截 (${esc(res.intent.status)})：</strong><ul style="margin:4px 0 0 18px;">${qs}</ul>`;
    }
    const toolCallsBlock = buildToolTraceHtml(res.tool_calls);
    const replyExtra = res.assistant_reply
      ? `<div class="agent-prose">${formatSafeMarkdown(res.assistant_reply)}</div>`
      : "";
    appendStreamCard(
      "warn",
      `安全护栏拦截 · Intent v${res.intent.intent_version}`,
      res.intent.status,
      `<strong>拦截待确认项：</strong><ul style="margin:4px 0 0 18px;">${qs}</ul>${toolCallsBlock}${replyExtra}`
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

  const outcome = (res.report && res.report.outcome_status) || "COMPLETED";
  const badgeOut = document.getElementById("badge-outcome");
  if (badgeOut) {
    badgeOut.textContent = outcome;
    badgeOut.className =
      outcome.includes("UNCOMPARABLE") || !state.candidates.length ? "badge mono warn" : "badge mono ok";
  }

  const notes = [];
  if (res.report && res.report.uncomparable_reason) {
    notes.push(`<strong>档位对比阻断：</strong>${esc(res.report.uncomparable_reason)}`);
  }
  if (res.report && res.report.honest_attribution_banner) {
    notes.push(`<strong>归因声明：</strong>${esc(res.report.honest_attribution_banner)}`);
  }
  (res.report?.action_transparency || []).forEach((t) => {
    if (t.proof_diff_summary && t.stage_applied === "recorded_only") {
      notes.push(`<strong>[${esc(t.requirement_id)} · ${esc(t.stage_applied)}]</strong> ${esc(t.proof_diff_summary)}`);
    }
  });
  (res.report?.pre_synthesis_advisories || []).forEach((adv) => {
    if (adv.category === "crs_safety_layer_boundary" || adv.category === "model_domain_boundary") {
      notes.push(`<strong>[${esc(adv.category)}]</strong> ${esc(adv.message)}`);
    }
  });
  if (banner) {
    if (notes.length) {
      banner.hidden = false;
      banner.className = "notice-banner";
      banner.innerHTML = notes.join("<br/>");
    } else {
      banner.hidden = true;
      banner.className = "notice-banner";
      banner.innerHTML = "";
    }
  }

  let diffHtml = "";
  if (res.intent_diff) {
    const d = res.intent_diff;
    diffHtml = `<div style="margin-top:4px; font-size:11.5px; color:var(--ok);">
      ✓ 继承核验：<code>${esc(d.v1_id)} → ${esc(d.v2_id)}</code> ·
      蛋白锁定: <code>${esc(d.protein_lock_preserved)}</code> ·
      化学修饰: <code>${esc(d.chemistry_preserved)}</code> ·
      禁切酶: <code>${esc(JSON.stringify(d.v1_enzymes))} → ${esc(JSON.stringify(d.v2_enzymes))}</code>
    </div>`;
  }
  const traceHtml = (res.report?.action_transparency || [])
    .map((t) => `<li><code>${esc(t.requirement_id)}</code> [${esc(t.stage_applied)}]: ${esc(t.proof_diff_summary)}</li>`)
    .join("");
  const traceBlock = traceHtml ? `<ul style="margin:4px 0 0 18px; font-size:11.5px;">${traceHtml}</ul>` : "";
  const toolCallsBlock = buildToolTraceHtml(res.tool_calls);
  const llmReplyHtml = res.assistant_reply
    ? `<div class="agent-prose">${formatSafeMarkdown(res.assistant_reply)}</div>`
    : "";

  const actorLabel = res.model
    ? `${res.provider || state.llmProvider}:${res.model} · v${res.intent.intent_version}`
    : `Harness 完成 · Intent v${res.intent ? res.intent.intent_version : "?"}`;

  appendStreamCard(
    "agent",
    actorLabel,
    `${state.candidates.length} 条合格`,
    `<div>指令：<em>${esc(userText)}</em> → <code>${esc(outcome)}</code></div>${diffHtml}${toolCallsBlock}${llmReplyHtml}${traceBlock}`
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
    pendingCard.innerHTML = `<div class="turn-meta"><span>RUNNING</span><span>0.0s</span></div><div><span class="spin"></span> ${esc(modeLabel)}</div>`;
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
    if (btnSend) btnSend.innerHTML = `<span class="spin"></span> 计算中… ${sec}s`;
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
  // Classify local auth errors strictly by outer HTTP 401 status (never regex on upstream 502 error strings)
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
        公开网页默认拦截未授权云端 DeepSeek 调用，以保护服务器额度。
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
        <strong>请求失败：</strong><code>${esc((err && err.message) || "Unknown error")}</code>
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
      : esc((err && err.message) || "请求异常")
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
    `当前处于确定性引擎模式且尚未生成设计版本；问题 <code>${esc(
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
    esc(reqSpec.utterance)
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
    allowMainDraftFallback && !submittedText && utteranceInput ? utteranceInput.value.trim() : "";
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

  const utterance =
    draftText ||
    (useR
      ? `基于参考构建 R 设计 eGFP mRNA (${chem}, ${prof}${late ? ", 后期表达" : ""}${detarget ? ", 肝脏去靶向" : ""})，避开 ${enz.join("/") || "无"}，生成保守最小改动 A 与探索方案 B`
      : `从零设计 eGFP 报告蛋白 mRNA (${chem}, ${prof}${late ? ", 后期表达" : ""})，蛋白 100% 锁定，${detarget ? "开启肝脏去靶向，" : ""}避开 ${enz.join("/") || "无"}`);

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
  const defaultProt = (state.presets?.egfp_reporter?.protein_sequence) || FALLBACK_EGFP;
  const namesEgfp = /\b(?:egfp|gfp|p42212|u55762)\b/i.test(draftText);
  const namesOtherTarget = /\b(?:mcherry|cas9|luciferase|fluc|spike|insulin|epo)\b/i.test(draftText);
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
    ? `\n[表单参数：compliance_track=${prof}${prof === "commercial_clean" ? " (use_commercial_surrogate=true)" : ""}, uridine_chemistry=${chem}, entry_mode=${useR ? "protein_with_reference_r" : "protein_only"}, restriction_enzymes=${JSON.stringify(enz)}, detarget_liver=${detarget}, prioritize_late_signal=${late}]`
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
        uridine_chemistry: effChem,
        compliance_track: effProf,
        use_commercial_surrogate: effSurrogate,
        restriction_enzymes: effEnz,
        detarget_liver: effDetarget,
        prioritize_late_signal: late,
        conflicting_single_pick: conflict,
        use_preset_reference_r: Boolean(useR && usePrefilledDefaultForDet && !namesOtherTarget),
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
    esc(utterance)
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
    allowMainDraftFallback && !submittedText && utteranceInput ? utteranceInput.value.trim() : "";
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
    esc(utterance)
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
  const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  const revTag = `v${String(revNum).padStart(2, "0")}`;
  a.download = `${revTag}_${bid}_${filename}`;
  document.body.appendChild(a);
  a.click();
  a.remove();
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
      name: "1. 7×24 Vultr HTTPS 健康与热力学/洁净内核就绪核验",
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
    row.innerHTML = `<span>${esc(step.name)}</span><span class="mono" style="color:var(--warn);">运行中...</span>`;
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
      `已拒绝不受信任的 API 源 <code>${esc(candidateApi)}</code>；网关配置与凭据未保存，也未向其他网关发送。`
    );
    const banner = document.getElementById("banner-notes");
    if (banner) {
      banner.hidden = false;
      banner.className = "notice-banner err";
      banner.innerHTML = `<strong>设置未保存：</strong>已拒绝不受信任的网关源 <code>${esc(candidateApi)}</code>，密钥未改动也未发送。`;
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
  document.getElementById("btn-theme-toggle").addEventListener("click", () => {
    applyTheme(state.theme === "graphite" ? "light" : "graphite");
  });

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

  document.addEventListener("click", (e) => {
    const insideDropdown = e.target.closest("details.dropdown-menu");
    closeOpenDropdowns(insideDropdown);
  });

  document.getElementById("btn-toggle-copilot")?.addEventListener("click", () => {
    const grid = document.getElementById("studio-grid");
    const btn = document.getElementById("btn-toggle-copilot");
    if (!grid || !btn) return;
    const collapsed = grid.classList.toggle("copilot-collapsed");
    btn.setAttribute("aria-pressed", String(!collapsed));
  });

  document.querySelectorAll(".engine-opt").forEach((btn) => {
    btn.addEventListener("click", () => {
      setExecMode(btn.dataset.mode, { userInitiated: true });
    });
  });
  document.getElementById("sel-exec-mode")?.addEventListener("change", (e) => {
    setExecMode(e.target.value, { userInitiated: true });
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
    if (cfgModel) {
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

  document.getElementById("input-protein").addEventListener("input", () => {
    state.userEditedProtein = true;
    updateProteinBadge();
  });
  document.getElementById("sel-chemistry").addEventListener("change", updateSetupSummary);
  document.getElementById("sel-profile").addEventListener("change", updateSetupSummary);
  document.getElementById("chk-detarget-liver").addEventListener("change", updateSetupSummary);

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

  const header = document.getElementById("cand-thead-row");
  function sortByHeader(e) {
    const th = e.target.closest("th[data-sort]");
    if (!th || (e.type === "keydown" && e.key !== "Enter" && e.key !== " ")) return;
    e.preventDefault();
    const col = th.dataset.sort;
    state.sortAsc = state.sortCol === col ? !state.sortAsc : col === "rank";
    state.sortCol = col;
    renderCandidates();
    if (e.type === "keydown") {
      Array.from(header.children).find((cell) => cell.dataset.sort === col)?.focus();
    }
  }
  header.addEventListener("click", sortByHeader);
  header.addEventListener("keydown", sortByHeader);

  document.getElementById("preset-egfp").addEventListener("click", () => {
    closeOpenDropdowns();
    document.getElementById("setup-fold").open = true;
    const seq = (state.presets?.egfp_reporter?.protein_sequence) || FALLBACK_EGFP;
    document.getElementById("input-protein").value = seq;
    setSelectedEnzymes(["BsaI"]);
    updateProteinBadge();
  });

  document.getElementById("preset-no-m").addEventListener("click", () => {
    closeOpenDropdowns();
    if (state.inFlight) return;
    const seq = (state.presets?.egfp_reporter?.protein_sequence) || FALLBACK_EGFP;
    document.getElementById("input-protein").value = seq.slice(1);
    updateProteinBadge();
    setExecMode("deterministic");
    runInitialTurn("测试缺起始 M 蛋白护栏拦截", { bypassReadOnlyClassifier: true });
  });

  document.getElementById("preset-conflict").addEventListener("click", () => {
    closeOpenDropdowns();
    if (state.inFlight) return;
    const seq = (state.presets?.egfp_reporter?.protein_sequence) || FALLBACK_EGFP;
    document.getElementById("input-protein").value = seq;
    updateProteinBadge();
    setExecMode("deterministic");
    state.forceConflictOnce = true;
    runInitialTurn("测试单候选目标冲突追问", { bypassReadOnlyClassifier: true });
  });

  // Starter Chips
  document.getElementById("starter-egfp")?.addEventListener("click", () => {
    if (state.inFlight) return;
    const seq = (state.presets?.egfp_reporter?.protein_sequence) || FALLBACK_EGFP;
    setEntryMode("protein_only");
    document.getElementById("input-protein").value = seq;
    state.userEditedProtein = false;
    document.getElementById("sel-profile").value = "research_max";
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

  document.getElementById("starter-ref-r")?.addEventListener("click", () => {
    if (state.inFlight) return;
    const seq = (state.presets?.egfp_reporter?.protein_sequence) || FALLBACK_EGFP;
    setEntryMode("with_r");
    document.getElementById("input-protein").value = seq;
    state.userEditedProtein = false;
    document.getElementById("sel-profile").value = "research_max";
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
    document.getElementById("sel-profile").value = "commercial_clean";
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
      document.getElementById("sel-profile").value = "research_max";
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
  wireDialogFallback(dlgSettings);
  wireDialogFallback(dlgSelftest);

  document.getElementById("banner-notes")?.addEventListener("click", (e) => {
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

  document.getElementById("btn-submit-initial").addEventListener("click", () => {
    if (state.inFlight) return;
    runInitialTurn(null, { fromForm: true, ignoreTextDraft: true });
  });
  document.getElementById("btn-send-turn").addEventListener("click", () => {
    submitComposerText(document.getElementById("input-utterance"));
  });
  document.getElementById("input-utterance").addEventListener("keydown", (e) => {
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

  document.getElementById("quick-refine-bspqi").addEventListener("click", () => {
    if (state.inFlight) return;
    setSelectedEnzymes(["BspQI"]);
    runRefineTurn({
      mode: "refine",
      user_utterance: "只改一项：把线性化内切酶从 BsaI 改成避开 BspQI，其余全部锁定项继承不变",
      restriction_enzymes: ["BspQI"],
    });
  });

  document.getElementById("quick-refine-lock-cds").addEventListener("click", () => {
    if (state.inFlight || !state.selectedBundleId) return;
    runRefineTurn({
      mode: "refine",
      user_utterance: `锁定当前候选 ${state.selectedBundleId} 的 CDS 核苷酸序列，仅允许重配 UTR`,
      lock_cds_from_bundle_id: state.selectedBundleId,
    });
  });

  document.getElementById("quick-refine-m15-surrogate").addEventListener("click", () => {
    if (state.inFlight) return;
    document.getElementById("sel-profile").value = "commercial_clean";
    updateSetupSummary();
    runRefineTurn({
      mode: "refine",
      user_utterance: "切换至 M1.5 四头商用洁净分轴代理栈 (5′UTR MRL + CDS + Saluki Full-Tx + fastUTR 3′UTR，零 ViennaRNA)",
      compliance_track: "commercial_clean",
      use_commercial_surrogate: true,
    });
  });

  document.getElementById("quick-refine-commercial").addEventListener("click", () => {
    if (state.inFlight) return;
    document.getElementById("sel-profile").value = "commercial_clean";
    updateSetupSummary();
    runRefineTurn({
      mode: "refine",
      user_utterance: "切换至 commercial_clean 纯商用档位核验组件放行状态",
      compliance_track: "commercial_clean",
    });
  });

  document.getElementById("quick-ask-tradeoffs")?.addEventListener("click", () => {
    if (state.inFlight || !state.intentHistory.length) return;
    setExecMode("llm_agent", { userInitiated: true });
    runRefineTurn(
      null,
      "请调用 explain_tradeoffs 简要总结当前 P1 前沿候选的指标权衡与最佳选择理由。",
      { readOnlyExplanation: true }
    );
  });

  document.getElementById("btn-new-session").addEventListener("click", () => {
    resetProjectWorkspaceState({ silent: false });
  });

  document.getElementById("btn-dl-gb").addEventListener("click", () => downloadExportFile("construct.gb"));
  document.getElementById("btn-dl-rna").addEventListener("click", () => {
    closeOpenDropdowns();
    downloadExportFile("full_rna.fasta");
  });
  document.getElementById("btn-dl-cds").addEventListener("click", () => {
    closeOpenDropdowns();
    downloadExportFile("cds_only.fasta");
  });
  document.getElementById("btn-dl-meta").addEventListener("click", () => {
    closeOpenDropdowns();
    downloadExportFile("chemistry_metadata.json");
  });
  document.getElementById("btn-copy-cli").addEventListener("click", async () => {
    if (!state.lastCliCommand) return;
    await navigator.clipboard.writeText(state.lastCliCommand);
    const b = document.getElementById("btn-copy-cli");
    const old = b.textContent;
    b.textContent = "✓ 已复制 CLI 命令";
    setTimeout(() => {
      b.textContent = old;
    }, 1500);
  });

  document.getElementById("btn-open-settings").addEventListener("click", () => {
    openSettingsModal(false);
  });
  document.getElementById("btn-close-settings").addEventListener("click", () => {
    dlgSettings.close();
  });
  document.getElementById("btn-save-settings").addEventListener("click", saveSettingsFromModal);

  document.getElementById("btn-run-selftest").addEventListener("click", () => {
    closeOpenDropdowns();
    dlgSelftest.showModal();
    runAutomatedBrowserSelfTest();
  });
  document.getElementById("btn-trigger-selftest").addEventListener("click", runAutomatedBrowserSelfTest);
  document.getElementById("btn-close-selftest").addEventListener("click", () => {
    dlgSelftest.close();
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

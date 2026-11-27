const DEFAULT_API_BASE = "https://149-248-39-200.sslip.io:8443";
const TRUSTED_PRODUCTION_ORIGINS = new Set([
  "https://149-248-39-200.sslip.io:8443",
]);
const FALLBACK_EGFP =
  "MVSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITLGMDELYK";

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
  sessionEpoch: 1,
  turnSequence: 0,
  appliedTurnSequence: 0,
  apiBase: DEFAULT_API_BASE,
  accessToken: "",
  llmProvider: "deepseek",
  byoKey: "",
  projectId: `proj_${Math.random().toString(36).slice(2, 8)}`,
  projectKey: mintProjectKey(),
  entryMode: "protein_only",
  presets: null,
  intentHistory: [],
  completedRevisions: [],
  activeVersion: 0,
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

function applyTheme(themeName) {
  state.theme = themeName === "graphite" ? "graphite" : "light";
  document.documentElement.dataset.theme = state.theme;
  localStorage.setItem("mrnax_theme", state.theme);
  const lbl = document.getElementById("txt-theme-label");
  if (lbl) {
    lbl.textContent = state.theme === "graphite" ? "☀ 清爽蓝白" : "◐ 石墨深色";
  }
  renderParetoScatter();
}

function initConfig() {
  const savedTheme = localStorage.getItem("mrnax_theme");
  applyTheme(savedTheme === "graphite" ? "graphite" : "light");

  const url = new URL(window.location.href);
  const rawQueryApi = url.searchParams.get("api");
  const rawStoredApi = localStorage.getItem("mrnax_api_base");
  if (rawQueryApi && isTrustedApiOrigin(rawQueryApi)) {
    state.apiBase = new URL(rawQueryApi).origin;
  } else if (rawStoredApi && isTrustedApiOrigin(rawStoredApi)) {
    state.apiBase = new URL(rawStoredApi).origin;
  } else {
    state.apiBase = DEFAULT_API_BASE;
    localStorage.removeItem("mrnax_api_base");
  }

  // Purge any legacy persisted tokens from earlier browser sessions
  localStorage.removeItem("mrnax_access_token");
  sessionStorage.removeItem("mrnax_byo_key");

  if (window.location.hash.startsWith("#token=")) {
    state.accessToken = decodeURIComponent(window.location.hash.slice(7)).trim();
    history.replaceState(null, "", window.location.pathname + window.location.search);
  }

  setProviderPreference(localStorage.getItem("mrnax_llm_provider"));

  document.getElementById("cfg-api-base").value = state.apiBase;
  document.getElementById("cfg-access-token").value = state.accessToken;
  document.getElementById("cfg-byo-key").value = state.byoKey;
  document.getElementById("txt-project-id").textContent = state.projectId;
  updateActiveRevBadge();
}

function updateActiveRevBadge() {
  const revEl = document.getElementById("badge-active-rev");
  if (revEl) {
    revEl.textContent = `v${String(state.activeVersion).padStart(2, "0")}`;
  }
}

function setProviderPreference(value) {
  const select = document.getElementById("cfg-llm-provider");
  const provider = (value || "").trim().toLowerCase();
  const supported = Array.from(select.options, (option) => option.value);
  state.llmProvider = supported.includes(provider) ? provider : "deepseek";
  select.value = state.llmProvider;
  localStorage.setItem("mrnax_llm_provider", state.llmProvider);
}

function buildHeaders(path, method, explicitProjectKey, includeAuth) {
  const h = { "Content-Type": "application/json" };
  if (includeAuth && isTrustedApiOrigin(state.apiBase)) {
    h["X-MRNAX-Project-Key"] = explicitProjectKey || state.projectKey;
    if (state.accessToken) h["X-MRNAX-Access-Token"] = state.accessToken;
    if (state.byoKey && method === "POST" && /^\/v1\/projects\/[^/]+\/agent-chat$/.test(path)) {
      h["X-MRNAX-LLM-Key"] = state.byoKey;
      h["X-MRNAX-LLM-Provider"] = state.llmProvider;
    }
  }
  return h;
}

async function apiFetch(path, options = {}) {
  const { explicitProjectKey, includeAuth = true, ...fetchOpts } = options;
  const resp = await fetch(`${state.apiBase}${path}`, {
    ...fetchOpts,
    headers: {
      ...buildHeaders(path, fetchOpts.method, explicitProjectKey, includeAuth),
      ...(options.headers || {}),
    },
  });
  const text = await resp.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = { detail: text };
  }
  if (!resp.ok) {
    throw new Error((data && data.detail) || `HTTP ${resp.status}`);
  }
  return data;
}

function getSelectedEnzymes() {
  return Array.from(document.querySelectorAll("#enzyme-chips .chip.selected")).map(
    (el) => el.dataset.enz
  );
}

function setSelectedEnzymes(list) {
  document.querySelectorAll("#enzyme-chips .chip").forEach((el) => {
    el.classList.toggle("selected", list.includes(el.dataset.enz));
  });
  updateSetupSummary();
}

function updateSetupSummary() {
  const raw = document.getElementById("input-protein").value.trim().toUpperCase();
  const chem = document.getElementById("sel-chemistry").value;
  const prof = document.getElementById("sel-profile").value;
  const enz = getSelectedEnzymes();
  const detarget = document.getElementById("chk-detarget-liver").checked;
  const summaryEl = document.getElementById("setup-collapsed-summary");
  if (summaryEl) {
    const enzLabel = enz.length ? `0-cut:${enz.join("+")}` : "无酶切禁区";
    summaryEl.textContent = `${raw.length} aa · ${chem} · ${prof} · ${enzLabel}${detarget ? " · miR-122" : ""}`;
  }
}

function updateProteinBadge() {
  const raw = document.getElementById("input-protein").value.trim().toUpperCase();
  const badge = document.getElementById("badge-aa-len");
  const startsM = raw.startsWith("M");
  badge.textContent = `${raw.length} aa · ${startsM ? "M✓" : "缺起始M⚠"}`;
  badge.className = startsM ? "badge mono ok" : "badge mono warn";
  updateSetupSummary();
}

function appendStreamCard(kind, metaLeft, metaRight, bodyHtml) {
  const box = document.getElementById("chat-stream");
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
}

function renderIntentLineage() {
  const bar = document.getElementById("intent-lineage-bar");
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
    wrap.innerHTML = `<div class="empty-hint mono">点击上方「生成候选构建」绘制多目标帕累托前沿</div>`;
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
  const H = Math.max(216, 80 + Math.ceil(4 * pts.length / 22) * spacing);
  const padL = 56;
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

  // Keep true coordinates for the axes and frontier. Displace colliding markers only.
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
      <text x="${padL + plotW / 2}" y="${H - 4}" text-anchor="middle" fill="var(--ink-secondary)" font-family="var(--font-mono)" font-size="10.5">${esc(axX.label)}</text>
      <text x="13" y="${padT + plotH / 2}" text-anchor="middle" transform="rotate(-90 13 ${padT + plotH / 2})" fill="var(--ink-secondary)" font-family="var(--font-mono)" font-size="10.5">${esc(axY.label)}</text>
    </svg>
  `;

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
  document.getElementById("txt-selected-bundle").textContent = "未选择候选";
  document.getElementById("chain-picker").hidden = true;
  document.getElementById("inspect-chain").replaceChildren();
  document.getElementById("enzyme-clearance-rail").innerHTML = "";
  document.getElementById("ruler-row").innerHTML = "";
  document.getElementById("prop-track-wrap").innerHTML =
    `<div class="empty-hint mono">生成后展示五段真实比例分子图谱</div>`;
  document.getElementById("seg-bar").innerHTML = "";
  document.getElementById("seg-inspector").textContent =
    "选择候选构建后自动对齐显示区段核苷酸与三联体密码子翻译轨";
}

function updateExportButtonLabels() {
  const gbBtn = document.getElementById("btn-dl-gb");
  const hasCand = Boolean(state.selectedBundleId && state.candidates.length);
  ["btn-dl-gb", "btn-dl-rna", "btn-dl-cds", "btn-dl-meta"].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.disabled = !hasCand;
  });
  document.getElementById("btn-copy-cli").disabled = !state.lastCliCommand;
  if (gbBtn) {
    if (hasCand) {
      const revTag = `v${String(state.activeVersion).padStart(2, "0")}`;
      gbBtn.textContent = `GenBank (${revTag}) ↓`;
      gbBtn.title = `${state.selectedBundleId}.gb`;
      gbBtn.setAttribute("aria-label", `GenBank ${revTag} · ${state.selectedBundleId}.gb`);
    } else {
      gbBtn.textContent = "GenBank (.gb) ↓";
      gbBtn.removeAttribute("title");
      gbBtn.removeAttribute("aria-label");
    }
  }
}

function renderCandidates() {
  const tbody = document.getElementById("cand-tbody");
  const countEl = document.getElementById("txt-cand-count");
  countEl.textContent = `${state.candidates.length} 条合格候选`;

  const axes = resolveMetricAxes(state.candidates);
  const header = document.getElementById("cand-thead-row");
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

  if (!state.candidates.length) {
    tbody.innerHTML = `<tr><td colspan="${header.children.length}" class="empty-cell mono">当前无合格推荐候选（详见上方状态横幅）</td></tr>`;
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

  tbody.innerHTML = "";
  sorted.forEach((c) => {
    const tr = document.createElement("tr");
    tr.tabIndex = 0;
    if (c.bundle_id === state.selectedBundleId) tr.classList.add("selected-row");
    const rankNum = c.pareto_rank;
    const isRec = rankNum === 1 && recSet.has(c.bundle_id);
    const metricCells = axes.map((axis, i) =>
      `<td class="num-col">${buildSparkCell(readAxisValue(c, axis), axis, ranges[i].min, ranges[i].max)}</td>`
    ).join("");

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
  const activeEnzymes = new Set(getSelectedEnzymes());
  rail.innerHTML = Object.entries(RESTRICTION_SITES_RNA)
    .map(([enz, motifs]) => {
      const cuts = countMotifOccurrences(fullSeq, motifs);
      const cls = cuts === 0 ? "clear" : "cut";
      const reqMark = activeEnzymes.has(enz) ? " [禁]" : "";
      return `<span class="enz-tag ${cls}">${esc(enz)}${esc(reqMark)}: ${cuts} cuts ${cuts === 0 ? "✓" : "⚠"}</span>`;
    })
    .join("");
}

function renderTrueScaleMap(cand) {
  const ruler = document.getElementById("ruler-row");
  const propWrap = document.getElementById("prop-track-wrap");
  const cardsBar = document.getElementById("seg-bar");
  const total = Math.max(1, cand.length_nt);

  ruler.innerHTML = [0, 0.25, 0.5, 0.75, 1]
    .map((f) => `<span>${Math.round(total * f)} nt</span>`)
    .join("");

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

  propWrap.innerHTML = stopPinHtml;
  propWrap.appendChild(propBar);

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

  document.getElementById("seg-inspector").textContent = `${header}\n${"─".repeat(78)}\n${body}`;
}

function selectCandidate(bundleId) {
  state.selectedBundleId = bundleId;
  renderCandidates();
  const cand = state.candidates.find((c) => c.bundle_id === bundleId);
  if (!cand) return;

  const chains = cand.molecules?.length ? cand.molecules : [cand];
  const select = document.getElementById("inspect-chain");
  select.replaceChildren();
  chains.forEach((chain, i) => {
    const option = document.createElement("option");
    option.value = String(i);
    option.textContent = chain.molecule_id || `分子 ${i + 1}`;
    select.appendChild(option);
  });
  document.getElementById("chain-picker").hidden = chains.length < 2;
  function inspectChain(index) {
    const chain = chains[index];
    const label = chains.length > 1 ? ` · ${chain.molecule_id}` : "";
    document.getElementById("txt-selected-bundle").textContent =
      `${cand.bundle_id}${label} · ${chain.length_nt} nt · GC ${chain.full_gc_pct}% · sha256:${cand.version_digest.slice(0, 10)}`;
    renderEnzymeClearance(chain.full_sequence);
    const segment = chain.segments.find((s) => s.segment_type === state.selectedSegType)
      || chain.segments.find((s) => s.segment_type === "cds") || chain.segments[0];
    if (segment) inspectSegment(chain, segment);
    else renderTrueScaleMap(chain);
  }
  select.onchange = () => inspectChain(Number(select.value));
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
  document.getElementById("ledger-ver-span").textContent = `${vPrev} → ${vCurr}`;

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
  deltaEl.innerHTML = deltaParts.join(" · ");

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
  locksBar.innerHTML = lockTokens.map((t) => `<span class="lock-token">${esc(t)}</span>`).join("");

  const dualWrap = document.getElementById("dual-scale-tracks");
  const prevChains = prev.topCandidate.molecules || [prev.topCandidate];
  const currChains = curr.topCandidate.molecules || [curr.topCandidate];
  const maxNt = Math.max(...prevChains.concat(currChains).map((chain) => chain.length_nt), 1);

  const buildScaledRow = (verTag, cand, other, isCurrent) => {
    const otherSeqMap = new Map(other?.segments.map((s) =>
      [s.segment_type, other.full_sequence.slice(s.start, s.end)]
    ));
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
  dualWrap.innerHTML = rows.join("");
}

function applyTurnResult(res, userText, sequence) {
  if (!res || !res.intent) {
    if (state.activeVersion > 0 || sequence < state.appliedTurnSequence) {
      return false;
    }
    state.appliedTurnSequence = sequence;
    if (res && res.assistant_reply) {
      appendStreamCard("ok", "Agent 回复 · 待立项", "INFO", esc(res.assistant_reply));
    }
    return true;
  }
  const version = res.intent.intent_version;
  if (version < state.activeVersion || (version === state.activeVersion && sequence < state.appliedTurnSequence)) {
    return false;
  }
  state.appliedTurnSequence = sequence;
  state.activeVersion = version;
  updateActiveRevBadge();
  let intentAdvanced = false;
  if (res.intent && !state.intentHistory.some((it) => it.intent_id === res.intent.intent_id)) {
    state.intentHistory.push(res.intent);
    intentAdvanced = true;
    renderIntentLineage();
    document.getElementById("sel-profile").value = res.intent.compliance_track;
    document.getElementById("sel-chemistry").value = res.intent.uridine_chemistry;
    updateSetupSummary();
  }

  const banner = document.getElementById("banner-notes");

  if (res.status === "NEEDS_CLARIFICATION") {
    state.latestReport = null;
    state.candidates = [];
    state.lastCliCommand = "";
    state.cachedExports = {};
    renderCandidates();
    document.getElementById("quick-refine-lock-cds").disabled = true;
    document.getElementById("revision-ledger").hidden = true;
    document.getElementById("badge-outcome").textContent = res.intent.status;
    document.getElementById("badge-outcome").className = "badge mono warn";
    const qs = (res.blocking_questions || []).map((q) => `<li>${esc(q)}</li>`).join("");
    banner.hidden = false;
    banner.innerHTML = `<strong>编译前护栏拦截 (${esc(res.intent.status)})：</strong><ul style="margin:4px 0 0 18px;">${qs}</ul>`;
    appendStreamCard(
      "warn",
      `安全护栏拦截 · Intent v${res.intent.intent_version}`,
      res.intent.status,
      `<strong>拦截待确认项：</strong><ul style="margin:4px 0 0 18px;">${qs}</ul>`
    );
    return true;
  }

  state.latestReport = res.report;
  state.candidates = res.candidates || [];
  state.lastCliCommand = res.cli_replay_command || "";

  const outcome = (res.report && res.report.outcome_status) || "COMPLETED";
  const badgeOut = document.getElementById("badge-outcome");
  badgeOut.textContent = outcome;
  badgeOut.className = outcome.includes("UNCOMPARABLE") || !state.candidates.length ? "badge mono warn" : "badge mono ok";

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
  if (notes.length) {
    banner.hidden = false;
    banner.innerHTML = notes.join("<br/>");
  } else {
    banner.hidden = true;
    banner.innerHTML = "";
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
  const llmReplyHtml = res.assistant_reply
    ? `<div style="margin-top:5px; white-space:pre-wrap;">${esc(res.assistant_reply)}</div>`
    : "";

  appendStreamCard(
    "agent",
    `Harness 完成 · Intent v${res.intent ? res.intent.intent_version : "?"}`,
    `${state.candidates.length} 条合格`,
    `<div>指令：<em>${esc(userText)}</em> → <code>${esc(outcome)}</code></div>${diffHtml}${traceBlock}${llmReplyHtml}`
  );

  ["quick-refine-bspqi", "quick-refine-lock-cds", "quick-refine-m15-surrogate", "quick-refine-commercial"].forEach(
    (id) => {
      document.getElementById(id).disabled = state.intentHistory.length === 0;
    }
  );

  const recFirst = res.report?.recommended_bundle_ids?.[0] || state.candidates[0]?.bundle_id || null;
  state.selectedBundleId = recFirst;

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

    const setupFold = document.getElementById("setup-fold");
    if (setupFold && state.completedRevisions.length === 1) {
      setupFold.open = false;
    }
  }

  renderCandidates();
  if (recFirst && state.candidates.length > 0) {
    selectCandidate(recFirst);
  }
  return true;
}

async function refreshSystemStatus() {
  const dot = document.getElementById("dot-gateway");
  const txt = document.getElementById("txt-gateway");
  const t0 = performance.now();
  try {
    const st = await apiFetch("/v1/system/status", { includeAuth: false });
    const ms = Math.round(performance.now() - t0);
    dot.className = "dot ok";
    txt.textContent = `${st.hostname} · ${ms}ms`;
    document.getElementById("pill-engine").textContent =
      `Engine: ${st.default_profile} · ViennaRNA ${st.viennarna_version || "lazy"}`;

    const presets = await apiFetch("/v1/presets", { includeAuth: false });
    state.presets = presets;
    if (!document.getElementById("input-protein").value.trim()) {
      document.getElementById("input-protein").value = presets.egfp_reporter.protein_sequence;
      updateProteinBadge();
    }
    document.getElementById("txt-ref-r-preview").textContent =
      `${presets.preset_reference_r.bundle_id} (${presets.preset_reference_r.length_nt} nt) · ${presets.preset_reference_r.sequence_preview}`;
  } catch (err) {
    dot.className = "dot err";
    txt.textContent = `离线 (${err.message})`;
  }
}

async function runInitialTurn() {
  const sequence = ++state.turnSequence;
  const reqEpoch = state.sessionEpoch;
  const reqProjectId = state.projectId;
  const prot = document.getElementById("input-protein").value.trim();
  const chem = document.getElementById("sel-chemistry").value;
  const prof = document.getElementById("sel-profile").value;
  const enz = getSelectedEnzymes();
  const detarget = document.getElementById("chk-detarget-liver").checked;
  const late = document.getElementById("chk-late-signal").checked;
  const useR = state.entryMode === "with_r";
  const conflict = state.forceConflictOnce;
  state.forceConflictOnce = false;

  const utterance =
    document.getElementById("input-utterance").value.trim() ||
    (useR
      ? `基于参考构建 R 设计 eGFP mRNA (${chem})，避开 ${enz.join("/")}，生成保守最小改动 A 与探索方案 B`
      : `从零设计 eGFP 报告蛋白 mRNA (${chem})，蛋白 100% 锁定，${detarget ? "开启肝脏去靶向，" : ""}避开 ${enz.join("/")}`);

  appendStreamCard("user", "首轮编译 (Initial)", state.entryMode, esc(utterance));
  const btn = document.getElementById("btn-submit-initial");
  btn.disabled = true;
  try {
    const execMode = document.getElementById("sel-exec-mode").value;
    let res;
    if (execMode === "llm_agent") {
      res = await apiFetch(`/v1/projects/${reqProjectId}/agent-chat`, {
        method: "POST",
        body: JSON.stringify({
          message: `${utterance}（目标蛋白序列：${prot}）`,
          provider: state.llmProvider,
        }),
      });
    } else {
      res = await apiFetch(`/v1/projects/${reqProjectId}/turn`, {
        method: "POST",
        body: JSON.stringify({
          mode: "initial",
          user_utterance: utterance,
          protein_sequence: prot,
          uridine_chemistry: chem,
          compliance_track: prof,
          use_commercial_surrogate: prof === "commercial_clean",
          restriction_enzymes: enz,
          detarget_liver: detarget,
          prioritize_late_signal: late,
          conflicting_single_pick: conflict,
          use_preset_reference_r: useR,
        }),
      });
    }
    if (reqEpoch !== state.sessionEpoch || reqProjectId !== state.projectId) return;
    applyTurnResult(res, utterance, sequence);
  } catch (err) {
    if (reqEpoch === state.sessionEpoch && sequence >= state.appliedTurnSequence) {
      appendStreamCard("warn", "请求异常", "ERROR", esc(err.message));
    }
  } finally {
    btn.disabled = false;
  }
}

async function runRefineTurn(overridePayload = null) {
  if (!state.intentHistory.length) {
    return runInitialTurn();
  }
  const sequence = ++state.turnSequence;
  const reqEpoch = state.sessionEpoch;
  const reqProjectId = state.projectId;
  const utterance =
    (overridePayload && overridePayload.user_utterance) ||
    document.getElementById("input-utterance").value.trim() ||
    `微调当前设计：内切酶避开 ${getSelectedEnzymes().join("/")}，其余锁定继承`;

  appendStreamCard("user", "单点增量微调 (Refine)", `基于 v${String(state.intentHistory.length).padStart(2, "0")}`, esc(utterance));
  const btn = document.getElementById("btn-send-turn");
  btn.disabled = true;
  try {
    const execMode = document.getElementById("sel-exec-mode").value;
    let res;
    if (execMode === "llm_agent" && !overridePayload) {
      res = await apiFetch(`/v1/projects/${reqProjectId}/agent-chat`, {
        method: "POST",
        body: JSON.stringify({ message: utterance, provider: state.llmProvider }),
      });
    } else {
      const payload = overridePayload || {
        mode: "refine",
        user_utterance: utterance,
        restriction_enzymes: getSelectedEnzymes(),
      };
      res = await apiFetch(`/v1/projects/${reqProjectId}/turn`, {
        method: "POST",
        body: JSON.stringify(payload),
      });
    }
    if (reqEpoch !== state.sessionEpoch || reqProjectId !== state.projectId) return;
    if (applyTurnResult(res, utterance, sequence)) document.getElementById("input-utterance").value = "";
  } catch (err) {
    if (reqEpoch === state.sessionEpoch && sequence >= state.appliedTurnSequence) {
      appendStreamCard("warn", "微调异常", "ERROR", esc(err.message));
    }
  } finally {
    btn.disabled = false;
  }
}

async function downloadExportFile(filename) {
  const bid = state.selectedBundleId;
  const projId = state.projectId;
  const revNum = state.activeVersion;
  const pKey = state.projectKey;
  if (!bid) return;
  const cacheKey = `${projId}:${revNum}:${bid}`;
  if (!state.cachedExports[cacheKey]) {
    const result = await apiFetch(
      `/v1/projects/${projId}/bundles/${encodeURIComponent(bid)}/export`,
      { explicitProjectKey: pKey }
    );
    if (projId !== state.projectId || revNum !== state.activeVersion || bid !== state.selectedBundleId) return;
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
  const egfp = (state.presets && state.presets.egfp_reporter.protein_sequence) || FALLBACK_EGFP;

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
  if (!("closedBy" in HTMLDialogElement.prototype)) {
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

function wireEvents() {
  document.getElementById("btn-theme-toggle").addEventListener("click", () => {
    applyTheme(state.theme === "graphite" ? "light" : "graphite");
  });

  document.addEventListener("click", (e) => {
    const insideDropdown = e.target.closest("details.dropdown-menu");
    closeOpenDropdowns(insideDropdown);
  });

  document.getElementById("input-protein").addEventListener("input", updateProteinBadge);
  document.getElementById("sel-chemistry").addEventListener("change", updateSetupSummary);
  document.getElementById("sel-profile").addEventListener("change", updateSetupSummary);
  document.getElementById("chk-detarget-liver").addEventListener("change", updateSetupSummary);

  document.querySelectorAll(".seg-tabs .tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".seg-tabs .tab").forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      state.entryMode = tab.dataset.entry;
      document.getElementById("box-ref-r").hidden = state.entryMode !== "with_r";
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
    const seq = (state.presets && state.presets.egfp_reporter.protein_sequence) || FALLBACK_EGFP;
    document.getElementById("input-protein").value = seq;
    setSelectedEnzymes(["BsaI"]);
    updateProteinBadge();
  });

  document.getElementById("preset-no-m").addEventListener("click", () => {
    closeOpenDropdowns();
    const seq = (state.presets && state.presets.egfp_reporter.protein_sequence) || FALLBACK_EGFP;
    document.getElementById("input-protein").value = seq.slice(1);
    updateProteinBadge();
    runInitialTurn();
  });

  document.getElementById("preset-conflict").addEventListener("click", () => {
    closeOpenDropdowns();
    const seq = (state.presets && state.presets.egfp_reporter.protein_sequence) || FALLBACK_EGFP;
    document.getElementById("input-protein").value = seq;
    updateProteinBadge();
    state.forceConflictOnce = true;
    runInitialTurn();
  });

  document.getElementById("btn-submit-initial").addEventListener("click", runInitialTurn);
  document.getElementById("btn-send-turn").addEventListener("click", () => runRefineTurn());
  document.getElementById("input-utterance").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      runRefineTurn();
    }
  });

  document.getElementById("quick-refine-bspqi").addEventListener("click", () => {
    setSelectedEnzymes(["BspQI"]);
    runRefineTurn({
      mode: "refine",
      user_utterance: "只改一项：把线性化内切酶从 BsaI 改成避开 BspQI，其余全部锁定项继承不变",
      restriction_enzymes: ["BspQI"],
    });
  });

  document.getElementById("quick-refine-lock-cds").addEventListener("click", () => {
    if (!state.selectedBundleId) return;
    runRefineTurn({
      mode: "refine",
      user_utterance: `锁定当前候选 ${state.selectedBundleId} 的 CDS 核苷酸序列，仅允许重配 UTR`,
      lock_cds_from_bundle_id: state.selectedBundleId,
    });
  });

  document.getElementById("quick-refine-m15-surrogate").addEventListener("click", () => {
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
    document.getElementById("sel-profile").value = "commercial_clean";
    updateSetupSummary();
    runRefineTurn({
      mode: "refine",
      user_utterance: "切换至 commercial_clean 纯商用档位核验组件放行状态",
      compliance_track: "commercial_clean",
    });
  });

  document.getElementById("btn-new-session").addEventListener("click", () => {
    state.sessionEpoch += 1;
    state.projectId = `proj_${Math.random().toString(36).slice(2, 8)}`;
    state.projectKey = mintProjectKey();
    state.intentHistory = [];
    state.completedRevisions = [];
    state.activeVersion = 0;
    state.appliedTurnSequence = 0;
    state.latestReport = null;
    state.candidates = [];
    state.selectedBundleId = null;
    state.lastCliCommand = "";
    state.cachedExports = {};
    state.logCount = 0;

    document.getElementById("txt-project-id").textContent = state.projectId;
    document.getElementById("badge-outcome").textContent = "IDLE";
    document.getElementById("badge-outcome").className = "badge mono";
    document.getElementById("banner-notes").hidden = true;
    document.getElementById("banner-notes").innerHTML = "";
    document.getElementById("revision-ledger").hidden = true;
    document.getElementById("setup-fold").open = true;
    document.getElementById("chat-stream").innerHTML = "";
    document.getElementById("txt-log-count").textContent = "0";
    clearSelectedConstructUI();
    ["quick-refine-bspqi", "quick-refine-lock-cds", "quick-refine-m15-surrogate", "quick-refine-commercial"].forEach(
      (id) => {
        document.getElementById(id).disabled = true;
      }
    );
    updateActiveRevBadge();
    renderIntentLineage();
    renderCandidates();
    appendStreamCard("agent", "新会话已创建", state.projectId, "已切换至全新隔离项目空间。");
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

  const dlgSettings = document.getElementById("modal-settings");
  const dlgSelftest = document.getElementById("modal-selftest");
  wireDialogFallback(dlgSettings);
  wireDialogFallback(dlgSelftest);

  document.getElementById("btn-open-settings").addEventListener("click", () => {
    dlgSettings.showModal();
  });
  document.getElementById("btn-close-settings").addEventListener("click", () => {
    dlgSettings.close();
  });
  document.getElementById("btn-save-settings").addEventListener("click", () => {
    const candidateApi =
      document.getElementById("cfg-api-base").value.trim().replace(/\/+$/, "") || DEFAULT_API_BASE;
    if (!isTrustedApiOrigin(candidateApi)) {
      appendStreamCard(
        "warn",
        "安全策略阻断",
        "UNTRUSTED_ORIGIN",
        `已拒绝不受信任的 API 源 <code>${esc(candidateApi)}</code>，回退至默认受信任网关。`
      );
      state.apiBase = DEFAULT_API_BASE;
      document.getElementById("cfg-api-base").value = DEFAULT_API_BASE;
      localStorage.removeItem("mrnax_api_base");
    } else {
      state.apiBase = new URL(candidateApi).origin;
      localStorage.setItem("mrnax_api_base", state.apiBase);
    }
    // Keep accessToken and byoKey strictly in tab memory (zero localStorage / sessionStorage)
    state.accessToken = document.getElementById("cfg-access-token").value.trim();
    setProviderPreference(document.getElementById("cfg-llm-provider").value);
    state.byoKey = document.getElementById("cfg-byo-key").value.trim();
    dlgSettings.close();
    refreshSystemStatus();
  });

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
  wireEvents();
  document.getElementById("input-protein").value = FALLBACK_EGFP;
  updateProteinBadge();
  refreshSystemStatus();
});

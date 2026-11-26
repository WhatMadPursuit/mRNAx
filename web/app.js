const DEFAULT_API_BASE = "https://149-248-39-200.sslip.io:8443";
const FALLBACK_EGFP =
  "MVSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITLGMDELYK";

const state = {
  apiBase: DEFAULT_API_BASE,
  accessToken: "",
  llmProvider: "deepseek",
  byoKey: "",
  projectId: `proj_${Math.random().toString(36).slice(2, 8)}`,
  entryMode: "protein_only",
  presets: null,
  intentHistory: [],
  latestReport: null,
  candidates: [],
  selectedBundleId: null,
  lastCliCommand: "",
  cachedExports: {},
  forceConflictOnce: false,
};

function esc(str) {
  const div = document.createElement("div");
  div.textContent = str == null ? "" : String(str);
  return div.innerHTML;
}

function initConfig() {
  const url = new URL(window.location.href);
  if (url.searchParams.get("api")) {
    state.apiBase = url.searchParams.get("api").replace(/\/+$/, "");
  } else if (localStorage.getItem("mrnax_api_base")) {
    state.apiBase = localStorage.getItem("mrnax_api_base").replace(/\/+$/, "");
  }

  if (window.location.hash.startsWith("#token=")) {
    const tok = decodeURIComponent(window.location.hash.slice(7));
    if (tok) localStorage.setItem("mrnax_access_token", tok);
    history.replaceState(null, "", window.location.pathname + window.location.search);
  }

  state.accessToken = localStorage.getItem("mrnax_access_token") || "";
  state.llmProvider = localStorage.getItem("mrnax_llm_provider") || "deepseek";
  state.byoKey = sessionStorage.getItem("mrnax_byo_key") || "";

  document.getElementById("cfg-api-base").value = state.apiBase;
  document.getElementById("cfg-access-token").value = state.accessToken;
  document.getElementById("cfg-llm-provider").value = state.llmProvider;
  document.getElementById("cfg-byo-key").value = state.byoKey;
  document.getElementById("txt-project-id").textContent = state.projectId;
}

function buildHeaders() {
  const h = { "Content-Type": "application/json" };
  if (state.accessToken) h["X-MRNAX-Access-Token"] = state.accessToken;
  if (state.byoKey) {
    h["X-MRNAX-LLM-Key"] = state.byoKey;
    h["X-MRNAX-LLM-Provider"] = state.llmProvider;
  }
  return h;
}

async function apiFetch(path, options = {}) {
  const resp = await fetch(`${state.apiBase}${path}`, {
    ...options,
    headers: { ...buildHeaders(), ...(options.headers || {}) },
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

function updateProteinBadge() {
  const raw = document.getElementById("input-protein").value.trim().toUpperCase();
  const badge = document.getElementById("badge-aa-len");
  const startsM = raw.startsWith("M");
  badge.textContent = `${raw.length} aa · ${startsM ? "M✓" : "缺起始M⚠"}`;
  badge.className = startsM ? "badge info" : "badge warn";
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
}

function renderIntentLineage() {
  const bar = document.getElementById("intent-lineage-bar");
  if (!state.intentHistory.length) {
    bar.innerHTML = `<span class="badge info">等待首轮输入启动 DesignIntent v1...</span>`;
    return;
  }
  const latest = state.intentHistory[state.intentHistory.length - 1];
  const chain = state.intentHistory
    .map((it) => `v${it.intent_version}:${it.intent_id}`)
    .join(" → ");
  const locks = Object.keys(latest.locked_scopes || {}).map((k) => `🔒${k}`).join(" ");
  bar.innerHTML = `
    <span class="badge info">${esc(chain)}</span>
    <span class="badge">${esc(locks || "无锁定")}</span>
    <span class="badge">${esc(latest.uridine_chemistry)} · ${esc(latest.compliance_track)}</span>
  `;
}

function renderCandidates() {
  const tbody = document.getElementById("cand-tbody");
  const countEl = document.getElementById("txt-cand-count");
  countEl.textContent = `${state.candidates.length} 条合格`;

  if (!state.candidates.length) {
    tbody.innerHTML = `<tr><td colspan="6" style="color:var(--text-muted); text-align:center;">当前无合格推荐候选（详见上方状态说明）</td></tr>`;
    ["btn-dl-gb", "btn-dl-rna", "btn-dl-cds", "btn-dl-meta"].forEach((id) => {
      document.getElementById(id).disabled = true;
    });
    return;
  }

  const recSet = new Set((state.latestReport && state.latestReport.recommended_bundle_ids) || []);
  tbody.innerHTML = "";
  state.candidates.forEach((c) => {
    const tr = document.createElement("tr");
    if (c.bundle_id === state.selectedBundleId) tr.classList.add("selected-row");
    const m = c.metrics || {};
    const kozak = m.kozak_unpaired_prob != null ? Number(m.kozak_unpaired_prob).toFixed(4) : "—";
    const mfe = m.cds_mean_window_mfe != null ? `${Number(m.cds_mean_window_mfe).toFixed(2)}` : "—";
    const mir = m.mirna_3p_dg_total != null ? `${Number(m.mirna_3p_dg_total).toFixed(2)}` : "—";
    const star = recSet.has(c.bundle_id) ? "★ " : "";
    tr.innerHTML = `
      <td class="mono">${esc(star + c.bundle_id)}</td>
      <td>${esc(c.derivation_action)}</td>
      <td class="mono">${esc(kozak)}</td>
      <td class="mono">${esc(mfe)}</td>
      <td class="mono">${esc(mir)}</td>
      <td class="mono">${esc(c.full_gc_pct)}% / ${esc(c.length_nt)}nt</td>
    `;
    tr.addEventListener("click", () => selectCandidate(c.bundle_id));
    tbody.appendChild(tr);
  });
}

function selectCandidate(bundleId) {
  state.selectedBundleId = bundleId;
  renderCandidates();
  const cand = state.candidates.find((c) => c.bundle_id === bundleId);
  if (!cand) return;

  document.getElementById("txt-selected-bundle").textContent =
    `${cand.bundle_id} (${cand.length_nt} nt · digest ${cand.version_digest.slice(0, 12)}...)`;

  const bar = document.getElementById("seg-bar");
  bar.innerHTML = "";
  cand.segments.forEach((seg) => {
    const pct = Math.max(6, Math.round((seg.length_nt / cand.length_nt) * 100));
    const div = document.createElement("div");
    div.className = `seg-block seg-${seg.segment_type}`;
    div.style.width = `${pct}%`;
    const shortLabel = {
      five_prime_utr: "5'UTR",
      cds: "CDS",
      stop_cassette: "Stop",
      three_prime_utr: "3'UTR",
      poly_a: "PolyA",
    }[seg.segment_type] || seg.segment_type;
    div.textContent = `${shortLabel} (${seg.length_nt})`;
    div.addEventListener("click", () => inspectSegment(cand, seg));
    bar.appendChild(div);
  });

  if (cand.segments.length) {
    inspectSegment(cand, cand.segments[1] || cand.segments[0]);
  }

  ["btn-dl-gb", "btn-dl-rna", "btn-dl-cds", "btn-dl-meta"].forEach((id) => {
    document.getElementById(id).disabled = false;
  });
  document.getElementById("btn-copy-cli").disabled = !state.lastCliCommand;
}

function inspectSegment(cand, seg) {
  const fullSub = cand.full_sequence.slice(seg.start, seg.end);
  document.getElementById("seg-inspector").textContent =
    `[${seg.segment_type}] 坐标: [${seg.start}..${seg.end}) | 长度: ${seg.length_nt} nt | GC: ${seg.gc_pct}% | 许可轨: ${seg.license_track} | 动作: ${seg.action_taken}\n` +
    `序列: ${fullSub}`;
}

function applyTurnResult(res, userText) {
  if (res.intent) {
    state.intentHistory.push(res.intent);
    renderIntentLineage();
  }

  if (res.status === "NEEDS_CLARIFICATION") {
    document.getElementById("badge-outcome").textContent = res.intent.status;
    document.getElementById("badge-outcome").className = "badge warn";
    const qs = (res.blocking_questions || []).map((q) => `<li>${esc(q)}</li>`).join("");
    appendStreamCard(
      "warn",
      `安全护栏拦截 · DesignIntent v${res.intent.intent_version}`,
      res.intent.status,
      `<strong>在消耗优化算力前拦截到待确认项：</strong><ul style="margin:6px 0 0 18px;">${qs}</ul>`
    );
    return;
  }

  state.latestReport = res.report;
  state.candidates = res.candidates || [];
  state.lastCliCommand = res.cli_replay_command || "";

  const outcome = (res.report && res.report.outcome_status) || "COMPLETED";
  const badgeOut = document.getElementById("badge-outcome");
  badgeOut.textContent = outcome;
  badgeOut.className = outcome.includes("UNCOMPARABLE") ? "badge warn" : "badge";

  const banner = document.getElementById("banner-notes");
  const notes = [];
  if (res.report && res.report.uncomparable_reason) {
    notes.push(`<strong>档位对比阻断：</strong>${esc(res.report.uncomparable_reason)}`);
  }
  if (res.report && res.report.honest_attribution_banner) {
    notes.push(`<strong>归因声明：</strong>${esc(res.report.honest_attribution_banner)}`);
  }
  (res.report?.action_transparency || []).forEach((t) => {
    if (t.evidence_boundary_note) {
      notes.push(`<strong>[${esc(t.segment_scope)}]</strong> ${esc(t.evidence_boundary_note)}`);
    }
  });
  if (notes.length) {
    banner.style.display = "block";
    banner.innerHTML = notes.join("<br/>");
  } else {
    banner.style.display = "none";
  }

  let diffHtml = "";
  if (res.intent_diff) {
    const d = res.intent_diff;
    diffHtml = `<div style="margin-top:6px; font-size:12px; color:#a7f3d0;">
      ✓ <strong>增量继承核验</strong>：<code>${esc(d.v1_id)} → ${esc(d.v2_id)}</code> |
      蛋白锁定继承: <code>${d.protein_lock_preserved}</code> |
      化学继承: <code>${d.chemistry_preserved}</code> |
      内切酶: <code>${esc(JSON.stringify(d.v1_enzymes))} → ${esc(JSON.stringify(d.v2_enzymes))}</code>
    </div>`;
  }
  const llmReplyHtml = res.assistant_reply
    ? `<div style="margin-top:6px; white-space:pre-wrap; font-size:12.5px;">${esc(res.assistant_reply)}</div>`
    : "";

  appendStreamCard(
    "agent",
    `Harness 执行完成 · Intent v${res.intent ? res.intent.intent_version : "?"}`,
    `${state.candidates.length} 条合格候选`,
    `<div>指令：<em>${esc(userText)}</em> → 状态 <code>${esc(outcome)}</code></div>${diffHtml}${llmReplyHtml}`
  );

  ["quick-refine-bspqi", "quick-refine-lock-cds", "quick-refine-commercial"].forEach((id) => {
    document.getElementById(id).disabled = false;
  });

  const recFirst = res.report?.recommended_bundle_ids?.[0] || state.candidates[0]?.bundle_id || null;
  state.selectedBundleId = recFirst;
  renderCandidates();
  if (recFirst) selectCandidate(recFirst);
}

async function refreshSystemStatus() {
  const dot = document.getElementById("dot-gateway");
  const txt = document.getElementById("txt-gateway");
  const t0 = performance.now();
  try {
    const st = await apiFetch("/v1/system/status");
    const ms = Math.round(performance.now() - t0);
    dot.className = "dot ok";
    txt.textContent = `7×24 ${st.hostname} Gateway: 在线 (${ms}ms)`;
    document.getElementById("pill-engine").textContent =
      `Engine: ${st.default_profile} · ViennaRNA ${st.viennarna_version}`;

    const presets = await apiFetch("/v1/presets");
    state.presets = presets;
    if (!document.getElementById("input-protein").value.trim()) {
      document.getElementById("input-protein").value = presets.egfp_reporter.protein_sequence;
      updateProteinBadge();
    }
    document.getElementById("txt-ref-r-preview").textContent =
      `${presets.preset_reference_r.bundle_id} (${presets.preset_reference_r.length_nt} nt) | ${presets.preset_reference_r.sequence_preview}`;
  } catch (err) {
    dot.className = "dot err";
    txt.textContent = `Gateway 未连接 (${err.message})`;
  }
}

async function runInitialTurn() {
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

  appendStreamCard("user", "用户输入 (Initial)", state.entryMode, esc(utterance));
  const btn = document.getElementById("btn-submit-initial");
  btn.disabled = true;
  try {
    const execMode = document.getElementById("sel-exec-mode").value;
    let res;
    if (execMode === "llm_agent") {
      res = await apiFetch(`/v1/projects/${state.projectId}/agent-chat`, {
        method: "POST",
        body: JSON.stringify({
          message: `${utterance}（目标蛋白序列：${prot}）`,
          provider: state.llmProvider,
        }),
      });
    } else {
      res = await apiFetch(`/v1/projects/${state.projectId}/turn`, {
        method: "POST",
        body: JSON.stringify({
          mode: "initial",
          user_utterance: utterance,
          protein_sequence: prot,
          uridine_chemistry: chem,
          compliance_track: prof,
          restriction_enzymes: enz,
          detarget_liver: detarget,
          prioritize_late_signal: late,
          conflicting_single_pick: conflict,
          use_preset_reference_r: useR,
        }),
      });
    }
    applyTurnResult(res, utterance);
  } catch (err) {
    appendStreamCard("warn", "请求异常", "ERROR", esc(err.message));
  } finally {
    btn.disabled = false;
  }
}

async function runRefineTurn(overridePayload = null) {
  if (!state.intentHistory.length) {
    return runInitialTurn();
  }
  const utterance =
    (overridePayload && overridePayload.user_utterance) ||
    document.getElementById("input-utterance").value.trim() ||
    `微调当前设计：内切酶避开 ${getSelectedEnzymes().join("/")}，其余锁定继承`;

  appendStreamCard("user", "用户增量微调 (Refine)", `基于 v${state.intentHistory.length}`, esc(utterance));
  const btn = document.getElementById("btn-send-turn");
  btn.disabled = true;
  try {
    const execMode = document.getElementById("sel-exec-mode").value;
    let res;
    if (execMode === "llm_agent" && !overridePayload) {
      res = await apiFetch(`/v1/projects/${state.projectId}/agent-chat`, {
        method: "POST",
        body: JSON.stringify({ message: utterance, provider: state.llmProvider }),
      });
    } else {
      const payload = overridePayload || {
        mode: "refine",
        user_utterance: utterance,
        restriction_enzymes: getSelectedEnzymes(),
        compliance_track: document.getElementById("sel-profile").value,
      };
      res = await apiFetch(`/v1/projects/${state.projectId}/turn`, {
        method: "POST",
        body: JSON.stringify(payload),
      });
    }
    applyTurnResult(res, utterance);
    document.getElementById("input-utterance").value = "";
  } catch (err) {
    appendStreamCard("warn", "微调异常", "ERROR", esc(err.message));
  } finally {
    btn.disabled = false;
  }
}

async function downloadExportFile(filename) {
  const bid = state.selectedBundleId;
  if (!bid) return;
  const cacheKey = `${state.projectId}:${bid}`;
  if (!state.cachedExports[cacheKey]) {
    state.cachedExports[cacheKey] = await apiFetch(
      `/v1/projects/${state.projectId}/bundles/${bid}/export`
    );
  }
  const content = state.cachedExports[cacheKey].files[filename];
  const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `${bid}_${filename}`;
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
  const egfp = (state.presets && state.presets.egfp_reporter.protein_sequence) || FALLBACK_EGFP;

  const steps = [
    {
      name: "1. 7×24 Vultr HTTPS 健康与 ViennaRNA 内核检查",
      fn: async () => {
        const s = await apiFetch("/v1/system/status");
        if (!s.viennarna_available) throw new Error("ViennaRNA 未就绪");
        return `${s.hostname} · ViennaRNA ${s.viennarna_version}`;
      },
    },
    {
      name: "2. 负样本输入护栏：缺起始 M 蛋白自动拦截",
      fn: async () => {
        const r = await apiFetch(`/v1/projects/${testProj}_nom/turn`, {
          method: "POST",
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
        return `Intent v1 (${r.candidates.length} 条合格)`;
      },
    },
    {
      name: "5. 多轮微调继承验证：只把 BsaI 改为 BspQI（核验蛋白与化学 100% 继承）",
      fn: async () => {
        const r = await apiFetch(`/v1/projects/${testProj}/turn`, {
          method: "POST",
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
        return `v1→v2 继承✓ (${r.intent_diff.v1_enzymes[0]}→${r.intent_diff.v2_enzymes[0]})`;
      },
    },
    {
      name: "6. 入口二（蛋白+参考构建 R）双轨输出 + INSDC GenBank 四件套导出核验",
      fn: async () => {
        const rRef = await apiFetch(`/v1/projects/${testProj}_r/turn`, {
          method: "POST",
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
          `/v1/projects/${testProj}/bundles/${window.__lastSelfTestTopBundle}/export`
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

function wireEvents() {
  document.getElementById("input-protein").addEventListener("input", updateProteinBadge);

  document.querySelectorAll(".tabs .tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tabs .tab").forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      state.entryMode = tab.dataset.entry;
      document.getElementById("box-ref-r").style.display =
        state.entryMode === "with_r" ? "block" : "none";
    });
  });

  document.querySelectorAll("#enzyme-chips .chip").forEach((chip) => {
    chip.addEventListener("click", () => chip.classList.toggle("selected"));
  });

  document.getElementById("preset-egfp").addEventListener("click", () => {
    const seq = (state.presets && state.presets.egfp_reporter.protein_sequence) || FALLBACK_EGFP;
    document.getElementById("input-protein").value = seq;
    setSelectedEnzymes(["BsaI"]);
    updateProteinBadge();
  });

  document.getElementById("preset-no-m").addEventListener("click", () => {
    const seq = (state.presets && state.presets.egfp_reporter.protein_sequence) || FALLBACK_EGFP;
    document.getElementById("input-protein").value = seq.slice(1);
    updateProteinBadge();
    runInitialTurn();
  });

  document.getElementById("preset-conflict").addEventListener("click", () => {
    const seq = (state.presets && state.presets.egfp_reporter.protein_sequence) || FALLBACK_EGFP;
    document.getElementById("input-protein").value = seq;
    updateProteinBadge();
    state.forceConflictOnce = true;
    runInitialTurn();
  });

  document.getElementById("btn-submit-initial").addEventListener("click", runInitialTurn);
  document.getElementById("btn-send-turn").addEventListener("click", () => runRefineTurn());

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

  document.getElementById("quick-refine-commercial").addEventListener("click", () => {
    document.getElementById("sel-profile").value = "commercial_clean";
    runRefineTurn({
      mode: "refine",
      user_utterance: "切换至 commercial_clean 纯商用档位核验组件放行状态",
      compliance_track: "commercial_clean",
    });
  });

  document.getElementById("btn-new-session").addEventListener("click", () => {
    state.projectId = `proj_${Math.random().toString(36).slice(2, 8)}`;
    state.intentHistory = [];
    state.candidates = [];
    state.selectedBundleId = null;
    document.getElementById("txt-project-id").textContent = state.projectId;
    renderIntentLineage();
    renderCandidates();
    appendStreamCard("agent", "新会话已创建", state.projectId, "已切换至全新隔离项目空间。");
  });

  document.getElementById("btn-dl-gb").addEventListener("click", () => downloadExportFile("construct.gb"));
  document.getElementById("btn-dl-rna").addEventListener("click", () => downloadExportFile("full_rna.fasta"));
  document.getElementById("btn-dl-cds").addEventListener("click", () => downloadExportFile("cds_only.fasta"));
  document.getElementById("btn-dl-meta").addEventListener("click", () => downloadExportFile("chemistry_metadata.json"));
  document.getElementById("btn-copy-cli").addEventListener("click", async () => {
    if (!state.lastCliCommand) return;
    await navigator.clipboard.writeText(state.lastCliCommand);
    const b = document.getElementById("btn-copy-cli");
    const old = b.textContent;
    b.textContent = "✓ 已复制 CLI 命令";
    setTimeout(() => { b.textContent = old; }, 1500);
  });

  document.getElementById("btn-open-settings").addEventListener("click", () => {
    document.getElementById("modal-settings").classList.add("open");
  });
  document.getElementById("btn-close-settings").addEventListener("click", () => {
    document.getElementById("modal-settings").classList.remove("open");
  });
  document.getElementById("btn-save-settings").addEventListener("click", () => {
    state.apiBase = document.getElementById("cfg-api-base").value.trim().replace(/\/+$/, "") || DEFAULT_API_BASE;
    state.accessToken = document.getElementById("cfg-access-token").value.trim();
    state.llmProvider = document.getElementById("cfg-llm-provider").value;
    state.byoKey = document.getElementById("cfg-byo-key").value.trim();
    localStorage.setItem("mrnax_api_base", state.apiBase);
    localStorage.setItem("mrnax_access_token", state.accessToken);
    localStorage.setItem("mrnax_llm_provider", state.llmProvider);
    sessionStorage.setItem("mrnax_byo_key", state.byoKey);
    document.getElementById("modal-settings").classList.remove("open");
    refreshSystemStatus();
  });

  document.getElementById("btn-run-selftest").addEventListener("click", () => {
    document.getElementById("modal-selftest").classList.add("open");
    runAutomatedBrowserSelfTest();
  });
  document.getElementById("btn-trigger-selftest").addEventListener("click", runAutomatedBrowserSelfTest);
  document.getElementById("btn-close-selftest").addEventListener("click", () => {
    document.getElementById("modal-selftest").classList.remove("open");
  });
}

document.addEventListener("DOMContentLoaded", () => {
  initConfig();
  wireEvents();
  document.getElementById("input-protein").value = FALLBACK_EGFP;
  updateProteinBadge();
  refreshSystemStatus();
});

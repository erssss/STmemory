const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

function escapeHtml(s) {
  return String(s ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function truncate(s, n) {
  const t = String(s ?? "");
  if (t.length <= n) return t;
  return t.slice(0, n) + "…";
}

function pill(id, text) {
  const el = $(id);
  if (el) el.textContent = text;
}

function dot(status) {
  const cls = status === "ok" ? "ok" : status === "bad" ? "bad" : status === "warn" ? "warn" : "";
  return `<span class="dot ${cls}"></span>`;
}

function checkRow(label, status, detail) {
  return `<div class="check">${dot(status)}<div><div>${escapeHtml(label)}</div><div class="mono" style="color: rgba(230,237,246,0.65); font-size: 12px">${escapeHtml(detail ?? "")}</div></div></div>`;
}

function plotlyLayout(title) {
  return {
    title: { text: title, font: { color: "#a8b3cf", size: 12 } },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    margin: { l: 42, r: 18, t: 42, b: 34 },
    font: { color: "#e6edf6" },
    xaxis: { gridcolor: "rgba(255,255,255,0.08)", zerolinecolor: "rgba(255,255,255,0.08)" },
    yaxis: { gridcolor: "rgba(255,255,255,0.08)", zerolinecolor: "rgba(255,255,255,0.08)" },
    legend: { font: { color: "#a8b3cf" } },
  };
}

let demoAuto = null;
let demoSeries = {
  turns: [],
  shallow: [],
  working: [],
  deep: [],
  meta: [],
  latency: [],
  memoryTokens: [],
  responseTokens: [],
  totalTokens: [],
  tokensSavedDelta: [],
};

function initMemoryCharts() {
  Plotly.newPlot(
    "memoryChart",
    [
      { x: [], y: [], name: "shallow", mode: "lines+markers" },
      { x: [], y: [], name: "working", mode: "lines+markers" },
      { x: [], y: [], name: "deep", mode: "lines+markers" },
      { x: [], y: [], name: "meta", mode: "lines+markers" },
    ],
    { ...plotlyLayout("各层记忆条目数"), yaxis: { ...plotlyLayout("").yaxis, title: { text: "entries" } } },
    { displayModeBar: false }
  );

  Plotly.newPlot(
    "latencyChart",
    [{ x: [], y: [], name: "latency_ms", type: "bar" }],
    { ...plotlyLayout("处理延迟（ms）"), yaxis: { ...plotlyLayout("").yaxis, title: { text: "ms" } } },
    { displayModeBar: false }
  );

  Plotly.newPlot(
    "tokenChart",
    [
      { x: [], y: [], name: "memory_tokens", mode: "lines+markers" },
      { x: [], y: [], name: "response_tokens", mode: "lines+markers" },
      { x: [], y: [], name: "total_tokens", mode: "lines+markers" },
      { x: [], y: [], name: "tokens_saved(Δ)", mode: "lines+markers" },
    ],
    { ...plotlyLayout("Token 预算与节省"), yaxis: { ...plotlyLayout("").yaxis, title: { text: "tokens" } } },
    { displayModeBar: false }
  );
}

function resetDemoUI(state) {
  demoSeries = {
    turns: [],
    shallow: [],
    working: [],
    deep: [],
    meta: [],
    latency: [],
    memoryTokens: [],
    responseTokens: [],
    totalTokens: [],
    tokensSavedDelta: [],
  };
  initMemoryCharts();
  $("#ioBox").textContent = "-";
  $("#ctxBox").textContent = "-";
  $("#checksBox").innerHTML = "-";
  updateDemoHeader(state);
}

function updateDemoHeader(state) {
  const idx = Number(state?.turn_idx ?? 0);
  const total = Number(state?.total_turns ?? 0);
  pill("#demoPill", `Demo: ${idx}/${total}`);
  $("#turnInfo").textContent = `${idx} / ${total}`;
}

function pushDemoPoint(payload) {
  const turn = Number(payload?.turn_idx ?? 0) + 1;
  const ms = Number(payload?.result?.latency_ms ?? 0);
  const stats = payload?.memory_stats ?? {};
  const perf = payload?.performance?.performance ?? {};
  const memoryTokens = Number(payload?.result?.memory_tokens ?? 0);
  const responseTokens = Number(payload?.result?.response_tokens ?? 0);
  const totalTokens = Number(payload?.result?.total_tokens ?? 0);
  const tokensSavedDelta = Number(payload?.changes?.tokens_saved_delta ?? 0);

  demoSeries.turns.push(turn);
  demoSeries.shallow.push(Number(stats?.shallow?.entries ?? 0));
  demoSeries.working.push(Number(stats?.working?.entries ?? 0));
  demoSeries.deep.push(Number(stats?.deep?.entries ?? 0));
  demoSeries.meta.push(Number(stats?.meta?.entries ?? 0));
  demoSeries.latency.push(ms);
  demoSeries.memoryTokens.push(memoryTokens);
  demoSeries.responseTokens.push(responseTokens);
  demoSeries.totalTokens.push(totalTokens);
  demoSeries.tokensSavedDelta.push(tokensSavedDelta);

  Plotly.react(
    "memoryChart",
    [
      { x: demoSeries.turns, y: demoSeries.shallow, name: "shallow", mode: "lines+markers" },
      { x: demoSeries.turns, y: demoSeries.working, name: "working", mode: "lines+markers" },
      { x: demoSeries.turns, y: demoSeries.deep, name: "deep", mode: "lines+markers" },
      { x: demoSeries.turns, y: demoSeries.meta, name: "meta", mode: "lines+markers" },
    ],
    { ...plotlyLayout("各层记忆条目数"), yaxis: { ...plotlyLayout("").yaxis, title: { text: "entries" } } },
    { displayModeBar: false }
  );

  Plotly.react(
    "latencyChart",
    [{ x: demoSeries.turns, y: demoSeries.latency, name: "latency_ms", type: "bar" }],
    { ...plotlyLayout("处理延迟（ms）"), yaxis: { ...plotlyLayout("").yaxis, title: { text: "ms" } } },
    { displayModeBar: false }
  );

  Plotly.react(
    "tokenChart",
    [
      { x: demoSeries.turns, y: demoSeries.memoryTokens, name: "memory_tokens", mode: "lines+markers" },
      { x: demoSeries.turns, y: demoSeries.responseTokens, name: "response_tokens", mode: "lines+markers" },
      { x: demoSeries.turns, y: demoSeries.totalTokens, name: "total_tokens", mode: "lines+markers" },
      { x: demoSeries.turns, y: demoSeries.tokensSavedDelta, name: "tokens_saved(Δ)", mode: "lines+markers" },
    ],
    { ...plotlyLayout("Token 预算与节省"), yaxis: { ...plotlyLayout("").yaxis, title: { text: "tokens" } } },
    { displayModeBar: false }
  );

  const cacheHit = Boolean(payload?.changes?.cache_hit ?? false);
  $("#cacheHit").textContent = cacheHit ? "是" : "否";

  const q = payload?.result?.query ?? "";
  const r = payload?.result?.response ?? "";
  const rel = Number(payload?.result?.relevant_memories_count ?? 0);
  const budget = payload?.result?.budget_check ?? {};
  const within = Boolean(budget?.within_limits ?? true);
  const maxTokens = budget?.max_tokens ?? "-";
  const io = [
    `Topic: ${payload?.conversation?.topic ?? ""}`,
    `Query: ${q}`,
    "",
    `Response: ${r}`,
    "",
    `latency_ms=${Number(payload?.result?.latency_ms ?? 0).toFixed(1)}  relevant_memories=${rel}`,
    `tokens: memory=${memoryTokens} response=${responseTokens} total=${totalTokens}  max=${maxTokens} within=${within}`,
    `perf: avg_latency_ms=${Number(perf?.average_latency_ms ?? 0).toFixed(1)} total_tokens_saved=${Number(perf?.total_tokens_saved ?? 0).toFixed(1)}`,
  ].join("\n");
  $("#ioBox").textContent = io;

  const ctx = payload?.result?.context ?? "";
  $("#ctxBox").textContent = truncate(ctx, 2200);

  const shallowOk = (stats?.shallow?.entries ?? 0) <= 50;
  const workingOk = (stats?.working?.entries ?? 0) <= 200;
  const budgetOk = within;
  const retrievalOk = typeof rel === "number" && rel >= 0;
  const checks = [
    checkRow("容量约束（浅层）", shallowOk ? "ok" : "bad", `entries=${stats?.shallow?.entries ?? 0}  max=50`),
    checkRow("容量约束（工作层）", workingOk ? "ok" : "bad", `entries=${stats?.working?.entries ?? 0}  max=200`),
    checkRow("预算约束", budgetOk ? "ok" : "warn", `within_limits=${within}  max_tokens=${maxTokens}`),
    checkRow("检索结果结构", retrievalOk ? "ok" : "bad", `relevant_memories_count=${rel}`),
  ];
  $("#checksBox").innerHTML = checks.join("");

  updateDemoHeader({ turn_idx: payload?.turn_idx + 1, total_turns: payload?.total_turns });
}

async function apiJson(url, opts) {
  const res = await fetch(url, opts);
  const t = await res.text();
  let data = null;
  try {
    data = JSON.parse(t);
  } catch {
    data = { raw: t };
  }
  if (!res.ok) {
    throw new Error(data?.error || `HTTP ${res.status}`);
  }
  return data;
}

async function demoReset() {
  const state = await apiJson("/api/demo/reset", { method: "POST" });
  resetDemoUI(state?.state);
}

async function demoStep() {
  const payload = await apiJson("/api/demo/step", { method: "POST" });
  if (payload?.done) {
    if (demoAuto) {
      clearInterval(demoAuto);
      demoAuto = null;
      $("#btnAuto").textContent = "自动播放";
    }
    return;
  }
  pushDemoPoint(payload);
}

async function demoChat(query) {
  const payload = await apiJson("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ query }) });
  const result = payload?.result ?? {};
  const stats = payload?.memory_stats ?? {};
  const perf = payload?.performance?.performance ?? {};
  const io = [
    `Query: ${result?.query ?? ""}`,
    "",
    `Response: ${result?.response ?? ""}`,
    "",
    `latency_ms=${Number(result?.latency_ms ?? 0).toFixed(1)}  relevant_memories=${Number(result?.relevant_memories_count ?? 0)}`,
    `tokens: memory=${Number(result?.memory_tokens ?? 0)} response=${Number(result?.response_tokens ?? 0)} total=${Number(result?.total_tokens ?? 0)}`,
    `perf: avg_latency_ms=${Number(perf?.average_latency_ms ?? 0).toFixed(1)} total_tokens_saved=${Number(perf?.total_tokens_saved ?? 0).toFixed(1)}`,
  ].join("\n");
  $("#ioBox").textContent = io;
  $("#ctxBox").textContent = truncate(result?.context ?? "", 2200);
  $("#cacheHit").textContent = (Number(result?.relevant_memories_count ?? 0) > 0) ? "是" : "否";
  const checks = [
    checkRow("容量约束（浅层）", (stats?.shallow?.entries ?? 0) <= 50 ? "ok" : "bad", `entries=${stats?.shallow?.entries ?? 0}  max=50`),
    checkRow("容量约束（工作层）", (stats?.working?.entries ?? 0) <= 200 ? "ok" : "bad", `entries=${stats?.working?.entries ?? 0}  max=200`),
  ];
  $("#checksBox").innerHTML = checks.join("");
}

let locomoWS = null;
let locomoLogBuf = [];

function appendLocomoLog(line) {
  locomoLogBuf.push(String(line ?? ""));
  if (locomoLogBuf.length > 500) locomoLogBuf = locomoLogBuf.slice(-500);
  const el = $("#locomoLog");
  el.textContent = locomoLogBuf.join("\n");
  el.scrollTop = el.scrollHeight;
}

function renderLocomoTable(rows) {
  const items = (rows ?? []).slice(0, 50);
  const html = [
    "<table>",
    "<thead><tr><th>category</th><th>question</th><th>answer</th><th>response</th><th>bleu</th><th>f1</th><th>llm</th></tr></thead>",
    "<tbody>",
    ...items.map((r) => {
      return `<tr>
        <td>${escapeHtml(r.category)}</td>
        <td>${escapeHtml(truncate(r.question, 120))}</td>
        <td>${escapeHtml(truncate(r.answer, 120))}</td>
        <td>${escapeHtml(truncate(r.response, 120))}</td>
        <td>${escapeHtml(r.bleu_score)}</td>
        <td>${escapeHtml(r.f1_score)}</td>
        <td>${escapeHtml(r.llm_score)}</td>
      </tr>`;
    }),
    "</tbody></table>",
  ].join("");
  $("#locomoTable").innerHTML = html;
}

function renderLocomoCharts(run) {
  const report = run?.report ?? {};
  const latency = report?.latency ?? {};
  const add = latency?.add ?? {};
  const search = latency?.search ?? {};
  const x = ["add_p95", "add_avg", "search_p95", "search_avg"];
  const y = [
    Number(add?.p95_s ?? NaN),
    Number(add?.avg_s ?? NaN),
    Number(search?.p95_s ?? NaN),
    Number(search?.avg_s ?? NaN),
  ].map((v) => (Number.isFinite(v) ? v : null));
  Plotly.react(
    "locomoLatencyChart",
    [{ x, y, type: "bar", name: "seconds" }],
    { ...plotlyLayout("延迟（秒）"), yaxis: { ...plotlyLayout("").yaxis, title: { text: "seconds" } } },
    { displayModeBar: false }
  );

  const metrics = report?.metrics ?? {};
  const sx = ["bleu_mean", "f1_mean", "llm_mean"];
  const sy = [metrics?.bleu_mean, metrics?.f1_mean, metrics?.llm_mean].map((v) => (v == null ? null : Number(v)));
  Plotly.react(
    "locomoScoreChart",
    [{ x: sx, y: sy, type: "bar", name: "mean" }],
    { ...plotlyLayout("总体得分均值"), yaxis: { ...plotlyLayout("").yaxis, title: { text: "score" } } },
    { displayModeBar: false }
  );

  const by = metrics?.by_category ?? {};
  const cats = Object.keys(by);
  const bleu = cats.map((c) => (by[c]?.bleu_mean == null ? null : Number(by[c]?.bleu_mean)));
  const f1 = cats.map((c) => (by[c]?.f1_mean == null ? null : Number(by[c]?.f1_mean)));
  const llm = cats.map((c) => (by[c]?.llm_mean == null ? null : Number(by[c]?.llm_mean)));
  Plotly.react(
    "locomoCategoryChart",
    [
      { x: cats, y: bleu, type: "bar", name: "bleu_mean" },
      { x: cats, y: f1, type: "bar", name: "f1_mean" },
      { x: cats, y: llm, type: "bar", name: "llm_mean" },
    ],
    { ...plotlyLayout("按类别聚合"), barmode: "group", yaxis: { ...plotlyLayout("").yaxis, title: { text: "score" } } },
    { displayModeBar: false }
  );
}

function renderThreshold(run) {
  const report = run?.report ?? {};
  const pass = Boolean(report?.threshold_pass ?? false);
  const reasons = report?.threshold_fail_reasons ?? [];
  const t = report?.thresholds ?? {};
  const addMax = t?.max_add_p95_s ?? null;
  const searchMax = t?.max_search_p95_s ?? null;
  const latency = report?.latency ?? {};
  const addP95 = latency?.add?.p95_s ?? null;
  const searchP95 = latency?.search?.p95_s ?? null;
  const tk1 = run?.token1?.token_count?.total_tokens ?? null;
  const tk2 = run?.token2?.token_count?.total_tokens ?? null;
  const tkDelta = (tk1 != null && tk2 != null) ? Number(tk2) - Number(tk1) : null;
  const detail = [
    `add_p95_s=${addP95 ?? "-"}  max=${addMax ?? "未设置"}`,
    `search_p95_s=${searchP95 ?? "-"}  max=${searchMax ?? "未设置"}`,
    (tkDelta != null ? `tokens(Δ)=${tkDelta}` : ""),
    reasons?.length ? `reasons=${JSON.stringify(reasons)}` : "",
  ].filter(Boolean).join("  ");
  $("#thresholdBox").innerHTML = `${dot(pass ? "ok" : (addMax || searchMax ? "bad" : "warn"))}<span style="margin-left: 8px">${pass ? "PASS" : "N/A or FAIL"}</span><div class="mono" style="margin-top: 6px; color: rgba(230,237,246,0.65)">${escapeHtml(detail)}</div>`;
}

function renderLocomoRun(run) {
  $("#locomoDir").textContent = run?.dir ?? "-";
  renderThreshold(run);
  renderLocomoCharts(run);
  renderLocomoTable(run?.evaluation_csv_rows ?? []);
  pill("#locomoPill", `LOCOMO: loaded`);
}

async function refreshRuns() {
  const data = await apiJson("/api/locomo/runs");
  const runs = data?.runs ?? [];
  const sel = $("#runSelect");
  sel.innerHTML = "";
  for (const r of runs) {
    const opt = document.createElement("option");
    opt.value = r.dir;
    const ts = r.mtime ? new Date(r.mtime * 1000).toLocaleString() : "-";
    opt.textContent = `${r.name}  (${ts})`;
    sel.appendChild(opt);
  }
}

function connectLocomoWS() {
  if (locomoWS) return;
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const url = `${proto}//${location.host}/ws/locomo`;
  const ws = new WebSocket(url);
  locomoWS = ws;

  ws.onopen = () => {
    pill("#locomoPill", "LOCOMO: ws connected");
  };
  ws.onclose = () => {
    pill("#locomoPill", "LOCOMO: ws closed");
    locomoWS = null;
  };
  ws.onerror = () => {};
  ws.onmessage = (ev) => {
    let msg = {};
    try { msg = JSON.parse(ev.data); } catch { msg = {}; }
    const type = msg?.type ?? "";
    if (type === "status") {
      $("#locomoDir").textContent = msg?.dir ?? "-";
      pill("#locomoPill", msg?.running ? "LOCOMO: running" : "LOCOMO: idle");
    } else if (type === "run_started") {
      locomoLogBuf = [];
      appendLocomoLog(`$ ${msg?.cmd ?? ""}`);
      $("#locomoDir").textContent = msg?.dir ?? "-";
      pill("#locomoPill", "LOCOMO: running");
    } else if (type === "log") {
      appendLocomoLog(msg?.line ?? "");
    } else if (type === "run_finished") {
      appendLocomoLog(`\n[finished] exit_code=${msg?.exit_code ?? "-"}\n`);
      pill("#locomoPill", "LOCOMO: finished");
      if (msg?.run) renderLocomoRun(msg.run);
    } else if (type === "run_loaded") {
      if (msg?.run) renderLocomoRun(msg.run);
    } else if (type === "error") {
      appendLocomoLog(`[error] ${msg?.message ?? ""}`);
    }
  };
}

function setupTabs() {
  $$(".tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      $$(".tab").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      const key = btn.dataset.tab;
      $$(".tabpane").forEach((p) => p.classList.remove("active"));
      if (key === "memory") $("#tab-memory").classList.add("active");
      if (key === "locomo") $("#tab-locomo").classList.add("active");
    });
  });
}

async function bootstrap() {
  setupTabs();
  initMemoryCharts();
  connectLocomoWS();
  await refreshRuns().catch(() => {});
  const state = await apiJson("/api/demo/state").catch(() => null);
  if (state) updateDemoHeader(state);

  $("#btnReset").addEventListener("click", async () => {
    await demoReset();
  });
  $("#btnStep").addEventListener("click", async () => {
    await demoStep();
  });
  $("#btnAuto").addEventListener("click", async () => {
    if (demoAuto) {
      clearInterval(demoAuto);
      demoAuto = null;
      $("#btnAuto").textContent = "自动播放";
      return;
    }
    $("#btnAuto").textContent = "暂停";
    demoAuto = setInterval(() => demoStep().catch(() => {}), 900);
  });
  $("#btnChat").addEventListener("click", async () => {
    const q = $("#chatInput").value.trim();
    if (!q) return;
    $("#chatInput").value = "";
    await demoChat(q);
  });
  $("#chatInput").addEventListener("keydown", async (ev) => {
    if (ev.key === "Enter") {
      const q = $("#chatInput").value.trim();
      if (!q) return;
      $("#chatInput").value = "";
      await demoChat(q);
    }
  });

  $("#btnRefreshRuns").addEventListener("click", async () => {
    await refreshRuns();
  });
  $("#btnLoadRun").addEventListener("click", async () => {
    const dir = $("#runSelect").value;
    if (!dir || !locomoWS || locomoWS.readyState !== 1) return;
    locomoWS.send(JSON.stringify({ action: "load", dir }));
  });
  $("#btnStartLocomo").addEventListener("click", async () => {
    if (!locomoWS || locomoWS.readyState !== 1) return;
    locomoWS.send(JSON.stringify({ action: "start", max_workers: 2 }));
  });
  $("#btnStopLocomo").addEventListener("click", async () => {
    if (!locomoWS || locomoWS.readyState !== 1) return;
    locomoWS.send(JSON.stringify({ action: "stop" }));
  });

  pill("#serverPill", "Server: online");
}

bootstrap();

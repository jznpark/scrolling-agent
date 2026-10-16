"use strict";
// Scrolling agent web demo: talks to webdemo/server.py and draws the environment.

const $ = (id) => document.getElementById(id);
const SESSION = (crypto.randomUUID ? crypto.randomUUID() : String(Math.random())).slice(0, 36);
const STORE_KEY = `scrolling-demo:${location.port}`;
const COMPACT = new URLSearchParams(location.search).get("compact") === "1";

let schema = null;        // settings + policies from the server
let state = null;         // last state from the server
let history = [];         // [t, return, mode] per step since reset
let events = [];
let lastEventId = 0;
let playing = false;
let inflight = false;
let budget = 0;           // fractional steps owed to the play loop
let lastFrame = 0;
const held = new Set();   // arrow keys held (human mode)
let pending = null;       // one-shot human action (click / back / stay)

// --- server -----------------------------------------------------------------------

async function api(path, body) {
  const res = await fetch(path, {
    method: body ? "POST" : "GET",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

function toast(message) {
  const el = $("toast");
  el.textContent = message;
  el.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => (el.hidden = true), 4000);
}

// --- settings panel ----------------------------------------------------------------

function fieldInput(field) {
  let input;
  if (field.choices) {
    input = document.createElement("select");
    for (const c of field.choices) input.add(new Option(c, c));
  } else if (field.kind === "bool") {
    input = document.createElement("input");
    input.type = "checkbox";
  } else {
    input = document.createElement("input");
    input.type = field.kind === "tuple" || field.kind === "str" ? "text" : "number";
    if (input.type === "number") input.step = "any";
  }
  input.dataset.key = field.key;
  setInput(input, field, field.default);
  input.addEventListener("input", () => markChanged(input, field));
  input.addEventListener("change", () => markChanged(input, field));
  return input;
}

function setInput(input, field, value) {
  if (field.kind === "bool") input.checked = Boolean(value);
  else if (field.kind === "tuple") input.value = value.join(", ");
  else input.value = value;
}

function readInput(input, field) {
  if (field.kind === "bool") return input.checked;
  if (field.kind === "tuple") return input.value.split(",").map((v) => parseFloat(v)).filter((v) => !Number.isNaN(v));
  if (field.kind === "int") return parseInt(input.value, 10);
  if (field.kind === "float") return parseFloat(input.value);
  return input.value;
}

function isChanged(input, field) {
  return JSON.stringify(readInput(input, field)) !== JSON.stringify(field.default);
}

function markChanged(input, field) {
  input.closest(".field").classList.toggle("changed", isChanged(input, field));
}

function buildSettings() {
  const root = $("settings");
  root.innerHTML = "";
  for (const section of schema.sections) {
    const det = document.createElement("details");
    det.open = !COMPACT && (section.fields.some((f) => f.tag === "ext") || section.name === "feed");
    const sum = document.createElement("summary");
    sum.textContent = section.name;
    if (section.fields.some((f) => f.tag === "ext")) sum.innerHTML += ' <span class="tag ext">ext</span>';
    det.appendChild(sum);
    for (const field of section.fields) {
      const row = document.createElement("div");
      row.className = "field";
      row.title = `${field.key}\n${field.comment}`;
      const name = document.createElement("span");
      name.className = "name";
      name.textContent = field.name;
      if (field.tag) {
        const tag = document.createElement("span");
        tag.className = `tag ${field.tag.toLowerCase()}`;
        tag.textContent = field.tag;
        name.appendChild(tag);
      }
      row.append(name, fieldInput(field));
      det.appendChild(row);
    }
    root.appendChild(det);
  }
}

function fieldsByKey() {
  const map = {};
  for (const s of schema.sections) for (const f of s.fields) map[f.key] = f;
  return map;
}

function overrides() {
  const map = fieldsByKey();
  const out = {};
  for (const input of document.querySelectorAll("#settings [data-key]")) {
    const field = map[input.dataset.key];
    if (isChanged(input, field)) out[field.key] = readInput(input, field);
  }
  return out;
}

function policySpec() {
  return {
    name: $("policy").value,
    rho: parseFloat($("rho").value),
    dwell: parseFloat($("dwell").value),
    seed: parseInt($("seed").value, 10),
  };
}

function saveSetup() {
  try {
    localStorage.setItem(STORE_KEY, JSON.stringify({ overrides: overrides(), policy: policySpec(),
      speed: $("speed").value, hidden: $("hidden").checked }));
  } catch (_) { /* storage may be unavailable */ }
}

function loadSetup() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORE_KEY) || "null");
    if (!saved) return;
    const map = fieldsByKey();
    for (const [key, value] of Object.entries(saved.overrides || {})) {
      const input = document.querySelector(`#settings [data-key="${key}"]`);
      if (input && map[key]) { setInput(input, map[key], value); markChanged(input, map[key]); }
    }
    if (saved.policy) {
      if ([...$("policy").options].some((o) => o.value === saved.policy.name)) $("policy").value = saved.policy.name;
      $("rho").value = saved.policy.rho;
      $("dwell").value = saved.policy.dwell;
      $("seed").value = saved.policy.seed;
    }
    if (saved.speed) $("speed").value = saved.speed;
    $("hidden").checked = Boolean(saved.hidden);
  } catch (_) { /* ignore broken storage */ }
}

// --- episode control -----------------------------------------------------------------

async function reset() {
  setPlaying(false);
  try {
    state = await api("/api/reset", { session: SESSION, seed: parseInt($("seed").value, 10),
      overrides: overrides(), policy: policySpec() });
  } catch (err) {
    toast(`다시 시작 실패: ${err.message}`);
    return;
  }
  history = [];
  events = [];
  lastEventId = 0;
  saveSetup();
  render();
}

function isHuman() {
  return $("policy").value === "human";
}

function humanAction() {
  const a = state.actions;
  if (pending !== null) {
    const p = pending;
    pending = null;
    return p;
  }
  if (held.has("ArrowDown")) return a.down[held.has("Shift") && a.down.length > 1 ? 1 : 0];
  if (held.has("ArrowUp")) return a.up[held.has("Shift") && a.up.length > 1 ? 1 : 0];
  return a.stay;
}

async function advance(n, action) {
  if (inflight || !state || state.done) return;
  inflight = true;
  try {
    const body = { session: SESSION, n, since_event: lastEventId };
    if (action !== undefined && action !== null) body.action = action;
    const next = await api("/api/step", body);
    history.push(...next.history);
    for (const e of next.events) events.push(e);
    if (next.events.length) lastEventId = next.events[next.events.length - 1].id;
    if (events.length > 300) events = events.slice(-300);
    state = next;
    if (state.done) setPlaying(false);
    render();
  } catch (err) {
    setPlaying(false);
    toast(err.message);
  } finally {
    inflight = false;
  }
}

function setPlaying(on) {
  playing = on && state && !state.done;
  $("play").textContent = playing ? "⏸ 일시정지 (P)" : "▶ 재생 (P)";
  budget = 0;
}

function loop(now) {
  const elapsed = lastFrame ? (now - lastFrame) / 1000 : 0;
  lastFrame = now;
  if (playing && state && !inflight) {
    budget += elapsed * parseFloat($("speed").value);
    if (budget >= 1) {
      if (isHuman()) {
        budget = Math.min(budget - 1, 2);
        advance(1, humanAction());
      } else {
        const n = Math.min(Math.floor(budget), 600);
        budget -= n;
        advance(n);
      }
    }
  }
  requestAnimationFrame(loop);
}

// --- drawing -----------------------------------------------------------------------

function css(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function fitCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const w = Math.max(1, Math.round(rect.width * dpr));
  const h = Math.max(1, Math.round(rect.height * dpr));
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w: rect.width, h: rect.height };
}

const fmt = (v, d = 2) => (typeof v === "number" ? v.toFixed(d) : v);

function drawList() {
  const { ctx, w, h } = fitCanvas($("list"));
  ctx.clearRect(0, 0, w, h);
  if (!state) return;
  const H = state.screen_h;
  const top = state.scroll - 1;
  const px = h / (H + 2);
  const showHidden = $("hidden").checked;
  const slots = state.slots;
  ctx.textBaseline = "middle";

  const first = Math.max(0, Math.floor(top));
  const last = Math.min(state.n_items - 1, Math.ceil(state.scroll + H + 1));
  for (let i = first; i <= last; i++) {
    const it = state.items[i];
    const y = (i - top) * px;
    const x0 = 12, cw = w - 24, ch = px - 8;
    const current = state.current === i;
    ctx.fillStyle = it.understood ? css("--card-known") : css("--card");
    ctx.strokeStyle = current ? css("--current") : css("--border");
    ctx.lineWidth = current ? 3 : 1;
    ctx.beginPath();
    ctx.roundRect(x0, y + 4, cw, ch, 8);
    ctx.fill();
    ctx.stroke();

    const slot = slots.indexOf(i);
    ctx.fillStyle = css("--text");
    ctx.font = "600 15px system-ui, sans-serif";
    ctx.fillText(`#${i}`, x0 + 12, y + 4 + ch * 0.22);
    if (slot >= 0) {
      ctx.font = "12px system-ui, sans-serif";
      ctx.fillStyle = state.mask[state.actions.slots[slot]] ? css("--known") : css("--muted");
      ctx.fillText(`[${slot + 1}] ${state.mask[state.actions.slots[slot]] ? "클릭 가능" : "보임"}`, x0 + 62, y + 4 + ch * 0.22);
    }

    ctx.font = "13px system-ui, sans-serif";
    if (it.understood) {
      ctx.fillStyle = css("--text");
      ctx.fillText(`A ${fmt(it.A)}   λ ${fmt(it.lam)}   w ${fmt(it.w, 1)}s   rate ${fmt(it.rate)}`, x0 + 12, y + 4 + ch * 0.48);
      const barW = (cw - 24) * Math.min(1, it.rate / 3);
      ctx.fillStyle = css("--known");
      ctx.fillRect(x0 + 12, y + 4 + ch * 0.7, barW, Math.max(4, ch * 0.1));
      if (showHidden && (Math.abs(it.A - it.hidden.A) > 1e-6 || Math.abs(it.lam - it.hidden.lam) > 1e-6)) {
        ctx.fillStyle = css("--current");
        ctx.font = "italic 12px system-ui, sans-serif";
        ctx.fillText(`실제: A ${fmt(it.hidden.A)}  λ ${fmt(it.hidden.lam)}  rate ${fmt(it.hidden.rate)}`, x0 + 12, y + 4 + ch * 0.88);
      }
    } else {
      ctx.fillStyle = css("--muted");
      ctx.fillText(it.seen ? `증거 ${fmt(it.x)}   노출 ${fmt(it.exposure, 1)}s` : "아직 안 봄", x0 + 12, y + 4 + ch * 0.48);
      ctx.fillStyle = css("--unseen");
      ctx.fillRect(x0 + 12, y + 4 + ch * 0.7, cw - 24, Math.max(4, ch * 0.1));
      ctx.fillStyle = css("--evidence");
      ctx.fillRect(x0 + 12, y + 4 + ch * 0.7, (cw - 24) * Math.max(0, Math.min(1, it.x)), Math.max(4, ch * 0.1));
      if (showHidden) {
        ctx.fillStyle = it.hidden.z ? css("--z1") : css("--hidden");
        ctx.font = "italic 12px system-ui, sans-serif";
        ctx.fillText(`숨은 값: A ${fmt(it.hidden.A)}  ν ${fmt(it.hidden.nu)}  z ${it.hidden.z}`, x0 + 12, y + 4 + ch * 0.88);
      }
    }
  }

  // the screen window
  ctx.setLineDash([8, 6]);
  ctx.strokeStyle = css("--screen");
  ctx.lineWidth = 2;
  ctx.strokeRect(4, px, w - 8, H * px);
  ctx.setLineDash([]);

  if (state.mode === "WATCH" && state.current !== null) drawWatchOverlay(ctx, w, h);
  if (state.mode === "TRANSITION") banner(ctx, w, h, "화면 전환 중 (클릭 / back, 0.2초)");
  if (state.done) banner(ctx, w, h, `세션 종료 · 누적 보상 ${fmt(state.return)} · R로 다시 시작`);
}

function banner(ctx, w, h, text) {
  ctx.fillStyle = "rgba(0,0,0,0.65)";
  ctx.fillRect(0, h / 2 - 22, w, 44);
  ctx.fillStyle = "#fff";
  ctx.font = "600 15px system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(text, w / 2, h / 2);
  ctx.textAlign = "left";
}

function drawWatchOverlay(ctx, w, h) {
  const it = state.items[state.current];
  const showHidden = $("hidden").checked;
  const x0 = 30, y0 = h * 0.18, bw = w - 60, bh = h * 0.64;
  ctx.fillStyle = css("--panel");
  ctx.globalAlpha = 0.96;
  ctx.fillRect(x0, y0, bw, bh);
  ctx.globalAlpha = 1;
  ctx.strokeStyle = css("--current");
  ctx.lineWidth = 3;
  ctx.strokeRect(x0, y0, bw, bh);

  ctx.fillStyle = css("--text");
  ctx.font = "600 18px system-ui, sans-serif";
  ctx.fillText(`시청 중 #${it.id}`, x0 + 16, y0 + 26);
  ctx.font = "13px system-ui, sans-serif";
  const gained = it.A / Math.max(it.lam, 1e-9) * (1 - Math.exp(-it.lam * it.w));
  ctx.fillText(`w ${fmt(it.w, 1)}s   지금 보상률 ${fmt(it.rate)}   (인지한 값 기준 누적 ${fmt(gained)})`, x0 + 16, y0 + 50);
  if (showHidden) {
    ctx.fillStyle = css("--current");
    ctx.fillText(`실제 보상률 ${fmt(it.hidden.rate)}`, x0 + 16, y0 + 70);
  }

  // rate curve A e^{-λ t}
  const px0 = x0 + 50, py0 = y0 + bh - 34, pw = bw - 80, ph = bh - 130;
  const tMax = Math.max(10, it.w + 5);
  const rMax = Math.max(it.A, showHidden ? it.hidden.A : 0, 0.1) * 1.1;
  const X = (t) => px0 + (t / tMax) * pw;
  const Y = (r) => py0 - (r / rMax) * ph;
  ctx.strokeStyle = css("--border");
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(px0, py0 - ph); ctx.lineTo(px0, py0); ctx.lineTo(px0 + pw, py0); ctx.stroke();
  const curve = (A, lam, color, dash) => {
    ctx.setLineDash(dash); ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath();
    for (let k = 0; k <= 100; k++) { const t = (k / 100) * tMax; const r = A * Math.exp(-lam * t); k ? ctx.lineTo(X(t), Y(r)) : ctx.moveTo(X(t), Y(r)); }
    ctx.stroke(); ctx.setLineDash([]);
  };
  curve(it.A, it.lam, css("--known"), []);
  if (showHidden) curve(it.hidden.A, it.hidden.lam, css("--current"), [5, 4]);
  // watched part
  ctx.fillStyle = css("--watch");
  ctx.globalAlpha = 0.25;
  ctx.fillRect(px0, py0 - ph, X(it.w) - px0, ph);
  ctx.globalAlpha = 1;
  ctx.beginPath(); ctx.arc(X(it.w), Y(it.rate), 5, 0, Math.PI * 2); ctx.fillStyle = css("--current"); ctx.fill();
  ctx.fillStyle = css("--muted");
  ctx.font = "11px system-ui, sans-serif";
  ctx.fillText("보상률 A·e^(−λw)", px0 + 4, py0 - ph - 8);
  ctx.fillText(`w (0 ~ ${tMax.toFixed(0)}s)`, px0 + pw - 70, py0 + 16);
}

function drawMinimap() {
  const { ctx, w, h } = fitCanvas($("minimap"));
  ctx.clearRect(0, 0, w, h);
  if (!state) return;
  const n = state.n_items;
  const rh = h / n;
  const showHidden = $("hidden").checked;
  const maxRate = Math.max(0.5, ...state.items.map((it) => it.rate));
  for (const it of state.items) {
    const y = it.id * rh;
    ctx.fillStyle = css("--unseen");
    ctx.fillRect(0, y, w, Math.max(1, rh - 0.5));
    if (it.understood) {
      ctx.fillStyle = css("--known");
      ctx.globalAlpha = 0.25 + 0.75 * Math.max(0, it.rate) / maxRate;
      ctx.fillRect(0, y, w, Math.max(1, rh - 0.5));
      ctx.globalAlpha = 1;
    } else if (it.seen) {
      ctx.fillStyle = css("--evidence");
      ctx.fillRect(0, y, w * Math.max(0.05, Math.min(1, it.x)), Math.max(1, rh - 0.5));
    }
    if (it.w > 0) {
      ctx.fillStyle = css("--watch");
      ctx.fillRect(0, y, Math.min(w * 0.5, 4 + it.w * 3), Math.max(1, rh - 0.5));
    }
    if (showHidden && it.hidden.z) {
      ctx.fillStyle = css("--z1");
      ctx.fillRect(w - 6, y, 6, Math.max(1, rh - 0.5));
    }
  }
  ctx.strokeStyle = css("--screen");
  ctx.lineWidth = 2;
  ctx.strokeRect(1, state.scroll * rh, w - 2, state.screen_h * rh);
  if (state.current !== null) {
    ctx.fillStyle = css("--current");
    ctx.fillRect(0, state.current * rh - 1, w, Math.max(2, rh + 1));
  }
}

function drawChart() {
  const { ctx, w, h } = fitCanvas($("chart"));
  ctx.clearRect(0, 0, w, h);
  if (!state) return;
  const T = state.T;
  const values = history.map((p) => p[1]);
  const lo = Math.min(0, ...values), hi = Math.max(1, ...values);
  const pad = 28;
  const X = (t) => pad + (t / T) * (w - pad - 6);
  const Y = (v) => h - 16 - ((v - lo) / (hi - lo)) * (h - 26);
  // watch periods
  ctx.fillStyle = css("--known");
  ctx.globalAlpha = 0.15;
  for (const [t, , mode] of history) if (mode === 1) ctx.fillRect(X(t - state.dt), 4, X(t) - X(t - state.dt) + 0.5, h - 20);
  ctx.globalAlpha = 1;
  ctx.strokeStyle = css("--border");
  ctx.beginPath(); ctx.moveTo(pad, Y(0)); ctx.lineTo(w - 6, Y(0)); ctx.stroke();
  ctx.strokeStyle = css("--text");
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  history.forEach(([t, v], k) => (k ? ctx.lineTo(X(t), Y(v)) : ctx.moveTo(X(t), Y(v))));
  ctx.stroke();
  ctx.fillStyle = css("--muted");
  ctx.font = "11px system-ui, sans-serif";
  ctx.fillText(fmt(hi, 1), 2, Y(hi) + 4);
  ctx.fillText(fmt(lo, 1), 2, Y(lo));
  ctx.fillText(`0 ~ ${T}s (초록 = 시청)`, pad, h - 3);
}

function renderStats() {
  const s = state;
  const rows = [
    ["시간", `${fmt(s.t, 1)} / ${s.T}s`],
    ["모드", s.mode + (s.current !== null ? ` #${s.current}` : "")],
    ["누적 보상", fmt(s.return)],
    ["시청 보상", fmt(s.watch_reward)],
    ["이해한 수", s.n_understood],
    ["클릭", s.n_clicks],
    ["목록 시간", `${fmt(s.time_by_mode.LIST, 1)}s`],
    ["시청 시간", `${fmt(s.time_by_mode.WATCH, 1)}s`],
    ["전환 시간", `${fmt(s.time_by_mode.TRANSITION, 1)}s`],
    ["직전 행동", `${s.last_action ?? "-"} (${fmt(s.last_reward, 3)})`],
  ];
  const table = $("stats");
  table.innerHTML = "";
  for (const [k, v] of rows) table.insertRow().innerHTML = `<td>${k}</td><td>${v}</td>`;
  for (const [k, v] of Object.entries(s.extra_metrics)) {
    const row = table.insertRow();
    row.className = "ext";
    row.innerHTML = `<td>${k}</td><td>${typeof v === "number" ? fmt(v) : v}</td>`;
  }
}

function renderEvents() {
  const list = $("events");
  list.innerHTML = "";
  for (const e of events.slice(-60).reverse()) {
    const li = document.createElement("li");
    li.className = e.kind;
    li.innerHTML = `<span class="t">${fmt(e.t, 1)}s</span>${e.text}`;
    list.appendChild(li);
  }
}

function renderKeys() {
  const human = isHuman();
  $("keys").innerHTML = human
    ? "직접 플레이: <kbd>↓</kbd>/<kbd>↑</kbd> 누르고 있으면 이동 (<kbd>Shift</kbd> = 빠르게) · <kbd>1</kbd>~<kbd>4</kbd> 보이는 칸 클릭 · <kbd>B</kbd> back · <kbd>Space</kbd> 멈춤 · 일시정지 중에는 키 한 번 = 한 스텝"
    : "<kbd>P</kbd> 재생/일시정지 · <kbd>N</kbd> 한 스텝 · <kbd>R</kbd> 다시 시작 · <kbd>H</kbd> 숨은 값 · 속도 10 = 실제 시간 (스텝 0.1초)";
  $("dwell").closest("label").style.opacity = $("policy").value === "greedy-dwell" ? 1 : 0.4;
}

function render() {
  if (!state) return;
  drawList();
  drawMinimap();
  drawChart();
  renderStats();
  renderEvents();
  renderKeys();
}

// --- input ---------------------------------------------------------------------------

function typing(e) {
  return ["INPUT", "SELECT", "TEXTAREA"].includes(e.target.tagName) && e.target.type !== "checkbox" && e.target.type !== "range";
}

document.addEventListener("keydown", (e) => {
  if (typing(e) || !state) return;
  const key = e.key;
  if (key === "Shift") { held.add("Shift"); return; }
  if (isHuman()) {
    let action = null;
    if (key === "ArrowDown" || key === "ArrowUp") {
      e.preventDefault();
      held.add(key);
      if (!playing) action = humanAction();
    } else if (/^[1-9]$/.test(key)) {
      action = state.actions.slots[parseInt(key, 10) - 1];
    } else if (key === "b" || key === "B" || key === "Backspace") {
      e.preventDefault();
      action = state.actions.back;
    } else if (key === " ") {
      e.preventDefault();
      action = state.actions.stay;
    }
    if (action !== null && action !== undefined) {
      if (playing) pending = action;
      else advance(1, action);
      return;
    }
  }
  if (key === "p" || key === "P") setPlaying(!playing);
  else if (key === "n" || key === "N") advance(1, isHuman() ? state.actions.stay : undefined);
  else if (key === "r" || key === "R") reset();
  else if (key === "h" || key === "H") { $("hidden").checked = !$("hidden").checked; saveSetup(); render(); }
});
document.addEventListener("keyup", (e) => held.delete(e.key));
window.addEventListener("blur", () => held.clear());
window.addEventListener("resize", render);

$("play").addEventListener("click", () => setPlaying(!playing));
$("step").addEventListener("click", () => advance(1, isHuman() ? humanAction() : undefined));
$("reset").addEventListener("click", reset);
$("seed-random").addEventListener("click", () => { $("seed").value = Math.floor(Math.random() * 100000); reset(); });
$("hidden").addEventListener("change", () => { saveSetup(); render(); });
$("speed").addEventListener("input", () => { $("speed-label").textContent = `${$("speed").value} 스텝/초`; saveSetup(); });
$("policy").addEventListener("change", () => { renderKeys(); });
$("defaults").addEventListener("click", () => {
  const map = fieldsByKey();
  for (const input of document.querySelectorAll("#settings [data-key]")) {
    setInput(input, map[input.dataset.key], map[input.dataset.key].default);
    markChanged(input, map[input.dataset.key]);
  }
});

// URL parameters (override saved settings): ?policy=greedy&seed=10001&speed=50&hidden=1
//   &rho=0.6&dwell=0.5&autoplay=1&skip=300&set=ddm.attention=log&set=feed.kappa=0.8
//   skip = run that many steps right after loading (jump into the episode)
function applyUrlParams() {
  const q = new URLSearchParams(location.search);
  if (q.has("policy") && [...$("policy").options].some((o) => o.value === q.get("policy"))) $("policy").value = q.get("policy");
  for (const k of ["seed", "speed", "rho", "dwell"]) if (q.has(k)) $(k).value = q.get(k);
  if (q.has("hidden")) $("hidden").checked = q.get("hidden") === "1";
  const map = fieldsByKey();
  for (const item of q.getAll("set")) {
    const [key, ...rest] = item.split("=");
    const input = document.querySelector(`#settings [data-key="${key}"]`);
    if (!input || !map[key]) { toast(`알 수 없는 설정: ${key}`); continue; }
    const raw = rest.join("=");
    const field = map[key];
    let value = raw;
    if (field.kind === "bool") value = ["1", "true", "True"].includes(raw);
    else if (field.kind === "tuple") value = raw.replace(/[()]/g, "").split(",").map(parseFloat).filter((v) => !Number.isNaN(v));
    setInput(input, field, value);
    markChanged(input, field);
  }
  return { autoplay: q.get("autoplay") === "1", skip: parseInt(q.get("skip") || "0", 10) };
}

// Commands from the branch comparison page (scripts/serve_branches.py):
// {type: "demo", cmd: "reset" | "play" | "pause" | "step", seed?, policy?, speed?, hidden?}
window.addEventListener("message", async (e) => {
  const m = e.data;
  if (!m || m.type !== "demo" || !schema) return;
  if (m.seed !== undefined) $("seed").value = m.seed;
  if (m.policy && [...$("policy").options].some((o) => o.value === m.policy)) $("policy").value = m.policy;
  if (m.speed !== undefined) { $("speed").value = m.speed; $("speed-label").textContent = `${m.speed} 스텝/초`; }
  if (m.hidden !== undefined) $("hidden").checked = m.hidden;
  if (m.cmd === "reset") await reset();
  else if (m.cmd === "play") setPlaying(true);
  else if (m.cmd === "pause") setPlaying(false);
  else if (m.cmd === "step") advance(1, isHuman() ? state.actions.stay : undefined);
  render();
});

// --- start ---------------------------------------------------------------------------

(async function start() {
  try {
    schema = await api("/api/schema");
  } catch (err) {
    toast(`서버에 연결할 수 없습니다: ${err.message}`);
    return;
  }
  if (COMPACT) document.body.classList.add("compact");
  $("label").textContent = schema.label;
  $("module").textContent = schema.module;
  document.title = `Scrolling Agent · ${schema.label || "demo"}`;
  for (const p of schema.policies) $("policy").add(new Option(p, p));
  $("policy").value = "greedy";
  buildSettings();
  loadSetup();
  const start = applyUrlParams();
  $("speed-label").textContent = `${$("speed").value} 스텝/초`;
  await reset();
  if (start.skip > 0 && !isHuman()) await advance(start.skip);
  if (start.autoplay) setPlaying(true);
  requestAnimationFrame(loop);
})();

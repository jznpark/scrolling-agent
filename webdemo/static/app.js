"use strict";
// Scrolling agent web demo: talks to webdemo/server.py and draws the environment.
// Left half: the phone screen (what the agent sees). Right half: reward rates.

const $ = (id) => document.getElementById(id);
const SESSION = (crypto.randomUUID ? crypto.randomUUID() : String(Math.random())).slice(0, 36);
const STORE_KEY = `scrolling-demo:${location.port}${location.pathname}`;  // per tab (/b/<k>/) on the comparison page
const COMPACT = new URLSearchParams(location.search).get("compact") === "1";
const RHO_POLICIES = ["scrollfirst", "greedy", "greedy-dwell"];

let schema = null;        // settings + policies + help texts from the server
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

function toast(message, kind = "error") {
  const el = $("toast");
  el.textContent = message;
  el.className = `toast ${kind}`;
  el.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => (el.hidden = true), kind === "info" ? 7000 : 4000);
}

// --- "?" tooltips ------------------------------------------------------------------

function helpText(el) {
  if (el.dataset.tip) return { key: el.dataset.key || "", text: el.dataset.tip };
  return { key: "", text: (schema && schema.ui_help[el.dataset.help]) || "" };
}

function showTip(el) {
  const { key, text } = helpText(el);
  if (!text) return;
  const tip = $("tooltip");
  tip.innerHTML = (key ? `<span class="key">${key}</span>` : "") + text;
  tip.hidden = false;
  const r = el.getBoundingClientRect();
  const w = tip.offsetWidth, h = tip.offsetHeight;
  let x = Math.min(r.left, window.innerWidth - w - 8);
  let y = r.bottom + 6;
  if (y + h > window.innerHeight - 8) y = r.top - h - 6;
  tip.style.left = `${Math.max(8, x)}px`;
  tip.style.top = `${Math.max(8, y)}px`;
}

document.addEventListener("mouseover", (e) => { const el = e.target.closest(".help"); if (el) showTip(el); });
document.addEventListener("mouseout", (e) => { if (e.target.closest(".help")) $("tooltip").hidden = true; });
document.addEventListener("focusin", (e) => { const el = e.target.closest(".help"); if (el) showTip(el); });
document.addEventListener("focusout", (e) => { if (e.target.closest(".help")) $("tooltip").hidden = true; });

function helpMark(tip, key) {
  const el = document.createElement("span");
  el.className = "help";
  el.tabIndex = 0;
  el.textContent = "?";
  el.dataset.tip = tip;
  if (key) el.dataset.key = key;
  return el;
}

// --- settings drawer ---------------------------------------------------------------

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
    const hasExt = section.fields.some((f) => f.tag === "ext");
    det.open = hasExt || section.name === "feed";
    const sum = document.createElement("summary");
    sum.textContent = section.name;
    if (hasExt) sum.innerHTML += ' <span class="tag ext">ext</span>';
    det.appendChild(sum);
    for (const field of section.fields) {
      const row = document.createElement("div");
      row.className = "field";
      const name = document.createElement("span");
      name.className = "name";
      name.textContent = field.label || field.name;
      if (field.tag) {
        const tag = document.createElement("span");
        tag.className = `tag ${field.tag.toLowerCase()}`;
        tag.textContent = field.tag;
        name.appendChild(tag);
      }
      const code = document.createElement("span");
      code.className = "code";
      code.textContent = field.key;
      name.appendChild(code);
      const tip = field.help + (field.comment && field.help !== field.comment ? `<br><span class="small" style="opacity:.7">config.py: ${field.comment}</span>` : "");
      row.append(name, helpMark(tip, `${field.label} · ${field.key}`), fieldInput(field));
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
    if (input.classList.contains("help")) continue;
    const field = map[input.dataset.key];
    if (isChanged(input, field)) out[field.key] = readInput(input, field);
  }
  return out;
}

function settingInput(key) {
  return document.querySelector(`#settings input[data-key="${key}"], #settings select[data-key="${key}"]`);
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
      const input = settingInput(key);
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

function openDrawer(open) {
  $("drawer").hidden = !open;
}

// --- situations (scenarios) ---------------------------------------------------------

function buildScenarios() {
  const sel = $("scenario");
  sel.innerHTML = "";
  sel.add(new Option("직접 설정 (지금 값 그대로)", ""));
  for (const sc of schema.scenarios) sel.add(new Option(sc.name, sc.id));
}

function setAllDefaults() {
  const map = fieldsByKey();
  for (const input of document.querySelectorAll("#settings input[data-key], #settings select[data-key]")) {
    setInput(input, map[input.dataset.key], map[input.dataset.key].default);
    markChanged(input, map[input.dataset.key]);
  }
}

async function applyScenario(id) {
  const sc = schema.scenarios.find((s) => s.id === id);
  const note = $("scenario-note");
  if (!sc) { note.hidden = true; return; }
  setAllDefaults();
  const map = fieldsByKey();
  for (const [key, value] of Object.entries(sc.set)) {
    const input = settingInput(key);
    if (input) { setInput(input, map[key], value); markChanged(input, map[key]); }
  }
  if (sc.policy) $("policy").value = sc.policy;
  if (sc.dwell !== undefined) $("dwell").value = sc.dwell;
  const changed = Object.keys(sc.set).map((k) => `<b>${map[k].label}</b>`).join(", ");
  note.innerHTML = `<b>${sc.name}</b>${changed ? ` — 바뀐 설정: ${changed}` : ""}<br>예상되는 행동: ${sc.expect}`;
  note.hidden = false;
  renderKeys();
  await reset();
  runCompare();
}

function changedSettingsText() {
  const map = fieldsByKey();
  const ov = overrides();
  const keys = Object.keys(ov);
  if (!keys.length) return "바꾼 설정이 없습니다 — 기본 설정끼리 비교합니다. 위에서 '상황'을 고르거나 ⚙ 설정에서 값을 바꿔 보세요.";
  return "기본과 다른 설정: " + keys.map((k) => {
    const v = Array.isArray(ov[k]) ? ov[k].join(", ") : ov[k];
    const d = Array.isArray(map[k].default) ? map[k].default.join(", ") : map[k].default;
    return `<b>${map[k].label}</b> ${fmt(d, 3)} → ${fmt(v, 3)}`;
  }).join(" · ");
}

// --- behaviour comparison ------------------------------------------------------------

let comparing = false;

async function runCompare() {
  if (comparing) return;
  comparing = true;
  const n = parseInt($("compare-n").value, 10);
  $("compare-run").disabled = true;
  $("compare-status").textContent = `피드 ${n}개를 기본 설정과 지금 설정으로 각각 돌리는 중… (몇 초 걸립니다)`;
  $("compare-changed").innerHTML = changedSettingsText();
  try {
    const res = await api("api/compare", { overrides: overrides(), policy: policySpec(), n,
      seed: parseInt($("seed").value, 10) });
    renderCompare(res);
    $("compare-status").textContent = `피드 ${res.n}개 (번호 ${res.seeds[0]}~${res.seeds[1]}) · 정책 ${$("policy").value} · ${res.seconds}초 걸림`;
  } catch (err) {
    $("compare-status").textContent = "";
    toast(`비교 실패: ${err.message}`);
  } finally {
    comparing = false;
    $("compare-run").disabled = false;
  }
}

function change(row) {
  const { base, now } = row;
  if (Math.abs(now - base) < 1e-9 || (Math.abs(base) < 0.05 && Math.abs(now) < 0.05)) return { text: "변화 없음", cls: "same", size: 0 };
  if (Math.abs(base) < 1e-6) return { text: `0 → ${fmt(now, 1)}`, cls: now > 0 ? "up" : "down", size: 9 };
  const ratio = now / base;
  if (base > 0 && ratio > 0 && (ratio >= 1.5 || ratio <= 0.67)) {
    const text = ratio >= 1 ? `${fmt(ratio, 1)}배 ↑` : `${fmt(1 / ratio, 1)}분의 1 ↓`;
    return { text, cls: ratio >= 1 ? "up" : "down", size: Math.abs(Math.log(ratio)) };
  }
  const pct = ((now - base) / Math.abs(base)) * 100;
  if (Math.abs(pct) < 5) return { text: "거의 같음", cls: "same", size: 0 };
  return { text: `${pct > 0 ? "+" : ""}${pct.toFixed(0)}%`, cls: pct > 0 ? "up" : "down", size: Math.abs(pct) / 100 };
}

function renderCompare(res) {
  const table = $("compare-table");
  table.innerHTML = "<tr><th>행동</th><th class='num'>기본 설정</th><th class='num'>지금 설정</th><th></th><th class='chg'>변화</th></tr>";
  const changes = [];
  for (const row of res.rows) {
    const c = change(row);
    if (row.key !== "return" && c.size > 0) changes.push({ row, c });
    const max = Math.max(Math.abs(row.base), Math.abs(row.now), 1e-9);
    const tr = table.insertRow();
    tr.innerHTML =
      `<td>${row.label}${row.hint ? `<div class="hint">${row.hint}</div>` : ""}</td>` +
      `<td class="num">${fmt(row.base, 1)}${row.unit}</td><td class="num"><b>${fmt(row.now, 1)}${row.unit}</b></td>` +
      `<td class="bars"><div class="bar base" style="width:${(100 * Math.abs(row.base)) / max}%"></div>` +
      `<div class="bar now" style="width:${(100 * Math.abs(row.now)) / max}%"></div></td>` +
      `<td class="chg ${c.cls}">${c.text}</td>`;
  }
  const ret = res.rows.find((r) => r.key === "return");
  const top = changes.sort((a, b) => b.c.size - a.c.size).slice(0, 3);
  const parts = top.map(({ row, c }) => `<b>${row.label}</b> ${c.text}`);
  let summary = parts.length ? `가장 크게 달라진 행동: ${parts.join(", ")}.` : "행동이 거의 달라지지 않았습니다.";
  if (ret) {
    const c = change(ret);
    summary += ` 얻은 재미는 ${fmt(ret.base, 1)} → ${fmt(ret.now, 1)} (${c.text}).`;
  }
  $("compare-summary").innerHTML = summary;
}

// --- episode control -----------------------------------------------------------------

async function reset() {
  setPlaying(false);
  try {
    state = await api("api/reset", { session: SESSION, seed: parseInt($("seed").value, 10),
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

const MOMENTS = {
  late: (e) => `${e.text.split(" ")[0]}을(를) 화면에서 지나간 뒤에야 알아차렸어요. ▶을 누르면 다시 올라가 여는지 볼 수 있습니다.`,
  click_late: (e) => `${e.text.split(" ")[0]}: 뒤늦게 알아챈 콘텐츠로 되돌아와 열었어요!`,
};

function momentOf(e) {
  if (e.kind === "late") return MOMENTS.late(e);
  if (e.kind === "click" && e.text.includes("뒤늦게")) return MOMENTS.click_late(e);
  return null;
}

async function advance(n, action) {
  if (inflight || !state || state.done) return;
  inflight = true;
  if ($("pause-moments").checked) n = Math.min(n, 2);  // small batches so the pause lands on the moment
  try {
    const body = { session: SESSION, n, since_event: lastEventId };
    if (action !== undefined && action !== null) body.action = action;
    const next = await api("api/step", body);
    history.push(...next.history);
    for (const e of next.events) events.push(e);
    if (next.events.length) lastEventId = next.events[next.events.length - 1].id;
    if (events.length > 300) events = events.slice(-300);
    state = next;
    if (state.done) setPlaying(false);
    if ($("pause-moments").checked && playing) {
      const moment = next.events.map(momentOf).find(Boolean);
      if (moment) { setPlaying(false); toast(moment, "info"); }
    }
    render();
  } catch (err) {
    setPlaying(false);
    toast(err.message);
  } finally {
    inflight = false;
  }
}

function setPlaying(on) {
  playing = Boolean(on && state && !state.done);
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

// --- drawing helpers -----------------------------------------------------------------

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
const clock = (t) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;
const usesRho = () => RHO_POLICIES.includes($("policy").value);

function text(ctx, str, x, y, { size = 12, weight = 400, color = css("--text"), align = "left", italic = false } = {}) {
  ctx.font = `${italic ? "italic " : ""}${weight} ${size}px system-ui, "Malgun Gothic", sans-serif`;
  ctx.fillStyle = color;
  ctx.textAlign = align;
  ctx.textBaseline = "middle";
  ctx.fillText(str, x, y);
  ctx.textAlign = "left";
}

// --- left half: the phone screen -------------------------------------------------------

function drawPhone() {
  const { ctx, w, h } = fitCanvas($("phone"));
  ctx.clearRect(0, 0, w, h);
  if (!state) return;
  const statusH = 30, homeH = 18;
  const feedTop = statusH, feedH = h - statusH - homeH;

  // status bar
  text(ctx, clock(state.t), 18, statusH / 2 + 2, { size: 13, weight: 600 });
  text(ctx, `${fmt(state.t, 1)} / ${state.T}s`, w - 16, statusH / 2 + 2, { size: 11, color: css("--muted"), align: "right" });

  ctx.save();
  ctx.beginPath();
  ctx.rect(0, feedTop, w, feedH);
  ctx.clip();
  const watchingView = state.mode === "WATCH" || (state.mode === "TRANSITION" && state.transition_to === "LIST");
  if (watchingView) drawPlayer(ctx, w, feedTop, feedH);
  else drawFeed(ctx, w, feedTop, feedH);
  if (state.mode === "TRANSITION") {
    ctx.fillStyle = "rgba(0,0,0,0.45)";
    ctx.fillRect(0, feedTop, w, feedH);
    text(ctx, state.transition_to === "WATCH" ? `#${state.current} 여는 중…` : "목록으로 돌아가는 중…",
      w / 2, feedTop + feedH / 2, { size: 15, weight: 600, color: "#fff", align: "center" });
  }
  if (state.done) {
    ctx.fillStyle = "rgba(0,0,0,0.65)";
    ctx.fillRect(0, feedTop, w, feedH);
    text(ctx, "세션 종료", w / 2, feedTop + feedH / 2 - 12, { size: 17, weight: 700, color: "#fff", align: "center" });
    text(ctx, `누적 보상 ${fmt(state.return)} · R로 다시 시작`, w / 2, feedTop + feedH / 2 + 14,
      { size: 12, color: "#fff", align: "center" });
  }
  ctx.restore();

  // home indicator
  ctx.fillStyle = css("--muted");
  ctx.globalAlpha = 0.5;
  ctx.beginPath();
  ctx.roundRect(w / 2 - 50, h - homeH / 2 - 2, 100, 4, 2);
  ctx.fill();
  ctx.globalAlpha = 1;

  const pill = $("phone-mode");
  pill.textContent = state.auto_preview
    ? `#${state.auto_preview.item} 자동 미리보기 (${fmt(state.auto_preview.seconds, 1)}초)`
    : state.mode === "WATCH" ? `#${state.current} 보는 중` : state.mode === "TRANSITION" ? "여닫는 중" : "목록 보는 중";
  pill.className = `pill ${state.auto_preview ? "PREVIEW" : state.mode}`;
}

function drawFeed(ctx, w, top, height) {
  const H = state.screen_h;
  const pitch = height / H;
  const showHidden = $("hidden").checked;
  const first = Math.max(0, Math.floor(state.scroll));
  const last = Math.min(state.n_items - 1, Math.ceil(state.scroll + H));
  for (let i = first; i <= last; i++) {
    const it = state.items[i];
    const y = top + (i - state.scroll) * pitch;
    const slot = state.slots.indexOf(i);
    const x0 = 10, cw = w - 20, ch = pitch - 10, cy = y + 5;
    ctx.globalAlpha = slot >= 0 ? 1 : 0.45;   // less than half on screen: not being processed
    ctx.fillStyle = it.understood ? css("--card-known") : css("--card");
    ctx.strokeStyle = css("--border");
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(x0, cy, cw, ch, 12);
    ctx.fill();
    ctx.stroke();

    // thumbnail placeholder
    const tx = x0 + 10, ty = cy + 10, tw = cw - 20, th = ch * 0.48;
    ctx.fillStyle = css("--unseen");
    ctx.beginPath();
    ctx.roundRect(tx, ty, tw, th, 8);
    ctx.fill();
    text(ctx, `#${i}`, tx + tw / 2, ty + th / 2, { size: Math.min(22, th * 0.35), weight: 700, color: css("--muted"), align: "center" });
    if (slot >= 0) {
      const canClick = state.mask[state.actions.slots[slot]];
      ctx.fillStyle = canClick ? css("--known") : css("--muted");
      ctx.beginPath();
      ctx.arc(tx + 16, ty + 16, 11, 0, Math.PI * 2);
      ctx.fill();
      text(ctx, String(slot + 1), tx + 16, ty + 16.5, { size: 12, weight: 700, color: "#fff", align: "center" });
    }

    // understanding progress (evidence) under the thumbnail
    const by = ty + th + 8, bh = 5;
    ctx.fillStyle = css("--unseen");
    ctx.fillRect(tx, by, tw, bh);
    ctx.fillStyle = it.understood ? css("--known") : css("--evidence");
    ctx.fillRect(tx, by, tw * Math.max(0, Math.min(1, it.understood ? 1 : it.x)), bh);

    const ly = by + bh + Math.max(12, (cy + ch - by - bh) * 0.32);
    if (it.late) {
      ctx.fillStyle = css("--late");
      ctx.beginPath();
      ctx.roundRect(tx + tw - 104, ty + 6, 98, 22, 11);
      ctx.fill();
      text(ctx, "★ 뒤늦게 알아챔", tx + tw - 55, ty + 17.5, { size: 11, weight: 700, color: "#111", align: "center" });
    }
    if (it.understood) {
      text(ctx, "알아봄", tx, ly, { size: 12, weight: 700, color: css("--known") });
      text(ctx, `지금 재미 ${fmt(it.rate)}` + (it.w > 0 ? ` · ${fmt(it.w, 1)}초 봄` : ""), tx + 48, ly, { size: 12 });
    } else {
      text(ctx, it.seen ? `알아보는 중 ${Math.round(Math.max(0, it.x) * 100)}% · ${fmt(it.exposure, 1)}초 봄` : "아직 안 봄", tx, ly,
        { size: 12, color: css("--muted") });
    }
    if (showHidden) {
      const differs = it.understood && (Math.abs(it.A - it.hidden.A) > 1e-6 || Math.abs(it.lam - it.hidden.lam) > 1e-6);
      const hidden = it.understood
        ? (differs ? `착각! 실제 지금 재미 ${fmt(it.hidden.rate)}` : "")
        : `실제: 재미 ${fmt(it.hidden.A)} · 알아보기 쉬움 ${fmt(it.hidden.nu)}${it.hidden.z ? " · 좋은 콘텐츠" : ""}`;
      if (hidden) text(ctx, hidden, tx, ly + 17, { size: 11, italic: true,
        color: differs ? css("--current") : it.hidden.z ? css("--z1") : css("--hidden") });
    }
    ctx.globalAlpha = 1;
  }
}

function drawPlayer(ctx, w, top, height) {
  const it = state.items[state.current];
  ctx.fillStyle = "#0b0c0f";
  ctx.fillRect(0, top, w, height);
  const cx = w / 2, cy = top + height * 0.42;
  ctx.fillStyle = "rgba(255,255,255,0.12)";
  ctx.beginPath();
  ctx.arc(cx, cy, 42, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = "#fff";
  ctx.beginPath();
  ctx.moveTo(cx - 13, cy - 20);
  ctx.lineTo(cx + 22, cy);
  ctx.lineTo(cx - 13, cy + 20);
  ctx.closePath();
  ctx.fill();
  text(ctx, `#${it.id}`, cx, cy + 72, { size: 20, weight: 700, color: "#fff", align: "center" });
  text(ctx, `본 시간 ${fmt(it.w, 1)}초`, cx, cy + 100, { size: 13, color: "#c9cdd4", align: "center" });
  // progress line at the bottom of the video
  const px = 16, py = top + height - 40, pw = w - 32;
  ctx.fillStyle = "rgba(255,255,255,0.25)";
  ctx.fillRect(px, py, pw, 4);
  ctx.fillStyle = css("--current");
  ctx.fillRect(px, py, pw * Math.min(1, it.w / Math.max(10, it.w + 4)), 4);
  text(ctx, "B = 목록으로", cx, py + 20, { size: 11, color: "#9aa1ab", align: "center" });
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
    const y = it.id * rh, hh = Math.max(1, rh - 0.5);
    ctx.fillStyle = css("--unseen");
    ctx.fillRect(0, y, w, hh);
    if (it.understood) {
      ctx.fillStyle = css("--known");
      ctx.globalAlpha = 0.25 + 0.75 * Math.max(0, it.rate) / maxRate;
      ctx.fillRect(0, y, w, hh);
      ctx.globalAlpha = 1;
    } else if (it.seen) {
      ctx.fillStyle = css("--evidence");
      ctx.fillRect(0, y, w * Math.max(0.05, Math.min(1, it.x)), hh);
    }
    if (it.w > 0) {
      ctx.fillStyle = css("--watch");
      ctx.fillRect(0, y, Math.min(w * 0.5, 4 + it.w * 3), hh);
    }
    if (showHidden && it.hidden.z) {
      ctx.fillStyle = css("--z1");
      ctx.fillRect(w - 6, y, 6, hh);
    }
    if (it.late) {
      ctx.fillStyle = css("--late");
      ctx.fillRect(w - 14, y - 1, 8, hh + 2);
    }
  }
  ctx.strokeStyle = css("--text");
  ctx.lineWidth = 2;
  ctx.strokeRect(1, state.scroll * rh, w - 2, state.screen_h * rh);
  if (state.current !== null) {
    ctx.fillStyle = css("--current");
    ctx.fillRect(0, state.current * rh - 1, w, Math.max(2, rh + 1));
  }
}

// --- right half: reward rates -----------------------------------------------------------

function axes(ctx, x0, y0, pw, ph) {
  ctx.strokeStyle = css("--border");
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(x0, y0 - ph);
  ctx.lineTo(x0, y0);
  ctx.lineTo(x0 + pw, y0);
  ctx.stroke();
}

function rhoLine(ctx, x0, pw, Y) {
  if (!usesRho()) return;
  const rho = parseFloat($("rho").value);
  ctx.setLineDash([5, 4]);
  ctx.strokeStyle = css("--muted");
  ctx.beginPath();
  ctx.moveTo(x0, Y(rho));
  ctx.lineTo(x0 + pw, Y(rho));
  ctx.stroke();
  ctx.setLineDash([]);
  text(ctx, `볼 만한 기준 ${fmt(rho)}`, x0 + pw - 2, Y(rho) - 8, { size: 11, color: css("--muted"), align: "right" });
}

function drawCurve() {
  const { ctx, w, h } = fitCanvas($("curve"));
  ctx.clearRect(0, 0, w, h);
  if (!state) return;
  const id = state.current !== null ? state.current : state.last_watched;
  const title = $("curve-title");
  if (id === null) {
    title.textContent = "보고 있는 콘텐츠의 재미";
    text(ctx, "아직 열어 본 콘텐츠가 없습니다.", w / 2, h / 2 - 10, { size: 13, color: css("--muted"), align: "center" });
    text(ctx, "콘텐츠를 열면, 볼수록 재미가 줄어드는 모습이 여기에 나타납니다.", w / 2, h / 2 + 12,
      { size: 12, color: css("--muted"), align: "center" });
    return;
  }
  const it = state.items[id];
  const watching = state.current !== null;
  title.textContent = watching ? `#${id}의 재미 (보는 중)` : `#${id}의 재미 (마지막으로 본 콘텐츠)`;
  const showHidden = $("hidden").checked;
  const x0 = 44, y0 = h - 26, pw = w - x0 - 14, ph = h - 46;
  const tMax = Math.max(10, it.w * 1.5 + 2);
  const rho = usesRho() ? parseFloat($("rho").value) : 0;
  const rMax = Math.max(it.A, showHidden ? it.hidden.A : 0, rho, 0.1) * 1.15;
  const X = (t) => x0 + (t / tMax) * pw;
  const Y = (r) => y0 - (r / rMax) * ph;
  axes(ctx, x0, y0, pw, ph);

  // watched part: area under the curve = reward collected
  ctx.fillStyle = css("--watch");
  ctx.globalAlpha = 0.3;
  ctx.beginPath();
  ctx.moveTo(X(0), Y(0));
  for (let k = 0; k <= 60; k++) { const t = (k / 60) * it.w; ctx.lineTo(X(t), Y(it.A * Math.exp(-it.lam * t))); }
  ctx.lineTo(X(it.w), Y(0));
  ctx.closePath();
  ctx.fill();
  ctx.globalAlpha = 1;

  const curve = (A, lam, color, dash) => {
    ctx.setLineDash(dash);
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let k = 0; k <= 120; k++) { const t = (k / 120) * tMax; const r = A * Math.exp(-lam * t); k ? ctx.lineTo(X(t), Y(r)) : ctx.moveTo(X(t), Y(r)); }
    ctx.stroke();
    ctx.setLineDash([]);
  };
  curve(it.A, it.lam, css("--known"), []);
  if (showHidden) curve(it.hidden.A, it.hidden.lam, css("--current"), [5, 4]);
  rhoLine(ctx, x0, pw, Y);

  ctx.fillStyle = css("--current");
  ctx.beginPath();
  ctx.arc(X(it.w), Y(it.rate), 5, 0, Math.PI * 2);
  ctx.fill();

  const gained = it.A / Math.max(it.lam, 1e-9) * (1 - Math.exp(-it.lam * it.w));
  text(ctx, `지금 재미 ${fmt(it.rate)}`, x0 + pw, 12, { size: 12, weight: 600, align: "right" });
  text(ctx, `본 시간 ${fmt(it.w, 1)}초 · 얻은 재미 ${fmt(gained)}` + (showHidden ? ` · 실제 지금 재미 ${fmt(it.hidden.rate)}` : ""),
    x0 + pw, 28, { size: 11, color: css("--muted"), align: "right" });
  text(ctx, fmt(rMax, 1), x0 - 6, Y(rMax), { size: 10, color: css("--muted"), align: "right" });
  text(ctx, "0", x0 - 6, y0, { size: 10, color: css("--muted"), align: "right" });
  text(ctx, `본 시간 (0 ~ ${tMax.toFixed(0)}초) →`, x0 + pw, y0 + 14, { size: 10, color: css("--muted"), align: "right" });
}

function drawRates() {
  const { ctx, w, h } = fitCanvas($("rates"));
  ctx.clearRect(0, 0, w, h);
  if (!state) return;
  const showHidden = $("hidden").checked;
  const known = state.items.filter((it) => it.understood);
  const seenMax = Math.max(...state.items.filter((it) => it.seen).map((it) => it.id), 0);
  const hiIdx = Math.min(state.n_items, Math.max(seenMax + 4, Math.ceil(state.scroll + state.screen_h) + 3, 12));
  const rho = usesRho() ? parseFloat($("rho").value) : 0;
  const rMax = Math.max(0.5, rho, ...known.map((it) => it.rate), ...(showHidden ? known.map((it) => it.hidden.rate) : [])) * 1.15;
  const x0 = 44, y0 = h - 22, pw = w - x0 - 10, ph = h - 34;
  const X = (i) => x0 + (i / hiIdx) * pw;
  const Y = (r) => y0 - (Math.max(0, r) / rMax) * ph;
  const bw = Math.max(2, pw / hiIdx - 2);

  // the screen window
  ctx.fillStyle = css("--evidence");
  ctx.globalAlpha = 0.12;
  ctx.fillRect(X(state.scroll), y0 - ph, X(state.scroll + state.screen_h) - X(state.scroll), ph);
  ctx.globalAlpha = 1;
  axes(ctx, x0, y0, pw, ph);

  for (const it of known) {
    const x = X(it.id) + 1;
    const current = state.current === it.id;
    ctx.fillStyle = current ? css("--current") : css("--known");
    ctx.fillRect(x, Y(it.rate), bw, y0 - Y(it.rate));
    if (state.slots.includes(it.id)) {
      ctx.strokeStyle = css("--evidence");
      ctx.lineWidth = 2;
      ctx.strokeRect(x, Y(it.rate), bw, y0 - Y(it.rate));
    }
    if (it.w > 0) {
      ctx.fillStyle = css("--watch");
      ctx.fillRect(x, Y(it.rate) - 3, bw, 3);
    }
    if (it.late) text(ctx, "★", x + bw / 2, Y(it.rate) - 10, { size: 12, color: css("--late"), align: "center" });
    if (showHidden) {
      ctx.fillStyle = css("--current");
      ctx.beginPath();
      ctx.arc(x + bw / 2, Y(it.hidden.rate), 2.5, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  rhoLine(ctx, x0, pw, Y);
  if (!known.length) text(ctx, "아직 뭔지 알아본 콘텐츠가 없습니다", x0 + pw / 2, y0 - ph / 2, { size: 12, color: css("--muted"), align: "center" });
  text(ctx, fmt(rMax, 1), x0 - 6, Y(rMax), { size: 10, color: css("--muted"), align: "right" });
  text(ctx, "0", x0 - 6, y0, { size: 10, color: css("--muted"), align: "right" });
  text(ctx, `목록에서의 위치 0 ~ ${hiIdx}번 (파란 배경 = 지금 화면)`, x0 + pw, y0 + 13, { size: 10, color: css("--muted"), align: "right" });
}

function drawChart() {
  const { ctx, w, h } = fitCanvas($("chart"));
  ctx.clearRect(0, 0, w, h);
  if (!state) return;
  const T = state.T;
  const values = history.map((p) => p[1]);
  const lo = Math.min(0, ...values), hi = Math.max(1, ...values);
  const x0 = 44, y0 = h - 18, pw = w - x0 - 10, ph = h - 28;
  const X = (t) => x0 + (t / T) * pw;
  const Y = (v) => y0 - ((v - lo) / (hi - lo)) * ph;
  ctx.fillStyle = css("--known");
  ctx.globalAlpha = 0.15;
  for (const [t, , mode] of history) if (mode === 1) ctx.fillRect(X(t - state.dt), y0 - ph, X(t) - X(t - state.dt) + 0.5, ph);
  ctx.globalAlpha = 1;
  axes(ctx, x0, y0, pw, ph);
  ctx.strokeStyle = css("--border");
  ctx.beginPath(); ctx.moveTo(x0, Y(0)); ctx.lineTo(x0 + pw, Y(0)); ctx.stroke();
  ctx.strokeStyle = css("--text");
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  history.forEach(([t, v], k) => (k ? ctx.lineTo(X(t), Y(v)) : ctx.moveTo(X(t), Y(v))));
  ctx.stroke();
  text(ctx, fmt(hi, 1), x0 - 6, Y(hi), { size: 10, color: css("--muted"), align: "right" });
  text(ctx, fmt(lo, 1), x0 - 6, Y(lo), { size: 10, color: css("--muted"), align: "right" });
  text(ctx, `시간 0 ~ ${T}초 →`, x0 + pw, y0 + 11, { size: 10, color: css("--muted"), align: "right" });
}

function renderStats() {
  const s = state;
  const rows = [
    ["지금", { LIST: "목록 보는 중", WATCH: `#${s.current} 보는 중`, TRANSITION: "여닫는 중" }[s.mode]],
    ["얻은 재미 (점수)", fmt(s.return)],
    ["본 것에서 얻은 재미", fmt(s.watch_reward)],
    ["뭔지 알아본 수", s.n_understood],
    ["열어 본 횟수", s.n_clicks],
    ["목록을 본 시간", `${fmt(s.time_by_mode.LIST, 1)}초`],
    ["콘텐츠를 본 시간", `${fmt(s.time_by_mode.WATCH, 1)}초`],
    ["여닫는 데 쓴 시간", `${fmt(s.time_by_mode.TRANSITION, 1)}초`],
    ["방금 한 행동", `${s.last_action ?? "-"}`],
  ];
  const table = $("stats");
  table.innerHTML = "";
  for (const [k, v] of rows) table.insertRow().innerHTML = `<td>${k}</td><td>${v}</td>`;
  for (const [k, v] of Object.entries(s.extra_metrics)) {
    const row = table.insertRow();
    row.className = "ext";
    row.innerHTML = `<td>${(schema.behaviours && schema.behaviours[k]) || k}</td><td>${typeof v === "number" ? fmt(v) : v}</td>`;
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
  $("keys").innerHTML = isHuman()
    ? "직접 해 보기: <kbd>↓</kbd>/<kbd>↑</kbd> 누르고 있으면 스크롤 (<kbd>Shift</kbd> 빠르게) · <kbd>1</kbd>~<kbd>4</kbd> 번호 붙은 카드 열기 · <kbd>B</kbd> 닫고 목록으로 · <kbd>Space</kbd> 멈춤<br>일시정지 중에는 키 한 번 = 0.1초"
    : "<kbd>P</kbd> 재생/일시정지 · <kbd>N</kbd> 0.1초 진행 · <kbd>R</kbd> 다시 시작 · <kbd>H</kbd> 숨은 값 · 흐린 카드 = 절반도 안 보여서 아직 눈에 안 들어온 카드";
  $("dwell-wrap").style.opacity = $("policy").value === "greedy-dwell" ? 1 : 0.4;
}

function render() {
  if (!state) return;
  drawPhone();
  drawMinimap();
  drawCurve();
  drawRates();
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
  if (e.key === "Escape") { openDrawer(false); return; }
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
$("apply").addEventListener("click", () => { openDrawer(false); reset(); });
$("open-settings").addEventListener("click", () => openDrawer($("drawer").hidden));
$("close-settings").addEventListener("click", () => openDrawer(false));
$("seed-random").addEventListener("click", () => { $("seed").value = Math.floor(Math.random() * 100000); reset(); });
$("hidden").addEventListener("change", () => { saveSetup(); render(); });
$("rho").addEventListener("input", render);
$("speed").addEventListener("input", () => { $("speed-label").textContent = `${$("speed").value}/초`; saveSetup(); });
$("policy").addEventListener("change", renderKeys);
$("defaults").addEventListener("click", setAllDefaults);
$("scenario").addEventListener("change", () => applyScenario($("scenario").value));
$("compare-run").addEventListener("click", runCompare);
$("settings").addEventListener("change", () => { $("scenario").value = ""; $("scenario-note").hidden = true; });
for (const btn of document.querySelectorAll(".step-btn[data-target]")) {
  btn.addEventListener("click", () => {
    const input = $(btn.dataset.target);
    const delta = parseFloat(btn.dataset.delta);
    const value = Math.max(0, parseFloat(input.value || "0") + delta);
    input.value = Number.isInteger(delta) ? String(Math.round(value)) : String(Math.round(value * 100) / 100);
    input.dispatchEvent(new Event("input"));
    if (btn.dataset.target === "seed") reset();
  });
}

// URL parameters (override saved settings): ?policy=greedy&seed=10001&speed=50&hidden=1
//   &rho=0.6&dwell=0.5&autoplay=1&skip=300&set=ddm.attention=log&set=feed.kappa=0.8
//   skip = run that many steps right after loading (jump into the episode)
//   scenario = id of a ready-made situation (webdemo/descriptions.py SCENARIOS)
function applyUrlParams() {
  const q = new URLSearchParams(location.search);
  if (q.has("policy") && [...$("policy").options].some((o) => o.value === q.get("policy"))) $("policy").value = q.get("policy");
  for (const k of ["seed", "speed", "rho", "dwell"]) if (q.has(k)) $(k).value = q.get(k);
  if (q.has("hidden")) $("hidden").checked = q.get("hidden") === "1";
  if (q.has("pause")) $("pause-moments").checked = q.get("pause") === "1";
  const map = fieldsByKey();
  for (const item of q.getAll("set")) {
    const [key, ...rest] = item.split("=");
    const input = settingInput(key);
    if (!input || !map[key]) { toast(`알 수 없는 설정: ${key}`); continue; }
    const raw = rest.join("=");
    const field = map[key];
    let value = raw;
    if (field.kind === "bool") value = ["1", "true", "True"].includes(raw);
    else if (field.kind === "tuple") value = raw.replace(/[()]/g, "").split(",").map(parseFloat).filter((v) => !Number.isNaN(v));
    setInput(input, field, value);
    markChanged(input, field);
  }
  return { autoplay: q.get("autoplay") === "1", skip: parseInt(q.get("skip") || "0", 10), scenario: q.get("scenario") };
}

// Commands from the branch comparison page (scripts/serve_branches.py):
// {type: "demo", cmd: "reset" | "play" | "pause" | "step", seed?, policy?, speed?, hidden?}
window.addEventListener("message", async (e) => {
  const m = e.data;
  if (!m || m.type !== "demo" || !schema) return;
  if (m.seed !== undefined) $("seed").value = m.seed;
  if (m.policy && [...$("policy").options].some((o) => o.value === m.policy)) $("policy").value = m.policy;
  if (m.speed !== undefined) { $("speed").value = m.speed; $("speed-label").textContent = `${m.speed}/초`; }
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
    schema = await api("api/schema");
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
  for (const el of document.querySelectorAll(".help[data-help]")) el.tabIndex = 0;
  buildSettings();
  buildScenarios();
  loadSetup();
  const start = applyUrlParams();
  $("speed-label").textContent = `${$("speed").value}/초`;
  if (start.scenario && schema.scenarios.some((sc) => sc.id === start.scenario)) {
    $("scenario").value = start.scenario;
    await applyScenario(start.scenario);
  } else {
    await reset();
  }
  if (start.skip > 0 && !isHuman()) await advance(start.skip);
  if (start.autoplay) setPlaying(true);
  $("compare-changed").innerHTML = changedSettingsText();
  requestAnimationFrame(loop);
})();

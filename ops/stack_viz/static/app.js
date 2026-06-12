const $ = (id) => document.getElementById(id);

let lastState = null;
let jsonpSeq = 0;
let failedPolls = 0;

function pct(value) {
  const n = Number(value || 0);
  return Math.max(0, Math.min(100, n));
}

function setBar(id, value, warnAt = 85) {
  const bar = $(id);
  if (!bar) return;
  const wrap = bar.parentElement;
  const p = pct(value);
  bar.style.width = `${p}%`;
  wrap.classList.toggle("warn", p >= warnAt);
  wrap.classList.toggle("hot", p >= 70 && p < warnAt);
}

function fmtMiB(mib) {
  const n = Number(mib || 0);
  if (n >= 1024) return `${(n / 1024).toFixed(1)} GB`;
  return `${Math.round(n)} MB`;
}

function ageLabel(seconds) {
  const s = Math.max(0, Number(seconds || 0));
  if (s < 60) return `${Math.round(s)}s`;
  if (s < 3600) return `${Math.round(s / 60)}m`;
  return `${(s / 3600).toFixed(1)}h`;
}

function shortText(text, limit = 190) {
  const value = String(text || "");
  return value.length > limit ? `${value.slice(0, limit - 3)}...` : value;
}

function updateTop(state) {
  const gpu = state.gpu || {};
  const stack = state.stack || {};
  const metrics = state.metrics || {};
  const admission = state.admission || {};
  const queue = state.idle_queue || {};

  $("clock").textContent = new Date(state.timestamp).toLocaleTimeString();

  const pill = $("admissionPill");
  pill.textContent = admission.status === "tight" ? "guard tight" : "guard open";
  pill.classList.toggle("tight", admission.status === "tight");

  $("gpuUtil").textContent = `${gpu.gpu_util_pct || 0}%`;
  $("gpuName").textContent = `30s avg ${gpu.gpu_util_avg_30s || 0}% | max ${gpu.gpu_util_max_30s || 0}% | ${Math.round(gpu.power_w || 0)}W | ${gpu.temp_c || 0}C`;
  setBar("gpuUtilBar", gpu.gpu_util_pct || 0, 92);

  $("vramUsed").textContent = `${(gpu.used_pct || 0).toFixed(1)}%`;
  $("vramFree").textContent = `${fmtMiB(gpu.free_mib)} free | allocated by loaded models/jobs`;
  setBar("vramBar", gpu.used_pct || 0, 85);

  $("modelCount").textContent = String(stack.active_training_count || 0);
  $("modelCaption").textContent = `${stack.active_llm_count || 0} local LLM servers | ${queue.queued || 0} queued`;
  setBar("modelBar", Math.min(100, (stack.active_training_count || 0) * 45), 90);

  const step = metrics.latest_step;
  if (step) {
    $("stepCount").textContent = `${step.step}/${step.total}`;
    setBar("stepBar", (step.step / Math.max(1, step.total)) * 100, 100);
  } else {
    $("stepCount").textContent = "--";
    setBar("stepBar", 0, 100);
  }
  const loss = metrics.latest_loss;
  const speed = metrics.latest_speed;
  $("lossCaption").textContent = loss == null ? "waiting for training signal" : `loss ${Number(loss).toFixed(4)} | ${speed || "-"} batches/sec`;
}

function nodeHtml(node) {
  const metrics = (node.metrics || []).map((m) => `
    <div class="node-metric">
      <span>${escapeHtml(m.label)}</span>
      <strong>${escapeHtml(m.value)}</strong>
    </div>
  `).join("");
  return `
    <article class="stack-node ${node.status || "idle"}" data-node="${escapeHtml(node.id)}">
      <div>
        <div class="node-top">
          <h3 class="node-title">${escapeHtml(node.label)}</h3>
          <i class="status-dot ${node.status || "idle"}"></i>
        </div>
        <p class="node-sub">${escapeHtml(node.sub || "")}</p>
      </div>
      <div>
        <div class="node-meter"><i style="width:${pct(node.meter || 0)}%"></i></div>
        <div class="node-metrics">${metrics}</div>
      </div>
    </article>
  `;
}

function linkHtml(flow) {
  const intensity = Number(flow.intensity || 0);
  const count = intensity > 0.72 ? 3 : intensity > 0.35 ? 2 : 1;
  const speed = Math.max(0.75, 2.6 - intensity * 1.7);
  let dots = "";
  for (let i = 0; i < count; i += 1) {
    dots += `<i class="flow-dot" style="animation-duration:${speed}s"></i>`;
  }
  return `<div class="pipe-link" data-from="${escapeHtml(flow.from)}" data-to="${escapeHtml(flow.to)}">${dots}</div>`;
}

function updateStack(state) {
  const stack = state.stack || {};
  const nodes = stack.nodes || [];
  const flows = stack.flows || [];
  const pieces = [];
  nodes.forEach((node, index) => {
    pieces.push(nodeHtml(node));
    if (index < flows.length) pieces.push(linkHtml(flows[index]));
  });
  $("stackStrip").innerHTML = pieces.join("");
}

function jobTitle(job) {
  if (job.model) return job.model;
  if (job.kind) return job.kind;
  return job.name || `pid ${job.pid}`;
}

function updateJobs(state) {
  const jobs = (state.processes || [])
    .filter((p) => p.vram_mib > 128 || ["training", "uncertainty", "model_server", "lakehouse", "bronze"].includes(p.stage))
    .slice(0, 12);
  if (!jobs.length) {
    $("jobList").innerHTML = `<p class="empty">No GPU-heavy jobs currently visible.</p>`;
    return;
  }
  $("jobList").innerHTML = jobs.map((job) => `
    <article class="job">
      <div class="job-head">
        <div class="job-title">${escapeHtml(jobTitle(job))}</div>
        <div class="job-stage">${escapeHtml(job.stage || "process")}</div>
      </div>
      <div class="job-stats">
        <div class="job-stat">VRAM<strong>${fmtMiB(job.vram_mib)}</strong></div>
        <div class="job-stat">RAM<strong>${fmtMiB(job.ram_mb)}</strong></div>
        <div class="job-stat">PID<strong>${job.pid}</strong></div>
      </div>
      <p class="job-command">${escapeHtml(shortText(job.command, 230))}</p>
    </article>
  `).join("");
}

function updateQueue(state) {
  const queue = state.idle_queue || {};
  const jobs = (queue.jobs || []).slice(0, 8);
  const target = $("queueList");
  if (!target) return;

  if (!queue.configured) {
    target.innerHTML = `<p class="empty">No idle queue configured.</p>`;
    return;
  }
  if (!jobs.length) {
    target.innerHTML = `<p class="empty">Idle queue is empty.</p>`;
    return;
  }

  target.innerHTML = jobs.map((job) => `
    <article class="queue-item">
      <p class="queue-title">
        <span>${escapeHtml(job.label || job.id)}</span>
        <b class="queue-status ${escapeHtml(job.status || "queued")}">${escapeHtml(job.status || "queued")}</b>
      </p>
      <p class="queue-reason">${escapeHtml(shortText(job.reason || `priority ${job.priority || 0}`, 150))}</p>
    </article>
  `).join("");
}

function updateEvents(state) {
  const events = (state.events || []).slice(0, 24);
  if (!events.length) {
    $("eventList").innerHTML = `<p class="empty">No recent model events found.</p>`;
    return;
  }
  $("eventList").innerHTML = events.map((event) => `
    <article class="event">
      <div class="event-badge ${event.severity || "info"}">${escapeHtml(event.stage || "event")}</div>
      <div>
        <p class="event-text">${escapeHtml(shortText(event.text, 320))}</p>
        <p class="event-source">${escapeHtml(event.source || "")}</p>
      </div>
    </article>
  `).join("");
}

function updateArtifacts(state) {
  const artifacts = (state.artifacts || []).slice(0, 22);
  if (!artifacts.length) {
    $("artifactList").innerHTML = `<p class="empty">No lakehouse artifacts found.</p>`;
    return;
  }
  $("artifactList").innerHTML = artifacts.map((artifact) => `
    <article class="artifact-row">
      <div class="artifact-main">
        <p class="artifact-name">${escapeHtml(artifact.name || artifact.kind || "artifact")}</p>
        <p class="artifact-path">${escapeHtml(shortText(artifact.path, 180))}</p>
      </div>
      <div class="artifact-meta">${escapeHtml(artifact.size_label || "")}<br>${ageLabel(artifact.age_s)}</div>
    </article>
  `).join("");
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function render(state) {
  lastState = state;
  failedPolls = 0;
  updateTop(state);
  updateStack(state);
  updateJobs(state);
  updateQueue(state);
  updateEvents(state);
  updateArtifacts(state);
}

async function refresh() {
  try {
    const state = await requestJson("/api/state");
    render(state);
  } catch (error) {
    failedPolls += 1;
    if (!lastState) {
      $("admissionPill").textContent = "waiting for server";
      $("admissionPill").classList.add("tight");
    }
    if (lastState && failedPolls >= 3) {
      window.setTimeout(() => window.location.reload(), 2000);
    }
  }
}

function requestJson(url) {
  if (typeof fetch === "function") {
    return fetch(url, { cache: "no-store" }).then((response) => {
      if (!response.ok) throw new Error(`state ${response.status}`);
      return response.json();
    });
  }
  if (typeof XMLHttpRequest === "function") {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("GET", `${url}?t=${Date.now()}`, true);
      xhr.setRequestHeader("Accept", "application/json");
      xhr.onload = () => {
        if (xhr.status < 200 || xhr.status >= 300) {
          reject(new Error(`state ${xhr.status}`));
          return;
        }
        try {
          resolve(JSON.parse(xhr.responseText));
        } catch (error) {
          reject(error);
        }
      };
      xhr.onerror = () => reject(new Error("network error"));
      xhr.send();
    });
  }
  return requestJsonp();
}

function requestJsonp() {
  return new Promise((resolve, reject) => {
    const callbackName = `StackVizState_${Date.now()}_${jsonpSeq += 1}`;
    const script = document.createElement("script");
    const cleanup = () => {
      delete window[callbackName];
      if (script.parentNode) script.parentNode.removeChild(script);
    };
    const timer = window.setTimeout(() => {
      cleanup();
      reject(new Error("state timeout"));
    }, 12000);
    window[callbackName] = (payload) => {
      window.clearTimeout(timer);
      cleanup();
      resolve(payload);
    };
    script.onerror = () => {
      window.clearTimeout(timer);
      cleanup();
      reject(new Error("script load error"));
    };
    script.src = `/api/state.js?callback=${encodeURIComponent(callbackName)}&t=${Date.now()}`;
    document.head.appendChild(script);
  });
}

if (window.__STACK_STATE__) {
  render(window.__STACK_STATE__);
}
refresh();
setInterval(refresh, 1500);

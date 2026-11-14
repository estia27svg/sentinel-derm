const state = {
  metrics: null,
  samples: [],
  currentFile: null,
  previewUrl: null,
  busy: false,
  camOpacity: 0.55,
};

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

const pct = (value, digits = 1) => {
  if (value === null || value === undefined || Number.isNaN(value)) return "--";
  const amount = value * 100;
  if (digits === 0 && amount > 0 && amount < 1) return "<1%";
  return `${amount.toFixed(digits)}%`;
};
const num = (value, digits = 2) =>
  value === null || value === undefined || Number.isNaN(value) ? "--" : Number(value).toFixed(digits);
const compact = (value) =>
  value === null || value === undefined || Number.isNaN(value)
    ? "--"
    : Math.abs(value) >= 1000
      ? `${(value / 1000).toFixed(1)}k`
      : Number(value).toFixed(1);
const int = (value) => Number(value || 0).toLocaleString("en-US");
const escapeHtml = (value) =>
  String(value).replace(/[&<>"']/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[character]));

/* ------------------------------------------------------------------ routing */

function showView(name) {
  $$(".view").forEach((view) => view.classList.toggle("is-active", view.id === `view-${name}`));
  $$(".tab").forEach((tab) => tab.classList.toggle("is-active", tab.dataset.view === name));
  if (location.hash !== `#${name}`) history.replaceState(null, "", `#${name}`);
  window.scrollTo({ top: 0, behavior: "smooth" });
}

$$("[data-view]").forEach((element) => {
  element.addEventListener("click", (event) => {
    event.preventDefault();
    showView(element.dataset.view);
  });
});

/* ------------------------------------------------------------------- upload */

const dropzone = $("#dropzone");
const fileInput = $("#file-input");
const previewWrap = $("#preview-wrap");
const preview = $("#preview");
const previewName = $("#preview-name");

function setFile(file) {
  if (!file) return;
  if (!file.type.startsWith("image/")) {
    renderError("That file is not an image. Please choose a JPEG or PNG.");
    return;
  }
  if (file.size > 12 * 1024 * 1024) {
    renderError("That image is larger than 12 MB.");
    return;
  }
  state.currentFile = file;
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.previewUrl = URL.createObjectURL(file);
  preview.src = state.previewUrl;
  previewName.textContent = file.name || "pasted image";
  previewWrap.hidden = false;
  dropzone.hidden = true;
  $$(".sample").forEach((node) => node.classList.remove("is-active"));
  analyze(file);
}

function clearFile() {
  state.currentFile = null;
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.previewUrl = null;
  preview.removeAttribute("src");
  previewWrap.hidden = true;
  dropzone.hidden = false;
  fileInput.value = "";
  $$(".sample").forEach((node) => node.classList.remove("is-active"));
  resetResult();
}

fileInput.addEventListener("change", () => setFile(fileInput.files[0]));
$("#clear-button").addEventListener("click", clearFile);

["dragenter", "dragover"].forEach((type) =>
  dropzone.addEventListener(type, (event) => {
    event.preventDefault();
    dropzone.classList.add("is-dragging");
  })
);
["dragleave", "drop"].forEach((type) =>
  dropzone.addEventListener(type, (event) => {
    event.preventDefault();
    dropzone.classList.remove("is-dragging");
  })
);
dropzone.addEventListener("drop", (event) => setFile(event.dataTransfer.files[0]));
dropzone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    fileInput.click();
  }
});

/* ------------------------------------------------------------------ analyze */

async function analyze(file) {
  if (state.busy) return;
  state.busy = true;
  showLoading();

  try {
    const body = new FormData();
    body.append("file", file, file.name || "upload.jpg");
    const response = await fetch("/api/analyze", { method: "POST", body });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.detail || `Request failed (${response.status})`);
    renderResult(payload);
  } catch (error) {
    renderError(error.message || "Something went wrong during analysis.");
  } finally {
    state.busy = false;
  }
}

function showLoading() {
  $("#empty-state").hidden = true;
  const body = $("#result-body");
  body.hidden = false;
  body.innerHTML = `
    <div class="result-loading">
      <div class="spinner" aria-hidden="true"></div>
      <span>Running the model, conformal calibration and out-of-domain check&hellip;</span>
    </div>`;
}

function resetResult() {
  const body = $("#result-body");
  body.hidden = true;
  body.innerHTML = "";
  $("#empty-state").hidden = false;
}

function renderError(message) {
  $("#empty-state").hidden = true;
  const body = $("#result-body");
  body.hidden = false;
  body.innerHTML = `<div class="error-box"><b>Could not analyse this image.</b><br />${escapeHtml(message)}</div>`;
}

/* ------------------------------------------------------------------- result */

const STATUS_COPY = {
  confident: "Confident single diagnosis",
  ambiguous: "Ambiguous - abstained",
  uncertain: "No reliable differential",
  out_of_distribution: "Out of domain - rejected",
  low_quality: "Image quality too low",
};

function renderResult(result) {
  const body = $("#result-body");
  body.hidden = false;
  $("#empty-state").hidden = true;

  const rejected = result.status === "out_of_distribution" || result.status === "low_quality";
  const coverage = state.metrics?.conformal?.coverage ?? result.guarantee.target_coverage;
  const targetCoverage = result.guarantee.target_coverage;

  const triage = result.triage || {};
  const triageHtml = `
    <div class="triage-card ${escapeHtml(triage.urgency || "none")}">
      <div class="triage-label">Triage recommendation</div>
      <div class="triage-title">${escapeHtml(triage.label || "Not assessed")}</div>
      <div class="triage-advice">${escapeHtml(triage.advice || "")}</div>
    </div>`;

  const chips = result.prediction_set.length
    ? result.prediction_set
        .map((code) => {
          const entry = result.probabilities.find((item) => item.code === code);
          const klass = result.status === "ambiguous" ? "chip in-set abstain" : "chip in-set";
          return `<span class="${klass}">${escapeHtml(entry.short)}<span class="chip-prob">${pct(entry.probability, 1)}</span></span>`;
        })
        .join("")
    : `<span class="chip abstain">No class cleared its calibrated threshold</span>`;

  const candidates = rejected
    ? ""
    : `
      <p class="section-label">Candidate diagnoses <span>class probability</span></p>
      <div class="candidates">
        ${result.probabilities
          .map(
            (item) => `
          <div class="candidate ${item.in_set ? "" : "is-out"}">
            <span class="candidate-name" title="${escapeHtml(item.name)}">${escapeHtml(item.short)}</span>
            <span class="candidate-bar"><span style="width:${Math.max(1.5, item.probability * 100)}%"></span></span>
            <span class="candidate-prob">${pct(item.probability, 1)}</span>
          </div>`
          )
          .join("")}
      </div>`;

  const squares = Array.from({ length: 10 }, (_, index) => `<i class="${index / 10 < targetCoverage - 0.001 ? "on" : ""}"></i>`).join("");

  const camHtml = result.gradcam
    ? `
      <div class="cam-block">
        <p class="section-label">Where the model looked <span>Grad-CAM</span></p>
        <div class="cam-frame">
          <img src="${state.previewUrl}" alt="Original lesion image" />
          <img class="cam-overlay" id="cam-overlay" src="${result.gradcam}" alt="Grad-CAM saliency overlay" style="opacity:${state.camOpacity}" />
        </div>
        <div class="cam-controls">
          <span>original</span>
          <input id="cam-range" type="range" min="0" max="1" step="0.01" value="${state.camOpacity}" aria-label="Grad-CAM overlay opacity" />
          <span>saliency</span>
        </div>
      </div>`
    : "";

  body.innerHTML = `
    <div class="result-head">
      <span class="status-pill ${escapeHtml(result.status)}"><span class="dot-sm"></span>${escapeHtml(STATUS_COPY[result.status] || result.status)}</span>
      <span class="hint">${num(result.timings_ms.total, 0)} ms on CPU</span>
    </div>
    ${triageHtml}
    <p class="section-label">${rejected ? "Input assessment" : `${pct(targetCoverage, 0)} prediction set`} <span>${rejected ? "no diagnosis" : `${result.prediction_set.length} candidate${result.prediction_set.length === 1 ? "" : "s"}`}</span></p>
    <div class="chips">${rejected ? `<span class="chip abstain">${escapeHtml(rejectedLabel(result))}</span>` : chips}</div>
    ${candidates}
    <div class="insight-box">
      <svg class="insight-icon" viewBox="0 0 20 20" aria-hidden="true"><circle cx="10" cy="10" r="7.5"/><path d="M10 6.2v4.6M10 13.6v.2"/></svg>
      <div class="insight-text">${escapeHtml(result.explanation)}</div>
    </div>
    ${
      rejected
        ? ""
        : `<div class="coverage-proof">
             <div class="coverage-squares" aria-hidden="true">${squares}</div>
             <div class="coverage-copy"><b>${pct(coverage, 1)}</b> empirical coverage on the held-out test split at a ${pct(targetCoverage, 0)} target. The true diagnosis lands in a set like this one that often.</div>
           </div>`
    }
    ${camHtml}
    <dl class="diagnostics">
      <div class="diagnostic"><dt>Backbone</dt><dd>${escapeHtml(result.model.backbone)}</dd></div>
      <div class="diagnostic"><dt>In-domain score</dt><dd>${compact(result.ood.score)} / ${compact(result.ood.threshold)}</dd></div>
      <div class="diagnostic"><dt>Sharpness</dt><dd>${num(result.quality.sharpness, 1)}</dd></div>
      <div class="diagnostic"><dt>Calibration n</dt><dd>${int(result.guarantee.n_calibration)}</dd></div>
    </dl>`;

  const range = $("#cam-range");
  if (range) {
    range.addEventListener("input", (event) => {
      state.camOpacity = Number(event.target.value);
      $("#cam-overlay").style.opacity = state.camOpacity;
    });
  }

  const panel = $("#result-panel");
  const bounds = panel.getBoundingClientRect();
  if (window.innerWidth < 980 || bounds.top > window.innerHeight - 80 || bounds.bottom < 80) {
    panel.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

function rejectedLabel(result) {
  return result.status === "low_quality" ? "Image did not pass the quality gate" : "Not a dermoscopic image";
}

/* ------------------------------------------------------------------ samples */

async function loadSamples() {
  const container = $("#samples");
  try {
    const response = await fetch("/api/samples");
    const payload = await response.json();
    state.samples = payload.samples || [];
    if (!state.samples.length) {
      container.innerHTML = `<div class="samples-loading">No sample cases found. Run training/train.py to generate them.</div>`;
      return;
    }
    container.innerHTML = "";
    state.samples.forEach((sample) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "sample";
      button.dataset.id = sample.image_id;
      button.innerHTML = `
        <img src="${sample.image_url}" alt="${escapeHtml(sample.true_class)} sample" />
        <span class="sample-meta"><span class="sample-tag">${escapeHtml(sample.tag)}</span></span>`;
      button.addEventListener("click", () => runSample(sample, button));
      container.appendChild(button);
    });

    const rejectButton = document.createElement("button");
    rejectButton.type = "button";
    rejectButton.className = "sample";
    rejectButton.innerHTML = `
      <img src="/api/demo/ood-image?v=${Date.now()}" alt="Non-skin photograph" />
      <span class="sample-meta"><span class="sample-tag">non-skin photo</span></span>`;
    rejectButton.addEventListener("click", () => runOodDemo(rejectButton));
    container.appendChild(rejectButton);
  } catch (error) {
    container.innerHTML = `<div class="samples-loading">Could not load sample cases: ${escapeHtml(error.message)}</div>`;
  }
}

async function runSample(sample, button) {
  if (state.busy) return;
  $$(".sample").forEach((node) => node.classList.toggle("is-active", node === button));
  const response = await fetch(sample.image_url);
  const blob = await response.blob();
  const file = new File([blob], `${sample.image_id}.jpg`, { type: "image/jpeg" });
  state.currentFile = file;
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.previewUrl = URL.createObjectURL(file);
  preview.src = state.previewUrl;
  previewName.textContent = `${sample.image_id} · true label: ${sample.true_class}`;
  previewWrap.hidden = false;
  dropzone.hidden = true;
  analyze(file);
}

async function runOodDemo(button) {
  if (state.busy) return;
  $$(".sample").forEach((node) => node.classList.toggle("is-active", node === button));
  const response = await fetch(`/api/demo/ood-image?v=${Date.now()}`, { cache: "no-store" });
  const blob = await response.blob();
  const file = new File([blob], "non_skin_photo.jpg", { type: "image/jpeg" });
  state.currentFile = file;
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.previewUrl = URL.createObjectURL(file);
  preview.src = state.previewUrl;
  previewName.textContent = "natural image · outside the dermoscopic domain";
  previewWrap.hidden = false;
  dropzone.hidden = true;
  analyze(file);
}

/* ------------------------------------------------------------------ metrics */

function metricCard(value, label, note = "") {
  return `<div class="metric-card">
    <div class="metric-value">${value}</div>
    <div class="metric-label">${label}</div>
    ${note ? `<div class="metric-note">${note}</div>` : ""}
  </div>`;
}

function renderMetrics(metrics) {
  const classifier = metrics.classifier;
  const conformal = metrics.conformal;
  const ood = metrics.ood || {};

  $("#metric-grid").innerHTML = [
    metricCard(pct(conformal.coverage, 1), "Empirical coverage", `target ${pct(conformal.target_coverage, 0)}`),
    metricCard(num(conformal.avg_set_size, 2), "Average prediction-set size", `${pct(conformal.singleton_rate, 0)} singleton`),
    metricCard(pct(conformal.selective_accuracy, 1), "Selective accuracy", "when the model commits"),
    metricCard(pct(classifier.macro_f1, 1), "Macro F1", "7 classes, imbalanced"),
    metricCard(pct(classifier.per_class_recall.mel, 1), "Melanoma recall", `${int(classifier.support.mel)} melanoma cases`),
    metricCard(ood.auroc !== null && ood.auroc !== undefined ? num(ood.auroc, 5) : "--", "Out-of-domain AUROC", "rejection of non-skin images"),
  ].join("");

  const tbody = $("#class-table tbody");
  const classInfo = {
    akiec: "Actinic keratosis",
    bcc: "Basal cell carcinoma",
    bkl: "Benign keratosis",
    df: "Dermatofibroma",
    mel: "Melanoma",
    nv: "Melanocytic nevus",
    vasc: "Vascular lesion",
  };
  tbody.innerHTML = metrics.classifier.support
    ? Object.keys(classInfo)
        .map((code) => {
          const support = classifier.support[code] ?? 0;
          const recall = classifier.per_class_recall[code];
          const precision = classifier.per_class_precision[code];
          const covered = conformal.coverage_by_class?.[code];
          return `<tr>
            <td>${classInfo[code]}</td>
            <td class="num">${int(support)}</td>
            <td class="num">${pct(recall, 1)}</td>
            <td class="num">${pct(precision, 1)}</td>
            <td><span class="pill-mini">${pct(covered, 1)}</span></td>
          </tr>`;
        })
        .join("")
    : "";

  const dataset = metrics.dataset || {};
  const splits = dataset.splits || {};
  $("#dataset-copy").textContent = `${int(dataset.total_images)} dermoscopic images from HAM10000, deduplicated and split by lesion into ${int(
    splits.train
  )} train, ${int(splits.cal)} calibration and ${int(splits.test)} test images. The calibration split is used only to set conformal thresholds; the test split is untouched until evaluation.`;
}

function renderShell(metrics) {
  const conformal = metrics.conformal;
  $("#hero-stats").innerHTML = [
    `<div class="stat"><span class="stat-value">${pct(conformal.target_coverage, 0)}</span><span class="stat-label">coverage target</span></div>`,
    `<div class="stat"><span class="stat-value">${pct(conformal.coverage, 1)}</span><span class="stat-label">achieved</span></div>`,
    `<div class="stat"><span class="stat-value">${num(conformal.avg_set_size, 2)}</span><span class="stat-label">avg set size</span></div>`,
    `<div class="stat"><span class="stat-value">${int(metrics.dataset.total_images)}</span><span class="stat-label">images</span></div>`,
  ].join("");

  $("#topbar-meta").textContent = `HAM10000 · n=${int(metrics.dataset.total_images)} · lesion-level splits`;
  $("#footer-meta").textContent = `backbone ${metrics.backbone} · calibrated ${new Date(metrics.generated_at).toLocaleDateString(
    "en-GB",
    { day: "2-digit", month: "short", year: "numeric" }
  )} · alpha ${metrics.alpha}`;
}

/* --------------------------------------------------------------------- init */

async function init() {
  const hash = (location.hash || "#analyze").replace("#", "");
  if (["analyze", "evidence", "method"].includes(hash)) showView(hash);

  try {
    const response = await fetch("/api/metrics");
    if (!response.ok) throw new Error((await response.json()).detail || "metrics unavailable");
    state.metrics = await response.json();
    renderMetrics(state.metrics);
    renderShell(state.metrics);
  } catch (error) {
    $("#topbar-meta").textContent = "model not trained yet";
    $("#metric-grid").innerHTML = `<div class="error-box">${escapeHtml(error.message)}<br />Run <code>python training/train.py</code> first.</div>`;
  }

  loadSamples();
}

window.addEventListener("hashchange", () => {
  const hash = (location.hash || "#analyze").replace("#", "");
  if (["analyze", "evidence", "method"].includes(hash)) showView(hash);
});

init();

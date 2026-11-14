const state = {
  metrics: null,
  samples: [],
  samplesError: null,
  currentFile: null,
  previewUrl: null,
  previewInfo: null,
  activeSample: null,
  lastResult: null,
  mode: "smartphone",
  journalEntries: [],
  journalFile: null,
  journalPreviewUrl: null,
  journalDb: null,
  demo: false,
  demoScenario: "good",
  demoResultScenario: null,
  demoCheckBusy: false,
  demoTimer: null,
  busy: false,
  camOpacity: 0.55,
};

const I18N = window.SentinelI18N;
const t = (key, params) => I18N.t(key, params);
const pct = (value, digits = 1) => I18N.percent(value, digits);
const num = (value, digits = 2) =>
  value === null || value === undefined || Number.isNaN(value) ? "--" : I18N.number(Number(value), digits);
const compact = (value) =>
  value === null || value === undefined || Number.isNaN(value)
    ? "--"
    : Math.abs(value) >= 1000
      ? `${I18N.number(value / 1000, 1)}k`
      : I18N.number(Number(value), 1);
const int = (value) => I18N.number(Number(value || 0), 0);
const classText = (code, field) => t(`class.${code}.${field}`);
const escapeHtml = (value) =>
  String(value).replace(/[&<>"']/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[character]));

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

const CLASS_CODES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"];

const BACKBONES = {
  efficientnet_b0: { label: "EfficientNet-B0", dimension: 1280 },
  convnext_tiny: { label: "ConvNeXt-Tiny", dimension: 768 },
};

const SAMPLE_TAG_KEYS = {
  "clear melanoma": "clear_melanoma",
  "clear nevus": "clear_nevus",
  ambiguous: "ambiguous",
  "basal cell carcinoma": "bcc",
  "benign keratosis": "bkl",
};

/* ------------------------------------------------------------------ routing */

const VIEW_NAMES = ["analyze", "journal", "evidence", "method", "demo"];

function showView(name) {
  $$(".view").forEach((view) => view.classList.toggle("is-active", view.id === `view-${name}`));
  $$(".tab").forEach((tab) => tab.classList.toggle("is-active", tab.dataset.view === name));
  if (location.hash !== `#${name}`) history.replaceState(null, "", `#${name}`);
  window.scrollTo({ top: 0, behavior: "smooth" });
}

$$("[data-view]").forEach((element) => {
  element.addEventListener("click", (event) => {
    event.preventDefault();
    const target = element.dataset.view;
    if (state.demo && target !== "demo") stopDemo();
    if (target === "demo") startDemo();
    else showView(target);
  });
});

/* ------------------------------------------------------------------- upload */

const dropzone = $("#dropzone");
const fileInput = $("#file-input");
const previewWrap = $("#preview-wrap");
const preview = $("#preview");
const previewName = $("#preview-name");
const modeOptions = $$(".mode-option");

function updateModeCopy() {
  const dermoscopy = state.mode === "dermoscopy";
  $("#submit-hint").textContent = t(dermoscopy ? "hint.dermoscopy" : "hint.phone");
  $("#mode-note").textContent = t(dermoscopy ? "mode.dermNote" : "mode.phoneNote");
  $("#drop-title").textContent = t(dermoscopy ? "drop.title" : "drop.phoneTitle");
  $("#drop-meta").textContent = t(dermoscopy ? "drop.meta" : "drop.phoneMeta");
  $("#samples-panel").hidden = !dermoscopy;
}

function setMode(mode, options = {}) {
  if (mode === state.mode) {
    updateModeCopy();
    return;
  }
  state.mode = mode;
  modeOptions.forEach((button) => {
    const active = button.dataset.mode === mode;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-pressed", String(active));
  });
  updateModeCopy();
  if (options.rerun === false) return;
  if (state.currentFile) analyze(state.currentFile);
  else resetResult();
}

modeOptions.forEach((button) => button.addEventListener("click", () => setMode(button.dataset.mode)));

function updatePreviewText() {
  if (!state.previewInfo) {
    previewName.textContent = "";
    return;
  }
  if (state.previewInfo.type === "sample") {
    previewName.textContent = t("preview.sample", {
      id: state.previewInfo.id,
      label: classText(state.previewInfo.label, "short"),
    });
    return;
  }
  if (state.previewInfo.type === "ood") {
    previewName.textContent = t("preview.ood");
    return;
  }
  previewName.textContent = state.previewInfo.name || t("preview.pasted");
}

function setPreviewSource(file, info) {
  state.currentFile = file;
  state.previewInfo = info;
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.previewUrl = URL.createObjectURL(file);
  preview.src = state.previewUrl;
  updatePreviewText();
  previewWrap.hidden = false;
  dropzone.hidden = true;
}

function setFile(file) {
  if (!file) return;
  if (!file.type.startsWith("image/")) {
    renderError(t("error.fileType"));
    return;
  }
  if (file.size > 12 * 1024 * 1024) {
    renderError(t("error.fileSize"));
    return;
  }
  state.activeSample = null;
  $$(".sample").forEach((node) => node.classList.remove("is-active"));
  setPreviewSource(file, { type: "file", name: file.name });
  analyze(file);
}

function clearFile() {
  state.currentFile = null;
  state.previewInfo = null;
  state.activeSample = null;
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
    const endpoint = state.mode === "dermoscopy" ? "/api/analyze" : "/api/guidance";
    const response = await fetch(endpoint, { method: "POST", body });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.detail || t("error.request", { status: response.status }));
    state.lastResult = payload;
    renderActiveResult(payload);
  } catch (error) {
    renderError(error.message || t("error.analysis"));
  } finally {
    state.busy = false;
  }
}

function renderActiveResult(result, options = {}) {
  if (result.mode === "smartphone" || result.scope === "guidance_only_no_diagnosis") {
    renderGuidanceResult(result, options);
  } else {
    renderResult(result, options);
  }
}

function showLoading() {
  state.lastResult = null;
  $("#empty-state").hidden = true;
  const body = $("#result-body");
  body.hidden = false;
  body.innerHTML = `
    <div class="result-loading">
      <div class="spinner" aria-hidden="true"></div>
      <span>${escapeHtml(t(state.mode === "dermoscopy" ? "loading" : "loading.guidance"))}</span>
    </div>`;
}

function resetResult() {
  state.lastResult = null;
  const body = $("#result-body");
  body.hidden = true;
  body.innerHTML = "";
  $("#empty-state").hidden = false;
}

function renderError(message) {
  state.lastResult = null;
  $("#empty-state").hidden = true;
  const body = $("#result-body");
  body.hidden = false;
  body.innerHTML = `<div class="error-box"><b>${escapeHtml(t("error.title"))}</b><br />${escapeHtml(message)}</div>`;
}

/* ------------------------------------------------------------------- result */

function triageCopy(result) {
  const triage = result.triage || {};
  const code = triage.code;
  if (code === "class" && triage.class_code) {
    return {
      label: classText(triage.class_code, "urgencyLabel"),
      advice: classText(triage.class_code, "advice"),
    };
  }
  if (code === "uncertain_refer") {
    return { label: t("triage.uncertain_refer"), advice: t("triage.uncertain_refer.advice") };
  }
  if (code === "not_assessed") {
    return { label: t("triage.not_assessed"), advice: t("triage.not_assessed.advice") };
  }
  return { label: triage.label || t("triage.not_assessed"), advice: triage.advice || "" };
}

function explanationCopy(result) {
  const code = result.explanation_code;
  if (!code) return result.explanation || "";
  const params = result.explanation_params || {};
  const coverage = pct(result.guarantee?.target_coverage ?? 0.9, 0);

  if (code === "low_quality") {
    const issues = (params.issues || []).map((issue) =>
      issue === "resolution"
        ? t("quality.resolution", {
            width: result.quality.width,
            height: result.quality.height,
            minimum: result.quality.minimum ?? 96,
          })
        : t("quality.focus")
    );
    const joined = I18N.list(issues);
    const capitalised = joined.charAt(0).toLocaleUpperCase() + joined.slice(1);
    return t("explanation.low_quality", { issues: capitalised });
  }
  if (code === "out_of_distribution") return t("explanation.out_of_distribution");
  if (code === "uncertain") return t("explanation.uncertain");
  if (code === "confident") {
    return t("explanation.confident", {
      coverage,
      class: params.class_code ? classText(params.class_code, "full") : "",
      probability: pct(params.probability, 0),
    });
  }
  if (code === "ambiguous") {
    const classes = (params.class_codes || []).map((classCode) => classText(classCode, "short"));
    return t("explanation.ambiguous", { coverage, classes: I18N.list(classes) });
  }
  return result.explanation || "";
}

function guidanceCheckLabel(check) {
  const params = check.params || {};
  if (check.code === "readability") {
    return check.detail === "small"
      ? t("guidance.readability.small", { width: int(params.width), height: int(params.height), minimum: int(params.minimum) })
      : t("guidance.readability.ok", { width: int(params.width), height: int(params.height) });
  }
  if (check.code === "focus") {
    return t(check.state === "pass" ? "guidance.focus.ok" : "guidance.focus.blurred");
  }
  if (check.code === "lighting") {
    return t(`guidance.lighting.${check.detail || "ok"}`);
  }
  if (check.code === "skin") {
    return t(check.detail === "none" ? "guidance.skin.none" : "guidance.skin.ok");
  }
  return "";
}

function renderGuidanceResult(result, options = {}) {
  const body = $("#result-body");
  body.hidden = false;
  $("#empty-state").hidden = true;

  const statusKey =
    result.status === "guidance_ready"
      ? "guidance.statusReady"
      : result.status === "guidance_unclear"
        ? "guidance.statusUnclear"
        : "guidance.statusRetake";
  const detailKey =
    result.status === "guidance_ready"
      ? "guidance.detailReady"
      : result.status === "guidance_unclear"
        ? "guidance.detailUnclear"
        : "guidance.detailRetake";
  const statusClass = result.status === "guidance_ready" ? "confident" : "low_quality";

  const checks = (result.checks || [])
    .map(
      (check) => `<li class="check-item ${escapeHtml(check.state)}">
        <span class="check-mark" aria-hidden="true">${check.state === "pass" ? "&check;" : "!"}</span>
        <span>${escapeHtml(guidanceCheckLabel(check))}</span>
      </li>`
    )
    .join("");

  const nextSteps = [];
  if (result.status !== "guidance_ready") nextSteps.push(t("guidance.nextRetake"));
  nextSteps.push(t("guidance.nextClinician"), t("guidance.nextNoReassurance"));

  const quality = result.quality || {};

  body.innerHTML = `
    <p class="safety-strip">
      <svg viewBox="0 0 20 20" aria-hidden="true"><path d="M10 2.5l6.5 2.4v5.6c0 4-2.7 6.8-6.5 8-3.8-1.2-6.5-4-6.5-8V4.9L10 2.5z"/><path d="M10 7v3.6M10 13.2v.2"/></svg>
      <span>${escapeHtml(t("guidance.safety"))}</span>
    </p>
    <div class="result-head">
      <span class="status-pill ${statusClass}"><span class="dot-sm"></span>${escapeHtml(t(statusKey))}</span>
      <span class="hint">${escapeHtml(t("guidance.noAi"))}</span>
    </div>
    <p class="section-label">${escapeHtml(t("guidance.checksTitle"))}</p>
    <ul class="check-list">${checks}</ul>
    <div class="insight-box">
      <svg class="insight-icon" viewBox="0 0 20 20" aria-hidden="true"><circle cx="10" cy="10" r="7.5"/><path d="M10 6.2v4.6M10 13.6v.2"/></svg>
      <div class="insight-text">${escapeHtml(t(detailKey))}</div>
    </div>
    <div class="guidance">
      <p class="section-label">${escapeHtml(t("guidance.nextTitle"))}</p>
      <ul class="guidance-list">${nextSteps.map((step) => `<li>${escapeHtml(step)}</li>`).join("")}</ul>
    </div>
    <div class="guidance">
      <p class="section-label">${escapeHtml(t("guidance.eduTitle"))}</p>
      <ul class="guidance-list">
        <li>${escapeHtml(t("guidance.edu1"))}</li>
        <li>${escapeHtml(t("guidance.edu2"))}</li>
        <li>${escapeHtml(t("guidance.edu3"))}</li>
      </ul>
    </div>
    <dl class="diagnostics">
      <div class="diagnostic"><dt>${escapeHtml(t("diag.size"))}</dt><dd>${int(quality.width)} &times; ${int(quality.height)}</dd></div>
      <div class="diagnostic"><dt>${escapeHtml(t("diag.sharpness"))}</dt><dd>${num(quality.sharpness, 1)}</dd></div>
      <div class="diagnostic"><dt>${escapeHtml(t("diag.brightness"))}</dt><dd>${num(quality.brightness, 0)}</dd></div>
      <div class="diagnostic"><dt>${escapeHtml(t("diag.skin"))}</dt><dd>${pct(quality.skin_like_fraction, 1)}</dd></div>
    </dl>`;

  if (options.scroll === false) return;
  const panel = $("#result-panel");
  const bounds = panel.getBoundingClientRect();
  if (window.innerWidth < 980 || bounds.top > window.innerHeight - 80 || bounds.bottom < 80) {
    panel.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

function renderResult(result, options = {}) {
  const body = $("#result-body");
  body.hidden = false;
  $("#empty-state").hidden = true;

  const rejected = result.status === "out_of_distribution" || result.status === "low_quality";
  const coverage = state.metrics?.conformal?.coverage ?? result.guarantee.target_coverage;
  const targetCoverage = result.guarantee.target_coverage;
  const triage = triageCopy(result);
  const nextSteps = (result.triage?.next_steps || []).map((code) => t(`next.${code}`));

  const triageHtml = `
    <div class="triage-card ${escapeHtml(result.triage?.urgency || "none")}">
      <div class="triage-label">${escapeHtml(t("triage.title"))}</div>
      <div class="triage-title">${escapeHtml(triage.label)}</div>
      <div class="triage-advice">${escapeHtml(triage.advice)}</div>
    </div>`;

  const guidanceHtml = nextSteps.length
    ? `<div class="guidance">
         <p class="section-label">${escapeHtml(t("result.nextTitle"))}</p>
         <ul class="guidance-list">${nextSteps
           .map((step) => `<li>${escapeHtml(step)}</li>`)
           .join("")}</ul>
       </div>`
    : "";

  const chips = result.prediction_set.length
    ? result.prediction_set
        .map((code) => {
          const entry = result.probabilities.find((item) => item.code === code);
          const klass = result.status === "ambiguous" ? "chip in-set abstain" : "chip in-set";
          return `<span class="${klass}">${escapeHtml(classText(code, "short"))}<span class="chip-prob">${pct(
            entry.probability,
            1
          )}</span></span>`;
        })
        .join("")
    : `<span class="chip abstain">${escapeHtml(t("result.noClass"))}</span>`;

  const candidates = rejected
    ? ""
    : `
      <p class="section-label">${escapeHtml(t("result.candidates"))} <span>${escapeHtml(t("result.probability"))}</span></p>
      <div class="candidates">
        ${result.probabilities
          .map(
            (item) => `
          <div class="candidate ${item.in_set ? "" : "is-out"}">
            <span class="candidate-name" title="${escapeHtml(classText(item.code, "full"))}">${escapeHtml(
              classText(item.code, "short")
            )}</span>
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
        <p class="section-label">${escapeHtml(t("result.looked"))} <span>${escapeHtml(t("result.camTag"))}</span></p>
        <div class="cam-frame">
          <img src="${state.previewUrl}" alt="${escapeHtml(t("preview.alt"))}" />
          <img class="cam-overlay" id="cam-overlay" src="${result.gradcam}" alt="${escapeHtml(
            t("result.camTag")
          )}" style="opacity:${state.camOpacity}" />
        </div>
        <div class="cam-controls">
          <span>${escapeHtml(t("result.original"))}</span>
          <input id="cam-range" type="range" min="0" max="1" step="0.01" value="${state.camOpacity}" aria-label="${escapeHtml(
            t("result.camTag")
          )}" />
          <span>${escapeHtml(t("result.saliency"))}</span>
        </div>
      </div>`
    : "";

  const setLabel = rejected ? t("result.inputAssessment") : t("result.predictionSet", { coverage: pct(targetCoverage, 0) });
  const setNote = rejected
    ? t("result.noDiagnosis")
    : result.prediction_set.length === 1
      ? t("result.candidateOne", { count: 1 })
      : t("result.candidateMany", { count: result.prediction_set.length });

  const statusText = t(`status.${result.status}`);

  body.innerHTML = `
    <p class="safety-strip">
      <svg viewBox="0 0 20 20" aria-hidden="true"><path d="M10 2.5l6.5 2.4v5.6c0 4-2.7 6.8-6.5 8-3.8-1.2-6.5-4-6.5-8V4.9L10 2.5z"/><path d="M10 7v3.6M10 13.2v.2"/></svg>
      <span>${escapeHtml(t("result.safety"))}</span>
    </p>
    <div class="result-head">
      <span class="status-pill ${escapeHtml(result.status)}"><span class="dot-sm"></span>${escapeHtml(statusText)}</span>
      <span class="hint">${escapeHtml(t("result.msCpu", { ms: num(result.timings_ms.total, 0) }))}</span>
    </div>
    ${triageHtml}
    ${guidanceHtml}
    <p class="section-label">${escapeHtml(setLabel)} <span>${escapeHtml(setNote)}</span></p>
    <div class="chips">${
      rejected ? `<span class="chip abstain">${escapeHtml(t(`rejected.${result.status}`))}</span>` : chips
    }</div>
    <p class="set-explainer">${escapeHtml(t("result.setExplainer"))}</p>
    ${candidates}
    <div class="insight-box">
      <svg class="insight-icon" viewBox="0 0 20 20" aria-hidden="true"><circle cx="10" cy="10" r="7.5"/><path d="M10 6.2v4.6M10 13.6v.2"/></svg>
      <div class="insight-text">${escapeHtml(explanationCopy(result))}</div>
    </div>
    ${
      rejected
        ? ""
        : `<div class="coverage-proof">
             <div class="coverage-squares" aria-hidden="true">${squares}</div>
             <div class="coverage-copy">${t("result.coverageProof", {
               coverage: pct(coverage, 1),
               target: pct(targetCoverage, 0),
             })}</div>
           </div>`
    }
    ${camHtml}
    <dl class="diagnostics">
      <div class="diagnostic"><dt>${escapeHtml(t("diag.backbone"))}</dt><dd>${escapeHtml(result.model.backbone)}</dd></div>
      <div class="diagnostic"><dt>${escapeHtml(t("diag.indomain"))}</dt><dd>${compact(result.ood.score)} / ${compact(
        result.ood.threshold
      )}</dd></div>
      <div class="diagnostic"><dt>${escapeHtml(t("diag.sharpness"))}</dt><dd>${num(result.quality.sharpness, 1)}</dd></div>
      <div class="diagnostic"><dt>${escapeHtml(t("diag.calibration"))}</dt><dd>${int(result.guarantee.n_calibration)}</dd></div>
    </dl>`;

  const range = $("#cam-range");
  if (range) {
    range.addEventListener("input", (event) => {
      state.camOpacity = Number(event.target.value);
      $("#cam-overlay").style.opacity = state.camOpacity;
    });
  }

  if (options.scroll === false) return;
  const panel = $("#result-panel");
  const bounds = panel.getBoundingClientRect();
  if (window.innerWidth < 980 || bounds.top > window.innerHeight - 80 || bounds.bottom < 80) {
    panel.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

/* ------------------------------------------------------------------ samples */

function sampleTagCopy(tag) {
  const key = SAMPLE_TAG_KEYS[String(tag || "").toLowerCase()];
  return key ? t(`sample.tags.${key}`) : tag || "";
}

function renderSamples() {
  const container = $("#samples");
  if (state.samplesError) {
    container.innerHTML = `<div class="samples-loading">${t("samples.error", {
      message: escapeHtml(state.samplesError),
    })}</div>`;
    return;
  }
  if (!state.samples.length) {
    container.innerHTML = `<div class="samples-loading">${escapeHtml(t("samples.empty"))}</div>`;
    return;
  }

  const fragment = document.createDocumentFragment();
  state.samples.forEach((sample) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "sample";
    button.dataset.sampleKey = sample.image_id;
    button.classList.toggle("is-active", state.activeSample === sample.image_id);
    button.innerHTML = `
      <img src="${sample.image_url}" alt="${escapeHtml(
        t("sample.alt", { label: classText(sample.true_class, "short") })
      )}" />
      <span class="sample-meta"><span class="sample-tag">${escapeHtml(sampleTagCopy(sample.tag))}</span></span>`;
    button.addEventListener("click", () => runSample(sample, button));
    fragment.appendChild(button);
  });

  const rejectButton = document.createElement("button");
  rejectButton.type = "button";
  rejectButton.className = "sample";
  rejectButton.dataset.sampleKey = "ood";
  rejectButton.classList.toggle("is-active", state.activeSample === "ood");
  rejectButton.innerHTML = `
    <img src="/api/demo/ood-image?v=${Date.now()}" alt="${escapeHtml(t("sample.oodAlt"))}" />
    <span class="sample-meta"><span class="sample-tag">${escapeHtml(t("sample.tags.non_skin"))}</span></span>`;
  rejectButton.addEventListener("click", () => runOodDemo(rejectButton));
  fragment.appendChild(rejectButton);

  container.replaceChildren(fragment);
}

async function loadSamples() {
  try {
    const response = await fetch("/api/samples");
    const payload = await response.json();
    state.samples = payload.samples || [];
    state.samplesError = null;
  } catch (error) {
    state.samples = [];
    state.samplesError = error.message || "unknown error";
  }
  renderSamples();
}

async function runSample(sample, button) {
  if (state.busy) return;
  setMode("dermoscopy", { rerun: false });
  state.activeSample = sample.image_id;
  $$(".sample").forEach((node) => node.classList.toggle("is-active", node === button));
  const response = await fetch(sample.image_url);
  const blob = await response.blob();
  const file = new File([blob], `${sample.image_id}.jpg`, { type: "image/jpeg" });
  setPreviewSource(file, { type: "sample", id: sample.image_id, label: sample.true_class });
  analyze(file);
}

async function runOodDemo(button) {
  if (state.busy) return;
  setMode("dermoscopy", { rerun: false });
  state.activeSample = "ood";
  $$(".sample").forEach((node) => node.classList.toggle("is-active", node === button));
  const response = await fetch(`/api/demo/ood-image?v=${Date.now()}`, { cache: "no-store" });
  const blob = await response.blob();
  const file = new File([blob], "non_skin_photo.jpg", { type: "image/jpeg" });
  setPreviewSource(file, { type: "ood" });
  analyze(file);
}

/* ------------------------------------------------------------------ abcde */

const ABCDE_LETTERS = ["a", "b", "c", "d", "e"];
let abcdeActive = "a";

function renderAbcde(letter) {
  abcdeActive = letter;
  $$(".abcde-tab").forEach((button) => {
    const active = button.dataset.letter === letter;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-selected", String(active));
  });
  const term = $("#abcde-term");
  const body = $("#abcde-body");
  const prompt = $("#abcde-prompt");
  if (!term || !body || !prompt) return;
  $("#abcde-letter").textContent = letter.toUpperCase();
  term.textContent = t(`abcde.${letter}.title`);
  body.textContent = t(`abcde.${letter}.body`);
  prompt.textContent = t(`abcde.${letter}.check`);
}

function initAbcde() {
  $$(".abcde-tab").forEach((button) => {
    button.addEventListener("click", () => renderAbcde(button.dataset.letter));
    button.addEventListener("keydown", (event) => {
      if (!["ArrowRight", "ArrowLeft"].includes(event.key)) return;
      event.preventDefault();
      const index = ABCDE_LETTERS.indexOf(abcdeActive);
      const step = event.key === "ArrowRight" ? 1 : -1;
      const next = ABCDE_LETTERS[(index + step + ABCDE_LETTERS.length) % ABCDE_LETTERS.length];
      renderAbcde(next);
      $(`.abcde-tab[data-letter="${next}"]`).focus();
    });
  });
  renderAbcde("a");
}

/* ----------------------------------------------------------------- journal */

const JOURNAL_DB_NAME = "sentinel-derm-journal";
const JOURNAL_STORE = "entries";
const JOURNAL_MAX_SIDE = 1200;

function openJournalDb() {
  if (state.journalDb) return Promise.resolve(state.journalDb);
  return new Promise((resolve, reject) => {
    if (!("indexedDB" in window)) {
      reject(new Error(t("journal.unsupported")));
      return;
    }
    const request = indexedDB.open(JOURNAL_DB_NAME, 1);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(JOURNAL_STORE)) {
        const store = db.createObjectStore(JOURNAL_STORE, { keyPath: "id", autoIncrement: true });
        store.createIndex("date", "date");
      }
    };
    request.onsuccess = () => {
      state.journalDb = request.result;
      resolve(request.result);
    };
    request.onerror = () => reject(request.error || new Error("IndexedDB error"));
  });
}

async function journalAll() {
  const db = await openJournalDb();
  return new Promise((resolve, reject) => {
    const request = db.transaction(JOURNAL_STORE, "readonly").objectStore(JOURNAL_STORE).getAll();
    request.onsuccess = () => {
      const entries = request.result || [];
      entries.sort((left, right) => {
        const byDate = String(right.date).localeCompare(String(left.date));
        return byDate !== 0 ? byDate : Number(right.id) - Number(left.id);
      });
      resolve(entries);
    };
    request.onerror = () => reject(request.error);
  });
}

async function journalAdd(entry) {
  const db = await openJournalDb();
  return new Promise((resolve, reject) => {
    const request = db.transaction(JOURNAL_STORE, "readwrite").objectStore(JOURNAL_STORE).add(entry);
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function journalDelete(id) {
  const db = await openJournalDb();
  return new Promise((resolve, reject) => {
    const request = db.transaction(JOURNAL_STORE, "readwrite").objectStore(JOURNAL_STORE).delete(id);
    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
  });
}

function todayValue() {
  const now = new Date();
  const local = new Date(now.getTime() - now.getTimezoneOffset() * 60000);
  return local.toISOString().slice(0, 10);
}

function journalDateLabel(value) {
  return I18N.formatDate(new Date(`${value}T12:00:00`));
}

function imageFileToJournalDataUrl(file) {
  return new Promise((resolve, reject) => {
    if (!file || !file.type.startsWith("image/")) {
      reject(new Error(t("journal.imageError")));
      return;
    }
    const url = URL.createObjectURL(file);
    const image = new Image();
    image.onload = () => {
      try {
        const scale = Math.min(1, JOURNAL_MAX_SIDE / image.width, JOURNAL_MAX_SIDE / image.height);
        const canvas = document.createElement("canvas");
        canvas.width = Math.max(1, Math.round(image.width * scale));
        canvas.height = Math.max(1, Math.round(image.height * scale));
        const context = canvas.getContext("2d");
        context.fillStyle = "#ffffff";
        context.fillRect(0, 0, canvas.width, canvas.height);
        context.drawImage(image, 0, 0, canvas.width, canvas.height);
        resolve(canvas.toDataURL("image/jpeg", 0.82));
      } catch (error) {
        reject(error);
      } finally {
        URL.revokeObjectURL(url);
      }
    };
    image.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error(t("journal.imageError")));
    };
    image.src = url;
  });
}

function updateJournalPreviewText() {
  const text = $("#journal-photo-text");
  if (text) text.textContent = t(state.journalFile ? "journal.change" : "journal.choose");
}

function renderJournal() {
  const container = $("#journal-entries");
  if (!container) return;
  const count = $("#journal-count");
  if (count)
    count.textContent = state.journalEntries.length
      ? state.journalEntries.length === 1
        ? t("journal.countOne")
        : t("journal.count", { n: int(state.journalEntries.length) })
      : "";

  if (!state.journalEntries.length) {
    container.innerHTML = `<div class="journal-empty"><strong>${escapeHtml(t("journal.emptyTitle"))}</strong>${escapeHtml(
      t("journal.emptyBody")
    )}</div>`;
    return;
  }

  container.innerHTML = state.journalEntries
    .map((entry) => {
      const date = journalDateLabel(entry.date);
      const note = (entry.note || "").trim();
      return `<article class="journal-card" data-id="${Number(entry.id)}">
        <button class="journal-thumb" type="button" data-action="view" data-id="${Number(entry.id)}">
          <img src="${entry.dataUrl}" alt="${escapeHtml(t("journal.viewAlt", { date }))}" />
        </button>
        <div class="journal-card-body">
          <time datetime="${escapeHtml(entry.date)}">${escapeHtml(date)}</time>
          ${note ? `<p>${escapeHtml(note)}</p>` : ""}
          <button class="journal-delete" type="button" data-action="delete" data-id="${Number(entry.id)}">${escapeHtml(
            t("journal.delete")
          )}</button>
        </div>
      </article>`;
    })
    .join("");
}

function openJournalLightbox(entry) {
  const lightbox = $("#journal-lightbox");
  const image = $("#journal-lightbox-image");
  if (!lightbox || !image) return;
  const date = journalDateLabel(entry.date);
  const note = (entry.note || "").trim();
  const isDemo = Boolean(entry.demo);
  image.src = entry.dataUrl || entry.image;
  image.alt = t(isDemo ? "demo.journal.viewAlt" : "journal.viewAlt", { date });
  const prefix = isDemo ? `${t("demo.journal.lightbox")} · ` : "";
  $("#journal-lightbox-caption").textContent = prefix + (note ? `${date} · ${note}` : date);
  lightbox.hidden = false;
  document.body.style.overflow = "hidden";
}

function closeJournalLightbox() {
  const lightbox = $("#journal-lightbox");
  if (!lightbox) return;
  lightbox.hidden = true;
  $("#journal-lightbox-image").removeAttribute("src");
  $("#journal-lightbox-caption").textContent = "";
  document.body.style.overflow = "";
}

async function initJournal() {
  const dateInput = $("#journal-date");
  if (dateInput && !dateInput.value) dateInput.value = todayValue();
  updateJournalPreviewText();
  try {
    state.journalEntries = await journalAll();
    renderJournal();
  } catch (error) {
    const status = $("#journal-status");
    if (status) status.textContent = t("journal.unsupported");
    const save = $("#journal-save");
    if (save) save.disabled = true;
  }
}

$("#journal-file")?.addEventListener("change", (event) => {
  const file = event.target.files?.[0];
  const status = $("#journal-status");
  if (!file) return;
  if (!file.type.startsWith("image/")) {
    if (status) status.textContent = t("journal.imageError");
    return;
  }
  if (state.journalPreviewUrl) URL.revokeObjectURL(state.journalPreviewUrl);
  state.journalFile = file;
  state.journalPreviewUrl = URL.createObjectURL(file);
  const preview = $("#journal-preview");
  preview.src = state.journalPreviewUrl;
  preview.hidden = false;
  updateJournalPreviewText();
  $("#journal-save").disabled = false;
  if (status) status.textContent = "";
});

$("#journal-form")?.addEventListener("submit", async (event) => {
  event.preventDefault();
  const status = $("#journal-status");
  const save = $("#journal-save");
  if (!state.journalFile || !save) return;

  save.disabled = true;
  if (status) status.textContent = t("journal.saving");
  try {
    const dataUrl = await imageFileToJournalDataUrl(state.journalFile);
    await journalAdd({
      date: $("#journal-date").value || todayValue(),
      note: $("#journal-note").value.trim().slice(0, 280),
      dataUrl,
      createdAt: new Date().toISOString(),
    });
    state.journalEntries = await journalAll();
    state.journalFile = null;
    if (state.journalPreviewUrl) URL.revokeObjectURL(state.journalPreviewUrl);
    state.journalPreviewUrl = null;
    $("#journal-file").value = "";
    $("#journal-preview").hidden = true;
    $("#journal-preview").removeAttribute("src");
    $("#journal-note").value = "";
    $("#journal-date").value = todayValue();
    updateJournalPreviewText();
    renderJournal();
    if (status) status.textContent = t("journal.saved");
  } catch (error) {
    if (status) status.textContent = t("journal.error", { message: error.message || "IndexedDB" });
    save.disabled = !state.journalFile;
  }
});

$("#journal-entries")?.addEventListener("click", async (event) => {
  const action = event.target.closest("[data-action]");
  if (!action) return;
  const entry = state.journalEntries.find((item) => Number(item.id) === Number(action.dataset.id));
  if (!entry) return;
  if (action.dataset.action === "view") {
    openJournalLightbox(entry);
    return;
  }
  if (action.dataset.action === "delete") {
    if (!window.confirm(t("journal.deleteConfirm"))) return;
    try {
      await journalDelete(entry.id);
      state.journalEntries = await journalAll();
      renderJournal();
    } catch (error) {
      const status = $("#journal-status");
      if (status) status.textContent = t("journal.error", { message: error.message || "IndexedDB" });
    }
  }
});

$("#journal-lightbox-close")?.addEventListener("click", closeJournalLightbox);
$("#journal-lightbox")?.addEventListener("click", (event) => {
  if (event.target.id === "journal-lightbox") closeJournalLightbox();
});
window.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !$("#journal-lightbox")?.hidden) closeJournalLightbox();
});

/* -------------------------------------------------------------------- demo */

const DEMO_SCENARIOS = {
  good: {
    status: "guidance.statusReady",
    tone: "confident",
    hint: "demo.phone.hintGood",
    guidance: "demo.phone.guidanceGood",
    checks: [
      { code: "readability", state: "pass", detail: "ok", params: { width: 1080, height: 1440, minimum: 96 } },
      { code: "focus", state: "pass", detail: "ok" },
      { code: "lighting", state: "pass", detail: "ok" },
      { code: "skin", state: "pass", detail: "ok" },
    ],
  },
  dim: {
    status: "guidance.statusRetake",
    tone: "low_quality",
    hint: "demo.phone.hintDim",
    guidance: "demo.phone.guidanceDim",
    checks: [
      { code: "readability", state: "pass", detail: "ok", params: { width: 1080, height: 1440, minimum: 96 } },
      { code: "focus", state: "pass", detail: "ok" },
      { code: "lighting", state: "warn", detail: "dark" },
      { code: "skin", state: "pass", detail: "ok" },
    ],
  },
  blurry: {
    status: "guidance.statusRetake",
    tone: "low_quality",
    hint: "demo.phone.hintBlurry",
    guidance: "demo.phone.guidanceBlurry",
    checks: [
      { code: "readability", state: "pass", detail: "ok", params: { width: 1080, height: 1440, minimum: 96 } },
      { code: "focus", state: "warn", detail: "blurred" },
      { code: "lighting", state: "pass", detail: "ok" },
      { code: "skin", state: "pass", detail: "ok" },
    ],
  },
};

const DEMO_LETTERS = ["a", "b", "c", "d", "e"];
let demoAbcdeActive = "a";

const DEMO_JOURNAL_ENTRIES = [
  {
    id: "demo-1",
    date: "2026-09-12",
    noteKey: "demo.journal.note1",
    image: "/static/assets/demo-skin-arm.svg",
  },
  {
    id: "demo-2",
    date: "2026-10-24",
    noteKey: "demo.journal.note2",
    image: "/static/assets/demo-skin-back.svg",
  },
];

function demoEmptyHtml() {
  return `<div class="demo-result-empty">
    <p class="section-label">${escapeHtml(t("demo.phone.resultEmptyTitle"))}</p>
    <p class="body-copy">${escapeHtml(t("demo.phone.resultEmpty"))}</p>
  </div>`;
}

function renderDemoPhone() {
  const container = $("#demo-phone-result");
  if (!container) return;
  if (!state.demoResultScenario) {
    container.innerHTML = demoEmptyHtml();
    return;
  }

  const scenario = DEMO_SCENARIOS[state.demoResultScenario];
  const checks = scenario.checks
    .map(
      (check) => `<li class="check-item ${escapeHtml(check.state)}">
        <span class="check-mark" aria-hidden="true">${check.state === "pass" ? "&check;" : "!"}</span>
        <span>${escapeHtml(guidanceCheckLabel(check))}</span>
      </li>`
    )
    .join("");

  container.innerHTML = `
    <div class="demo-result-head">
      <span class="status-pill ${escapeHtml(scenario.tone)}"><span class="dot-sm"></span>${escapeHtml(
        t(scenario.status)
      )}</span>
      <span class="demo-sim-tag">${escapeHtml(t("demo.simulated"))}</span>
    </div>
    <p class="section-label">${escapeHtml(t("guidance.checksTitle"))}</p>
    <ul class="check-list">${checks}</ul>
    <div class="insight-box">
      <svg class="insight-icon" viewBox="0 0 20 20" aria-hidden="true"><circle cx="10" cy="10" r="7.5"/><path d="M10 6.2v4.6M10 13.6v.2"/></svg>
      <div class="insight-text">${escapeHtml(t(scenario.guidance))}</div>
    </div>
    <p class="demo-fineprint">${escapeHtml(t("demo.phone.resultNote"))}</p>`;
}

function updateDemoScenario() {
  const frame = $("#demo-photo-frame");
  if (frame) frame.dataset.scenario = state.demoScenario;
  $$(".demo-scenario").forEach((button) => {
    const active = button.dataset.demoScenario === state.demoScenario;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-pressed", String(active));
  });
  const hint = $("#demo-scenario-hint");
  if (hint) hint.textContent = t(DEMO_SCENARIOS[state.demoScenario].hint);
}

function selectDemoScenario(value) {
  if (state.demoCheckBusy || !DEMO_SCENARIOS[value]) return;
  state.demoScenario = value;
  state.demoResultScenario = null;
  updateDemoScenario();
  renderDemoPhone();
}

function runDemoPhoneCheck() {
  if (state.demoCheckBusy) return;
  const container = $("#demo-phone-result");
  if (!container) return;
  state.demoCheckBusy = true;
  state.demoResultScenario = null;
  container.innerHTML = `<div class="demo-checking"><span class="spinner" aria-hidden="true"></span><span>${escapeHtml(
    t("demo.phone.running")
  )}</span></div>`;
  const button = $("#demo-run");
  if (button) button.disabled = true;

  clearTimeout(state.demoTimer);
  state.demoTimer = setTimeout(() => {
    state.demoCheckBusy = false;
    state.demoResultScenario = state.demoScenario;
    if (button) button.disabled = false;
    renderDemoPhone();
  }, 700);
}

function renderDemoAbcde(letter) {
  demoAbcdeActive = letter;
  $$("[data-demo-letter]").forEach((button) => {
    const active = button.dataset.demoLetter === letter;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-selected", String(active));
  });
  const term = $("#demo-abcde-term");
  const body = $("#demo-abcde-body");
  const prompt = $("#demo-abcde-prompt");
  const observation = $("#demo-abcde-obs");
  if (!term || !body || !prompt || !observation) return;
  $("#demo-abcde-letter").textContent = letter.toUpperCase();
  term.textContent = t(`abcde.${letter}.title`);
  body.textContent = t(`abcde.${letter}.body`);
  prompt.textContent = t(`abcde.${letter}.check`);
  observation.textContent = t(`demo.abcde.obs.${letter}`);
}

function initDemoAbcde() {
  $$("[data-demo-letter]").forEach((button) => {
    button.addEventListener("click", () => renderDemoAbcde(button.dataset.demoLetter));
    button.addEventListener("keydown", (event) => {
      if (!["ArrowRight", "ArrowLeft"].includes(event.key)) return;
      event.preventDefault();
      const index = DEMO_LETTERS.indexOf(demoAbcdeActive);
      const step = event.key === "ArrowRight" ? 1 : -1;
      const next = DEMO_LETTERS[(index + step + DEMO_LETTERS.length) % DEMO_LETTERS.length];
      renderDemoAbcde(next);
      $(`[data-demo-letter="${next}"]`).focus();
    });
  });
  renderDemoAbcde("a");
}

function renderDemoJournal() {
  const container = $("#demo-journal-entries");
  if (!container) return;
  container.innerHTML = DEMO_JOURNAL_ENTRIES.map((entry) => {
    const date = journalDateLabel(entry.date);
    return `<article class="journal-card demo-journal-card">
      <span class="demo-card-tag">${escapeHtml(t("demo.journal.tag"))}</span>
      <button class="journal-thumb" type="button" data-demo-journal="${entry.id}">
        <img src="${entry.image}" alt="${escapeHtml(t("demo.journal.viewAlt", { date }))}" />
      </button>
      <div class="journal-card-body">
        <time datetime="${entry.date}">${escapeHtml(date)}</time>
        <p>${escapeHtml(t(entry.noteKey))}</p>
      </div>
    </article>`;
  }).join("");
}

function resetDemoTour() {
  clearTimeout(state.demoTimer);
  state.demoCheckBusy = false;
  state.demoScenario = "good";
  state.demoResultScenario = null;
  const run = $("#demo-run");
  if (run) run.disabled = false;
  updateDemoScenario();
  renderDemoPhone();
  renderDemoAbcde("a");
  renderDemoJournal();
}

function startDemo() {
  resetDemoTour();
  state.demo = true;
  document.body.classList.add("demo-mode");
  const bar = $("#demo-bar");
  if (bar) bar.hidden = false;
  showView("demo");
}

function stopDemo() {
  if (!state.demo) return;
  state.demo = false;
  document.body.classList.remove("demo-mode");
  const bar = $("#demo-bar");
  if (bar) bar.hidden = true;
  const lightbox = $("#journal-lightbox");
  if (lightbox && !lightbox.hidden) closeJournalLightbox();
}

function exitDemo() {
  stopDemo();
  showView("analyze");
}

function scrollToDemoSection(id) {
  document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
}

function updateDemoCopy() {
  updateDemoScenario();
  if (!state.demoCheckBusy) renderDemoPhone();
  renderDemoAbcde(demoAbcdeActive);
  renderDemoJournal();
}

$("#try-demo")?.addEventListener("click", startDemo);
$("#demo-exit")?.addEventListener("click", exitDemo);
$("#demo-exit-inline")?.addEventListener("click", exitDemo);
$("#demo-restart")?.addEventListener("click", () => {
  resetDemoTour();
  window.scrollTo({ top: 0, behavior: "smooth" });
});
$("#demo-start")?.addEventListener("click", () => {
  scrollToDemoSection("demo-phone");
  runDemoPhoneCheck();
});
$("#demo-run")?.addEventListener("click", runDemoPhoneCheck);
$$("[data-demo-scenario]").forEach((button) =>
  button.addEventListener("click", () => selectDemoScenario(button.dataset.demoScenario))
);
$$("[data-demo-jump]").forEach((button) =>
  button.addEventListener("click", () => scrollToDemoSection(button.dataset.demoJump))
);
$("#demo-journal-entries")?.addEventListener("click", (event) => {
  const button = event.target.closest("[data-demo-journal]");
  if (!button) return;
  const entry = DEMO_JOURNAL_ENTRIES.find((item) => item.id === button.dataset.demoJournal);
  if (!entry) return;
  openJournalLightbox({ ...entry, demo: true, note: t(entry.noteKey) });
});

/* ------------------------------------------------------------------ metrics */

function metricCard(value, label, note = "") {
  return `<div class="metric-card">
    <div class="metric-value">${value}</div>
    <div class="metric-label">${escapeHtml(label)}</div>
    ${note ? `<div class="metric-note">${escapeHtml(note)}</div>` : ""}
  </div>`;
}

function renderMetrics(metrics) {
  const classifier = metrics.classifier;
  const conformal = metrics.conformal;
  const ood = metrics.ood || {};

  $("#metric-grid").innerHTML = [
    metricCard(
      pct(conformal.coverage, 1),
      t("metric.coverage"),
      t("metric.coverageNote", { target: pct(conformal.target_coverage, 0) })
    ),
    metricCard(
      num(conformal.avg_set_size, 2),
      t("metric.avgSet"),
      t("metric.avgSetNote", { pct: pct(conformal.singleton_rate, 0) })
    ),
    metricCard(pct(conformal.selective_accuracy, 1), t("metric.selective"), t("metric.selectiveNote")),
    metricCard(pct(classifier.macro_f1, 1), t("metric.macroF1"), t("metric.macroF1Note")),
    metricCard(
      pct(classifier.per_class_recall.mel, 1),
      t("metric.melRecall"),
      t("metric.melRecallNote", { n: int(classifier.support.mel) })
    ),
    metricCard(
      ood.auroc !== null && ood.auroc !== undefined ? num(ood.auroc, 5) : "--",
      t("metric.oodAuroc"),
      t("metric.oodNote")
    ),
  ].join("");

  const tbody = $("#class-table tbody");
  tbody.innerHTML = classifier.support
    ? CLASS_CODES.map((code) => {
        const support = classifier.support[code] ?? 0;
        const recall = classifier.per_class_recall[code];
        const precision = classifier.per_class_precision[code];
        const covered = conformal.coverage_by_class?.[code];
        return `<tr>
          <td>${escapeHtml(classText(code, "short"))}</td>
          <td class="num">${int(support)}</td>
          <td class="num">${pct(recall, 1)}</td>
          <td class="num">${pct(precision, 1)}</td>
          <td><span class="pill-mini">${pct(covered, 1)}</span></td>
        </tr>`;
      }).join("")
    : "";

  const melRecall = classifier.per_class_recall.mel;
  const melSupport = classifier.support.mel ?? 0;
  const warning = $("#evidence-warning");
  if (warning) {
    if (typeof melRecall === "number" && melRecall < 0.6) {
      const missed = Math.round((1 - melRecall) * melSupport);
      warning.hidden = false;
      warning.innerHTML = `<strong>${escapeHtml(t("evidence.warningTitle"))}.</strong> ${escapeHtml(
        t("evidence.warningMel", {
          recall: pct(melRecall, 1),
          missed: int(missed),
          support: int(melSupport),
        })
      )}`;
    } else {
      warning.hidden = true;
      warning.innerHTML = "";
    }
  }

  const dataset = metrics.dataset || {};
  const splits = dataset.splits || {};
  $("#dataset-copy").textContent = t("dataset.copy", {
    total: int(dataset.total_images),
    train: int(splits.train),
    cal: int(splits.cal),
    test: int(splits.test),
  });
}

function renderShell(metrics) {
  const conformal = metrics.conformal;
  const backbone = BACKBONES[metrics.backbone] || { label: metrics.backbone, dimension: "\u2014" };

  $("#hero-stats").innerHTML = [
    `<div class="stat"><span class="stat-value">${pct(conformal.target_coverage, 0)}</span><span class="stat-label">${escapeHtml(
      t("stat.coverageTarget")
    )}</span></div>`,
    `<div class="stat"><span class="stat-value">${pct(conformal.coverage, 1)}</span><span class="stat-label">${escapeHtml(
      t("stat.achieved")
    )}</span></div>`,
    `<div class="stat"><span class="stat-value">${num(conformal.avg_set_size, 2)}</span><span class="stat-label">${escapeHtml(
      t("stat.avgSetSize")
    )}</span></div>`,
    `<div class="stat"><span class="stat-value">${int(metrics.dataset.total_images)}</span><span class="stat-label">${escapeHtml(
      t("stat.images")
    )}</span></div>`,
  ].join("");

  $("#topbar-meta").textContent = t("topbar.meta", { n: int(metrics.dataset.total_images) });
  $("#footer-meta").textContent = t("footer.meta", {
    backbone: backbone.label,
    date: I18N.formatDate(metrics.generated_at),
    alpha: metrics.alpha,
  });
  $("#step1-body").textContent = t("step1.body", {
    backbone: backbone.label,
    dimension: backbone.dimension,
  });
}

/* --------------------------------------------------------------------- init */

function buildLanguageSelect() {
  const select = $("#language-select");
  if (!select) return;
  select.innerHTML = I18N.LANGUAGES.map(
    (language) => `<option value="${language.code}">${escapeHtml(language.label)}</option>`
  ).join("");
  select.value = I18N.getLanguage();
  select.addEventListener("change", () => I18N.setLanguage(select.value));
}

I18N.onChange(() => {
  if (state.metrics) {
    renderMetrics(state.metrics);
    renderShell(state.metrics);
  } else {
    $("#topbar-meta").textContent = t("topbar.noModel");
  }
  renderSamples();
  updatePreviewText();
  updateModeCopy();
  updateJournalPreviewText();
  renderAbcde(abcdeActive);
  renderJournal();
  updateDemoCopy();
  if (state.lastResult) renderActiveResult(state.lastResult, { scroll: false });
});

async function init() {
  I18N.setLanguage(I18N.detectLanguage(), { silent: true });
  I18N.applyTranslations(document);
  buildLanguageSelect();
  modeOptions.forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.mode === state.mode)));
  updateModeCopy();
  initAbcde();
  initDemoAbcde();
  initJournal();
  updateDemoCopy();

  const hash = (location.hash || "#analyze").replace("#", "");
  if (hash === "demo") startDemo();
  else if (VIEW_NAMES.includes(hash)) showView(hash);

  try {
    const response = await fetch("/api/metrics");
    if (!response.ok) throw new Error((await response.json()).detail || t("topbar.noModel"));
    state.metrics = await response.json();
    renderMetrics(state.metrics);
    renderShell(state.metrics);
  } catch (error) {
    $("#topbar-meta").textContent = t("topbar.noModel");
    $("#hero-stats").innerHTML = `<div class="stat"><span class="stat-value">&mdash;</span><span class="stat-label">${escapeHtml(
      t("stat.images")
    )}</span></div>`;
    $("#metric-grid").innerHTML = `<div class="error-box">${escapeHtml(error.message)}<br />${t(
      "error.runTraining"
    )}</div>`;
  }

  loadSamples();
}

window.addEventListener("hashchange", () => {
  const hash = (location.hash || "#analyze").replace("#", "");
  if (hash === "demo") {
    startDemo();
    return;
  }
  if (VIEW_NAMES.includes(hash)) {
    if (state.demo) stopDemo();
    showView(hash);
  }
});

init();

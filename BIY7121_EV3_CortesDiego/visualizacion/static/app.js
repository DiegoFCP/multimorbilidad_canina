const API_URL = window.PREDICT_API_URL || "http://127.0.0.1:5050/predict";
const RETRAIN_URL = window.RETRAIN_API_URL || "http://127.0.0.1:5001/retrain";

const PRESETS = {
  bajo: {
    pa_activity_level: 4,
    pa_avg_activity_intensity: 3.0,
    total_daily_active_minutes: 180,
    dental_brushing_freq: 5,
    daily_supplements: 2,
    diet_primary: 1,
    weight_kg: 12,
    age_derived: 3,
  },
  medio: {
    pa_activity_level: 2,
    pa_avg_activity_intensity: 1.5,
    total_daily_active_minutes: 75,
    dental_brushing_freq: 2,
    daily_supplements: 1,
    diet_primary: 0,
    weight_kg: 18,
    age_derived: 7,
  },
  alto: {
    pa_activity_level: 1,
    pa_avg_activity_intensity: 0.5,
    total_daily_active_minutes: 25,
    dental_brushing_freq: 0,
    daily_supplements: 0,
    diet_primary: 0,
    weight_kg: 30,
    age_derived: 14,
  },
};

const form = document.getElementById("predict-form");
const resultSection = document.getElementById("result-section");
const errorSection = document.getElementById("error-section");
const resultBadge = document.getElementById("result-badge");
const resultProbability = document.getElementById("result-probability");
const resultJson = document.getElementById("result-json");
const riskBar = document.getElementById("risk-bar");
const errorMessage = document.getElementById("error-message");
const loadingOverlay = document.getElementById("loading-overlay");
const loadingMessage = document.getElementById("loading-message");
const submitBtn = document.getElementById("submit-btn");
const csvBtn = document.getElementById("csv-predict-btn");
const retrainBtn = document.getElementById("retrain-btn");
const retrainResultSection = document.getElementById("retrain-result-section");
const retrainSummary = document.getElementById("retrain-summary");
const retrainJson = document.getElementById("retrain-json");

let activeButton = null;

function setLoading(isLoading, message = "Consultando el modelo…") {
  loadingOverlay.hidden = !isLoading;
  loadingMessage.textContent = message;

  if (activeButton) {
    activeButton.disabled = isLoading;
    activeButton.classList.toggle("btn-loading", isLoading);
  }
  submitBtn.disabled = isLoading;
  csvBtn.disabled = isLoading;
  retrainBtn.disabled = isLoading;
}

function hideRetrainResult() {
  retrainResultSection.hidden = true;
}

function showError(msg) {
  errorSection.hidden = false;
  resultSection.hidden = true;
  hideRetrainResult();
  errorMessage.textContent = msg;
  errorSection.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function showResult(data) {
  errorSection.hidden = true;
  hideRetrainResult();
  resultSection.hidden = false;
  resultSection.classList.add("visible");

  const high = data.multimorbidity_flag === 1;
  resultBadge.textContent = data.label;
  resultBadge.className = "badge " + (high ? "high" : "low");

  const pct = Math.round((data.probability || 0) * 100);
  resultProbability.textContent = `Probabilidad de multimorbilidad: ${pct}%`;

  riskBar.style.width = "0%";
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      riskBar.style.width = pct + "%";
    });
  });

  resultJson.textContent = JSON.stringify(data, null, 2);
  resultSection.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function showRetrainResult(data) {
  errorSection.hidden = true;
  resultSection.hidden = true;
  retrainResultSection.hidden = false;
  retrainResultSection.classList.add("visible");

  const f1cv = typeof data.f1_cv === "number" ? (data.f1_cv * 100).toFixed(1) : "—";
  const f1test = typeof data.f1_test === "number" ? (data.f1_test * 100).toFixed(1) : "—";
  const rate = typeof data.multimorbidity_rate === "number"
    ? (data.multimorbidity_rate * 100).toFixed(1)
    : "—";

  retrainSummary.textContent =
    `Filas usadas: ${data.rows_used ?? "—"} · F1 CV: ${f1cv}% · F1 test: ${f1test}% · ` +
    `Tasa multimorbilidad: ${rate}% · ${data.metric_reference || ""}`;

  retrainJson.textContent = JSON.stringify(data, null, 2);
  retrainResultSection.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

async function callPredict(payload) {
  const response = await fetch(API_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || `HTTP ${response.status}`);
  }
  return data;
}

async function callRetrain(file) {
  const formData = new FormData();
  formData.append("file", file);

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 600000);

  try {
    const response = await fetch(RETRAIN_URL, {
      method: "POST",
      body: formData,
      signal: controller.signal,
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || `HTTP ${response.status}`);
    }
    return data;
  } finally {
    clearTimeout(timeoutId);
  }
}

function formToPayload(formEl) {
  const fd = new FormData(formEl);
  const payload = {};
  for (const [key, val] of fd.entries()) {
    payload[key] = parseFloat(val);
  }
  return payload;
}

function applyPreset(name) {
  const preset = PRESETS[name];
  if (!preset) return;
  for (const [field, value] of Object.entries(preset)) {
    const input = form.elements.namedItem(field);
    if (input) input.value = value;
  }
}

async function runPredict(payload, button, loadingText) {
  activeButton = button;
  setLoading(true, loadingText);
  try {
    const data = await callPredict(payload);
    showResult(data);
  } catch (err) {
    showError("No se pudo obtener predicción. ¿Está activa la API? " + err.message);
  } finally {
    setLoading(false);
    activeButton = null;
  }
}

document.querySelectorAll(".btn-preset").forEach((btn) => {
  btn.addEventListener("click", () => {
    applyPreset(btn.dataset.preset);
  });
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  await runPredict(formToPayload(form), submitBtn, "Analizando perfil del perro…");
});

function csvRowToPayload(row) {
  const map = {
    pa_activity_level: ["pa_activity_level"],
    pa_avg_activity_intensity: ["pa_avg_activity_intensity"],
    total_daily_active_minutes: ["pa_avg_daily_active_minutes", "total_daily_active_minutes"],
    dental_brushing_freq: ["mp_dental_brushing_frequency", "dental_brushing_freq"],
    daily_supplements: ["mp_dental_examination_frequency", "daily_supplements"],
    diet_primary: ["df_primary_diet_component_organic", "diet_primary"],
    weight_kg: ["weight_kg"],
    age_derived: ["dd_edad_anios", "age_derived"],
  };

  const payload = {};
  for (const [apiKey, candidates] of Object.entries(map)) {
    for (const col of candidates) {
      if (row[col] !== undefined && row[col] !== "") {
        let v = parseFloat(row[col]);
        if (apiKey === "weight_kg" && col === "dd_weight_lbs") {
          v = v * 0.453592;
        }
        if (!Number.isNaN(v)) {
          payload[apiKey] = v;
          break;
        }
      }
    }
  }

  const required = Object.keys(map);
  const missing = required.filter((k) => payload[k] === undefined);
  if (missing.length) {
    throw new Error("Faltan columnas en el CSV: " + missing.join(", "));
  }
  return payload;
}

function parseCsvLine(line) {
  return line.split(",").map((s) => s.trim().replace(/^"|"$/g, ""));
}

csvBtn.addEventListener("click", async () => {
  const fileInput = document.getElementById("csv-file");
  const file = fileInput.files[0];
  if (!file) {
    showError("Seleccione un archivo CSV.");
    return;
  }
  activeButton = csvBtn;
  setLoading(true, "Leyendo CSV y consultando API…");
  try {
    const text = await file.text();
    const lines = text.trim().split(/\r?\n/).filter(Boolean);
    if (lines.length < 2) throw new Error("El CSV debe tener encabezado y al menos una fila.");
    const headers = parseCsvLine(lines[0]);
    const values = parseCsvLine(lines[1]);
    const row = {};
    headers.forEach((h, i) => { row[h] = values[i]; });
    const payload = csvRowToPayload(row);
    loadingMessage.textContent = "Consultando modelo con datos del CSV…";
    const data = await callPredict(payload);
    showResult(data);
  } catch (err) {
    showError(err.message);
  } finally {
    setLoading(false);
    activeButton = null;
  }
});

retrainBtn.addEventListener("click", async () => {
  const fileInput = document.getElementById("retrain-csv-file");
  const file = fileInput.files[0];
  if (!file) {
    showError("Seleccione un CSV de entrenamiento.");
    return;
  }

  const confirmed = window.confirm(
    "¿Reentrenar el modelo con este archivo? Se actualizará el modelo en el servidor y puede tardar varios minutos."
  );
  if (!confirmed) return;

  activeButton = retrainBtn;
  setLoading(true, "Reentrenando modelo (GridSearchCV)… esto puede tardar varios minutos.");
  try {
    const data = await callRetrain(file);
    showRetrainResult(data);
  } catch (err) {
    const msg = err.name === "AbortError"
      ? "El reentrenamiento superó el tiempo máximo de espera (10 min)."
      : err.message;
    showError("No se pudo reentrenar. ¿Está activa la API de reentrenamiento? " + msg);
  } finally {
    setLoading(false);
    activeButton = null;
  }
});

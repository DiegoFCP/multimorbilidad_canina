const API_URL = window.PREDICT_API_URL || "http://127.0.0.1:5050/predict";

const form = document.getElementById("predict-form");
const resultSection = document.getElementById("result-section");
const errorSection = document.getElementById("error-section");
const resultBadge = document.getElementById("result-badge");
const resultProbability = document.getElementById("result-probability");
const resultJson = document.getElementById("result-json");
const riskBar = document.getElementById("risk-bar");
const errorMessage = document.getElementById("error-message");

function showError(msg) {
  errorSection.hidden = false;
  resultSection.hidden = true;
  errorMessage.textContent = msg;
}

function showResult(data) {
  errorSection.hidden = true;
  resultSection.hidden = false;
  const high = data.multimorbidity_flag === 1;
  resultBadge.textContent = data.label;
  resultBadge.className = "badge " + (high ? "high" : "low");
  const pct = Math.round((data.probability || 0) * 100);
  resultProbability.textContent = `Probabilidad de multimorbilidad: ${pct}%`;
  riskBar.style.width = pct + "%";
  resultJson.textContent = JSON.stringify(data, null, 2);
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

function formToPayload(formEl) {
  const fd = new FormData(formEl);
  const payload = {};
  for (const [key, val] of fd.entries()) {
    payload[key] = parseFloat(val);
  }
  return payload;
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  try {
    const data = await callPredict(formToPayload(form));
    showResult(data);
  } catch (err) {
    showError("No se pudo obtener predicción. ¿Está activa la API? " + err.message);
  }
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
        payload[apiKey] = v;
        break;
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

document.getElementById("csv-predict-btn").addEventListener("click", async () => {
  const fileInput = document.getElementById("csv-file");
  const file = fileInput.files[0];
  if (!file) {
    showError("Seleccione un archivo CSV.");
    return;
  }
  try {
    const text = await file.text();
    const lines = text.trim().split(/\r?\n/).filter(Boolean);
    if (lines.length < 2) throw new Error("El CSV debe tener encabezado y al menos una fila.");
    const headers = parseCsvLine(lines[0]);
    const values = parseCsvLine(lines[1]);
    const row = {};
    headers.forEach((h, i) => { row[h] = values[i]; });
    const payload = csvRowToPayload(row);
    const data = await callPredict(payload);
    showResult(data);
  } catch (err) {
    showError(err.message);
  }
});

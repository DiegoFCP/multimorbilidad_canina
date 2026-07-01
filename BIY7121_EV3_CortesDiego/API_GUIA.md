# Guía para consumir la API — Multimorbilidad canina (EV3 BIY7121)

Documentación para integrar la API de predicción (y opcionalmente reentrenamiento) en otra aplicación web, script o herramienta externa.

**Proyecto:** Dog Aging Project — insight de cuidado preventivo y actividad física del tutor vs. carga de enfermedades del perro.  
**Modelo:** `RandomForestClassifier` (scikit-learn), entrenado con `maestro_dap.csv`.

---

## URLs base

| Entorno | API predicción | API reentrenamiento |
|---------|----------------|---------------------|
| **Producción (Render)** | `https://dap-predict-api.onrender.com` | `https://dap-retrain-api.onrender.com` |
| **Local (Windows)** | `http://127.0.0.1:5050` | `http://127.0.0.1:5001` |

**Frontend en producción:** https://multimorbilidad-canina.vercel.app

> En el plan gratuito de Render, el servicio puede tardar ~30–60 s en responder tras un período de inactividad (*cold start*). Antes de una demo, llame a `GET /health`.

---

## CORS

La API de predicción envía estas cabeceras en todas las respuestas:

```
Access-Control-Allow-Origin: *
Access-Control-Allow-Headers: Content-Type
Access-Control-Allow-Methods: GET, POST, OPTIONS
```

Puede consumirla desde el navegador con `fetch()` aunque la web esté en otro dominio (Vercel, Netlify, etc.).

---

## API de predicción

### `GET /health`

Comprueba que el servicio esté activo.

**Ejemplo:**

```http
GET https://dap-predict-api.onrender.com/health
```

**Respuesta 200:**

```json
{
  "status": "ok",
  "service": "api_prediccion"
}
```

---

### `POST /predict`

Recibe los hábitos del tutor y datos básicos del perro, aplica el mismo preprocesamiento del notebook (`preprocess.py` + `KNNImputer`) y devuelve la predicción de **multimorbilidad** (`multimorbidity_flag`: perro con ≥2 condiciones de salud).

#### Cabeceras

| Cabecera | Valor |
|----------|-------|
| `Content-Type` | `application/json` |

#### Cuerpo (JSON)

Todos los campos son **obligatorios** y deben ser **numéricos**.

| Campo | Tipo | Descripción | Rango orientativo |
|-------|------|-------------|-------------------|
| `pa_activity_level` | number | Nivel de actividad física del tutor | 1 (bajo) – 4 (alto) |
| `pa_avg_activity_intensity` | number | Intensidad promedio de la actividad | 0 – 4 |
| `total_daily_active_minutes` | number | Minutos activos diarios del perro | 0 – 300+ |
| `dental_brushing_freq` | number | Frecuencia de cepillado dental | 0 – 7 |
| `daily_supplements` | number | Proxy de suplementos / examen dental | 0 – 3 |
| `diet_primary` | number | Dieta orgánica o premium | `0` = no, `1` = sí |
| `weight_kg` | number | Peso del perro en kilogramos | ej. 5 – 40 |
| `age_derived` | number | Edad del perro en años | ej. 1 – 20 |

**Nota:** No envíe `preventive_care_score`. La API lo calcula internamente a partir de los campos anteriores.

#### Ejemplo de petición

```http
POST https://dap-predict-api.onrender.com/predict
Content-Type: application/json

{
  "pa_activity_level": 3,
  "pa_avg_activity_intensity": 2,
  "total_daily_active_minutes": 90,
  "dental_brushing_freq": 4,
  "daily_supplements": 1,
  "diet_primary": 1,
  "weight_kg": 12.5,
  "age_derived": 5
}
```

#### Respuesta 200 (éxito)

```json
{
  "multimorbidity_flag": 0,
  "label": "Sin multimorbilidad significativa",
  "probability": 0.2139,
  "model": "RandomForestClassifier",
  "metric_reference": "F1_cv=0.5596"
}
```

| Campo | Tipo | Significado |
|-------|------|-------------|
| `multimorbidity_flag` | int | `0` = sin multimorbilidad significativa, `1` = multimorbilidad probable |
| `label` | string | Etiqueta legible del resultado |
| `probability` | float | Probabilidad estimada de la clase positiva (0–1) |
| `model` | string | Algoritmo utilizado |
| `metric_reference` | string | Referencia de la métrica principal (F1 en validación cruzada) |

#### Errores

| Código | Causa típica | Ejemplo de cuerpo |
|--------|--------------|-------------------|
| `400` | JSON inválido, `Content-Type` incorrecto o campos faltantes/no numéricos | `{"error": "Campo requerido faltante: weight_kg"}` |
| `500` | Modelo no encontrado o error interno | `{"error": "No existe .../model.joblib. Ejecute train_and_export.py"}` |

---

## API de reentrenamiento (opcional)

### `GET /health`

```http
GET https://dap-retrain-api.onrender.com/health
```

**Respuesta 200:**

```json
{
  "status": "ok",
  "service": "api_reentrenamiento"
}
```

---

### `POST /retrain`

Recibe un CSV con datos DAP (`maestro_dap.csv` o `dap_consolidado.csv`), reentrena el `RandomForestClassifier` con el mismo `param_grid` del notebook y **sobrescribe** `model.joblib`, `imputer.joblib` y los metadatos JSON en el servidor.

#### Opción A — Subir archivo (recomendada en producción)

| Cabecera | Valor |
|----------|-------|
| `Content-Type` | `multipart/form-data` |

| Campo formulario | Tipo | Descripción |
|------------------|------|-------------|
| `file` | archivo | CSV de entrenamiento |

#### Opción B — Ruta local (solo servidor local)

```json
{
  "csv_path": "data/maestro_dap.csv"
}
```

`Content-Type: application/json`. La ruta debe existir **en el disco del servidor** donde corre la API.

#### Respuesta 200 (éxito)

```json
{
  "status": "ok",
  "rows_used": 1234,
  "best_params": {
    "max_depth": 14,
    "min_samples_leaf": 20,
    "n_estimators": 150
  },
  "f1_cv": 0.5596,
  "f1_test": 0.5942,
  "roc_auc_test": 0.8217,
  "metric_reference": "F1_cv=0.5596",
  "multimorbidity_rate": 0.42
}
```

---

## Ejemplos de integración

### JavaScript (navegador)

```javascript
const PREDICT_URL = "https://dap-predict-api.onrender.com/predict";

async function predecirMultimorbilidad(datos) {
  const response = await fetch(PREDICT_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(datos),
  });

  const result = await response.json();

  if (!response.ok) {
    throw new Error(result.error || `HTTP ${response.status}`);
  }

  return result;
}

// Uso
predecirMultimorbilidad({
  pa_activity_level: 2,
  pa_avg_activity_intensity: 1.5,
  total_daily_active_minutes: 45,
  dental_brushing_freq: 1,
  daily_supplements: 0,
  diet_primary: 0,
  weight_kg: 18,
  age_derived: 8,
}).then((data) => {
  console.log(data.label, `${Math.round(data.probability * 100)}%`);
});
```

Para configurar la URL sin hardcodearla (como en la visualización del proyecto):

```html
<script>
  window.PREDICT_API_URL = "https://dap-predict-api.onrender.com/predict";
</script>
<script src="app.js"></script>
```

---

### Python (`requests`)

```python
import requests

URL = "https://dap-predict-api.onrender.com/predict"

payload = {
    "pa_activity_level": 3,
    "pa_avg_activity_intensity": 2,
    "total_daily_active_minutes": 90,
    "dental_brushing_freq": 4,
    "daily_supplements": 1,
    "diet_primary": 1,
    "weight_kg": 12.5,
    "age_derived": 5,
}

response = requests.post(URL, json=payload, timeout=120)
response.raise_for_status()
print(response.json())
```

---

### PowerShell

**Health check:**

```powershell
Invoke-RestMethod -Uri "https://dap-predict-api.onrender.com/health"
```

**Predicción:**

```powershell
$body = @{
  pa_activity_level = 2
  pa_avg_activity_intensity = 1.5
  total_daily_active_minutes = 45
  dental_brushing_freq = 1
  daily_supplements = 0
  diet_primary = 0
  weight_kg = 18
  age_derived = 8
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri "https://dap-predict-api.onrender.com/predict" `
  -Method POST `
  -Body $body `
  -ContentType "application/json"
```

**Reentrenamiento con CSV:**

```powershell
curl.exe -X POST `
  -F "file=@maestro_dap.csv" `
  https://dap-retrain-api.onrender.com/retrain
```

---

### cURL

```bash
curl -X POST "https://dap-predict-api.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "pa_activity_level": 1,
    "pa_avg_activity_intensity": 0.5,
    "total_daily_active_minutes": 25,
    "dental_brushing_freq": 0,
    "daily_supplements": 0,
    "diet_primary": 0,
    "weight_kg": 30,
    "age_derived": 14
  }'
```

---

## Casos de prueba sugeridos

| Perfil | `pa_activity_level` | `pa_avg_activity_intensity` | `total_daily_active_minutes` | `dental_brushing_freq` | `daily_supplements` | `diet_primary` | `weight_kg` | `age_derived` |
|--------|--------------------|-----------------------------|------------------------------|------------------------|---------------------|----------------|-------------|---------------|
| Bajo riesgo (joven) | 4 | 3.0 | 180 | 5 | 2 | 1 | 12 | 3 |
| Riesgo medio | 2 | 1.5 | 75 | 2 | 1 | 0 | 18 | 7 |
| Alto riesgo (senior) | 1 | 0.5 | 25 | 0 | 0 | 0 | 30 | 14 |

También puede usar el archivo `casos_prueba.csv` en la raíz del repositorio (columnas sin el campo `caso`).

---

## Flujo interno (referencia)

```
JSON de entrada
    → validate_input()          # campos obligatorios y tipos
    → preventive_care_score     # calculado en servidor
    → KNNImputer (imputer.joblib)
    → RandomForest (model.joblib)
    → JSON de salida
```

El preprocesamiento está centralizado en `modelo_y_preprocesamiento/preprocess.py` y debe coincidir con el notebook de entrenamiento.

---

## Levantar la API en local

```powershell
cd BIY7121_EV3_CortesDiego
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Terminal 1 — predicción (puerto 5050)
python api_prediccion\app.py

# Terminal 2 — reentrenamiento (puerto 5001)
python api_reentrenamiento\app.py
```

Sustituya las URLs de producción por `http://127.0.0.1:5050/predict` y `http://127.0.0.1:5001/retrain`.

---

## Buenas prácticas al integrar

1. **Siempre** envíe los 8 campos numéricos; la API no acepta campos parciales.
2. Use `GET /health` antes de la primera predicción en demos con Render free tier.
3. Configure un **timeout** generoso (60–120 s) en clientes HTTP por el cold start.
4. Muestre `probability` como porcentaje y `label` al usuario final; use `multimorbidity_flag` para lógica binaria.
5. No exponga la API de reentrenamiento en frontends públicos sin autenticación (puede sobrescribir el modelo en producción).

---

## Referencias en el repositorio

| Archivo | Contenido |
|---------|-----------|
| `api_prediccion/app.py` | Implementación del endpoint `/predict` |
| `api_reentrenamiento/app.py` | Implementación del endpoint `/retrain` |
| `modelo_y_preprocesamiento/preprocess.py` | Validación, features e imputación |
| `visualizacion/static/app.js` | Cliente de referencia con `fetch()` |
| `README.md` | Instrucciones generales del proyecto |

---

*BIY7121 — Evaluación 3 — CortesDiego*

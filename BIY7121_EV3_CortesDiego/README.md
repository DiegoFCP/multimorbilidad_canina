# BIY7121 — Evaluación 3 — CortesDiego

## Insight seleccionado

**Proyecto grupal:** Dog Aging Project (DAP) 2024.

> El nivel de cuidado preventivo y la actividad física que provee el tutor se asocian con la carga de enfermedades del perro. Los perfiles de menor actividad y menor cuidado preventivo concentran mayor proporción de salud/enfermedad alta (evidencia en clusters y cruces de la sección 8 del notebook grupal).

**Modelo desplegado:** predice `multimorbidity_flag` (perro con 2 o más condiciones de salud registradas).

## Modelo utilizado

- **Algoritmo:** `RandomForestClassifier` (scikit-learn)
- **Hiperparámetros:** seleccionados con `GridSearchCV` (cv=5, scoring=F1)
- **Métrica principal:** **F1-score** en validación cruzada 5-fold

## Variables usadas

| Campo API | Columna DAP | Descripción |
|-----------|-------------|-------------|
| `preventive_care_score` | derivada | Promedio z-score de cuidado preventivo |
| `pa_activity_level` | `pa_activity_level` | Nivel de actividad del tutor (1-4) |
| `pa_avg_activity_intensity` | `pa_avg_activity_intensity` | Intensidad de actividad |
| `total_daily_active_minutes` | `pa_avg_daily_active_minutes` | Minutos activos diarios |
| `dental_brushing_freq` | `mp_dental_brushing_frequency` | Frecuencia cepillado dental |
| `daily_supplements` | `mp_dental_examination_frequency` | Proxy de suplementos/examen |
| `diet_primary` | `df_primary_diet_component_organic` | Dieta orgánica (0/1) |
| `weight_kg` | `dd_weight_lbs × 0.453592` | Peso del perro en kg |
| `age_derived` | `dd_edad_anios` | Edad del perro en años |

**Target:** `multimorbidity_flag` = 1 si `salud_num_condiciones >= 2`, else 0.

## Instrucciones de ejecución (Windows PowerShell)

```powershell
cd BIY7121_EV3_CortesDiego
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Coloque `dap_consolidado.csv` en `data\` (opcional; si no existe, el entrenamiento usa datos sintéticos).

**Entrenar / exportar modelo:**

```powershell
python modelo_y_preprocesamiento\train_and_export.py
```

**Terminal 1 — API de predicción (puerto 5050; el 5000 suele estar reservado en Windows):**

```powershell
python api_prediccion\app.py
```

**Terminal 2 — API de reentrenamiento (puerto 5001):**

```powershell
python api_reentrenamiento\app.py
```

**Terminal 3 — Visualización web (puerto 8080):**

```powershell
python visualizacion\app.py
```

Abrir en el navegador: http://127.0.0.1:8080

## Ejemplo de entrada y salida de la API

**POST** `http://127.0.0.1:5050/predict`

```json
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

**Respuesta (200):**

```json
{
  "multimorbidity_flag": 1,
  "label": "Multimorbilidad probable",
  "probability": 0.73,
  "model": "RandomForestClassifier",
  "metric_reference": "F1_cv=0.6040"
}
```

**Ejemplo con curl (PowerShell):**

```powershell
curl.exe -X POST -H "Content-Type: application/json" -d "{\"pa_activity_level\":2,\"pa_avg_activity_intensity\":1.5,\"total_daily_active_minutes\":45,\"dental_brushing_freq\":1,\"daily_supplements\":0,\"diet_primary\":0,\"weight_kg\":18,\"age_derived\":8}" http://127.0.0.1:5050/predict
```

**Reentrenamiento — POST** `http://127.0.0.1:5001/retrain` con `multipart/form-data`, campo `file` = CSV DAP.

## Estructura del proyecto

```
BIY7121_EV3_CortesDiego/
├── api_prediccion/app.py
├── api_reentrenamiento/app.py
├── visualizacion/
├── modelo_y_preprocesamiento/
├── requirements.txt
└── README.md
```

## Notas

- El preprocesamiento en la API replica `preprocess.py` usado en el notebook.
- El reentrenamiento reutiliza el mismo `param_grid` guardado en `best_params.json`.
- Compatible con scikit-learn 1.3+; use las mismas versiones al cargar `model.joblib`.

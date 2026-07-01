# Documentación completa — BIY7121 EV3 CortesDiego

**Asignatura:** BIY7121 — Minería de datos  
**Entrega:** Evaluación 3 — Despliegue de modelos  
**Autor:** Diego Cortes  
**Proyecto grupal de origen:** Dog Aging Project (DAP) 2024  

---

## Tabla de contenidos

1. [Contexto del caso](#1-contexto-del-caso)
2. [Insight seleccionado](#2-insight-seleccionado)
3. [Objetivo de la EV3](#3-objetivo-de-la-ev3)
4. [Arquitectura de la solución](#4-arquitectura-de-la-solución)
5. [Ingesta de datos de entrenamiento](#5-ingesta-de-datos-de-entrenamiento)
6. [Preprocesamiento y limpieza de datos](#6-preprocesamiento-y-limpieza-de-datos)
7. [Ingeniería de variables](#7-ingeniería-de-variables)
8. [Entrenamiento, validación y selección del modelo](#8-entrenamiento-validación-y-selección-del-modelo)
9. [Exportación de artefactos](#9-exportación-de-artefactos)
10. [API de predicción](#10-api-de-predicción)
11. [API de reentrenamiento](#11-api-de-reentrenamiento)
12. [Visualización web](#12-visualización-web)
13. [Despliegue en producción](#13-despliegue-en-producción)
14. [Pruebas y casos de ejemplo](#14-pruebas-y-casos-de-ejemplo)
15. [Estructura del repositorio](#15-estructura-del-repositorio)
16. [Ejecución local](#16-ejecución-local)
17. [Decisiones de diseño y limitaciones](#17-decisiones-de-diseño-y-limitaciones)
18. [Referencias](#18-referencias)

---

## 1. Contexto del caso

El **Dog Aging Project (DAP)** es un estudio longitudinal sobre envejecimiento, salud y comportamiento de perros. El proyecto grupal del semestre analizó datos consolidados del DAP para identificar patrones entre:

- Hábitos del **tutor** (actividad física, cuidado preventivo, alimentación).
- Indicadores de **salud del perro** (condiciones crónicas, sistemas afectados, carga de enfermedad).

La Evaluación 3 (EV3) exige tomar **un insight** de ese proyecto grupal, entrenar un modelo de minería de datos, exportarlo y **desplegarlo mediante APIs** consumidas por una **visualización web**.

Este trabajo individual toma como base la **sección 8** del notebook grupal (`Proyecto_Mineria_DAPv3_MARKDOWNS.ipynb`), donde se observa que perfiles de menor cuidado preventivo y menor actividad se asocian con mayor carga de enfermedades.

---

## 2. Insight seleccionado

> **El nivel de cuidado preventivo y la actividad física del tutor predice la carga de enfermedades del perro.**

### Traducción operacional al modelo

| Concepto | Implementación |
|----------|----------------|
| Carga de enfermedades | `multimorbidity_flag`: 1 si el perro tiene **2 o más** condiciones de salud registradas |
| Cuidado preventivo | `preventive_care_score` + variables de cepillado dental, suplementos y dieta |
| Actividad del tutor | `pa_activity_level`, `pa_avg_activity_intensity`, `total_daily_active_minutes` |
| Tipo de problema | **Clasificación binaria** (multimorbilidad sí/no) |

### Por qué este insight

- Proviene directamente del análisis grupal (sección 8: *menor cuidado → mayor carga de salud*).
- Target binario claro y balanceable (~26% positivos en `maestro_dap.csv`).
- Encaja de forma natural con un formulario web: el usuario ingresa hábitos del tutor y datos del perro → recibe una predicción de riesgo.

---

## 3. Objetivo de la EV3

La solución entregada cumple los requisitos de la asignatura:

| Requerimiento | Cumplimiento |
|---------------|--------------|
| Notebook de entrenamiento | `BIY7121_EV3_CortesDiego.ipynb` |
| Preprocesamiento documentado | `modelo_y_preprocesamiento/preprocess.py` |
| Validación cruzada | `cross_val_score` con 5 folds |
| Tuning de hiperparámetros | `GridSearchCV` |
| Exportación del modelo | `model.joblib` + metadatos JSON |
| API de predicción | `api_prediccion/app.py` → `POST /predict` |
| API de reentrenamiento | `api_reentrenamiento/app.py` → `POST /retrain` |
| Visualización web | `visualizacion/` consumiendo la API real |
| README | `README.md` |

---

## 4. Arquitectura de la solución

```
┌─────────────────────────────────────────────────────────────────┐
│                        USUARIO / PROFESOR                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
   ┌──────────────────────┐      ┌──────────────────────┐
   │  Visualización web   │      │   Cliente HTTP       │
   │  (Vercel / Flask)    │      │   (curl, Postman)    │
   └──────────┬───────────┘      └──────────┬───────────┘
              │  POST /predict               │
              └──────────────┬───────────────┘
                             ▼
              ┌──────────────────────────────┐
              │   API de predicción (Flask)  │
              │   Render — puerto 5050       │
              └──────────┬───────────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
  preprocess.py    imputer.joblib   model.joblib
  (features)       (KNNImputer)     (Random Forest)
```

### Flujo de predicción

1. El usuario envía 8 campos numéricos (hábitos + datos del perro).
2. La API valida el JSON y construye el vector de 9 features (incluye `preventive_care_score` calculado).
3. Se aplica el **KNNImputer** exportado (consistencia con entrenamiento).
4. El **Random Forest** predice clase y probabilidad.
5. Se retorna JSON con `multimorbidity_flag`, `label`, `probability`.

### Componentes desacoplados

| Componente | Tecnología | Rol |
|------------|------------|-----|
| Entrenamiento | Python, scikit-learn, pandas | Offline — genera artefactos |
| Preprocesamiento compartido | `preprocess.py` | Notebook, APIs y reentrenamiento usan la misma lógica |
| API predicción | Flask + gunicorn | Inferencia en tiempo real |
| API reentrenamiento | Flask + gunicorn | Actualiza modelo con nuevos CSV |
| Frontend | HTML/CSS/JS (+ Flask local) | Formulario, CSV, guía de variables |

---

## 5. Ingesta de datos de entrenamiento

### Fuente principal utilizada

**Archivo:** `maestro_dap.csv`  
**Ubicación:** raíz del workspace o `BIY7121_EV3_CortesDiego/data/`  
**Registros:** ~50.188 filas  
**Tasa de multimorbilidad:** ~25.6%

El script `train_and_export.py` busca datos en este orden:

1. `../maestro_dap.csv`
2. `data/maestro_dap.csv`
3. `data/dap_consolidado.csv`
4. `../outputs/dap_consolidado.csv`
5. Si ninguno existe → genera **datos sintéticos** (solo para desarrollo)

### Formato maestro vs DAP crudo

El sistema detecta automáticamente el formato:

```python
def is_maestro_format(df):
    return "multimorbidity_flag" in df.columns and "preventive_care_score" in df.columns
```

| Aspecto | `maestro_dap.csv` | `dap_consolidado.csv` |
|---------|-------------------|------------------------|
| Target | `multimorbidity_flag` ya calculado | Se deriva de `salud_num_condiciones >= 2` |
| Features | Columnas alineadas al modelo | Se construyen desde columnas DAP crudas |
| `preventive_care_score` | Precalculado | Se calcula con z-scores de variables preventivas |

### Columnas relevantes del maestro

Entre otras, el dataset incluye: `age_derived`, `weight_kg`, `pa_activity_level`, `pa_avg_activity_intensity`, `total_daily_active_minutes`, `dental_brushing_freq`, `daily_supplements`, `diet_primary`, `preventive_care_score`, `multimorbidity_flag`.

---

## 6. Preprocesamiento y limpieza de datos

Toda la lógica está centralizada en `modelo_y_preprocesamiento/preprocess.py` para garantizar **consistencia** entre notebook, entrenamiento, predicción y reentrenamiento.

### 6.1 Imputación de valores faltantes — KNNImputer

**Método actual:** `sklearn.impute.KNNImputer`

| Parámetro | Valor |
|-----------|-------|
| `n_neighbors` | 5 |
| `weights` | `distance` |

**Motivación:** En `maestro_dap.csv` se detectaron faltantes relevantes:

- `daily_supplements`: ~19.7% NaN
- `diet_primary`: ~23.0% NaN

La imputación por **mediana** (método inicial) reemplazaba cada hueco con un valor fijo, ignorando correlaciones entre variables. **KNN** estima el valor faltante a partir de los *k* registros más similares en el espacio de features, lo que mejoró el rendimiento del modelo.

**Flujo:**

1. Construir matriz `X` **preservando NaNs** (no imputar durante la ingeniería de features).
2. Ajustar el imputer en el conjunto de entrenamiento.
3. Transformar train/test y, para producción, reajustar con **todos los datos** y exportar `imputer.joblib`.
4. En la API, aplicar el mismo imputer antes de `model.predict()`.

### 6.2 Codificación de variables categóricas

`diet_primary` puede venir como texto (`seco_comercial`, `mixto`, `casero_cocido`, etc.). Se mapea a numérico:

| Categoría | Valor |
|-----------|-------|
| `casero_cocido`, `mixto` | 1.0 |
| `seco_comercial`, `humedo_comercial`, `no_sabe` | 0.0 |

### 6.3 Validación de rangos (DAP crudo)

Para datos no-maestro:

- **Edad:** derivada de `dd_edad_anios`, `dd_age` o `dd_birth_year`; valores fuera de [0, 25] → NaN.
- **Peso:** conversión `dd_weight_lbs` → `weight_kg` (factor 0.453592).
- **Tipos:** `pd.to_numeric(..., errors="coerce")` en columnas de actividad, preventivo y salud.

### 6.4 Target

```python
multimorbidity_flag = 1  si  salud_num_condiciones >= 2
multimorbidity_flag = 0  en caso contrario
```

En el maestro, el target ya viene como `multimorbidity_flag`.

---

## 7. Ingeniería de variables

### Variables finales del modelo (orden fijo)

| # | Variable | Origen / significado |
|---|----------|----------------------|
| 1 | `preventive_care_score` | Score de cuidado preventivo (0–4 en API; precalculado en maestro) |
| 2 | `pa_activity_level` | Nivel de actividad del tutor (1–4) |
| 3 | `pa_avg_activity_intensity` | Intensidad de actividad (0–4) |
| 4 | `total_daily_active_minutes` | Minutos activos diarios del perro |
| 5 | `dental_brushing_freq` | Frecuencia de cepillado dental (0–6) |
| 6 | `daily_supplements` | Suplementos / examen dental (0–5) |
| 7 | `diet_primary` | Dieta orgánica o premium (0/1) |
| 8 | `weight_kg` | Peso del perro en kilogramos |
| 9 | `age_derived` | Edad del perro en años |

### Cálculo de `preventive_care_score` en la API

Cuando el usuario envía datos por formulario (sin el score precalculado):

```text
preventive_care_score = mean(
    dental_brushing_freq / 6,
    daily_supplements,
    diet_primary
) × 4
```

### Importancia de variables (modelo actual)

| Variable | Importancia |
|----------|-------------|
| `age_derived` | ~48.1% |
| `preventive_care_score` | ~15.1% |
| `weight_kg` | ~8.0% |
| `pa_avg_activity_intensity` | ~7.9% |
| `daily_supplements` | ~7.5% |
| Resto | < 6% c/u |

La **edad** es el predictor dominante; el cuidado preventivo y la actividad siguen siendo relevantes y alineados con el insight del proyecto grupal.

---

## 8. Entrenamiento, validación y selección del modelo

### 8.1 Partición de datos

- **75% entrenamiento / 25% prueba** (`train_test_split`, `stratify=y`, `random_state=42`).

### 8.2 Comparación de modelos candidatos

Se evaluaron tres algoritmos con **validación cruzada 5-fold** y métrica **F1-score**:

| Modelo | Descripción |
|--------|-------------|
| Naive Bayes Gaussiano | Pipeline con `StandardScaler` |
| Árbol de decisión | `max_depth=4`, `min_samples_leaf=50`, `class_weight=balanced` |
| Random Forest | Configuración base antes del tuning |

**Selección:** Random Forest — mejor F1 promedio en CV entre los candidatos.

### 8.3 Ajuste de hiperparámetros — GridSearchCV

```python
PARAM_GRID = {
    "n_estimators": [100, 150, 200],
    "max_depth": [6, 10, 14, None],
    "min_samples_leaf": [5, 10, 20],
}
```

- **Scoring:** F1
- **CV:** 5 folds
- **class_weight:** `balanced` (clase positiva minoritaria)

### 8.4 Mejor configuración encontrada

| Hiperparámetro | Valor |
|----------------|-------|
| `n_estimators` | 150 |
| `max_depth` | 14 |
| `min_samples_leaf` | 20 |

### 8.5 Métricas del modelo desplegado

| Métrica | Valor |
|---------|-------|
| **F1 CV (5-fold)** | **0.5596** ← métrica principal |
| F1 test (hold-out) | 0.5942 |
| ROC-AUC test | 0.8217 |
| Accuracy test | ~0.74 |

### 8.6 Justificación de F1 como métrica principal

El dataset tiene desbalance (~26% multimorbilidad). **F1** equilibra precisión y recall en la clase positiva, más informativo que accuracy sola para este problema clínico.

---

## 9. Exportación de artefactos

Tras el entrenamiento, `train_and_export.py` guarda en `modelo_y_preprocesamiento/`:

| Archivo | Contenido |
|---------|-----------|
| `model.joblib` | Random Forest entrenado con todos los datos |
| `imputer.joblib` | KNNImputer ajustado |
| `best_params.json` | Mejores hiperparámetros, métricas, `param_grid` |
| `imputation_medians.json` | Medianas de referencia (metadatos / DAP crudo) |
| `score_params.json` | Fuente de datos, método de imputación, stats preventivos |
| `feature_columns.json` | Orden de columnas del modelo |
| `feature_importances.csv` | Importancia Gini por variable |

El notebook y las APIs cargan estos artefactos con **joblib** — misma práctica vista en clases (serialización de modelos para despliegue).

---

## 10. API de predicción

**Archivo:** `api_prediccion/app.py`  
**Endpoint principal:** `POST /predict`  
**Health check:** `GET /health`

### Entrada (JSON)

Campos **obligatorios** (todos numéricos):

```json
{
  "pa_activity_level": 3,
  "pa_avg_activity_intensity": 2.0,
  "total_daily_active_minutes": 90,
  "dental_brushing_freq": 4,
  "daily_supplements": 1,
  "diet_primary": 1,
  "weight_kg": 12.5,
  "age_derived": 5
}
```

### Salida (JSON)

```json
{
  "multimorbidity_flag": 0,
  "label": "Sin multimorbilidad significativa",
  "probability": 0.2139,
  "model": "RandomForestClassifier",
  "metric_reference": "F1_cv=0.5596"
}
```

| Campo | Significado |
|-------|-------------|
| `multimorbidity_flag` | 0 = no, 1 = sí (umbral 50% en probabilidad) |
| `probability` | P(multimorbilidad) ∈ [0, 1] |
| `label` | Texto legible del resultado |
| `metric_reference` | Calidad del modelo en entrenamiento |

### Manejo de errores

- JSON inválido o vacío → 400
- Campos faltantes o no numéricos → 400
- Modelo no exportado → 500
- **CORS** habilitado para consumo desde Vercel

### URLs

| Entorno | URL |
|---------|-----|
| Local | `http://127.0.0.1:5050/predict` |
| Producción (Render) | `https://dap-predict-api.onrender.com/predict` |

> El puerto 5050 se usa en local porque el 5000 suele estar reservado en Windows.

---

## 11. API de reentrenamiento

**Archivo:** `api_reentrenamiento/app.py`  
**Endpoint:** `POST /retrain`  
**Puerto local:** 5001

### Entrada

- `multipart/form-data` con campo `file` (CSV), o
- JSON con `{"csv_path": "ruta/al/archivo.csv"}` (solo local)

### Comportamiento

1. Carga el CSV (maestro o DAP crudo).
2. Aplica el mismo preprocesamiento e imputación KNN.
3. Ejecuta **GridSearchCV** con el `param_grid` guardado en `best_params.json` (no improvisa otro modelo).
4. Reentrena con todos los datos y **sobrescribe** `model.joblib`, `imputer.joblib` y metadatos.

### Respuesta exitosa (ejemplo)

```json
{
  "status": "ok",
  "rows_used": 50188,
  "best_params": { "max_depth": 14, "min_samples_leaf": 20, "n_estimators": 150 },
  "f1_cv": 0.5596,
  "f1_test": 0.5942,
  "roc_auc_test": 0.8217,
  "metric_reference": "F1_cv=0.5596",
  "multimorbidity_rate": 0.256
}
```

> En Render free tier, el reentrenamiento con ~50k filas puede exceder el timeout; para la presentación se recomienda demo de **predicción** en producción y reentrenamiento en local si es necesario.

---

## 12. Visualización web

**Carpeta:** `visualizacion/`  
**Framework:** Flask (local) / sitio estático (Vercel en producción)

### Funcionalidades

1. **Guía y diccionario** — tabla explicativa de cada variable, rangos y significado del resultado.
2. **Formulario manual** — 8 campos con hints; botones de casos predefinidos (bajo/medio/alto riesgo).
3. **Carga CSV** — lee la primera fila de datos y llama a la API (columnas API o nombres DAP).
4. **Resultado visual** — badge, probabilidad, barra de riesgo animada, JSON crudo.
5. **Estados de carga** — overlay con spinner mientras la API responde.

### Consumo real de la API

El frontend usa `fetch()` contra `PREDICT_API_URL`:

- **Local:** inyectada por Flask (`{{ predict_api_url }}`).
- **Vercel:** generada en build desde variable de entorno `PREDICT_API_URL` → `config.js`.

### URL de producción

**https://multimorbilidad-canina.vercel.app**

---

## 13. Despliegue en producción

### Repositorio GitHub

`https://github.com/DiegoFCP/multimorbilidad_canina`

### API — Render

- Blueprint: `render.yaml` en la raíz del repo.
- Servicio: `dap-predict-api` (Python + gunicorn).
- `rootDir`: `BIY7121_EV3_CortesDiego`
- Comando: `gunicorn --chdir api_prediccion --bind 0.0.0.0:$PORT app:app`

### Frontend — Vercel

- `vercel.json` en la raíz del repo.
- Build: `node BIY7121_EV3_CortesDiego/scripts/vercel-build.js`
- Output: `BIY7121_EV3_CortesDiego/visualizacion`
- Variable de entorno: `PREDICT_API_URL=https://dap-predict-api.onrender.com/predict`

### Diagrama de despliegue

```
GitHub (push) ──► Render (API Flask + model.joblib)
              └──► Vercel (HTML/CSS/JS estático)
                        │
                        └── fetch ──► Render /predict
```

---

## 14. Pruebas y casos de ejemplo

### Caso bajo riesgo (perro joven, buen cuidado)

```json
{
  "pa_activity_level": 4,
  "pa_avg_activity_intensity": 3.0,
  "total_daily_active_minutes": 180,
  "dental_brushing_freq": 5,
  "daily_supplements": 2,
  "diet_primary": 1,
  "weight_kg": 12,
  "age_derived": 3
}
```

### Caso alto riesgo (senior, poco cuidado)

```json
{
  "pa_activity_level": 1,
  "pa_avg_activity_intensity": 0.5,
  "total_daily_active_minutes": 25,
  "dental_brushing_freq": 0,
  "daily_supplements": 0,
  "diet_primary": 0,
  "weight_kg": 30,
  "age_derived": 14
}
```

### CSV de prueba

Archivo `casos_prueba.csv` con columnas API. La web procesa **solo la primera fila** de datos.

### Verificación rápida

```powershell
# Health check
Invoke-RestMethod -Uri "https://dap-predict-api.onrender.com/health"

# Predicción
$body = '{"pa_activity_level":2,"pa_avg_activity_intensity":1.5,"total_daily_active_minutes":60,"dental_brushing_freq":2,"daily_supplements":1,"diet_primary":0,"weight_kg":15,"age_derived":6}'
Invoke-RestMethod -Uri "https://dap-predict-api.onrender.com/predict" -Method POST -Body $body -ContentType "application/json"
```

---

## 15. Estructura del repositorio

```
BIY7121_EV3_CortesDiego.ipynb          ← Notebook de entrenamiento (entrega EV3)

BIY7121_EV3_CortesDiego/
├── api_prediccion/
│   └── app.py                         ← API Flask predicción
├── api_reentrenamiento/
│   └── app.py                         ← API Flask reentrenamiento
├── visualizacion/
│   ├── app.py                         ← Servidor Flask local
│   ├── index.html                     ← Versión estática (Vercel)
│   ├── templates/index.html           ← Template Flask
│   └── static/
│       ├── style.css
│       └── app.js
├── modelo_y_preprocesamiento/
│   ├── preprocess.py                  ← Preprocesamiento compartido
│   ├── train_and_export.py            ← Entrenamiento + exportación
│   ├── model.joblib
│   ├── imputer.joblib
│   └── *.json                         ← Metadatos
├── data/
│   └── maestro_dap.csv                ← Dataset de entrenamiento
├── scripts/
│   └── vercel-build.js
├── requirements.txt
├── README.md
└── DOCUMENTACION_PROYECTO.md          ← Este documento
```

---

## 16. Ejecución local

```powershell
cd BIY7121_EV3_CortesDiego
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Entrenar (opcional, si hay datos nuevos):**

```powershell
python modelo_y_preprocesamiento\train_and_export.py
```

**Tres terminales:**

```powershell
# Terminal 1 — API predicción
python api_prediccion\app.py

# Terminal 2 — API reentrenamiento (opcional)
python api_reentrenamiento\app.py

# Terminal 3 — Visualización
python visualizacion\app.py
```

Abrir: **http://127.0.0.1:8080**

---

## 17. Decisiones de diseño y limitaciones

### Decisiones

| Decisión | Razón |
|----------|-------|
| `preprocess.py` compartido | Misma lógica en notebook, train, predict y retrain |
| KNNImputer vs mediana | Mejor aprovechamiento de correlaciones entre features; mejora F1 y ROC-AUC |
| Random Forest | Mejor F1 en CV; maneja no linealidad e interacciones |
| `class_weight=balanced` | Mitiga desbalance de clases |
| Flask | Framework visto en clases; simple para APIs REST |
| Frontend estático en Vercel | sklearn/joblib no es adecuado para serverless |
| API en Render | Hosting Python persistente con modelo en memoria |

### Limitaciones

1. **Edad domina el modelo** — el insight de cuidado preventivo es válido pero con menor peso relativo que `age_derived`.
2. **CSV en la web** — solo primera fila; no predicción masiva.
3. **Render free tier** — cold start (~30–60 s tras inactividad); timeout limitado para reentrenamiento pesado.
4. **Generalización** — modelo entrenado con datos DAP; no sustituye diagnóstico veterinario.
5. **`diet_primary` en CSV** — debe ser numérico (0/1) en la carga web; texto categórico solo en pipeline de entrenamiento maestro.

---

## 18. Referencias

- Notebook grupal: `Proyecto_Mineria_DAPv3_MARKDOWNS.ipynb` (sección 8 — salud y multimorbilidad).
- Material de curso: despliegue de modelos, serialización joblib, APIs Flask (BIY7121).
- Dataset: `maestro_dap.csv` — Dog Aging Project 2024.
- Documentación técnica: `README.md` (instrucciones rápidas de ejecución).

---

*Documento generado para la presentación EV3 — BIY7121 — CortesDiego. Última actualización: modelo con KNNImputer, Random Forest tunado, despliegue Vercel + Render.*

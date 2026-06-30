"""
Preprocesamiento compartido — EV3 BIY7121 CortesDiego
Dog Aging Project: multimorbilidad canina desde hábitos del tutor.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

RANDOM_STATE = 42
LBS_TO_KG = 0.453592

# Columnas crudas en dap_consolidado
RAW_ACTIVITY = [
    "pa_activity_level",
    "pa_avg_activity_intensity",
    "pa_avg_daily_active_minutes",
]
RAW_PREVENTIVE = [
    "mp_dental_brushing_frequency",
    "mp_dental_examination_frequency",
    "mp_professional_grooming",
    "df_primary_diet_component_organic",
]
RAW_DOG = ["dd_edad_anios", "dd_weight_lbs"]

# Proxy para daily_supplements si no existe en el CSV
SUPPLEMENT_PROXY = "mp_dental_examination_frequency"

# Campos esperados en la API (entrada del usuario)
API_INPUT_FIELDS = [
    "pa_activity_level",
    "pa_avg_activity_intensity",
    "total_daily_active_minutes",
    "dental_brushing_freq",
    "daily_supplements",
    "diet_primary",
    "weight_kg",
    "age_derived",
]

# Columnas finales del modelo (orden fijo)
FEATURE_ORDER = [
    "preventive_care_score",
    "pa_activity_level",
    "pa_avg_activity_intensity",
    "total_daily_active_minutes",
    "dental_brushing_freq",
    "daily_supplements",
    "diet_primary",
    "weight_kg",
    "age_derived",
]

PARAM_GRID = {
    "n_estimators": [100, 150, 200],
    "max_depth": [6, 10, 14, None],
    "min_samples_leaf": [5, 10, 20],
}

DIET_PRIMARY_MAP = {
    "casero_cocido": 1.0,
    "mixto": 1.0,
    "seco_comercial": 0.0,
    "humedo_comercial": 0.0,
    "no_sabe": 0.0,
}


def artifact_dir(base: Path | None = None) -> Path:
    return base or Path(__file__).resolve().parent


def imputar_mediana_segura(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.isna().all():
        return s.fillna(0)
    return s.fillna(s.median())


def zscore_seguro(s: pd.Series) -> pd.Series:
    s = imputar_mediana_segura(s)
    std = s.std()
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std


def score_promedio_z(df_base: pd.DataFrame, columnas: list[str]) -> pd.Series:
    columnas = [c for c in columnas if c in df_base.columns]
    if not columnas:
        return pd.Series(0.0, index=df_base.index)
    matriz = pd.DataFrame({c: zscore_seguro(df_base[c]) for c in columnas}, index=df_base.index)
    return matriz.mean(axis=1)


def ensure_dog_age(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "dd_edad_anios" not in out.columns:
        if "dd_age" in out.columns:
            out["dd_edad_anios"] = pd.to_numeric(out["dd_age"], errors="coerce")
        elif "dd_birth_year" in out.columns:
            out["dd_edad_anios"] = 2024 - pd.to_numeric(out["dd_birth_year"], errors="coerce")
        else:
            out["dd_edad_anios"] = np.nan
    out["dd_edad_anios"] = pd.to_numeric(out["dd_edad_anios"], errors="coerce")
    out.loc[(out["dd_edad_anios"] < 0) | (out["dd_edad_anios"] > 25), "dd_edad_anios"] = np.nan
    return out


def ensure_weight(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "dd_weight_lbs" not in out.columns and "dd_weight_kg" in out.columns:
        out["dd_weight_lbs"] = pd.to_numeric(out["dd_weight_kg"], errors="coerce") / LBS_TO_KG
    out["dd_weight_lbs"] = pd.to_numeric(out.get("dd_weight_lbs", 0), errors="coerce")
    return out


def resolve_supplements_column(df: pd.DataFrame) -> str:
    if "daily_supplements" in df.columns:
        return "daily_supplements"
    if "mp_daily_supplements" in df.columns:
        return "mp_daily_supplements"
    return SUPPLEMENT_PROXY


def is_maestro_format(df: pd.DataFrame) -> bool:
    return "multimorbidity_flag" in df.columns and "preventive_care_score" in df.columns


def encode_diet_primary(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return imputar_mediana_segura(series)
    mapped = series.astype(str).str.strip().str.lower().map(DIET_PRIMARY_MAP)
    return imputar_mediana_segura(mapped)


def estimate_preventive_care_score_api(cleaned: dict[str, float]) -> float:
    """Aproxima preventive_care_score (escala 0-4) desde campos de la API."""
    parts = [
        cleaned.get("dental_brushing_freq", 0) / 6.0,
        cleaned.get("daily_supplements", 0),
        cleaned.get("diet_primary", 0),
    ]
    return float(np.mean(parts)) * 4.0


def prepare_maestro_features(
    df: pd.DataFrame,
    medians: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.Series, dict[str, float], dict[str, dict[str, float]]]:
    """Construye X e y desde maestro_dap.csv (columnas ya alineadas al modelo)."""
    data = df.copy()
    X = pd.DataFrame(index=data.index)
    for col in FEATURE_ORDER:
        if col == "diet_primary":
            X[col] = encode_diet_primary(data["diet_primary"])
        else:
            X[col] = imputar_mediana_segura(data[col]) if col in data.columns else 0.0

    if medians is None:
        medians = {c: float(X[c].median()) for c in FEATURE_ORDER}

    y = pd.to_numeric(data["multimorbidity_flag"], errors="coerce").fillna(0).astype(int)
    return X, y, medians, {}


def build_target(df: pd.DataFrame) -> pd.Series:
    cond = pd.to_numeric(df.get("salud_num_condiciones", 0), errors="coerce").fillna(0)
    return (cond >= 2).astype(int)


def compute_preventive_stats(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Calcula media/std por columna preventiva para z-score en inferencia."""
    stats: dict[str, dict[str, float]] = {}
    for col in RAW_PREVENTIVE:
        if col in df.columns:
            s = imputar_mediana_segura(df[col])
            stats[col] = {"mean": float(s.mean()), "std": float(s.std() if s.std() else 1.0)}
    return stats


def preventive_care_score_from_stats(
    row: dict[str, Any],
    preventive_stats: dict[str, dict[str, float]],
    medians: dict[str, float],
) -> float:
    z_vals = []
    mapping = {
        "mp_dental_brushing_frequency": row.get("dental_brushing_freq", row.get("mp_dental_brushing_frequency")),
        "mp_dental_examination_frequency": row.get("daily_supplements", row.get("mp_dental_examination_frequency")),
        "mp_professional_grooming": row.get("mp_professional_grooming", medians.get("mp_professional_grooming", 0)),
        "df_primary_diet_component_organic": row.get("diet_primary", row.get("df_primary_diet_component_organic")),
    }
    for col, val in mapping.items():
        if col not in preventive_stats:
            continue
        v = pd.to_numeric(val, errors="coerce")
        if pd.isna(v):
            v = medians.get(col, 0)
        mean = preventive_stats[col]["mean"]
        std = preventive_stats[col]["std"] or 1.0
        z_vals.append((float(v) - mean) / std)
    return float(np.mean(z_vals)) if z_vals else 0.0


def prepare_dataframe_features(
    df: pd.DataFrame,
    medians: dict[str, float] | None = None,
    preventive_stats: dict[str, dict[str, float]] | None = None,
) -> tuple[pd.DataFrame, pd.Series, dict[str, float], dict[str, dict[str, float]]]:
    """Construye X e y desde un DataFrame DAP crudo o maestro_dap."""
    if is_maestro_format(df):
        return prepare_maestro_features(df, medians)

    df = ensure_dog_age(ensure_weight(df.copy()))

    for col in RAW_ACTIVITY + RAW_PREVENTIVE + RAW_DOG + ["salud_num_condiciones"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if medians is None:
        medians = {}
        for col in RAW_ACTIVITY + RAW_PREVENTIVE + RAW_DOG:
            if col in df.columns:
                medians[col] = float(imputar_mediana_segura(df[col]).median())

    if preventive_stats is None:
        preventive_stats = compute_preventive_stats(df)

    supp_col = resolve_supplements_column(df)

    rows = []
    for idx in df.index:
        row_raw = df.loc[idx]
        dental = row_raw.get("mp_dental_brushing_frequency", medians.get("mp_dental_brushing_frequency", 0))
        if pd.isna(dental):
            dental = medians.get("mp_dental_brushing_frequency", 0)

        supplements = row_raw.get(supp_col, medians.get(supp_col, medians.get(SUPPLEMENT_PROXY, 0)))
        if pd.isna(supplements):
            supplements = medians.get(supp_col, medians.get(SUPPLEMENT_PROXY, 0))

        diet = row_raw.get("df_primary_diet_component_organic", medians.get("df_primary_diet_component_organic", 0))
        if pd.isna(diet):
            diet = medians.get("df_primary_diet_component_organic", 0)

        weight_lbs = row_raw.get("dd_weight_lbs", medians.get("dd_weight_lbs", 0))
        if pd.isna(weight_lbs):
            weight_lbs = medians.get("dd_weight_lbs", 0)

        age = row_raw.get("dd_edad_anios", medians.get("dd_edad_anios", 0))
        if pd.isna(age):
            age = medians.get("dd_edad_anios", 0)

        act_level = row_raw.get("pa_activity_level", medians.get("pa_activity_level", 0))
        act_int = row_raw.get("pa_avg_activity_intensity", medians.get("pa_avg_activity_intensity", 0))
        act_min = row_raw.get("pa_avg_daily_active_minutes", medians.get("pa_avg_daily_active_minutes", 0))

        api_row = {
            "pa_activity_level": act_level,
            "pa_avg_activity_intensity": act_int,
            "total_daily_active_minutes": act_min,
            "dental_brushing_freq": dental,
            "daily_supplements": supplements,
            "diet_primary": diet,
            "weight_kg": float(weight_lbs) * LBS_TO_KG,
            "age_derived": age,
            "mp_professional_grooming": row_raw.get(
                "mp_professional_grooming", medians.get("mp_professional_grooming", 0)
            ),
            "mp_dental_examination_frequency": supplements,
            "df_primary_diet_component_organic": diet,
            "mp_dental_brushing_frequency": dental,
        }
        api_row["preventive_care_score"] = preventive_care_score_from_stats(
            api_row, preventive_stats, medians
        )
        rows.append({k: api_row[k] for k in FEATURE_ORDER})

    X = pd.DataFrame(rows, index=df.index)[FEATURE_ORDER]
    y = build_target(df)
    return X, y, medians, preventive_stats


def validate_input(data: dict[str, Any]) -> tuple[dict[str, float], list[str]]:
    errors: list[str] = []
    if not isinstance(data, dict):
        return {}, ["El cuerpo debe ser un objeto JSON."]

    cleaned: dict[str, float] = {}
    for field in API_INPUT_FIELDS:
        if field not in data:
            errors.append(f"Campo requerido faltante: {field}")
            continue
        try:
            cleaned[field] = float(data[field])
        except (TypeError, ValueError):
            errors.append(f"Campo '{field}' debe ser numérico.")

    return cleaned, errors


def build_features_from_api(
    data: dict[str, Any],
    medians: dict[str, float],
    preventive_stats: dict[str, dict[str, float]],
    data_source: str = "dap",
) -> pd.DataFrame:
    cleaned, errors = validate_input(data)
    if errors:
        raise ValueError("; ".join(errors))

    row = dict(cleaned)
    if data_source == "maestro" or not preventive_stats:
        row["preventive_care_score"] = estimate_preventive_care_score_api(cleaned)
    else:
        row["mp_dental_brushing_frequency"] = row["dental_brushing_freq"]
        row["mp_dental_examination_frequency"] = row["daily_supplements"]
        row["df_primary_diet_component_organic"] = row["diet_primary"]
        row["mp_professional_grooming"] = medians.get("mp_professional_grooming", 0)
        row["preventive_care_score"] = preventive_care_score_from_stats(
            row, preventive_stats, medians
        )

    return pd.DataFrame([[row[c] for c in FEATURE_ORDER]], columns=FEATURE_ORDER)


def load_metadata(base: Path | None = None) -> tuple[dict[str, float], dict[str, dict[str, float]], list[str], dict, str]:
    base = artifact_dir(base)
    with open(base / "imputation_medians.json", encoding="utf-8") as f:
        medians = json.load(f)
    with open(base / "score_params.json", encoding="utf-8") as f:
        score_params = json.load(f)
    with open(base / "feature_columns.json", encoding="utf-8") as f:
        features = json.load(f)
    best_path = base / "best_params.json"
    best = {}
    if best_path.exists():
        with open(best_path, encoding="utf-8") as f:
            best = json.load(f)
    data_source = score_params.get("data_source", "dap")
    return medians, score_params.get("preventive_stats", {}), features, best, data_source


def save_metadata(
    medians: dict[str, float],
    preventive_stats: dict[str, dict[str, float]],
    best_params: dict,
    metric_reference: str,
    base: Path | None = None,
    data_source: str = "dap",
) -> None:
    base = artifact_dir(base)
    base.mkdir(parents=True, exist_ok=True)
    with open(base / "imputation_medians.json", "w", encoding="utf-8") as f:
        json.dump(medians, f, indent=2)
    with open(base / "score_params.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "preventive_stats": preventive_stats,
                "supplement_proxy": SUPPLEMENT_PROXY,
                "data_source": data_source,
            },
            f,
            indent=2,
        )
    with open(base / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(FEATURE_ORDER, f, indent=2)
    payload = dict(best_params)
    payload["metric_reference"] = metric_reference
    payload["param_grid"] = PARAM_GRID
    with open(base / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

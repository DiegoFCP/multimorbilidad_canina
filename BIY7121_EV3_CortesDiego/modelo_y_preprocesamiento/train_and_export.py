"""
Entrena el modelo y exporta artefactos a modelo_y_preprocesamiento/.
Ejecutar desde la raíz del proyecto EV3:
  python modelo_y_preprocesamiento/train_and_export.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "modelo_y_preprocesamiento"))

from preprocess import (  # noqa: E402
    FEATURE_ORDER,
    PARAM_GRID,
    RANDOM_STATE,
    apply_imputer,
    artifact_dir,
    fit_imputer,
    is_maestro_format,
    prepare_dataframe_features,
    save_metadata,
)

DATA_CANDIDATES = [
    ROOT.parent / "maestro_dap.csv",
    ROOT / "data" / "maestro_dap.csv",
    ROOT / "data" / "dap_consolidado.csv",
    ROOT.parent / "outputs" / "dap_consolidado.csv",
]


def generate_synthetic_dap(n: int = 8000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Dataset sintético con relación cuidado bajo -> mayor multimorbilidad."""
    rng = np.random.default_rng(seed)
    care = rng.uniform(0, 1, n)
    df = pd.DataFrame(
        {
            "pa_activity_level": np.clip(rng.integers(1, 4, n) + (care > 0.5).astype(int), 1, 4),
            "pa_avg_activity_intensity": np.clip(rng.normal(1.5 + care, 0.6, n), 0, 4),
            "pa_avg_daily_active_minutes": np.clip(rng.normal(40 + 80 * care, 25, n), 5, 300),
            "mp_dental_brushing_frequency": np.clip(rng.integers(0, 5, n) + (care > 0.4).astype(int), 0, 6),
            "mp_dental_examination_frequency": np.clip(rng.integers(0, 4, n) + (care > 0.5).astype(int), 0, 5),
            "mp_professional_grooming": (care > 0.55).astype(int),
            "df_primary_diet_component_organic": (care > 0.45).astype(int),
            "dd_edad_anios": np.clip(rng.normal(6, 3, n), 1, 18),
            "dd_weight_lbs": np.clip(rng.normal(45, 18, n), 8, 120),
        }
    )
  # multimorbilidad inversamente relacionada con cuidado y actividad
    logit = -1.2 + 1.8 * (1 - care) + 0.02 * df["dd_edad_anios"] + rng.normal(0, 0.35, n)
    prob = 1 / (1 + np.exp(-logit))
    multimorb = (rng.random(n) < prob).astype(int)
    df["salud_num_condiciones"] = np.where(
        multimorb == 1,
        rng.integers(2, 6, n),
        rng.integers(0, 2, n),
    )
    return df


def load_data() -> pd.DataFrame:
    for path in DATA_CANDIDATES:
        if path.exists():
            print(f"Cargando datos desde: {path}")
            return pd.read_csv(path, low_memory=False)
    print("No se encontró dap_consolidado.csv — usando dataset sintético para exportar artefactos.")
    return generate_synthetic_dap()


def main() -> None:
    df = load_data()
    X_raw, y, medians, preventive_stats = prepare_dataframe_features(df)

    mask = y.notna()
    X_raw, y = X_raw.loc[mask], y.loc[mask]

    if y.nunique() < 2:
        raise ValueError("El target multimorbidity_flag requiere al menos dos clases.")

    print(f"Filas: {len(X_raw)} | Features: {len(FEATURE_ORDER)} | Tasa multimorbilidad: {y.mean():.3f}")
    missing_pct = X_raw.isna().mean().mul(100).round(2)
    if missing_pct.max() > 0:
        print("NaNs antes de KNNImputer (% por columna):")
        print(missing_pct[missing_pct > 0].to_string())

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    imputer_cv = fit_imputer(X_train_raw)
    X_train = apply_imputer(X_train_raw, imputer_cv)
    X_test = apply_imputer(X_test_raw, imputer_cv)

    models = {
        "Naive Bayes Gaussian": make_pipeline(StandardScaler(), GaussianNB()),
        "Árbol de Decisión": DecisionTreeClassifier(
            max_depth=4, min_samples_leaf=50, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    results = []
    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1", n_jobs=-1)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        auc = roc_auc_score(y_test, proba) if proba is not None else float("nan")
        results.append(
            {
                "Modelo": name,
                "F1_test": f1_score(y_test, pred),
                "Accuracy_test": accuracy_score(y_test, pred),
                "ROC_AUC_test": auc,
                "F1_cv_mean": float(cv_scores.mean()),
                "F1_cv_std": float(cv_scores.std()),
            }
        )
        print(f"\n{name}: F1_cv={cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    results_df = pd.DataFrame(results).sort_values("F1_cv_mean", ascending=False)
    print("\nComparación de modelos:")
    print(results_df.round(4).to_string(index=False))

    rf = RandomForestClassifier(
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    )
    grid = GridSearchCV(
        rf, PARAM_GRID, cv=5, scoring="f1", n_jobs=-1, refit=True
    )
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    imputer_prod = fit_imputer(X_raw)
    X_full = apply_imputer(X_raw, imputer_prod)
    best_model.fit(X_full, y)  # reentrenar con todos los datos para despliegue

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    f1_test = f1_score(y_test, y_pred)
    metric_ref = f"F1_cv={grid.best_score_:.4f}"

    print(f"\nMejor RF: {grid.best_params_}")
    print(f"F1 test: {f1_test:.4f} | ROC-AUC test: {roc_auc_score(y_test, y_proba):.4f}")
    print(classification_report(y_test, y_pred, target_names=["Sin multimorbilidad", "Multimorbilidad"]))
    print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))

    out = artifact_dir()
    joblib.dump(best_model, out / "model.joblib")
    joblib.dump(imputer_prod, out / "imputer.joblib")

    best_payload = {
        "best_params": grid.best_params_,
        "best_cv_f1": float(grid.best_score_),
        "f1_test": float(f1_test),
        "roc_auc_test": float(roc_auc_score(y_test, y_proba)),
    }
    save_metadata(
        medians, preventive_stats, best_payload, metric_ref, out,
        data_source="maestro" if is_maestro_format(df) else "dap",
    )

    importancias = pd.DataFrame(
        {"variable": FEATURE_ORDER, "importancia": best_model.feature_importances_}
    ).sort_values("importancia", ascending=False)
    importancias.to_csv(out / "feature_importances.csv", index=False)
    print(f"\nArtefactos guardados en: {out}")


if __name__ == "__main__":
    main()

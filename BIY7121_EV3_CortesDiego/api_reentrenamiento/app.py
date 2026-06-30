"""
API de reentrenamiento — EV3 BIY7121 CortesDiego
Puerto 5001
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "modelo_y_preprocesamiento"
sys.path.insert(0, str(MODEL_DIR))

from preprocess import (  # noqa: E402
    PARAM_GRID,
    RANDOM_STATE,
    apply_imputer,
    artifact_dir,
    fit_imputer,
    is_maestro_format,
    load_metadata,
    prepare_dataframe_features,
    save_metadata,
)

app = Flask(__name__)


def run_retrain(df: pd.DataFrame) -> dict:
    X_raw, y, medians, preventive_stats = prepare_dataframe_features(df)
    mask = y.notna() & (y.nunique() >= 1)
    X_raw, y = X_raw.loc[mask], y.loc[mask]

    if y.nunique() < 2:
        raise ValueError("El CSV debe producir al menos dos clases en multimorbidity_flag.")

    _, _, _, best_meta, _, _ = load_metadata(MODEL_DIR)
    param_grid = best_meta.get("param_grid", PARAM_GRID)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    imputer_cv = fit_imputer(X_train_raw)
    X_train = apply_imputer(X_train_raw, imputer_cv)
    X_test = apply_imputer(X_test_raw, imputer_cv)

    rf = RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
    grid = GridSearchCV(rf, param_grid, cv=5, scoring="f1", n_jobs=-1, refit=True)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    imputer_prod = fit_imputer(X_raw)
    X_full = apply_imputer(X_raw, imputer_prod)
    best_model.fit(X_full, y)

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    f1_test = float(f1_score(y_test, y_pred))
    auc_test = float(roc_auc_score(y_test, y_proba))
    metric_ref = f"F1_cv={grid.best_score_:.4f}"

    out = artifact_dir(MODEL_DIR)
    joblib.dump(best_model, out / "model.joblib")
    joblib.dump(imputer_prod, out / "imputer.joblib")

    best_payload = {
        "best_params": grid.best_params_,
        "best_cv_f1": float(grid.best_score_),
        "f1_test": f1_test,
        "roc_auc_test": auc_test,
    }
    save_metadata(
        medians, preventive_stats, best_payload, metric_ref, out,
        data_source="maestro" if is_maestro_format(df) else "dap",
    )

    return {
        "status": "ok",
        "rows_used": int(len(X)),
        "best_params": grid.best_params_,
        "f1_cv": float(grid.best_score_),
        "f1_test": f1_test,
        "roc_auc_test": auc_test,
        "metric_reference": metric_ref,
        "multimorbidity_rate": float(y.mean()),
    }


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "api_reentrenamiento"})


@app.route("/retrain", methods=["POST"])
def retrain():
    df = None

    if "file" in request.files and request.files["file"].filename:
        file = request.files["file"]
        try:
            content = file.read()
            df = pd.read_csv(io.BytesIO(content), low_memory=False)
        except Exception as exc:
            return jsonify({"error": f"No se pudo leer el CSV: {exc}"}), 400
    elif request.is_json:
        body = request.get_json(silent=True) or {}
        path = body.get("csv_path")
        if path:
            p = Path(path)
            if not p.exists():
                return jsonify({"error": f"Archivo no encontrado: {path}"}), 400
            try:
                df = pd.read_csv(p, low_memory=False)
            except Exception as exc:
                return jsonify({"error": f"No se pudo leer el CSV: {exc}"}), 400

    if df is None or df.empty:
        return jsonify(
            {
                "error": "Envíe un CSV en multipart/form-data (campo 'file') "
                "o JSON con {'csv_path': 'ruta/al/archivo.csv'}"
            }
        ), 400

    try:
        result = run_retrain(df)
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Error en reentrenamiento: {exc}"}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False)

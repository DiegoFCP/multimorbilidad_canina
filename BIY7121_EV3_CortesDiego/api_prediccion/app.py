"""
API de predicción — EV3 BIY7121 CortesDiego
Puerto 5000
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
from flask import Flask, jsonify, request

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "modelo_y_preprocesamiento"
sys.path.insert(0, str(MODEL_DIR))

from preprocess import build_features_from_api, load_metadata  # noqa: E402

app = Flask(__name__)


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


_model = None
_imputer = None
_medians = None
_preventive_stats = None
_data_source = "dap"
_metric_reference = "F1_cv"


def get_model():
    global _model, _imputer, _medians, _preventive_stats, _data_source, _metric_reference
    if _model is None:
        _medians, _preventive_stats, _, best, _data_source, _imputer = load_metadata(MODEL_DIR)
        _metric_reference = best.get("metric_reference", "F1_cv")
        model_path = MODEL_DIR / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(
                f"No existe {model_path}. Ejecute modelo_y_preprocesamiento/train_and_export.py"
            )
        _model = joblib.load(model_path)
    return _model, _imputer, _medians, _preventive_stats, _data_source


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "api_prediccion"})


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return "", 204

    if not request.is_json:
        return jsonify({"error": "Content-Type debe ser application/json"}), 400

    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "JSON inválido o vacío"}), 400

    try:
        model, imputer, medians, preventive_stats, data_source = get_model()
        X = build_features_from_api(
            data, medians, preventive_stats, data_source=data_source, imputer=imputer
        )
        pred = int(model.predict(X)[0])
        proba = float(model.predict_proba(X)[0][1])
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 500
    except Exception as exc:
        return jsonify({"error": f"Error interno: {exc}"}), 500

    label = "Multimorbilidad probable" if pred == 1 else "Sin multimorbilidad significativa"
    return jsonify(
        {
            "multimorbidity_flag": pred,
            "label": label,
            "probability": round(proba, 4),
            "model": "RandomForestClassifier",
            "metric_reference": _metric_reference,
        }
    )


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PREDICT_PORT", "5050"))
    app.run(host="127.0.0.1", port=port, debug=False)

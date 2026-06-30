"""
Visualización web — EV3 BIY7121 CortesDiego
Puerto 8080 — consume api_prediccion en puerto 5000
"""
import os
from pathlib import Path

from flask import Flask, render_template

ROOT = Path(__file__).resolve().parent
app = Flask(__name__, template_folder=str(ROOT / "templates"), static_folder=str(ROOT / "static"))


@app.route("/")
def index():
    api_url = os.environ.get("PREDICT_API_URL", "http://127.0.0.1:5050/predict")
    return render_template("index.html", predict_api_url=api_url)


@app.context_processor
def inject_api_url():
    return {"predict_api_url": os.environ.get("PREDICT_API_URL", "http://127.0.0.1:5050/predict")}


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=False)

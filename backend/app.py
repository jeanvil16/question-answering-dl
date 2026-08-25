"""Flask backend for the Question Answering system.

Endpoints
---------
GET  /api/health          : status check, whether a model is loaded
POST /api/predict         : run inference on {context, question}
GET  /api/model/info      : architecture, hyper-parameters, best metrics
GET  /api/model/history   : full training history for the frontend charts

The server starts on port 5000 by default (configurable via the PORT
environment variable). CORS headers are added for local Vite dev servers.
"""

import json
import os
import sys
from pathlib import Path

# Allow importing model/ and preprocessing/ from sibling packages.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flask import Flask, jsonify, request
from flask_cors import CORS

import backend.inference as inference_module
from backend.inference import load_engine

# ---------------------------------------------------------------------------
#  App factory
# ---------------------------------------------------------------------------

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

SAVED = ROOT / "saved_models"
APP_VERSION = "1.0.0"


def _error(code, message):
    resp = jsonify({"error": {"code": code, "message": message}})
    resp.status_code = code
    return resp


# ---------------------------------------------------------------------------
#  Health
# ---------------------------------------------------------------------------

@app.route("/api/health")
def health():
    eng = inference_module.engine
    return jsonify({
        "status": "ok",
        "model_loaded": eng.loaded if eng else False,
        "device": str(eng.device) if eng and eng.loaded else None,
        "backend_version": APP_VERSION,
    })


# ---------------------------------------------------------------------------
#  Predict
# ---------------------------------------------------------------------------

@app.route("/api/predict", methods=["POST"])
def predict():
    body = request.get_json(silent=True)
    if not body or not isinstance(body, dict):
        return _error(400, "Request body must be a JSON object.")

    context = (body.get("context") or "").strip()
    question = (body.get("question") or "").strip()

    if not context:
        return _error(400, "The 'context' field is required and must not be empty.")
    if len(context) < 10:
        return _error(400, "Context is too short. Please provide at least 10 characters.")
    if len(context) > 25_000:
        return _error(400, "Context exceeds the 25 000 character limit.")
    if not question:
        return _error(400, "The 'question' field is required and must not be empty.")
    if len(question) < 3:
        return _error(400, "Question is too short (minimum 3 characters).")

    if not inference_module.engine or not inference_module.engine.loaded:
        return _error(503, "No trained model loaded. Run  python training/train.py  first.")

    try:
        result = inference_module.engine.predict(context, question)
    except Exception as exc:
        # Never leak internal tracebacks to the client.
        print(f"[predict] ERROR: {exc}", file=sys.stderr)
        return _error(500, "An internal error occurred during prediction.")

    return jsonify(result)


# ---------------------------------------------------------------------------
#  Model information
# ---------------------------------------------------------------------------

@app.route("/api/model/info")
def model_info():
    config_path = SAVED / "model_config.json"
    if not config_path.exists():
        return _error(503, "Model not found. Train first.")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    from model.qa_model import ExtractiveQAModel, QAModelConfig, count_parameters
    cfg = QAModelConfig(**config)
    temp_model = ExtractiveQAModel(cfg)
    param_count = count_parameters(temp_model)

    metrics_path = SAVED / "metrics.json"
    best = {}
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            best = json.load(f)

    return jsonify({
        "config": config,
        "trainable_parameters": param_count,
        "best_metrics": best,
    })


@app.route("/api/model/history")
def model_history():
    hist_path = SAVED / "training_history.json"
    if not hist_path.exists():
        return jsonify({"available": False, "history": []})
    with open(hist_path, "r", encoding="utf-8") as f:
        history = json.load(f)
    return jsonify({"available": True, "history": history})


# ---------------------------------------------------------------------------
#  Run
# ---------------------------------------------------------------------------

def create_app():
    load_engine()
    eng = inference_module.engine
    status = "LOADED" if (eng and eng.loaded) else "NOT FOUND"
    print("=" * 58)
    print(f"  QA Backend  v{APP_VERSION}")
    print(f"  Model : {status}")
    if eng and eng.loaded:
        print(f"  Device: {eng.device}")
        bm = eng.best_metrics
        if bm:
            print(f"  Best  : EM={bm.get('val_em',0):.3f}  F1={bm.get('val_f1',0):.3f}"
                  f"  (epoch {bm.get('epoch','-')})")
    print("=" * 58)
    return app


if __name__ == "__main__":
    create_app().run(
        host="127.0.0.1",
        port=int(os.environ.get("PORT", 5000)),
        debug=False,
        use_reloader=False,   # avoid double-loading the model
    )

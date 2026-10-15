"""
api.py - Step 9: serve the model as a REST API with Flask.

Other programs (a dashboard, a gateway, a mobile app) can send sensor readings
over HTTP and get a failure prediction back.

Start the server from the project folder:
    python -m src.api

Send a test request from a second terminal:
    curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d @docs/sample_request.json
(Windows PowerShell: use curl.exe instead of curl.)

Security note: this demo has NO login. Do not expose it to the internet as it is.
"""
from flask import Flask, jsonify, request

from src import config
from src.predict import predict_latest
from src.train import load_model_bundle

app = Flask(__name__)
_bundle = None  # the model is loaded once and reused for every request


def get_bundle():
    """Load the trained model the first time it is needed."""
    global _bundle
    if _bundle is None:
        if not config.MODEL_PATH.exists():
            raise FileNotFoundError("No trained model found. Run 'python main.py' first.")
        _bundle = load_model_bundle()
    return _bundle


@app.get("/")
def home():
    """Home page: lists the endpoints so visitors know where to go."""
    return jsonify({
        "project": "AI-Powered Predictive Maintenance for IoT Devices",
        "endpoints": {
            "GET /health": "check that the server and the trained model are ready",
            "POST /predict": "send hourly sensor readings and get a failure prediction (JSON body needed)",
        },
        "try_this": 'curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "@docs/sample_request.json"',
    })


@app.errorhandler(404)
def page_not_found(error):
    return jsonify({"error": "Page not found.", "try": ["/", "/health", "POST /predict"]}), 404


@app.errorhandler(405)
def wrong_method(error):
    return jsonify({"error": "This endpoint needs a POST request with JSON. "
                             "Browsers only send GET requests, so use curl (see the home page '/')."}), 405


@app.get("/health")
def health():
    """Quick check that the server is running."""
    return jsonify({"status": "ok", "model_file_found": config.MODEL_PATH.exists()})


@app.post("/predict")
def predict():
    """Input: {"machine_id": "...", "readings": [hourly sensor readings, oldest first]}."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Send a JSON object with 'machine_id' and 'readings'."}), 400

    machine_id = payload.get("machine_id")
    if not isinstance(machine_id, str) or not machine_id:
        return jsonify({"error": "'machine_id' must be a non-empty string."}), 400

    try:
        result = predict_latest(get_bundle(), machine_id, payload.get("readings"))
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    except FileNotFoundError as error:
        return jsonify({"error": str(error)}), 503
    return jsonify(result)


if __name__ == "__main__":
    # 127.0.0.1 = only this computer can connect. Debug mode stays OFF,
    # because debug mode can allow remote code execution.
    app.run(host="127.0.0.1", port=5000, debug=False)

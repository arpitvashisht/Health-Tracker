from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import mimetypes
from urllib.parse import unquote, urlparse

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT / "frontend"
MODEL_PATH = ROOT / "models" / "calorie_predictor.joblib"

HOST = "127.0.0.1"
PORT = 8000

MODEL_CACHE = None


def load_saved_model() -> dict:
    global MODEL_CACHE
    if MODEL_CACHE is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError("Run python src/train_calorie_model.py before starting predictions.")
        MODEL_CACHE = joblib.load(MODEL_PATH)
    return MODEL_CACHE


def as_float(payload: dict, key: str, required: bool = True) -> float | None:
    value = payload.get(key)
    if value in ("", None):
        if required:
            raise ValueError(f"{key} is required")
        return None

    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a number") from exc


def build_prediction_row(payload: dict, saved: dict) -> pd.DataFrame:
    features = saved["features"]
    feature_defaults = saved.get("feature_defaults", {})
    row = {feature: feature_defaults.get(feature, 0) for feature in features}

    day_of_week = int(as_float(payload, "dayOfWeek"))
    row.update(
        {
            "Id": int(as_float(payload, "id")),
            "StepTotal": as_float(payload, "steps"),
            "TotalIntensity": as_float(payload, "totalIntensity"),
            "AverageIntensity": as_float(payload, "averageIntensity"),
            "Hour": int(as_float(payload, "hour")),
            "DayOfWeek": day_of_week,
            "IsWeekend": int(day_of_week in [5, 6]),
        }
    )

    heart_rate_mean = as_float(payload, "heartRateMean", required=False)
    if heart_rate_mean is not None:
        row["HeartRateMean"] = heart_rate_mean
        row["HeartRateMin"] = heart_rate_mean
        row["HeartRateMax"] = heart_rate_mean
        row["HeartRateCount"] = max(row.get("HeartRateCount", 0), 1)
        row["HasHeartRate"] = 1

    sleep_minutes = as_float(payload, "sleepMinutes", required=False)
    if sleep_minutes is not None:
        row["SleepMinuteCount"] = sleep_minutes
        row["SleepValueMean"] = max(row.get("SleepValueMean", 1), 1)
        row["SleepValueMax"] = max(row.get("SleepValueMax", 1), 1)
        row["HasSleepLog"] = int(sleep_minutes > 0)

    weight_kg = as_float(payload, "weightKg", required=False)
    bmi = as_float(payload, "bmi", required=False)
    if weight_kg is not None:
        row["WeightKg"] = weight_kg
        row["WeightLogCount"] = max(row.get("WeightLogCount", 0), 1)
        row["HasWeightLog"] = 1
    if bmi is not None:
        row["BMI"] = bmi
        row["WeightLogCount"] = max(row.get("WeightLogCount", 0), 1)
        row["HasWeightLog"] = 1

    return pd.DataFrame([row], columns=features)


class FrontendHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return

    def send_json(self, status_code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)

        if parsed.path == "/api/status":
            try:
                saved = load_saved_model()
                self.send_json(
                    200,
                    {
                        "modelReady": True,
                        "featureCount": len(saved["features"]),
                    },
                )
            except Exception as exc:
                self.send_json(503, {"modelReady": False, "error": str(exc)})
            return

        requested_path = "/index.html" if parsed.path == "/" else parsed.path
        target = (FRONTEND_DIR / unquote(requested_path).lstrip("/")).resolve()

        if not str(target).startswith(str(FRONTEND_DIR.resolve())) or not target.is_file():
            self.send_error(404)
            return

        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        content = target.read_bytes()

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/predict":
            self.send_error(404)
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
            saved = load_saved_model()
            row = build_prediction_row(payload, saved)
            prediction = saved["model"].predict(row)[0]
            self.send_json(200, {"predictedCalories": round(float(prediction), 1)})
        except Exception as exc:
            self.send_json(400, {"error": str(exc)})


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), FrontendHandler)
    print(f"Health Tracker AI running at http://{HOST}:{PORT}")
    print("Press Ctrl+C to stop the server.")
    server.serve_forever()


if __name__ == "__main__":
    main()

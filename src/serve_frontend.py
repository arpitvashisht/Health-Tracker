from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import mimetypes
import os
import urllib.error
import urllib.request
from urllib.parse import unquote, urlparse

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT / "frontend"
MODEL_PATH = ROOT / "models" / "calorie_predictor.joblib"
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.2")

HOST = "127.0.0.1"
PORT = 8000

MODEL_CACHE = None

HEALTH_ASSISTANT_INSTRUCTIONS = (
    "You are a friendly AI health assistant for a student health tracker app. "
    "Use the provided health score, habits, and suggestions to respond conversationally. "
    "Keep the advice practical, supportive, and non-medical. Do not diagnose disease. "
    "Recommend seeing a qualified professional for severe symptoms or medical concerns."
)


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


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def calculate_health_score(steps: float, sleep_hours: float, water_liters: float) -> int:
    step_score = clamp(steps / 10000, 0, 1) * 40

    if 7 <= sleep_hours <= 9:
        sleep_score = 30
    elif sleep_hours < 7:
        sleep_score = clamp(sleep_hours / 7, 0, 1) * 30
    else:
        sleep_score = clamp(1 - ((sleep_hours - 9) / 5), 0.4, 1) * 30

    water_score = clamp(water_liters / 2.5, 0, 1) * 30
    if water_liters > 4.5:
        water_score = max(18, water_score - 6)

    return int(round(step_score + sleep_score + water_score))


def analyze_health_patterns(steps: float, sleep_hours: float, water_liters: float, history: list[dict]) -> list[str]:
    habits = []

    if steps < 5000:
        habits.append("Low daily movement")
    elif steps < 7500:
        habits.append("Steps are below the active range")

    if sleep_hours < 6:
        habits.append("Very low sleep")
    elif sleep_hours < 7:
        habits.append("Sleep is slightly short")
    elif sleep_hours > 9.5:
        habits.append("Sleep is above the usual healthy range")

    if water_liters < 1.5:
        habits.append("Low hydration")
    elif water_liters > 4.5:
        habits.append("Water intake is unusually high")

    recent = history[-3:] if isinstance(history, list) else []
    if len(recent) >= 3:
        low_step_days = sum(1 for day in recent if float(day.get("steps", 0)) < 5000)
        low_sleep_days = sum(1 for day in recent if float(day.get("sleepHours", 0)) < 7)
        low_water_days = sum(1 for day in recent if float(day.get("waterLiters", 0)) < 1.5)

        if low_step_days >= 2:
            habits.append("Repeated low movement pattern")
        if low_sleep_days >= 2:
            habits.append("Repeated short sleep pattern")
        if low_water_days >= 2:
            habits.append("Repeated low hydration pattern")

    if not habits:
        habits.append("No major unhealthy habits detected")

    return habits


def build_suggestions(steps: float, sleep_hours: float, water_liters: float, habits: list[str]) -> list[str]:
    suggestions = []

    if steps < 7500:
        suggestions.append("Add a 10-15 minute walk after a meal to raise your step count without changing your whole day.")
    else:
        suggestions.append("Keep your movement steady and add light stretching to support recovery.")

    if sleep_hours < 7:
        suggestions.append("Set a fixed sleep window tonight and reduce screens for the last 30 minutes before bed.")
    elif sleep_hours > 9.5:
        suggestions.append("Track whether long sleep is linked with low energy, stress, or irregular bedtime.")
    else:
        suggestions.append("Your sleep duration is in a healthy range; protect that routine.")

    if water_liters < 2:
        suggestions.append("Drink one glass of water after waking and one glass before each main meal.")
    elif water_liters > 4.5:
        suggestions.append("Avoid forcing extra water; balance hydration with thirst, meals, and activity level.")
    else:
        suggestions.append("Hydration looks reasonable; keep it consistent through the day.")

    if "No major unhealthy habits detected" in habits:
        suggestions.append("Choose one small goal for tomorrow so the routine stays easy to repeat.")

    return suggestions


def fallback_health_reply(score: int, habits: list[str], suggestions: list[str], message: str) -> str:
    habit_text = ", ".join(habits).lower()
    suggestion_text = " ".join(suggestions[:2])
    if message:
        return (
            f"Based on your latest data, your health score is {score}/100. "
            f"I noticed {habit_text}. {suggestion_text} "
            "A small, repeatable improvement is better than trying to fix everything in one day."
        )

    return (
        f"Your health score today is {score}/100. "
        f"The main pattern I see is {habit_text}. {suggestion_text}"
    )


def extract_response_text(response_payload: dict) -> str:
    if response_payload.get("output_text"):
        return str(response_payload["output_text"]).strip()

    output = response_payload.get("output", [])
    for item in output:
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"} and content.get("text"):
                return str(content["text"]).strip()

    return ""


def call_openai_health_assistant(
    inputs: dict,
    score: int,
    habits: list[str],
    suggestions: list[str],
    message: str,
    history: list[dict],
) -> str | None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    prompt = {
        "daily_health_data": inputs,
        "health_score": score,
        "detected_habits": habits,
        "suggestions": suggestions,
        "recent_history": history[-7:] if isinstance(history, list) else [],
        "user_message": message or "Analyze my daily health data and give me personalized guidance.",
    }
    request_body = {
        "model": OPENAI_MODEL,
        "instructions": HEALTH_ASSISTANT_INSTRUCTIONS,
        "input": json.dumps(prompt),
        "max_output_tokens": 260,
    }
    request = urllib.request.Request(
        OPENAI_RESPONSES_URL,
        data=json.dumps(request_body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
            return extract_response_text(payload) or None
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
        return None


def build_health_chat_response(payload: dict) -> dict:
    steps = as_float(payload, "steps")
    sleep_hours = as_float(payload, "sleepHours")
    water_liters = as_float(payload, "waterLiters")
    message = str(payload.get("message", "")).strip()
    history = payload.get("history", [])

    if steps < 0 or sleep_hours < 0 or water_liters < 0:
        raise ValueError("Health values cannot be negative")
    if sleep_hours > 24:
        raise ValueError("sleepHours cannot be more than 24")

    inputs = {
        "steps": int(round(steps)),
        "sleepHours": round(sleep_hours, 1),
        "waterLiters": round(water_liters, 1),
    }
    score = calculate_health_score(steps, sleep_hours, water_liters)
    habits = analyze_health_patterns(steps, sleep_hours, water_liters, history)
    suggestions = build_suggestions(steps, sleep_hours, water_liters, habits)
    ai_reply = call_openai_health_assistant(inputs, score, habits, suggestions, message, history)

    return {
        "inputs": inputs,
        "healthScore": score,
        "habits": habits,
        "suggestions": suggestions,
        "reply": ai_reply or fallback_health_reply(score, habits, suggestions, message),
        "aiPowered": bool(ai_reply),
    }


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
            model_ready = MODEL_PATH.exists()
            try:
                saved = load_saved_model() if model_ready else {"features": []}
                self.send_json(
                    200,
                    {
                        "modelReady": model_ready,
                        "featureCount": len(saved["features"]),
                        "aiReady": bool(os.environ.get("OPENAI_API_KEY")),
                        "openaiModel": OPENAI_MODEL,
                    },
                )
            except Exception as exc:
                self.send_json(
                    200,
                    {
                        "modelReady": False,
                        "featureCount": 0,
                        "aiReady": bool(os.environ.get("OPENAI_API_KEY")),
                        "openaiModel": OPENAI_MODEL,
                        "error": str(exc),
                    },
                )
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
        path = urlparse(self.path).path
        if path not in {"/api/predict", "/api/health-chat"}:
            self.send_error(404)
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))

            if path == "/api/health-chat":
                self.send_json(200, build_health_chat_response(payload))
                return

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

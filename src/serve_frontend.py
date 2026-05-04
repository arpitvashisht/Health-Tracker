from __future__ import annotations

from datetime import date, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import mimetypes
import os
import sqlite3
import urllib.error
import urllib.request
from urllib.parse import parse_qs, unquote, urlparse

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT / "frontend"
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "health_tracker.db"
MODEL_PATH = ROOT / "models" / "calorie_predictor.joblib"
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.2")

HOST = "127.0.0.1"
PORT = 8000

MODEL_CACHE = None

HEALTH_ASSISTANT_INSTRUCTIONS = (
    "You are a friendly AI health assistant for a smart health tracker. "
    "Use the user's goals, daily logs, meals, reminders, score trend, habits, and recent chat memory. "
    "Respond conversationally with practical, personalized, non-medical guidance. "
    "Do not diagnose disease. Recommend a qualified professional for severe symptoms or medical concerns."
)

WEEKLY_REPORT_INSTRUCTIONS = (
    "You generate concise weekly wellness reports for a health tracking app. "
    "Use the supplied seven-day data, goals, streaks, meals, and habits. "
    "Return supportive, practical advice in short paragraphs. Do not provide medical diagnosis."
)


def connect_db() -> sqlite3.Connection:
    DATA_DIR.mkdir(exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def init_db() -> None:
    with connect_db() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS goals (
                user_id INTEGER PRIMARY KEY,
                steps_goal INTEGER NOT NULL DEFAULT 8000,
                sleep_goal REAL NOT NULL DEFAULT 7.5,
                water_goal REAL NOT NULL DEFAULT 2.5,
                calorie_goal INTEGER NOT NULL DEFAULT 2200,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS daily_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                log_date TEXT NOT NULL,
                steps INTEGER NOT NULL,
                sleep_hours REAL NOT NULL,
                water_liters REAL NOT NULL,
                mood INTEGER NOT NULL,
                stress INTEGER NOT NULL,
                energy INTEGER NOT NULL,
                health_score INTEGER NOT NULL,
                habits_json TEXT NOT NULL,
                suggestions_json TEXT NOT NULL,
                notes TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, log_date),
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS meals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                meal_date TEXT NOT NULL,
                name TEXT NOT NULL,
                calories INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS reminders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                reminder_time TEXT NOT NULL,
                category TEXT NOT NULL,
                enabled INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                message TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS weekly_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                week_start TEXT NOT NULL,
                week_end TEXT NOT NULL,
                report TEXT NOT NULL,
                ai_powered INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            """
        )

        user_count = connection.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        if user_count == 0:
            cursor = connection.execute("INSERT INTO users (name) VALUES (?)", ("Guest",))
            connection.execute("INSERT INTO goals (user_id) VALUES (?)", (cursor.lastrowid,))


def row_to_dict(row: sqlite3.Row | None) -> dict | None:
    return dict(row) if row else None


def rows_to_dicts(rows: list[sqlite3.Row]) -> list[dict]:
    return [dict(row) for row in rows]


def parse_body(handler: BaseHTTPRequestHandler) -> dict:
    content_length = int(handler.headers.get("Content-Length", "0"))
    if content_length == 0:
        return {}
    return json.loads(handler.rfile.read(content_length).decode("utf-8"))


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


def as_int(payload: dict, key: str, required: bool = True) -> int | None:
    value = as_float(payload, key, required)
    return None if value is None else int(round(value))


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def today_iso() -> str:
    return date.today().isoformat()


def validate_date(value: str | None) -> str:
    if not value:
        return today_iso()
    datetime.strptime(value, "%Y-%m-%d")
    return value


def load_saved_model() -> dict:
    global MODEL_CACHE
    if MODEL_CACHE is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError("Run python src/train_calorie_model.py before starting predictions.")
        MODEL_CACHE = joblib.load(MODEL_PATH)
    return MODEL_CACHE


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


def get_user_id(payload: dict | None = None, query: dict | None = None) -> int:
    if payload and payload.get("userId"):
        return int(payload["userId"])
    if query and query.get("userId"):
        return int(query["userId"][0])
    with connect_db() as connection:
        row = connection.execute("SELECT id FROM users ORDER BY id LIMIT 1").fetchone()
        if row is None:
            init_db()
            row = connection.execute("SELECT id FROM users ORDER BY id LIMIT 1").fetchone()
        return int(row["id"])


def get_goals(connection: sqlite3.Connection, user_id: int) -> dict:
    row = connection.execute("SELECT * FROM goals WHERE user_id = ?", (user_id,)).fetchone()
    if row is None:
        connection.execute("INSERT INTO goals (user_id) VALUES (?)", (user_id,))
        row = connection.execute("SELECT * FROM goals WHERE user_id = ?", (user_id,)).fetchone()
    goals = dict(row)
    return {
        "stepsGoal": goals["steps_goal"],
        "sleepGoal": goals["sleep_goal"],
        "waterGoal": goals["water_goal"],
        "calorieGoal": goals["calorie_goal"],
    }


def calculate_health_score(
    steps: float,
    sleep_hours: float,
    water_liters: float,
    mood: float = 3,
    stress: float = 3,
    energy: float = 3,
    goals: dict | None = None,
) -> int:
    goals = goals or {"stepsGoal": 8000, "sleepGoal": 7.5, "waterGoal": 2.5}
    step_score = clamp(steps / max(float(goals["stepsGoal"]), 1), 0, 1) * 30

    sleep_goal = max(float(goals["sleepGoal"]), 1)
    sleep_delta = abs(sleep_hours - sleep_goal)
    sleep_score = clamp(1 - (sleep_delta / 4), 0, 1) * 24

    water_score = clamp(water_liters / max(float(goals["waterGoal"]), 0.1), 0, 1) * 22
    if water_liters > float(goals["waterGoal"]) * 1.9:
        water_score = max(13, water_score - 5)

    mood_score = clamp(mood / 5, 0, 1) * 8
    stress_score = clamp((6 - stress) / 5, 0, 1) * 8
    energy_score = clamp(energy / 5, 0, 1) * 8
    return int(round(step_score + sleep_score + water_score + mood_score + stress_score + energy_score))


def analyze_health_patterns(
    steps: float,
    sleep_hours: float,
    water_liters: float,
    mood: float,
    stress: float,
    energy: float,
    history: list[dict],
    goals: dict,
) -> list[str]:
    habits = []

    if steps < float(goals["stepsGoal"]) * 0.6:
        habits.append("Low daily movement")
    elif steps < float(goals["stepsGoal"]):
        habits.append("Steps are below your goal")

    if sleep_hours < float(goals["sleepGoal"]) - 1:
        habits.append("Sleep is too short")
    elif sleep_hours > float(goals["sleepGoal"]) + 2:
        habits.append("Sleep is above your usual target")

    if water_liters < float(goals["waterGoal"]) * 0.75:
        habits.append("Low hydration")
    elif water_liters > float(goals["waterGoal"]) * 1.9:
        habits.append("Water intake is unusually high")

    if mood <= 2:
        habits.append("Low mood")
    if stress >= 4:
        habits.append("High stress")
    if energy <= 2:
        habits.append("Low energy")

    recent = history[-3:] if isinstance(history, list) else []
    if len(recent) >= 3:
        low_step_days = sum(1 for day in recent if float(day.get("steps", 0)) < float(goals["stepsGoal"]))
        low_sleep_days = sum(1 for day in recent if float(day.get("sleepHours", 0)) < float(goals["sleepGoal"]))
        low_water_days = sum(1 for day in recent if float(day.get("waterLiters", 0)) < float(goals["waterGoal"]))
        high_stress_days = sum(1 for day in recent if float(day.get("stress", 0)) >= 4)

        if low_step_days >= 2:
            habits.append("Repeated movement gap")
        if low_sleep_days >= 2:
            habits.append("Repeated short sleep pattern")
        if low_water_days >= 2:
            habits.append("Repeated low hydration pattern")
        if high_stress_days >= 2:
            habits.append("Repeated high stress pattern")

    if not habits:
        habits.append("No major unhealthy habits detected")

    return habits


def build_suggestions(
    steps: float,
    sleep_hours: float,
    water_liters: float,
    mood: float,
    stress: float,
    energy: float,
    goals: dict,
    habits: list[str],
) -> list[str]:
    suggestions = []

    if steps < float(goals["stepsGoal"]):
        gap = max(500, int(float(goals["stepsGoal"]) - steps))
        suggestions.append(f"Add about {gap} steps with a short walk or two small movement breaks.")
    else:
        suggestions.append("Your movement goal is met; keep the routine steady and add light stretching.")

    if sleep_hours < float(goals["sleepGoal"]):
        suggestions.append("Move bedtime earlier by 20-30 minutes and reduce screens before sleep.")
    elif sleep_hours > float(goals["sleepGoal"]) + 2:
        suggestions.append("Keep wake-up time consistent and watch whether long sleep matches low energy.")
    else:
        suggestions.append("Sleep duration is close to your goal; protect that routine.")

    if water_liters < float(goals["waterGoal"]):
        suggestions.append("Place a water bottle nearby and drink one glass after waking and before meals.")
    else:
        suggestions.append("Hydration is on track; spread it evenly through the day.")

    if stress >= 4:
        suggestions.append("Use a 3-minute breathing reset or a quiet walk to lower stress today.")
    if mood <= 2 or energy <= 2:
        suggestions.append("Choose one low-effort win, such as sunlight, a short walk, or a simple meal.")
    if "No major unhealthy habits detected" in habits:
        suggestions.append("Pick one tiny goal for tomorrow so the healthy pattern continues.")

    return suggestions[:5]


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


def call_openai(instructions: str, prompt: dict, max_output_tokens: int = 320) -> str | None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    request_body = {
        "model": OPENAI_MODEL,
        "instructions": instructions,
        "input": json.dumps(prompt),
        "max_output_tokens": max_output_tokens,
    }
    request = urllib.request.Request(
        OPENAI_RESPONSES_URL,
        data=json.dumps(request_body).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
            return extract_response_text(payload) or None
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
        return None


def fallback_health_reply(score: int, habits: list[str], suggestions: list[str], message: str, trend: dict) -> str:
    habit_text = ", ".join(habits).lower()
    suggestion_text = " ".join(suggestions[:2])
    trend_text = ""
    if trend.get("scoreDelta") is not None:
        direction = "up" if trend["scoreDelta"] >= 0 else "down"
        trend_text = f" Your score is {direction} {abs(trend['scoreDelta'])} points from your previous log."

    if message:
        return (
            f"Your current health score is {score}/100.{trend_text} "
            f"I noticed {habit_text}. {suggestion_text} "
            "Keep the next action small enough that you can actually repeat it tomorrow."
        )

    return f"Your health score today is {score}/100.{trend_text} The main pattern I see is {habit_text}. {suggestion_text}"


def get_logs(connection: sqlite3.Connection, user_id: int, limit: int = 30) -> list[dict]:
    rows = connection.execute(
        """
        SELECT * FROM daily_logs
        WHERE user_id = ?
        ORDER BY log_date DESC
        LIMIT ?
        """,
        (user_id, limit),
    ).fetchall()
    logs = []
    for row in rows:
        item = dict(row)
        item["habits"] = json.loads(item.pop("habits_json"))
        item["suggestions"] = json.loads(item.pop("suggestions_json"))
        item["stepsGoalMet"] = False
        logs.append(item)
    return list(reversed(logs))


def get_meals(connection: sqlite3.Connection, user_id: int, limit: int = 50) -> list[dict]:
    rows = connection.execute(
        "SELECT * FROM meals WHERE user_id = ? ORDER BY meal_date DESC, id DESC LIMIT ?",
        (user_id, limit),
    ).fetchall()
    return rows_to_dicts(rows)


def get_reminders(connection: sqlite3.Connection, user_id: int) -> list[dict]:
    rows = connection.execute(
        "SELECT * FROM reminders WHERE user_id = ? ORDER BY reminder_time, id",
        (user_id,),
    ).fetchall()
    reminders = rows_to_dicts(rows)
    for reminder in reminders:
        reminder["enabled"] = bool(reminder["enabled"])
    return reminders


def get_chat_messages(connection: sqlite3.Connection, user_id: int, limit: int = 12) -> list[dict]:
    rows = connection.execute(
        "SELECT role, message, created_at FROM chat_messages WHERE user_id = ? ORDER BY id DESC LIMIT ?",
        (user_id, limit),
    ).fetchall()
    return list(reversed(rows_to_dicts(rows)))


def get_latest_report(connection: sqlite3.Connection, user_id: int) -> dict | None:
    row = connection.execute(
        "SELECT * FROM weekly_reports WHERE user_id = ? ORDER BY id DESC LIMIT 1",
        (user_id,),
    ).fetchone()
    report = row_to_dict(row)
    if report:
        report["aiPowered"] = bool(report.pop("ai_powered"))
    return report


def build_trends(logs: list[dict]) -> dict:
    if not logs:
        return {
            "averages": {"steps": 0, "sleepHours": 0, "waterLiters": 0, "healthScore": 0},
            "scoreDelta": None,
            "series": [],
        }

    recent = logs[-7:]
    count = len(recent)
    averages = {
        "steps": round(sum(log["steps"] for log in recent) / count),
        "sleepHours": round(sum(log["sleep_hours"] for log in recent) / count, 1),
        "waterLiters": round(sum(log["water_liters"] for log in recent) / count, 1),
        "healthScore": round(sum(log["health_score"] for log in recent) / count),
    }
    score_delta = None
    if len(logs) >= 2:
        score_delta = logs[-1]["health_score"] - logs[-2]["health_score"]

    return {
        "averages": averages,
        "scoreDelta": score_delta,
        "series": [
            {
                "date": log["log_date"],
                "steps": log["steps"],
                "sleepHours": log["sleep_hours"],
                "waterLiters": log["water_liters"],
                "healthScore": log["health_score"],
                "mood": log["mood"],
                "stress": log["stress"],
                "energy": log["energy"],
            }
            for log in recent
        ],
    }


def calculate_streaks(logs: list[dict], goals: dict) -> dict:
    sorted_logs = sorted(logs, key=lambda item: item["log_date"], reverse=True)
    streaks = {"steps": 0, "sleep": 0, "water": 0, "overall": 0}

    for log in sorted_logs:
        if log["steps"] >= goals["stepsGoal"]:
            streaks["steps"] += 1
        else:
            break

    for log in sorted_logs:
        if log["sleep_hours"] >= goals["sleepGoal"]:
            streaks["sleep"] += 1
        else:
            break

    for log in sorted_logs:
        if log["water_liters"] >= goals["waterGoal"]:
            streaks["water"] += 1
        else:
            break

    for log in sorted_logs:
        if log["health_score"] >= 75:
            streaks["overall"] += 1
        else:
            break

    return streaks


def build_app_state(user_id: int) -> dict:
    with connect_db() as connection:
        users = rows_to_dicts(connection.execute("SELECT * FROM users ORDER BY id").fetchall())
        active_user = row_to_dict(connection.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone())
        if active_user is None and users:
            user_id = users[0]["id"]
            active_user = users[0]

        goals = get_goals(connection, user_id)
        logs = get_logs(connection, user_id, 30)
        meals = get_meals(connection, user_id)
        reminders = get_reminders(connection, user_id)
        chat_messages = get_chat_messages(connection, user_id)
        trends = build_trends(logs)
        streaks = calculate_streaks(logs, goals)
        latest_report = get_latest_report(connection, user_id)

    return {
        "users": users,
        "activeUser": active_user,
        "goals": goals,
        "logs": logs,
        "meals": meals,
        "reminders": reminders,
        "chatMessages": chat_messages,
        "trends": trends,
        "streaks": streaks,
        "weeklyReport": latest_report,
        "today": today_iso(),
        "aiReady": bool(os.environ.get("OPENAI_API_KEY")),
        "openaiModel": OPENAI_MODEL,
    }


def save_chat_message(connection: sqlite3.Connection, user_id: int, role: str, message: str) -> None:
    connection.execute(
        "INSERT INTO chat_messages (user_id, role, message) VALUES (?, ?, ?)",
        (user_id, role, message),
    )


def save_daily_log(payload: dict) -> dict:
    user_id = get_user_id(payload=payload)
    log_date = validate_date(payload.get("date"))
    steps = as_int(payload, "steps")
    sleep_hours = as_float(payload, "sleepHours")
    water_liters = as_float(payload, "waterLiters")
    mood = as_int(payload, "mood")
    stress = as_int(payload, "stress")
    energy = as_int(payload, "energy")
    notes = str(payload.get("notes", "")).strip()

    if steps < 0 or sleep_hours < 0 or water_liters < 0:
        raise ValueError("Health values cannot be negative")
    if sleep_hours > 24:
        raise ValueError("sleepHours cannot be more than 24")
    if not (1 <= mood <= 5 and 1 <= stress <= 5 and 1 <= energy <= 5):
        raise ValueError("mood, stress, and energy must be between 1 and 5")

    with connect_db() as connection:
        goals = get_goals(connection, user_id)
        logs = get_logs(connection, user_id, 30)
        history = [
            {
                "steps": log["steps"],
                "sleepHours": log["sleep_hours"],
                "waterLiters": log["water_liters"],
                "mood": log["mood"],
                "stress": log["stress"],
                "energy": log["energy"],
            }
            for log in logs
        ]
        score = calculate_health_score(steps, sleep_hours, water_liters, mood, stress, energy, goals)
        habits = analyze_health_patterns(steps, sleep_hours, water_liters, mood, stress, energy, history, goals)
        suggestions = build_suggestions(steps, sleep_hours, water_liters, mood, stress, energy, goals, habits)

        connection.execute(
            """
            INSERT INTO daily_logs (
                user_id, log_date, steps, sleep_hours, water_liters, mood, stress, energy,
                health_score, habits_json, suggestions_json, notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, log_date) DO UPDATE SET
                steps = excluded.steps,
                sleep_hours = excluded.sleep_hours,
                water_liters = excluded.water_liters,
                mood = excluded.mood,
                stress = excluded.stress,
                energy = excluded.energy,
                health_score = excluded.health_score,
                habits_json = excluded.habits_json,
                suggestions_json = excluded.suggestions_json,
                notes = excluded.notes,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                user_id,
                log_date,
                steps,
                sleep_hours,
                water_liters,
                mood,
                stress,
                energy,
                score,
                json.dumps(habits),
                json.dumps(suggestions),
                notes,
            ),
        )

    state = build_app_state(user_id)
    state["latestAnalysis"] = {
        "inputs": {"steps": steps, "sleepHours": sleep_hours, "waterLiters": water_liters, "mood": mood, "stress": stress, "energy": energy},
        "healthScore": score,
        "habits": habits,
        "suggestions": suggestions,
    }
    return state


def create_user(payload: dict) -> dict:
    name = str(payload.get("name", "")).strip()
    if not name:
        raise ValueError("name is required")
    with connect_db() as connection:
        cursor = connection.execute("INSERT INTO users (name) VALUES (?)", (name,))
        user_id = cursor.lastrowid
        connection.execute("INSERT INTO goals (user_id) VALUES (?)", (user_id,))
    return build_app_state(user_id)


def update_goals(payload: dict) -> dict:
    user_id = get_user_id(payload=payload)
    steps_goal = as_int(payload, "stepsGoal")
    sleep_goal = as_float(payload, "sleepGoal")
    water_goal = as_float(payload, "waterGoal")
    calorie_goal = as_int(payload, "calorieGoal")
    if steps_goal <= 0 or sleep_goal <= 0 or water_goal <= 0 or calorie_goal <= 0:
        raise ValueError("goals must be positive")

    with connect_db() as connection:
        connection.execute(
            """
            INSERT INTO goals (user_id, steps_goal, sleep_goal, water_goal, calorie_goal)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                steps_goal = excluded.steps_goal,
                sleep_goal = excluded.sleep_goal,
                water_goal = excluded.water_goal,
                calorie_goal = excluded.calorie_goal,
                updated_at = CURRENT_TIMESTAMP
            """,
            (user_id, steps_goal, sleep_goal, water_goal, calorie_goal),
        )
    return build_app_state(user_id)


def add_meal(payload: dict) -> dict:
    user_id = get_user_id(payload=payload)
    meal_date = validate_date(payload.get("date"))
    name = str(payload.get("name", "")).strip()
    calories = as_int(payload, "calories")
    if not name:
        raise ValueError("meal name is required")
    if calories < 0:
        raise ValueError("calories cannot be negative")

    with connect_db() as connection:
        connection.execute(
            "INSERT INTO meals (user_id, meal_date, name, calories) VALUES (?, ?, ?, ?)",
            (user_id, meal_date, name, calories),
        )
    return build_app_state(user_id)


def add_reminder(payload: dict) -> dict:
    user_id = get_user_id(payload=payload)
    title = str(payload.get("title", "")).strip()
    reminder_time = str(payload.get("time", "")).strip()
    category = str(payload.get("category", "general")).strip() or "general"
    if not title:
        raise ValueError("reminder title is required")
    if not reminder_time:
        raise ValueError("reminder time is required")

    with connect_db() as connection:
        connection.execute(
            "INSERT INTO reminders (user_id, title, reminder_time, category, enabled) VALUES (?, ?, ?, ?, 1)",
            (user_id, title, reminder_time, category),
        )
    return build_app_state(user_id)


def toggle_reminder(payload: dict) -> dict:
    user_id = get_user_id(payload=payload)
    reminder_id = as_int(payload, "id")
    with connect_db() as connection:
        row = connection.execute(
            "SELECT enabled FROM reminders WHERE id = ? AND user_id = ?",
            (reminder_id, user_id),
        ).fetchone()
        if row is None:
            raise ValueError("reminder not found")
        connection.execute(
            "UPDATE reminders SET enabled = ? WHERE id = ? AND user_id = ?",
            (0 if row["enabled"] else 1, reminder_id, user_id),
        )
    return build_app_state(user_id)


def build_health_chat_response(payload: dict) -> dict:
    user_id = get_user_id(payload=payload)
    message = str(payload.get("message", "")).strip() or "Analyze my health data and recommend my next best action."

    with connect_db() as connection:
        goals = get_goals(connection, user_id)
        logs = get_logs(connection, user_id, 30)
        meals = get_meals(connection, user_id, 20)
        reminders = get_reminders(connection, user_id)
        chat_memory = get_chat_messages(connection, user_id, 10)
        trends = build_trends(logs)
        latest = logs[-1] if logs else None

        if latest:
            score = latest["health_score"]
            habits = latest["habits"]
            suggestions = latest["suggestions"]
        else:
            steps = as_float(payload, "steps", required=False) or 0
            sleep_hours = as_float(payload, "sleepHours", required=False) or 0
            water_liters = as_float(payload, "waterLiters", required=False) or 0
            mood = as_float(payload, "mood", required=False) or 3
            stress = as_float(payload, "stress", required=False) or 3
            energy = as_float(payload, "energy", required=False) or 3
            history = []
            score = calculate_health_score(steps, sleep_hours, water_liters, mood, stress, energy, goals)
            habits = analyze_health_patterns(steps, sleep_hours, water_liters, mood, stress, energy, history, goals)
            suggestions = build_suggestions(steps, sleep_hours, water_liters, mood, stress, energy, goals, habits)

        prompt = {
            "user_message": message,
            "latest_log": latest,
            "goals": goals,
            "recent_logs": logs[-7:],
            "recent_meals": meals[:10],
            "reminders": reminders,
            "trends": trends,
            "chat_memory": chat_memory,
            "health_score": score,
            "detected_habits": habits,
            "suggestions": suggestions,
        }
        ai_reply = call_openai(HEALTH_ASSISTANT_INSTRUCTIONS, prompt, max_output_tokens=320)
        reply = ai_reply or fallback_health_reply(score, habits, suggestions, message, trends)

        save_chat_message(connection, user_id, "user", message)
        save_chat_message(connection, user_id, "assistant", reply)

    state = build_app_state(user_id)
    return {
        "inputs": latest,
        "healthScore": score,
        "habits": habits,
        "suggestions": suggestions,
        "reply": reply,
        "aiPowered": bool(ai_reply),
        "state": state,
    }


def fallback_weekly_report(logs: list[dict], goals: dict, streaks: dict, meals: list[dict]) -> str:
    if not logs:
        return "No daily logs are available yet. Add a few days of steps, sleep, water, mood, stress, and energy to unlock a useful weekly report."

    avg_steps = round(sum(log["steps"] for log in logs) / len(logs))
    avg_sleep = round(sum(log["sleep_hours"] for log in logs) / len(logs), 1)
    avg_water = round(sum(log["water_liters"] for log in logs) / len(logs), 1)
    avg_score = round(sum(log["health_score"] for log in logs) / len(logs))
    total_calories = sum(meal["calories"] for meal in meals)

    focus = []
    if avg_steps < goals["stepsGoal"]:
        focus.append("movement")
    if avg_sleep < goals["sleepGoal"]:
        focus.append("sleep")
    if avg_water < goals["waterGoal"]:
        focus.append("hydration")
    if not focus:
        focus.append("consistency")

    return (
        f"Weekly summary: your average health score was {avg_score}/100, with {avg_steps} steps, "
        f"{avg_sleep} hours of sleep, and {avg_water}L water on average. "
        f"Your current streaks are {streaks['steps']} step-goal days, {streaks['sleep']} sleep-goal days, "
        f"and {streaks['water']} hydration-goal days. "
        f"Meal logging captured {total_calories} calories this week. "
        f"Next week's focus should be {', '.join(focus)}. Choose one small daily action and repeat it for seven days."
    )


def generate_weekly_report(payload: dict) -> dict:
    user_id = get_user_id(payload=payload)
    end_date = datetime.strptime(validate_date(payload.get("date")), "%Y-%m-%d").date()
    start_date = end_date - timedelta(days=6)

    with connect_db() as connection:
        goals = get_goals(connection, user_id)
        logs = get_logs(connection, user_id, 30)
        week_logs = [log for log in logs if start_date.isoformat() <= log["log_date"] <= end_date.isoformat()]
        meals = [
            meal
            for meal in get_meals(connection, user_id, 80)
            if start_date.isoformat() <= meal["meal_date"] <= end_date.isoformat()
        ]
        streaks = calculate_streaks(logs, goals)
        prompt = {
            "week_start": start_date.isoformat(),
            "week_end": end_date.isoformat(),
            "goals": goals,
            "logs": week_logs,
            "meals": meals,
            "streaks": streaks,
        }
        ai_report = call_openai(WEEKLY_REPORT_INSTRUCTIONS, prompt, max_output_tokens=420)
        report = ai_report or fallback_weekly_report(week_logs, goals, streaks, meals)
        connection.execute(
            "INSERT INTO weekly_reports (user_id, week_start, week_end, report, ai_powered) VALUES (?, ?, ?, ?, ?)",
            (user_id, start_date.isoformat(), end_date.isoformat(), report, int(bool(ai_report))),
        )

    return build_app_state(user_id)


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

    def serve_static(self, path: str) -> None:
        requested_path = "/index.html" if path == "/" else path
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

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)

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

        if parsed.path == "/api/app-state":
            self.send_json(200, build_app_state(get_user_id(query=query)))
            return

        self.serve_static(parsed.path)

    def do_POST(self) -> None:
        path = urlparse(self.path).path

        try:
            payload = parse_body(self)
            if path == "/api/predict":
                saved = load_saved_model()
                row = build_prediction_row(payload, saved)
                prediction = saved["model"].predict(row)[0]
                self.send_json(200, {"predictedCalories": round(float(prediction), 1)})
                return

            routes = {
                "/api/users": create_user,
                "/api/goals": update_goals,
                "/api/daily-log": save_daily_log,
                "/api/meals": add_meal,
                "/api/reminders": add_reminder,
                "/api/reminders/toggle": toggle_reminder,
                "/api/health-chat": build_health_chat_response,
                "/api/weekly-report": generate_weekly_report,
            }

            handler = routes.get(path)
            if handler is None:
                self.send_error(404)
                return

            self.send_json(200, handler(payload))
        except Exception as exc:
            self.send_json(400, {"error": str(exc)})


def main() -> None:
    init_db()
    server = ThreadingHTTPServer((HOST, PORT), FrontendHandler)
    print(f"AI Smart Health Tracker running at http://{HOST}:{PORT}")
    print("Press Ctrl+C to stop the server.")
    server.serve_forever()


if __name__ == "__main__":
    main()

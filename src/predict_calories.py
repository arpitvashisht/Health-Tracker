import argparse
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "calorie_predictor.joblib"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict hourly calories burned.")
    parser.add_argument("--id", type=int, required=True, help="User id from the Fitbit dataset.")
    parser.add_argument("--steps", type=int, required=True, help="Steps taken in the hour.")
    parser.add_argument("--total-intensity", type=int, required=True, help="Total activity intensity for the hour.")
    parser.add_argument("--average-intensity", type=float, required=True, help="Average activity intensity for the hour.")
    parser.add_argument("--hour", type=int, required=True, choices=range(24), help="Hour of day, 0 to 23.")
    parser.add_argument("--day-of-week", type=int, required=True, choices=range(7), help="Monday=0 through Sunday=6.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not MODEL_PATH.exists():
        raise SystemExit("Model not found. Run: python src/train_calorie_model.py")

    saved = joblib.load(MODEL_PATH)
    model = saved["model"]

    row = pd.DataFrame(
        [
            {
                "Id": args.id,
                "StepTotal": args.steps,
                "TotalIntensity": args.total_intensity,
                "AverageIntensity": args.average_intensity,
                "Hour": args.hour,
                "DayOfWeek": args.day_of_week,
                "IsWeekend": int(args.day_of_week in [5, 6]),
            }
        ]
    )

    predicted_calories = model.predict(row)[0]
    print(f"Predicted calories burned: {predicted_calories:.1f}")


if __name__ == "__main__":
    main()

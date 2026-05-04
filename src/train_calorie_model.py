from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "calorie_predictor.joblib"


def load_training_data() -> pd.DataFrame:
    calories = pd.read_csv(ROOT / "hourlyCalories_merged.csv")
    steps = pd.read_csv(ROOT / "hourlySteps_merged.csv")
    intensities = pd.read_csv(ROOT / "hourlyIntensities_merged.csv")

    data = calories.merge(steps, on=["Id", "ActivityHour"])
    data = data.merge(intensities, on=["Id", "ActivityHour"])

    data["ActivityHour"] = pd.to_datetime(data["ActivityHour"], format="%m/%d/%Y %I:%M:%S %p")
    data["Hour"] = data["ActivityHour"].dt.hour
    data["DayOfWeek"] = data["ActivityHour"].dt.dayofweek
    data["IsWeekend"] = data["DayOfWeek"].isin([5, 6]).astype(int)

    return data


def build_model() -> Pipeline:
    features = ["Id", "StepTotal", "TotalIntensity", "AverageIntensity", "Hour", "DayOfWeek", "IsWeekend"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("user", OneHotEncoder(handle_unknown="ignore"), ["Id"]),
            ("numeric", "passthrough", ["StepTotal", "TotalIntensity", "AverageIntensity", "Hour", "DayOfWeek", "IsWeekend"]),
        ]
    )

    regressor = RandomForestRegressor(
        n_estimators=150,
        random_state=42,
        min_samples_leaf=2,
        n_jobs=1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", regressor),
        ]
    ), features


def main() -> None:
    data = load_training_data()
    model, features = build_model()

    x = data[features]
    y = data["Calories"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
    )

    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": model, "features": features}, MODEL_PATH)

    print(f"Rows used: {len(data)}")
    print(f"Mean absolute error: {mae:.2f} calories")
    print(f"R2 score: {r2:.3f}")
    print(f"Saved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()

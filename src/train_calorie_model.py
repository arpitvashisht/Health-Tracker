from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from preprocess_health_data import preprocess_hourly_health_data


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "calorie_predictor.joblib"


def load_training_data() -> pd.DataFrame:
    return preprocess_hourly_health_data(save=True)


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

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DATA_PATH = PROCESSED_DIR / "hourly_health_features.csv"
REPORT_PATH = PROCESSED_DIR / "preprocessing_report.txt"

REQUIRED_COLUMNS = [
    "Id",
    "ActivityHour",
    "Calories",
    "StepTotal",
    "TotalIntensity",
    "AverageIntensity",
]
NUMERIC_COLUMNS = ["Calories", "StepTotal", "TotalIntensity", "AverageIntensity"]


def load_raw_hourly_data() -> pd.DataFrame:
    calories = pd.read_csv(ROOT / "hourlyCalories_merged.csv")
    steps = pd.read_csv(ROOT / "hourlySteps_merged.csv")
    intensities = pd.read_csv(ROOT / "hourlyIntensities_merged.csv")

    data = calories.merge(steps, on=["Id", "ActivityHour"], how="inner")
    return data.merge(intensities, on=["Id", "ActivityHour"], how="inner")


def cap_iqr_outliers(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    capped_counts = {}

    for column in NUMERIC_COLUMNS:
        q1 = data[column].quantile(0.25)
        q3 = data[column].quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            capped_counts[column] = 0
            continue

        lower_bound = max(0, q1 - 1.5 * iqr)
        upper_bound = q3 + 1.5 * iqr
        outlier_mask = (data[column] < lower_bound) | (data[column] > upper_bound)

        capped_counts[column] = int(outlier_mask.sum())
        data[column] = data[column].clip(lower=lower_bound, upper=upper_bound)

    return data, capped_counts


def preprocess_hourly_health_data(save: bool = True) -> pd.DataFrame:
    data = load_raw_hourly_data()
    raw_rows = len(data)

    data = data.drop_duplicates(subset=["Id", "ActivityHour"]).copy()
    duplicate_rows_removed = raw_rows - len(data)

    data["ActivityHour"] = pd.to_datetime(
        data["ActivityHour"],
        format="%m/%d/%Y %I:%M:%S %p",
        errors="coerce",
    )

    data["Id"] = pd.to_numeric(data["Id"], errors="coerce")
    for column in NUMERIC_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    missing_before = data[REQUIRED_COLUMNS].isna().sum()
    data = data.dropna(subset=REQUIRED_COLUMNS).copy()

    invalid_values = (
        (data["Calories"] < 0)
        | (data["StepTotal"] < 0)
        | (data["TotalIntensity"] < 0)
        | (data["AverageIntensity"] < 0)
    )
    invalid_rows_removed = int(invalid_values.sum())
    data = data.loc[~invalid_values].copy()

    data["Id"] = data["Id"].astype("int64")
    data["Hour"] = data["ActivityHour"].dt.hour
    data["DayOfWeek"] = data["ActivityHour"].dt.dayofweek
    data["IsWeekend"] = data["DayOfWeek"].isin([5, 6]).astype(int)

    data, capped_counts = cap_iqr_outliers(data)
    data = data.sort_values(["Id", "ActivityHour"]).reset_index(drop=True)

    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        data.to_csv(PROCESSED_DATA_PATH, index=False)

        report = [
            "Health dataset preprocessing report",
            f"Raw merged rows: {raw_rows}",
            f"Duplicate rows removed: {duplicate_rows_removed}",
            f"Rows with missing required values removed: {int(missing_before.sum())}",
            f"Invalid negative rows removed: {invalid_rows_removed}",
            f"Final clean rows: {len(data)}",
            "",
            "Missing values before cleanup:",
        ]
        report.extend(f"- {column}: {int(count)}" for column, count in missing_before.items())
        report.append("")
        report.append("Outliers capped with IQR bounds:")
        report.extend(f"- {column}: {count}" for column, count in capped_counts.items())

        REPORT_PATH.write_text("\n".join(report), encoding="utf-8")

    return data


def main() -> None:
    data = preprocess_hourly_health_data(save=True)
    print(f"Clean rows saved: {len(data)}")
    print(f"Processed dataset: {PROCESSED_DATA_PATH}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()

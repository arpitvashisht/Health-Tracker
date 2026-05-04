from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DATA_PATH = PROCESSED_DIR / "hourly_health_features.csv"
REPORT_PATH = PROCESSED_DIR / "preprocessing_report.txt"

DATETIME_FORMAT = "%m/%d/%Y %I:%M:%S %p"

BASE_COLUMNS = ["Id", "ActivityHour", "Calories", "StepTotal", "TotalIntensity", "AverageIntensity"]
BASE_NUMERIC_COLUMNS = ["Calories", "StepTotal", "TotalIntensity", "AverageIntensity"]

ALL_FEATURE_COLUMNS = [
    "Id",
    "StepTotal",
    "TotalIntensity",
    "AverageIntensity",
    "Hour",
    "DayOfWeek",
    "IsWeekend",
    "MinuteCaloriesSum",
    "MinuteCaloriesMean",
    "MinuteCaloriesMax",
    "MinuteCaloriesStd",
    "MinuteCaloriesCount",
    "MinuteStepsSum",
    "MinuteStepsMean",
    "MinuteStepsMax",
    "MinuteStepsStd",
    "MinuteStepsCount",
    "MinuteIntensitySum",
    "MinuteIntensityMean",
    "MinuteIntensityMax",
    "MinuteIntensityStd",
    "MinuteIntensityCount",
    "MinuteMETsSum",
    "MinuteMETsMean",
    "MinuteMETsMax",
    "MinuteMETsStd",
    "MinuteMETsCount",
    "SleepMinuteCount",
    "SleepValueMean",
    "SleepValueMax",
    "SleepValueStd",
    "HasSleepLog",
    "HeartRateMean",
    "HeartRateMin",
    "HeartRateMax",
    "HeartRateStd",
    "HeartRateCount",
    "HasHeartRate",
    "WeightKg",
    "BMI",
    "Fat",
    "ManualWeightLogCount",
    "WeightLogCount",
    "HasWeightLog",
    "HasFatLog",
]

MODEL_FEATURE_COLUMNS = [
    column for column in ALL_FEATURE_COLUMNS if not column.startswith("MinuteCalories")
]

OUTLIER_EXCLUDED_COLUMNS = {
    "Id",
    "Hour",
    "DayOfWeek",
    "IsWeekend",
    "HasSleepLog",
    "HasHeartRate",
    "HasWeightLog",
    "HasFatLog",
}


def parse_datetime(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, format=DATETIME_FORMAT, errors="coerce")


def load_raw_hourly_data() -> pd.DataFrame:
    calories = pd.read_csv(ROOT / "hourlyCalories_merged.csv", usecols=["Id", "ActivityHour", "Calories"])
    steps = pd.read_csv(ROOT / "hourlySteps_merged.csv", usecols=["Id", "ActivityHour", "StepTotal"])
    intensities = pd.read_csv(
        ROOT / "hourlyIntensities_merged.csv",
        usecols=["Id", "ActivityHour", "TotalIntensity", "AverageIntensity"],
    )

    data = calories.merge(steps, on=["Id", "ActivityHour"], how="inner")
    return data.merge(intensities, on=["Id", "ActivityHour"], how="inner")


def aggregate_minute_data(filename: str, value_column: str, prefix: str) -> pd.DataFrame:
    data = pd.read_csv(ROOT / filename, usecols=["Id", "ActivityMinute", value_column])
    data["Id"] = pd.to_numeric(data["Id"], errors="coerce")
    data[value_column] = pd.to_numeric(data[value_column], errors="coerce")
    data["ActivityHour"] = parse_datetime(data["ActivityMinute"]).dt.floor("h")
    data = data.dropna(subset=["Id", "ActivityHour", value_column]).copy()
    data["Id"] = data["Id"].astype("int64")

    hourly = (
        data.groupby(["Id", "ActivityHour"])[value_column]
        .agg(["sum", "mean", "max", "std", "count"])
        .reset_index()
        .rename(
            columns={
                "sum": f"{prefix}Sum",
                "mean": f"{prefix}Mean",
                "max": f"{prefix}Max",
                "std": f"{prefix}Std",
                "count": f"{prefix}Count",
            }
        )
    )
    hourly[f"{prefix}Std"] = hourly[f"{prefix}Std"].fillna(0)
    return hourly


def aggregate_sleep_data() -> pd.DataFrame:
    data = pd.read_csv(ROOT / "minuteSleep_merged.csv", usecols=["Id", "date", "value"])
    data["Id"] = pd.to_numeric(data["Id"], errors="coerce")
    data["SleepValue"] = pd.to_numeric(data["value"], errors="coerce")
    data["ActivityHour"] = parse_datetime(data["date"]).dt.floor("h")
    data = data.dropna(subset=["Id", "ActivityHour", "SleepValue"]).copy()
    data["Id"] = data["Id"].astype("int64")

    hourly = (
        data.groupby(["Id", "ActivityHour"])
        .agg(
            SleepMinuteCount=("SleepValue", "count"),
            SleepValueMean=("SleepValue", "mean"),
            SleepValueMax=("SleepValue", "max"),
            SleepValueStd=("SleepValue", "std"),
        )
        .reset_index()
    )
    hourly["SleepValueStd"] = hourly["SleepValueStd"].fillna(0)
    return hourly


def aggregate_heart_rate_data() -> pd.DataFrame:
    data = pd.read_csv(ROOT / "heartrate_seconds_merged.csv", usecols=["Id", "Time", "Value"])
    data["Id"] = pd.to_numeric(data["Id"], errors="coerce")
    data["HeartRate"] = pd.to_numeric(data["Value"], errors="coerce")
    data["ActivityHour"] = parse_datetime(data["Time"]).dt.floor("h")
    data = data.dropna(subset=["Id", "ActivityHour", "HeartRate"]).copy()
    data["Id"] = data["Id"].astype("int64")

    hourly = (
        data.groupby(["Id", "ActivityHour"])
        .agg(
            HeartRateMean=("HeartRate", "mean"),
            HeartRateMin=("HeartRate", "min"),
            HeartRateMax=("HeartRate", "max"),
            HeartRateStd=("HeartRate", "std"),
            HeartRateCount=("HeartRate", "count"),
        )
        .reset_index()
    )
    hourly["HeartRateStd"] = hourly["HeartRateStd"].fillna(0)
    return hourly


def aggregate_weight_data() -> pd.DataFrame:
    data = pd.read_csv(
        ROOT / "weightLogInfo_merged.csv",
        usecols=["Id", "Date", "WeightKg", "Fat", "BMI", "IsManualReport"],
    )
    data["Id"] = pd.to_numeric(data["Id"], errors="coerce")
    data["ActivityDate"] = parse_datetime(data["Date"]).dt.normalize()
    for column in ["WeightKg", "Fat", "BMI"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["IsManualReport"] = data["IsManualReport"].astype(str).str.lower().eq("true").astype(int)
    data = data.dropna(subset=["Id", "ActivityDate"]).copy()
    data["Id"] = data["Id"].astype("int64")

    return (
        data.groupby(["Id", "ActivityDate"])
        .agg(
            WeightKg=("WeightKg", "mean"),
            BMI=("BMI", "mean"),
            Fat=("Fat", "mean"),
            ManualWeightLogCount=("IsManualReport", "sum"),
            WeightLogCount=("WeightKg", "count"),
        )
        .reset_index()
    )


def add_all_dataset_features(data: pd.DataFrame) -> pd.DataFrame:
    minute_sources = [
        ("minuteCaloriesNarrow_merged.csv", "Calories", "MinuteCalories"),
        ("minuteStepsNarrow_merged.csv", "Steps", "MinuteSteps"),
        ("minuteIntensitiesNarrow_merged.csv", "Intensity", "MinuteIntensity"),
        ("minuteMETsNarrow_merged.csv", "METs", "MinuteMETs"),
    ]

    for filename, value_column, prefix in minute_sources:
        data = data.merge(
            aggregate_minute_data(filename, value_column, prefix),
            on=["Id", "ActivityHour"],
            how="left",
        )

    data = data.merge(aggregate_sleep_data(), on=["Id", "ActivityHour"], how="left")
    data = data.merge(aggregate_heart_rate_data(), on=["Id", "ActivityHour"], how="left")
    data = data.merge(aggregate_weight_data(), on=["Id", "ActivityDate"], how="left")

    data["HasSleepLog"] = data["SleepMinuteCount"].fillna(0).gt(0).astype(int)
    data["HasHeartRate"] = data["HeartRateCount"].fillna(0).gt(0).astype(int)
    data["HasWeightLog"] = data["WeightLogCount"].fillna(0).gt(0).astype(int)
    data["HasFatLog"] = data["Fat"].notna().astype(int)
    return data


def fill_missing_feature_values(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    for column in ALL_FEATURE_COLUMNS:
        if column not in data.columns:
            data[column] = pd.NA

    missing_before_fill = data[ALL_FEATURE_COLUMNS].isna().sum()
    zero_fill_columns = [
        column
        for column in ALL_FEATURE_COLUMNS
        if column.startswith("Minute")
        or column.startswith("Sleep")
        or column.endswith("Count")
        or column in OUTLIER_EXCLUDED_COLUMNS
    ]

    for column in zero_fill_columns:
        if column != "Id":
            data[column] = data[column].fillna(0)

    remaining_columns = [column for column in ALL_FEATURE_COLUMNS if column not in zero_fill_columns and column != "Id"]
    if remaining_columns:
        user_medians = data.groupby("Id")[remaining_columns].transform("median")
        data[remaining_columns] = data[remaining_columns].fillna(user_medians)
        data[remaining_columns] = data[remaining_columns].fillna(data[remaining_columns].median(numeric_only=True))
        data[remaining_columns] = data[remaining_columns].fillna(0)

    return data, missing_before_fill


def cap_iqr_outliers(data: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, dict[str, int]]:
    capped_counts = {}

    for column in columns:
        if column in OUTLIER_EXCLUDED_COLUMNS or column not in data.columns:
            continue
        if not pd.api.types.is_numeric_dtype(data[column]) or data[column].nunique(dropna=True) <= 2:
            continue

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

    data["ActivityHour"] = parse_datetime(data["ActivityHour"])
    data["Id"] = pd.to_numeric(data["Id"], errors="coerce")
    for column in BASE_NUMERIC_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    missing_required = data[BASE_COLUMNS].isna().any(axis=1)
    missing_required_rows_removed = int(missing_required.sum())
    missing_required_cells = data[BASE_COLUMNS].isna().sum()
    data = data.loc[~missing_required].copy()

    data["Id"] = data["Id"].astype("int64")
    data["ActivityDate"] = data["ActivityHour"].dt.normalize()
    data["Hour"] = data["ActivityHour"].dt.hour
    data["DayOfWeek"] = data["ActivityHour"].dt.dayofweek
    data["IsWeekend"] = data["DayOfWeek"].isin([5, 6]).astype(int)

    data = add_all_dataset_features(data)
    data, missing_features_before_fill = fill_missing_feature_values(data)

    numeric_model_columns = ["Calories"] + [column for column in ALL_FEATURE_COLUMNS if column != "Id"]
    invalid_values = (data[numeric_model_columns] < 0).any(axis=1)
    invalid_rows_removed = int(invalid_values.sum())
    data = data.loc[~invalid_values].copy()

    data, capped_counts = cap_iqr_outliers(data, numeric_model_columns)
    data = data.sort_values(["Id", "ActivityHour"]).reset_index(drop=True)

    remaining_missing = int(data[BASE_COLUMNS + ALL_FEATURE_COLUMNS].isna().sum().sum())

    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        data.to_csv(PROCESSED_DATA_PATH, index=False)

        report = [
            "Health dataset preprocessing report",
            f"Raw merged hourly rows: {raw_rows}",
            f"Duplicate rows removed: {duplicate_rows_removed}",
            f"Rows with missing required base values removed: {missing_required_rows_removed}",
            f"Invalid negative rows removed: {invalid_rows_removed}",
            f"Final clean rows: {len(data)}",
            f"Remaining missing values after cleanup: {remaining_missing}",
            "",
            "Datasets used:",
            "- hourlyCalories_merged.csv",
            "- hourlySteps_merged.csv",
            "- hourlyIntensities_merged.csv",
            "- minuteCaloriesNarrow_merged.csv",
            "- minuteStepsNarrow_merged.csv",
            "- minuteIntensitiesNarrow_merged.csv",
            "- minuteMETsNarrow_merged.csv",
            "- minuteSleep_merged.csv",
            "- heartrate_seconds_merged.csv",
            "- weightLogInfo_merged.csv",
            "",
            "Note: minuteCaloriesNarrow_merged.csv is saved in the processed data but excluded from model features because it directly duplicates the calorie prediction target.",
            "",
            "Missing required base values before cleanup:",
        ]
        report.extend(f"- {column}: {int(count)}" for column, count in missing_required_cells.items())
        report.append("")
        report.append("Optional feature missing values filled before training:")
        filled_features = missing_features_before_fill[missing_features_before_fill > 0]
        if filled_features.empty:
            report.append("- none")
        else:
            report.extend(f"- {column}: {int(count)}" for column, count in filled_features.items())
        report.append("")
        report.append("Outliers capped with IQR bounds:")
        if not capped_counts:
            report.append("- none")
        else:
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

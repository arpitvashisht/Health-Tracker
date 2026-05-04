# Health-Tracker

AI smart health tracker project using Fitbit-style activity datasets.

## Dataset files

This repo currently includes hourly and minute-level health tracking CSV files, including:

- `hourlyCalories_merged.csv`
- `hourlySteps_merged.csv`
- `hourlyIntensities_merged.csv`
- `heartrate_seconds_merged.csv`
- `minuteCaloriesNarrow_merged.csv`
- `minuteIntensitiesNarrow_merged.csv`
- `minuteMETsNarrow_merged.csv`
- `minuteSleep_merged.csv`
- `minuteStepsNarrow_merged.csv`
- `weightLogInfo_merged.csv`

## ML model

The first model predicts hourly calories burned from all available datasets:

- hourly calories, steps, and intensity data
- minute-level calories, steps, intensity, and METs
- minute-level sleep logs
- second-level heart rate data
- weight, BMI, and fat logs
- time features such as hour of day and day of week

Install dependencies:

```bash
pip install -r requirements.txt
```

Preprocess the datasets:

```bash
python src/preprocess_health_data.py
```

Train the model:

```bash
python src/train_calorie_model.py
```

Run a prediction:

```bash
python src/predict_calories.py --id 1503960366 --steps 500 --total-intensity 20 --average-intensity 0.33 --hour 18 --day-of-week 2
```

Run the frontend:

```bash
python src/serve_frontend.py
```

Then open `http://127.0.0.1:8000`.

The frontend is built with HTML, CSS, and React. It is served from the `frontend/` folder and calls the Python API in `src/serve_frontend.py`.

The trained model is saved to `models/calorie_predictor.joblib`.

The preprocessing step creates a cleaned dataset at `data/processed/hourly_health_features.csv`.
It combines every uploaded dataset into hourly features, removes duplicate rows, removes missing required values, fills optional sensor gaps, removes invalid negative rows, and caps numeric outliers using the IQR method.

`minuteCaloriesNarrow_merged.csv` is included in the cleaned dataset for validation and analysis, but its calorie columns are excluded from model training because they directly duplicate the value the model is trying to predict.

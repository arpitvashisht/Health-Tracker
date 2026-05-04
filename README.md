# Health-Tracker

AI smart health tracker project using Fitbit-style activity datasets.

## Dataset files

This repo currently includes hourly and minute-level health tracking CSV files, including:

- `hourlyCalories_merged.csv`
- `hourlySteps_merged.csv`
- `hourlyIntensities_merged.csv`
- `minuteCaloriesNarrow_merged.csv`
- `minuteIntensitiesNarrow_merged.csv`
- `minuteMETsNarrow_merged.csv`
- `minuteSleep_merged.csv`
- `minuteStepsNarrow_merged.csv`
- `weightLogInfo_merged.csv`

## ML model

The first model predicts hourly calories burned from:

- user id
- steps
- activity intensity
- hour of day
- day of week

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python src/train_calorie_model.py
```

Run a prediction:

```bash
python src/predict_calories.py --id 1503960366 --steps 500 --total-intensity 20 --average-intensity 0.33 --hour 18 --day-of-week 2
```

The trained model is saved to `models/calorie_predictor.joblib`.

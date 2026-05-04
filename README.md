# AI Smart Health Tracker Chatbot using Generative AI

A web-based AI smart health tracker that analyzes daily habits and responds like a conversational health assistant.

The app lets users enter:

- steps walked
- sleep hours
- water intake in liters

It calculates a health score, detects unhealthy habits, generates personalized suggestions, and shows the results in a clean React UI.

Full project documentation is available in [`PROJECT_REPORT.md`](PROJECT_REPORT.md).

## Application features

- user profiles for multiple people
- personal goals for steps, sleep, water, and calories
- daily logs for steps, sleep, water, mood, stress, energy, and notes
- health score calculation from daily health data
- unhealthy habit detection from current and recent logs
- progress charts for the last seven days
- streaks for steps, sleep, hydration, and strong health scores
- meal and calorie logging
- reminders for healthy habits
- chatbot memory using saved chat history and daily logs
- weekly AI report with a local fallback when no API key is set

The backend uses SQLite and stores local app data in `data/health_tracker.db`.

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

## Generative AI chatbot

Run the web app:

```bash
python src/serve_frontend.py
```

Then open `http://127.0.0.1:8000`.

The frontend is built with HTML, CSS, and React. It is served from the `frontend/` folder and calls the Python API in `src/serve_frontend.py`.

The chatbot works in two modes:

- OpenAI mode when `OPENAI_API_KEY` is set
- local fallback mode when no API key is available

Set your OpenAI API key in PowerShell:

```powershell
$env:OPENAI_API_KEY="your_api_key_here"
python src/serve_frontend.py
```

You can optionally choose a model:

```powershell
$env:OPENAI_MODEL="gpt-5.2"
```

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

The trained model is saved to `models/calorie_predictor.joblib`.

The preprocessing step creates a cleaned dataset at `data/processed/hourly_health_features.csv`.
It combines every uploaded dataset into hourly features, removes duplicate rows, removes missing required values, fills optional sensor gaps, removes invalid negative rows, and caps numeric outliers using the IQR method.

`minuteCaloriesNarrow_merged.csv` is included in the cleaned dataset for validation and analysis, but its calorie columns are excluded from model training because they directly duplicate the value the model is trying to predict.

# Project Report: AI Smart Health Tracker Chatbot using Generative AI

## 1. Project Title

**AI Smart Health Tracker Chatbot using Generative AI**

## 2. Project Overview

This project is a web-based smart health tracking application that helps users monitor daily lifestyle habits and receive personalized health guidance. Users can enter daily health data such as steps walked, sleep hours, water intake, mood, stress, energy, meals, and reminders.

The system calculates a health score, detects unhealthy habits, shows progress charts and streaks, stores user history, and provides chatbot-style health suggestions. The chatbot can use the OpenAI Responses API when an API key is available, and it also includes a local fallback assistant when no API key is configured.

## 3. Problem Statement

Many people track health data but do not know how to interpret it or turn it into useful daily actions. This project solves that problem by combining health tracking, machine learning, simple habit analysis, and Generative AI into one interactive assistant.

## 4. Objectives

- Collect daily health data from users.
- Calculate a personalized health score.
- Detect unhealthy patterns such as low steps, low sleep, low hydration, high stress, or low energy.
- Predict hourly calories burned using a machine learning model.
- Generate personalized suggestions using Generative AI.
- Store user history using a backend database.
- Show charts, streaks, meals, reminders, and weekly reports.
- Provide a clean and modern React-based user interface.

## 5. Datasets Used

The project uses **10 Fitbit-style dataset files**:

1. `hourlyCalories_merged.csv`
2. `hourlySteps_merged.csv`
3. `hourlyIntensities_merged.csv`
4. `heartrate_seconds_merged.csv`
5. `minuteCaloriesNarrow_merged.csv`
6. `minuteIntensitiesNarrow_merged.csv`
7. `minuteMETsNarrow_merged.csv`
8. `minuteSleep_merged.csv`
9. `minuteStepsNarrow_merged.csv`
10. `weightLogInfo_merged.csv`

`minuteCaloriesNarrow_merged.csv` is included during preprocessing and analysis, but its calorie-derived columns are excluded from model training because they directly duplicate the prediction target and would cause data leakage.

## 6. Dataset Preprocessing

Preprocessing is implemented in:

`src/preprocess_health_data.py`

Preprocessing steps:

- Merge hourly calories, steps, and intensity datasets.
- Aggregate minute-level datasets into hourly features.
- Aggregate heart-rate seconds into hourly heart-rate features.
- Aggregate sleep data into hourly sleep features.
- Merge weight, BMI, and fat log information.
- Convert date/time fields into standard datetime format.
- Remove duplicate rows.
- Remove rows with missing required base values.
- Fill optional missing sensor values.
- Remove invalid negative rows.
- Cap outliers using the IQR method.
- Save cleaned output to `data/processed/hourly_health_features.csv`.

## 7. Machine Learning Model

The calorie prediction model is implemented in:

`src/train_calorie_model.py`

### Model Used

**Random Forest Regressor**

Library:

`scikit-learn`

### Why Random Forest?

Random Forest was selected because:

- It works well with tabular health/activity data.
- It can model non-linear relationships.
- It handles mixed feature importance well.
- It is easy to train and evaluate.
- It performs strongly without requiring deep learning infrastructure.

### ML Pipeline

The training pipeline includes:

- `ColumnTransformer`
- `OneHotEncoder` for user ID
- numeric passthrough features
- `RandomForestRegressor`
- `train_test_split`
- `mean_absolute_error`
- `r2_score`
- `joblib` model saving

### Model Configuration

```python
RandomForestRegressor(
    n_estimators=150,
    random_state=42,
    min_samples_leaf=2,
    n_jobs=1,
)
```

### Target Variable

The model predicts:

`Calories`

### Features Used

The model uses **40 features**, including:

- steps
- total intensity
- average intensity
- hour of day
- day of week
- weekend flag
- minute-level steps
- minute-level intensity
- METs
- sleep features
- heart-rate features
- weight/BMI/fat features
- availability flags such as `HasSleepLog`, `HasHeartRate`, and `HasWeightLog`

## 8. Model Evaluation

After preprocessing and removing target leakage, the trained model achieved:

```text
Rows used: 24084
Features used: 40
Mean absolute error: 0.76 calories
R2 score: 0.994
```

### Metrics Used

**Mean Absolute Error (MAE):**

Measures the average prediction error in calories.

**R2 Score:**

Measures how well the model explains the variance in calorie values.

## 9. Generative AI Chatbot

The chatbot backend is implemented in:

`src/serve_frontend.py`

### AI API Used

**OpenAI Responses API**

The app uses OpenAI when this environment variable is set:

```powershell
$env:OPENAI_API_KEY="your_api_key_here"
```

Default model:

```powershell
gpt-5.2
```

The model can be changed using:

```powershell
$env:OPENAI_MODEL="model_name"
```

### Local Fallback

If no OpenAI API key is available, the app still works using a local rule-based assistant. This fallback creates useful health suggestions based on health score, goals, recent logs, habits, and trends.

## 10. Health Score Calculation

The health score is calculated from:

- steps
- sleep hours
- water intake
- mood
- stress
- energy
- user goals

The final score is out of **100**.

The app also detects patterns such as:

- low daily movement
- repeated movement gap
- short sleep
- repeated low hydration
- high stress
- low mood
- low energy

## 11. Backend Technology

Backend file:

`src/serve_frontend.py`

### Backend Tools

- Python
- `http.server`
- SQLite
- JSON APIs
- `urllib` for OpenAI API requests
- `joblib` for loading the ML model
- `pandas` for prediction input formatting

### Database

Database:

`SQLite`

Local database path:

`data/health_tracker.db`

### Database Tables

- `users`
- `goals`
- `daily_logs`
- `meals`
- `reminders`
- `chat_messages`
- `weekly_reports`

## 12. Backend API Routes

Main API endpoints:

- `GET /api/status`
- `GET /api/app-state`
- `POST /api/users`
- `POST /api/goals`
- `POST /api/daily-log`
- `POST /api/meals`
- `POST /api/reminders`
- `POST /api/reminders/toggle`
- `POST /api/health-chat`
- `POST /api/weekly-report`
- `POST /api/predict`

## 13. Frontend Technology

Frontend files:

- `frontend/index.html`
- `frontend/app.js`
- `frontend/styles.css`

### Frontend Tools

- HTML
- CSS
- JavaScript
- React
- Browser Fetch API
- SVG chart rendering

### UI Features

- Purple-pink gradient theme
- User profile selection
- Personal goals form
- Daily health log form
- Health score ring
- Streak cards
- Progress charts
- Habit detection chips
- Personalized suggestion list
- Meal and calorie logging
- Reminder management
- Weekly AI report panel
- Chatbot interface with memory

## 14. Tools and Technologies Used

### Programming Languages

- Python
- JavaScript
- HTML
- CSS

### Python Libraries

- `pandas`
- `scikit-learn`
- `joblib`
- `sqlite3`

### Machine Learning

- Random Forest Regressor
- One-hot encoding
- Train/test split
- MAE and R2 evaluation

### Generative AI

- OpenAI Responses API
- Local fallback chatbot logic

### Frontend

- React
- CSS Grid
- SVG charts
- Responsive design

### Backend

- Python HTTP server
- REST-style JSON API
- SQLite database

### Version Control

- Git
- GitHub
- Pull request branch: `add-health-datasets`

## 15. System Architecture

```text
User
  |
  v
React Frontend
  |
  v
Python Backend API
  |
  |-- SQLite database
  |-- ML calorie prediction model
  |-- Health score and habit analysis logic
  |-- OpenAI Responses API or local fallback assistant
```

## 16. How the System Works

1. User opens the web app.
2. User enters daily health information.
3. Backend calculates the health score.
4. Backend detects unhealthy habits.
5. Data is saved in SQLite.
6. Frontend displays charts, streaks, score, suggestions, and history.
7. User asks the chatbot a question.
8. Backend sends context to OpenAI if an API key is available.
9. If OpenAI is unavailable, local fallback generates a response.
10. Chat history is saved for future context.

## 17. How to Run the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the ML model:

```bash
python src/train_calorie_model.py
```

Run the app:

```bash
python src/serve_frontend.py
```

Open:

```text
http://127.0.0.1:8000
```

Optional OpenAI setup:

```powershell
$env:OPENAI_API_KEY="your_api_key_here"
python src/serve_frontend.py
```

## 18. Strengths of the Project

- Complete full-stack application.
- Uses real health/activity datasets.
- Includes preprocessing and outlier handling.
- Uses a trained ML model.
- Uses Generative AI for chatbot responses.
- Works even without an OpenAI API key.
- Stores user data locally.
- Includes charts, streaks, goals, reminders, and weekly reports.
- Clean responsive UI.

## 19. Limitations

- The health score is rule-based, not clinically validated.
- The app is for lifestyle guidance only, not medical diagnosis.
- SQLite is local and not suitable for large multi-user production deployment.
- React is loaded through CDN instead of a full build system.
- The model is trained on historical Fitbit-style data and may not generalize perfectly to all users.
- Authentication is not implemented yet.

## 20. Future Scope

- Add secure login and authentication.
- Deploy backend on a cloud platform.
- Add database hosting such as PostgreSQL.
- Add wearable device integration.
- Add email or notification reminders.
- Add advanced nutrition tracking.
- Add doctor/export report PDF.
- Add better charting library.
- Add mobile app version.
- Add model monitoring and retraining.
- Add personalized long-term health trend forecasting.

## 21. Conclusion

The AI Smart Health Tracker Chatbot using Generative AI is a complete health-focused web application that combines machine learning, data preprocessing, Generative AI, full-stack backend development, and a modern React frontend. It helps users understand their daily habits, identify unhealthy patterns, track progress, and receive personalized suggestions in a conversational way.

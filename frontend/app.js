const { useEffect, useMemo, useState } = React;

const starterMessage = {
  role: "assistant",
  message: "Add your daily log or ask me about your habits. I will use your saved history, goals, meals, and reminders to guide you.",
};

const emptyState = {
  users: [],
  activeUser: null,
  goals: { stepsGoal: 8000, sleepGoal: 7.5, waterGoal: 2.5, calorieGoal: 2200 },
  logs: [],
  meals: [],
  reminders: [],
  chatMessages: [],
  trends: { averages: {}, scoreDelta: null, series: [] },
  streaks: { steps: 0, sleep: 0, water: 0, overall: 0 },
  weeklyReport: null,
  today: new Date().toISOString().slice(0, 10),
  aiReady: false,
  openaiModel: "gpt-5.2",
};

function h(type, props, ...children) {
  return React.createElement(type, props, ...children);
}

function numberValue(value) {
  return Number(value || 0);
}

function formatNumber(value) {
  return Number(value || 0).toLocaleString();
}

function apiPost(path, payload) {
  return fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  }).then(async (response) => {
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Request failed");
    return data;
  });
}

function scoreLabel(score) {
  if (score >= 85) return "Excellent";
  if (score >= 70) return "Good";
  if (score >= 50) return "Needs attention";
  return "Needs care";
}

function Field({ label, hint, children }) {
  return h(
    "label",
    null,
    h("span", null, label),
    children,
    hint ? h("small", null, hint) : null
  );
}

function StatCard({ label, value, tone }) {
  return h(
    "div",
    { className: `stat-card ${tone || ""}` },
    h("span", null, label),
    h("strong", null, value)
  );
}

function SimpleLineChart({ series }) {
  const points = series.length ? series : [{ date: "No data", healthScore: 0 }];
  const width = 360;
  const height = 150;
  const padding = 18;
  const max = 100;
  const xStep = points.length > 1 ? (width - padding * 2) / (points.length - 1) : 0;
  const coords = points.map((point, index) => {
    const x = padding + index * xStep;
    const y = height - padding - (Number(point.healthScore || 0) / max) * (height - padding * 2);
    return { x, y, point };
  });
  const path = coords.map((coord, index) => `${index === 0 ? "M" : "L"} ${coord.x} ${coord.y}`).join(" ");

  return h(
    "div",
    { className: "chart-box" },
    h(
      "svg",
      { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": "Health score trend" },
      h("path", { className: "chart-grid", d: `M ${padding} ${height / 2} L ${width - padding} ${height / 2}` }),
      h("path", { className: "chart-line", d: path }),
      coords.map((coord) => h("circle", { key: `${coord.point.date}-${coord.x}`, className: "chart-dot", cx: coord.x, cy: coord.y, r: 4 }))
    ),
    h(
      "div",
      { className: "chart-labels" },
      points.map((point) => h("span", { key: point.date }, point.date.slice(5)))
    )
  );
}

function GoalBars({ latest, goals }) {
  const bars = [
    { label: "Steps", value: latest?.steps || 0, goal: goals.stepsGoal, suffix: "" },
    { label: "Sleep", value: latest?.sleep_hours || 0, goal: goals.sleepGoal, suffix: "h" },
    { label: "Water", value: latest?.water_liters || 0, goal: goals.waterGoal, suffix: "L" },
  ];

  return h(
    "div",
    { className: "goal-bars" },
    bars.map((bar) => {
      const pct = Math.min(100, Math.round((Number(bar.value) / Number(bar.goal || 1)) * 100));
      return h(
        "div",
        { className: "goal-bar", key: bar.label },
        h("div", null, h("span", null, bar.label), h("strong", null, `${bar.value}${bar.suffix} / ${bar.goal}${bar.suffix}`)),
        h("div", { className: "bar-track" }, h("div", { className: "bar-fill", style: { width: `${pct}%` } }))
      );
    })
  );
}

function App() {
  const [state, setState] = useState(emptyState);
  const [dailyForm, setDailyForm] = useState({
    date: emptyState.today,
    steps: "6500",
    sleepHours: "6.5",
    waterLiters: "1.8",
    mood: "3",
    stress: "3",
    energy: "3",
    notes: "",
  });
  const [goalForm, setGoalForm] = useState(emptyState.goals);
  const [mealForm, setMealForm] = useState({ date: emptyState.today, name: "Breakfast", calories: "450" });
  const [reminderForm, setReminderForm] = useState({ title: "Drink water", time: "10:00", category: "hydration" });
  const [newUserName, setNewUserName] = useState("");
  const [chatInput, setChatInput] = useState("");
  const [statusText, setStatusText] = useState("Loading");
  const [busy, setBusy] = useState(false);

  const userId = state.activeUser?.id || 1;
  const latestLog = state.logs[state.logs.length - 1] || null;
  const messages = state.chatMessages.length ? state.chatMessages : [starterMessage];
  const todayMeals = state.meals.filter((meal) => meal.meal_date === dailyForm.date);
  const todayCalories = todayMeals.reduce((total, meal) => total + Number(meal.calories || 0), 0);

  useEffect(() => {
    loadState();
  }, []);

  useEffect(() => {
    setGoalForm(state.goals);
    setDailyForm((current) => ({ ...current, date: state.today || current.date }));
    setMealForm((current) => ({ ...current, date: state.today || current.date }));
  }, [state.goals, state.today]);

  async function loadState(selectedUserId = userId) {
    try {
      const response = await fetch(`/api/app-state?userId=${selectedUserId}`);
      const data = await response.json();
      setState(data);
      setStatusText(data.aiReady ? "OpenAI connected" : "Local AI fallback");
    } catch (error) {
      setStatusText(error.message);
    }
  }

  function updateDaily(name, value) {
    setDailyForm((current) => ({ ...current, [name]: value }));
  }

  function updateGoals(name, value) {
    setGoalForm((current) => ({ ...current, [name]: value }));
  }

  async function runAction(action) {
    setBusy(true);
    try {
      await action();
    } catch (error) {
      setStatusText(error.message);
    } finally {
      setBusy(false);
    }
  }

  function saveDailyLog(event) {
    event.preventDefault();
    runAction(async () => {
      const data = await apiPost("/api/daily-log", {
        userId,
        date: dailyForm.date,
        steps: numberValue(dailyForm.steps),
        sleepHours: numberValue(dailyForm.sleepHours),
        waterLiters: numberValue(dailyForm.waterLiters),
        mood: numberValue(dailyForm.mood),
        stress: numberValue(dailyForm.stress),
        energy: numberValue(dailyForm.energy),
        notes: dailyForm.notes,
      });
      setState(data);
      setStatusText("Daily log saved");
    });
  }

  function saveGoals(event) {
    event.preventDefault();
    runAction(async () => {
      const data = await apiPost("/api/goals", {
        userId,
        stepsGoal: numberValue(goalForm.stepsGoal),
        sleepGoal: numberValue(goalForm.sleepGoal),
        waterGoal: numberValue(goalForm.waterGoal),
        calorieGoal: numberValue(goalForm.calorieGoal),
      });
      setState(data);
      setStatusText("Goals updated");
    });
  }

  function createUser(event) {
    event.preventDefault();
    runAction(async () => {
      const data = await apiPost("/api/users", { name: newUserName });
      setState(data);
      setNewUserName("");
      setStatusText("Profile created");
    });
  }

  function addMeal(event) {
    event.preventDefault();
    runAction(async () => {
      const data = await apiPost("/api/meals", { userId, ...mealForm, calories: numberValue(mealForm.calories) });
      setState(data);
      setMealForm((current) => ({ ...current, name: "", calories: "" }));
      setStatusText("Meal added");
    });
  }

  function addReminder(event) {
    event.preventDefault();
    runAction(async () => {
      const data = await apiPost("/api/reminders", { userId, ...reminderForm });
      setState(data);
      setReminderForm({ title: "", time: "09:00", category: "general" });
      setStatusText("Reminder added");
    });
  }

  function toggleReminder(id) {
    runAction(async () => {
      const data = await apiPost("/api/reminders/toggle", { userId, id });
      setState(data);
      setStatusText("Reminder updated");
    });
  }

  function generateWeeklyReport() {
    runAction(async () => {
      const data = await apiPost("/api/weekly-report", { userId, date: dailyForm.date });
      setState(data);
      setStatusText("Weekly report generated");
    });
  }

  function sendChat(event) {
    event.preventDefault();
    const message = chatInput.trim();
    if (!message) return;
    setChatInput("");
    runAction(async () => {
      const data = await apiPost("/api/health-chat", {
        userId,
        message,
        steps: numberValue(dailyForm.steps),
        sleepHours: numberValue(dailyForm.sleepHours),
        waterLiters: numberValue(dailyForm.waterLiters),
        mood: numberValue(dailyForm.mood),
        stress: numberValue(dailyForm.stress),
        energy: numberValue(dailyForm.energy),
      });
      setState(data.state);
      setStatusText(data.aiPowered ? "OpenAI response" : "Local assistant response");
    });
  }

  const score = latestLog?.health_score || 0;
  const scoreStyle = useMemo(
    () => ({ background: `conic-gradient(#147c72 ${score * 3.6}deg, #e5ebe8 0deg)` }),
    [score]
  );

  return h(
    React.Fragment,
    null,
    h(
      "header",
      { className: "app-header" },
      h("div", null, h("p", { className: "eyebrow" }, "Generative AI Health Assistant"), h("h1", null, "AI Smart Health Tracker Chatbot using Generative AI")),
      h("div", { className: `model-status ${state.aiReady ? "ready" : "error"}` }, statusText)
    ),
    h(
      "main",
      { className: "app-shell tracker-shell" },
      h(
        "section",
        { className: "panel profile-panel" },
        h("div", { className: "panel-heading" }, h("h2", null, "Profile"), h("span", { className: "run-state" }, state.activeUser?.name || "Guest")),
        h(
          "div",
          { className: "profile-row" },
          h(
            "select",
            { value: userId, onChange: (event) => loadState(event.target.value) },
            state.users.map((user) => h("option", { key: user.id, value: user.id }, user.name))
          ),
          h(
            "form",
            { className: "inline-form", onSubmit: createUser },
            h("input", { value: newUserName, placeholder: "New profile name", onChange: (event) => setNewUserName(event.target.value) }),
            h("button", { className: "ghost-button", type: "submit", disabled: busy }, "Add")
          )
        )
      ),
      h(
        "section",
        { className: "panel score-panel" },
        h("div", { className: "score-ring", style: scoreStyle }, h("strong", null, latestLog ? score : "--"), h("span", null, "/100")),
        h("div", null, h("h2", null, latestLog ? scoreLabel(score) : "No log yet"), h("p", null, latestLog ? `Last log: ${latestLog.log_date}` : "Save your first daily log to start tracking."))
      ),
      h(
        "section",
        { className: "panel streak-panel" },
        h("div", { className: "panel-heading" }, h("h2", null, "Streaks"), h("span", { className: "run-state" }, "Live")),
        h(
          "div",
          { className: "streak-grid" },
          h(StatCard, { label: "Steps", value: `${state.streaks.steps} days` }),
          h(StatCard, { label: "Sleep", value: `${state.streaks.sleep} days` }),
          h(StatCard, { label: "Water", value: `${state.streaks.water} days` }),
          h(StatCard, { label: "Score 75+", value: `${state.streaks.overall} days` })
        )
      ),
      h(
        "section",
        { className: "panel input-panel" },
        h("div", { className: "panel-heading" }, h("h2", null, "Daily Health Log"), h("span", { className: "run-state" }, busy ? "Saving" : "Ready")),
        h(
          "form",
          { className: "form-grid daily-form", onSubmit: saveDailyLog },
          h(Field, { label: "Date" }, h("input", { type: "date", value: dailyForm.date, onChange: (event) => updateDaily("date", event.target.value), required: true })),
          h(Field, { label: "Steps walked", hint: `Goal ${formatNumber(state.goals.stepsGoal)}` }, h("input", { type: "number", min: "0", value: dailyForm.steps, onChange: (event) => updateDaily("steps", event.target.value), required: true })),
          h(Field, { label: "Sleep hours", hint: `Goal ${state.goals.sleepGoal}h` }, h("input", { type: "number", min: "0", max: "24", step: "0.1", value: dailyForm.sleepHours, onChange: (event) => updateDaily("sleepHours", event.target.value), required: true })),
          h(Field, { label: "Water intake", hint: `Goal ${state.goals.waterGoal}L` }, h("input", { type: "number", min: "0", step: "0.1", value: dailyForm.waterLiters, onChange: (event) => updateDaily("waterLiters", event.target.value), required: true })),
          ["mood", "stress", "energy"].map((field) =>
            h(
              Field,
              { key: field, label: `${field[0].toUpperCase()}${field.slice(1)} (${dailyForm[field]}/5)` },
              h("input", { type: "range", min: "1", max: "5", value: dailyForm[field], onChange: (event) => updateDaily(field, event.target.value) })
            )
          ),
          h(Field, { label: "Notes" }, h("input", { value: dailyForm.notes, placeholder: "Optional note", onChange: (event) => updateDaily("notes", event.target.value) })),
          h("button", { className: "primary-button", type: "submit", disabled: busy }, "Save Daily Log")
        )
      ),
      h(
        "section",
        { className: "panel goals-panel" },
        h("div", { className: "panel-heading" }, h("h2", null, "Personal Goals"), h("span", { className: "run-state" }, "Custom")),
        h(
          "form",
          { className: "form-grid goals-form", onSubmit: saveGoals },
          h(Field, { label: "Step goal" }, h("input", { type: "number", min: "1", value: goalForm.stepsGoal, onChange: (event) => updateGoals("stepsGoal", event.target.value) })),
          h(Field, { label: "Sleep goal" }, h("input", { type: "number", min: "1", step: "0.1", value: goalForm.sleepGoal, onChange: (event) => updateGoals("sleepGoal", event.target.value) })),
          h(Field, { label: "Water goal" }, h("input", { type: "number", min: "0.1", step: "0.1", value: goalForm.waterGoal, onChange: (event) => updateGoals("waterGoal", event.target.value) })),
          h(Field, { label: "Calorie target" }, h("input", { type: "number", min: "1", value: goalForm.calorieGoal, onChange: (event) => updateGoals("calorieGoal", event.target.value) })),
          h("button", { className: "primary-button", type: "submit", disabled: busy }, "Update Goals")
        ),
        h(GoalBars, { latest: latestLog, goals: state.goals })
      ),
      h(
        "section",
        { className: "panel chart-panel" },
        h("div", { className: "panel-heading" }, h("h2", null, "Progress Charts"), h("span", { className: "run-state" }, "7 days")),
        h(SimpleLineChart, { series: state.trends.series }),
        h(
          "div",
          { className: "metric-grid" },
          h(StatCard, { label: "Avg steps", value: formatNumber(state.trends.averages.steps) }),
          h(StatCard, { label: "Avg sleep", value: `${state.trends.averages.sleepHours || 0}h` }),
          h(StatCard, { label: "Avg water", value: `${state.trends.averages.waterLiters || 0}L` }),
          h(StatCard, { label: "Avg score", value: state.trends.averages.healthScore || 0 })
        )
      ),
      h(
        "section",
        { className: "panel result-panel" },
        h("div", { className: "panel-heading" }, h("h2", null, "Habits & Suggestions"), h("span", { className: "run-state" }, "Personalized")),
        h("div", { className: "habit-list" }, (latestLog?.habits || ["No analysis yet"]).map((habit) => h("span", { key: habit }, habit))),
        h("ul", { className: "suggestion-list" }, (latestLog?.suggestions || ["Save a daily log to generate suggestions."]).map((suggestion) => h("li", { key: suggestion }, suggestion)))
      ),
      h(
        "section",
        { className: "panel meals-panel" },
        h("div", { className: "panel-heading" }, h("h2", null, "Food & Calories"), h("span", { className: "run-state" }, `${todayCalories}/${state.goals.calorieGoal}`)),
        h(
          "form",
          { className: "inline-form stacked", onSubmit: addMeal },
          h("input", { type: "date", value: mealForm.date, onChange: (event) => setMealForm((current) => ({ ...current, date: event.target.value })) }),
          h("input", { value: mealForm.name, placeholder: "Meal name", onChange: (event) => setMealForm((current) => ({ ...current, name: event.target.value })) }),
          h("input", { type: "number", min: "0", value: mealForm.calories, placeholder: "Calories", onChange: (event) => setMealForm((current) => ({ ...current, calories: event.target.value })) }),
          h("button", { className: "primary-button", type: "submit", disabled: busy }, "Add Meal")
        ),
        h("div", { className: "compact-list" }, state.meals.slice(0, 6).map((meal) => h("div", { key: meal.id }, h("span", null, `${meal.meal_date} - ${meal.name}`), h("strong", null, `${meal.calories} cal`))))
      ),
      h(
        "section",
        { className: "panel reminder-panel" },
        h("div", { className: "panel-heading" }, h("h2", null, "Reminders"), h("span", { className: "run-state" }, `${state.reminders.length} total`)),
        h(
          "form",
          { className: "inline-form stacked", onSubmit: addReminder },
          h("input", { value: reminderForm.title, placeholder: "Reminder title", onChange: (event) => setReminderForm((current) => ({ ...current, title: event.target.value })) }),
          h("input", { type: "time", value: reminderForm.time, onChange: (event) => setReminderForm((current) => ({ ...current, time: event.target.value })) }),
          h("input", { value: reminderForm.category, placeholder: "Category", onChange: (event) => setReminderForm((current) => ({ ...current, category: event.target.value })) }),
          h("button", { className: "primary-button", type: "submit", disabled: busy }, "Add Reminder")
        ),
        h(
          "div",
          { className: "compact-list" },
          state.reminders.map((reminder) =>
            h(
              "button",
              { className: `list-button ${reminder.enabled ? "enabled" : ""}`, key: reminder.id, type: "button", onClick: () => toggleReminder(reminder.id) },
              h("span", null, `${reminder.reminder_time} - ${reminder.title}`),
              h("strong", null, reminder.enabled ? "On" : "Off")
            )
          )
        )
      ),
      h(
        "section",
        { className: "panel report-panel" },
        h("div", { className: "panel-heading" }, h("h2", null, "Weekly AI Report"), h("button", { className: "ghost-button", type: "button", onClick: generateWeeklyReport, disabled: busy }, "Generate")),
        h("p", { className: "report-text" }, state.weeklyReport?.report || "Generate a weekly report after saving a few daily logs.")
      ),
      h(
        "section",
        { className: "panel history-panel" },
        h("div", { className: "panel-heading" }, h("h2", null, "Daily History"), h("span", { className: "run-state" }, `${state.logs.length} logs`)),
        h(
          "div",
          { className: "table-wrap" },
          h(
            "table",
            null,
            h("thead", null, h("tr", null, ["Date", "Score", "Steps", "Sleep", "Water", "Mood", "Stress", "Energy"].map((head) => h("th", { key: head }, head)))),
            h(
              "tbody",
              null,
              state.logs
                .slice()
                .reverse()
                .map((log) =>
                  h(
                    "tr",
                    { key: log.id },
                    h("td", null, log.log_date),
                    h("td", null, log.health_score),
                    h("td", null, formatNumber(log.steps)),
                    h("td", null, `${log.sleep_hours}h`),
                    h("td", null, `${log.water_liters}L`),
                    h("td", null, log.mood),
                    h("td", null, log.stress),
                    h("td", null, log.energy)
                  )
                )
            )
          )
        )
      ),
      h(
        "section",
        { className: "panel chatbot-panel" },
        h("div", { className: "panel-heading" }, h("h2", null, "Health Chatbot Memory"), h("span", { className: "run-state" }, busy ? "Thinking" : "Ready")),
        h(
          "div",
          { className: "chat-window" },
          messages.map((message, index) => h("div", { className: `chat-message ${message.role}`, key: `${message.role}-${index}` }, message.message))
        ),
        h(
          "form",
          { className: "chat-form", onSubmit: sendChat },
          h("input", { value: chatInput, placeholder: "Ask about your weekly habits, goals, meals, or reminders", onChange: (event) => setChatInput(event.target.value) }),
          h("button", { className: "primary-button", type: "submit", disabled: busy }, "Send")
        )
      )
    )
  );
}

ReactDOM.createRoot(document.querySelector("#root")).render(h(App));

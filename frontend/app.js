const { useEffect, useMemo, useState } = React;

const initialForm = {
  id: "1503960366",
  steps: "500",
  totalIntensity: "20",
  averageIntensity: "0.33",
  hour: "18",
  dayOfWeek: "2",
  heartRateMean: "",
  sleepMinutes: "",
  weightKg: "",
  bmi: "",
};

const days = [
  ["0", "Monday"],
  ["1", "Tuesday"],
  ["2", "Wednesday"],
  ["3", "Thursday"],
  ["4", "Friday"],
  ["5", "Saturday"],
  ["6", "Sunday"],
];

const datasets = [
  "Hourly calories",
  "Hourly steps",
  "Hourly intensity",
  "Minute steps",
  "Minute intensity",
  "METs",
  "Sleep",
  "Heart rate",
  "Weight",
];

function numberOrNull(value) {
  return value === "" ? null : Number(value);
}

function Field({ label, children }) {
  return React.createElement("label", null, label, children);
}

function TextInput(props) {
  return React.createElement("input", {
    ...props,
    onChange: (event) => props.onChange(props.name, event.target.value),
  });
}

function App() {
  const [form, setForm] = useState(initialForm);
  const [modelStatus, setModelStatus] = useState({ label: "Checking model", state: "" });
  const [runState, setRunState] = useState("Ready");
  const [predictedCalories, setPredictedCalories] = useState(null);
  const [featureCount, setFeatureCount] = useState(40);
  const [isLoading, setIsLoading] = useState(false);

  const hours = useMemo(
    () => Array.from({ length: 24 }, (_, hour) => [String(hour), `${String(hour).padStart(2, "0")}:00`]),
    []
  );

  useEffect(() => {
    async function refreshModelStatus() {
      try {
        const response = await fetch("/api/status");
        const status = await response.json();

        if (!response.ok || !status.modelReady) {
          throw new Error(status.error || "Model not ready");
        }

        setModelStatus({ label: "Model ready", state: "ready" });
        setFeatureCount(status.featureCount);
      } catch (error) {
        setModelStatus({ label: "Train model first", state: "error" });
      }
    }

    refreshModelStatus();
  }, []);

  function updateField(name, value) {
    setForm((current) => ({ ...current, [name]: value }));
  }

  function resetForm() {
    setForm(initialForm);
    setPredictedCalories(null);
    setRunState("Ready");
  }

  function predictionPayload() {
    return {
      id: Number(form.id),
      steps: Number(form.steps),
      totalIntensity: Number(form.totalIntensity),
      averageIntensity: Number(form.averageIntensity),
      hour: Number(form.hour),
      dayOfWeek: Number(form.dayOfWeek),
      heartRateMean: numberOrNull(form.heartRateMean),
      sleepMinutes: numberOrNull(form.sleepMinutes),
      weightKg: numberOrNull(form.weightKg),
      bmi: numberOrNull(form.bmi),
    };
  }

  async function predict(event) {
    event.preventDefault();
    setIsLoading(true);
    setRunState("Running");

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(predictionPayload()),
      });
      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.error || "Prediction failed");
      }

      setPredictedCalories(Number(result.predictedCalories));
      setRunState("Complete");
    } catch (error) {
      setPredictedCalories(null);
      setRunState(error.message);
    } finally {
      setIsLoading(false);
    }
  }

  const meterWidth = predictedCalories === null ? 0 : Math.min(100, Math.max(4, (predictedCalories / 180) * 100));

  return React.createElement(
    React.Fragment,
    null,
    React.createElement(
      "header",
      { className: "app-header" },
      React.createElement(
        "div",
        null,
        React.createElement("p", { className: "eyebrow" }, "Health Tracker AI"),
        React.createElement("h1", null, "Calorie Prediction Dashboard")
      ),
      React.createElement("div", { className: `model-status ${modelStatus.state}` }, modelStatus.label)
    ),
    React.createElement(
      "main",
      { className: "app-shell" },
      React.createElement(
        "section",
        { className: "panel input-panel" },
        React.createElement(
          "div",
          { className: "panel-heading" },
          React.createElement("h2", null, "Prediction Inputs"),
          React.createElement("button", { className: "ghost-button", type: "button", onClick: resetForm }, "Reset")
        ),
        React.createElement(
          "form",
          { className: "form-grid", onSubmit: predict },
          React.createElement(
            Field,
            { label: "User ID" },
            React.createElement(TextInput, {
              name: "id",
              type: "number",
              value: form.id,
              required: true,
              onChange: updateField,
            })
          ),
          React.createElement(
            Field,
            { label: "Steps" },
            React.createElement(TextInput, {
              name: "steps",
              type: "number",
              min: "0",
              value: form.steps,
              required: true,
              onChange: updateField,
            })
          ),
          React.createElement(
            Field,
            { label: "Total intensity" },
            React.createElement(TextInput, {
              name: "totalIntensity",
              type: "number",
              min: "0",
              value: form.totalIntensity,
              required: true,
              onChange: updateField,
            })
          ),
          React.createElement(
            Field,
            { label: "Average intensity" },
            React.createElement(TextInput, {
              name: "averageIntensity",
              type: "number",
              min: "0",
              step: "0.01",
              value: form.averageIntensity,
              required: true,
              onChange: updateField,
            })
          ),
          React.createElement(
            Field,
            { label: "Hour" },
            React.createElement(
              "select",
              {
                name: "hour",
                value: form.hour,
                required: true,
                onChange: (event) => updateField("hour", event.target.value),
              },
              hours.map(([value, label]) => React.createElement("option", { key: value, value }, label))
            )
          ),
          React.createElement(
            Field,
            { label: "Day" },
            React.createElement(
              "select",
              {
                name: "dayOfWeek",
                value: form.dayOfWeek,
                required: true,
                onChange: (event) => updateField("dayOfWeek", event.target.value),
              },
              days.map(([value, label]) => React.createElement("option", { key: value, value }, label))
            )
          ),
          React.createElement(
            "div",
            { className: "field-group" },
            React.createElement("h3", null, "Heart"),
            React.createElement(
              Field,
              { label: "Avg bpm" },
              React.createElement(TextInput, {
                name: "heartRateMean",
                type: "number",
                min: "0",
                value: form.heartRateMean,
                placeholder: "optional",
                onChange: updateField,
              })
            )
          ),
          React.createElement(
            "div",
            { className: "field-group" },
            React.createElement("h3", null, "Sleep"),
            React.createElement(
              Field,
              { label: "Sleep minutes" },
              React.createElement(TextInput, {
                name: "sleepMinutes",
                type: "number",
                min: "0",
                max: "60",
                value: form.sleepMinutes,
                placeholder: "optional",
                onChange: updateField,
              })
            )
          ),
          React.createElement(
            "div",
            { className: "field-group" },
            React.createElement("h3", null, "Body"),
            React.createElement(
              Field,
              { label: "Weight kg" },
              React.createElement(TextInput, {
                name: "weightKg",
                type: "number",
                min: "0",
                step: "0.1",
                value: form.weightKg,
                placeholder: "optional",
                onChange: updateField,
              })
            ),
            React.createElement(
              Field,
              { label: "BMI" },
              React.createElement(TextInput, {
                name: "bmi",
                type: "number",
                min: "0",
                step: "0.1",
                value: form.bmi,
                placeholder: "optional",
                onChange: updateField,
              })
            )
          ),
          React.createElement(
            "button",
            { className: "primary-button", type: "submit", disabled: isLoading },
            isLoading ? "Predicting" : "Predict Calories"
          )
        )
      ),
      React.createElement(
        "section",
        { className: "panel result-panel" },
        React.createElement(
          "div",
          { className: "panel-heading" },
          React.createElement("h2", null, "Prediction"),
          React.createElement("span", { className: "run-state" }, runState)
        ),
        React.createElement(
          "div",
          { className: "calorie-readout" },
          React.createElement("span", null, predictedCalories === null ? "--" : predictedCalories.toFixed(1)),
          React.createElement("small", null, "calories")
        ),
        React.createElement(
          "div",
          { className: "meter", "aria-hidden": "true" },
          React.createElement("div", { className: "meter-fill", style: { width: `${meterWidth}%` } })
        ),
        React.createElement(
          "div",
          { className: "metric-grid" },
          React.createElement("div", null, React.createElement("span", null, "Model features"), React.createElement("strong", null, featureCount)),
          React.createElement("div", null, React.createElement("span", null, "Rows trained"), React.createElement("strong", null, "24,084")),
          React.createElement("div", null, React.createElement("span", null, "MAE"), React.createElement("strong", null, "0.76")),
          React.createElement("div", null, React.createElement("span", null, "R2"), React.createElement("strong", null, "0.994"))
        ),
        React.createElement(
          "div",
          { className: "dataset-strip", "aria-label": "Datasets used" },
          datasets.map((dataset) => React.createElement("span", { key: dataset }, dataset))
        )
      )
    )
  );
}

ReactDOM.createRoot(document.querySelector("#root")).render(React.createElement(App));

const form = document.querySelector("#predictionForm");
const resetButton = document.querySelector("#resetButton");
const modelStatus = document.querySelector("#modelStatus");
const runState = document.querySelector("#runState");
const calorieValue = document.querySelector("#calorieValue");
const meterFill = document.querySelector("#meterFill");
const featureCount = document.querySelector("#featureCount");
const hourSelect = form.elements.hour;

for (let hour = 0; hour < 24; hour += 1) {
  const option = document.createElement("option");
  option.value = String(hour);
  option.textContent = `${String(hour).padStart(2, "0")}:00`;
  option.selected = hour === 18;
  hourSelect.append(option);
}

function numberOrNull(value) {
  return value === "" ? null : Number(value);
}

function formPayload() {
  const data = new FormData(form);

  return {
    id: Number(data.get("id")),
    steps: Number(data.get("steps")),
    totalIntensity: Number(data.get("totalIntensity")),
    averageIntensity: Number(data.get("averageIntensity")),
    hour: Number(data.get("hour")),
    dayOfWeek: Number(data.get("dayOfWeek")),
    heartRateMean: numberOrNull(data.get("heartRateMean")),
    sleepMinutes: numberOrNull(data.get("sleepMinutes")),
    weightKg: numberOrNull(data.get("weightKg")),
    bmi: numberOrNull(data.get("bmi")),
  };
}

async function refreshModelStatus() {
  try {
    const response = await fetch("/api/status");
    const status = await response.json();

    if (!response.ok || !status.modelReady) {
      throw new Error(status.error || "Model not ready");
    }

    modelStatus.textContent = "Model ready";
    modelStatus.className = "model-status ready";
    featureCount.textContent = String(status.featureCount);
  } catch (error) {
    modelStatus.textContent = "Train model first";
    modelStatus.className = "model-status error";
  }
}

async function predict(event) {
  event.preventDefault();

  const submitButton = form.querySelector("button[type='submit']");
  submitButton.disabled = true;
  runState.textContent = "Running";

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(formPayload()),
    });

    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.error || "Prediction failed");
    }

    const calories = Number(result.predictedCalories);
    calorieValue.textContent = calories.toFixed(1);
    meterFill.style.width = `${Math.min(100, Math.max(4, (calories / 180) * 100))}%`;
    runState.textContent = "Complete";
  } catch (error) {
    calorieValue.textContent = "--";
    meterFill.style.width = "0";
    runState.textContent = error.message;
  } finally {
    submitButton.disabled = false;
  }
}

function resetForm() {
  form.reset();
  hourSelect.value = "18";
  calorieValue.textContent = "--";
  meterFill.style.width = "0";
  runState.textContent = "Ready";
}

form.addEventListener("submit", predict);
resetButton.addEventListener("click", resetForm);
refreshModelStatus();

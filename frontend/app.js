const { useEffect, useMemo, useState } = React;

const initialForm = {
  steps: "6500",
  sleepHours: "6.5",
  waterLiters: "1.8",
};

const starterMessages = [
  {
    role: "assistant",
    text: "Hi, I am your AI health assistant. Enter today's steps, sleep, and water intake, then I will calculate your score and suggest a realistic next move.",
  },
];

function toNumber(value) {
  return Number(value || 0);
}

function scoreLabel(score) {
  if (score >= 85) return "Excellent";
  if (score >= 70) return "Good";
  if (score >= 50) return "Needs attention";
  return "Needs care";
}

function Field({ label, hint, children }) {
  return React.createElement(
    "label",
    null,
    React.createElement("span", null, label),
    children,
    hint ? React.createElement("small", null, hint) : null
  );
}

function App() {
  const [form, setForm] = useState(initialForm);
  const [chatInput, setChatInput] = useState("");
  const [messages, setMessages] = useState(starterMessages);
  const [analysis, setAnalysis] = useState(null);
  const [history, setHistory] = useState([]);
  const [status, setStatus] = useState({ aiReady: false, modelReady: false });
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    async function loadStatus() {
      try {
        const response = await fetch("/api/status");
        const data = await response.json();
        setStatus(data);
      } catch (error) {
        setStatus({ aiReady: false, modelReady: false });
      }
    }

    loadStatus();
  }, []);

  const score = analysis?.healthScore ?? 0;
  const scoreStyle = useMemo(
    () => ({
      background: `conic-gradient(#147c72 ${score * 3.6}deg, #e5ebe8 0deg)`,
    }),
    [score]
  );

  function updateField(name, value) {
    setForm((current) => ({ ...current, [name]: value }));
  }

  function payload(extraMessage = "") {
    return {
      steps: toNumber(form.steps),
      sleepHours: toNumber(form.sleepHours),
      waterLiters: toNumber(form.waterLiters),
      message: extraMessage,
      history: history.slice(-6),
    };
  }

  async function callHealthAssistant(extraMessage = "") {
    setIsLoading(true);

    const userMessage = extraMessage || "Analyze my daily health data.";
    setMessages((current) => [...current, { role: "user", text: userMessage }]);

    try {
      const response = await fetch("/api/health-chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload(extraMessage)),
      });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Health assistant failed");
      }

      setAnalysis(data);
      setHistory((current) =>
        [
          ...current,
          {
            steps: data.inputs.steps,
            sleepHours: data.inputs.sleepHours,
            waterLiters: data.inputs.waterLiters,
            healthScore: data.healthScore,
          },
        ].slice(-7)
      );
      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          text: data.reply,
        },
      ]);
    } catch (error) {
      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          text: error.message,
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  }

  function submitDailyData(event) {
    event.preventDefault();
    callHealthAssistant();
  }

  function sendChat(event) {
    event.preventDefault();
    const trimmed = chatInput.trim();
    if (!trimmed) return;
    setChatInput("");
    callHealthAssistant(trimmed);
  }

  function resetApp() {
    setForm(initialForm);
    setChatInput("");
    setMessages(starterMessages);
    setAnalysis(null);
    setHistory([]);
  }

  return React.createElement(
    React.Fragment,
    null,
    React.createElement(
      "header",
      { className: "app-header" },
      React.createElement(
        "div",
        null,
        React.createElement("p", { className: "eyebrow" }, "Generative AI Health Assistant"),
        React.createElement("h1", null, "AI Smart Health Tracker Chatbot using Generative AI")
      ),
      React.createElement(
        "div",
        { className: `model-status ${status.aiReady ? "ready" : "error"}` },
        status.aiReady ? "OpenAI connected" : "Local AI fallback"
      )
    ),
    React.createElement(
      "main",
      { className: "app-shell health-app-shell" },
      React.createElement(
        "section",
        { className: "panel input-panel" },
        React.createElement(
          "div",
          { className: "panel-heading" },
          React.createElement("h2", null, "Daily Health Data"),
          React.createElement("button", { className: "ghost-button", type: "button", onClick: resetApp }, "Reset")
        ),
        React.createElement(
          "form",
          { className: "form-grid daily-form", onSubmit: submitDailyData },
          React.createElement(
            Field,
            { label: "Steps walked", hint: "Daily target: 8,000-10,000 steps" },
            React.createElement("input", {
              name: "steps",
              type: "number",
              min: "0",
              value: form.steps,
              required: true,
              onChange: (event) => updateField("steps", event.target.value),
            })
          ),
          React.createElement(
            Field,
            { label: "Sleep hours", hint: "Healthy range: 7-9 hours" },
            React.createElement("input", {
              name: "sleepHours",
              type: "number",
              min: "0",
              max: "24",
              step: "0.1",
              value: form.sleepHours,
              required: true,
              onChange: (event) => updateField("sleepHours", event.target.value),
            })
          ),
          React.createElement(
            Field,
            { label: "Water intake", hint: "Measured in liters" },
            React.createElement("input", {
              name: "waterLiters",
              type: "number",
              min: "0",
              step: "0.1",
              value: form.waterLiters,
              required: true,
              onChange: (event) => updateField("waterLiters", event.target.value),
            })
          ),
          React.createElement(
            "button",
            { className: "primary-button", type: "submit", disabled: isLoading },
            isLoading ? "Analyzing" : "Analyze My Health"
          )
        ),
        React.createElement(
          "div",
          { className: "score-card" },
          React.createElement(
            "div",
            { className: "score-ring", style: scoreStyle },
            React.createElement("strong", null, analysis ? score : "--"),
            React.createElement("span", null, "/100")
          ),
          React.createElement(
            "div",
            null,
            React.createElement("h3", null, analysis ? scoreLabel(score) : "Waiting for data"),
            React.createElement(
              "p",
              null,
              analysis
                ? "Your score combines movement, recovery, and hydration into one simple daily signal."
                : "Submit your daily data to calculate a health score."
            )
          )
        )
      ),
      React.createElement(
        "section",
        { className: "panel result-panel" },
        React.createElement(
          "div",
          { className: "panel-heading" },
          React.createElement("h2", null, "Health Insights"),
          React.createElement("span", { className: "run-state" }, analysis?.aiPowered ? "OpenAI" : "Rule based")
        ),
        React.createElement(
          "div",
          { className: "metric-grid health-metrics" },
          React.createElement("div", null, React.createElement("span", null, "Steps"), React.createElement("strong", null, form.steps || "0")),
          React.createElement("div", null, React.createElement("span", null, "Sleep"), React.createElement("strong", null, `${form.sleepHours || 0}h`)),
          React.createElement("div", null, React.createElement("span", null, "Water"), React.createElement("strong", null, `${form.waterLiters || 0}L`))
        ),
        React.createElement(
          "div",
          { className: "insight-block" },
          React.createElement("h3", null, "Unhealthy Habits Detected"),
          React.createElement(
            "div",
            { className: "habit-list" },
            (analysis?.habits?.length ? analysis.habits : ["No analysis yet"]).map((habit) =>
              React.createElement("span", { key: habit }, habit)
            )
          )
        ),
        React.createElement(
          "div",
          { className: "insight-block" },
          React.createElement("h3", null, "Personalized Suggestions"),
          React.createElement(
            "ul",
            { className: "suggestion-list" },
            (analysis?.suggestions?.length ? analysis.suggestions : ["Suggestions will appear after analysis."]).map((suggestion) =>
              React.createElement("li", { key: suggestion }, suggestion)
            )
          )
        )
      ),
      React.createElement(
        "section",
        { className: "panel chatbot-panel" },
        React.createElement(
          "div",
          { className: "panel-heading" },
          React.createElement("h2", null, "Health Chatbot"),
          React.createElement("span", { className: "run-state" }, isLoading ? "Typing" : "Ready")
        ),
        React.createElement(
          "div",
          { className: "chat-window" },
          messages.map((message, index) =>
            React.createElement(
              "div",
              { className: `chat-message ${message.role}`, key: `${message.role}-${index}` },
              message.text
            )
          )
        ),
        React.createElement(
          "form",
          { className: "chat-form", onSubmit: sendChat },
          React.createElement("input", {
            value: chatInput,
            placeholder: "Ask about your habits, score, or next goal",
            onChange: (event) => setChatInput(event.target.value),
          }),
          React.createElement("button", { className: "primary-button", type: "submit", disabled: isLoading }, "Send")
        )
      )
    )
  );
}

ReactDOM.createRoot(document.querySelector("#root")).render(React.createElement(App));

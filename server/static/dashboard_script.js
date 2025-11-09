const NA_STRING = "N/A";
const trainingForm = document.getElementById("training-form");
const startBtn = document.getElementById("start-training-btn");
const stopBtn = document.getElementById("stop-training-btn");
const trainingStatusText = document.getElementById("training-status-text");
const trainingMessageEl = document.getElementById("training-message");
const trainingInputs = {
  features: document.getElementById("features-input"),
  classes: document.getElementById("classes-input"),
  rounds: document.getElementById("rounds-input"),
  clients: document.getElementById("clients-input"),
};

function setTrainingMessage(message, isError = false) {
  if (!trainingMessageEl) return;
  trainingMessageEl.textContent = message;
  trainingMessageEl.style.color = isError ? "#f48771" : "#b0b0b0";
}

function updateInputValue(input, value) {
  if (!input || value === undefined || document.activeElement === input) {
    return;
  }
  input.value = value;
}

async function updateDashboard() {
  try {
    const response = await fetch("/metrics", { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const metrics = await response.json();

    if (!document.getElementById("server-status")) {
      location.reload();
      return;
    }

    document.getElementById("server-status").textContent =
      metrics.server_status || NA_STRING;
    document.getElementById("current-round").textContent =
      metrics.current_round || "0";

    const latestAccuracy =
      metrics.latest_accuracy_percent === null
        ? NA_STRING
        : `${metrics.latest_accuracy_percent}%`;
    document.getElementById("latest-accuracy").textContent = latestAccuracy;

    // --- NEW: Logic to control the download button state ---
    const downloadBtn = document.getElementById("download-button");
    if (downloadBtn) {
      if (metrics.server_status === "TRAINING_COMPLETE") {
        // If training is complete, enable the button
        downloadBtn.classList.remove("disabled");
        downloadBtn.setAttribute(
          "title",
          "Download the final aggregated model"
        );
      } else {
        // Otherwise, ensure the button is disabled
        downloadBtn.classList.add("disabled");
        downloadBtn.setAttribute(
          "title",
          "Model is available for download once training is complete"
        );
      }
    }
    // --- End of new logic ---

    const historyTbody = document.getElementById("history-tbody");
    let historyHtml = "";
    if (metrics.training_history && metrics.training_history.length > 0) {
      metrics.training_history.forEach((entry) => {
        historyHtml += `<tr><td>${entry.round}</td><td>${entry.accuracy}%</td></tr>`;
      });
    }
    historyTbody.innerHTML = historyHtml;

    if (trainingStatusText && typeof metrics.training_active !== "undefined") {
      trainingStatusText.textContent = metrics.training_active
        ? `Running (PID ${metrics.training_pid ?? "N/A"})`
        : "Idle";
    }

    if (startBtn && stopBtn && typeof metrics.training_active !== "undefined") {
      startBtn.disabled = !!metrics.training_active;
      stopBtn.disabled = !metrics.training_active;
    }

    if (metrics.training_args) {
      updateInputValue(trainingInputs.features, metrics.training_args.features);
      updateInputValue(trainingInputs.classes, metrics.training_args.classes);
      updateInputValue(trainingInputs.rounds, metrics.training_args.rounds);
      updateInputValue(trainingInputs.clients, metrics.training_args.clients);
    }
  } catch (error) {
    console.error("Failed to fetch dashboard metrics:", error);
    const container = document.getElementById("dashboard-container");
    if (container) {
      container.innerHTML = `<div class="error-container"><h1>Connection Error</h1><p>Could not fetch live data from the server. Is it running? Retrying automatically...</p></div>`;
    }
  }
}

setInterval(updateDashboard, 150);

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.detail || data.message || "Request failed.");
  }
  return data;
}

function readTrainingPayload() {
  const payload = {};
  Object.entries(trainingInputs).forEach(([key, input]) => {
    if (!input) return;
    const value = parseInt(input.value, 10);
    if (!Number.isFinite(value) || value < 1) {
      throw new Error(`"${key}" must be a positive number.`);
    }
    payload[key] = value;
  });
  return payload;
}

if (trainingForm && startBtn) {
  trainingForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    try {
      const payload = readTrainingPayload();
      setTrainingMessage("Starting training...");
      startBtn.disabled = true;
      const data = await postJson("/training/start", payload);
      setTrainingMessage(`Training started (PID ${data.pid ?? "N/A"}).`);
    } catch (error) {
      console.error(error);
      setTrainingMessage(error.message, true);
      startBtn.disabled = false;
    }
  });
}

if (stopBtn) {
  stopBtn.addEventListener("click", async () => {
    try {
      setTrainingMessage("Stopping training...");
      stopBtn.disabled = true;
      await postJson("/training/stop", {});
      setTrainingMessage("Training stop requested.");
    } catch (error) {
      console.error(error);
      setTrainingMessage(error.message, true);
      stopBtn.disabled = false;
    }
  });
}

updateDashboard();

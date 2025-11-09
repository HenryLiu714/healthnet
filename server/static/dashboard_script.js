document.addEventListener("DOMContentLoaded", () => {
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

  /**
   * Formats a raw server status string into a human-readable version.
   * @param {string} status The raw status string (e.g., "FITTING_ROUND_1").
   * @returns {string} The formatted string (e.g., "Fitting R1").
   */
  function formatServerStatus(status) {
    if (!status || status === "None") {
      return "Idle";
    }

    const roundMatch = status.match(/(\w+)_ROUND_(\d+)/);
    if (roundMatch) {
      const action = roundMatch[1];
      const roundNum = roundMatch[2];
      const capitalizedAction =
        action.charAt(0) + action.slice(1).toLowerCase();
      return `${capitalizedAction} R${roundNum}`;
    }

    switch (status) {
      case "TRAINING_COMPLETE":
        return "Complete";
      case "INITIALIZING":
        return "Initializing";
      case "IDLE":
        return "Idle";
      default:
        return status
          .replace(/_/g, " ")
          .replace(
            /\w\S*/g,
            (txt) => txt.charAt(0).toUpperCase() + txt.substr(1).toLowerCase()
          );
    }
  }

  /**
   * Reads the initial data from the HTML and formats it to prevent flickering.
   */
  function initializeDashboard() {
    const statusEl = document.getElementById("server-status");
    if (statusEl) {
      const initialStatus = statusEl.getAttribute("data-initial-status");
      statusEl.textContent = formatServerStatus(initialStatus);
    }
  }

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

      document.getElementById("server-status").textContent = formatServerStatus(
        metrics.server_status
      );
      document.getElementById("current-round").textContent =
        metrics.current_round || "0";

      const latestAccuracy =
        metrics.latest_accuracy_percent === null
          ? NA_STRING
          : `${metrics.latest_accuracy_percent}%`;
      document.getElementById("latest-accuracy").textContent = latestAccuracy;

      const downloadBtn = document.getElementById("download-button");
      if (downloadBtn) {
        if (metrics.server_status === "TRAINING_COMPLETE") {
          downloadBtn.classList.remove("disabled");
          downloadBtn.setAttribute(
            "title",
            "Download the final aggregated model"
          );
        } else {
          downloadBtn.classList.add("disabled");
          downloadBtn.setAttribute(
            "title",
            "Model is available for download once training is complete"
          );
        }
      }

      const historyTbody = document.getElementById("history-tbody");
      let historyHtml = "";
      if (metrics.training_history && metrics.training_history.length > 0) {
        const reversedHistory = metrics.training_history.slice().reverse();
        reversedHistory.forEach((entry) => {
          historyHtml += `<tr><td>${entry.round}</td><td>${entry.accuracy}%</td></tr>`;
        });
      }
      historyTbody.innerHTML = historyHtml;

      if (
        trainingStatusText &&
        typeof metrics.training_active !== "undefined"
      ) {
        trainingStatusText.textContent = metrics.training_active
          ? `Running (PID ${metrics.training_pid ?? "N/A"})`
          : "Idle";
      }

      if (
        startBtn &&
        stopBtn &&
        typeof metrics.training_active !== "undefined"
      ) {
        startBtn.disabled = !!metrics.training_active;
        stopBtn.disabled = !metrics.training_active;
      }

      if (metrics.training_args) {
        updateInputValue(
          trainingInputs.features,
          metrics.training_args.features
        );
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

  // --- SCRIPT EXECUTION ---
  initializeDashboard(); // Format the initial server-rendered text
  updateDashboard(); // Fetch the first set of live data
  setInterval(updateDashboard, 150); // Poll for new data every 1.5 seconds
});

const NA_STRING = "N/A";

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

    const historyTbody = document.getElementById("history-tbody");
    let historyHtml = "";
    if (metrics.training_history && metrics.training_history.length > 0) {
      metrics.training_history.forEach((entry) => {
        historyHtml += `<tr><td>${entry.round}</td><td>${entry.accuracy}%</td></tr>`;
      });
    }
    historyTbody.innerHTML = historyHtml;
  } catch (error) {
    console.error("Failed to fetch dashboard metrics:", error);
    const container = document.getElementById("dashboard-container");
    if (container) {
      container.innerHTML = `<div class="error-container"><h1>Connection Error</h1><p>Could not fetch live data from the server. Is it running? Retrying automatically...</p></div>`;
    }
  }
}

setInterval(updateDashboard, 200);

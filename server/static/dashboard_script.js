document.addEventListener("DOMContentLoaded", function () {
  const serverStatusEl = document.getElementById("server-status");
  const currentRoundEl = document.getElementById("current-round");
  const latestAccuracyEl = document.getElementById("latest-accuracy");
  const historyTbodyEl = document.getElementById("history-tbody");
  const downloadButtonEl = document.getElementById("download-button");
  const errorBoxEl = document.getElementById("error-box");

  async function updateDashboard() {
    try {
      const response = await fetch("/metrics");
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();

      // Hide error box if we successfully connect
      errorBoxEl.style.display = "none";

      // 1. Update metric cards
      serverStatusEl.textContent = data.server_status || "UNKNOWN";
      currentRoundEl.textContent =
        data.current_round !== undefined ? data.current_round : "N/A";

      const accuracy = data.latest_accuracy_percent;
      latestAccuracyEl.textContent =
        accuracy !== null && accuracy !== undefined ? `${accuracy}%` : "N/A";

      // 2. Update the download button state
      if (data.server_status === "COMPLETE") {
        downloadButtonEl.classList.remove("disabled");
      } else {
        downloadButtonEl.classList.add("disabled");
      }

      // 3. Update the training history table
      historyTbodyEl.innerHTML = ""; // Clear existing rows
      if (data.training_history && data.training_history.length > 0) {
        data.training_history.forEach((entry) => {
          const row = document.createElement("tr");
          row.innerHTML = `
            <td>${entry.round}</td>
            <td>${entry.accuracy}%</td>
          `;
          historyTbodyEl.appendChild(row);
        });
      } else {
        // Optional: show a message when history is empty
        const row = document.createElement("tr");
        row.innerHTML = `<td colspan="2" style="text-align:center;">No training history yet.</td>`;
        historyTbodyEl.appendChild(row);
      }
    } catch (error) {
      console.error("Failed to fetch metrics:", error);
      // Show error box if we fail to connect
      errorBoxEl.style.display = "block";
      // Also disable the button on error
      downloadButtonEl.classList.add("disabled");
    }
  }

  // Fetch data immediately on page load, then every 5 seconds
  updateDashboard();
  setInterval(updateDashboard, 150);
});

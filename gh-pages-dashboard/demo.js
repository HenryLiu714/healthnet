const historyData = [
  { round: 1, accuracy: 71.32, status: "Starting Round 1" },
  { round: 2, accuracy: 78.15, status: "Fitting Round 2" },
  { round: 3, accuracy: 82.67, status: "Evaluating Round 3" },
  { round: 4, accuracy: 85.91, status: "Aggregating" },
  { round: 5, accuracy: 87.42, status: "Training Complete" },
];

const historyTbody = document.getElementById("history-tbody");
const serverStatusEl = document.getElementById("server-status");
const currentRoundEl = document.getElementById("current-round");
const accuracyEl = document.getElementById("latest-accuracy");
const messageEl = document.getElementById("training-message");
const trainingStatusEl = document.getElementById("training-status-text");
const startBtn = document.getElementById("start-training-btn");
const stopBtn = document.getElementById("stop-training-btn");

function renderHistory() {
  if (!historyTbody) {
    return;
  }
  historyTbody.innerHTML = historyData
    .slice()
    .reverse()
    .map((entry) => `
      <tr>
        <td>${entry.round}</td>
        <td>${entry.accuracy.toFixed(2)}%</td>
      </tr>
    `)
    .join("");
}

let animationInterval = null;
let animationIndex = 0;

function renderFrame(index) {
  const frame = historyData[index % historyData.length];
  if (serverStatusEl) {
    serverStatusEl.textContent = frame.status;
  }
  if (currentRoundEl) {
    currentRoundEl.textContent = frame.round;
  }
  if (accuracyEl) {
    accuracyEl.textContent = `${frame.accuracy.toFixed(2)}%`;
  }
}

function startAnimation() {
  if (animationInterval) {
    return;
  }
  animationIndex = 0;
  renderFrame(animationIndex);
  animationInterval = setInterval(() => {
    animationIndex += 1;
    renderFrame(animationIndex);
    if (animationIndex >= historyData.length * 3) {
      stopAnimation("Demo training complete.");
      if (serverStatusEl) {
        serverStatusEl.textContent = "Training Complete";
      }
    }
  }, 1800);
  if (trainingStatusEl) {
    trainingStatusEl.textContent = "Running demo animation";
  }
  showStaticMessage("Demo training started. Real training requires the backend.");
}

function stopAnimation(message = "Demo training paused.") {
  if (animationInterval) {
    clearInterval(animationInterval);
    animationInterval = null;
  }
  if (serverStatusEl) {
    serverStatusEl.textContent = "Idle";
  }
  if (currentRoundEl) {
    currentRoundEl.textContent = "0";
  }
  if (accuracyEl) {
    accuracyEl.textContent = "N/A";
  }
  if (trainingStatusEl) {
    trainingStatusEl.textContent = "Idle (Static Preview)";
  }
  showStaticMessage(message);
}

function showStaticMessage(text) {
  if (messageEl) {
    messageEl.textContent = `${text} GitHub Pages can only host static files, so the FastAPI/Flower backend isn't running here.`;
    messageEl.style.color = "#f48771";
  }
}

if (startBtn) {
  startBtn.addEventListener("click", (event) => {
    event.preventDefault();
    startAnimation();
  });
}

if (stopBtn) {
  stopBtn.addEventListener("click", (event) => {
    event.preventDefault();
    stopAnimation();
  });
}

renderHistory();

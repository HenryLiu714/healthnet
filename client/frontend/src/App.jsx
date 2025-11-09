import React, { useState, useRef, useEffect } from "react";

function App() {
  const [file, setFile] = useState(null);
  const [runId, setRunId] = useState(null);
  const [server, setServer] = useState("localhost:8080");
  const [logs, setLogs] = useState([]);
  const [metrics, setMetrics] = useState([]);
  const wsRef = useRef(null);

  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [inferFile, setInferFile] = useState(null);
  const [predictions, setPredictions] = useState([]);

  useEffect(() => {
    fetchModels();
  }, []);

  async function fetchModels() {
    try {
      const res = await fetch("http://localhost:8000/models");
      const data = await res.json();
      setModels(data.models || []);
    } catch {
      console.error("Failed to load models list");
    }
  }

  function appendLog(entry) {
    setLogs((prev) => [...prev, entry].slice(-500));
  }

  async function upload() {
    if (!file) {
      alert("Select CSV");
      return;
    }
    try {
      const fd = new FormData();
      fd.append("file", file);
      const res = await fetch("http://localhost:8000/upload", {
        method: "POST",
        body: fd,
      });
      const data = await res.json();
      setRunId(data.run_id);
      appendLog({ type: "status", message: "Uploaded: " + data.filename });
    } catch (err) {
      appendLog({ type: "error", message: "Upload failed" });
    }
  }

  async function start() {
    if (!runId) {
      alert("Upload first");
      return;
    }
    try {
      await fetch(
        `http://localhost:8000/start?run_id=${encodeURIComponent(
          runId
        )}&server=${encodeURIComponent(server)}`,
        { method: "POST" }
      );
      appendLog({ type: "status", message: "Started run " + runId });
      startWs(runId);
    } catch (err) {
      appendLog({ type: "error", message: "Failed to start run" });
    }
  }

  function startWs(id) {
    if (wsRef.current) wsRef.current.close();
    const ws = new WebSocket(`ws://localhost:8000/ws/${id}`);
    ws.onopen = () =>
      appendLog({ type: "status", message: "WebSocket connected" });
    ws.onmessage = (event) => {
      try {
        const obj = JSON.parse(event.data);
        if (obj.type === "log") appendLog(obj);
        else if (obj.type === "metrics" || obj.epoch !== undefined)
          setMetrics((m) => [...m, obj]);
        else appendLog(obj);
      } catch {
        appendLog({ type: "raw", message: event.data });
      }
    };
    ws.onclose = () =>
      appendLog({ type: "status", message: "WebSocket closed" });
    ws.onerror = () =>
      appendLog({ type: "error", message: "WebSocket error" });
    wsRef.current = ws;
  }

  const latestMetric = metrics.length ? metrics[metrics.length - 1] : null;

  async function runInference() {
    if (!selectedModel) {
      alert("Select a model run");
      return;
    }
    if (!inferFile) {
      alert("Upload a CSV for inference");
      return;
    }
    const fd = new FormData();
    fd.append("model_run_id", selectedModel);
    fd.append("file", inferFile);
    try {
      const res = await fetch("http://localhost:8000/infer", {
        method: "POST",
        body: fd,
      });
      const data = await res.json();
      setPredictions(data.predictions || []);
    } catch (err) {
      alert("Inference failed");
    }
  }

  const buttonStyle = {
    width: "100%",
    padding: "12px 20px",
    borderRadius: 10,
    border: "none",
    fontWeight: 600,
    color: "#fff",
    background: "linear-gradient(90deg, #EF5D60 0%, #D94B4F 100%)",
    cursor: "pointer",
    transition: "opacity 0.2s ease",
  };

  const cardStyle = {
    flex: 1,
    minWidth: 280,
    backgroundColor: "#fff",
    padding: 24,
    borderRadius: 16,
    boxShadow: "0 2px 16px rgba(0,0,0,0.08)",
  };

  const inputStyle = {
    width: "100%",
    padding: "12px 14px",
    borderRadius: 8,
    border: "1px solid #ddd",
    backgroundColor: "#F9F9F9",
    fontSize: 14,
    boxSizing: "border-box",
  };

  const fileUploadStyle = {
    position: "relative",
    width: "100%",
    marginTop: 10,
  };

  const customFileBtn = {
    display: "inline-block",
    padding: "10px 16px",
    borderRadius: 8,
    backgroundColor: "#54494B",
    color: "#fff",
    fontWeight: 500,
    cursor: "pointer",
    textAlign: "center",
    transition: "background 0.2s ease",
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        backgroundColor: "#F1F7ED",
        fontFamily: "'Inter', sans-serif",
        color: "#54494B",
        padding: "40px 20px",
      }}
    >
      <div style={{ maxWidth: 1200, margin: "0 auto" }}>
        <img
          src="/assets/logo.png"
          alt="Federated Learning Logo"
          style={{
            width: 150, // adjust size as needed
            height: 'auto',
            marginBottom: -30,
            marginLeft: -18
          }}
        />

        {/* TRAINING */}
        <section style={{ marginBottom: 50 }}>
          <h2 style={{ fontSize: 22, fontWeight: 600, marginBottom: 20 }}>
            Training
          </h2>
          <div style={{ display: "flex", gap: 30, flexWrap: "wrap" }}>
            <div style={cardStyle}>
              <label style={{ fontWeight: 600 }}>CSV File</label>
              <div style={fileUploadStyle}>
                <input
                  id="trainFile"
                  type="file"
                  accept=".csv"
                  onChange={(e) => setFile(e.target.files?.[0])}
                  style={{
                    opacity: 0,
                    position: "absolute",
                    left: 0,
                    top: 0,
                    width: "100%",
                    height: "100%",
                    cursor: "pointer",
                  }}
                />
                <label htmlFor="trainFile" style={customFileBtn}>
                  {file ? file.name : "Choose CSV"}
                </label>
              </div>
              <button
                onClick={upload}
                style={{ ...buttonStyle, marginTop: 18 }}
                onMouseOver={(e) => (e.currentTarget.style.opacity = 0.85)}
                onMouseOut={(e) => (e.currentTarget.style.opacity = 1)}
              >
                Upload
              </button>
            </div>

            <div style={cardStyle}>
              <label style={{ fontWeight: 600 }}>Aggregator Server</label>
              <input
                value={server}
                onChange={(e) => setServer(e.target.value)}
                placeholder="localhost:8080"
                style={{ ...inputStyle, marginTop: 10 }}
              />
              <button
                onClick={start}
                disabled={!runId}
                style={{
                  ...buttonStyle,
                  marginTop: 18,
                  opacity: runId ? 1 : 0.5,
                  cursor: runId ? "pointer" : "not-allowed",
                }}
              >
                Start Federated Learning
              </button>
            </div>
          </div>
        </section>

        {/* LOGS */}
        <section style={{ marginBottom: 50 }}>
          <h2 style={{ fontSize: 22, fontWeight: 600, marginBottom: 15 }}>
            Live Logs
          </h2>
          <div
            style={{
              backgroundColor: "#1E1E1E",
              color: "#EAEAEA",
              borderRadius: 12,
              padding: 18,
              fontFamily: "monospace",
              maxHeight: 300,
              overflowY: "auto",
              boxShadow: "0 2px 12px rgba(0,0,0,0.1)",
            }}
          >
            {logs.map((l, i) => (
              <div key={i} style={{ marginBottom: 6, fontSize: 13 }}>
                [{l.type || "log"}] {l.message || JSON.stringify(l)}
              </div>
            ))}
          </div>
        </section>

        {/* METRICS */}
        <section style={{ marginBottom: 50 }}>
          <h2 style={{ fontSize: 22, fontWeight: 600, marginBottom: 15 }}>
            Latest Metrics
          </h2>
          <pre
            style={{
              backgroundColor: "#fff",
              borderRadius: 12,
              padding: 18,
              fontSize: 14,
              border: "1px solid #eee",
              boxShadow: "0 2px 12px rgba(0,0,0,0.04)",
              overflowX: "auto",
            }}
          >
            {latestMetric ? JSON.stringify(latestMetric, null, 2) : "No metrics yet"}
          </pre>
        </section>

        {/* INFERENCE */}
        <section>
          <h2 style={{ fontSize: 22, fontWeight: 600, marginBottom: 20 }}>
            Inference
          </h2>
          <div style={{ display: "flex", gap: 30, flexWrap: "wrap" }}>
            <div style={cardStyle}>
              <label style={{ fontWeight: 600 }}>Select Model Run</label>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                style={{ ...inputStyle, marginTop: 10 }}
              >
                <option value="">Choose...</option>
                {models.map((m) => (
                  <option key={m.run_id} value={m.run_id}>
                    {m.run_id}
                  </option>
                ))}
              </select>

              <label
                style={{ fontWeight: 600, marginTop: 20, display: "block" }}
              >
                Inference CSV
              </label>
              <div style={fileUploadStyle}>
                <input
                  id="inferFile"
                  type="file"
                  accept=".csv"
                  onChange={(e) => setInferFile(e.target.files?.[0])}
                  style={{
                    opacity: 0,
                    position: "absolute",
                    left: 0,
                    top: 0,
                    width: "100%",
                    height: "100%",
                    cursor: "pointer",
                  }}
                />
                <label htmlFor="inferFile" style={customFileBtn}>
                  {inferFile ? inferFile.name : "Choose CSV"}
                </label>
              </div>

              <button
                onClick={runInference}
                style={{ ...buttonStyle, marginTop: 18 }}
                onMouseOver={(e) => (e.currentTarget.style.opacity = 0.85)}
                onMouseOut={(e) => (e.currentTarget.style.opacity = 1)}
              >
                Run Inference
              </button>
            </div>

            <div
              style={{
                ...cardStyle,
                maxHeight: 400,
                overflowY: "auto",
              }}
            >
              <h3 style={{ fontWeight: 600, marginBottom: 10 }}>Predictions</h3>
              {predictions.length === 0
                ? "No predictions yet"
                : predictions.map((p, i) => (
                    <div key={i} style={{ marginBottom: 6 }}>
                      Row {i + 1}: <b>{p}</b>
                    </div>
                  ))}
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}

export default App;
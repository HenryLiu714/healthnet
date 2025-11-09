import React, { useState, useRef, useEffect } from "react";

function App() {
  const [file, setFile] = useState(null);
  const [runId, setRunId] = useState(null);
  const [server, setServer] = useState("localhost:8080");

  const [logs, setLogs] = useState([]);
  const [metrics, setMetrics] = useState([]);

  const wsRef = useRef(null);

  // NEW for inference panel
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
    setLogs(prev => [...prev, entry].slice(-500));
  }

  // -------------------------
  // TRAINING FLOW
  // -------------------------

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
        body: fd
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
      const res = await fetch(
        `http://localhost:8000/start?run_id=${encodeURIComponent(
          runId
        )}&server=${encodeURIComponent(server)}`,
        { method: "POST" }
      );
      const data = await res.json();

      appendLog({ type: "status", message: "Started run " + runId });
      startWs(runId);
    } catch (err) {
      appendLog({ type: "error", message: "Failed to start run" });
    }
  }

  function startWs(id) {
    if (wsRef.current) {
      wsRef.current.close();
    }

    const ws = new WebSocket(`ws://localhost:8000/ws/${id}`);

    ws.onopen = () => {
      appendLog({ type: "status", message: "WebSocket connected" });
    };

    ws.onmessage = event => {
      try {
        const obj = JSON.parse(event.data);

        if (obj.type === "log") {
          appendLog(obj);
        } else if (obj.type === "metrics" || obj.epoch !== undefined) {
          setMetrics(m => [...m, obj]);
        } else {
          appendLog(obj);
        }
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

  // -------------------------
  // INFERENCE FLOW
  // -------------------------

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
        body: fd
      });
      const data = await res.json();
      setPredictions(data.predictions || []);
    } catch (err) {
      alert("Inference failed");
    }
  }

  return (
    <div
      style={{
        padding: 20,
        fontFamily: "sans-serif",
        maxWidth: 1200,
        margin: "auto"
      }}
    >
      <h2 style={{ marginBottom: 5 }}>Federated Learning Client</h2>
      <p style={{ color: "#777", marginTop: 0 }}>
        Upload data, connect, train, and perform inference.
      </p>

      {/* -------------------------------- */}
      {/* TRAINING SECTION */}
      {/* -------------------------------- */}
      <h3>Training</h3>
      <div style={{ display: "flex", gap: 30, marginTop: 20 }}>
        {/* Upload Area */}
        <div style={{ flex: 1 }}>
          <label style={{ fontWeight: 600 }}>CSV File</label>
          <input
            type="file"
            accept=".csv"
            onChange={e => setFile(e.target.files?.[0])}
            style={{ marginTop: 8 }}
          />

          <button
            onClick={upload}
            style={{ marginTop: 12, padding: "6px 12px" }}
          >
            Upload
          </button>
        </div>

        {/* Server Settings */}
        <div style={{ flex: 1 }}>
          <label style={{ fontWeight: 600 }}>Aggregator Server</label>
          <input
            value={server}
            onChange={e => setServer(e.target.value)}
            placeholder="localhost:8080"
            style={{ marginTop: 8, width: "100%", padding: 6 }}
          />

          <button
            onClick={start}
            disabled={!runId}
            style={{ marginTop: 12, padding: "6px 12px" }}
          >
            Start Federated Learning
          </button>
        </div>
      </div>

      <hr style={{ margin: "30px 0" }} />

      {/* Logs */}
      <h3 style={{ marginBottom: 5 }}>Live Logs</h3>

      <div
        style={{
          height: 300,
          overflow: "auto",
          background: "#111",
          color: "#eee",
          padding: 10,
          borderRadius: 6
        }}
      >
        {logs.map((l, i) => (
          <div key={i} style={{ fontFamily: "monospace", fontSize: 12 }}>
            [{l.type || "log"}] {l.message || JSON.stringify(l)}
          </div>
        ))}
      </div>

      {/* Metrics */}
      <h3 style={{ marginTop: 30 }}>Latest Metrics</h3>

      <pre
        style={{
          background: "#f6f6f6",
          padding: 10,
          borderRadius: 6,
          fontSize: 13
        }}
      >
        {latestMetric
          ? JSON.stringify(latestMetric, null, 2)
          : "No metrics yet"}
      </pre>

      {/* -------------------------------- */}
      {/* INFERENCE SECTION */}
      {/* -------------------------------- */}
      <hr style={{ margin: "40px 0" }} />

      <h3>Inference</h3>

      {/* Model Selector */}
      <div style={{ marginTop: 10 }}>
        <label style={{ fontWeight: 600 }}>Select Model Run</label>
        <select
          value={selectedModel}
          onChange={e => setSelectedModel(e.target.value)}
          style={{ marginLeft: 10, padding: 6 }}
        >
          <option value="">Choose...</option>
          {models.map(m => (
            <option key={m.run_id} value={m.run_id}>
              {m.run_id}
            </option>
          ))}
        </select>
      </div>

      {/* CSV for inference */}
      <div style={{ marginTop: 20 }}>
        <label style={{ fontWeight: 600 }}>Inference CSV</label>
        <input
          type="file"
          accept=".csv"
          onChange={e => setInferFile(e.target.files?.[0])}
          style={{ marginLeft: 10 }}
        />
      </div>

      <button
        onClick={runInference}
        style={{ marginTop: 20, padding: "6px 12px" }}
      >
        Run Inference
      </button>

      {/* Predictions */}
      <h3 style={{ marginTop: 30 }}>Predictions</h3>

      <div
        style={{
          background: "#f6f6f6",
          padding: 10,
          borderRadius: 6,
          fontSize: 13,
          maxHeight: 300,
          overflow: "auto"
        }}
      >
        {predictions.length === 0
          ? "No predictions yet"
          : predictions.map((p, i) => (
              <div key={i} style={{ padding: "4px 0" }}>
                Row {i + 1}: <b>{p}</b>
              </div>
            ))}
      </div>
    </div>
  );
}

export default App;

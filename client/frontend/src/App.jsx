// src/App.jsx
import React, {useState, useRef, useEffect} from "react";

function App(){
  const [file, setFile] = useState(null);
  const [runId, setRunId] = useState(null);
  const [server, setServer] = useState("localhost:8080");
  const [logs, setLogs] = useState([]);
  const [metrics, setMetrics] = useState([]); // array of metric objects
  const wsRef = useRef();

  function appendLog(line){
    setLogs(l => [...l, line].slice(-500)); // cap
  }

  async function upload(){
    if(!file) return alert("Select CSV");
    const fd = new FormData();
    fd.append("file", file);
    const res = await fetch("http://localhost:8000/upload", {method:"POST", body:fd});
    const j = await res.json();
    setRunId(j.run_id);
    appendLog({type:"status", message:"Uploaded: " + j.filename});
  }

  async function start(){
    if(!runId) return alert("Upload first");
    const res = await fetch("http://localhost:8000/start?run_id=" + encodeURIComponent(runId) + "&server=" + encodeURIComponent(server), {method:"POST"});
    const j = await res.json();
    appendLog({type:"status", message:"Started run " + runId});
    startWs(runId);
  }

  function startWs(id){
    if(wsRef.current) wsRef.current.close();
    const ws = new WebSocket(`ws://localhost:8000/ws/${id}`);
    ws.onopen = () => appendLog({type:"status", message:"WebSocket connected"});
    ws.onmessage = (ev) => {
      try {
        const obj = JSON.parse(ev.data);
        if(obj.type === "log"){
          appendLog(obj);
        } else if(obj.type === "metrics" || obj.epoch !== undefined){
          setMetrics(m => [...m, obj]);
        } else {
          appendLog(obj);
        }
      } catch(e){
        appendLog({type:"raw", message: ev.data});
      }
    };
    ws.onclose = () => appendLog({type:"status", message:"WebSocket closed"});
    ws.onerror = (e) => appendLog({type:"error", message: "WebSocket error"});
    wsRef.current = ws;
  }

  // Simple chart rendering (textual) — replace with recharts or chart.js for nicer visuals
  const latestMetric = metrics.length ? metrics[metrics.length-1] : null;

  return (
    <div style={{padding:20, fontFamily:"sans-serif", maxWidth:1000, margin:"auto"}}>
      <h2>Federated Learning Runner</h2>
      <div style={{display:"flex", gap:20}}>
        <div style={{flex:1}}>
          <label>CSV file</label><br/>
          <input type="file" accept=".csv" onChange={e=>setFile(e.target.files?.[0])} />
          <div style={{marginTop:10}}>
            <button onClick={upload}>Upload</button>
          </div>
        </div>

        <div style={{flex:1}}>
          <label>Aggregator server:port</label><br/>
          <input value={server} onChange={e=>setServer(e.target.value)} />
          <div style={{marginTop:10}}>
            <button onClick={start} disabled={!runId}>Start Federated Learning</button>
          </div>
        </div>
      </div>

      <hr/>
      <h3>Live logs</h3>
      <div style={{height:300, overflow:"auto", background:"#111", color:"#eee", padding:10, borderRadius:6}}>
        {logs.map((l,i)=>(
          <div key={i} style={{fontFamily:"monospace", fontSize:12}}>
            [{l.type || "log"}] {l.message || JSON.stringify(l)}
          </div>
        ))}
      </div>

      <h3>Latest metrics</h3>
      <pre style={{background:"#f6f6f6", padding:10, borderRadius:6}}>
        {latestMetric ? JSON.stringify(latestMetric, null, 2) : "No metrics yet"}
      </pre>
    </div>
  );
}

export default App;

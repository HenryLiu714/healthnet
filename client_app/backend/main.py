# backend/main.py
import asyncio
import uuid
import os
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import json
import uvicorn

ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs"
RUNS_DIR.mkdir(exist_ok=True)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory run registry
runs = {}  # run_id -> {"status": "idle|running|done|error", "ws_clients": set(), "artifacts": {...}}

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files accepted")
    run_id = str(uuid.uuid4())
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    file_path = run_dir / file.filename
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    runs[run_id] = {"status": "uploaded", "file": str(file_path), "ws_clients": set(), "artifacts": {}}
    return {"run_id": run_id, "filename": file.filename}

@app.post("/start")
async def start_run(run_id: str, server: str):
    if run_id not in runs:
        raise HTTPException(404, "run_id not found")
    run = runs[run_id]
    if run.get("status") == "running":
        raise HTTPException(400, "Run already running")
    run_dir = Path(run["file"]).parent
    run["status"] = "running"
    asyncio.create_task(_run_pipeline(run_id, run_dir, run["file"], server))
    return {"run_id": run_id, "status": "started"}

# WebSocket for logs & metrics
@app.websocket("/ws/{run_id}")
async def websocket_endpoint(websocket: WebSocket, run_id: str):
    await websocket.accept()
    if run_id not in runs:
        await websocket.send_text(json.dumps({"type":"error","message":"run_id not found"}))
        await websocket.close()
        return
    runs[run_id]["ws_clients"].add(websocket)
    try:
        while True:
            # we don't expect incoming messages but keep the socket open
            await websocket.receive_text()
    except WebSocketDisconnect:
        runs[run_id]["ws_clients"].discard(websocket)

# Endpoint to download artifact (e.g. trained model or metrics)
@app.get("/artifact/{run_id}/{name}")
async def get_artifact(run_id: str, name: str):
    run = runs.get(run_id)
    if not run:
        raise HTTPException(404, "run not found")
    path = Path(run.get("artifacts", {}).get(name, ""))
    if not path or not path.exists():
        raise HTTPException(404, "artifact not found")
    return FileResponse(path, filename=path.name)

# Internal helper: run pipeline
async def _run_pipeline(run_id: str, run_dir: Path, csv_path: str, server: str):
    run = runs[run_id]
    try:
        # 1) Run CSVDataset.py
        await _broadcast(run_id, {"type":"status","message":"Preparing dataset"})
        cmd_prep = ["python", "CSVDataset.py", "--input", csv_path, "--out", str(run_dir / "dataset")]
        await _run_subprocess_and_forward(run_id, cmd_prep, parse_metrics=False)

        # 2) Run Flower client (assume run_client.py exists and accepts --server)
        await _broadcast(run_id, {"type":"status","message":"Starting federated learning client"})
        cmd_client = ["python", "run_client.py", "--server", server, "--data", str(run_dir / "dataset")]
        await _run_subprocess_and_forward(run_id, cmd_client, parse_metrics=True)

        run["status"] = "done"
        await _broadcast(run_id, {"type":"status","message":"Run complete", "status":"done"})
    except Exception as e:
        run["status"] = "error"
        await _broadcast(run_id, {"type":"error","message": str(e)})
        raise

# Helper: spawn subprocess, stream stdout to websocket, optionally parse metric JSON lines
async def _run_subprocess_and_forward(run_id: str, cmd, parse_metrics=False):
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    assert process.stdout is not None
    while True:
        line = await process.stdout.readline()
        if not line:
            break
        text = line.decode(errors="ignore").rstrip()
        # If the child process produces JSON metric updates, try to parse them
        sent = False
        if parse_metrics:
            try:
                obj = json.loads(text)
                # assume metrics messages are like {"type":"metrics","epoch":1,"loss":0.5,"acc":0.8}
                await _broadcast(run_id, obj)
                sent = True
            except Exception:
                pass
        if not sent:
            await _broadcast(run_id, {"type":"log","message":text})
    rc = await process.wait()

    await _broadcast(run_id, {"type":"log","message":f"Process exited with code {rc}"})

# Broadcast utility
async def _broadcast(run_id: str, payload):
    for ws in list(runs[run_id]["ws_clients"]):
        try:
            await ws.send_text(json.dumps(payload))
        except Exception:
            runs[run_id]["ws_clients"].discard(ws)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

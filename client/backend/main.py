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
import pandas as pd
import numpy as np
from fastapi import File, UploadFile
from fastapi import Form


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

@app.get("/models")
async def list_models():
    models = []
    for run_dir in RUNS_DIR.iterdir():
        model_path = run_dir / "models" / "final_model.npz"
        meta_path = run_dir / "dataset" / "metadata.json"
        if model_path.exists() and meta_path.exists():
            models.append({
                "run_id": run_dir.name,
                "model": str(model_path),
                "metadata": str(meta_path)
            })
    return {"models": models}

import json

def preprocess_for_inference(df: pd.DataFrame, metadata_path: Path):
    with open(metadata_path, "r") as f:
        meta = json.load(f)

    feature_cols = meta["feature_columns"]
    categorical_feature_cols = meta["categorical_feature_columns"]

    # original columns from training script
    numerical_cols = ['Age', 'BMI', 'Alcohol Consumption', 'Physical Activity', 'Sleep Duration', 'Stress Level']
    categorical_cols = ['Gender', 'Smoking Status', 'Chronic Disease History']

    # fill missing numeric vals
    for col in numerical_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # fill missing categorical vals
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    # scale numeric columns with *training* scale approximated as z-score without centering
    # because actual StandardScaler params aren't saved
    # Instead, we do consistency: center and scale using the new data,
    # then align dims afterwards. This works numerically for trees/NNs.
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df_num = scaler.fit_transform(df[numerical_cols])
    df_num = pd.DataFrame(df_num, columns=numerical_cols)

    # one-hot categorical columns
    df_cat = pd.get_dummies(df[categorical_cols], drop_first=False)

    # combine
    df_final = pd.concat([df_num, df_cat], axis=1)

    # align final columns with training metadata
    for col in feature_cols:
        if col not in df_final.columns:
            df_final[col] = 0.0

    # drop extra columns that training didn't have
    df_final = df_final[feature_cols]

    return df_final.astype(np.float32).values

def load_model_npz(path: Path):
    data = np.load(path, allow_pickle=True)
    return data

@app.post("/infer")
async def infer_with_model(
    model_run_id: str = Form(...),
    file: UploadFile = File(...)
):
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files accepted")

    model_path = RUNS_DIR / model_run_id / "models" / "final_model.npz"
    metadata_path = RUNS_DIR / model_run_id / "dataset" / "metadata.json"

    if not model_path.exists():
        raise HTTPException(404, "model not found")
    if not metadata_path.exists():
        raise HTTPException(404, "metadata.json not found for this run")

    # read CSV
    df = pd.read_csv(file.file)

    # preprocess based on metadata
    X = preprocess_for_inference(df, metadata_path)

    # load trained model
    model = load_model_npz(model_path)

    weights_and_biases = [model[key] for key in model.files]

    # Example: linear layer model
    preds = forward_nn(X, weights_and_biases)

    # if classification, optionally:
    pred_classes = np.argmax(preds, axis=1)

    return {
        "predictions": pred_classes.tolist(),
        "count": len(pred_classes)
    }

def forward_nn(X, weights_and_biases):
    A = X
    num_layers = len(weights_and_biases) // 2  # each layer has W, b
    for i in range(num_layers):
        W = weights_and_biases[2*i]
        b = weights_and_biases[2*i + 1]
        A = A.dot(W.T) + b  # transpose W to align dims
        if i < num_layers - 1:
            # ReLU hidden layers
            A = np.maximum(0, A)
    return A

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

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

import html
import os
import re
import signal
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from env_loader import get_env_int, get_env_str, load_env_file

# --- Configuration ---
SCRIPT_DIR = Path(__file__).resolve().parent
load_env_file(SCRIPT_DIR / ".env")

FASTAPI_HOST = get_env_str("FASTAPI_HOST", "0.0.0.0")
FASTAPI_PORT = get_env_int("FASTAPI_PORT", 8002)
PUBLIC_DASHBOARD_DOMAIN = get_env_str(
    "PUBLIC_DASHBOARD_DOMAIN",
    f"http://{FASTAPI_HOST}:{FASTAPI_PORT}",
)

FLOWER_SERVER_HOST = get_env_str("FLOWER_SERVER_HOST", "0.0.0.0")
FLOWER_SERVER_PORT = get_env_int("FLOWER_SERVER_PORT", 8080)
FLOWER_SERVER_ADDRESS = f"{FLOWER_SERVER_HOST}:{FLOWER_SERVER_PORT}"

LOG_FILE_PATH = str(SCRIPT_DIR / "server.log")
FILTER_KEYWORDS = ["uvicorn.access"]
MODEL_FILE_PATH = SCRIPT_DIR / "final_model.npz"
DEFAULT_TRAINING_ARGS = {
    "features": 15,
    "classes": 3,
    "rounds": 3,
    "clients": 1,
}
_DEFAULT_TEST_PATH = SCRIPT_DIR.parent / "test" / "server.py"
TRAINING_SCRIPT = _DEFAULT_TEST_PATH if _DEFAULT_TEST_PATH.exists() else SCRIPT_DIR / "server.py"
TRAINING_WORKDIR = SCRIPT_DIR  # Keep logs/models aligned with dashboard expectations

STATIC_DIR = SCRIPT_DIR / "static"
TEMPLATE_DIR = SCRIPT_DIR / "templates"

training_process: Optional[subprocess.Popen] = None
training_lock = threading.Lock()
current_training_args = DEFAULT_TRAINING_ARGS.copy()


class TrainingConfig(BaseModel):
    features: int = Field(DEFAULT_TRAINING_ARGS["features"], ge=1)
    classes: int = Field(DEFAULT_TRAINING_ARGS["classes"], ge=1)
    rounds: int = Field(DEFAULT_TRAINING_ARGS["rounds"], ge=1)
    clients: int = Field(DEFAULT_TRAINING_ARGS["clients"], ge=1)


def _build_training_command(config: TrainingConfig) -> List[str]:
    return [
        "python3",
        str(TRAINING_SCRIPT),
        "--features",
        str(config.features),
        "--classes",
        str(config.classes),
        "--rounds",
        str(config.rounds),
        "--clients",
        str(config.clients),
    ]


def _cleanup_training_process_locked() -> None:
    """Reset process reference if the subprocess already finished."""
    global training_process
    if training_process and training_process.poll() is not None:
        training_process = None


def _get_training_state() -> Dict[str, Any]:
    with training_lock:
        _cleanup_training_process_locked()
        running = training_process is not None
        pid = training_process.pid if running else None
        return {
            "training_active": running,
            "training_pid": pid,
            "training_args": current_training_args.copy(),
            "training_command": (
                _build_training_command(TrainingConfig(**current_training_args))
                if running
                else []
            ),
        }


def _terminate_process(proc: subprocess.Popen) -> None:
    """Attempt a graceful stop and fall back to kill if needed."""
    try:
        if os.name != "nt":
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        else:
            proc.terminate()
    except ProcessLookupError:
        return

app = FastAPI()

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))


# --- MODIFIED: The DASHBOARD_TEMPLATE string is now removed ---

# --- UNCHANGED: The LOG_VIEW_TEMPLATE remains as it was ---
LOG_VIEW_TEMPLATE = """
<!DOCTYPE html><html><head><title>Flower Server Log Viewer</title><meta http-equiv="refresh" content="5">
<style>
    body {font-family:monospace; background-color:#2b2b2b; color:#a9b7c6;}
    .container {width:90%; margin:auto;}
    h1 {color:#cc7832;}
    .header {padding:10px; margin-bottom:10px;}
    .header a {color:#6a8759;}
    .log-entry {padding:4px; border-bottom:1px solid #323232;}
    .error {color:#ff6b68;}
</style></head>
<body><div class="header">{toggle_link}</div><div class="container"><h1>Flower Server Log Viewer</h1><h3>Displaying: {log_view_mode} from code>{log_file}</code></h3><div>{log_content}</div></div></body></html>
"""


# --- UNCHANGED: The log parsing function is identical ---
def parse_log_file() -> Dict[str, Any]:
    try:
        with open(LOG_FILE_PATH, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return {"error": f"Log file not found at '{LOG_FILE_PATH}'. Is the Flower server running?"}
    except Exception as e:
        return {"error": f"An unexpected error occurred while reading the file: {e}"}

    round_regex = re.compile(r"\[ROUND (\d+)\]")
    aggregated_accuracy_regex = re.compile(r"Server: Aggregated weighted accuracy: ([\d.]+)")
    configure_fit_regex = re.compile(r"configure_fit: strategy sampled")
    run_finished_regex = re.compile(r"Run finished")

    server_status = "INITIALIZING"
    current_round = 0
    latest_accuracy = None
    history_dict = {}

    for line in lines:
        if round_regex.search(line):
            match = round_regex.search(line)
            current_round = int(match.group(1))
            server_status = f"STARTING_ROUND_{current_round}"
        elif configure_fit_regex.search(line):
            server_status = f"FITTING_ROUND_{current_round}"
        elif aggregated_accuracy_regex.search(line):
            match = aggregated_accuracy_regex.search(line)
            accuracy_value = round(float(match.group(1)) * 100, 2)
            latest_accuracy = accuracy_value
            server_status = f"EVALUATING_ROUND_{current_round}"
            if current_round > 0:
                history_dict[current_round] = accuracy_value
        elif run_finished_regex.search(line):
            server_status = "TRAINING_COMPLETE"
    
    training_history = [{"round": r, "accuracy": acc} for r, acc in sorted(history_dict.items())]

    return {
        "server_status": server_status,
        "current_round": current_round,
        "latest_accuracy_percent": latest_accuracy,
        "training_history": training_history
    }


# --- MODIFIED: The /dashboard endpoint now uses the Jinja2 template ---
@app.get("/dashboard", response_class=HTMLResponse)
def get_dashboard(request: Request):
    metrics = parse_log_file()
    if "error" in metrics:
        return HTMLResponse(content=f"<h1>Error</h1><p>{metrics['error']}</p>", status_code=500)

    metrics.update(_get_training_state())
    # Use the template response to render dashboard.html
    return templates.TemplateResponse("dashboard.html", {"request": request, "metrics": metrics})


# --- UNCHANGED: The /metrics endpoint is identical ---
@app.get("/metrics", response_class=JSONResponse)
def get_metrics_api():
    metrics = parse_log_file()
    if "error" in metrics:
        raise HTTPException(status_code=500, detail=metrics["error"])
    metrics.update(_get_training_state())
    return metrics


@app.post("/training/start", response_class=JSONResponse)
def start_training(config: TrainingConfig):
    if not TRAINING_SCRIPT.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Training script not found at {TRAINING_SCRIPT}.",
        )

    with training_lock:
        global training_process, current_training_args
        _cleanup_training_process_locked()
        if training_process is not None:
            raise HTTPException(status_code=409, detail="Training is already running.")

        command = _build_training_command(config)
        popen_kwargs: Dict[str, Any] = {
            "cwd": TRAINING_WORKDIR,
            "stderr": subprocess.STDOUT,
        }
        if os.name != "nt":
            popen_kwargs["preexec_fn"] = os.setsid
        else:
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]

        try:
            training_process = subprocess.Popen(command, **popen_kwargs)
            current_training_args = config.dict()
        except Exception as exc:
            training_process = None
            raise HTTPException(status_code=500, detail=f"Failed to start training: {exc}") from exc

        return {
            "status": "started",
            "pid": training_process.pid,
            "command": command,
            "args": current_training_args,
        }


@app.post("/training/stop", response_class=JSONResponse)
def stop_training():
    with training_lock:
        global training_process
        _cleanup_training_process_locked()
        if training_process is None:
            raise HTTPException(status_code=409, detail="No active training process to stop.")

        proc = training_process

    # Terminate outside the lock to avoid blocking other requests during wait.
    _terminate_process(proc)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    finally:
        with training_lock:
            if training_process is proc:
                training_process = None

    return {"status": "stopped"}


@app.get("/training/status", response_class=JSONResponse)
def training_status():
    return _get_training_state()


# --- UNCHANGED: The /log_view endpoint is identical ---
@app.get("/log_view", response_class=HTMLResponse)
def get_log_website(request: Request):
    show_all = request.query_params.get("show") == "all"
    log_html_entries = ""
    try:
        with open(LOG_FILE_PATH, "r") as f:
            all_logs = f.readlines()
    except FileNotFoundError:
        log_html_entries = f'<div class="log-entry error">ERROR: Log file not found.</div>'
        all_logs = []

    if not show_all:
        logs_to_display = [log for log in all_logs if not any(kw in log for kw in FILTER_KEYWORDS)]
        log_view_mode = "Filtered"
        toggle_link = '<a href="/log_view?show=all">Show All Logs</a>'
    else:
        logs_to_display = all_logs
        log_view_mode = "All (Raw)"
        toggle_link = '<a href="/log_view">Show Filtered Logs</a>'

    logs_to_display.reverse()
    log_html_entries += "".join(f'<div class="log-entry">{html.escape(log)}</div>' for log in logs_to_display)
    
    return LOG_VIEW_TEMPLATE.format(log_content=log_html_entries, log_view_mode=log_view_mode, toggle_link=toggle_link, log_file=LOG_FILE_PATH)


# --- UNCHANGED: The root endpoint is identical ---
@app.get("/", response_class=HTMLResponse)
def root():
    return f"""
    <html><head><title>Flower App</title></head><body style="font-family:sans-serif;background-color:#1e1e1e;color:#d4d4d4;text-align:center;padding-top:50px;">
    <h1>Federated Learning Monitor</h1><p><a href="/dashboard" style="font-size:1.2em;color:#4ec9b0;">Go to Live Dashboard</a></p><p><a href="/log_view" style="color:#9cdcfe;">View Raw Logs</a></p><p><a href="/metrics" style="color:#9cdcfe;">Access Metrics API (JSON)</a></p>
    <p style="margin-top:30px;color:#9cdcfe;">Public dashboard URL: {PUBLIC_DASHBOARD_DOMAIN}</p>
    </body></html>
    """

@app.get("/download_model")
async def download_model():
    """
    This endpoint allows clients to download the final aggregated model.
    It returns the file if it exists, otherwise returns a 404 error.
    """
    if not MODEL_FILE_PATH.exists():
        # Raise an HTTPException, which FastAPI turns into a proper 404 response
        raise HTTPException(
            status_code=404,
            detail="Model file not found. The training process may not be complete yet."
        )

    # Use FileResponse to efficiently stream the file to the client
    return FileResponse(
        path=MODEL_FILE_PATH,
        filename="model.npz",
        media_type="application/octet-stream" # A generic type for binary files
    )

# --- UNCHANGED: The main execution block is identical ---
if __name__ == "__main__":
    print(f"Starting web app. Go to {PUBLIC_DASHBOARD_DOMAIN} for options.")
    print(f"Reading logs from: {LOG_FILE_PATH}")
    print(f"Flower server address (for reference): {FLOWER_SERVER_ADDRESS}")
    uvicorn.run(app, host=FASTAPI_HOST, port=FASTAPI_PORT)

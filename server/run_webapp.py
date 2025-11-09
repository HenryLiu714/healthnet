import html
import uvicorn
import re
from typing import Dict, Any
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
# --- MODIFIED: Add new imports for templating and static files ---
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

# --- Configuration ---
SCRIPT_DIR = Path(__file__).resolve().parent
LOG_FILE_PATH = str(SCRIPT_DIR / "server.log")
FILTER_KEYWORDS = ["uvicorn.access"]
MODEL_FILE_PATH = SCRIPT_DIR / "final_model.npz"

app = FastAPI()

# --- MODIFIED: Set up static file and template directories ---
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


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

    # Use the template response to render dashboard.html
    return templates.TemplateResponse("dashboard.html", {"request": request, "metrics": metrics})


# --- UNCHANGED: The /metrics endpoint is identical ---
@app.get("/metrics", response_class=JSONResponse)
def get_metrics_api():
    metrics = parse_log_file()
    if "error" in metrics:
        raise HTTPException(status_code=500, detail=metrics["error"])
    return metrics


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
    return """
    <html><head><title>Flower App</title></head><body style="font-family:sans-serif;background-color:#1e1e1e;color:#d4d4d4;text-align:center;padding-top:50px;">
    <h1>Federated Learning Monitor</h1><p><a href="/dashboard" style="font-size:1.2em;color:#4ec9b0;">Go to Live Dashboard</a></p><p><a href="/log_view" style="color:#9cdcfe;">View Raw Logs</a></p><p><a href="/metrics" style="color:#9cdcfe;">Access Metrics API (JSON)</a></p>
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
    print("Starting web app. Go to http://127.0.0.1:8000 for options.")
    print(f"Reading logs from: {LOG_FILE_PATH}")
    uvicorn.run(app, host="0.0.0.0", port=8002)
import html
import uvicorn
import re
import logging
from typing import Dict, Any
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

# --- Configuration ---
SCRIPT_DIR = Path(__file__).resolve().parent
LOG_FILE_PATH = str(SCRIPT_DIR / "server.log")
FILTER_KEYWORDS = ["uvicorn.access"]

app = FastAPI()

# --- (HTML Templates are unchanged) ---

DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Flower Training Dashboard</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; background-color: #1e1e1e; color: #d4d4d4; margin: 0; padding: 20px; }
        .container { max-width: 900px; margin: auto; }
        h1 { color: #4ec9b0; border-bottom: 2px solid #4ec9b0; padding-bottom: 10px; }
        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .metric-card { background-color: #252526; border-left: 5px solid #9cdcfe; padding: 20px; border-radius: 5px; }
        .metric-card h3 { margin-top: 0; color: #9cdcfe; }
        .metric-card p { font-size: 2em; margin: 0; font-weight: bold; color: #ffffff; }
        .history-table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        .history-table th, .history-table td { padding: 12px; border: 1px solid #3c3c3c; text-align: left; }
        .history-table th { background-color: #3c3c3c; color: #4ec9b0; }
        .error-container { background-color: #5A2D2D; border: 1px solid #f48771; padding: 20px; border-radius: 5px; }
        .error-container h1 { color: #f48771; border-bottom-color: #f48771; }
        .status-badge { background-color: #ce9178; color: #1e1e1e; padding: 5px 10px; border-radius: 12px; font-size: 0.8em; font-weight: bold; }
    </style>
</head>
<body>
    <div class="container" id="dashboard-container">
        {content}
    </div>

    <script>
        const NA_STRING = "N/A";

        async function updateDashboard() {
            try {
                const response = await fetch('/metrics', { cache: 'no-store' });
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                const metrics = await response.json();

                if (!document.getElementById('server-status')) {
                    location.reload();
                    return;
                }

                document.getElementById('server-status').textContent = metrics.server_status || NA_STRING;
                document.getElementById('current-round').textContent = metrics.current_round || "0";
                
                const latestAccuracy = metrics.latest_accuracy_percent === null ? NA_STRING : `${metrics.latest_accuracy_percent}%`;
                const finalAccuracy = metrics.final_accuracy_percent === null ? NA_STRING : `${metrics.final_accuracy_percent}%`;

                document.getElementById('latest-accuracy').textContent = latestAccuracy;
                document.getElementById('final-accuracy').textContent = finalAccuracy;

                const historyTbody = document.getElementById('history-tbody');
                let historyHtml = "";
                if (metrics.training_history && metrics.training_history.length > 0) {
                    metrics.training_history.forEach(entry => {
                        historyHtml += `<tr><td>${entry.round}</td><td>${entry.accuracy}%</td></tr>`;
                    });
                }
                historyTbody.innerHTML = historyHtml;

            } catch (error) {
                console.error("Failed to fetch dashboard metrics:", error);
                const container = document.getElementById('dashboard-container');
                container.innerHTML = `<div class="error-container"><h1>Connection Error</h1><p>Could not fetch live data from the server. Is it running? Retrying automatically...</p></div>`;
            }
        }

        setInterval(updateDashboard, 500);
    </script>
</body>
</html>
"""

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
<body><div class="header">{toggle_link}</div><div class="container"><h1>Flower Server Log Viewer</h1><h3>Displaying: {log_view_mode} from <code>{log_file}</code></h3><div>{log_content}</div></div></body></html>
"""

def parse_log_file() -> Dict[str, Any]:
    try:
        with open(LOG_FILE_PATH, "r") as f:
            # Read the entire file content for multi-line parsing
            file_content = f.read()
    except FileNotFoundError:
        return {"error": f"Log file not found at '{LOG_FILE_PATH}'. Is the Flower server running?"}
    except Exception as e:
        return {"error": f"An unexpected error occurred while reading the file: {e}"}

    # --- Part 1: Parse per-round metrics line-by-line ---
    lines = file_content.splitlines()
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

    # --- Part 2: Parse final accuracy from the entire file content ---
    final_accuracy = None
    # Use re.DOTALL to allow `.` to match newlines for multi-line parsing
    summary_accuracy_regex = re.compile(r"'accuracy':\s*(\[.*?\])", re.DOTALL)
    match = summary_accuracy_regex.search(file_content)
    if match:
        try:
            list_string = match.group(1)
            # The captured string is a valid Python literal, so eval is safe here
            accuracy_list = eval(list_string)
            if accuracy_list:
                # Get the accuracy value from the last tuple, e.g., (20, 0.305)
                final_accuracy_float = accuracy_list[-1][1]
                final_accuracy = round(final_accuracy_float * 100, 2)
        except (SyntaxError, IndexError, TypeError) as e:
            # This handles cases where the log file is incomplete
            logging.warning(f"Could not parse final accuracy from summary block: {e}")

    return {
        "server_status": server_status,
        "current_round": current_round,
        "latest_accuracy_percent": latest_accuracy,
        "final_accuracy_percent": final_accuracy,
        "training_history": training_history
    }


# --- (The rest of the file is unchanged) ---
@app.get("/dashboard", response_class=HTMLResponse)
def get_dashboard():
    metrics = parse_log_file()
    if "error" in metrics:
        error_content = f'<div class="error-container"><h1>Error</h1><p>{metrics["error"]}</p></div>'
        final_html = DASHBOARD_TEMPLATE.replace('{content}', error_content)
        return HTMLResponse(content=final_html)

    latest_acc_str = metrics.get("latest_accuracy_percent", "N/A")
    if latest_acc_str != "N/A":
        latest_acc_str = f"{latest_acc_str}%"

    final_acc_str = metrics.get("final_accuracy_percent", "N/A")
    if final_acc_str != "N/A":
        final_acc_str = f"{final_acc_str}%"

    history_rows_html = "".join(f"<tr><td>{entry['round']}</td><td>{entry['accuracy']}%</td></tr>" for entry in metrics["training_history"])
    dashboard_content = f"""
        <h1>Federated Learning Dashboard</h1>
        <div class="metrics-grid">
            <div class="metric-card"><h3>Server Status</h3><p><span id="server-status" class="status-badge">{metrics["server_status"]}</span></p></div>
            <div class="metric-card"><h3>Current Round</h3><p id="current-round">{metrics["current_round"]}</p></div>
            <div class="metric-card"><h3>Latest Accuracy</h3><p id="latest-accuracy">{latest_acc_str}</p></div>
            <div class="metric-card"><h3>Final Accuracy</h3><p id="final-accuracy">{final_acc_str}</p></div>
        </div>
        <h2>Training History</h2>
        <table class="history-table">
            <thead><tr><th>Round</th><th>Accuracy (%)</th></tr></thead>
            <tbody id="history-tbody">{history_rows_html}</tbody>
        </table>
    """
    final_html = DASHBOARD_TEMPLATE.replace('{content}', dashboard_content)
    return HTMLResponse(content=final_html)

@app.get("/metrics", response_class=JSONResponse)
def get_metrics_api():
    metrics = parse_log_file()
    if "error" in metrics:
        raise HTTPException(status_code=500, detail=metrics["error"])
    return metrics

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

@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <html><head><title>Flower App</title></head><body style="font-family:sans-serif;background-color:#1e1e1e;color:#d4d4d4;text-align:center;padding-top:50px;">
    <h1>Federated Learning Monitor</h1><p><a href="/dashboard" style="font-size:1.2em;color:#4ec9b0;">Go to Live Dashboard</a></p><p><a href="/log_view" style="color:#9cdcfe;">View Raw Logs</a></p><p><a href="/metrics" style="color:#9cdcfe;">Access Metrics API (JSON)</a></p>
    </body></html>
    """

if __name__ == "__main__":
    print("Starting web app. Go to http://127.0.0.1:8000 for options.")
    print(f"Reading logs from: {LOG_FILE_PATH}")
    uvicorn.run(app, host="0.0.0.0", port=8002)
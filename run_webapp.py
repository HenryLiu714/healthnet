import html
import uvicorn
from typing import List
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

# --- Configuration ---
LOG_FILE_PATH = "server.log"
FILTER_KEYWORDS = ["uvicorn.access"]

app = FastAPI()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
    <head>
        <title>Flower Server Log Viewer</title>
        <meta http-equiv="refresh" content="5">
        <style>
            body {{ font-family: 'Courier New', monospace; background-color: #2b2b2b; color: #a9b7c6; font-size: 14px; margin: 0; padding: 0; }}
            .container {{ width: 95%; margin: auto; }}
            h1 {{ color: #cc7832; }}
            .header {{ background-color: #3c3f41; padding: 10px 2.5%; margin-bottom: 20px; border-bottom: 2px solid #cc7832; }}
            .header a {{ color: #6a8759; text-decoration: none; font-weight: bold; }}
            .log-entry {{ padding: 4px; border-bottom: 1px solid #3c3f41; white-space: pre-wrap; }}
            .error {{ color: #ff6b68; }}
        </style>
    </head>
    <body>
        <div class="header">
            {toggle_link}
        </div>
        <div class="container">
            <h1>Flower Server Log Viewer</h1>
            <h3>Displaying: {log_view_mode} from <code>{log_file}</code></h3>
            <div>
                {log_content}
            </div>
        </div>
    </body>
</html>
"""

# --- NEW: API Endpoint to return raw logs as JSON ---
@app.get("/log", response_class=JSONResponse)
def get_logs_api():
    """
    Reads the log file and returns all log entries as a JSON array of strings.
    """
    try:
        with open(LOG_FILE_PATH, "r") as f:
            # .strip() removes trailing newline characters for a cleaner API response
            logs = [line.strip() for line in f.readlines()]
        return logs
    except FileNotFoundError:
        # Return a standard 404 error if the log file doesn't exist
        raise HTTPException(
            status_code=404,
            detail=f"Log file not found at '{LOG_FILE_PATH}'. The Flower server may not be running yet."
        )


@app.get("/log_view", response_class=HTMLResponse)
def get_log_website(request: Request):
    """
    Serves a self-refreshing HTML page by reading the log file.
    """
    show_all = request.query_params.get("show") == "all"
    log_html_entries = ""
    
    try:
        # Read all lines from the log file
        with open(LOG_FILE_PATH, "r") as f:
            all_logs = f.readlines()
            
    except FileNotFoundError:
        log_html_entries = f'<div class="log-entry error">ERROR: Log file not found at "{LOG_FILE_PATH}". Have you run the Flower server yet?</div>'
        all_logs = []

    # Determine which logs to display
    if not show_all:
        logs_to_display = [log for log in all_logs if not any(kw in log for kw in FILTER_KEYWORDS)]
        log_view_mode = "Filtered"; toggle_link = '<a href="/log_view?show=all">Show All Logs</a>'
    else:
        logs_to_display = all_logs; log_view_mode = "All (Raw)"; toggle_link = '<a href="/log_view">Show Filtered Logs</a>'
    
    # Reverse to show newest logs first and format for HTML
    logs_to_display.reverse()
    log_html_entries += "".join(f'<div class="log-entry">{html.escape(log)}</div>' for log in logs_to_display)
    
    # Inject content into the template
    return HTML_TEMPLATE.format(
        log_content=log_html_entries,
        log_view_mode=log_view_mode,
        toggle_link=toggle_link,
        log_file=LOG_FILE_PATH
    )

# --- NEW: Root endpoint to guide users ---
@app.get("/", response_class=HTMLResponse)
def root():
    """
    Provides simple navigation to the available endpoints.
    """
    return """
    <html>
        <head><title>Log Viewer App</title></head>
        <body style="font-family: sans-serif; background-color: #2b2b2b; color: #a9b7c6; text-align: center; padding-top: 50px;">
            <h1>Log Viewer Application</h1>
            <p>Two endpoints are available:</p>
            <p><a href="/log_view" style="color: #6a8759;">/log_view</a> - A web-based log viewer with auto-refresh.</p>
            <p><a href="/log" style="color: #6a8759;">/log</a> - A JSON API endpoint for programmatic access to the logs.</p>
        </body>
    </html>
    """


if __name__ == "__main__":
    print("Starting web app. Available endpoints:")
    print("  - Web UI:    http://127.0.0.1:8000/log_view")
    print("  - JSON API:  http://127.0.0.1:8000/log")
    uvicorn.run(app, host="0.0.0.0", port=8000)
import os
import csv
import logging
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

logger = logging.getLogger("Andvari.Auditor")

app = FastAPI(title="Andvari Rapid Review UI")

# In a real deployment, these paths would be passed via environment variables or a config file
OUTPUT_DIR = "./data/output"
CSV_PATH = os.path.join(OUTPUT_DIR, "verified_candidates.csv")
FINAL_CSV_PATH = os.path.join(OUTPUT_DIR, "final_deployment_targets.csv")
THUMB_DIR = os.path.join(OUTPUT_DIR, "thumbnails")

# Mount the thumbnails directory so the web server can display the images
os.makedirs(THUMB_DIR, exist_ok=True)
app.mount("/thumbnails", StaticFiles(directory=THUMB_DIR), name="thumbnails")

# Minimal HTML template baked directly into the script to avoid extra file dependencies
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Andvari Auditor</title>
    <style>
        body { font-family: sans-serif; background-color: #1e1e1e; color: #fff; text-align: center; margin-top: 50px; }
        img { max-width: 512px; border: 2px solid #555; border-radius: 8px; margin-bottom: 20px; }
        .data-panel { background-color: #2d2d2d; padding: 20px; border-radius: 8px; display: inline-block; text-align: left; margin-bottom: 20px; }
        .btn { padding: 15px 30px; font-size: 18px; cursor: pointer; border: none; border-radius: 5px; margin: 0 10px; color: white; font-weight: bold; }
        .btn-approve { background-color: #2e7d32; }
        .btn-approve:hover { background-color: #1b5e20; }
        .btn-reject { background-color: #c62828; }
        .btn-reject:hover { background-color: #b71c1c; }
    </style>
</head>
<body>
    <h1>Candidate Review ({{ current_idx + 1 }} / {{ total }})</h1>
    
    {% if candidate %}
        <img src="/thumbnails/{{ candidate['Thumbnail'] }}" alt="Meteorite Candidate">
        <br>
        <div class="data-panel">
            <strong>ID:</strong> {{ candidate['ID'] }} <br>
            <strong>Confidence:</strong> {{ candidate['Confidence'] }} <br>
            <strong>Lat/Lon:</strong> {{ candidate['Latitude'] }}, {{ candidate['Longitude'] }} <br>
            <strong>Source:</strong> {{ candidate['Parent_Image'] }}
        </div>
        <br>
        <form action="/submit" method="post">
            <input type="hidden" name="candidate_id" value="{{ candidate['ID'] }}">
            <button type="submit" name="decision" value="approve" class="btn btn-approve">APPROVE (Deploy)</button>
            <button type="submit" name="decision" value="reject" class="btn btn-reject">REJECT (Junk)</button>
        </form>
    {% else %}
        <h2>All candidates reviewed!</h2>
        <p>The final deployment list has been saved to: <br> <code>{{ final_csv }}</code></p>
    {% endif %}
</body>
</html>
"""

# We use a hacky in-memory state for simplicity in the field
class ReviewState:
    def __init__(self):
        self.candidates = []
        self.current_idx = 0
        self.load_data()
        
    def load_data(self):
        if not os.path.exists(CSV_PATH):
            return
        with open(CSV_PATH, 'r') as f:
            reader = csv.DictReader(f)
            self.candidates = list(reader)
            
    def save_approval(self, candidate_id):
        candidate = next((c for c in self.candidates if c['ID'] == candidate_id), None)
        if candidate:
            file_exists = os.path.exists(FINAL_CSV_PATH)
            with open(FINAL_CSV_PATH, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=candidate.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(candidate)

state = ReviewState()
templates = Jinja2Templates(directory=".")

# Write the template to a temporary file so Jinja can render it
with open("template.html", "w") as f:
    f.write(HTML_TEMPLATE)

@app.get("/", response_class=HTMLResponse)
async def review_ui(request: Request):
    """Renders the main review interface."""
    candidate = None
    if state.current_idx < len(state.candidates):
        candidate = state.candidates[state.current_idx]
        # Fix path formatting for the web server by retaining only the filename since it mounts to /thumbnails
        candidate['Thumbnail'] = os.path.basename(candidate['Thumbnail'])
        
    return templates.TemplateResponse(request, "template.html", {
        "candidate": candidate,
        "current_idx": state.current_idx,
        "total": len(state.candidates),
        "final_csv": FINAL_CSV_PATH
    })

@app.post("/submit")
async def process_decision(candidate_id: str = Form(...), decision: str = Form(...)):
    """Handles the user's click and advances the queue."""
    if decision == "approve":
        state.save_approval(candidate_id)
        logger.info(f"Candidate {candidate_id} APPROVED for recovery.")
    else:
        logger.info(f"Candidate {candidate_id} REJECTED.")
        
    state.current_idx += 1
    return RedirectResponse(url="/", status_code=303)

def launch_auditor():
    """Starts the local web server."""
    logger.info("Auditor Agent online. Open a browser to http://127.0.0.1:8000")
    # uvicorn must be run programmatically if called from another script
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")

if __name__ == "__main__":
    launch_auditor()

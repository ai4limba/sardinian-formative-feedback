"""
Web application: a writing environment with formative spelling feedback.

Serves a single page (src/app/static) and three endpoints:
    GET  /api/config    interface texts, feedback categories and default threshold
    POST /api/analyze   issues found in a text, with explained suggestions
    POST /api/session   saves the record of a writing session on this computer
"""

import json
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.config import SESSIONS_DIR
from src.feedback.categorize import build_issues
from src.feedback.messages import CATEGORIES, CONTEXT_FIT, PATTERNS, UI
from src.pipeline.loader import load_spellchecker
from src.utils import timestamp

STATIC_DIR = Path(__file__).parent / "static"
MAX_TEXT_LENGTH = 20_000
MAX_SESSION_BYTES = 2_000_000
LINE_CACHE_SIZE = 2_000

spellchecker = None
# The models are shared: one analysis at a time
model_lock = threading.Lock()
# Lines already analysed, so that typing in one paragraph does not re-analyse the others
line_cache = OrderedDict()


@asynccontextmanager
async def lifespan(app):
    global spellchecker
    print("Loading the spellchecker models...")
    spellchecker = load_spellchecker()
    print("Ready.")
    yield


app = FastAPI(title="Sardinian Formative Feedback", lifespan=lifespan)


class AnalyzeRequest(BaseModel):
    text: str = Field(max_length=MAX_TEXT_LENGTH)
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)


def analyze_line(line, threshold):
    """ Issues of a single line, with positions relative to the line. """
    key = (line, threshold)
    if key in line_cache:
        line_cache.move_to_end(key)
        return line_cache[key]

    spellchecker.detection_threshold = threshold
    issues = build_issues(spellchecker.process_text(line), spellchecker.vocabulary)

    line_cache[key] = issues
    if len(line_cache) > LINE_CACHE_SIZE:
        line_cache.popitem(last=False)
    return issues


@app.get("/api/config")
def get_config():
    # The lock makes sure the threshold is not read while an analysis is temporarily changing it
    with model_lock:
        threshold = spellchecker.detection_threshold
    return {
        "strings": UI,
        "categories": CATEGORIES,
        "patterns": PATTERNS,
        "context_fit": CONTEXT_FIT,
        "threshold": threshold
    }


@app.post("/api/analyze")
def analyze(request: AnalyzeRequest):
    start_time = time.perf_counter()
    issues = []
    words = 0

    with model_lock:
        default_threshold = spellchecker.detection_threshold
        threshold = request.threshold if request.threshold is not None else default_threshold
        try:
            # Each line is analysed on its own: the detector reads at most 512 tokens at a time
            offset = 0
            for line in request.text.split("\n"):
                tokens = spellchecker.split_text(line)
                if tokens:
                    words += sum(1 for token in tokens if token.replace("'", "").replace("-", "").isalpha())
                    for issue in analyze_line(line, threshold):
                        issues.append({**issue, "start": issue["start"] + offset, "end": issue["end"] + offset})
                offset += len(line) + 1
        finally:
            spellchecker.detection_threshold = default_threshold

    return {
        "issues": issues,
        "words": words,
        "latency_ms": round((time.perf_counter() - start_time) * 1000, 2)
    }


@app.post("/api/session")
def save_session(session: dict):
    """ Saves the record of a writing session (text, feedback events, summary) as a local JSON file. """
    content = json.dumps(session, ensure_ascii=False, indent=2)
    if len(content.encode("utf-8")) > MAX_SESSION_BYTES:
        raise HTTPException(status_code=413, detail="Session record too large.")

    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = SESSIONS_DIR / f"session_{timestamp()}.json"
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return {"saved_to": f"{SESSIONS_DIR.name}/{path.name}"}


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

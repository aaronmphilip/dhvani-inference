"""DHVANI inference API for Hugging Face Spaces."""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from analyzer import VoiceAnalyzer

ALLOWED = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"}
MAX_BYTES = 20 * 1024 * 1024

app = FastAPI(title="DHVANI Inference", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

analyzer = VoiceAnalyzer()


@app.get("/")
@app.get("/health")
def health():
    return {
        "status": "ok",
        "product": "DHVANI",
        "mode": "hf-space",
        "version": "0.1.0",
    }


@app.post("/analyze")
async def analyze(audio: UploadFile = File(...)):
    if not audio.filename:
        raise HTTPException(status_code=400, detail="Empty filename.")

    suffix = Path(audio.filename).suffix.lower()
    if suffix and suffix not in ALLOWED:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    data = await audio.read()
    if not data:
        raise HTTPException(status_code=400, detail="No audio file provided.")
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=400, detail="File exceeds 20 MB limit.")

    saved_path = Path(tempfile.gettempdir()) / f"{uuid.uuid4().hex}{suffix or '.wav'}"
    saved_path.write_bytes(data)

    try:
        result = analyzer.analyze_file(saved_path)
        return {
            "synthetic_probability": result.synthetic_probability,
            "authentic_probability": result.authentic_probability,
            "verdict": result.verdict,
            "verdict_en": result.verdict_en,
            "hindi_message": result.hindi_message,
            "model_id": result.model_id,
            "duration_sec": result.duration_sec,
        }
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc
    finally:
        if saved_path.exists():
            saved_path.unlink()
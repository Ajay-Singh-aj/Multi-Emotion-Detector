# app_api.py
"""
Optional FastAPI server to serve the modular inference as REST endpoints.
Run: uvicorn app_api:app --reload --port 8000
"""

import tempfile, io, os
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from models.face_model import detect_face_emotion_from_bytes
from models.audio_model import detect_audio_emotion_from_file
from models.text_model import detect_text_emotion_from_text
from fusion import fuse_modalities
from typing import Optional

app = FastAPI(title="MoodSpark API")

@app.post("/infer")
async def infer(image: Optional[UploadFile] = File(None), audio: Optional[UploadFile] = File(None), text: Optional[str] = Form(None)):
    mod_outputs = {"face": None, "audio": None, "text": None}
    tmp_files = []
    if image:
        b = await image.read()
        mod_outputs["face"] = detect_face_emotion_from_bytes(b)
    if audio:
        # write temp file
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        content = await audio.read()
        tmp.write(content)
        tmp.flush()
        tmp_files.append(tmp.name)
        mod_outputs["audio"] = detect_audio_emotion_from_file(tmp.name)
    if text:
        mod_outputs["text"] = detect_text_emotion_from_text(text)
    fused = fuse_modalities({k: v for k, v in mod_outputs.items() if v is not None})
    for f in tmp_files:
        try:
            os.remove(f)
        except Exception:
            pass
    if fused is None:
        return JSONResponse({"error": "No valid input provided"}, status_code=400)
    return {"modalities": mod_outputs, "fusion": fused}

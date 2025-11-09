# models/text_model.py
"""
Text sentiment module: uses HF sentiment pipeline (distilbert finetuned sst-2).
Maps to COMMON_EMOTIONS heuristically.
"""

from typing import Optional, Dict
import torch
from transformers import pipeline

COMMON_EMOTIONS = ["happy", "sad", "angry", "surprise", "fear", "disgust", "neutral"]
TEXT_MODEL_ID = "distilbert-base-uncased-finetuned-sst-2-english"

_text_pipe = None

def init_text_pipe():
    global _text_pipe
    if _text_pipe is not None:
        return
    _text_pipe = pipeline("sentiment-analysis", model=TEXT_MODEL_ID, device=0 if torch.cuda.is_available() else -1)

def detect_text_emotion_from_text(text: str) -> Optional[Dict[str, float]]:
    if not text or text.strip() == "":
        return None
    try:
        init_text_pipe()
    except Exception:
        return None
    out = _text_pipe(text[:512])[0]
    label = out.get("label", "").lower()
    score = float(out.get("score", 0.0))
    vec = {k: 0.0 for k in COMMON_EMOTIONS}
    if "positive" in label or "pos" in label:
        vec["happy"] = score
        vec["neutral"] = 1.0 - score
    elif "negative" in label or "neg" in label:
        # baseline: negative -> sad
        vec["sad"] = score
        vec["neutral"] = 1.0 - score
    else:
        vec["neutral"] = 1.0
    total = sum(vec.values()) + 1e-9
    vec = {k: float(v / total) for k, v in vec.items()}
    return vec

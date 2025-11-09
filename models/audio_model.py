# models/audio_model.py
"""
Audio emotion module: loads HF audio classification model and maps outputs
to COMMON_EMOTIONS heuristically.
"""

from typing import Optional, Dict
import os
import numpy as np
import soundfile as sf
import librosa
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

COMMON_EMOTIONS = ["happy", "sad", "angry", "surprise", "fear", "disgust", "neutral"]
AUDIO_MODEL_ID = "superb/hubert-large-superb-er"

# lazy-loaded objects
_feature_extractor = None
_audio_model = None

def init_audio_model():
    global _feature_extractor, _audio_model
    if _audio_model is not None:
        return
    _feature_extractor = AutoFeatureExtractor.from_pretrained(AUDIO_MODEL_ID)
    _audio_model = AutoModelForAudioClassification.from_pretrained(AUDIO_MODEL_ID)
    if torch.cuda.is_available():
        _audio_model.to("cuda")

def _map_labels_to_common(labels_map: Dict[str, float]) -> Dict[str, float]:
    out = {k: 0.0 for k in COMMON_EMOTIONS}
    for lab, v in labels_map.items():
        l = lab.lower()
        if "happy" in l or "joy" in l:
            out["happy"] += v
        elif "sad" in l:
            out["sad"] += v
        elif "angry" in l or "anger" in l:
            out["angry"] += v
        elif "surprise" in l:
            out["surprise"] += v
        elif "fear" in l or "afraid" in l:
            out["fear"] += v
        elif "disgust" in l:
            out["disgust"] += v
        elif "neutral" in l:
            out["neutral"] += v
        else:
            out["neutral"] += v * 0.5
    return out

def detect_audio_emotion_from_file(path: str) -> Optional[Dict[str, float]]:
    try:
        init_audio_model()
    except Exception:
        return None
    try:
        wav, sr = sf.read(path)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
    except Exception:
        return None
    target_sr = _feature_extractor.sampling_rate if hasattr(_feature_extractor, "sampling_rate") else 16000
    if sr != target_sr:
        wav = librosa.resample(wav.astype(float), orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    inputs = _feature_extractor(wav, sampling_rate=sr, return_tensors="pt", padding=True)
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        logits = _audio_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    labels = _audio_model.config.id2label if hasattr(_audio_model.config, "id2label") else {i: str(i) for i in range(len(probs))}
    labels_map = {labels[i].lower(): float(probs[i]) for i in range(len(probs))}
    mapped = _map_labels_to_common(labels_map)
    total = sum(mapped.values()) + 1e-9
    mapped = {k: float(v / total) for k, v in mapped.items()}
    return mapped

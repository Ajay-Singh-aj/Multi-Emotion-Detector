# fusion.py
"""
Fusion utilities: handle modality fusion, weight normalization, suggestions, mapping helpers.
"""

from typing import Dict, Optional
COMMON_EMOTIONS = ["happy", "sad", "angry", "surprise", "fear", "disgust", "neutral"]
DEFAULT_WEIGHTS = {"face": 0.35, "audio": 0.35, "text": 0.30}

SUGGESTIONS = {
    "happy": "Nice — user seems upbeat! 🎉 Try matching energy.",
    "sad": "User seems low — respond with warmth and empathy. 🤝",
    "angry": "Calm responses; acknowledge frustration. 🧘",
    "surprise": "Ask a clarifying question. 🤔",
    "fear": "Offer reassurance and safety cues. 🕊️",
    "disgust": "Consider changing approach or content. 🛑",
    "neutral": "Neutral — ask an engaging question. 💬"
}

def fuse_modalities(mod_outputs: Dict[str, Dict[str, float]], weights: Dict[str, float] = DEFAULT_WEIGHTS):
    present = [m for m in mod_outputs.keys() if mod_outputs[m] is not None]
    if not present:
        return None
    w = {m: weights.get(m, 1.0) for m in present}
    s = sum(w.values()) + 1e-9
    w = {k: float(v / s) for k, v in w.items()}
    fused = {k: 0.0 for k in COMMON_EMOTIONS}
    modality_conf = {}
    for m in present:
        vec = mod_outputs[m]
        conf = max(vec.values()) if vec else 0.0
        modality_conf[m] = float(conf)
        for emo in COMMON_EMOTIONS:
            fused[emo] += w[m] * float(vec.get(emo, 0.0))
    total = sum(fused.values()) + 1e-9
    fused = {k: float(v / total) for k, v in fused.items()}
    final_label = max(fused.items(), key=lambda x: x[1])[0]
    overall_conf = sum(modality_conf[m] * w[m] for m in present)
    return {"fused": fused, "final_label": final_label, "confidence": float(overall_conf),
            "modalities": modality_conf, "weights_used": w}

def suggestion_from_label(label: str) -> str:
    return SUGGESTIONS.get(label, "")

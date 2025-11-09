# models/face_model.py
import cv2
import numpy as np
from fer import FER

detector = FER(mtcnn=True)

COMMON_EMOTIONS = ["happy","sad","angry","surprise","fear","disgust","neutral"]

def detect_face_emotion_from_bytes(image_bytes):
    """
    Always return a dict of emotions.
    If no face is detected, returns neutral=1.0 (fallback)
    """
    try:
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return {"neutral": 1.0}

        results = detector.detect_emotions(img)

        if not results:
            return {"neutral": 1.0}

        # If multiple faces -> choose the one with the biggest bounding box
        biggest = max(results, key=lambda x: (x["box"][2] * x["box"][3]))
        emotions = biggest["emotions"]

        total = sum(emotions.values()) + 1e-9
        normalized = {k: float(v / total) for k, v in emotions.items()}

        # Ensure all COMMON_EMOTIONS exist
        output = {k: normalized.get(k, 0.0) for k in COMMON_EMOTIONS}

        return output

    except Exception:
        return {"neutral": 1.0}

# app_streamlit.py
"""
MoodSpark — Fun Multimodal Emotion Detector (Final Stable Version)
"""

import os
import json
import time
import tempfile
import numpy as np
import soundfile as sf
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase

from models.face_model import detect_face_emotion_from_bytes
from models.audio_model import detect_audio_emotion_from_file
from models.text_model import detect_text_emotion_from_text
from fusion import fuse_modalities, DEFAULT_WEIGHTS, suggestion_from_label
from utils import format_result_json


# ✅ Replace SVG Logo → Use Uploaded Image
HEADER_IMAGE = "/content/app_header.jpg"   # <--- Change this filename if needed


# ✅ UI Header
st.set_page_config(page_title="MoodSpark", page_icon="✨", layout="centered")

if os.path.exists(HEADER_IMAGE):
    st.image(HEADER_IMAGE, use_column_width=True)
else:
    st.warning("⚠️ Header image not found. Upload and update HEADER_IMAGE path.")

st.markdown("<h5 style='text-align:center;margin-top:-10px;'>Developed by <b>Ajay Singh</b> • Guided by <b>Prof. Amar Behera (DES646)</b></h5>", unsafe_allow_html=True)


# ✅ Emotion Icons
EMOJI = {"happy": "😄","sad": "😢","angry": "😡","surprise": "😲","fear": "😨","disgust": "🤢","neutral": "😐"}

RTC_CONFIGURATION = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}


# ✅ Stable Audio Processor (Fixes Start/Stop Problem)
class MicProcessor(AudioProcessorBase):
    def __init__(self):
        self.frames = []

    def recv_audio(self, frame):
        arr = frame.to_ndarray().astype(np.float32).mean(axis=1) / 32768.0
        self.frames.append(arr)
        return frame


def save_wav(frames, sr=16000, path="/tmp/rec.wav"):
    if frames:
        wav = np.concatenate(frames)
        sf.write(path, wav, sr)
        return path
    return None


def analyze(image_bytes, audio_path, text, weights):
    outputs = {"face": None, "audio": None, "text": None}

    if image_bytes:
        outputs["face"] = detect_face_emotion_from_bytes(image_bytes)

    if audio_path:
        outputs["audio"] = detect_audio_emotion_from_file(audio_path)

    if text:
        outputs["text"] = detect_text_emotion_from_text(text)

    available = {k: v for k, v in outputs.items() if v}

    if not available:
        return outputs, None

    return outputs, fuse_modalities(available, weights)


# -------------------- Sidebar --------------------
st.sidebar.markdown("### Select Inputs")
use_image = st.sidebar.checkbox("Use Image", True)
use_audio = st.sidebar.checkbox("Use Audio", True)
use_text = st.sidebar.checkbox("Use Text", True)

weights = {
    "face": st.sidebar.slider("Face Weight", 0.0, 1.0, DEFAULT_WEIGHTS["face"]),
    "audio": st.sidebar.slider("Audio Weight", 0.0, 1.0, DEFAULT_WEIGHTS["audio"]),
    "text": st.sidebar.slider("Text Weight", 0.0, 1.0, DEFAULT_WEIGHTS["text"]),
}


# -------------------- Inputs UI --------------------
st.write("## Inputs")
col1, col2 = st.columns(2)

image_bytes = None
audio_path = None
typed_text = None


# ✅ IMAGE Input
if use_image:
    with col1:
        st.subheader("📷 Image Input")
        img = st.file_uploader("Upload Image (JPG/PNG)", ["jpg","jpeg","png"])
        cam = st.camera_input("Or Take Photo")
        chosen = cam or img
        if chosen:
            image_bytes = chosen.read()
            st.image(image_bytes)


# ✅ AUDIO Input (Improved Stable Recording)
if use_audio:
    with col2:
        st.subheader("🎤 Live Audio Recording")

        ctx = webrtc_streamer(
            key="mic_capture",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            audio_processor_factory=MicProcessor,
            media_stream_constraints={"audio": True, "video": False},
        )

        if ctx and ctx.audio_processor:
            if st.button("Stop Recording"):
                audio_path = save_wav(ctx.audio_processor.frames)
                st.success("✅ Audio Recorded")

        uploaded_audio = st.file_uploader("Or Upload Audio", ["wav","mp3"])
        if uploaded_audio:
            temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            temp.write(uploaded_audio.read())
            temp.flush()
            audio_path = temp.name
            st.audio(audio_path)


# ✅ TEXT Input
if use_text:
    st.subheader("💬 Text Input")
    typed_text = st.text_area("Enter text here:")


# -------------------- Analyze Button --------------------
st.markdown("---")
if st.button("✨ Analyze Emotion"):
    outputs, fused = analyze(image_bytes, audio_path, typed_text, weights)

    if fused is None:
        st.error("⚠️ No valid input detected. Please upload image, speak, or enter text.")
    else:
        emo = fused["final_label"]; conf = fused["confidence"]
        st.markdown(f"## {EMOJI[emo]} {emo.upper()} — {conf*100:.1f}%")
        st.write(suggestion_from_label(emo))
        st.download_button("Download Results JSON", format_result_json(outputs, fused), "result.json")

st.markdown("<div style='text-align:center'>Developed by Ajay Singh — MoodSpark AI</div>", unsafe_allow_html=True)

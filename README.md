# MoodSpark — Multimodal Emotion Detector (Modular)

This repository contains a modular multimodal emotion detection demo (face + audio + text) with:
- Streamlit UI (`app_streamlit.py`) — fun themed
- Optional FastAPI server (`app_api.py`)
- Modular model wrappers under `models/`
- Fusion logic in `fusion.py`

## Quick setup (Google Colab)
1. Create a Colab notebook and run:
```bash
!pip install -q -r requirements.txt

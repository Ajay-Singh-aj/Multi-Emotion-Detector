# utils.py
"""
Utility helpers used by both Streamlit & API apps.
"""

import json
from typing import Dict

def format_result_json(mod_outputs: Dict[str, Dict], fusion_result: Dict) -> str:
    out = {"modalities": mod_outputs, "fusion": fusion_result}
    return json.dumps(out, indent=2)

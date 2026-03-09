 # agent/summarizer.py
from typing import List, Dict, Any
from openai import OpenAI
from .openai_utils import img_to_data_url, call_with_retry, parse_json_loose

CASE_SCHEMA_HINT = """
Return JSON ONLY with keys:
{
 "visible_observations": [string],
 "suspected_abnormal_signs": [string],
 "behavior_state": string,
 "breathing_effort": string,
 "mobility": string,
 "notable_red_flags_seen": [string],
 "audio_clues": [string],
 "summary_1to2_sentences": string
}
Rules:
- Only describe what is visible/audible; do NOT diagnose.
- No medication names, no dosing.
"""

def build_case_summary(frames: List[str], audio_text: str, client: OpenAI, model="gpt-4o-mini") -> Dict[str, Any]:
    content = [
        {"type": "input_text", "text": "Extract structured observations for a vet assistant system."},
        {"type": "input_text", "text": f"Audio transcript:\n{audio_text or '(none)'}"},
        {"type": "input_text", "text": CASE_SCHEMA_HINT},
    ]
    for fp in frames:
        content.append({"type": "input_image", "image_url": img_to_data_url(fp)})

    def _call():
        return client.responses.create(
            model=model,
            input=[{"role": "user", "content": content}],
        )

    resp = call_with_retry(_call)
    # For Responses API, output_text is the simplest string extraction
    return parse_json_loose(resp.output_text)
 # agent/openai_utils.py
import base64, mimetypes, json, time
from openai import RateLimitError, APIError

def img_to_data_url(path: str) -> str:
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def call_with_retry(fn, max_retries=3, sleep_sec=2):
    last_err = None
    for i in range(max_retries):
        try:
            return fn()
        except Exception as e:
            last_err = e
            print(f"[call_with_retry] attempt {i+1}/{max_retries} failed: {repr(e)}")
            time.sleep(sleep_sec)

    raise RuntimeError(f"API call failed after retries. Last error: {repr(last_err)}")

def parse_json_loose(text: str):
    text = (text or "").strip()
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e != -1 and e > s:
        return json.loads(text[s:e+1])
    raise ValueError("No JSON object found.")
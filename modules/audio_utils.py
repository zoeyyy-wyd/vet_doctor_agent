# agent/audio_utils.py
import whisper

def transcribe_video_audio(video_path: str, whisper_model="base") -> str:
    model = whisper.load_model(whisper_model)
    result = model.transcribe(video_path)
    return (result.get("text") or "").strip()
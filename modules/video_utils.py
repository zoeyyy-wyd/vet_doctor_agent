# agent/video_utils.py
import cv2
from pathlib import Path

def extract_frames_by_seconds(video_path: str, out_dir="frames", every_sec=2.0, max_frames=10):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 30.0
    interval = max(int(round(fps * every_sec)), 1)

    frames = []
    idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            fp = out_dir / f"frame_{saved:04d}.jpg"
            cv2.imwrite(str(fp), frame)
            frames.append(str(fp))
            saved += 1
            if saved >= max_frames:
                break
        idx += 1

    cap.release()
    return frames
from pathlib import Path
import cv2


def frames(source, every_sec=1.0):
    """
    Yield (name, frame) pairs from:
      - a folder of images (sorted by filename; PKLot names are timestamps)
      - a video file
      - a webcam index like "0"
    For video, only one frame every `every_sec` seconds is returned,
    because parked cars barely move and this saves CPU.
    """
    source = str(source)
    p = Path(source)

    if p.is_dir():
        files = sorted(list(p.rglob("*.jpg")) + list(p.rglob("*.png")))
        for f in files:
            img = cv2.imread(str(f))
            if img is not None:
                yield f.name, img
        return

    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open source: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step = max(1, int(fps * every_sec))
    i = 0
    while True:
        if not cap.grab():
            break
        if i % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                yield f"frame_{i}", frame
        i += 1
    cap.release()
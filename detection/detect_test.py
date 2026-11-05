import cv2
from pathlib import Path
from ultralytics import YOLO

VIDEO = Path("data/demo.mp4")
OUT = Path("data/detect_preview.jpg")
WIDTH = 1280                      # shrink 4K frames to this width
CAR_CLASSES = {2, 5, 7}           # car, bus, truck
SECOND = 5                        # which moment of the video to test

cap = cv2.VideoCapture(str(VIDEO))
cap.set(cv2.CAP_PROP_POS_MSEC, SECOND * 1000)
ok, frame = cap.read()
cap.release()
if not ok:
    raise SystemExit("Could not read the video. Check the path data/demo.mp4")

h, w = frame.shape[:2]
frame = cv2.resize(frame, (WIDTH, int(h * WIDTH / w)))
print("Frame size after resize:", frame.shape[1], "x", frame.shape[0])

model = YOLO("yolov8n.pt")        # downloads ~6 MB the first time
res = model(frame, imgsz=WIDTH, conf=0.25, verbose=False)[0]

count = 0
for box, cls, conf in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(), res.boxes.conf.tolist()):
    if int(cls) not in CAR_CLASSES:
        continue
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    count += 1

cv2.imwrite(str(OUT), frame)
print(f"Vehicles detected: {count}")
print("Saved", OUT)
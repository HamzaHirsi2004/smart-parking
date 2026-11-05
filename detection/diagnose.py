import cv2
from collections import Counter
from ultralytics import YOLO

cap = cv2.VideoCapture("data/demo.mp4")
cap.set(cv2.CAP_PROP_POS_MSEC, 5000)
ok, frame = cap.read()
cap.release()
h, w = frame.shape[:2]
frame = cv2.resize(frame, (1280, int(h * 1280 / w)))

for name in ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]:
    model = YOLO(name)
    for imgsz in (1280, 1920):
        res = model(frame, imgsz=imgsz, conf=0.05, verbose=False)[0]
        names = [model.names[int(c)] for c in res.boxes.cls.tolist()]
        print(f"{name:11} imgsz={imgsz}: {len(names)} detections ->", dict(Counter(names)))
        
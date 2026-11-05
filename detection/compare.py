import cv2
import numpy as np
from collections import Counter
from ultralytics import YOLO

cap = cv2.VideoCapture("data/demo.mp4")
cap.set(cv2.CAP_PROP_POS_MSEC, 5000)
ok, frame = cap.read()
cap.release()
assert ok, "Could not read data/demo.mp4"
h, w = frame.shape[:2]
frame = cv2.resize(frame, (1280, int(h * 1280 / w)))

# A: standard model, treating "cell phone" (67) as a vehicle too
a = frame.copy()
m = YOLO("yolov8m.pt")
res = m(frame, imgsz=1280, conf=0.10, classes=[2, 3, 5, 7, 67], verbose=False)[0]
for (x1, y1, x2, y2) in res.boxes.xyxy.tolist():
    cv2.rectangle(a, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
cv2.imwrite("data/compare_A.jpg", a)
print("A (yolov8m + cell phone):", len(res.boxes), "detections")

# B: aerial-trained oriented-box model
try:
    b = frame.copy()
    o = YOLO("yolov8m-obb.pt")        # downloads ~50 MB first time
    r = o(frame, imgsz=1280, conf=0.25, verbose=False)[0]
    for poly in r.obb.xyxyxyxy.tolist():
        cv2.polylines(b, [np.array(poly, dtype=np.int32)], True, (0, 255, 0), 2)
    cv2.imwrite("data/compare_B.jpg", b)
    names = [o.names[int(c)] for c in r.obb.cls.tolist()]
    print("B (aerial model):", len(names), "detections ->", dict(Counter(names)))
except Exception as e:
    print("B failed:", e)
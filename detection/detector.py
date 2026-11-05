from ultralytics import YOLO

# COCO: car, motorcycle, bus, truck, plus "cell phone" (67).
# From directly above, YOLO often labels a car roof as a cell phone.
VEHICLE_CLASSES = [2, 3, 5, 7, 67]


class VehicleDetector:
    def __init__(self, model_path="yolov8m.pt", conf=0.10, imgsz=1280):
        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz

    def detect(self, frame):
        """frame: BGR image (numpy array). Returns a list of detections."""
        result = self.model(
            frame, classes=VEHICLE_CLASSES, conf=self.conf,
            imgsz=self.imgsz, verbose=False,
        )[0]
        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "box": (int(x1), int(y1), int(x2), int(y2)),
                "conf": float(box.conf[0]),
                "cls": int(box.cls[0]),
            })
        return detections
import json
from pathlib import Path
from ultralytics import YOLO

folder = Path("data/pklot/archive/test")
data = json.load(open(folder / "_annotations.coco.json"))
model = YOLO("yolov8n.pt")          # downloads ~6 MB the first time

CAR_CLASSES = {2, 5, 7}             # car, bus, truck
N = 60                              # images to test

boxes = {}
for a in data["annotations"]:
    boxes.setdefault(a["image_id"], []).append(a)

same = swapped = total = 0
for img in data["images"][:N]:
    res = model(str(folder / img["file_name"]), imgsz=640, conf=0.25, verbose=False)[0]
    centres = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
               for b, c in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist())
               if int(c) in CAR_CLASSES]
    for a in boxes.get(img["id"], []):
        x, y, w, h = a["bbox"]
        has_car = any(x <= cx <= x + w and y <= cy <= y + h for cx, cy in centres)
        labelled_occ = a["category_id"] == 2
        total += 1
        same += (has_car == labelled_occ)
        swapped += (has_car != labelled_occ)

print(f"{total} spaces checked")
print(f"agree with labels as-is : {same / total:.1%}")
print(f"agree if labels swapped : {swapped / total:.1%}")
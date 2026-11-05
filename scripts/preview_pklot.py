import json
import cv2
from pathlib import Path

folder = Path("data/pklot/archive/test")
data = json.load(open(folder / "_annotations.coco.json"))

img_info = data["images"][0]
img = cv2.imread(str(folder / img_info["file_name"]))

for a in data["annotations"]:
    if a["image_id"] != img_info["id"]:
        continue
    x, y, w, h = [int(round(v)) for v in a["bbox"]]
    color = (0, 200, 0) if a["category_id"] == 1 else (0, 0, 255)  # green = empty, red = occupied (BGR)
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)

out = Path("data/preview.jpg")
cv2.imwrite(str(out), img)
print("Saved", out, "-", img_info["file_name"])
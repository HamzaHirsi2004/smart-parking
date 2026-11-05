import json
from collections import Counter

d = json.load(open("data/pklot/archive/test/_annotations.coco.json"))
c = Counter(a["category_id"] for a in d["annotations"])
print("empty (1):", c[1], " occupied (2):", c[2])
print("occupied share:", round(c[2] / (c[1] + c[2]), 3))
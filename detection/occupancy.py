import json
import cv2
import numpy as np


class BayMatcher:
    """Matches vehicle boxes to bay polygons."""

    def __init__(self, spaces, frame_shape, ref_size=None, min_overlap=0.30):
        h, w = frame_shape[:2]
        sx = sy = 1.0
        if ref_size and tuple(ref_size) != (w, h):      # video size differs from reference frame
            sx, sy = w / ref_size[0], h / ref_size[1]

        self.min_overlap = min_overlap
        self.labels, self.polys, self.rects, self.masks, self.areas = [], [], [], [], []

        for s in spaces:
            if not s["polygon"]:
                continue
            pts = (np.array(json.loads(s["polygon"]), dtype=np.float32) * [sx, sy]).astype(np.int32)
            x, y, bw, bh = cv2.boundingRect(pts)
            mask = np.zeros((bh, bw), dtype=np.uint8)    # small mask, not full-frame (saves RAM)
            cv2.fillPoly(mask, [pts - [x, y]], 1)
            area = int(mask.sum())
            if area == 0:
                continue
            self.labels.append(s["label"])
            self.polys.append(pts)
            self.rects.append((x, y, bw, bh))
            self.masks.append(mask)
            self.areas.append(area)

    def _overlap(self, i, box):
        """Fraction of bay i's area that is covered by the box."""
        x, y, bw, bh = self.rects[i]
        ix1, iy1 = max(box[0], x), max(box[1], y)
        ix2, iy2 = min(box[2], x + bw), min(box[3], y + bh)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        return float(self.masks[i][iy1 - y:iy2 - y, ix1 - x:ix2 - x].sum()) / self.areas[i]

    def match(self, detections):
        """Return {label: True/False} (True = occupied)."""
        status = {label: False for label in self.labels}
        if not self.labels:
            return status
        for d in detections:
            scores = [self._overlap(i, d["box"]) for i in range(len(self.labels))]
            best = int(np.argmax(scores))
            if scores[best] >= self.min_overlap:
                status[self.labels[best]] = True
        return status


class StatusSmoother:
    """A bay only changes status after `confirm` consecutive frames disagree."""

    def __init__(self, confirm=3):
        self.confirm = confirm
        self.state = {}
        self.pending = {}

    def update(self, raw):
        for label in list(self.state):                   # forget removed bays
            if label not in raw:
                self.state.pop(label)
                self.pending.pop(label, None)
        for label, occ in raw.items():
            if label not in self.state:                  # first sighting: trust it
                self.state[label], self.pending[label] = occ, 0
            elif occ == self.state[label]:
                self.pending[label] = 0
            else:
                self.pending[label] += 1
                if self.pending[label] >= self.confirm:
                    self.state[label], self.pending[label] = occ, 0
        return dict(self.state)
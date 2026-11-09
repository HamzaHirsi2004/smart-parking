import sys, os, time, argparse
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from core import database as db
from detection.detector import VehicleDetector
from detection.frame_source import frames
from detection.occupancy import BayMatcher, StatusSmoother

LATEST = Path(__file__).resolve().parent.parent / "data" / "latest_frame.jpg"
LATEST_TMP = LATEST.with_name("latest_frame_tmp.jpg")


def annotate(frame, matcher, status, dets, ms):
    out = frame.copy()
    overlay = out.copy()
    for poly, label in zip(matcher.polys, matcher.labels):
        color = (0, 0, 255) if status.get(label) else (0, 200, 0)   # BGR: red / green
        cv2.fillPoly(overlay, [poly], color)
    out = cv2.addWeighted(overlay, 0.35, out, 0.65, 0)

    for poly, label in zip(matcher.polys, matcher.labels):
        color = (0, 0, 255) if status.get(label) else (0, 200, 0)
        cv2.polylines(out, [poly], True, color, 2)
        cx, cy = poly.mean(axis=0).astype(int)
        cv2.putText(out, label, (cx - 12, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    for d in dets:
        x1, y1, x2, y2 = d["box"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 165, 255), 1)

    occ = sum(status.values())
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, f"Occupied {occ}/{len(status)} | {ms:.0f} ms", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="data/demo.mp4")
    ap.add_argument("--every", type=float, default=1.0, help="video seconds between analysed frames")
    ap.add_argument("--interval", type=float, default=1.0, help="minimum real seconds per cycle")
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--min-overlap", type=float, default=0.30)
    ap.add_argument("--confirm", type=int, default=3, help="frames needed to change a bay's status")
    ap.add_argument("--log-every", type=float, default=10.0, help="seconds between history rows")
    ap.add_argument("--no-loop", action="store_true")
    ap.add_argument("--show", action="store_true", help="show a live preview window")
    args = ap.parse_args()

    db.init_db()
    detector = VehicleDetector(conf=args.conf, imgsz=args.imgsz)
    smoother = StatusSmoother(args.confirm)
    matcher, signature, last_reload, last_log = None, None, 0.0, 0.0

    print("Worker started. Press Ctrl+C to stop.")
    try:
        while True:
            for name, frame in frames(args.source, args.every):
                t0 = time.time()

                # reload bays every 10 s so edits in the admin appear without a restart
                if matcher is None or t0 - last_reload > 10:
                    spaces = [s for s in db.get_spaces() if s["polygon"]]
                    sig = (frame.shape[:2], tuple((s["label"], s["polygon"]) for s in spaces))
                    if sig != signature:
                        matcher = BayMatcher(spaces, frame.shape, db.get_setting("frame_size"),
                                             args.min_overlap)
                        signature = sig
                        print(f"Loaded {len(matcher.labels)} bays.")
                    last_reload = t0

                if not matcher.labels:
                    print("No bays configured yet. Draw them in the admin app. Waiting...")
                    time.sleep(3)
                    continue

                dets = detector.detect(frame)
                status = smoother.update(matcher.match(dets))
                db.update_statuses({k: int(v) for k, v in status.items()})

                if t0 - last_log >= args.log_every:
                    db.log_occupancy()
                    last_log = t0

                ms = (time.time() - t0) * 1000
                out = annotate(frame, matcher, status, dets, ms)
                cv2.imwrite(str(LATEST_TMP), out)
                os.replace(LATEST_TMP, LATEST)          # atomic: the dashboard never sees half a file

                occ = sum(status.values())
                print(f"{name}: {len(dets)} vehicles | {occ}/{len(status)} bays occupied | {ms:.0f} ms")

                if args.show:
                    cv2.imshow("Smart Parking Worker (q to quit)", out)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        return
                time.sleep(max(0.0, args.interval - (time.time() - t0)))

            if args.no_loop:
                break
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

    
import sys, time, argparse
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2
from detection.detector import VehicleDetector
from detection.frame_source import frames


def draw(frame, detections, name, ms):
    out = frame.copy()
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 165, 255), 2)
        cv2.putText(out, f"{d['conf']:.2f}", (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)
    label = f"{name} | vehicles: {len(detections)} | {ms:.0f} ms"
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="video file, image folder, or webcam index")
    ap.add_argument("--every", type=float, default=1.0, help="seconds between analysed video frames")
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--max", type=int, default=0, help="stop after N frames (0 = no limit)")
    ap.add_argument("--loop", action="store_true", help="restart when the source ends (good for demos)")
    args = ap.parse_args()

    detector = VehicleDetector(conf=args.conf, imgsz=args.imgsz)
    cv2.namedWindow("Detections (press q to quit)", cv2.WINDOW_NORMAL)

    count = 0
    while True:
        for name, frame in frames(args.source, args.every):
            t = time.time()
            dets = detector.detect(frame)
            ms = (time.time() - t) * 1000
            cv2.imshow("Detections (press q to quit)", draw(frame, dets, name, ms))
            print(f"{name}: {len(dets)} vehicles ({ms:.0f} ms)")
            count += 1
            if cv2.waitKey(1) & 0xFF == ord("q") or (args.max and count >= args.max):
                cv2.destroyAllWindows()
                return
        if not args.loop:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
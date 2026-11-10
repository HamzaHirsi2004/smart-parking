# AI-Powered Smart Parking Management and Vehicle Monitoring System

Final year project prototype. A fixed camera feed is analysed with YOLO, each
parking bay is marked free or occupied, and the result is shown to drivers on a
responsive website and to administrators on a control dashboard.

## Features
**Customer website (Smart Parking Finder)**
- Total, occupied and available spaces, refreshing every 5 seconds
- Green/red parking map, with the nearest free bay from a chosen entrance
- "Last updated" time, occupancy history, and a 24-hour predicted busy-times chart
- Works on phone, tablet and computer with no app installation

**Administrator dashboard (Smart Parking Control Centre)**
- Live video with AI vehicle detections and bay status
- Click-to-draw bay boundary editor and entrance placement
- Occupancy statistics, trends, and CSV export
- Database log of every bay change
- Train and review a scikit-learn occupancy forecast

## Architecture
```
camera/video -> YOLO detection -> bay matching + smoothing -> SQLite
                                                                |
                                     customer site  <-----------+-----------> admin dashboard
```
Both interfaces read the same SQLite database. The detection worker is the only
component that writes bay statuses.

## Setup
Python 3.10 or 3.11 recommended.

    python -m venv venv
    venv\Scripts\activate        (Mac/Linux: source venv/bin/activate)
    pip install -r requirements.txt

Place a fixed-camera, top-down parking video at `data/demo.mp4`.

The database and bay layout are not stored in Git, so on a fresh copy you must
create them: open the admin dashboard -> Space configuration, choose a
reference frame, then draw your bays and entrances.

Optional demo history for the statistics and forecast tabs (synthetic data):

    python scripts/seed_history_only.py

Do not run `scripts/seed_demo_data.py` once real bays exist. It replaces them
with 40 fake bays.

## Run
Windows: double-click `scripts/run_demo.bat`, or run each part manually:

    python detection/worker.py --source data/demo.mp4
    streamlit run customer/app.py --server.port 8501
    streamlit run admin/app.py --server.port 8502

Before a presentation, run `python scripts/reset_demo.py` to clear the activity log.

## Project structure
    core/        database, routing, forecasting
    detection/   YOLO detector, frame reader, bay matching, worker
    customer/    customer website
    admin/       dashboard (monitor, statistics, bay editor, predictions)
    scripts/     demo data, simulator, reset, launcher

## Limitations
- Tested on one fixed, top-down camera view; no multi-camera or multi-lot support
- Uses a pretrained COCO YOLO model (yolov8m), not fine-tuned on parking footage.
  From directly above it often labels cars as "cell phone", so that class is
  counted as a vehicle. This suits the demo footage and may not suit other views
- A vehicle driving through a bay for several seconds can briefly mark it occupied,
  and occasional missed detections cause short flickers; a 3-frame confirmation
  reduces but does not remove this
- Bay shapes are drawn on one reference frame, so a moving camera breaks them
- "Nearest bay" uses straight-line distance, not walking distance along the aisles
- The demo video loops, so live occupancy has no real daily pattern. The forecast
  shown in the demo is trained on synthetic data and demonstrates the method only;
  its accuracy figure is not a real-world result
- No user accounts or login on the admin dashboard

## Credits
- Ultralytics YOLOv8 (AGPL-3.0)
- PKLot dataset, used during development to test detection: Almeida, P., Oliveira,
  L. S., Silva Jr, E., Britto Jr, A., Koerich, A., "PKLot - A robust dataset for
  parking lot classification", Expert Systems with Applications, 42(11), 2015
- Demo video: <source and author>
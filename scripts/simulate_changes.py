import sys, time, random
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from core import database as db

while True:
    spaces = db.get_spaces()
    for sp in random.sample(spaces, 3):          # flip 3 random bays
        db.update_statuses({sp["label"]: 1 - sp["is_occupied"]})
    db.log_occupancy()
    print("Updated:", db.get_summary())
    time.sleep(4)
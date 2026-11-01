import sys, random
from pathlib import Path
from datetime import datetime, timedelta
sys.path.append(str(Path(__file__).resolve().parent.parent))

from core import database as db

random.seed(42)
db.init_db()

# --- 40 bays: rows A-D, 10 bays each. Map coords are a simple grid. ---
for r, row in enumerate("ABCD"):
    for c in range(10):
        db.add_space(
            label=f"{row}{c+1}",
            map_x=c * 1.0,
            map_y=r * 1.5,
            is_occupied=1 if random.random() < 0.55 else 0,
        )

db.set_setting("entrances", {"Main Gate": [0.0, -1.5], "Side Gate": [9.0, -1.5]})

# --- 3 days of fake history (every 15 min) with a realistic daily pattern ---
import math
total = 40
start = datetime.now() - timedelta(days=3)
with db.get_conn() as conn:
    conn.execute("DELETE FROM occupancy_log")
    t = start
    while t < datetime.now():
        hour = t.hour + t.minute / 60
        base = 0.15 + 0.7 * math.exp(-((hour - 13) ** 2) / 18)   # peak around 1 PM
        occ = max(0, min(total, int(total * base + random.randint(-3, 3))))
        conn.execute(
            "INSERT INTO occupancy_log (timestamp,total,occupied,available) VALUES (?,?,?,?)",
            (t.strftime("%Y-%m-%d %H:%M:%S"), total, occ, total - occ),
        )
        t += timedelta(minutes=15)

print("Done:", db.get_summary())
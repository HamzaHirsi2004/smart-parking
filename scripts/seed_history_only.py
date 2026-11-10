import sys, math, random
from pathlib import Path
from datetime import datetime, timedelta
sys.path.append(str(Path(__file__).resolve().parent.parent))

from core import database as db

random.seed(42)
db.init_db()
total = len(db.get_spaces())
if total == 0:
    raise SystemExit("No bays configured. Draw them in the admin app first.")

start = datetime.now() - timedelta(days=3)
end = datetime.now() - timedelta(minutes=5)

with db.get_conn() as conn:
    conn.execute("DELETE FROM occupancy_log")
    t = start
    while t < end:
        hour = t.hour + t.minute / 60
        base = 0.15 + 0.7 * math.exp(-((hour - 13) ** 2) / 18)   # peak around 1 PM
        occ = max(0, min(total, int(total * base + random.randint(-2, 2))))
        conn.execute(
            "INSERT INTO occupancy_log (timestamp,total,occupied,available) VALUES (?,?,?,?)",
            (t.strftime("%Y-%m-%d %H:%M:%S"), total, occ, total - occ),
        )
        t += timedelta(minutes=15)

print(f"Inserted demo history for {total} bays.")
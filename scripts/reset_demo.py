import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from core import database as db

db.init_db()
db.clear_events()
print("Bay events cleared. Bays, entrances, history and the trained model are untouched.")
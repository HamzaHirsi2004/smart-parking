import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st
from core import database as db
from admin import bay_editor, live_monitor, stats_view

st.set_page_config(page_title="Smart Parking Control Centre", page_icon="🛠️", layout="wide")
db.init_db()

st.title("🛠️ Smart Parking Control Centre")

tab_live, tab_stats, tab_config, tab_pred = st.tabs(
    ["📹 Live monitor", "📊 Statistics & trends", "⚙️ Space configuration", "🔮 Predictions"]
)

with tab_live:
    live_monitor.render()
with tab_stats:
    stats_view.render()
with tab_config:
    bay_editor.render()
with tab_pred:
    st.info("Occupancy forecasting arrives in Step 8.")
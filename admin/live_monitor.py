import time
from pathlib import Path

import pandas as pd
import streamlit as st

from core import database as db

LATEST = Path(__file__).resolve().parent.parent / "data" / "latest_frame.jpg"


@st.fragment(run_every=2)          # refreshes every 2 seconds
def render():
    s = db.get_summary()
    pct = s["occupied"] / s["total"] * 100 if s["total"] else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total bays", s["total"])
    c2.metric("🔴 Occupied", s["occupied"])
    c3.metric("🟢 Available", s["available"])
    c4.metric("Occupancy", f"{pct:.0f}%")

    left, right = st.columns([3, 2])

    with left:
        st.subheader("AI vehicle detection")
        if LATEST.exists():
            age = time.time() - LATEST.stat().st_mtime
            if age < 15:
                st.success(f"🟢 AI worker running (last frame {age:.0f}s ago)")
            else:
                st.warning("🟠 AI worker looks offline. Start it with: "
                           "`python detection/worker.py --source data/demo.mp4`")
            try:
                st.image(LATEST.read_bytes(),
                         caption="Green = free, red = occupied, orange boxes = detected vehicles")
            except OSError:
                st.caption("Loading frame...")
        else:
            st.info("No video frame yet. Start the AI worker to see live detections.")

    with right:
        st.subheader("Bay status")
        spaces = db.get_spaces()
        if spaces:
            df = pd.DataFrame(spaces)
            df["Status"] = df["is_occupied"].map({0: "🟢 Free", 1: "🔴 Occupied"})
            df = df.rename(columns={"label": "Bay", "updated_at": "Updated"})
            st.dataframe(df[["Bay", "Status", "Updated"]], hide_index=True, height=300)
        else:
            st.info("No bays configured yet.")

        st.subheader("Recent activity")
        ev = db.get_events(8)
        if ev.empty:
            st.caption("No changes recorded yet.")
        else:
            ev["event"] = ev["event"].map({"occupied": "🔴 occupied", "vacated": "🟢 vacated"})
            st.dataframe(ev[["timestamp", "label", "event"]].rename(
                columns={"timestamp": "Time", "label": "Bay", "event": "Event"}),
                hide_index=True, height=260)
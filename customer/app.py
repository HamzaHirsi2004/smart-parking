import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from core import database as db

st.set_page_config(page_title="Smart Parking Finder", page_icon="🅿️", layout="wide")
db.init_db()


def build_map(spaces, entrances):
    df = pd.DataFrame(spaces)
    df["status"] = df["is_occupied"].map({0: "Available", 1: "Occupied"})
    df["color"] = df["is_occupied"].map({0: "#2ecc71", 1: "#e74c3c"})

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["map_x"], y=df["map_y"], mode="markers+text",
        text=df["label"], textposition="middle center",
        textfont=dict(color="white", size=11),
        marker=dict(symbol="square", size=34, color=df["color"]),
        customdata=df[["label", "status"]],
        hovertemplate="Bay %{customdata[0]}<br>%{customdata[1]}<extra></extra>",
        showlegend=False,
    ))
    if entrances:
        fig.add_trace(go.Scatter(
            x=[p[0] for p in entrances.values()],
            y=[p[1] for p in entrances.values()],
            mode="markers+text", text=list(entrances.keys()),
            textposition="top center",
            marker=dict(symbol="diamond", size=18, color="#3498db"),
            hoverinfo="skip", showlegend=False,
        ))
    fig.update_layout(
        height=480, margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(visible=False, fixedrange=True),
        yaxis=dict(visible=False, autorange="reversed",
                   scaleanchor="x", fixedrange=True),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


@st.fragment(run_every=5)          # re-reads the database every 5 seconds
def live_view():
    s = db.get_summary()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total spaces", s["total"])
    c2.metric("🟢 Available", s["available"])
    c3.metric("🔴 Occupied", s["occupied"])

    pct = s["occupied"] / s["total"] if s["total"] else 0
    st.progress(pct, text=f"{pct:.0%} full")
    st.caption(f"🕒 Last updated: {s['last_updated']}")

    st.subheader("Parking map")
    st.plotly_chart(build_map(db.get_spaces(), db.get_setting("entrances", {})))


st.title("🅿️ Smart Parking Finder")
live_view()

with st.expander("📈 Occupancy history (preview)"):
    hist = db.get_history()
    if hist.empty:
        st.info("No history yet.")
    else:
        recent = hist[hist["timestamp"] > hist["timestamp"].max() - pd.Timedelta(hours=48)]
        st.line_chart(recent.set_index("timestamp")["occupied"])
        
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from core import database as db
from core.routing import find_nearest
from core import forecast

st.set_page_config(page_title="Smart Parking Finder", page_icon="🅿️", layout="wide")
db.init_db()


def build_map(spaces, entrances, best_label=None):
    df = pd.DataFrame(spaces)
    df["status"] = df["is_occupied"].map({0: "Available", 1: "Occupied"})
    df["color"] = df["is_occupied"].map({0: "#2ecc71", 1: "#e74c3c"})

    fig = go.Figure()

    # gold halo behind the recommended bay
    if best_label:
        best = df[df["label"] == best_label]
        fig.add_trace(go.Scatter(
            x=best["map_x"], y=best["map_y"], mode="markers",
            marker=dict(symbol="square", size=46, color="#f1c40f"),
            hoverinfo="skip", showlegend=False,
        ))

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
def live_view(entrance_name):
    entrances = db.get_setting("entrances", {})
    s = db.get_summary()
    spaces = db.get_spaces()
    if not spaces:
        st.info("No parking bays have been configured yet.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Total spaces", s["total"])
    c2.metric("🟢 Available", s["available"])
    c3.metric("🔴 Occupied", s["occupied"])

    pct = s["occupied"] / s["total"] if s["total"] else 0
    st.progress(pct, text=f"{pct:.0%} full")
    st.caption(f"🕒 Last updated: {s['last_updated']}")

    nearest = find_nearest(spaces, entrances[entrance_name]) if entrance_name in entrances else []
    if nearest:
        best = nearest[0]
        st.success(f"⭐ Closest free bay from **{entrance_name}**: "
                   f"**{best['label']}** (about {best['distance_m']} m)")
    elif s["total"] and s["available"] == 0:
        st.error("The car park is currently full.")

    st.subheader("Parking map")
    st.plotly_chart(build_map(spaces, entrances, nearest[0]["label"] if nearest else None))

    if len(nearest) > 1:
        st.caption("Next best options: " + ", ".join(
            f"{n['label']} ({n['distance_m']} m)" for n in nearest[1:]))


st.title("🅿️ Smart Parking Finder")

entrances = db.get_setting("entrances", {})
entrance_name = st.selectbox("Which entrance are you using?", list(entrances.keys())) if entrances else None
live_view(entrance_name)

with st.expander("📈 Occupancy history (preview)"):
    hist = db.get_history()
    if hist.empty:
        st.info("No history yet.")
    else:
        recent = hist[hist["timestamp"] > hist["timestamp"].max() - pd.Timedelta(hours=48)]
        st.line_chart(recent.set_index("timestamp")["occupied"])

with st.expander("🔮 Predicted busy times (next 24 hours)"):
    pred = forecast.predict_future(24)
    if pred.empty:
        st.info("Predictions are not available yet.")
    else:
        st.line_chart(pred.set_index("timestamp")["predicted_pct"])
        quiet = pred.loc[pred["predicted_pct"].idxmin()]
        st.caption(f"Quietest expected time: **{quiet['timestamp']:%a %H:%M}** "
                   f"(about {quiet['predicted_pct']:.0f}% full)")
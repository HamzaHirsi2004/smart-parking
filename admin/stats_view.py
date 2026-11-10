import pandas as pd
import plotly.express as px
import streamlit as st

from core import database as db

RANGES = {"Last hour": 1 / 24, "Last 6 hours": 0.25, "Last 24 hours": 1,
          "Last 7 days": 7, "All time": None}


def render():
    full = db.get_history()
    if full.empty:
        st.info("No occupancy records yet. Start the AI worker to begin recording.")
        return
    full["pct"] = full["occupied"] / full["total"].where(full["total"] > 0) * 100

    choice = st.radio("Period", list(RANGES), index=2, horizontal=True)
    days = RANGES[choice]
    hist = full if days is None else full[full["timestamp"] >= pd.Timestamp.now() - pd.Timedelta(days=days)]
    if hist.empty:
        st.warning("No records in this period.")
        return

    peak = hist.loc[hist["pct"].idxmax()]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Average occupancy", f"{hist['pct'].mean():.0f}%")
    k2.metric("Peak occupancy", f"{peak['pct']:.0f}%", help=f"at {peak['timestamp']}")
    k3.metric("Lowest occupancy", f"{hist['pct'].min():.0f}%")
    k4.metric("Records", f"{len(hist):,}")

    # ---- trend over time ----
    ts = hist.set_index("timestamp")["pct"]
    span = ts.index.max() - ts.index.min()
    rule = "1min" if span <= pd.Timedelta(hours=2) else "15min" if span <= pd.Timedelta(days=2) else "1h"
    trend = ts.resample(rule).mean().dropna().reset_index()
    fig = px.line(trend, x="timestamp", y="pct", title="Occupancy over time",
                  labels={"pct": "Occupancy (%)", "timestamp": ""})
    fig.update_yaxes(range=[0, 100])
    st.plotly_chart(fig)

    # ---- typical day (uses ALL history, since patterns need lots of data) ----
    by_hour = full.groupby(full["timestamp"].dt.hour)["pct"].mean().reset_index()
    by_hour.columns = ["Hour of day", "Average occupancy (%)"]
    fig2 = px.bar(by_hour, x="Hour of day", y="Average occupancy (%)",
                  title="Average occupancy by hour of day")
    fig2.update_yaxes(range=[0, 100])
    st.plotly_chart(fig2)
    busiest = by_hour.loc[by_hour["Average occupancy (%)"].idxmax()]
    st.caption(f"Busiest hour on record: **{int(busiest['Hour of day']):02d}:00** "
               f"(about {busiest['Average occupancy (%)']:.0f}% full)")

    # ---- records and export ----
    with st.expander("📋 Bay events (database records)"):
        ev = db.get_events(500)
        if ev.empty:
            st.caption("No bay changes recorded yet.")
        else:
            st.dataframe(ev, hide_index=True)
            st.download_button("Download events (CSV)", ev.to_csv(index=False), "bay_events.csv")
    st.download_button("Download occupancy history (CSV)", full.to_csv(index=False),
                       "occupancy_history.csv")
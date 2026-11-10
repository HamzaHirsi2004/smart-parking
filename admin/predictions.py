import pandas as pd
import plotly.express as px
import streamlit as st

from core import database as db
from core import forecast


def render():
    st.subheader("Occupancy forecasting")

    sources = ["Database history (occupancy_log)"]
    if forecast.PKLOT_CSV.exists():
        sources.append("PKLot history (data/pklot_history.csv)")
    src = st.radio("Training data", sources, horizontal=True)

    if st.button("🧠 Train / retrain model", type="primary"):
        try:
            df = pd.read_csv(forecast.PKLOT_CSV) if src.startswith("PKLot") else db.get_history()
            if df.empty:
                raise ValueError("That data source is empty.")
            with st.spinner("Training..."):
                forecast.train(forecast.prepare(df))
            st.success("Model trained and saved.")
        except ValueError as e:
            st.error(str(e))

    b = forecast.load_model()
    if b is None:
        st.info("No model yet. Choose a data source and click Train.")
        return

    st.markdown("**Model accuracy** (tested on the newest 20% of the data)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model error (MAE)", f"{b['mae']:.1f} pts",
              help="Average difference between predicted and real occupancy, in percentage points.")
    c2.metric("Baseline error", f"{b['baseline_mae']:.1f} pts",
              help="A simple 'average for this hour' guess. The model should beat this.")
    c3.metric("Training data", f"{b['span_days']} days")
    c4.metric("Calendar features", "Day + hour" if b["use_dow"] else "Hour only")
    if b["mae"] < b["baseline_mae"]:
        st.caption(f"✅ The model beats the baseline. Trained {b['trained_at']} on {b['n_points']:,} slots.")
    else:
        st.caption("⚠️ The model does not beat the simple baseline. More history is needed.")
    if b["test_points"] < 300:
        st.caption("Note: the test window is short, so treat the accuracy figure as a rough guide.")

    st.markdown("**Forecast: next 24 hours**")
    pred = forecast.predict_future(24)
    if pred.empty:
        st.warning("No forecast available for the coming hours.")
        return
    fig = px.line(pred, x="timestamp", y="predicted_pct", markers=True,
                  labels={"predicted_pct": "Predicted occupancy (%)", "timestamp": ""})
    fig.update_yaxes(range=[0, 100])
    st.plotly_chart(fig)

    peak = pred.loc[pred["predicted_pct"].idxmax()]
    quiet = pred.loc[pred["predicted_pct"].idxmin()]
    k1, k2 = st.columns(2)
    k1.metric("Predicted busiest time", peak["timestamp"].strftime("%a %H:%M"),
              f"{peak['predicted_pct']:.0f}% full", delta_color="off")
    k2.metric("Predicted quietest time", quiet["timestamp"].strftime("%a %H:%M"),
              f"{quiet['predicted_pct']:.0f}% full", delta_color="off")
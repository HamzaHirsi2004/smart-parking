from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "data" / "occupancy_model.joblib"
PKLOT_CSV = ROOT / "data" / "pklot_history.csv"
STEP = "15min"


def prepare(df):
    """occupancy table -> a 15-minute time series of occupancy %."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["pct"] = df["occupied"] / df["total"].where(df["total"] > 0) * 100
    return df.set_index("timestamp")["pct"].dropna().sort_index().resample(STEP).mean().dropna()


def make_features(index, use_dow):
    hour = index.hour + index.minute / 60
    X = pd.DataFrame({
        "hour": hour,
        "h_sin": np.sin(2 * np.pi * hour / 24),
        "h_cos": np.cos(2 * np.pi * hour / 24),
    }, index=index)
    if use_dow:
        X["dow"] = index.dayofweek
        X["is_weekend"] = (index.dayofweek >= 5).astype(int)
    return X


def train(series):
    if len(series) < 192:          # 2 days of 15-minute slots
        raise ValueError("Not enough history. Need at least 2 days of records "
                         f"(have {len(series)} slots of 15 minutes).")

    span_days = (series.index.max() - series.index.min()).days + 1
    use_dow = span_days >= 14

    cut = int(len(series) * 0.8)
    train_s, test_s = series.iloc[:cut], series.iloc[cut:]

    def fit(s):
        m = RandomForestRegressor(n_estimators=200, min_samples_leaf=3,
                                  random_state=42, n_jobs=-1)
        m.fit(make_features(s.index, use_dow), s.values)
        return m

    # honest evaluation on the newest 20 %
    model = fit(train_s)
    pred = model.predict(make_features(test_s.index, use_dow))
    mae = mean_absolute_error(test_s.values, pred)

    hour_mean = train_s.groupby(train_s.index.hour).mean()
    base_pred = test_s.index.hour.map(hour_mean).to_numpy(dtype=float)
    base_pred = np.where(np.isnan(base_pred), train_s.mean(), base_pred)
    baseline_mae = mean_absolute_error(test_s.values, base_pred)

    # final model uses ALL the data
    final = fit(series)
    bundle = {
        "model": final,
        "use_dow": use_dow,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "n_points": int(len(series)),
        "span_days": int(span_days),
        "test_points": int(len(test_s)),
        "mae": float(mae),
        "baseline_mae": float(baseline_mae),
        "hours": sorted(set(int(h) for h in series.index.hour)),   # hours the data covers
    }
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    return bundle


def load_model():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


def predict_future(hours=24):
    """Predicted occupancy % for the next `hours` hours (15-minute steps)."""
    b = load_model()
    if b is None:
        return pd.DataFrame(columns=["timestamp", "predicted_pct"])
    start = pd.Timestamp.now().ceil(STEP)
    idx = pd.date_range(start, periods=hours * 4, freq=STEP)
    preds = np.clip(b["model"].predict(make_features(idx, b["use_dow"])), 0, 100)
    out = pd.DataFrame({"timestamp": idx, "predicted_pct": preds})
    # don't predict hours the training data never covered (e.g. night, if the camera data stops)
    return out[out["timestamp"].dt.hour.isin(b["hours"])].reset_index(drop=True)
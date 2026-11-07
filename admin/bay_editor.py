import json
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

from core import database as db

ROOT = Path(__file__).resolve().parent.parent
REF_FRAME = ROOT / "data" / "reference_frame.jpg"
DISPLAY_W = 1000


# ---------------- helpers ----------------
def grab_frame(video_path, seconds):
    p = Path(video_path)
    if not p.is_absolute():
        p = ROOT / p
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def recompute_map_coords():
    """Convert pixel positions into customer-map units (1 unit ~ one bay)."""
    polys = {}
    for s in db.get_spaces():
        if s["polygon"]:
            polys[s["label"]] = np.array(json.loads(s["polygon"]), dtype=np.float32)

    if polys:
        unit = float(np.median([np.sqrt(cv2.contourArea(p)) for p in polys.values()]))
        unit = unit if unit > 1 else 60.0
        db.set_setting("map_unit", unit)
        db.update_map_coords({
            label: (float(p[:, 0].mean() / unit), float(p[:, 1].mean() / unit))
            for label, p in polys.items()
        })
    else:
        unit = db.get_setting("map_unit", 60.0)

    ent_px = db.get_setting("entrances_px", {})
    db.set_setting("entrances", {n: [x / unit, y / unit] for n, (x, y) in ent_px.items()})


def next_label():
    existing = {s["label"] for s in db.get_spaces()}
    n = len(existing) + 1
    while f"P{n}" in existing:
        n += 1
    return f"P{n}"


def draw_overlay(frame, spaces, points, entrances_px, entrance_pt):
    img = frame.copy()
    for s in spaces:
        if not s["polygon"]:
            continue
        pts = np.array(json.loads(s["polygon"]), dtype=np.int32)
        cv2.polylines(img, [pts], True, (0, 255, 255), 2)
        cx, cy = pts.mean(axis=0)
        cv2.putText(img, s["label"], (int(cx) - 12, int(cy) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    if points:
        cv2.polylines(img, [np.array(points, dtype=np.int32)], False, (0, 120, 255), 2)
        for x, y in points:
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
    for name, (x, y) in entrances_px.items():
        cv2.circle(img, (int(x), int(y)), 9, (255, 120, 0), -1)
        cv2.putText(img, name, (int(x) + 12, int(y) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 120, 0), 2)
    if entrance_pt:
        cv2.circle(img, tuple(entrance_pt), 9, (255, 255, 0), 2)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# ---------------- button callbacks ----------------
def undo_point():
    if st.session_state.points:
        st.session_state.points.pop()


def clear_points():
    st.session_state.points = []


def save_bay():
    pts = st.session_state.points
    label = st.session_state.bay_label.strip()
    if len(pts) < 3 or not label:
        st.session_state.msg = ("error", "Click at least 3 corners and enter a label.")
        return
    db.add_space(label, 0.0, 0.0, polygon=pts)
    recompute_map_coords()
    st.session_state.points = []
    st.session_state.msg = ("success", f"Saved bay {label}.")
    st.session_state.bay_label = next_label()


def save_entrance():
    pt = st.session_state.entrance_pt
    name = st.session_state.entrance_name.strip()
    if not pt or not name:
        st.session_state.msg = ("error", "Click the entrance on the image and enter a name.")
        return
    ent = db.get_setting("entrances_px", {})
    ent[name] = pt
    db.set_setting("entrances_px", ent)
    recompute_map_coords()
    st.session_state.entrance_pt = None
    st.session_state.msg = ("success", f"Saved entrance {name}.")


def delete_bay(label):
    db.delete_space(label)
    recompute_map_coords()
    st.session_state.msg = ("success", f"Deleted bay {label}.")


def delete_entrance(name):
    ent = db.get_setting("entrances_px", {})
    ent.pop(name, None)
    db.set_setting("entrances_px", ent)
    recompute_map_coords()
    st.session_state.msg = ("success", f"Deleted entrance {name}.")


# ---------------- page ----------------
def render():
    st.session_state.setdefault("points", [])
    st.session_state.setdefault("entrance_pt", None)
    st.session_state.setdefault("bay_label", next_label())
    st.session_state.setdefault("entrance_name", "Main Gate")

    with st.expander("1. Choose the reference frame", expanded=not REF_FRAME.exists()):
        c1, c2, c3 = st.columns([3, 1, 1])
        video = c1.text_input("Video path", "data/demo.mp4")
        sec = c2.number_input("Second", 0.0, 3600.0, 0.0, 1.0)
        c3.write("")
        if c3.button("Use this frame"):
            frame = grab_frame(video, sec)
            if frame is None:
                st.error("Could not read that video. Check the path.")
            else:
                cv2.imwrite(str(REF_FRAME), frame)
                h, w = frame.shape[:2]
                db.set_setting("frame_size", [w, h])
                st.session_state.points = []
                st.rerun()
        st.caption("If you change the frame after drawing bays, the old boundaries "
                   "won't match. Only change it if the camera view is the same.")

    if not REF_FRAME.exists():
        st.info("Pick a reference frame above to start.")
        return

    frame = cv2.imread(str(REF_FRAME))
    fh, fw = frame.shape[:2]
    spaces = db.get_spaces()
    entrances_px = db.get_setting("entrances_px", {})

    msg = st.session_state.pop("msg", None)
    if msg:
        getattr(st, msg[0])(msg[1])

    st.markdown("**2. Draw the bays**")
    mode = st.radio("Mode", ["Draw a bay", "Place an entrance"], horizontal=True)

    left, right = st.columns([3, 1])
    with left:
        img = draw_overlay(frame, spaces, st.session_state.points,
                           entrances_px, st.session_state.entrance_pt)
        click = streamlit_image_coordinates(img, width=min(fw, DISPLAY_W), key="canvas")
        if click:
            sig = (click["x"], click["y"], click.get("unix_time"))
            if sig != st.session_state.get("last_click"):
                st.session_state.last_click = sig
                scale = fw / click["width"]
                x, y = int(click["x"] * scale), int(click["y"] * scale)
                if mode == "Draw a bay":
                    st.session_state.points.append([x, y])
                else:
                    st.session_state.entrance_pt = [x, y]
                st.rerun()

    with right:
        if mode == "Draw a bay":
            st.write(f"Corners clicked: **{len(st.session_state.points)}**")
            st.text_input("Bay label", key="bay_label")
            b1, b2 = st.columns(2)
            b1.button("↩ Undo", on_click=undo_point)
            b2.button("✖ Clear", on_click=clear_points)
            st.button("💾 Save bay", type="primary", on_click=save_bay)
            st.caption("Click the 4 corners around the bay, then Save.")
        else:
            st.write("Click where cars enter, then name it.")
            st.text_input("Entrance name", key="entrance_name")
            st.button("💾 Save entrance", type="primary", on_click=save_entrance)
        st.divider()
        st.metric("Bays configured", len(spaces))
        st.metric("Entrances", len(entrances_px))

    st.markdown("**3. Manage**")
    m1, m2 = st.columns(2)
    with m1:
        labels = [s["label"] for s in spaces]
        if labels:
            sel = st.selectbox("Bay to delete", labels)
            st.button("🗑 Delete bay", on_click=delete_bay, args=(sel,))
    with m2:
        if entrances_px:
            ent_sel = st.selectbox("Entrance to delete", list(entrances_px.keys()))
            st.button("🗑 Delete entrance", on_click=delete_entrance, args=(ent_sel,))

    with st.expander("⚠️ Danger zone"):
        confirm = st.checkbox("I understand this permanently deletes data")
        d1, d2 = st.columns(2)
        if d1.button("Remove ALL bays and entrances", disabled=not confirm):
            db.clear_spaces()
            db.set_setting("entrances_px", {})
            db.set_setting("entrances", {})
            st.session_state.points = []
            st.rerun()
        if d2.button("Clear occupancy history", disabled=not confirm):
            db.clear_history()
            st.success("History cleared.")
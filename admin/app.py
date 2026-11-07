import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st
from core import database as db
from admin import bay_editor

st.set_page_config(page_title="Smart Parking Control Centre", page_icon="🛠️", layout="wide")
db.init_db()

st.title("🛠️ Smart Parking Control Centre")

tab_config, = st.tabs(["⚙️ Space configuration"])
with tab_config:
    bay_editor.render()
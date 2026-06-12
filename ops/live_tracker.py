import streamlit as st
import pandas as pd
import json
import time
import streamlit.components.v1 as components
from pathlib import Path

# Add project root to path so we can import ops
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ops.data_fetch.cli import _status_for_source, REGISTRY

st.set_page_config(page_title="MBAL Live Tracker", layout="wide", page_icon="🌊")

st.title("🌊 MBAL Data Fetch Live Tracker")
st.markdown("Live view of data fetch operations and status.")

# Auto-refresh every 5 seconds
components.html("<script>setTimeout(function(){window.parent.location.reload()}, 5000)</script>", height=0)

st.write(f"**Last Updated:** {time.strftime('%Y-%m-%d %H:%M:%S')}")

try:
    entries = [_status_for_source(k) for k in sorted(REGISTRY, key=lambda k: REGISTRY[k].priority)]
    
    if not entries:
        st.warning("No data fetch sources found in registry.")
    else:
        df = pd.DataFrame(entries)
        
        # Reorder and format columns for better readability
        cols = ["priority", "source", "status", "rows", "type", "mode", "date_min", "date_max"]
        # Only keep columns that exist
        cols = [c for c in cols if c in df.columns]
        
        # Move any extra columns to the end
        extra_cols = [c for c in df.columns if c not in cols and c != "title"]
        df = df[cols + extra_cols]
        
        # Apply some basic styling
        def color_status(val):
            color = ""
            if val == "READY_FOR_MODELING":
                color = "green"
            elif val == "FAILED":
                color = "red"
            elif val == "FETCHED_NEEDS_REVIEW":
                color = "orange"
            elif val == "NOT_STARTED":
                color = "gray"
            return f'color: {color}'

        styled_df = df.style.map(color_status, subset=['status'])
        
        st.dataframe(styled_df, use_container_width=True, height=600)
        
except Exception as e:
    st.error(f"Error loading tracker data: {e}")

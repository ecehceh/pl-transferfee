"""
Entry point and navigation for the scouting app.

    streamlit run app.py

Two pages: the club fit finder (scouting tool, default) and the fee-vs-value
analysis the project grew out of. Page titles are set here so the sidebar nav
reads "Club fit finder" / "Fee analysis" rather than the script filenames.
"""

import streamlit as st

st.set_page_config(page_title="Scout — PL transfer value", page_icon=":material/search:", layout="wide")

nav = st.navigation([
    st.Page("pages/1_Club_fit_finder.py", title="Club fit finder", icon=":material/search:", default=True),
    st.Page("pages/2_Fee_analysis.py", title="Fee analysis", icon=":material/payments:"),
])
nav.run()

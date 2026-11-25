"""Dedicated mobile-friendly broker workspace for property publishing."""

import streamlit as st

st.set_page_config(
    page_title="AI Estate | Broker Portal",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @media (max-width: 768px) {
        .block-container {
            padding: 1rem 0.75rem 2rem !important;
        }
        .stButton > button {
            min-height: 48px;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.25rem;
            overflow-x: auto;
            flex-wrap: nowrap;
        }
        .stTabs [data-baseweb="tab"] {
            white-space: nowrap;
            padding: 0.5rem 0.65rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🏢 Broker Portal")
st.caption("Publish and manage verified property listings from your phone or desktop.")

from broker_portal_ui import render_broker_portal

render_broker_portal(
    default_lat=12.9716,
    default_lon=77.5946,
)

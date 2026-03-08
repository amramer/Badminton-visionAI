# webapp/pages/coach_report.py
from __future__ import annotations
from typing import Any, Dict
import streamlit as st

from webapp.tabs import positioning, shot_profile
from webapp.report import generate_coach_report


def render(dashboard, tracking_data: Dict[str, Any], config: Dict[str, Any]) -> None:
    """
    Coach Report page.
    Thin router that delegates each tab to its own module.
    """

    st.markdown('<div class="section-header">Coach Report</div>', unsafe_allow_html=True)

    
    pdf_bytes = generate_coach_report(tracking_data, config)
    st.download_button(
        "📄 Download Coach Report",
        data=pdf_bytes,
        file_name="coach_assistant_report.pdf",
        mime="application/pdf",
    )

    tab1, tab2 = st.tabs([
        "POSITIONING",
        "SHOT PROFILE",
    ])

    with tab1:
        positioning.render(dashboard, tracking_data, config)

    with tab2:
        shot_profile.render(dashboard, tracking_data, config)  

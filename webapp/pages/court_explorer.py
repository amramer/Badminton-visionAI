# webapp/pages/court_explorer.py
from __future__ import annotations

from typing import Any, Dict

import os
import streamlit as st


def render(config: Dict[str, Any]) -> None:
    st.markdown('<div class="section-header">🏸 3D COURT EXPLORER</div>', unsafe_allow_html=True)
    video_path = config["streamlit_dashboard"].get("video_path_3d", "data/court3d.mp4")

    if os.path.exists(video_path):
        with st.container(border=True):
            st.video(video_path, autoplay=True, loop=True)
    else:
        st.error(f"Video asset not found at: {video_path}")
        st.image("data/images/court-view.jpg", caption="Standard Badminton Court Layout", use_column_width=True)

    st.divider()
    st.subheader("Court Specifications", divider="orange")

    col_dim, col_rules = st.columns(2, gap="large")
    with col_dim:
        with st.container(border=True, height=300):
            st.markdown("### 📏 Dimensions")
            st.markdown(
                """
- **Full length**: 13.4 m
- **Doubles width**: 6.1 m
- **Singles width**: 5.18 m
- **Service line**: 1.98 m from net
- **Back boundary**: 0.76 m (doubles)
- **Net height**: 1.55 m (edges), 1.524 m (center)
                """
            )
    with col_rules:
        with st.container(border=True, height=300):
            st.markdown("### 📜 Game Rules")
            st.markdown(
                """
- **Format**: Best of 3 games (21 points)
- **Winning**: By 2 points (capped at 30)
- **Scoring**: Rally point system
- **Service**: Underhand, below waist
- **Landing**: Diagonal service court
- **Rotation**: Change ends after each game
                """
            )

    st.caption("All measurements follow BWF tournament standards")

# webapp/pages/match_replay.py
from __future__ import annotations

from typing import Any, Dict

import os
import numpy as np
import streamlit as st


def render(dashboard, tracking_data: Dict[str, Any], config: Dict[str, Any]) -> None:
    def _fmt_mmss(t: float) -> str:
        if t is None or not np.isfinite(t) or t < 0:
            return "—"
        m = int(t // 60)
        s = int(t % 60)
        return f"{m:02d}:{s:02d}"

    col_video, col_sidecourt = st.columns([8, 2])

    # ============================================================
    # LEFT: VIDEO + RALLY SUMMARY
    # ============================================================
    with col_video:
        st.markdown('<div class="section-header">Live Tracking</div>', unsafe_allow_html=True)

        video_path = config.get("video_path", "outputs/tracking_results/final_analysis.mp4")
        if os.path.exists(video_path):
            st.video(video_path, autoplay=True, loop=True)
        else:
            st.warning(f"Video not found: {video_path}")

        rs = tracking_data.get("rally_summary", {}) or {}
        shots = int(rs.get("shots", 0))
        duration_s = float(rs.get("duration_s", 0.0))
        shot_rate = float(rs.get("shot_rate", 0.0))

        duration_str = "—"
        if duration_s > 0:
            if duration_s >= 60:
                m = int(duration_s // 60)
                s = int(duration_s % 60)
                duration_str = f"{m}m {s:02d}s"
            else:
                duration_str = f"{duration_s:.0f}s"

        shot_rate_str = f"{shot_rate:.2f}/s" if duration_s > 0 else "—"

        st.markdown(
            f"""
            <div style="
                margin-top: 11px;
                background: var(--secondary-color);
                padding: 11px;
                margin-bottom: 12px;
                border-radius: 8px;
                border: 1px solid #333;
                min-height: 145px;
            ">
                <div style="
                    color: var(--primary-color);
                    text-align: center;
                    font-weight: bold;
                    font-size: 1.2rem;
                    margin-bottom: 14px;
                ">
                    RALLY SUMMARY
                </div>
                <div style="display: flex; justify-content: space-between; text-align: center;">
                    <div style="flex: 1;">
                        <div style="color: #ffa500; font-size: 1.3rem;">{shots}</div>
                        <div style="font-size: 0.9rem;">Shots</div>
                    </div>
                    <div style="flex: 1; border-left: 1px solid #333; border-right: 1px solid #333;">
                        <div style="color: #ffa500; font-size: 1.3rem;">{duration_str}</div>
                        <div style="font-size: 0.9rem;">Duration</div>
                    </div>
                    <div style="flex: 1;">
                        <div style="color: #ffa500; font-size: 1.3rem;">{shot_rate_str}</div>
                        <div style="font-size: 0.9rem;">Shot Rate</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ============================================================
    # RIGHT: COURT HEATMAP + CURRENT P1/P2 + SLIDER BELOW IMAGE
    # ============================================================
    with col_sidecourt:
        st.markdown('<div class="section-header">Heatmap View</div>', unsafe_allow_html=True)

        target_w = int(config["streamlit_dashboard"].get("court_width", 385))
        target_h = int(config["streamlit_dashboard"].get("court_height", 840))

        court_img = dashboard.load_and_resize_image(
            config["streamlit_dashboard"].get("court_image_path", "data/images/court-view.jpg"),
            target_width=target_w,
            target_height=target_h,
            mode="stretch",
        )

        p1_positions = tracking_data.get("player1_positions", []) or []
        p2_positions = tracking_data.get("player2_positions", []) or []
        frame_ids = tracking_data.get("frame_ids", []) or []

        if court_img is None:
            st.warning("Court image not available")
            return
        if not p1_positions or not p2_positions:
            st.warning("No player positions available")
            return

        last_idx = min(len(p1_positions), len(p2_positions)) - 1
        if last_idx < 0:
            st.warning("Not enough points to render")
            return

        fps = float(config.get("video_fps", config.get("fps", 30.0)) or 30.0)
        fps = fps if fps > 1e-6 else 30.0

        state_key = "heatmap_frame_idx"
        if state_key not in st.session_state:
            st.session_state[state_key] = last_idx

        idx = int(st.session_state[state_key])
        idx = max(0, min(last_idx, idx))

        if len(frame_ids) == max(len(p1_positions), len(p2_positions)) and idx < len(frame_ids):
            frame_num = int(frame_ids[idx])
        else:
            frame_num = idx

        t_sec = frame_num / fps
        t_str = _fmt_mmss(t_sec)

        heatmap_img = dashboard.create_heatmap(court_img, p1_positions, p2_positions)
        final_img = dashboard.annotate_court(heatmap_img.copy(), p1_positions[idx], p2_positions[idx])
        st.image(final_img)

        dashboard.show_player_legend()

        st.slider(
            f"Time {t_str}  •  Frame {frame_num}",
            0,
            last_idx,
            idx,
            key=state_key,
        )

        with st.expander("📍 Players Positions", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Player 1 X(px)", p1_positions[idx][0])
                st.metric("Player 1 Y(px)", p1_positions[idx][1])
            with c2:
                st.metric("Player 2 X(px)", p2_positions[idx][0])
                st.metric("Player 2 Y(px)", p2_positions[idx][1])

    # ============================================================
    # BOTTOM: PLAYER + BALL PANELS
    # ============================================================
    with st.container(height=1350, border=True):
        col_p1, col_p2, col_ball = st.columns([4, 4, 2])

        p1_static = config["streamlit_dashboard"].get("player_data", {}).get("CHOU TIEN CHEN", {})
        p2_static = config["streamlit_dashboard"].get("player_data", {}).get("ALEXANDRE LANIER", {})

        p1_dyn = (tracking_data.get("player_dyn", {}) or {}).get(1, {})
        p2_dyn = (tracking_data.get("player_dyn", {}) or {}).get(2, {})

        p1_shots = tracking_data.get("player1_shots", {})
        p2_shots = tracking_data.get("player2_shots", {})

        tsec = tracking_data.get("time_s", []) or []

        with col_p1:
            st.markdown(
                '<div style="font-size: 1.2rem; color: #ffa500; font-weight: bold; text-align: center;margin-top: 15px ;margin-bottom: 15px;">PLAYER 1</div>',
                unsafe_allow_html=True,
            )
            dashboard.display_player_card(
                player_name="CHOU TIEN CHEN",
                static_info=p1_static,
                dynamic_info=p1_dyn,
                shots=p1_shots,
                time_s=tsec,
            )

        with col_p2:
            st.markdown(
                '<div style="font-size: 1.2rem; color: #ffa500; font-weight: bold; text-align: center;margin-top: 15px; margin-bottom: 15px;">PLAYER 2</div>',
                unsafe_allow_html=True,
            )
            dashboard.display_player_card(
                player_name="ALEXANDRE LANIER",
                static_info=p2_static,
                dynamic_info=p2_dyn,
                shots=p2_shots,
                time_s=tsec,
            )

        with col_ball:
            st.markdown(
                '<div style="font-size: 1.2rem; color: #ffa500; font-weight: bold; text-align: center; margin-top: 15px;margin-bottom: 15px;">BALL</div>',
                unsafe_allow_html=True,
            )
            dashboard.display_ball_stats(tracking_data)

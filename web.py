#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from webapp import StreamlitDashboard
from webapp.pages import coach_report, court_explorer, match_replay
from config import load_config
from utils import get_logger

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(layout="wide", page_title="Badminton AI Analytics")

# -------------------------------
# Custom CSS (dark theme)
# -------------------------------
st.markdown(
    """
    <style>
        :root {
            --primary-color: #ffa500;
            --secondary-color: #111111;
            --text-color: #ffffff;
            --metric-bg: #222222;
            --border-color: #333333;
            --highlight-color: #ffc04d;
        }

        body {
            background-color: #000000;
            color: var(--text-color);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        .main { background-color: #000000; }

        .stats-frame {
            position: relative;
            border: 1px solid #444;
            border-radius: 8px;
            padding: 25px 20px 20px;
            margin: 30px 0;
            background: var(--secondary-color);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        .stats-header {
            position: absolute;
            top: -12px;
            left: 25px;
            background: var(--secondary-color);
            padding: 0 15px;
            color: var(--primary-color);
            font-weight: bold;
            font-size: 1.1rem;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }
        .stats-content { margin-top: 10px; }

        .player-card {
            background-color: var(--secondary-color);
            border-radius: 5px;
            margin-bottom: 5px;
            height: 100%;
            display: flex;
            flex-direction: column;
            border: 1px solid var(--border-color);
            transition: all 0.3s ease;
        }
        .player-card:hover {
            border-color: var(--primary-color);
            box-shadow: 0 0 15px rgba(255,165,0,0.2);
        }

        .ball-stats {
            background-color: var(--secondary-color);
            border-radius: 5px;
            margin-bottom: 35px;
            border: 1px solid var(--border-color);
            height: 100%;
        }

        .section-header {
            font-size: 1.2rem;
            font-weight: 600;
            color: var(--primary-color);
            margin-bottom: 15px;
            text-align: center;
            text-transform: uppercase;
            letter-spacing: 1px;
            position: relative;
        }
        .section-header::after {
            content: "";
            position: absolute;
            bottom: -8px;
            left: 50%;
            transform: translateX(-50%);
            width: 80px;
            height: 3px;
            background: var(--primary-color);
            border-radius: 3px;
        }

        .metric-card {
            background-color: var(--metric-bg);
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
            border-left: 4px solid var(--primary-color);
            transition: transform 0.2s ease;
        }
        .metric-card:hover { transform: translateY(-3px); }

        .player-header {
            display: flex;
            align-items: center;
            margin-bottom: 20px;
            gap: 15px;
            padding: 10px;
        }

        .player-image {
            width: 220px;
            height: 250px;
            border-radius: 45%;
            object-fit: cover;
            border: 4px solid var(--primary-color);
            margin-right: 8px;
            transition: all 0.3s ease;
        }
        .player-image:hover {
            transform: scale(1.02);
            box-shadow: 0 0 20px rgba(255,165,0,0.3);
        }

        .player-name {
            font-size: 1.7rem;
            font-weight: 700;
            color: var(--primary-color);
            margin: -40px 0 5px 0;
            text-transform: uppercase;
            letter-spacing: 1px;
            flex-grow: 1;
            text-shadow: 0 0 10px rgba(255,165,0,0.3);
        }

        .player-info {
            font-family: 'Segoe UI', Roboto, -apple-system, sans-serif;
            font-size: 15px;
            line-height: 1.9;
            color: #999;
            margin-top: 10px;
        }

        .stTabs [data-baseweb="tab"] {
            color: white;
            font-weight: bold;
            transition: all 0.2s ease;
        }
        .stTabs [data-baseweb="tab"]:hover { color: var(--highlight-color); }

        .stSelectbox > div > div {
            background-color: var(--metric-bg) !important;
            color: white !important;
            border: 1px solid var(--border-color) !important;
        }

        .court-image {
            width: 100%;
            height: auto;
            object-fit: contain;
            border-radius: 8px;
            border: 1px solid var(--border-color);
        }

        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(255,165,0,0.4); }
            70% { box-shadow: 0 0 0 10px rgba(255,165,0,0); }
            100% { box-shadow: 0 0 0 0 rgba(255,165,0,0); }
        }
        .highlight { animation: pulse 1.5s infinite; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------
# ASCII logo (keep it)
# -------------------------------
logo_html = """
<div style="
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    font-family: 'Courier New', monospace;
    white-space: pre;
    text-align: center;
    line-height: 1.3;
    font-size: 1.0vw;
    letter-spacing: 1px;
    margin-bottom: 12px;
    color: #ffffff;
    text-shadow: 2px 2px 4px rgba(255, 165, 0, 0.8);
">
<pre>
██████╗  █████╗ ██████╗ ███╗   ███╗██╗███╗   ██╗████████╗ ██████╗ ███╗   ██╗
██╔══██╗██╔══██╗██╔══██╗████╗ ████║██║████╗  ██║╚══██╔══╝██╔═══██╗████╗  ██║
██████╔╝███████║██║  ██║██╔████╔██║██║██╔██╗ ██║   ██║   ██║   ██║██╔██╗ ██║
██╔══██╗██╔══██║██║  ██║██║╚██╔╝██║██║██║╚██╗██║   ██║   ██║   ██║██║╚██╗██║
██████╔╝██║  ██║██████╔╝██║ ╚═╝ ██║██║██║ ╚████║   ██║   ╚██████╔╝██║ ╚████║
╚═════╝ ╚═╝  ╚═╝╚═════╝ ╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝   ╚═╝    ╚═════╝ ╚═╝  ╚═══╝
</pre>
    <div style="
        font-size: 1.1vw;
        font-weight: bold;
        color: #FD9F00;
        border-bottom: 3px solid #333333;
        padding-bottom: 7px;
        margin-top: -10px;
    ">
🎾 AI-Powered Badminton Performance Analysis System 🎾
    </div>
</div>
"""

# -------------------------------
# Caching + IO
# -------------------------------
@st.cache_data(show_spinner=False)
def _read_json_cached(path_str: str) -> Any:
    path = Path(path_str)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def discover_json_files(json_dir: Path) -> List[Path]:
    if not json_dir.exists() or not json_dir.is_dir():
        return []
    return sorted([p for p in json_dir.glob("*.json") if p.is_file()])


def load_json_bundle(json_paths: List[Path], logger) -> Dict[str, Any]:
    bundle: Dict[str, Any] = {}

    for p in json_paths:
        key = p.stem
        try:
            bundle[key] = _read_json_cached(str(p.resolve()))
        except Exception as e:
            logger.exception("Failed to load JSON: %s", str(p))
            bundle[key] = {"__error__": str(e), "__path__": str(p.resolve())}

    bundle["__meta__"] = {
        "source_dir": str(json_paths[0].parent.resolve()) if json_paths else str(None),
        "files": [p.name for p in json_paths],
    }
    return bundle


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default="config/config.yaml", help="Path to config YAML")
    parser.add_argument("--json_dir", default="data/json", help="Directory containing pipeline JSON outputs (default: data/json)")
    return parser.parse_known_args(argv)[0]


def select_jsons(json_dir: Path) -> List[Path]:
    expected_order = [
        "court_keypoints.json",
        "players_tracking.json",
        "ball_tracking.json",
        "shot_events.json",
        "player_shot_history.json",
        "players_final_metrics.json",
        "ball_final_metrics.json",
        "final_shots_stats.json",
    ]

    all_files = {p.name: p for p in discover_json_files(json_dir)}
    selected = [all_files[name] for name in expected_order if name in all_files]
    return selected if selected else list(all_files.values())


def main():
    # -------------------------------
    # Sidebar (brand + profile)
    # -------------------------------
    with st.sidebar:

        # logo
        logo_path = Path("assets/logo.png")
        if logo_path.exists():
            logo_bytes = logo_path.read_bytes()
            logo_base64 = base64.b64encode(logo_bytes).decode()

            st.markdown(
                f"""
                <div style="
                    display:flex;
                    justify-content:center;
                    align-items:center;
                    margin:-46px -46px -46px -46px;
                ">
                    <img src="data:image/png;base64,{logo_base64}"
                        style="
                            max-width:190px;
                            width:100%;
                            height:auto;
                            object-fit:contain;
                            display:block;
                        ">
                </div>
                """,
                unsafe_allow_html=True,
            )

        # profile image
        profile_path = Path("assets/sidebar-profile.jpeg")
        profile_base64 = ""
        if profile_path.exists():
            profile_bytes = profile_path.read_bytes()
            profile_base64 = base64.b64encode(profile_bytes).decode()

        st.markdown(
            f"""
            <div style='text-align:center; margin:6px 0 8px 0;'>
                <img src="data:image/jpeg;base64,{profile_base64}"
                    width='145'
                    style='border-radius:50%; border:2px solid #FEE700; margin-bottom:4px;'>
                <h2 style='margin:3px 0; color:#fff;'>Amr Amer</h2>
                <div style='font-size:14px; color:#999; margin-bottom:8px;'>
                    <span style='color:#FEE700;'>●</span> Badminton Enthusiast
                    <span style='color:#FEE700;'>●</span> AI Developer
                </div>
            </div>

            <div style='background: rgba(254,231,0,0.1);
                        padding:12px;
                        border-radius:10px;
                        border-left:4px solid #FEE700;
                        margin:6px 0 10px 0;'>
                <p style='font-style:italic; margin-bottom:5px;'>
                "I built this because I wanted to combine my two passions -
                badminton and technology."
                </p>
                <p style='text-align:right; font-weight:bold;'>- Amr</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        with st.expander("🏸 Badminton Fun Fact"):
            st.write(
                """
                Did you know?
                - Fastest smash ever recorded: **419 km/h**
                - Shuttlecocks have **16 goose feathers**
                """
            )

    st.markdown(logo_html, unsafe_allow_html=True)

    # -------------------------------
    # Load config + data
    # -------------------------------
    args = parse_args(sys.argv[1:])
    logger = get_logger(__name__)
    config = load_config(args.config)

    dashboard = StreamlitDashboard(config)

    json_dir = Path(args.json_dir)
    selected_paths = select_jsons(json_dir)

    if not selected_paths:
        st.error(f"No JSON files found in: {json_dir.resolve()}")
        st.info("Run your pipeline first or point --json_dir to the folder containing the JSON outputs.")
        st.stop()

    tracking_bundle = load_json_bundle(selected_paths, logger)
    tracking_data = dashboard.adapt_bundle(tracking_bundle)

    # -------------------------------
    # Navigation + routing
    # -------------------------------
    st.sidebar.markdown("## Navigation")
    mode = st.sidebar.radio(
        "Choose your analysis mode:",
        ["🔴 Match Replay", "🔵 Coach Report", "🖥️ Court Explorer"],
        index=0,
        help="Switch between real-time tracking, detailed analytics, and 3D court visualization.",
    )

    if mode == "🔴 Match Replay":
        match_replay.render(dashboard, tracking_data, config)
    elif mode == "🔵 Coach Report":
        coach_report.render(dashboard, tracking_data, config)
    else:
        court_explorer.render(config)

    # -------------------------------
    # Footer
    # -------------------------------
    st.sidebar.markdown(
        """
        <div style="
            text-align: center;
            margin-top: 18px;
            padding-top: 12px;
            border-top: 1px solid #333;
            color: #888;
            font-size: 12px;
            line-height: 1.6;
        ">
            <div>
                Made with ❤️ and 🏸
            </div>
            <div style="margin-top: 6px; font-size: 11px; color: #777;">
                © 2026 Amr Amer — Badminton-vision AI
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
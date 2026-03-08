# webapp/report.py
# ============================================================
# Badminton-Vision AI — Coach Assistant Report (PDF)
# ============================================================

from __future__ import annotations

from io import BytesIO
from typing import Dict, Any, List, Optional
import os

import numpy as np
from PIL import Image

# Reuse implemented KPIs/tables/insights from tabs (single source of truth)
from webapp.tabs import positioning as positioning_tab
from webapp.tabs import shot_profile as shot_profile_tab

from webapp.reports.helpers import (
    _patch_reportlab_md5_py38,
    build_styles,
    build_doc_template,
    build_toc,
    _draw_header_footer,
    _clean_insights,
    _recommendations_from_signals,
    _build_shot_points_for_report,
    _normalize_xy_sequence,
    _pct_from_counts,
    _attack_defense_counts,
    _pil_to_png_bytes,
    _create_heatmap_overlay,
    _mpl_group_bar,
    _mpl_shot_timeline,
    _mpl_positioning_court_map,
    _mpl_shot_map,
    _find_player_photo,
)

from webapp.reports.pages import (
    build_cover_page,
    build_contents_page,
    build_match_snapshot_page,
    build_executive_overview_page,
    build_positioning_pages,
    build_shot_profile_pages,
    build_attack_defense_page,
    build_recommendations_page,
    build_appendix_page,
)

from webapp.reports.placeholders import build_under_development_page


def generate_coach_report(tracking_data: Dict[str, Any], config: Dict[str, Any]) -> bytes:
    _patch_reportlab_md5_py38()

    buffer = BytesIO()
    styles = build_styles()
    doc = build_doc_template(buffer)

    # --------------------------------------------------------
    # Reuse tab assets (single source of truth)
    # --------------------------------------------------------
    pos_assets = positioning_tab.build_report_assets(None, tracking_data, config, include_figures=False) or {}
    shot_assets = shot_profile_tab.build_report_assets(None, tracking_data, config, include_figures=False) or {}

    # --------------------------------------------------------
    # Core config
    # --------------------------------------------------------
    w = int(config.get("streamlit_dashboard", {}).get("court_width", 385) or 385)
    h = int(config.get("streamlit_dashboard", {}).get("court_height", 840) or 840)
    fps = float(config.get("video_fps", config.get("fps", 30.0)) or 30.0)
    fps = fps if fps > 1e-6 else 30.0

    # --------------------------------------------------------
    # Read inputs
    # --------------------------------------------------------
    names = tracking_data.get("player_names", {1: "Player 1", 2: "Player 2"}) or {1: "Player 1", 2: "Player 2"}
    p1_name = str(names.get(1, "Player 1"))
    p2_name = str(names.get(2, "Player 2"))

    rally = tracking_data.get("rally_summary", {}) or {}
    dyn = tracking_data.get("player_dyn", {}) or {}
    p1_dyn = dyn.get(1, {}) or {}
    p2_dyn = dyn.get(2, {}) or {}

    ball_speed = tracking_data.get("ball_speed_smooth_kmh", []) or tracking_data.get("ball_speed_kmh", []) or []
    bs_arr = np.array(ball_speed, dtype=float) if ball_speed else np.array([], dtype=float)
    ball_peak = float(np.nanmax(bs_arr)) if bs_arr.size and np.isfinite(np.nanmax(bs_arr)) else float("nan")
    ball_avg = float(np.nanmean(bs_arr[np.isfinite(bs_arr)])) if bs_arr.size and np.any(np.isfinite(bs_arr)) else float("nan")

    p1_pos_raw = tracking_data.get("player1_positions", []) or []
    p2_pos_raw = tracking_data.get("player2_positions", []) or []

    p1_shots = tracking_data.get("player1_shots", {}) or {}
    p2_shots = tracking_data.get("player2_shots", {}) or {}

    shot_points = _build_shot_points_for_report(tracking_data, fps=fps)

    # Tab KPIs/insights/tables
    pos_k = (pos_assets or {}).get("kpis", {}) or {}
    shot_k = (shot_assets or {}).get("kpis", {}) or {}
    zone_rows = (pos_assets.get("tables") or {}).get("positioning_zone_table", []) or []

    pos_insights = _clean_insights((pos_assets.get("insights") or []), limit=10)
    shot_insights = _clean_insights((shot_assets.get("insights") or []), limit=10)

    # --------------------------------------------------------
    # Player photos
    # --------------------------------------------------------
    player_data = (config.get("streamlit_dashboard", {}) or {}).get("player_data", {}) or {}
    p1_photo = _find_player_photo(player_data, p1_name)
    p2_photo = _find_player_photo(player_data, p2_name)

    # --------------------------------------------------------
    # Court background for heatmap
    # --------------------------------------------------------
    court_img_path = (config.get("streamlit_dashboard", {}) or {}).get("court_image_path", "data/images/court-view.jpg")
    court_img = None
    try:
        if court_img_path and os.path.exists(court_img_path):
            court_img = Image.open(court_img_path).convert("RGBA").resize((w, h), Image.LANCZOS)
    except Exception:
        court_img = None

    heatmap_buf: Optional[BytesIO] = None
    if court_img is not None:
        p1_norm = _normalize_xy_sequence(p1_pos_raw)
        p2_norm = _normalize_xy_sequence(p2_pos_raw)
        hm = _create_heatmap_overlay(court_img, p1_norm, p2_norm)
        heatmap_buf = _pil_to_png_bytes(hm, max_w=1600, max_h=1600)

    # --------------------------------------------------------
    # Build figures
    # --------------------------------------------------------
    figs: Dict[str, Optional[BytesIO]] = {
        "pos_zone": None,
        "pos_width": None,
        "pos_court": None,
        "shot_share": None,
        "shot_timeline": None,
        "shot_map": None,
        "attack_defense": None,
    }

    # Positioning charts
    if zone_rows:
        labels = [str(r.get("Zone (player-relative)", "")) for r in zone_rows][:3]
        y1 = [float(r.get("P1 Occ %", 0.0) or 0.0) for r in zone_rows][:3]
        y2 = [float(r.get("P2 Occ %", 0.0) or 0.0) for r in zone_rows][:3]
        figs["pos_zone"] = _mpl_group_bar(
            "Zone Occupancy (Rear / Middle / Front)",
            labels,
            y1,
            y2,
            p1_name,
            p2_name,
            "Occupancy (%)",
        )

        p1_left = float(pos_k.get("p1_left_pct", 0.0) or 0.0)
        p2_left = float(pos_k.get("p2_left_pct", 0.0) or 0.0)
        figs["pos_width"] = _mpl_group_bar(
            "Width Usage (General Left / Right)",
            ["Left", "Right"],
            [p1_left, 100.0 - p1_left],
            [p2_left, 100.0 - p2_left],
            p1_name,
            p2_name,
            "Usage (%)",
        )

    figs["pos_court"] = _mpl_positioning_court_map(w, h, p1_pos_raw, p2_pos_raw, p1_name, p2_name)

    # Shot share
    p1_pct = _pct_from_counts(p1_shots)
    p2_pct = _pct_from_counts(p2_shots)
    all_types = list(dict.fromkeys(list(p1_pct.keys()) + list(p2_pct.keys())))
    if all_types:
        comb = [(t, float(p1_pct.get(t, 0.0)) + float(p2_pct.get(t, 0.0))) for t in all_types]
        comb.sort(key=lambda x: x[1], reverse=True)
        top = [t for t, _ in comb[:10]]
        y1 = [float(p1_pct.get(t, 0.0)) for t in top]
        y2 = [float(p2_pct.get(t, 0.0)) for t in top]
        figs["shot_share"] = _mpl_group_bar(
            "Shot Share by Type (Top 10)",
            top,
            y1,
            y2,
            p1_name,
            p2_name,
            "Share (%)",
        )

    figs["shot_timeline"] = _mpl_shot_timeline(shot_points, p1_name, p2_name)
    figs["shot_map"] = _mpl_shot_map(w, h, shot_points, p1_name, p2_name)

    # Attack vs Defense
    a1, d1 = _attack_defense_counts(p1_shots)
    a2, d2 = _attack_defense_counts(p2_shots)
    figs["attack_defense"] = _mpl_group_bar(
        "Attack vs Defense (Match)",
        ["Attack (Smash)", "Defense (Lift/Clear)"],
        [a1, d1],
        [a2, d2],
        p1_name,
        p2_name,
        "Count",
    )

    # --------------------------------------------------------
    # Positioning detailed table (6 zones)
    # --------------------------------------------------------
    detail_rows: List[Dict[str, Any]] = []
    try:
        p_dyn = tracking_data.get("player_dyn", {}) or {}
        p1_speed = (p_dyn.get(1, {}) or {}).get("speed_series_kmh", None)
        p2_speed = (p_dyn.get(2, {}) or {}).get("speed_series_kmh", None)
        p1_pos_norm = _normalize_xy_sequence(p1_pos_raw)
        p2_pos_norm = _normalize_xy_sequence(p2_pos_raw)

        p1_detail = positioning_tab._compute_detail_stats6(p1_pos_norm, p1_speed, w, h, fps, player="P1")  # type: ignore[attr-defined]
        p2_detail = positioning_tab._compute_detail_stats6(p2_pos_norm, p2_speed, w, h, fps, player="P2")  # type: ignore[attr-defined]
        detail_rows = positioning_tab._detail_table6(p1_detail, p2_detail)  # type: ignore[attr-defined]
    except Exception:
        detail_rows = []

    # --------------------------------------------------------
    # Executive findings
    # --------------------------------------------------------
    headline: List[str] = []
    if pos_insights:
        headline.append(pos_insights[0])
    if shot_insights:
        headline.append(shot_insights[0])

    headline.append(
        f"{p1_name}: main zone <b>{pos_k.get('p1_main_zone','—')}</b>, main shot <b>{shot_k.get('p1_main_shot','—')}</b> • "
        f"{p2_name}: main zone <b>{pos_k.get('p2_main_zone','—')}</b>, main shot <b>{shot_k.get('p2_main_shot','—')}</b>."
    )
    headline = headline[:5]

    recs = _recommendations_from_signals(p1_name, p2_name, pos_k, shot_k, p1_shots, p2_shots)

    # --------------------------------------------------------
    # Build PDF story
    # --------------------------------------------------------
    story: List[Any] = []

    build_cover_page(
        story=story,
        styles=styles,
        rally=rally,
        p1_name=p1_name,
        p2_name=p2_name,
        ball_peak=ball_peak,
    )

    toc = build_toc()
    build_contents_page(story=story, styles=styles, toc=toc)

    # ======================================================
    # Keep full page code in pages.py, but do not use yet
    # ======================================================

    # build_match_snapshot_page(
    #     story=story,
    #     styles=styles,
    #     p1_name=p1_name,
    #     p2_name=p2_name,
    #     p1_photo=p1_photo,
    #     p2_photo=p2_photo,
    #     p1_dyn=p1_dyn,
    #     p2_dyn=p2_dyn,
    #     ball_peak=ball_peak,
    #     ball_avg=ball_avg,
    #     rally=rally,
    #     heatmap_buf=heatmap_buf,
    # )
    build_under_development_page(story, styles, "Match Snapshot")

    # build_executive_overview_page(
    #     story=story,
    #     styles=styles,
    #     pos_k=pos_k,
    #     shot_k=shot_k,
    #     headline=headline,
    #     recs=recs,
    # )
    build_under_development_page(story, styles, "Executive Overview")

    # build_positioning_pages(
    #     story=story,
    #     styles=styles,
    #     pos_assets=pos_assets,
    #     figs=figs,
    #     pos_insights=pos_insights,
    #     zone_rows=zone_rows,
    #     detail_rows=detail_rows,
    # )
    build_under_development_page(story, styles, "Positioning & Movement")

    # build_shot_profile_pages(
    #     story=story,
    #     styles=styles,
    #     shot_assets=shot_assets,
    #     figs=figs,
    #     shot_insights=shot_insights,
    # )
    build_under_development_page(story, styles, "Shot Profile")

    # build_attack_defense_page(story=story, styles=styles, figs=figs)
    build_under_development_page(story, styles, "Attack vs Defense")

    # build_recommendations_page(story=story, styles=styles, recs=recs)
    build_under_development_page(story, styles, "Tactical Recommendations")

    # build_appendix_page(story=story, styles=styles)
    build_under_development_page(story, styles, "Appendix")

    doc.multiBuild(
        story,
        onFirstPage=_draw_header_footer,
        onLaterPages=_draw_header_footer,
    )
    buffer.seek(0)
    return buffer.getvalue()
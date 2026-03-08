# webapp/reports/pages.py

from __future__ import annotations

from typing import Dict, Any, List, Optional
from datetime import datetime
import numpy as np

from webapp.reports.helpers import (
    _safe_float,
    _fmt_num,
    _sanitize_reportlab_para,
    _load_image_as_png_bytes,
    _boxed,
    _kv_table,
    _bullet_list,
    _rl_img,
)


def build_cover_page(
    story,
    styles,
    rally: Dict[str, Any],
    p1_name: str,
    p2_name: str,
    ball_peak: float,
) -> None:
    from reportlab.platypus import Paragraph, Spacer, PageBreak

    story.append(Paragraph("Coach Assistant Report", styles["CoverSubtitle"]))
    story.append(Spacer(1, 18))

    cover_rows = [
        ("Match", f"{p1_name} vs {p2_name}"),
        ("Generated", datetime.now().strftime("%Y-%m-%d %H:%M")),
        ("Shots", str(int(rally.get("shots", 0) or 0))),
        ("Duration", f"{_safe_float(rally.get('duration_s', 0.0)):.1f} s"),
        ("Shot rate", f"{_safe_float(rally.get('shot_rate', 0.0)):.2f} / s"),
        ("Ball peak speed", f"{ball_peak:.1f} km/h" if np.isfinite(ball_peak) else "—"),
    ]
    story.append(_boxed("Match Summary", [_kv_table(cover_rows, styles)], styles))
    story.append(Spacer(1, 10))

    story.append(_boxed("Report Overview", [
        Paragraph(
            _sanitize_reportlab_para(
                "This report provides a tactical overview of the match based on automated player tracking and shot recognition. "
                "It analyzes movement patterns, court positioning, and shot selection to highlight tactical tendencies and key performance differences between players.<br/><br/>"
                "The report combines court visualizations, statistical summaries, and coaching insights to support match evaluation and training planning."
            ),
            styles["Small"],
        )
    ], styles))
    story.append(PageBreak())


def build_contents_page(story, styles, toc) -> None:
    from reportlab.platypus import Paragraph, PageBreak

    story.append(Paragraph("Contents", styles["H1"]))
    story.append(toc)
    story.append(PageBreak())


def build_match_snapshot_page(
    story,
    styles,
    p1_name: str,
    p2_name: str,
    p1_photo: Optional[str],
    p2_photo: Optional[str],
    p1_dyn: Dict[str, Any],
    p2_dyn: Dict[str, Any],
    ball_peak: float,
    ball_avg: float,
    rally: Dict[str, Any],
    heatmap_buf,
) -> None:
    from reportlab.platypus import Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.platypus import Image as RLImage
    from reportlab.lib.units import cm

    story.append(Paragraph("Match Snapshot", styles["H1"]))

    def _player_card(name: str, photo_path: Optional[str], dyn_info: Dict[str, Any]) -> Table:
        photo = _load_image_as_png_bytes(photo_path or "", max_w=320, max_h=320) if photo_path else None
        img = RLImage(photo, width=3.2 * cm, height=3.2 * cm) if photo else Paragraph("No photo", styles["Tiny"])

        max_spd = dyn_info.get("max_speed_kmh", None)
        avg_spd = dyn_info.get("avg_speed_kmh", None)
        dist_m = dyn_info.get("distance_m", None)
        hi_pct = dyn_info.get("high_speed_pct", None)
        hi_time = dyn_info.get("high_speed_time_s", None)
        thr = dyn_info.get("high_speed_thr_kmh", None)

        rows = [
            ("Max speed", f"{_fmt_num(max_spd, '{:.1f}')} km/h"),
            ("Avg speed", f"{_fmt_num(avg_spd, '{:.1f}')} km/h"),
            ("Distance", f"{_fmt_num(dist_m, '{:.0f}')} m"),
        ]
        if hi_pct is not None:
            thr_txt = f">{_fmt_num(thr, '{:.0f}')} km/h" if thr is not None else "High intensity"
            rows.append(("High intensity", f"{_fmt_num(hi_pct, '{:.1f}')}%  ({thr_txt})"))
            if hi_time is not None:
                rows.append(("High intensity time", f"{_fmt_num(hi_time, '{:.1f}')} s"))

        t = Table(
            [[img, Paragraph(f"<b>{_sanitize_reportlab_para(name)}</b>", styles["Normal"])]]
            + [[Paragraph(f"<b>{k}</b>", styles["Tiny"]), Paragraph(v, styles["Tiny"])] for k, v in rows],
            colWidths=[3.4 * cm, 5.0 * cm],
        )
        t.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("SPAN", (1, 0), (1, 0)),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        return t

    hm_panel = _rl_img(heatmap_buf, 8.2, 18.5, styles) if heatmap_buf else Paragraph("Heatmap unavailable.", styles["Small"])

    left_col = Table(
        [[_player_card(p1_name, p1_photo, p1_dyn)],
         [Spacer(1, 6)],
         [_player_card(p2_name, p2_photo, p2_dyn)]],
        colWidths=[8.4 * cm],
    )

    ball_rows = [
        ("Peak speed", f"{ball_peak:.1f} km/h" if np.isfinite(ball_peak) else "—"),
        ("Avg speed", f"{ball_avg:.1f} km/h" if np.isfinite(ball_avg) else "—"),
        ("Shots (total)", str(int(rally.get("shots", 0) or 0))),
    ]
    ball_box = _boxed("Ball Summary", [_kv_table(ball_rows, styles, colw=(6.2, 2.2))], styles)

    story.append(
        Table(
            [[
                Table([[left_col], [Spacer(1, 8)], [ball_box]], colWidths=[8.4 * cm]),
                _boxed(
                    "Court Heatmap (occupancy)",
                    [hm_panel, Spacer(1, 6), Paragraph("Heatmap from match positions (P1 amber, P2 blue).", styles["Small"])],
                    styles,
                ),
            ]],
            colWidths=[8.4 * cm, 8.4 * cm],
        )
    )
    story.append(PageBreak())


def build_executive_overview_page(
    story,
    styles,
    pos_k: Dict[str, Any],
    shot_k: Dict[str, Any],
    headline: List[str],
    recs: List[str],
) -> None:
    from reportlab.platypus import Paragraph, Spacer, PageBreak

    story.append(Paragraph("Executive Overview", styles["H1"]))

    kpi_rows = [
        ("P1 main zone", str(pos_k.get("p1_main_zone", "—"))),
        ("P2 main zone", str(pos_k.get("p2_main_zone", "—"))),
        ("P1 positioning balance", _fmt_num(pos_k.get("p1_balance", np.nan), "{:.1f}/100")),
        ("P2 positioning balance", _fmt_num(pos_k.get("p2_balance", np.nan), "{:.1f}/100")),
        ("P1 main shot", str(shot_k.get("p1_main_shot", "—"))),
        ("P2 main shot", str(shot_k.get("p2_main_shot", "—"))),
        ("P1 shot balance", _fmt_num(shot_k.get("p1_shot_balance", np.nan), "{:.1f}/100")),
        ("P2 shot balance", _fmt_num(shot_k.get("p2_shot_balance", np.nan), "{:.1f}/100")),
    ]
    story.append(_boxed("Key KPIs", [_kv_table(kpi_rows, styles)], styles))
    story.append(Spacer(1, 10))

    story.append(_boxed("Headline Findings", [_bullet_list(headline, styles)], styles))
    story.append(Spacer(1, 10))

    story.append(_boxed("Recommendations (next session)", [_bullet_list(recs[:6], styles)], styles))
    story.append(PageBreak())


def build_positioning_pages(
    story,
    styles,
    pos_assets: Dict[str, Any],
    figs: Dict[str, Any],
    pos_insights: List[str],
    zone_rows: List[Dict[str, Any]],
    detail_rows: List[Dict[str, Any]],
) -> None:
    from reportlab.platypus import Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib import colors
    from reportlab.lib.units import cm

    story.append(Paragraph("Positioning & Movement", styles["H1"]))
    if pos_assets.get("error"):
        story.append(_boxed(
            "Data availability",
            [Paragraph(_sanitize_reportlab_para(f"Positioning unavailable: {pos_assets.get('error')}"), styles["Normal"])],
            styles,
        ))
        story.append(PageBreak())
    else:
        story.append(_boxed("Overview Charts", [
            _rl_img(figs["pos_zone"], 16.8, 6.1, styles),
            Spacer(1, 8),
            _rl_img(figs["pos_width"], 16.8, 6.1, styles),
        ], styles))
        story.append(Spacer(1, 10))

        story.append(_boxed("Coach Notes (Positioning)", [
            Paragraph("Key observations:", styles["Small"]),
            _bullet_list(pos_insights[:6] if pos_insights else ["No positioning insights available from current data."], styles),
        ], styles))
        story.append(PageBreak())

        story.append(Paragraph("Positioning Map", styles["H1"]))
        story.append(_boxed("Court Map (Movement Paths)", [
            _rl_img(figs["pos_court"], 16.8, 20.2, styles),
            Spacer(1, 6),
            Paragraph("Markers: ■ start, ✖ end. Zones are shown using the same service-line geometry rules as the dashboard tabs.", styles["Small"]),
        ], styles))
        story.append(PageBreak())

        story.append(Paragraph("Positioning Tables", styles["H1"]))
        story.append(Paragraph("Zone Comparison (Rear / Middle / Front)", styles["H2"]))

        if zone_rows:
            cols = ["Zone", "P1 Occ %", "P2 Occ %", "Δ (P1-P2) %", "P1 Time (s)", "P2 Time (s)"]
            data = [cols]
            for r in zone_rows:
                data.append([
                    str(r.get("Zone (player-relative)", "")),
                    f"{_safe_float(r.get('P1 Occ %', 0.0)):.1f}",
                    f"{_safe_float(r.get('P2 Occ %', 0.0)):.1f}",
                    f"{_safe_float(r.get('Δ (P1-P2) %', 0.0)):+.1f}",
                    (_fmt_num(r.get("P1 Time (s)", np.nan), "{:.1f}") if r.get("P1 Time (s)") is not None else "—"),
                    (_fmt_num(r.get("P2 Time (s)", np.nan), "{:.1f}") if r.get("P2 Time (s)") is not None else "—"),
                ])
            t = Table(data, colWidths=[4.2 * cm, 2.1 * cm, 2.1 * cm, 2.3 * cm, 2.2 * cm, 2.2 * cm])
            t.setStyle(TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.35, colors.lightgrey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#efefef")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(t)
        else:
            story.append(Paragraph("No positioning zone table available.", styles["Small"]))
        story.append(PageBreak())

        story.append(Paragraph("Positioning Details Table", styles["H1"]))
        story.append(Paragraph("Rear/Middle/Front × Left/Right breakdown", styles["H2"]))

        if detail_rows:
            cols = ["Zone (detailed)", "P1 Occ %", "P2 Occ %", "Δ (P1-P2) %", "P1 Time (s)", "P2 Time (s)"]
            data = [cols]
            for r in detail_rows:
                data.append([
                    str(r.get("Zone (detailed)", "")),
                    f"{_safe_float(r.get('P1 Occ %', 0.0)):.1f}",
                    f"{_safe_float(r.get('P2 Occ %', 0.0)):.1f}",
                    f"{_safe_float(r.get('Δ (P1-P2) %', 0.0)):+.1f}",
                    (_fmt_num(r.get("P1 Time (s)", np.nan), "{:.1f}") if r.get("P1 Time (s)") is not None else "—"),
                    (_fmt_num(r.get("P2 Time (s)", np.nan), "{:.1f}") if r.get("P2 Time (s)") is not None else "—"),
                ])
            t2 = Table(data, colWidths=[5.2 * cm, 2.0 * cm, 2.0 * cm, 2.2 * cm, 2.2 * cm, 2.2 * cm])
            t2.setStyle(TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.35, colors.lightgrey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#efefef")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(t2)
        else:
            story.append(Paragraph("Detailed positioning table not available in current build.", styles["Small"]))
        story.append(PageBreak())


def build_shot_profile_pages(
    story,
    styles,
    shot_assets: Dict[str, Any],
    figs: Dict[str, Any],
    shot_insights: List[str],
) -> None:
    from reportlab.platypus import Paragraph, Spacer, PageBreak

    story.append(Paragraph("Shot Profile", styles["H1"]))
    if shot_assets.get("error"):
        story.append(_boxed(
            "Data availability",
            [Paragraph(_sanitize_reportlab_para(f"Shot profile unavailable: {shot_assets.get('error')}"), styles["Normal"])],
            styles,
        ))
        story.append(PageBreak())
    else:
        story.append(_boxed("Overview Charts", [
            _rl_img(figs["shot_share"], 16.8, 6.1, styles),
            Spacer(1, 8),
            _rl_img(figs["shot_timeline"], 16.8, 5.6, styles),
        ], styles))
        story.append(Spacer(1, 10))

        story.append(_boxed("Coach Notes (Shot Profile)", [
            Paragraph("Key observations:", styles["Small"]),
            _bullet_list(shot_insights[:6] if shot_insights else ["No shot profile insights available from current data."], styles),
        ], styles))
        story.append(PageBreak())

        story.append(Paragraph("Shot Map", styles["H1"]))
        story.append(_boxed("Shot Map (Match)", [
            _rl_img(figs["shot_map"], 16.8, 20.2, styles),
            Spacer(1, 6),
            Paragraph("Color = player, marker shape = player. Court geometry matches the dashboard tab rules.", styles["Small"]),
        ], styles))
        story.append(PageBreak())


def build_attack_defense_page(story, styles, figs: Dict[str, Any]) -> None:
    from reportlab.platypus import Paragraph, Spacer, PageBreak

    story.append(Paragraph("Attack vs Defense", styles["H1"]))
    story.append(_boxed("Tactical Tendency", [
        _rl_img(figs["attack_defense"], 16.8, 6.8, styles),
        Spacer(1, 6),
        Paragraph("Definition: Attack = Smash. Defense = Lift/Clear. Other shots are excluded from this split.", styles["Small"]),
    ], styles))
    story.append(PageBreak())


def build_recommendations_page(story, styles, recs: List[str]) -> None:
    from reportlab.platypus import Paragraph, Spacer, PageBreak

    story.append(Paragraph("Tactical Recommendations", styles["H1"]))
    story.append(_boxed("Action Plan (based on the analysis)", [
        Paragraph("These priorities are derived from the measured movement/shot tendencies in this match.", styles["Small"]),
        Spacer(1, 6),
        _bullet_list(recs, styles),
    ], styles))
    story.append(Spacer(1, 10))

    story.append(_boxed("Suggested additions to increase coaching depth", [
        _bullet_list([
            "Add landing zones (court bins) per shot to link shot choice → pressure → outcomes.",
            "Add rally segmentation + point outcome labels to connect patterns to scoring efficiency.",
            "Add serve/return phase tagging (first 3 shots) to highlight opening patterns.",
        ], styles)
    ], styles))
    story.append(PageBreak())


def build_appendix_page(story, styles) -> None:
    from reportlab.platypus import Paragraph

    story.append(Paragraph("Appendix", styles["H1"]))
    story.append(_boxed("Definitions", [
        _bullet_list([
            "Rear/Middle/Front zones are computed player-relative using the positioning tab logic.",
            "Width usage is general Left/Right over the match (not per-zone).",
            "Shot share and balances are computed from the shot profile tab logic.",
            "Court visuals are rendered for print using the same badminton service-line geometry rules as the dashboard tabs.",
        ], styles)
    ], styles))
# webapp/tabs/positioning.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import streamlit as st
import plotly.graph_objects as go

from constants import COURT_WIDTH, COURT_LENGTH, SHORT_SERVICE_LINE, LONG_SERVICE_LINE

# Optional: singles sidelines
try:
    from constants import SIDELINE_OFFSET
except Exception:
    SIDELINE_OFFSET = 0.46


# ============================================================
# Model
# - Court drawing shows only 3 zones per side (Rear/Middle/Front) with distinct colors
# - Comparison table compares ONLY Rear/Middle/Front (player-relative, comparable)
# - Left/Right is computed as GENERAL width usage (overall), NOT per zone
# - Detailed table (optional) provides Rear/Middle/Front × Left/Right breakdown
#
# Assumption:
# - Player 1 is bottom (near y=h)
# - Player 2 is top (near y=0)
# ============================================================

BANDS = ["Rear", "Middle", "Front"]
P1_COLOR = "#F28E2B"
P2_COLOR = "#6FA4D9"
SIDES = ["Left", "Right"]
ZONE6 = [f"{b} {s}" for b in BANDS for s in SIDES]  # detail table only


@dataclass
class BandStats3:
    counts: np.ndarray     # (3,)
    pct: np.ndarray        # (3,)
    avg_speed: np.ndarray  # (3,) NaN if missing
    time_s: np.ndarray     # (3,) NaN if missing


@dataclass
class DetailStats6:
    counts: np.ndarray     # (6,)
    pct: np.ndarray        # (6,)
    avg_speed: np.ndarray  # (6,)
    time_s: np.ndarray     # (6,)


# ============================================================
# Geometry helpers
# ============================================================

def _m_to_px_y(m: float, h: int) -> float:
    return (m / float(COURT_LENGTH)) * float(h)


def _m_to_px_x(m: float, w: int) -> float:
    return (m / float(COURT_WIDTH)) * float(w)


def _court_geometry_px(w: int, h: int) -> Dict[str, float]:
    net_y = h / 2.0
    long_px = _m_to_px_y(LONG_SERVICE_LINE, h)
    short_px = _m_to_px_y(SHORT_SERVICE_LINE, h)

    # From TOP baseline
    y_top_long = long_px
    y_top_short = short_px

    # From BOTTOM baseline
    y_bot_long = h - long_px
    y_bot_short = h - short_px

    x_singles_left = _m_to_px_x(SIDELINE_OFFSET, w)
    x_singles_right = w - x_singles_left

    return dict(
        net_y=net_y,
        y_top_long=y_top_long,
        y_top_short=y_top_short,
        y_bot_long=y_bot_long,
        y_bot_short=y_bot_short,
        x_center=w / 2.0,
        x_singles_left=x_singles_left,
        x_singles_right=x_singles_right,
    )


# ============================================================
# Player-relative band index + general left/right
# ============================================================

def _band_idx_player(y: int, h: int, player: str) -> int:
    """
    0=Rear, 1=Middle, 2=Front (player-relative)

    Clamp into player half so net-cross artifacts don't break stats.
    """
    net_y = h / 2.0
    long_px = _m_to_px_y(LONG_SERVICE_LINE, h)
    short_px = _m_to_px_y(SHORT_SERVICE_LINE, h)

    if player == "P1":  # bottom
        y_clamped = max(float(y), net_y)
        d = float(h) - y_clamped
    else:  # "P2" top
        y_clamped = min(float(y), net_y)
        d = y_clamped

    if d <= long_px:
        return 0
    if d <= short_px:
        return 1
    return 2


def _is_left(x: int, w: int) -> bool:
    return float(x) < (float(w) / 2.0)


# ============================================================
# Stats computation
# ============================================================

def _compute_band_stats3(
    positions: List[Tuple[int, int]],
    speed_series_kmh: Optional[List[Optional[float]]],
    w: int,
    h: int,
    fps: float,
    player: str,
) -> BandStats3:
    counts = np.zeros(3, dtype=float)
    sum_speed = np.zeros(3, dtype=float)
    n_speed = np.zeros(3, dtype=float)

    for i, (_x, y) in enumerate(positions):
        bi = _band_idx_player(int(y), h, player)
        counts[bi] += 1.0

        if speed_series_kmh and i < len(speed_series_kmh):
            v = speed_series_kmh[i]
            if v is not None and np.isfinite(float(v)):
                sum_speed[bi] += float(v)
                n_speed[bi] += 1.0

    total = float(counts.sum())
    pct = (counts / total * 100.0) if total > 0 else counts.copy()

    avg_speed = np.full(3, np.nan, dtype=float)
    mask = n_speed > 0
    avg_speed[mask] = sum_speed[mask] / np.maximum(n_speed[mask], 1.0)

    time_s = np.full(3, np.nan, dtype=float)
    if fps and fps > 1e-6:
        time_s = counts / float(fps)

    return BandStats3(counts=counts, pct=pct, avg_speed=avg_speed, time_s=time_s)


def _compute_general_lr(
    positions: List[Tuple[int, int]],
    w: int,
) -> Tuple[float, float]:
    """
    General left/right usage across the full match for that player (percent).
    """
    if not positions:
        return 0.0, 0.0
    left = 0.0
    right = 0.0
    for x, _y in positions:
        if _is_left(int(x), w):
            left += 1.0
        else:
            right += 1.0
    total = left + right
    if total <= 0:
        return 0.0, 0.0
    return (left / total * 100.0), (right / total * 100.0)


def _compute_detail_stats6(
    positions: List[Tuple[int, int]],
    speed_series_kmh: Optional[List[Optional[float]]],
    w: int,
    h: int,
    fps: float,
    player: str,
) -> "DetailStats6":
    """
    Detailed breakdown: (Rear/Middle/Front) × (Left/Right) => 6 bins.
    Used only in the "Detailed" table below.
    """
    counts = np.zeros(6, dtype=float)
    sum_speed = np.zeros(6, dtype=float)
    n_speed = np.zeros(6, dtype=float)

    for i, (x, y) in enumerate(positions):
        bi = _band_idx_player(int(y), h, player)        # 0..2
        si = 0 if _is_left(int(x), w) else 1            # 0/1
        zi = bi * 2 + si                                # 0..5
        counts[zi] += 1.0

        if speed_series_kmh and i < len(speed_series_kmh):
            v = speed_series_kmh[i]
            if v is not None and np.isfinite(float(v)):
                sum_speed[zi] += float(v)
                n_speed[zi] += 1.0

    total = float(counts.sum())
    pct = (counts / total * 100.0) if total > 0 else counts.copy()

    avg_speed = np.full(6, np.nan, dtype=float)
    mask = n_speed > 0
    avg_speed[mask] = sum_speed[mask] / np.maximum(n_speed[mask], 1.0)

    time_s = np.full(6, np.nan, dtype=float)
    if fps and fps > 1e-6:
        time_s = counts / float(fps)

    return DetailStats6(counts=counts, pct=pct, avg_speed=avg_speed, time_s=time_s)


def _balance_score_entropy(pct: np.ndarray, eps: float = 1e-12) -> float:
    """
    Entropy-based balance score (0..100).
    100 = perfectly even across zones, 0 = concentrated in one zone.
    """
    if pct is None or len(pct) == 0:
        return 0.0

    p = np.array(pct, dtype=float)
    p = p[np.isfinite(p)]
    if p.size == 0:
        return 0.0

    p = np.clip(p / 100.0, 0.0, 1.0)
    s = float(p.sum())
    if s <= eps:
        return 0.0
    p = p / s

    p_nz = p[p > 0.0]
    H = -float(np.sum(p_nz * np.log(p_nz)))

    Hmax = float(np.log(len(p)))  # ln(3) for 3 zones
    if Hmax <= eps:
        return 0.0

    return float(100.0 * (H / Hmax))


# ============================================================
# Drawing (3-zones court)
# ============================================================

def _add_text_badge(
    fig: go.Figure,
    x: float,
    y: float,
    text: str,
    font_size: int = 13,
    font_color: str = "white",
    bg: str = "rgba(0,0,0,0.55)",
    border: str = "rgba(255,255,255,0.35)",
    pad_px: int = 6,
) -> None:
    fig.add_annotation(
        x=x, y=y,
        text=f"<b>{text}</b>",
        showarrow=False,
        xanchor="center",
        yanchor="middle",
        align="center",
        font=dict(size=font_size, color=font_color),
        bgcolor=bg,
        bordercolor=border,
        borderwidth=1,
        borderpad=pad_px,
        opacity=1.0,
    )


def _add_line(fig: go.Figure, x0, y0, x1, y1, width=1, dash="solid", opacity=1.0) -> None:
    fig.add_shape(
        type="line",
        x0=x0, y0=y0, x1=x1, y1=y1,
        line=dict(width=width, dash=dash),
        opacity=opacity,
    )


def _add_rect(fig: go.Figure, x0, y0, x1, y1, fill, width=1, dash="solid", opacity=1.0, layer="below") -> None:
    yy0, yy1 = (y0, y1) if y0 <= y1 else (y1, y0)
    xx0, xx1 = (x0, x1) if x0 <= x1 else (x1, x0)
    fig.add_shape(
        type="rect",
        x0=xx0, y0=yy0, x1=xx1, y1=yy1,
        line=dict(width=width, dash=dash),
        fillcolor=fill,
        opacity=opacity,
        layer=layer,
    )


def _draw_court_3zones(fig: go.Figure, w: int, h: int, show_zones: bool = True) -> Dict[str, float]:
    g = _court_geometry_px(w, h)

    _add_rect(fig, 0, 0, w, h, fill="rgba(30, 120, 70, 0.10)", width=0, opacity=0.9, layer="below")
    _add_rect(fig, 0, 0, w, h, fill="rgba(0,0,0,0)", width=2, dash="solid", opacity=0.9, layer="above")

    # Net: bold dotted split, with label
    gap = max(18, 0.03 * w)
    mid = g["x_center"]
    y = g["net_y"]
    _add_line(fig, 0, y, mid - gap / 2, y, width=3, dash="dot", opacity=0.7)
    _add_line(fig, mid + gap / 2, y, w, y, width=3, dash="dot", opacity=0.7)

    _add_text_badge(
        fig,
        x=mid,
        y=y,
        text="Net Line",
        font_size=13,
        bg="rgba(0,0,0,0.50)",
        border="rgba(155,155,155,0.50)",
        pad_px=5,
    )

    _add_line(fig, 0, g["y_top_short"], w, g["y_top_short"], width=2, opacity=0.80)
    _add_line(fig, 0, g["y_bot_short"], w, g["y_bot_short"], width=2, opacity=0.80)
    _add_line(fig, 0, g["y_top_long"], w, g["y_top_long"], width=2, opacity=0.80)
    _add_line(fig, 0, g["y_bot_long"], w, g["y_bot_long"], width=2, opacity=0.80)

    _add_line(fig, g["x_center"], g["y_top_long"], g["x_center"], g["y_top_short"], width=2, opacity=0.80)
    _add_line(fig, g["x_center"], g["y_bot_short"], g["x_center"], g["y_bot_long"], width=2, opacity=0.80)

    if 0 < g["x_singles_left"] < w / 2:
        _add_line(fig, g["x_singles_left"], 0, g["x_singles_left"], h, width=1, opacity=0.50)
        _add_line(fig, g["x_singles_right"], 0, g["x_singles_right"], h, width=1, opacity=0.50)

    if not show_zones:
        return g

    band_fill = {
        "Rear": "rgba(70, 140, 255, 0.18)",
        "Middle": "rgba(255, 255, 255, 0.12)",
        "Front": "rgba(255, 170, 70, 0.18)",
    }

    # Top half (P2)
    _add_rect(fig, 0, 0, w, g["y_top_long"], fill=band_fill["Rear"], width=1, dash="dot", opacity=0.9)
    _add_rect(fig, 0, g["y_top_long"], w, g["y_top_short"], fill=band_fill["Middle"], width=1, dash="dot", opacity=0.9)
    _add_rect(fig, 0, g["y_top_short"], w, g["net_y"], fill=band_fill["Front"], width=1, dash="dot", opacity=0.9)

    # Bottom half (P1)
    _add_rect(fig, 0, g["y_bot_long"], w, h, fill=band_fill["Rear"], width=1, dash="dot", opacity=0.9)
    _add_rect(fig, 0, g["y_bot_short"], w, g["y_bot_long"], fill=band_fill["Middle"], width=1, dash="dot", opacity=0.9)
    _add_rect(fig, 0, g["net_y"], w, g["y_bot_short"], fill=band_fill["Front"], width=1, dash="dot", opacity=0.9)

    return g


def _band_centers(fig_g: Dict[str, float], w: int, h: int) -> Dict[Tuple[str, str], Tuple[float, float]]:
    g = fig_g
    cx = w / 2.0

    cy_p2 = {
        "Rear": (0 + g["y_top_long"]) / 2.0,
        "Middle": (g["y_top_long"] + g["y_top_short"]) / 2.0,
        "Front": (g["y_top_short"] + g["net_y"]) / 2.0,
    }
    cy_p1 = {
        "Rear": (g["y_bot_long"] + h) / 2.0,
        "Middle": (g["y_bot_short"] + g["y_bot_long"]) / 2.0,
        "Front": (g["net_y"] + g["y_bot_short"]) / 2.0,
    }

    out: Dict[Tuple[str, str], Tuple[float, float]] = {}
    for b in BANDS:
        out[("P2", b)] = (cx, cy_p2[b])
        out[("P1", b)] = (cx, cy_p1[b])
    return out


def _build_court_figure(
    title: str,
    w: int,
    h: int,
    p1_positions: List[Tuple[int, int]],
    p2_positions: List[Tuple[int, int]],
    p1: BandStats3,
    p2: BandStats3,
    show: str = "Both",
    max_points: int = 900,
) -> go.Figure:
    fig = go.Figure()
    g = _draw_court_3zones(fig, w, h, show_zones=True)
    centers = _band_centers(g, w, h)

    def _fmt(t, v):
        t_str = f"{t:.1f}s" if np.isfinite(t) else "—"
        v_str = f"{v:.1f} km/h" if np.isfinite(v) else "—"
        return t_str, v_str

    # Labels (top+bottom) with consistent hover
    for i, band in enumerate(BANDS):
        t1, v1 = _fmt(p1.time_s[i], p1.avg_speed[i])
        t2, v2 = _fmt(p2.time_s[i], p2.avg_speed[i])

        tip = (
            f"<b>{band} (player-relative)</b><br>"
            f"<span style='color:{P1_COLOR}'><b>Player 1</b></span>: {p1.pct[i]:.1f}% • {t1} • {v1}<br>"
            f"<span style='color:{P2_COLOR}'><b>Player 2</b></span>: {p2.pct[i]:.1f}% • {t2} • {v2}"
            "<extra></extra>"
        )

        cx, cy = centers[("P2", band)]
        fig.add_trace(go.Scatter(
            x=[cx], y=[cy],
            mode="text",
            text=[f"{band.upper()}<br><span style='font-size:12px'>P2 {p2.pct[i]:.1f}%</span>"],
            textposition="middle center",
            hovertemplate=tip,
            showlegend=False,
        ))
        cx, cy = centers[("P1", band)]
        fig.add_trace(go.Scatter(
            x=[cx], y=[cy],
            mode="text",
            text=[f"{band.upper()}<br><span style='font-size:12px'>P1 {p1.pct[i]:.1f}%</span>"],
            textposition="middle center",
            hovertemplate=tip,
            showlegend=False,
        ))

    def _downsample(points: List[Tuple[int, int]], n: int) -> List[Tuple[int, int]]:
        if not points:
            return []
        if len(points) <= n:
            return points
        idx = np.linspace(0, len(points) - 1, n).astype(int)
        return [points[j] for j in idx]

    p1d = _downsample(p1_positions, max_points)
    p2d = _downsample(p2_positions, max_points)

    if show in ("Both", "Player 1"):
        fig.add_trace(go.Scatter(
            x=[x for x, _ in p1d],
            y=[y for _, y in p1d],
            mode="lines+markers",
            name="Player 1 Path",
            line=dict(width=2, color=P1_COLOR),
            marker=dict(size=4, color=P1_COLOR),
            hovertemplate="P1<br>x=%{x:.0f}, y=%{y:.0f}<extra></extra>",
        ))
    if show in ("Both", "Player 2"):
        fig.add_trace(go.Scatter(
            x=[x for x, _ in p2d],
            y=[y for _, y in p2d],
            mode="lines+markers",
            name="Player 2 Path",
            line=dict(width=2, color=P2_COLOR),
            marker=dict(size=4, color=P2_COLOR),
            hovertemplate="P2<br>x=%{x:.0f}, y=%{y:.0f}<extra></extra>",
        ))

    # Zone legend (hidden points)
    fig.add_trace(go.Scatter(x=[-9999], y=[-9999], mode="markers",
        marker=dict(size=10, color="rgba(70, 140, 255, 0.55)"),
        name="Rear zone", showlegend=True, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(x=[-9999], y=[-9999], mode="markers",
        marker=dict(size=10, color="rgba(255, 255, 255, 0.35)"),
        name="Middle zone", showlegend=True, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(x=[-9999], y=[-9999], mode="markers",
        marker=dict(size=10, color="rgba(255, 170, 70, 0.55)"),
        name="Front zone", showlegend=True, hoverinfo="skip",
    ))

    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", x=0.02, xanchor="left"),
        height=800,
        margin=dict(l=16, r=16, t=82, b=35),  
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        legend=dict(orientation="h", yanchor="top", y=1.04, xanchor="left", x=0.0),
    )
    fig.update_xaxes(range=[0, w], showgrid=False, zeroline=False, visible=False)
    fig.update_yaxes(range=[h, 0], showgrid=False, zeroline=False, visible=False)


    return fig


# ============================================================
# Charts
# ============================================================

def _build_band_bar(p1: BandStats3, p2: BandStats3) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(name="Player 1", x=BANDS, y=p1.pct, marker_color=P1_COLOR)
    fig.add_bar(name="Player 2", x=BANDS, y=p2.pct, marker_color=P2_COLOR)
    fig.update_layout(
        barmode="group",
        height=260,
        margin=dict(l=16, r=16, t=70, b=18),  
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        title=dict(text="<b>Zone Occupancy (Rear / Middle / Front)</b>", x=0.02, xanchor="left"),
        yaxis_title="Occupancy (%)",
        legend=dict(orientation="h", yanchor="top", y=1.20, xanchor="left", x=0.0),
    )
    fig.update_yaxes(rangemode="tozero")
    return fig


def _build_lr_bar(p1_lr: Tuple[float, float], p2_lr: Tuple[float, float]) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(name="Player 1", x=["Left", "Right"], y=[p1_lr[0], p1_lr[1]], marker_color=P1_COLOR)
    fig.add_bar(name="Player 2", x=["Left", "Right"], y=[p2_lr[0], p2_lr[1]], marker_color=P2_COLOR)
    fig.update_layout(
        barmode="group",
        height=260,
        margin=dict(l=16, r=16, t=70, b=18),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        title=dict(text="<b>Width Usage (General Left / Right)</b>", x=0.02, xanchor="left"),
        yaxis_title="Usage (%)",
        legend=dict(orientation="h", yanchor="top", y=1.20, xanchor="left", x=0.0),
    )
    fig.update_yaxes(rangemode="tozero")
    return fig


# ============================================================
# Tables + coach insights (unchanged logic)
# ============================================================

def _band_comparison_table(p1: BandStats3, p2: BandStats3) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for i, band in enumerate(BANDS):
        d = float(p1.pct[i] - p2.pct[i])
        rows.append(
            {
                "Zone (player-relative)": band,
                "P1 Occ %": float(p1.pct[i]),
                "P2 Occ %": float(p2.pct[i]),
                "Δ (P1-P2) %": d,
                "P1 Time (s)": float(p1.time_s[i]) if np.isfinite(p1.time_s[i]) else None,
                "P2 Time (s)": float(p2.time_s[i]) if np.isfinite(p2.time_s[i]) else None,
                "P1 Avg Speed (km/h)": float(p1.avg_speed[i]) if np.isfinite(p1.avg_speed[i]) else None,
                "P2 Avg Speed (km/h)": float(p2.avg_speed[i]) if np.isfinite(p2.avg_speed[i]) else None,
            }
        )
    return rows


def _coach_insights_smart(
    p1: BandStats3,
    p2: BandStats3,
    p1_lr: Tuple[float, float],
    p2_lr: Tuple[float, float],
) -> List[str]:
    # --- keep your original implementation exactly ---
    # (Paste your existing _coach_insights_smart here unchanged)
    # For brevity in this snippet, I’m keeping it as-is; do NOT change the logic.
    # -----------------------------------------------
    notes: List[str] = []

    def _fmt_pct(x: float) -> str:
        return f"{x:.1f}%" if np.isfinite(x) else "—"

    def _fmt_speed(x: float) -> str:
        return f"{x:.1f} km/h" if np.isfinite(x) else "—"

    def _fmt_time(x: float) -> str:
        return f"{x:.1f}s" if np.isfinite(x) else "—"

    def _p1_html(s: str) -> str:
        return f"<span style='color:#ffa500'><b>{s}</b></span>"

    def _p2_html(s: str) -> str:
        return f"<span style='color:#3aa0ff'><b>{s}</b></span>"

    def _best(v1: float, v2: float, eps: float = 1e-9) -> int:
        if not (np.isfinite(v1) and np.isfinite(v2)):
            return 0
        if abs(v1 - v2) <= eps:
            return 0
        return 1 if v1 > v2 else 2

    def _pname(best_id: int, player_id: int) -> str:
        if best_id == player_id:
            return _p1_html("Player_1") if player_id == 1 else _p2_html("Player_2")
        return "Player_1" if player_id == 1 else "Player_2"

    def _stat(best_id: int, player_id: int, txt: str) -> str:
        if best_id == player_id:
            return _p1_html(txt) if player_id == 1 else _p2_html(txt)
        return txt

    def _safe_nanargmax(a: np.ndarray) -> int:
        if a is None or len(a) == 0:
            return 0
        if not np.any(np.isfinite(a)):
            return 0
        return int(np.nanargmax(a))

    def _overall_avg_speed(bs: BandStats3) -> float:
        v = np.array(bs.avg_speed, dtype=float)
        c = np.array(bs.counts, dtype=float)
        m = np.isfinite(v) & np.isfinite(c) & (c > 0)
        if not np.any(m):
            return float("nan")
        return float(np.sum(v[m] * c[m]) / np.sum(c[m]))

    def _lr_bias(lr: Tuple[float, float]) -> float:
        return float(lr[0] - lr[1])

    def _pref_side(bias: float) -> str:
        if bias >= 10.0:
            return "left"
        if bias <= -10.0:
            return "right"
        return "balanced"

    def _zone_txt(band: str) -> str:
        return {"Front": "front-court", "Middle": "mid-court", "Rear": "back-court"}.get(band, band.lower())

    i1 = _safe_nanargmax(p1.pct)
    i2 = _safe_nanargmax(p2.pct)

    z1 = _zone_txt(BANDS[i1])
    z2 = _zone_txt(BANDS[i2])

    if i1 == i2:
        p1_p = float(p1.pct[i1]) if np.isfinite(p1.pct[i1]) else np.nan
        p2_p = float(p2.pct[i2]) if np.isfinite(p2.pct[i2]) else np.nan
        best_share = _best(p1_p, p2_p)

        notes.append(
            f"- Both players were most active in the {z1}: "
            f"{_pname(best_share, best_share) if best_share in (1,2) else '—'} had the higher share "
            f"({_pname(best_share, 1)} {_stat(best_share, 1, _fmt_pct(p1_p))} vs "
            f"{_pname(best_share, 2)} {_stat(best_share, 2, _fmt_pct(p2_p))})."
        )
    else:
        notes.append(
            f"- Player_1 dominant/main zone was the {z1} ({_fmt_pct(p1.pct[i1])}, {_fmt_time(p1.time_s[i1])}), "
            f"while Player_2 dominant/main zone was the {z2} ({_fmt_pct(p2.pct[i2])}, {_fmt_time(p2.time_s[i2])})."
        )

    diffs_time = (p1.time_s - p2.time_s) if (p1.time_s is not None and p2.time_s is not None) else np.zeros(3)
    i_sep = _safe_nanargmax(np.abs(diffs_time))
    t1 = float(p1.time_s[i_sep]) if np.isfinite(p1.time_s[i_sep]) else np.nan
    t2 = float(p2.time_s[i_sep]) if np.isfinite(p2.time_s[i_sep]) else np.nan

    best_time = _best(t1, t2)
    z_sep = _zone_txt(BANDS[i_sep])

    notes.append(
        f"- The clearest separation was in the {z_sep}: {_pname(best_time, best_time) if best_time in (1,2) else '—'} spent more time there than "
        f"{_pname(best_time, 2 if best_time == 1 else 1) if best_time in (1,2) else '—'} "
        f"({_pname(best_time, 1)} {_stat(best_time, 1, _fmt_time(t1))} vs {_pname(best_time, 2)} {_stat(best_time, 2, _fmt_time(t2))})."
    )

    p1_v = _overall_avg_speed(p1)
    p2_v = _overall_avg_speed(p2)

    if np.isfinite(p1_v) and np.isfinite(p2_v) and abs(p1_v - p2_v) > 1e-6:
        best_spd = _best(p1_v, p2_v)
        notes.append(
            f"- Overall, {_pname(best_spd, best_spd) if best_spd in (1,2) else '—'} moved faster on average "
            f"({_pname(best_spd, 1)} {_stat(best_spd, 1, _fmt_speed(p1_v))} vs "
            f"{_pname(best_spd, 2)} {_stat(best_spd, 2, _fmt_speed(p2_v))}, "
            f"Δ {abs(p1_v - p2_v):.1f} km/h)."
        )
    else:
        notes.append("- Overall movement speed was similar or not available from the current data.")

    p1_speed_vec = np.array(p1.avg_speed, dtype=float)
    p2_speed_vec = np.array(p2.avg_speed, dtype=float)

    i1_fast = _safe_nanargmax(np.where(np.isfinite(p1_speed_vec), p1_speed_vec, -np.inf))
    i2_fast = _safe_nanargmax(np.where(np.isfinite(p2_speed_vec), p2_speed_vec, -np.inf))

    z1_fast = _zone_txt(BANDS[i1_fast])
    z2_fast = _zone_txt(BANDS[i2_fast])

    v1_fast = float(p1.avg_speed[i1_fast]) if np.isfinite(p1.avg_speed[i1_fast]) else np.nan
    v2_fast = float(p2.avg_speed[i2_fast]) if np.isfinite(p2.avg_speed[i2_fast]) else np.nan

    if i1_fast == i2_fast:
        best_zone_spd = _best(v1_fast, v2_fast)
        notes.append(
            f"- Both players reached their highest average speed in the {z1_fast}, but {_pname(best_zone_spd, best_zone_spd) if best_zone_spd in (1,2) else '—'} was faster there "
            f"({_pname(best_zone_spd, 1)} {_stat(best_zone_spd, 1, _fmt_speed(v1_fast))} vs "
            f"{_pname(best_zone_spd, 2)} {_stat(best_zone_spd, 2, _fmt_speed(v2_fast))})."
        )
    else:
        notes.append(
            f"- Player_1 was fastest in the {z1_fast} ({_fmt_speed(v1_fast)}), "
            f"whereas Player_2 was fastest in the {z2_fast} ({_fmt_speed(v2_fast)})."
        )

    b1 = _lr_bias(p1_lr)
    b2 = _lr_bias(p2_lr)
    s1 = _pref_side(b1)
    s2 = _pref_side(b2)

    best_bias = 0
    if np.isfinite(b1) and np.isfinite(b2):
        if abs(b1) > abs(b2) + 1e-9:
            best_bias = 1
        elif abs(b2) > abs(b1) + 1e-9:
            best_bias = 2

    p1L = _fmt_pct(p1_lr[0])
    p1R = _fmt_pct(p1_lr[1])
    p2L = _fmt_pct(p2_lr[0])
    p2R = _fmt_pct(p2_lr[1])

    if best_bias == 1:
        p1L, p1R = _p1_html(p1L), _p1_html(p1R)
    elif best_bias == 2:
        p2L, p2R = _p2_html(p2L), _p2_html(p2R)

    if s1 == s2 and s1 != "balanced":
        if best_bias:
            notes.append(
                f"- Both players leaned to the {s1} side, but {_pname(best_bias, best_bias)} showed a stronger lean "
                f"({('Player_1' if best_bias != 1 else _p1_html('Player_1'))} Left {p1L} vs Right {p1R} | "
                f"{('Player_2' if best_bias != 2 else _p2_html('Player_2'))} Left {p2L} vs Right {p2R})."
            )
        else:
            notes.append(
                f"- Both players leaned to the {s1} side "
                f"(Player_1 Left {p1L} vs Right {p1R} | "
                f"Player_2 Left {p2L} vs Right {p2R})."
            )
    elif s1 == "balanced" and s2 == "balanced":
        notes.append(
            f"- Both players used left and right fairly evenly "
            f"(Player_1 {p1L}–{p1R} | Player_2 {p2L}–{p2R})."
        )
    else:
        notes.append(
            f"- Player_1 was {s1}-leaning (Left {p1L} vs Right {p1R}), "
            f"while Player_2 was {s2}-leaning (Left {p2L} vs Right {p2R})."
        )

    parts: List[str] = []
    if i1 == i2:
        parts.append(f"both players main activity was in the {z1}")
    else:
        parts.append(f"Player_1 focused on the {z1}, while Player_2 focused on the {z2}")
    parts.append(f"the biggest share gap appeared in the {z_sep}")
    if np.isfinite(p1_v) and np.isfinite(p2_v) and abs(p1_v - p2_v) > 1e-6:
        parts.append(f"{('Player_1' if p1_v > p2_v else 'Player_2')} led overall movement speed")
    if not (s1 == "balanced" and s2 == "balanced"):
        if s1 == s2 and s1 != "balanced":
            parts.append("both leaned the same side, but one lean was stronger")
        else:
            parts.append(f"side usage differed (P1 {s1} vs P2 {s2})")
    notes.append(f"- Summary: " + "; ".join(parts) + ".")
    return notes


def _detail_table6(p1: DetailStats6, p2: DetailStats6) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for i, zn in enumerate(ZONE6):
        rows.append(
            {
                "Zone (detailed)": zn,
                "P1 Occ %": float(p1.pct[i]),
                "P2 Occ %": float(p2.pct[i]),
                "Δ (P1-P2) %": float(p1.pct[i] - p2.pct[i]),
                "P1 Time (s)": float(p1.time_s[i]) if np.isfinite(p1.time_s[i]) else None,
                "P2 Time (s)": float(p2.time_s[i]) if np.isfinite(p2.time_s[i]) else None,
                "P1 Avg Speed (km/h)": float(p1.avg_speed[i]) if np.isfinite(p1.avg_speed[i]) else None,
                "P2 Avg Speed (km/h)": float(p2.avg_speed[i]) if np.isfinite(p2.avg_speed[i]) else None,
            }
        )
    return rows


#========================================
# Report Reusable Tabs
#========================================

def build_report_assets(
    dashboard,
    tracking_data: Dict[str, Any],
    config: Dict[str, Any],
    include_figures: bool = True
) -> Dict[str, Any]:
    p1_pos = tracking_data.get("player1_positions", []) or []
    p2_pos = tracking_data.get("player2_positions", []) or []

    if not p1_pos or not p2_pos:
        return {"error": "No tracking data available for positioning.", "figures": {}, "tables": {}, "kpis": {}, "insights": []}

    w = int(config.get("streamlit_dashboard", {}).get("court_width", 385) or 385)
    h = int(config.get("streamlit_dashboard", {}).get("court_height", 840) or 840)
    fps = float(config.get("video_fps", config.get("fps", 30.0)) or 30.0)
    fps = fps if fps > 1e-6 else 30.0

    p_dyn = tracking_data.get("player_dyn", {}) or {}
    p1_speed = (p_dyn.get(1, {}) or {}).get("speed_series_kmh", None)
    p2_speed = (p_dyn.get(2, {}) or {}).get("speed_series_kmh", None)

    # --- Stats (safe / non-plotly)
    p1_band = _compute_band_stats3(p1_pos, p1_speed, w, h, fps, player="P1")
    p2_band = _compute_band_stats3(p2_pos, p2_speed, w, h, fps, player="P2")
    p1_lr = _compute_general_lr(p1_pos, w)
    p2_lr = _compute_general_lr(p2_pos, w)

    table_primary = _band_comparison_table(p1_band, p2_band)
    insights = _coach_insights_smart(p1_band, p2_band, p1_lr, p2_lr)

    # --- KPIs (safe / non-plotly)
    kpis = {
        "p1_main_zone": BANDS[int(np.nanargmax(p1_band.pct))] if np.any(np.isfinite(p1_band.pct)) else "—",
        "p2_main_zone": BANDS[int(np.nanargmax(p2_band.pct))] if np.any(np.isfinite(p2_band.pct)) else "—",
        "p1_balance": float(_balance_score_entropy(p1_band.pct)),
        "p2_balance": float(_balance_score_entropy(p2_band.pct)),
        "p1_left_pct": float(p1_lr[0]),
        "p2_left_pct": float(p2_lr[0]),
    }

    figures: Dict[str, Any] = {}

    # --- Figures (Plotly) are OPTIONAL
    if include_figures:
        try:
            fig_court = _build_court_figure(
                title="Court Map (3 Zones + Movement Path)",
                w=w, h=h,
                p1_positions=p1_pos, p2_positions=p2_pos,
                p1=p1_band, p2=p2_band,
                show="Both",
            )
            fig_band = _build_band_bar(p1_band, p2_band)
            fig_lr = _build_lr_bar(p1_lr, p2_lr)

            figures = {
                "positioning_court": fig_court,
                "positioning_zone_bar": fig_band,
                "positioning_width_bar": fig_lr,
            }
        except Exception as e:
            # Do NOT fail the whole report if Plotly/pandas are broken
            figures = {}
            # keep an error hint for debugging (optional)
            # You can remove this line if you don't want it exposed anywhere.
            return {
                "error": f"Positioning figures unavailable: {type(e).__name__}: {e}",
                "figures": {},
                "tables": {"positioning_zone_table": table_primary},
                "kpis": kpis,
                "insights": insights,
            }

    return {
        "figures": figures,
        "tables": {
            "positioning_zone_table": table_primary,
        },
        "kpis": kpis,
        "insights": insights,
    }


# ============================================================
# MAIN TAB
# ============================================================

def render(dashboard, tracking_data: Dict[str, Any], config: Dict[str, Any]) -> None:
    st.markdown("## Positioning & Movement")

    p1_pos = tracking_data.get("player1_positions", []) or []
    p2_pos = tracking_data.get("player2_positions", []) or []
    if not p1_pos or not p2_pos:
        st.warning("No tracking data available for positioning.")
        return

    w = int(config["streamlit_dashboard"].get("court_width", 385))
    h = int(config["streamlit_dashboard"].get("court_height", 840))

    fps = float(config.get("video_fps", config.get("fps", 30.0)) or 30.0)
    fps = fps if fps > 1e-6 else 30.0

    p_dyn = tracking_data.get("player_dyn", {}) or {}
    p1_speed = (p_dyn.get(1, {}) or {}).get("speed_series_kmh", None)
    p2_speed = (p_dyn.get(2, {}) or {}).get("speed_series_kmh", None)

    # Primary comparisons: Rear/Middle/Front only (player-relative)
    p1_band = _compute_band_stats3(p1_pos, p1_speed, w, h, fps, player="P1")
    p2_band = _compute_band_stats3(p2_pos, p2_speed, w, h, fps, player="P2")

    # General width usage (left/right overall)
    p1_lr = _compute_general_lr(p1_pos, w)
    p2_lr = _compute_general_lr(p2_pos, w)

    # Detailed breakdown (optional below)
    p1_detail = _compute_detail_stats6(p1_pos, p1_speed, w, h, fps, player="P1")
    p2_detail = _compute_detail_stats6(p2_pos, p2_speed, w, h, fps, player="P2")

    # ============================================================
    # Executive summary (same structure; cleaner panel)
    # ============================================================
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([1.15, 1.15, 1.0, 1.0], gap="small")

        def _safe_dom(pct):
            return int(np.nanargmax(pct)) if pct is not None and np.any(np.isfinite(pct)) else 0

        p1_dom_i = _safe_dom(p1_band.pct)
        p2_dom_i = _safe_dom(p2_band.pct)

        dom_help = (
            "Main zone = where the player spent most of the match (Rear/Middle/Front), "
            "measured relative to each player's baseline."
        )
        bal_help = (
            "Balance is an entropy score (0–100). Higher means the player used all zones more evenly."
        )

        with c1:
            st.metric("P1 main zone", BANDS[p1_dom_i], f"{p1_band.pct[p1_dom_i]:.1f}% of time", help=dom_help)
        with c2:
            st.metric("P2 main zone", BANDS[p2_dom_i], f"{p2_band.pct[p2_dom_i]:.1f}% of time", help=dom_help)
        with c3:
            st.metric("P1 court balance", f"{_balance_score_entropy(p1_band.pct):.1f}/100", help=bal_help)
        with c4:
            st.metric("P2 court balance", f"{_balance_score_entropy(p2_band.pct):.1f}/100", help=bal_help)

        
        st.caption("**Main zone** = where the player spends most time (Rear/Middle/Front, relative to their own baseline). **Balance score** = how evenly time is split across zones (0–100; higher = more evenly distributed).")

    st.divider()

    # ============================================================
    # Court + Charts + Insights (same structure; improved panels)
    # ============================================================
    left, right = st.columns([1.3, 1.7], gap="large")

    with left:
        with st.container(border=True):
            st.markdown("#### Court View")
            view = st.radio(
                "Court overlay",
                ["Both", "Player 1", "Player 2"],
                horizontal=True,
                index=0,
                help="Shows the movement path overlay for the selected player(s).",
            )

        fig_court = _build_court_figure(
            title="Court Map (3 Zones + Movement Path)",
            w=w,
            h=h,
            p1_positions=p1_pos,
            p2_positions=p2_pos,
            p1=p1_band,
            p2=p2_band,
            show=view,
        )
        st.plotly_chart(fig_court, use_container_width=True)

    with right:
        with st.container(border=True):
            st.plotly_chart(_build_band_bar(p1_band, p2_band), use_container_width=True)
            st.plotly_chart(_build_lr_bar(p1_lr, p2_lr), use_container_width=True)

        with st.container(border=True):
            st.markdown("### Coach Insights")
            st.markdown("\n".join(_coach_insights_smart(p1_band, p2_band, p1_lr, p2_lr)), unsafe_allow_html=True)

    st.divider()

    # ============================================================
    # Primary comparison table (Rear/Middle/Front)
    # ============================================================
    with st.container(border=True):
        st.markdown("### Zone Comparison (Rear / Middle / Front)")
        st.dataframe(_band_comparison_table(p1_band, p2_band), use_container_width=True, hide_index=True)

    # ============================================================
    # Detailed table (optional)
    # ============================================================
    with st.expander("Detailed breakdown (Rear/Middle/Front × Left/Right)"):
        st.dataframe(_detail_table6(p1_detail, p2_detail), use_container_width=True, hide_index=True)
# webapp/tabs/shot_profile.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import bisect

import numpy as np
import streamlit as st
import plotly.graph_objects as go

# Palette (your updated pro colors)
P1_COLOR = "#F28E2B"   # Player 1 (amber)
P2_COLOR = "#6FA4D9"   # Player 2 (light blue)

# Court constants
from constants import COURT_WIDTH, COURT_LENGTH, SHORT_SERVICE_LINE, LONG_SERVICE_LINE

try:
    from constants import SIDELINE_OFFSET
except Exception:
    SIDELINE_OFFSET = 0.46

BANDS = ["Rear", "Middle", "Front"]


# ============================================================
# Small text helpers
# ============================================================

def _norm_label(s: Any) -> str:
    s = ("" if s is None else str(s)).strip().lower()
    s = s.replace("-", " ")
    return " ".join(s.split())


def _pretty_label(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return "Unknown"
    return s[:1].upper() + s[1:]


# ============================================================
# Geometry + court drawing
# ============================================================

def _m_to_px_y(m: float, h: int) -> float:
    return (m / float(COURT_LENGTH)) * float(h)


def _m_to_px_x(m: float, w: int) -> float:
    return (m / float(COURT_WIDTH)) * float(w)


def _court_geometry_px(w: int, h: int) -> Dict[str, float]:
    net_y = h / 2.0
    long_px = _m_to_px_y(LONG_SERVICE_LINE, h)
    short_px = _m_to_px_y(SHORT_SERVICE_LINE, h)

    y_top_long = long_px
    y_top_short = short_px

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


def _add_text_badge(
    fig: go.Figure,
    x: float,
    y: float,
    text: str,
    font_size: int = 12,
    font_color: str = "white",
    bg: str = "rgba(0,0,0,0.55)",
    border: str = "rgba(255,255,255,0.35)",
    pad_px: int = 5,
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


def _draw_court_3zones(fig: go.Figure, w: int, h: int, show_zones: bool = True) -> Dict[str, float]:
    g = _court_geometry_px(w, h)

    _add_rect(fig, 0, 0, w, h, fill="rgba(30, 120, 70, 0.10)", width=0, opacity=0.9, layer="below")
    _add_rect(fig, 0, 0, w, h, fill="rgba(0,0,0,0)", width=2, dash="solid", opacity=0.9, layer="above")

    gap = max(18, 0.03 * w)
    mid = g["x_center"]
    y = g["net_y"]
    _add_line(fig, 0, y, mid - gap / 2, y, width=3, dash="dot", opacity=0.7)
    _add_line(fig, mid + gap / 2, y, w, y, width=3, dash="dot", opacity=0.7)
    _add_text_badge(fig, x=mid, y=y, text="Net Line", font_size=12, bg="rgba(0,0,0,0.50)", border="rgba(155,155,155,0.50)")

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
        "Rear": "rgba(70, 140, 255, 0.16)",
        "Middle": "rgba(255, 255, 255, 0.10)",
        "Front": "rgba(255, 170, 70, 0.16)",
    }

    _add_rect(fig, 0, 0, w, g["y_top_long"], fill=band_fill["Rear"], width=1, dash="dot", opacity=0.9)
    _add_rect(fig, 0, g["y_top_long"], w, g["y_top_short"], fill=band_fill["Middle"], width=1, dash="dot", opacity=0.9)
    _add_rect(fig, 0, g["y_top_short"], w, g["net_y"], fill=band_fill["Front"], width=1, dash="dot", opacity=0.9)

    _add_rect(fig, 0, g["y_bot_long"], w, h, fill=band_fill["Rear"], width=1, dash="dot", opacity=0.9)
    _add_rect(fig, 0, g["y_bot_short"], w, g["y_bot_long"], fill=band_fill["Middle"], width=1, dash="dot", opacity=0.9)
    _add_rect(fig, 0, g["net_y"], w, g["y_bot_short"], fill=band_fill["Front"], width=1, dash="dot", opacity=0.9)

    centers = _band_centers(g, w, h)
    for b in BANDS:
        x2, y2 = centers[("P2", b)]
        x1, y1 = centers[("P1", b)]
        _add_text_badge(fig, x2, y2, b.upper(), font_size=11, bg="rgba(0,0,0,0.35)", border="rgba(255,255,255,0.25)")
        _add_text_badge(fig, x1, y1, b.upper(), font_size=11, bg="rgba(0,0,0,0.35)", border="rgba(255,255,255,0.25)")

    return g


# ============================================================
# Stats helpers
# ============================================================

def _pct_dict(counts: Dict[str, Any]) -> Dict[str, float]:
    if not counts:
        return {}
    total = float(sum(float(v) for v in counts.values() if v is not None))
    if total <= 0:
        return {k: 0.0 for k in counts.keys()}
    return {k: float(v) / total * 100.0 for k, v in counts.items()}


def _entropy_balance_from_pct(pct: Dict[str, float], eps: float = 1e-12) -> float:
    if not pct:
        return 0.0
    p = np.array(list(pct.values()), dtype=float)
    p = p[np.isfinite(p)]
    if p.size == 0:
        return 0.0
    p = np.clip(p / 100.0, 0.0, 1.0)
    s = float(p.sum())
    if s <= eps:
        return 0.0
    p = p / s
    p = p[p > 0.0]
    H = -float(np.sum(p * np.log(p)))
    Hmax = float(np.log(len(p))) if len(p) > 1 else 0.0
    return float(100.0 * (H / Hmax)) if Hmax > eps else 0.0


def _main_label(pct: Dict[str, float]) -> Tuple[str, float]:
    if not pct:
        return "—", 0.0
    k = max(pct.keys(), key=lambda kk: float(pct.get(kk, 0.0)))
    return k, float(pct.get(k, 0.0))


def _unified_shot_types(p1: Dict[str, Any], p2: Dict[str, Any], total: Dict[str, Any]) -> List[str]:
    keys = set()
    keys.update(p1.keys() if p1 else [])
    keys.update(p2.keys() if p2 else [])
    keys.update(total.keys() if total else [])
    out = list(keys)
    if total:
        out.sort(key=lambda k: float(total.get(k, 0.0) or 0.0), reverse=True)
    else:
        out.sort()
    return out


# ============================================================
# Shot-event ↔ frame alignment helpers
# ============================================================

def _nearest_index(sorted_frames: List[int], target: int) -> int:
    if not sorted_frames:
        return 0
    i = bisect.bisect_left(sorted_frames, int(target))
    if i <= 0:
        return 0
    if i >= len(sorted_frames):
        return len(sorted_frames) - 1
    return i if abs(sorted_frames[i] - target) < abs(sorted_frames[i - 1] - target) else i - 1


def _nearest_ball_speed(ball_frames: List[int], ball_speed: List[float], frame_idx: int) -> float:
    if not ball_frames or not ball_speed:
        return float("nan")
    if len(ball_frames) != len(ball_speed):
        return float("nan")
    j = _nearest_index(ball_frames, frame_idx)
    v = ball_speed[j]
    return float(v) if np.isfinite(v) else float("nan")


def _best_type_display(ev_type: str, canonical_display: List[str]) -> str:
    evn = _norm_label(ev_type)
    if not evn:
        return "Unknown"

    for d in canonical_display:
        if _norm_label(d) == evn:
            return d

    best = None
    best_len = -1
    for d in canonical_display:
        dn = _norm_label(d)
        if not dn:
            continue
        if dn in evn or evn in dn:
            L = min(len(dn), len(evn))
            if L > best_len:
                best = d
                best_len = L

    return best if best else _pretty_label(ev_type)


def _select_events_to_match_counts(
    events: List[Dict[str, Any]],
    p1_counts: Dict[str, Any],
    p2_counts: Dict[str, Any],
    canonical_types: List[str],
    min_conf: float = 0.0,
) -> List[Dict[str, Any]]:
    def _desired(pid: int) -> Dict[str, int]:
        src = p1_counts if pid == 1 else p2_counts
        out: Dict[str, int] = {}
        for k, v in (src or {}).items():
            try:
                n = int(v)
            except Exception:
                continue
            if n > 0:
                out[_norm_label(k)] = n
        return out

    desired1 = _desired(1)
    desired2 = _desired(2)

    evs = []
    for e in events:
        try:
            pid = int(e.get("player_id", 0))
            f = int(e.get("frame_index", -1))
            conf = float(e.get("confidence", 1.0))
        except Exception:
            continue
        if pid not in (1, 2) or f < 0:
            continue
        if not np.isfinite(conf):
            conf = 1.0
        if conf < float(min_conf):
            continue
        e2 = dict(e)
        e2["confidence"] = conf
        evs.append(e2)

    by_player: Dict[int, List[Dict[str, Any]]] = {1: [], 2: []}
    for e in evs:
        by_player[int(e["player_id"])].append(e)

    selected: List[Dict[str, Any]] = []
    used_frame = set()

    def _select_for_player(pid: int, desired: Dict[str, int]) -> None:
        nonlocal selected, used_frame
        pool = by_player.get(pid, [])
        if not pool or not desired:
            return

        pool_by_type: Dict[str, List[Dict[str, Any]]] = {}
        for e in pool:
            disp = _best_type_display(str(e.get("shot_type", "Unknown")), canonical_types)
            keyn = _norm_label(disp)
            e["_shot_display"] = disp
            pool_by_type.setdefault(keyn, []).append(e)

        for keyn, need in desired.items():
            if need <= 0:
                continue
            candidates = pool_by_type.get(keyn, [])
            if not candidates:
                continue
            candidates = sorted(candidates, key=lambda x: float(x.get("confidence", 1.0)), reverse=True)

            taken = 0
            for e in candidates:
                f = int(e["frame_index"])
                if f in used_frame:
                    continue
                selected.append(e)
                used_frame.add(f)
                taken += 1
                if taken >= need:
                    break

        want_total = int(sum(desired.values()))
        have_total = sum(1 for e in selected if int(e["player_id"]) == pid)
        if have_total < want_total:
            remaining = [e for e in pool if int(e["frame_index"]) not in used_frame]
            remaining.sort(key=lambda x: float(x.get("confidence", 1.0)), reverse=True)
            for e in remaining[: (want_total - have_total)]:
                f = int(e["frame_index"])
                if f in used_frame:
                    continue
                disp = _best_type_display(str(e.get("shot_type", "Unknown")), canonical_types)
                e["_shot_display"] = disp
                selected.append(e)
                used_frame.add(f)

    _select_for_player(1, desired1)
    _select_for_player(2, desired2)

    selected.sort(key=lambda x: int(x.get("frame_index", 0)))
    return selected


def _shot_points_from_tracking(
    tracking_data: Dict[str, Any],
    fps: float,
    p1_counts: Dict[str, Any],
    p2_counts: Dict[str, Any],
    canonical_types: List[str],
    min_conf: float = 0.0,
) -> List[Dict[str, Any]]:
    frames: List[int] = tracking_data.get("frame_ids", []) or []
    if not frames:
        return []

    p1_by: List[Optional[Tuple[int, int]]] = tracking_data.get("p1_pos_by_frame", []) or []
    p2_by: List[Optional[Tuple[int, int]]] = tracking_data.get("p2_pos_by_frame", []) or []
    if (not p1_by or not p2_by) or (len(p1_by) != len(frames) or len(p2_by) != len(frames)):
        return []

    p_dyn = tracking_data.get("player_dyn", {}) or {}
    p1_speed = (p_dyn.get(1, {}) or {}).get("speed_series_kmh", []) or []
    p2_speed = (p_dyn.get(2, {}) or {}).get("speed_series_kmh", []) or []

    ball_frames: List[int] = tracking_data.get("ball_frame_ids", []) or []
    ball_speed = tracking_data.get("ball_speed_smooth_kmh", []) or tracking_data.get("ball_speed_kmh", []) or []

    time_s: List[float] = tracking_data.get("time_s", []) or []
    use_time_series = bool(time_s) and (len(time_s) == len(frames))
    f0 = int(frames[0]) if frames else 0
    fps_eff = float(fps) if fps and fps > 1e-6 else 30.0

    events = tracking_data.get("shot_events_parsed", []) or []
    if not events:
        return []

    chosen = _select_events_to_match_counts(
        events=events,
        p1_counts=p1_counts,
        p2_counts=p2_counts,
        canonical_types=canonical_types,
        min_conf=float(min_conf),
    )
    if not chosen:
        return []

    pts: List[Dict[str, Any]] = []
    for ev in chosen:
        try:
            pid = int(ev.get("player_id", 0))
            f = int(ev.get("frame_index", -1))
        except Exception:
            continue
        if pid not in (1, 2) or f < 0:
            continue

        i = _nearest_index(frames, f)
        pos = p1_by[i] if pid == 1 else p2_by[i]
        if pos is None:
            continue
        x, y = int(pos[0]), int(pos[1])

        ps = None
        if pid == 1 and i < len(p1_speed):
            ps = p1_speed[i]
        elif pid == 2 and i < len(p2_speed):
            ps = p2_speed[i]
        ps_val = float(ps) if (ps is not None and np.isfinite(float(ps))) else float("nan")

        bs_val = _nearest_ball_speed(ball_frames, ball_speed, f)

        if use_time_series:
            try:
                t_val = float(time_s[i])
            except Exception:
                t_val = float("nan")
        else:
            t_val = float(f - f0) / fps_eff

        shot_disp = str(ev.get("_shot_display", "")) or _best_type_display(str(ev.get("shot_type", "Unknown")), canonical_types)
        conf = float(ev.get("confidence", 1.0))
        if not np.isfinite(conf):
            conf = 1.0

        pts.append(
            dict(
                x=x, y=y,
                player_id=pid,
                frame=f,
                time_s=t_val,
                shot_type=shot_disp,
                shot_type_norm=_norm_label(shot_disp),
                confidence=conf,
                player_speed_kmh=ps_val,
                ball_speed_kmh=bs_val,
            )
        )

    return pts


def _normalize_marker_sizes(vals: List[float], min_px: int = 8, max_px: int = 22) -> List[float]:
    a = np.array(vals, dtype=float)
    m = np.isfinite(a)
    if not np.any(m):
        return [float(min_px)] * len(vals)
    lo = float(np.nanpercentile(a[m], 10))
    hi = float(np.nanpercentile(a[m], 90))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-9:
        return [float((min_px + max_px) / 2.0)] * len(vals)
    s = (a - lo) / (hi - lo)
    s = np.clip(s, 0.0, 1.0)
    out = min_px + s * (max_px - min_px)
    out[~m] = float(min_px)
    return out.tolist()


def _shot_symbol_map(types: List[str]) -> Dict[str, str]:
    symbols = [
        "x", "circle", "triangle-up", "star", "square",
        "diamond", "hexagon", "pentagon", "triangle-left", "triangle-right",
    ]
    return {t: symbols[i % len(symbols)] for i, t in enumerate(types)}


# ============================================================
# Filtering helpers (LEFT ONLY)
# ============================================================

def _filter_points(points: List[Dict[str, Any]], player_filter: str, selected_types: List[str]) -> List[Dict[str, Any]]:
    selected_norm = set(_norm_label(t) for t in (selected_types or []))
    out: List[Dict[str, Any]] = []
    for p in points or []:
        pid = int(p.get("player_id", 0))
        if player_filter == "Player 1" and pid != 1:
            continue
        if player_filter == "Player 2" and pid != 2:
            continue
        if selected_norm and _norm_label(p.get("shot_type", "")) not in selected_norm:
            continue
        out.append(p)
    return out


def _counts_from_points(points: List[Dict[str, Any]], pid: Optional[int] = None) -> Dict[str, int]:
    c: Dict[str, int] = {}
    for p in points or []:
        if pid is not None and int(p.get("player_id", 0)) != int(pid):
            continue
        k = str(p.get("shot_type", "Unknown"))
        c[k] = int(c.get(k, 0)) + 1
    return c


# ============================================================
# Figures
# ============================================================

def _build_shot_court_figure(
    title: str,
    w: int,
    h: int,
    points: List[Dict[str, Any]],
    view_player: str,
    selected_types: List[str],
    size_by: str,
) -> go.Figure:
    fig = go.Figure()
    _draw_court_3zones(fig, w, h, show_zones=True)

    pts = _filter_points(points, player_filter=view_player, selected_types=selected_types)

    if not pts:
        fig.add_annotation(
            x=0.5, y=0.52, xref="paper", yref="paper",
            text="<b>No shots match the current court filters</b>",
            showarrow=False,
            bgcolor="rgba(0,0,0,0.55)",
            bordercolor="rgba(255,255,255,0.25)",
            borderwidth=1,
            font=dict(color="white", size=13),
        )
        fig.update_layout(
            title=dict(text=f"<b>{title}</b>", x=0.02, xanchor="left"),
            height=760,
            margin=dict(l=16, r=16, t=60, b=16),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
        )
        fig.update_xaxes(range=[0, w], showgrid=False, zeroline=False, visible=False)
        fig.update_yaxes(range=[h, 0], showgrid=False, zeroline=False, visible=False)
        return fig

    if size_by == "Ball speed (km/h)":
        size_vals = [float(p.get("ball_speed_kmh", np.nan)) for p in pts]
        size_label = "Ball speed"
    elif size_by == "Player speed (km/h)":
        size_vals = [float(p.get("player_speed_kmh", np.nan)) for p in pts]
        size_label = "Player speed"
    elif size_by == "Confidence":
        size_vals = [float(p.get("confidence", np.nan)) for p in pts]
        size_label = "Confidence"
    else:
        size_vals = [np.nan] * len(pts)
        size_label = "Fixed"

    sizes = _normalize_marker_sizes(size_vals, min_px=8, max_px=22)

    by_type: Dict[str, List[int]] = {}
    for idx, p in enumerate(pts):
        by_type.setdefault(str(p["shot_type"]), []).append(idx)

    type_order = sorted(by_type.keys(), key=lambda t: len(by_type[t]), reverse=True)
    sym_map = _shot_symbol_map(type_order)

    hover = (
        "<b>%{customdata[0]}</b><br>"
        "Player: %{customdata[1]}<br>"
        "Time: %{customdata[2]:.2f}s • Frame: %{customdata[3]}<br>"
        "Ball speed: %{customdata[4]} km/h<br>"
        "Player speed: %{customdata[5]} km/h<br>"
        "Conf: %{customdata[6]:.2f}<extra></extra>"
    )

    for stype in type_order:
        idxs = by_type[stype]
        xs = [pts[i]["x"] for i in idxs]
        ys = [pts[i]["y"] for i in idxs]
        pids = [int(pts[i]["player_id"]) for i in idxs]

        line_colors = []
        fill_colors = []
        for pid in pids:
            if pid == 1:
                line_colors.append(P1_COLOR)
                fill_colors.append("rgba(242, 142, 43, 0.20)")
            else:
                line_colors.append(P2_COLOR)
                fill_colors.append("rgba(111, 164, 217, 0.20)")

        customdata = []
        for i in idxs:
            p = pts[i]
            bs = p.get("ball_speed_kmh", np.nan)
            ps = p.get("player_speed_kmh", np.nan)
            customdata.append([
                str(p.get("shot_type", "Unknown")),
                "P1" if int(p["player_id"]) == 1 else "P2",
                float(p.get("time_s", np.nan)),
                int(p.get("frame", -1)),
                f"{float(bs):.1f}" if np.isfinite(bs) else "—",
                f"{float(ps):.1f}" if np.isfinite(ps) else "—",
                float(p.get("confidence", 1.0)) if np.isfinite(float(p.get("confidence", 1.0))) else 1.0,
            ])

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                name=str(stype),
                marker=dict(
                    size=[sizes[i] for i in idxs],
                    symbol=sym_map.get(stype, "circle"),
                    color=fill_colors,
                    line=dict(color=line_colors, width=2),
                ),
                customdata=customdata,
                hovertemplate=hover,
                showlegend=True,
            )
        )

    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", x=0.02, xanchor="left"),
        height=800,
        margin=dict(l=16, r=16, t=80, b=25),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        legend=dict(orientation="h", yanchor="top", y=1.05, xanchor="left", x=0.0),
    )
    fig.update_xaxes(range=[0, w], showgrid=False, zeroline=False, visible=False)
    fig.update_yaxes(range=[h, 0], showgrid=False, zeroline=False, visible=False)

    fig.add_annotation(
        x=0.99, y=1.028, xref="paper", yref="paper",
        text=f"<span style='font-size:12px;color:#bdbdbd'>Marker size: {size_label} •  Color: Player •  Symbol: Shot type</span>",
        showarrow=False,
    )

    return fig


def _build_shot_share_bar_both(
    types: List[str],
    p1_pct: Dict[str, float],
    p2_pct: Dict[str, float],
    p1_name: str,
    p2_name: str,
) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(name=p1_name, x=types, y=[p1_pct.get(t, 0.0) for t in types], marker_color=P1_COLOR)
    fig.add_bar(name=p2_name, x=types, y=[p2_pct.get(t, 0.0) for t in types], marker_color=P2_COLOR)

    fig.update_layout(
        barmode="group",
        height=280,
        margin=dict(l=16, r=16, t=74, b=18),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        title=dict(text="<b>Shot Share by Type (Match)</b>", x=0.02, xanchor="left"),
        yaxis_title="Share (%)",
        legend=dict(orientation="h", yanchor="top", y=1.18, xanchor="left", x=0.0),
    )
    fig.update_yaxes(rangemode="tozero")
    fig.update_xaxes(tickangle=-30)
    return fig


def _build_shot_timeline_match(points: List[Dict[str, Any]], title: str = "Shot Timeline (Match)") -> go.Figure:
    """
    Timeline for the full match (no filters). Uses shot-event points if available.
    Shows cumulative shots over time per player.
    """
    fig = go.Figure()

    if not points:
        fig.update_layout(
            height=240,
            margin=dict(l=16, r=16, t=70, b=16),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            title=dict(text=f"<b>{title}</b>", x=0.02, xanchor="left"),
        )
        fig.add_annotation(
            x=0.5, y=0.5, xref="paper", yref="paper",
            text="<b>No shot-event timeline available</b>",
            showarrow=False,
            bgcolor="rgba(0,0,0,0.55)",
            bordercolor="rgba(255,255,255,0.25)",
            borderwidth=1,
            font=dict(color="white", size=12),
        )
        return fig

    pts = sorted(points, key=lambda p: float(p.get("time_s", 0.0) if p.get("time_s") is not None else 0.0))
    t = [float(p.get("time_s", np.nan)) for p in pts]
    pid = [int(p.get("player_id", 0)) for p in pts]

    t1, y1 = [], []
    t2, y2 = [], []
    c1 = 0
    c2 = 0
    for ti, pi in zip(t, pid):
        if not np.isfinite(ti):
            continue
        if pi == 1:
            c1 += 1
            t1.append(ti)
            y1.append(c1)
        elif pi == 2:
            c2 += 1
            t2.append(ti)
            y2.append(c2)

    if t1:
        fig.add_trace(go.Scatter(x=t1, y=y1, mode="lines+markers", name="P1", line=dict(width=3, color=P1_COLOR)))
    if t2:
        fig.add_trace(go.Scatter(x=t2, y=y2, mode="lines+markers", name="P2", line=dict(width=3, color=P2_COLOR)))

    fig.update_layout(
        height=240,
        margin=dict(l=16, r=16, t=70, b=16),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        title=dict(text=f"<b>{title}</b>", x=0.02, xanchor="left"),
        xaxis_title="Time (s)",
        yaxis_title="Cumulative shots",
        legend=dict(orientation="h", yanchor="top", y=1.18, xanchor="left", x=0.0),
    )
    fig.update_yaxes(rangemode="tozero")
    return fig


# ============================================================
# Coach insights (match-level, both players only; no filters)
# ============================================================

def _fmt_pct(x: float) -> str:
    try:
        return f"{float(x):.1f}%"
    except Exception:
        return "—"


def _coach_insights_match(p1_pct: Dict[str, float], p2_pct: Dict[str, float], p1_name: str, p2_name: str) -> List[str]:
    if not p1_pct or not p2_pct:
        return ["- Not enough data for a full match comparison."]

    p1_main, p1_main_pct = _main_label(p1_pct)
    p2_main, p2_main_pct = _main_label(p2_pct)

    notes: List[str] = []
    notes.append(f"- {p1_name} is leaning mostly on **{p1_main}** ({_fmt_pct(p1_main_pct)}), while {p2_name} prefers **{p2_main}** ({_fmt_pct(p2_main_pct)}).")

    types = sorted(set(p1_pct.keys()) | set(p2_pct.keys()))
    diffs = [(t, float(p1_pct.get(t, 0.0)) - float(p2_pct.get(t, 0.0))) for t in types]
    t_big, delta = max(diffs, key=lambda x: abs(x[1])) if diffs else ("—", 0.0)
    if t_big != "—":
        if delta > 0:
            notes.append(f"- Key difference: {p1_name} uses **{t_big}** more (+{_fmt_pct(abs(delta))} vs {p2_name}).")
        else:
            notes.append(f"- Key difference: {p2_name} uses **{t_big}** more (+{_fmt_pct(abs(delta))} vs {p1_name}).")

    b1 = _entropy_balance_from_pct(p1_pct)
    b2 = _entropy_balance_from_pct(p2_pct)
    if abs(b1 - b2) > 1e-6:
        leader = p1_name if b1 > b2 else p2_name
        notes.append(f"- Variation: **{leader}** shows a broader mix (balance {max(b1, b2):.1f}/100).")
    else:
        notes.append("- Variation: both players show a similar level of variety.")

    notes.append("- Coaching note: anticipate the opponent’s most repeated pattern under pressure, and take the shuttle earlier to break their timing.")
    return notes

#========================================
# Report Reusable Tabs
#========================================

def build_report_assets(
    dashboard,
    tracking_data: Dict[str, Any],
    config: Dict[str, Any],
    include_figures: bool = True
) -> Dict[str, Any]:
    p1 = tracking_data.get("player1_shots", {}) or {}
    p2 = tracking_data.get("player2_shots", {}) or {}
    total = tracking_data.get("total_shots", {}) or {}

    if not p1 and not p2:
        return {"error": "No shot data.", "figures": {}, "tables": {}, "kpis": {}, "insights": [], "meta": {}}

    names = tracking_data.get("player_names", {1: "Player 1", 2: "Player 2"}) or {1: "Player 1", 2: "Player 2"}
    p1_name = str(names.get(1, "Player 1"))
    p2_name = str(names.get(2, "Player 2"))

    w = int(config.get("streamlit_dashboard", {}).get("court_width", 385) or 385)
    h = int(config.get("streamlit_dashboard", {}).get("court_height", 840) or 840)
    fps = float(config.get("video_fps", config.get("fps", 30.0)) or 30.0)
    fps = fps if fps > 1e-6 else 30.0

    shot_types = _unified_shot_types(p1, p2, total)
    shot_types = [str(t) for t in shot_types if str(t).strip()]

    min_conf = float(config.get("streamlit_dashboard", {}).get("shot_min_confidence", 0.0) or 0.0)

    # ---- Build points (non-plotly, safe)
    points = _shot_points_from_tracking(
        tracking_data=tracking_data,
        fps=fps,
        p1_counts=p1,
        p2_counts=p2,
        canonical_types=shot_types,
        min_conf=min_conf,
    )

    # ---- Match-level counts (safe)
    if points:
        p1_counts_m = _counts_from_points(points, pid=1)
        p2_counts_m = _counts_from_points(points, pid=2)
    else:
        p1_counts_m = dict(p1)
        p2_counts_m = dict(p2)

    p1_pct_m = _pct_dict(p1_counts_m)
    p2_pct_m = _pct_dict(p2_counts_m)

    axis_types = [t for t in shot_types if (t in p1_counts_m or t in p2_counts_m)] or shot_types

    # ---- Insights + KPIs (safe)
    insights = _coach_insights_match(p1_pct_m, p2_pct_m, p1_name=p1_name, p2_name=p2_name)

    kpis = {
        "p1_main_shot": _main_label(_pct_dict(p1))[0] if p1 else "—",
        "p2_main_shot": _main_label(_pct_dict(p2))[0] if p2 else "—",
        "p1_shot_balance": float(_entropy_balance_from_pct(_pct_dict(p1))),
        "p2_shot_balance": float(_entropy_balance_from_pct(_pct_dict(p2))),
    }

    meta = {
        "p1_name": p1_name,
        "p2_name": p2_name,
        "axis_types": axis_types,
        "shot_types": shot_types,
        "min_conf": min_conf,
        "has_points": bool(points),
    }

    figures: Dict[str, Any] = {}

    # ---- Figures (Plotly) OPTIONAL
    if include_figures:
        try:
            fig_share = _build_shot_share_bar_both(
                axis_types, p1_pct_m, p2_pct_m, p1_name=p1_name, p2_name=p2_name
            )
            fig_timeline = _build_shot_timeline_match(points, title="Shot Timeline (Match)")

            figures = {
                "shot_share": fig_share,
                "shot_timeline": fig_timeline,
            }

            # Court shot map: match-level "Both"
            if points:
                fig_shot_map = _build_shot_court_figure(
                    title="Shot Map (Match)",
                    w=w, h=h,
                    points=points,
                    view_player="Both",
                    selected_types=shot_types,
                    size_by="Ball speed (km/h)",
                )
                figures["shot_map"] = fig_shot_map

        except Exception as e:
            # Do NOT fail the PDF/report if Plotly/pandas are broken
            return {
                "error": f"Shot figures unavailable: {type(e).__name__}: {e}",
                "figures": {},
                "tables": {},
                "kpis": kpis,
                "insights": insights,
                "meta": meta,
            }

    return {
        "figures": figures,
        "tables": {},
        "kpis": kpis,
        "insights": insights,
        "meta": meta,
    }

# ============================================================
# MAIN TAB
# ============================================================

def render(dashboard, tracking_data: Dict[str, Any], config: Dict[str, Any]) -> None:
    st.markdown("## Shot Profile")

    p1 = tracking_data.get("player1_shots", {}) or {}
    p2 = tracking_data.get("player2_shots", {}) or {}
    total = tracking_data.get("total_shots", {}) or {}
    rally = tracking_data.get("rally_summary", {}) or {}

    names = tracking_data.get("player_names", {1: "Player 1", 2: "Player 2"}) or {1: "Player 1", 2: "Player 2"}
    p1_name = str(names.get(1, "Player 1"))
    p2_name = str(names.get(2, "Player 2"))

    if not p1 and not p2:
        st.warning("No shot data.")
        return

    w = int(config.get("streamlit_dashboard", {}).get("court_width", 385))
    h = int(config.get("streamlit_dashboard", {}).get("court_height", 840))

    fps = float(config.get("video_fps", config.get("fps", 30.0)) or 30.0)
    fps = fps if fps > 1e-6 else 30.0

    shot_types = _unified_shot_types(p1, p2, total)
    shot_types = [str(t) for t in shot_types if str(t).strip()]

    # Summary (compact)
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([1.15, 1.15, 1.0, 1.0], gap="small")

        p1_pct_all = _pct_dict(p1)
        p2_pct_all = _pct_dict(p2)
        p1_main, p1_main_pct = _main_label(p1_pct_all)
        p2_main, p2_main_pct = _main_label(p2_pct_all)

        dom_help = "Main shot = the most frequent shot type. Share = count / total shots."
        bal_help = "Shot balance is an entropy score (0–100). Higher means more variety."

        with c1:
            st.metric(f"{p1_name} main shot", p1_main, f"{p1_main_pct:.1f}% share", help=dom_help)
        with c2:
            st.metric(f"{p2_name} main shot", p2_main, f"{p2_main_pct:.1f}% share", help=dom_help)
        with c3:
            st.metric(f"{p1_name} shot balance", f"{_entropy_balance_from_pct(p1_pct_all):.1f}/100", help=bal_help)
        with c4:
            st.metric(f"{p2_name} shot balance", f"{_entropy_balance_from_pct(p2_pct_all):.1f}/100", help=bal_help)

        st.caption("**Main shot** = most frequently used type (% of total shots). **Balance score** = how evenly shots are distributed (0–100; higher = more balanced mix).")

    st.divider()

    # Build shot points for visualizations
    min_conf = float(config.get("streamlit_dashboard", {}).get("shot_min_confidence", 0.0) or 0.0)
    points = _shot_points_from_tracking(
        tracking_data=tracking_data,
        fps=fps,
        p1_counts=p1,
        p2_counts=p2,
        canonical_types=shot_types,
        min_conf=min_conf,
    )

    left, right = st.columns([1.25, 1.75], gap="large")

    # ---------------- Left: Court ONLY (filters exist only here) ----------------
    with left:
        with st.container(border=True):
            st.markdown("#### Court View")
            view_player = st.radio(
                "Show on court",
                ["Both", "Player 1", "Player 2"],
                horizontal=True,
                index=0,
                help="Spatial shot distribution. Each marker = one shot. Color = player, symbol = shot type, size = selected metric.",
            )

            view_types = st.multiselect(
                "Shot types on court",
                options=shot_types,
                default=shot_types,
                help="Affects ONLY the court map.",
            )

            size_by = st.selectbox(
                "Marker size by",
                ["Ball speed (km/h)", "Player speed (km/h)", "Confidence", "Fixed"],
                index=0,
            )

        if points:
            st.plotly_chart(
                _build_shot_court_figure(
                    title="Shot Map (Court View)",
                    w=w, h=h,
                    points=points,
                    view_player=view_player,
                    selected_types=view_types,
                    size_by=size_by,
                ),
                use_container_width=True,
            )
        else:
            st.warning(
                "No shot-event positions available for the court view. "
                "This requires aligned positions (p1_pos_by_frame / p2_pos_by_frame) and shot_events_parsed."
            )

    # ---------------- Right: Match analysis ONLY (no filters) ----------------
    with right:
        # Prefer event-derived counts if points exist; otherwise use summaries.
        if points:
            match_points = points  # no filtering on the right
            p1_counts_m = _counts_from_points(match_points, pid=1)
            p2_counts_m = _counts_from_points(match_points, pid=2)
        else:
            match_points = []
            p1_counts_m = dict(p1)
            p2_counts_m = dict(p2)

        p1_pct_m = _pct_dict(p1_counts_m)
        p2_pct_m = _pct_dict(p2_counts_m)

        axis_types = [t for t in shot_types if (t in p1_counts_m or t in p2_counts_m)] or shot_types

        with st.container(border=True):
            st.plotly_chart(
                _build_shot_share_bar_both(
                    axis_types,
                    p1_pct_m,
                    p2_pct_m,
                    p1_name=p1_name,
                    p2_name=p2_name,
                ),
                use_container_width=True,
            )

            st.markdown("#### Shot Timeline (Match)", 
            help="Shows cumulative shots over match time. "
                 "X-axis = time (seconds). Y-axis = total shots so far. "
                 "Each point represents one detected shot. "
                 "A steeper line indicates higher shot tempo or sustained pressure.")

            st.plotly_chart(
                _build_shot_timeline_match(match_points, title="Shot Timeline (Match)"),
                use_container_width=True,
            )

        with st.container(border=True):
            st.markdown("### Coach Insights ")
            st.markdown("\n".join(_coach_insights_match(p1_pct_m, p2_pct_m, p1_name=p1_name, p2_name=p2_name)))

        st.divider()

        cA, cB = st.columns(2, gap="medium")

        profile_help = (
            "Attack vs Defense breakdown based on shot types over the full match. "
            "Shows each player's tactical tendency. "
            "Higher attacking share indicates a more aggressive playing style."
        )

        with cA:
            st.markdown(f"#### {p1_name} Profile", help=profile_help)
            if p1_counts_m:
                dashboard.render_attack_defense_profile(
                    p1_counts_m,
                    title="Attack vs Defense",
                    use_columns=False
                )
            else:
                st.info("Not enough shots to show a profile.")

        with cB:
            st.markdown(f"#### {p2_name} Profile")
            if p2_counts_m:
                dashboard.render_attack_defense_profile(
                    p2_counts_m,
                    title="Attack vs Defense",
                    use_columns=False
                )
            else:
                st.info("Not enough shots to show a profile.")

    if rally:
        st.divider()
        st.subheader("Rally Stats")
        c1, c2, c3 = st.columns(3)
        c1.metric("Shots", rally.get("shots", 0))
        c2.metric("Duration", f"{rally.get('duration_s', 0):.1f}s")
        c3.metric("Shot Rate", f"{rally.get('shot_rate', 0):.2f}/s")
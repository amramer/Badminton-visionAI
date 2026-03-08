# webapp/reports/helpers.py

from __future__ import annotations

from io import BytesIO
from typing import Dict, Any, List, Tuple, Optional
import re
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image, ImageFilter

# Court constants (same family used by your tabs)
from constants.court_dimensions import COURT_WIDTH, COURT_LENGTH
from constants import SHORT_SERVICE_LINE, LONG_SERVICE_LINE

try:
    from constants import SIDELINE_OFFSET  # singles sideline offset (meters)
except Exception:
    SIDELINE_OFFSET = 0.46


# ============================================================
# ReportLab Python 3.8 md5 patch
# ============================================================
def _patch_reportlab_md5_py38() -> None:
    """
    Fix ReportLab calling hashlib.md5(usedforsecurity=False) on Python 3.8.
    Patch both pdfdoc.md5 and lib.utils.md5.
    """
    try:
        import hashlib

        def _md5_compat(*args, **kwargs):
            kwargs.pop("usedforsecurity", None)
            return hashlib.md5(*args, **kwargs)

        try:
            import reportlab.pdfbase.pdfdoc as pdfdoc
            if hasattr(pdfdoc, "md5"):
                pdfdoc.md5 = _md5_compat
        except Exception:
            pass

        try:
            import reportlab.lib.utils as rl_utils
            if hasattr(rl_utils, "md5"):
                rl_utils.md5 = _md5_compat
        except Exception:
            pass
    except Exception:
        pass


# ============================================================
# ReportLab Paragraph sanitation
# ============================================================
def _sanitize_reportlab_para(text: Any) -> str:
    """
    ReportLab Paragraph supports a limited HTML subset.
    - Convert markdown **bold** to <b>bold</b>
    - Convert unsupported <span style='color:#xxxxxx'> to <font color='...'>
    - Normalize <br> to <br/>
    """
    s = "" if text is None else str(text)

    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(
        r"<\s*span[^>]*style=(['\"])(?:(?!\1).)*?color\s*:\s*([#0-9a-fA-F]{3,8})(?:(?!\1).)*?\1[^>]*>",
        r"<font color='\2'>",
        s,
        flags=re.IGNORECASE | re.DOTALL,
    )
    s = re.sub(r"</\s*span\s*>", r"</font>", s, flags=re.IGNORECASE)
    s = re.sub(r"<\s*span[^>]*>", "", s, flags=re.IGNORECASE)
    s = re.sub(r"</\s*span\s*>", "", s, flags=re.IGNORECASE)
    s = re.sub(r"<\s*br\s*>", "<br/>", s, flags=re.IGNORECASE)
    return s


# ============================================================
# Image helpers
# ============================================================
def _load_image_as_png_bytes(path: str, max_w: int = 1400, max_h: int = 1400) -> Optional[BytesIO]:
    try:
        if not path:
            return None
        if not os.path.exists(path):
            return None
        img = Image.open(path).convert("RGBA")
        w, h = img.size
        scale = min(max_w / max(w, 1), max_h / max(h, 1), 1.0)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        if (nw, nh) != (w, h):
            img = img.resize((nw, nh), Image.LANCZOS)

        buf = BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return buf
    except Exception:
        return None


def _pil_to_png_bytes(img: Image.Image, max_w: int = 1400, max_h: int = 1400) -> Optional[BytesIO]:
    try:
        if img is None:
            return None
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        w, h = img.size
        scale = min(max_w / max(w, 1), max_h / max(h, 1), 1.0)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        if (nw, nh) != (w, h):
            img = img.resize((nw, nh), Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return buf
    except Exception:
        return None


# ============================================================
# Matplotlib -> PNG buffer
# ============================================================
def _mpl_to_buf(fig) -> BytesIO:
    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=200)
    plt.close(fig)
    buf.seek(0)
    return buf


# ============================================================
# Numeric helpers
# ============================================================
def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else float(default)
    except Exception:
        return float(default)


def _fmt_num(x: Any, fmt: str = "{:.1f}") -> str:
    v = _safe_float(x, default=float("nan"))
    return fmt.format(v) if np.isfinite(v) else "—"


# ============================================================
# Robust position normalization
# ============================================================
def _normalize_xy_sequence(seq: Any) -> List[Tuple[int, int]]:
    """
    Accepts many formats:
      - [(x,y), (x,y), ...]
      - [[x,y], [x,y], ...]
      - [(x,y,w,h), ...] -> uses first two
      - list with None / empty items -> skips
      - dicts with {x,y}
    Returns clean list of (int x, int y).
    """
    out: List[Tuple[int, int]] = []
    if not isinstance(seq, list) or not seq:
        return out

    for item in seq:
        if item is None:
            continue
        try:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                x = item[0]
                y = item[1]
            elif isinstance(item, dict) and ("x" in item) and ("y" in item):
                x = item.get("x")
                y = item.get("y")
            else:
                continue

            x = float(x)
            y = float(y)
            if not (np.isfinite(x) and np.isfinite(y)):
                continue

            out.append((int(round(x)), int(round(y))))
        except Exception:
            continue
    return out


# ============================================================
# Heatmap
# ============================================================
def _create_heatmap_overlay(
    court_img: Image.Image,
    p1_positions: List[Tuple[int, int]],
    p2_positions: List[Tuple[int, int]],
    bins: Tuple[int, int] = (32, 64),
    sigma_bins: float = 2.0,
    pixel_blur: float = 6.0,
    alpha_max: int = 235,
    norm_percentile: float = 98.0,
    gamma: float = 0.65,
    floor: float = 0.03,
) -> Image.Image:
    """
    Creates a combined heatmap overlay similar to StreamlitDashboard.create_heatmap,
    but without Streamlit dependencies (pure PIL/NumPy).
    """
    base = court_img.convert("RGBA")
    w, h = base.size

    def _valid_xy(pos):
        xs, ys = [], []
        for x, y in pos or []:
            if x is None or y is None:
                continue
            try:
                x = int(x)
                y = int(y)
            except Exception:
                continue
            if 0 <= x < w and 0 <= y < h:
                xs.append(x)
                ys.append(y)
        return np.array(xs, dtype=float), np.array(ys, dtype=float)

    x1, y1 = _valid_xy(p1_positions)
    x2, y2 = _valid_xy(p2_positions)

    if x1.size < 2 and x2.size < 2:
        return base

    xbins, ybins = int(bins[0]), int(bins[1])
    xbins = max(8, xbins)
    ybins = max(8, ybins)

    def _hist(x, y):
        if x.size < 2:
            return np.zeros((xbins, ybins), dtype=float)
        H, _, _ = np.histogram2d(x, y, bins=[xbins, ybins], range=[[0, w], [0, h]])
        return H

    H1 = _hist(x1, y1)
    H2 = _hist(x2, y2)

    def _smooth(A: np.ndarray, iters: int = 2) -> np.ndarray:
        if not np.any(A):
            return A
        B = A.copy()
        for _ in range(max(1, iters)):
            B = (
                B
                + np.roll(B, 1, axis=0) + np.roll(B, -1, axis=0)
                + np.roll(B, 1, axis=1) + np.roll(B, -1, axis=1)
            ) / 5.0
        return B

    H1 = _smooth(H1, iters=max(1, int(round(float(sigma_bins)))))
    H2 = _smooth(H2, iters=max(1, int(round(float(sigma_bins)))))

    def _robust_norm(A: np.ndarray) -> np.ndarray:
        if not np.any(A):
            return A * 0.0
        nz = A[A > 0]
        if nz.size == 0:
            return A * 0.0
        p = float(np.percentile(nz, float(norm_percentile)))
        denom = float(max(p, 1e-12))
        return np.clip(A / denom, 0.0, 1.0)

    N1 = _robust_norm(H1)
    N2 = _robust_norm(H2)

    def _boost(N: np.ndarray) -> np.ndarray:
        if not np.any(N):
            return N
        N = np.clip(N, 0.0, 1.0)
        N = np.where(N < float(floor), 0.0, N)
        return np.power(N, float(gamma))

    N1 = _boost(N1)
    N2 = _boost(N2)

    def _to_overlay(N: np.ndarray, color_rgb: Tuple[int, int, int]) -> Image.Image:
        if not np.any(N):
            return Image.new("RGBA", (w, h), (0, 0, 0, 0))

        rgba = np.zeros((xbins, ybins, 4), dtype=np.uint8)
        rgba[..., 0] = color_rgb[0]
        rgba[..., 1] = color_rgb[1]
        rgba[..., 2] = color_rgb[2]
        rgba[..., 3] = np.clip(N * (float(alpha_max)), 0.0, 255.0).astype(np.uint8)

        rgba_img = np.transpose(rgba, (1, 0, 2))
        small = Image.fromarray(rgba_img, mode="RGBA")
        overlay = small.resize((w, h), resample=Image.BILINEAR)

        if pixel_blur and pixel_blur > 0:
            overlay = overlay.filter(ImageFilter.GaussianBlur(radius=float(pixel_blur)))

        return overlay

    o1 = _to_overlay(N1, (255, 165, 0))
    o2 = _to_overlay(N2, (0, 100, 255))

    def _screen_blend(a: Image.Image, b: Image.Image) -> Image.Image:
        A = np.asarray(a).astype(np.float32) / 255.0
        B = np.asarray(b).astype(np.float32) / 255.0
        rgb = 1.0 - (1.0 - A[..., :3]) * (1.0 - B[..., :3])
        alpha = np.maximum(A[..., 3], B[..., 3])[..., None]
        out = np.concatenate([rgb, alpha], axis=-1)
        return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8), mode="RGBA")

    combined = _screen_blend(o1, o2)
    return Image.alpha_composite(base, combined)


# ============================================================
# Shot aggregation helpers
# ============================================================
def _attack_defense_counts(shots: Dict[str, Any]) -> Tuple[int, int]:
    """
    Definition:
      Attack = Smash
      Defense = Lift or Clear
    """
    A = 0
    D = 0
    for k, v in (shots or {}).items():
        name = str(k).strip().lower()
        try:
            n = int(v)
        except Exception:
            n = 0
        if "smash" in name:
            A += n
        elif ("lift" in name) or ("clear" in name):
            D += n
    return A, D


def _pct_from_counts(counts: Dict[str, Any]) -> Dict[str, float]:
    if not counts:
        return {}
    total = 0.0
    for v in counts.values():
        try:
            total += float(v)
        except Exception:
            pass
    if total <= 1e-9:
        return {str(k): 0.0 for k in counts.keys()}
    out: Dict[str, float] = {}
    for k, v in counts.items():
        try:
            out[str(k)] = float(v) / total * 100.0
        except Exception:
            out[str(k)] = 0.0
    return out


# ============================================================
# Matplotlib figure builders
# ============================================================
def _mpl_group_bar(
    title: str,
    labels: List[str],
    y1: List[float],
    y2: List[float],
    n1: str,
    n2: str,
    ylabel: str,
) -> BytesIO:
    fig, ax = plt.subplots(figsize=(7.6, 3.25), dpi=200)
    x = np.arange(len(labels))
    w = 0.36
    ax.bar(x - w / 2, y1, w, label=n1)
    ax.bar(x + w / 2, y2, w, label=n2)
    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(fontsize=9, frameon=False, ncol=2, loc="upper right")
    return _mpl_to_buf(fig)


def _mpl_shot_timeline(points: List[Dict[str, Any]], p1_name: str, p2_name: str) -> BytesIO:
    fig, ax = plt.subplots(figsize=(7.6, 3.1), dpi=200)
    if not points:
        ax.text(0.5, 0.5, "No shot-event timeline available", ha="center", va="center")
        ax.set_axis_off()
        return _mpl_to_buf(fig)

    pts = sorted(points, key=lambda p: float(p.get("time_s", 0.0) if p.get("time_s") is not None else 0.0))
    t1, y1, t2, y2 = [], [], [], []
    c1 = 0
    c2 = 0
    for p in pts:
        t = p.get("time_s", None)
        if t is None:
            continue
        try:
            t = float(t)
        except Exception:
            continue
        if not np.isfinite(t):
            continue
        pid = int(p.get("player_id", 0))
        if pid == 1:
            c1 += 1
            t1.append(t)
            y1.append(c1)
        elif pid == 2:
            c2 += 1
            t2.append(t)
            y2.append(c2)

    if t1:
        ax.plot(t1, y1, marker="o", linewidth=2, markersize=3, label=p1_name)
    if t2:
        ax.plot(t2, y2, marker="o", linewidth=2, markersize=3, label=p2_name)

    ax.set_title("Shot Timeline (Match)", fontsize=11, pad=10)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Cumulative shots", fontsize=9)
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(alpha=0.22)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    return _mpl_to_buf(fig)


# ============================================================
# Court geometry
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


def _mpl_draw_court_like_tabs(ax, w: int, h: int, title: str = "", show_zone_labels: bool = True) -> None:
    """
    Court drawing.
    Includes:
      - Outer border
      - Net line (dotted split)
      - Service lines (short/long)
      - Center service lines
      - Singles sidelines (optional)
      - 3-zone fills (Rear/Middle/Front) per half
    """
    g = _court_geometry_px(w, h)

    ax.add_patch(Rectangle((0, 0), w, h, fill=False, linewidth=2.0))

    mid = g["x_center"]
    y = g["net_y"]
    gap = max(18, 0.03 * w)
    ax.plot([0, mid - gap / 2], [y, y], linewidth=2.6, linestyle=":", alpha=0.75)
    ax.plot([mid + gap / 2, w], [y, y], linewidth=2.6, linestyle=":", alpha=0.75)

    ax.plot([0, w], [g["y_top_short"], g["y_top_short"]], linewidth=1.8, alpha=0.85)
    ax.plot([0, w], [g["y_bot_short"], g["y_bot_short"]], linewidth=1.8, alpha=0.85)
    ax.plot([0, w], [g["y_top_long"], g["y_top_long"]], linewidth=1.8, alpha=0.85)
    ax.plot([0, w], [g["y_bot_long"], g["y_bot_long"]], linewidth=1.8, alpha=0.85)

    ax.plot([g["x_center"], g["x_center"]], [g["y_top_long"], g["y_top_short"]], linewidth=1.8, alpha=0.85)
    ax.plot([g["x_center"], g["x_center"]], [g["y_bot_short"], g["y_bot_long"]], linewidth=1.8, alpha=0.85)

    if 0 < g["x_singles_left"] < w / 2:
        ax.plot([g["x_singles_left"], g["x_singles_left"]], [0, h], linewidth=1.2, alpha=0.55)
        ax.plot([g["x_singles_right"], g["x_singles_right"]], [0, h], linewidth=1.2, alpha=0.55)

    band_fill = {
        "Rear": (70 / 255, 140 / 255, 255 / 255, 0.14),
        "Middle": (1.0, 1.0, 1.0, 0.08),
        "Front": (1.0, 170 / 255, 70 / 255, 0.14),
    }

    ax.add_patch(Rectangle((0, 0), w, g["y_top_long"], facecolor=band_fill["Rear"], edgecolor="none"))
    ax.add_patch(Rectangle((0, g["y_top_long"]), w, g["y_top_short"] - g["y_top_long"], facecolor=band_fill["Middle"], edgecolor="none"))
    ax.add_patch(Rectangle((0, g["y_top_short"]), w, g["net_y"] - g["y_top_short"], facecolor=band_fill["Front"], edgecolor="none"))

    ax.add_patch(Rectangle((0, g["y_bot_long"]), w, h - g["y_bot_long"], facecolor=band_fill["Rear"], edgecolor="none"))
    ax.add_patch(Rectangle((0, g["y_bot_short"]), w, g["y_bot_long"] - g["y_bot_short"], facecolor=band_fill["Middle"], edgecolor="none"))
    ax.add_patch(Rectangle((0, g["net_y"]), w, g["y_bot_short"] - g["net_y"], facecolor=band_fill["Front"], edgecolor="none"))

    if show_zone_labels:
        def _badge(x, y, text):
            ax.text(
                x, y, text,
                ha="center", va="center",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.25", facecolor=(0, 0, 0, 0.32), edgecolor=(1, 1, 1, 0.25)),
                alpha=0.95,
            )

        cx = w / 2.0
        _badge(cx, (0 + g["y_top_long"]) / 2.0, "REAR")
        _badge(cx, (g["y_top_long"] + g["y_top_short"]) / 2.0, "MIDDLE")
        _badge(cx, (g["y_top_short"] + g["net_y"]) / 2.0, "FRONT")

        _badge(cx, (g["y_bot_long"] + h) / 2.0, "REAR")
        _badge(cx, (g["y_bot_short"] + g["y_bot_long"]) / 2.0, "MIDDLE")
        _badge(cx, (g["net_y"] + g["y_bot_short"]) / 2.0, "FRONT")

        ax.text(mid, y - 8, "NET", ha="center", va="bottom", fontsize=9, alpha=0.75)

        ax.text(w - 6, 10, "Top side (P2)", fontsize=8, alpha=0.6, ha="right", va="top")
        ax.text(w - 6, h - 10, "Bottom side (P1)", fontsize=8, alpha=0.6, ha="right", va="bottom")

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xticks([])
    ax.set_yticks([])


def _mpl_positioning_court_map(
    w: int,
    h: int,
    p1_positions_raw: Any,
    p2_positions_raw: Any,
    p1_name: str,
    p2_name: str,
    max_points: int = 900,
) -> Optional[BytesIO]:
    p1_positions = _normalize_xy_sequence(p1_positions_raw)
    p2_positions = _normalize_xy_sequence(p2_positions_raw)
    if len(p1_positions) < 5 and len(p2_positions) < 5:
        return None

    def _downsample(pts: List[Tuple[int, int]], n: int) -> List[Tuple[int, int]]:
        if not pts:
            return []
        if len(pts) <= n:
            return pts
        idx = np.linspace(0, len(pts) - 1, n).astype(int)
        return [pts[i] for i in idx]

    p1d = _downsample(p1_positions, max_points)
    p2d = _downsample(p2_positions, max_points)

    fig, ax = plt.subplots(figsize=(7.6, 9.0), dpi=200)
    _mpl_draw_court_like_tabs(ax, w, h, title="Court Map — Movement Paths", show_zone_labels=True)

    p1_color = "#F28E2B"
    p2_color = "#6FA4D9"

    if p1d:
        ax.plot([x for x, _y in p1d], [_y for _x, _y in p1d], linewidth=2.0, alpha=0.85, color=p1_color, label=p1_name)
        ax.scatter([p1d[0][0]], [p1d[0][1]], s=60, marker="s", color=p1_color, edgecolors="white", linewidths=0.8, zorder=5)
        ax.scatter([p1d[-1][0]], [p1d[-1][1]], s=70, marker="X", color=p1_color, edgecolors="white", linewidths=0.8, zorder=5)

    if p2d:
        ax.plot([x for x, _y in p2d], [_y for _x, _y in p2d], linewidth=2.0, alpha=0.85, color=p2_color, label=p2_name)
        ax.scatter([p2d[0][0]], [p2d[0][1]], s=60, marker="s", color=p2_color, edgecolors="white", linewidths=0.8, zorder=5)
        ax.scatter([p2d[-1][0]], [p2d[-1][1]], s=70, marker="X", color=p2_color, edgecolors="white", linewidths=0.8, zorder=5)

    ax.legend(fontsize=9, frameon=False, loc="upper center", ncol=2)
    ax.text(
        0.01,
        -0.04,
        "Markers: ■ start, ✖ end. Paths are downsampled for PDF readability.",
        transform=ax.transAxes,
        fontsize=8,
        alpha=0.7,
        va="top",
    )
    return _mpl_to_buf(fig)


def _mpl_shot_map(
    w: int,
    h: int,
    points: List[Dict[str, Any]],
    p1_name: str,
    p2_name: str,
    max_points: int = 700,
) -> BytesIO:
    fig, ax = plt.subplots(figsize=(7.6, 9.0), dpi=200)
    _mpl_draw_court_like_tabs(ax, w, h, title="Shot Map — Event Locations", show_zone_labels=True)

    if not points:
        ax.text(0.5, 0.5, "Shot map unavailable (no aligned shot events)", ha="center", va="center", transform=ax.transAxes)
        return _mpl_to_buf(fig)

    pts = points[:]
    if len(pts) > max_points:
        idx = np.linspace(0, len(pts) - 1, max_points).astype(int)
        pts = [pts[i] for i in idx]

    p1_color = "#F28E2B"
    p2_color = "#6FA4D9"

    x1 = [p["x"] for p in pts if int(p.get("player_id", 0)) == 1]
    y1 = [p["y"] for p in pts if int(p.get("player_id", 0)) == 1]
    x2 = [p["x"] for p in pts if int(p.get("player_id", 0)) == 2]
    y2 = [p["y"] for p in pts if int(p.get("player_id", 0)) == 2]

    if x1:
        ax.scatter(x1, y1, s=22, alpha=0.75, marker="o", color=p1_color, edgecolors="white", linewidths=0.4, label=p1_name)
    if x2:
        ax.scatter(x2, y2, s=22, alpha=0.75, marker="^", color=p2_color, edgecolors="white", linewidths=0.4, label=p2_name)

    ax.legend(fontsize=9, frameon=False, loc="upper center", ncol=2)
    ax.text(
        0.01,
        -0.04,
        "Markers: ○ Player 1, △ Player 2. Points are downsampled for PDF readability.",
        transform=ax.transAxes,
        fontsize=8,
        alpha=0.7,
        va="top",
    )
    return _mpl_to_buf(fig)


# ============================================================
# Shot points builder
# ============================================================
def _nearest_index(sorted_frames: List[int], target: int) -> int:
    if not sorted_frames:
        return 0
    lo, hi = 0, len(sorted_frames) - 1
    if target <= sorted_frames[0]:
        return 0
    if target >= sorted_frames[-1]:
        return hi
    while lo <= hi:
        mid = (lo + hi) // 2
        if sorted_frames[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    i = max(1, min(lo, len(sorted_frames) - 1))
    return i if abs(sorted_frames[i] - target) < abs(sorted_frames[i - 1] - target) else i - 1


def _build_shot_points_for_report(tracking_data: Dict[str, Any], fps: float) -> List[Dict[str, Any]]:
    frames: List[int] = tracking_data.get("frame_ids", []) or []
    if not frames:
        return []

    p1_by = tracking_data.get("p1_pos_by_frame", []) or []
    p2_by = tracking_data.get("p2_pos_by_frame", []) or []
    if not p1_by or not p2_by:
        return []
    if len(p1_by) != len(frames) or len(p2_by) != len(frames):
        return []

    events = tracking_data.get("shot_events_parsed", []) or []
    if not events:
        return []

    time_s = tracking_data.get("time_s", []) or []
    use_time = bool(time_s) and (len(time_s) == len(frames))
    fps_eff = float(fps) if fps and fps > 1e-6 else 30.0
    f0 = int(frames[0])

    evs = []
    for e in events:
        if not isinstance(e, dict):
            continue
        try:
            pid = int(e.get("player_id", 0))
            f = int(e.get("frame_index", -1))
        except Exception:
            continue
        if pid not in (1, 2) or f < 0:
            continue
        conf = e.get("confidence", 1.0)
        try:
            conf = float(conf)
        except Exception:
            conf = 1.0
        if not np.isfinite(conf):
            conf = 1.0
        stype = str(e.get("shot_type", "Unknown"))
        evs.append((f, pid, conf, stype))

    evs.sort(key=lambda x: x[0])
    pts: List[Dict[str, Any]] = []

    for f, pid, conf, stype in evs:
        i = _nearest_index(frames, f)
        pos = p1_by[i] if pid == 1 else p2_by[i]
        if pos is None:
            continue
        try:
            x, y = float(pos[0]), float(pos[1])
        except Exception:
            continue
        if not (np.isfinite(x) and np.isfinite(y)):
            continue

        if use_time:
            try:
                t = float(time_s[i])
            except Exception:
                t = float(f - f0) / fps_eff
        else:
            t = float(f - f0) / fps_eff

        pts.append(
            dict(
                x=int(round(x)),
                y=int(round(y)),
                player_id=pid,
                frame=f,
                time_s=t,
                shot_type=stype,
                confidence=conf,
            )
        )
    return pts


# ============================================================
# Insight hygiene
# ============================================================
def _clean_insights(raw: List[Any], limit: int = 8) -> List[str]:
    out: List[str] = []
    for x in (raw or []):
        s = str(x).strip()
        if not s:
            continue
        s = s.lstrip("-• ").strip()
        s = re.sub(r"\s+", " ", s)
        out.append(s)

    seen = set()
    uniq = []
    for s in out:
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(s)
    return uniq[: max(1, int(limit))]


# ============================================================
# Recommendations derived from analysis
# ============================================================
def _recommendations_from_signals(
    p1_name: str,
    p2_name: str,
    pos_k: Dict[str, Any],
    shot_k: Dict[str, Any],
    p1_shots: Dict[str, Any],
    p2_shots: Dict[str, Any],
) -> List[str]:
    recs: List[str] = []

    p1_left = _safe_float(pos_k.get("p1_left_pct", np.nan), np.nan)
    p2_left = _safe_float(pos_k.get("p2_left_pct", np.nan), np.nan)

    def _width_bias(left_pct: float) -> Optional[str]:
        if not np.isfinite(left_pct):
            return None
        if left_pct >= 65:
            return "left"
        if left_pct <= 35:
            return "right"
        return None

    b1 = _width_bias(p1_left)
    b2 = _width_bias(p2_left)
    if b1:
        recs.append(f"{p1_name}: noticeable {b1}-side bias — add neutral recovery to center after corner shots + weak-side first step drills.")
    if b2:
        recs.append(f"{p2_name}: noticeable {b2}-side bias — add neutral recovery to center after corner shots + weak-side first step drills.")

    p1_bal = _safe_float(pos_k.get("p1_balance", np.nan), np.nan)
    p2_bal = _safe_float(pos_k.get("p2_balance", np.nan), np.nan)
    if np.isfinite(p1_bal) and p1_bal < 35:
        recs.append(f"{p1_name}: low zone balance — use structured rear↔mid↔front footwork blocks with fixed targets (10–12 reps per pattern).")
    if np.isfinite(p2_bal) and p2_bal < 35:
        recs.append(f"{p2_name}: low zone balance — use structured rear↔mid↔front footwork blocks with fixed targets (10–12 reps per pattern).")

    def _dominant(shot_kpis: Dict[str, Any], counts: Dict[str, Any]) -> Tuple[str, float]:
        main = str(shot_kpis or "")
        pct = 0.0
        pct_map = _pct_from_counts(counts)
        if main and main != "—":
            pct = float(pct_map.get(main, 0.0))
        return main, pct

    p1_main = str(shot_k.get("p1_main_shot", "—"))
    p2_main = str(shot_k.get("p2_main_shot", "—"))
    p1_dom, p1_dom_pct = _dominant(p1_main, p1_shots)
    p2_dom, p2_dom_pct = _dominant(p2_main, p2_shots)

    if p1_dom and p1_dom != "—" and p1_dom_pct >= 55:
        recs.append(f"{p1_name}: heavy reliance on {p1_dom} (~{p1_dom_pct:.0f}%) — add 1–2 change-up options off the same preparation to reduce predictability.")
    if p2_dom and p2_dom != "—" and p2_dom_pct >= 55:
        recs.append(f"{p2_name}: heavy reliance on {p2_dom} (~{p2_dom_pct:.0f}%) — add 1–2 change-up options off the same preparation to reduce predictability.")

    a1, d1 = _attack_defense_counts(p1_shots)
    a2, d2 = _attack_defense_counts(p2_shots)

    def _ad_note(name: str, A: int, D: int) -> Optional[str]:
        denom = A + D
        if denom <= 0:
            return None
        a_pct = 100.0 * A / denom
        if a_pct >= 65:
            return f"{name}: attack-heavy profile — prioritize shot quality (steepness + placement) and ensure recovery step after smashes."
        if a_pct <= 35:
            return f"{name}: defense-heavy profile — add transition patterns (lift/clear → mid-court intercept → first attack)."
        return None

    n1 = _ad_note(p1_name, a1, d1)
    n2 = _ad_note(p2_name, a2, d2)
    if n1:
        recs.append(n1)
    if n2:
        recs.append(n2)

    if not recs:
        recs.append("No strong single-axis bias detected — keep training balanced: early contact, recovery to base, and pattern variation under pressure.")
    return recs[:10]


# ============================================================
# ReportLab styles + layout
# ============================================================
def build_styles():
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors

    styles = getSampleStyleSheet()
    styles["Normal"].fontName = "Helvetica"
    styles["Normal"].fontSize = 10
    styles["Normal"].leading = 13

    styles.add(ParagraphStyle(
        name="Small", parent=styles["Normal"],
        fontSize=9, leading=11.5, textColor=colors.grey
    ))
    styles.add(ParagraphStyle(
        name="Tiny", parent=styles["Normal"],
        fontSize=8, leading=10, textColor=colors.grey
    ))
    styles.add(ParagraphStyle(
        name="CoverTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=28,
        leading=34,
        alignment=1,
        spaceBefore=20,
        spaceAfter=6,
        textColor=colors.HexColor("#FD9F00"),
    ))
    styles.add(ParagraphStyle(
        name="CoverSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=22,
        alignment=1,
        spaceBefore=0,
        spaceAfter=8,
        textColor=colors.HexColor("#333333"),
    ))
    styles.add(ParagraphStyle(
        name="CoverKicker",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=14,
        alignment=1,
        spaceBefore=0,
        spaceAfter=0,
        textColor=colors.HexColor("#666666"),
    ))
    styles.add(ParagraphStyle(
        name="H1", parent=styles["Heading1"],
        fontSize=16, leading=20, spaceBefore=10, spaceAfter=8,
        textColor=colors.HexColor("#FD9F00")
    ))
    styles.add(ParagraphStyle(
        name="H2", parent=styles["Heading2"],
        fontSize=13, leading=16, spaceBefore=10, spaceAfter=6,
        textColor=colors.HexColor("#333333")
    ))
    styles.add(ParagraphStyle(
        name="BoxTitle", parent=styles["Normal"],
        fontSize=10, leading=12, textColor=colors.HexColor("#333333"), spaceAfter=6
    ))
    return styles


def _draw_logo(canvas, doc_, logo_path: str) -> None:
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.lib.utils import ImageReader

    try:
        if not logo_path or (not os.path.exists(logo_path)):
            return

        ir = ImageReader(logo_path)
        iw, ih = ir.getSize()
        if not iw or not ih:
            return

        target_h = 1.3 * cm
        scale = float(target_h) / float(ih)
        target_w = float(iw) * scale

        page_w, page_h = doc_.pagesize
        x = page_w - doc_.rightMargin - target_w
        y = page_h - doc_.topMargin + 1.7 * cm

        canvas.saveState()
        canvas.setFillColor(colors.white)
        canvas.setStrokeColor(colors.Color(0, 0, 0, alpha=0.14))
        canvas.setLineWidth(0.6)
        canvas.restoreState()

        canvas.drawImage(ir, x, y, width=target_w, height=target_h, mask="auto")
    except Exception:
        return


def _draw_header_footer(canvas, doc_):
    from reportlab.lib import colors
    from reportlab.lib.units import cm

    canvas.saveState()

    page_w, page_h = doc_.pagesize
    header_text_y = page_h - 0.95 * cm
    divider_y = page_h - 1.35 * cm

    if doc_.page == 1:
        title = "Coach Assistant Report"
        title_color = colors.HexColor("#FD9F00")
        title_font = "Helvetica-Bold"
        title_size = 10
    else:
        title = "Badminton-vision AI — Coach Assistant Report"
        title_color = colors.HexColor("#333333")
        title_font = "Helvetica-Bold"
        title_size = 10

    canvas.setFillColor(title_color)
    canvas.setFont(title_font, title_size)
    canvas.drawString(doc_.leftMargin, header_text_y, title)

    _draw_logo(canvas, doc_, "assets/logo.png")

    canvas.setStrokeColor(colors.Color(0, 0, 0, alpha=0.12))
    canvas.setLineWidth(0.8)
    canvas.line(doc_.leftMargin, divider_y, page_w - doc_.rightMargin, divider_y)

    canvas.setFillColor(colors.grey)
    canvas.setFont("Helvetica", 9)
    canvas.drawString(doc_.leftMargin, 0.85 * cm, "Badminton-vision AI")
    canvas.drawRightString(page_w - doc_.rightMargin, 0.85 * cm, f"Page {doc_.page}")

    canvas.restoreState()


def build_doc_template(buffer: BytesIO):
    from reportlab.platypus import SimpleDocTemplate, Paragraph
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm

    class _MyDocTemplate(SimpleDocTemplate):
        def afterFlowable(self, flowable):
            if isinstance(flowable, Paragraph):
                style = flowable.style.name
                text = flowable.getPlainText()
                if style == "H1":
                    self.notify("TOCEntry", (0, text, self.page))
                elif style == "H2":
                    self.notify("TOCEntry", (1, text, self.page))

    doc = _MyDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=1.2 * cm,
        rightMargin=1.0 * cm,
        topMargin=3.0 * cm,
        bottomMargin=1.5 * cm,
        title="Badminton-vision AI Coach Assistant Report",
        author="Badminton-vision AI",
    )
    return doc


def build_toc():
    from reportlab.platypus.tableofcontents import TableOfContents
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib import colors

    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle(fontName="Helvetica", fontSize=10, name="TOC0", leftIndent=10, firstLineIndent=-10, spaceBefore=2, leading=12),
        ParagraphStyle(fontName="Helvetica", fontSize=9, name="TOC1", leftIndent=24, firstLineIndent=-10, spaceBefore=1, leading=11, textColor=colors.grey),
    ]
    return toc


def _bullet_list(items: List[str], styles):
    from reportlab.platypus import Paragraph, ListFlowable, ListItem

    flow: List[ListItem] = []
    for t in items:
        flow.append(ListItem(Paragraph(_sanitize_reportlab_para(t), styles["Normal"]), leftIndent=14))
    return ListFlowable(flow, bulletType="bullet", leftIndent=14, bulletFontName="Helvetica", bulletFontSize=10)


def _kv_table(rows: List[Tuple[str, str]], styles, colw=(7.2, 8.0)):
    from reportlab.platypus import Paragraph, Table, TableStyle
    from reportlab.lib import colors
    from reportlab.lib.units import cm

    data = [[
        Paragraph(f"<b>{_sanitize_reportlab_para(k)}</b>", styles["Normal"]),
        Paragraph(_sanitize_reportlab_para(v), styles["Normal"])
    ] for k, v in rows]

    t = Table(data, colWidths=[colw[0] * cm, colw[1] * cm])
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.35, colors.lightgrey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def _boxed(title: str, elements: List[Any], styles):
    from reportlab.platypus import Paragraph, Table, TableStyle
    from reportlab.lib import colors
    from reportlab.lib.units import cm

    inner = [Paragraph(f"<b>{_sanitize_reportlab_para(title)}</b>", styles["BoxTitle"])]
    inner.extend(elements)
    t = Table([[inner]], colWidths=[16.8 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.whitesmoke),
        ("BOX", (0, 0), (-1, -1), 0.6, colors.lightgrey),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
    ]))
    return t


def _rl_img(png_buf: Optional[BytesIO], width_cm: float, height_cm: float, styles) -> Any:
    from reportlab.platypus import Paragraph
    from reportlab.platypus import Image as RLImage
    from reportlab.lib.units import cm

    if not png_buf:
        return Paragraph("Figure unavailable for current data.", styles["Small"])
    try:
        return RLImage(png_buf, width=width_cm * cm, height=height_cm * cm)
    except Exception:
        return Paragraph("Figure unavailable for current data.", styles["Small"])


def _find_player_photo(player_data: Dict[str, Any], name: str) -> Optional[str]:
    if not name:
        return None

    if name in player_data and isinstance(player_data[name], dict):
        p = player_data[name].get("photo", "")
        return p if p else None

    name_l = name.lower()
    for k, v in player_data.items():
        if not isinstance(v, dict):
            continue
        if str(k).lower() in name_l or name_l in str(k).lower():
            p = v.get("photo", "")
            return p if p else None
    return None
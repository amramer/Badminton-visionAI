# webapp/streamlitdashboard.py
import os
import base64
import numpy as np
import streamlit as st
import matplotlib.cm as cm
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from constants.court_dimensions import COURT_WIDTH, COURT_LENGTH
from typing import Dict, Tuple, List, Literal, Optional, Any
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops

ResizeMode = Literal["fit", "cover", "stretch"]


class StreamlitDashboard:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    # ============================================================
    # Utils
    # ============================================================
    @staticmethod
    def _to_float(x: Any, default: float = np.nan) -> float:
        try:
            return float(x)
        except Exception:
            return default

    @staticmethod
    def _is_intlike(x: Any) -> bool:
        try:
            int(x)
            return True
        except Exception:
            return False

    @staticmethod
    def _norm_label(s: Any) -> str:
        s = ("" if s is None else str(s)).strip().lower()
        s = s.replace("-", " ")
        return " ".join(s.split())

    @staticmethod
    def _fmt_time_mmss(t: Optional[float]) -> str:
        if t is None or not np.isfinite(t) or t < 0:
            return "—"
        m = int(t // 60)
        s = int(round(t - 60 * m))
        if m > 0:
            return f"{m}m {s:02d}s"
        return f"{s}s"

    # ============================================================
    # Attack/Defense bucketing
    # ============================================================
    @classmethod
    def _shot_bucket(cls, shot_type: Any) -> str:
        stype = cls._norm_label(shot_type)
        if "smash" in stype:
            return "attack"
        if "lift" in stype or "clear" in stype:
            return "defense"
        return "other"

    @classmethod
    def _ad_counts(cls, shots_dict: Dict[str, int]) -> Dict[str, int]:
        c = {"attack": 0, "defense": 0}
        for stype, n in (shots_dict or {}).items():
            b = cls._shot_bucket(stype)
            if b in c:
                c[b] += int(n)
        return c

    # ============================================================
    # Shots source
    # ============================================================
    @classmethod
    def _extract_player_shots_from_final(cls, player_obj: Dict[str, Any]) -> Dict[str, int]:
        if not isinstance(player_obj, dict):
            return {}

        out: Dict[str, int] = {}
        for k, v in player_obj.items():
            if k in ("id", "name", "total_shots"):
                continue
            if cls._is_intlike(v):
                key = str(k).strip()
                if key:
                    out[key] = int(v)
        return out

    @classmethod
    def _aggregate_total_shots(cls, p1: Dict[str, int], p2: Dict[str, int]) -> Dict[str, int]:
        total: Dict[str, int] = {}
        for d in (p1 or {}, p2 or {}):
            for k, v in d.items():
                total[k] = total.get(k, 0) + int(v)
        return total

    @classmethod
    def _try_final_shots_stats(cls, bundle: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        final_stats = bundle.get("final_shots_stats")
        if not isinstance(final_stats, dict):
            return None

        players = final_stats.get("players")
        if not isinstance(players, dict):
            return None

        p1_raw = players.get("player1")
        p2_raw = players.get("player2")
        if not isinstance(p1_raw, dict) or not isinstance(p2_raw, dict):
            return None

        try:
            p1_id = int(p1_raw.get("id", 1))
            p2_id = int(p2_raw.get("id", 2))
        except Exception:
            return None
        if p1_id != 1 or p2_id != 2:
            return None

        p1_name = str(p1_raw.get("name", "Player 1"))
        p2_name = str(p2_raw.get("name", "Player 2"))

        p1_total = p1_raw.get("total_shots")
        p2_total = p2_raw.get("total_shots")
        if not cls._is_intlike(p1_total) or not cls._is_intlike(p2_total):
            return None

        p1_shots = cls._extract_player_shots_from_final(p1_raw)
        p2_shots = cls._extract_player_shots_from_final(p2_raw)

        total_from_file = final_stats.get("total_shots")
        total_n = int(total_from_file) if cls._is_intlike(total_from_file) else (int(p1_total) + int(p2_total))
        total_dist = cls._aggregate_total_shots(p1_shots, p2_shots)

        return {
            "total_shots_n": int(total_n),
            "players": {
                1: {"id": 1, "name": p1_name, "total_shots": int(p1_total), "shots": p1_shots},
                2: {"id": 2, "name": p2_name, "total_shots": int(p2_total), "shots": p2_shots},
            },
            "total_distribution": total_dist,
        }

    # ============================================================
    # Shot events parsing
    # ============================================================
    @classmethod
    def _parse_shot_event(cls, ev: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(ev, dict):
            return None
        ps = ev.get("primary_shot")
        if not isinstance(ps, dict):
            return None

        pid = ps.get("player_id", None)
        if pid not in (1, 2):
            return None

        f = ps.get("frame_index", ev.get("frame_index", None))
        if f is None:
            return None
        try:
            f = int(f)
        except Exception:
            return None

        shot_type = str(ps.get("shot_type", "Unknown")).strip() or "Unknown"
        conf = cls._to_float(ps.get("confidence", ps.get("shot_confidence", 1.0)), 1.0)
        if not np.isfinite(conf):
            conf = 1.0

        return {"player_id": int(pid), "frame_index": int(f), "shot_type": shot_type, "confidence": float(conf)}

    @staticmethod
    def _dedupe_events_by_frame_best_conf(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        by_f: Dict[int, Dict[str, Any]] = {}
        for e in events:
            f = int(e["frame_index"])
            c = float(e.get("confidence", 1.0))
            if f not in by_f or c > float(by_f[f].get("confidence", 1.0)):
                by_f[f] = e
        return sorted(by_f.values(), key=lambda x: int(x["frame_index"]))

    @staticmethod
    def _find_closest_event_by_frame(events_sorted: List[Dict[str, Any]], target_frame: int) -> Optional[Dict[str, Any]]:
        if not events_sorted:
            return None
        frames = [int(e["frame_index"]) for e in events_sorted]
        import bisect

        i = bisect.bisect_left(frames, int(target_frame))
        cand = []
        if 0 <= i < len(events_sorted):
            cand.append(events_sorted[i])
        if 0 <= i - 1 < len(events_sorted):
            cand.append(events_sorted[i - 1])
        if not cand:
            return None
        cand.sort(key=lambda e: abs(int(e["frame_index"]) - int(target_frame)))
        return cand[0]

    # ============================================================
    # Rally duration
    # ============================================================
    @staticmethod
    def _parse_iso_datetime(x: Any) -> Optional[datetime]:
        if not x:
            return None
        try:
            return datetime.fromisoformat(str(x))
        except Exception:
            return None

    def _compute_rally_duration_s_datetime_only(self, players_seq: Any, frame_ids: List[int], fps: float) -> float:
        if isinstance(players_seq, list) and players_seq:
            first_dt: Optional[datetime] = None
            last_dt: Optional[datetime] = None
            for rec in players_seq:
                if not isinstance(rec, dict):
                    continue
                dt = self._parse_iso_datetime(rec.get("timestamp"))
                if dt is None:
                    continue
                if first_dt is None:
                    first_dt = dt
                last_dt = dt
            if first_dt is not None and last_dt is not None:
                return max(0.0, float((last_dt - first_dt).total_seconds()))

        fps = float(fps) if fps and fps > 1e-6 else 30.0
        if frame_ids and len(frame_ids) >= 2:
            return max(0.0, float(max(frame_ids) - min(frame_ids)) / fps)

        return 0.0

    # ============================================================
    # Bundle adapter (PUBLIC)
    # ============================================================
    def adapt_bundle(self, bundle: Dict[str, Any]) -> Dict[str, Any]:
        tracking_data: Dict[str, Any] = {}

        players_seq = bundle.get("players_final_metrics") or bundle.get("players_tracking") or []
        ball_seq = bundle.get("ball_final_metrics") or bundle.get("ball_tracking") or []
        shot_events = bundle.get("shot_events") or []

        # ============================================================
        # Extract player world positions (aligned by frame) + frame_ids
        # ============================================================
        frame_ids: List[int] = []
        timestamps: List[str] = []

        # Per-frame aligned WORLD positions (may contain None)
        p1_world_by_frame: Dict[int, Tuple[float, float]] = {}
        p2_world_by_frame: Dict[int, Tuple[float, float]] = {}

        if isinstance(players_seq, list):
            for rec in players_seq:
                if not isinstance(rec, dict):
                    continue

                frame = rec.get("frame")
                ts = rec.get("timestamp")

                try:
                    f = int(frame) if frame is not None else None
                except Exception:
                    f = None

                # We need a frame id to align everything; skip records without it
                if f is None:
                    continue

                frame_ids.append(f)
                timestamps.append(str(ts) if ts is not None else "")

                players = rec.get("players", {}) if isinstance(rec.get("players", {}), dict) else {}

                p1 = players.get("player1", {}) if isinstance(players.get("player1", {}), dict) else {}
                p2 = players.get("player2", {}) if isinstance(players.get("player2", {}), dict) else {}

                p1p = p1.get("position", {}) if isinstance(p1.get("position", {}), dict) else {}
                p2p = p2.get("position", {}) if isinstance(p2.get("position", {}), dict) else {}

                x1, y1 = self._to_float(p1p.get("x", np.nan)), self._to_float(p1p.get("y", np.nan))
                x2, y2 = self._to_float(p2p.get("x", np.nan)), self._to_float(p2p.get("y", np.nan))

                if np.isfinite(x1) and np.isfinite(y1):
                    p1_world_by_frame[f] = (float(x1), float(y1))
                if np.isfinite(x2) and np.isfinite(y2):
                    p2_world_by_frame[f] = (float(x2), float(y2))

        tracking_data["frame_ids"] = frame_ids
        tracking_data["timestamps"] = timestamps

        # ============================================================
        # Normalize world coords (meters) to pixel coords for court image
        # ============================================================
        court_w = int(self.config["streamlit_dashboard"].get("court_width", 385))
        court_h = int(self.config["streamlit_dashboard"].get("court_height", 840))

        half_w = float(COURT_WIDTH) / 2.0
        half_l = float(COURT_LENGTH) / 2.0

        def _world_to_px(pos: Optional[Tuple[float, float]]) -> Optional[Tuple[int, int]]:
            if pos is None:
                return None
            x, y = pos
            if not (np.isfinite(x) and np.isfinite(y)):
                return None

            x = float(np.clip(x, -half_w, half_w))
            y = float(np.clip(y, -half_l, half_l))

            nx = (x + half_w) / (2.0 * half_w)
            ny = (y + half_l) / (2.0 * half_l)

            px = int(round(nx * (court_w - 1)))
            py = int(round(ny * (court_h - 1)))
            return (px, py)

        # NEW: per-frame aligned pixel positions (for shots map)
        p1_pos_by_frame: List[Optional[Tuple[int, int]]] = []
        p2_pos_by_frame: List[Optional[Tuple[int, int]]] = []
        for f in frame_ids:
            p1_pos_by_frame.append(_world_to_px(p1_world_by_frame.get(int(f))))
            p2_pos_by_frame.append(_world_to_px(p2_world_by_frame.get(int(f))))

        tracking_data["p1_pos_by_frame"] = p1_pos_by_frame
        tracking_data["p2_pos_by_frame"] = p2_pos_by_frame
        tracking_data["frame_to_idx"] = {int(f): i for i, f in enumerate(frame_ids)}

        # Keep old compact lists used by positioning tab (unchanged behavior)
        tracking_data["player1_positions"] = [p for p in p1_pos_by_frame if p is not None]
        tracking_data["player2_positions"] = [p for p in p2_pos_by_frame if p is not None]

        # ============================================================
        # Extract ball metrics
        # ============================================================
        ball_speed_kmh: List[float] = []
        ball_speed_smooth_kmh: List[float] = []
        ball_frame_ids: List[int] = []

        if isinstance(ball_seq, list):
            for rec in ball_seq:
                if not isinstance(rec, dict):
                    continue
                frame = rec.get("frame", None)
                if frame is not None:
                    try:
                        ball_frame_ids.append(int(frame))
                    except Exception:
                        pass

                ball = rec.get("ball", {}) if isinstance(rec.get("ball", {}), dict) else {}
                ball_speed_kmh.append(self._to_float(ball.get("speed_kmh", np.nan), np.nan))
                ball_speed_smooth_kmh.append(self._to_float(ball.get("speed_smooth_kmh", np.nan), np.nan))

        tracking_data["ball_speed_kmh"] = ball_speed_kmh
        tracking_data["ball_speed_smooth_kmh"] = ball_speed_smooth_kmh
        tracking_data["ball_frame_ids"] = ball_frame_ids

        # ============================================================
        # Shots: prefer final_shots_stats
        # ============================================================
        final = self._try_final_shots_stats(bundle)
        if final is None:
            tracking_data["player1_shots"] = {}
            tracking_data["player2_shots"] = {}
            tracking_data["total_shots"] = {}
            tracking_data["shots_total_n"] = 0
            tracking_data["player_names"] = {1: "Player 1", 2: "Player 2"}
        else:
            tracking_data["player1_shots"] = final["players"][1]["shots"]
            tracking_data["player2_shots"] = final["players"][2]["shots"]
            tracking_data["total_shots"] = final["total_distribution"]
            tracking_data["shots_total_n"] = int(final["total_shots_n"])
            tracking_data["player_names"] = {1: final["players"][1]["name"], 2: final["players"][2]["name"]}

        # ============================================================
        # Per-player dynamic metrics
        # ============================================================
        t0: Optional[datetime] = None
        time_s: List[float] = []

        p_dyn = {
            1: {"speed_kmh": [], "distance_m": []},
            2: {"speed_kmh": [], "distance_m": []},
        }

        if isinstance(players_seq, list):
            for rec in players_seq:
                if not isinstance(rec, dict):
                    continue

                # NOTE: keep behavior consistent: dynamic metrics follow players_seq as before.
                ts = rec.get("timestamp")
                tsec = None
                if ts:
                    try:
                        dt = datetime.fromisoformat(str(ts))
                        if t0 is None:
                            t0 = dt
                        tsec = (dt - t0).total_seconds()
                    except Exception:
                        tsec = None

                if tsec is not None:
                    time_s.append(float(tsec))

                players = rec.get("players", {}) if isinstance(rec.get("players", {}), dict) else {}

                for pid, key in [(1, "player1"), (2, "player2")]:
                    p = players.get(key, {}) if isinstance(players.get(key, {}), dict) else {}
                    vel = p.get("velocity", {}) if isinstance(p.get("velocity", {}), dict) else {}

                    speed_norm = self._to_float(vel.get("norm", np.nan), np.nan)
                    dist_m = self._to_float(p.get("distance", np.nan), np.nan)

                    speed_kmh = float(speed_norm) if np.isfinite(speed_norm) else np.nan
                    p_dyn[pid]["speed_kmh"].append(speed_kmh if np.isfinite(speed_kmh) else np.nan)
                    p_dyn[pid]["distance_m"].append(float(dist_m) if np.isfinite(dist_m) else np.nan)

        def _nanmax(arr: List[float]) -> float:
            a = np.array(arr, dtype=float)
            if a.size == 0:
                return 0.0
            m = np.nanmax(a)
            return float(m) if np.isfinite(m) else 0.0

        def _last_finite(arr: List[float]) -> float:
            for v in reversed(arr):
                if np.isfinite(v):
                    return float(v)
            return 0.0

        def _nanmean(arr: List[float]) -> float:
            a = np.array(arr, dtype=float)
            if a.size == 0:
                return 0.0
            m = np.nanmean(a)
            return float(m) if np.isfinite(m) else 0.0

        def _high_speed_stats(speed_series: List[float], thr_kmh: float) -> Tuple[float, int, int]:
            a = np.array(speed_series, dtype=float)
            valid = np.isfinite(a)
            valid_n = int(valid.sum())
            if valid_n == 0:
                return 0.0, 0, 0
            high_n = int((a[valid] > float(thr_kmh)).sum())
            pct = 100.0 * high_n / max(valid_n, 1)
            return float(pct), int(high_n), int(valid_n)

        tracking_data["time_s"] = time_s

        thr_kmh = float(self.config.get("streamlit_dashboard", {}).get("high_speed_threshold_kmh", 12.0) or 12.0)
        fps = float(self.config.get("video_fps", self.config.get("fps", 30.0)) or 30.0)
        fps = fps if fps > 1e-6 else 30.0

        p1_high_pct, p1_high_n, p1_valid_n = _high_speed_stats(p_dyn[1]["speed_kmh"], thr_kmh)
        p2_high_pct, p2_high_n, p2_valid_n = _high_speed_stats(p_dyn[2]["speed_kmh"], thr_kmh)

        tracking_data["player_dyn"] = {
            1: {
                "max_speed_kmh": _nanmax(p_dyn[1]["speed_kmh"]),
                "avg_speed_kmh": _nanmean(p_dyn[1]["speed_kmh"]),
                "distance_m": _last_finite(p_dyn[1]["distance_m"]),
                "speed_series_kmh": [float(v) if np.isfinite(v) else None for v in p_dyn[1]["speed_kmh"]],
                "high_speed_thr_kmh": float(thr_kmh),
                "high_speed_pct": float(p1_high_pct),
                "high_speed_frames": int(p1_high_n),
                "valid_speed_frames": int(p1_valid_n),
                "high_speed_time_s": float(p1_high_n / fps) if fps > 1e-6 else 0.0,
            },
            2: {
                "max_speed_kmh": _nanmax(p_dyn[2]["speed_kmh"]),
                "avg_speed_kmh": _nanmean(p_dyn[2]["speed_kmh"]),
                "distance_m": _last_finite(p_dyn[2]["distance_m"]),
                "speed_series_kmh": [float(v) if np.isfinite(v) else None for v in p_dyn[2]["speed_kmh"]],
                "high_speed_thr_kmh": float(thr_kmh),
                "high_speed_pct": float(p2_high_pct),
                "high_speed_frames": int(p2_high_n),
                "valid_speed_frames": int(p2_valid_n),
                "high_speed_time_s": float(p2_high_n / fps) if fps > 1e-6 else 0.0,
            },
        }

        # ============================================================
        # Keep shot_events normalized for peak-shot lookup
        # ============================================================
        parsed_events: List[Dict[str, Any]] = []
        if isinstance(shot_events, list) and shot_events:
            for ev in shot_events:
                pe = self._parse_shot_event(ev)
                if pe is not None:
                    parsed_events.append(pe)
        tracking_data["shot_events_parsed"] = self._dedupe_events_by_frame_best_conf(parsed_events)

        # ============================================================
        # Rally summary
        # ============================================================
        duration_s = self._compute_rally_duration_s_datetime_only(players_seq=players_seq, frame_ids=frame_ids, fps=fps)

        shots_total = int(tracking_data.get("shots_total_n", 0))
        shot_rate = (shots_total / duration_s) if duration_s > 1e-6 else 0.0

        tracking_data["rally_summary"] = {
            "shots": shots_total,
            "duration_s": float(duration_s),
            "shot_rate": float(shot_rate),
        }

        return tracking_data

    # ============================================================
    # Pro Attack/Defense
    # ============================================================
    def render_attack_defense_profile(
        self,
        shots: Dict[str, int],
        title: str = "ATTACK vs DEFENSE",
        use_columns: bool = True,   # ✅ new
    ):
        ad = self._ad_counts(shots or {})
        A, D = int(ad["attack"]), int(ad["defense"])
        denom = A + D

        a_pct = 100.0 * A / denom if denom > 0 else 0.0
        d_pct = 100.0 * D / denom if denom > 0 else 0.0

        st.markdown(
            f"""
            <div style='text-align:center; font-weight:800; color: var(--primary-color); margin-top:10px; line-height:1.05;'>
                {title}
            </div>
            <div style='text-align:center; font-size:0.82rem; color:#bdbdbd; margin-top:4px;'>
                Attack = Smash &nbsp;•&nbsp; Defense = Lift/Clear &nbsp;•&nbsp; Other excluded
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ✅ SAFE: if we're inside another column, don't create nested columns
        if use_columns:
            c1, c2, c3 = st.columns([1, 1, 1.2], gap="small")
            with c1:
                st.metric("Attack", f"{A}/{denom}" if denom else f"{A}/0", f"{a_pct:.0f}%")
            with c2:
                st.metric("Defense", f"{D}/{denom}" if denom else f"{D}/0", f"{d_pct:.0f}%")
            with c3:
                st.metric("Attack share", f"{a_pct:.0f}%", f"A:D = {A}:{D}")
        else:
            # Compact + readable
            st.markdown(
                f"<div style='text-align:center; color:#d0d0d0; margin-top:6px;'>"
                f"<b>Attack</b> {A} ({a_pct:.0f}%) &nbsp;|&nbsp; "
                f"<b>Defense</b> {D} ({d_pct:.0f}%) &nbsp;|&nbsp; "
                f"<b>A:D</b> {A}:{D}"
                f"</div>",
                unsafe_allow_html=True,
            )

        fig = go.Figure()
        fig.add_trace(go.Bar(name="Attack", x=[A], y=[""], orientation="h", hovertemplate="Attack: %{x}<extra></extra>"))
        fig.add_trace(go.Bar(name="Defense", x=[D], y=[""], orientation="h", hovertemplate="Defense: %{x}<extra></extra>"))
        fig.update_layout(
            barmode="stack",
            height=85,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ============================================================
    # Player card
    # ============================================================
    def display_player_card(
        self,
        player_name: str,
        static_info: Dict[str, Any],
        dynamic_info: Dict[str, Any],
        shots: Dict[str, int],
        time_s: List[float],
    ):
        photo = static_info.get("photo", "")
        nationality = static_info.get("nationality", "—")
        age = static_info.get("age", "—")
        height = static_info.get("height", "—")
        world_rank = static_info.get("world_rank", "—")

        max_speed_kmh = dynamic_info.get("max_speed_kmh", None)
        avg_speed_kmh = dynamic_info.get("avg_speed_kmh", None)
        distance_m = dynamic_info.get("distance_m", None)
        speed_series_kmh = dynamic_info.get("speed_series_kmh", []) or []

        high_speed_pct = dynamic_info.get("high_speed_pct", None)
        high_speed_time_s = dynamic_info.get("high_speed_time_s", None)
        high_speed_thr_kmh = dynamic_info.get("high_speed_thr_kmh", 12.0)

        def fmt(v, unit="", dec=1):
            try:
                v = float(v)
                if not np.isfinite(v):
                    return "—"
                return f"{v:.{dec}f}{unit}"
            except Exception:
                return "—"

        img_b64 = self.image_to_base64(photo)

        with st.container():
            st.markdown('<div class="player-card">', unsafe_allow_html=True)

            st.markdown(
                """
                <div class="player-header">
                    <img class="player-image" src="data:image/png;base64,{image}" alt="{name}">
                    <div style="flex-grow: 1;">
                        <div class="player-name">{name}</div>
                        <div class="player-info">
                            <strong>Nationality:</strong> {nationality}<br>
                            <strong>Age:</strong> {age}<br>
                            <strong>Height:</strong> {height}<br>
                            <strong>World Rank:</strong> {world_rank}
                        </div>
                    </div>
                </div>
                """.format(
                    image=img_b64,
                    name=player_name,
                    nationality=nationality,
                    age=age,
                    height=height,
                    world_rank=world_rank,
                ),
                unsafe_allow_html=True,
            )

            delta_speed = None
            try:
                if max_speed_kmh is not None and avg_speed_kmh is not None:
                    mx = float(max_speed_kmh)
                    av = float(avg_speed_kmh)
                    if np.isfinite(mx) and np.isfinite(av):
                        delta_speed = f"+{(mx - av):.1f} km/h vs avg"
            except Exception:
                delta_speed = None

            st.markdown(self.create_metric("Max. Speed", fmt(max_speed_kmh, " km/h", 1), delta=delta_speed), unsafe_allow_html=True)
            st.markdown(self.create_metric("Avg. Speed", fmt(avg_speed_kmh, " km/h", 1)), unsafe_allow_html=True)
            st.markdown(self.create_metric("Distance Covered", fmt(distance_m, " m", 0)), unsafe_allow_html=True)

            delta_wr = None
            try:
                thr = float(high_speed_thr_kmh)
                if high_speed_time_s is not None and np.isfinite(float(high_speed_time_s)):
                    delta_wr = f">{thr:.0f} km/h • {float(high_speed_time_s):.1f}s"
                else:
                    delta_wr = f">{thr:.0f} km/h"
            except Exception:
                delta_wr = None

            hs_val = "—"
            try:
                if high_speed_pct is not None and np.isfinite(float(high_speed_pct)):
                    hs_val = f"{float(high_speed_pct):.1f}%"
            except Exception:
                hs_val = "—"

            st.markdown(self.create_metric("High-Intensity Movement", hs_val, delta=delta_wr), unsafe_allow_html=True)

            chart_type = st.selectbox(
                f"{player_name.split()[0]} STATS",
                ["Speed Trend", "Shot Distribution"],
                key=f"{player_name}_chart",
            )

            if chart_type == "Speed Trend":
                y = [v if v is not None else np.nan for v in speed_series_kmh]
                if time_s and len(time_s) == len(y):
                    fig = px.line(x=time_s, y=y, title=f"{player_name.split()[0]} Speed Over Time (km/h)")
                    fig.update_xaxes(title_text="Time (s)")
                else:
                    fig = px.line(y=y, title=f"{player_name.split()[0]} Speed Over Time (km/h)")
                    fig.update_xaxes(title_text="Samples")

                fig.update_yaxes(title_text="Speed (km/h)")
                fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white")
                st.plotly_chart(fig, use_container_width=True)

            else:
                if shots:
                    fig = px.pie(values=list(shots.values()), names=list(shots.keys()), title=f"{player_name.split()[0]} Shot Type", hole=0.35)
                else:
                    fig = px.pie(values=[1], names=["No shots"], title=f"{player_name.split()[0]} Shot Type", hole=0.35)

                fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white")
                st.plotly_chart(fig, use_container_width=True)

                self.render_attack_defense_profile(shots, title="Attack vs Defense")

            st.markdown("</div>", unsafe_allow_html=True)

    # ============================================================
    # Ball card
    # ============================================================
    def display_ball_stats(self, tracking_data: Dict[str, Any]):
        smooth = tracking_data.get("ball_speed_smooth_kmh", []) or []
        raw = tracking_data.get("ball_speed_kmh", []) or []
        time_s = tracking_data.get("time_s", []) or []
        ball_frames = tracking_data.get("ball_frame_ids", []) or []
        shot_events = tracking_data.get("shot_events_parsed", []) or []
        player_names = tracking_data.get("player_names", {1: "Player 1", 2: "Player 2"}) or {1: "Player 1", 2: "Player 2"}

        fps = float(self.config.get("video_fps", self.config.get("fps", 30.0)) or 30.0)
        fps = fps if fps > 1e-6 else 30.0

        series = np.array(smooth, dtype=float)
        if series.size == 0 or not np.isfinite(np.nanmax(series)):
            series = np.array(raw, dtype=float)

        if time_s and len(time_s) == len(series):
            ball_t = np.array(time_s, dtype=float)
        elif ball_frames and len(ball_frames) == len(series):
            f0 = float(min(ball_frames)) if ball_frames else 0.0
            ball_t = (np.array(ball_frames, dtype=float) - f0) / fps
        else:
            ball_t = np.arange(series.size, dtype=float) / fps if series.size else np.array([], dtype=float)

        mask = np.isfinite(series) & np.isfinite(ball_t) & (ball_t > 1.0)
        series_eff = series[mask] if series.size else np.array([], dtype=float)
        t_eff = ball_t[mask] if ball_t.size else np.array([], dtype=float)
        frames_eff: Optional[np.ndarray] = None
        if ball_frames and len(ball_frames) == len(series):
            frames_eff = np.array(ball_frames, dtype=int)[mask]
        else:
            frames_eff = None

        if series_eff.size == 0 or not np.isfinite(np.nanmax(series_eff)):
            series_eff = series[np.isfinite(series)]
            t_eff = ball_t[np.isfinite(series)]
            if ball_frames and len(ball_frames) == len(series):
                frames_eff = np.array(ball_frames, dtype=int)[np.isfinite(series)]
            else:
                frames_eff = None

        max_speed = float(np.nanmax(series_eff)) if series_eff.size else 0.0
        avg_speed = float(np.nanmean(series_eff)) if series_eff.size else 0.0
        delta = (max_speed - avg_speed) if np.isfinite(max_speed) and np.isfinite(avg_speed) else 0.0

        max_idx_eff = int(np.nanargmax(series_eff)) if series_eff.size else -1
        max_time = float(t_eff[max_idx_eff]) if (max_idx_eff >= 0 and t_eff.size) else None

        max_frame: Optional[int] = None
        if frames_eff is not None and max_idx_eff >= 0 and frames_eff.size:
            max_frame = int(frames_eff[max_idx_eff])
        elif ball_frames and len(ball_frames) == len(series) and series.size:
            gi = int(np.nanargmax(series)) if np.isfinite(np.nanmax(series)) else -1
            if gi >= 0:
                max_frame = int(ball_frames[gi])

        strongest_shot_summary = None
        if max_frame is not None and shot_events:
            closest = self._find_closest_event_by_frame(shot_events, max_frame)
            if closest:
                pid = int(closest.get("player_id", 0)) if closest.get("player_id") in (1, 2) else 0
                p_name = player_names.get(pid, f"Player {pid}" if pid else "Unknown")
                shot_type = str(closest.get("shot_type", "Unknown")).strip() or "Unknown"
                shot_frame = int(closest.get("frame_index", -1))

                shot_time = None
                if ball_frames and len(ball_frames) == len(series):
                    f0 = min(ball_frames) if ball_frames else 0
                    shot_time = (shot_frame - f0) / fps
                elif time_s and len(time_s) == len(series) and max_frame is not None:
                    shot_time = max_time

                strongest_shot_summary = {
                    "player": p_name,
                    "shot_type": shot_type,
                    "shot_frame": shot_frame,
                    "shot_time_s": float(shot_time) if (shot_time is not None and np.isfinite(shot_time)) else None,
                    "peak_speed_kmh": float(max_speed) if np.isfinite(max_speed) else None,
                    "peak_time_s": float(max_time) if (max_time is not None and np.isfinite(max_time)) else None,
                    "peak_frame": int(max_frame),
                }

        total_shots = tracking_data.get("total_shots", {}) or {}
        shots_total_n = int(tracking_data.get("shots_total_n", 0))

        with st.container():
            st.markdown('<div class="ball-stats">', unsafe_allow_html=True)

            st.markdown(self.create_metric("Max Shuttle Speed", f"{max_speed:.1f} km/h", delta=f"+{delta:.1f} km/h vs avg"), unsafe_allow_html=True)
            st.markdown(self.create_metric("Avg. Shuttle Speed", f"{avg_speed:.1f} km/h"), unsafe_allow_html=True)
            st.markdown(self.create_metric("Shots in Rally", f"{shots_total_n}"), unsafe_allow_html=True)

            if series_eff.size and t_eff.size:
                y = [float(v) if np.isfinite(v) else np.nan for v in series_eff.tolist()]
                x = [float(v) if np.isfinite(v) else np.nan for v in t_eff.tolist()]

                fig = px.line(x=x, y=y, title="Shuttle Speed (km/h)")
                fig.update_xaxes(title_text="Time (s)")
                fig.update_yaxes(title_text="Speed (km/h)")
                fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white")
                st.plotly_chart(fig, use_container_width=True)

            if total_shots:
                pie = px.pie(values=list(total_shots.values()), names=list(total_shots.keys()), title="TOTAL Shot Type", hole=0.35)
            else:
                pie = px.pie(values=[1], names=["No shots"], title="Total Shot Type", hole=0.35)

            pie.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white")
            st.plotly_chart(pie, use_container_width=True)

            st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)
            st.markdown(
                "<div style='text-align:center; font-weight:800; color: var(--primary-color); margin-top:6px;'>STRONGEST SHOT (Peak Speed)</div>",
                unsafe_allow_html=True,
            )

            if strongest_shot_summary and strongest_shot_summary.get("peak_speed_kmh") is not None:
                p = strongest_shot_summary["player"]
                stype = strongest_shot_summary["shot_type"]
                peak = strongest_shot_summary["peak_speed_kmh"]
                peak_t = strongest_shot_summary.get("peak_time_s")
                peak_f = strongest_shot_summary.get("peak_frame")

                shot_t = strongest_shot_summary.get("shot_time_s")
                shot_f = strongest_shot_summary.get("shot_frame")

                c1, c2 = st.columns(2, gap="small")
                with c1:
                    st.metric("Peak Speed", f"{peak:.1f} km/h", self._fmt_time_mmss(peak_t))
                with c2:
                    st.metric("Attributed to", p, stype)

                st.caption(
                    f"Closest shot-event match at frame {shot_f} (peak frame {peak_f}). "
                    + (f"Shot time ≈ {shot_t:.2f}s." if (shot_t is not None and np.isfinite(shot_t)) else "")
                )
            else:
                st.info("No shot-events available to attribute the peak speed to a player/shot type.")

            st.markdown("</div>", unsafe_allow_html=True)

    # ============================================================
    # Helpers
    # ============================================================
    def create_metric(self, label, value, delta=None):
        return f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {f'<div style="font-size: 0.9rem; color: #aaaaaa;">{delta}</div>' if delta else ''}
        </div>
        """

    def image_to_base64(self, image_path: str) -> str:
        try:
            if not image_path or not os.path.exists(image_path):
                return ""
            with open(image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode("utf-8")
        except Exception:
            return ""

    def load_and_resize_image(
        self,
        image_path: str,
        target_width: Optional[int] = None,
        target_height: Optional[int] = None,
        mode: str = "cover",
        allow_upscale: bool = True,
    ) -> Optional[Image.Image]:
        try:
            img = Image.open(image_path).convert("RGBA")
            w, h = img.size

            if target_width is None and target_height is None:
                return img

            if target_width is None and target_height is not None:
                scale = target_height / h
                if not allow_upscale:
                    scale = min(scale, 1.0)
                new_w = max(1, int(round(w * scale)))
                new_h = max(1, int(round(h * scale)))
                return img.resize((new_w, new_h), Image.LANCZOS)

            if target_height is None and target_width is not None:
                scale = target_width / w
                if not allow_upscale:
                    scale = min(scale, 1.0)
                new_w = max(1, int(round(w * scale)))
                new_h = max(1, int(round(h * scale)))
                return img.resize((new_w, new_h), Image.LANCZOS)

            tw, th = int(target_width), int(target_height)

            if mode == "stretch":
                return img.resize((tw, th), Image.LANCZOS)

            if mode == "fit":
                scale = min(tw / w, th / h)
                if not allow_upscale:
                    scale = min(scale, 1.0)
                new_w = max(1, int(round(w * scale)))
                new_h = max(1, int(round(h * scale)))
                return img.resize((new_w, new_h), Image.LANCZOS)

            if mode == "cover":
                scale = max(tw / w, th / h)
                if not allow_upscale:
                    scale = min(scale, 1.0)
                rw = max(1, int(round(w * scale)))
                rh = max(1, int(round(h * scale)))
                resized = img.resize((rw, rh), Image.LANCZOS)

                left = (rw - tw) // 2
                top = (rh - th) // 2
                return resized.crop((left, top, left + tw, top + th))

            raise ValueError(f"Unknown mode: {mode}")

        except Exception as e:
            st.error(f"Error loading image {image_path}: {e}")
            return None

    def create_heatmap(
        self,
        court_img: Image.Image,
        p1_positions: List[Tuple[int, int]],
        p2_positions: List[Tuple[int, int]],
        bins: Tuple[int, int] = (32, 64),
        sigma_bins: float = 2.0,
        pixel_blur: float = 6.0,
        alpha_max: int = 235,
        p1_cmap_name: str = "YlOrRd",
        p2_cmap_name: str = "Blues",
        norm_percentile: float = 98.0,
        gamma: float = 0.65,
        floor: float = 0.03,
        blend_mode: str = "screen",
    ) -> Image.Image:
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
            H, _, _ = np.histogram2d(
                x, y,
                bins=[xbins, ybins],
                range=[[0, w], [0, h]],
            )
            return H

        H1 = _hist(x1, y1)
        H2 = _hist(x2, y2)

        if np.any(H1):
            H1 = gaussian_filter(H1, sigma=float(sigma_bins))
        if np.any(H2):
            H2 = gaussian_filter(H2, sigma=float(sigma_bins))

        def _robust_norm(A: np.ndarray) -> np.ndarray:
            if not np.any(A):
                return A * 0.0
            p = np.percentile(A[A > 0], norm_percentile) if np.any(A > 0) else 0.0
            denom = float(max(p, 1e-12))
            N = np.clip(A / denom, 0.0, 1.0)
            return N

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

        def _to_overlay(N: np.ndarray, cmap_name: str) -> Image.Image:
            if not np.any(N):
                return Image.new("RGBA", (w, h), (0, 0, 0, 0))

            cmap = cm.get_cmap(cmap_name)
            rgba = cmap(N)
            rgba[..., 3] = np.clip(N, 0.0, 1.0) * (float(alpha_max) / 255.0)

            rgba8 = (np.clip(rgba, 0.0, 1.0) * 255.0).astype(np.uint8)
            rgba8_img = np.transpose(rgba8, (1, 0, 2))

            small = Image.fromarray(rgba8_img, mode="RGBA")
            overlay = small.resize((w, h), resample=Image.BILINEAR)

            if pixel_blur and pixel_blur > 0:
                overlay = overlay.filter(ImageFilter.GaussianBlur(radius=float(pixel_blur)))

            return overlay

        o1 = _to_overlay(N1, p1_cmap_name)
        o2 = _to_overlay(N2, p2_cmap_name)

        if blend_mode == "add":
            combined = ImageChops.add(o1, o2)
            return Image.alpha_composite(base, combined)

        def _screen_blend(a: Image.Image, b: Image.Image) -> Image.Image:
            A = np.asarray(a).astype(np.float32) / 255.0
            B = np.asarray(b).astype(np.float32) / 255.0
            rgb = 1.0 - (1.0 - A[..., :3]) * (1.0 - B[..., :3])
            alpha = np.maximum(A[..., 3], B[..., 3])[..., None]
            out = np.concatenate([rgb, alpha], axis=-1)
            return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8), mode="RGBA")

        combined = _screen_blend(o1, o2)
        return Image.alpha_composite(base, combined)

    def show_player_legend(self):
        st.markdown(
            """
            <div style="background: var(--secondary-color); padding: 5px; border-radius: 8px; margin-top: 2px;margin-bottom: -8px;">
                <div style="display: flex; align-items: center; margin-bottom: 2px;">
                    <div style="width: 6px; height: 8px; background: #ffa500; border-radius: 50%; margin-right: 4px;"></div>
                    <span style="font-weight: bold;">Player 1</span>
                </div>
                <div style="display: flex; align-items: center;">
                    <div style="width: 6px; height: 8px; background: #0064ff; border-radius: 50%; margin-right: 4px;"></div>
                    <span style="font-weight: bold;">Player 2</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    def annotate_court(self, court_img, p1_pos, p2_pos):
        draw = ImageDraw.Draw(court_img)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except Exception:
            font = ImageFont.load_default()

        draw.ellipse([(p1_pos[0] - 10, p1_pos[1] - 10), (p1_pos[0] + 10, p1_pos[1] + 10)], fill="#ffa500", outline="white")
        draw.text((p1_pos[0] + 15, p1_pos[1] - 5), "P1", fill="white", font=font)

        draw.ellipse([(p2_pos[0] - 10, p2_pos[1] - 10), (p2_pos[0] + 10, p2_pos[1] + 10)], fill="#0064ff", outline="white")
        draw.text((p2_pos[0] + 15, p2_pos[1] - 5), "P2", fill="white", font=font)

        return court_img

    def calculate_coverage(self, positions: List[Tuple[float, float]], bin_size: int = 20, court_width: int = 380, court_height: int = 530) -> float:
        if bin_size <= 0:
            raise ValueError("bin_size must be positive")
        if court_width <= 0 or court_height <= 0:
            raise ValueError("Court dimensions must be positive")
        if not positions:
            return 0.0

        x_bins = max(1, court_width // bin_size)
        y_bins = max(1, court_height // bin_size)
        total_zones = x_bins * y_bins

        unique_zones = set()
        for x, y in positions:
            x_clamped = max(0, min(x, court_width - 1))
            y_clamped = max(0, min(y, court_height - 1))
            x_bin = min(int(x_clamped / bin_size), x_bins - 1)
            y_bin = min(int(y_clamped / bin_size), y_bins - 1)
            unique_zones.add((x_bin, y_bin))

        return len(unique_zones) / total_zones

    def calculate_base_position(self, positions):
        if not positions:
            return (0, 0)
        x = np.median([p[0] for p in positions])
        y = np.median([p[1] for p in positions])
        return (x, y)

    def calculate_movement_efficiency(self, positions):
        if len(positions) < 2:
            return 0.0
        total_distance = sum(
            np.sqrt((positions[i][0] - positions[i - 1][0]) ** 2 + (positions[i][1] - positions[i - 1][1]) ** 2)
            for i in range(1, len(positions))
        )
        optimal_distance = np.sqrt((positions[-1][0] - positions[0][0]) ** 2 + (positions[-1][1] - positions[0][1]) ** 2)
        return min(100, (optimal_distance / total_distance) * 100) if total_distance > 0 else 0.0

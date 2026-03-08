import cv2
import numpy as np
import pandas as pd
import os
import logging
from datetime import datetime
from typing import Dict, Tuple, List, Optional, Any

from tracking import Player, Ball, Keypoints
from .metrics import Metrics
from .sidecourt import SideCourt
from utils import save_json, get_logger

logger = get_logger(__name__, log_file="logs/dashboard.log", level=logging.DEBUG)


class Dashboard:
    def __init__(self, width: int, height: int, fps: int, court_keypoints: Keypoints, config: Dict):
        """
        Initialize the dashboard with all visualization panels.
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.config = config

        self.alpha = config["dashboard"].get("alpha", 0.7)

        self.player_info = {
            1: {
                "id": config["player_mapping"]["player_1"]["id"],
                "name": config["player_mapping"]["player_1"]["name"],
                "color": config["dashboard"]["charts_panel"]["bar_colors"]["player1"],
            },
            2: {
                "id": config["player_mapping"]["player_2"]["id"],
                "name": config["player_mapping"]["player_2"]["name"],
                "color": config["dashboard"]["charts_panel"]["bar_colors"]["player2"],
            },
        }

        # --- Shot counter state (original) ---
        self.shot_count_total = 0
        self.shot_count_p1 = 0
        self.shot_count_p2 = 0
        self._last_counted_shot_frame: Optional[int] = None  # debouncing key

        # --- Turn-based shot counting state (original) ---
        self._turn_player: Optional[int] = None
        self._turn_best_event: Optional[Dict[str, Any]] = None
        self._turn_best_conf: float = -1.0

        # --- Shot type stats (final JSON) ---
        # Counts are applied ONLY when a shot is counted (i.e., when a turn is finalized).
        self.shot_type_counts = {
            1: {"Smash": 0, "Lift": 0, "Clear": 0, "Drop": 0, "Unknown": 0},
            2: {"Smash": 0, "Lift": 0, "Clear": 0, "Drop": 0, "Unknown": 0},
        }

        self.metrics_panel = MetricsPanel(
            width,
            height,
            config["dashboard"]["metrics_panel"],
            self.alpha,
            player_info=self.player_info,
        )

        self.shot_panel = ShotTypePanel(
            width,
            height,
            fps,
            config["dashboard"]["shot_panel"],
            self.alpha,
            self.player_info,
        )

        self.shot_counter_panel = ShotCounterPanel(
            width,
            height,
            config["dashboard"].get("shot_counter_panel", {}),
            self.alpha,
            self.player_info,
        )

        self.metrics = Metrics()

        self.side_court = SideCourt(
            width=width,
            height=height,
            position=str(config["side_court"]["position"]),
            scale_factor=config["side_court"].get("scale_factor", 1.0),
            alpha=config["side_court"]["alpha"],
        )

        # Static homography once
        self.side_court.H = self.side_court.homography_matrix(court_keypoints)

        # Metrics buffers
        self.all_player_metrics: List[Dict[str, Any]] = []
        self.all_ball_metrics: List[Dict[str, Any]] = []

        # Output data directory
        self.output_dir = config["dashboard"]["dashboard_output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)

        self.frame_counter = 0

    def draw(
        self,
        frame: np.ndarray,
        players_detection: Optional[List[Player]],
        ball_detection: Optional[Ball],
        shot_data: Optional[Dict[str, Any]] = None,
        save_metrics: bool = False,
    ) -> np.ndarray:
        """
        Draw the dashboard overlay on the frame (frame-accurate metrics).
        """
        if shot_data is None:
            shot_data = {}

        overlay = frame.copy()

        try:
            self.metrics.begin_frame(self.frame_counter)

            if players_detection:
                self.side_court.update_players_position(players_detection, self.metrics)

            if ball_detection:
                self.side_court.update_ball_position(ball_detection, self.metrics)

            self.metrics.end_frame()
            self.frame_counter += 1

            metrics_df = self.metrics.into_dataframe(self.fps)
            realtime_metrics = metrics_df.iloc[-1] if not metrics_df.empty else pd.Series(dtype=float)

            # Original turn-based counting (no consecutive-frame gating)
            self._update_shot_counter(shot_data)

            # Draw panels
            self.shot_counter_panel.draw(
                overlay,
                total_shots=self.shot_count_total,
                p1_shots=self.shot_count_p1,
                p2_shots=self.shot_count_p2,
            )

            self.metrics_panel.draw(overlay, realtime_metrics)
            self.shot_panel.draw(overlay, shot_data, realtime_metrics)

            if save_metrics:
                self._collect_metrics()

        except Exception as e:
            logger.error(f"Error in dashboard draw: {e}", exc_info=True)

        return cv2.addWeighted(overlay, self.alpha, frame, 1 - self.alpha, 0)

    def _collect_metrics(self) -> None:
        try:
            player_metrics = self._prepare_players_metrics()
            if player_metrics:
                self.all_player_metrics.append(player_metrics)

            ball_metrics = self._prepare_ball_metrics()
            if ball_metrics:
                self.all_ball_metrics.append(ball_metrics)

        except Exception as e:
            logger.error(f"Error collecting metrics: {e}", exc_info=True)

    def save_final_metrics(self) -> None:
        """
        Save:
          - players_final_metrics.json
          - ball_final_metrics.json
          - final_shots_stats.json
        NOTE: This keeps the ORIGINAL counting behavior (no explicit end-of-video finalize).
        """
        try:
            if self.all_player_metrics:
                players_path = os.path.join(self.output_dir, "players_final_metrics.json")
                save_json(self.all_player_metrics, players_path)
                logger.info(f"Players metrics saved to {players_path}")

            if self.all_ball_metrics:
                ball_path = os.path.join(self.output_dir, "ball_final_metrics.json")
                save_json(self.all_ball_metrics, ball_path)
                logger.info(f"Ball metrics saved to {ball_path}")

            # --- Save final shots stats (clean structure as requested) ---
            shots_stats_path = os.path.join(self.output_dir, "final_shots_stats.json")

            def _shot_counts(pid: int) -> Dict[str, int]:
                c = self.shot_type_counts.get(pid, {})
                return {
                    "Smash": int(c.get("Smash", 0)),
                    "Lift": int(c.get("Lift", 0)),
                    "Clear": int(c.get("Clear", 0)),
                    "Drop": int(c.get("Drop", 0)),
                    "Unknown": int(c.get("Unknown", 0)),
                }

            payload = {
                "timestamp": datetime.now().isoformat(),
                "total_shots": int(self.shot_count_total),
                "players": {
                    "player1": {
                        "id": 1,
                        "name": self.player_info[1]["name"],
                        "total_shots": int(self.shot_count_p1),
                        **_shot_counts(1),
                    },
                    "player2": {
                        "id": 2,
                        "name": self.player_info[2]["name"],
                        "total_shots": int(self.shot_count_p2),
                        **_shot_counts(2),
                    },
                },
            }

            save_json(payload, shots_stats_path)
            logger.info(f"Final shots stats saved to {shots_stats_path}")

        except Exception as e:
            logger.error(f"Error saving final metrics: {e}", exc_info=True)

    def _prepare_players_metrics(self) -> Dict[str, Any]:
        df = self.metrics.into_dataframe(self.fps)
        if df.empty:
            return {}

        last = df.iloc[-1].to_dict()
        return {
            "timestamp": datetime.now().isoformat(),
            "frame": int(last.get("frame", -1)),
            "players": {
                "player1": {
                    "id": self.player_info[1]["id"],
                    "name": self.player_info[1]["name"],
                    "position": {"x": last.get("player1_x"), "y": last.get("player1_y")},
                    "velocity": {
                        "x": last.get("player1_Vx1"),
                        "y": last.get("player1_Vy1"),
                        "norm": last.get("player1_speed_kmh_smooth"),
                    },
                    "acceleration": {
                        "x": last.get("player1_Ax1"),
                        "y": last.get("player1_Ay1"),
                        "norm": last.get("player1_acc_ms2_smooth"),
                    },
                    "distance": last.get("player1_distance"),
                },
                "player2": {
                    "id": self.player_info[2]["id"],
                    "name": self.player_info[2]["name"],
                    "position": {"x": last.get("player2_x"), "y": last.get("player2_y")},
                    "velocity": {
                        "x": last.get("player2_Vx1"),
                        "y": last.get("player2_Vy1"),
                        "norm": last.get("player2_speed_kmh_smooth"),
                    },
                    "acceleration": {
                        "x": last.get("player2_Ax1"),
                        "y": last.get("player2_Ay1"),
                        "norm": last.get("player2_acc_ms2_smooth"),
                    },
                    "distance": last.get("player2_distance"),
                },
            },
        }

    def _prepare_ball_metrics(self) -> Dict[str, Any]:
        df = self.metrics.into_dataframe(self.fps)
        if df.empty:
            return {}

        last = df.iloc[-1].to_dict()
        return {
            "timestamp": datetime.now().isoformat(),
            "frame": int(last.get("frame", -1)),
            "ball": {
                "position": {"x": last.get("ball_x"), "y": last.get("ball_y")},
                "velocity": {"x": last.get("ball_Vx1"), "y": last.get("ball_Vy1"), "norm": last.get("ball_velocity1")},
                "speed_kmh": last.get("ball_speed_kmh"),
                "speed_smooth_kmh": last.get("ball_speed_smooth_kmh"),
                "shot_power": last.get("ball_shot_power1"),
            },
        }

    # ----------------------------
    # Shot counting + stats
    # ----------------------------
    @staticmethod
    def _normalize_shot_type(raw: Any) -> str:
        """
        Normalize incoming shot labels to:
          Smash, Lift, Clear, Drop, Unknown
        """
        if raw is None:
            return "Unknown"
        s = str(raw).strip().lower()
        if "smash" in s:
            return "Smash"
        if "lift" in s:
            return "Lift"
        if "clear" in s:
            return "Clear"
        if "drop" in s:
            return "Drop"
        return "Unknown"

    def _apply_count_for_event(self, player_id: int, event: Dict[str, Any]) -> None:
        """
        Apply the counting effect of a finalized turn event:
        - increment totals
        - increment per-player totals
        - increment per-player shot-type bucket (normalized)
        """
        self.shot_count_total += 1
        if player_id == 1:
            self.shot_count_p1 += 1
        elif player_id == 2:
            self.shot_count_p2 += 1

        st = self._normalize_shot_type(event.get("shot_type", None))
        if player_id in (1, 2):
            self.shot_type_counts[player_id][st] = self.shot_type_counts[player_id].get(st, 0) + 1

    def _update_shot_counter(self, shot_data: Dict[str, Any]) -> None:
        """
        ORIGINAL turn-based counting (no consecutive-frame gating, no explicit end finalize):
        - Aggregate consecutive detections for the same player into ONE shot
        - Choose the highest-confidence event within that player's turn
        - Finalize (count) when the other player appears
        """
        if not shot_data:
            return

        pid = shot_data.get("player_id", None)
        frame_idx = shot_data.get("frame_index", None)
        if pid not in (1, 2) or frame_idx is None:
            return

        frame_idx = int(frame_idx)

        conf = shot_data.get("confidence", shot_data.get("shot_confidence", 1.0))
        try:
            conf = float(conf)
        except Exception:
            conf = 1.0

        # Start first turn
        if self._turn_player is None:
            self._turn_player = pid
            self._turn_best_event = dict(shot_data)
            self._turn_best_conf = conf
            return

        # Same player's turn -> keep only the max-confidence event
        if pid == self._turn_player:
            if conf > self._turn_best_conf:
                self._turn_best_conf = conf
                self._turn_best_event = dict(shot_data)
            return

        # Different player appeared -> finalize previous turn (count once)
        if self._turn_best_event is not None:
            best_frame = int(self._turn_best_event.get("frame_index", -1))

            # Optional debounce in case finalize repeats the same frame
            if best_frame >= 0 and self._last_counted_shot_frame != best_frame:
                self._last_counted_shot_frame = best_frame
                self._apply_count_for_event(self._turn_player, self._turn_best_event)

        # Start new turn for the new player
        self._turn_player = pid
        self._turn_best_event = dict(shot_data)
        self._turn_best_conf = conf


class MetricsPanel:
    def __init__(self, width: int, height: int, config: Dict, alpha: float, player_info: Dict):
        self.panel_width = int(width * config["width"])
        self.panel_height = int(height * config["height"])
        self.position = self.determine_position(config["position"], width, height, self.panel_width, self.panel_height)
        self.background_color = tuple(config["background_color"])
        self.text_color = tuple(config["text_color"])
        self.alpha = alpha
        self.player_info = player_info

        self.refresh_every_n = int(config.get("refresh_every_n", 9))
        self._tick = 0
        self._cached_metrics: Optional[pd.Series] = None

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.value_scale = 0.55
        self.unit_scale = 0.40
        self.value_thickness = 2
        self.unit_thickness = 1

    def determine_position(
        self,
        position: str,
        frame_width: int,
        frame_height: int,
        panel_width: int,
        panel_height: int,
    ) -> Tuple[int, int]:
        positions = {
            "top_left": (20, 250),
            "top_right": (frame_width - panel_width - 50, 50),
            "bottom_left": (50, frame_height - panel_height - 50),
            "bottom_right": (frame_width - panel_width - 70, frame_height - panel_height - 50),
        }
        return positions.get(position, (20, 250))

    @staticmethod
    def _safe_float(v, default=0.0) -> float:
        try:
            if v is None:
                return default
            if isinstance(v, float) and np.isnan(v):
                return default
            return float(v)
        except Exception:
            return default

    def _maybe_cache(self, metrics: pd.Series) -> pd.Series:
        self._tick += 1

        if metrics is None or getattr(metrics, "empty", True):
            return self._cached_metrics if self._cached_metrics is not None else pd.Series(dtype=float)

        if self.refresh_every_n <= 1:
            self._cached_metrics = metrics
            return metrics

        if self._cached_metrics is None:
            self._cached_metrics = metrics
            return metrics

        if (self._tick % self.refresh_every_n) == 0:
            self._cached_metrics = metrics

        return self._cached_metrics

    def _draw_value_with_unit(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        value_str: str,
        unit_str: str,
    ) -> None:
        cv2.putText(frame, value_str, (x, y), self.font, self.value_scale, self.text_color, self.value_thickness)
        (tw, _), _ = cv2.getTextSize(value_str, self.font, self.value_scale, self.value_thickness)
        cv2.putText(frame, unit_str, (x + tw + 8, y + 2), self.font, self.unit_scale, self.text_color, self.unit_thickness)

    def draw(self, frame: np.ndarray, metrics: pd.Series) -> np.ndarray:
        metrics = self._maybe_cache(metrics)
        if metrics is None or getattr(metrics, "empty", True):
            return frame

        x, y = self.position
        padding = 90
        line_spacing = 40
        column_spacing = self.panel_width // 2

        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + self.panel_width, y + self.panel_height), self.background_color, -1)
        cv2.addWeighted(overlay, self.alpha, frame, 1 - self.alpha, 0, frame)

        cv2.putText(frame, "Players Metrics", (x + self.panel_width // 3, y + 35), self.font, 0.6, self.text_color, 2)
        cv2.line(frame, (x + 10, y + 45), (x + self.panel_width - 10, y + 45), (180, 180, 180), 1)

        cv2.putText(frame, f"P1: {self.player_info[1]['name']}", (x + padding, y + 75), self.font, 0.55, self.text_color, 2)
        cv2.putText(frame, f"P2: {self.player_info[2]['name']}", (x + column_spacing + padding, y + 75), self.font, 0.55, self.text_color, 2)

        rows = [
            ("Speed", "player1_speed_kmh_smooth", "player2_speed_kmh_smooth", "km/h",  lambda v: f"{v:5.1f}"),
            ("Accel.", "player1_acc_ms2_smooth", "player2_acc_ms2_smooth", "m/s^2", lambda v: f"{v:5.2f}"),
            ("Distance", "player1_distance", "player2_distance", "m", lambda v: f"{v:6.1f}"),
        ]

        for i, (label, k1, k2, unit, fmt) in enumerate(rows):
            y_offset = y + 110 + i * line_spacing

            v1 = self._safe_float(metrics.get(k1, 0.0), 0.0)
            v2 = self._safe_float(metrics.get(k2, 0.0), 0.0)

            cv2.putText(frame, label, (x + 15, y_offset), self.font, 0.55, self.text_color, 2)
            self._draw_value_with_unit(frame, x + padding, y_offset, fmt(v1), unit)
            self._draw_value_with_unit(frame, x + column_spacing + padding, y_offset, fmt(v2), unit)

        cv2.line(frame, (x + column_spacing, y + 60), (x + column_spacing, y + self.panel_height - 30), (220, 220, 220), 2)
        return frame


class ShotTypePanel:
    def __init__(self, width: int, height: int, fps: int, config: Dict, alpha: float, player_info: Dict):
        self.fps = fps

        self.panel_width = int(width * config["width"])
        self.panel_height = int(height * config["height"])
        self.position = self.determine_position(config["position"], width, height, self.panel_width, self.panel_height)

        self.background_color = tuple(config["background_color"])
        self.text_color = tuple(config["text_color"])
        self.alpha = alpha
        self.player_info = player_info

        self.max_power_kmh = float(config.get("max_power_kmh", 230.0))
        self.bar_height = int(config.get("bar_height", 18))
        self.bar_bg_color = tuple(config.get("bar_bg_color", (70, 70, 70)))
        self.bar_fill_color = tuple(config.get("bar_fill_color", (250, 250, 250)))

        self.padding_x = int(config.get("padding_x", 20))
        self.line_color = tuple(config.get("line_color", (180, 180, 180)))

        self.title_scale = float(config.get("title_scale", 0.62))
        self.label_scale = float(config.get("label_scale", 0.55))
        self.value_scale = float(config.get("value_scale", 0.55))
        self.unit_scale = float(config.get("unit_scale", 0.40))
        self.thickness = int(config.get("thickness", 2))

        self.speed_key_smooth_kmh = str(config.get("speed_key_smooth_kmh", "ball_speed_smooth_kmh"))
        self.speed_key_ms = str(config.get("speed_key_ms", "ball_velocity1"))

        self.display_hold_frames = int(config.get("display_hold_frames", 25))
        self._hold_counter = 0
        self._last_event: Dict[str, Any] = {}

        self.refresh_every_n = int(config.get("refresh_every_n", 1))
        self._tick = 0
        self._cached_speed_kmh: float = 0.0

        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def determine_position(self, position: str, frame_width: int, frame_height: int, panel_width: int, panel_height: int):
        positions = {
            "top_left": (20, 160),
            "top_right": (frame_width - panel_width - 20, 160),
            "bottom_left": (20, frame_height - panel_height - 260),
            "bottom_right": (frame_width - panel_width - 20, frame_height - panel_height - 260),
        }
        return positions.get(position, (20, frame_height - panel_height - 280))

    @staticmethod
    def _format_time_mmss(seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m:02d}:{s:02d}"

    @staticmethod
    def _safe_float(v, default: float = 0.0) -> float:
        try:
            if v is None:
                return default
            if isinstance(v, float) and np.isnan(v):
                return default
            return float(v)
        except Exception:
            return default

    def _get_player_display(self, player_id: Optional[int]) -> str:
        if player_id in self.player_info:
            mapped_id = self.player_info[player_id].get("id", player_id)
            mapped_name = self.player_info[player_id].get("name", f"Player {mapped_id}")
            return f"{mapped_name} (ID: {mapped_id})"
        if player_id is None:
            return "Unknown"
        return f"Player {player_id} (ID: {player_id})"

    def _update_hold_state(self, shot_data: Dict) -> Dict:
        if shot_data:
            self._last_event = dict(shot_data)
            self._hold_counter = self.display_hold_frames
            return self._last_event

        if self._hold_counter > 0 and self._last_event:
            self._hold_counter -= 1
            return self._last_event

        return {}

    def _get_speed_kmh_throttled(self, metrics: pd.Series) -> float:
        self._tick += 1

        if metrics is None or getattr(metrics, "empty", True):
            return self._cached_speed_kmh

        smooth = self._safe_float(metrics.get(self.speed_key_smooth_kmh, None), default=np.nan)
        if not np.isnan(smooth):
            current = max(0.0, smooth)
        else:
            ms = self._safe_float(metrics.get(self.speed_key_ms, 0.0), 0.0)
            current = max(0.0, ms * 3.6)

        if self.refresh_every_n <= 1:
            self._cached_speed_kmh = current
            return current

        if (self._tick % self.refresh_every_n) == 0:
            self._cached_speed_kmh = current

        return self._cached_speed_kmh

    def _draw_value_with_unit(self, frame: np.ndarray, x: int, y: int, value_str: str, unit_str: str) -> None:
        cv2.putText(frame, value_str, (x, y), self.font, self.value_scale, self.text_color, 2)
        (tw, _), _ = cv2.getTextSize(value_str, self.font, self.value_scale, 2)
        cv2.putText(frame, unit_str, (x + tw + 8, y + 2), self.font, self.unit_scale, self.text_color, 1)

    def draw(self, frame: np.ndarray, shot_data: Dict, metrics: pd.Series) -> np.ndarray:
        event = self._update_hold_state(shot_data)
        if not event:
            return frame

        x, y = self.position

        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + self.panel_width, y + self.panel_height), self.background_color, -1)
        cv2.addWeighted(overlay, self.alpha, frame, 1 - self.alpha, 0, frame)

        cv2.putText(frame, "Shot Analysis", (x + self.panel_width // 4, y + 32), self.font, self.title_scale, self.text_color, self.thickness)
        cv2.line(frame, (x + 10, y + 45), (x + self.panel_width - 10, y + 45), self.line_color, 1)

        player_id = event.get("player_id", None)
        shot_type = event.get("shot_type", "Unknown")
        frame_index = int(event.get("frame_index", -1))

        t_seconds = (frame_index / float(self.fps)) if (frame_index >= 0 and self.fps > 0) else 0.0
        t_mmss = self._format_time_mmss(t_seconds)

        player_display = self._get_player_display(player_id)

        label_x = x + self.padding_x
        value_x = x + int(self.panel_width * 0.42)
        row_y = y + 78
        row_gap = 30

        def row(label: str, value: str, y_pos: int):
            cv2.putText(frame, label, (label_x, y_pos), self.font, self.label_scale, self.text_color, 2)
            cv2.putText(frame, value, (value_x, y_pos), self.font, self.value_scale, self.text_color, 2)

        row("Player", player_display, row_y); row_y += row_gap
        row("Shot Type", str(shot_type), row_y); row_y += row_gap
        row("Frame", f"{frame_index}", row_y); row_y += row_gap
        row("Time", t_mmss, row_y)

        div_y = row_y + 18
        cv2.line(frame, (x + 10, div_y), (x + self.panel_width - 10, div_y), self.line_color, 1)

        speed_kmh = self._get_speed_kmh_throttled(metrics)

        speed_title_y = div_y + 30
        cv2.putText(frame, "Shuttle Speed", (x + self.padding_x, speed_title_y), self.font, self.label_scale, self.text_color, 2)

        bar_x = x + self.padding_x
        bar_y = speed_title_y + 14
        bar_w = self.panel_width - (2 * self.padding_x)
        bar_h = self.bar_height

        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), self.bar_bg_color, -1)

        norm = 0.0 if self.max_power_kmh <= 0 else max(0.0, min(speed_kmh / self.max_power_kmh, 1.0))
        fill_w = int(bar_w * norm)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), self.bar_fill_color, -1)

        value_text = f"{speed_kmh:.1f}"
        value_size = cv2.getTextSize(value_text, self.font, self.value_scale, 2)[0]
        value_x_text = bar_x + bar_w - (value_size[0] + 40)
        value_y_text = bar_y - 6
        self._draw_value_with_unit(frame, value_x_text, value_y_text, value_text, "km/h")

        if not shot_data and self._hold_counter > 0:
            hold_text = "Last shot"
            hold_size = cv2.getTextSize(hold_text, self.font, 0.5, 1)[0]
            hold_x = x + self.panel_width - self.padding_x - hold_size[0]
            hold_y = y + 32
            cv2.putText(frame, hold_text, (hold_x, hold_y), self.font, 0.5, self.text_color, 1)

        return frame


class ShotCounterPanel:
    """
    Compact top overlay that shows rally/shot count.
    """

    def __init__(self, width: int, height: int, config: Dict, alpha: float, player_info: Dict):
        self.width = width
        self.height = height
        self.alpha = alpha
        self.player_info = player_info

        self.enabled = bool(config.get("enabled", True))
        self.background_color = tuple(config.get("background_color", (18, 18, 18)))
        self.text_color = tuple(config.get("text_color", (255, 255, 255)))
        self.line_color = tuple(config.get("line_color", (180, 180, 180)))

        self.panel_w = int(width * float(config.get("width", 0.22)))
        self.panel_h = int(height * float(config.get("height", 0.07)))
        self.margin_top = int(config.get("margin_top", 18))

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.title_scale = float(config.get("title_scale", 0.55))
        self.value_scale = float(config.get("value_scale", 0.72))
        self.thickness = int(config.get("thickness", 2))

        self.show_per_player = bool(config.get("show_per_player", False))

    def _panel_position(self) -> Tuple[int, int]:
        x = (self.width - self.panel_w) // 2
        y = self.margin_top
        return x, y

    def draw(self, frame: np.ndarray, total_shots: int, p1_shots: int = 0, p2_shots: int = 0) -> np.ndarray:
        if not self.enabled:
            return frame

        x, y = self._panel_position()

        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + self.panel_w, y + self.panel_h), self.background_color, -1)
        cv2.addWeighted(overlay, self.alpha, frame, 1 - self.alpha, 0, frame)

        cv2.line(frame, (x + 10, y + self.panel_h - 10), (x + self.panel_w - 10, y + self.panel_h - 10), self.line_color, 1)

        cv2.putText(frame, "Shots", (x + 14, y + int(self.panel_h * 0.62)),
                    self.font, self.title_scale, self.text_color, 2)

        value_text = f"{int(total_shots)}"
        (tw, _), _ = cv2.getTextSize(value_text, self.font, self.value_scale, self.thickness)
        vx = x + self.panel_w - 14 - tw
        vy = y + int(self.panel_h * 0.66)
        cv2.putText(frame, value_text, (vx, vy), self.font, self.value_scale, self.text_color, self.thickness)

        if self.show_per_player:
            sub = f"P1 {p1_shots}  |  P2 {p2_shots}"
            cv2.putText(frame, sub, (x + 14, y + self.panel_h - 13),
                        self.font, 0.45, self.text_color, 1)

        return frame

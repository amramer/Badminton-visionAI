"""
Player Tracking and Motion Analysis (Frame-Accurate + Stable)

Key fixes vs previous version:
1) Frame alignment is correct: you must call begin_frame(frame_idx) -> add positions -> end_frame().
2) No off-by-one: we do not "step" after adding positions.
3) Robust handling of missing players/ball using NaNs.
4) Stable ball speed for UI:
   - ball_speed_kmh (instant)
   - ball_speed_smooth_kmh (rolling median + mean)

Public API used by your Dashboard/SideCourt:
- begin_frame(frame_idx: int)
- add_player_position(player_id: int, position: tuple[float, float])
- add_ball_position(position: tuple[float, float])
- end_frame()
- into_dataframe(fps: int) -> pd.DataFrame
"""

from __future__ import annotations

from typing import List, Optional
import logging
from dataclasses import dataclass
import pandas as pd
import numpy as np
from collections import defaultdict

from utils import get_logger

logger = get_logger(__name__, log_file="logs/metrics.log", level=logging.DEBUG)


class InvalidDataPoint(Exception):
    """Custom exception for invalid data points."""
    pass


@dataclass
class PlayerPosition:
    """Player position (meters) in a given frame."""
    id: int
    position: tuple[float, float]

    def __post_init__(self):
        if self.id not in (1, 2):
            raise ValueError("Player ID must be 1 or 2")
        if not isinstance(self.position[0], (float, int)) or not isinstance(self.position[1], (float, int)):
            raise ValueError("Position coordinates must be numeric")


@dataclass
class BallPosition:
    """Ball position (meters) in a given frame."""
    position: tuple[float, float]

    def __post_init__(self):
        if not isinstance(self.position[0], (float, int)) or not isinstance(self.position[1], (float, int)):
            raise ValueError("Position coordinates must be numeric")


@dataclass
class DataPoint:
    """Data collected for a single frame."""
    frame: int
    players_position: Optional[List[PlayerPosition]] = None
    ball_position: Optional[BallPosition] = None

    def validate(self) -> None:
        if self.frame is None:
            raise InvalidDataPoint("Unknown frame")

        if self.players_position is None:
            self.players_position = []

        # Keep only first occurrence of each valid player id (1,2)
        valid: List[PlayerPosition] = []
        seen = set()
        for p in self.players_position:
            try:
                if p.id in (1, 2) and p.id not in seen:
                    valid.append(p)
                    seen.add(p.id)
            except Exception:
                continue

        self.players_position = valid

    def add_player_position(self, p: PlayerPosition) -> None:
        if self.players_position is None:
            self.players_position = []
        self.players_position.append(p)


class Metrics:
    """
    Stable frame-accurate metrics collector.

    Expected steps for per-frame usage:
        metrics.begin_frame(frame_idx)
        metrics.add_player_position(...)
        metrics.add_ball_position(...)
        metrics.end_frame()
    """

    def __init__(self):
        self._data = defaultdict(list)
        self.current: Optional[DataPoint] = None

    def restart(self) -> None:
        self.__init__()

    def begin_frame(self, frame_idx: int) -> None:
        """Start collecting data for a specific frame index."""
        self.current = DataPoint(frame=int(frame_idx), players_position=[], ball_position=None)

    def end_frame(self) -> None:
        """Validate and commit current frame datapoint into storage."""
        if self.current is None:
            logger.warning("end_frame() called without an active frame. Ignoring.")
            return

        try:
            self.current.validate()
            frame = self.current.frame
            self._data["frame"].append(frame)

            players = self.current.players_position or []

            # Always append exactly one entry for each player id (1,2) per frame
            for pid in (1, 2):
                xk = f"player{pid}_x"
                yk = f"player{pid}_y"
                pos = next((p for p in players if p.id == pid), None)

                if pos is None:
                    self._data[xk].append(np.nan)
                    self._data[yk].append(np.nan)
                else:
                    self._data[xk].append(float(pos.position[0]))
                    self._data[yk].append(float(pos.position[1]))

            # Ball
            if self.current.ball_position is None:
                self._data["ball_x"].append(np.nan)
                self._data["ball_y"].append(np.nan)
            else:
                self._data["ball_x"].append(float(self.current.ball_position.position[0]))
                self._data["ball_y"].append(float(self.current.ball_position.position[1]))

        except Exception as e:
            logger.error(
                f"Error committing frame {getattr(self.current, 'frame', None)}: {e}",
                exc_info=True
            )
        finally:
            self.current = None

    def add_player_position(self, player_id: int, position: tuple[float, float]) -> None:
        if self.current is None:
            logger.warning("add_player_position() called before begin_frame(). Ignoring.")
            return
        try:
            self.current.add_player_position(PlayerPosition(id=int(player_id), position=position))
        except Exception as e:
            logger.error(f"Error adding player position: {e}", exc_info=True)

    def add_ball_position(self, position: tuple[float, float]) -> None:
        if self.current is None:
            logger.warning("add_ball_position() called before begin_frame(). Ignoring.")
            return
        try:
            self.current.ball_position = BallPosition(position=position)
        except Exception as e:
            logger.error(f"Error adding ball position: {e}", exc_info=True)

    def into_dataframe(self, fps: int) -> pd.DataFrame:
        """
        Convert stored tracking data into a DataFrame with kinematics features.

        Notes:
        - Uses interval=1 for realtime stability and clarity.
        - Distance is cumulative per player (ignores NaN steps safely).
        - Player speed includes km/h and smoothed km/h for UI.
        - Player acceleration (km/h/s) for readable UI.
        - Ball speed includes km/h and smoothed km/h for UI.
        """
        
        if not self._data or not self._data.get("frame"):
            return pd.DataFrame()

        try:
            df = pd.DataFrame(self._data).sort_values("frame").reset_index(drop=True)

            fps = max(1, int(fps))
            df["time"] = df["frame"] / float(fps)

            # Guard against dt=0
            dt = df["time"].diff(1)
            dt = dt.replace(0, np.nan)

            # --- Player metrics (interval=1) ---
            for pid in (1, 2):
                x = f"player{pid}_x"
                y = f"player{pid}_y"

                dx = df[x].diff(1)
                dy = df[y].diff(1)

                df[f"player{pid}_delta_x1"] = dx
                df[f"player{pid}_delta_y1"] = dy

                df[f"player{pid}_Vx1"] = dx / dt
                df[f"player{pid}_Vy1"] = dy / dt
                df[f"player{pid}_Vnorm1"] = np.sqrt(df[f"player{pid}_Vx1"] ** 2 + df[f"player{pid}_Vy1"] ** 2)

                df[f"player{pid}_Ax1"] = df[f"player{pid}_Vx1"].diff(1) / dt
                df[f"player{pid}_Ay1"] = df[f"player{pid}_Vy1"].diff(1) / dt
                df[f"player{pid}_Anorm1"] = np.sqrt(df[f"player{pid}_Ax1"] ** 2 + df[f"player{pid}_Ay1"] ** 2)

                # Distance (meters), cumulative
                step_dist = np.sqrt(dx ** 2 + dy ** 2)
                df[f"player{pid}_distance"] = step_dist.fillna(0.0).cumsum()

                # --- UI-friendly units + smoothing ---
                # Speed in km/h
                df[f"player{pid}_speed_kmh"] = df[f"player{pid}_Vnorm1"] * 3.6

                # Smoothed speed (median then mean => reduces flicker but stays responsive)
                df[f"player{pid}_speed_kmh_smooth"] = (
                    df[f"player{pid}_speed_kmh"]
                    .rolling(window=9, min_periods=1).median()
                    .rolling(window=7, min_periods=1).mean()
                )

                # Smoothed acceleration (m/s^2)
                df[f"player{pid}_acc_ms2_smooth"] = (
                    df[f"player{pid}_Anorm1"]
                    .rolling(window=9, min_periods=1).median()
                    .rolling(window=7, min_periods=1).mean()
                )

                # Optional more intuitive acceleration: km/h per second
                df[f"player{pid}_acc_kmhps_smooth"] = df[f"player{pid}_acc_ms2_smooth"] * 3.6

            # --- Ball metrics (interval=1) ---
            if "ball_x" in df.columns and "ball_y" in df.columns:
                bdx = df["ball_x"].diff(1)
                bdy = df["ball_y"].diff(1)
                bdt = dt  # same dt

                df["ball_delta_x1"] = bdx
                df["ball_delta_y1"] = bdy

                df["ball_Vx1"] = bdx / bdt
                df["ball_Vy1"] = bdy / bdt

                df["ball_velocity1"] = np.sqrt(df["ball_Vx1"] ** 2 + df["ball_Vy1"] ** 2)
                df["ball_shot_power1"] = df["ball_velocity1"] ** 2

                df["ball_speed_kmh"] = df["ball_velocity1"] * 3.6

                # Robust smoothing for UI (reduces flicker/spikes)
                df["ball_speed_smooth_kmh"] = (
                    df["ball_speed_kmh"]
                    .rolling(window=7, min_periods=1).median()
                    .rolling(window=5, min_periods=1).mean()
                )

            return df

        except Exception as e:
            logger.error(f"Error creating DataFrame: {e}", exc_info=True)
            return pd.DataFrame()
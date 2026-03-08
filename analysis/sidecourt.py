from math import log
import numpy as np
import logging
import cv2
from typing import List, Literal, Iterable, Optional, Type
from dataclasses import dataclass
from utils import convert_meters_to_pixel_distance, convert_pixel_distance_to_meters
from constants import COURT_WIDTH, COURT_LENGTH, SHORT_SERVICE_LINE, LONG_SERVICE_LINE
from tracking import Player, Players, Ball, Keypoint, Keypoints
from utils import get_logger
from analysis import Metrics

logger = get_logger(__name__, log_file="logs/sidecourt.log", level=logging.DEBUG)

class InconsistentPredictedKeypoints(Exception):
    pass

PointPixels = tuple[int, int]

@dataclass
class Rectangle:
    """
    Rectangle geometry utilities
    """
    top_left: PointPixels
    bottom_right: PointPixels

    @property
    def width(self) -> int:
        return self.bottom_right[0] - self.top_left[0]

    @property
    def height(self) -> int:
        return self.bottom_right[1] - self.top_left[1]

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def perimeter(self) -> int:
        return 2 * self.width + 2 * self.height

@dataclass
class SideCourtKeypoints:
    """
    badminton Sidecourt 32 points of interest
    """
    k1: PointPixels
    k2: PointPixels
    k3: PointPixels
    k4: PointPixels
    k5: PointPixels
    k6: PointPixels
    k7: PointPixels
    k8: PointPixels
    k9: PointPixels
    k10: PointPixels
    k11: PointPixels
    k12: PointPixels
    k13: PointPixels
    k14: PointPixels
    k15: PointPixels
    k16: PointPixels
    k17: PointPixels
    k18: PointPixels
    k19: PointPixels
    k20: PointPixels
    k21: PointPixels
    k22: PointPixels
    k23: PointPixels
    k24: PointPixels
    k25: PointPixels
    k26: PointPixels
    k27: PointPixels
    k28: PointPixels
    k29: PointPixels
    k30: PointPixels
    k31: PointPixels
    k32: PointPixels

    def __post_init__(self):
        self.origin = self._get_origin()

    @property
    def width(self) -> int:
        return abs(self.k17[0] - self.k16[0])

    @property
    def height(self) -> int:
        return abs(self.k1[1] - self.k28[1])

    def _get_origin(self) -> tuple[float, float]:
        x0 = (self.k16[0] + self.k17[0]) / 2.0
        y0 = (self.k16[1] + self.k17[1]) / 2.0
        return (x0, y0)

    def keypoints(self, number_keypoints: int = 32) -> List[Keypoint]:
        """Returns all keypoints as a list of Keypoint objects."""
        points = [
            Keypoint(id=i, xy=tuple(float(p) for p in v))
            for i, (k, v) in enumerate(self.__dict__.items())
            if "k" in k
        ]

        if len(points) != number_keypoints:
            raise ValueError(f"Expected {number_keypoints} keypoints, found {len(points)}")

        return points

    def __getitem__(self, k: str) -> Keypoint:
        id = int(k.replace("k", "")) - 1
        return Keypoint(
            id=id,
            xy=tuple(float(p) for p in self.__dict__[k]),
        )

    def lines(self) -> List[tuple[PointPixels, PointPixels]]:
        """
        Court lines start and end keypoints
        """
        return [
            (self.k1, self.k5),
            (self.k1, self.k28),
            (self.k5, self.k32),
            (self.k28, self.k32),
            (self.k16, self.k17),
            (self.k2, self.k29),
            (self.k4, self.k31),
            (self.k6, self.k10),
            (self.k11, self.k15),
            (self.k23, self.k27),
            (self.k18, self.k22),
            (self.k3, self.k13),
            (self.k20, self.k30),
        ]

    def shift_point_origin(
        self,
        point: tuple[float, float],
        dimension: Literal["pixels", "meters"],
    ) -> tuple[float, float]:
        """
        Shift a mini-court pixel point so that (0,0) is the court center.
        Optionally convert to meters.
        """
        dx = float(point[0] - self.origin[0])
        dy = float(point[1] - self.origin[1])

        if dimension == "meters":
            dx = convert_pixel_distance_to_meters(dx, COURT_WIDTH, self.width)
            dy = convert_pixel_distance_to_meters(dy, COURT_LENGTH, self.height)

        return dx, dy


class SideCourt:
    """
    Side court abstraction with utilities to project and draw
    objects of interest in a 2d plane.

    Attributes:
        width: Width of the video frame.
        height: Height of the video frame.
        position: Position of the canvas ("top_right" or "bottom_left").
    """

    DEFAULT_WIDTH_MULTIPLIER = 0.17
    DEFAULT_HEIGHT_MULTIPLIER = 0.5
    BUFFER = 45  # Margin from the edges
    PADDING = 20

    def __init__(self,
                 width: int,
                 height: int,
                 position: str = "top_right",
                 scale_factor: float = 1.0,
                 alpha: float = 0.5):
        self.width = width
        self.height = height
        self.position = position
        self.scale_factor = scale_factor
        self.alpha = alpha

        # Adjust multipliers with scale_factor
        self.WIDTH_MULTIPLIER = self.DEFAULT_WIDTH_MULTIPLIER * self.scale_factor
        self.HEIGHT_MULTIPLIER = self.DEFAULT_HEIGHT_MULTIPLIER * self.scale_factor

        # Canvas background points in pixels
        self.WIDTH = int(self.WIDTH_MULTIPLIER * self.width)
        self.HEIGHT = int(self.WIDTH / (COURT_WIDTH / COURT_LENGTH)) - self.BUFFER

        self._set_canvas_background_position()
        self._set_side_court_position()
        self._set_side_court_keypoints()

        # Initialize the homography matrix
        self.H = None
        # self.H_ball = None

    def _set_canvas_background_position(self) -> None:
        """
        Set the canvas background position relative to the video frame.
        The canvas can be either in the top-right or bottom-left corner,
        with a margin of BUFFER pixels.
        """
        if self.position == "top_right":
            # Position canvas in the top-right corner
            start_x = self.width - self.WIDTH - self.BUFFER
            start_y = 3.7 * self.BUFFER
        elif self.position == "bottom_left":
            # Position canvas in the bottom-left corner
            start_x = self.BUFFER
            start_y = self.height - self.HEIGHT - 3 * self.BUFFER
        else:
            raise ValueError("Invalid position. Use 'top_right' or 'bottom_left'.")

        end_x = start_x + self.WIDTH
        end_y = start_y + self.HEIGHT

        self.background_position = Rectangle(top_left=(int(start_x), int(start_y)), bottom_right=(int(end_x), int(end_y)))

    def _set_side_court_position(self) -> None:
        """
        Set the side court position (respecting measurements in meters)
        inside the canvas background
        """
        court_start_x = self.background_position.top_left[0] + self.PADDING
        court_start_y = self.background_position.top_left[1] + self.PADDING
        court_end_x = self.background_position.bottom_right[0] - self.PADDING
        court_width = court_end_x - court_start_x
        court_height = convert_meters_to_pixel_distance(
            COURT_LENGTH,
            reference_in_meters=COURT_WIDTH,
            reference_in_pixels=court_width
        )
        court_end_y = court_start_y + court_height

        self.court_position = Rectangle(
            top_left=(int(court_start_x), int(court_start_y)),
            bottom_right=(int(court_end_x), int(court_end_y)),
        )

    def _set_side_court_keypoints(self) -> None:
        """
        Set the side court keypoints
        """
        long_service_height = convert_meters_to_pixel_distance(
            LONG_SERVICE_LINE,
            reference_in_meters=COURT_WIDTH,
            reference_in_pixels=self.court_position.width,
        )

        short_service_height = convert_meters_to_pixel_distance(
            SHORT_SERVICE_LINE,
            reference_in_meters=COURT_WIDTH,
            reference_in_pixels=self.court_position.width
        )

        self.court_keypoints = SideCourtKeypoints(
            # corner keypoints
            k1=(
                self.court_position.top_left[0],
                self.court_position.bottom_right[1],
            ),
            k5=(
                self.court_position.bottom_right[0],
                self.court_position.bottom_right[1],
            ),
            k28=(
                self.court_position.top_left[0],
                self.court_position.top_left[1],
            ),
            k32=(
                self.court_position.bottom_right[0],
                self.court_position.top_left[1],
            ),

            # Net keypoints
            k16=(
                self.court_position.top_left[0],
                int(self.court_position.top_left[1] + self.court_position.height / 2),
            ),
            k17=(
                self.court_position.bottom_right[0],
                int(self.court_position.top_left[1] + self.court_position.height / 2),
            ),

            # Service lines keypoints for lower court region
            k2=(
                int(self.court_position.top_left[0] + self.court_position.width / 10),
                self.court_position.bottom_right[1],
            ),
            k3=(
                int(self.court_position.top_left[0] + self.court_position.width / 2),
                self.court_position.bottom_right[1],
            ),
            k4=(
                int(self.court_position.bottom_right[0] - self.court_position.width / 10),
                self.court_position.bottom_right[1],
            ),
            k6=(
                self.court_position.top_left[0],
                int(self.court_position.bottom_right[1] - long_service_height),
            ),
            k7=(
                int(self.court_position.top_left[0] + self.court_position.width / 10),
                int(self.court_position.bottom_right[1] - long_service_height),
            ),
            k8=(
                int(self.court_position.top_left[0] + self.court_position.width / 2),
                int(self.court_position.bottom_right[1] - long_service_height),
            ),
            k9=(
                int(self.court_position.bottom_right[0] - self.court_position.width / 10),
                int(self.court_position.bottom_right[1] - long_service_height),
            ),
            k10=(
                self.court_position.bottom_right[0],
                int(self.court_position.bottom_right[1] - long_service_height),
            ),
            k11=(
                self.court_position.top_left[0],
                int(self.court_position.bottom_right[1] - short_service_height),
            ),
            k12=(
                int(self.court_position.top_left[0] + self.court_position.width / 10),
                int(self.court_position.bottom_right[1] - short_service_height),
            ),
            k13=(
                int(self.court_position.top_left[0] + self.court_position.width / 2),
                int(self.court_position.bottom_right[1] - short_service_height),
            ),
            k14=(
                int(self.court_position.bottom_right[0] - self.court_position.width / 10),
                int(self.court_position.bottom_right[1] - short_service_height),
            ),
            k15=(
                self.court_position.bottom_right[0],
                int(self.court_position.bottom_right[1] - short_service_height),
            ),

            # Service lines keypoints for upper court region
            k18=(
                self.court_position.top_left[0],
                int(self.court_position.top_left[1] + short_service_height),
            ),
            k19=(
                int(self.court_position.top_left[0] + self.court_position.width / 10),
                int(self.court_position.top_left[1] + short_service_height),
            ),
            k20=(
                int(self.court_position.top_left[0] + self.court_position.width / 2),
                int(self.court_position.top_left[1] + short_service_height),
            ),
            k21=(
                int(self.court_position.bottom_right[0] - self.court_position.width / 10),
                int(self.court_position.top_left[1] + short_service_height),
            ),
            k22=(
                self.court_position.bottom_right[0],
                int(self.court_position.top_left[1] + short_service_height),
            ),
            k23=(
                self.court_position.top_left[0],
                int(self.court_position.top_left[1] + long_service_height),
            ),
            k24=(
                int(self.court_position.top_left[0] + self.court_position.width / 10),
                int(self.court_position.top_left[1] + long_service_height),
            ),
            k25=(
                int(self.court_position.top_left[0] + self.court_position.width / 2),
                int(self.court_position.top_left[1] + long_service_height),
            ),
            k26=(
                int(self.court_position.bottom_right[0] - self.court_position.width / 10),
                int(self.court_position.top_left[1] + long_service_height),
            ),
            k27=(
                self.court_position.bottom_right[0],
                int(self.court_position.top_left[1] + long_service_height),
            ),
            k29=(
                int(self.court_position.top_left[0] + self.court_position.width / 10),
                self.court_position.top_left[1],
            ),
            k30=(
                int(self.court_position.top_left[0] + self.court_position.width / 2),
                self.court_position.top_left[1],
            ),
            k31=(
                int(self.court_position.bottom_right[0] - self.court_position.width / 10),
                self.court_position.top_left[1],
            ),
        )

    def _draw_dashed_line(
        self,
        frame: np.ndarray,
        start: PointPixels,
        end: PointPixels,
        color: tuple[int, int, int],
        thickness: int = 2,
        dash_length: int = 10,
        gap_length: int = 5,
    ):
        """
        Draw a dashed line between two points.
        """
        x1, y1 = start
        x2, y2 = end

        length = int(np.hypot(x2 - x1, y2 - y1))
        if length == 0:
            return

        dx = (x2 - x1) / length
        dy = (y2 - y1) / length

        dist = 0
        while dist < length:
            x_start = int(x1 + dx * dist)
            y_start = int(y1 + dy * dist)
            dist += dash_length
            x_end = int(x1 + dx * min(dist, length))
            y_end = int(y1 + dy * min(dist, length))

            cv2.line(frame, (x_start, y_start), (x_end, y_end), color, thickness)
            dist += gap_length

    def draw_background_single_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw the side court background on the given frame
        """
        shapes = np.zeros_like(frame, np.uint8)
        cv2.rectangle(
            shapes,
            self.background_position.top_left,
            self.background_position.bottom_right,
            (190, 215, 190),
            -1,
        )
        output_frame = frame.copy()
        mask = shapes.astype(bool)
        output_frame[mask] = cv2.addWeighted(
            output_frame,
            self.alpha,
            shapes,
            1 - self.alpha,
            0,
        )[mask]

        return output_frame
    
    def draw_side_court_single_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw minicourt points of interest and lines
        """
        # Draw court keypoints (k1..k32)
        for k, v in vars(self.court_keypoints).items():
            if not k.startswith("k"):
                continue
            cx, cy = int(v[0]), int(v[1])
            cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)

        # Draw origin (center) – subtle
        # ox, oy = map(int, map(round, self.court_keypoints.origin))
        # cv2.circle(frame, (ox, oy), 3, (0, 180, 0), -1)

        # Draw lines
        for start_point, end_point in self.court_keypoints.lines():

            # --- NET LINE (k16 → k17) ---
            if start_point == self.court_keypoints.k16 and end_point == self.court_keypoints.k17:
                self._draw_dashed_line(
                    frame,
                    start_point,
                    end_point,
                    color=(60, 60, 60),   # dark gray
                    thickness=3,
                    dash_length=10,
                    gap_length=6
                )
            else:
                cv2.line(frame, start_point, end_point, (0, 0, 0), 2)

        return frame

    def homography_matrix(self, keypoints_detection: Keypoints) -> np.ndarray:
        """
        Calculates the homography matrix that projects the court keypoints detected
        on a given frame into the 2D court.

        Parameters:
            keypoints_detection: Predicted keypoints on a single frame.
        """
        if len(keypoints_detection) != 32:
            raise ValueError(f"Expected 32 keypoints, but got {len(keypoints_detection)}")

        src_points = np.array([keypoint for keypoint in keypoints_detection])
        dst_keypoints = self.court_keypoints.keypoints()
        dst_points = np.array([keypoint.xy for keypoint in dst_keypoints])

        if src_points.shape != dst_points.shape:
            raise InconsistentPredictedKeypoints("Mismatch in number of source and destination points")

        H, _ = cv2.findHomography(src_points, dst_points)

        return H

    def project_point(
        self,
        point: tuple[float, float],
        homography_matrix: np.ndarray,
    ) -> tuple[float, float]:
        """
        Project point given a homography matrix H.

        Parameters:
            point: point to be projected
            homography_matrix: homography matrix that projects into the court 2d plane

        Returns:
            projected point
        """
        assert homography_matrix.shape == (3, 3)

        src_point = np.array([float(p) for p in point])
        src_point = np.append(
            src_point,
            np.array([1]),
            axis=0,
        )

        dst_point = np.matmul(homography_matrix, src_point)
        dst_point = dst_point / dst_point[2]

        return (dst_point[0], dst_point[1])

    def project_player(
        self,
        player_detection: Player,
        homography_matrix: np.ndarray,
    ) -> Player:
        """
        Player detection 2d court projection
        """
        projected_point = self.project_point(
            point=player_detection.feet,
            homography_matrix=homography_matrix,
        )
        player_detection.projection = tuple(float(v) for v in projected_point)
        return player_detection

    def project_ball(
        self,
        ball_detection: Ball,
        homography_matrix: np.ndarray,
    ) -> Ball:
        """
        Ball detection 2d court projection
        """
        projected_point = self.project_point(
            point=ball_detection.asint(),
            homography_matrix=homography_matrix,
        )
        ball_detection.projection = tuple(float(v) for v in projected_point)
        return ball_detection

    def draw_projected_player(
        self,
        frame: np.ndarray,
        player_detection: Player,
        homography_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Project and draw a single player
        """
        projected_player = self.project_player(
            player_detection=player_detection,
            homography_matrix=homography_matrix,
        )

        return projected_player.draw_projection(frame)

    def draw_projected_players(
        self,
        frame: np.ndarray,
        players_detection: List[Player],
        homography_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Project and draw players
        """
        for player_detection in players_detection:
            frame = self.draw_projected_player(
                frame=frame,
                player_detection=player_detection,
                homography_matrix=homography_matrix,
            )

        return frame

    def draw_projected_ball(
        self,
        frame: np.ndarray,
        ball_detection: Ball,
        homography_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Project and draw ball
        """
        projected_ball = self.project_ball(
            ball_detection=ball_detection,
            homography_matrix=homography_matrix,
        )

        return projected_ball.draw_projection(frame)

    def draw_court(
        self,
        frame: np.ndarray,
        keypoints_detection: Keypoints,
        players_detection: Optional[List[Player]],
        ball_detection: Optional[Ball],
    ) -> np.ndarray:
        """
        Project and draw court and objects of interest.
        Updated to match reference implementation logic.

        Parameters:
            frame: video frame
            keypoints_detection: court keypoints detection
            players_detection: players bounding box detection
            ball_detection: ball position detection
        """
        output_frame = self.draw_background_single_frame(frame)
        output_frame = self.draw_side_court_single_frame(output_frame)

        if keypoints_detection:
            try:                
                # Calculate the homography matrix using the transformed keypoints
                self.H = self.homography_matrix(keypoints_detection)
                
            except Exception as e:
                logger.error(f"Error in perspective transform: {str(e)}")
                return output_frame

        # Rest of the drawing logic remains the same...
        if self.H is not None and players_detection:
            output_frame = self.draw_projected_players(output_frame, players_detection, self.H)

        # if self.H_ball is not None and ball_detection:
        #     output_frame = self.draw_projected_ball(output_frame, ball_detection, self.H_ball)

        return output_frame

    def update_players_position(
        self,
        players_detection: List[Player],
        metrics: Optional[Metrics] = None
    ) -> None:
        """
        Project all players and update their positions for analytics using the existing homography matrix.

        Args:
            players_detection (List[Player]): List of detected players to project.
            metrics (Optional[Metrics]): Metrics object to update player positions.
        """
        if self.H is None:
            logger.warning("Homography matrix is not available. Cannot update player positions.")
            return

        for player_detection in players_detection:
            # Project the player using the existing homography matrix
            projected_player = self.project_player(
                player_detection=player_detection,
                homography_matrix=self.H,
            )

            # Shift the projected players position to the mini court origin
            shifted_projected_player_pos = self.court_keypoints.shift_point_origin(
                point=tuple(float(v) for v in projected_player.projection),
                dimension="meters",
            )

            # Update metrics if provided
            if metrics is not None:
                metrics.add_player_position(
                    player_id=projected_player.id,
                    position=shifted_projected_player_pos
                )

    def update_ball_position(
        self,
        ball_detection: Ball,
        metrics: Optional[Metrics] = None
    ) -> None:
        """
        Project the ball's position and update its metrics for analytics.

        Args:
            ball_detection (Ball): The detected ball to project.
            metrics (Optional[Metrics]): Metrics object to update the ball's position.
        """
        if self.H is None:
            logger.warning("Homography matrix is not available. Cannot update ball position.")
            return

        # Project the ball using the homography matrix
        projected_ball = self.project_ball(
            ball_detection=ball_detection,
            homography_matrix=self.H,
        )

        # Shift the projected position to the origin in meters
        shifted_projected_ball_pos = self.court_keypoints.shift_point_origin(
            point=tuple(float(v) for v in projected_ball.projection),
            dimension="meters",
        )

        # Update metrics if provided
        if metrics is not None:
            metrics.add_ball_position(shifted_projected_ball_pos)

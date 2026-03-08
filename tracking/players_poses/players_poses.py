import numpy as np
import torch
import cv2
import logging
import supervision as sv
from ultralytics import YOLO
from typing import List, Literal, Iterable, Optional, Type
from PIL import Image
from utils.io_utils import save_json, load_json
from utils import read_video,save_video,get_logger
from config import load_config


config = load_config()
logger = get_logger(__name__, log_file="logs/players_poses.log", level=logging.DEBUG)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Keypoint:
    """
    Represents a single keypoint in a player's pose.
    """
    def __init__(self, id: int, name: str, xy: tuple[float, float]):
        self.id = id
        self.name = name
        self.xy = xy

    def asint(self) -> tuple[int, int]:
        """Return the coordinates as integers."""
        return tuple(map(int, self.xy))

    def serialize(self) -> dict:
        """Serialize the keypoint for saving or debugging."""
        return {"id": self.id, "name": self.name, "xy": self.xy}

    def draw(self, frame: np.ndarray, color: tuple[int, int, int], radius: int = 3) -> np.ndarray:
        """
        Draw the keypoint on the given frame.

        Args:
            frame (np.ndarray): The frame to draw on.
            color (tuple[int, int, int]): Color of the keypoint.
            radius (int): Radius of the keypoint circle.

        Returns:
            np.ndarray: Frame with the keypoint drawn.
        """
        cv2.circle(frame, self.asint(), radius = radius, color = color, thickness = 1)
        return frame


class Pose:
    """
    Represents a player's pose with multiple keypoints and connections or edges.
    """
    KEYPOINTS_NAMES = [
        "left_foot", "right_foot", "torso", "right_shoulder", "left_shoulder",
        "head", "neck", "left_hand", "right_hand", "right_knee", "left_knee",
        "right_elbow", "left_elbow",
    ]
    CONNECTIONS = [
        ("left_foot", "left_knee"), ("left_knee", "torso"), ("right_foot", "right_knee"),
        ("right_knee", "torso"), ("torso", "left_shoulder"), ("torso", "right_shoulder"),
        ("left_hand", "left_elbow"), ("left_elbow", "left_shoulder"), ("left_shoulder", "neck"),
        ("neck", "head"), ("right_hand", "right_elbow"), ("right_elbow", "right_shoulder"),
        ("right_shoulder", "neck"),
    ]

    def __init__(self, keypoints: List[Keypoint]):
        self.keypoints = keypoints
        self.keypoints_by_name = {kp.name: kp for kp in keypoints}

    def serialize(self) -> dict:
        """Serialize the pose for saving or debugging."""
        return {"keypoints": [kp.serialize() for kp in self.keypoints]}
    
    def get_keypoint_by_name(self, name: str) -> Keypoint:
        
        assert name in self.KEYPOINTS_NAMES

        return self.keypoints_by_name[name]    
    
    def get_connections_as_points(self) -> List[tuple[tuple[int, int], tuple[int, int]]]:
            """
            Get all connections as pairs of points.

            Returns:
                List[tuple[tuple[int, int], tuple[int, int]]]: List of connections as point pairs.
            """
            keypoints = {kp.name: kp.asint() for kp in self.keypoints}
            connections = []
            for conn in self.CONNECTIONS:
                if conn[0] in keypoints and conn[1] in keypoints:
                    connections.append((keypoints[conn[0]], keypoints[conn[1]]))
            return connections
    
    def distance_between_keypoints(self, keypoint1_name: str, keypoint2_name: str) -> float:
        """
        Calculate the Euclidean distance between two keypoints.

        Args:
            keypoint1_name (str): The name of the first keypoint.
            keypoint2_name (str): The name of the second keypoint.

        Returns:
            float: The Euclidean distance between the two keypoints.
        """
        keypoint1 = self.get_keypoint_by_name(keypoint1_name)
        keypoint2 = self.get_keypoint_by_name(keypoint2_name)

        if keypoint1 and keypoint2:
            return np.linalg.norm(np.array(keypoint1.asint()) - np.array(keypoint2.asint()))
        return 0.0

    def draw(
        self,
        frame: np.ndarray,
        keypoint_color: tuple[int, int, int],
        keypoint_radius: int,
        connection_color: tuple[int, int, int],
        connection_thickness: int,
        draw_keypoints: bool = True,
        draw_connections: bool = True,
    ) -> np.ndarray:
        """
        Draw the pose by connecting keypoints with lines and highlighting keypoints.

        Args:
            frame (np.ndarray): The frame to draw on.
            keypoint_color (tuple[int, int, int]): Color for keypoints.
            keypoint_radius (int): Radius of keypoints.
            connection_color (tuple[int, int, int]): Color for connections.
            connection_thickness (int): Thickness of connection lines.
            draw_keypoints (bool): Whether to draw keypoints. Defaults to True.
            draw_connections (bool): Whether to draw connections. Defaults to True.

        Returns:
            np.ndarray: Frame with the pose drawn.
        """
        keypoints = {kp.name: kp.asint() for kp in self.keypoints}
        frame = frame.copy()

        # Draw connections
        if draw_connections:
            for conn in self.CONNECTIONS:
                if conn[0] in keypoints and conn[1] in keypoints:
                    cv2.line(frame, keypoints[conn[0]], keypoints[conn[1]],
                             color = connection_color, thickness = connection_thickness)

        # Draw keypoints
        if draw_keypoints:
            for keypoint in self.keypoints:
                frame = keypoint.draw(frame, color = keypoint_color, radius = keypoint_radius)

        return frame


class PlayersPoses:
    """
    Represents and manages the poses of multiple players,
    allowing annotation and visualization.
    """
    def __init__(self, players_poses: List[Pose]):
        if not all(isinstance(player_pose, Pose) for player_pose in players_poses):
            raise ValueError("All elements must be instances of the Pose class.")
        self.players_poses = players_poses

    def __iter__(self):
        """Make PlayersPoses iterable."""
        return iter(self.players_poses)
    
    def serialize(self) -> dict:
        """Serialize the poses for saving or debugging."""
        return {"poses": [pose.serialize() for pose in self.players_poses]}
    
    def get_pose_by_index(self, index: int) -> Pose:
        """Get a pose by its index."""
        if index < 0 or index >= len(self.players_poses):
            logger.error(f"Index {index} is out of range.")
            return None
        else:
            logger.debug(f"Getting pose at index {index}")
            return self.players_poses[index]
        
    def remove_pose_by_index(self, index: int):
        if index < 0 or index >= len(self.players_poses):
            logger.error(f"Index {index} is out of range.")
        else:
            logger.debug(f"Removing pose at index {index}")
            del self.players_poses[index]

    def add_pose(self, pose: Pose):
        if isinstance(pose, Pose):
            self.players_poses.append(pose)
        else:
            logger.error(f"Invalid pose type: {type(pose)}")

    def clear_poses(self):
        self.players_poses = []

    def get_frame_keypoints(self) -> List[dict]:
        """ Get the keypoints of all player poses in the frame. """
        return [kp.serialize() for pose in self.players_poses for kp in pose.keypoints]

    def annotate_frame(
        self,
        frame: np.ndarray,
        keypoint_color: tuple[int, int, int] = (255, 255, 0),
        keypoint_radius: int = 3,
        connection_color: tuple[int, int, int] = (0, 255, 255),
        connection_thickness: int = 2,
        draw_keypoints: bool = True,
        draw_connections: bool = True,
    ) -> np.ndarray:
        """
        Annotate the frame with player pose keypoints and connections.

        Args:
            frame (np.ndarray): The current video frame.
            keypoint_color (tuple[int, int, int], optional): RGB color for keypoints. Defaults to (255, 255, 0).
            keypoint_radius (int, optional): Radius of keypoints. Defaults to 3.
            connection_color (tuple[int, int, int], optional): RGB color for connections. Defaults to (0, 255, 255).
            connection_thickness (int, optional): Thickness of connection lines. Defaults to 2.
            draw_keypoints (bool, optional): Whether to draw keypoints. Defaults to True.
            draw_connections (bool, optional): Whether to draw connections. Defaults to True.

        Returns:
            np.ndarray: The frame with annotations.
        """
        try:
            for player_pose in self.players_poses:
                frame = player_pose.draw(
                    frame,
                    keypoint_color=keypoint_color,
                    keypoint_radius=keypoint_radius,
                    connection_color=connection_color,
                    connection_thickness=connection_thickness,
                    draw_keypoints=draw_keypoints,
                    draw_connections=draw_connections,
                )
            return frame
        except Exception as e:
            logger.error(f"Error annotating frame: {e}")
            return frame
    

class PoseTracker:

    """
    Tracker for multiple players' poses in a video.
    """
    
    CONF = 0.25
    IOU = 0.7

    def __init__(
            self,
            model_path: str,
            batch_size: int = 4,
            frame_rate: int = 30,
            image_size: int = 1280
            ):
        
        """
        Initialize PoseTracker with YOLOv8 model and other configurations.

        Args:
            model_path (str): Path to the YOLO model.
            batch_size (int): Batch size for processing frames.
            frame_rate (int): Frame rate of the video.
        """
        self.model = YOLO(model_path).to(device)
        self.batch_size = batch_size
        self.frame_rate = frame_rate
        self.image_size = image_size
        self.court_keypoints = None

    def set_court_zone(self, court_keypoints: List[tuple[float, float]]):
        """
        Set court zone keypoints for tracking.

        Args:
            court_keypoints (List[tuple[float, float]]): List of court keypoints.
        """
        self.court_keypoints = court_keypoints

    def run_tracker(self, frames: List[np.ndarray]) -> List[PlayersPoses]:
        """
        Run the pose tracker on a batch of frames.

        Args:
            frames (List[np.ndarray]): List of video frames.

        Returns:
            List[List[Pose]]: Detected poses for each frame.
        """
        
        logger.info("Running poses tracker on %d frames...", len(frames))

        all_poses = []

        try:
            h_frame, w_frame = frames[0].shape[:2]
            ratio_x = w_frame / self.image_size
            ratio_y = h_frame / self.image_size

            for i in range(0, len(frames), self.batch_size):
                batch = frames[i:i+self.batch_size]
                # Preprocess the batch for YOLO inference
                processed_batch = [
                    Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize(
                        (self.image_size, self.image_size)
                    ) for frame in batch
                ]
                results = self.model.predict(processed_batch, conf=self.CONF, iou=self.IOU, imgsz = self.image_size, classes=[0])
                
                for result in results:
                    poses = []
                    detections = result.keypoints.xy  # Keypoints for all detected players (shape: [num_players, 13, 2])

                    if detections.ndim == 2:  # Case: Single player detected, add batch dimension
                        detections = detections.unsqueeze(0)

                    for detection in detections:  # Iterate over each player's keypoints
                        keypoints = [
                            Keypoint(
                                id=i,
                                name=Pose.KEYPOINTS_NAMES[i],
                                xy=(
                                    kp[0].item() * ratio_x,
                                    kp[1].item() * ratio_y
                                )
                            )
                            for i, kp in enumerate(detection)
                        ]
                        poses.append(Pose(keypoints))  # Create a Pose object for the player

                    all_poses.append(PlayersPoses(poses))

            logger.info("Players poses tracking completed.")
            return all_poses
        
        except Exception as e:
            logger.error(f"Error running poses tracker: {e}")
            return []

    def save_poses_results(self, predictions: List[List[Pose]], save_path: str):
        logger.info(f"Players poses results successfully saved to {save_path}")

    def load_poses_data(self, file_path: str) -> Optional[List[List[dict]]]:
        logger.info("Loading poses data from %s...", file_path)

    def update_tracker(self, tracking_data: List[List[dict]]) -> List[List[Pose]]:
        logger.info("Updating poses tracker with %d frames...", len(tracking_data))
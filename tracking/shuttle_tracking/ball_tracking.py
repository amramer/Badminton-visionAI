# tools/shuttle_tracking/ball_tracking.py
import json
import os
import numpy as np
import torch
import cv2
import logging
import math
from tqdm import tqdm
import supervision as sv
from ultralytics import YOLO
from collections import deque
from typing import List, Literal, Iterable, Optional, Type
from torch.utils.data import DataLoader, IterableDataset
from PIL import Image, ImageDraw
from utils.io_utils import save_json, load_json
from utils import read_video, save_video, get_logger
from config import load_config

# Local imports from shuttle_tracking
from tracking.shuttle_tracking.models import TrackNet, InpaintNet
from tracking.shuttle_tracking.dataset import BallTrajectoryDataset
from tracking.shuttle_tracking.iterable import BallTrajectoryIterable
from tracking.shuttle_tracking.predict import predict, predict_modified

config = load_config()
logger = get_logger(__name__, log_file="logs/ball_tracking.log", level=logging.DEBUG)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model(
    model_name: Literal["TrackNet", "InpaintNet"],
    seq_len: Optional[int] = None,
    bg_mode: Optional[Literal["", "subtract", "subtract_concat", "concat"]] = None,
) -> torch.nn.Module:
    """
    Create and configure a model by name.

    Parameters:
        model_name (Literal):
            - "TrackNet": Returns a TrackNet model.
            - "InpaintNet": Returns an InpaintNet model.

        seq_len (Optional[int]): Length of the input sequence for TrackNet.
            Required for "TrackNet".

        bg_mode (Optional[Literal]): Background mode for TrackNet.
            Only applicable if model_name is "TrackNet".
            Choices:
                - "": Input consists of L x 3 channels (RGB).
                - "subtract": Input consists of L x 1 channel (Difference frame).
                - "subtract_concat": Input consists of L x 4 channels (RGB + Difference frame).
                - "concat": Input consists of (L+1) x 3 channels (RGB).

    Returns:
        torch.nn.Module: Configured model instance.

    Raises:
        ValueError: If an invalid model_name is provided or required parameters are missing.
    """
    if model_name == "TrackNet":
        if seq_len is None:
            raise ValueError("`seq_len` must be specified for TrackNet.")
        if bg_mode == "subtract":
            model = TrackNet(in_dim=seq_len, out_dim=seq_len).to(device)
        elif bg_mode == "subtract_concat":
            model = TrackNet(in_dim=seq_len * 4, out_dim=seq_len).to(device)
        elif bg_mode == "concat":
            model = TrackNet(in_dim=(seq_len + 1) * 3, out_dim=seq_len).to(device)
        elif bg_mode == "":
            model = TrackNet(in_dim=seq_len * 3, out_dim=seq_len).to(device)
        else:
            raise ValueError(f"Invalid `bg_mode` value: {bg_mode}")
    elif model_name == "InpaintNet":
        if bg_mode is not None or seq_len is not None:
            logger.warning("`bg_mode` and `seq_len` are ignored for InpaintNet.")
        model = InpaintNet().to(device)
    else:
        raise ValueError(f"Invalid `model_name`: {model_name}. Must be 'TrackNet' or 'InpaintNet'.")

    return model.to(device)

def get_ensemble_weight(
    seq_len: int,
    eval_mode: Literal["average", "weight"],
) -> torch.Tensor:
    """
    Get weight for temporal ensemble.

    Parameters:
        seq_len: Length of input sequence
        eval_mode: Mode of temporal ensemble
            Choices:
                - 'average': return uniform weight
                - 'weight': return positional weight

        Returns:
            weight for temporal ensemble
    """

    if eval_mode == 'average':
        weight = torch.ones(seq_len) / seq_len
    elif eval_mode == 'weight':
        weight = torch.ones(seq_len).to(device)
        for i in range(math.ceil(seq_len/2)):
            weight[i] = (i+1)
            weight[seq_len-i-1] = (i+1)
        weight = weight / weight.sum()
    else:
        raise ValueError('Invalid mode')

    return weight.to(device)

def generate_inpaint_mask(pred_dict: dict, th_h: float=30) -> list:
    """
    Generate inpaint mask form predicted trajectory.

    Parameters:
        pred_dict: prediction result
            Format: {'Frame':[], 'X':[], 'Y':[], 'Visibility':[]}
        th_h: height threshold (pixels) for y coordinate

    Returns:
        inpaint mask
    """
    y = np.array(pred_dict['y'])
    vis_pred = np.array(pred_dict['visibility'])
    inpaint_mask = np.zeros_like(y)
    i = 0 # index that ball start to disappear
    j = 0 # index that ball start to appear
    threshold = th_h
    while j < len(vis_pred):
        while i < len(vis_pred)-1 and vis_pred[i] == 1:
            i += 1
        j = i
        while j < len(vis_pred)-1 and vis_pred[j] == 0:
            j += 1
        if j == i:
            break
        elif i == 0 and y[j] > threshold:
            # start from the first frame that ball disappear
            inpaint_mask[:j] = 1
        elif (i > 1 and y[i-1] > threshold) and (j < len(vis_pred) and y[j] > threshold):
            inpaint_mask[i:j] = 1
        else:
            # ball is out of the field of camera view
            pass
        i = j

    return inpaint_mask.tolist()

class Ball:
    """
    Represents a ball in a video frame.

    Attributes:
        frame (int): Frame number.
        xy (tuple[float, float]): Coordinates of the ball.
        visibility (Literal[1, 0]): Visibility of the ball.
    """
    def __init__(
        self,
        frame: int,
        xy: tuple[float, float],
        visibility: Literal[1,0],
        projection: Optional[tuple[int, int]] = None 
    ):
        self.frame = frame
        self.xy = xy
        self.visibility = visibility
        self.projection = projection

    def serialize(self) -> dict:
        """Serialize the ball for saving or debugging."""
        return {"frame": self.frame, "xy": self.xy, "visibility": self.visibility}

    def asint(self) -> tuple[int, int]:
        """Return the coordinates as integers."""
        return tuple(map(int, self.xy))

    def draw(self, frame: np.ndarray, color: tuple[int, int, int], radius: int = 5) -> np.ndarray:
        """
        Draw the ball on the given frame.

        Args:
            frame (np.ndarray): The frame to draw on.
            color (tuple[int, int, int]): Color of the ball.
            radius (int): Radius of the ball circle.

        Returns:
            np.ndarray: Frame with the ball drawn.
        """
        cv2.circle(frame, self.asint(), radius=radius, color=color, thickness=2)
        return frame

    def draw_projection(self, frame: np.ndarray) -> np.ndarray:
        """
        Draws the ball's projected position on the frame using configured styling.
        
        Args:
            frame: The video frame to draw on (modified in-place)
            
        Returns:
            The modified frame with ball projection drawn
        """
        # Get styling configuration
        cfg = config["side_court"]["ball_projection"]
        # Draw projection circle
        cv2.circle(
            frame,
            self.projection,
            cfg["radius"],
            tuple(cfg["circle_color"]),  
            -1
        )
            
        return frame

class BallTracker:
    """
    Tracker for a single ball in a video.

    Attributes:
        tracking_model_path (str): Path to the TrackNet model checkpoint.
        inpaint_model_path (str): Path to the InpaintNet model checkpoint (optional).
        batch_size (int): Batch size for model inference.
        frame_rate (int): Frame rate of the video being processed.
    """

    TRAJECTORY_LENGTH = 8
    HEIGHT = 360
    WIDTH = 640
    SIGMA = 2.5
    DEVICE = device  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EVAL_MODE = "weight"

    def __init__(
        self,
        tracking_model_path: str,
        inpaint_model_path: Optional[str] = None,
        batch_size: int = 4,
        frame_rate: int = 30
    ):
        self.tracking_model_path = tracking_model_path
        self.inpaint_model_path = inpaint_model_path
        self.batch_size = batch_size
        self.frame_rate = frame_rate

        # Load the TrackNet model
        self.tracknet = self._load_tracknet()

        # Load the InpaintNet model if provided
        self.inpaintnet = self._load_inpaintnet() if inpaint_model_path else None

        self.DELTA_T: float = 1 / math.sqrt(self.HEIGHT**2 + self.WIDTH**2)
        self.COOR_TH = self.DELTA_T * 50

    def _load_tracknet(self) -> TrackNet:
        """Load the TrackNet model from the checkpoint."""
        checkpoint = torch.load(self.tracking_model_path, map_location=self.DEVICE)
        tracknet = get_model("TrackNet", self.TRAJECTORY_LENGTH, checkpoint["param_dict"]["bg_mode"])
        tracknet.load_state_dict(checkpoint["model"])
        tracknet.eval().to(self.DEVICE)
        return tracknet

    def _load_inpaintnet(self) -> Optional[InpaintNet]:
        """Load the InpaintNet model from the checkpoint."""
        checkpoint = torch.load(self.inpaint_model_path, map_location=self.DEVICE)
        inpaintnet = get_model("InpaintNet")
        inpaintnet.load_state_dict(checkpoint["model"])
        inpaintnet.eval().to(self.DEVICE)
        return inpaintnet

    def run_tracker(self,frames: List[np.ndarray]) -> List[Ball]:
        """
        Predict ball positions and visibility for each frame.

        Args:
            frames (List[np.ndarray]): List of video frames.

        Returns:
            List[Ball]: List of detected ball objects for each frame.
        """
        total_frames = len(frames)
        self.video_height, self.video_width = frames[0].shape[:2]
        seq_len = self.TRAJECTORY_LENGTH
        w_scaler, h_scaler = self.video_width / self.WIDTH, self.video_height / self.HEIGHT           
        img_scaler = (w_scaler, h_scaler)

        median_max_sample_num = 2400
        median = None

        # Initialize buffer and output dictionary
        tracknet_pred_dict = {
            'frame': [],
            'x': [],
            'y': [],
            'visibility': [],
            'inpaint_mask': [],
            'img_scaler': img_scaler,
            'img_shape': (self.video_width, self.video_height),
        }

        # Data loader for sequence preparation
        data_loader = DataLoader(
            BallTrajectoryIterable(
                seq_len=seq_len,
                sliding_step=1,
                data_mode="heatmap",
                bg_mode="concat",
                frames = frames,
                HEIGHT=self.HEIGHT,
                WIDTH=self.WIDTH,
                SIGMA=self.SIGMA,
                IMG_FORMAT="png",
                median=median,
                median_range=median_max_sample_num
            ),
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )

        # Predict using TrackNet
        try:

            logger.info("Running TrackNet prediction...")
            num_sample, sample_count = total_frames - seq_len + 1, 0
            buffer_size = seq_len - 1
            sample_indices = torch.arange(seq_len).to(self.DEVICE)
            frame_indices = torch.arange(seq_len-1, -1, -1).to(self.DEVICE)
            y_pred_buffer = torch.zeros(
                (buffer_size, seq_len, self.HEIGHT, self.WIDTH),
                dtype=torch.float32,
            ).to(self.DEVICE)
            weight = get_ensemble_weight(seq_len, self.EVAL_MODE).to(self.DEVICE)

            for batch in tqdm(data_loader, total=total_frames // seq_len):
                batch = batch.float().to(self.DEVICE)
                with torch.no_grad():
                    y_pred = self.tracknet(batch).to(self.DEVICE)

                y_pred_buffer = torch.cat((y_pred_buffer, y_pred.to(self.DEVICE)), dim=0).to(self.DEVICE)
                ensemble_y_pred = torch.empty((0, 1, self.HEIGHT, self.WIDTH), dtype=torch.float32).to(self.DEVICE)

                for sample_i in range(batch.shape[0]):
                    if sample_count < buffer_size:
                        y_pred = (y_pred_buffer[sample_indices + sample_i, frame_indices].sum(0) / (sample_count + 1)).to(self.DEVICE)
                    else:
                        y_pred = ((y_pred_buffer[sample_indices + sample_i, frame_indices] * weight[:, None, None]).sum(0)).to(self.DEVICE)

                    ensemble_y_pred = torch.cat((ensemble_y_pred, y_pred.reshape(1, 1, self.HEIGHT, self.WIDTH)), dim=0).to(self.DEVICE)
                    sample_count += 1

                    if sample_count == num_sample:
                        y_zero_pad = torch.zeros((buffer_size, seq_len, self.HEIGHT, self.WIDTH), dtype=torch.float32).to(self.DEVICE)
                        y_pred_buffer = torch.cat((y_pred_buffer, y_zero_pad), dim=0).to(self.DEVICE)
                        for frame_i in range(1, seq_len):
                            y_pred = (y_pred_buffer[sample_indices + sample_i + frame_i, frame_indices].sum(0) / (seq_len - frame_i)).to(self.DEVICE)
                            ensemble_y_pred = torch.cat((ensemble_y_pred, y_pred.reshape(1, 1, self.HEIGHT, self.WIDTH)), dim=0).to(self.DEVICE)

                tmp_pred = predict_modified(
                    y_pred=ensemble_y_pred,
                    img_scaler=img_scaler,
                    WIDTH=self.WIDTH,
                    HEIGHT=self.HEIGHT,
                )

                for key in tmp_pred.keys():
                    tracknet_pred_dict[key].extend(tmp_pred[key])

                y_pred_buffer = y_pred_buffer[-buffer_size:].to(self.DEVICE)

        except Exception as e:
            logger.error(f"Failed to run TrackNet prediction: {e}")
        # If inpainting is enabled, refine predictions
        if self.inpaintnet:
            logger.info("Refining predictions using InpaintNet...")
            tracknet_pred_dict = self._refine_predictions(tracknet_pred_dict, img_scaler)

        # Convert predictions to Ball objects
        try:

            logger.info("Converting predictions to Ball objects...")
            ball_detections = []
            for frame_idx in range(total_frames):
                if frame_idx in tracknet_pred_dict["Frame"]:
                    idx = tracknet_pred_dict["Frame"].index(frame_idx)
                    ball_detections.append(
                        Ball(
                            frame=frame_idx,
                            xy=(tracknet_pred_dict["X"][idx], tracknet_pred_dict["Y"][idx]),
                            visibility = tracknet_pred_dict["Visibility"][idx],
                        )
                    )
                else:
                    # If the frame is not in the predictions, set visibility to 0
                    # and xy to (0, 0)
                    logger.warning(f"Frame {frame_idx} not found in predictions")
                    ball_detections.append(
                        Ball(frame=frame_idx, xy=(0, 0), visibility = 0)
                    )

            return ball_detections

        except Exception as e:
            logger.error(f"Failed to convert predictions to Ball objects: {e}")

        return []

    def _refine_predictions(self, tracknet_pred_dict: dict, img_scaler: tuple[float, float]) -> dict:
        """
        Refine predictions using InpaintNet.

        Args:
            tracknet_pred_dict (dict): Predictions from TrackNet.
            img_scaler (tuple[float, float]): Scale factors for x and y dimensions.

        Returns:
            dict: Refined predictions.
        """
        try:
            tracknet_pred_dict["inpaint_mask"] = generate_inpaint_mask(tracknet_pred_dict, th_h = self.video_height * 0.05)
            dataset = BallTrajectoryDataset(
                seq_len=self.TRAJECTORY_LENGTH,
                sliding_step=1,
                data_mode="coordinate",
                pred_dict=self.modify_pred_dict(tracknet_pred_dict),
                HEIGHT=self.HEIGHT,
                WIDTH=self.WIDTH,
                SIGMA=self.SIGMA,
                IMG_FORMAT="png"
            )

            refined_dict = {
                "Frame": [],
                "X": [],
                "Y": [],
                "Visibility": []
            }
            data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)

            weight = get_ensemble_weight(self.TRAJECTORY_LENGTH, self.EVAL_MODE).to(self.DEVICE)
            num_sample, sample_count = len(dataset), 0
            buffer_size = self.TRAJECTORY_LENGTH - 1
            sample_indices = torch.arange(self.TRAJECTORY_LENGTH).to(self.DEVICE)
            frame_indices = torch.arange(self.TRAJECTORY_LENGTH-1, -1, -1).to(self.DEVICE)
            coor_inpaint_buffer = torch.zeros(
                (buffer_size, self.TRAJECTORY_LENGTH, 2),
                dtype=torch.float32
            ).to(self.DEVICE)

            for (i, coord_pred, mask) in tqdm(data_loader):
                coord_pred = coord_pred.float().to(self.DEVICE)
                mask = mask.float().to(self.DEVICE)
                i = i.to(self.DEVICE)  # Ensure `i` is also on the same device
                with torch.no_grad():
                    refined_pred = self.inpaintnet(coord_pred, mask).to(self.DEVICE)
                    refined_pred = refined_pred * mask + coord_pred * (1 - mask)

                th_mask = (
                    (
                        (refined_pred[:, :, 0] < self.COOR_TH)
                        &
                        (refined_pred[:, :, 1] < self.COOR_TH)
                    )
                )
                refined_pred[th_mask] = 0.

                coor_inpaint_buffer = torch.cat((coor_inpaint_buffer, refined_pred), dim=0).to(self.DEVICE)
                ensemble_i = torch.empty((0, 1, 2), dtype=torch.float32).to(self.DEVICE)
                ensemble_coor_inpaint = torch.empty((0, 1, 2), dtype=torch.float32).to(self.DEVICE)

                for sample_i in range(coord_pred.shape[0]):
                    if sample_count < buffer_size:
                        refined_pred = coor_inpaint_buffer[
                            sample_indices + sample_i,
                            frame_indices].sum(0) / (sample_count + 1)
                    else:
                        refined_pred = (
                            coor_inpaint_buffer[
                                sample_indices + sample_i,
                                frame_indices] * weight[:, None]
                        ).sum(0)

                    ensemble_i = torch.cat((ensemble_i, i[sample_i][0].view(1, 1, 2)), dim=0).to(self.DEVICE)
                    ensemble_coor_inpaint = torch.cat(
                        (ensemble_coor_inpaint, refined_pred.view(1, 1, 2)),
                        dim=0).to(self.DEVICE)
                    sample_count += 1

                    if sample_count == num_sample:
                        coor_zero_pad = torch.zeros(
                            (buffer_size, self.TRAJECTORY_LENGTH, 2),
                            dtype=torch.float32).to(self.DEVICE)
                        coor_inpaint_buffer = torch.cat((coor_inpaint_buffer, coor_zero_pad), dim=0).to(self.DEVICE)

                        for frame_i in range(1, self.TRAJECTORY_LENGTH):
                            refined_pred = coor_inpaint_buffer[
                                sample_indices + sample_i + frame_i,
                                frame_indices].sum(0) / (self.TRAJECTORY_LENGTH - frame_i)
                            ensemble_i = torch.cat(
                                (ensemble_i, i[-1][frame_i].view(1, 1, 2)),
                                dim=0).to(self.DEVICE)
                            ensemble_coor_inpaint = torch.cat(
                                (ensemble_coor_inpaint, refined_pred.view(1, 1, 2)),
                                dim=0).to(self.DEVICE)

                th_mask = (
                    (ensemble_coor_inpaint[:, :, 0] < self.COOR_TH)
                    &
                    (ensemble_coor_inpaint[:, :, 1] < self.COOR_TH)
                )
                ensemble_coor_inpaint[th_mask] = 0.

                tmp_pred = predict(ensemble_i, c_pred=ensemble_coor_inpaint, img_scaler=img_scaler, WIDTH=self.WIDTH, HEIGHT=self.HEIGHT)

                for key in tmp_pred.keys():
                    refined_dict[key].extend(tmp_pred[key])

                coor_inpaint_buffer = coor_inpaint_buffer[-buffer_size:].to(self.DEVICE)

            return refined_dict
        except Exception as e:
            logger.error(f"Failed to refine predictions using InpaintNet: {e}")
            return None

    def draw_traj(
            self, 
            img: np.ndarray, 
            traj: deque, 
            color: tuple[int, int, int], 
            radius: int = 3
        ) -> np.ndarray:
        """Draw trajectory on the image with configurable styling.

        Args:
            img: Image array (H, W, C)
            traj: Trajectory points to draw
            color: BGR color tuple for trajectory
            radius: Radius of trajectory points

        Returns:
            Image with trajectory drawn
        """
        img = img.copy()
        for i in range(len(traj)):
            if traj[i] is not None:
                cv2.circle(
                    img, 
                    tuple(map(int, traj[i])), 
                    radius=radius, 
                    color=color, 
                    thickness=1
                )
        return img

    def annotate_frames(
            self, 
            frames: List[np.ndarray], 
            ball_detections: List[Ball], 
            trail_length: int = 8,
            ball_color: str = 'YELLOW',
            trail_color: str = 'CYAN',
            ball_radius: int = 5
        ) -> List[np.ndarray]:
        """
        Draw ball trajectories on multiple frames with configurable styling.

        Args:
            frames: List of video frames
            ball_detections: List of detected ball objects
            trail_length: Length of trajectory to draw (default: 8)
            ball_color: Color name for ball marker (default: "YELLOW")
            trail_color: Color name for ball trail (default: "CYAN")
            ball_radius: Radius of ball marker in pixels (default: 5)

        Returns:
            List of frames with drawn trajectories
        """
        # Color name to BGR mapping
        color_map = {
            'YELLOW': (0, 255, 255),
            'CYAN': (255, 255, 0),
            'RED': (0, 0, 255),
            'GREEN': (0, 255, 0),
            'BLUE': (255, 0, 0)
        }
        
        ball_bgr = color_map.get(ball_color.upper(), (0, 255, 255))
        trail_bgr = color_map.get(trail_color.upper(), (255, 255, 0))

        pred_queue = deque(maxlen=trail_length)
        output_frames = []

        for frame, ball_detection in zip(frames, ball_detections):
            frame = frame.copy()
            
            # Draw current ball position
            if ball_detection.visibility == 1:
                cv2.circle(
                    frame,
                    ball_detection.asint(),
                    radius=ball_radius,
                    color=ball_bgr,
                    thickness=2
                )
                pred_queue.appendleft(list(ball_detection.xy))
            else:
                pred_queue.appendleft(None)

            # Draw trajectory
            output_frames.append(self.draw_traj(frame, pred_queue, trail_bgr, max(1, ball_radius-2)))

        return output_frames

    def modify_pred_dict(self, pred_dict: dict):

        mapping = {
            "X": "x",
            "Y": "y",
            "Visibility": "visibility",
            "Inpaint_Mask": "inpaint_mask",
            "Img_scaler": "img_scaler",
            "Img_shape": "img_shape",
        }

        return {
            k: pred_dict[v]
            for k, v in mapping.items()
        }

    def save_tracking_results(self, detections: List[Ball], save_path: str) -> bool:
        """
        Serialize and save ball tracking results to JSON file.
        
        Args:
            detections: List of Ball objects to save
            save_path: Output file path for JSON data
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Validate input
            if not detections:
                logger.warning("⚠️ Empty detections - no data to save")
                return False
                
            logger.info(f"💾 Saving ball tracking to {save_path}")
            logger.debug(f"  - Input contains {len(detections)} detections")
            
            # Prepare serializable data structure
            results = []
            total_visible = 0
            
            with logger.context("Serializing ball data"):
                for ball in detections:
                    ball_data = {
                        "frame": ball.frame,
                        "xy": ball.xy,
                        "visibility": ball.visibility,
                        "projection": ball.projection
                    }
                    results.append(ball_data)
                    if ball.visibility == 1:
                        total_visible += 1
                    
            # Save to JSON file
            with logger.context(f"Writing to {save_path}"):
                save_json(results, save_path)
                
                # Calculate file stats
                file_size = os.path.getsize(save_path) / (1024 * 1024)  # in MB
                logger.info(
                    f"✅ Successfully saved ball tracking:\n"
                    f"  - Total detections: {len(results):,}\n"
                    f"  - Visible balls: {total_visible:,}\n"
                    f"  - File size: {file_size:.2f}MB"
                )
                return True
                
        except IOError as e:
            logger.error(f"❌ File system error saving to {save_path}: {str(e)}")
            return False
        except Exception as e:
            logger.error(
                f"❌ Unexpected error saving ball tracking:\n"
                f"  - Path: {save_path}\n"
                f"  - Error: {str(e)}"
            )
            return False

    def load_tracking_data(self, file_path: str) -> Optional[List[dict]]:
        """
        Load ball tracking data from JSON file with validation.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            List of dicts containing ball data, or None if invalid
        """
        try:
            logger.info(f"🔍 Loading ball tracking from {file_path}")
            
            # Check if file exists
            if not os.path.exists(file_path):
                logger.error("❌ File not found")
                return None
            
            # Load file   
            data = load_json(file_path)
                
            # Validate data structure
            if not isinstance(data, list):
                logger.error("❌ Invalid data format - expected list")
                return None
                
            logger.info(f"✅ Loaded {len(data)} ball detections")
            return data
            
        except json.JSONDecodeError:
            logger.error("❌ Invalid JSON format")
            return None
        except Exception as e:
            logger.error(f"❌ Unexpected error loading data: {str(e)}")
            return None

    def update_tracker(self, tracking_data: List[dict]) -> List[Ball]:
        """
        Convert loaded tracking data into Ball objects.
        
        Args:
            tracking_data: List of dicts containing serialized ball data
            
        Returns:
            List of Ball objects
        """
        if not tracking_data:
            logger.warning("⚠️ Empty tracking data")
            return []
            
        try:
            logger.info(f"🔄 Converting {len(tracking_data)} detections to Ball objects")
            balls = []
            valid_detections = 0
            
            for item in tracking_data:
                # Validate required fields
                if not all(k in item for k in ['frame', 'xy', 'visibility']):
                    logger.warning("⚠️ Skipping invalid ball data")
                    continue
                    
                balls.append(Ball(
                    frame=item['frame'],
                    xy=tuple(item['xy']),
                    visibility=item['visibility'],
                    projection=tuple(item['projection']) if item.get('projection') else None
                ))
                if item['visibility'] == 1:
                    valid_detections += 1
                    
            logger.info(
                f"✅ Created {len(balls)} Ball objects\n"
                f"  - Visible balls: {valid_detections}"
            )
            return balls
            
        except Exception as e:
            logger.error(f"❌ Conversion failed: {str(e)}")
            return []
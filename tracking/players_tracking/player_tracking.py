import json
import os
import numpy as np
import torch
import logging
import cv2
from ultralytics import YOLO
import supervision as sv
from collections import deque
from typing import List, Optional
from segment_anything import sam_model_registry, SamPredictor
from typing import List, Optional
from utils.io_utils import save_json
from config import load_config
from utils import get_logger

# Initialize logger
logger = get_logger(__name__)

# Load SAM model for player segmentation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = load_config()
sam_model_type = config["players"].get("sam_model_type", "vit_h")
sam_model = sam_model_registry[sam_model_type](checkpoint=config["players"]["sam_model"]).to(device=device)
mask_predictor = SamPredictor(sam_model)

class Player:
    """Player class for storing player information."""
    net_line_y = None
    def __init__(self, 
                 detection: sv.Detections, 
                 player_id: Optional[int] = None, 
                 player_name: Optional[str] = "Unknown", 
                 projection: Optional[tuple[float, float]] = None
    ):
        
        self.detection = detection
        self.projection = projection
        self.xyxy = detection.xyxy[0]
        self.id = player_id
        self.name = player_name
        self.class_id = int(detection.class_id[0])
        self.confidence = float(detection.confidence[0])
        self.player_mapping = config.get("player_mapping", {})
        
        self.assign_player()
        logger.debug(f"👤 Created Player {self.id} ({self.name}) at {self.feet}")

    @property
    def top_left(self):
        return tuple(self.xyxy[:2])

    @property
    def bottom_right(self):
        return tuple(self.xyxy[2:])

    @property
    def height(self):
        return self.bottom_right[1] - self.top_left[1]

    @property
    def width(self):
        return self.bottom_right[0] - self.top_left[0]

    @property
    def feet(self) -> tuple[int, int]:
        return (
            int(round(self.top_left[0] + self.width / 2)),
            int(round(self.bottom_right[1])),
        )

    def assign_player(self):
        """Assign player ID and name."""
        try:
            if Player.net_line_y is None:
                logger.warning("⚠️ Net line position not set - using default mapping")
                self.id = 0
                self.name = "Unknown"
                return

            if not self.player_mapping:
                raise ValueError("Player mapping missing in config")

            if self.feet[1] > Player.net_line_y:
                self.id = self.player_mapping.get("player_1", {}).get("id", 1)
                self.name = self.player_mapping.get("player_1", {}).get("name", "Player 1")
            else:
                self.id = self.player_mapping.get("player_2", {}).get("id", 2)
                self.name = self.player_mapping.get("player_2", {}).get("name", "Player 2")
                
                logger.debug(f"  ↳ Assigned {self.name} (ID: {self.id}) at position (X:{self.feet[0]}, Y:{self.feet[1]})")
                
        except Exception as e:
            logger.error(f"❌ Player assignment error: {str(e)}")
            self.id = None
            self.name = "Error"

    def generate_mask(self, frame: np.ndarray) -> np.ndarray:
        """"
        Generate mask for the player using the bounding box.
        
        Args:
            frame (np.ndarray): The current video frame.
            
        Returns:
            np.ndarray: The mask for the player.    
        """
        try:

            x1, y1 = self.top_left
            x2, y2 = self.bottom_right

            # Ensure valid box
            if x2 <= x1 or y2 <= y1:
                logger.debug(f"⚠️ Invalid bbox for {self.name}: {(x1,y1,x2,y2)}")
                return None

            box = np.array([x1, y1, x2, y2], dtype=np.float32)

            masks, scores, _ = mask_predictor.predict(
                box=box,
                multimask_output=True
            )

            if masks is None or len(masks) == 0:
                return None

            # Pick best mask by SAM score
            best_idx = int(np.argmax(scores)) if scores is not None else 0
            best_mask = masks[best_idx]  # (H, W) bool

            logger.debug(f"  ↳ SAM mask for {self.name} | best_idx={best_idx} | score={float(scores[best_idx]) if scores is not None else 'NA'}")
            return best_mask

        except Exception as e:
            logger.error(f"❌ Mask generation failed for {self.name}: {str(e)}", exc_info=True)
            return None

    def draw(self, 
            frame: np.ndarray, 
            mask_annotator: sv.MaskAnnotator,
            label_annotator: sv.LabelAnnotator, 
            ellipse_annotator: sv.EllipseAnnotator,
            generate_masks: bool = False,
            show_confidence: bool = True 
    ) -> np.ndarray:
        """
        Annotate a single player on the frame with optional mask generation.
        
        Args:
            frame: The current video frame
            mask_annotator: Annotator for masks
            label_annotator: Annotator for labels 
            ellipse_annotator: Annotator for ellipses
            generate_masks: Whether to generate and draw masks (default: False)
            show_confidence: Whether to show confidence score in label (default: True)
        """
        try:

            label = f"P{self.id} | {self.name}"
                
            frame = ellipse_annotator.annotate(frame, self.detection)
            frame = label_annotator.annotate(frame, self.detection, labels=[label])

            if generate_masks:
                mask = self.generate_mask(frame)
                if mask is not None:
                    detections = sv.Detections(
                        xyxy=sv.mask_to_xyxy(masks=np.array([mask])),
                        mask=np.array([mask])
                    )
                    frame = mask_annotator.annotate(scene=frame, detections=detections)
            return frame
        except Exception as e:
            logger.error(f"❌ Annotation failed for {self.name}: {str(e)}")
            return frame

    def draw_projection(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw the player's projected position on the side court view.

        Annotates the frame with a circle and player ID at their projected
        position on the side court visualization.

        Args:
            frame (np.ndarray): The current video frame to be annotated.

        Returns:
            np.ndarray: The frame with the player's side court projection added.

        Raises:
            ValueError: If projection coordinates haven't been calculated.
        """
        try:
            if not self.projection:
                return frame
            
            px, py = self.projection

            cfg = config["side_court"]["player_projection"]
            cv2.circle(
                frame, 
                (int(round(px)), int(round(py))), 
                int(cfg["radius"]),
                tuple(cfg["circle_color"]), 
                -1
            )
            cv2.putText(frame, str(self.id), (int(round(px)), int(round(py)) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,cfg["font_scale"], cfg["font_color"], 2)
            return frame
        except Exception as e:
            logger.error(f"❌ Player projection drawing failed: {str(e)}")
            return frame

class Players:
    """
    A class representing a list of players.

    Attributes:
        players (List[Player]): A list of Player objects.
    """
    def __init__(self, players: List[Player]):
        if not all(isinstance(player, Player) for player in players):
            logger.error("❌ Invalid player objects in list")
            raise ValueError("All elements must be Player instances")
        self.players = players
        logger.debug(f"👥 Created Players list with {len(players)} players")

    def __iter__(self):
        return iter(self.players)

    def annotate_frame(
        self,
        frame: np.ndarray,
        generate_masks: bool = False,
        mask_color: sv.Color = sv.Color.RED,
        mask_color_lookup: sv.ColorLookup = sv.ColorLookup.INDEX,
        label_color: sv.Color = sv.Color.BLUE,
        label_text_scale: float = 0.7,
        ellipse_thickness: int = 2,
        ellipse_color: sv.Color = sv.Color.BLUE,
        show_confidence: bool = True
    ) -> np.ndarray:
        
        """
        Annotate the frame with players information and masks.
        
        Args:
            frame (np.ndarray): The current video frame.
            mask_color (sv.Color, optional): Color for the mask annotations. Defaults to RED.
            mask_color_lookup (sv.ColorLookup, optional): Color lookup for masks. Defaults to INDEX.
            label_color (sv.Color, optional): Color for the label text. Defaults to BLUE.
            label_text_scale (float, optional): Scale for the label text. Defaults to 0.8.
            ellipse_thickness (int, optional): Thickness of ellipse annotations. Defaults to 2.
            ellipse_color (sv.Color, optional): Color for ellipse annotations. Defaults to BLUE.
            
        Returns:
            np.ndarray: The frame with annotations.
        """
        try:
            mask_annotator = sv.MaskAnnotator(color=mask_color, color_lookup=mask_color_lookup)
            label_annotator = sv.LabelAnnotator(
                text_scale=label_text_scale,
                text_position=sv.Position.CENTER,
                color=label_color
            )
            ellipse_annotator = sv.EllipseAnnotator(thickness=ellipse_thickness, color=ellipse_color)

            # Use a clean copy for SAM (no drawings on it)
            sam_rgb = None
            if generate_masks:
                sam_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mask_predictor.set_image(sam_rgb)

            # We'll annotate on this frame
            out = frame

            for player in self.players:
                # First: (optional) mask on the clean image
                if generate_masks and sam_rgb is not None:
                    best_mask = player.generate_mask(sam_rgb)  # (H,W) or None

                    if best_mask is not None:
                        # supervision expects mask shape: (N, H, W)
                        masks_n = best_mask[None, ...].astype(bool)

                        det = sv.Detections(
                            xyxy=sv.mask_to_xyxy(masks=masks_n),
                            mask=masks_n
                        )
                        out = mask_annotator.annotate(scene=out, detections=det)

                # Then: draw ellipse + label (these can be on the same output)
                label = f"P{player.id} | {player.name}"
                out = ellipse_annotator.annotate(out, player.detection)
                out = label_annotator.annotate(out, player.detection, labels=[label])

            return out

        except Exception as e:
            logger.error(f"❌ Frame annotation failed: {str(e)}", exc_info=True)
            return frame

        finally:
            # Reset only once per frame (optional, but clean)
            if generate_masks:
                try:
                    mask_predictor.reset_image()
                except Exception:
                    pass

class PlayerTracker:
    """
    A class representing a player tracker.
    
    Attributes:
        model (YOLO): The YOLO model for object detection.
        byte_track (ByteTrack): The ByteTrack object for player tracking.
        batch_size (int): The batch size for object detection.
        court_keypoints (List[tuple]): The list of court keypoints.
        polygon_zone (PolygonZone): The polygon zone for player tracking.
        detection_history (Dict): The history of detected players.
    """
    CONF = 0.5
    IOU = 0.7
    IMGSZ = 640

    def __init__(self, 
                 model_path: str, batch_size: int = 4, 
                 frame_rate: int = 30
    ):
        logger.info("🔄 Initializing PlayerTracker:")
        logger.debug(f"  - Model path: {model_path}")
        logger.debug(f"  - Batch size: {batch_size}")
        logger.debug(f"  - Frame rate: {frame_rate}")
        
        # Load YOLO model for player detection
        self.model = YOLO(model_path).to(device)
        
        # Intialize ByteTrack for player tracking
        self.byte_track = sv.ByteTrack(frame_rate=frame_rate)

        # Initialize variables
        self.batch_size = batch_size
        self.court_keypoints = None
        self.polygon_zone = None
        self.detection_history = {}

    def set_court_zone(self, court_keypoints: List[tuple]):
        """
        Set court zone keypoints for tracking.
        
        Args:
            court_keypoints (List[tuple]): A list of 32 tuple representing the court's keypoints.
            
        Raises:
            ValueError: If not enough court keypoints are provided.
        """
        try:
            if len(court_keypoints) != 32:
                raise ValueError("Expected 32 court keypoints")
                
            self.court_keypoints = court_keypoints
            polygon_zone = sv.PolygonZone(
                np.concatenate((
                    np.expand_dims(court_keypoints[0], axis=0),
                    np.expand_dims(court_keypoints[4], axis=0),
                    np.expand_dims(court_keypoints[-1], axis=0),
                    np.expand_dims(court_keypoints[-5], axis=0),
                ), axis=0)
            )
            self.polygon_zone = polygon_zone
            # Calculate the position of the net line
            Player.net_line_y = (court_keypoints[15][1] + court_keypoints[16][1]) / 2
            logger.info("✅ Court zone configured")
            logger.debug(f"  - Net line Y: {Player.net_line_y:.2f}")
        except Exception as e:
            logger.error(f"❌ Court zone setup failed: {str(e)}")
            raise

    def smooth_detections(self, tracked_detections: sv.Detections) -> sv.Detections:
        """
        Apply temporal smoothing to player trajectories using moving average.
        
        Maintains a history of previous positions for each player and calculates
        smoothed positions to reduce jitter in tracking results.
        
        Args:
            tracked_detections (sv.Detections): Raw player detections from current frame
            
        Returns:
            sv.Detections: Smoothed player positions with reduced jitter
            
        Note:
            Uses a deque with maxlen=2 to maintain only the most recent positions
            for calculating the moving average
        """
        logger.debug(
            f"🔄 Applying temporal smoothing to {len(tracked_detections)} player tracks "
            f"(History window: {self.detection_history.maxlen if hasattr(self.detection_history, 'maxlen') else 'N/A'})"
        )
        
        smoothed_xyxy = []
        smoothing_effects = []
        
        for i, tracker_id in enumerate(tracked_detections.tracker_id):
            # Initialize history for new tracks
            if tracker_id not in self.detection_history:
                self.detection_history[tracker_id] = deque(maxlen=2)
                logger.debug(f"  🆕 New track ID: {tracker_id}")
            
            # Store current detection and calculate smoothed position
            original_box = tracked_detections.xyxy[i]
            self.detection_history[tracker_id].append(original_box)
            history = np.array(self.detection_history[tracker_id])
            smoothed_box = history.mean(axis=0)
            smoothed_xyxy.append(smoothed_box)
        
        tracked_detections.xyxy = np.array(smoothed_xyxy, dtype=np.float32)
        
        logger.info(
            f"✅ Smoothed {len(smoothed_xyxy)} player positions "
            f"(Avg Δ: {np.mean(smoothing_effects):.2f}px)" if smoothing_effects else ""
        )
        return tracked_detections

    def run_tracker(self, frames: List[np.ndarray]) -> List[List[Player]]:
        """
        Process video frames to detect and track players across the sequence.
        
        Performs batched inference using YOLO model, applies court boundary filtering,
        runs ByteTrack for temporal consistency, and smooths detections.
        
        Args:
            frames (List[np.ndarray]): List of video frames (H,W,3) to process
            
        Returns:
            List[List[Player]]: List of Player objects for each frame (empty list on failure)
        """
        logger.info("🔍 Starting player tracking pipeline")
        logger.debug(
            f"  - Frames: {len(frames)} | "
            f"Batch size: {self.batch_size} | "
            f"Conf: {self.CONF} | "
            f"IoU: {self.IOU}"
        )
        
        predictions = []
        total_players = 0
        total_filtered = 0
        
        try:
            with logger.context(f"Processing {len(frames)} frames"):
                for batch_idx, i in enumerate(range(0, len(frames), self.batch_size), 1):
                    batch = frames[i:i+self.batch_size]
                    
                    with logger.context(f"Batch {batch_idx}/{len(frames)//self.batch_size + 1}"):
                        # Model inference
                        with logger.context("YOLO inference"):
                            results = self.model.predict(
                                batch, 
                                conf=self.CONF,
                                iou=self.IOU,
                                imgsz=self.IMGSZ,
                                classes=[0],
                                verbose=False
                            )
                            logger.debug(f"  🖥️  Batch inference completed ({len(results)} results)")
                        
                        # Process each frame in batch
                        batch_players = []
                        for frame_idx, result in enumerate(results):
                            detections = sv.Detections.from_ultralytics(result)
                            initial_count = len(detections)
                            
                            # Court boundary filtering
                            if self.polygon_zone:
                                detections = detections[self.polygon_zone.trigger(detections)]
                                filtered = initial_count - len(detections)
                                total_filtered += filtered
                                logger.debug(
                                    f"  🏟️  Frame {frame_idx}: "
                                    f"Filtered {filtered}/{initial_count} players "
                                    f"(Kept: {len(detections)})"
                                )
                            
                            # Tracking and smoothing
                            tracked_detections = self.byte_track.update_with_detections(detections)
                            smoothed_detections = self.smooth_detections(tracked_detections)
                            
                            # Create Player objects
                            players = [
                                Player(detection=smoothed_detections[i]) 
                                for i in range(len(smoothed_detections))
                            ]
                            batch_players.append(Players(players))
                            total_players += len(players)
                        
                        predictions.extend(batch_players)
            
            logger.info(
                f"✅ Tracking completed - "
                f"{total_players} player instances across {len(predictions)} frames"
            )
            if total_filtered > 0:
                logger.debug(f"  🚫 Filtered {total_filtered} out-of-court detections")
            return predictions
            
        except Exception as e:
            logger.error(
                f"❌ Tracking pipeline failed after processing {len(predictions)} frames: "
                f"{str(e)}",
                exc_info=True
            )
            return []

    def save_tracking_results(self, predictions: List[List[Player]], save_path: str):
        """
        Serialize and save player tracking results to JSON file.
        
        Args:
            predictions (List[List[Player]]): List of player predictions per frame
            save_path (str): Output file path for JSON data
            
        Raises:
            IOError: If file writing fails
            ValueError: If predictions data is malformed
        """
        try:
            # Validate input
            if not predictions:
                logger.warning("⚠️ Empty predictions - no data to save")
                return
                
            logger.info(f"💾 Saving tracking data to {save_path}")
            logger.debug(f"  - Input contains {len(predictions)} frames")
            
            # Prepare serializable data structure
            results = []
            player_ids = set()
            total_detections = 0
            
            with logger.context("Serializing player data"):
                for frame_idx, frame_predictions in enumerate(predictions, 1):
                    frame_data = []
                    for player in frame_predictions:
                        player_data = {
                            "xyxy": player.xyxy.tolist(),
                            "class_id": player.class_id,
                            "confidence": player.confidence,
                            "player_id": player.id,
                            "player_name": player.name,
                        }
                        frame_data.append(player_data)
                        player_ids.add(player.id)
                        total_detections += 1
                        
                    results.append(frame_data)
                    
                    if frame_idx % 10 == 0 or frame_idx == len(predictions):
                        logger.debug(
                            f"  📊 Frame {frame_idx:04d}/{len(predictions)} "
                            f"({min(frame_idx/len(predictions)*100, 100):.0f}%)"
                        )
            
            # Save to JSON file
            with logger.context(f"Writing to {save_path}"):
                save_json(results, save_path)
                
                # Calculate file stats
                file_size = os.path.getsize(save_path) / (1024 * 1024)  # in MB
                logger.info(
                    f"✅ Successfully saved tracking data:\n"
                    f"  - Frames: {len(predictions):,}\n"
                    f"  - Unique players: {len(player_ids)}\n"
                    f"  - Total detections: {total_detections:,}\n"
                    f"  - File size: {file_size:.2f}MB"
                )
                
        except IOError as e:
            logger.error(f"❌ File system error saving to {save_path}: {str(e)}")
            raise
        except ValueError as e:
            logger.error(f"❌ Invalid tracking data format: {str(e)}")
            raise
        except Exception as e:
            logger.error(
                f"❌ Unexpected error saving tracking data:\n"
                f"  - Path: {save_path}\n"
                f"  - Error: {str(e)}",
                exc_info=True
            )
            raise

    def load_tracking_data(self, file_path: str) -> Optional[List[List[dict]]]:
        """📦 Load tracking data with validation
        
        Args:
            file_path (str): Path to the JSON file containing tracking data.
            
        Returns:
            Optional[List[List[dict]]]: Loaded tracking data or None if the file doesn't exist or is invalid.
        """
        try:
            logger.info(f"🔍 Loading: {file_path}")
            
            # Check if file exists
            if not os.path.exists(file_path):
                logger.error("❌ File not found")
                return None
            
            # Load file   
            with open(file_path, "r") as f:
                data = json.load(f)
                
            # Validate data    
            if not (isinstance(data, list) and all(isinstance(frame, list) for frame in data)):
                logger.error("❌ Invalid data structure")
                return None
                
            logger.info(f"✅ Loaded {len(data)} frames")
            return data
            
        except json.JSONDecodeError:
            logger.error("❌ Invalid JSON format")
            return None
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
            return None


    def update_tracker(self, tracking_data: List[List[dict]]) -> List[List[Player]]:
        """
        🔄 Update file data into Players Objects.
        
        Args:
            tracking_data (List[List[dict]]): The loaded tracking data.

        Returns:
            List[List[Player]]: List of players objects per frame.
        """
        if not tracking_data:
            logger.warning("⚠️ Empty input")
            return []
            
        try:
            logger.info(f"Processing {len(tracking_data)} frames")
            predictions = []
            
            for frame_data in tracking_data:
                frame_players = []
                for data in frame_data:
                    # validate
                    if not all(k in data for k in ['xyxy', 'class_id', 'confidence']):
                        logger.warning("⚠️ Skipping invalid detection")
                        continue
                        
                    detection = sv.Detections(
                        xyxy=np.array([data["xyxy"]], dtype=np.float32),
                        class_id=np.array([data["class_id"]], dtype=np.int32),
                        confidence=np.array([data["confidence"]], dtype=np.float32),
                    )
                    frame_players.append(Player(
                        detection=detection,
                        player_id=data.get("player_id"),
                        player_name=data.get("player_name", "Unknown"),
                    ))
                    
                predictions.append(Players(frame_players))
                
            logger.info(f"✅ Processed {len(predictions)} frames")
            return predictions
            
        except Exception as e:
            logger.error(f"❌ Processing failed: {e}")
            return []
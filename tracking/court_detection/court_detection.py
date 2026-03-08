import cv2
import os
import copy
import torch
import numpy as np
from typing import List, Literal, Iterable, Optional, Type, Tuple
from utils.io_utils import check_file_exists, load_json, save_json
from utils.video import save_video
from torchvision import models
from ultralytics import YOLO

class Keypoint:

    """
    Badminton court keypoint detection in a given video frame

    Attributes:
        id: keypoint unique identifier
        xy: keypoint position coordinates in pixels
    """

    def __init__(self, id: int, xy: Tuple[float, float]):
        self.id = id
        self.xy = xy

    @classmethod
    def from_json(cls, x: dict):
        return cls(**x)
    
    def serialize(self) -> dict:
        return {
            "id": self.id,
            "xy": self.xy,
        }
    
    def asint(self) -> Tuple[int, int]:
        return Tuple(int(v) for v in self.xy)

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw keypoint detection in a given frame with clear ID label
        """
        x, y = self.asint()
        
        # First draw the circle
        cv2.circle(
            frame,
            (x, y),
            radius=8,  # Slightly larger radius
            color=(0, 0, 255),  # Red color for better visibility
            thickness=-1,
        )
        
        # Then draw the text with contrasting background
        text = str(self.id + 1)
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        
        # Draw background rectangle for text
        cv2.rectangle(
            frame,
            (x - 5, y - text_height - 10),
            (x + text_width + 5, y - 5),
            (0, 0, 0),  # Black background
            -1
        )
        
        # Draw the text
        cv2.putText(
            frame, 
            text,
            (x, y - 10),  # Position above the point
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,  # Larger font size
            (255, 255, 255),  # White text
            2,  # Thicker text
            cv2.LINE_AA
        )
        
        return frame

class Keypoints:

    """
    Court multiple keypoint detection in a given video frame
    """

    def __init__(self, keypoints: List[Keypoint]):
        super().__init__()

        self.keypoints = sorted(keypoints, key=lambda x: x.id)
        self.keypoints_by_id = {
            keypoint.id: keypoint
            for keypoint in keypoints
        }

    @classmethod
    def from_json(cls, x: List[dict]) -> "Keypoints":
        return cls(
            keypoints=[
                Keypoint.from_json(keypoint_json)
                for keypoint_json in x
            ]
        )
    
    def serialize(self) -> List[dict]:
        return [
            keypoint.serialize()
            for keypoint in self.keypoints
        ]
    
    def __len__(self) -> int:
        return len(self.keypoints)
    
    def __iter__(self) -> Iterable[Keypoint]:
        return (keypoint for keypoint in self.keypoints)
    
    def __getitem__(self, id: int) -> Keypoint:
        return self.keypoints_by_id[id]

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw court keypoints detection in a given frame 
        """
        for keypoint in self.keypoints:
            frame = keypoint.draw(frame)

        return frame
    
class CourtDetection:
    def __init__(self, config, logger):
        """
        Initialize the CourtDetection class with configuration parameters.

        Parameters:
        - config (dict): Project configuration loaded from config.yaml
        - logger: Logger instance for logging information
        """
        self.logger = logger
        self.config = config
        self.selected_keypoints = []

        # Access video-related configurations
        self.court_video_path = config["video"].get("court_detection_output")

        # Access court keypoints-related configurations
        self.court_keypoints_path = config["court_keypoints"].get("court_keypoints_path")
        self.use_fixed_keypoints = config["court_keypoints"].get("use_fixed_keypoints")
        self.model_type = config["court_keypoints"].get("model_type", "yolo")  # Default to YOLO if not specified

        self.model = None


    def click_event(self, event, x, y, flags, params):
        """
        Mouse click event to capture keypoints manually.

        Parameters:
        - event: Type of mouse event
        - x, y: Coordinates of the mouse click
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selected_keypoints.append((x, y))
            cv2.circle(params['img'], (x, y), 5, (0, 0, 255), -1)
            cv2.putText(params['img'], f"{x},{y}", (x-5, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.imshow('Select Court Keypoints', params['img'])

    def get_fixed_keypoints(self, img):
        """
        Load fixed keypoints from a file or collect them manually.

        Parameters:
        - img: Frame image for manual selection

        Returns:
        - list: List of selected keypoints
        """

        window_name = 'Select Court Keypoints'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        # Get screen dimensions
        screen_width = cv2.getWindowImageRect(window_name)[2]
        screen_height = cv2.getWindowImageRect(window_name)[3]
        
        cv2.resizeWindow(window_name, int(screen_width * 0.8), int(screen_height * 0.8))
        
        # Check if the keypoints file path is provided
        if not self.court_keypoints_path:
            self.logger.warning("No path provided for court keypoints. Please set it in the configuration.")
            return []

        # Check if the keypoints file exists
        if check_file_exists(self.court_keypoints_path):
            self.logger.info("Loading court keypoints from file.")
            self.selected_keypoints = load_json(self.court_keypoints_path)
        else:
            # Manual keypoint selection
            self.logger.info("Keypoints file not found. Initiating manual keypoint selection.")
            cv2.imshow('Select Court Keypoints', img)
            cv2.setMouseCallback('Select Court Keypoints', self.click_event, {'img': img})
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # Save the selected keypoints to the specified file path
            save_json(self.selected_keypoints, self.court_keypoints_path)
            self.logger.info("Selected keypoints saved successfully.")

        return self.selected_keypoints

    def load_model(self):
        """
        Load a model based on the specified type in the configuration.
        """
        self.logger.info(f"Loading model of type: {self.model_type}")
        if self.model_type == "resnet":
            model = models.resnet50(pretrained=True)
            model.fc = torch.nn.Linear(model.fc.in_features, 24)  # 12 keypoints * 2 coordinates
            self.model = model.eval()
        elif self.model_type == "yolo":
            self.model = YOLO(self.keypoints_tracker_model)
        else:
            raise ValueError("Unknown model type")

    def detect_keypoints(self, frame):
        """
        Detect keypoints using the loaded model.

        Parameters:
        - frame (numpy array): A single video frame

        Returns:
        - list: Detected keypoints
        """
        if self.model_type == "yolo":
            results = self.model.predict(frame)
            keypoints = [(kp.x, kp.y) for kp in results[0].keypoints.xy.squeeze(0)]
        else:
            tensor = torch.tensor(frame).float().permute(2, 0, 1).unsqueeze(0)  # Reshape for model input
            with torch.no_grad():
                output = self.model(tensor).view(-1, 2)
            keypoints = output.numpy().tolist()
        
        self.logger.info(f"Detected {len(keypoints)} keypoints in the frame.")
        return keypoints

    def get_court_keypoints(self, frames):
        """
        Choose between fixed keypoints or model-based keypoint detection.

        Parameters:
        - frames (list): List of video frames

        Returns:
        - list: Court keypoints
        """
        if self.use_fixed_keypoints:
            return self.get_fixed_keypoints(frames[0])
        else:
            self.load_model()
            return self.detect_keypoints(frames[0])
        
    def court_detection_exists(self):
        """
        Check if the court detection output file already exists.

        Returns:
        - bool: True if the file exists, False otherwise.
        """
        return os.path.exists(self.court_keypoints_path)        
        
    def save_video_with_keypoints(self, frames, court_keypoints):
        """
        Save a video with court keypoints marked on each frame
        """
        if not court_keypoints:
            self.logger.warning("No court keypoints provided to save video.")
            return

        annotated_frames = copy.deepcopy(frames)
        
        # Create Keypoints objects for consistent drawing
        keypoints_obj = Keypoints([
            Keypoint(idx, (float(x), float(y))) 
            for idx, (x, y) in enumerate(court_keypoints)
        ])

        # Annotate each frame using the Keypoints draw method
        for frame in annotated_frames:
            frame = keypoints_obj.draw(frame)

        save_video(
            frames=annotated_frames,
            path=self.court_video_path,
            fps=30,
            width=annotated_frames[0].shape[1],
            height=annotated_frames[0].shape[0]
        )
        self.logger.info(f"Video with court points saved to {self.court_video_path}")

    def load_court_keypoints(self) -> Optional[List[Tuple[int, int]]]:
        """Load existing court keypoints from JSON file"""
        try:
            return load_json(self.court_keypoints_path)
        except Exception as e:
            self.logger.error(f"Failed to load court keypoints: {str(e)}")
            return None

from typing import Literal,List,Union,Tuple
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os


def read_video(
    path: Union[str, Path], 
    max_frames: Union[int, str, None] = None, 
    show_progress: bool = True
) -> Tuple[List[np.ndarray], int, int, int]:
    """
    Reads a video from the specified path and returns frames along with video metadata.

    Parameters:
    - path (str | Path): The path to the video file.
    - max_frames (int, optional): Maximum number of frames to read. Reads all frames if None.
    - show_progress (bool, optional): Whether to show a progress bar during reading.

    Returns:
    - frames (list[np.ndarray]): A list of frames (RGB format).
    - fps (int): Frames per second of the video.
    - width (int): Width of the video frames.
    - height (int): Height of the video frames.
    """
    path = str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Video file not found: {path}")
    
    print("Reading Video ...")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Failed to open video file: {path}")

    # Video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if isinstance(max_frames, str):
        max_frames = None if max_frames.lower() == "none" else int(max_frames)
    elif max_frames is not None:
        max_frames = int(max_frames)

    # Determine frame count to read
    frame_count = total_frames if max_frames is None else min(max_frames, total_frames)
    
    frames = []
    progress_bar = tqdm(total=frame_count, desc="Reading Frames", disable=not show_progress)

    try:
        while len(frames) < frame_count:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            progress_bar.update(1)
    finally:
        cap.release()
        progress_bar.close()

    print("Done.")
    return frames, fps, width, height


def save_video(
    frames: List[np.ndarray],
    path: Union[str, Path],
    fps: int,
    width: int,
    height: int,
    codec: Literal['mp4v', 'avc1', 'XVID'] = 'mp4v',
    show_progress: bool = True
):
    """
    Saves a list of frames as a video file.

    Parameters:
    - frames (list[np.ndarray]): A list of frames (RGB format).
    - path (str | Path): The path where the video will be saved.
    - fps (int): Frames per second for the output video.
    - width (int): Width of the video frames.
    - height (int): Height of the video frames.
    - codec (str): FourCC code for the video codec. Defaults to 'mp4v'.
    - show_progress (bool, optional): Whether to show a progress bar during saving.

    Returns:
    - None
    """
    path = str(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(path, fourcc, float(fps), (width, height))

    if not out.isOpened():
        raise IOError(f"Failed to create video file: {path}")

    print(f"Saving Video to {path} ...")
    progress_bar = tqdm(total=len(frames), desc="Writing Frames", disable=not show_progress)

    try:
        for frame in frames:
            # Convert frame back to BGR for saving
            out.write(frame)
            progress_bar.update(1)
    finally:
        out.release()
        progress_bar.close()

    print("Video saved successfully.")

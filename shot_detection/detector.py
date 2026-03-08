import json
import numpy as np
from tqdm import tqdm
from roboflow import Roboflow
import supervision as sv
from utils import read_video, save_video, calculate_iou
from .stabilizer import LabelStabilizer
from .visualizer import ShotVisualizer

class ShotDetector:
    def __init__(self, api_key, project_name, version):
        self.stabilizer = LabelStabilizer()
        self.visualizer = ShotVisualizer()
        rf = Roboflow(api_key=api_key)
        self.model = rf.workspace().project(project_name).version(version).model

    def run(self, frames, track_json, conf_threshold=0.3):
        with open(track_json, 'r') as f:
            player_tracks = json.load(f)

        annotated_frames = []
        frame_results = []
        player_shot_history = {}
        progress_bar = tqdm(total=len(frames), desc="Processing frames", unit="frame")

        for frame_idx, frame in enumerate(frames):
            current_players = player_tracks[frame_idx] if frame_idx < len(player_tracks) else []
            try:
                result = self.model.predict(frame, confidence=conf_threshold, overlap=40).json()
                detections = sv.Detections.from_inference(result)
            except Exception:
                detections = sv.Detections.empty()

            primary_shot = self._get_primary_shot(detections, result, current_players, frame_idx)

            if primary_shot['player_id']:
                stabilized_label = self.stabilizer.update(
                    primary_shot['player_id'],
                    primary_shot['shot_type'],
                    primary_shot['confidence']
                )
                if stabilized_label:
                    primary_shot['shot_type'] = stabilized_label
                    player_shot_history.setdefault(primary_shot['player_id'], []).append(primary_shot)

            stabilized_labels = {}
            if primary_shot['player_id'] and primary_shot['shot_type']:
                stabilized_labels[primary_shot['player_id']] = primary_shot['shot_type']

            annotated_frames.append(self.visualizer.annotate_frame(frame.copy(), current_players, stabilized_labels))

            if primary_shot['player_id']:
                frame_results.append({"frame_index": frame_idx, "primary_shot": primary_shot})

            progress_bar.update(1)

        progress_bar.close()
        self._save_outputs(annotated_frames, frame_results, player_shot_history)
        return frame_results, annotated_frames

    def _get_primary_shot(self, detections, result, current_players, frame_idx):
        shot = {'player_id': None, 'player_name': None, 'shot_type': None,
                'confidence': 0, 'iou': 0, 'frame_index': frame_idx}
        if len(detections.xyxy) == 0:
            return shot

        highest_conf_idx = np.argmax(detections.confidence)
        best_det_box = detections.xyxy[highest_conf_idx]
        best_shot_type = result["predictions"][highest_conf_idx]["class"]
        best_confidence = float(detections.confidence[highest_conf_idx])

        best_player, best_iou = None, 0
        for player in current_players:
            iou = calculate_iou(best_det_box, player["xyxy"])
            if iou > best_iou:
                best_iou, best_player = iou, player

        if best_player:
            shot.update({
                'player_id': best_player["player_id"],
                'player_name': best_player.get("name", f"Player {best_player['player_id']}"),
                'shot_type': best_shot_type,
                'confidence': best_confidence,
                'iou': best_iou
            })
        return shot

    def _save_outputs(self, annotated_frames, frame_results, player_shot_history):
        # save_video(annotated_frames, f"{output_dir}/shots_annotated.mp4")
        with open(f"data/json/shot_events.json", "w") as f:
            json.dump(frame_results, f, indent=2)
        with open(f"data/json/player_shot_history.json", "w") as f:
            json.dump(player_shot_history, f, indent=2)

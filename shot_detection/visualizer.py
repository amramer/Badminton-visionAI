import cv2
import supervision as sv

class ShotVisualizer:
    def __init__(self):
        self.label_annotator = sv.LabelAnnotator()
        self.palette = sv.ColorPalette.from_matplotlib("Set1", 7)
        self.shot_type_color_map = {}

    def get_power_color(self, power):
        if power < 30:
            return (0, 0, 255)
        elif power < 70:
            return (0, int(255 * power / 100), 255)
        else:
            return (0, 255, int(255 * (100 - power) / 30))

    def draw_power_bar(self, frame, box, power):
        x1, y1, x2, _ = map(int, box)
        bar_height = 16
        bar_width = x2 - x1
        filled_width = int(bar_width * (power / 100))

        for i in range(filled_width):
            current_power = int((i / bar_width) * 100)
            color = self.get_power_color(current_power)
            cv2.rectangle(frame, (x1 + i, y1), (x1 + i + 1, y1 + bar_height), color, -1)

        if filled_width < bar_width:
            cv2.rectangle(frame, (x1 + filled_width, y1), (x2, y1 + bar_height), (50, 50, 50), -1)

        cv2.rectangle(frame, (x1, y1), (x2, y1 + bar_height), (30, 30, 30), 1)
        cv2.rectangle(frame, (x1+1, y1+1), (x2-1, y1 + bar_height-1), (70, 70, 70), 1)

        text = f"{power}%"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = x1 + (bar_width - text_size[0]) // 2
        text_y = y1 + (bar_height + text_size[1]) // 2 - 2
        cv2.putText(frame, text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    def annotate_frame(self, frame, current_players, player_labels):
        """
        current_players: list of dicts, each like:
        {
            "xyxy": [x1,y1,x2,y2],
            "confidence": 0.88,
            "player_id": 1,
            "player_name": "..."
        }
        player_labels: dict {player_id: shot_type}
        """
        if not current_players:
            return frame

        for player in current_players:
            player_id = player.get("player_id")
            if player_id is None:
                continue

            shot_type = player_labels.get(player_id)
            if shot_type is None:
                continue

            shot_type_key = str(shot_type).lower()
            if shot_type_key not in self.shot_type_color_map:
                color_idx = len(self.shot_type_color_map)
                self.shot_type_color_map[shot_type_key] = self.palette.by_idx(color_idx).as_bgr()
            color = self.shot_type_color_map[shot_type_key]

            box = player.get("xyxy", None)
            if not box or len(box) != 4:
                continue

            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label_text = f"{str(shot_type).upper()}"
            (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

            text_x = x1 + 33
            text_y = y1 - 33
            padding = 8
            bg_x1 = max(0, text_x - padding)
            bg_y1 = max(0, text_y - text_height - padding)
            bg_x2 = min(frame.shape[1], text_x + text_width + padding)
            bg_y2 = min(frame.shape[0], text_y + padding)

            overlay = frame.copy()
            cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

            cv2.putText(
                frame, label_text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA
            )

            # Power: use player's confidence from tracking JSON if present, else fallback to 0
            power = int(float(player.get("confidence", 0.0)) * 100)
            self.draw_power_bar(frame, (x1, y1 - 20, x2, y1 - 4), power)

        return frame


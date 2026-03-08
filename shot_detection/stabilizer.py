from collections import defaultdict, deque

class LabelStabilizer:
    def __init__(
        self,
        history_length: int = 8,
        min_consensus: float = 0.31,
        min_stick_frames: int = 5,
        decay_factor: float = 0.87,
        linger_frames: int = 11,
    ):
        self.history = defaultdict(deque)
        self.current_label = defaultdict(lambda: None)
        self.current_confidence = defaultdict(float)
        self.stick_counter = defaultdict(int)
        self.linger_counter = defaultdict(int)
        self.history_length = history_length
        self.min_consensus = min_consensus
        self.min_stick_frames = min_stick_frames
        self.decay_factor = decay_factor
        self.linger_frames = linger_frames

    def _apply_ema(self, player_id, new_confidence):
        if self.current_confidence[player_id] == 0:
            return new_confidence
        return (
            self.decay_factor * self.current_confidence[player_id] +
            (1 - self.decay_factor) * new_confidence
        )

    def _get_dominant_label(self, player_id):
        if not self.history[player_id]:
            return None, 0.0

        label_confidences = defaultdict(float)
        for label, conf in self.history[player_id]:
            label_confidences[label] = self._apply_ema(player_id, conf)

        best_label, best_conf = max(label_confidences.items(), key=lambda x: x[1])
        if best_conf < self.min_consensus:
            return None, 0.0
        return best_label, best_conf

    def update(self, player_id, new_label, new_confidence):
        if new_label is None:
            if self.current_label[player_id] is not None and self.linger_counter[player_id] > 0:
                self.linger_counter[player_id] -= 1
                return self.current_label[player_id]
            return None

        self.history[player_id].append((new_label, new_confidence))
        if len(self.history[player_id]) > self.history_length:
            self.history[player_id].popleft()

        best_label, best_conf = self._get_dominant_label(player_id)
        if best_label is None:
            return None

        current_label = self.current_label[player_id]

        if current_label is None:
            self.current_label[player_id] = best_label
            self.current_confidence[player_id] = best_conf
            self.stick_counter[player_id] = 1
            if self.stick_counter[player_id] >= self.min_stick_frames:
                self.linger_counter[player_id] = self.linger_frames
                return best_label
            return None

        if best_label == current_label:
            self.stick_counter[player_id] += 1
            self.current_confidence[player_id] = best_conf
            self.linger_counter[player_id] = self.linger_frames
            return best_label
        else:
            self.current_label[player_id] = best_label
            self.current_confidence[player_id] = best_conf
            self.stick_counter[player_id] = 1

        if self.stick_counter[player_id] >= self.min_stick_frames:
            self.linger_counter[player_id] = self.linger_frames
            return self.current_label[player_id]
        return None

    def reset_player(self, player_id):
        self.history[player_id].clear()
        self.current_label[player_id] = None
        self.current_confidence[player_id] = 0.0
        self.stick_counter[player_id] = 0
        self.linger_counter[player_id] = 0

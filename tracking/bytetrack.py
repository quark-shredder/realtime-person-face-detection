"""ByteTrack implementation for face tracking."""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from scipy.optimize import linear_sum_assignment


@dataclass
class Track:
    """Represents a tracked object."""
    track_id: int
    bbox: List[int]  # [x1, y1, x2, y2]
    confidence: float
    label: Optional[str] = None  # For face recognition
    age: int = 0  # Frames since last update
    hits: int = 1  # Number of successful matches
    time_since_update: int = 0
    state: str = "tracked"  # tracked, lost, removed
    last_recognized_frame: int = -1  # Frame number of last recognition


class ByteTracker:
    """
    Simplified ByteTrack implementation.
    Tracks faces across frames using IoU matching.
    """

    def __init__(
        self,
        track_thresh: float = 0.5,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        min_box_area: int = 100,
    ):
        """
        Initialize ByteTracker.

        Args:
            track_thresh: Detection confidence threshold
            track_buffer: Number of frames to keep lost tracks
            match_thresh: IoU threshold for matching
            min_box_area: Minimum bounding box area
        """
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.min_box_area = min_box_area

        self.tracks: List[Track] = []
        self.next_id = 0
        self.frame_count = 0

    def update(
        self, detections: List[Tuple[List[int], float]]
    ) -> List[Track]:
        """
        Update tracks with new detections.

        Args:
            detections: List of (bbox, confidence) tuples

        Returns:
            List of active tracks
        """
        self.frame_count += 1

        # Filter detections by area
        valid_dets = []
        for bbox, conf in detections:
            x1, y1, x2, y2 = bbox
            area = (x2 - x1) * (y2 - y1)
            if area >= self.min_box_area:
                valid_dets.append((bbox, conf))

        # Separate detections into high and low confidence
        high_dets = [(bbox, conf) for bbox, conf in valid_dets if conf >= self.track_thresh]
        low_dets = [(bbox, conf) for bbox, conf in valid_dets if conf < self.track_thresh]

        # Separate tracks into tracked and lost
        tracked_tracks = [t for t in self.tracks if t.state == "tracked"]
        lost_tracks = [t for t in self.tracks if t.state == "lost"]

        # First association: match high-confidence detections with tracked tracks
        matched_tracks, unmatched_tracks, unmatched_dets = self._match(
            tracked_tracks, high_dets
        )

        # Second association: match remaining tracks with low-confidence detections
        if len(unmatched_tracks) > 0 and len(low_dets) > 0:
            matched_low, unmatched_tracks_final, _ = self._match(
                unmatched_tracks, low_dets
            )
            matched_tracks.extend(matched_low)
        else:
            unmatched_tracks_final = unmatched_tracks

        # Third association: match lost tracks with remaining detections
        if len(lost_tracks) > 0 and len(unmatched_dets) > 0:
            matched_lost, _, unmatched_dets_final = self._match(
                lost_tracks, [high_dets[i] for i in unmatched_dets]
            )
            matched_tracks.extend(matched_lost)
        else:
            unmatched_dets_final = unmatched_dets

        # Update matched tracks
        for track_idx, det_idx in matched_tracks:
            if track_idx < len(tracked_tracks):
                track = tracked_tracks[track_idx]
            else:
                track = lost_tracks[track_idx - len(tracked_tracks)]

            bbox, conf = high_dets[det_idx] if det_idx < len(high_dets) else low_dets[det_idx - len(high_dets)]
            track.bbox = bbox
            track.confidence = conf
            track.time_since_update = 0
            track.hits += 1
            track.state = "tracked"

        # Mark unmatched tracks as lost
        for track_idx in unmatched_tracks_final:
            if track_idx < len(tracked_tracks):
                track = tracked_tracks[track_idx]
                track.time_since_update += 1
                if track.time_since_update > self.track_buffer:
                    track.state = "removed"
                else:
                    track.state = "lost"

        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets_final:
            if det_idx < len(high_dets):
                bbox, conf = high_dets[det_idx]
                new_track = Track(
                    track_id=self.next_id,
                    bbox=bbox,
                    confidence=conf,
                )
                self.tracks.append(new_track)
                self.next_id += 1

        # Remove old tracks
        self.tracks = [t for t in self.tracks if t.state != "removed"]

        # Return only actively tracked tracks
        return [t for t in self.tracks if t.state == "tracked"]

    def _match(
        self,
        tracks: List[Track],
        detections: List[Tuple[List[int], float]]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Match tracks with detections using IoU.

        Returns:
            matched_pairs, unmatched_tracks, unmatched_detections
        """
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))

        # Compute IoU cost matrix
        iou_matrix = np.zeros((len(tracks), len(detections)))
        for i, track in enumerate(tracks):
            for j, (bbox, _) in enumerate(detections):
                iou_matrix[i, j] = self._compute_iou(track.bbox, bbox)

        # Use Hungarian algorithm for matching
        # Convert IoU to cost (higher IoU = lower cost)
        cost_matrix = 1.0 - iou_matrix

        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        matched = []
        unmatched_tracks = list(range(len(tracks)))
        unmatched_dets = list(range(len(detections)))

        for row, col in zip(row_indices, col_indices):
            if iou_matrix[row, col] >= self.match_thresh:
                matched.append((row, col))
                unmatched_tracks.remove(row)
                unmatched_dets.remove(col)

        return matched, unmatched_tracks, unmatched_dets

    @staticmethod
    def _compute_iou(bbox1: List[int], bbox2: List[int]) -> float:
        """Compute IoU between two bounding boxes."""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2

        # Intersection area
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

        # Union area
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area

        if union_area == 0:
            return 0.0

        return inter_area / union_area

    def get_tracks_for_recognition(self, recognition_interval: int = 30) -> List[Track]:
        """
        Get tracks that need face recognition.

        Args:
            recognition_interval: Frames between re-recognition

        Returns:
            List of tracks that need recognition
        """
        tracks_to_recognize = []
        for track in self.tracks:
            if track.state != "tracked":
                continue

            # New track (never recognized)
            if track.last_recognized_frame == -1:
                tracks_to_recognize.append(track)
                continue

            # Re-recognize periodically
            frames_since_last = self.frame_count - track.last_recognized_frame
            if frames_since_last >= recognition_interval:
                tracks_to_recognize.append(track)

        return tracks_to_recognize

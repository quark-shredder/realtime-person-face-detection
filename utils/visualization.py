"""Visualization utilities for drawing bounding boxes and labels."""

import cv2
import numpy as np
from typing import List, Tuple, Dict
from tracking.bytetrack import Track
import time


class Visualizer:
    """Handles all visualization and rendering."""

    def __init__(
        self,
        show_fps: bool = True,
        show_latency: bool = True,
        show_track_ids: bool = True,
        show_confidence: bool = True,
        box_thickness: int = 2,
        font_scale: float = 0.6,
        colors: Dict[str, List[int]] = None,
    ):
        """
        Initialize visualizer.

        Args:
            show_fps: Display FPS counter
            show_latency: Display processing latency
            show_track_ids: Display track IDs on faces
            show_confidence: Display confidence scores
            box_thickness: Bounding box line thickness
            font_scale: Font scale for text
            colors: Color dictionary for different object types
        """
        self.show_fps = show_fps
        self.show_latency = show_latency
        self.show_track_ids = show_track_ids
        self.show_confidence = show_confidence
        self.box_thickness = box_thickness
        self.font_scale = font_scale

        # Default colors (BGR format)
        self.colors = colors or {
            "object": [0, 255, 0],      # green
            "face": [255, 165, 0],      # orange
            "recognized": [0, 255, 255], # yellow
            "unknown": [128, 128, 128]  # gray
        }

        # FPS calculation
        self.fps_history = []
        self.max_fps_samples = 30

    def draw_object_boxes(
        self,
        frame: np.ndarray,
        detections: List[Tuple[List[int], int, float, str]]
    ) -> np.ndarray:
        """
        Draw bounding boxes for object detections.

        Args:
            frame: Image to draw on
            detections: List of (bbox, class_id, confidence, class_name)

        Returns:
            Frame with drawn boxes
        """
        for bbox, cls_id, conf, cls_name in detections:
            x1, y1, x2, y2 = bbox
            color = self.colors.get("object", [0, 255, 0])

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.box_thickness)

            # Draw label
            label = f"{cls_name}"
            if self.show_confidence:
                label += f" {conf:.2f}"

            self._draw_label(frame, label, (x1, y1 - 5), color)

        return frame

    def draw_face_tracks(
        self,
        frame: np.ndarray,
        tracks: List[Track]
    ) -> np.ndarray:
        """
        Draw bounding boxes for tracked faces.

        Args:
            frame: Image to draw on
            tracks: List of face tracks

        Returns:
            Frame with drawn boxes
        """
        for track in tracks:
            x1, y1, x2, y2 = track.bbox

            # Determine color based on recognition status
            if track.label and track.label != "unknown":
                color = self.colors.get("recognized", [0, 255, 255])
            elif track.label == "unknown":
                color = self.colors.get("unknown", [128, 128, 128])
            else:
                color = self.colors.get("face", [255, 165, 0])

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.box_thickness)

            # Build label
            label_parts = []
            if self.show_track_ids:
                label_parts.append(f"ID:{track.track_id}")
            if track.label:
                label_parts.append(track.label)
            if self.show_confidence:
                label_parts.append(f"{track.confidence:.2f}")

            label = " ".join(label_parts)

            if label:
                self._draw_label(frame, label, (x1, y1 - 5), color)

        return frame

    def draw_info_overlay(
        self,
        frame: np.ndarray,
        fps: float = 0.0,
        latency: float = 0.0,
        num_objects: int = 0,
        num_faces: int = 0,
    ) -> np.ndarray:
        """
        Draw information overlay on frame.

        Args:
            frame: Image to draw on
            fps: Frames per second
            latency: Processing latency in milliseconds
            num_objects: Number of detected objects
            num_faces: Number of tracked faces

        Returns:
            Frame with info overlay
        """
        h, w = frame.shape[:2]
        overlay_height = 100
        overlay = frame.copy()

        # Semi-transparent background
        cv2.rectangle(
            overlay,
            (0, 0),
            (w, overlay_height),
            (0, 0, 0),
            -1
        )
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        y_offset = 25
        text_color = (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        # FPS
        if self.show_fps:
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10, y_offset),
                font,
                self.font_scale,
                text_color,
                2
            )
            y_offset += 25

        # Latency
        if self.show_latency:
            cv2.putText(
                frame,
                f"Latency: {latency:.1f}ms",
                (10, y_offset),
                font,
                self.font_scale,
                text_color,
                2
            )
            y_offset += 25

        # Detection counts
        cv2.putText(
            frame,
            f"Objects: {num_objects} | Faces: {num_faces}",
            (10, y_offset),
            font,
            self.font_scale,
            text_color,
            2
        )

        return frame

    def draw_controls_help(
        self,
        frame: np.ndarray
    ) -> np.ndarray:
        """
        Draw keyboard controls help.

        Args:
            frame: Image to draw on

        Returns:
            Frame with controls overlay
        """
        h, w = frame.shape[:2]
        controls = [
            "Q: Quit",
            "S: Save frame",
            "R: Record",
            "I: Toggle info",
            "F: Toggle faces",
            "O: Toggle objects",
        ]

        x_start = w - 200
        y_start = h - (len(controls) * 25) - 10

        # Background
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (x_start - 10, y_start - 25),
            (w, h),
            (0, 0, 0),
            -1
        )
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        # Draw controls
        for i, control in enumerate(controls):
            y = y_start + i * 25
            cv2.putText(
                frame,
                control,
                (x_start, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

        return frame

    def _draw_label(
        self,
        frame: np.ndarray,
        text: str,
        position: Tuple[int, int],
        color: List[int]
    ):
        """Draw a text label with background."""
        x, y = position
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Get text size
        (text_w, text_h), baseline = cv2.getTextSize(
            text, font, self.font_scale, 2
        )

        # Draw background
        cv2.rectangle(
            frame,
            (x, y - text_h - baseline),
            (x + text_w, y),
            color,
            -1
        )

        # Draw text
        cv2.putText(
            frame,
            text,
            (x, y - baseline // 2),
            font,
            self.font_scale,
            (0, 0, 0),  # Black text
            2
        )

    def calculate_fps(self, frame_time: float) -> float:
        """
        Calculate running average FPS.

        Args:
            frame_time: Time taken for frame in seconds

        Returns:
            Average FPS
        """
        if frame_time > 0:
            fps = 1.0 / frame_time
            self.fps_history.append(fps)

            # Keep only recent samples
            if len(self.fps_history) > self.max_fps_samples:
                self.fps_history.pop(0)

            return sum(self.fps_history) / len(self.fps_history)

        return 0.0

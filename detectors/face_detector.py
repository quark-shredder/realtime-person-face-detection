"""MediaPipe face detector wrapper."""

import cv2
import numpy as np
import mediapipe as mp
from typing import List, Tuple, Optional


class MediaPipeFaceDetector:
    """Fast face detector using MediaPipe."""

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        model_selection: int = 0,
    ):
        """
        Initialize MediaPipe face detector.

        Args:
            min_detection_confidence: Minimum confidence for face detection
            model_selection: 0 for short-range (2m), 1 for full-range (5m)
        """
        self.min_confidence = min_detection_confidence
        self.model_selection = model_selection
        self._closed = False

        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.detector = self.mp_face_detection.FaceDetection(
            min_detection_confidence=min_detection_confidence,
            model_selection=model_selection,
        )

        print(f"MediaPipe Face Detector initialized (model_selection={model_selection})")

    def __call__(self, image: np.ndarray) -> List[Tuple[List[int], float]]:
        """
        Detect faces in image.

        Args:
            image: Input image (BGR format)

        Returns:
            List of (bbox, confidence) tuples
            bbox is [x1, y1, x2, y2] in pixel coordinates
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect faces
        results = self.detector.process(image_rgb)

        detections = []

        if results.detections:
            h, w = image.shape[:2]

            for detection in results.detections:
                # Get confidence score
                confidence = detection.score[0]

                # Get bounding box
                bbox = detection.location_data.relative_bounding_box

                # Convert to pixel coordinates
                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                x2 = int((bbox.xmin + bbox.width) * w)
                y2 = int((bbox.ymin + bbox.height) * h)

                # Ensure bbox is within image bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                detections.append(([x1, y1, x2, y2], confidence))

        return detections

    def close(self):
        """Release resources."""
        if not self._closed and hasattr(self, 'detector') and self.detector is not None:
            try:
                self.detector.close()
            except (ValueError, Exception):
                pass  # Already closed or error during cleanup
            finally:
                self._closed = True

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.close()
        except:
            pass  # Suppress errors during interpreter shutdown

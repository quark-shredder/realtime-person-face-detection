"""Camera management with threading for non-blocking capture."""

import cv2
import threading
import queue
import time
from typing import Optional, Tuple
import numpy as np


class CameraManager:
    """Thread-safe camera capture manager."""

    def __init__(
        self,
        device_id: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        buffer_size: int = 2,
    ):
        """
        Initialize camera manager.

        Args:
            device_id: Camera device index
            width: Frame width
            height: Frame height
            fps: Target frames per second
            buffer_size: Frame buffer size (smaller = lower latency)
        """
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps
        self.buffer_size = buffer_size

        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_queue: queue.Queue = queue.Queue(maxsize=buffer_size)
        self.capture_thread: Optional[threading.Thread] = None
        self.running = False
        self.frame_count = 0
        self.start_time = time.time()

    def start(self) -> bool:
        """Start camera capture thread."""
        if self.running:
            return True

        # Initialize camera
        self.cap = cv2.VideoCapture(self.device_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.device_id}")

        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        # Disable auto-focus for stability (if supported)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

        # Get actual properties
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))

        print(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps} FPS")

        # Start capture thread
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

        return True

    def _capture_loop(self):
        """Background thread for continuous frame capture."""
        while self.running:
            ret, frame = self.cap.read()

            if not ret:
                print("Warning: Failed to read frame from camera")
                time.sleep(0.1)
                continue

            # Drop oldest frame if queue is full
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass

            # Add new frame
            try:
                self.frame_queue.put_nowait((time.time(), frame))
                self.frame_count += 1
            except queue.Full:
                pass

    def read(self, timeout: float = 1.0) -> Optional[Tuple[float, np.ndarray]]:
        """
        Read latest frame from buffer.

        Args:
            timeout: Maximum time to wait for frame

        Returns:
            Tuple of (timestamp, frame) or None if timeout
        """
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_fps(self) -> float:
        """Get actual capture FPS."""
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            return self.frame_count / elapsed
        return 0.0

    def stop(self):
        """Stop camera capture and release resources."""
        self.running = False

        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)

        if self.cap:
            self.cap.release()

        # Clear queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()

    def is_running(self) -> bool:
        """Check if camera is running."""
        return self.running and self.capture_thread and self.capture_thread.is_alive()

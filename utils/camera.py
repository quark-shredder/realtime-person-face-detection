"""Camera management with threading for non-blocking capture."""

import cv2
import threading
import queue
import time
import platform
from typing import Optional, Tuple
import numpy as np


class CameraPermissionError(Exception):
    """Raised when camera permission is denied."""
    pass


class CameraNotFoundError(Exception):
    """Raised when camera device is not found."""
    pass


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

    def start(self, retry_count: int = 3, retry_delay: float = 1.0) -> bool:
        """
        Start camera capture thread.

        Args:
            retry_count: Number of retry attempts
            retry_delay: Delay between retries in seconds

        Returns:
            True if successful

        Raises:
            CameraPermissionError: If camera access is denied
            CameraNotFoundError: If camera device not found
        """
        if self.running:
            return True

        last_error = None

        for attempt in range(retry_count):
            try:
                # Initialize camera
                self.cap = cv2.VideoCapture(self.device_id)

                if not self.cap.isOpened():
                    if attempt < retry_count - 1:
                        print(f"Camera not available, retrying in {retry_delay}s... ({attempt + 1}/{retry_count})")
                        time.sleep(retry_delay)
                        continue
                    else:
                        # Determine error type
                        if platform.system() == 'Darwin':  # macOS
                            raise CameraPermissionError(
                                f"Camera access denied. Please grant camera permission:\n"
                                f"  1. Open System Settings → Privacy & Security → Camera\n"
                                f"  2. Enable permission for Terminal (or your Python IDE)\n"
                                f"  3. Run this program again\n\n"
                                f"Quick fix: Run 'sudo killall VDCAssistant' and try again"
                            )
                        else:
                            raise CameraNotFoundError(
                                f"Camera {self.device_id} not found. "
                                f"Please check your camera connection or try a different device ID"
                            )

                # Test if we can actually read from camera
                ret, test_frame = self.cap.read()
                if not ret or test_frame is None:
                    self.cap.release()
                    if attempt < retry_count - 1:
                        print(f"Camera test failed, retrying... ({attempt + 1}/{retry_count})")
                        time.sleep(retry_delay)
                        continue
                    else:
                        raise CameraPermissionError(
                            "Camera opened but cannot read frames. Permission may be denied."
                        )

                # Set camera properties
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                # Don't force FPS - let camera use native rate to avoid driver issues

                # Disable auto-focus for stability (if supported)
                self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

                # Get actual properties
                actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))

                print(f"✓ Camera initialized: {actual_width}x{actual_height} @ {actual_fps} FPS")

                # Start capture thread
                self.running = True
                self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
                self.capture_thread.start()

                return True

            except (CameraPermissionError, CameraNotFoundError):
                raise
            except Exception as e:
                last_error = e
                if attempt < retry_count - 1:
                    print(f"Error starting camera: {e}, retrying... ({attempt + 1}/{retry_count})")
                    time.sleep(retry_delay)
                    continue

        # All retries failed
        if last_error:
            raise CameraNotFoundError(f"Failed to start camera after {retry_count} attempts: {last_error}")

        return False

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

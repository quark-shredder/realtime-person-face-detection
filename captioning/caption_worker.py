"""Background worker for SmolVLM captioning."""

import threading
import time
import numpy as np
from typing import Optional


class CaptionWorker:
    """Background worker thread for captioning with live config updates."""

    def __init__(self, config):
        """
        Initialize caption worker.

        Args:
            config: CaptioningConfig instance
        """
        self.config = config
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_result = {
            "text": "Initializing caption...",
            "latency_ms": 0,
            "error": None,
            "timestamp": time.time()
        }
        self.is_processing = False  # Prevent overlapping requests
        self.running = False
        self.lock = threading.Lock()
        self.worker_thread: Optional[threading.Thread] = None

        # Live config (can be updated at runtime)
        self.prompt = config.prompt
        self.interval = config.interval_seconds

    def update_frame(self, frame: np.ndarray):
        """
        Update latest frame (called from main thread).

        Args:
            frame: Current video frame (BGR format)
        """
        with self.lock:
            self.latest_frame = frame.copy()

    def get_latest_caption(self) -> dict:
        """
        Get current caption with age (non-blocking).

        Returns:
            Dict with text, latency_ms, error, age_ms
        """
        with self.lock:
            result = self.latest_result.copy()
            # Calculate age since caption was generated
            result["age_ms"] = (time.time() - result["timestamp"]) * 1000
            return result

    def set_prompt(self, prompt: str):
        """
        Update caption prompt at runtime.

        Args:
            prompt: New instruction text
        """
        with self.lock:
            self.prompt = prompt
            print(f"[CAPTION] Prompt updated: {prompt}")

    def set_interval(self, interval_seconds: float):
        """
        Update caption interval at runtime.

        Args:
            interval_seconds: Time between caption requests
        """
        with self.lock:
            self.interval = interval_seconds
            print(f"[CAPTION] Interval updated: {interval_seconds}s")

    def _worker_loop(self):
        """Background thread: request caption every interval_seconds."""
        from captioning.smolvlm_api import get_caption

        while self.running:
            # Get current interval (may change at runtime)
            with self.lock:
                sleep_time = self.interval

            time.sleep(sleep_time)

            # Skip if previous request still running (overlap prevention)
            if self.is_processing:
                continue

            # Check if captioning is still enabled
            with self.lock:
                if not self.config.enabled:
                    self.latest_frame = None  # Free memory when disabled
                    continue

                frame = self.latest_frame
                current_prompt = self.prompt

            if frame is None:
                continue

            self.is_processing = True

            # Get caption from SmolVLM
            result = get_caption(
                frame,
                current_prompt,
                self.config.server_url,
                self.config.jpeg_quality,
                self.config.max_tokens,
                self.config.timeout_seconds
            )

            with self.lock:
                self.latest_result = result

            self.is_processing = False

    def start(self):
        """Start background worker thread."""
        if self.running:
            return

        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        print("[CAPTION] Worker started")

    def stop(self):
        """Stop background worker thread."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=2.0)
        print("[CAPTION] Worker stopped")

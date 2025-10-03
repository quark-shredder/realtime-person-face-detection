"""Main application for real-time object detection and face recognition."""

import cv2
import time
import argparse
import warnings
from pathlib import Path
from datetime import datetime

# Suppress warnings from dependencies
warnings.filterwarnings('ignore', category=FutureWarning, module='insightface')

from utils.config import Config
from utils.camera import CameraManager, CameraPermissionError, CameraNotFoundError
from utils.visualization import Visualizer
from detectors.yolox_detector import YOLOXDetector
from detectors.face_detector import MediaPipeFaceDetector
from tracking.bytetrack import ByteTracker
from recognition.face_recognition import FaceRecognitionSystem, RecognitionWorker
from captioning.caption_worker import CaptionWorker
from web import server, launcher


class RealtimeDetectionApp:
    """Main application orchestrating all components."""

    def __init__(self, config: Config):
        """Initialize application with config."""
        self.config = config

        # Initialize camera
        print("Initializing camera...")
        self.camera = CameraManager(
            device_id=config.camera.device_id,
            width=config.camera.width,
            height=config.camera.height,
            fps=config.camera.fps,
            buffer_size=config.camera.buffer_size,
        )

        # Initialize object detector
        self.object_detector = None
        if config.object_detector.enabled:
            print("Initializing object detector...")
            try:
                self.object_detector = YOLOXDetector(
                    model_path=config.object_detector.model_path,
                    input_size=config.object_detector.input_size,
                    confidence_threshold=config.object_detector.confidence_threshold,
                    nms_threshold=config.object_detector.nms_threshold,
                    max_detections=config.object_detector.max_detections,
                    device=config.object_detector.device,
                    model_name=config.object_detector.model,
                )
            except Exception as e:
                print(f"Warning: Failed to load object detector: {e}")
                print("Continuing without object detection...")
                self.object_detector = None

        # Initialize face detector
        self.face_detector = None
        if config.face_detector.enabled:
            print("Initializing face detector...")
            self.face_detector = MediaPipeFaceDetector(
                min_detection_confidence=config.face_detector.min_detection_confidence,
                model_selection=config.face_detector.model_selection,
            )

        # Initialize face tracker
        self.tracker = None
        if config.tracking.enabled and self.face_detector:
            print("Initializing face tracker...")
            self.tracker = ByteTracker(
                track_thresh=config.tracking.track_thresh,
                track_buffer=config.tracking.track_buffer,
                match_thresh=config.tracking.match_thresh,
                min_box_area=config.tracking.min_box_area,
            )

        # Initialize face recognition
        self.face_recognition = None
        self.recognition_worker = None
        if config.face_recognition.enabled and self.tracker:
            print("Initializing face recognition...")
            try:
                self.face_recognition = FaceRecognitionSystem(
                    model_pack=config.face_recognition.model_pack,
                    enrolled_faces_path=config.face_recognition.enrolled_faces_path,
                    similarity_threshold=config.face_recognition.similarity_threshold,
                    ctx_id=config.face_recognition.ctx_id,
                    det_size=config.face_recognition.det_size,
                )
                self.recognition_worker = RecognitionWorker(self.face_recognition)
                self.recognition_worker.start()
            except Exception as e:
                print(f"Warning: Failed to initialize face recognition: {e}")
                print("Continuing without face recognition...")
                self.face_recognition = None

        # Initialize caption worker
        self.caption_worker = None
        if config.captioning.enabled:
            print("Initializing caption worker...")
            try:
                self.caption_worker = CaptionWorker(config.captioning)
            except Exception as e:
                print(f"Warning: Failed to initialize caption worker: {e}")
                print("Continuing without captioning...")
                self.caption_worker = None

        # Initialize web server
        self.web_server = None
        if config.web_ui.enabled:
            print("Initializing web server...")
            try:
                self.web_server = launcher.start_web_server(
                    config.web_ui.host,
                    config.web_ui.port,
                    config.web_ui.stream_fps
                )
            except Exception as e:
                print(f"Warning: Failed to start web server: {e}")
                print("Continuing in OpenCV mode...")
                self.web_server = None

        # Initialize visualizer
        self.visualizer = Visualizer(
            show_fps=config.visualization.show_fps,
            show_latency=config.visualization.show_latency,
            show_track_ids=config.visualization.show_track_ids,
            show_confidence=config.visualization.show_confidence,
            box_thickness=config.visualization.box_thickness,
            font_scale=config.visualization.font_scale,
            colors=config.visualization.colors,
        )

        # State
        self.running = False
        self.show_info = True
        self.show_controls = True
        self.show_profiling = False  # Toggle for detailed timing breakdown
        self.recording = False
        self.video_writer = None

        # Frame skipping for performance
        self.frame_count = 0
        self.last_object_detections = []
        self.last_face_detections = []

        print("\nInitialization complete!")
        print("=" * 60)

    def run(self):
        """Main application loop."""
        self.running = True

        # Start camera with error handling
        try:
            print("\nStarting camera...")
            self.camera.start()
            print("✓ Camera ready")
        except CameraPermissionError as e:
            print("\n" + "=" * 60)
            print("❌ CAMERA PERMISSION DENIED")
            print("=" * 60)
            print(f"\n{e}\n")
            print("=" * 60)
            return
        except CameraNotFoundError as e:
            print("\n" + "=" * 60)
            print("❌ CAMERA NOT FOUND")
            print("=" * 60)
            print(f"\n{e}\n")
            print("Available troubleshooting:")
            print("  • Check if camera is connected")
            print("  • Try a different camera device ID: python main.py --camera 1")
            print("  • Close other apps using the camera (Zoom, Skype, etc.)")
            print("=" * 60)
            return
        except Exception as e:
            print(f"\n❌ Unexpected error starting camera: {e}")
            return

        # Wait for camera to warm up
        time.sleep(0.5)

        # Start caption worker if initialized
        if self.caption_worker:
            self.caption_worker.start()
            print("✓ Caption worker started")

        print("\nStarting detection...")
        print("Press 'Q' to quit\n")

        try:
            while self.running:
                loop_start = time.perf_counter()
                stage_timings = {}
                self.frame_count += 1

                # Get frame
                result = self.camera.read(timeout=1.0)
                if result is None:
                    print("Warning: No frame received")
                    continue

                timestamp, frame = result

                # Object detection (skip every 3rd frame)
                object_detections = []
                if self.object_detector and self.config.object_detector.enabled:
                    if self.frame_count % 3 == 0:
                        t0 = time.perf_counter()
                        object_detections = self.object_detector(frame)
                        self.last_object_detections = object_detections
                        stage_timings['object_det'] = (time.perf_counter() - t0) * 1000
                    else:
                        # Reuse last detections
                        object_detections = self.last_object_detections
                        stage_timings['object_det'] = 0.0  # Skipped

                # Face detection (skip every 2nd frame)
                face_detections = []
                tracks = []
                if self.face_detector and self.config.face_detector.enabled:
                    if self.frame_count % 2 == 0:
                        t0 = time.perf_counter()
                        face_detections = self.face_detector(frame)
                        self.last_face_detections = face_detections
                        stage_timings['face_det'] = (time.perf_counter() - t0) * 1000
                    else:
                        # Reuse last detections
                        face_detections = self.last_face_detections
                        stage_timings['face_det'] = 0.0  # Skipped

                    # Track faces (only if faces detected)
                    if self.tracker and face_detections:
                        t0 = time.perf_counter()
                        tracks = self.tracker.update(face_detections)
                        stage_timings['tracking'] = (time.perf_counter() - t0) * 1000

                        # Face recognition
                        if self.recognition_worker and self.config.face_recognition.enabled:
                            t0 = time.perf_counter()
                            # Get tracks that need recognition
                            tracks_to_recognize = self.tracker.get_tracks_for_recognition(
                                self.config.face_recognition.recognition_interval
                            )

                            # Submit recognition tasks
                            for track in tracks_to_recognize:
                                success = self.recognition_worker.submit(
                                    track.track_id, frame, track.bbox
                                )
                                if success:
                                    track.last_recognized_frame = self.tracker.frame_count

                            # Process recognition results
                            results = self.recognition_worker.get_results()
                            for track_id, name, confidence in results:
                                # Find track and update label
                                for track in tracks:
                                    if track.track_id == track_id:
                                        track.label = name
                                        break
                            stage_timings['recognition'] = (time.perf_counter() - t0) * 1000
                    elif self.tracker and not face_detections:
                        # No faces - skip tracking
                        stage_timings['tracking'] = 0.0

                # Initialize caption (updated later after visualization)
                caption = {"text": "", "error": None, "latency_ms": 0, "age_ms": 0}

                # Handle browser config updates (if web UI enabled)
                if self.web_server:
                    config_update = server.pop_config_update()
                    if config_update:
                        if "prompt" in config_update:
                            self.config.captioning.prompt = config_update["prompt"]
                            if self.caption_worker:
                                self.caption_worker.set_prompt(config_update["prompt"])
                        if "interval" in config_update:
                            self.config.captioning.interval_seconds = config_update["interval"]
                            if self.caption_worker:
                                self.caption_worker.set_interval(config_update["interval"])
                        if "feed_mode" in config_update:
                            # Validate feed_mode (prevent invalid values)
                            if config_update["feed_mode"] in {"raw", "processed"}:
                                self.config.captioning.feed_mode = config_update["feed_mode"]

                    toggle = server.pop_toggle()
                    if toggle:
                        feature = toggle.get("feature")
                        enabled = toggle.get("enabled", False)
                        if feature == "caption":
                            self.config.captioning.enabled = enabled
                        elif feature == "objects":
                            self.config.object_detector.enabled = enabled
                        elif feature == "faces":
                            self.config.face_detector.enabled = enabled

                # Copy frame only before drawing (saves ~5-10ms per frame)
                display_frame = frame.copy()

                # Visualization
                if object_detections:
                    display_frame = self.visualizer.draw_object_boxes(
                        display_frame, object_detections
                    )

                if tracks:
                    display_frame = self.visualizer.draw_face_tracks(
                        display_frame, tracks
                    )

                # Caption worker update (after visualization, so display_frame has face boxes)
                if self.caption_worker and self.config.captioning.enabled:
                    # Choose frame based on feed mode
                    if self.config.captioning.feed_mode == "processed":
                        # Send frame with face boxes (copy to avoid tearing)
                        caption_frame = display_frame.copy()
                    else:
                        # Send raw camera feed
                        caption_frame = frame

                    self.caption_worker.update_frame(caption_frame)
                    caption = self.caption_worker.get_latest_caption()

                # Caption overlay (OpenCV mode only)
                if (caption.get("text") or caption.get("error")) and not self.web_server:
                    display_frame = self.visualizer.draw_caption(
                        display_frame, caption
                    )

                # Info overlay
                if self.show_info:
                    loop_time = time.perf_counter() - loop_start
                    fps = self.visualizer.calculate_fps(loop_time)
                    latency = loop_time * 1000  # Convert to ms

                    display_frame = self.visualizer.draw_info_overlay(
                        display_frame,
                        fps=fps,
                        latency=latency,
                        num_objects=len(object_detections),
                        num_faces=len(tracks),
                        stage_timings=stage_timings if self.show_profiling else None,
                    )

                # Controls overlay
                if self.show_controls:
                    display_frame = self.visualizer.draw_controls_help(display_frame)

                # Recording
                if self.recording and self.video_writer:
                    self.video_writer.write(display_frame)

                # Update web UI state
                if self.web_server:
                    # Choose frame based on config (with or without overlays)
                    web_frame = display_frame if self.config.web_ui.send_overlays else frame
                    fps = self.visualizer.calculate_fps(time.perf_counter() - loop_start)
                    server.update_state(
                        frame=web_frame,
                        objects=object_detections,
                        faces=tracks,
                        caption=caption,
                        fps=fps,
                        stats=stage_timings,
                        frame_quality=self.config.web_ui.frame_quality
                    )

                # Display (OpenCV mode)
                if not self.web_server:
                    cv2.imshow("Real-time Detection", display_frame)

                    # Handle keyboard input (OpenCV mode only)
                    key = cv2.waitKey(1) & 0xFF
                    if not self.handle_key(key, display_frame):
                        break
                else:
                    # Web mode: small sleep to allow interrupts (Ctrl+C)
                    time.sleep(0.001)

                # FPS limiting (avoid excessive CPU usage)
                target_frame_time = 1.0 / self.config.performance.target_fps
                elapsed = time.time() - loop_start
                if elapsed < target_frame_time:
                    time.sleep(target_frame_time - elapsed)

        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()

    def handle_key(self, key: int, frame) -> bool:
        """
        Handle keyboard input.

        Args:
            key: Key code
            frame: Current frame

        Returns:
            True to continue, False to quit
        """
        if key == ord('q') or key == ord('Q'):
            # Quit
            return False

        elif key == ord('s') or key == ord('S'):
            # Save frame
            self.save_frame(frame)

        elif key == ord('r') or key == ord('R'):
            # Toggle recording
            self.toggle_recording()

        elif key == ord('i') or key == ord('I'):
            # Toggle info overlay
            self.show_info = not self.show_info
            print(f"Info overlay: {'ON' if self.show_info else 'OFF'}")

        elif key == ord('h') or key == ord('H'):
            # Toggle controls help
            self.show_controls = not self.show_controls

        elif key == ord('f') or key == ord('F'):
            # Toggle face detection
            if self.face_detector:
                self.config.face_detector.enabled = not self.config.face_detector.enabled
                print(f"Face detection: {'ON' if self.config.face_detector.enabled else 'OFF'}")

        elif key == ord('o') or key == ord('O'):
            # Toggle object detection
            if self.object_detector:
                self.config.object_detector.enabled = not self.config.object_detector.enabled
                print(f"Object detection: {'ON' if self.config.object_detector.enabled else 'OFF'}")

        elif key == ord('p') or key == ord('P'):
            # Toggle profiling overlay
            self.show_profiling = not self.show_profiling
            print(f"Profiling overlay: {'ON' if self.show_profiling else 'OFF'}")

        elif key == ord('c') or key == ord('C'):
            # Toggle captioning (OpenCV mode only)
            if self.caption_worker and not self.web_server:
                self.config.captioning.enabled = not self.config.captioning.enabled
                print(f"Captioning: {'ON' if self.config.captioning.enabled else 'OFF'}")

        return True

    def save_frame(self, frame):
        """Save current frame to disk."""
        save_dir = Path(self.config.recording.save_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"frame_{timestamp}.{self.config.recording.image_format}"
        filepath = save_dir / filename

        cv2.imwrite(str(filepath), frame)
        print(f"Saved frame: {filepath}")

    def toggle_recording(self):
        """Toggle video recording."""
        if not self.recording:
            # Start recording
            save_dir = Path(self.config.recording.save_path)
            save_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}.mp4"
            filepath = save_dir / filename

            fourcc = cv2.VideoWriter_fourcc(*self.config.recording.video_codec)
            self.video_writer = cv2.VideoWriter(
                str(filepath),
                fourcc,
                self.config.camera.fps,
                (self.config.camera.width, self.config.camera.height)
            )

            self.recording = True
            print(f"Recording started: {filepath}")
        else:
            # Stop recording
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None

            self.recording = False
            print("Recording stopped")

    def cleanup(self):
        """Clean up resources."""
        print("\nCleaning up...")

        self.running = False

        # Stop recording
        if self.video_writer:
            self.video_writer.release()

        # Stop camera
        if self.camera:
            self.camera.stop()

        # Stop recognition worker
        if self.recognition_worker:
            self.recognition_worker.stop()

        # Stop caption worker
        if self.caption_worker:
            self.caption_worker.stop()

        # Close face detector
        if self.face_detector:
            self.face_detector.close()

        # Close windows
        cv2.destroyAllWindows()

        print("Shutdown complete")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Real-time object detection and face recognition"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="Camera device ID (overrides config)"
    )

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        config = Config.from_yaml(config_path)
    else:
        print(f"Warning: Config file not found: {config_path}")
        print("Using default configuration")
        config = Config()

    # Override camera if specified
    if args.camera is not None:
        config.camera.device_id = args.camera

    # Create and run app
    app = RealtimeDetectionApp(config)
    app.run()


if __name__ == "__main__":
    main()

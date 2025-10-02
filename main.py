"""Main application for real-time object detection and face recognition."""

import cv2
import time
import argparse
from pathlib import Path
from datetime import datetime

from utils.config import Config
from utils.camera import CameraManager, CameraPermissionError, CameraNotFoundError
from utils.visualization import Visualizer
from detectors.object_detector import ObjectDetector
from detectors.face_detector import MediaPipeFaceDetector
from tracking.bytetrack import ByteTracker
from recognition.face_recognition import FaceRecognitionSystem, RecognitionWorker


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
                self.object_detector = ObjectDetector(
                    confidence_threshold=config.object_detector.confidence_threshold,
                    max_detections=config.object_detector.max_detections,
                    device=config.object_detector.device,
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
                )
                self.recognition_worker = RecognitionWorker(self.face_recognition)
                self.recognition_worker.start()
            except Exception as e:
                print(f"Warning: Failed to initialize face recognition: {e}")
                print("Continuing without face recognition...")
                self.face_recognition = None

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
        self.recording = False
        self.video_writer = None

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

        print("\nStarting detection...")
        print("Press 'Q' to quit\n")

        try:
            while self.running:
                loop_start = time.time()

                # Get frame
                result = self.camera.read(timeout=1.0)
                if result is None:
                    print("Warning: No frame received")
                    continue

                timestamp, frame = result
                display_frame = frame.copy()

                # Object detection
                object_detections = []
                if self.object_detector and self.config.object_detector.enabled:
                    object_detections = self.object_detector(frame)

                # Face detection
                face_detections = []
                tracks = []
                if self.face_detector and self.config.face_detector.enabled:
                    face_detections = self.face_detector(frame)

                    # Track faces
                    if self.tracker:
                        tracks = self.tracker.update(face_detections)

                        # Face recognition
                        if self.recognition_worker and self.config.face_recognition.enabled:
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

                # Visualization
                if object_detections:
                    display_frame = self.visualizer.draw_object_boxes(
                        display_frame, object_detections
                    )

                if tracks:
                    display_frame = self.visualizer.draw_face_tracks(
                        display_frame, tracks
                    )

                # Info overlay
                if self.show_info:
                    loop_time = time.time() - loop_start
                    fps = self.visualizer.calculate_fps(loop_time)
                    latency = loop_time * 1000  # Convert to ms

                    display_frame = self.visualizer.draw_info_overlay(
                        display_frame,
                        fps=fps,
                        latency=latency,
                        num_objects=len(object_detections),
                        num_faces=len(tracks),
                    )

                # Controls overlay
                if self.show_controls:
                    display_frame = self.visualizer.draw_controls_help(display_frame)

                # Recording
                if self.recording and self.video_writer:
                    self.video_writer.write(display_frame)

                # Display
                cv2.imshow("Real-time Detection", display_frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if not self.handle_key(key, display_frame):
                    break

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

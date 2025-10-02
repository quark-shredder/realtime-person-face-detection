"""Face enrollment CLI - capture and enroll faces for recognition."""

import cv2
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from recognition.face_recognition import FaceRecognitionSystem
from utils.config import Config


def main():
    """Main enrollment function."""
    print("=" * 60)
    print("FACE ENROLLMENT")
    print("=" * 60)

    # Load config
    config_path = Path("config/default.yaml")
    if config_path.exists():
        config = Config.from_yaml(config_path)
    else:
        print("Warning: Config not found, using defaults")
        config = Config()

    # Get person's name
    name = input("\nEnter person's name: ").strip()
    if not name:
        print("Error: Name cannot be empty")
        sys.exit(1)

    # Initialize recognition system
    print("\nInitializing face recognition system...")
    face_system = FaceRecognitionSystem(
        model_pack=config.face_recognition.model_pack,
        enrolled_faces_path=config.face_recognition.enrolled_faces_path,
        similarity_threshold=config.face_recognition.similarity_threshold,
    )

    # Initialize camera
    print("\nStarting camera...")
    cap = cv2.VideoCapture(config.camera.device_id)

    if not cap.isOpened():
        print("Error: Could not open camera")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera.height)

    # Capture frames
    num_samples = config.face_recognition.enrollment_samples
    samples = []

    print(f"\nCollecting {num_samples} samples...")
    print("Instructions:")
    print("  - Look at the camera")
    print("  - Press SPACE to capture a sample")
    print("  - Press ESC to cancel")
    print("  - After capturing all samples, enrollment will complete")
    print()

    while len(samples) < num_samples:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read from camera")
            break

        # Display
        display_frame = frame.copy()
        cv2.putText(
            display_frame,
            f"Samples: {len(samples)}/{num_samples}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            display_frame,
            "SPACE: capture | ESC: cancel",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        cv2.imshow("Enrollment", display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            print("\nEnrollment cancelled")
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)
        elif key == 32:  # SPACE
            samples.append(frame.copy())
            print(f"  Captured sample {len(samples)}/{num_samples}")

    cap.release()
    cv2.destroyAllWindows()

    # Enroll face
    print(f"\nEnrolling {name}...")
    success = face_system.enroll_face(name, samples)

    if success:
        print(f"\n✓ Successfully enrolled {name}!")
        print(f"  Total enrolled faces: {len(face_system.enrolled_faces)}")
        print("\nYou can now run the main detection:")
        print("  python main.py")
    else:
        print("\n✗ Enrollment failed")
        print("  Please ensure your face is clearly visible in all samples")
        sys.exit(1)


if __name__ == "__main__":
    main()

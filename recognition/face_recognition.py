"""Face recognition using InsightFace embeddings."""

import cv2
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from insightface.app import FaceAnalysis
import threading
import queue


class FaceRecognitionSystem:
    """Face recognition with InsightFace embeddings."""

    def __init__(
        self,
        model_pack: str = "buffalo_l",
        enrolled_faces_path: str = "data/enrolled",
        similarity_threshold: float = 0.50,
    ):
        """
        Initialize face recognition system.

        Args:
            model_pack: InsightFace model pack (buffalo_l, buffalo_s)
            enrolled_faces_path: Path to enrolled face embeddings
            similarity_threshold: Cosine similarity threshold for matching
        """
        self.model_pack = model_pack
        self.enrolled_path = Path(enrolled_faces_path)
        self.threshold = similarity_threshold

        # Ensure directory exists
        self.enrolled_path.mkdir(parents=True, exist_ok=True)

        # Initialize InsightFace
        print(f"Loading InsightFace model pack: {model_pack}")
        self.app = FaceAnalysis(name=model_pack)
        self.app.prepare(ctx_id=0, det_size=(256, 256))

        # Load enrolled faces
        self.enrolled_faces: Dict[str, np.ndarray] = {}
        self._load_enrolled_faces()

        print(f"Face recognition ready. {len(self.enrolled_faces)} enrolled faces.")

    def _load_enrolled_faces(self):
        """Load enrolled face embeddings from disk."""
        manifest_path = self.enrolled_path / "manifest.json"

        if not manifest_path.exists():
            print("No enrolled faces found. Use enrollment script first.")
            return

        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        for name, embedding_file in manifest.items():
            embedding_path = self.enrolled_path / embedding_file
            if embedding_path.exists():
                data = np.load(embedding_path)
                self.enrolled_faces[name] = data['embedding']
                print(f"  Loaded: {name}")

    def enroll_face(self, name: str, images: List[np.ndarray]) -> bool:
        """
        Enroll a person's face from multiple images.

        Args:
            name: Person's name
            images: List of face images (BGR format)

        Returns:
            True if successful
        """
        embeddings = []

        for img in images:
            # Convert to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Get face embedding
            faces = self.app.get(img_rgb)

            if len(faces) == 0:
                print(f"Warning: No face detected in one image")
                continue

            # Use the largest face
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            embeddings.append(face.normed_embedding)

        if len(embeddings) < 3:
            print(f"Error: Need at least 3 valid face samples, got {len(embeddings)}")
            return False

        # Average embeddings
        mean_embedding = np.mean(np.vstack(embeddings), axis=0).astype(np.float32)

        # Normalize
        mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)

        # Save embedding
        embedding_file = f"{name}.npz"
        embedding_path = self.enrolled_path / embedding_file
        np.savez(embedding_path, embedding=mean_embedding)

        # Update manifest
        manifest_path = self.enrolled_path / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
        else:
            manifest = {}

        manifest[name] = embedding_file

        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        # Add to in-memory database
        self.enrolled_faces[name] = mean_embedding

        print(f"✓ Enrolled {name} with {len(embeddings)} samples")
        return True

    def recognize_face(self, image: np.ndarray, bbox: List[int]) -> Tuple[str, float]:
        """
        Recognize a face in an image.

        Args:
            image: Full image (BGR format)
            bbox: Face bounding box [x1, y1, x2, y2]

        Returns:
            (name, confidence) tuple. Returns ("unknown", 0.0) if no match.
        """
        # Crop and align face
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)

        face_crop = image[y1:y2, x1:x2]

        if face_crop.size == 0:
            return "unknown", 0.0

        # Convert to RGB
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

        # Get embedding
        faces = self.app.get(face_rgb)

        if len(faces) == 0:
            return "unknown", 0.0

        # Use the largest face
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        embedding = face.normed_embedding

        # Match against enrolled faces
        best_match = ("unknown", 0.0)

        for name, enrolled_emb in self.enrolled_faces.items():
            # Cosine similarity
            similarity = float(np.dot(embedding, enrolled_emb))

            if similarity > best_match[1]:
                best_match = (name, similarity)

        # Check threshold
        if best_match[1] >= self.threshold:
            return best_match
        else:
            return "unknown", best_match[1]


class RecognitionWorker:
    """Background worker for face recognition to avoid blocking main loop."""

    def __init__(self, recognition_system: FaceRecognitionSystem):
        """
        Initialize recognition worker.

        Args:
            recognition_system: Face recognition system instance
        """
        self.system = recognition_system
        self.task_queue: queue.Queue = queue.Queue(maxsize=10)
        self.result_queue: queue.Queue = queue.Queue()
        self.running = False
        self.worker_thread: Optional[threading.Thread] = None

    def start(self):
        """Start worker thread."""
        if self.running:
            return

        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def _worker_loop(self):
        """Background recognition loop."""
        while self.running:
            try:
                task = self.task_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            track_id, image, bbox = task

            # Perform recognition
            name, confidence = self.system.recognize_face(image, bbox)

            # Put result
            try:
                self.result_queue.put_nowait((track_id, name, confidence))
            except queue.Full:
                pass

            self.task_queue.task_done()

    def submit(self, track_id: int, image: np.ndarray, bbox: List[int]) -> bool:
        """
        Submit a recognition task.

        Args:
            track_id: Track ID
            image: Full image
            bbox: Face bounding box

        Returns:
            True if task was queued
        """
        try:
            self.task_queue.put_nowait((track_id, image.copy(), bbox))
            return True
        except queue.Full:
            return False

    def get_results(self) -> List[Tuple[int, str, float]]:
        """
        Get all available recognition results.

        Returns:
            List of (track_id, name, confidence) tuples
        """
        results = []
        while not self.result_queue.empty():
            try:
                results.append(self.result_queue.get_nowait())
            except queue.Empty:
                break
        return results

    def stop(self):
        """Stop worker thread."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=2.0)

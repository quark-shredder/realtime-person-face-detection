"""Configuration management for real-time detection system."""

import yaml
from pathlib import Path
from typing import Any, Dict
from dataclasses import dataclass, field


@dataclass
class CameraConfig:
    device_id: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    buffer_size: int = 2


@dataclass
class ObjectDetectorConfig:
    enabled: bool = True
    model: str = "yolox_tiny"
    input_size: int = 640
    confidence_threshold: float = 0.50
    nms_threshold: float = 0.45
    max_detections: int = 50
    device: str = "mps"
    model_path: str = "data/models/yolox_tiny.pth"


@dataclass
class FaceDetectorConfig:
    enabled: bool = True
    model: str = "mediapipe"
    min_detection_confidence: float = 0.5
    model_selection: int = 0


@dataclass
class FaceRecognitionConfig:
    enabled: bool = True
    model_pack: str = "buffalo_l"
    embedding_model: str = "mobilefacenet"
    similarity_threshold: float = 0.50
    enrollment_samples: int = 15
    recognition_interval: int = 30
    enrolled_faces_path: str = "data/enrolled"
    ctx_id: int = -1  # -1 for CPU, 0 for GPU
    det_size: int = 128  # Detection size for InsightFace


@dataclass
class TrackingConfig:
    enabled: bool = True
    tracker_type: str = "bytetrack"
    track_thresh: float = 0.5
    track_buffer: int = 30
    match_thresh: float = 0.8
    min_box_area: int = 100


@dataclass
class PerformanceConfig:
    target_fps: int = 30
    num_threads: int = 3
    enable_profiling: bool = False
    adaptive_quality: bool = True


@dataclass
class VisualizationConfig:
    show_fps: bool = True
    show_latency: bool = True
    show_track_ids: bool = True
    show_confidence: bool = True
    box_thickness: int = 2
    font_scale: float = 0.6
    colors: Dict[str, list] = field(default_factory=lambda: {
        "object": [0, 255, 0],
        "face": [255, 165, 0],
        "recognized": [0, 255, 255],
        "unknown": [128, 128, 128]
    })


@dataclass
class RecordingConfig:
    save_path: str = "logs/recordings"
    video_codec: str = "mp4v"
    image_format: str = "jpg"


@dataclass
class LoggingConfig:
    level: str = "INFO"
    log_file: str = "logs/app.log"
    rotation: str = "10 MB"


@dataclass
class CaptioningConfig:
    enabled: bool = False
    server_url: str = "http://localhost:8080"
    prompt: str = "Describe what you see in one sentence."
    interval_seconds: float = 0.5
    jpeg_quality: int = 80
    max_tokens: int = 100
    timeout_seconds: float = 8.0


@dataclass
class WebUIConfig:
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 8000
    frame_quality: int = 70
    stream_fps: int = 15
    send_overlays: bool = True


@dataclass
class Config:
    """Main configuration class."""
    camera: CameraConfig = field(default_factory=CameraConfig)
    object_detector: ObjectDetectorConfig = field(default_factory=ObjectDetectorConfig)
    face_detector: FaceDetectorConfig = field(default_factory=FaceDetectorConfig)
    face_recognition: FaceRecognitionConfig = field(default_factory=FaceRecognitionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    recording: RecordingConfig = field(default_factory=RecordingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    captioning: CaptioningConfig = field(default_factory=CaptioningConfig)
    web_ui: WebUIConfig = field(default_factory=WebUIConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        return cls(
            camera=CameraConfig(**data.get('camera', {})),
            object_detector=ObjectDetectorConfig(**data.get('object_detector', {})),
            face_detector=FaceDetectorConfig(**data.get('face_detector', {})),
            face_recognition=FaceRecognitionConfig(**data.get('face_recognition', {})),
            tracking=TrackingConfig(**data.get('tracking', {})),
            performance=PerformanceConfig(**data.get('performance', {})),
            visualization=VisualizationConfig(**data.get('visualization', {})),
            recording=RecordingConfig(**data.get('recording', {})),
            logging=LoggingConfig(**data.get('logging', {})),
            captioning=CaptioningConfig(**data.get('captioning', {})),
            web_ui=WebUIConfig(**data.get('web_ui', {})),
        )

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        from dataclasses import asdict

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False)

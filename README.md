# Real-time Object Detection & Face Recognition

High-performance real-time object detection and face recognition system optimized for Apple Silicon (M4 MacBook Air). Achieves **30+ FPS at 480p** using YOLOX-Nano for object detection, MediaPipe for face detection, and InsightFace for face recognition.

## Features

- **Real-time object detection** with YOLOX-Nano (80 COCO classes)
- **Fast face detection** using MediaPipe
- **Face recognition** with InsightFace embeddings
- **Multi-object tracking** using ByteTrack
- **Threaded architecture** for maximum performance
- **MPS acceleration** (Apple Metal Performance Shaders)
- **Live video recording** and frame capture
- **Interactive controls** with keyboard shortcuts

## System Requirements

- **Hardware**: Apple Silicon Mac (M1/M2/M3/M4) or Intel Mac with GPU
- **OS**: macOS 12.0+
- **RAM**: 8GB minimum, 16GB recommended
- **Python**: 3.10 or higher

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/quark-shredder/realtime-person-face-detection.git
cd realtime-person-face-detection
```

### 2. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -e .
```

### 4. Download models

```bash
python -m scripts.download_models
```

This will download:
- YOLOX-Nano weights (~4MB)
- InsightFace models (auto-downloaded on first use, ~100MB)

## Quick Start

### 1. Enroll faces (optional, for face recognition)

```bash
python -m scripts.enroll
```

Follow the prompts to capture 15 samples of each person you want to recognize.

### 2. Run detection

```bash
python main.py
```

### 3. Keyboard controls

| Key | Action |
|-----|--------|
| `Q` | Quit application |
| `S` | Save current frame |
| `R` | Start/stop video recording |
| `I` | Toggle info overlay |
| `H` | Toggle controls help |
| `F` | Toggle face detection |
| `O` | Toggle object detection |

## Configuration

Edit `config/default.yaml` to customize:

- **Camera settings**: resolution, FPS, device ID
- **Model parameters**: confidence thresholds, input sizes
- **Performance**: threading, target FPS
- **Visualization**: colors, overlay options
- **Recording**: video codec, save paths

Example:

```yaml
camera:
  device_id: 0
  width: 640
  height: 480
  fps: 30

object_detector:
  enabled: true
  confidence_threshold: 0.30
  device: "mps"  # mps, cuda, or cpu

face_recognition:
  enabled: true
  similarity_threshold: 0.50
  enrollment_samples: 15
```

## Architecture

```
┌─────────────┐
│   Camera    │  ← Threaded capture (non-blocking)
│  (Thread 1) │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────┐
│         Main Processing Loop            │
│  ┌────────────┐      ┌───────────────┐ │
│  │   YOLOX    │      │   MediaPipe   │ │
│  │ (Objects)  │      │    (Faces)    │ │
│  │   [MPS]    │      │     [CPU]     │ │
│  └────────────┘      └───────┬───────┘ │
│                              │          │
│                      ┌───────▼────────┐ │
│                      │   ByteTrack    │ │
│                      │   (Tracking)   │ │
│                      └───────┬────────┘ │
└──────────────────────────────┼──────────┘
                               │
                       ┌───────▼─────────┐
                       │  Recognition    │
                       │   Worker        │
                       │  (Thread 3)     │
                       │  InsightFace    │
                       └─────────────────┘
```

## Performance Tips

### Achieving 30+ FPS

The system is optimized for real-time performance:

1. **Object detection** runs on MPS (GPU)
2. **Face detection** runs on CPU (leaves GPU free)
3. **Face recognition** runs in background thread (non-blocking)
4. **Tracking** reduces unnecessary recognition calls

### If experiencing low FPS:

1. **Lower camera resolution**:
   ```yaml
   camera:
     width: 416
     height: 416
   ```

2. **Increase confidence thresholds**:
   ```yaml
   object_detector:
     confidence_threshold: 0.40  # Fewer detections
   ```

3. **Disable object detection** (face-only mode):
   ```yaml
   object_detector:
     enabled: false
   ```

4. **Reduce recognition frequency**:
   ```yaml
   face_recognition:
     recognition_interval: 60  # Re-recognize every 60 frames
   ```

## Advanced Usage

### Custom camera

```bash
python main.py --camera 1  # Use external webcam
```

### Custom configuration

```bash
python main.py --config my_config.yaml
```

### Programmatic usage

```python
from utils.config import Config
from main import RealtimeDetectionApp

config = Config.from_yaml("config/default.yaml")
app = RealtimeDetectionApp(config)
app.run()
```

## Project Structure

```
.
├── config/
│   └── default.yaml          # Configuration file
├── data/
│   ├── enrolled/             # Enrolled face embeddings
│   └── models/               # Downloaded model weights
├── detectors/
│   ├── yolox_detector.py     # YOLOX object detector
│   └── face_detector.py      # MediaPipe face detector
├── recognition/
│   └── face_recognition.py   # InsightFace recognition
├── tracking/
│   └── bytetrack.py          # ByteTrack tracker
├── utils/
│   ├── camera.py             # Threaded camera manager
│   ├── config.py             # Configuration management
│   └── visualization.py      # Rendering utilities
├── scripts/
│   ├── download_models.py    # Model download script
│   └── enroll.py             # Face enrollment CLI
└── main.py                   # Main application
```

## Troubleshooting

### "Failed to open camera"

- Check camera permissions: System Settings → Privacy & Security → Camera
- Try different camera ID: `python main.py --camera 0`
- Ensure no other app is using the camera

### "Model not found"

```bash
python -m scripts.download_models
```

### "MPS not available"

The system will automatically fall back to CPU. To force CPU:

```yaml
object_detector:
  device: "cpu"
```

### Low FPS on Intel Mac

Intel Macs don't have MPS. Use CPU mode and reduce resolution:

```yaml
camera:
  width: 416
  height: 416
object_detector:
  device: "cpu"
```

## License

MIT License - see LICENSE file for details

## Acknowledgments

- **YOLOX**: [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
- **MediaPipe**: [Google MediaPipe](https://github.com/google/mediapipe)
- **InsightFace**: [deepinsight/insightface](https://github.com/deepinsight/insightface)
- **ByteTrack**: [ifzhang/ByteTrack](https://github.com/ifzhang/ByteTrack)

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## Privacy & Ethics

This software is intended for:
- ✅ Personal use and experimentation
- ✅ Security and access control (with proper consent)
- ✅ Research and development

Always obtain consent before enrolling faces. Follow local privacy laws and regulations.

# Real-time Object Detection & Face Recognition

High-performance real-time object detection and face recognition system optimized for Apple Silicon (M4 MacBook Air). Achieves **30+ FPS at 480p** using YOLOX-Nano for object detection, MediaPipe for face detection, and InsightFace for face recognition.

## Features

- **Real-time object detection** with YOLOX-Nano (80 COCO classes)
- **Fast face detection** using MediaPipe
- **Face recognition** with InsightFace embeddings
- **Multi-object tracking** using ByteTrack
- **SmolVLM image captioning** with live scene descriptions
- **Browser UI** with WebSocket streaming (optional)
- **Threaded architecture** for maximum performance
- **MPS acceleration** (Apple Metal Performance Shaders)
- **Live video recording** and frame capture
- **Interactive controls** with keyboard shortcuts or browser UI

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

### 3. Keyboard controls (OpenCV mode)

| Key | Action |
|-----|--------|
| `Q` | Quit application |
| `S` | Save current frame |
| `R` | Start/stop video recording |
| `I` | Toggle info overlay |
| `H` | Toggle controls help |
| `F` | Toggle face detection |
| `O` | Toggle object detection |
| `C` | Toggle captioning |

**Note:** Keyboard shortcuts only work in OpenCV window mode. When using Browser UI (see below), use the web interface controls instead.

## Browser UI & SmolVLM Captioning

### Overview

The system supports an optional **browser-based UI** with real-time streaming and **SmolVLM image captioning**. This mode:

- Streams video to your browser via WebSocket (configurable FPS)
- Displays detected objects and faces in sidebar lists
- Shows live captions with latency/age metrics
- Allows runtime config changes (prompt, interval, feature toggles)
- Works on headless servers or remote machines

### Prerequisites

To use captioning, you need:

1. **llama.cpp** with server support
2. **SmolVLM model** (GGUF format with mmproj)

### Installation: llama.cpp + SmolVLM

#### Option 1: Homebrew (macOS)

```bash
# Install llama.cpp
brew install llama.cpp

# Download SmolVLM model (HuggingFace-to-GGUF format)
# Example using HF CLI:
huggingface-cli download HuggingFaceTB/SmolVLM-Instruct-GGUF \
  smolvlm-instruct-q4_k_m.gguf smolvlm-instruct-mmproj-f16.gguf \
  --local-dir ./data/models/smolvlm
```

#### Option 2: Build from source

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make LLAMA_METAL=1  # macOS with Metal

# Download SmolVLM (see Option 1 above)
```

### Configuration

Edit `config/default.yaml`:

```yaml
# Enable captioning
captioning:
  enabled: true
  server_url: "http://localhost:8080"
  prompt: "Describe what you see in one sentence."
  interval_seconds: 0.5        # Caption refresh rate
  jpeg_quality: 80             # Encoding quality
  max_tokens: 100
  timeout_seconds: 8.0

# Enable browser UI
web_ui:
  enabled: true
  host: "127.0.0.1"
  port: 8000
  frame_quality: 70            # Browser stream quality
  stream_fps: 15               # Browser stream rate
  send_overlays: true          # Show detection boxes in browser
```

**Important Notes:**
- Set `captioning.enabled: true` to enable SmolVLM captions
- Set `web_ui.enabled: true` to use browser UI (disables OpenCV window)
- When `web_ui.enabled: true`, keyboard shortcuts are disabled
- Use browser controls for feature toggles when in web mode

### Running with Browser UI

#### 1. Start llama.cpp server

```bash
# Navigate to llama.cpp directory
cd llama.cpp

# Start server with SmolVLM
./llama-server \
  --model ./data/models/smolvlm/smolvlm-instruct-q4_k_m.gguf \
  --mmproj ./data/models/smolvlm/smolvlm-instruct-mmproj-f16.gguf \
  --port 8080 \
  --host 127.0.0.1 \
  --ctx-size 2048
```

**Verify server is running:**
```bash
curl http://localhost:8080/health
# Should return: {"status":"ok"}
```

#### 2. Run detection with web UI

```bash
# In your project directory
python main.py
```

**Expected output:**
```
Initializing camera...
Initializing object detector...
Initializing face detector...
Initializing face tracker...
Initializing face recognition...
Initializing caption worker...
Initializing web server...
✓ Web UI available at http://127.0.0.1:8000
✓ Caption worker started

Starting detection...
Press 'Q' to quit (Ctrl+C to stop)
```

#### 3. Open browser

Navigate to: **http://localhost:8000**

### Browser UI Features

The web interface shows:

- **Live video feed** with detection overlays (left panel)
- **Caption display** with latency and age metrics (below video)
- **Toggle buttons** for Objects / Faces / Caption detection (right panel)
- **Prompt editor** for customizing caption instructions (live updates)
- **Interval selector** for caption refresh rate (300ms - 2000ms)
- **Detected objects list** with confidence scores (right panel)
- **Detected faces list** with track IDs and names (right panel)
- **FPS counter** showing detection performance

### Troubleshooting Browser UI

#### "WebSocket connection failed"

- Ensure detection app is running (`python main.py`)
- Check port 8000 is not in use: `lsof -i :8000`
- Verify config: `web_ui.enabled: true` and `web_ui.port: 8000`

#### "Disconnected from server" with auto-reconnect

This is normal during:
- Server restarts
- Network hiccups
- Excessive lag

The browser will auto-reconnect with exponential backoff (1s → 2s → 4s → 8s → 16s → 30s max).

#### "Caption error: Connection refused"

- llama.cpp server not running → Start `llama-server` (see step 1 above)
- Wrong port in config → Check `captioning.server_url` matches llama.cpp port
- Server crashed → Check llama.cpp logs

#### "Caption error: Timeout"

- Model too slow for interval → Increase `captioning.interval_seconds` to 1.0 or 2.0
- Increase timeout: `captioning.timeout_seconds: 15.0`
- Use smaller model (Q4_K_M instead of F16)

#### Captions are stale/slow

- Check "Age" metric in browser (should be < 1000ms)
- Increase interval: `captioning.interval_seconds: 1.0`
- Model inference too slow → Use smaller quantization (Q4_K_S)
- Toggle caption OFF when not needed (saves CPU/bandwidth)

#### Low browser FPS (choppy video)

- Increase `web_ui.stream_fps` from 15 to 20 or 30
- Reduce `web_ui.frame_quality` from 70 to 50 (less bandwidth)
- Check network if accessing remotely

### Remote Access

To access browser UI from another device:

```yaml
web_ui:
  host: "0.0.0.0"  # Listen on all interfaces
  port: 8000
```

Then open: `http://<your-mac-ip>:8000`

**Security note:** This exposes your camera feed. Only use on trusted networks or behind a firewall.

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
┌──────────────────────────────────────────────────────────┐
│              Main Processing Loop                        │
│  ┌────────────┐      ┌───────────────┐                  │
│  │   YOLOX    │      │   MediaPipe   │                  │
│  │ (Objects)  │      │    (Faces)    │                  │
│  │   [MPS]    │      │     [CPU]     │                  │
│  └────────────┘      └───────┬───────┘                  │
│                              │                           │
│                      ┌───────▼────────┐                  │
│                      │   ByteTrack    │                  │
│                      │   (Tracking)   │                  │
│                      └───────┬────────┘                  │
└──────────────────────────────┼───────────────────────────┘
                               │
          ┌────────────────────┼──────────────────┐
          │                    │                  │
  ┌───────▼─────────┐  ┌───────▼────────┐  ┌─────▼──────┐
  │  Recognition    │  │  Caption        │  │ Web Server │
  │   Worker        │  │  Worker         │  │ (Optional) │
  │  (Thread 2)     │  │  (Thread 3)     │  │ (Thread 4) │
  │  InsightFace    │  │  SmolVLM API    │  │  FastAPI   │
  └─────────────────┘  └─────────────────┘  └────┬───────┘
                                                  │
                                            WebSocket
                                                  │
                                            ┌─────▼──────┐
                                            │  Browser   │
                                            │  (Chrome)  │
                                            └────────────┘
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
├── captioning/
│   ├── smolvlm_api.py        # SmolVLM API client
│   └── caption_worker.py     # Caption worker thread
├── web/
│   ├── server.py             # FastAPI WebSocket server
│   ├── launcher.py           # Web server launcher
│   └── index.html            # Browser UI
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
- **SmolVLM**: [HuggingFaceTB/SmolVLM](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct)
- **llama.cpp**: [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## Privacy & Ethics

This software is intended for:
- ✅ Personal use and experimentation
- ✅ Security and access control (with proper consent)
- ✅ Research and development

Always obtain consent before enrolling faces. Follow local privacy laws and regulations.

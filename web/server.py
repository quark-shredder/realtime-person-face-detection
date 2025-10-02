"""FastAPI WebSocket server for browser UI."""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pathlib import Path
import asyncio
import base64
import cv2
import threading
from typing import List
import numpy as np


app = FastAPI()

# Shared state (thread-safe with lock)
state_lock = threading.Lock()
stream_fps = 15  # Default, can be updated via set_stream_fps()
state = {
    "frame": None,
    "objects": [],
    "faces": [],
    "caption": {"text": "", "error": None, "latency_ms": 0, "age_ms": 0},
    "fps": 0.0,
    "stats": {},
    "config_update": None,  # Pending config changes from browser
    "toggle": None  # Pending toggle commands
}

active_connections: List[WebSocket] = []


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for streaming data to browser."""
    await websocket.accept()
    active_connections.append(websocket)

    try:
        while True:
            # Send state to client (thread-safe read)
            with state_lock:
                data = state.copy()

            await websocket.send_json(data)
            await asyncio.sleep(1.0 / stream_fps)  # Dynamic FPS from config

    except WebSocketDisconnect:
        active_connections.remove(websocket)


@app.post("/api/config")
async def update_config(data: dict):
    """Receive config updates from browser (prompt, interval)."""
    with state_lock:
        state["config_update"] = data
    return {"status": "ok"}


@app.post("/api/toggle")
async def toggle_feature(data: dict):
    """Toggle detection features (objects/faces/caption)."""
    with state_lock:
        state["toggle"] = data  # {"feature": "caption", "enabled": true}
    return {"status": "ok"}


@app.get("/")
async def get_index():
    """Serve main UI page."""
    index_path = Path(__file__).parent / "index.html"
    return FileResponse(index_path)


def update_state(
    frame: np.ndarray,
    objects: list,
    faces: list,
    caption: dict,
    fps: float,
    stats: dict,
    frame_quality: int = 70  # Default, should be overridden
):
    """
    Called by main.py detection loop (thread-safe write).

    Args:
        frame: Current frame (with or without overlays based on config)
        objects: List of detected objects
        faces: List of face tracks
        caption: Caption dict with text, error, latency, age
        fps: Current FPS
        stats: Profiling stats
        frame_quality: JPEG quality for browser streaming
    """
    # Encode frame to base64 JPEG
    _, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, frame_quality])
    frame_b64 = base64.b64encode(jpg.tobytes()).decode()

    with state_lock:
        state["frame"] = frame_b64
        state["objects"] = [
            {"label": name, "conf": conf, "bbox": bbox}
            for bbox, cls_id, conf, name in objects
        ]
        state["faces"] = [
            {"name": track.label or "Unknown", "track_id": track.track_id, "bbox": track.bbox}
            for track in faces
        ]
        state["caption"] = caption
        state["fps"] = fps
        state["stats"] = stats


def pop_config_update():
    """Get pending config update (called from main loop)."""
    with state_lock:
        update = state.get("config_update")
        state["config_update"] = None
        return update


def pop_toggle():
    """Get pending toggle command."""
    with state_lock:
        toggle = state.get("toggle")
        state["toggle"] = None
        return toggle


def set_stream_fps(fps: int):
    """Set browser stream FPS (called from launcher)."""
    global stream_fps
    stream_fps = fps

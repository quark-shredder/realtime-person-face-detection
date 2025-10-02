"""SmolVLM API client for llama.cpp server."""

import requests
import base64
import cv2
import time
import numpy as np


def get_caption(
    frame_bgr: np.ndarray,
    prompt: str,
    server_url: str,
    jpeg_quality: int = 80,
    max_tokens: int = 100,
    timeout: float = 8.0
) -> dict:
    """
    Get caption from llama.cpp server.

    Args:
        frame_bgr: Input frame (BGR format)
        prompt: Caption instruction
        server_url: llama.cpp server URL (e.g., http://localhost:8080)
        jpeg_quality: JPEG compression quality (0-100)
        max_tokens: Maximum tokens in response
        timeout: Request timeout in seconds

    Returns:
        Dict with keys: text, latency_ms, error, timestamp
    """
    # Encode frame to JPEG
    ok, jpg = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        return {
            "text": "",
            "latency_ms": 0,
            "error": "Failed to encode frame",
            "timestamp": time.time()
        }

    # Base64 encode
    b64 = base64.b64encode(jpg.tobytes()).decode('utf-8')

    # Build OpenAI-compatible payload
    payload = {
        "max_tokens": max_tokens,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            ]
        }]
    }

    start = time.perf_counter()

    try:
        resp = requests.post(
            f"{server_url}/v1/chat/completions",
            json=payload,
            timeout=timeout
        )
        resp.raise_for_status()

        # Extract caption text
        text = resp.json()["choices"][0]["message"]["content"]
        latency = (time.perf_counter() - start) * 1000

        return {
            "text": text,
            "latency_ms": latency,
            "error": None,
            "timestamp": time.time()
        }

    except Exception as e:
        return {
            "text": "",
            "latency_ms": 0,
            "error": str(e),
            "timestamp": time.time()
        }

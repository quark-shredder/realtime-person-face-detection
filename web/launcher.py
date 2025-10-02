"""Web server launcher (runs in background thread)."""

import threading
import uvicorn


def start_web_server(host: str, port: int, stream_fps: int = 15):
    """
    Start FastAPI server in background thread.

    Args:
        host: Server host address
        port: Server port
        stream_fps: Browser stream FPS (frames per second)

    Returns:
        Thread instance
    """
    # Set stream FPS before starting server
    from web import server
    server.set_stream_fps(stream_fps)

    def run():
        uvicorn.run(
            "web.server:app",
            host=host,
            port=port,
            log_level="warning"
        )

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    print(f"✓ Web UI available at http://{host}:{port}")
    return thread

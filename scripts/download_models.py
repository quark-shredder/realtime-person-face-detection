"""Download and setup required models."""

import os
import sys
import hashlib
from pathlib import Path
from urllib.request import urlretrieve
from tqdm import tqdm


class DownloadProgress:
    """Progress bar for downloads."""

    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar = tqdm(total=total_size, unit='B', unit_scale=True)
        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(block_size)
        else:
            self.pbar.close()


def verify_checksum(filepath: Path, expected_md5: str) -> bool:
    """Verify file MD5 checksum."""
    md5 = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            md5.update(chunk)
    return md5.hexdigest() == expected_md5


def download_yolox_nano():
    """Download YOLOX-Nano pretrained weights."""
    print("\n[1/2] Downloading YOLOX-Nano model...")

    model_dir = Path("data/models")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "yolox_nano.pth"

    if model_path.exists():
        print(f"✓ Model already exists: {model_path}")
        return True

    # Official YOLOX-Nano weights from Megvii GitHub releases
    url = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth"

    try:
        print(f"  Downloading from: {url}")
        urlretrieve(url, model_path, DownloadProgress())
        print(f"✓ Downloaded: {model_path}")
        return True
    except Exception as e:
        print(f"✗ Failed to download YOLOX-Nano: {e}")
        return False


def setup_insightface():
    """Setup InsightFace models (will auto-download on first use)."""
    print("\n[2/2] Setting up InsightFace...")

    models_dir = Path.home() / ".insightface/models"

    print(f"  InsightFace models will be downloaded to: {models_dir}")
    print(f"  Models auto-download on first use (~100MB)")

    # Create directory structure
    models_dir.mkdir(parents=True, exist_ok=True)

    print("✓ InsightFace setup complete")
    return True


def download_all():
    """Download all required models."""
    print("=" * 60)
    print("MODEL DOWNLOAD SCRIPT")
    print("=" * 60)

    success = True

    # YOLOX
    if not download_yolox_nano():
        success = False

    # InsightFace
    if not setup_insightface():
        success = False

    print("\n" + "=" * 60)
    if success:
        print("✓ All models ready!")
        print("\nYou can now run:")
        print("  python -m scripts.enroll  # to enroll faces")
        print("  python main.py            # to start detection")
    else:
        print("✗ Some downloads failed. Please check errors above.")
        sys.exit(1)
    print("=" * 60)


def main():
    """Main entry point."""
    download_all()


if __name__ == "__main__":
    main()

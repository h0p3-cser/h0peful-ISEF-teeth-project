"""OpenCV image I/O through pathlib, including non-ASCII Windows paths."""
from pathlib import Path

import cv2
import numpy as np


def read_image(path, flags):
    path = Path(path)
    data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(data, flags) if data.size else None
    if image is None:
        raise ValueError(f"Unreadable image: {path}")
    return image


def write_png(path, image):
    ok, data = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise OSError(f"Cannot encode {path}")
    Path(path).write_bytes(data.tobytes())

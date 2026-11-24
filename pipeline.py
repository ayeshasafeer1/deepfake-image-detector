"""Image preprocessing for the deepfake detector (no torch needed).

Steps for an uploaded photo:
  1. load, fix EXIF rotation, convert to RGB, cap the size
  2. optionally find the largest face and crop around it with a margin
  3. resize/crop to 256x256, the size of the training images
The model code then centre-crops to 224x224 (see model_utils.py).
"""
import io

import numpy as np
from PIL import Image, ImageOps
from matplotlib import colormaps
from skimage import data
from skimage.feature import Cascade

IMG_SIZE = 256        # size of the training images
CROP_SIZE = 224       # network input size (centre crop of the 256x256 image)
MAX_SIDE = 2000       # larger uploads are shrunk to this many pixels on the long side
DETECT_SIDE = 640     # face detection runs on a copy shrunk to this size

_detector = None


def _get_detector():
    global _detector
    if _detector is None:
        _detector = Cascade(data.lbp_frontal_face_cascade_filename())
    return _detector


def load_image(raw: bytes) -> Image.Image:
    """Open uploaded bytes as an RGB image with the right orientation."""
    img = Image.open(io.BytesIO(raw))
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    return img


def detect_faces(img: Image.Image):
    """Return face boxes (x, y, w, h) in the coordinates of `img`, largest first."""
    scale = min(1.0, DETECT_SIDE / max(img.size))
    small = img if scale == 1.0 else img.resize(
        (max(1, round(img.width * scale)), max(1, round(img.height * scale))))
    gray = np.asarray(small.convert("L"))
    h, w = gray.shape
    if min(h, w) < 60:
        return []
    try:
        found = _get_detector().detect_multi_scale(
            img=gray, scale_factor=1.2, step_ratio=1,
            min_size=(60, 60), max_size=(w, h))      # sizes are (width, height) tuples
    except Exception:
        return []
    boxes = [(d["c"] / scale, d["r"] / scale, d["width"] / scale, d["height"] / scale) for d in found]
    return sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)


def prepare(img: Image.Image, detect: bool = True, margin: float = 0.75):
    """Return (256x256 RGB image for the model, info dict)."""
    info = {"faces": 0, "cropped": False}
    work = img
    if detect:
        faces = detect_faces(img)
        info["faces"] = len(faces)
        if faces:
            x, y, w, h = faces[0]
            cx, cy, half = x + w / 2, y + h / 2, max(w, h) * margin
            box = (max(cx - half, 0), max(cy - half, 0),
                   min(cx + half, img.width), min(cy + half, img.height))
            work = img.crop(tuple(int(round(v)) for v in box))
            info["cropped"] = True
    return ImageOps.fit(work, (IMG_SIZE, IMG_SIZE), Image.BICUBIC), info


def center_crop(img: Image.Image) -> np.ndarray:
    """The exact 224x224 region the network sees, as a uint8 array."""
    off = (IMG_SIZE - CROP_SIZE) // 2
    return np.asarray(img.crop((off, off, off + CROP_SIZE, off + CROP_SIZE)))


def overlay(rgb224: np.ndarray, cam: np.ndarray, opacity: float = 0.45) -> np.ndarray:
    """Blend a [0, 1] heat map (jet colours) onto a 224x224 RGB image."""
    heat = (colormaps["jet"](cam)[..., :3] * 255).astype(np.float32)
    out = (1 - opacity) * rgb224.astype(np.float32) + opacity * heat
    return np.clip(out, 0, 255).astype(np.uint8)

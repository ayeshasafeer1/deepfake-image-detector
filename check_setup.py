"""Quick self-test: run `python check_setup.py` before starting the app."""
import time

import numpy as np
from PIL import Image
from skimage import data

from pipeline import detect_faces, prepare


def main():
    print("1. Face detection on a test photo ...")
    img = Image.fromarray(data.astronaut())
    faces = detect_faces(img)
    print(f"   faces found: {len(faces)} (expected 1)")
    model_img, info = prepare(img)
    print(f"   prepared image: {model_img.size}, info: {info}")

    print("2. Loading the model ...")
    from model_utils import WEIGHTS_PATH, Detector
    if not WEIGHTS_PATH.exists():
        print(f"   MISSING: {WEIGHTS_PATH}. Copy best.pt into the model/ folder.")
        return
    det = Detector()
    print(f"   loaded, threshold = {det.threshold:.5f}")

    print("3. Running one prediction with Grad-CAM ...")
    t0 = time.time()
    p_real, cam = det.predict(model_img)
    print(f"   P(real) = {p_real:.4f}, heat map {cam.shape}, "
          f"range {cam.min():.2f}-{cam.max():.2f}, {time.time() - t0:.2f}s")
    print("All good. Start the app with:  streamlit run app.py")


if __name__ == "__main__":
    main()

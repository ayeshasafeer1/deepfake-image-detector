"""Model loading, inference and Grad-CAM (needs torch)."""
from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

MODEL_DIR = Path(__file__).parent / "model"
WEIGHTS_PATH = MODEL_DIR / "best.pt"
SETTINGS_PATH = MODEL_DIR / "final_settings.json"
DEFAULT_THRESHOLD = 0.07897      # threshold chosen on TestVal for the final model

_to_input = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_threshold() -> float:
    try:
        return float(json.loads(SETTINGS_PATH.read_text())["threshold"])
    except Exception:
        return DEFAULT_THRESHOLD


class Detector:
    """ResNet50 with one output logit = evidence for REAL (Fake = 0, Real = 1)."""

    def __init__(self, weights_path: Path = WEIGHTS_PATH, threshold: float | None = None):
        self.threshold = load_threshold() if threshold is None else threshold
        self.device = torch.device("cpu")
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 1)
        state = torch.load(weights_path, map_location=self.device, weights_only=True)
        model.load_state_dict(state)
        self.model = model.to(self.device).eval()
        self._lock = threading.Lock()        # the model and its hooks are shared between sessions
        self._acts = None
        self._grads = None
        self.model.layer4[-1].register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        self._acts = output
        if output.requires_grad:
            output.register_hook(self._save_grad)

    def _save_grad(self, grad):
        self._grads = grad

    def predict(self, img, with_cam: bool = True):
        """img: 256x256 PIL image. Returns (P(real), heat map in [0, 1] of shape 224x224 or None)."""
        x = _to_input(img).unsqueeze(0).to(self.device)
        with self._lock:
            self.model.zero_grad(set_to_none=True)
            if not with_cam:
                with torch.no_grad():
                    logit = self.model(x).squeeze()
                return torch.sigmoid(logit).item(), None
            with torch.enable_grad():
                logit = self.model(x).squeeze()
                p_real = torch.sigmoid(logit).item()
                (-logit).backward()                       # gradient of the FAKE evidence
                weights = self._grads.mean(dim=(2, 3), keepdim=True)
                cam = torch.relu((weights * self._acts).sum(dim=1, keepdim=True)).detach()
            self._acts = self._grads = None
            self.model.zero_grad(set_to_none=True)
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return p_real, cam.astype(np.float32)

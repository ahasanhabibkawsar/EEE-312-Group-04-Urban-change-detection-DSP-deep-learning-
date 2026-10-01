"""
Several change-detection models behind one interface (v1, v2, ensembles).

Why this exists
---------------
* v2 (final model) works at LEVIR-CD's native 0.5 m/pixel with a sliding
  window. It is far more accurate on LEVIR-CD (F1 91.2 % vs 77.1 %).
* v1 (first prototype, kept in _v1_baseline/) resizes EVERY image to
  256 x 256 and was trained on LEVIR-CD down-sampled 4x (about 2 m/pixel).
  It sees coarse, blob-like structure and turned out to transfer better
  to some Bangladeshi image pairs (Purbachal), which are 256 x 256 tiles.

Methods available (keys of METHODS):

    v2       v2, RGB + Canny (4 channels)          checkpoints/best_model.pth
    v2rgb    v2, RGB only (3 channels, ablation)   checkpoints/ablation_rgb_only/best_model.pth
    v1       v1 prototype (raw RGB + Canny, 256 px) _v1_baseline/checkpoints/best_model_v1.pth
    ens_avg  average of v1 and v2 probabilities
    ens_max  pixel-wise maximum (union) of v1 and v2 probabilities

All methods return a probability map on the grid of the input images.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import config
from models import build_model_from_checkpoint, predict_probabilities
from preprocessing.dsp_processing import compute_edge_map, prepare_model_input

ROOT = Path(config.PROJECT_ROOT)

V2_CHECKPOINT = Path(config.CHECKPOINT_PATH)
V2_RGB_CHECKPOINT = ROOT / "checkpoints" / "ablation_rgb_only" / "best_model.pth"
V1_CHECKPOINT = ROOT / "_v1_baseline" / "checkpoints" / "best_model_v1.pth"
V1_SIZE = 256
V1_THRESHOLD = 0.45          # threshold used by the v1 prototype
ENSEMBLE_FILE = ROOT / "checkpoints" / "ensemble_thresholds.json"

METHODS = {
    "v2": "v2 – RGB + Canny, 4-ch (final)",
    "v2rgb": "v2 – RGB only, 3-ch (ablation)",
    "v1": "v1 – old model (256 px, raw RGB + Canny)",
    "ens_avg": "Ensemble v1 + v2 (average)",
    "ens_max": "Ensemble v1 + v2 (union / max)",
}
NEEDS = {"v2": ["v2"], "v2rgb": ["v2rgb"], "v1": ["v1"],
         "ens_avg": ["v1", "v2"], "ens_max": ["v1", "v2"]}


def _threshold_from_json(path, default):
    try:
        return float(json.loads(Path(path).read_text())["threshold"])
    except Exception:
        return default


def default_thresholds():
    """Decision threshold per method (validation-selected where available)."""

    th = {
        "v2": config.load_best_threshold(),
        "v2rgb": _threshold_from_json(V2_RGB_CHECKPOINT.parent / "best_threshold.json", 0.41),
        "v1": V1_THRESHOLD,
        "ens_avg": 0.40,
        "ens_max": 0.50,
    }
    if ENSEMBLE_FILE.exists():           # written by compare_models.py (LEVIR-CD validation)
        try:
            th.update({k: float(v) for k, v in json.loads(ENSEMBLE_FILE.read_text()).items() if k in th})
        except Exception:
            pass
    return th


# ---------------------------------------------------------------- loading

def load_v1(device, path=V1_CHECKPOINT):
    """Build the v1 prototype from _v1_baseline/ (its own model code)."""

    from _v1_baseline.models.siamese_unet import SiameseUNet as SiameseUNetV1

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model = SiameseUNetV1(in_channels=4, pretrained=False)
    model.load_state_dict(state)
    return model.to(device).eval()


class ModelZoo:
    """Loads models lazily and once; predicts with any method."""

    def __init__(self, device):
        self.device = device
        self.models = {}

    def available(self):
        ok = {"v2": V2_CHECKPOINT.exists(), "v2rgb": V2_RGB_CHECKPOINT.exists(), "v1": V1_CHECKPOINT.exists()}
        return [m for m in METHODS if all(ok[n] for n in NEEDS[m])]

    def get(self, name):
        if name not in self.models:
            if name == "v1":
                self.models[name] = load_v1(self.device)
            else:
                path = V2_CHECKPOINT if name == "v2" else V2_RGB_CHECKPOINT
                self.models[name], _ = build_model_from_checkpoint(path, self.device)
        return self.models[name]

    def load_for(self, method):
        for name in NEEDS[method]:
            self.get(name)

    # ------------------------------------------------------------ v2
    @torch.no_grad()
    def predict_v2(self, name, before, after, tta=True, scale=1.0):
        """uint8 RGB pair (H x W x 3) -> probability H x W (same grid)."""

        model = self.get(name)
        h, w = before.shape[:2]
        if scale != 1.0:
            size = (max(32, round(w * scale)), max(32, round(h * scale)))
            interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
            before = cv2.resize(before, size, interpolation=interp)
            after = cv2.resize(after, size, interpolation=interp)

        use_canny = model.in_channels == 4
        tb = torch.from_numpy(prepare_model_input(before, use_canny).transpose(2, 0, 1).copy())[None].to(self.device)
        ta = torch.from_numpy(prepare_model_input(after, use_canny).transpose(2, 0, 1).copy())[None].to(self.device)
        prob = predict_probabilities(model, tb, ta, tta=tta)[0, 0].float().cpu().numpy()

        if prob.shape != (h, w):
            prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
        return np.clip(prob, 0.0, 1.0)

    # ------------------------------------------------------------ v1
    @torch.no_grad()
    def predict_v1(self, before, after, tta=True):
        """
        Exactly as the v1 GUI: resize to 256 x 256, input = raw RGB (0-255)
        + Canny edge map (0/1), batch size 1, 3-view flip TTA.
        Probability is resized back to the input grid.
        """

        model = self.get("v1")
        h, w = before.shape[:2]

        def to_tensor(rgb):
            rgb = cv2.resize(rgb, (V1_SIZE, V1_SIZE), interpolation=cv2.INTER_LINEAR)
            x = np.concatenate([rgb.astype(np.float32), compute_edge_map(rgb)[..., None]], axis=-1)
            return torch.from_numpy(x.transpose(2, 0, 1).copy())[None].float().to(self.device)

        tb, ta = to_tensor(before), to_tensor(after)
        views = [(), (3,), (2,)] if tta else [()]
        total = 0.0
        for dims in views:
            b = torch.flip(tb, dims) if dims else tb
            a = torch.flip(ta, dims) if dims else ta
            out = model(b, a)
            out = out[0] if isinstance(out, (tuple, list)) else out
            p = torch.sigmoid(out)
            total = total + (torch.flip(p, dims) if dims else p)
        prob = (total / len(views))[0, 0].float().cpu().numpy()

        if prob.shape != (h, w):
            prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
        return np.clip(prob, 0.0, 1.0)

    # ------------------------------------------------------------ any method
    def predict(self, method, before, after, tta=True, scale=1.0, cache=None):
        """
        Probability map for `method`. `cache` (dict) lets callers reuse the
        v1 / v2 maps when evaluating several methods on the same pair.
        """

        cache = {} if cache is None else cache

        def single(name):
            key = (name, tta, scale if name != "v1" else None)
            if key not in cache:
                cache[key] = (self.predict_v1(before, after, tta) if name == "v1"
                              else self.predict_v2(name, before, after, tta, scale))
            return cache[key]

        if method in ("v2", "v2rgb", "v1"):
            return single(method)
        p1, p2 = single("v1"), single("v2")
        if method == "ens_avg":
            return 0.5 * (p1 + p2)
        if method == "ens_max":
            return np.maximum(p1, p2)
        raise ValueError(f"Unknown method {method}")

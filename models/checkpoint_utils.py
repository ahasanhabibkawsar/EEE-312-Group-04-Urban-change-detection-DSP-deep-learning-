"""
Saving / loading model checkpoints (v2 format).

A v2 checkpoint is a dict:

    {
        "model_state_dict": ...,
        "model_config": {"in_channels": 4, "use_canny": True, "version": 2},
        "epoch": int,
        "metrics": {...},            # validation metrics
        "threshold": float,          # validation-selected threshold
        "optimizer_state_dict": ..., # only in latest_model.pth (for resuming)
    }
"""

from pathlib import Path

import torch

from config import CHECKPOINT_PATH, DEFAULT_THRESHOLD

from .siamese_unet import SiameseUNet, MODEL_VERSION


class IncompatibleCheckpointError(RuntimeError):
    pass


def _is_v1_state_dict(state_dict):
    return (
        any(k.startswith("fusion.bams.0.out_conv") for k in state_dict)
        or "decoder.decoder4.conv.0.weight" in state_dict
    )


def load_checkpoint_file(path=CHECKPOINT_PATH, map_location="cpu"):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            "Train the v2 model first:  python train.py"
        )

    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    if not (isinstance(checkpoint, dict) and "model_state_dict" in checkpoint):
        checkpoint = {"model_state_dict": checkpoint}

    if _is_v1_state_dict(checkpoint["model_state_dict"]):
        raise IncompatibleCheckpointError(
            f"{path} is a v1 checkpoint (old architecture). It cannot be loaded "
            "into the fixed v2 model. Retrain with `python train.py`; the v1 "
            "model and code are preserved in _v1_baseline/."
        )

    return checkpoint


def build_model_from_checkpoint(path=CHECKPOINT_PATH, device="cpu"):
    """
    Create a SiameseUNet matching the checkpoint, load weights, eval().

    Returns
    -------
    model, checkpoint_dict
    """

    checkpoint = load_checkpoint_file(path, map_location=device)
    state_dict = checkpoint["model_state_dict"]

    config = checkpoint.get("model_config", {})
    in_channels = config.get(
        "in_channels",
        state_dict["encoder.stage0.0.weight"].shape[1],
    )

    model = SiameseUNet(in_channels=in_channels, pretrained=False)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    return model, checkpoint


def checkpoint_threshold(checkpoint, default=DEFAULT_THRESHOLD):
    return float(checkpoint.get("threshold", default) or default)


def save_model_checkpoint(path, model, epoch, metrics, threshold,
                          use_canny, optimizer=None, scheduler=None, extra=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "in_channels": model.in_channels,
            "use_canny": use_canny,
            "version": MODEL_VERSION,
        },
        "epoch": epoch,
        "metrics": metrics,
        "threshold": threshold,
    }

    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if extra:
        payload.update(extra)

    torch.save(payload, path)

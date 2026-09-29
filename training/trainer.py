"""
Training and validation loops.

Training   : random 256 x 256 native-resolution crops, multi-task loss
             (BCE + Dice on the change map, class-balanced BCE on the
             change boundaries), AdamW, warm-up + cosine LR schedule
             stepped every iteration, gradient clipping.

Validation : full 1024 x 1024 images predicted with the sliding window.
             Dataset-level (micro) metrics are accumulated with a
             ThresholdSweep, so every epoch reports both the metrics at
             the current threshold and the best validation threshold.
"""

import math
import time

import torch

from training.metrics import ThresholdSweep


def warmup_cosine_lambda(total_steps, warmup_steps, min_ratio=0.01):
    """LR multiplier: linear warm-up, then cosine decay to min_ratio."""

    def fn(step):
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return fn


class Trainer:

    def __init__(
        self,
        model,
        criterion,
        optimizer,
        device,
        scheduler=None,
        grad_clip=1.0,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.grad_clip = grad_clip

    # =====================================================
    # TRAIN ONE EPOCH
    # =====================================================

    def train_one_epoch(self, dataloader, max_batches=None, log_every=50):
        self.model.train()

        totals = {"loss": 0.0, "bce": 0.0, "dice": 0.0, "edge": 0.0}
        steps = 0
        start = time.time()
        n_batches = len(dataloader) if max_batches is None else min(len(dataloader), max_batches)

        for batch_index, batch in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break

            before = batch["before"].to(self.device, non_blocking=True)
            after = batch["after"].to(self.device, non_blocking=True)
            mask = batch["mask"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            change_logits, edge_logits = self.model(before, after, return_edges=True)
            loss, parts = self.criterion(change_logits, mask, edge_logits)

            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at batch {batch_index}: {parts}")

            loss.backward()

            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            totals["loss"] += loss.item()
            for key in ("bce", "dice", "edge"):
                totals[key] += parts[key]
            steps += 1

            if log_every and (batch_index + 1) % log_every == 0:
                rate = (time.time() - start) / steps
                print(f"    batch {batch_index + 1:4d}/{n_batches} | "
                      f"loss {totals['loss'] / steps:.4f} | "
                      f"{rate:.2f}s/batch | ETA {rate * (n_batches - steps) / 60:.1f} min",
                      flush=True)

        return {key: value / max(steps, 1) for key, value in totals.items()}

    # =====================================================
    # VALIDATION
    # =====================================================

    @torch.no_grad()
    def validate(self, dataloader, threshold=0.5, max_batches=None):
        self.model.eval()

        sweep = ThresholdSweep()
        total_loss = 0.0
        batches = 0

        for batch_index, batch in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break

            before = batch["before"].to(self.device)
            after = batch["after"].to(self.device)
            mask = batch["mask"].to(self.device)

            logits = self.model(before, after)        # sliding window for 1024 x 1024
            loss, _ = self.criterion(logits, mask)

            total_loss += loss.item()
            batches += 1

            sweep.update(torch.sigmoid(logits), mask)

        at_threshold = sweep.metrics_at(threshold)
        best = sweep.best("f1")

        return {
            "loss": total_loss / max(batches, 1),
            **{k: at_threshold[k] for k in ("accuracy", "precision", "recall", "f1", "iou", "kappa")},
            "threshold": threshold,
            "best_threshold": best["threshold"],
            "best_f1": best["f1"],
            "best_iou": best["iou"],
            "best_precision": best["precision"],
            "best_recall": best["recall"],
            "sweep": sweep,
        }

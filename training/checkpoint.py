"""
Model checkpoint utilities.
"""

import os
import torch


class CheckpointManager:

    def __init__(
        self,
        directory="checkpoints",
    ):

        self.directory = directory

        os.makedirs(
            self.directory,
            exist_ok=True,
        )

        self.best_f1 = -1.0

    def save_best(
        self,
        model,
        optimizer,
        epoch,
        metrics,
    ):
        """
        Save model if validation F1 improves.
        """

        current_f1 = metrics["f1"]

        if current_f1 <= self.best_f1:

            return False

        self.best_f1 = current_f1

        checkpoint = {
            "epoch": epoch,
            "model_state_dict":
                model.state_dict(),
            "optimizer_state_dict":
                optimizer.state_dict(),
            "metrics": metrics,
            "best_f1": self.best_f1,
        }

        path = os.path.join(
            self.directory,
            "best_model.pth",
        )

        torch.save(
            checkpoint,
            path,
        )

        return True

    def save_latest(
        self,
        model,
        optimizer,
        epoch,
        metrics,
    ):
        """
        Save the latest training state.
        """

        checkpoint = {
            "epoch": epoch,
            "model_state_dict":
                model.state_dict(),
            "optimizer_state_dict":
                optimizer.state_dict(),
            "metrics": metrics,
            "best_f1": self.best_f1,
        }

        path = os.path.join(
            self.directory,
            "latest_model.pth",
        )

        torch.save(
            checkpoint,
            path,
        )
"""
Training and validation loops (simplified, without edge loss).
"""

import torch


class Trainer:

    def __init__(
        self,
        model,
        criterion,
        optimizer,
        device,
        use_bcl=False,
        edge_weight=0.0,
        bcl_margin=2.0,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.use_bcl = use_bcl
        self.edge_weight = edge_weight

        if use_bcl:
            from losses import BatchBalancedContrastiveLoss
            self.bcl_loss_fn = BatchBalancedContrastiveLoss(margin=bcl_margin)

    # =====================================================
    # TRAIN ONE EPOCH
    # =====================================================

    def train_one_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        for batch in dataloader:
            before = batch["before"].to(self.device)
            after = batch["after"].to(self.device)
            mask = batch["mask"].to(self.device)

            self.optimizer.zero_grad()

            if self.use_bcl:
                distance_map = self.model(before, after, return_distance=True)
                loss = self.bcl_loss_fn(distance_map, mask)
            else:
                logits = self.model(before, after, return_edges=False)
                loss = self.criterion(logits, mask)

            loss.backward()
            self.optimizer.step()

            batch_size = before.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        return total_loss / total_samples

    # =====================================================
    # VALIDATION
    # =====================================================

    def validate(self, dataloader, metrics_function):
        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        total_tp = 0
        total_tn = 0
        total_fp = 0
        total_fn = 0

        with torch.no_grad():
            for batch in dataloader:
                before = batch["before"].to(self.device)
                after = batch["after"].to(self.device)
                mask = batch["mask"].to(self.device)

                if self.use_bcl:
                    distance_map = self.model(before, after, return_distance=True)
                    loss = self.bcl_loss_fn(distance_map, mask)
                    # For metrics, we need logits. We'll compute logits separately.
                    logits = self.model(before, after, return_edges=False)
                else:
                    logits = self.model(before, after, return_edges=False)
                    loss = self.criterion(logits, mask)

                metrics = metrics_function(logits, mask)

                batch_size = before.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                total_tp += metrics["tp"]
                total_tn += metrics["tn"]
                total_fp += metrics["fp"]
                total_fn += metrics["fn"]

        avg_loss = total_loss / total_samples

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        iou = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0.0
        accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn) if (total_tp + total_tn + total_fp + total_fn) > 0 else 0.0

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou,
        }
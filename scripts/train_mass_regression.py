"""Train the mass regression head.

Usage::

    python scripts/train_mass_regression.py --config configs/models/mass_regression_training.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from edge_ai_mass.data.dataset import MassDataset
from edge_ai_mass.evaluation.metrics import mape, r_squared, rmse
from edge_ai_mass.modules.mass.regression_estimator import MassRegressionHead
from edge_ai_mass.utils.config import load_config
from edge_ai_mass.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)

    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data
    train_ds = MassDataset(data_cfg["train_csv"], input_dim=model_cfg["input_dim"])
    val_ds = MassDataset(data_cfg["val_csv"], input_dim=model_cfg["input_dim"])
    train_dl = DataLoader(train_ds, batch_size=train_cfg["batch_size"], shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=train_cfg["batch_size"])

    # Model
    model = MassRegressionHead(
        input_dim=model_cfg["input_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        num_classes=model_cfg["num_classes"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    loss_fn = nn.HuberLoss(delta=train_cfg.get("huber_delta", 0.5))

    best_val_rmse = float("inf")
    ckpt_dir = Path(cfg["evaluation"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(train_cfg["epochs"]):
        # Train
        model.train()
        train_losses = []
        for batch in train_dl:
            feats = batch["features"].to(device)
            cls_ids = batch["class_id"].to(device)
            targets = batch["mass_kg"].to(device)

            preds = model(feats, cls_ids)
            loss = loss_fn(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validate
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in val_dl:
                feats = batch["features"].to(device)
                cls_ids = batch["class_id"].to(device)
                preds = model(feats, cls_ids)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(batch["mass_kg"].numpy())

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_targets)
        val_rmse = rmse(y_true, y_pred)
        val_mape = mape(y_true, y_pred)
        val_r2 = r_squared(y_true, y_pred)

        logger.info(
            "Epoch %3d | train_loss=%.4f | val RMSE=%.4f MAPE=%.2f%% R²=%.4f",
            epoch + 1,
            np.mean(train_losses),
            val_rmse,
            val_mape,
            val_r2,
        )

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(model.state_dict(), ckpt_dir / "mass_regression.pt")
            logger.info("  → Saved best checkpoint (RMSE=%.4f)", val_rmse)

    logger.info("Training complete. Best val RMSE: %.4f", best_val_rmse)


if __name__ == "__main__":
    main()

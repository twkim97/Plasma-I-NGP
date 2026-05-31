from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from src.model import build_model
from src.dataset import create_condition_dataloaders

import os
import random
import numpy as np
import torch
import yaml
from typing import Dict, Any

def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def load_config(config_path: str | Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def train(config):
    seed = config["experiment"].get("seed", 42)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(config["model"])
    model = model.to(device)

    epochs = config["training"]["epochs"]
    lr = config["training"]["learning_rate"]
    batch_size = config["training"]["batch_size"]

    train_loader, test_loader = create_condition_dataloaders(
        input_path=config["data"]["input_path"],
        output_path=config["data"]["output_path"],
        batch_size=batch_size,
        num_workers=config["training"].get("num_workers", 4),
        pin_memory=True,
    )

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_loss = float("inf")
    save_every = config["checkpoint"].get("save_every", 10)
    save_dir = Path(config["checkpoint"].get("save_dir", "checkpoints"))
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_count = 0

        for step, (xt_batch, y_batch) in enumerate(train_loader, start=1):
            xt_batch = xt_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                y_pred = model(xt_batch)
                loss = criterion(y_pred, y_batch)

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=config["training"].get("grad_clip", 1.0))

            scaler.step(optimizer)
            scaler.update()

            bs = xt_batch.size(0)
            total_loss += loss.item() * bs
            total_count += bs

            if step % 200 == 0:
                avg_step_loss = total_loss / max(total_count, 1)
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Step {step} | "
                    f"Loss(avg): {avg_step_loss:.6f} | "
                    f"LR: {current_lr:.2e}"
                )

        avg_loss = total_loss / max(total_count, 1)
        print(f"[Epoch {epoch+1}/{epochs}] Train Loss: {avg_loss:.6f}")

        scheduler.step()

        if (epoch + 1) % save_every == 0:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model": model.state_dict(),
                    "opt": optimizer.state_dict(),
                    "loss": avg_loss,
                    "config": config,
                },
                save_dir / f"epoch_{epoch+1:04d}.pt",
            )

    torch.save(
        {
            "epoch": epochs,
            "model": model.state_dict(),
            "opt": optimizer.state_dict(),
            "final_loss": avg_loss,
            "config": config,
        },
        save_dir / "final.pt",
    )

def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment config yaml file.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    train(config)
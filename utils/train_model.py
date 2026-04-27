"""
utils.py — shared experiment harness for teamMav CV project
Owns: logging setup, the training loop, weight saving, seeding, device.
Does NOT own: model, optimizer, loss, data. Those are the notebook's job.
"""

import os
import random
import logging
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


# ── SEEDING ──────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── DEVICE ───────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] using {dev}")
    return dev


# ── LOGGING SETUP ─────────────────────────────────────────────────────────────
# log_to: "wandb" | "file" | "both"

def setup_logging(config: dict, log_to: str = "both") -> logging.Logger:
    exp_id  = config.get("exp_id", "EXP_UNKNOWN")
    log_dir = Path(config.get("log_dir", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(exp_id)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%H:%M:%S")
    fh  = logging.FileHandler(log_dir / f"{exp_id}.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    

    if log_to in ("wandb", "both"):
        logger.info("setting up wandb")
        try:
            import wandb
            # wandb.init(
            #     project=config.get("sek188", ""),
            #     name=exp_id,
            #     config=config,
            #     entity=config.get("wandb_entity", None),
            # )
            
            # run = wandb.init(
            #     # entity="sek188",
            #     project="cs2770_teamMav",
            #     config=config,
            # )
            
            run = wandb.init(
                entity="teamMaverick",
                project="cs2770_teamMav",
                config=config,
            )
            
        except ImportError:
            logger.warning("wandb not installed, falling back to file-only logging")

    return logger


def log_metrics(metrics: dict, step: int, logger: logging.Logger, log_to: str = "both"):
    """Log a dict of metrics to file and/or wandb."""
    msg = " | ".join(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                     for k, v in metrics.items())
    logger.info(f"step={step} | {msg}")

    if log_to in ("wandb", "both"):
        try:
            import wandb
            if wandb.run is not None:
                wandb.log(metrics, step=step)
        except ImportError:
            pass


# ── CHECKPOINTING ─────────────────────────────────────────────────────────────

def save_weights(model: torch.nn.Module, config: dict, tag: str = "final"):
    """Save model weights. tag is usually 'final' or a note like 'best_val'."""
    exp_id   = config.get("exp_id", "EXP_UNKNOWN")
    save_dir = Path(config.get("weights_dir", f"weights/{exp_id}"))
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"{exp_id}_{tag}.pt"
    torch.save(model.state_dict(), path)
    return path


def load_weights(model: torch.nn.Module, path: str | Path) -> torch.nn.Module:
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model


# ── TRAINING HARNESS ──────────────────────────────────────────────────────────

def run_experiment(
    model:        torch.nn.Module,
    optimizer:    torch.optim.Optimizer,
    loss_fn,                              # callable: (batch, model, mode, **kwargs) -> tensor
    train_loader: torch.utils.data.DataLoader,
    val_loader:   torch.utils.data.DataLoader,
    config:       dict,
    log_to:       str = "both",
    loss_kwargs:  dict = None,            # passed to loss_fn every call
):
    """
    Minimal, data-agnostic training harness.

    Contract for loss_fn:
        loss = loss_fn(batch, model, mode="train"|"val", **loss_kwargs)
        - must return a scalar tensor during "train" (backward is called on it)
        - during "val" it can return a tensor or a plain float/dict —
          if it returns a dict, all keys are logged as metrics

    The harness never unpacks batch. That's loss_fn's job entirely.
    SLURM note: to submit as a job, wrap this call in a submit.py that
    builds model/optimizer/loss identically to the notebook and calls here.
    """
    loss_kwargs = loss_kwargs or {}
    epochs      = config.get("epochs", 10)
    exp_id      = config.get("exp_id", "EXP_UNKNOWN")
    device      = get_device()

    logger = setup_logging(config, log_to)
    model.to(device)

    logger.info(f"Starting experiment {exp_id} | epochs={epochs} | device={device}")
    start = time.time()

    for epoch in range(1, epochs + 1):

        # ── TRAIN ────────────────────────────────────────────────────────────
        model.train()
        train_losses = []
        pbar = tqdm(train_loader, desc=f"[{exp_id}] epoch {epoch}/{epochs}",
                    leave=False, dynamic_ncols=True)

        for batch in pbar:
            # move tensors to device if batch is a list/tuple of tensors
            if isinstance(batch, (list, tuple)):
                batch = [b.to(device) if isinstance(b, torch.Tensor) else b
                         for b in batch]

            optimizer.zero_grad()
            loss = loss_fn(batch, model, epoch=epoch, total_epochs=epochs, mode="train", **loss_kwargs)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train = sum(train_losses) / len(train_losses)

        # ── VAL ──────────────────────────────────────────────────────────────
        model.eval()
        val_outputs = []
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (list, tuple)):
                    batch = [b.to(device) if isinstance(b, torch.Tensor) else b
                             for b in batch]
                out = loss_fn(batch, model, epoch=epoch, total_epochs=epochs, mode="val", **loss_kwargs)
                val_outputs.append(out)

        # val metric assembly — loss_fn can return scalar, tensor, or dict
        if val_outputs and isinstance(val_outputs[0], dict):
            val_metrics = {
                k: sum(d[k] for d in val_outputs) / len(val_outputs)
                for k in val_outputs[0]
            }
        else:
            avg_val = sum(
                v.item() if isinstance(v, torch.Tensor) else v
                for v in val_outputs
            ) / len(val_outputs)
            val_metrics = {"val_loss": avg_val}

        metrics = {"epoch": epoch, "train_loss": avg_train, **val_metrics}
        log_metrics(metrics, step=epoch, logger=logger, log_to=log_to)

    # ── SAVE ─────────────────────────────────────────────────────────────────
    save_path = save_weights(model, config, tag="final")
    elapsed   = time.time() - start
    logger.info(f"Done. weights saved to {save_path} | total time: {elapsed:.1f}s")

    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except ImportError:
        pass

    return model
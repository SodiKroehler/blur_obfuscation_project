"""
new_experiment.py — scaffold a new experiment in one line
Usage:
    python new_experiment.py --epic 3 --id 01 --desc "curriculum learning" --owner sodi
Creates:
    Epic_3/EX_03_01/        ← folder structure + README + notebook
    configs/EX_03_01.yaml   ← experiment config with variant stubs
"""
import argparse
import json
from datetime import date
from pathlib import Path


def make_exp_id(epic: int, exp: str) -> str:
    return f"EX_{epic:02d}_{exp.zfill(2)}"


def scaffold(epic: int, exp: str, desc: str, owner: str):
    exp_id = make_exp_id(epic, exp)
    root = Path(f"Epic_{epic}") / exp_id

    # ── folder structure ─────────────────────────────────────────────────────
    for d in ["Raw", "Model", "Results/Graphs", "logs"]:
        p = root / d
        p.mkdir(parents=True, exist_ok=True)
        (p / ".gitkeep").touch()

    # ── config ───────────────────────────────────────────────────────────────
    cfg_dir = Path("configs")
    cfg_dir.mkdir(exist_ok=True)
    cfg_path = cfg_dir / f"{exp_id}.yaml"
    cfg_path.write_text(
        f"exp_id: {exp_id}\n"
        f"epic: {epic}\n"
        f"description: \"{desc}\"\n"
        f"date: {date.today()}\n"
        f"owner: {owner}\n"
        f"seed: 42\n"
        f"img_size: 28\n"
        f"raw_dir: Raw\n"
        f"grad_dir: ../../Epic_1/EX_01_01/Raw/gradients\n"
        f"csv_path: ../../Epic_1/EX_01_01/Raw/pairings.csv\n"
        f"batch_size: 512\n"
        f"epochs: 20\n"
        f"# override any defaults.yaml values below this line\n"
        f"variants:\n"
        f"  a:\n"
        f"    description: \"\"\n"
        f"    alpha: -0.55\n"
        f"    beta: 0.45\n"
        f"    dropout_rate: 0.5\n"
        f"    sparse_lambda: 0.2\n"
        f"    blur_kernel_size: 3\n"
        f"    epochs: 10\n"
        f"  b:\n"
        f"    description: \"\"\n"
        f"    alpha: -0.55\n"
        f"    beta: 0.45\n"
        f"    dropout_rate: 0.5\n"
        f"    sparse_lambda: 0.2\n"
        f"    blur_kernel_size: 3\n"
        f"    epochs: 10\n"
        f"  c:\n"
        f"    description: \"\"\n"
        f"    alpha: -0.55\n"
        f"    beta: -0.45\n"
        f"    dropout_rate: 0.5\n"
        f"    sparse_lambda: 0.2\n"
        f"    blur_kernel_size: 3\n"
        f"    epochs: 10\n"
    )

    # ── notebook ─────────────────────────────────────────────────────────────
    cell_imports = (
        "import yaml, torch, sys, json, random, os, re\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import torch.nn as nn\n"
        "import torch.nn.functional as F\n"
        "from pathlib import Path\n"
        "from types import SimpleNamespace\n"
        "from PIL import Image\n"
        "from sklearn.metrics import classification_report, confusion_matrix\n"
        "from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split\n"
        "from torchvision import datasets, transforms\n"
        "import torchvision.transforms.functional as TF\n"
        "sys.path.insert(0, '../../')\n"
        "from utils.train_model import run_experiment, set_seed, get_device"
    )

    cell_config = (
        "def to_namespace(d):\n"
        "    if isinstance(d, dict):\n"
        "        return SimpleNamespace(**{k: to_namespace(v) for k, v in d.items()})\n"
        "    return d\n\n"
        "with open('../../configs/defaults.yaml') as f:\n"
        "    config = yaml.safe_load(f)\n\n"
        f"with open('../../configs/{exp_id}.yaml') as f:\n"
        "    config.update(yaml.safe_load(f))\n\n"
        "config = to_namespace(config)\n"
        "set_seed(config.seed)\n"
        "device = get_device()\n"
        "RAW_DIR  = Path(config.raw_dir)\n"
        "GRAD_DIR = Path(config.grad_dir)\n"
        f"data_dir = Path('./{exp_id}/Raw/')"
    )

    def make_cell(source: str) -> dict:
        return {
            "cell_type": "code",
            "metadata": {},
            "source": source,
            "outputs": [],
            "execution_count": None
        }

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "teamMavConda",
                "language": "python",
                "name": "teamMavConda"
            },
            "language_info": {"name": "python"}
        },
        "cells": [
            make_cell(cell_imports),
            make_cell(cell_config),
        ]
    }

    with open(root / f"{exp_id}.ipynb", "w") as f:
        json.dump(nb, f, indent=1)

    # ── README ────────────────────────────────────────────────────────────────
    (root / "README.md").write_text(
        f"# {exp_id} — {desc}\n\n"
        f"**Epic:** {epic} | **Date:** {date.today()} | **Owner:** {owner}\n\n"
        "## Hypothesis\n\n## Results\n\n## Notes\n"
    )

    print(f"✓ {root}")
    print(f"✓ configs/{exp_id}.yaml")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epic",  type=int, required=True)
    p.add_argument("--id",    type=str, required=True)
    p.add_argument("--desc",  type=str, default="")
    p.add_argument("--owner", type=str, default="")
    args = p.parse_args()
    scaffold(args.epic, args.id, args.desc, args.owner)
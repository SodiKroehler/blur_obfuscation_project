"""
new_experiment.py — scaffold a new experiment in one line

Usage:
    python new_experiment.py --epic 1 --id 001 --desc "box blur baseline soft mask" --owner alex

Creates:
    Epic_1/EX_001/        ← folder structure + README + notebook
    configs/EX_001.yaml   ← identity-only, overrides defaults.yaml
"""

import argparse
import json
from datetime import date
from pathlib import Path


def scaffold(epic: int, exp_id: str, desc: str, owner: str):
    root = Path(f"Epic_{epic}") / exp_id

    # folder structure
    for d in ["Raw", "Model", "Results/Graphs", "logs"]:
        p = root / d
        p.mkdir(parents=True, exist_ok=True)
        (p / ".gitkeep").touch()

    # configs/EX_001.yaml — identity only, rest inherited from defaults.yaml
    cfg_dir = Path("configs")
    cfg_dir.mkdir(exist_ok=True)
    cfg_path = cfg_dir / f"{exp_id}.yaml"
    cfg_path.write_text(
        f"exp_id: {exp_id}\n"
        f"epic: {epic}\n"
        f"description: \"{desc}\"\n"
        f"date: {date.today()}\n"
        f"owner: {owner}\n"
        f"# override any defaults.yaml values below this line\n"
    )

    # notebook — loads defaults then overrides with experiment config
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
            {
                "cell_type": "code",
                "metadata": {},
                "source": (
                    "import yaml, torch, sys\n"
                    "sys.path.insert(0, '../../')\n"
                    "from utils.train_model import set_seed, get_device\n\n"
                    "with open('../../configs/defaults.yaml') as f:\n"
                    "    config = yaml.safe_load(f)\n\n"
                    f"with open('../../configs/{exp_id}.yaml') as f:\n"
                    "    config.update(yaml.safe_load(f))\n\n"
                    "set_seed(config['seed'])\n"
                    "device = get_device()"
                ),
                "outputs": [],
                "execution_count": None
            }
        ]
    }
    with open(root / f"{exp_id}.ipynb", "w") as f:
        json.dump(nb, f, indent=1)

    # README
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
    scaffold(args.epic, f"EX_{args.id}", args.desc, args.owner)
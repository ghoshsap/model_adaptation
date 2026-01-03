"""Plot training/eval loss and perplexity from log_history_full.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-darkgrid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training/eval metrics")
    parser.add_argument("log_path", help="Path to log_history_full.json")
    parser.add_argument("--output", default=None, help="Optional path to save the plot")
    return parser.parse_args()


def load_logs(path: Path):
    records = []
    with path.open() as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def main():
    args = parse_args()
    path = Path(args.log_path)
    records = load_logs(path)
    if not records:
        print("No records found.")
        return

    steps = [r.get("step", r.get("global_step")) for r in records]
    train_loss = [r.get("loss") for r in records if "loss" in r]
    train_steps = [r.get("step", r.get("global_step")) for r in records if "loss" in r]
    eval_loss = [r.get("eval_loss") for r in records if "eval_loss" in r]
    eval_steps = [r.get("step", r.get("global_step")) for r in records if "eval_loss" in r]
    eval_ppl = [r.get("eval_perplexity") for r in records if r.get("eval_perplexity")]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(train_steps, train_loss, marker="o", markersize=3, label="Train Loss")
    axes[0].plot(eval_steps, eval_loss, marker="s", markersize=4, linewidth=2, label="Eval Loss")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Eval Loss Over Steps")
    axes[0].legend()

    axes[1].plot(eval_steps, eval_ppl, marker="o", color="darkgreen", markersize=4, linewidth=2)
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Perplexity")
    axes[1].set_title("Eval Perplexity Over Training Steps")

    plt.tight_layout()
    if args.output:
        plt.savefig(args.output)
    else:
        plt.show()


if __name__ == "__main__":
    main()

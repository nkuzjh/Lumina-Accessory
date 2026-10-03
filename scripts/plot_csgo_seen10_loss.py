"""Plot logged Seen-10 train and full validation losses without starting work."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path)
    p.add_argument("--events", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args(argv)
    if args.events is None:
        if args.run_root is None:
            p.error("Specify --events or --run-root")
        args.events = args.run_root / "train/events.jsonl"
    if args.output is None:
        args.output = args.events.with_name("loss.png")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [json.loads(line) for line in args.events.read_text().splitlines() if line.strip()]
    train = [(r["step"], r["loss"]) for r in rows if r.get("event") == "train"]
    validation = [(r["step"], r["loss"]) for r in rows if r.get("event") == "validation"]
    if not train:
        raise ValueError("No training loss records")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(*zip(*train), label="Training flow loss", alpha=0.8)
    if validation:
        ax.plot(*zip(*validation), marker="o", label="Validation flow loss (count in events.jsonl)")
    ax.set(xlabel="Optimizer update", ylabel="Mean generation loss")
    if any(r.get("structural_cpu_probe") for r in rows):
        ax.set_title("CPU structural probe (batch 1; excluded from formal budget)")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()

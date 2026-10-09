"""Turn runs/*/log.jsonl into a results table (markdown) and loss-curve plots.

    python scripts/summarize.py runs/ [--plot docs/img/curves.png] [--induction docs/img/ind.png]

A row per run: params, best val loss, best val bits-per-byte, the step it happened at,
the final train/val gap (overfitting), and wall time. Runs whose names differ only by a
"-s<seed>" suffix are also aggregated into mean ± spread.
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def load(run_dir):
    recs = [json.loads(l) for l in (run_dir / "log.jsonl").read_text().splitlines() if l]
    cfg = json.loads((run_dir / "config.json").read_text())
    best = min(recs, key=lambda r: r["val_loss"])
    last = recs[-1]
    times = [r["time"] for r in recs if r.get("time")]
    return {"name": run_dir.name, "data": cfg["data"], "recs": recs, "best": best,
            "gap": last["val_loss"] - last["train_loss"],
            "minutes": (times[-1] - times[0]) / 60 if len(times) > 1 else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--plot")
    ap.add_argument("--induction", help="also plot the induction-probe curves to this path")
    args = ap.parse_args()
    dirs = []
    for r in args.runs:
        p = Path(r)
        dirs += [d for d in sorted(p.iterdir()) if (d / "log.jsonl").exists()] \
            if not (p / "log.jsonl").exists() else [p]
    runs = [load(d) for d in dirs]
    runs.sort(key=lambda r: r["best"]["val_bpb"])

    print("| run | data | best val loss | best val bpb | at step | final val-train gap | min |")
    print("|---|---|---|---|---|---|---|")
    for r in runs:
        b = r["best"]
        print(f"| {r['name']} | {Path(r['data']).name} | {b['val_loss']:.4f} | "
              f"**{b['val_bpb']:.4f}** | {b['step']} | {r['gap']:+.3f} | {r['minutes']:.0f} |")

    groups = defaultdict(list)
    for r in runs:
        groups[re.sub(r"-s\d+$", "", r["name"])].append(r["best"]["val_bpb"])
    multi = {k: v for k, v in groups.items() if len(v) > 1}
    if multi:
        print("\n| config (over seeds) | mean val bpb | min | max | n |")
        print("|---|---|---|---|---|")
        for k, v in sorted(multi.items(), key=lambda kv: sum(kv[1]) / len(kv[1])):
            print(f"| {k} | **{sum(v) / len(v):.4f}** | {min(v):.4f} | {max(v):.4f} | {len(v)} |")

    if args.plot:
        plot(runs, args.plot)
    if args.induction:
        plot_induction(runs, args.induction)


def plot_induction(runs, path):
    """Two panels from the in-training probe (train.induction_probe): the best head's
    attention on the induction target, and how much easier the repeated half is (nats)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    runs = [r for r in runs if "ind_score" in r["recs"][-1]]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for r in runs:
        steps = [x["step"] for x in r["recs"]]
        axes[0].plot(steps, [x["ind_score"] for x in r["recs"]], lw=1.6, label=r["name"])
        axes[1].plot(steps, [x["copy_gain"] for x in r["recs"]], lw=1.6, label=r["name"])
    axes[0].set_title("induction score: best head's attention on token i-P+1")
    axes[1].set_title("copy gain: loss(first copy) - loss(second copy), nats")
    axes[1].axhline(0, color="grey", lw=0.8)
    for ax in axes:
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"induction plot -> {path}")


def plot(runs, path):
    """Two panels: the whole run, and a zoom on the second half where configs separate.
    Seeds of one config share a colour; the line is their mean, the band their min..max."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    groups = defaultdict(list)
    for r in runs:
        groups[re.sub(r"-s\d+$", "", r["name"])].append(r)
    order = sorted(groups, key=lambda k: np.mean([r["best"]["val_bpb"] for r in groups[k]]))
    colours = list(plt.get_cmap("tab10").colors) + list(plt.get_cmap("Dark2").colors)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for i, k in enumerate(order):
        steps = [x["step"] for x in groups[k][0]["recs"]]
        ys = np.array([[x["val_bpb"] for x in r["recs"]][:len(steps)] for r in groups[k]])
        mean = ys.mean(0)
        n = len(groups[k])
        label = f"{k} ({mean[-1]:.3f})" + (f" ×{n} seeds" if n > 1 else "")
        for ax in axes:
            ax.plot(steps, mean, color=colours[i % len(colours)], label=label, lw=1.6)
            if n > 1:
                ax.fill_between(steps, ys.min(0), ys.max(0), color=colours[i % len(colours)],
                                alpha=0.25, lw=0)
    last = max(x["step"] for r in runs for x in r["recs"])
    tail = [x["val_bpb"] for r in runs for x in r["recs"] if x["step"] >= last / 2]
    axes[0].set_ylim(top=min(4.5, max(tail) + 1.5))
    axes[1].set_xlim(last / 2, last * 1.01)
    lo, hi = min(tail), sorted(tail)[int(0.9 * len(tail)) - 1]
    axes[1].set_ylim(lo - 0.02, hi + 0.04)
    axes[0].set_title("validation bits per byte")
    axes[1].set_title("zoom: second half of training")
    for ax in axes:
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("val bpb (lower is better)")
    axes[0].legend(fontsize=8)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"\nplot -> {path}")


if __name__ == "__main__":
    main()

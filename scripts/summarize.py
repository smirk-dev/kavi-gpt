"""Turn runs/*/log.jsonl into a results table (markdown) and loss-curve plots.

    python scripts/summarize.py runs/ [--plot docs/img/curves.png] [--group seed]

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
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 5))
        for r in runs:
            steps = [x["step"] for x in r["recs"]]
            ax.plot(steps, [x["val_bpb"] for x in r["recs"]], label=r["name"])
        ax.set_xlabel("step")
        ax.set_ylabel("validation bits per byte")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot, dpi=130, bbox_inches="tight")
        print(f"\nplot -> {args.plot}")


if __name__ == "__main__":
    main()

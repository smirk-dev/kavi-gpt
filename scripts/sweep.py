"""Run a list of training runs described by a sweep file, one after another.

    python scripts/sweep.py configs/sweep_ablation.json --backend cupy [--only name1 name2]

Sweep file: {"base": "configs/x.json", "set": [...common overrides...],
             "runs": [{"name": "...", "set": ["model.pos=\"learned\"", ...]}, ...]}
Every run is the base config plus *its own* overrides, so each ablation is a single-knob
diff that can be read straight off the sweep file. Finished runs (samples.txt exists) are
skipped, so a sweep can be resumed after a timeout.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep")
    ap.add_argument("--backend", default="numpy")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    sweep = json.loads(Path(args.sweep).read_text(encoding="utf-8"))
    for run in sweep["runs"]:
        name = run["name"]
        if args.only and name not in args.only:
            continue
        if (Path(args.out) / name / "samples.txt").exists():
            print(f"== {name}: already done, skipping")
            continue
        cmd = [sys.executable, str(ROOT / "scripts" / "train.py"), sweep["base"],
               "--backend", args.backend, "--out", args.out,
               "--set", f'name="{name}"', *sweep.get("set", []), *run.get("set", [])]
        print(f"\n== {name}: {' '.join(run.get('set', [])) or '(base)'}", flush=True)
        t0 = time.time()
        subprocess.run(cmd, check=True)
        print(f"== {name}: done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()

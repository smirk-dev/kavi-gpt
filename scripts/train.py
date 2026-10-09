"""Train a Kavi model.

    python scripts/train.py configs/char_kavi.json
    python scripts/train.py configs/char_kavi.json --set train.max_steps=200 model.n_layer=2
    python scripts/train.py configs/char_kavi.json --backend cupy        # GPU (Kaggle)

Outputs runs/<name>/: config.json, log.jsonl (one line per eval), best.npz (lowest val
loss), last.npz (for resuming with --resume), samples.txt.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kavi import backend as B  # noqa: E402
from kavi.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from kavi.data import TokenData  # noqa: E402
from kavi.model import GPT, GPTConfig  # noqa: E402
from kavi.optim import SGD, AdamW, clip_grad_norm, lr_at  # noqa: E402
from kavi.probes import induction, induction_batch  # noqa: E402

DEFAULT_TRAIN = dict(
    optimizer="adamw", batch_size=32, max_steps=5000, lr=1e-3, min_lr=1e-4, warmup=200,
    weight_decay=0.1, beta1=0.9, beta2=0.95, momentum=0.9, grad_clip=1.0,
    eval_interval=250, eval_iters=20, seed=1337, sample_tokens=300, sample_prompt="\n",
    log_interval=25, induction_probe=False,
)


def parse_value(v):
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return v


def load_config(path, overrides):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    cfg.setdefault("name", Path(path).stem)
    cfg["train"] = {**DEFAULT_TRAIN, **cfg.get("train", {})}
    for item in overrides or []:
        key, value = item.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = parse_value(value)
    return cfg


def evaluate(model, data, tc, block_size):
    """Mean loss over a *fixed* set of batches per split, so evals are comparable."""
    model.eval()
    out = {}
    for split in ("train", "val"):
        rng = np.random.default_rng(1234)
        losses = []
        for _ in range(tc["eval_iters"]):
            x, y = data.batch(split, tc["batch_size"], block_size, rng)
            losses.append(float(model.forward(x, y)[1]))
        out[split] = float(np.mean(losses))
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--set", nargs="*", default=[], help="overrides like train.lr=3e-4")
    ap.add_argument("--backend", default="numpy", choices=["numpy", "cupy"])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    B.set_backend(args.backend, precision="float32")
    tc = cfg["train"]
    B.seed(tc["seed"])

    data = TokenData(ROOT / cfg["data"])
    tok = data.tokenizer()
    mcfg = GPTConfig.preset(cfg["model"].pop("preset", "kavi"),
                            **{**cfg["model"], "vocab_size": data.vocab_size})
    cfg["model"] = mcfg.to_dict()
    model = GPT(mcfg, seed=tc["seed"])
    params = model.params()
    if tc["optimizer"] not in ("adamw", "sgd", "momentum"):
        raise ValueError(f"train.optimizer={tc['optimizer']!r}; expected adamw | sgd | momentum")
    if tc["optimizer"] == "adamw":
        opt = AdamW(params, lr=tc["lr"], betas=(tc["beta1"], tc["beta2"]),
                    weight_decay=tc["weight_decay"])
    else:
        if tc["optimizer"] == "sgd":
            tc["momentum"] = 0.0     # plain SGD: record what actually runs (config.json is written below)
        opt = SGD(params, lr=tc["lr"], momentum=tc["momentum"])

    run = Path(args.out) / cfg["name"]
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    start, best = 0, float("inf")
    if args.resume and (run / "last.npz").exists():
        start, best = load_checkpoint(run / "last.npz", model, opt)
        print(f"resumed at step {start} (best val {best:.4f})")

    n = model.num_params()
    print(f"[{cfg['name']}] {n / 1e6:.2f}M params | backend {B.name} | "
          f"data {cfg['data']} (vocab {data.vocab_size}, {len(data.train):,} train tokens, "
          f"{len(data.train) / n:.2f} tokens/param)")
    print(f"model: {mcfg}")

    rng = np.random.default_rng(tc["seed"] + start)
    T, bs = mcfg.block_size, tc["batch_size"]
    # induction_probe: at every eval, log how strongly the model copies a repeated sequence,
    # to catch the induction-head "phase change" as it happens (docs/14-scaling.md). The same
    # fixed batch every time, so the numbers form a curve.
    probe_ids = (induction_batch(data.val, min(50, T // 2), 8, np.random.default_rng(4321))
                 if tc["induction_probe"] else None)
    # fresh run: start a new log. Resume: keep only records before `start` (the resumed
    # step is evaluated again, and a crashed run may have logged past its last checkpoint)
    log_path = run / "log.jsonl"
    kept = []
    if start and log_path.exists():
        kept = [l for l in log_path.read_text(encoding="utf-8").splitlines()
                if l and json.loads(l)["step"] < start]
    log_path.write_text("".join(l + "\n" for l in kept), encoding="utf-8")
    log = open(log_path, "a", encoding="utf-8")
    t_last, tokens_since, lr = time.time(), 0, None
    for step in range(start, tc["max_steps"] + 1):
        if step % tc["eval_interval"] == 0 or step == tc["max_steps"]:
            ev = evaluate(model, data, tc, T)
            rec = {"step": step, "train_loss": ev["train"], "val_loss": ev["val"],
                   "val_bpb": data.bpb(ev["val"]), "lr": lr, "time": time.time()}
            if probe_ids is not None:
                pr = induction(model, probe_ids)
                rec.update(ind_score=pr["best_ind"], ind_head=list(pr["best_head"]),
                           copy_gain=pr["loss_first"] - pr["loss_second"])
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"  eval step {step:5d} | train {ev['train']:.4f} | val {ev['val']:.4f} "
                  f"| val bpb {data.bpb(ev['val']):.4f}"
                  + (f" | induction {rec['ind_score']:.3f} L{rec['ind_head'][0]}H{rec['ind_head'][1]}"
                     f" copy gain {rec['copy_gain']:.2f}" if probe_ids is not None else ""))
            if ev["val"] < best:
                best = ev["val"]
                save_checkpoint(run / "best.npz", model, opt, step, best, cfg)
            save_checkpoint(run / "last.npz", model, opt, step, best, cfg)
            if step == tc["max_steps"]:
                break
        lr = lr_at(step, tc["lr"], tc["min_lr"], tc["warmup"], tc["max_steps"])
        opt.lr = lr
        x, y = data.batch("train", bs, T, rng)
        model.zero_grad()
        _, loss = model.forward(x, y)
        model.backward()
        gnorm = clip_grad_norm(params, tc["grad_clip"])
        opt.step()
        tokens_since += bs * T
        if step % tc["log_interval"] == 0:
            dt = time.time() - t_last
            print(f"step {step:5d} | loss {float(loss):.4f} | lr {lr:.2e} | gnorm {gnorm:.2f} "
                  f"| {tokens_since / max(dt, 1e-9):,.0f} tok/s")
            t_last, tokens_since = time.time(), 0
    log.close()

    # final sample from the best checkpoint
    load_checkpoint(run / "best.npz", model)
    prompt = np.array([tok.encode(tc["sample_prompt"])])
    out = model.generate(prompt, tc["sample_tokens"], temperature=0.8, top_k=50,
                         rng=np.random.default_rng(0))
    text = tok.decode(B.to_numpy(out[0]).tolist())
    (run / "samples.txt").write_text(text, encoding="utf-8")
    print(f"\nbest val loss {best:.4f} (bpb {data.bpb(best):.4f}). sample:\n{text}")


if __name__ == "__main__":
    main()

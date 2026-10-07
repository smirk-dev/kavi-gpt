"""Checkpoints as plain .npz files: parameters, optimiser state, step, config."""
import json

import numpy as np

from .model import GPT, GPTConfig


def save_checkpoint(path, model, opt, step, best_val, cfg):
    arrays = {f"param/{k}": v for k, v in model.state_dict().items()}
    if opt is not None:
        arrays.update({f"opt/{k}": np.asarray(v) for k, v in opt.state_dict().items()})
    arrays["meta"] = np.array(json.dumps({"step": step, "best_val": best_val, "config": cfg}))
    tmp = str(path) + ".tmp.npz"
    np.savez(tmp, **arrays)
    import os
    os.replace(tmp, path)        # atomic: a crash mid-save never corrupts the old file


def load_checkpoint(path, model, opt=None):
    z = np.load(path)
    model.load_state_dict({k[6:]: z[k] for k in z.files if k.startswith("param/")})
    if opt is not None:
        opt.load_state_dict({k[4:]: z[k] for k in z.files if k.startswith("opt/")})
    meta = json.loads(str(z["meta"]))
    return meta["step"], meta["best_val"]


def load_model(path):
    """Rebuild a model from a checkpoint alone (config is stored inside)."""
    z = np.load(path)
    cfg = json.loads(str(z["meta"]))["config"]
    model = GPT(GPTConfig(**cfg["model"]))
    load_checkpoint(path, model)
    return model, cfg

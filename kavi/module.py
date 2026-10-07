"""Tiny Module/Param system — the "Lego" interface every brick implements.

There is no autograd. Each Module:
  * ``forward(x)``  computes its output and caches whatever its backward pass needs;
  * ``backward(dout) -> dx``  receives dLoss/dOutput, *accumulates* dLoss/dParam into each
    ``Param.grad`` and returns dLoss/dInput for the module below it.

Gradients accumulate (``+=``) rather than overwrite so that a parameter used twice — the
tied token embedding is both the input lookup table and the output projection — collects
the gradient from both uses. Call ``zero_grad()`` once per optimisation step.
"""
from . import backend as B


class Param:
    """A trainable tensor plus its gradient buffer."""
    __slots__ = ("data", "grad", "decay")

    def __init__(self, data, decay: bool = True):
        self.data = data
        self.grad = B.xp.zeros_like(data)
        # AdamW applies weight decay only to matrices (decay=True), never to biases/gains.
        self.decay = decay


class Module:
    training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    # -- traversal -------------------------------------------------------------------
    def _children(self):
        for key, value in vars(self).items():
            if isinstance(value, (Param, Module)):
                yield key, value
            elif isinstance(value, (list, tuple)):
                for i, item in enumerate(value):
                    if isinstance(item, (Param, Module)):
                        yield f"{key}.{i}", item

    def named_params(self, prefix: str = ""):
        """Yield (dotted_name, Param), each Param once even if shared (weight tying)."""
        seen = set()
        stack = [(prefix, self)]
        out = []
        while stack:
            pre, mod = stack.pop(0)
            for key, child in mod._children():
                if isinstance(child, Param):
                    if id(child) not in seen:
                        seen.add(id(child))
                        out.append((pre + key, child))
                else:
                    stack.append((pre + key + ".", child))
        return out

    def params(self):
        return [p for _, p in self.named_params()]

    def modules(self):
        yield self
        for _, child in self._children():
            if isinstance(child, Module):
                yield from child.modules()

    # -- state -----------------------------------------------------------------------
    def zero_grad(self) -> None:
        for p in self.params():
            p.grad.fill(0)

    def train(self, mode: bool = True):
        for m in self.modules():
            m.training = mode
        return self

    def eval(self):
        return self.train(False)

    def num_params(self) -> int:
        return int(sum(p.data.size for p in self.params()))

    def state_dict(self) -> dict:
        return {name: B.to_numpy(p.data) for name, p in self.named_params()}

    def load_state_dict(self, state: dict) -> None:
        params = dict(self.named_params())
        missing = set(params) - set(state)
        if missing:
            raise KeyError(f"checkpoint is missing parameters: {sorted(missing)}")
        for name, p in params.items():
            if tuple(state[name].shape) != tuple(p.data.shape):
                raise ValueError(f"{name}: shape {state[name].shape} != {p.data.shape}")
            p.data[...] = B.asarray(state[name])

# 11 · GPU: the same NumPy code on a free Kaggle GPU

> **What you'll learn.** How Kavi runs its hand-written forward and backward passes on a GPU
> without rewriting them: a one-variable backend switch to CuPy. You'll learn the three
> traps that switch has to avoid, how a multi-file project is shipped to Kaggle as a single
> script and driven from the command line, and why the GPU helps, starting from a CPU
> performance finding about how NumPy executes matrix multiplies.

Anton's NumPy GPT trained for 11 hours on a CPU, and moving to a GPU meant rewriting it in
PyTorch. Kavi keeps the *same* NumPy-style code and gets roughly a 10× speed-up on a free
Kaggle T4.

---

## 1. The idea: CuPy is NumPy on the GPU

[CuPy](https://cupy.dev) mirrors NumPy's API: `cupy.exp`, `cupy.zeros_like`, `a @ b`,
`a.reshape`, `a.sum(axis=-1)` all exist with the same signatures, and operate on arrays in
GPU memory. If every line of the model calls "the array library" through one indirection,
swapping that library swaps the hardware.

That indirection is `kavi/backend.py`:

```python
xp = np                 # active array module (numpy or cupy)
dtype = np.float32

def set_backend(backend="numpy", precision=None):
    global xp, name
    xp = np if backend == "numpy" else cupy
```

Every module writes `B.xp.exp(...)`, `B.xp.zeros_like(...)`, or `xp = B.xp` at the top of a
method, never `np.` on the hot path. `scripts/train.py --backend cupy` calls
`B.set_backend("cupy")` and then builds the model, and from then on every array lives on
the GPU.

---

## 2. Trap 1: look `xp` up at call time

The tempting way to write a module is:

```python
from kavi.backend import xp        # WRONG: binds whatever xp is *right now*
def forward(self, x): return xp.exp(x)
```

`from ... import xp` copies a reference to the object `xp` points to *at import time*, which
is NumPy, into the importing module's own namespace. `set_backend("cupy")` later rebinds
the name `xp` *inside `kavi.backend`*, but the copy in the other module still points to
NumPy. You'd then call `numpy.exp` on a CuPy array. CuPy refuses implicit device-to-host
conversion, so you'd get a `TypeError` at best, or a half-GPU, half-CPU model shuffling data
across the bus at worst.

`from . import backend as B` followed by `B.xp` instead looks the attribute up on the module
object **every time the line runs**, so it always sees the current backend. That's the rule
stated in `backend.py`'s docstring, and every file in `kavi/` follows it.

The same timing issue applies to **allocations**. `Param.__init__` does
`B.xp.zeros_like(data)`, and `B.normal` returns `B.asarray(...)`, both on the backend
active *when the model is built*. So `train.py` and `backend_parity.py` call `set_backend`
first and construct `GPT` second. Swap the order and you get NumPy parameters fed CuPy
batches.

## 3. Host-side seeded initialisation

```python
def normal(rng: np.random.Generator, shape, std):
    return asarray(rng.normal(0.0, std, size=shape))
```

All weights are drawn on the **host** with a seeded NumPy `Generator` and then copied to the
active device. CuPy has its own RNG, but its streams differ from NumPy's. Drawing on the
device would give different initial weights on CPU and GPU for the same seed. Drawing on the
host makes `GPT(cfg, seed=7)` **bit-identical** on both backends. That gives:

* CPU and GPU runs of the same config start from the same point, so differences in results
  come from hardware arithmetic, not luck;
* `scripts/backend_parity.py` (§6) can demand near machine-precision agreement.

RoPE's `cos`/`sin` tables are built the same way (NumPy float64 on the host, then
`B.asarray`). Dropout masks and batch indices are random *during* training, and those come
from `B.seed`/`B.rand` (dropout) and a host NumPy RNG (batches). They're reproducible per
backend, but not across backends.

## 4. Trap 2: `W[idx] += v` loses gradient

The embedding's backward has to add each position's gradient into the row of the token at
that position. When a token appears several times in a batch (" the" appears hundreds of
times), its row must receive the **sum**. The obvious code is wrong:

```python
W = np.zeros((5, 2)); idx = np.array([1, 3, 1, 1]); v = np.ones((4, 2))
W[idx] += v            # W[:,0] -> [0, 1, 0, 1, 0]   row 1 got 1, not 3!
np.add.at(W, idx, v)   # W[:,0] -> [0, 3, 0, 1, 0]   correct
```

`W[idx] += v` expands to `W[idx] = W[idx] + v`. The right-hand side gathers three copies of
row 1, each plus one, and the assignment writes row 1 three times with the same value, so
the last write wins. No error, no warning, and every repeated token silently gets only one
position's gradient. Gradcheck on a batch with no repeated ids would even pass.

`B.scatter_add(target, idx, values)` does it properly: `np.add.at` on CPU, and on GPU
`cupy.add.at` (CuPy ≥ 13) with a fallback to `cupyx.scatter_add` for older versions. On the
GPU the accumulation uses atomic adds, so many threads can safely add into the same row.
`Embedding.backward` is the only caller. `test_gradcheck.py::test_embedding` uses ids with
repeats, which is what makes it a meaningful test of this.

## 5. Trap 3: `float()` is a synchronisation

A GPU runs asynchronously: Python enqueues kernels and moves on. Converting a device scalar
to a Python number (`float(x)`, `x.item()`, printing it) forces the host to **wait** for every
queued kernel to finish and then copy the value back. A few per step is fine. Hundreds per
step (one per parameter tensor) would stall the pipeline each time.

`clip_grad_norm` in `kavi/optim.py` is written with that in mind:

```python
# sum on-device and convert once: on a GPU every float() is a host<->device sync
total = float(xp.sqrt(sum((p.grad * p.grad).sum() for p in params)))
```

The per-parameter sums stay as 0-d device arrays and are added on the device. There's
**one** `float()` per step, which the `if total > max_norm` branch needs anyway. Elsewhere
the training loop only converts the loss to a float every `log_interval` steps, and
evaluation once per eval batch.

---

## 6. Proving the GPU computes the same thing: `scripts/backend_parity.py`

For each preset (`gpt2`, `kavi`) it builds a small model (vocab 50, $C=32$, 2 layers, $T=16$)
with `seed=7` on NumPy *and* on CuPy, both in **float64**, runs one forward/backward on the
same batch, and compares the loss and every parameter gradient:

```
assert abs(l_cpu - l_gpu) < 1e-9 and worst < 1e-8     # worst = max relative grad diff
```

Float64 is what makes this a sharp test. Real bugs (a NumPy-only call, a lost scatter-add,
a missing sync) produce differences many orders of magnitude bigger than float64 rounding.
On the Kaggle probe the result was:

```
gpt2: loss cpu 3.919125383659 gpu 3.919125383659 | worst grad rel diff 5.96e-16
kavi: loss cpu 3.899363421992 gpu 3.899363421992 | worst grad rel diff 7.58e-16
```

That's a few ulps: the GPU and CPU differ only in floating-point summation order. The
kernel runs this **before** any training, and with `check=True`, so if parity fails the
kernel stops and no GPU hours are wasted on a wrong model.

---

## 7. Why the GPU helps: how matmuls actually run

### 7.1 A CPU finding: flatten before you multiply

`Linear.forward` in `kavi/layers.py` does:

```python
y = x.reshape(-1, self.n_in) @ self.weight.data      # (B·T, C) @ (C, N): ONE GEMM
return y.reshape(*x.shape[:-1], self.n_out)
```

Mathematically `x @ W` with `x` of shape `(B, T, C)` gives the same answer. But NumPy runs a
stacked `(B, T, C) @ (C, N)` matmul as a loop of $B$ separate `(T, C) @ (C, N)` BLAS calls.
On this laptop that measured **12.9 ms**, against **2.9 ms** for the flattened 2-D version,
a **4.4×** difference for identical arithmetic. A single big GEMM lets OpenBLAS tile the
problem for its caches and spread it over every core. Many small GEMMs each pay call
overhead and are too small to parallelise well. Every projection in the model (qkv,
attention output, all MLP matrices) goes through `Linear`, so this one reshape speeds up
most of the FLOPs.

### 7.2 What can't be flattened: attention

Attention's two big products are genuinely batched:

$$
\underbrace{(B, H, T, D) @ (B, H, D, T)}_{\text{scores}} \rightarrow (B, H, T, T),
\qquad
\underbrace{(B, H, T, T) @ (B, H, T, D)}_{\text{output}} \rightarrow (B, H, T, D).
$$

Each of the $B \cdot H$ (batch, head) pairs has its *own* matrices, so there is no single
GEMM to flatten into. For `bpe_base.json` that's $32 \times 8 = 256$ separate
$256 \times 32 \times 256$ products in each of the two forward matmuls, and four more in the
backward pass. These small, skinny GEMMs are exactly what a CPU does worst.

A GPU does them well: CuPy dispatches a batched matmul to cuBLAS's **strided batched GEMM**,
which runs all 256 small products in a single kernel launch spread across thousands of
cores. The same is true of the elementwise work (softmax, GELU, norms), where every element
gets its own thread.

### 7.3 The numbers

| Hardware | Training throughput |
|---|---|
| laptop CPU (NumPy + OpenBLAS) | ~1.4–3.8 k tok/s (char model) |
| Kaggle Tesla T4 (CuPy 14.0.1) | **~32–33 k tok/s** per GPU (BPE model, 5.8 M params, B=32, T=256) |

The laptop figure is for the much smaller char model and it still lost by about 10×. The
300-step probe of each preset took about 1.5 minutes on one T4. A complete 4000-step BPE run
is $4000 \times 8192 / 32{,}500 \approx 17$ minutes plus evals. Anton needed 11 hours on a
CPU.

---

## 8. The Kaggle pipeline (`kaggle/build_kernel.py`)

### 8.1 One file, a whole package

A Kaggle **script** kernel is a single `.py` file, but Kavi is a package (`kavi/`,
`scripts/`, `configs/`). `build()`:

1. zips `kavi/`, `scripts/` and `configs/` in memory (skipping `__pycache__`);
2. base64-encodes the zip into a string;
3. pastes it into `TEMPLATE` as `BUNDLE = "..."`, along with the sweep file to run, and
   writes `kaggle/build/kavi_kernel.py` (about 43 KB) plus `kernel-metadata.json` (private,
   `enable_gpu`, `enable_internet`, id `smirkxd/kavi-gpt`).

At runtime the kernel decodes and unzips itself into `/tmp/kavi-gpt` and runs ordinary
scripts from there. No dataset upload is needed, because the kernel re-downloads
Shakespeare (internet is enabled). The data pipeline is deterministic, and the Kaggle log
reproduced the laptop's numbers exactly: 5,359,343 chars, 1,578,753 BPE train tokens,
3.03 bytes/token.

### 8.2 What the kernel does

1. Unpack the bundle, then run `nvidia-smi`, which logs which GPU(s) were assigned.
2. `import cupy`, `pip install cupy-cuda12x` only if that fails. The probe found **CuPy
   14.0.1 preinstalled**.
3. `prepare_data.py`, `tokenize_data.py char`, `tokenize_data.py bpe --vocab 4096`.
4. `backend_parity.py`. A failure here aborts the kernel.
5. **One sweep shard per GPU, in parallel.** For each GPU `g` of `n_gpu`, launch
   `sweep.py SWEEP --backend cupy --out /kaggle/working/runs --shard g/n_gpu` with
   `CUDA_VISIBLE_DEVICES=g`, so each process sees a single GPU (as its device 0) and takes
   every `n_gpu`-th run. Each shard's output goes to `runs/sweep_gpu<g>.log`, and the tail
   of each is echoed into the kernel log at the end.
6. `summarize.py` writes the table and `curves.png`.
7. Delete every `last.npz` to keep the download small (`best.npz` is kept).

Everything under `/kaggle/working/` becomes the kernel's downloadable output.

### 8.3 Driving it from the laptop

```bash
python kaggle/build_kernel.py build --sweep configs/sweep_ablation.json
python kaggle/build_kernel.py push        # upload + start a new version
python kaggle/build_kernel.py status      # queued / running / complete / error
python kaggle/build_kernel.py output      # download into kaggle/output/
```

These use the official `kaggle` package's `KaggleApi` (`kernels_push`, `kernels_status`,
`kernels_output`), with credentials from the usual `kaggle.json`.

### 8.4 Gotchas (each one cost time)

* **Which GPU you get varies.** The code comments, written from earlier projects, say
  API-pushed kernels get a single Tesla P100 (sm_60). The Kavi probe was given **2× Tesla
  T4** instead. The kernel therefore asks CuPy how many devices exist and shards across
  all of them, rather than assuming either.
* **Only the Windows Store Python works with the Kaggle API on this machine.** Run the
  `push/status/output` commands with that interpreter (it's the `python` on `PATH` here).
* **`kernels_output` skips files that already exist locally.** A re-download would silently
  keep the previous version's files. `output` deletes `kaggle/output/` first.
* **The output `.log` write can fail on cp1252.** Windows' default console encoding can't
  represent some characters in the log. `output` sets `PYTHONUTF8=1` and catches the
  exception, since the data files have usually landed by then.

### 8.5 Free-tier budget

Kaggle gives about **30 GPU-hours per week** and at most **12 hours per session**. That
shapes the workflow:

* run a **probe** (`sweep_probe.json`: 2 runs × 300 steps, ~4 minutes end to end) before
  any long sweep, to catch pipeline bugs cheaply;
* keep runs short and resumable. The sweep skips finished runs, so a timed-out kernel can
  be pushed again and continue with the rest;
* two T4s per session double the sweep throughput for the same quota.

---

## 9. How we know it's right

* `scripts/backend_parity.py`: float64 loss and every gradient agree to ~1e-15 relative,
  on both presets, every time the kernel runs.
* `tests/test_gradcheck.py::test_embedding`: catches a non-accumulating scatter (ids
  repeat).
* The probe: the same 300-step config on GPU gave val bpb **2.156 (`kavi`) vs 2.335
  (`gpt2`)**, plausible, ordered, and consistent with the CPU loss scale.

## 10. Pitfalls

* Building the model before `set_backend` (NumPy weights, CuPy data).
* `np.` instead of `B.xp.` in a new module: works on CPU, breaks or crawls on GPU. Run
  `backend_parity.py` after adding any brick.
* `float(...)` inside a per-parameter loop: one host-device sync per tensor per step.
* Benchmarking a GPU without synchronising: kernels are asynchronous, so time a whole step
  (the per-step `float()` in clipping syncs anyway) rather than one call.

## 11. Try it yourself

1. **Reproduce the flattening finding.** Time `x @ W` vs `(x.reshape(-1, C) @ W)` for
   `x = np.random.randn(32, 256, 256).astype(np.float32)`, `W` of shape `(256, 768)`, with
   `timeit`. *Expected:* the flattened version is several times faster (4.4× was measured
   on this laptop. The exact ratio depends on the BLAS and core count, and is smaller when
   another process is using the CPU).
2. **See the lost gradient.** Run `Embedding.backward` with `ids = [[1, 1, 1]]` and
   `dout = ones`, once as written and once with `scatter_add` replaced by
   `grad[ids] += dout`. *Expected:* row 1 receives 3 per column in the correct version and 1
   in the broken one.
3. **Count syncs.** Rewrite `clip_grad_norm` in a scratch file to call `float()` on each
   parameter's sum and add them in Python. On a GPU, time 100 steps of each. *Expected:* the
   per-parameter version is measurably slower (dozens of syncs per step instead of one).
   On CPU there's no difference, which is why the bug would go unnoticed locally.
4. **Make the clip sync-free.** Compute `scale = xp.minimum(1.0, max_norm / (total + 1e-6))`
   on the device and always multiply. *Expected:* same results, but you lose the Python
   `float` for logging. Decide whether it's worth it, given that `train.py` prints `gnorm`.

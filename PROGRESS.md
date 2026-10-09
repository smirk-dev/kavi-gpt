# PROGRESS — read this first in a new session

State of each phase in [PLAN.md](PLAN.md). Updated 2026-10-08.

| Phase | Status | Evidence |
|---|---|---|
| 0 Foundations | ✅ | backend switch, Module/Param, gradcheck; data prepared |
| 1 Bricks | ✅ | `tests/test_gradcheck.py`: every brick < 1e-6 rel err (float64) |
| 2 Assemble | ✅ | whole-model gradcheck < 1e-5; `tests/test_torch_parity.py` grads == torch autograd to 1e-7 |
| 3 Train | ✅ | overfit-one-batch + SGD-vs-AdamW tests; CPU char run (`runs/char_kavi`) |
| 4 BPE | ✅ | `tests/test_bpe.py`: incremental == naive merges, exact round trip; 4096 vocab in 9.2 s |
| 5 GPU | ✅ | Kaggle probe v1: 2× T4, CPU/GPU grad parity 7.6e-16, ~32k tok/s per T4 |
| 6 Ablations | ✅ | Kaggle v2, 14 runs in 160 min on 2× T4; `results/ablation/`, table in `docs/journal.md` (kavi 1.627 vs gpt2 1.707 bpb) |
| 7 Inference & insight | ✅ | KV cache (7.2×), flash == naive, `docs/img/attention.png`, `scripts/induction.py` (no induction heads yet) |
| Docs | ✅ | 15 chapters + journal, links checked |
| 8 Scale-up | ⏳ | corpus built: 137 texts, 11.9× Shakespeare, 18.7M BPE-8192 train tokens, leakage-guarded (`scripts/build_corpus.py`, `docs/14-scaling.md`); scale probe + long run pending |

## How to resume
- Tests: `python -m pytest tests -q` (48 pass; torch optional).
- Kaggle loop: `python kaggle/build_kernel.py build --sweep <file>` → `push` → `status` → `output`
  (downloads to `kaggle/output/`, cleared first). Run from the Windows Store `python` (it has
  the kaggle API auth). Copy results you keep into `results/<name>/` (git-tracked, no .npz).
- Results table: `python scripts/summarize.py results/<name> --plot docs/img/<name>.png`.

## Checkpoints
`ckpt/` (gitignored, weights only, ~23 MB each): kavi-s2, gpt2-s2, anton-s2, kavi-char-s1.
Load with `kavi.checkpoint.load_model`. Optimizer state was stripped, so they can't be resumed.

## Next
- Done 2026-10-09: Kaggle v3 follow-up (`results/followup/`, 59 min). Anton's edge is ReLU (−0.008; biases −0.002), and ReLU ties SwiGLU. SGD+momentum at lr 0.1 gives 1.722 vs plain 2.238 vs AdamW 1.627. Written up in the journal.
- Then `configs/sweep_scale_probe.json` (300 steps × S/M/L, measures T4 throughput), then the long run + `scripts/induction.py`.

## Known facts / gotchas found while building
- NumPy stacked matmul `(B,T,C)@(C,N)` is 4.4× slower than flattening to 2-D → `Linear` flattens.
- Gradcheck: use h=1e-4 (h=1e-6 is rounding-dominated for small grads); zero-by-symmetry
  gradients (attention key bias without RoPE) need an absolute check.
- Kaggle assigned 2× T4 to the API-pushed kernel this time (earlier projects got a P100);
  the kernel handles 1..n GPUs. CuPy 14 is preinstalled.
- Laptop RAM is tight (~2 GB free): CPU training ran at 1.4–3.8k tok/s depending on load.

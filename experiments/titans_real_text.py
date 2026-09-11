"""Roadmap item 3: does the forgetting-curve / non-localization story from
titans_memdiff.py and titans_ablation.py hold when the memory reads REAL
text instead of random vectors?

Everything so far (this branch) fed the memory random 64-dim vectors, which
sidesteps a real question: with actual language, is there structure in
*what* gets written (do certain byte patterns get preferentially retained)
that random-vector documents can't show at all? This trains a small, real,
byte-level language model with a Titans neural memory wired in
(`titans_pytorch.MemoryAsContextTransformer`, the "MAC" architecture), on
real enwik8 text, then re-runs the same early-vs-end snapshot diff on the
memory as it reads a real held-out passage.

Scaled WAY down from the library's own `train_mac.py` recipe on purpose:
that one is dim=384, depth=8, 100k batches, wandb, flex-attention -- a real
multi-hour+ training run, not a Colab demo. This version:
- dim=128, depth=4, ONE memory layer (layer 2) instead of three
- the memory's own MLP is dim_head=64, MemoryMLP(64, depth=2) -- the exact
  same 64->256->64 shape used in titans_memdiff.py / titans_ablation.py, so
  results here are directly comparable to the toy-vector experiments
- neural_memory_segment_len=8 (independent of the attention segment_len) --
  closer to titans_memdiff.py's CHUNK=16 than to titans_ablation.py's
  chunk_size=1, since the MAC transformer ties memory chunking to segment
  boundaries rather than exposing a free chunk_size=1 option
- no flex-attention, no wandb, no gradient accumulation, plain Adam
- a few thousand steps on a short sequence length -- minutes on a Colab T4,
  not hours. Loss will NOT reach the library's reported numbers; this is a
  correctness/qualitative-structure check, not a real language model.

Needs the enwik8 dataset: point --data at a local `enwik8.gz`
(e.g. the one bundled with a clone of github.com/lucidrains/titans-pytorch
at data/enwik8.gz) or pass --data to a different path.

Run
---
    pip install titans-pytorch
    python experiments/titans_real_text.py --data /path/to/enwik8.gz [--steps 2000]
"""
from __future__ import annotations

import argparse
import gzip

import numpy as np
import torch
import torch.nn.functional as F
from titans_pytorch import MemoryAsContextTransformer, MemoryMLP

DIM_HEAD, HIDDEN = 64, 256  # matches titans_memdiff.py / titans_ablation.py exactly


def build_model(neural_memory_segment_len: int = 8) -> MemoryAsContextTransformer:
    # dim=64 (not the library's usual 384) so the transformer's width matches
    # the memory's dim_head exactly -- NeuralMemory requires dim_head == dim
    # when heads=1, and keeping the memory at 64->256->64 keeps every result
    # here directly comparable to titans_memdiff.py / titans_ablation.py.
    return MemoryAsContextTransformer(
        num_tokens=256,               # raw bytes
        dim=DIM_HEAD,
        depth=4,
        segment_len=32,                # local attention window
        neural_memory_segment_len=neural_memory_segment_len,
        num_persist_mem_tokens=4,
        num_longterm_mem_tokens=4,
        neural_memory_layers=(2,),     # a single memory layer, for simplicity
        dim_head=32,
        heads=2,
        neural_memory_model=MemoryMLP(DIM_HEAD, depth=2, expansion_factor=4.),  # 64 -> 256 -> 64: MemoryMLP defaults to expansion_factor=2 (hidden=128) unless told otherwise -- NeuralMemory's OWN internal default of 4 only applies when it builds the MLP itself
        neural_memory_kwargs=dict(dim_head=DIM_HEAD, heads=1),
        use_flex_attn=False,
    )


def load_enwik8(path: str, n_bytes: int = int(20e6)):
    with gzip.open(path) as f:
        data = np.frombuffer(f.read(n_bytes), dtype=np.uint8).copy()
    split = int(len(data) * 0.9)
    return torch.from_numpy(data[:split]).long(), torch.from_numpy(data[split:]).long()


def sample_batch(data: torch.Tensor, seq_len: int, batch_size: int):
    starts = torch.randint(0, data.size(0) - seq_len - 1, (batch_size,))
    return torch.stack([data[s: s + seq_len + 1] for s in starts])


def train(model, data_train, data_val, steps: int, seq_len: int, batch_size: int, lr: float, device: str):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for step in range(steps):
        batch = sample_batch(data_train, seq_len, batch_size).to(device)
        loss = model(batch, return_loss=True)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step()
        if step % max(1, steps // 20) == 0 or step == steps - 1:
            model.eval()
            with torch.no_grad():
                vbatch = sample_batch(data_val, seq_len, batch_size).to(device)
                vloss = model(vbatch, return_loss=True)
            model.train()
            print(f"  step {step:5d}  train loss {loss.item():.3f}  val loss {vloss.item():.3f}"
                  f"  (byte-uniform baseline: {np.log(256):.3f} nats)")


def _cos(a, b, axis):
    return np.sum(a * b, axis) / (np.linalg.norm(a, axis=axis) * np.linalg.norm(b, axis=axis) + 1e-9)


@torch.no_grad()
def diff_memory_on_passage(model: MemoryAsContextTransformer, passage: torch.Tensor, device: str):
    """Same early-vs-end weight-snapshot diff as titans_memdiff.py, but the
    snapshots come from a real trained memory reading real held-out text
    instead of a synthetic NeuralMemory reading random vectors."""
    model.eval()
    _, cache = model(passage.unsqueeze(0).to(device), return_cache=True)
    _, _, neural_mem_caches = cache
    state = neural_mem_caches[0]  # the one memory layer

    U0 = state.updates["model.weights.0"].detach()[0].cpu().numpy()  # (chunks, 64, 256)
    U1 = state.updates["model.weights.1"].detach()[0].cpu().numpy()  # (chunks, 256, 64)

    print(f"passage length {passage.shape[0]} tokens -> {U0.shape[0]} memory weight snapshots")

    g_in, g_out = U0[1], U0[-1]
    d_in, d_out = U1[1], U1[-1]
    gate_cos = _cos(g_in, g_out, axis=0)
    down_cos = _cos(d_in, d_out, axis=1)
    nr = (np.linalg.norm(g_out, axis=0) + 1e-9) / (np.linalg.norm(g_in, axis=0) + 1e-9)
    moved = gate_cos < 0.99

    print(f"hidden units moved (gate_cos < 0.99): {moved.sum()} / {len(gate_cos)}")
    print(f"gate_cos  min {gate_cos.min():+.3f}   median {np.median(gate_cos):+.3f}")
    print(f"norm_ratio (end/early write) on moved units: mean {nr[moved].mean() if moved.any() else float('nan'):.2f}")

    incr_norm = np.linalg.norm(np.diff(U1, axis=0), axis=2)
    early = np.argsort(incr_norm[0])[::-1][:20]
    w_early, w_end = U1[1, early, :], U1[-1, early, :]
    scos = _cos(w_early, w_end, axis=1)
    smag = np.linalg.norm(w_end, axis=1) / (np.linalg.norm(w_early, axis=1) + 1e-9)
    print(f"\nthe 20 units the first chunk wrote hardest:")
    print(f"  direction retained (cos): mean {scos.mean():+.3f}")
    print(f"  magnitude retained (ratio): mean {smag.mean():.2f}  min {smag.min():.2f}  max {smag.max():.2f}")
    print("(compare to titans_memdiff.py's random-vector numbers: does real text forget faster, slower, or the same?)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to enwik8.gz")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("loading enwik8...")
    data_train, data_val = load_enwik8(args.data)

    model = build_model().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params / 1e6:.1f}M params, device={device}\n")

    print(f"training {args.steps} steps...")
    train(model, data_train, data_val, args.steps, args.seq_len, args.batch_size, args.lr, device)

    print("\ndiffing the memory on a real held-out passage...")
    passage = sample_batch(data_val, 512, 1)[0]
    diff_memory_on_passage(model, passage, device)


if __name__ == "__main__":
    main()

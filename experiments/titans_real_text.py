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

Result (2026-09-11), two independent runs, effect STRENGTHENING with more
training (rules out "undertrained artifact"):

    local, 1000 steps, CPU, val loss 2.27:  norm_ratio 1.81, direction
        retained 0.87, magnitude retained min 1.30
    Colab, 3000 steps, T4, val loss 1.88:   norm_ratio 2.20, direction
        retained 0.95, magnitude retained min 1.66 (never once shrinks)

Compare to titans_memdiff.py's toy recall-task-trained memory: norm_ratio
~0.07, magnitude always < 1 (aggressive exponential forgetting). A memory
trained on real language modeling does NOT develop that forgetting -- it
behaves like the *untrained* random-vector case (writes accumulate), and
more real training pushes further in that direction, not less. The earlier
"trained memory forgets exponentially" finding is real for that specific
toy task, but is not a general property of trained Titans memories -- see
experiments/README.md roadmap item 3 for the full writeup.

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


@torch.no_grad()
def inspect_decay_gate(model: MemoryAsContextTransformer, passage: torch.Tensor, device: str):
    """WHY does a real-text-trained memory forget less -- is the forget gate
    genuinely reading the input and choosing to retain predictable/real-
    looking content, or did training just settle on a fixed low-forgetting
    habit regardless of what it's shown? Hooks the memory layer's own
    `to_decay_factor` (the thing that produces alpha_t) during a real
    forward pass on real text, then again on random bytes through the SAME
    trained weights, and compares the two. If real text produces a
    noticeably lower gate value than random bytes, that's direct evidence
    for "the gate reads the input." If they're similar, the low forgetting
    is a fixed, input-independent habit the training run settled into."""
    mem_layer = next(group[4] for group in model.layers if group[4] is not None)

    captured = {}

    def hook(_module, _inp, out):
        captured["decay"] = out.sigmoid().detach()

    handle = mem_layer.to_decay_factor.register_forward_hook(hook)

    model(passage.unsqueeze(0).to(device), return_cache=True)
    decay_real = captured["decay"]

    random_bytes = torch.randint(0, 256, passage.shape, device=device)
    model(random_bytes.unsqueeze(0), return_cache=True)
    decay_random = captured["decay"]

    handle.remove()

    print(f"learned decay gate (alpha_t) on REAL TEXT:    "
          f"mean={decay_real.mean():.4f}  min={decay_real.min():.4f}  max={decay_real.max():.4f}")
    print(f"learned decay gate (alpha_t) on RANDOM BYTES: "
          f"mean={decay_random.mean():.4f}  min={decay_random.min():.4f}  max={decay_random.max():.4f}")
    gap = decay_random.mean() - decay_real.mean()
    print(f"\ngap (random - real): {gap:+.4f}")
    if gap > 0.02:
        print("-> the gate IS input-sensitive: it forgets less specifically for real text.")
    elif gap < -0.02:
        print("-> unexpected: the gate forgets MORE for real text than random bytes.")
    else:
        print("-> the gate looks roughly the SAME regardless of input: low forgetting is a")
        print("   fixed habit from training, not a live per-input decision.")


@torch.no_grad()
def consecutive_write_alignment(model: MemoryAsContextTransformer, passage: torch.Tensor, device: str):
    """Is real text's accumulation (Finding 5) coming from correlated,
    content-driven structure alone (should show up even UNTRAINED, since
    the raw bytes are correlated regardless of what the network learned),
    or does something learned during training create it? Measures the
    cosine similarity between EACH chunk's write direction and the
    PREVIOUS chunk's write direction -- consistently aligned (near +1)
    means writes reinforce each other (constructive, explains
    accumulation); scattered/negative means they partly cancel."""
    _, cache = model(passage.unsqueeze(0).to(device), return_cache=True)
    _, _, neural_mem_caches = cache
    state = neural_mem_caches[0]

    U1 = state.updates["model.weights.1"].detach()[0].cpu().numpy()  # (chunks, hidden, dim_head)
    incr = np.diff(U1, axis=0)  # (chunks-1, hidden, dim_head) -- each chunk's write
    incr_flat = incr.reshape(incr.shape[0], -1)  # flatten hidden x dim_head into one write vector per chunk

    norms = np.linalg.norm(incr_flat, axis=1, keepdims=True) + 1e-9
    unit = incr_flat / norms
    consecutive_cos = (unit[1:] * unit[:-1]).sum(axis=1)  # cos(write_t, write_{t-1}) for each t

    return consecutive_cos


def report_write_alignment(label: str, cos_values: np.ndarray):
    print(f"{label:<28}  consecutive-write cos: mean {cos_values.mean():+.3f}  "
          f"median {np.median(cos_values):+.3f}  (n={len(cos_values)} chunk-pairs)")


def titans_logit_lens(model: MemoryAsContextTransformer, direction: np.ndarray, k: int = 8):
    """Direct port of marv.probe.logit_lens for a live Titans memory unit
    instead of a static FFN column: normalize through the model's OWN
    final-norm gain (not plain RMS scaling -- see marv/probe.py's docstring
    on why that gain matters), then project through the vocabulary head.
    Returns (byte_ids, their_logits) for the top k bytes this direction
    currently pushes toward."""
    eps = 1e-6
    v = direction / np.sqrt((direction ** 2).mean() + eps)
    gain = model.norm.weight.detach().cpu().numpy()
    v = v * gain
    lm_head = model.to_logits.weight.detach().cpu().numpy()  # (num_tokens, dim)
    logits = lm_head @ v
    idx = np.argsort(-logits)[:k]
    return idx, logits[idx]


def _printable_byte(b: int) -> str:
    return repr(chr(b))[1:-1] if 32 <= b < 127 else f"\\x{b:02x}"


@torch.no_grad()
def describe_memory_units(model: MemoryAsContextTransformer, passage: torch.Tensor, device: str,
                           units: list[int] | None = None, k: int = 6):
    """describe_feature()-style readout for a live memory: for each unit,
    what does its CURRENT output direction (at the end of the passage)
    promote in byte-space? A direct readout, not an inference from an
    experiment -- the thing roadmap item 3 flagged as still open."""
    _, cache = model(passage.unsqueeze(0).to(device), return_cache=True)
    _, _, neural_mem_caches = cache
    state = neural_mem_caches[0]
    U1 = state.updates["model.weights.1"].detach()[0].cpu().numpy()  # (chunks, hidden, dim_head)
    final = U1[-1]  # (hidden, dim_head) -- the memory's current state, end of passage

    units = units if units is not None else list(range(final.shape[0]))
    for u in units:
        idx, logits = titans_logit_lens(model, final[u], k=k)
        bytes_str = " ".join(f"'{_printable_byte(b)}'" for b in idx)
        print(f"  unit {u:>4}  ->  {bytes_str}")


def compute_memory_jacobian(model: MemoryAsContextTransformer, passages: list[torch.Tensor], device: str,
                             position: int = -1) -> torch.Tensor:
    """Simplified version of Anthropic's J-lens (arXiv:2607.15495,
    "Verbalizable Representations Form a Global Workspace in Language
    Models"): J_l = E[d(final logits)/d(activation at layer l)], averaged
    over a corpus -- instead of assuming a naive logit lens's shortcut
    (project an intermediate vector straight through the FINAL norm, as if
    nothing downstream could change its meaning), this measures the REAL,
    exact transport through whatever layers actually come after the memory,
    via real backprop through the model's own true computation graph.

    Simplifications vs. the paper, stated plainly: uses the SAME position as
    both source and target (the paper averages over source position t AND
    later positions t' > t; this only looks at t'=t) and computes an exact
    per-example Jacobian via 256 backward passes per passage rather than the
    approximation techniques a Claude-scale model would need -- tractable
    here specifically because this model is tiny (dim_head=64, 0.4M params).

    Returns G, shape (num_tokens, dim_head): G @ v approximates how
    injecting direction v into the memory's retrieved output would move
    each token's logit, properly accounting for the remaining attention/
    feedforward layers -- unlike titans_logit_lens, which skips them."""
    mem_layer = next(group[4] for group in model.layers if group[4] is not None)
    num_tokens = model.to_logits.out_features
    G_sum = torch.zeros(num_tokens, DIM_HEAD, device=device)

    for passage in passages:
        captured = {}

        # NOTE: mac_transformer.py calls `mem.forward(...)` directly, not
        # `mem(...)` -- that bypasses nn.Module.__call__, so a normal
        # register_forward_hook here silently never fires. Monkey-patching
        # .forward itself is the only way to intercept it.
        original_forward = mem_layer.forward

        def patched_forward(*args, **kwargs):
            result = original_forward(*args, **kwargs)
            captured["retrieved"] = result[0]
            return result

        mem_layer.forward = patched_forward
        model.zero_grad(set_to_none=True)
        try:
            logits = model(passage.unsqueeze(0).to(device))
        finally:
            mem_layer.forward = original_forward

        retrieved = captured["retrieved"]  # (1, seq, dim_head), requires_grad
        pos = position if position >= 0 else logits.shape[1] + position
        target_logits = logits[0, pos, :]

        for tok in range(num_tokens):
            grad = torch.autograd.grad(target_logits[tok], retrieved, retain_graph=(tok < num_tokens - 1))[0]
            G_sum[tok] += grad[0, pos].detach()

    return G_sum / len(passages)


def describe_units_via_jacobian(model: MemoryAsContextTransformer, G: torch.Tensor, U1_final: np.ndarray,
                                 units: list[int], k: int = 6, valid_bytes: np.ndarray | None = None):
    """Same units, same current output directions as describe_memory_units,
    but decoded through the measured Jacobian transport instead of a naive
    same-layer projection. Compare the two printouts directly -- agreement
    would suggest the naive lens was fine here; disagreement would mean the
    layers after the memory are meaningfully reshaping what a unit's
    contribution ends up promoting.

    `valid_bytes`: a boolean mask over the 256 byte ids restricting which
    ones can be reported. Necessary in practice -- gradient-based methods
    give wildly unstable, oversized gradients for tokens the model has
    almost no real experience with (verified: byte 0x00 never appears at
    all in 5M bytes of enwik8, and control bytes like 0xa2-0xa6 appear a
    few hundred times out of 5 million). Without this filter, every unit's
    top-k collapses onto the same handful of near-unseen bytes regardless
    of what the unit actually encodes -- a known general pitfall of
    gradient-based interpretability, not specific to this implementation."""
    G_np = G.cpu().numpy()
    for u in units:
        v = U1_final[u]
        scores = G_np @ v
        if valid_bytes is not None:
            scores = np.where(valid_bytes, scores, -np.inf)
        idx = np.argsort(-scores)[:k]
        bytes_str = " ".join(f"'{_printable_byte(int(b))}'" for b in idx)
        print(f"  unit {u:>4}  ->  {bytes_str}")


def common_byte_mask(data: torch.Tensor, min_count: int = 50) -> np.ndarray:
    """Which of the 256 byte ids actually occur often enough in real data
    to trust a gradient-based ranking involving them."""
    counts = np.bincount(data.numpy(), minlength=256)
    return counts >= min_count


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

    print("\nchecking WHY it forgets less: is the gate input-sensitive, or a fixed habit?")
    inspect_decay_gate(model, passage, device)

    print("\nchecking whether real text's writes reinforce each other more than random bytes' do...")
    random_passage = torch.randint(0, 256, passage.shape, device=device)
    report_write_alignment("real text", consecutive_write_alignment(model, passage, device))
    report_write_alignment("random bytes", consecutive_write_alignment(model, random_passage, device))

    print("\nlogit lens: what do a few memory units currently promote (end of passage)?")
    describe_memory_units(model, passage, device, units=list(range(8)))

    print("\njacobian-transported lens (arXiv:2607.15495-style): same units, properly")
    print("transported through the layers AFTER the memory instead of a naive shortcut...")
    mask = common_byte_mask(data_train)
    extra_passages = [sample_batch(data_val, 256, 1)[0] for _ in range(4)]
    G = compute_memory_jacobian(model, [passage] + extra_passages, device)
    with torch.no_grad():
        _, cache = model(passage.unsqueeze(0).to(device), return_cache=True)
        U1_final = cache[2][0].updates["model.weights.1"].detach()[0].cpu().numpy()[-1]
    describe_units_via_jacobian(model, G, U1_final, units=list(range(8)), valid_bytes=mask)


if __name__ == "__main__":
    main()

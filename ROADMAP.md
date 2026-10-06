# Roadmap

What comes next for MARV and its sibling packages. Updated 2026-10-06.
Released versions and their breaking changes are in `CHANGELOG.md`.

## Done

**v0.3.0:** History (`commit`, `log`, `gate`, `bisect`, `blame` with revert
proof, Trainer callback, Colab notebook).

**v0.2.0:**

Architecture adapters with refusal of unsupported models; exact tracing
through RMSNorm and LayerNorm; target-scoring, contextual-query and
evaluation fixes; held-out edit evaluation (`split_probes`); lineage check
for `diff`; vindex provenance; Colab test notebook. marv-audio (Whisper)
builds on it, pinned to `v0.2.0`.

## Next

**Order:** small trust items, then History (its revert and gate tools are what
the reranker study needs), then the WSDM reranker study.

### Small, trust-building items

- [x] Test that recording hooks leave the model bit-identical (output with
      hooks == output without).
- [x] Permanent Qwen3 exactness test.
- [x] Signed ranking option for `active_features` (positive / negative /
      both).
- [x] marv-audio E3: the matched-units audio split (real-clip split minus
      silence split). Predictions committed before running; cross-attention
      123%, MLPs −25%.

### History (`docs/history-design.md`)

Test models before shipping them; find what changed and why. Weights stay on
Hugging Face; History keeps test results. Colab-first: the results file is
downloaded or exported as `results.jsonl` and committed to GitHub.

1. [x] Results database, `commit`, `log`, JSONL export/import, download/upload.
2. [x] `bisect`, `blame` with the revert test.
3. [x] `gate` with paired statistics.
4. [x] Trainer integration and a Colab notebook finding a planted fact (v0.3.0).
5. [ ] Optional `marv[bench]`: standard benchmarks via lm-evaluation-harness.

### WSDM study: why did a reranker update change the ranking?

`Qwen3-Reranker-0.6B` scores relevance as logit("yes") − logit("no"), so
MARV explains each score exactly today. A public product-search fine-tune
(`codefactory4791/Qwen3-Reranker-HomeDepot`) is the same lineage.

- [ ] Base vs fine-tune on in-domain and general queries: ranking quality and
      per-query regressions.
- [ ] Explain each regression exactly (layers and neurons that moved the score).
- [ ] Revert the suspect neurons: does the regression go away while the
      fine-tune's gains stay?
- [ ] Would a gate have blocked the update?
- [ ] Needs only a small "revert these neurons" function, not full History.

Pass `"yes"`/`"no"` as token ids (no leading space), and use the model
card's exact prompt template. Check the submission deadline before starting.

### Ideas from LARQL (Apache-2.0; credit when adapting)

- [ ] Gemma adapter from declared family settings (norm offset, embedding
      scale, GELU-tanh, softcap), still refusing anything unverified.
- [ ] Edits as patch files that refuse to apply to the wrong base model and
      carry their measured collateral.
- [ ] Exact / approximate / refused label on every result; graded weight
      fidelity; checksums of the source weight files.
- [ ] Relation labels for features (word fired on → word promoted, clustered).

### marv-audio

- [ ] Neuron tools for Whisper: a LayerNorm logit lens and ungated-FFN support
      (with biases) in MARV core, then Whisper's `fc1`/`fc2` wiring in
      marv-audio.
- [ ] Automatic tests on every push (GitHub Actions) for both repos.

## Not now

GitHub Actions for models, LoRA diff, merge attribution, a local weight
store. Real models run in Colab, not on a laptop.

# Changelog

Versions are git tags (`v0.2.0`). Depend on a tag, not on `main`:

```
marv @ git+https://github.com/thebnbrkr/marv@v0.2.0
```

## Unreleased

### Added

- `marv.History` (`docs/history-design.md`): a results database for testing
  model versions. `commit` (never reruns a stored test), `log`, `gate` (paired
  exact McNemar on right->wrong flips, bootstrap interval on scores, warns when
  a test is too small), `bisect` (binary search over versions, reusing stored
  results), `blame` (weight diff as a suspect list, then revert tests:
  all-suspects ceiling, then halving to the smallest set that restores the
  result, reporting when neurons only matter jointly), JSONL export/import,
  Colab download/upload. Results record the weights' SHA-256 and the code
  versions that produced them.
- `revert_neurons`, `changed_neurons`; `ArchAdapter.neuron_params`.
- `active_features(sign=...)`: select by sign from the start.
- Tests: hooks leave the model bit-identical; Qwen3 decomposes exactly.
- `marv.__version__`, single-sourced in `marv/_version.py`.

## 0.2.0 (2026-10-06)

### Breaking

Numbers produced before 0.2.0, including those in the older notebooks
(Llama-3.2-1B, Qwen2.5-0.5B, AMD-135M, edit-eval), will differ if rerun.

- **Target scoring.** A target with a leading space (`" Paris"`) was scored as
  the bare-space token. Case and no-space variants now count only as single
  whole tokens (`"Paris"` no longer adds `" par"`, `"euro"` no longer adds
  `"e"`). Target probabilities that were inflated are lower, and fewer probes
  pass a "the model knows this" rank filter.
- **`hidden_states_at_layers` returns a different vector**: the FFN's real
  input (post-attention, normalised), not HF's post-FFN hidden state. This
  changes `describe_prompt`, `constellation`, the heatmaps and `toolcall`.
- **`constellation(model=...)` ranks by real activation** by default
  (`by="cosine"` for the old behaviour); `sim` then holds the signed
  activation.
- **`diff()` refuses unrelated checkpoints** (median gate cosine below
  `min_lineage=0.5`); pass `min_lineage=None` to override.
- **`detect_adapter` refuses models it cannot represent exactly**, including
  Gemma (Llama module names, different maths) and non-SiLU Llama configs.
- **`_verdict`**: the relative "degraded/improved" rule applies only when a
  probability is at least `floor=0.01`.
- **`mean_writes` skips position 0** by default (`skip_first=False` for the
  old behaviour), which changes `mean_ablate` and `load_bearing`.
- **`dead_features` is relative** to each layer's median peak (`rel=0.01`);
  `tol=` gives the old absolute cut-off.
- **`capital_edit_battery`** has 8 target rephrasings, not 4.
- **`extract_streaming`** reads `norm_eps` from `config.json` (was a fixed 1e-5).
- **`rank_by_ablation_effect`** lost its unused `restore_between` parameter.

### Added

- Architecture adapters own every module path; `register_adapter` lets other
  packages add architectures (marv-audio adds Whisper). Final norm may be
  RMSNorm or LayerNorm; a `prompt` may be a dict of model inputs.
- `split_probes`: choose an edit on some prompts, score it on others.
- `active_features`, `feature_activations_at_layers`: real signed activations
  and what each feature pushes up and down.
- `lineage_score`; `compare_scale` (check a null model's scale before
  trusting it).
- Vindex provenance: `revision`, `source_dtype`, `exact`; a tied `lm_head` is
  stored once.
- `notebooks/marv_tests_colab.ipynb`.

### Fixed

- `extract_streaming` reads bf16 checkpoints.
- README `trace_by_depth` example.

## 0.1.0

Initial version: vindex extraction, gate-KNN and logit lens, checkpoint diff,
edits and probe batteries, residual tracing and diagnostics ported from
marv-hyena.

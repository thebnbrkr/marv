# MARV History: design note (draft for review, 2026-10-06)

Status: **proposal**. No code exists yet. See `ROADMAP.md` for where it sits.

## What it is for

Test a model before you ship it, and understand what changed when a test
result moves. History records every version of a model you care about, the
tests run on each, and their per-question results. On top of that record it
answers: what changed (`log`), when (`bisect`), which neurons are
responsible (`blame`, proved by a revert test), and whether a new version
may replace the current one (`gate`).

## Colab-first: what lives where

Real models run on Colab, not on a laptop. Weights are never stored by
History: Hugging Face already versions them (a model repo is a git
repository, with commits, tags and deduplicated storage). History stores
only what nothing else records, the test results.

| What | Where | Kept? |
|---|---|---|
| Public model weights | Hugging Face, pinned by revision | re-downloaded each session |
| Your own fine-tune checkpoints | a private HF repo, one commit per save (`Trainer(push_to_hub=True, hub_strategy="every_save")`) | yes, by HF |
| Vindex | built on Colab from the weights each session | no: a cache, rebuilt identically from (revision, MARV version) |
| Test results | `.marv/history.sqlite` (created automatically; `/content/.marv/` on Colab) | yes: the only thing History must keep |

### A Colab session

```python
h = marv.History()                 # creates /content/.marv/history.sqlite if missing
# or continue a previous session:
h = marv.History.upload()          # pick a history.sqlite or results.jsonl from your computer
h = marv.History.from_jsonl(path)  # or from a file pulled from GitHub

h.commit(model, label="base", tests=[capitals, health])
...
h.download()                       # save history.sqlite to your computer
h.export_jsonl("results.jsonl")    # plain text, one line per result: commit this to GitHub
```

- `/content` is wiped when the Colab session ends: download or export
  before closing.
- **GitHub:** commit `results.jsonl`, not the SQLite file. A database is a
  binary blob in git (no readable diff, no merging); the JSONL is readable,
  diffable and mergeable, and History rebuilds its database from it.
- Pushing from Colab needs a GitHub token. Keep it in Colab's **Secrets**
  panel, never in the notebook text.
- SQLite runs in WAL mode, so a notebook can read while another process
  writes.

## What each version records

`model_id`, HF `revision` (or the commit of a pushed checkpoint and its
training step), SHA-256 of every source weight file (the HF API reports
them), dtype, exact/approximate, and the code that produced the numbers:
MARV version (tag/commit), torch, transformers, and for benchmarks the
lm-eval and task versions.

## Tables (sketch)

```
versions(id, model_id, revision, step, parent_id, dtype, exact, created_at,
         marv_version, torch_version, transformers_version)
source_files(version_id, filename, sha256)
tests(id, name, kind, spec_json, spec_hash)     kind: battery | benchmark | health | custom
runs(id, version_id, test_id, env_json, check_ok, started_at)
results(run_id, item_id, correct, score, target_prob)   one row per question
```

A run whose exactness check failed is stored with `check_ok = 0` and never
shown as a result.

## Commands

| Command | Does |
|---|---|
| `commit(model, label, tests)` | record the version, run any tests it has not seen, save per-question results |
| `log(test)` | one test's result across versions, oldest first |
| `bisect(test, good, bad)` | binary search over HF revisions for where a result changed (~7 runs for 100 checkpoints) |
| `blame(test, before, after)` | diff the two versions (the **suspect list**), then put old neurons back and check the result returns. Halving the suspect set (group testing) needs ~log₂(n) runs instead of n |
| `gate(candidate, current, tests)` | run the same tests on both, compare with paired statistics, fail on a significant regression |

Old versions are never retested: their results are already stored.

`bisect` and `blame` read only the layers they need. A `.safetensors`
file's header lists every tensor's byte range, so one layer can be fetched
with a range request instead of a full download.

## Comparing fairly

- Same questions on both versions; per-question results are paired.
- Right/wrong items: McNemar's test on the flips (right→wrong vs wrong→right).
  Continuous scores: paired bootstrap. Report the effect with an interval.
- Warn when a test set is too small to detect the change asked about.
- Compare two versions in the same environment (GPU runs are not bit-repeatable).

## Rules carried over from MARV

- No numbers from a run whose check failed.
- A diff is a suspect list, not an explanation. Only the revert test
  (total effect) names a cause.
- Label direct vs total.
- **Group testing assumes neurons act roughly independently.** If halving
  stops converging (both halves restore part of the effect), say so and fall
  back to testing the survivors one by one. A neuron that matters only
  jointly with others is a real outcome to report.

## Speed

The slow part is running tests, not reading weights.

- `run_battery` scores one prompt at a time. Batch it, with **left padding**
  so "the last token" is the same position in every row.
- Run tests on a Colab GPU; keep exactness checks on CPU.
- Diffs stream one layer at a time.
- Bisect and group testing keep the number of runs logarithmic.

## Tests (tiny fake model, no network)

- Export to JSONL and rebuild: identical database contents.
- Plant a change at step k of 16 fake checkpoints: bisect finds k, blame
  names the planted neurons, the revert test restores the result, gate fails
  the bad version and passes a good one.
- A run with a failed check never appears in `log`.

## Build order

1. Results database, `commit`, `log`, JSONL export/import, download/upload.
2. `bisect`, `blame` with the revert test.
3. `gate` with paired statistics.
4. `Trainer` integration (push every save, commit results) and a Colab
   notebook that fine-tunes a small model on a made-up fact and shows History
   finding it.
5. Optional `marv[bench]`: standard benchmarks through lm-evaluation-harness
   as another kind of test (per-question results kept, versions recorded,
   MARV's edit hooks active during runs). Small models get HellaSwag,
   ARC-Easy, WikiText perplexity; MMLU is at chance below ~1B. Benchmarks are
   downloaded at run time, never committed; check each dataset's licence.

## Not now

GitHub Actions for models, LoRA diff, merge attribution. A local weight
store (hash-named `.safetensors` objects) only if offline use is ever needed.

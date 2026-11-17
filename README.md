# TurboVLA LIBERO latency reproduction on RTX 4090

Measurements collected on **2026-11-17** with the **released LIBERO checkpoint**, upstream commit [`fe9f5eb94ed6b16bb2f0228533f64864c1673a05`](https://github.com/H-EmbodVis/TurboVLA/tree/fe9f5eb94ed6b16bb2f0228533f64864c1673a05). This is an independent reproduction report, not an official TurboVLA benchmark release.

Prepared in response to [TurboVLA issue #30](https://github.com/H-EmbodVis/TurboVLA/issues/30), with thanks to the authors for releasing `benchmark_latency.py` and clarifying its scope.

## Main results

All numbers below are **milliseconds per 12-action chunk, batch size 1**, not per action or simulator step.

| Experiment | Input | Median ms | p95 ms | Samples |
|---|---|---:|---:|---:|
| Official script, no CUDA Graph | Fixed synthetic GPU tensors | **11.397** | See block results | 3 blocks × 50 |
| Original real-observation replay: model only | 566 decisions from 40 episodes | 11.420 | 12.619 | 1,698 |
| Original replay: model + stats-BF16 GPU postprocessing | Same observations | **11.473** | 12.640 | 1,698 |
| Original replay: CPU preparation through CPU actions | Same observations | **12.852** | 15.211 | 1,698 |
| Separate, profiler-free paired experiment: CUDA Graph OFF | Same observations | **11.492** | 12.292 | 1,698 |
| Same paired experiment: CUDA Graph ON | Same observations | **4.849** | 4.880 | 1,698 |

The synthetic number is the **median of three block medians**; real-observation numbers pool individual calls. These are different aggregation rules. Full precision and per-task/per-suite results are retained.

**The difference from the paper's 31.2 ms is unresolved.** We do not claim that CUDA Graph explains that difference: our ordinary approximately 11.5 ms run already has CUDA Graph OFF. Graph mode is a separate optional optimization experiment with a small capture-specific text-wrapper adaptation.

## Contents

- [PROTOCOL.md](PROTOCOL.md): timing boundaries, hardware, preprocessing, samples, aggregation, and validation.
- [RESULTS.md](RESULTS.md): detailed comparisons, cache/pacing diagnostics, CUDA Graph verification, limitations.
- [EPISODES.md](EPISODES.md), [episodes.csv](episodes.csv), [episode-identities.json](episode-identities.json): all 40 instructions, task IDs, initial states, hashes, decision counts, and success flags.
- [summary.json](summary.json): the authors' official synthetic benchmark output (machine-specific identifiers redacted).
- [replay-aggregate.json](replay-aggregate.json) and [graph-summary.json](graph-summary.json): full-precision results and checks.
- `raw-evidence.zip`: original timing CSVs, summaries, manifests, runtime checks and GPU monitoring evidence. No timing outliers were removed.
- `observations-01.zip` through `observations-05.zip`: **all 566 actual observation NPZs**, with [observation-hashes.json](observation-hashes.json). Extract all into one directory.
- `episode-metadata.zip`: per-task capture/result metadata and available original command records. Extract alongside observations.
- Python scripts in the repository root: collection, replay, cache diagnostics, CUDA Graph experiments, and offline verification.
- [REPRODUCE.md](REPRODUCE.md): setup and commands; read before executing GPU experiments.
- [provenance.json](provenance.json), [publication.patch](publication.patch), `SHA256SUMS`: original/published content hashes and publication edits.

## Scope and provenance

Checkpoint SHA-256: `d031ad7be05a2f5d04afb3194ed26b0cb46083685edee7a5e145078a37d26bab`.

All timing results use the public checkpoint, **not our separately trained English model**. The input collection contains one policy-driven episode per task, seed 7 and benchmark initial-state index 0. This is not a repeat of the 2,000-trial success benchmark, an independent test of generalization, or an end-to-end robot control benchmark.

Upstream implementation and weights remain subject to their original licenses. This repository does not redistribute model weights. Observation arrays were generated in LIBERO; consult [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) and its asset licenses for reuse conditions. No new license is asserted over upstream material.

Publication preparation replaced local server paths and the machine-specific GPU UUID. The runnable timing scripts differ only in reading the selected UUID from `CUDA_VISIBLE_DEVICES`; their measurement/model logic is unchanged. The original scripts were executed on GPU; the publication edits were syntax-checked and the packaged evidence verified offline, not used to generate a new claimed GPU measurement. See provenance for exact hashes.

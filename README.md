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

## Repository layout

```text
README.md                    # overview and main results
SHA256SUMS                   # integrity manifest for published files
docs/                        # protocol, results, episode table, reproduction guide
scripts/                     # collection, benchmarks, diagnostics, offline verifier
results/                     # official summary, aggregate results, raw timing archive
data/                        # episode metadata, identities and observation hashes
  observations/              # five archives containing all 566 observation arrays
provenance/                  # measured/published hashes and publication patch
```

| Start here | Contents |
|---|---|
| [Detailed protocol](docs/PROTOCOL.md) | Hardware, timing boundaries, sampling, aggregation and validation |
| [Results and limitations](docs/RESULTS.md) | Timing comparisons, cache/pacing diagnostics and CUDA Graph checks |
| [Exact episode inventory](docs/EPISODES.md) | All 40 tasks, instructions and decision counts |
| [Reproduction guide](docs/REPRODUCE.md) | Environment setup, extraction and execution commands |
| [Official summary.json](results/summary.json) | Authors' synthetic benchmark output |
| [Real-observation results](results/replay-aggregate.json) / [Graph results](results/graph-summary.json) | Full-precision aggregates and validation |
| [Raw timing evidence](results/raw-evidence.zip) | CSVs, manifests, runtime checks and monitoring |
| [Episode identities](data/episode-identities.json) / [CSV](data/episodes.csv) | Initial-state hashes, instructions, IDs and success flags |
| [Observation archives](data/observations/) / [Metadata](data/episode-metadata.zip) | Original measured inputs; extract all archives into one directory |
| [Scripts](scripts/) / [Offline verifier](scripts/verify_artifacts.py) | Collection, timing and integrity checks |
| [Provenance](provenance/provenance.json) / [Publication patch](provenance/publication.patch) | Original and published content hashes and publication edits |

From the repository root, run `python scripts/verify_artifacts.py` to verify the packaged evidence without a GPU.

## Scope and provenance

Checkpoint SHA-256: `d031ad7be05a2f5d04afb3194ed26b0cb46083685edee7a5e145078a37d26bab`.

All timing results use the public checkpoint, **not our separately trained English model**. The input collection contains one policy-driven episode per task, seed 7 and benchmark initial-state index 0. This is not a repeat of the 2,000-trial success benchmark, an independent test of generalization, or an end-to-end robot control benchmark.

Upstream implementation and weights remain subject to their original licenses. This repository does not redistribute model weights. Observation arrays were generated in LIBERO; consult [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) and its asset licenses for reuse conditions. No new license is asserted over upstream material.

Publication preparation replaced local server paths and the machine-specific GPU UUID. The runnable timing scripts differ only in reading the selected UUID from `CUDA_VISIBLE_DEVICES`; their measurement/model logic is unchanged. The original scripts were executed on GPU; the publication edits were syntax-checked and the packaged evidence verified offline, not used to generate a new claimed GPU measurement. See provenance for exact hashes.

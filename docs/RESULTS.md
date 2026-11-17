# Results and interpretation

## Original real-observation replay

| Scope | n | Median ms | p95 ms |
|---|---:|---:|---:|
| A: model only | 1,698 | 11.419667 | 12.619065 |
| B: stats-BF16 GPU postprocessing | 1,698 | 11.473488 | 12.639720 |
| C: public-policy FP32 GPU postprocessing | 1,698 | 11.478259 | 12.508575 |
| E: suite-statistics FP32 GPU postprocessing | 1,698 | 11.475392 | 12.569038 |
| D: CPU preparation through CPU actions | 1,698 | 12.851673 | 15.211437 |

B's equal-task mean of task medians is 11.464894 ms. Per-task and per-suite values are in `replay-aggregate.json`. The first observation of each task was checked against the collector's recorded canonical actions and CPU/GPU suite postprocessing (40/40 passed). This original replay check is not described as checking all 566 recorded action chunks; the separate graph equality check does cover all 566 inputs.

Small differences between these separate medians are descriptive, not a rigorous component microbenchmark. In particular, C changes both dtype/sanitization and normalization constants.

## Synthetic comparison and unresolved paper discrepancy

The official script gives **11.396594 ms** (median of block medians) on fixed synthetic input. Original real-observation B gives **11.473488 ms** (pooled calls). Their proximity is an observation, not proof that all tasks, hardware or workloads have identical latency.

Older results of 11.265 ms synthetic and 11.375 ms on one fixed real initial scene were separate historical runs. Neither represents the 40-episode distribution. An earlier approximately 17 ms run on a different host/thread setup is not a controlled causal comparison.

We have not reproduced the reported paper value of 31.2 ms. We request comparison of the accompanying official-script `summary.json`, environment and precise timing boundary. We do not infer a measurement error by the authors or identify a single cause from these results.

## Cache and pacing diagnostics

Changing observations or starting a fresh process does not remove every hardware/runtime cache. Fresh processes separate process-local state but do not constitute a cache-free hardware protocol. Default allocator reuse is part of valid warmed steady-state results here.

- Runtime hooks over 20 forwards (10 repeated and 10 changed inputs) counted tokenization 20, BERT 20, vision 40, each of six fusion layers 20, and ACT 20. No feature reuse was found in these inspected paths.
- After warm-up, **566 distinct observations, each measured once in shuffled order**, gave median **11.550398 ms**, p95 **12.146443 ms**. A single fixed observation is not needed to obtain the approximately 11.5 ms result. This does not eliminate allocator/hardware caching.
- Cleaner pacing experiment (120 calls/condition, monitoring once per second): no gap **11.371528 / 11.809081 ms** median/p95; 100 ms gap **13.416349 / 18.269557**; 500 ms gap **13.705949 / 16.463275**. Reported median SM clock was 2520 MHz before/after in each condition. The cause of the gap effect remains unresolved; CPU wake-up/scheduling/frequency effects were not isolated.
- An earlier exploratory comparison invoked `nvidia-smi` between calls, confounding cache conditions with inter-call delay. Its 15–16 ms fresh-input/empty-cache/memory-sweep/thread-count results are retained in the archive but **must not be used to isolate cache causality**.
- First forward after model load: **186.281726 ms, n=1**. Model loading/context initialization are excluded; this is not a reliable cold-start distribution.
- Independent allocator-enabled run: **11.395094 / 11.533542 ms**, n=120. `PYTORCH_NO_CUDA_MEMORY_CACHING=1` failed with CUDA illegal memory access/SIGABRT. **There is no valid allocator-disabled latency result.** Recovery with normal allocation gave **11.647181 / 11.738513 ms**, n=120.

## CUDA Graph: verified separate optimization

| Profiler-free experiment | OFF median / p95 ms | ON median / p95 ms | n per mode |
|---|---:|---:|---:|
| Paired, alternating order | 11.491932 / 12.291898 | 4.848868 / 4.879918 | 1,698 |
| Unmixed blocks | 11.502806 / 12.228306 | 4.840700 / 4.870682 | 1,698 |

Paired median ratio: **2.370×**, a **57.81%** reduction. This is a request producing an action chunk, with online text work and static-buffer updates included, but CPU image/state preparation excluded.

Separate profiler verification for three requests: OFF `cudaGraphLaunch=0`, `cudaLaunchKernel=4086`; ON `cudaGraphLaunch=3`, `cudaLaunchKernel=102`. GPU kernel events were 4086 vs 4083; the small difference comes with the capture-wrapper adaptation. Reduced CPU launch count is not evidence that BERT/vision computation was skipped. All 566 eager/graph output comparisons were bitwise equal under the exact-zero tolerance checks.

Graph capture itself cost median **92.193 ms per task**, excluding preceding model load/warm-ups. A replay-only fixed-buffer lower bound was approximately **4.431 ms**, n=2,000, but excludes online tokenization/mask preparation/input updates and is **not the headline request latency**.

A preliminary run with profiling in the measurement process gave OFF 15.026 ms / ON 5.004 ms. The final experiment was rerun in a separate profiler-free process; the preliminary data remain labeled in the archive. The exact mechanism of that process difference was not isolated.

## Limits

One GPU machine/software configuration, one collection seed and one initial state per task; repeated measurements are correlated. This does not evaluate generalization or reproduce 2,000 success trials. GPU/CPU clocks were not locked, CPU affinity was not fixed, and observer overhead can affect timing. Graph capture is per instruction/layout and its capture/start-up cost is excluded from steady-state results. No claim is made about arbitrary instruction switching, all possible input shapes, physical robot end-to-end control, or allocator-disabled performance.

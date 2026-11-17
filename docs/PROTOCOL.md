# Detailed measurement protocol

## Hardware, software and checkpoint

Timing ran on a dedicated RTX 4090 selected by UUID on a machine also containing an RTX 3090. CPU: AMD Ryzen 9 7950X3D. Python 3.9.25, PyTorch 2.3.1+cu121, CUDA runtime 12.1, Transformers 4.56.0, tokenizers 0.22.2, timm 1.0.25, NumPy 1.26.4. The official `summary.json` includes driver/GPU snapshots, package versions, thread settings and model configuration.

CPU intra-op/OMP/MKL threads: 1 for primary timing. TF32 and cuDNN benchmarking disabled in the custom replay scripts. BF16, inference mode, batch 1, two 256×256 image views, state dimension 8, chunk length 12, action dimension 7, six fusion layers. No `torch.compile` or text/vision embedding cache. Default allocator caching remains enabled for valid primary runs. Model initialization and warm-up are excluded.

TurboVLA revision: `fe9f5eb94ed6b16bb2f0228533f64864c1673a05`. LIBERO revision: `8f1084e3132a39270c3a13ebe37270a43ece2a01`. The working tree had unrelated training-file edits; inference/model file hashes are recorded in the manifests and were compared between timing runs. Do not describe this as an entirely clean working tree.

Released checkpoint: `checkpoints/libero/turbovla_libero.pth` under the downloaded release assets; SHA-256 `d031ad7be05a2f5d04afb3194ed26b0cb46083685edee7a5e145078a37d26bab`. Official benchmark script SHA-256 `4db1a98cf3680c3d36f0adeaf182e93af0c3ee79012d436123ac75948b453dcd`.

## Actual episode inputs

LIBERO-Spatial, Object, Goal and Long (`libero_10`): ten tasks per suite, task order index 0, **one episode per task**, seed 7, initial-state list index 0. Ten settling dummy steps; 256-pixel camera observations; action chunk and open-loop execution length both 12. Upstream timeouts are Spatial 220, Object 280, Goal 300 and Long 520 action/environment steps, plus ten settling steps; standard success predicates are used; collection is through the upstream `vla_adapter.rollout` and `TurboVLAPolicy.predict_env_action_chunk`, without changing returned actions. Exact commands accompany the metadata where available; the portable collection launcher constructs the same settings.

The collector runs the released suite policy and stores the actual observation at every action-chunk decision, including later frames as the robot moves. Collection happened on a different RTX 4090 host from timing. Collection was not timed. Timing replays these saved inputs on an otherwise unoccupied selected GPU; this separates simulator cost from inference cost.

Each NPZ contains `primary`, `wrist` (officially rotated images), actual `state`, `eef_pos`, `eef_quat`, `gripper_qpos`, exact `instruction`, canonical `actions`, `environment_step = 10 + 12 * policy_call`, and `policy_call`. There are 566 decisions: Spatial 88, Object 128, Goal 109, Long 241. All observations, including those from two unsuccessful episodes, are retained. No random synthetic observation is labeled an episode input.

[EPISODES.md](EPISODES.md) lists every task/instruction. [episode-identities.json](../data/episode-identities.json) additionally records the task BDDL, initial-state asset hash, selected row index, dtype, shape and SHA-256 of its contiguous array bytes. State identity hashes were added during publication by inspecting the same pinned LIBERO assets; they were not logged during the original collector run. NPZ hashes directly identify the original measured arrays.

## Five timing boundaries

The original custom replay tests each saved decision three times per scope after ten warm-up calls per task:

| Scope | Start → end | Included / excluded |
|---|---|---|
| A_forward | GPU-ready images/state → normalized GPU action chunk | Online instruction tokenization, token transfer, BERT, vision, fusion and ACT; no action postprocessing |
| B_stats_bf16 | GPU-ready images/state → GPU actions | A plus mixed-suite statistics BF16 unnormalization and gripper conversion; closest to official script boundary |
| C_public_fp32 | GPU-ready images/state → GPU actions | A plus FP32 conversion, sanitization and public-policy constants; constants differ from B, so this is not a pure dtype ablation |
| E_suite_fp32 | GPU-ready images/state → GPU actions | A plus FP32 sanitization/conversion using suite statistics |
| D_cpu_to_cpu | CPU observation preparation → CPU environment actions | Image rotation, state construction, image/state preparation and H2D, full model and D2H/suite action conversion |

A/B/C/E exclude image/state CPU preprocessing and H2D. All include online text handling. D excludes actual camera acquisition, disk IO, simulator `env.step`, network and robot communication. D begins from already-loaded arrays and reconstructs raw orientation from stored rotated arrays inside `cpu()` before applying the standard rotation. Do not call D full robot end-to-end latency.

Wall-clock timing uses `time.perf_counter()` with `torch.cuda.synchronize()` before starting and before stopping. Therefore CPU text work is included; this is not CUDA-event-only kernel timing. Outputs are 12×7 action chunks, not a single control action.

Primary summaries pool calls (1,698 per scope). Longer episodes contribute more observations. `replay-aggregate.json` also provides per-task/per-suite medians/p95s and equal-task mean/median of task medians. Equal-task aggregates and pooled distributions are explicitly distinct. p95 is NumPy's default percentile interpolation. Repeated timing calls are not independent episodes; no confidence interval over independent episodes is claimed.

## Synthetic control

Run the authors' unmodified `scripts/libero/benchmark_latency.py` with the released checkpoint, CPU threads 1 and three blocks; each block uses ten warm-ups and 50 measurements. Inputs are fixed synthetic GPU tensors, not real episodes. Full configuration and validation are in `summary.json`; raw 150 timings are in `raw-evidence.zip/synthetic-control/raw_latency.csv`. The 11.396594 ms headline is median of block medians.

## CUDA Graph experiment

A separate task-specific graph captures BERT, DINOv3, fusion, ACT and BF16 GPU postprocessing. Tokenization and dynamic special-token masks are rebuilt **online on every request** outside the capture, then copied with fresh image/state tensors into static GPU buffers before replay. These operations are included in graph-request timing. There is no embedding cache. Graph creation and its warm-up are excluded and reported separately.

The existing text wrapper's CPU-index-based assignment was not capturable. During capture only, the batch-1 wrapper uses fixed slices to assemble the same padded text features/masks. Eager inference retains the original wrapper. This is a small explicit code adaptation, not merely toggling a framework flag. Each task has a fixed instruction and token layout; changed instruction/layout requires compatible recapture and is not benchmarked here.

Ten eager warm-ups, ten capture-body warm-ups on a side stream, graph capture, and ten graph-request warm-ups precede measurements. Clean-run validation checks **every one of the 566 observations**, comparing eager and graph GPU actions with `rtol=0, atol=0`; all are identical. Three blocks alternate OFF/ON order by observation/block. Additional unmixed blocks measure each mode separately, three passes each. Primary process uses no profiler. Runtime traces were obtained in a separate process, and the clean script imports their counts only after measurement.

GPU occupancy is checked at start/end and sampled during primary replay/graph execution. No foreign process was detected on the selected GPU. Sampling cannot prove that no sub-second activity ever occurred; no global clock locking or CPU affinity claim is made.

# Reproduce or verify

## Verify the published evidence without a GPU

Clone/download this repository. With Python and NumPy installed:

```sh
python verify_artifacts.py
```

This checks all published-file hashes, all 566 NPZ hashes, episode counts, and recomputes the primary eager/graph and five-scope medians/p95s directly from the raw files. It does not run the model or establish fresh performance on another machine.

## Prepare inference environment

Use the official [TurboVLA setup](https://github.com/H-EmbodVis/TurboVLA/tree/fe9f5eb94ed6b16bb2f0228533f64864c1673a05) and LIBERO renderer setup, pinned revisions from PROTOCOL.md and the recorded package versions in summary.json. Obtain public weights from the official project; verify the checkpoint hash. No weights are included here. Environment setup, gated DINO assets and graphics libraries must already work before these scripts run.

The scripts expect this layout (or symlinks to your assets):

```text
PROJECT/
  LIBERO/                      # pinned LIBERO checkout
  pretrained/dinov3-vitb/       # downloaded DINOv3 assets
  TurboVLA/                    # pinned upstream checkout
    pretrained/bert-base-uncased/
    pretrained/release/checkpoints/libero/turbovla_libero.pth
    reproduction/              # copy this repository's .py scripts here
```

Install upstream `third_party/vla_adapter` as required by its setup. Configure `LIBERO_CONFIG_PATH` for your own asset locations and the renderer (the original collection used OSMesa). Set `PYTHONPATH` to include the TurboVLA and LIBERO roots, `PYTHONNOUSERSITE=1`, `USE_TORCH=1`, `USE_TF=0`, `USE_FLAX=0`. Run all following commands from the upstream TurboVLA root in the prepared environment.

Select the RTX 4090 UUID from `nvidia-smi -L` and export it:

```sh
export CUDA_VISIBLE_DEVICES=GPU-YOUR-RTX4090-UUID
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
```

The scripts assert an RTX 4090 and reject detected foreign work on the selected GPU. They must run separately, not concurrently. Use new output directories for independent runs; do not reuse a partial output without inspecting it. For long remote runs, use tmux or another durable process wrapper.

## Extract the exact measured observations

```sh
mkdir -p inputs-release
for archive in /path/to/turbovla-latency-reproduction/observations-*.zip; do
  unzip -n "$archive" -d inputs-release
done
unzip -n /path/to/turbovla-latency-reproduction/episode-metadata.zip -d inputs-release
```

`inputs-release/inputs` is the `--inputs` directory below. Its task folders must contain both NPZ files and capture/result JSONs. The original 566 arrays are included; regenerating trajectories is optional and may not produce byte-identical arrays across different software/hardware.

## Official synthetic control

```sh
python scripts/libero/benchmark_latency.py \
  --checkpoint pretrained/release/checkpoints/libero/turbovla_libero.pth \
  --dinov3-path ../pretrained/dinov3-vitb \
  --bert-path pretrained/bert-base-uncased \
  --cpu-threads 1 --blocks 3 --output-dir outputs/publication/synthetic
```

## Real-observation replay

```sh
python reproduction/replay_latency.py --inputs inputs-release/inputs --output outputs/publication/replay
```

`--smoke` limits the replay/graph scripts to one task. The script writes per-task raw CSVs/completion checks and a manifest. `replay-aggregate.json` is the original full aggregation; the independent verifier demonstrates its raw-data calculation.

## CUDA Graph verification and clean timing

The clean script deliberately reads runtime traces from a separately profiled run at this fixed upstream-relative location. Generate those traces first, then run clean timing in a fresh process:

```sh
python reproduction/benchmark_cuda_graph.py --inputs inputs-release/inputs --output outputs/evaluation/cuda-graph-full-20261117
python reproduction/benchmark_cuda_graph_clean.py --inputs inputs-release/inputs --output outputs/publication/graph-clean
```

The clean script's `if False` profiler blocks are intentional. Runtime proof comes from the separate process; measurement itself does not enable profiling. Do not merge the processes or present the preliminary profiled timing as the clean result.

## Optional cache/pacing diagnostics

```sh
python reproduction/audit_cache_latency.py --inputs inputs-release/inputs --output outputs/publication/cache
python reproduction/audit_pacing.py --inputs inputs-release/inputs --output outputs/publication/pacing
python reproduction/audit_allocator.py --inputs inputs-release/inputs --output outputs/publication/allocator-normal
```

Read RESULTS.md before interpreting these diagnostics. The exploratory cache script has a known pacing confound. We do not recommend treating `empty_cache()` as a complete cache purge. Allocator-disabled execution failed on our stack; no valid disabled-allocator result is supplied.

## Optional regeneration of the 40 episodes

```sh
python reproduction/collect_all.py --output outputs/publication/episodes
```

`collect_episode_inputs.py` is the executed collector. `collect_all.py` is a new portable orchestration helper constructing the original settings, syntax-checked but not used to generate the reported numbers. The original orchestration used a local command template; per-task command metadata and PROTOCOL.md preserve the settings without requiring that private template.

## Publication changes

Machine paths/UUIDs are replaced in evidence text; numerical values, raw timings and observation NPZ bytes are unchanged. In the seven measured scripts, only the hardcoded UUID becomes `os.environ['CUDA_VISIBLE_DEVICES']`; see publication.patch (its removed UUID is itself redacted). `provenance.json` lists original and published hashes. Setup must be adapted to your machine; these scripts are research artifacts, not a standalone dependency installer. No new GPU result is claimed for the publication packaging edits.

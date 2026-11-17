"""Publication launcher equivalent to recorded collection commands (new, not GPU-run)."""
import argparse, os, subprocess, sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[1]
for suite in ['libero_spatial','libero_object','libero_goal','libero_10']:
 for task in range(10):
  dest=a.output.resolve()/f'{suite}-task{task:02}';dest.mkdir(parents=True,exist_ok=True)
  if (dest/'capture.json').exists() and (dest/'result.json').exists():continue
  if list(dest.glob('call*.npz')):raise RuntimeError(f'Inspect incomplete task before restarting: {dest}')
  cmd=[sys.executable,str(root/'reproduction/collect_episode_inputs.py'),'--ckpt_path',str(root/'pretrained/release/checkpoints/libero/turbovla_libero.pth'),'--libero_root',str(root.parent/'LIBERO'),'--dinov3_path',str(root.parent/'pretrained/dinov3-vitb'),'--bert_path',str(root/'pretrained/bert-base-uncased'),'--stats_path',str(root/'experiments/libero/configs/libero_all4_stats.json'),'--stats_key','libero_all4_no_noops','--task_suite_name',suite,'--task_ids',str(task),'--num_trials_per_task','1','--num_steps_wait','10','--env_img_res','256','--chunk_size','12','--num_open_loop_steps','12','--seed','7','--precision','bf16','--save_video','false','--result_json_path',str(dest/'result.json')]
  subprocess.run(cmd,cwd=root,env=dict(os.environ,CAPTURE_DIR=str(dest)),check=True)

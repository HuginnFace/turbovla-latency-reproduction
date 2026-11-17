"""Paired replay timing on real LIBERO policy-decision observations."""
import os
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',USE_TORCH='1',USE_TF='0',USE_FLAX='0',TOKENIZERS_PARALLELISM='false')
import argparse,csv,json,time,hashlib,subprocess,threading,fcntl,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from turbovla.evaluation.suite_policy import TurboVLAPolicy
from turbovla.evaluation.policy import ACTION_MIN,ACTION_MAX,normalized_action_to_env_action,sanitize_pred_chunk,rotate_libero_image,state_from_libero_obs
p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true');args=p.parse_args()
args.output.mkdir(parents=True,exist_ok=True)
lock=(args.output/'runner.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
(args.output/'runner.pid').write_text(str(os.getpid()))
torch.set_num_threads(1);torch.manual_seed(7);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False
uuid=os.environ['CUDA_VISIBLE_DEVICES']
assert os.environ.get('CUDA_VISIBLE_DEVICES')==uuid
assert '4090' in torch.cuda.get_device_name(0)
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
def occupancy():
 s=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
 return [int(row.split(',')[1]) for row in s.splitlines() if row.split(',')[0].strip()==uuid and int(row.split(',')[1])!=os.getpid()]
assert not occupancy(),'selected GPU busy'
ckpt=ROOT/'pretrained/release/checkpoints/libero/turbovla_libero.pth'
policy=TurboVLAPolicy(ckpt_path=str(ckpt),dinov3_path=str(ROOT.parent/'pretrained/dinov3-vitb'),bert_path=str(ROOT/'pretrained/bert-base-uncased'),stats_path=str(ROOT/'experiments/libero/configs/libero_all4_stats.json'),stats_key='libero_all4_no_noops',precision='bf16',device='cuda:0')
lowbf=torch.tensor(policy.action_min,device='cuda',dtype=torch.bfloat16);highbf=torch.tensor(policy.action_max,device='cuda',dtype=torch.bfloat16)
low=torch.tensor(policy.action_min,device='cuda');high=torch.tensor(policy.action_max,device='cuda');publiclow=torch.tensor(ACTION_MIN,device='cuda');publichigh=torch.tensor(ACTION_MAX,device='cuda')
def post(raw,lo,hi,fp32):
 if fp32:raw=torch.nan_to_num(raw.float(),nan=0.,posinf=1.,neginf=-1.)
 arm=0.5*(raw[...,:6]+1.);arm=arm*(hi[:6]-lo[:6])+lo[:6]
 grip=raw[...,6:7];grip=grip.new_ones(grip.shape).masked_fill(grip<0,-1.)
 return torch.cat((arm,grip),dim=-1)

files=[sorted(p.parent.glob('call*.npz'))[0] for p in sorted(args.inputs.glob('*/capture.json'))]
data=[dict(np.load(f,allow_pickle=False)) for f in files];rows=[];rng=np.random.default_rng(19)
with torch.inference_mode():
 d=data[0];images,states=policy._prepare_model_inputs(*policy._build_batch([d['primary']],[d['wrist']],[d['state']]))
 for _ in range(10):y=post(policy.model([str(d['instruction'])],images,states),lowbf,highbf,False);del y
 for block in range(3):
  assert not occupancy()
  for i in rng.permutation(40):
   d=data[int(i)];images,states=policy._prepare_model_inputs(*policy._build_batch([d['primary']],[d['wrist']],[d['state']]))
   torch.cuda.synchronize();start=time.perf_counter();y=post(policy.model([str(d['instruction'])],images,states),lowbf,highbf,False);torch.cuda.synchronize();ms=(time.perf_counter()-start)*1000;del y
   rows.append({'block':block,'task':int(i),'ms':ms})
 assert not occupancy()
v=[r['ms'] for r in rows];summary={'n':len(v),'median_ms':float(np.median(v)),'p95_ms':float(np.percentile(v,95)),'PYTORCH_NO_CUDA_MEMORY_CACHING':os.environ.get('PYTORCH_NO_CUDA_MEMORY_CACHING'),'raw':rows,'script_sha256':sha(Path(__file__)),'scope':'model+stats BF16 postprocessing; changed observation on each call;10warmup;inputprep excluded;3blocks;CPUthreads1'}
(args.output/'summary.json').write_text(json.dumps(summary,indent=2));print({k:v for k,v in summary.items() if k!='raw'})

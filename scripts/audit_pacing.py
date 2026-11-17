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

import ctypes
lib=ctypes.CDLL('libnvidia-ml.so.1');assert lib.nvmlInit_v2()==0
handle=ctypes.c_void_p();assert lib.nvmlDeviceGetHandleByUUID(uuid.encode(),ctypes.byref(handle))==0
def clock():
 value=ctypes.c_uint();assert lib.nvmlDeviceGetClockInfo(handle,ctypes.c_uint(1),ctypes.byref(value))==0;return value.value
files=[sorted(p.parent.glob('call*.npz'))[0] for p in sorted(args.inputs.glob('*/capture.json'))];data=[dict(np.load(f,allow_pickle=False)) for f in files];assert len(data)==40
stop=threading.Event();errors=[]
def monitor():
 with (args.output/'monitor.jsonl').open('w') as f:
  while not stop.is_set():
   others=occupancy()
   if others:errors.append(others)
   f.write(json.dumps({'time':time.time(),'sm_clock':clock(),'foreign':others})+'\n');f.flush();stop.wait(1)
thread=threading.Thread(target=monitor,daemon=True);thread.start()
rows=[];rng=np.random.default_rng(17)
with torch.inference_mode():
 for block in range(3):
  for gap in [0.,.1,.5][block:]+[0.,.1,.5][:block]:
   for _ in range(10):
    d=data[0];images,states=policy._prepare_model_inputs(*policy._build_batch([d['primary']],[d['wrist']],[d['state']]));y=post(policy.model([str(d['instruction'])],images,states),lowbf,highbf,False);del y
   for index in rng.permutation(40):
    d=data[int(index)];images,states=policy._prepare_model_inputs(*policy._build_batch([d['primary']],[d['wrist']],[d['state']]))
    torch.cuda.synchronize()
    if gap:time.sleep(gap)
    before=clock();start=time.perf_counter();y=post(policy.model([str(d['instruction'])],images,states),lowbf,highbf,False);torch.cuda.synchronize();ms=(time.perf_counter()-start)*1000;after=clock();del y
    rows.append({'gap_s':gap,'block':block,'task':int(index),'ms':ms,'sm_clock_before':before,'sm_clock_after':after})
   (args.output/'progress.json').write_text(json.dumps({'rows':len(rows),'block':block,'gap':gap}))
stop.set();thread.join();assert not errors and not occupancy()
with (args.output/'raw.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary={}
for gap in [0.,.1,.5]:
 rr=[r for r in rows if r['gap_s']==gap];v=[r['ms'] for r in rr];summary[str(gap)]={'n':len(v),'median_ms':float(np.median(v)),'p95_ms':float(np.percentile(v,95)),'sm_clock_before_median':float(np.median([r['sm_clock_before'] for r in rr])),'sm_clock_after_median':float(np.median([r['sm_clock_after'] for r in rr]))}
(args.output/'summary.json').write_text(json.dumps({'status':'completed','gaps':summary,'script_sha256':sha(Path(__file__)),'checkpoint_sha256':sha(ckpt),'scope':'model+stats BF16 postprocessing; sleep and NVML readings excluded; no nvidia-smi between each call; monitor once/second','cpu_threads':1},indent=2));print(json.dumps(summary),flush=True)

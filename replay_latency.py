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
stop=threading.Event();poll_errors=[];foreign=[]
def monitor():
 with (args.output/'gpu-monitor.jsonl').open('a') as f:
  while not stop.is_set():
   try:
    others=occupancy();foreign.extend(others)
    gpu=subprocess.check_output(['nvidia-smi','--id='+uuid,'--query-gpu=clocks.sm,temperature.gpu,power.draw,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True).strip()
    f.write(json.dumps({'time':time.time(),'foreign_pids':others,'gpu':gpu})+'\n');f.flush()
   except Exception as e:poll_errors.append(str(e))
   stop.wait(1)
thread=threading.Thread(target=monitor,daemon=True);thread.start()
scopes=['A_forward','B_stats_bf16','C_public_fp32','E_suite_fp32','D_cpu_to_cpu']
folders=sorted(d.parent for d in args.inputs.glob('*/capture.json'))
if args.smoke:folders=folders[:1]
else:assert len(folders)==40
manifest={'checkpoint':str(ckpt),'checkpoint_sha256':sha(ckpt),'script_sha256':sha(Path(__file__)),'gpu_uuid':uuid,'torch':torch.__version__,'cuda':torch.version.cuda,'cpu_threads':1,'seed':7,'warmup_per_task':10,'blocks':3,'smoke':args.smoke,'scopes':{'A':'model only','B':'model + official stats BF16 GPU postprocess','C':'model + public-policy FP32 GPU postprocess','E':'model + suite-stat FP32 GPU postprocess','D':'raw CPU observation rotation/state conversion/preparation/H2D/model/CPU return/suite FP32 actions; excludes env.step and IO'},'source_sha256':{str(f.relative_to(ROOT)):sha(f) for f in (ROOT/'turbovla').rglob('*.py')},'task_inputs':{}}
try:
 with torch.inference_mode():
  for folder in folders:
   target=args.output/folder.name;target.mkdir(exist_ok=True)
   if (target/'complete.json').exists():continue
   result=json.loads((folder/'result.json').read_text());capture=json.loads((folder/'capture.json').read_text());files=sorted(folder.glob('call*.npz'));assert result['total_episodes']==1 and len(files)==capture['calls'] and files
   manifest['task_inputs'][folder.name]={f.name:sha(f) for f in files}
   data=[dict(np.load(f,allow_pickle=False)) for f in files]
   def prepare(d):return policy._prepare_model_inputs(*policy._build_batch([d['primary']],[d['wrist']],[d['state']]))
   def cpu(d):
    obs={'robot0_eef_pos':d['eef_pos'],'robot0_eef_quat':d['eef_quat'],'robot0_gripper_qpos':d['gripper_qpos']}
    primary=rotate_libero_image(d['primary'][::-1,::-1]);wrist=rotate_libero_image(d['wrist'][::-1,::-1])
    return policy.predict_env_action_chunk(primary,wrist,str(d['instruction']),obs)
   d=data[0];images,states=prepare(d);instruction=str(d['instruction'])
   for _ in range(10):cpu(d)
   raw=policy.model([instruction],images,states)
   assert tuple(raw.shape)==(1,12,7) and torch.isfinite(raw).all()
   expected=np.stack([policy._normalized_row_to_env_action(row) for row in sanitize_pred_chunk(raw[0].float().cpu().numpy())])
   np.testing.assert_allclose(post(raw,low,high,True)[0].cpu().numpy(),expected,rtol=0,atol=1e-6)
   expectedpublic=np.stack([normalized_action_to_env_action(row) for row in sanitize_pred_chunk(raw[0].float().cpu().numpy())])
   np.testing.assert_allclose(post(raw,publiclow,publichigh,True)[0].cpu().numpy(),expectedpublic,rtol=0,atol=1e-6)
   np.testing.assert_allclose(cpu(d),expected,rtol=0,atol=1e-6)
   rows=[]
   for block in range(3):
    for i,d in enumerate(data):
     images,states=prepare(d);instruction=str(d['instruction'])
     def run(scope):
      if scope=='D_cpu_to_cpu':return cpu(d)
      raw=policy.model([instruction],images,states)
      if scope=='A_forward':return raw
      if scope=='B_stats_bf16':return post(raw,lowbf,highbf,False)
      if scope=='C_public_fp32':return post(raw,publiclow,publichigh,True)
      return post(raw,low,high,True)
     offset=(block+i)%len(scopes);order=scopes[offset:]+scopes[:offset]
     for scope in order:
      torch.cuda.synchronize();start=time.perf_counter();output=run(scope);torch.cuda.synchronize();ms=(time.perf_counter()-start)*1000;del output
      rows.append([folder.name,block,i,int(d['environment_step']),scope,ms])
     if foreign or poll_errors:raise RuntimeError('GPU contention or monitoring failure; discard affected run')
   assert not occupancy()
   with (target/'raw.csv').open('w') as f:
    w=csv.writer(f);w.writerow(['task','block','call','environment_step','scope','ms']);w.writerows(rows)
   tokens=policy.model.text_encoder.tokenizer([str(data[0]['instruction'])])
   (target/'complete.json').write_text(json.dumps({'calls':len(data),'rows':len(rows),'episode_success':result['total_successes'],'instruction':str(data[0]['instruction']),'tokens':len(tokens['input_ids'][0]),'checks':'suite/public GPU postprocess and CPU wrapper match respective references on first observation','canonical_action_max_abs_difference_first_observation':float(np.max(np.abs(expected-data[0]['actions'])))},indent=2))
   (args.output/'progress.json').write_text(json.dumps({'completed_tasks':len(list(args.output.glob('*/complete.json'))),'total_tasks':len(folders),'last_task':folder.name,'time':time.time()}))
   (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
 stop.set();thread.join();assert not foreign and not poll_errors
 (args.output/'status.json').write_text(json.dumps({'status':'completed','time':time.time()}))
except Exception as e:
 stop.set();thread.join();(args.output/'status.json').write_text(json.dumps({'status':'failed','error':str(e),'time':time.time()}));raise

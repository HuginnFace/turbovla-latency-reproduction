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

# Every condition times forward plus the same official stats-BF16 postprocessing.
files=sorted(args.inputs.glob('*/call*.npz'));assert len(files)==566
all_data=[dict(np.load(f,allow_pickle=False)) for f in files]
indices=[next(i for i,f in enumerate(files) if f.parent==folder) for folder in sorted({f.parent for f in files})]
def prepare(d):return policy._prepare_model_inputs(*policy._build_batch([d['primary']],[d['wrist']],[d['state']]))
def infer(d,inputs=None):
 images,states=prepare(d) if inputs is None else inputs
 return post(policy.model([str(d['instruction'])],images,states),lowbf,highbf,False)
def timed(d,inputs):
 torch.cuda.synchronize();t=time.perf_counter();y=infer(d,inputs);torch.cuda.synchronize();ms=(time.perf_counter()-t)*1000
 assert y.shape==(1,12,7) and bool(torch.isfinite(y).all())
 return ms
rows=[]
with torch.inference_mode():
 d=all_data[indices[0]];inputs=prepare(d)
 first=timed(d,inputs)
 # Untimed repeated-input and changed-input path audit. Hooks removed before timing.
 counts={};handles=[]
 def hook(name):
  def fn(*a):counts[name]=counts.get(name,0)+1
  return fn
 modules={'bert':policy.model.text_encoder.bert,'vision':policy.model.vision_encoder.backbone,'act':policy.model.action_head}
 for i,layer in enumerate(policy.model.vision_language_interaction.fusion_layers):modules[f'fusion{i}']=layer
 for name,module in modules.items():handles.append(module.register_forward_hook(hook(name)))
 original_tokenize=policy.model.text_encoder._tokenize_group
 def tokenize(*a,**kw):counts['tokenize']=counts.get('tokenize',0)+1;return original_tokenize(*a,**kw)
 policy.model.text_encoder._tokenize_group=tokenize
 for i in range(20):infer(d,inputs) if i<10 else infer(all_data[indices[i]])
 for handle in handles:handle.remove()
 policy.model.text_encoder._tokenize_group=original_tokenize
 assert counts==dict(tokenize=20,bert=20,vision=40,act=20,**{f'fusion{i}':20 for i in range(6)}),counts
 # Hardware cache pressure is diagnostic, not a guaranteed full-cache reset.
 sweep=torch.empty(256*1024*1024,dtype=torch.uint8,device='cuda')
 conditions=['fresh_input','same_tensor_repeated','allocator_empty','memory_sweep','paced_100ms','cpu_threads4','mha_fastpath_off']
 rng=np.random.default_rng(7)
 for block in range(3):
  for task_index in rng.permutation(indices):
   d=all_data[int(task_index)];inputs=prepare(d)
   for cond in rng.permutation(conditions):
    assert not occupancy(),'GPU became busy'
    torch.set_num_threads(4 if cond=='cpu_threads4' else 1)
    torch.backends.mha.set_fastpath_enabled(cond!='mha_fastpath_off')
    if cond=='same_tensor_repeated':
     for _ in range(5):y=infer(d,inputs);del y
    if cond=='allocator_empty':torch.cuda.synchronize();torch.cuda.empty_cache()
    if cond=='memory_sweep':sweep.fill_(block+1);torch.cuda.synchronize()
    if cond=='paced_100ms':torch.cuda.synchronize();time.sleep(.1)
    ms=timed(d,inputs);rows.append({'condition':str(cond),'block':block,'input_index':int(task_index),'ms':ms})
  (args.output/'progress.json').write_text(json.dumps({'blocks_complete':block+1,'rows':len(rows)}))
 torch.set_num_threads(1);torch.backends.mha.set_fastpath_enabled(True)
 # Corrected primary measurement: each real observation exactly once, fresh preparation,
 # shuffled across tasks, no interleaved alternative scopes for the same observation.
 for _ in range(10):infer(d,inputs)
 for i in rng.permutation(len(files)):
  d=all_data[int(i)];inputs=prepare(d);ms=timed(d,inputs)
  rows.append({'condition':'unique_one_pass','block':0,'input_index':int(i),'ms':ms})
 assert not occupancy()
with (args.output/'raw.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=['condition','block','input_index','ms']);w.writeheader();w.writerows(rows)
summary={}
for condition in sorted({r['condition'] for r in rows}):
 vals=[r['ms'] for r in rows if r['condition']==condition];summary[condition]={'n':len(vals),'median_ms':float(np.median(vals)),'p95_ms':float(np.percentile(vals,95))}
result={'status':'completed','first_forward_ms_after_model_load':first,'counts_20_forwards':counts,'conditions':summary,'checkpoint_sha256':sha(ckpt),'script_sha256':sha(Path(__file__)),'source_sha256':{str(f.relative_to(ROOT)):sha(f) for f in (ROOT/'turbovla').rglob('*.py')},'model_config':policy.model.config.to_dict(),'torch':torch.__version__,'cuda':torch.version.cuda,'gpu_uuid':uuid,'cpu_threads_primary':1,'timing':'synchronized wall clock, model+stats BF16 GPU postprocessing; input preparation excluded','notes':['empty_cache clears unused allocator blocks, not features or hardware caches','memory sweep256MiB is cache pressure not guaranteed full L2 reset; may affect clocks','unique_one_pass uses566 changing observations in shuffled order once each after10warmups','diagnostic counts are untimed; no hooks during timing','first forward excludes model loading and CUDA-context initialization']}
(args.output/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result['conditions'],indent=2),flush=True)

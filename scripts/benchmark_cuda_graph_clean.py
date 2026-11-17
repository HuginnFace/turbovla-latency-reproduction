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

folders=sorted(d.parent for d in args.inputs.glob('*/capture.json'))
if args.smoke:folders=folders[:1]
original_encode=policy.model.text_encoder.encode_bert_hidden
original_tokenize=policy.model.text_encoder._tokenize_group
rows=[];checks=[];unmixed=[];stop=threading.Event();errors=[]
def monitor():
 with (args.output/'gpu-monitor.jsonl').open('w') as f:
  while not stop.is_set():
   others=occupancy()
   if others:errors.append(others)
   f.write(json.dumps({'time':time.time(),'foreign_pids':others})+'\n');f.flush();stop.wait(1)
thread=threading.Thread(target=monitor,daemon=True);thread.start()
def prepare(d):return policy._prepare_model_inputs(*policy._build_batch([d['primary']],[d['wrist']],[d['state']]))
try:
 with torch.inference_mode():
  for folder in folders:
   data=[dict(np.load(f,allow_pickle=False)) for f in sorted(folder.glob('call*.npz'))]
   instruction=str(data[0]['instruction']);cfg=policy.model.text_encoder.config
   length=cfg.padding_length_by_instruction.get(instruction,cfg.padding_length)
   static_images,static_states=prepare(data[0]);static_tokens,static_masks,static_pos=original_tokenize([instruction],torch.device('cuda'),length)
   def eager(images,states):
    policy.model.text_encoder._tokenize_group=original_tokenize
    policy.model.text_encoder.encode_bert_hidden=original_encode
    return post(policy.model([instruction],images,states),lowbf,highbf,False)
   def graph_body():
    return post(policy.model([instruction],static_images,static_states),lowbf,highbf,False)
   for _ in range(10):eager(static_images,static_states)
   if False: # Profiler verification was completed in separate process.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
     for _ in range(3):eager(static_images,static_states)
     torch.cuda.synchronize()
    prof.export_chrome_trace(str(args.output/'eager-trace.json'))
   policy.model.text_encoder._tokenize_group=lambda *a,**kw:(static_tokens,static_masks,static_pos)
   def encode_static(instructions,device):
    gh,ga,gs=policy.model.text_encoder._encode_group(instructions,device,length)
    output_length=cfg.padding_length
    hidden=gh.new_zeros((1,output_length,gh.shape[-1]));hidden[:,:length]=gh
    attention=torch.zeros((1,output_length),device=device,dtype=torch.bool);attention[:,:length]=ga
    masks=torch.eye(output_length,device=device,dtype=torch.bool).unsqueeze(0).clone();masks[:,:length,:length]=gs
    return hidden,attention,masks
   policy.model.text_encoder.encode_bert_hidden=encode_static
   side=torch.cuda.Stream();side.wait_stream(torch.cuda.current_stream())
   with torch.cuda.stream(side):
    for _ in range(10):graph_body()
   torch.cuda.current_stream().wait_stream(side);torch.cuda.synchronize()
   graph=torch.cuda.CUDAGraph();started=time.perf_counter()
   with torch.cuda.graph(graph,stream=side):static_output=graph_body()
   torch.cuda.synchronize();capture_ms=(time.perf_counter()-started)*1000
   policy.model.text_encoder._tokenize_group=original_tokenize
   policy.model.text_encoder.encode_bert_hidden=original_encode
   def graph_request(images,states):
    # Re-tokenize online on EVERY request. No embedding cache.
    tokens,masks,pos=original_tokenize([instruction],torch.device('cuda'),length)
    for k in static_tokens:static_tokens[k].copy_(tokens[k])
    static_masks.copy_(masks);static_pos.copy_(pos)
    static_images['dinov3'].copy_(images['dinov3']);static_states.copy_(states)
    graph.replay();return static_output
   for _ in range(10):graph_request(static_images,static_states)
   if False: # Profiler verification was completed in separate process.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
     for _ in range(3):graph_request(static_images,static_states)
     torch.cuda.synchronize()
    prof.export_chrome_trace(str(args.output/'graph-trace.json'))
   for mode in ['eager','graph_request']:
    for _ in range(10):eager(static_images,static_states) if mode=='eager' else graph_request(static_images,static_states)
    for repeat in range(3):
     for j,d in enumerate(data):
      images,states=prepare(d)
      torch.cuda.synchronize();t=time.perf_counter();v=eager(images,states) if mode=='eager' else graph_request(images,states);torch.cuda.synchronize();ms=(time.perf_counter()-t)*1000;del v
      unmixed.append({'task':folder.name,'mode':mode,'repeat':repeat,'call':j,'ms':ms})
   maxdiff=0.;changed_outputs=[]
   for d in data:
    images,states=prepare(d);a=eager(images,states).clone();b=graph_request(images,states).clone();torch.cuda.synchronize()
    diff=float((a.float()-b.float()).abs().max());maxdiff=max(maxdiff,diff)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
   for block in range(3):
    for i,d in enumerate(data):
     images,states=prepare(d)
     for mode in (['eager','graph_request'] if (i+block)%2==0 else ['graph_request','eager']):
      torch.cuda.synchronize();t=time.perf_counter();value=eager(images,states) if mode=='eager' else graph_request(images,states);torch.cuda.synchronize();ms=(time.perf_counter()-t)*1000
      rows.append({'task':folder.name,'block':block,'call':i,'mode':mode,'ms':ms});del value
     if errors:raise RuntimeError('foreign GPU activity')
   # Replay-only lower bound: current fixed buffers; online input/token work excluded.
   replay_times=[]
   for _ in range(50):
    torch.cuda.synchronize();t=time.perf_counter();graph.replay();torch.cuda.synchronize();replay_times.append((time.perf_counter()-t)*1000)
   checks.append({'task':folder.name,'observations':len(data),'max_abs_difference':maxdiff,'capture_ms':capture_ms,'replay_only_median_ms':float(np.median(replay_times)),'replay_only_raw_ms':replay_times})
   (args.output/'checks.json').write_text(json.dumps(checks,indent=2));(args.output/'progress.json').write_text(json.dumps({'tasks':len(checks),'rows':len(rows)}))
   graph.reset();del graph,static_output
 with (args.output/'raw.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 with (args.output/'unmixed.json').open('w') as f:json.dump(unmixed,f)
 stop.set();thread.join();assert not errors and not occupancy()
 traces={}
 for mode in ['eager','graph']:
  events=json.loads((ROOT/'outputs/evaluation/cuda-graph-full-20261117'/(mode+'-trace.json')).read_text())['traceEvents']
  traces[mode]={'cudaGraphLaunch':sum('cudaGraphLaunch' in x.get('name','') for x in events),'cudaLaunchKernel':sum('cudaLaunchKernel' in x.get('name','') for x in events)}
 assert traces['eager']['cudaGraphLaunch']==0 and traces['graph']['cudaGraphLaunch']>=3,traces
 stats={}
 for mode in ['eager','graph_request']:
  v=[r['ms'] for r in rows if r['mode']==mode];stats[mode]={'n':len(v),'median_ms':float(np.median(v)),'p95_ms':float(np.percentile(v,95))}
 unmixed_stats={}
 for mode in ['eager','graph_request']:
  v=[r['ms'] for r in unmixed if r['mode']==mode];unmixed_stats[mode]={'n':len(v),'median_ms':float(np.median(v)),'p95_ms':float(np.percentile(v,95))}
 result={'unmixed_timings':unmixed_stats,'status':'completed','tasks':len(checks),'observations':sum(c['observations'] for c in checks),'timings':stats,'profiler_runtime_counts_from_separate_verified_run':traces,'checks':checks,'checkpoint_sha256':sha(ckpt),'script_sha256':sha(Path(__file__)),'model_source_sha256':{str(f.relative_to(ROOT)):sha(f) for f in (ROOT/'turbovla').rglob('*.py')},'scope':'GPU-ready images/state + online tokenization/mask-generation/token-transfer + model + stats-BF16 GPU postprocess; graph path additionally copies into static buffers; image/state CPU preparation excluded','graph_scope':'BERT + DINOv3 + fusion + ACT + GPU postprocessing; tokenization and dynamic text masks outside graph; task-specific capture; no embedding caching','torch':torch.__version__,'cuda':torch.version.cuda,'gpu_uuid':uuid,'cpu_threads':1}
 (args.output/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(stats),flush=True)
except Exception as e:
 stop.set();thread.join();policy.model.text_encoder._tokenize_group=original_tokenize
 (args.output/'failure.json').write_text(json.dumps({'error':str(e)}));raise

"""Offline integrity and result verification; does not execute TurboVLA."""
import csv,hashlib,io,json,zipfile
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1]
for line in (root/'SHA256SUMS').read_text().splitlines():
 digest,name=line.split('  ',1)
 assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
ident=json.loads((root/'data/episode-identities.json').read_text())
assert len(ident)==40 and sum(x['observations'] for x in ident)==566
assert sum(x['successes'] for x in ident)==38
expected={r['path']:r['sha256'] for r in json.loads((root/'data/observation-hashes.json').read_text())};seen=set()
for p in sorted(root.glob('data/observations/observations-*.zip')):
 with zipfile.ZipFile(p) as z:
  for name in z.namelist():
   assert name not in seen and hashlib.sha256(z.read(name)).hexdigest()==expected[name],name
   seen.add(name)
assert seen==set(expected)
def compare(values,ref):
 assert len(values)==ref['n']
 assert abs(float(np.median(values))-ref['median_ms'])<1e-9
 assert abs(float(np.percentile(values,95))-ref['p95_ms'])<1e-9
with zipfile.ZipFile(root/'results/raw-evidence.zip') as z:
 graph=json.loads(z.read('cuda-graph-clean/summary.json'))
 rows=list(csv.DictReader(io.StringIO(z.read('cuda-graph-clean/raw.csv').decode())))
 for mode,ref in graph['timings'].items():compare([float(r['ms']) for r in rows if r['mode']==mode],ref)
 assert graph['observations']==566 and all(c['max_abs_difference']==0 for c in graph['checks'])
 agg=json.loads(z.read('replay-full/aggregate.json'));pooled={}
 for name in z.namelist():
  if name.startswith('replay-full/') and name.endswith('/raw.csv'):
   for r in csv.DictReader(io.StringIO(z.read(name).decode())):pooled.setdefault(r['scope'],[]).append(float(r['ms']))
 for scope,ref in agg['pooled_call_weighted'].items():compare(pooled[scope],ref)
 assert sum(map(len,pooled.values()))==8490
print('PASS: published files; 40 episodes / 566 exact observations; 8490 replay timings; 1698 paired timings per graph mode; all medians/p95 and zero-difference checks.')

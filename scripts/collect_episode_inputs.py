"""Capture actual policy decision observations without changing official rollout actions."""
import os,json,hashlib
from pathlib import Path
import numpy as np
from turbovla.evaluation.suite_policy import TurboVLAPolicy
from turbovla.evaluation.policy import state_from_libero_obs
from vla_adapter import rollout
out=Path(os.environ['CAPTURE_DIR']);out.mkdir(parents=True,exist_ok=True)
original=TurboVLAPolicy.predict_env_action_chunk
calls=0
def capture(self,primary_image,wrist_image,instruction,state_or_obs,execute_steps=None):
 global calls
 assert isinstance(state_or_obs,dict) and execute_steps==12
 obs=state_or_obs
 actions=original(self,primary_image,wrist_image,instruction,obs,execute_steps)
 np.savez_compressed(out/f'call{calls:04}.npz',primary=primary_image,wrist=wrist_image,state=state_from_libero_obs(obs),eef_pos=obs['robot0_eef_pos'],eef_quat=obs['robot0_eef_quat'],gripper_qpos=obs['robot0_gripper_qpos'],instruction=np.asarray(instruction),actions=actions,environment_step=10+12*calls,policy_call=calls)
 calls+=1
 return actions
TurboVLAPolicy.predict_env_action_chunk=capture
rollout.main()
(out/'capture.json').write_text(json.dumps({'calls':calls,'seed':7,'episode_index':0,'settling_steps':10,'openloop':12,'input_source':'actual policy-driven LIBERO decision observations','collector_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))

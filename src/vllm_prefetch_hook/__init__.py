import logging 
import os 
import subprocess  
import sys  
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

_PREFETCH_TRIGGERED_ENV = "VLLM_PREFETCH_HOOK_TRIGGERED"

#vllm plugins, called before vllm startup
def register():
	if os.environ.get(_PREFETCH_TRIGGERED_ENV) == "1":
		return
	model_path = os.environ.get("VLLM_PREFETCH_MODEL_PATH")
	if not model_path:
		return 
	boostrap = ("from vllm_prefetch_hook import _prefetch_worker_main as m; m()")
	try:
		subprocess.Popen([sys.executable,"-c",boostrap,model_path], start_new_session=True,stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
		logger.info("vllm_prefetch_hook: launced independent prefetch subprocess for %s",model_path)
	except Exception as e:
		logger.warning("vllm_prefetch_hook: failed to launch prefetch subprocess: %s",e)	

	os.environ[_PREFETCH_TRIGGERED_ENV] = "1"

def _prefetch_one(path: Path):
	try:
		proc = subprocess.run(
            ["cat", str(path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
		if proc.returncode != 0:
			return path, f"cat exited with code {proc.returncode}"
		return path,None
	except OSError as e:
		return path,e 

#prefetch in independent os process
def _prefetch_worker_main():
	if len(sys.argv) < 2:
		return  
	model_path = Path(sys.argv[1])
	if not model_path.exists():
		return  
	patterns = ("**/*.safetensors","**/*.bin")
	files = []
	for pat in patterns:
		files.extend(model_path.glob(pat))
	files = sorted(set(files))
	threads = int(os.environ.get("VLLM_PREFETCH_THREADS","8"))
	threads = max(1,min(threads,len(files)))
	with ThreadPoolExecutor(max_workers=threads) as pool:
		futures = [pool.submit(_prefetch_one,f) for f in files]
		for fut in as_completed(futures):
			path,err = fut.result()
			if err:
				logger.warning("prefetch thread error: %s", err)




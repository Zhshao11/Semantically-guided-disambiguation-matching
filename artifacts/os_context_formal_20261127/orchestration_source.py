import os,subprocess,time,json
from pathlib import Path
root=Path("reports/os_context_formal_20261127")
logs=Path("reports/os_context_formal_logs_20261127")
env=os.environ.copy();env["PYTHONPATH"]="."
first_pid=json.loads((logs/"seed2026112706_pid.json").read_text())["pid"]
while not (root/"seed2026112706/completion.json").exists():
    os.kill(first_pid,0)
    time.sleep(10)
with (logs/"seed2026112706_verify.log").open("x") as f:
    subprocess.run([".venv/bin/python","scripts/train_os_context_formal.py","--seed","2026112706","--verify"],stdout=f,stderr=subprocess.STDOUT,env=env,check=True)
assert json.loads((root/"first_seed_engineering_gate.json").read_text())["passed"]
jobs=[]
for seed in [2026112707,2026112708]:
    f=(logs/f"seed{seed}.log").open("x")
    cmd=[".venv/bin/python","scripts/train_os_context_formal.py","--seed",str(seed),"--cuda-device","1"]
    job=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
    (logs/f"seed{seed}_pid.json").write_text(json.dumps(dict(pid=job.pid,cuda_device=1)))
    jobs.append((seed,job,f))
for seed,job,f in jobs:
    code=job.wait();f.close()
    if code:raise RuntimeError(f"Seed{seed} failed code{code}")
    with (logs/f"seed{seed}_verify.log").open("x") as log:
        subprocess.run([".venv/bin/python","scripts/train_os_context_formal.py","--seed",str(seed),"--verify"],stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
with (logs/"summary.log").open("x") as f:
    subprocess.run([".venv/bin/python","scripts/summarize_os_context_formal.py"],stdout=f,stderr=subprocess.STDOUT,env=env,check=True)
print("ALL_THREE_SEEDS_TRAINED_VERIFIED_SUMMARIZED",flush=True)

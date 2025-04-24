# --- main.py ---
import subprocess, glob, os, json
import numpy as np
from input_generator import generate_inputs

# setup
generate_inputs(10,100,0.1,15)
for d in ['output/dvhop','output/hco_vvs','output/hco_dpso','output/hco_vvs_custom']:
    os.makedirs(d, exist_ok=True)
# run
subprocess.run(['python','dvhop_only.py'])
subprocess.run(['python','hco_vvs.py'])
subprocess.run(['python','hco_dpso.py'])
subprocess.run(['python','hco_vvs_custom.py'])
# compare and tabulate
print(f"{'Input':<8} {'DV-Hop':>8} {'VVS':>8} {'DPSO':>8} {'Custom':>8}")
for inp in sorted(glob.glob('input/*.txt')):
    data=json.load(open(inp)); coords=np.array(data['coords'])
    outname=os.path.basename(inp).replace('input','output')
    errs=[]
    for alg in ['dvhop','hco_vvs','hco_dpso','hco_vvs_custom']:
        res=np.loadtxt(f'output/{alg}/{outname}')
        errs.append(np.linalg.norm(res-coords,axis=1).mean())
    print(f"{outname[6:]:<8} {errs[0]:>8.3f} {errs[1]:>8.3f} {errs[2]:>8.3f} {errs[3]:>8.3f}")

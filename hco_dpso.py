# --- hco_dpso.py ---
import os,json
import numpy as np
from common_utils import compute_hop_counts, dvhop_estimate
from dpso import DPSO_Optimizer
os.makedirs('output/hco_dpso',exist_ok=True)

def run_hco_dpso():
    for fname in sorted(os.listdir('input')):
        data=json.load(open(f'input/{fname}'))
        coords=np.array(data['coords']); anchors=data['anchors']
        comm_range=data['comm_range']; area_size=data['area_size']
        hop=compute_hop_counts(coords,comm_range,anchors)
        d_est=dvhop_estimate(coords,anchors,hop)
        unk=[]
        for u in range(len(coords)):
            if u in anchors: unk.append(coords[u])
            else:
                def obj_fn(x): return np.sum(np.abs(np.linalg.norm(x-coords[anchors],axis=1)-d_est[:,u]))
                opt=DPSO_Optimizer(obj_fn,[[0,area_size],[0,area_size]])
                unk.append(opt.optimize()[0])
        arr=np.stack(unk)
        np.savetxt(f'output/hco_dpso/output{fname[5:]}',arr,'%.6f')

if __name__=='__main__': run_hco_dpso()
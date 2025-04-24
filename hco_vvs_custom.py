# --- hco_vvs_custom.py ---
import os,json
import numpy as np
from common_utils import compute_hop_counts, dvhop_estimate
from vvs_hco import VVSHCO_Optimizer
os.makedirs('output/hco_vvs_custom',exist_ok=True)

def run_custom():
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
                pop=50
                def init_pop():
                    half=pop//2
                    P=np.random.uniform(0,area_size,(half,2))
                    centroid=np.dot(d_est[:,u],coords[anchors])/d_est[:,u].sum()
                    Q=np.clip(centroid+np.random.randn(pop-half,2)*area_size*0.1,0,area_size)
                    return np.vstack([P,Q])
                opt=VVSHCO_Optimizer(obj_fn,[[0,area_size],[0,area_size]],pop_size=pop)
                opt.initialize=lambda:init_pop()
                unk.append(opt.optimize()[0])
        arr=np.stack(unk)
        np.savetxt(f'output/hco_vvs_custom/output{fname[5:]}',arr,'%.6f')

if __name__=='__main__': run_custom()
# --- dvhop_only.py (Enhanced DV-Hop with Weighted LS) ---
import os, json
import numpy as np
from common_utils import compute_hop_counts
from scipy.spatial.distance import cdist

OUT_DIR = 'output/dvhop'
if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)

def run_enhanced_dvhop():
    for fname in sorted(os.listdir('input')):
        data = json.load(open(f'input/{fname}'))
        coords = np.array(data['coords'])
        anchors = data['anchors']
        comm_range = data['comm_range']
        area_size = data['area_size']
        anchor_coords = coords[anchors]
        A = len(anchors)
        hop = compute_hop_counts(coords, comm_range, anchors)
        full_dist = cdist(anchor_coords, anchor_coords)
        avg_anchor_dist = (full_dist.sum() - np.trace(full_dist)) / (A*(A-1))
        HpSz = np.zeros(A)
        for i in range(A):
            sumd = sumh = 0.0
            for j in range(A):
                if i == j: continue
                h_ij = hop[i, anchors[j]]
                if np.isfinite(h_ij) and h_ij > 0:
                    sumd += full_dist[i,j]; sumh += h_ij
            HpSz[i] = sumd/sumh if sumh > 0 else avg_anchor_dist
        HpSz_eav = np.zeros((A,A))
        for i in range(A):
            valid = [(j, hop[i,anchors[j]]) for j in range(A) if j!=i and np.isfinite(hop[i,anchors[j]]) and hop[i,anchors[j]]>0]
            n_i = min(valid, key=lambda x: x[1])[0] if valid else None
            for j in range(A):
                h_ij = hop[i,anchors[j]] if np.isfinite(hop[i,anchors[j]]) else 0
                h_in = hop[i,anchors[n_i]] if n_i is not None else 0
                err_ij = HpSz[i]*h_ij - full_dist[i,j]
                err_in = HpSz[i]*h_in - (full_dist[i,n_i] if n_i is not None else 0)
                denom = (h_ij + h_in) if (h_ij + h_in)>0 else 1e-9
                HpSz_eav[i,j] = HpSz[i] - (err_ij + err_in)/denom
        unk = []
        for u in range(len(coords)):
            if u in anchors:
                unk.append(coords[u])
                continue
            cand = [j for j in range(A) if np.isfinite(hop[j,u]) and hop[j,u]>0]
            if len(cand) < 3:
                pos = anchor_coords.mean(axis=0)
            else:
                sel = sorted(cand, key=lambda j: hop[j,u])[:3]
                pts = anchor_coords[sel]
                d_sel = np.array([HpSz_eav[z,j] * hop[j,u] for z,j in enumerate(sel)])
                A_arr = pts[:,0]; B_arr = pts[:,1]
                E_arr = A_arr**2 + B_arr**2
                A1, B1, E1, d1 = A_arr[0], B_arr[0], E_arr[0], d_sel[0]
                G = []
                h_c = []
                w = []
                for k in range(1, len(sel)):
                    Ai, Bi, Ei, di = A_arr[k], B_arr[k], E_arr[k], d_sel[k]
                    G.append([-2*(Ai-A1), -2*(Bi-B1)])
                    h_c.append(di**2 - d1**2 - (Ei - E1))
                    hopval = hop[sel[k], u]
                    w.append((1.0/hopval)**2)
                G = np.array(G)
                h_c = np.array(h_c)
                W = np.diag(w)
                GTWG = G.T.dot(W).dot(G)
                GTWc = G.T.dot(W).dot(h_c)
                sol = np.linalg.solve(GTWG, GTWc)
                pos = sol
            pos = np.clip(pos, 0, area_size)
            unk.append(pos)
        arr = np.stack(unk)
        np.savetxt(f'{OUT_DIR}/output{fname[5:]}', arr, fmt='%.6f')

if __name__ == '__main__':
    run_enhanced_dvhop()
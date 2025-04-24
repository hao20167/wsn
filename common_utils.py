import numpy as np
from scipy.spatial.distance import cdist


def generate_network(num_nodes, area_size, comm_range, anchor_ratio):
    coords = np.random.rand(num_nodes, 2) * area_size
    num_anchors = max(int(anchor_ratio * num_nodes), 3)
    anchors = np.random.choice(num_nodes, num_anchors, replace=False).tolist()
    return coords, anchors


def compute_hop_counts(coords, comm_range, anchors):
    N = coords.shape[0]
    dist = cdist(coords, coords)
    adj = (dist <= comm_range).astype(int)
    hop_counts = np.full((len(anchors), N), np.inf)
    for i, a in enumerate(anchors):
        hops = np.full(N, np.inf)
        hops[a] = 0
        frontier = {a}
        step = 0
        while frontier:
            next_front = set()
            for u in frontier:
                for v in np.where(adj[u])[0]:
                    if hops[v] == np.inf:
                        hops[v] = step + 1
                        next_front.add(v)
            frontier = next_front
            step += 1
        hop_counts[i] = hops
    return hop_counts


def dvhop_estimate(coords, anchors, hop_counts):
    """
    Compute estimated distances using the adjusted DV-Hop separation:
      d_ij = (hops_ij + theta) * HpSz_avg
    where theta = (sum_est_pairs - sum_real_pairs) / n_pairs
    """
    A, N = hop_counts.shape
    anchor_coords = coords[anchors]
    # 1) per-anchor hop size
    HpSz = np.zeros(A)
    for i in range(A):
        num = den = 0.0
        for j in range(A):
            if i == j:
                continue
            d_true = np.linalg.norm(anchor_coords[i] - anchor_coords[j])
            h_ij = hop_counts[i, anchors[j]]
            if np.isfinite(h_ij) and h_ij > 0:
                num += d_true
                den += h_ij
        HpSz[i] = num / den if den > 0 else 0.0
    # 2) global average hop size
    HpSz_avg = HpSz.sum() / A if A > 0 else 0.0
    # 3) finite hop counts matrix
    hop_finite = np.where(np.isfinite(hop_counts), hop_counts, 0.0)
    # 4) compute adjustment theta based on beacon-beacon pairs
    sum_est = 0.0
    sum_real = 0.0
    count = 0
    for i in range(A):
        for j in range(i+1, A):
            sum_est += HpSz_avg * hop_finite[i, anchors[j]]
            sum_real += np.linalg.norm(anchor_coords[i] - anchor_coords[j])
            count += 1
    theta = (sum_est - sum_real) / count if count > 0 else 0.0
    # 5) adjusted separation matrix
    return (hop_finite + theta) * HpSz_avg


def trilaterate(anchors_coords, dists):
    m = anchors_coords.shape[0]
    ref = m - 1
    xk, yk = anchors_coords[ref]
    dk = dists[ref]
    A = []
    B = []
    for i in range(m-1):
        xi, yi = anchors_coords[i]
        di = dists[i]
        A.append([2*(xi-xk), 2*(yi-yk)])
        B.append(di*di - dk*dk - xi*xi + xk*xk - yi*yi + yk*yk)
    A = np.array(A)
    B = np.array(B)
    sol, *_ = np.linalg.lstsq(A, B, rcond=None)
    return sol

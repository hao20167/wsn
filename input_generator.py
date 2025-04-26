# --- input_generator.py ---
import os
import json
import time
import numpy as np

INPUT_DIR = 'input'
if not os.path.exists(INPUT_DIR):
    os.makedirs(INPUT_DIR)

def generate_inputs(num_files, num_nodes, anchor_ratio, comm_range, area_size=100):
    # seed once from current time (in seconds)
    seed = int(time.time())
    np.random.seed(seed)
    print(f"Seeding random generator with {seed}")

    for i in range(1, num_files + 1):
        coords = (np.random.rand(num_nodes, 2) * area_size).tolist()
        num_anchors = max(int(anchor_ratio * num_nodes), 3)
        anchors = np.random.choice(num_nodes, num_anchors, replace=False).tolist()
        cfg = {
            'coords': coords,
            'anchors': anchors,
            'comm_range': comm_range,
            'area_size': area_size
        }
        with open(f'{INPUT_DIR}/input{i}.txt', 'w') as f:
            json.dump(cfg, f)

    print(f"Generated {num_files} input files in '{INPUT_DIR}'")

if __name__ == '__main__':
    generate_inputs(3, 100, 0.1, 15)

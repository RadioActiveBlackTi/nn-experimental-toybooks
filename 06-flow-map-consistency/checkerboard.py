import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

def make_checkerboard(n_samples_per_tile=5000, num_tiles_per_side=4, single_tile=True):

    total_tiles = num_tiles_per_side ** 2
    total_samples = total_tiles * n_samples_per_tile
    
    points = np.random.uniform(-1, 1, size=(total_samples, 2))
    
    scaled_points = (points + 1) / 2 * num_tiles_per_side
    ints = np.floor(scaled_points).astype(int)
    
    labels = (ints[:, 0] + ints[:, 1]) % 2
    
    if single_tile:
        points = points[labels == 0]
        labels = labels[labels == 0]
    return points, labels

def approximate_chunked_pairing(x0, x1, chunk_size=256):
    assert x0.shape == x1.shape, "x0 and x1 must have the same shape"
    n = x0.shape[0]
    perm0 = torch.randperm(n, device=x0.device)
    perm1 = torch.randperm(n, device=x1.device)
    x0_perm = x0[perm0]
    x1_perm = x1[perm1]

    paired_x0 = []
    paired_x1 = []

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        a = x0_perm[start:end]
        b = x1_perm[start:end]
        cost = torch.cdist(a, b).cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost)
        paired_x0.append(a[row_ind])
        paired_x1.append(b[col_ind])

    return torch.cat(paired_x0, dim=0), torch.cat(paired_x1, dim=0)
    
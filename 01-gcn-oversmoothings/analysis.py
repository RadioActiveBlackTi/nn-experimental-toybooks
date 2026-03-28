from matplotlib import pyplot as plt
import torch
from torch_geometric.utils import get_laplacian, to_dense_adj

def plot_gcn_oversmoothing_analysis(activations_dict, edge_index, num_nodes=2708):
    
    # 1. Laplacian Matrix 
    # L = I - D^(-0.5) A D^(-0.5) (Normalized Laplacian)
    edge_index_lap, edge_weight_lap = get_laplacian(edge_index, normalization='sym')
    L = to_dense_adj(edge_index_lap, edge_attr=edge_weight_lap, max_num_nodes=num_nodes)[0]
    L = L.to(list(activations_dict.values())[0].device) 

    layers = list(activations_dict.keys())
    energies = []
    svd_results = []

    print("Analyzing layers...")
    for layer_name in layers:
        H = activations_dict[layer_name] # Shape: [2708, F]
        
        # --- Analysis 1: Dirichlet Energy ---
        # E = trace(H^T * L * H)
        # normalize by number of nodes and feature dimension for fair comparison
        energy = torch.trace(H.T @ L @ H) / (H.shape[0] * H.shape[1])
        energies.append(energy.item())

        # --- Analysis 2: SVD Spectrum (Effective Rank) ---
        _, S, _ = torch.svd(H)
        # Normalize singular values to sum to 1 (probability distribution like)
        S = S / S.sum()
        svd_results.append(S.cpu().numpy())

    # --- Plotting ---
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Dirichlet Energy Decay
    axes[0].plot(layers, energies, marker='o', linestyle='-', color='r', linewidth=2)
    axes[0].set_title("Dirichlet Energy (Roughness) by Layer")
    axes[0].set_xlabel("Layer Depth")
    axes[0].set_ylabel("Dirichlet Energy (Log Scale)")
    axes[0].set_yscale('log') 
    axes[0].grid(True, which="both", ls="--")
    axes[0].tick_params(axis='x', rotation=45)

    # Plot 2: Singular Value Distributio
    for i, sv in enumerate(svd_results):
        k = min(len(sv), 10) 
        axes[1].plot(range(1, k+1), sv[:k], label=layers[i], alpha=0.7)
    
    axes[1].set_title("Singular Value Spectrum (Feature Collapse)")
    axes[1].set_xlabel("Singular Value Rank")
    axes[1].set_ylabel("Normalized Magnitude")
    axes[1].set_yscale('log')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()
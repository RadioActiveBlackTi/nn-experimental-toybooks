import torch
import torch.nn as nn
from torchdiffeq import odeint
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_trajectory(model, num_samples):
    model.eval()
    with torch.no_grad():
        zT = torch.randn(num_samples, 2).to(device)
        logp_zT = torch.zeros(num_samples, 1).to(device)

        if model.dopri:
            integration_times = torch.tensor([1., 0.]).to(device)
            z0, _ = odeint(model.ffjord, (zT, logp_zT), integration_times, rtol=model.rtol, atol=model.atol)
        else:
            integration_times = torch.linspace(1., 0., steps=model.num_steps).to(device)
            z0, _ = odeint(model.ffjord, (zT, logp_zT), integration_times, method='rk4')
        return z0

def plot_trajectories(z0, num_plot=100):
    traj = z0.detach().cpu().numpy()
    
    plt.figure(figsize=(6, 6))
    
    final_z = traj[-1]
    plt.scatter(final_z[:, 0], final_z[:, 1], c='green', s=5, alpha=0.1, label='Final Samples')
    
    for i in range(num_plot):
        plt.plot(traj[:, i, 0], traj[:, i, 1], c='black', alpha=0.3, linewidth=0.8)
        
    plt.scatter(traj[0, :num_plot, 0], traj[0, :num_plot, 1], c='red', s=10, label='Start (Noise)')
    plt.scatter(traj[-1, :num_plot, 0], traj[-1, :num_plot, 1], c='blue', s=10, label='End (Data)')

    plt.title(f"Particle Trajectories (RK4 Steps: {traj.shape[0]-1})")
    plt.legend()
    plt.show()
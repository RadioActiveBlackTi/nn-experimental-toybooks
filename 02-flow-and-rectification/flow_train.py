import torch
import torch.optim as optim
import numpy as np
from FFJORD import ODEFunc, ODEModel
from swissroll import train_loader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, batch_data in enumerate(train_loader):
            batch_data = batch_data.to(device)
            batch_size = batch_data.size(0)

            # Forward pass
            z0 = batch_data
            zT, logp_zT = model(z0)

            # Compute log likelihood under standard normal
            logp_z0 = -0.5 * torch.sum(zT**2, dim=1, keepdim=True) - 0.5 * 2 * np.log(2 * np.pi)
            logp_z0 = logp_z0 - logp_zT
            loss = -torch.mean(logp_z0)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')

def train_reflow(model, optimizer, epochs, Z0, Z1, batch_size):
    model.train()
    dataset_size = Z0.size(0)
    for epoch in range(epochs):
        perm = torch.randperm(dataset_size)
        total_loss = 0
        for i in range(0, dataset_size, batch_size):
            indices = perm[i:i+batch_size]
            batch_z0 = Z0[indices].to(device)
            batch_z1 = Z1[indices].to(device)

            t = torch.rand(batch_z0.size(0), 1).to(device)
            batch_zt = t * batch_z1 + (1 - t) * batch_z0
            pred_zt = model(t, batch_zt)
            loss = F.mse_loss(pred_zt, batch_z1 - batch_z0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / (dataset_size / batch_size)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}')
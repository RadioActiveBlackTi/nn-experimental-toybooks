import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from abc import ABC, abstractmethod

from encoders import PointNetEncoder, PointNetVarEncoder
from decoders import PointDecoder, PointGMMDecoder

def chamfer_distance(points1, points2):
    # points1, points2: (B, N, 3)
    dist = torch.cdist(points1, points2, p=2) ** 2  # squared euclidean
    cd1 = torch.min(dist, dim=2)[0].mean(dim=1)
    cd2 = torch.min(dist, dim=1)[0].mean(dim=1)
    return (cd1 + cd2).mean()

def orthogonal_regularization(t, reg_weight=1e-3):
    # t: (B, k, k)
    batch_size, k, _ = t.size()
    identity = torch.eye(k, device=t.device).unsqueeze(0).repeat(batch_size, 1, 1)  # (B, k, k)
    t_transpose = t.transpose(1, 2)  # (B, k, k)
    product = torch.bmm(t, t_transpose)  # (B, k, k)

    mat_diff = (identity - product).reshape(batch_size, -1)  # (B, k*k)
    return reg_weight * torch.mean(torch.norm(mat_diff, dim=1))


class AEInterface(nn.Module, ABC):
    """Common interface for Autoencoders / Variational Autoencoders."""
    def __init__(self):
        super().__init__()

    @abstractmethod
    def encode(self, x):
        # Returns latent representation (or tuple/dict for VAE variants)
        raise NotImplementedError

    @abstractmethod
    def decode(self, latent):
        # Decodes latent to point cloud
        raise NotImplementedError

    @abstractmethod
    def reconstruct(self, x):
        # End-to-end reconstruction only
        raise NotImplementedError

    @abstractmethod
    def loss_forward(self, x, **kwargs):
        # Computes training loss (e.g., CD + regularizers [+ KL for VAE])
        raise NotImplementedError

    def forward(self, x):
        # Keep forward behavior consistent for inference/visualization
        return self.reconstruct(x)


def train_PAE(pae, dataloader, n_epochs=120, lr=3e-4, weight_decay=1e-4):
    pae.train()

    optimizer = optim.AdamW(
        list(pae.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-5)

    losses = []
    for epoch in range(n_epochs):
        total_loss = 0.0
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(device).float()  # (B, N, 3)
            optimizer.zero_grad()

            loss = pae.loss_forward(data)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(pae.parameters()), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.6f}, LR: {current_lr:.6e}")

    return losses


class PointCloudAutoencoder(AEInterface):
    def __init__(self, latent_dim=64, n_points=2048, reg_weight=1e-3):
        super().__init__()
        self.encoder = PointNetEncoder(latent_dim=latent_dim)
        self.decoder = PointDecoder(latent_dim=latent_dim, n_points=n_points)
        self.reg_weight = reg_weight

    def encode(self, x):
        latent, _, _ = self.encoder(x)
        return latent

    def decode(self, latent):
        return self.decoder(latent)

    def reconstruct(self, x):
        latent, _, _ = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    def forward_with_aux(self, x):
        # Optional: expose transform matrices for regularization/debugging
        latent, t3, t64 = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, t3, t64

    def loss_forward(self, x):
        reconstructed, t3, t64 = self.forward_with_aux(x)
        cd_loss = chamfer_distance(reconstructed, x)
        ortho_loss = orthogonal_regularization(t3, reg_weight=self.reg_weight) + orthogonal_regularization(t64, reg_weight=self.reg_weight)
        return cd_loss + ortho_loss


class PointCloudVAE(AEInterface):
    def __init__(self, latent_dim=64, n_points=2048, reg_weight=1e-3, kl_weight=1e-2):
        super().__init__()
        self.encoder = PointNetVarEncoder(latent_dim=latent_dim)
        self.decoder = PointDecoder(latent_dim=latent_dim, n_points=n_points)
        self.reg_weight = reg_weight
        self.kl_weight = kl_weight  # KL divergence weight

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        mu, logvar, _, _ = self.encoder(x)
        return mu, logvar

    def decode(self, latent):
        return self.decoder(latent)

    def reconstruct(self, x):
        mu, logvar, _, _ = self.encoder(x)
        latent = self.reparameterize(mu, logvar)
        reconstructed = self.decoder(latent)
        return reconstructed

    def forward_with_aux(self, x):
        mu, logvar, t3, t64 = self.encoder(x)
        latent = self.reparameterize(mu, logvar)
        reconstructed = self.decoder(latent)
        return reconstructed, mu, logvar, t3, t64

    def loss_forward(self, x):
        reconstructed, mu, logvar, t3, t64 = self.forward_with_aux(x)
        cd_loss = chamfer_distance(reconstructed, x)
        ortho_loss = orthogonal_regularization(t3, reg_weight=self.reg_weight) + orthogonal_regularization(t64, reg_weight=self.reg_weight)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return cd_loss + ortho_loss + self.kl_weight * kl_loss


class PointGMMVAE(AEInterface):
    def __init__(self, latent_dim=64, attn_dim=16, value_dim=32, branch_factor=2, level=3, reg_weight=1e-3, kl_weight=1e-2, sigma_min=1e-2):
        super().__init__()
        self.encoder = PointNetVarEncoder(latent_dim=latent_dim)
        self.decoder = PointGMMDecoder(latent_dim=latent_dim, attn_dim=attn_dim, value_dim=value_dim, branch_factor=branch_factor, level=level)
        self.reg_weight = reg_weight
        self.kl_weight = kl_weight
        self.sigma_min = sigma_min

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        mu, logvar, _, _ = self.encoder(x)
        return mu, logvar

    def decode(self, latent):
        return self.decoder(latent)
    
    def gmm_loss(self, x, levelwise_results):
        # x: [B, N, 3]
        # levelwise_results: list of (mu [B, K, 3], sigma [B, K, 3], logpi [B, K]) for each level
        # Full Gaussian mixture NLL (with normalization constant) + sigma floor for numerical stability
        total_loss = 0.0
        log_2pi = np.log(2.0 * np.pi)

        for mu, sigma, logpi in levelwise_results:
            sigma = torch.clamp(sigma, min=self.sigma_min)
            x_expanded = x.unsqueeze(2)  # [B, N, 1, 3]
            mu_expanded = mu.unsqueeze(1)  # [B, 1, K, 3]
            sigma_expanded = sigma.unsqueeze(1)  # [B, 1, K, 3]

            squared_mahal = torch.sum(((x_expanded - mu_expanded) / sigma_expanded) ** 2, dim=-1)  # [B, N, K]
            log_det = torch.sum(torch.log(sigma_expanded), dim=-1)  # [B, 1, K]

            # log N(x|mu,sigma^2I) = -0.5 * (mahal + D*log(2pi)) - sum(log sigma)
            log_gauss = -0.5 * (squared_mahal + 3.0 * log_2pi) - log_det  # [B, N, K]
            log_mix = torch.logsumexp(logpi.unsqueeze(1) + log_gauss, dim=-1)  # [B, N]

            nll = -log_mix.mean()
            total_loss += nll

        return total_loss / max(len(levelwise_results), 1)

    def gmm_sample(self, mu, sigma, logpi, n_points=2048):
        # mu, sigma: [B, K, 3], logpi: [B, K]
        B, K, _ = mu.size()
        pi = torch.softmax(logpi, dim=1)  # [B, K]
        categorical = torch.distributions.Categorical(pi)
        component_indices = categorical.sample((n_points,)).transpose(0, 1)  # [B, N]

        batch_indices = torch.arange(B, device=mu.device).unsqueeze(1).expand(B, component_indices.size(1))  # [B, N]
        selected_mu = mu[batch_indices, component_indices]  # [B, N, 3]
        selected_sigma = sigma[batch_indices, component_indices]  # [B, N, 3]
        selected_sigma = torch.clamp(selected_sigma, min=self.sigma_min)

        eps = torch.randn_like(selected_sigma)
        sampled_points = selected_mu + eps * selected_sigma
        return sampled_points

    def reconstruct(self, x):
        mu, logvar, _, _ = self.encoder(x)
        latent = self.reparameterize(mu, logvar)
        decoded_mu, decoded_sigma, decoded_logpi, _ = self.decoder(latent)
        return self.gmm_sample(decoded_mu, decoded_sigma, decoded_logpi)
    
    def decoder_sample(self, latent, n_points=2048):
        decoded_mu, decoded_sigma, decoded_logpi, _ = self.decoder(latent)
        return self.gmm_sample(decoded_mu, decoded_sigma, decoded_logpi, n_points=n_points)

    def forward_with_aux(self, x):
        mu, logvar, t3, t64 = self.encoder(x)
        latent = self.reparameterize(mu, logvar)
        decoded_mu, decoded_sigma, decoded_logpi, levelwise_results = self.decoder(latent)
        return decoded_mu, levelwise_results, mu, logvar, t3, t64

    def loss_forward(self, x):
        decoded_mu, levelwise_results, mu, logvar, t3, t64 = self.forward_with_aux(x)

        gmm_loss = self.gmm_loss(x, levelwise_results)

        ortho_loss = orthogonal_regularization(t3) + orthogonal_regularization(t64)

        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        return gmm_loss + self.reg_weight * ortho_loss + self.kl_weight * kl_loss
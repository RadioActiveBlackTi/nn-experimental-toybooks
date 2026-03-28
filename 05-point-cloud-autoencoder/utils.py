import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def reconstruct_and_visualize(AE, data):
    AE.eval()
    with torch.no_grad():
        data = data.to(device).float().unsqueeze(0)  # (1, N, 3)
        reconstructed = AE.reconstruct(data).squeeze(0).cpu().numpy()  # (N, 3)

    original = data.squeeze(0).cpu().numpy()

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    depth = np.linalg.norm(original, axis=1)
    ax1.scatter(original[:, 0], original[:, 1], original[:, 2], s=5, c=depth, cmap='viridis')
    ax1.set_title('Original Point Cloud')

    ax2 = fig.add_subplot(122, projection='3d')
    depth_recon = np.linalg.norm(reconstructed, axis=1)
    ax2.scatter(reconstructed[:, 0], reconstructed[:, 1], reconstructed[:, 2], s=5, c=depth_recon, cmap='viridis')
    ax2.set_title('Reconstructed Point Cloud')
    plt.show()


def latent_plot(AE, dataloader, n_batches=5, dim0=0, dim1=1):
    plt.figure(figsize=(6, 4))
    AE.eval()
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(dataloader):
            if batch_idx >= n_batches:
                break

            data = data.to(device).float()  # (B, N, 3)
            latent, _, _ = AE.encoder(data)  # (B, latent_dim)
            latent = latent.cpu().numpy()
            plt.scatter(latent[:, dim0], latent[:, dim1], s=10, alpha=0.5)
    plt.title(f'Latent Space (Dim {dim0} vs Dim {dim1})')
    plt.xlabel(f'Latent Dim {dim0}')
    plt.ylabel(f'Latent Dim {dim1}')
    plt.grid(True, alpha=0.3)
    plt.show()


def latent_interpolation(AE, latent_dim_idx=0, input_latent=None, min=-1.0, max=1.0, steps=10):
    AE.eval()
    with torch.no_grad():
        latent_vectors = []
        for alpha in np.linspace(min, max, steps):
            if input_latent is not None:
                latent = input_latent.clone().detach()
                if latent.dim() == 1:
                    latent = latent.unsqueeze(0)  # (D,) -> (1, D)
                latent = latent.to(device).float()
            else:
                latent = torch.randn(1, AE.decoder.fc_in[0].in_features, device=device)

            latent[0, latent_dim_idx] = float(alpha)
            latent_vectors.append(latent)

        latent_vectors = torch.cat(latent_vectors, dim=0)  # (steps, latent_dim)
        reconstructed_points = AE.decode(latent_vectors)  # (steps, N, 3)

    fig = plt.figure(figsize=(12, 6))
    for i in range(steps):
        ax = fig.add_subplot(2, steps // 2, i + 1, projection='3d')
        depth = np.linalg.norm(reconstructed_points[i].cpu().numpy(), axis=1)
        ax.scatter(
            reconstructed_points[i][:, 0].cpu(),
            reconstructed_points[i][:, 1].cpu(),
            reconstructed_points[i][:, 2].cpu(),
            s=5, c=depth, cmap='viridis'
        )
        ax.set_title(f'Alpha: {np.linspace(min, max, steps)[i]:.2f}')
    plt.tight_layout()
    plt.show()


def input_transform_visualization(AE, data):
    AE.eval()
    with torch.no_grad():
        data = data.to(device).float().unsqueeze(0)  # (1, N, 3)
        _, t3, _ = AE.encoder(data)  # (1, 3, 3)
        transformed = torch.bmm(data, t3).squeeze(0).cpu().numpy()  # (N, 3)

    original = data.squeeze(0).cpu().numpy()

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    depth = np.linalg.norm(original, axis=1)
    ax1.scatter(original[:, 0], original[:, 1], original[:, 2], s=5, c=depth, cmap='viridis')
    ax1.set_title('Original Point Cloud')

    ax2 = fig.add_subplot(122, projection='3d')
    depth_transformed = np.linalg.norm(transformed, axis=1)
    ax2.scatter(transformed[:, 0], transformed[:, 1], transformed[:, 2], s=5, c=depth_transformed, cmap='viridis')
    ax2.set_title('Transformed Point Cloud (Input T-Net)')
    plt.show()


def vae_latent_plot(VAE, dataloader, n_batches=5, dim0=0, dim1=1):
    plt.figure(figsize=(6, 4))
    VAE.eval()
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(dataloader):
            if batch_idx >= n_batches:
                break

            data = data.to(device).float()  # (B, N, 3)
            latent, _, _, _ = VAE.encoder(data)  # (B, latent_dim)
            latent = latent.cpu().numpy()
            plt.scatter(latent[:, dim0], latent[:, dim1], s=10, alpha=0.5)
    plt.title(f'Latent Space (Dim {dim0} vs Dim {dim1})')
    plt.xlabel(f'Latent Dim {dim0}')
    plt.ylabel(f'Latent Dim {dim1}')
    plt.grid(True, alpha=0.3)
    plt.show()


def draw_ellipsoid(ax, center, sigma, color='cyan', alpha=0.3, scale=2.0):
    """Draw a semi-transparent ellipsoid."""
    # Parametric ellipsoid
    u = np.linspace(0, 2 * np.pi, 25)
    v = np.linspace(0, np.pi, 20)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    
    # Scale by sigma and translate to center
    # scale parameter: how many standard deviations to display
    x_scaled = center[0] + x * sigma[0] * scale
    y_scaled = center[1] + y * sigma[1] * scale
    z_scaled = center[2] + z * sigma[2] * scale
    
    ax.plot_surface(x_scaled, y_scaled, z_scaled, color=color, alpha=alpha, edgecolor='none')

def reconstruct_and_visualize_means(PGVAE, data, use_ellipsoid=True, ellipsoid_scale=1.5):
    PGVAE.eval()
    with torch.no_grad():
        data = data.to(device).float().unsqueeze(0)  # (1, N, 3)
        decoded_mu, levelwise_results, _, _, _, _ = PGVAE.forward_with_aux(data)
        decoded_mu = decoded_mu.squeeze(0).cpu().numpy()  # (K, 3)
        decoded_sigma = levelwise_results[-1][1].squeeze(0).cpu().numpy()  # (K, 3) - sigma from last level
        sampled_points = PGVAE.reconstruct(data).squeeze(0).cpu().numpy()  # (N, 3)

    original = data.squeeze(0).cpu().numpy()

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(131, projection='3d')
    depth = np.linalg.norm(original, axis=1)
    ax1.scatter(original[:, 0], original[:, 1], original[:, 2], s=5, c=depth, cmap='viridis')
    ax1.set_xlim(-1.0, 1.0)
    ax1.set_ylim(-1.0, 1.0)
    ax1.set_zlim(-1.0, 1.0)
    ax1.set_title('Original Point Cloud')

    ax2 = fig.add_subplot(132, projection='3d')
    if use_ellipsoid:
        # Draw ellipsoids for each GMM component
        colors = plt.cm.viridis(np.linspace(0, 1, len(decoded_mu)))
        for i, (mu, sigma) in enumerate(zip(decoded_mu, decoded_sigma)):
            draw_ellipsoid(ax2, mu, sigma, color=colors[i], alpha=0.1, scale=ellipsoid_scale)
        ax2.scatter(decoded_mu[:, 0], decoded_mu[:, 1], decoded_mu[:, 2], s=30, c='black', marker='x')
    else:
        depth_means = np.linalg.norm(decoded_mu, axis=1)
        ax2.scatter(decoded_mu[:, 0], decoded_mu[:, 1], decoded_mu[:, 2], s=30, c=depth_means, cmap='viridis')
    ax2.set_xlim(-1.0, 1.0)
    ax2.set_ylim(-1.0, 1.0)
    ax2.set_zlim(-1.0, 1.0)
    ax2.set_title('GMM Components (Ellipsoids)')

    ax3 = fig.add_subplot(133, projection='3d')
    depth_sampled = np.linalg.norm(sampled_points, axis=1)
    ax3.scatter(sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2], s=5, c=depth_sampled, cmap='viridis')
    ax3.set_xlim(-1.0, 1.0)
    ax3.set_ylim(-1.0, 1.0)
    ax3.set_zlim(-1.0, 1.0)
    ax3.set_title('Reconstructed Point Cloud (Sampled)')

    plt.show()


def gmm_latent_interpolation(PGVAE, latent_dim_idx=0, input_latent=None, min=-1.0, max=1.0, steps=10):
    # draw the GMM components (means and ellipsoids)
    PGVAE.eval()
    with torch.no_grad():
        latent_vectors = []
        for alpha in np.linspace(min, max, steps):
            if input_latent is not None:
                latent = input_latent.clone().detach()
                if latent.dim() == 1:
                    latent = latent.unsqueeze(0)  # (D,) -> (1, D)
                latent = latent.to(device).float()
            else:
                latent = torch.randn(1, PGVAE.decoder.fc_in[0].in_features, device=device)

            latent[0, latent_dim_idx] = float(alpha)
            latent_vectors.append(latent)

        latent_vectors = torch.cat(latent_vectors, dim=0)  # (steps, latent_dim)
        decoded_mu, decoded_sigma, decoded_logpi, _ = PGVAE.decoder(latent_vectors)
        

    fig = plt.figure(figsize=(12, 6))
    for i in tqdm(range(steps)):
        ax = fig.add_subplot(2, steps // 2, i + 1, projection='3d')
        # Draw ellipsoids for each GMM component
        colors = plt.cm.viridis(np.linspace(0, 1, decoded_mu.size(1)))
        for j in range(decoded_mu.size(1)):
            mu = decoded_mu[i, j].cpu().numpy()
            sigma = decoded_sigma[i, j].cpu().numpy()
            draw_ellipsoid(ax, mu, sigma, color=colors[j], alpha=0.1, scale=1.5)
            ax.scatter(mu[0], mu[1], mu[2], s=30, c='black', marker='x')
    plt.tight_layout()
    plt.show()


def gmmvae_segmentation(PGVAE, data):
    PGVAE.eval()
    with torch.no_grad():
        data = data.to(device).float().unsqueeze(0)  # (1, N, 3)
        decoded_mu, levelwise_results, _, _, _, _ = PGVAE.forward_with_aux(data)
        decoded_mu = decoded_mu.squeeze(0).cpu().numpy()  # (K, 3)
        decoded_sigma = levelwise_results[-1][1].squeeze(0).cpu().numpy()  # (K, 3)
        decoded_logpi = levelwise_results[-1][2].squeeze(0).cpu().numpy()  # (K,)

    original = data.squeeze(0).cpu().numpy()

    # Assign each original point to the most likely GMM component
    x_expanded = data.unsqueeze(2)  # [1, N, 1, 3]
    mu_expanded = torch.tensor(decoded_mu).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, K, 3]
    sigma_expanded = torch.tensor(decoded_sigma).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, K, 3]
    logpi_expanded = torch.tensor(decoded_logpi).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, K]

    squared_mahal = torch.sum(((x_expanded - mu_expanded) / sigma_expanded) ** 2, dim=-1)  # [1, N, K]
    log_det = torch.sum(torch.log(sigma_expanded), dim=-1)  # [1, 1, K]
    log_gauss = -0.5 * (squared_mahal + 3.0 * np.log(2.0 * np.pi)) - log_det
    log_mix = logpi_expanded + log_gauss
    assigned_components = torch.argmax(log_mix.squeeze(0), dim=-1).cpu().numpy()  # [N]

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    depth = np.linalg.norm(original, axis=1)
    ax1.scatter(original[:, 0], original[:, 1], original[:, 2], s=5, c=depth, cmap='viridis')
    ax1.set_title('Point Cloud (Color by depth)')

    ax2 = fig.add_subplot(122, projection='3d')
    colors = plt.cm.tab10(assigned_components / np.max(assigned_components))
    ax2.scatter(original[:, 0], original[:, 1], original[:, 2], s=5, c=colors)
    ax2.set_title('Point Cloud Segmentation (Color by GMM Component)')
    plt.show()
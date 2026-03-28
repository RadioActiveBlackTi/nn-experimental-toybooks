import torch.nn as nn

from PointNet import PointNetFeature

class PointNetEncoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.pointnet_feature = PointNetFeature()
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
    
    def forward(self, x):
        # Input: (B, N, 3)
        # Output: (B, latent_dim)
        features, t3, t64 = self.pointnet_feature(x)  # (B, 1024)
        x = self.fc(features)  # (B, latent_dim)

        return x, t3, t64 # should regularize t3, t64 to orthogonal matrices

class PointNetVarEncoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.pointnet_feature = PointNetFeature()
        self.fc_mu = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.fc_logvar = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )

    def forward(self, x):
        features, t3, t64 = self.pointnet_feature(x)  # (B, 1024)
        mu = self.fc_mu(features)  # (B, latent_dim)
        logvar = self.fc_logvar(features)  # (B, latent_dim)
        return mu, logvar, t3, t64
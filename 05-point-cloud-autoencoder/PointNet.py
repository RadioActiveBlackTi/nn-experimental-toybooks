import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class STNKd(nn.Module):
    # Spatial Transformer Network, a.k.a. T-Net for point clouds
    def __init__(self, k):
        super().__init__()
        self.k = k
        self.conv1 = nn.Sequential(nn.Conv1d(k, 64, 1), nn.BatchNorm1d(64))
        self.conv2 = nn.Sequential(nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024))

        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, k * k)
        )
    
    def forward(self, x):
        # Input: (B, k, N)
        # Output: (B, k, k)
        batch_size = x.size(0)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.max(x, 2)[0]  # (B, 1024)
        x = self.fc(x)  # (B, k*k)

        identity = Variable(torch.eye(self.k, device=x.device).view(1, self.k * self.k).repeat(batch_size, 1))

        x = x + identity  # Initialize as identity
        x = x.view(-1, self.k, self.k)  # (B, k, k)
        return x

class PointNetFeature(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_transform = STNKd(3)
        self.conv1 = nn.Sequential(nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64))
        self.feature_transform = STNKd(64)
        self.conv2 = nn.Sequential(nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024))

    def forward(self, x):
        # Input: (B, N, 3)
        # Output: (B, latent_dim)
        t3 = self.input_transform(x.transpose(1, 2))  # (B, 3, 3)
        x = torch.bmm(x, t3)  # (B, N, 3)
        x = F.relu(self.conv1(x.transpose(1, 2)))  # (B, 64, N)

        t64 = self.feature_transform(x)  # (B, 64, 64)
        x = torch.bmm(x.transpose(1, 2), t64).transpose(1, 2)  # (B, 64, N)
        x = F.relu(self.conv2(x))  # (B, 128, N)
        x = F.relu(self.conv3(x))  # (B, 1024, N)
        x = torch.max(x, 2)[0]  # (B, 1024)

        return x, t3, t64 # should regularize t3, t64 to orthogonal matrices
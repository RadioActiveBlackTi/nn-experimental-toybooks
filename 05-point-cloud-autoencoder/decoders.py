import torch
import torch.nn as nn
import torch.nn.functional as F

class PointDecoder(nn.Module):
    def __init__(self, latent_dim=64, n_points=2048):
        super().__init__()
        self.n_points = n_points

        self.fc_in = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU()
        )

        self.fc_res1 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024)
        )

        self.fc_res2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024)
        )

        self.fc_out = nn.Sequential(
            nn.ReLU(),
            nn.Linear(1024, n_points * 3),
            nn.Tanh()  # normalized output range
        )

    def forward(self, x):
        # Input: (B, latent_dim)
        # Output: (B, N, 3)
        x = self.fc_in(x)
        x = x + self.fc_res1(x)
        x = x + self.fc_res2(x)
        x = self.fc_out(x)
        x = x.view(-1, self.n_points, 3)
        return x


class HierachicalSplitBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, attn_dim, branching_factor=2):
        super().__init__()
        self.bf = branching_factor
        self.attn_dim = attn_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim

        self.fc = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim * self.bf)
        )  # [..., in_dim] -> [..., hidden_dim * bf]

        # grouped 1x1 conv: each branch is processed independently
        self.Q = nn.Conv1d(hidden_dim * self.bf, attn_dim * self.bf, kernel_size=1, groups=self.bf, bias=True)
        self.K = nn.Conv1d(hidden_dim * self.bf, attn_dim * self.bf, kernel_size=1, groups=self.bf, bias=True)
        self.V = nn.Conv1d(hidden_dim * self.bf, out_dim * self.bf, kernel_size=1, groups=self.bf, bias=True)

        self.mu_head = nn.Linear(out_dim, 3)
        self.sigma_head = nn.Linear(out_dim, 3)
        self.logit_head = nn.Linear(out_dim, 1)

    def forward(self, x):
        # x: [B, Np, Din] or [B, Din]
        if x.dim() == 2:
            x = x.unsqueeze(1)
        B, Np, _ = x.shape

        h = self.fc(x)  # [B, Np, H*bf]
        h = h.reshape(B * Np, self.hidden_dim * self.bf, 1)  # [B*Np, H*bf, 1]

        # [B*Np, A*bf, 1] -> [B, Np, bf, A]
        q = self.Q(h).squeeze(-1).view(B, Np, self.bf, self.attn_dim)
        k = self.K(h).squeeze(-1).view(B, Np, self.bf, self.attn_dim)
        v = self.V(h).squeeze(-1).view(B, Np, self.bf, self.out_dim)

        # sibling attention inside each parent node
        attn_scores = torch.einsum('bnia,bnja->bnij', q, k) / (self.attn_dim ** 0.5)  # [B, Np, bf, bf]
        attn_weights = F.softmax(attn_scores, dim=-1)
        feat_child = torch.einsum('bnij,bnjo->bnio', attn_weights, v)  # [B, Np, bf, O]

        delta_mu = torch.tanh(self.mu_head(feat_child))                 # [B, Np, bf, 3]
        sigma = F.softplus(self.sigma_head(feat_child)) + 1e-6         # [B, Np, bf, 3]
        logit_local = self.logit_head(feat_child).squeeze(-1)          # [B, Np, bf] (raw logits)

        # flatten child nodes for next level
        feat_next = feat_child.reshape(B, Np * self.bf, self.out_dim)
        delta_mu = delta_mu.reshape(B, Np * self.bf, 3)
        sigma = sigma.reshape(B, Np * self.bf, 3)

        return feat_next, delta_mu, sigma, logit_local

class PointGMMDecoder(nn.Module):
    def __init__(self, latent_dim=64, attn_dim=16, value_dim=32, branch_factor=2, level=3):
        super().__init__()
        self.branch_factor = branch_factor
        self.level = level
        self.n_branches = branch_factor ** level

        self.fc_in = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU()
        )

        self.mu_init = nn.Linear(1024, 3)  # root mean

        self.hbs = nn.ModuleList([
            HierachicalSplitBlock(
                in_dim=1024 if i == 0 else value_dim,
                hidden_dim=128,
                out_dim=value_dim,
                attn_dim=attn_dim,
                branching_factor=branch_factor
            ) for i in range(level)
        ])

    def forward(self, z):
        # Input: z [B, latent_dim]
        # Output: mu [B, K, 3], sigma [B, K, 3], logpi [B, K] (K = branch_factor^level)
        B = z.size(0)
        root_feat = self.fc_in(z)                 # [B, 1024]
        feat = root_feat.unsqueeze(1)             # [B, 1, 1024]

        mu = self.mu_init(root_feat).unsqueeze(1) # [B, 1, 3]
        sigma = torch.ones_like(mu) * 0.1         # [B, 1, 3]
        logpi = torch.zeros(B, 1, device=z.device) # [B, 1]

        levelwise_results = []

        for hb in self.hbs:
            # feat_next: [B, Np*bf, D], delta/sigma_rel: [B, Np*bf, 3], logit_local: [B, Np, bf]
            feat, delta_mu, sigma_rel, logit_local = hb(feat)

            Np = mu.size(1)
            bf = self.branch_factor

            parent_mu = mu.unsqueeze(2).expand(B, Np, bf, 3).reshape(B, Np * bf, 3)
            parent_sigma = sigma.unsqueeze(2).expand(B, Np, bf, 3).reshape(B, Np * bf, 3)

            mu = parent_mu + delta_mu
            sigma = parent_sigma * sigma_rel

            # accumulate mixture logits in log-space
            child_logpi = logpi.unsqueeze(-1) + F.log_softmax(logit_local, dim=-1)  # [B, Np, bf]
            logpi = child_logpi.reshape(B, Np * bf)

            levelwise_results.append((mu, sigma, logpi))

        return mu, sigma, logpi, levelwise_results
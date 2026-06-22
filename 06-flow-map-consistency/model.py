import torch
import torch.nn as nn

class FourierFeatureFlow(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, num_frequencies=8):
        super(FourierFeatureFlow, self).__init__()
        freqs = (2.0 ** torch.arange(num_frequencies, dtype=torch.float32)) * torch.pi
        self.register_buffer("freqs", freqs)
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        self.pos_dim = input_dim * (1 + 2 * num_frequencies)
        self.time_dim = 2 * num_frequencies
        self.net = nn.Sequential(
            nn.Linear(self.pos_dim + self.time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def time_features(self, t):
        t = t.reshape(t.shape[0], -1)
        angles = t * self.freqs.view(1, -1)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

    def pos_features(self, x):
        x = x.reshape(x.shape[0], -1)
        angles = x.unsqueeze(-1) * self.freqs.view(1, 1, -1)
        fourier = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1).flatten(1)
        return torch.cat([x, fourier], dim=-1)

    def forward(self, x, t):
        t_feat = self.time_features(t)
        x_feat = self.pos_features(x)
        h = torch.cat([x_feat, t_feat], dim=-1)
        return self.net(h)

class FlowMap(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, num_frequencies=8):
        # input: xs(2dim), s, t
        # output: xt(2dim)
        super(FlowMap, self).__init__()
        freqs = (2.0 ** torch.arange(num_frequencies, dtype=torch.float32)) * torch.pi
        self.register_buffer("freqs", freqs)
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        self.pos_dim = input_dim * (1 + 2 * num_frequencies)
        self.time_dim = 2 * num_frequencies
        self.net = nn.Sequential(
            nn.Linear(self.pos_dim + 2 * self.time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )
    
    def time_features(self, t):
        t = t.reshape(t.shape[0], -1)
        angles = t * self.freqs.view(1, -1)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

    def pos_features(self, x):
        x = x.reshape(x.shape[0], -1)
        angles = x.unsqueeze(-1) * self.freqs.view(1, 1, -1)
        fourier = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1).flatten(1)
        return torch.cat([x, fourier], dim=-1)

    def velocity(self, x, s, t):
        x_feat = self.pos_features(x)
        s_feat = self.time_features(s)
        t_feat = self.time_features(t)
        h = torch.cat([x_feat, s_feat, t_feat], dim=-1)
        return self.net(h)
    
    def forward(self, x, s, t):
        return x + self.velocity(x, s, t) * (t - s)
        
        

class SelfDistilledFlowMap(FlowMap):
    def __init__(self, input_dim=2, hidden_dim=128, num_frequencies=8):
        super().__init__(input_dim, hidden_dim, num_frequencies)
        
        self.w_net = nn.Sequential(
            nn.Linear(2 * self.time_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )

    def velocity_and_w(self, x, s, t):
        v = self.velocity(x, s, t) 
        
        s_feat = self.time_features(s)
        t_feat = self.time_features(t)
        w_input = torch.cat([s_feat, t_feat], dim=-1)
        w = self.w_net(w_input)
        
        return v, w
import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.nn import GCNConv

class GCNBlurSharpen(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNBlurSharpen, self).__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.norm = torch_geometric.nn.LayerNorm(out_channels)
        self.alpha = nn.Parameter(torch.ones(out_channels) * 0.5)  # Learnable parameter to balance blur and sharpen
    
    def forward(self, x, edge_index):
        x_norm = self.norm(x)
        out = self.conv(x_norm, edge_index)

        high_freq = x_norm - out
        return x + out + self.alpha * high_freq  # residual with blur and sharpen components


class BSGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2):
        super(BSGCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2_list = nn.ModuleList(GCNBlurSharpen(hidden_channels, hidden_channels) for _ in range(num_layers - 2))
        self.conv3 = GCNConv(hidden_channels, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        for conv in self.conv2_list:
            x = conv(x, edge_index)
            x = self.relu(x)
        x = self.conv3(x, edge_index)
        return x
    
    def forward_inspect(self, x, edge_index):
        inspections = {}
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        inspections['conv1'] = x.detach().cpu()
        for i, conv in enumerate(self.conv2_list):
            x = conv(x, edge_index)
            x = self.relu(x)
            inspections[f'conv{i+2}'] = x.detach().cpu()
        x = self.conv3(x, edge_index)
        inspections['conv_out'] = x.detach().cpu()
        return x, inspections
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, GlobalAttention
from torch import nn

class TowerGNN(nn.Module):
    def __init__(self, in_channels=8, hidden_channels=64, out_channels=2, use_attention=False):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)

        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)

        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.bn3 = nn.BatchNorm1d(hidden_channels)

        self.use_attention = use_attention
        if use_attention:
            self.pool = GlobalAttention(gate_nn=nn.Linear(hidden_channels, 1))
        else:
            self.pool = global_mean_pool

        self.regressor = nn.Sequential(
            nn.Linear(hidden_channels, 64),
            nn.ReLU(),
            nn.Linear(64, out_channels)  # Outputs: [deadzone, cost]
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=0.2, training=self.training)

        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.dropout(x, p=0.2, training=self.training)

        x = F.relu(self.bn3(self.conv3(x, edge_index)))
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.pool(x, batch) if not self.use_attention else self.pool(x, batch)

        return self.regressor(x)  # Shape: [batch_size, 2]

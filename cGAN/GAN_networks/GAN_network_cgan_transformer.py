import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, BatchNorm, global_mean_pool


class Generator(nn.Module):
    def __init__(self, feature_dim, noise_dim, hidden_dim, dropout_prob=0.3):
        """
        Args:
            feature_dim: dimension of generated node features
            noise_dim:   dimension of per-node random noise
            hidden_dim:  hidden size for the GNN backbone
            dropout_prob: dropout probability (default 0.3)
        """
        super(Generator, self).__init__()

        # GNN layer
        self.conv1 = TransformerConv(feature_dim + noise_dim, hidden_dim, heads=1, dropout=0.0)
        self.bn1 = BatchNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)

        # MLP head mapping to feature_dim
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.fc3 = nn.Linear(hidden_dim // 4, hidden_dim // 4)
        self.fc4 = nn.Linear(hidden_dim // 4, feature_dim)

    def forward(self, normal_features, noise, edge_index, batch):
        """
        Args:
            normal_features: (num_nodes, feature_dim)
            noise:           (num_nodes, noise_dim)
            edge_index:      (2, num_edges)
            batch:           (num_nodes,) graph id per node (not used here)

        Returns:
            (num_nodes, feature_dim)
        """
        x = torch.cat([normal_features, noise], dim=1)
        x = F.leaky_relu(self.bn1(self.conv1(x, edge_index)), negative_slope=0.2)
        x = self.dropout(x)

        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.2)
        x = F.leaky_relu(self.fc3(x), negative_slope=0.2)
        x = F.leaky_relu(self.fc4(x), negative_slope=0.2)
        return x


class Discriminator(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        """
        Args:
            in_dim:     node feature dimension (per branch)
            hidden_dim: hidden size for the GNN/MLP
        """
        super(Discriminator, self).__init__()

        # GNN branch for structural cues
        self.conv1 = TransformerConv(in_dim * 2, hidden_dim, heads=1, dropout=0.0)

        # MLP branch on concatenated features
        self.joint_fc1 = nn.Linear(in_dim * 2, hidden_dim)

        # Graph readout and final classifier
        self.readout = global_mean_pool
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, normal_features, extreme_features, edge_index, batch):
        """
        Args:
            normal_features:  (num_nodes, in_dim)
            extreme_features: (num_nodes, in_dim)
            edge_index:       (2, num_edges)
            batch:            (num_nodes,) graph id per node

        Returns:
            (num_graphs, 1) probability that the graph is real
        """
        x = torch.cat([normal_features, extreme_features], dim=1)

        x_gnn = F.relu(self.conv1(x, edge_index))
        x_mlp = F.relu(self.joint_fc1(x))

        x_combined = x_gnn + x_mlp
        graph_feature = self.readout(x_combined, batch)
        out = torch.sigmoid(self.fc(graph_feature))
        return out

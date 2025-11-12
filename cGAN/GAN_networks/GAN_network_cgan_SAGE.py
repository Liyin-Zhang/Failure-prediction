import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, BatchNorm, global_mean_pool


class Generator(nn.Module):
    def __init__(self, feature_dim, noise_dim, hidden_dim, dropout_prob=0.3):
        """
        noise_dim: dimension of random noise
        feature_dim: dimension of generated node features
        hidden_dim: hidden size for GNN
        dropout_prob: dropout probability, default 0.3
        """
        super(Generator, self).__init__()

        # GNN layer
        self.conv1 = SAGEConv(feature_dim + noise_dim, hidden_dim)
        self.bn1 = BatchNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)

        # MLP head mapping to feature_dim
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.fc3 = nn.Linear(hidden_dim // 4, hidden_dim // 4)
        self.fc4 = nn.Linear(hidden_dim // 4, feature_dim)

        # Optional output activation for [0, 1] range
        self.activation = nn.Sigmoid()

    def forward(self, normal_features, noise, edge_index, batch):
        """
        normal_features: (num_nodes, feature_dim)
        noise: (num_nodes, noise_dim)
        edge_index: (2, num_edges)
        batch: (num_nodes,)
        """
        x = torch.cat([normal_features, noise], dim=1)

        x = F.leaky_relu(self.bn1(self.conv1(x, edge_index)), negative_slope=0.2)
        x = self.dropout(x)

        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.2)
        x = F.leaky_relu(self.fc3(x), negative_slope=0.2)
        x = F.leaky_relu(self.fc4(x), negative_slope=0.2)

        # x = self.activation(x)
        return x  # (num_nodes, feature_dim)


class Discriminator(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(Discriminator, self).__init__()

        # GNN branch for structural features
        self.conv1 = SAGEConv(in_dim * 2, hidden_dim)

        # MLP branch for joint features
        self.joint_fc1 = nn.Linear(in_dim * 2, hidden_dim)

        # Graph level readout and final classifier
        self.readout = global_mean_pool
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, normal_features, extreme_features, edge_index, batch):
        """
        normal_features: (num_nodes, in_dim)
        extreme_features: (num_nodes, in_dim)
        edge_index: (2, num_edges)
        batch: (num_nodes,)
        """
        x = torch.cat([normal_features, extreme_features], dim=1)

        x_gnn = F.relu(self.conv1(x, edge_index))
        x_mlp = F.relu(self.joint_fc1(x))

        x_combined = x_gnn + x_mlp
        graph_feature = self.readout(x_combined, batch)  # (num_graphs, hidden_dim)

        out = self.fc(graph_feature)                     # (num_graphs, 1)
        out = torch.sigmoid(out)
        return out

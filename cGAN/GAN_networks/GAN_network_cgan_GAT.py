import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool, BatchNorm, GINConv
from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


class Generator(nn.Module):
    def __init__(self, feature_dim, noise_dim, hidden_dim, dropout_prob=0.3):
        """
        noise_dim: dimension of the random noise
        feature_dim: dimension of generated node features
        hidden_dim: hidden dimension of the GNN backbone
        dropout_prob: dropout probability
        """
        super(Generator, self).__init__()

        # 1. GNN layer
        self.conv1 = GATConv(feature_dim + noise_dim, hidden_dim, heads=2, concat=False, dropout=0.0)

        # 2. Batch Normalization
        self.bn1 = BatchNorm(hidden_dim)
        self.bn2 = BatchNorm(hidden_dim)
        self.bn3 = BatchNorm(hidden_dim)

        # 3. Dropout
        self.dropout = nn.Dropout(dropout_prob)

        # 4. Final MLP to map to feature_dim
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.fc3 = nn.Linear(hidden_dim // 4, hidden_dim // 4)
        self.fc4 = nn.Linear(hidden_dim // 4, feature_dim)

        # 5. Optional output activation
        self.activation = nn.Sigmoid()  # keep output in [0, 1] if needed

    def forward(self, normal_features, noise, edge_index, batch):
        """
        normal_features: (num_nodes, feature_dim)
        noise: (num_nodes, noise_dim)
        edge_index: (2, num_edges)
        batch: (num_nodes,)
        """
        # concatenate normal features and noise
        x = torch.cat([normal_features, noise], dim=1)

        # GNN propagation + BN + Dropout
        x = F.leaky_relu(self.bn1(self.conv1(x, edge_index)), negative_slope=0.2)
        x = self.dropout(x)

        # MLP refinement
        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.2)

        # produce final node features
        x = F.leaky_relu(self.fc3(x), negative_slope=0.2)
        x = F.leaky_relu(self.fc4(x), negative_slope=0.2)

        # x = self.activation(x)  # enable if you need outputs in [0, 1]
        return x  # (num_nodes, feature_dim)


class Discriminator(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(Discriminator, self).__init__()

        # 1. GNN branch for structural information
        self.conv1 = GATConv(in_dim * 2, hidden_dim, heads=2, concat=False, dropout=0.0)

        # 2. MLP branch on concatenated features
        self.joint_fc1 = nn.Linear(in_dim * 2, hidden_dim)
        self.joint_fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.lin1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.lin2 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.lin3 = nn.Linear(hidden_dim // 4, 1)

        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()

        # 3. Graph level readout
        self.readout = global_mean_pool

        # 4. Final decision on the graph feature
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, normal_features, extreme_features, edge_index, batch):
        """
        normal_features: (num_nodes, in_dim)
        extreme_features: (num_nodes, in_dim)
        edge_index: (2, num_edges)
        batch: (num_nodes,)
        """
        # concatenate inputs
        x = torch.cat([normal_features, extreme_features], dim=1)

        # structural branch
        x_gnn = F.relu(self.conv1(x, edge_index))

        # joint feature branch
        x_mlp = F.relu(self.joint_fc1(x))

        # fuse branches
        x_combined = x_mlp + x_gnn

        # graph level aggregation
        graph_feature = self.readout(x_combined, batch)  # (num_graphs, hidden_dim)

        # final decision
        out = self.fc(graph_feature)  # (num_graphs, 1)
        out = torch.sigmoid(out)
        return out

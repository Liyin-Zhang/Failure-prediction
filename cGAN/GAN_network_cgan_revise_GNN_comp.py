import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    SAGEConv, GATConv, GINConv, TransformerConv,
    global_mean_pool, BatchNorm
)

# ------------ 工具：按名称创建一层“输入维 -> 输出维”的图卷积 ------------
def make_gnn_layer(kind: str, in_dim: int, out_dim: int, *, gat_heads: int = 4):
    """
    kind: 'sage' | 'gat' | 'gin' | 'transformer'
    这里的 TransformerConv 版本不使用 edge_attr（与你的要求一致）
    """
    kind = (kind or "sage").lower()
    if kind == "sage":
        return SAGEConv(in_dim, out_dim)
    elif kind == "gat":
        # concat=False 以保证输出维度 = out_dim
        return GATConv(in_dim, out_dim, heads=gat_heads, concat=False, dropout=0.0)
    elif kind == "gin":
        # GIN 需要一个 MLP 聚合器
        mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim)
        )
        return GINConv(mlp)
    elif kind == "transformer":
        # 不使用 edge_attr 的 TransformerConv
        return TransformerConv(in_channels=in_dim, out_channels=out_dim, heads=1, dropout=0.0)
    else:
        raise ValueError(f"Unsupported gnn kind: {kind}. Choose from sage|gat|gin|transformer.")


# =========================== Generator ===========================
class Generator(nn.Module):
    def __init__(self, feature_dim, noise_dim, hidden_dim, dropout_prob=0.3,
                 gnn_type: str = "sage", gat_heads: int = 4):
        """
        gnn_type: 'sage' | 'gat' | 'gin' | 'transformer'
        """
        super().__init__()
        self.gnn_type = gnn_type.lower()

        # 1) 图卷积骨干（可切换；默认与原实现一致：SAGE）
        self.conv1 = make_gnn_layer(self.gnn_type, feature_dim + noise_dim, hidden_dim,
                                    gat_heads=gat_heads)

        # 2) BN / Dropout
        self.bn1 = BatchNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)

        # 3) MLP 头
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.fc3 = nn.Linear(hidden_dim // 4, hidden_dim // 4)
        self.fc4 = nn.Linear(hidden_dim // 4, feature_dim)

        # self.activation = nn.Sigmoid()  # 如需限制到[0,1]可放开

    def forward(self, normal_features, noise, edge_index, batch):
        """
        noise: (num_nodes, noise_dim)
        edge_index: (2, E)
        batch: (num_nodes,)
        """
        x = torch.cat([normal_features, noise], dim=1)  # [N, feature+noise]

        # GNN + BN + Dropout
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(self.bn1(x), negative_slope=0.2)
        x = self.dropout(x)

        # MLP
        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.2)
        x = F.leaky_relu(self.fc3(x), negative_slope=0.2)
        x = F.leaky_relu(self.fc4(x), negative_slope=0.2)
        # x = self.activation(x)
        return x  # (num_nodes, feature_dim)


# ========================== Discriminator ==========================
class Discriminator(nn.Module):
    def __init__(self, in_dim, hidden_dim,
                 gnn_type: str = "sage", gat_heads: int = 4):
        """
        gnn_type: 'sage' | 'gat' | 'gin' | 'transformer'
        """
        super().__init__()
        self.gnn_type = gnn_type.lower()

        # 1) GNN 分支（将 normal/extreme 拼接后送入）
        self.conv1 = make_gnn_layer(self.gnn_type, in_dim * 2, hidden_dim, gat_heads=gat_heads)
        self.bn1   = BatchNorm(hidden_dim)

        # 2) MLP 分支：直接对拼接后的节点特征建模（与 GNN 分支融合）
        self.joint_fc1 = nn.Linear(in_dim * 2, hidden_dim)
        self.joint_fc2 = nn.Linear(hidden_dim, hidden_dim)

        # 3) 图级读出
        self.readout = global_mean_pool

        # 4) 判别头
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, normal_features, extreme_features, edge_index, batch):
        """
        normal_features / extreme_features: [num_nodes, in_dim]
        edge_index: (2, E)
        batch: (num_nodes,)
        """
        x_cat = torch.cat([normal_features, extreme_features], dim=1)  # [N, 2*in_dim]

        # GNN 分支
        x_gnn = self.conv1(x_cat, edge_index)
        x_gnn = F.relu(self.bn1(x_gnn))

        # MLP 分支
        x_mlp = F.relu(self.joint_fc1(x_cat))
        x_mlp = F.relu(self.joint_fc2(x_mlp))

        # 融合（简单相加，和你现有逻辑一致）
        x_combined = x_gnn + x_mlp

        # 图级读出 -> 判别
        graph_feature = self.readout(x_combined, batch)  # [num_graphs, hidden_dim]
        out = torch.sigmoid(self.fc(graph_feature))      # [num_graphs, 1]
        return out

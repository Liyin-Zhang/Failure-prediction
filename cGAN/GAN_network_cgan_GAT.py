import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv,global_mean_pool, BatchNorm,GINConv
from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


class Generator(nn.Module):
    def __init__(self, feature_dim,noise_dim, hidden_dim, dropout_prob=0.3):
        """
        noise_dim: 随机噪声的维度
        in_dim: 生成的节点特征维度
        hidden_dim: GNN 隐藏层维度
        dropout_prob: Dropout 的概率，默认 0.3
        """
        super(Generator, self).__init__()

        # 1️⃣ GNN 层（3 层 SAGEConv）
        self.conv1 = GATConv(feature_dim+noise_dim, hidden_dim, heads=2, concat=False, dropout=0.0) #############################
        # self.conv2 = GCNConv(hidden_dim, hidden_dim)
        # self.conv3 = GCNConv(hidden_dim, hidden_dim)

        # 2️⃣ Batch Normalization 层
        self.bn1 = BatchNorm(hidden_dim)
        self.bn2 = BatchNorm(hidden_dim)
        self.bn3 = BatchNorm(hidden_dim)

        # 3️⃣ Dropout 防止过拟合
        self.dropout = nn.Dropout(dropout_prob)

        # 4️⃣ 最终的全连接层（映射到 in_dim 维度）
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.fc3 = nn.Linear(hidden_dim // 4, hidden_dim // 4)  # 最终输出
        self.fc4 = nn.Linear(hidden_dim // 4, feature_dim)  # 最终输出

        # 5️⃣ Tanh 激活函数，使输出范围在 [-1,1]
        self.activation = nn.Sigmoid()  # 让 G 输出 [0,1]

    def forward(self, normal_features,  noise, edge_index, batch):
        """
        noise: (num_nodes, noise_dim) - 每个节点的随机噪声
        edge_index: (2, num_edges) - 图的边索引
        batch: (num_nodes,) - 每个节点对应的图 ID
        """

        x = torch.cat([normal_features, noise], dim=1)  # 连接一般天气特征和噪声

        # 1️⃣ 通过 GNN 传播 + BatchNorm + Dropout
        x = F.leaky_relu(self.bn1(self.conv1(x, edge_index)), negative_slope=0.2)
        x = self.dropout(x)

        # 2️⃣ 通过 MLP 进一步映射，提高非线性表达能力
        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)

        x = F.leaky_relu(self.fc2(x), negative_slope=0.2)

        # 3️⃣ 生成最终节点特征
        x =  F.leaky_relu(self.fc3(x), negative_slope=0.2)

        # 3️⃣ 生成最终节点特征
        x = F.leaky_relu(self.fc4(x), negative_slope=0.2)

        # x = self.activation(x)#*2-1  # 归一化到 [-1,1]

        return x  # (num_nodes, in_dim)


class Discriminator(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(Discriminator, self).__init__()

        # 1️⃣ GNN 处理节点特征

        self.conv1 =  GATConv(in_dim*2, hidden_dim, heads=2, concat=False, dropout=0.0)  # 第一层 GCN
        # self.conv2 = SAGEConv(hidden_dim, hidden_dim)  # 第二层 GCN
        # self.conv3 = SAGEConv(hidden_dim, hidden_dim)  # 第二层 GCN

        # 2️⃣ MLP 直接判别 `x` 的联合分布
        self.joint_fc1 = nn.Linear(in_dim*2, hidden_dim)  # 处理整个特征向量
        self.joint_fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.lin1 = torch.nn.Linear(hidden_dim, hidden_dim//2)
        self.lin2 = torch.nn.Linear(hidden_dim//2, hidden_dim//4)
        self.lin3 = torch.nn.Linear( hidden_dim//4, 1)

        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.ReLU()

        # 3️⃣ 图级聚合
        self.readout = global_mean_pool  # 这里用 `mean_pool`，也可以用 `max_pool`

        # 4️⃣ 判别整个 `x`
        self.fc = nn.Linear(hidden_dim, 1)  # 最终判别 (真/假)


    def forward(self, normal_features, extreme_features, edge_index, batch):
        """
        x: (num_nodes, feature_dim) - 节点特征矩阵
        edge_index: (2, num_edges) - 边索引
        batch: (num_nodes,) - 每个节点属于哪个图的索引
        """

        x = torch.cat([normal_features, extreme_features], dim=1)  # 连接输入

        # 1️⃣ **让 GNN 处理结构信息**
        x_gnn = F.relu(self.conv1(x, edge_index))

        # 2️⃣ **同时用 MLP 处理 `x` 的联合特征**
        x_mlp = F.relu(self.joint_fc1(x))  # 直接输入 `x`，学习特征之间的关系

        # 3️⃣ **融合 GNN 处理后的 `x_gnn` 和 MLP 处理后的 `x_mlp`**
        x_combined =x_mlp+ x_gnn   # 让 `D` 既关注邻居特征，也关注 `x` 内部关系

        # 4️⃣ **全图级别的特征聚合**
        graph_feature = self.readout(x_combined, batch)  # (num_graphs, hidden_dim)

        # 5️⃣ **最终判别**
        out = self.fc(graph_feature)  # (num_graphs, 1)
        out = torch.sigmoid(out)  # 转换为概率

        return out
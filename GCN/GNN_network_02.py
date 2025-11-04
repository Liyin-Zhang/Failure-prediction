# 单纯有向图
# GNN02_network.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATConv,
    TransformerConv,              # ← 使用 TransformerConv 以接入 edge_attr
    global_mean_pool as GAP,
    global_max_pool  as GMP
)
from GNN_args import *

class NetA_NodeOnly(nn.Module):
    """
    方案A（节点图两阶段，无支路图）——无向→有向，均在节点层面计算：
      Stage-1（无向，不池化）: GATConv ×2 -> H_und  (N, d)
      Stage-2（有向，不池化）: TransformerConv ×2 (使用 edge_attr) -> H_dir (N, d)
      Readout（图级）: 对 H_dir 做 GAP + GMP -> MLP -> Sigmoid 概率
      （可选）一致性正则: λ * || H_und - stopgrad(H_dir) ||_F^2  （λ=0 时关闭）
    """
    def __init__(self, lambda_consistency: float = 0.0, hidden_dim: int = 256, edge_dim: int = 6, dropout_p: float = 0.5):
        super().__init__()
        self.lambda_consistency = float(lambda_consistency)
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.dropout_p = dropout_p

        # Embedding 输入（与你的数据离散化方案一致）
        self.item_embedding = nn.Embedding(num_embeddings=max_node_number, embedding_dim=embed_dim)
        d_in = embed_dim * feature_num
        d_h  = hidden_dim

        # --- Stage-1: 无向（节点级，不池化） ---
        # self.und_conv1 = GATConv(d_in, d_h)
        # self.und_conv2 = GATConv(d_h, d_h)
        self.und_conv1 = TransformerConv(d_in, d_h, edge_dim=edge_dim, heads=1, dropout=0.0)
        self.und_conv2 = TransformerConv(d_h, d_h, edge_dim=edge_dim, heads=1, dropout=0.0)

        # --- Stage-2: 有向（节点级，不池化，使用边特征） ---
        # heads=1 保持与维度一致；需要多头可自行调整并相应修改维度
        self.dir_conv1 = TransformerConv(d_in, d_h, edge_dim=edge_dim, heads=1, dropout=0.0)
        self.dir_conv2 = TransformerConv(d_h, d_h, edge_dim=edge_dim, heads=1, dropout=0.0)

        # --- Readout & 分类头 ---
        readout_dim = 2 * d_h  # GAP + GMP
        self.lin1 = nn.Linear(readout_dim, 256)
        self.lin2 = nn.Linear(256, 128)
        self.lin3 = nn.Linear(128, 1)

    def forward(self, data, return_reg: bool = False):
        """
        需要 data 字段：
          - x: [N, feature_num]（离散索引）
          - edge_index_und: [2, E_und]（无向边）
          - edge_index_dir: [2, E_dir]（有向边）
          - edge_attr: [E_dir, edge_dim]（边特征：WIN, PRE, TMP, PRS, SHU, flow）
          - batch: [N]（节点到图的映射）
        """
        x = data.x
        # ei_und = data.edge_index_und
        ei_dir = data.edge_index_dir
        eattr  = data.edge_attr
        batch_nodes = data.batch

        # --- Embedding 展开 ---
        x = self.item_embedding(x)                  # [N, feature_num, embed_dim]
        x = x.reshape(-1, embed_dim * feature_num)  # [N, d_in]

        # # --- Stage-1: 无向 ---
        # h = F.relu(self.und_conv1(x, ei_und))
        # H_und = F.relu(self.und_conv2(h, ei_und))   # [N, d_h]

        # --- Stage-2: 有向（以 H_und 为输入，使用 edge_attr） ---
        z = F.relu(self.dir_conv1(x, ei_dir, edge_attr=eattr))
        H_dir = F.relu(self.dir_conv2(z, ei_dir, edge_attr=eattr))     # [N, d_h]

        # --- Readout: 对节点做 GAP + GMP ---
        readout = torch.cat([GAP(H_dir, batch_nodes), GMP(H_dir, batch_nodes)], dim=1)  # [B, 2*d_h]

        # --- 图级分类头 ---
        out = F.relu(self.lin1(readout))
        out = F.relu(self.lin2(out))
        out = F.dropout(out, p=self.dropout_p, training=self.training)
        out = torch.sigmoid(self.lin3(out)).squeeze(1)  # [B], 概率

        if (not return_reg) or self.lambda_consistency <= 0.0:
            return out

        # （可选）一致性正则：约束 H_und 与 H_dir 不要偏离过大（稳定器）
        # reg = (H_und - H_dir.detach()).pow(2).sum(dim=1).mean()
        return out, self.lambda_consistency# * reg

# 无向有向并联
# GNN02_network.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    TransformerConv,          # 无向/有向都用它来吃 edge_attr
    global_mean_pool as GAP,
    global_max_pool  as GMP,
)
from torch_geometric.utils import to_undirected
from GNN_args import *

class NetA_NodeOnly(nn.Module):
    """
    节点图两阶段（无向→有向），无池化；支持门控加权融合：
      - Stage-1 无向：TransformerConv ×2 (使用无向 edge_attr_und)
      - Stage-2 有向：TransformerConv ×2 (使用有向 edge_attr)
      - Readout：分别对 H_und / H_dir 做 GAP+GMP，再门控融合（feature 或 logit）
      - 可选一致性正则: λ * || H_und - stopgrad(H_dir) ||_F^2
    需要 data 字段：
      - x: [N, feature_num]
      - edge_index_dir: [2, E_dir]
      - edge_attr: [E_dir, edge_dim]  （6维：WIN, PRE, TMP, PRS, SHU, flow）
      - （可选）edge_index_und / edge_attr_und：若未提供，将由有向边现场生成（按端对端合并，reduce='mean'）
      - batch: [N]
    """
    def __init__(self, lambda_consistency: float = 0.0, hidden_dim: int = 256,
                 fuse_mode: str = 'feature', dropout_p: float = 0.5, edge_dim: int = 6):
        super().__init__()
        assert fuse_mode in ('logit', 'feature')
        self.lambda_consistency = float(lambda_consistency)
        self.hidden_dim = hidden_dim
        self.fuse_mode = fuse_mode
        self.dropout_p = dropout_p
        self.edge_dim = edge_dim  # 与 edge_attr 列数一致

        # 输入仍采用你的 Embedding 设计
        self.item_embedding = nn.Embedding(num_embeddings=max_node_number, embedding_dim=embed_dim)
        d_in = embed_dim * feature_num
        d_h  = hidden_dim

        # --- Stage-1: 无向（节点级，不池化，使用边特征） ---
        self.und_conv1 = TransformerConv(d_in, d_h, edge_dim=edge_dim, heads=1, dropout=0.0)
        self.und_conv2 = TransformerConv(d_h, d_h, edge_dim=edge_dim, heads=1, dropout=0.0)

        # --- Stage-2: 有向（节点级，不池化，使用边特征） ---
        self.dir_conv1 = TransformerConv(d_in, d_h, edge_dim=edge_dim, heads=1, dropout=0.0)
        self.dir_conv2 = TransformerConv(d_h, d_h, edge_dim=edge_dim, heads=1, dropout=0.0)

        # --- 读出 & 分类头 ---
        readout_dim = 2 * d_h  # GAP+GMP
        self.gate_mlp = nn.Sequential(
            nn.Linear(2 * readout_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        if self.fuse_mode == 'feature':
            self.head = nn.Sequential(
                nn.Linear(readout_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(128, 1)
            )
        else:
            self.head_und = nn.Sequential(
                nn.Linear(readout_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(128, 1)
            )
            self.head_dir = nn.Sequential(
                nn.Linear(readout_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(128, 1)
            )

    def _get_undirected_with_attrs(self, data):
        """
        优先使用 data.edge_index_und / data.edge_attr_und；
        若未提供，则基于有向边 (edge_index_dir, edge_attr) 现场生成：
          to_undirected(..., reduce='mean')
        """
        if hasattr(data, "edge_index_und") and hasattr(data, "edge_attr_und") \
           and data.edge_index_und is not None and data.edge_attr_und is not None:
            return data.edge_index_und, data.edge_attr_und

        # 兜底：用有向边聚合成无向边与其特征（同一无向对合并取均值）
        ei_und, eattr_und = to_undirected(
            data.edge_index_dir, edge_attr=data.edge_attr, reduce='mean'
        )
        return ei_und, eattr_und

    def forward(self, data, return_reg: bool = False):
        x = data.x
        ei_dir = data.edge_index_dir
        eattr  = data.edge_attr            # [E_dir, edge_dim]
        batch_nodes = data.batch

        # --- 准备无向边与无向边特征 ---
        ei_und, eattr_und = self._get_undirected_with_attrs(data)

        # --- Embedding 展开 ---
        x = self.item_embedding(x)                  # [N, feature_num, embed_dim]
        x = x.reshape(-1, embed_dim * feature_num)  # [N, d_in]

        # Stage-1: 无向（使用 eattr_und）
        h = F.relu(self.und_conv1(x, ei_und, edge_attr=eattr_und))
        H_und = F.relu(self.und_conv2(h, ei_und, edge_attr=eattr_und))   # [N, d_h]

        # Stage-2: 有向（以原始 x 为输入，使用 edge_attr）
        # 如需“串联”效果，可把下一行的输入改为 H_und
        z = F.relu(self.dir_conv1(x, ei_dir, edge_attr=eattr))
        H_dir = F.relu(self.dir_conv2(z, ei_dir, edge_attr=eattr))       # [N, d_h]

        # Readout（分别对 H_und / H_dir）
        r_und = torch.cat([GAP(H_und, batch_nodes), GMP(H_und, batch_nodes)], dim=1)  # [B, 2*d_h]
        r_dir = torch.cat([GAP(H_dir, batch_nodes), GMP(H_dir, batch_nodes)], dim=1)  # [B, 2*d_h]

        # 门控（按图生成 0~1 的权重）
        gate_logit = self.gate_mlp(torch.cat([r_und, r_dir], dim=1))   # [B, 1]
        g = torch.sigmoid(gate_logit)                                  # [B, 1]

        if self.fuse_mode == 'feature':
            r = g * r_dir + (1.0 - g) * r_und                          # [B, 2*d_h]
            logit = self.head(r).squeeze(1)                            # [B]
        else:
            logit_und = self.head_und(r_und).squeeze(1)                # [B]
            logit_dir = self.head_dir(r_dir).squeeze(1)                # [B]
            logit = (g.squeeze(1) * logit_dir) + ((1.0 - g.squeeze(1)) * logit_und)

        out = torch.sigmoid(logit)                                     # 概率 [B]

        if (not return_reg) or self.lambda_consistency <= 0.0:
            return out

        # 一致性正则（节点级）——如需启用把 lambda_consistency 设为 >0
        reg = (H_und - H_dir.detach()).pow(2).sum(dim=1).mean()
        return out, self.lambda_consistency #* reg

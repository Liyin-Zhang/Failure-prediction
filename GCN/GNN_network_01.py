# 单纯无向图（动态边权）

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    TransformerConv,              # 无向/有向都用它来吃 edge_attr
    global_mean_pool as GAP,
    global_max_pool  as GMP
)
from torch_geometric.utils import to_undirected
from GNN_args import *

class NetA_NodeOnly(nn.Module):
    """
    两阶段（无向→有向），节点层，不池化：
      - Stage-1 无向：TransformerConv×2，使用无向 edge_attr_und
      - Stage-2 有向：TransformerConv×2，使用有向 edge_attr
      - Readout：对 H_dir 做 GAP+GMP → MLP → 概率
      - 一致性正则：λ * || H_und - stopgrad(H_dir) ||_F^2（λ=0 关闭）
    期望 Data 字段：
      x: [N, feature_num]（离散索引，经 embedding 展开）
      edge_index_dir: [2, E_dir]
      edge_attr: [E_dir, edge_dim]（6维：WIN, PRE, TMP, PRS, SHU, flow）
      （可选）edge_index_und / edge_attr_und：若未提供，将由 dir 边现场生成无向版本
      batch: [N]
    """
    def __init__(self, lambda_consistency: float = 0.0,
                 hidden_dim: int = 256, edge_dim: int = 6, dropout_p: float = 0.5):
        super().__init__()
        self.lambda_consistency = float(lambda_consistency)
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.dropout_p = dropout_p

        # 节点输入 embedding（保持你的离散化方案）
        self.item_embedding = nn.Embedding(num_embeddings=max_node_number, embedding_dim=embed_dim)
        d_in = embed_dim * feature_num
        d_h  = hidden_dim

        # --- Stage-1: 无向（节点级，不池化，使用边特征） ---
        self.und_conv1 = TransformerConv(d_in, d_h, edge_dim=edge_dim, heads=1, dropout=0.0)
        self.und_conv2 = TransformerConv(d_h,  d_h, edge_dim=edge_dim, heads=1, dropout=0.0)

        # --- Stage-2: 有向（节点级，不池化，使用边特征） ---
        self.dir_conv1 = TransformerConv(d_h, d_h, edge_dim=edge_dim, heads=1, dropout=0.0)
        self.dir_conv2 = TransformerConv(d_h, d_h, edge_dim=edge_dim, heads=1, dropout=0.0)

        # --- Readout & 分类头 ---
        readout_dim = 2 * d_h  # GAP + GMP
        self.lin1 = nn.Linear(readout_dim, 256)
        self.lin2 = nn.Linear(256, 128)
        self.lin3 = nn.Linear(128, 1)

    def _get_undirected_with_attrs(self, data):
        """
        优先使用 data.edge_index_und / data.edge_attr_und；
        若未提供，则基于有向边 (edge_index_dir, edge_attr) 现场生成：
          to_undirected(..., reduce='mean')
        """
        has_provided = hasattr(data, "edge_index_und") and hasattr(data, "edge_attr_und")
        if has_provided and data.edge_index_und is not None and data.edge_attr_und is not None:
            return data.edge_index_und, data.edge_attr_und

        # 兜底：由有向边构建无向边及其特征（按端对端合并，默认求均值）
        ei_dir = data.edge_index_dir
        eattr  = data.edge_attr
        ei_und_new, eattr_und = to_undirected(ei_dir, edge_attr=eattr, reduce='mean')
        return ei_und_new, eattr_und

    def forward(self, data, return_reg: bool = False):
        x = data.x
        ei_dir = data.edge_index_dir
        eattr  = data.edge_attr
        batch_nodes = data.batch

        # 无向边 & 无向边特征（现场或使用提供的）
        ei_und, eattr_und = self._get_undirected_with_attrs(data)

        # Embedding 展开
        x = self.item_embedding(x)                  # [N, feature_num, embed_dim]

        # ★ 如果解释时传了 node_gate（N,1），就做节点级门控
        gate = getattr(data, 'node_gate', None)
        if gate is not None:
            # gate: [N,1] -> [N,1,1]，对每个节点的 embedding 全通道缩放
            x = x * gate.view(-1, 1, 1)

        x = x.reshape(-1, embed_dim * feature_num)  # [N, d_in]

        # --- Stage-1: 无向（使用 eattr_und） ---
        h = F.relu(self.und_conv1(x, ei_und, edge_attr=eattr_und))
        H_und = F.relu(self.und_conv2(h, ei_und, edge_attr=eattr_und))  # [N, d_h]

        # # --- Stage-2: 有向（以 H_und 为输入，使用 eattr） ---
        # z = F.relu(self.dir_conv1(H_und, ei_dir, edge_attr=eattr))
        # H_dir = F.relu(self.dir_conv2(z,     ei_dir, edge_attr=eattr))  # [N, d_h]

        # --- Readout: 对节点做 GAP + GMP ---
        readout = torch.cat([GAP(H_und, batch_nodes), GMP(H_und, batch_nodes)], dim=1)  # [B, 2*d_h]

        # --- 图级分类头 ---
        out = F.relu(self.lin1(readout))
        out = F.relu(self.lin2(out))
        out = F.dropout(out, p=self.dropout_p, training=self.training)
        out = torch.sigmoid(self.lin3(out)).squeeze(1)  # [B], 概率

        if (not return_reg) or self.lambda_consistency <= 0.0:
            return out

        # 一致性正则（稳定器）：λ * || H_und - stopgrad(H_dir) ||_F^2
        # reg = (H_und - H_dir.detach()).pow(2).sum(dim=1).mean()
        return out, self.lambda_consistency

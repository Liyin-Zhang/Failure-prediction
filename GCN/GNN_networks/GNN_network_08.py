"""

Parallel Undirected & Directed (GINE) design: starting from the same embedded node features,
two GINEConv layers encode the undirected graph into H_und and another two GINEConv layers
encode the directed graph into H_dir. Both are pooled via GAP/GMP and fused by a gating MLP
(feature-level or logit-level) to produce a binary probability. An optional node-level
consistency regularizer is supported.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GINEConv,
    global_mean_pool as GAP,
    global_max_pool as GMP,
)
from torch_geometric.utils import to_undirected


def mlp(in_dim, out_dim, hidden=None, dropout=0.0):
    """Two-layer MLP helper for GINE."""
    h = out_dim if hidden is None else hidden
    return nn.Sequential(
        nn.Linear(in_dim, h),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(h, out_dim),
    )


class NetA_NodeOnly(nn.Module):
    """
    Parallel (Undirected & Directed) node-level encoder with GINE and gated fusion.

    Expected `data` fields:
      - x: LongTensor [N, feature_num] (discrete indices expanded via embedding)
      - edge_index_dir: LongTensor [2, E_dir] (directed edges)
      - edge_attr: FloatTensor [E_dir, edge_dim] (e.g., WIN, PRE, TMP, PRS, SHU, flow)
      - (optional) edge_index_und / edge_attr_und: if not provided, undirected edges/attrs
        are derived from directed ones via to_undirected(..., reduce='mean')
      - batch: LongTensor [N] (node-to-graph mapping)

    Notes:
      - `max_node_number`, `embed_dim`, and `feature_num` must be defined at module scope
        (e.g., injected from your args/config before constructing the model).
      - Set `lambda_consistency > 0` to enable the node-level consistency regularizer.
    """
    def __init__(
        self,
        lambda_consistency: float = 0.0,
        hidden_dim: int = 256,
        fuse_mode: str = 'feature',   # 'feature' or 'logit'
        dropout_p: float = 0.5,
        edge_dim: int = 6,
    ):
        super().__init__()
        assert fuse_mode in ('logit', 'feature')
        self.lambda_consistency = float(lambda_consistency)
        self.hidden_dim = hidd_

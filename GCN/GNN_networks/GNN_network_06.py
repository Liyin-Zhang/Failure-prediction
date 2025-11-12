"""

Parallel Undirected & Directed design: starting from the same embedded node features,
two TransformerConv stacks encode the undirected and directed graphs to obtain H_und and H_dir.
Both branches are pooled via GAP/GMP and fused by a gating MLP (feature-level or logit-level)
to produce binary probabilities. An optional node-level consistency regularizer is supported.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    TransformerConv,
    global_mean_pool as GAP,
    global_max_pool as GMP,
)
from torch_geometric.utils import to_undirected


class NetA_NodeOnly(nn.Module):
    """
    Parallel (Undirected & Directed) node-level encoder without internal pooling.

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
        self.hidden_dim = hidden_dim
        self.fuse_mode = fuse_mode
        self.dropout_p = dropout_p
        self.edge_dim = edge_dim  # must match the number of columns in edge_attr

        # Embedding for discrete node features
        self.item_embedding = nn.Embedding(num_embeddings=max_node_number, embedding_dim=embed_dim)
        d_in = embed_dim * feature_num
        d_h = hidden_dim

        # Undirected branch (uses undirected edge attributes)
        self.und_conv1 = TransformerConv(d_in, d_h, edge_dim=edge_dim, heads=1, dropout=0.0)
        self.und_conv2 = TransformerConv(d_h,  d_h, edge_dim=edge_dim, heads=1, dropout=0.0)

        # Directed branch (uses directed edge attributes)
        self.dir_conv1 = TransformerConv(d_in, d_h, edge_dim=edge_dim, heads=1, dropout=0.0)
        self.dir_conv2 = TransformerConv(d_h, d_h, edge_dim=edge_dim, heads=1, dropout=0.0)

        # Readout and heads
        readout_dim = 2 * d_h  # GAP + GMP
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
        Return undirected edges and attributes. Prefer provided fields
        (edge_index_und / edge_attr_und); otherwise derive them from directed
        edges/attrs via to_undirected(..., reduce='mean').
        """
        if (
            hasattr(data, "edge_index_und") and hasattr(data, "edge_attr_und")
            and data.edge_index_und is not None and data.edge_attr_und is not None
        ):
            return data.edge_index_und, data.edge_attr_und

        ei_und, eattr_und = to_undirected(
            data.edge_index_dir, edge_attr=data.edge_attr, reduce='mean'
        )
        return ei_und, eattr_und

    def forward(self, data, return_reg: bool = False):
        x = data.x
        ei_dir = data.edge_index_dir
        eattr = data.edge_attr
        batch_nodes = data.batch

        # Prepare undirected edges and attributes
        ei_und, eattr_und = self._get_undirected_with_attrs(data)

        # Embedding expansion
        x = self.item_embedding(x)                  # [N, feature_num, embed_dim]
        x = x.reshape(-1, embed_dim * feature_num)  # [N, d_in]

        # Undirected branch
        h = F.relu(self.und_conv1(x, ei_und, edge_attr=eattr_und))
        H_und = F.relu(self.und_conv2(h, ei_und, edge_attr=eattr_und))   # [N, d_h]

        # Directed branch
        z = F.relu(self.dir_conv1(x, ei_dir, edge_attr=eattr))
        H_dir = F.relu(self.dir_conv2(z, ei_dir, edge_attr=eattr))       # [N, d_h]

        # Readout (both branches)
        r_und = torch.cat([GAP(H_und, batch_nodes), GMP(H_und, batch_nodes)], dim=1)  # [B, 2*d_h]
        r_dir = torch.cat([GAP(H_dir, batch_nodes), GMP(H_dir, batch_nodes)], dim=1)  # [B, 2*d_h]

        # Gated fusion
        gate_logit = self.gate_mlp(torch.cat([r_und, r_dir], dim=1))  # [B, 1]
        g = torch.sigmoid(gate_logit)                                 # [B, 1]

        if self.fuse_mode == 'feature':
            r = g * r_dir + (1.0 - g) * r_und                         # [B, 2*d_h]
            logit = self.head(r).squeeze(1)                           # [B]
        else:
            logit_und = self.head_und(r_und).squeeze(1)               # [B]
            logit_dir = self.head_dir(r_dir).squeeze(1)               # [B]
            logit = (g.squeeze(1) * logit_dir) + ((1.0 - g.squeeze(1)) * logit_und)

        out = torch.sigmoid(logit)                                    # [B], probability

        if (not return_reg) or self.lambda_consistency <= 0.0:
            return out

        # Node-level consistency regularizer
        reg = (H_und - H_dir.detach()).pow(2).sum(dim=1).mean()
        return out, self.lambda_consistency * reg

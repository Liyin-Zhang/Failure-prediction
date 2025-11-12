"""

Single-branch undirected graph with dynamic edge weights: discrete node features are embedded,
processed by two TransformerConv layers on the undirected graph with edge attributes, then
GAP/GMP readout feeds a small MLP for binary classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    TransformerConv,
    global_mean_pool as GAP,
    global_max_pool as GMP
)
from torch_geometric.utils import to_undirected


class NetA_NodeOnly(nn.Module):
    """
    Two-stage (Undirected) node-level encoder without pooling inside layers:
      - Undirected Stage: TransformerConv ×2 using undirected edge attributes.
      - Readout: GAP + GMP over nodes, followed by an MLP to output probability.

    Expected fields in `data`:
      - x: LongTensor [N, feature_num] (discrete indices, expanded via embedding)
      - edge_index_dir: LongTensor [2, E_dir] (directed edges)
      - edge_attr: FloatTensor [E_dir, edge_dim] (e.g., 6: WIN, PRE, TMP, PRS, SHU, flow)
      - (optional) edge_index_und / edge_attr_und: if not provided, will be derived from directed edges
      - batch: LongTensor [N] (node-to-graph mapping)
      - (optional) node_gate: FloatTensor [N, 1] node-level gate to modulate the embedding

    Note:
      - `max_node_number`, `embed_dim`, and `feature_num` are expected to be defined at module scope
        (e.g., injected from args before model construction).
    """
    def __init__(self, lambda_consistency: float = 0.0,
                 hidden_dim: int = 256, edge_dim: int = 6, dropout_p: float = 0.5):
        super().__init__()
        self.lambda_consistency = float(lambda_consistency)
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.dropout_p = dropout_p

        # Embedding for discrete node features
        self.item_embedding = nn.Embedding(num_embeddings=max_node_number, embedding_dim=embed_dim)
        d_in = embed_dim * feature_num
        d_h = hidden_dim

        # Undirected stage (node-level, no pooling inside the stage, uses undirected edge_attr)
        self.und_conv1 = TransformerConv(d_in, d_h, edge_dim=edge_dim, heads=1, dropout=0.0)
        self.und_conv2 = TransformerConv(d_h,  d_h, edge_dim=edge_dim, heads=1, dropout=0.0)

        # Readout & classifier (GAP + GMP -> MLP -> sigmoid)
        readout_dim = 2 * d_h
        self.lin1 = nn.Linear(readout_dim, 256)
        self.lin2 = nn.Linear(256, 128)
        self.lin3 = nn.Linear(128, 1)

    def _get_undirected_with_attrs(self, data):
        """
        Prefer provided undirected graph (edge_index_und / edge_attr_und).
        Otherwise, derive undirected edges and attributes from the directed ones
        via to_undirected(..., reduce='mean').
        """
        has_provided = hasattr(data, "edge_index_und") and hasattr(data, "edge_attr_und")
        if has_provided and data.edge_index_und is not None and data.edge_attr_und is not None:
            return data.edge_index_und, data.edge_attr_und

        ei_dir = data.edge_index_dir
        eattr = data.edge_attr
        ei_und_new, eattr_und = to_undirected(ei_dir, edge_attr=eattr, reduce='mean')
        return ei_und_new, eattr_und

    def forward(self, data, return_reg: bool = False):
        x = data.x
        batch_nodes = data.batch

        # Build or fetch undirected edges & attributes
        ei_und, eattr_und = self._get_undirected_with_attrs(data)

        # Embedding expansion
        x = self.item_embedding(x)  # [N, feature_num, embed_dim]

        # Optional node-wise gate
        gate = getattr(data, 'node_gate', None)
        if gate is not None:
            x = x * gate.view(-1, 1, 1)

        x = x.reshape(-1, embed_dim * feature_num)  # [N, d_in]

        # Undirected stage with edge attributes
        h = F.relu(self.und_conv1(x, ei_und, edge_attr=eattr_und))
        H_und = F.relu(self.und_conv2(h, ei_und, edge_attr=eattr_und))  # [N, d_h]

        # Readout: GAP + GMP over nodes
        readout = torch.cat([GAP(H_und, batch_nodes), GMP(H_und, batch_nodes)], dim=1)  # [B, 2*d_h]

        # Graph-level classifier
        out = F.relu(self.lin1(readout))
        out = F.relu(self.lin2(out))
        out = F.dropout(out, p=self.dropout_p, training=self.training)
        out = torch.sigmoid(self.lin3(out)).squeeze(1)  # [B], probability

        if (not return_reg) or self.lambda_consistency <= 0.0:
            return out

        # No explicit regularizer in this undirected-only variant; return constant weight for compatibility
        return out, self.lambda_consistency

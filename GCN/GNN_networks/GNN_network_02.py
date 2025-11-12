"""

Single-branch directed-graph (GAT) model: discrete node features are embedded, encoded by
two GATConv layers on the directed graph, then pooled with GAP/GMP and passed through an
MLP for binary classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATConv,
    global_mean_pool as GAP,
    global_max_pool as GMP
)

class NetA_NodeOnly(nn.Module):
    """
    Directed-branch encoder with GATConv and GAP/GMP readout.

    Expected `data` fields:
      - x: LongTensor [N, feature_num] (discrete indices, expanded via embedding)
      - edge_index_dir: LongTensor [2, E_dir] (directed edges)
      - batch: LongTensor [N] (node-to-graph mapping)

    Notes:
      - `max_node_number`, `embed_dim`, and `feature_num` must be defined at module scope
        (e.g., injected from your args/config before constructing the model).
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
        d_h  = hidden_dim

        # Directed branch (node-level, no internal pooling)
        self.dir_conv1 = GATConv(d_in, d_h, heads=1, dropout=0.0)
        self.dir_conv2 = GATConv(d_h, d_h, heads=1, dropout=0.0)

        # Readout & classifier (GAP + GMP -> MLP -> sigmoid)
        readout_dim = 2 * d_h
        self.lin1 = nn.Linear(readout_dim, 256)
        self.lin2 = nn.Linear(256, 128)
        self.lin3 = nn.Linear(128, 1)

    def forward(self, data, return_reg: bool = False):
        x = data.x
        ei_dir = data.edge_index_dir
        batch_nodes = data.batch

        # Embedding expansion
        x = self.item_embedding(x)                  # [N, feature_num, embed_dim]
        x = x.reshape(-1, embed_dim * feature_num)  # [N, d_in]

        # Directed branch
        z = F.relu(self.dir_conv1(x, ei_dir))
        H_dir = F.relu(self.dir_conv2(z, ei_dir))   # [N, d_h]

        # Readout: GAP + GMP over nodes
        readout = torch.cat([GAP(H_dir, batch_nodes), GMP(H_dir, batch_nodes)], dim=1)  # [B, 2*d_h]

        # Graph-level classifier
        out = F.relu(self.lin1(readout))
        out = F.relu(self.lin2(out))
        out = F.dropout(out, p=self.dropout_p, training=self.training)
        out = torch.sigmoid(self.lin3(out)).squeeze(1)  # [B], probability

        if (not return_reg) or self.lambda_consistency <= 0.0:
            return out

        # No explicit regularizer in this directed-only variant; return constant weight for compatibility
        return out, self.lambda_consistency

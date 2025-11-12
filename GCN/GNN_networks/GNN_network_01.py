"""
This is a single-branch undirected graph model: node features are embedded from
discrete indices, passed through two GATConv layers on the undirected graph,
then pooled with GAP/GMP and classified by a 3-layer MLP for binary prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATConv,
    TransformerConv,
    global_mean_pool as GAP,
    global_max_pool as GMP
)

class NetA_NodeOnly(nn.Module):
    """
    Undirected-branch GNN with GATConv encoder and GAP/GMP readout.
    - Inputs from data:
        x: LongTensor [N, feature_num] (discrete indices)
        edge_index_und: LongTensor [2, E_und] (undirected edges)
        edge_index_dir: LongTensor [2, E_dir] (directed edges)
        edge_attr: FloatTensor [E_dir, edge_dim] (directed edge features)
        batch: LongTensor [N] (node-to-graph mapping)
    - Note: max_node_number, embed_dim, feature_num are expected to be defined
      at module scope (injected via args before model construction).
    """
    def __init__(self, lambda_consistency: float = 0.0, hidden_dim: int = 256,
                 edge_dim: int = 6, dropout_p: float = 0.5):
        super().__init__()
        self.lambda_consistency = float(lambda_consistency)
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.dropout_p = dropout_p

        # Embedding for discrete node features
        self.item_embedding = nn.Embedding(num_embeddings=max_node_number, embedding_dim=embed_dim)
        d_in = embed_dim * feature_num
        d_h = hidden_dim

        # Stage-1: undirected encoder (node-level, no pooling)
        self.und_conv1 = GATConv(d_in, d_h)
        self.und_conv2 = GATConv(d_h, d_h)

        # Stage-2: directed encoder (defined for compatibility; not used in forward)
        self.dir_conv1 = TransformerConv(d_in, d_h, edge_dim=edge_dim, heads=1, dropout=0.0)
        self.dir_conv2 = TransformerConv(d_h, d_h, edge_dim=edge_dim, heads=1, dropout=0.0)

        # Readout & classifier (GAP + GMP -> MLP -> sigmoid)
        readout_dim = 2 * d_h
        self.lin1 = nn.Linear(readout_dim, 256)
        self.lin2 = nn.Linear(256, 128)
        self.lin3 = nn.Linear(128, 1)

    def forward(self, data, return_reg: bool = False):
        x = data.x
        ei_und = data.edge_index_und
        batch_nodes = data.batch

        # Embedding expansion
        x = self.item_embedding(x)                 # [N, feature_num, embed_dim]
        x = x.reshape(-1, embed_dim * feature_num) # [N, d_in]

        # Undirected branch
        h = F.relu(self.und_conv1(x, ei_und))
        H_und = F.relu(self.und_conv2(h, ei_und))  # [N, d_h]

        # Readout: GAP + GMP over nodes
        readout = torch.cat([GAP(H_und, batch_nodes), GMP(H_und, batch_nodes)], dim=1)  # [B, 2*d_h]

        # Graph-level classifier
        out = F.relu(self.lin1(readout))
        out = F.relu(self.lin2(out))
        out = F.dropout(out, p=self.dropout_p, training=self.training)
        out = torch.sigmoid(self.lin3(out)).squeeze(1)  # [B], probability

        if (not return_reg) or self.lambda_consistency <= 0.0:
            return out

        # No regularizer in this undirected-only variant; return a constant weight (compatibility)
        return out, self.lambda_consistency

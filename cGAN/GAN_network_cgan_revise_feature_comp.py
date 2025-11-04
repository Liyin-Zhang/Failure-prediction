import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool, BatchNorm

# -------------------------------
# 你的 Generator：保持不变（仅小清理）
# -------------------------------
class Generator(nn.Module):
    def __init__(self, feature_dim, noise_dim, hidden_dim, dropout_prob=0.3):
        super().__init__()
        self.conv1 = SAGEConv(feature_dim + noise_dim, hidden_dim)
        self.bn1 = BatchNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.fc3 = nn.Linear(hidden_dim // 4, hidden_dim // 4)
        self.fc4 = nn.Linear(hidden_dim // 4, feature_dim)

    def forward(self, normal_features, noise, edge_index, batch):
        x = torch.cat([normal_features, noise], dim=1)
        x = F.leaky_relu(self.bn1(self.conv1(x, edge_index)), negative_slope=0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.2)
        x = F.leaky_relu(self.fc3(x), negative_slope=0.2)
        x = F.leaky_relu(self.fc4(x), negative_slope=0.2)
        return x


# ----------------------------------------
# 判别器：门控融合 + 对齐 + decor 正则 + 读出
# ----------------------------------------
class Discriminator(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        proj_dim: int = None,
        num_gnn_layers: int = 2,
        channel_gate: bool = True,      # True: 逐通道门控；False: 标量门控
        use_film: bool = False,         # (c) 可选：FiLM/SE 调制（默认关）
        film_hidden: int = None
    ):
        """
        in_dim:     节点特征维度（单侧），输入会拼接 normal/extreme → 2*in_dim
        hidden_dim: GNN/MLP 的中间维度
        proj_dim:   融合前的对齐维度（默认 = hidden_dim）
        """
        super().__init__()
        if proj_dim is None:
            proj_dim = hidden_dim
        if film_hidden is None:
            film_hidden = hidden_dim

        self.channel_gate = channel_gate
        self.use_film = use_film

        # ---- GNN 分支（结构）: 输入拼接后的 x=[x_norm, x_ext] ----
        self.gnn_convs = nn.ModuleList()
        in_gnn = in_dim * 2
        for li in range(num_gnn_layers):
            self.gnn_convs.append(SAGEConv(in_gnn if li == 0 else hidden_dim, hidden_dim))
        self.gnn_ln = nn.LayerNorm(hidden_dim)     # (a) 对齐里的 LN（先在各自空间）

        # ---- MLP 分支（属性联合）----
        self.mlp_joint = nn.Sequential(
            nn.Linear(in_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.mlp_ln = nn.LayerNorm(hidden_dim)     # (a) 对齐里的 LN

        # ---- (a) 通道对齐：线性投影到公共维度 ----
        self.proj_g = nn.Linear(hidden_dim, proj_dim, bias=True)  # W_G
        self.proj_m = nn.Linear(hidden_dim, proj_dim, bias=True)  # W_M

        # ---- (b) 门控融合 ----
        if channel_gate:
            # 逐通道门控：输出与 proj_dim 对齐
            self.gate = nn.Linear(2 * proj_dim, proj_dim)  # W_alpha
        else:
            # 标量门控：更稳，α \in [0,1] 标量
            self.gate = nn.Linear(2 * proj_dim, 1)

        # ---- (c) 可选：FiLM 调制（用 M 分支调制 G 分支）----
        if self.use_film:
            self.film = nn.Sequential(
                nn.Linear(proj_dim, film_hidden),
                nn.ReLU(),
                nn.Linear(film_hidden, 2 * proj_dim)  # 输出 [gamma, beta]
            )

        # ---- (e) 读出 + 判别头 ----
        self.readout = global_mean_pool
        self.fc = nn.Linear(proj_dim, 1)

        # 存放“最近一次前向”的中间量，用于 decor 正则
        self._cache_hg = None  # h~_G
        self._cache_hm = None  # h~_M

    # ---------- 辅助：协方差与 decor 正则 ----------
    @staticmethod
    def _cov(mat: torch.Tensor, eps: float = 1e-6):
        """
        mat: [N, D]，先中心化，再计算 (X^T X)/(N-1)
        返回 [D, D]
        """
        if mat.ndim != 2:
            mat = mat.view(mat.size(0), -1)
        X = mat - mat.mean(dim=0, keepdim=True)
        N = max(1, X.size(0) - 1)
        cov = (X.T @ X) / N
        # 数值稳定：给对角加一点
        cov = cov + eps * torch.eye(cov.size(0), device=cov.device, dtype=cov.dtype)
        return cov

    def decor_loss(self, lam_cross: float = 0.1):
        """
        (d) decor 正则：
        ||Cov(hG)-I||_F^2 + ||Cov(hM)-I||_F^2 + lam * ||Cov(hG,hM)||_F^2
        若还未前向，返回 0
        """
        if (self._cache_hg is None) or (self._cache_hm is None):
            return torch.tensor(0.0, device=next(self.parameters()).device)

        hG = self._cache_hg      # [N, D]
        hM = self._cache_hm      # [N, D]
        I = torch.eye(hG.size(1), device=hG.device, dtype=hG.dtype)

        # 自身去相关
        covG = self._cov(hG)
        covM = self._cov(hM)
        lossG = torch.norm(covG - I, p='fro') ** 2
        lossM = torch.norm(covM - I, p='fro') ** 2

        # 交叉相关：用拼接构造联合协方差，再取 off-diagonal 块，或直接对 X= [hG | hM] 做协方差
        X = torch.cat([hG, hM], dim=1)  # [N, 2D]
        covX = self._cov(X)
        D = hG.size(1)
        cov_cross = covX[:D, D:]        # Cov(hG, hM)
        lossC = torch.norm(cov_cross, p='fro') ** 2

        return lossG + lossM + lam_cross * lossC

    # ---------- 前向 ----------
    def forward(self, normal_features, extreme_features, edge_index, batch):
        """
        输入：
          - normal_features: [num_nodes, in_dim]
          - extreme_features:[num_nodes, in_dim]
          - edge_index:      [2, num_edges]
          - batch:           [num_nodes], 图分配
        输出：
          - prob:            [num_graphs, 1]（sigmoid 概率）
        说明：
          - decor 正则请在训练时调用 self.decor_loss(lam)
        """
        device = normal_features.device
        # 拼接节点输入 x = [x_norm, x_ext]，让两支看到同样输入（结构/属性不同处理）
        x = torch.cat([normal_features, extreme_features], dim=1)

        # ---- 结构分支（GNN）----
        hG = x
        for conv in self.gnn_convs:
            hG = F.relu(conv(hG, edge_index))
        hG = self.gnn_ln(hG)                # LayerNorm
        hG = self.proj_g(hG)                # 线性投影到公共维度 D_p

        # ---- 属性分支（MLP）----
        hM = self.mlp_joint(x)
        hM = self.mlp_ln(hM)                # LayerNorm
        hM = self.proj_m(hM)                # 线性投影到公共维度 D_p

        # ---- (c) 可选：FiLM/SE 调制（用 M 调制 G）----
        if self.use_film:
            gamma_beta = self.film(hM)                  # [N, 2*D_p]
            gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
            hG = gamma * hG + beta

        # ---- (b) 门控融合 ----
        gate_in = torch.cat([hG, hM], dim=1)            # [N, 2*D_p]
        alpha = torch.sigmoid(self.gate(gate_in))       # [N, D_p] 或 [N, 1]
        if not self.channel_gate:                       # 标量门控需要扩展到通道
            alpha = alpha.expand_as(hG)
        h_fuse = alpha * hG + (1.0 - alpha) * hM        # [N, D_p]

        # ---- (e) 读出 + 判别头 ----
        g = global_mean_pool(h_fuse, batch)             # [num_graphs, D_p]
        out = torch.sigmoid(self.fc(g))                 # [num_graphs, 1]

        # 缓存对齐后的两路表征，供 decor 正则使用
        self._cache_hg = hG.detach()                    # 不影响主梯度（正则会单独反传时可不 detach）
        self._cache_hm = hM.detach()

        return out

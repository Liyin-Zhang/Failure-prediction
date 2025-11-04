# -*- coding: utf-8 -*-
"""
判别器（可切换融合策略）：
支持 5 种结构，保持 forward 接口不变，训练主程序几乎无需修改。
(a) 通道对齐（LayerNorm+Linear）对所有策略默认启用
(b) 门控融合（默认）
(1) concat_mlp         : [hG~ || hM~] -> 小MLP
(2) gated              : 逐通道/标量门控
(3) hadamard_concat    : hG~⊙hM~  +  Concat-MLP
(4) cross_stitch       : 软参数共享（2×2可学习矩阵），再求和
(5) film               : 用属性分支调制结构分支（FiLM），再与 hM~ 相加

附：decor_loss() 可选的去相关正则（Orth/Decorr），不调用则不生效。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool, BatchNorm


# -----------------------
# 你现有的 Generator 保持不变
# -----------------------
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

#
# # --------------------------------------
# # 判别器：可切换融合策略（保持类名 Discriminator）
# # --------------------------------------
# class Discriminator(nn.Module):
#     def __init__(
#         self,
#         in_dim: int,
#         hidden_dim: int,
#         proj_dim: int = None,         # 对齐后的公共维度（默认=hidden_dim）
#         num_gnn_layers: int = 2,
#         fusion_type: str = "gated",   # 'concat_mlp' | 'gated' | 'hadamard_concat' | 'cross_stitch' | 'film'
#         channel_gate: bool = True,    # gated 策略下是否逐通道门控；False=标量门控
#         film_hidden: int = None       # FiLM 的中间宽度
#     ):
#         super().__init__()
#         assert fusion_type in {"concat_mlp", "gated", "hadamard_concat", "cross_stitch", "film"}
#         self.fusion_type = fusion_type
#         self.channel_gate = channel_gate
#
#         if proj_dim is None:
#             proj_dim = hidden_dim
#         if film_hidden is None:
#             film_hidden = hidden_dim
#
#         # ===== 结构分支（GNN） =====
#         self.gnn_convs = nn.ModuleList()
#         in_gnn = in_dim * 2  # 输入为 [x_norm, x_ext] 拼接
#         for li in range(num_gnn_layers):
#             self.gnn_convs.append(SAGEConv(in_gnn if li == 0 else hidden_dim, hidden_dim))
#         self.gnn_ln = nn.LayerNorm(hidden_dim)
#
#         # ===== 属性分支（MLP） =====
#         self.mlp_joint = nn.Sequential(
#             nn.Linear(in_dim * 2, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#         )
#         self.mlp_ln = nn.LayerNorm(hidden_dim)
#
#         # ===== (a) 通道对齐（两支 → proj_dim）=====
#         self.proj_g = nn.Linear(hidden_dim, proj_dim)  # W_G
#         self.proj_m = nn.Linear(hidden_dim, proj_dim)  # W_M
#         self.proj_dim = proj_dim
#
#         # ===== 各融合策略需要的额外层 =====
#         # 1) concat+mlp
#         self.concat_mlp = nn.Sequential(
#             nn.Linear(2 * proj_dim, proj_dim),
#             nn.ReLU(),
#             # nn.Linear(proj_dim, proj_dim)
#         )
#
#         # 2) gated
#         self.gate = nn.Linear(2 * proj_dim, 1) #(proj_dim if channel_gate else 1)
#
#         # 3) hadamard + concat-mlp
#         self.hadamard_mlp = nn.Sequential(
#             nn.Linear(3 * proj_dim, proj_dim),
#             nn.ReLU(),
#             # nn.Linear(proj_dim, proj_dim)
#         )
#
#         # 4) cross-stitch（2×2 参数矩阵）
#         #    a11 a12
#         #    a21 a22
#         self.cs_a11 = nn.Parameter(torch.tensor(0.9))
#         self.cs_a12 = nn.Parameter(torch.tensor(0.1))
#         self.cs_a21 = nn.Parameter(torch.tensor(0.1))
#         self.cs_a22 = nn.Parameter(torch.tensor(0.9))
#
#         # 5) FiLM（用 M 调制 G）
#         self.film = nn.Sequential(
#             nn.Linear(proj_dim, 2 * proj_dim), #film_hidden
#             nn.ReLU(),
#             # nn.Linear(film_hidden, 2 * proj_dim)  # -> [gamma, beta]
#         )
#
#         # ===== (e) 读出 + 判别头 =====
#         self.readout = global_mean_pool
#         self.fc = nn.Linear(proj_dim, 1)
#
#         # decor 正则缓存
#         self._cache_hg = None
#         self._cache_hm = None
#
#     # ---------- decor 正则 ----------
#     @staticmethod
#     def _cov(mat: torch.Tensor, eps: float = 1e-6):
#         if mat.ndim != 2:
#             mat = mat.view(mat.size(0), -1)
#         X = mat - mat.mean(dim=0, keepdim=True)
#         N = max(1, X.size(0) - 1)
#         cov = (X.T @ X) / N
#         cov = cov + eps * torch.eye(cov.size(0), device=cov.device, dtype=cov.dtype)
#         return cov
#
#     def decor_loss(self, lam_cross: float = 0.1):
#         if (self._cache_hg is None) or (self._cache_hm is None):
#             return torch.tensor(0.0, device=next(self.parameters()).device)
#         hG, hM = self._cache_hg, self._cache_hm
#         I = torch.eye(hG.size(1), device=hG.device, dtype=hG.dtype)
#         covG = self._cov(hG); covM = self._cov(hM)
#         lossG = torch.norm(covG - I, p='fro') ** 2
#         lossM = torch.norm(covM - I, p='fro') ** 2
#         X = torch.cat([hG, hM], dim=1)                 # 交叉相关块
#         covX = self._cov(X); D = hG.size(1)
#         cov_cross = covX[:D, D:]
#         lossC = torch.norm(cov_cross, p='fro') ** 2
#         return lossG + lossM + lam_cross * lossC
#
#     # ---------- 前向 ----------
#     def forward(self, normal_features, extreme_features, edge_index, batch):
#         # x = [x_norm, x_ext]（两支看到相同输入）
#         x = torch.cat([normal_features, extreme_features], dim=1)
#
#         # 结构分支
#         hG = x
#         for conv in self.gnn_convs:
#             hG = F.relu(conv(hG, edge_index))
#         hG = self.gnn_ln(hG)
#         hG = self.proj_g(hG)                 # hG~
#
#         # 属性分支
#         hM = self.mlp_joint(x)
#         hM = self.mlp_ln(hM)
#         hM = self.proj_m(hM)                 # hM~
#
#         # ============= 融合 =============
#         if self.fusion_type == "concat_mlp":
#             h_fuse = self.concat_mlp(torch.cat([hG, hM], dim=1))
#
#         elif self.fusion_type == "gated":
#             alpha = torch.sigmoid(self.gate(torch.cat([hG, hM], dim=1)))
#             if alpha.shape[1] == 1:          # 标量门控
#                 alpha = alpha.expand_as(hG)
#             h_fuse = alpha * hG + (1.0 - alpha) * hM
#
#         elif self.fusion_type == "hadamard_concat":
#             had = hG * hM
#             h_fuse = self.hadamard_mlp(torch.cat([hG, hM, had], dim=1))
#
#         elif self.fusion_type == "cross_stitch":
#             # 两支线性混合，再求和得到融合向量
#             hG2 = self.cs_a11 * hG + self.cs_a12 * hM
#             hM2 = self.cs_a21 * hG + self.cs_a22 * hM
#             h_fuse = 0.5 * (hG2 + hM2)
#
#         elif self.fusion_type == "film":
#             gamma_beta = self.film(hM)                 # 用属性分支调制结构分支
#             gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
#             hG_mod = gamma * hG + beta
#             h_fuse = hG_mod + hM
#
#         else:
#             raise ValueError(f"Unknown fusion_type: {self.fusion_type}")
#
#         # 读出 + 判别头
#         g = self.readout(h_fuse, batch)                # [num_graphs, proj_dim]
#         out = torch.sigmoid(self.fc(g))                # [num_graphs, 1]
#
#         # 缓存对齐后的两支（供 decor 正则）
#         self._cache_hg = hG.detach()
#         self._cache_hm = hM.detach()
#         return out
class Discriminator(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        proj_dim: int = None,
        num_gnn_layers: int = 2,
        fusion_type: str = "gated",      # 'concat_mlp' | 'gated' | 'hadamard_concat' | 'cross_stitch' | 'film'
        channel_gate: bool = True,
        film_hidden: int = None,
        # ==== 全局“温和化”参数（可按需调） ====
        lambda_avg: float = 0.7,         # 与 avg 的混合权重（越大越保守）
        fuse_l2: float = 1e-3,           # L2 正则：||h_fuse - avg||^2 的系数；0=关闭
        # gated（延续你已有的温度/夹紧/动量平滑）
        gate_tau: float = 2.0,
        gate_eps: float = 0.05,
        gate_momentum: float = 0.0,
        gate_l2: float = 1e-3,
        # hadamard
        had_eta: float = 0.2,            # ⊙项的幅度（越小越保守）
        # cross-stitch
        cs_rho: float = 0.1,             # 偏离 I 的幅度上界（越小越保守）
        # film
        film_scale: float = 0.1          # γ/β 的调制幅度（越小越保守）
    ):
        super().__init__()
        assert fusion_type in {"concat_mlp", "gated", "hadamard_concat", "cross_stitch", "film"}
        self.fusion_type = fusion_type
        self.channel_gate = channel_gate

        if proj_dim is None:
            proj_dim = hidden_dim
        if film_hidden is None:
            film_hidden = hidden_dim

        # 保存温和化参数
        self.lambda_avg = float(lambda_avg)
        self.fuse_l2 = float(fuse_l2)

        self.gate_tau = float(gate_tau)
        self.gate_eps = float(gate_eps)
        self.gate_momentum = float(gate_momentum)
        self.gate_l2 = float(gate_l2)

        self.had_eta = float(had_eta)
        self.cs_rho = float(cs_rho)
        self.film_scale = float(film_scale)

        # ===== 结构分支（GNN） =====
        self.gnn_convs = nn.ModuleList()
        in_gnn = in_dim * 2
        for li in range(num_gnn_layers):
            self.gnn_convs.append(SAGEConv(in_gnn if li == 0 else hidden_dim, hidden_dim))
        self.gnn_ln = nn.LayerNorm(hidden_dim)

        # ===== 属性分支（MLP） =====
        self.mlp_joint = nn.Sequential(
            nn.Linear(in_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.mlp_ln = nn.LayerNorm(hidden_dim)

        # ===== (a) 对齐到同一维 =====
        self.proj_g = nn.Linear(hidden_dim, proj_dim)
        self.proj_m = nn.Linear(hidden_dim, proj_dim)
        self.proj_dim = proj_dim

        # ===== 额外层 =====
        self.concat_mlp = nn.Sequential(
            nn.Linear(2 * proj_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )

        self.gate = nn.Linear(2 * proj_dim, (proj_dim if channel_gate else 1))
        nn.init.zeros_(self.gate.weight); nn.init.zeros_(self.gate.bias)

        self.hadamard_mlp = nn.Sequential(
            nn.Linear(3 * proj_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )

        # cross-stitch 基矩阵（后面做 I + ρ*tanh(B)）
        self.cs_B = nn.Parameter(torch.zeros(2, 2))  # 初始化为 0 => 初始即近似恒等

        self.film = nn.Sequential(
            nn.Linear(proj_dim, 2 * proj_dim),
            nn.ReLU(),
            nn.Linear(film_hidden, 2 * proj_dim)  # -> [gamma, beta]
        )

        self.readout = global_mean_pool
        self.fc = nn.Linear(proj_dim, 1)

        # 缓存
        self._cache_hg = None
        self._cache_hm = None
        self._cache_alpha = None
        self._alpha_ma = None
        self._cache_fuse = None
        self._cache_avg = None

    # ---------- decor 正则（如需） ----------
    @staticmethod
    def _cov(mat: torch.Tensor, eps: float = 1e-6):
        if mat.ndim != 2:
            mat = mat.view(mat.size(0), -1)
        X = mat - mat.mean(dim=0, keepdim=True)
        N = max(1, X.size(0) - 1)
        cov = (X.T @ X) / N
        cov = cov + eps * torch.eye(cov.size(0), device=mat.device, dtype=mat.dtype)
        return cov

    def decor_loss(self, lam_cross: float = 0.1):
        if (self._cache_hg is None) or (self._cache_hm is None):
            return torch.tensor(0.0, device=next(self.parameters()).device)
        hG, hM = self._cache_hg, self._cache_hm
        I = torch.eye(hG.size(1), device=hG.device, dtype=hG.dtype)
        covG = self._cov(hG); covM = self._cov(hM)
        lossG = torch.norm(covG - I, p='fro') ** 2
        lossM = torch.norm(covM - I, p='fro') ** 2
        X = torch.cat([hG, hM], dim=1); D = hG.size(1)
        covX = self._cov(X); cov_cross = covX[:D, D:]
        lossC = torch.norm(cov_cross, p='fro') ** 2
        return lossG + lossM + lam_cross * lossC

    # ---------- 门控正则（可选） ----------
    def gate_reg_loss(self):
        if (self._cache_alpha is None) or (self.gate_l2 <= 0):
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return self.gate_l2 * ((self._cache_alpha - 0.5) ** 2).mean()

    # ---------- 融合“回正”正则（可选）----------
    def fuse_reg_loss(self):
        if (self._cache_fuse is None) or (self._cache_avg is None) or (self.fuse_l2 <= 0):
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return self.fuse_l2 * ((self._cache_fuse - self._cache_avg) ** 2).mean()

    # ---------- 前向 ----------
    def forward(self, normal_features, extreme_features, edge_index, batch):
        x = torch.cat([normal_features, extreme_features], dim=1)

        # 结构分支
        hG = x
        for conv in self.gnn_convs:
            hG = F.relu(conv(hG, edge_index))
        # hG = self.gnn_ln(hG)
        # hG = self.proj_g(hG)

        # 属性分支
        hM = self.mlp_joint(x)
        # hM = self.mlp_ln(hM)
        # hM = self.proj_m(hM)

        # 基线：更温和的“平均融合”
        avg = 0.5 * (hG + hM)
        lam = self.lambda_avg  # 与 avg 的混合权重

        if self.fusion_type == "concat_mlp":
            core = self.concat_mlp(torch.cat([hG, hM], dim=1))
            h_fuse = lam * avg + (1 - lam) * core

        elif self.fusion_type == "gated":
            gz = self.gate(torch.cat([hG, hM], dim=1)) / max(1e-6, self.gate_tau)
            alpha = torch.sigmoid(gz)
            if alpha.shape[1] == 1:
                alpha = alpha.expand_as(hG)
            if self.gate_eps > 0:
                eps = min(0.49, max(0.0, self.gate_eps))
                alpha = alpha * (1 - 2 * eps) + eps
            if self.training and self.gate_momentum > 0:
                if (self._alpha_ma is None) or (self._alpha_ma.shape != alpha.shape):
                    self._alpha_ma = alpha.detach()
                m = self.gate_momentum
                alpha = (1 - m) * alpha + m * self._alpha_ma
                self._alpha_ma = alpha.detach()
            core = alpha * hG + (1.0 - alpha) * hM
            # print(alpha)
            h_fuse = lam * avg + (1 - lam) * core
            self._cache_alpha = alpha.detach()

        elif self.fusion_type == "hadamard_concat":
            had = self.had_eta * (hG * hM)          # 把 ⊙ 项限制很小
            core = self.hadamard_mlp(torch.cat([hG, hM, had], dim=1))
            h_fuse = lam * avg + (1 - lam) * core

        elif self.fusion_type == "cross_stitch":
            # A = I + ρ * tanh(B) （ρ 小，保证更温和）
            B = torch.tanh(self.cs_B)
            A = torch.eye(2, device=B.device, dtype=B.dtype) + self.cs_rho * B
            # [hG, hM] @ A
            h_stack = torch.stack([hG, hM], dim=-1)       # [N, D, 2]
            mixed   = torch.matmul(h_stack, A)            # [N, D, 2]
            hG2, hM2 = mixed[..., 0], mixed[..., 1]
            core = 0.5 * (hG2 + hM2)
            h_fuse = lam * avg + (1 - lam) * core

        elif self.fusion_type == "film":
            gamma_beta = self.film(hM)
            gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
            # 把调制幅度限制在很小区间
            gamma = 1.0 + self.film_scale * torch.tanh(gamma)  # ~ [1-δ, 1+δ]
            beta  = self.film_scale * torch.tanh(beta)         # ~ [-δ, +δ]
            core = gamma * hG + beta + hM
            h_fuse = lam * avg + (1 - lam) * core

        else:
            raise ValueError(f"Unknown fusion_type: {self.fusion_type}")

        # 读出 + 判别头
        g = self.readout(h_fuse, batch)
        out = torch.sigmoid(self.fc(g)) ###########g

        # 缓存（正则/分析）
        self._cache_hg = hG.detach()
        self._cache_hm = hM.detach()
        self._cache_fuse = h_fuse.detach()
        self._cache_avg  = avg.detach()

        return out

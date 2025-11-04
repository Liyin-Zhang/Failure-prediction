# -*- coding: utf-8 -*-
"""
多模型对比训练版本（按程序2的采样与加权机制改写）：
- 分别训练 6 个模型：
  (1) 单独生成 WIN
  (2) 单独生成 PRE
  (3) 单独生成 PRS
  (4) 单独生成 SHU
  (5) 单独生成 TMP
  (6) 同时生成 5 个特征（ALL）
训练细节与程序2对齐：
  - 在线随机采样 iterate_batches（锚事件限次 + 批内去重）
  - 锚事件单位质量加权（BCE 使用 reduction='none' 再乘权重）
  - 判别器梯度裁剪
  - 一致性损失仅在 ALL 时按指定通道生效（CONS_USE_CHANNELS）
  - 噪声接口兼容 per_node / graph 广播与可选拉普拉斯平滑（默认 per_node，无平滑）
"""

import sys, os, re, random, pickle
sys.path.append(os.path.dirname(__file__))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from GAN_network_cgan_revise import Generator, Discriminator
from GAN_dataset import create_graph
from visualizer import GANDemoVisualizer
import matplotlib.pyplot as plt

# ========================= 开关与超参数 =========================
# —— 功能开关 ——（沿用程序2）
USE_ONLINE_SAMPLING      = False    # 在线随机采样
LIMIT_REPEAT_PER_ANCHOR  = False
BATCH_DEDUP              = False
ANCHOR_UNIT_MASS         = False

# —— 采样与去重参数 ——（沿用程序2）
BATCH_SIZE        = 800
CAP_PER_ANCHOR    = 2
DT_THR            = 6
SIM_THR           = 0.98

# —— 训练参数 ——（沿用程序1/2）
HIDDEN_DIM   = 72
NUM_EPOCHS   = 3000
LR_G         = 5e-4
LR_D         = 5e-4
BETAS        = (0.5, 0.9)
STEP_SIZE    = 150
GAMMA        = 0.8
SEED         = 42

# —— 一致性/相关性损失（与程序2保持一致）——
USE_CONSISTENCY_LOSS = False        # r = x_fake - x_norm 的图拉普拉斯一致性
USE_CORR_LOSS        = False        # 图级余弦相关（宏观相关性）
LAMBDA_CONS          = 1e-2        # 建议 1e-2 起步
LAMBDA_CORR          = 5e-3        # 建议 5e-3 起步
CONS_USE_CHANNELS    = [1, 4]      # 参与一致性的通道：如 PRE(1)、SHU(4)
CONS_SIGMA           = 0.2         # 条件权重的高斯尺度

# —— 噪声形态开关 ——（与程序2一致，默认 per_node / 不平滑）
NOISE_MODE           = "per_node"  # "per_node" | "graph"（图级广播噪声）
SMOOTH_NOISE         = False       # 是否对噪声做一次拉普拉斯平滑（抑制破碎）
SMOOTH_ALPHA         = 0.5         # 噪声平滑的邻域权重系数（0~1）

# ========================= 工具函数（与程序2对齐） =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def feeder_base_name(name: str):
    m = re.match(r"^(.*?)(\d{5})$", str(name))
    return m.group(1) if m else str(name)

def cosine_sim(a: torch.Tensor, b: torch.Tensor, eps=1e-8):
    a = a.view(-1).float(); b = b.view(-1).float()
    na = a.norm(p=2) + eps; nb = b.norm(p=2) + eps
    return torch.dot(a, b).item() / (na.item() * nb.item())

def get_graph_meta(d: Data):
    fname = getattr(d, "feeder_name", None)
    feeder_full = str(fname) if fname is not None else "unknown00000"
    base = feeder_base_name(feeder_full)
    ne = getattr(d, "norm_event", None)
    ee = getattr(d, "ext_event", None)
    if isinstance(ne, torch.Tensor): ne = int(ne.view(-1)[0].item())
    if isinstance(ee, torch.Tensor): ee = int(ee.view(-1)[0].item())
    if ne is None: ne = -1
    if ee is None: ee = -1
    return base, ne, ee, feeder_full

def build_index(dataset):
    items = []
    by_anchor = {}
    for i in range(len(dataset)):
        d = dataset[i]
        base, ne, ee, _ = get_graph_meta(d)
        items.append((i, base, ne, ee))
        by_anchor.setdefault((base, ne), []).append(i)
    return items, by_anchor

def iterate_batches(dataset, batch_size=BATCH_SIZE, cap_per_anchor=CAP_PER_ANCHOR,
                    dt_thr=DT_THR, sim_thr=SIM_THR, seed=SEED):
    rng = np.random.default_rng(seed)
    items, by_anchor = build_index(dataset)
    used_cnt = {k: 0 for k in by_anchor.keys()}
    order = np.arange(len(items)); rng.shuffle(order)

    batch_buf = []
    picked_meta = {}  # (base, ne) -> list of {'ext': int, 'pre_ext': Tensor}

    for pos in order:
        idx, base, ne, ee = items[pos]
        anchor = (base, ne)

        if LIMIT_REPEAT_PER_ANCHOR and ne >= 0:
            if used_cnt.get(anchor, 0) >= cap_per_anchor:
                continue

        if BATCH_DEDUP and ne >= 0:
            plist = picked_meta.get(anchor, [])
            if len(plist) > 0:
                d = dataset[idx]
                pre_ext = d.x[:, 1, 1].detach().cpu()
                dup = False
                for rec in plist:
                    if ee >= 0 and rec['ext'] >= 0 and abs(ee - rec['ext']) <= dt_thr:
                        dup = True; break
                    sim = cosine_sim(pre_ext, rec['pre_ext'])
                    if sim >= sim_thr:
                        dup = True; break
                if dup:
                    continue

        batch_buf.append(idx)
        if BATCH_DEDUP and ne >= 0:
            pre_ext = dataset[idx].x[:, 1, 1].detach().cpu()
            picked_meta.setdefault(anchor, []).append({'ext': ee, 'pre_ext': pre_ext})
        if LIMIT_REPEAT_PER_ANCHOR and ne >= 0:
            used_cnt[anchor] = used_cnt.get(anchor, 0) + 1

        if len(batch_buf) >= batch_size:
            yield Batch.from_data_list([dataset[i] for i in batch_buf])
            batch_buf = []

    if len(batch_buf) > 0:
        yield Batch.from_data_list([dataset[i] for i in batch_buf])

def anchor_ids_from_batch(batch: Batch):
    feeder_names = batch.feeder_name
    norm_events  = batch.norm_event
    ids = []
    anchor_map = {}
    next_id = 0
    for i in range(batch.num_graphs):
        fname = feeder_names[i]
        if isinstance(fname, bytes):
            fname = fname.decode('utf-8', errors='ignore')
        base = feeder_base_name(str(fname))
        ne = norm_events[i]
        if isinstance(ne, torch.Tensor): ne = int(ne.item())
        key = (base, int(ne))
        if key not in anchor_map:
            anchor_map[key] = next_id; next_id += 1
        ids.append(anchor_map[key])
    return torch.tensor(ids, dtype=torch.long, device=batch.x.device), anchor_map

def compute_anchor_weights(batch, device):
    if not ANCHOR_UNIT_MASS:
        return torch.full((batch.num_graphs, 1), 1.0 / max(1, batch.num_graphs), device=device)

    if (not hasattr(batch, "feeder_name")) or (not hasattr(batch, "norm_event")):
        return torch.full((batch.num_graphs, 1), 1.0 / max(1, batch.num_graphs), device=device)

    anchor_ids, _ = anchor_ids_from_batch(batch)
    unique_anchors, counts = torch.unique(anchor_ids, return_counts=True)
    per_anchor_w = {int(a.item()): 1.0 / float(c.item()) for a, c in zip(unique_anchors, counts)}
    graph_weights = torch.tensor([per_anchor_w[int(a.item())] for a in anchor_ids],
                                 dtype=torch.float32, device=device).view(-1, 1)
    return graph_weights

# ---------------------- 条件一致性损失（与程序2一致） ----------------------
def _pairwise_weight(xi, xj, sigma=0.2):
    dist2 = ((xi - xj) ** 2).sum(dim=-1)
    return torch.exp(- dist2 / (sigma**2 + 1e-8))

def consistency_loss(x_fake, x_norm, edge_index, use_channels=None, sigma=0.2):
    """
    r = x_fake - x_norm 的图拉普拉斯一致性：sum_ij w_ij || r_i - r_j ||^2
    x_fake, x_norm: [N, C]
    edge_index: [2, E]
    """
    if use_channels is not None:
        x_fake = x_fake[:, use_channels]
        x_norm = x_norm[:, use_channels]
    r = x_fake - x_norm
    src, dst = edge_index[0], edge_index[1]
    wij = _pairwise_weight(x_norm[src], x_norm[dst], sigma=sigma)     # [E]
    diff = (r[src] - r[dst]).pow(2).sum(dim=-1)                       # [E]
    return (wij * diff).mean()

def graphwise_cosine_loss(x_fake, x_norm, batch):
    """
    图级余弦相关：(1 - cos(vec_fake_g, vec_norm_g)) 的平均
    简单做法：图内求和向量作为图表示
    """
    num_graphs = int(batch.max().item()) + 1
    fake_g = torch.zeros(num_graphs, x_fake.size(1), device=x_fake.device)\
        .index_add_(0, batch, x_fake)
    norm_g = torch.zeros(num_graphs, x_norm.size(1), device=x_norm.device)\
        .index_add_(0, batch, x_norm)
    dot = (fake_g * norm_g).sum(dim=-1)
    nf  = fake_g.norm(p=2, dim=-1) + 1e-8
    nn  = norm_g.norm(p=2, dim=-1) + 1e-8
    cos = dot / (nf * nn)
    return (1.0 - cos).mean()

# ---------------------- 噪声策略（与程序2一致） ----------------------
def make_noise(mode, num_nodes, num_graphs, feat_dim, device, batch, edge_index,
               smooth=False, alpha=0.5):
    """
    返回 [num_nodes, feat_dim] 的噪声
    - mode="per_node"：逐节点独立高斯
    - mode="graph"   ：图级高斯 z_g -> 广播到节点
    - smooth=True    ：对噪声做一次拉普拉斯平滑：x <- (1-alpha)*x + alpha*邻域均值
    """
    if mode == "graph":
        z_g = torch.randn(num_graphs, feat_dim, device=device)
        noise = z_g[batch]                                  # 广播
    else:
        noise = torch.randn(num_nodes, feat_dim, device=device)

    if smooth:
        src, dst = edge_index[0], edge_index[1]
        deg = torch.zeros(num_nodes, device=device).index_add_(0, src, torch.ones_like(src, dtype=torch.float))
        neigh_sum = torch.zeros_like(noise).index_add_(0, src, noise[dst])  # 按 src 聚合 dst 的噪声
        neigh_mean = neigh_sum / (deg.clamp_min(1).unsqueeze(-1))
        noise = (1 - alpha) * noise + alpha * neigh_mean
    return noise

# ========================= 主程序 =========================
if __name__ == '__main__':
    set_seed(SEED)
    dataset = create_graph(file_name='GAN_data/train/train_0313_1_clustered.csv')
    visualizer = GANDemoVisualizer('GAN Graph Visualization (WIN+PRE)')

    # 六个训练配置
    feature_names = ['WIN', 'PRE', 'PRS', 'SHU', 'TMP']
    feature_indices = {'WIN': 0, 'PRE': 1, 'PRS': 2, 'SHU': 3, 'TMP': 4}
    experiments = ['ALL']#feature_names#['ALL']#feature_names + ['ALL']  # 可按需改为仅 ['ALL']

    for feat in experiments:
        print("="*90)
        print(f"🔹 Training model for feature: {feat}")
        print("="*90)

        # 设置输入输出维度
        IN_DIM = 1 if feat != 'ALL' else 5
        OUT_TAG = feat.lower()

        generator = Generator(feature_dim=IN_DIM, noise_dim=IN_DIM, hidden_dim=HIDDEN_DIM).cuda()
        discriminator = Discriminator(
            in_dim=IN_DIM,
            hidden_dim=HIDDEN_DIM,
            proj_dim=HIDDEN_DIM,
            channel_gate=True,
            use_film=False
        ).cuda()

        optim_G = torch.optim.Adam(generator.parameters(), lr=LR_G, betas=BETAS)
        optim_D = torch.optim.Adam(discriminator.parameters(), lr=LR_D, betas=BETAS)
        scheduler_G = torch.optim.lr_scheduler.StepLR(optim_G, step_size=STEP_SIZE, gamma=GAMMA)
        scheduler_D = torch.optim.lr_scheduler.StepLR(optim_D, step_size=STEP_SIZE, gamma=GAMMA)

        # 关键：与程序2一致，使用 reduction='none'，再乘以锚事件权重
        bce = nn.BCELoss(reduction='none')

        loss_D_hist, loss_G_hist = [], []

        for epoch in range(NUM_EPOCHS):
            epoch_loss_D = 0.0
            epoch_loss_G = 0.0
            nb = 0

            # —— 与程序2一致：在线采样或普通 DataLoader ——（默认使用在线采样）
            if USE_ONLINE_SAMPLING:
                batch_iter = iterate_batches(dataset,
                                             batch_size=BATCH_SIZE,
                                             cap_per_anchor=CAP_PER_ANCHOR,
                                             dt_thr=DT_THR,
                                             sim_thr=SIM_THR,
                                             seed=SEED + epoch)
            else:
                loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
                batch_iter = (b for b in loader)

            for batch in batch_iter:
                batch = batch.cuda()

                x = batch.x.to(torch.float)          # (num_nodes, 5, 2)
                edge_index  = batch.edge_index.to(torch.long)
                batch_index = batch.batch

                # 单特征 vs ALL
                if feat != 'ALL':
                    idx = feature_indices[feat]
                    x_normal  = x[:, [idx], 0]       # [N, 1]
                    x_extreme = x[:, [idx], 1]       # [N, 1]
                else:
                    x_normal  = x[:, :, 0]           # [N, 5]
                    x_extreme = x[:, :, 1]           # [N, 5]

                num_nodes  = x_normal.size(0)
                num_graphs = batch.num_graphs

                # —— 锚事件单位质量归一权重（与程序2一致）——
                graph_weights = compute_anchor_weights(batch, device=x.device)  # [num_graphs, 1]

                # ================= 判别器（D） =================
                for _ in range(3):
                    noise  = make_noise(NOISE_MODE, num_nodes, num_graphs, IN_DIM, x.device,
                                        batch_index, edge_index, smooth=SMOOTH_NOISE, alpha=SMOOTH_ALPHA)
                    x_fake = generator(x_normal, noise, edge_index, batch_index)

                    d_real = discriminator(x_normal, x_extreme, edge_index, batch_index)  # [G,1]
                    d_fake = discriminator(x_normal, x_fake.detach(), edge_index, batch_index)  # [G,1]

                    y_real = torch.ones_like(d_real)
                    y_fake = torch.zeros_like(d_fake)

                    loss_real = (bce(d_real, y_real) * graph_weights).mean()
                    loss_fake = (bce(d_fake, y_fake) * graph_weights).mean()
                    loss_D = loss_real + loss_fake

                    optim_D.zero_grad()
                    loss_D.backward()
                    # 与程序2一致：判别器梯度裁剪
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=10)
                    optim_D.step()

                # ================= 生成器（G） =================
                noise  = make_noise(NOISE_MODE, num_nodes, num_graphs, IN_DIM, x.device,
                                    batch_index, edge_index, smooth=SMOOTH_NOISE, alpha=SMOOTH_ALPHA)
                x_fake = generator(x_normal, noise, edge_index, batch_index)
                g_fake = discriminator(x_normal, x_fake, edge_index, batch_index)

                # 对抗损失（加权）
                adv_G = (bce(g_fake, torch.ones_like(g_fake)) * graph_weights).mean()
                loss_G = adv_G

                # —— 一致性/相关性：仅 ALL 时启用（与之前设计保持一致）——
                if feat == 'ALL' and USE_CONSISTENCY_LOSS:
                    L_cons = consistency_loss(x_fake, x_normal, edge_index,
                                              use_channels=CONS_USE_CHANNELS, sigma=CONS_SIGMA)
                    loss_G = loss_G + LAMBDA_CONS * L_cons
                if feat == 'ALL' and USE_CORR_LOSS:
                    # 建议只对 PRE 做宏观相关；如需多通道可改为 [:, CONS_USE_CHANNELS]
                    L_corr = graphwise_cosine_loss(x_fake[:, [1]] if IN_DIM > 1 else x_fake,
                                                   x_normal[:, [1]] if IN_DIM > 1 else x_normal,
                                                   batch_index)
                    loss_G = loss_G + LAMBDA_CORR * L_corr

                optim_G.zero_grad()
                loss_G.backward()
                optim_G.step()

                epoch_loss_D += float(loss_D.item())
                epoch_loss_G += float(loss_G.item())
                nb += 1

            scheduler_G.step()
            scheduler_D.step()
            avgD = epoch_loss_D / max(1, nb)
            avgG = epoch_loss_G / max(1, nb)
            loss_D_hist.append(avgD)
            loss_G_hist.append(avgG)

            if epoch % 10 == 0:
                print(f"[{feat}] Epoch {epoch:04d}/{NUM_EPOCHS} | D={avgD:.4f} | G={avgG:.4f}  "
                      f"(cons={USE_CONSISTENCY_LOSS and feat=='ALL'}, corr={USE_CORR_LOSS and feat=='ALL'}, "
                      f"noise={NOISE_MODE}, smooth={SMOOTH_NOISE})")
                if feat == 'ALL':
                    real_samples = x_extreme.detach().cpu().numpy()[:, :2]
                    gen_samples  = x_fake.detach().cpu().numpy()[:, :2]
                    visualizer.draw(real_samples, gen_samples, f"{feat} Epoch {epoch:04d}")

        # ===== 保存结果 =====
        suffix = f"compare_{OUT_TAG}_1028"
        os.makedirs("GAN_data/model", exist_ok=True)
        torch.save(generator.state_dict(), f"GAN_data/model/cGAN_generator_{suffix}.pth")
        torch.save(discriminator.state_dict(), f"GAN_data/model/cGAN_discriminator_{suffix}.pth")

        with open(f"GAN_data/model/cGAN_loss_D_list_{suffix}.pickle", "wb") as f:
            pickle.dump(loss_D_hist, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f"GAN_data/model/cGAN_loss_G_list_{suffix}.pickle", "wb") as f:
            pickle.dump(loss_G_hist, f, protocol=pickle.HIGHEST_PROTOCOL)

        # loss 曲线
        plt.figure(figsize=(6,4))
        plt.plot(loss_D_hist, label="Loss_D")
        plt.plot(loss_G_hist, label="Loss_G")
        plt.title(f"Training Loss ({feat})")
        plt.xlabel("Epoch"); plt.ylabel("Loss")
        plt.legend(); plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"GAN_data/model/loss_curve_{suffix}.png", dpi=200)
        plt.close()

    print("✅ All models trained and saved successfully.")

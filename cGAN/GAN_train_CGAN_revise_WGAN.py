# -*- coding: utf-8 -*-
"""
cGAN → WGAN-GP 训练（保持原有采样/去重/锚事件归一不变）
要点：
  - 判别器 D 必须输出 raw score（无 sigmoid）
  - 损失：D: E[D(fake)] - E[D(real)] + λ*GP；G: -E[D(fake)]
  - GP 只在 “被生成的分支” 上做插值（extreme），条件 normal 保持不变
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
# 判别器必须输出 raw score（不要在 forward 里做 sigmoid）
from GAN_network_cgan_revise_WGAN import Generator, Discriminator   # 确保 Discriminator 最后一层不做 sigmoid！
from GAN_dataset import create_graph
from visualizer import GANDemoVisualizer
import matplotlib.pyplot as plt

# ========================= 开关与超参数 =========================
# —— 功能开关 ——
USE_ONLINE_SAMPLING      = False
LIMIT_REPEAT_PER_ANCHOR  = False
BATCH_DEDUP              = False
ANCHOR_UNIT_MASS         = False

# —— 采样与去重参数 ——
BATCH_SIZE        = 150
CAP_PER_ANCHOR    = 2
DT_THR            = 6
SIM_THR           = 0.98

# —— 训练参数（WGAN-GP 常用设定） ——
IN_DIM       = 5
HIDDEN_DIM   = 128
NUM_EPOCHS   = 3000
LR_G         = 1e-4
LR_D         = 1e-4
BETAS        = (0.0, 0.9)    # WGAN-GP 推荐
STEP_SIZE    = 200
GAMMA        = 0.8
SEED         = 42
N_CRITIC     = 3             # 每次更新 G 前，先更新 D 的次数
LAMBDA_GP    = 10.0          # 梯度惩罚系数（常用 5~10）
LAMBDA_DECOR = 0.0           # 若要叠加 decor 正则，这里给一个很小的值，如 1e-4；默认关闭

# ========================= 工具函数 =========================
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

def weighted_mean(scores: torch.Tensor, weights: torch.Tensor):
    """对图级分数做加权平均，权重通常是锚事件单位质量归一后的 graph_weights"""
    return (scores * weights).sum() / (weights.sum() + 1e-8)

# --------- 关键：WGAN-GP（条件图）梯度惩罚 ----------
def gradient_penalty(discriminator, x_norm, real_ext, fake_ext, edge_index, batch):
    """
    仅在“被生成分支”extreme 上插值；D 输出 raw score（[num_graphs,1]）
    GP 在图级做聚合：节点梯度范数 → 同图内均值 → 跨图均值
    """
    device = real_ext.device
    alpha = torch.rand(real_ext.size(0), 1, device=device)
    inter_ext = (alpha * real_ext + (1 - alpha) * fake_ext).requires_grad_(True)

    d_inter = discriminator(x_norm, inter_ext, edge_index, batch)  # [num_graphs,1]
    grad_outputs = torch.ones_like(d_inter, device=device)

    grads = torch.autograd.grad(
        outputs=d_inter,
        inputs=inter_ext,
        grad_outputs=grad_outputs,
        create_graph=True,
        only_inputs=True
    )[0]  # [num_nodes, feat]

    node_norm = grads.view(grads.size(0), -1).norm(2, dim=1)               # [num_nodes]
    num_graphs = int(batch.max().item()) + 1
    graph_sum = torch.zeros(num_graphs, device=device).index_add_(0, batch, node_norm)
    graph_cnt = torch.zeros(num_graphs, device=device).index_add_(0, batch, torch.ones_like(node_norm))
    graph_norm = graph_sum / (graph_cnt + 1e-8)                             # [num_graphs]
    gp = ((graph_norm - 1.0) ** 2).mean()
    return gp

# ========================= 主程序 =========================
if __name__ == '__main__':
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    visualizer = GANDemoVisualizer('WGAN-GP Graph Visualization of WIN+PRE')

    dataset = create_graph(file_name='GAN_data/train/train_1023_clustered.csv')

    # 模型（确保 Discriminator 不做 sigmoid，输出 raw score）
    generator     = Generator(feature_dim=IN_DIM, noise_dim=IN_DIM, hidden_dim=HIDDEN_DIM).to(device)
    discriminator = Discriminator(
        in_dim=IN_DIM, hidden_dim=HIDDEN_DIM,
        # 若你的 Discriminator 有 fusion_type / proj_dim 等可选项，这里照旧传递
        # fusion_type="gated", channel_gate=True, proj_dim=HIDDEN_DIM
    ).to(device)

    optim_G = torch.optim.Adam(generator.parameters(), lr=LR_G, betas=BETAS)
    optim_D = torch.optim.Adam(discriminator.parameters(), lr=LR_D, betas=BETAS)
    scheduler_G = torch.optim.lr_scheduler.StepLR(optim_G, step_size=STEP_SIZE, gamma=GAMMA)
    scheduler_D = torch.optim.lr_scheduler.StepLR(optim_D, step_size=STEP_SIZE, gamma=GAMMA)

    loss_D_hist, loss_G_hist = [], []

    for epoch in range(NUM_EPOCHS):
        epoch_loss_D = 0.0
        epoch_loss_G = 0.0
        nb = 0

        # —— 在线采样 or 普通 DataLoader ——
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
            batch = batch.to(device)

            x = batch.x.to(torch.float)          # (num_nodes, 5, 2)
            edge_index  = batch.edge_index
            batch_index = batch.batch

            x_normal  = x[:, :, 0]
            x_extreme = x[:, :, 1]
            num_nodes = x_normal.shape[0]

            # —— 锚事件单位质量归一权重（图级）——
            graph_weights = compute_anchor_weights(batch, device=device)  # [num_graphs,1]

            # ==================== 训练 D（N_CRITIC 次） ====================
            for _ in range(N_CRITIC):
                noise  = torch.randn(num_nodes, IN_DIM, device=device)
                x_fake = generator(x_normal, noise, edge_index, batch_index).detach()  # 停梯度给 D

                d_real = discriminator(x_normal, x_extreme, edge_index, batch_index)   # [num_graphs,1]
                d_fake = discriminator(x_normal, x_fake,    edge_index, batch_index)   # [num_graphs,1]

                # Wasserstein critic（加权均值）
                wm_real = weighted_mean(d_real, graph_weights)
                wm_fake = weighted_mean(d_fake, graph_weights)

                # Gradient Penalty
                gp = gradient_penalty(discriminator, x_normal, x_extreme, x_fake, edge_index, batch_index)

                # decor 正则（可选）
                decor = torch.tensor(0.0, device=device)
                if LAMBDA_DECOR > 0.0 and hasattr(discriminator, "decor_loss"):
                    decor = discriminator.decor_loss(lam_cross=0.1)

                loss_D = (wm_fake - wm_real) + LAMBDA_GP * gp + LAMBDA_DECOR * decor

                optim_D.zero_grad()
                loss_D.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=10)
                optim_D.step()

            # ==================== 训练 G ====================
            noise  = torch.randn(num_nodes, IN_DIM, device=device)
            x_fake = generator(x_normal, noise, edge_index, batch_index)
            d_fake = discriminator(x_normal, x_fake, edge_index, batch_index)
            wm_fake = weighted_mean(d_fake, graph_weights)

            loss_G = - wm_fake

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
            print(f"[Epoch {epoch:04d}/{NUM_EPOCHS}]  D={avgD:.4f}  G={avgG:.4f}")
            # 可视化（使用上一循环的 x_extreme/x_fake）
            real_samples = x_extreme.detach().cpu().numpy()[:, :2]
            gen_samples  = x_fake.detach().cpu().numpy()[:, :2]
            visualizer.draw(real_samples, gen_samples,
                            f"Epoch {epoch:04d} | D={avgD:.3f} G={avgG:.3f}")

    visualizer.show()

    suffix = '1025_wgan_gp_sampler'
    torch.save(generator.state_dict(), f'GAN_data/model/cGAN_generator_model_{suffix}.pth')
    torch.save(discriminator.state_dict(), f'GAN_data/model/cGAN_discriminator_model_{suffix}.pth')

    with open(f'GAN_data/model/cGAN_loss_D_list_{suffix}.pickle', 'wb') as f:
        pickle.dump(loss_D_hist, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'GAN_data/model/cGAN_loss_G_list_{suffix}.pickle', 'wb') as f:
        pickle.dump(loss_G_hist, f, protocol=pickle.HIGHEST_PROTOCOL)

    # —— loss 曲线 ——
    plt.figure(figsize=(8,5))
    plt.plot(loss_D_hist, label="Critic (D)", color="tab:blue")
    plt.plot(loss_G_hist, label="Generator (G)", color="tab:red")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("WGAN-GP (with Online Pair Sampling & Anchor Normalization)")
    plt.legend(); plt.grid(True); plt.show()

    # —— 生成直方图（最后一个 batch 的 x_fake） ——
    plt.figure()
    plt.hist(x_fake.detach().cpu().numpy().flatten(), bins=20, alpha=0.7, label="Generated Features")
    plt.legend(); plt.show()

# -*- coding: utf-8 -*-
"""
cGAN 训练（可开关）：
  - 随机采样（在线配对采样，非离线穷举）
  - 重复度限制（每个锚事件采样上限）
  - 批内去重（时间/相似度阈值）
  - 锚事件（normal）单位质量归一（损失加权）
  - 【新增】条件一致性损失（Laplacian 平滑 + 图级相关）可开关
  - 【新增】噪声形态可选：节点独立 / 图级广播；可选对噪声做拉普拉斯平滑

修复：统一把 batch 搬到 CUDA
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
# from GAN_network_cgan import Generator, Discriminator
from GAN_network_cgan_revise import Generator, Discriminator
from GAN_dataset import create_graph
from visualizer import GANDemoVisualizer
import matplotlib.pyplot as plt

# ========================= 开关与超参数 =========================
# —— 功能开关 ——
USE_ONLINE_SAMPLING      = False    # 在线随机采样
LIMIT_REPEAT_PER_ANCHOR  = False
BATCH_DEDUP              = False
ANCHOR_UNIT_MASS         = False

# —— 采样与去重参数 ——
BATCH_SIZE        = 800
CAP_PER_ANCHOR    = 2
DT_THR            = 6
SIM_THR           = 0.98

# —— 训练参数 ——
IN_DIM       = 5
HIDDEN_DIM   = 128
NUM_EPOCHS   = 520#3000
LR_G         = 5e-4
LR_D         = 5e-4
BETAS        = (0.5, 0.9)
STEP_SIZE    = 200
GAMMA        = 0.8
SEED         = 42

# ——【新增】条件一致性损失开关与权重 ——
USE_CONSISTENCY_LOSS = False        # r = x_fake - x_norm 的图拉普拉斯一致性
USE_CORR_LOSS        = False       # 图级余弦相关（宏观相关性）
LAMBDA_CONS          = 1e-2        # 建议 1e-2 起步
LAMBDA_CORR          = 5e-3        # 建议 5e-3 起步
CONS_USE_CHANNELS    = [1, 4]      # 参与一致性的通道：如 PRE(1)、SHU(4)
CONS_SIGMA           = 0.2         # 条件权重的高斯尺度

# ——【新增】噪声形态开关 ——
NOISE_MODE           = "per_node"  # "per_node" | "graph"（图级广播噪声）
SMOOTH_NOISE         = False       # 是否对噪声做一次拉普拉斯平滑（抑制破碎）
SMOOTH_ALPHA         = 0.5         # 噪声平滑的邻域权重系数（0~1）

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

# ---------------------- 新增：条件一致性损失 ----------------------
def _pairwise_weight(xi, xj, sigma=0.2):
    # xi,xj: (..., C)
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
    # 图内求和
    fake_g = torch.zeros(num_graphs, x_fake.size(1), device=x_fake.device)\
        .index_add_(0, batch, x_fake)
    norm_g = torch.zeros(num_graphs, x_norm.size(1), device=x_norm.device)\
        .index_add_(0, batch, x_norm)
    dot = (fake_g * norm_g).sum(dim=-1)
    nf  = fake_g.norm(p=2, dim=-1) + 1e-8
    nn  = norm_g.norm(p=2, dim=-1) + 1e-8
    cos = dot / (nf * nn)
    return (1.0 - cos).mean()

# ---------------------- 新增：噪声策略 ----------------------
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

    visualizer = GANDemoVisualizer('GAN Graph Visualization of WIN+PRE')

    # 数据
    dataset = create_graph(file_name='GAN_data/train/train_0313_1_clustered.csv')

    # 模型
    generator     = Generator(feature_dim=IN_DIM, noise_dim=IN_DIM, hidden_dim=HIDDEN_DIM).cuda()
    discriminator = Discriminator(
        in_dim=IN_DIM,
        hidden_dim=HIDDEN_DIM,
        proj_dim=HIDDEN_DIM,
        channel_gate=True,
        use_film=False
    ).cuda()

    LAMBDA_DECOR = 0.0  # 如需叠加 decor，设置为 1e-4 ~ 1e-3

    optim_G = torch.optim.Adam(generator.parameters(), lr=LR_G, betas=BETAS)
    optim_D = torch.optim.Adam(discriminator.parameters(), lr=LR_D, betas=BETAS)
    scheduler_G = torch.optim.lr_scheduler.StepLR(optim_G, step_size=STEP_SIZE, gamma=GAMMA)
    scheduler_D = torch.optim.lr_scheduler.StepLR(optim_D, step_size=STEP_SIZE, gamma=GAMMA)

    bce = nn.BCELoss(reduction='none')  # 逐图，便于施加锚事件权重

    loss_D_hist, loss_G_hist = [], []

    # ✅ 2. 训练前定义一个总的字典，用来存每个 epoch 的 2 维特征
    samples_by_epoch = {}
    for epoch in range(NUM_EPOCHS):
        epoch_loss_D = 0.0
        epoch_loss_G = 0.0
        nb = 0

        # —— 构造 batch 迭代器：在线采样 or 普通 DataLoader ——
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

            x_normal  = x[:, :, 0]
            x_extreme = x[:, :, 1]
            num_nodes = x_normal.shape[0]
            num_graphs = batch.num_graphs

            # —— 锚事件单位质量归一权重 ——
            graph_weights = compute_anchor_weights(batch, device=x.device)  # [num_graphs, 1]

            # ================= 判别器（D） =================
            for _ in range(3):
                noise = make_noise(NOISE_MODE, num_nodes, num_graphs, IN_DIM, x.device,
                                   batch_index, edge_index, smooth=SMOOTH_NOISE, alpha=SMOOTH_ALPHA)
                x_fake = generator(x_normal, noise, edge_index, batch_index)

                d_real = discriminator(x_normal, x_extreme, edge_index, batch_index)  # [num_graphs, 1]
                d_fake = discriminator(x_normal, x_fake,    edge_index, batch_index)  # [num_graphs, 1]

                y_real = torch.ones_like(d_real)
                y_fake = torch.zeros_like(d_fake)

                loss_real = (bce(d_real, y_real) * graph_weights).mean()
                loss_fake = (bce(d_fake, y_fake) * graph_weights).mean()
                loss_D = loss_real + loss_fake

                if LAMBDA_DECOR > 0.0 and hasattr(discriminator, "decor_loss"):
                    loss_D = loss_D + LAMBDA_DECOR * discriminator.decor_loss(lam_cross=0.1)

                optim_D.zero_grad()
                loss_D.backward(retain_graph=False)
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=10)
                optim_D.step()

            # ================= 生成器（G） =================
            noise = make_noise(NOISE_MODE, num_nodes, num_graphs, IN_DIM, x.device,
                               batch_index, edge_index, smooth=SMOOTH_NOISE, alpha=SMOOTH_ALPHA)
            x_fake = generator(x_normal, noise, edge_index, batch_index)
            g_fake = discriminator(x_normal, x_fake, edge_index, batch_index)

            # adversarial
            adv_G = (bce(g_fake, torch.ones_like(g_fake)) * graph_weights).mean()
            loss_G = adv_G

            # ——【新增】条件一致性正则（可开关）——
            if USE_CONSISTENCY_LOSS:
                L_cons = consistency_loss(x_fake, x_normal, edge_index,
                                          use_channels=CONS_USE_CHANNELS, sigma=CONS_SIGMA)
                loss_G = loss_G + LAMBDA_CONS * L_cons
            if USE_CORR_LOSS:
                # 建议只对 PRE 做宏观相关；如需多通道可改为 [:, CONS_USE_CHANNELS]
                L_corr = graphwise_cosine_loss(x_fake[:, [1]], x_normal[:, [1]], batch_index)
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
            print(f"[Epoch {epoch:04d}/{NUM_EPOCHS}]  Loss_D={avgD:.4f}  Loss_G={avgG:.4f}  "
                  f"(cons={USE_CONSISTENCY_LOSS}, corr={USE_CORR_LOSS}, noise={NOISE_MODE}, smooth={SMOOTH_NOISE})")
            real_samples = x_extreme.detach().cpu().numpy()[:, :2]
            gen_samples  = x_fake.detach().cpu().numpy()[:, :2]
            visualizer.draw(real_samples, gen_samples,
                            f"Epoch {epoch:04d} | D={avgD:.3f} G={avgG:.3f}")
            # ✅ 存到总字典里，key 就是 epoch
            samples_by_epoch[epoch] = {
                "real": real_samples,
                "gen": gen_samples,
            }

    visualizer.show()
    # ✅ 训练结束后，把整张表一次性存成 pickle
    with open("samples_by_epoch_bce.pkl", "wb") as f:
        pickle.dump(samples_by_epoch, f)

    suffix = '1102_gated' #1027_gated
    torch.save(generator.state_dict(), f'GAN_data/model/cGAN_generator_model_{suffix}.pth')
    torch.save(discriminator.state_dict(), f'GAN_data/model/cGAN_discriminator_model_{suffix}.pth')

    with open(f'GAN_data/model/cGAN_loss_D_list_{suffix}.pickle', 'wb') as f:
        pickle.dump(loss_D_hist, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'GAN_data/model/cGAN_loss_G_list_{suffix}.pickle', 'wb') as f:
        pickle.dump(loss_G_hist, f, protocol=pickle.HIGHEST_PROTOCOL)

    # —— loss 曲线 ——
    plt.figure(figsize=(8,5))
    plt.plot(loss_D_hist, label="Loss_D", color="tab:blue")
    plt.plot(loss_G_hist, label="Loss_G", color="tab:red")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("cGAN Training with Consistency Loss & Structured Noise (switchable)")
    plt.legend(); plt.grid(True); plt.show()

    # —— 生成直方图（最后一个 batch 的 x_fake） ——
    plt.figure()
    plt.hist(x_fake.detach().cpu().numpy().flatten(), bins=20, alpha=0.7, label="Generated Features")
    plt.legend(); plt.show()

# -*- coding: utf-8 -*-
"""
依次训练 5 种判别器融合策略，加入 WGAN-GP（可开关）：
  fusion_types = ['concat_mlp', 'gated', 'hadamard_concat', 'cross_stitch', 'film']

- USE_WGAN_GP=True 时：采用 Wasserstein 损失 + 梯度惩罚（GP）
- USE_WGAN_GP=False 时：回退到原来的 BCE 训练

注意：你的 Discriminator.forward() 最后带 sigmoid。
为最小改动，这里用 logit(clip(p)) 恢复“原始打分”作为 critic 分数。
"""

import sys, os, random, pickle
sys.path.append(os.path.dirname(__file__))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd

# ====== 使用“可切换融合策略”的判别器与原 Generator ======
# 请确保这里引用的是你上一条提供的文件名
from GAN_network_fuse_comp_1 import Generator, Discriminator

from GAN_dataset import create_graph
from visualizer import GANDemoVisualizer


# ----------------- 小工具 -----------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def critic_score_from_prob(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    你的 D.forward() 输出的是 sigmoid 概率。WGAN 需要“未过 Sigmoid 的打分 f”。
    有 p = sigmoid(f)，因此 f = logit(p)。防止 0/1 溢出，做 clamp。
    """
    p_safe = torch.clamp(p, eps, 1 - eps)
    return torch.log(p_safe) - torch.log(1 - p_safe)  # logit


def gradient_penalty(D, x_norm, x_real_ext, x_fake_ext, edge_index, batch_index, lambda_gp=10.0):
    """
    WGAN-GP 梯度惩罚：对“极端分支”的插值样本做梯度范数约束。
    注意：D 的输入是 (x_norm, x_ext, edge_index, batch)。
    """
    device = x_real_ext.device
    N, C = x_real_ext.shape
    alpha = torch.rand(N, 1, device=device)
    alpha = alpha.expand_as(x_real_ext)
    interpolates = alpha * x_real_ext + (1 - alpha) * x_fake_ext
    interpolates.requires_grad_(True)

    # 判别器输出概率 -> 转换为 critic 分数
    d_interp_prob = D(x_norm, interpolates, edge_index, batch_index)
    d_interp = critic_score_from_prob(d_interp_prob)

    grad_outputs = torch.ones_like(d_interp, device=device)

    # 这里对“极端节点特征”求梯度
    gradients = torch.autograd.grad(
        outputs=d_interp,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    if gradients is None:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # GP： (||grad||_2 - 1)^2
    gradients = gradients.view(N, -1)
    grad_norm = gradients.norm(2, dim=1)
    gp = ((grad_norm - 1.0) ** 2).mean()
    return lambda_gp * gp


# ----------------- 训练一个融合策略 -----------------
def train_one_fusion(fusion_type: str,
                     input_csv='GAN_data/train/train_0313_1_clustered.csv',
                     batch_size=400,
                     in_dim=5,
                     hidden_dim=72,
                     proj_dim=8,
                     lambda_avg=0.7,
                     num_epochs=1000,
                     lr=5e-4,
                     betas=(0.5, 0.9),
                     stepsize=150,
                     gamma=0.8,
                     save_tag='1026',
                     # ===== WGAN-GP 相关开关与超参 =====
                     USE_WGAN_GP=True,
                     N_CRITIC=5,
                     LAMBDA_GP=10.0):
    """
    训练单个融合策略；返回 (loss_D_list, loss_G_list)
    """
    print(f"\n====== Train fusion: {fusion_type} | WGAN-GP={USE_WGAN_GP} ======")
    set_seed(42)

    # 重新加载数据
    dataset    = create_graph(file_name=input_csv)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 可视化器
    visualizer = GANDemoVisualizer(f'GAN Graph Visualization | fusion={fusion_type}')

    # 构建模型
    G = Generator(feature_dim=in_dim, noise_dim=in_dim, hidden_dim=hidden_dim).cuda()
    D = Discriminator(in_dim=in_dim, hidden_dim=hidden_dim,
                      proj_dim=proj_dim, fusion_type=fusion_type,lambda_avg=lambda_avg).cuda()

    # 优化器与调度器
    if USE_WGAN_GP:
        # 常见做法：Adam(lr=1e-4, betas=(0.0, 0.9))；这里保持与你原配置相近，也可自行调参
        optim_G = torch.optim.Adam(G.parameters(), lr=lr, betas=betas)
        optim_D = torch.optim.Adam(D.parameters(), lr=lr, betas=betas)
    else:
        optim_G = torch.optim.Adam(G.parameters(), lr=lr, betas=betas)
        optim_D = torch.optim.Adam(D.parameters(), lr=lr , betas=betas)

    scheduler_G = torch.optim.lr_scheduler.StepLR(optim_G, step_size=stepsize, gamma=gamma)
    scheduler_D = torch.optim.lr_scheduler.StepLR(optim_D, step_size=stepsize, gamma=gamma)

    bce = nn.BCELoss().cuda()

    # 训练前，放在 train_one_fusion 开头构图和优化器之后
    fixed_z = None
    div_hist = []  # 记录批内多样性

    def batch_diversity(x_fake_np):
        # 批内简单多样性指标: 每列标准差的均值 与 样本两两距离的均值
        if x_fake_np.ndim == 1:
            x_fake_np = x_fake_np[:, None]
        std_mean = np.std(x_fake_np, axis=0).mean()
        # 采样一小部分节点计算两两距离
        idx = np.random.choice(x_fake_np.shape[0], size=min(256, x_fake_np.shape[0]), replace=False)
        xx = x_fake_np[idx]
        d = ((xx[None, :, :] - xx[:, None, :]) ** 2).sum(-1) ** 0.5
        return float(std_mean), float(d.mean())


    # 记录
    loss_D_list, loss_G_list = [], []

    for epoch in range(num_epochs):
        epoch_loss_D = 0.0
        epoch_loss_G = 0.0
        num_batches  = 0

        for batch in dataloader:
            batch = batch.cuda()

            # 数据拆分
            x = batch.x.to(torch.float)              # [N, 5, 2]
            edge_index  = batch.edge_index.to(torch.long)
            batch_index = batch.batch

            x_norm = x[:, :, 0]                     # 条件
            x_ext  = x[:, :, 1]                     # 真实极端
            N = x_norm.shape[0]

            if USE_WGAN_GP:
                # -------- 训练 D：WGAN-GP --------
                for _ in range(N_CRITIC):
                    z = torch.randn(N, in_dim, device=x.device)
                    x_fake = G(x_norm, z, edge_index, batch_index)

                    # 判别器输出概率 -> 转换为 critic 分数
                    d_real_prob = D(x_norm, x_ext,  edge_index, batch_index)  # [G,1]
                    d_fake_prob = D(x_norm, x_fake, edge_index, batch_index)

                    d_real = critic_score_from_prob(d_real_prob)
                    d_fake = critic_score_from_prob(d_fake_prob)

                    # Wasserstein 损失
                    loss_D_w = d_fake.mean() - d_real.mean()

                    # GP
                    gp = gradient_penalty(D, x_norm, x_ext, x_fake, edge_index, batch_index, lambda_gp=LAMBDA_GP)

                    loss_D = loss_D_w + gp

                    optim_D.zero_grad()
                    loss_D.backward(retain_graph=False)
                    torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=10)
                    optim_D.step()

                # -------- 训练 G：Wasserstein --------
                z = torch.randn(N, in_dim, device=x.device)
                x_fake = G(x_norm, z, edge_index, batch_index)
                g_fake_prob = D(x_norm, x_fake, edge_index, batch_index)
                g_fake = critic_score_from_prob(g_fake_prob)

                loss_G = - g_fake.mean()

                optim_G.zero_grad()
                loss_G.backward()
                optim_G.step()

                # ====== 诊断统计，每个 batch 累积到 epoch 末打印 ======
                with torch.no_grad():
                    # 概率与分数的均值和方差
                    p_real_m = float(d_real_prob.mean().item());
                    p_fake_m = float(d_fake_prob.mean().item())
                    f_real_m = float(d_real.mean().item());
                    f_fake_m = float(d_fake.mean().item())
                    # GP 与 Wasserstein gap
                    gp_val = float(gp.item());
                    w_gap = float((-loss_D_w).item())  # 期望 d_real - d_fake
                    # 批内多样性
                    std_m, dist_m = batch_diversity(x_fake.detach().cpu().numpy())
                    div_hist.append((std_m, dist_m))
            else:
                # -------- 原 BCE 训练路径（回退）--------
                # 训练 D
                for _ in range(N_CRITIC):
                    z = torch.randn(N, in_dim, device=x.device)
                    x_fake = G(x_norm, z, edge_index, batch_index)

                    d_real = D(x_norm, x_ext,  edge_index, batch_index)
                    d_fake = D(x_norm, x_fake, edge_index, batch_index)

                    y_real = torch.ones_like(d_real)
                    y_fake = torch.zeros_like(d_fake)

                    loss_D = bce(d_real, y_real) + bce(d_fake, y_fake)
                    optim_D.zero_grad()
                    loss_D.backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=10)
                    optim_D.step()

                # 训练 G
                z = torch.randn(N, in_dim, device=x.device)
                x_fake = G(x_norm, z, edge_index, batch_index)
                g_fake = D(x_norm, x_fake, edge_index, batch_index)
                loss_G = bce(g_fake, torch.ones_like(g_fake))
                optim_G.zero_grad()
                loss_G.backward()
                optim_G.step()




            # 统计
            epoch_loss_D += float(loss_D.item())
            epoch_loss_G += float(loss_G.item())
            num_batches  += 1

        # 学习率调度
        scheduler_G.step(); scheduler_D.step()

        avgD = epoch_loss_D / max(1, num_batches)
        avgG = epoch_loss_G / max(1, num_batches)
        loss_D_list.append(avgD); loss_G_list.append(avgG)

        if epoch % 10 == 0 and epoch>0:
            mode = "WGAN-GP" if USE_WGAN_GP else "BCE"
            msg = f"[{fusion_type}][{mode}] Epoch {epoch:04d}/{num_epochs} | Loss_D={avgD:.4f} | Loss_G={avgG:.4f}"
            print(msg)
            # 可视化两个通道（仅取前2维）
            real_samples = x_ext.detach().cpu().numpy()[:, :2]
            gen_samples  = x_fake.detach().cpu().numpy()[:, :2]
            visualizer.draw(real_samples, gen_samples, msg)

            # std_avg = np.mean([s for s, _ in div_hist]) if div_hist else float('nan')
            # dist_avg = np.mean([d for _, d in div_hist]) if div_hist else float('nan')
            # print(
            #     f"[{fusion_type}][WGAN-GP] Ep {epoch:04d} "
            #     f"| LossD {avgD:.4f} LossG {avgG:.4f} "
            #     f"| w_gap≈{w_gap:.4f} gp≈{gp_val:.4f} "
            #     f"| p_real≈{p_real_m:.3f} p_fake≈{p_fake_m:.3f} "
            #     f"| f_real≈{f_real_m:.3f} f_fake≈{f_fake_m:.3f} "
            #     f"| div_std≈{std_avg:.4f} div_dist≈{dist_avg:.4f}"
            # )
            # # div_hist.clear()
            #
            # # 固定噪声评估，观察生成是否塌到单模
            # if fixed_z is None:
            #     fixed_z = torch.randn(min(2048, N), in_dim, device=x.device)
            # with torch.no_grad():
            #     subN = min(N, fixed_z.size(0))
            #     x_fake_fix = G(x_norm[:subN], fixed_z[:subN], edge_index, batch_index[:subN])
            #     std_fix, dist_fix = batch_diversity(x_fake_fix.detach().cpu().numpy())
            #     print(f"    fixed_noise diversity: std={std_fix:.4f} dist={dist_fix:.4f}")

    # 保存
    os.makedirs('GAN_data/model', exist_ok=True)
    suffix = f'{save_tag}_{fusion_type}' + ('_wgangp' if USE_WGAN_GP else '_bce')
    torch.save(G.state_dict(), f'GAN_data/model/cGAN_generator_model_{suffix}.pth')
    torch.save(D.state_dict(), f'GAN_data/model/cGAN_discriminator_model_{suffix}.pth')

    with open(f'GAN_data/model/cGAN_loss_D_list_{suffix}.pickle', 'wb') as f:
        pickle.dump(loss_D_list, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'GAN_data/model/cGAN_loss_G_list_{suffix}.pickle', 'wb') as f:
        pickle.dump(loss_G_list, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Loss 曲线
    plt.figure(figsize=(8,5))
    plt.plot(loss_D_list, label="Loss_D", color="tab:blue")
    plt.plot(loss_G_list, label="Loss_G", color="tab:red")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title(f"Training | fusion={fusion_type} | {'WGAN-GP' if USE_WGAN_GP else 'BCE'}")
    plt.legend(); plt.grid(True)
    plt.savefig(f'GAN_data/model/cGAN_losses_{suffix}.png', dpi=180, bbox_inches='tight')
    plt.close()

    return loss_D_list, loss_G_list


# ----------------- 主入口：循环 5 种融合策略 -----------------
if __name__ == '__main__':
    set_seed(42)

    fusion_types = ['gated'] #'concat_mlp', 'gated', 'hadamard_concat', 'cross_stitch', 'film' 'concat_mlp', 'hadamard_concat', 'cross_stitch', 'film'
    input_csv = 'GAN_data/train/train_0313_1_clustered.csv'

    for ftype in fusion_types:
        torch.cuda.empty_cache()
        try:
            train_one_fusion(
                fusion_type=ftype,
                input_csv=input_csv,
                batch_size=400,
                in_dim=5,
                hidden_dim=72,
                proj_dim=72,
                num_epochs=3000,
                lr=5e-4,
                betas=(0.5, 0.9), #(0.5, 0.9)
                stepsize=150,#150
                gamma=0.8,#0.8
                save_tag='1104',
                USE_WGAN_GP=False,     # <<< 开关：改 False 回到 BCE
                N_CRITIC=1,           # 常见 3~5
                LAMBDA_GP=5.0,       # 常见 10
                lambda_avg=0.5,
            )
        finally:
            # 彻底释放当前策略对象
            for k in list(globals().keys()):
                v = globals()[k]
                if isinstance(v, (torch.nn.Module, torch.optim.Optimizer)):
                    del globals()[k]
            torch.cuda.empty_cache()

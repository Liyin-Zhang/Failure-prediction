# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.path.dirname(__file__))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
import torch.nn as nn
from torch_geometric.data import Data, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random

from GAN_network_cgan_SAGE import Generator, Discriminator   # ← 使用下方给的新网络文件
from GAN_dataset import create_graph
from visualizer import GANDemoVisualizer

# =============== 可选：固定随机种子 ===============
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# =============== 选择要生成的单通道特征 ===============
# 支持：WIN PRE PRS SHU TMP
TARGET_FEAT = "SHU"
FEAT_INDEX = {"WIN": 0, "PRE": 1, "PRS": 2, "SHU": 3, "TMP": 4}[TARGET_FEAT]

if __name__ == '__main__':
    set_seed(42)
    visualizer = GANDemoVisualizer(f'Graph cGAN single feature: {TARGET_FEAT}')

    # 数据
    dataset = create_graph(file_name='GAN_data/train/train_0313_1_clustered.csv')
    dataloader = DataLoader(dataset, batch_size=800, shuffle=True)

    # 单通道设置
    in_dim = 1
    hidden_dim = 24

    # 模型
    generator = Generator(feature_dim=in_dim, noise_dim=in_dim, hidden_dim=hidden_dim).cuda()
    discriminator = Discriminator(in_dim=in_dim, hidden_dim=hidden_dim).cuda()

    # 优化器与调度
    stepsize = 200
    gama = 0.8
    optim_G = torch.optim.Adam(generator.parameters(), lr=0.0005, betas=(0.5, 0.9))
    optim_D = torch.optim.Adam(discriminator.parameters(), lr=0.0005, betas=(0.5, 0.9))
    scheduler_G = torch.optim.lr_scheduler.StepLR(optim_G, step_size=stepsize, gamma=gama)
    scheduler_D = torch.optim.lr_scheduler.StepLR(optim_D, step_size=stepsize, gamma=gama)

    # 损失
    bce = nn.BCELoss().cuda()

    # 训练
    num_epochs = 3000
    loss_D_list, loss_G_list = [], []

    for epoch in range(num_epochs):
        epoch_loss_D = 0.0
        epoch_loss_G = 0.0
        num_batches = 0

        for batch in dataloader:
            batch = batch.cuda()
            x = batch.x.to(torch.float)                 # [N, 5, 2]
            edge_index = batch.edge_index.to(torch.long)
            batch_index = batch.batch

            # 单通道切片：条件和目标均为该通道
            normal_features = x[:, [FEAT_INDEX], 0]     # [N, 1]
            extreme_features = x[:, [FEAT_INDEX], 1]    # [N, 1]

            N = normal_features.size(0)

            # 训练判别器
            for _ in range(3):
                noise = torch.randn(N, in_dim, device=x.device)     # [N, 1]
                fake_extreme = generator(normal_features, noise, edge_index, batch_index)  # [N, 1]

                d_real = discriminator(normal_features, extreme_features, edge_index, batch_index)  # [G, 1]
                d_fake = discriminator(normal_features, fake_extreme.detach(), edge_index, batch_index)  # [G, 1]

                y_real = torch.ones_like(d_real)
                y_fake = torch.zeros_like(d_fake)

                loss_D = bce(d_real, y_real) + bce(d_fake, y_fake)

                optim_D.zero_grad()
                loss_D.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=10)
                optim_D.step()

            # 训练生成器
            noise = torch.randn(N, in_dim, device=x.device)
            fake_extreme = generator(normal_features, noise, edge_index, batch_index)       # [N, 1]
            g_fake = discriminator(normal_features, fake_extreme, edge_index, batch_index)  # [G, 1]
            loss_G = bce(g_fake, torch.ones_like(g_fake))

            optim_G.zero_grad()
            loss_G.backward()
            optim_G.step()

            epoch_loss_D += float(loss_D.item())
            epoch_loss_G += float(loss_G.item())
            num_batches += 1

        scheduler_G.step()
        scheduler_D.step()

        avgD = epoch_loss_D / max(1, num_batches)
        avgG = epoch_loss_G / max(1, num_batches)
        loss_D_list.append(avgD)
        loss_G_list.append(avgG)

        if epoch % 10 == 0:
            print(f"[{TARGET_FEAT}] Epoch {epoch:04d}/{num_epochs}  D={avgD:.4f}  G={avgG:.4f}")
            # 可视化需要二维，这里把单通道复制一份形成二维
            real_samples = extreme_features.detach().cpu().numpy()
            gen_samples = fake_extreme.detach().cpu().numpy()
            real_vis = np.c_[real_samples, real_samples]     # [N, 2]
            gen_vis  = np.c_[gen_samples, gen_samples]       # [N, 2]
            visualizer.draw(real_vis, gen_vis, f"{TARGET_FEAT} Epoch {epoch:04d}")

    visualizer.show()

    # 保存
    suffix = f"1028_sage_single_{TARGET_FEAT.lower()}"
    os.makedirs("GAN_data/model", exist_ok=True)
    torch.save(generator.state_dict(), f"GAN_data/model/cGAN_generator_{suffix}.pth")
    torch.save(discriminator.state_dict(), f"GAN_data/model/cGAN_discriminator_{suffix}.pth")

    with open(f"GAN_data/model/cGAN_loss_D_list_{suffix}.pickle", "wb") as f:
        pickle.dump(loss_D_list, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f"GAN_data/model/cGAN_loss_G_list_{suffix}.pickle", "wb") as f:
        pickle.dump(loss_G_list, f, protocol=pickle.HIGHEST_PROTOCOL)

    # 简单直方图
    plt.figure()
    plt.hist(fake_extreme.detach().cpu().numpy().flatten(), bins=20, alpha=0.7, label=f"Generated {TARGET_FEAT}")
    plt.legend(); plt.tight_layout(); plt.show()

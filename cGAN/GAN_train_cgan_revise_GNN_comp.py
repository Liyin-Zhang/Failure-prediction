import sys, os, re, random, pickle
sys.path.append(os.path.dirname(__file__))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from GAN_network_cgan_revise_GNN_comp import *           # 兼容：若该文件已支持 gnn_type，将被用到；否则回退为原SAGE
from GAN_dataset import create_graph
from visualizer import GANDemoVisualizer

# ---------------- utils ----------------
def variance_matching_loss(real_x, fake_x):
    return torch.abs(torch.var(real_x) - torch.var(fake_x))

def feature_matching_loss(real_x, fake_x):
    return F.mse_loss(real_x.mean(dim=0), fake_x.mean(dim=0))

def gradient_penalty(discriminator, real_x, fake_x, edge_index, batch_index):
    alpha = torch.rand(real_x.shape[0], 1, device=real_x.device).expand_as(real_x)
    interpolates = alpha * real_x + (1 - alpha) * fake_x
    interpolates.requires_grad_(True)
    d_interpolates = discriminator(interpolates, edge_index, batch_index)
    grad_outputs = torch.ones_like(d_interpolates, device=real_x.device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates, inputs=interpolates, grad_outputs=grad_outputs,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    if gradients is None:
        return torch.tensor(0.0, device=real_x.device, requires_grad=True)
    gradients_norm = gradients.view(gradients.shape[0], -1).norm(2, dim=1)
    penalty = ((gradients_norm - 1) ** 2).mean()
    return penalty

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ---------------- train one backbone ----------------
def train_one_backbone(backbone: str,
                       dataset_path='GAN_data/train/train_0313_1_clustered.csv',
                       batch_size=400, in_dim=5, hidden_dim=72,
                       num_epochs=3000, lr=5e-4, betas=(0.5,0.9),
                       stepsize=200, gama=0.8, n_critic=3, suffix_base='1026'):
    """
    复用你现有训练循环；仅在模型实例化与保存名上区分骨架
    """
    print(f"\n====== Train Backbone: {backbone.upper()} ======")
    visualizer = GANDemoVisualizer(f'GAN Graph Visualization ({backbone})')

    dataset = create_graph(file_name=dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    set_seed(42)

    # ---- 定义模型（兼容：若类不支持 gnn_type 参数，则回退为原SAGE构造）----
    try:
        generator = Generator(feature_dim=in_dim, noise_dim=in_dim, hidden_dim=hidden_dim,
                              gnn_type=backbone).cuda()
    except TypeError:
        generator = Generator(feature_dim=in_dim, noise_dim=in_dim, hidden_dim=hidden_dim).cuda()

    try:
        discriminator = Discriminator(in_dim=in_dim, hidden_dim=hidden_dim,
                                      gnn_type=backbone).cuda()
    except TypeError:
        discriminator = Discriminator(in_dim=in_dim, hidden_dim=hidden_dim).cuda()

    optim_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=betas)
    optim_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=betas)

    loss_fn = nn.BCELoss().cuda()
    scheduler_G = torch.optim.lr_scheduler.StepLR(optim_G, step_size=stepsize, gamma=gama)
    scheduler_D = torch.optim.lr_scheduler.StepLR(optim_D, step_size=stepsize, gamma=gama)

    loss_D_list, loss_G_list = [], []

    for epoch in range(num_epochs):
        epoch_loss_D = 0.0
        epoch_loss_G = 0.0
        num_batches  = 0

        for data in dataloader:
            data = data.cuda()

            real_x = data.x.to(torch.float)       # (num_nodes, 5, 2)
            edge_index  = data.edge_index.to(torch.long)
            batch_index = data.batch

            normal_features  = real_x[:, :, 0]    # (num_nodes, 5)
            extreme_features = real_x[:, :, 1]    # (num_nodes, 5)
            num_nodes = normal_features.shape[0]

            # ---- 训练判别器 D ----
            for _ in range(n_critic):
                noise = torch.randn(num_nodes, in_dim, device=real_x.device)
                fake_extreme_features = generator(normal_features, noise, edge_index, batch_index)

                d_real = discriminator(normal_features, extreme_features, edge_index, batch_index)
                d_fake = discriminator(normal_features, fake_extreme_features, edge_index, batch_index)

                real_labels = torch.ones(data.num_graphs, 1, device=real_x.device)
                fake_labels = torch.zeros(data.num_graphs, 1, device=real_x.device)

                loss_D = loss_fn(d_real, real_labels) + loss_fn(d_fake, fake_labels)

                optim_D.zero_grad()
                loss_D.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=10)
                optim_D.step()

            # ---- 训练生成器 G ----
            noise = torch.randn(num_nodes, in_dim, device=real_x.device)
            fake_extreme_features = generator(normal_features, noise, edge_index, batch_index)
            g_fake = discriminator(normal_features, fake_extreme_features, edge_index, batch_index)

            loss_G = loss_fn(g_fake, real_labels)

            optim_G.zero_grad()
            loss_G.backward()
            optim_G.step()

            # ---- 记录 ----
            epoch_loss_D += loss_D.item()
            epoch_loss_G += loss_G.item()
            num_batches  += 1

        # ---- 学习率调度 ----
        scheduler_G.step()
        scheduler_D.step()

        avg_loss_D = epoch_loss_D / num_batches
        avg_loss_G = epoch_loss_G / num_batches
        loss_D_list.append(avg_loss_D)
        loss_G_list.append(avg_loss_G)

        if epoch % 10 == 0:
            msg = f"[{backbone.upper()}] Epoch [{epoch}/{num_epochs}] | D: {avg_loss_D:.4f} | G: {avg_loss_G:.4f}"
            print(msg)
            # 可视化两维散点（与原逻辑一致）
            real_samples = extreme_features.detach().cpu().numpy()[:, :2]
            gen_samples  = fake_extreme_features.detach().cpu().numpy()[:, :2]
            visualizer.draw(real_samples, gen_samples, msg)

    # visualizer.show()

    # ---- 保存模型与loss ----
    suffix = f'{suffix_base}_{backbone}'
    os.makedirs('GAN_data/model', exist_ok=True)
    torch.save(generator.state_dict(),     f'GAN_data/model/cGAN_generator_model_{suffix}.pth')
    torch.save(discriminator.state_dict(), f'GAN_data/model/cGAN_discriminator_model_{suffix}.pth')

    with open(f'GAN_data/model/cGAN_loss_D_list_{suffix}.pickle', 'wb') as f:
        pickle.dump(loss_D_list, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'GAN_data/model/cGAN_loss_G_list_{suffix}.pickle', 'wb') as f:
        pickle.dump(loss_G_list, f, protocol=pickle.HIGHEST_PROTOCOL)

    # ---- 绘制并保存损失曲线 ----
    plt.figure(figsize=(8,5))
    plt.plot(loss_D_list, label="Loss_D (Discriminator)", color="blue")
    plt.plot(loss_G_list, label="Loss_G (Generator)", color="red")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title(f"{backbone.upper()} | GAN Training Loss Curve")
    plt.legend(); plt.grid(True)
    plt.savefig(f'GAN_data/model/cGAN_losses_{suffix}.png', dpi=180, bbox_inches='tight')
    plt.close()


# ---------------- main：依次训练四种骨架 ----------------
if __name__ == '__main__':
    set_seed(42)

    # 公用的数据路径与超参（与你原脚本一致）
    DATASET_PATH = 'GAN_data/train/train_0313_1_clustered.csv'
    BACKBONES = ['sage', 'gat', 'gin', 'transformer']   # 依次训练并保存

    for bk in BACKBONES:
        train_one_backbone(
            backbone=bk,
            dataset_path=DATASET_PATH,
            batch_size=400,
            in_dim=5,
            hidden_dim=72,
            num_epochs=3000,
            lr=5e-4,
            betas=(0.5,0.9),
            stepsize=200,
            gama=0.8,
            n_critic=3,
            suffix_base='1026'
        )

import sys, os, random, pickle
sys.path.append(os.path.dirname(__file__))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pickle5
import pandas as pd
from GAN_network_0305 import *
from GAN_dataset import create_graph
import random
import pickle
from visualizer import GANDemoVisualizer
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def variance_matching_loss(real_x, fake_x):
    return torch.abs(torch.var(real_x) - torch.var(fake_x))

def feature_matching_loss(real_x, fake_x):
    return F.mse_loss(real_x.mean(dim=0), fake_x.mean(dim=0))  # 计算特征均值的匹配

def gradient_penalty(discriminator, real_x, fake_x, edge_index, batch_index):
    """计算 WGAN-GP 梯度惩罚"""
    alpha = torch.rand(real_x.shape[0], 1, device=real_x.device).expand_as(real_x)  # 生成随机权重 alpha
    interpolates = alpha * real_x + (1 - alpha) * fake_x  # 在 real_x 和 fake_x 之间插值
    interpolates.requires_grad_(True)  # 需要梯度计算

    d_interpolates = discriminator(real_x, interpolates, edge_index, batch_index)  # 判别器得分

    # 计算梯度
    grad_outputs = torch.ones_like(d_interpolates, device=real_x.device)  # 确保 shape 兼容
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    # 确保梯度不为 None
    if gradients is None:
        return torch.tensor(0.0, device=real_x.device, requires_grad=True)

    # 计算梯度范数的偏离 1 的惩罚项
    gradients_norm = gradients.view(gradients.shape[0], -1).norm(2, dim=1)  # 展平再计算 L2 范数
    penalty = ((gradients_norm - 1) ** 2).mean()

    return penalty


def set_seed(seed=42):
    random.seed(seed)  # Python 内置随机数
    np.random.seed(seed)  # NumPy 随机数
    torch.manual_seed(seed)  # PyTorch 随机数（CPU）

    # 如果使用 GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 固定 CUDA 随机数
        torch.cuda.manual_seed_all(seed)  # 多 GPU 也固定
        torch.backends.cudnn.deterministic = True  # 让 cuDNN 计算可复现
        torch.backends.cudnn.benchmark = False  # 禁止 cuDNN 自动优化（会影响复现性）




if __name__ == '__main__':

    visualizer = GANDemoVisualizer('GAN Grapn Visualization of WIN+PRE')


    # 将数据转换为图结构
    dataset = create_graph(file_name='GAN_data/train/train_0313_1_clustered.csv')
    # 使用DataLoader进行批量处理
    dataloader = DataLoader(dataset, batch_size=400, shuffle=True)

    # 训练前调用
    set_seed(42)

    # 定义模型
    in_dim=5
    hidden_dim=128

    generator = Generator(feature_dim=in_dim,noise_dim=in_dim, hidden_dim=hidden_dim)
    discriminator = Discriminator(in_dim=in_dim, hidden_dim=hidden_dim)
    generator=generator.cuda()
    discriminator=discriminator.cuda()

    # 定义优化器
    stepsize=200
    gama=0.7
    # learning_rate=0.0008
    lambda_gp =  5  # 梯度惩罚系数
    n_critic = 3  # 训练 D 的次数

    optim_G = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.9))
    optim_D = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.9))

    loss_fn = nn.BCELoss()
    loss_fn=loss_fn.cuda()

    scheduler_G = torch.optim.lr_scheduler.StepLR(optim_G, step_size=stepsize, gamma=gama)
    scheduler_D = torch.optim.lr_scheduler.StepLR(optim_D, step_size=stepsize, gamma=gama)

    # 训练循环
    num_epochs = 520
    # 记录 Loss
    loss_D_list = []
    loss_G_list = []
    D_real_list=[]
    D_fake_list=[]

    # ✅ 2. 训练前定义一个总的字典，用来存每个 epoch 的 2 维特征
    samples_by_epoch = {}

    for epoch in range(num_epochs):
        epoch_loss_D = 0.0
        epoch_loss_G = 0.0
        num_batches = 0
        D_real_list = []
        D_fake_list = []

        for data in dataloader:
            data = data.cuda()

            # **1️⃣ 处理输入数据**
            real_x = data.x.to(torch.float)  # 原始输入数据 (num_nodes, 5, 2)
            edge_index = data.edge_index.to(torch.long)  # 图的边索引
            batch_index = data.batch  # 每个节点对应的图 ID

            normal_features = real_x[:, :, 0]  # 一般天气特征 (num_nodes, 5)
            extreme_features = real_x[:, :, 1]  # 极端天气特征 (num_nodes, 5)

            batch_size = normal_features.shape[0]

            # **2️⃣ 训练判别器（D）**
            for _ in range(1):
                noise = torch.randn(batch_size, in_dim).cuda()  # 生成随机噪声 (batch_size, 5)
                fake_extreme_features = generator(normal_features, noise, edge_index, batch_index)  # 生成极端天气特征

                d_real = discriminator(normal_features, extreme_features, edge_index, batch_index)  # 判别真实数据
                d_fake = discriminator(normal_features, fake_extreme_features, edge_index, batch_index)  # 判别生成数据

                real_labels = torch.ones(data.num_graphs, 1).cuda()  # 真实样本的标签 = 1
                fake_labels = torch.zeros(data.num_graphs, 1).cuda()  # 生成样本的标签 = 0

                # 计算梯度惩罚
                gp = gradient_penalty(discriminator, extreme_features, fake_extreme_features, edge_index, batch_index)

                # WGAN-GP 判别器损失
                loss_D =  d_fake.mean() - d_real.mean() + lambda_gp * gp


                # loss_D = loss_fn(d_real, real_labels) + loss_fn(d_fake, fake_labels)

                optim_D.zero_grad()
                loss_D.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=10)  # 限制梯度范数，防止梯度爆炸
                optim_D.step()

            # **3️⃣ 训练生成器（G）**
            for _ in range(1):
                noise = torch.randn(batch_size, in_dim).cuda()
                fake_extreme_features = generator(normal_features, noise, edge_index, batch_index)  # 生成假的极端天气特征

                g_fake = discriminator(normal_features, fake_extreme_features, edge_index, batch_index)  # 生成数据的评分
                # loss_G = loss_fn(g_fake, real_labels)  # 让生成数据尽可能像真实数据
                la=-g_fake.mean()
                # lb=feature_matching_loss(real_x, fake_x)
                # lc=torch.mean(fake_x ** 2) # torch.mean(torch.abs(fake_x)) 鼓励生成数据远离0
                # ld=variance_matching_loss(real_x, fake_x)
                loss_G = la#+w_fm*ld#+w_reg*lc+0.1*ld
                optim_G.zero_grad()
                loss_G.backward()
                optim_G.step()

            # **4️⃣ 记录损失**
            epoch_loss_D += loss_D.item()
            epoch_loss_G += loss_G.item()
            num_batches += 1
            D_real_list.append(round(sum(d_real.detach().flatten().tolist()), 2))
            D_fake_list.append(round(sum(d_fake.detach().flatten().tolist()), 2))

        # **5️⃣ 计算平均损失**
        avg_loss_D = epoch_loss_D / num_batches
        avg_loss_G = epoch_loss_G / num_batches
        loss_D_list.append(avg_loss_D)
        loss_G_list.append(avg_loss_G)

        # **6️⃣ 更新学习率**
        scheduler_G.step()
        scheduler_D.step()

        # **7️⃣ 可视化**
        if epoch % 10 == 0:
            msg = f"Epoch [{epoch}/{num_epochs}], Loss_D: {avg_loss_D:.4f}, Loss_G: {avg_loss_G:.4f}"
            print(msg)
            print(gp)
            # 仅取前 2 维特征进行可视化
            real_samples = np.array(extreme_features.detach().tolist())[:, :2]
            gen_samples = np.array(fake_extreme_features.detach().tolist())[:, :2]

            # ✅ 存到总字典里，key 就是 epoch
            samples_by_epoch[epoch] = {
                "real": real_samples,
                "gen": gen_samples,
            }

            visualizer.draw(real_samples, gen_samples, msg)
            # visualizer.savefig(f"viz_out/epoch_{epoch}.png")





    visualizer.show()

    # ✅ 训练结束后，把整张表一次性存成 pickle
    with open("samples_by_epoch_wgan.pkl", "wb") as f:
        pickle.dump(samples_by_epoch, f)


    suffix='1102_WGAN'
    torch.save(generator.state_dict(), 'GAN_data/model/generator_model_'+suffix+'.pth')
    torch.save(discriminator.state_dict(), 'GAN_data/model/discriminator_model_'+suffix+'.pth')

    # with open('GAN_data/model/loss_D_list_'+suffix+'.pickle', 'wb') as f:
    #     pickle.dump(loss_D_list, f, protocol=pickle.HIGHEST_PROTOCOL)
    # with open('GAN_data/model/loss_G_list_'+suffix+'.pickle', 'wb') as f:
    #     pickle.dump(loss_G_list, f, protocol=pickle.HIGHEST_PROTOCOL)


    # 绘制损失曲线
    plt.figure(figsize=(8, 5))
    plt.plot(loss_D_list, label="Loss_D (Discriminator)", color="blue")
    plt.plot(loss_G_list, label="Loss_G (Generator)", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("GAN Training Loss Curve")
    plt.legend()
    plt.grid()


    # 测试生成新的节点特征
    for batch in dataloader:
        batch=batch.cuda()
        # **1️⃣ 处理输入数据**
        real_x = batch.x.to(torch.float)  # 原始输入数据 (num_nodes, 5, 2)
        edge_index = batch.edge_index.to(torch.long)  # 图的边索引
        batch_index = batch.batch  # 每个节点对应的图 ID

        normal_features = real_x[:, :, 0]  # 一般天气特征 (num_nodes, 5)
        extreme_features = real_x[:, :, 1]  # 极端天气特征 (num_nodes, 5)

        batch_size = normal_features.shape[0]

        noise = torch.randn(batch_size, in_dim).cuda()  # 生成随机噪声 (batch_size, 5)
        fake_extreme_features = generator(normal_features, noise, edge_index, batch_index)  # 生成极端天气特征

    plt.figure()
    generated_features=fake_extreme_features.cpu()
    plt.hist(generated_features.detach().numpy().flatten(), bins=20, alpha=0.7, label="Generated Features")
    plt.legend()
    plt.show()
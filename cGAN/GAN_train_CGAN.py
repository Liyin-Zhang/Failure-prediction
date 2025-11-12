
import sys
import os
sys.path.append(os.path.dirname(__file__))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pickle
from GAN_networks.GAN_network_cgan_SAGE import *
from GAN_dataset import create_graph
from visualizer import GANDemoVisualizer
import random

from GAN_args import get_train_args

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
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    if gradients is None:
        return torch.tensor(0.0, device=real_x.device, requires_grad=True)
    gradients_norm = gradients.view(gradients.shape[0], -1).norm(2, dim=1)
    return ((gradients_norm - 1) ** 2).mean()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    args = get_train_args()
    set_seed(args.seed)

    dataset = create_graph(csv_path=args.data_csv, pickle_path=args.topo_pkl)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    generator = Generator(feature_dim=args.in_dim, noise_dim=args.in_dim, hidden_dim=args.hidden_dim).cuda()
    discriminator = Discriminator(in_dim=args.in_dim, hidden_dim=args.hidden_dim).cuda()

    optim_G = torch.optim.Adam(generator.parameters(), lr=args.lr_g, betas=tuple(args.betas))
    optim_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr_d, betas=tuple(args.betas))
    scheduler_G = torch.optim.lr_scheduler.StepLR(optim_G, step_size=args.step_size, gamma=args.gamma)
    scheduler_D = torch.optim.lr_scheduler.StepLR(optim_D, step_size=args.step_size, gamma=args.gamma)
    bce = nn.BCELoss().cuda()

    visualizer = GANDemoVisualizer(args.vis_title)

    loss_D_list, loss_G_list = [], []

    NUM_EPOCHS = args.epochs
    N_CRITIC   = args.n_critic
    IN_DIM     = args.in_dim

    for epoch in range(NUM_EPOCHS):
        epoch_loss_D = 0.0
        epoch_loss_G = 0.0
        num_batches = 0

        for data in dataloader:
            data = data.cuda()
            real_x = data.x.to(torch.float)
            edge_index = data.edge_index.to(torch.long)
            batch_index = data.batch

            normal = real_x[:, :, 0]
            extreme = real_x[:, :, 1]
            bs = normal.shape[0]

            for _ in range(N_CRITIC):
                noise = torch.randn(bs, IN_DIM, device=normal.device)
                fake_extreme = generator(normal, noise, edge_index, batch_index)

                d_real = discriminator(normal, extreme, edge_index, batch_index)
                d_fake = discriminator(normal, fake_extreme.detach(), edge_index, batch_index)

                real_labels = torch.ones(data.num_graphs, 1, device=normal.device)
                fake_labels = torch.zeros(data.num_graphs, 1, device=normal.device)

                loss_D = bce(d_real, real_labels) + bce(d_fake, fake_labels)
                optim_D.zero_grad()
                loss_D.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=10.0)
                optim_D.step()

            noise = torch.randn(bs, IN_DIM, device=normal.device)
            fake_extreme = generator(normal, noise, edge_index, batch_index)
            g_fake = discriminator(normal, fake_extreme, edge_index, batch_index)
            loss_G = bce(g_fake, real_labels)

            optim_G.zero_grad()
            loss_G.backward()
            optim_G.step()

            epoch_loss_D += loss_D.item()
            epoch_loss_G += loss_G.item()
            num_batches += 1

        avg_loss_D = epoch_loss_D / num_batches
        avg_loss_G = epoch_loss_G / num_batches
        loss_D_list.append(avg_loss_D)
        loss_G_list.append(avg_loss_G)

        scheduler_G.step()
        scheduler_D.step()

        if epoch % 10 == 0:
            msg = f"GRAPH-cGAN Epoch [{epoch}/{NUM_EPOCHS}]  Loss_D: {avg_loss_D:.4f}  Loss_G: {avg_loss_G:.4f}"
            print(msg)
            real_samples = extreme.detach().cpu().numpy()[:, :2]
            gen_samples  = fake_extreme.detach().cpu().numpy()[:, :2]
            visualizer.draw(real_samples, gen_samples, msg)

    visualizer.show()
    if args.save_gif:
        os.makedirs(args.model_dir, exist_ok=True)
        visualizer.save_gif(os.path.join(args.model_dir, "conditional_Graph_training.gif"), duration=100)

    os.makedirs(args.model_dir, exist_ok=True)
    torch.save(generator.state_dict(),     os.path.join(args.model_dir, f'cGAN_generator_model_{args.model_suffix}.pth'))
    torch.save(discriminator.state_dict(), os.path.join(args.model_dir, f'cGAN_discriminator_model_{args.model_suffix}.pth'))

    with open(os.path.join(args.model_dir, f'cGAN_loss_D_list_{args.model_suffix}.pickle'), 'wb') as f:
        pickle.dump(loss_D_list, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(args.model_dir, f'cGAN_loss_G_list_{args.model_suffix}.pickle'), 'wb') as f:
        pickle.dump(loss_G_list, f, protocol=pickle.HIGHEST_PROTOCOL)

    plt.figure(figsize=(8, 5))
    plt.plot(loss_D_list, label="Loss_D")
    plt.plot(loss_G_list, label="Loss_G")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("GAN Training Loss")
    plt.legend()
    plt.grid()

    for batch in dataloader:
        batch = batch.cuda()
        real_x = batch.x.to(torch.float)
        edge_index = batch.edge_index.to(torch.long)
        batch_index = batch.batch
        normal = real_x[:, :, 0]
        bs = normal.shape[0]
        noise = torch.randn(bs, IN_DIM, device=normal.device)
        fake_extreme = generator(normal, noise, edge_index, batch_index)
        break

    plt.figure()
    gen_feat = fake_extreme.detach().cpu().numpy().flatten()
    plt.hist(gen_feat, bins=20, alpha=0.7, label="Generated Features")
    plt.legend()
    plt.show()

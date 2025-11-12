

import os, sys, warnings
warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(__file__))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
import numpy as np
import pandas as pd
from torch_geometric.data import DataLoader
from GAN_networks.GAN_network_cgan_SAGE import Generator
from GAN_dataset import create_graph
import matplotlib.pyplot as plt
import seaborn as sns
import random

from GAN_args import get_eval_args

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    args = get_eval_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # data
    dataset = create_graph(csv_path=args.data_csv, pickle_path=args.topo_pkl)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    ref_df = pd.read_csv(args.data_csv, encoding='utf-8')
    feeders = ref_df['feeder_name'].drop_duplicates().tolist()

    # model
    generator = Generator(feature_dim=args.in_dim, noise_dim=args.in_dim, hidden_dim=args.hidden_dim).to(device)
    generator.load_state_dict(torch.load(args.model_path, map_location=device))
    generator.eval()

    # generate
    generated_parts = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch = batch.to(device)
            real_x = batch.x.to(torch.float)          # [N, 5, 2]
            edge_index = batch.edge_index
            batch_index = batch.batch

            normal = real_x[:, :, 0]                  # [N, 5]
            noise = torch.randn(normal.shape[0], args.in_dim, device=device)
            fake_extreme = generator(normal, noise, edge_index, batch_index)
            fake_np = fake_extreme.detach().cpu().numpy()

            cur_df = ref_df[ref_df['feeder_name'] == feeders[i]].copy()
            for j, col in enumerate(args.feature_cols):
                cur_df[col] = fake_np[:, j]

            cur_df['feeder_name'] = feeders[i][:-5] + 'G' + feeders[i][-4:]
            generated_parts.append(cur_df)

    # save
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    final_df = pd.concat(generated_parts, ignore_index=True)
    final_df.to_csv(args.output_csv, index=False, encoding='utf-8')
    print(f"Saved generated data to: {args.output_csv}")

    # plots
    plt.figure()
    sns.histplot(final_df[args.feature_cols[0]], kde=True)
    plt.title(f"Distribution of {args.feature_cols[0]}")

    plt.figure()
    sns.histplot(final_df[args.feature_cols[1]], kde=True)
    plt.title(f"Distribution of {args.feature_cols[1]}")

    plt.figure()
    plt.hist(fake_np.flatten(), bins=20, alpha=0.7, label="Generated Features")
    plt.legend()
    plt.title("Histogram of All Generated Features")

    plt.show()

import torch
from torch_geometric.data import Data
import pickle5
import pandas as pd
import numpy as np

def standardize(x, mean=None, std=None):
    """Standardize to zero mean and unit variance, then map to [0, 1]."""
    x = np.array(x)
    if mean is None:
        mean = x.mean()
    if std is None:
        std = x.std()
    print('mean:', mean)
    print('std:', std)
    x_standardized = (x - mean) / (std if std != 0 else 1.0)
    return normalize_to_0_1(x_standardized)

def normalize_to_0_1(x, min_val=None, max_val=None):
    """Min-max normalize to [0, 1]."""
    x = np.array(x)
    if min_val is None:
        min_val = x.min()
    if max_val is None:
        max_val = x.max()
    print('min:', min_val)
    print('max:', max_val)
    denom = (max_val - min_val) if (max_val - min_val) != 0 else 1.0
    return np.round((x - min_val) / denom, 2)

def create_graph(csv_path, pickle_path):
    """
    Build a PyG graph list from node CSV and topology pickle.
    csv_path: node feature CSV with feeder_name and label
    pickle_path: pickle dict keyed by base feeder name -> edge list [(src, dst), ...]
    """
    with open(pickle_path, 'rb') as f:
        feeder_edge = pickle5.load(f)

    data_list = []
    df = pd.read_csv(csv_path, encoding='utf-8').fillna(0)

    header_normal  = ['WIN_normal', 'PRE_normal', 'PRS_normal', 'SHU_normal', 'TMP_normal']
    header_extreme = ['WIN_extreme', 'PRE_extreme', 'PRS_extreme', 'SHU_extreme', 'TMP_extreme']

    grouped = df.groupby('feeder_name', sort=False)

    for feeder_name, group in grouped:
        normal_features  = torch.FloatTensor(group[header_normal].values)   # (num_nodes, 5)
        extreme_features = torch.FloatTensor(group[header_extreme].values)  # (num_nodes, 5)

        # (num_nodes, 5, 2): [:, :, 0] normal, [:, :, 1] extreme
        node_features = torch.stack([normal_features, extreme_features], dim=2)

        # topology from pickle: key is feeder_name without last 5 chars
        base_key = feeder_name[:-5] if len(feeder_name) > 5 else feeder_name
        edges = feeder_edge[base_key]
        src = [e[0] for e in edges]
        dst = [e[1] for e in edges]
        edge_index = torch.tensor([src, dst], dtype=torch.long)

        y = torch.LongTensor([group['label'].iloc[0]])

        data = Data(x=node_features, edge_index=edge_index, y=y)
        data_list.append(data)

    return data_list

# main.py 使用 logits 版 Focal Loss
import pickle
import torch
import random
import numpy as np
from torch_geometric.loader import DataLoader
from GNN_args import *
from GNN01_apply_dataset import MyBinaryDataset
from GNN_network_08 import NetA_NodeOnly  # 使用无池化版本
import shutil, os
import torch.nn.functional as F

lambda_consistency = 0.0  # >0 启用一致性正则

SEED = 12100

def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def seed_worker(worker_id: int):
    worker_seed = (SEED + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

set_seed(SEED)

# ========= 二分类 Focal Loss（logits 输入） =========
class FocalLossBinaryLogits(torch.nn.Module):
    """
    公式: FL = - alpha_t * (1 - p_t)^gamma * log(p_t)
    其中 log(p_t) 用 BCEWithLogits 的稳定实现
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits: [B] 或 [B,1] 未过 sigmoid
        targets: [B] 或 [B,1] 取值 0 或 1
        """
        logits = logits.view(-1)
        targets = targets.view(-1).float()

        # CE = -log(p_t) 的数值稳定实现
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        # p_t 与 alpha_t
        p = torch.sigmoid(logits)
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)

        focal_weight = alpha_t * (1.0 - p_t).pow(self.gamma)
        loss = focal_weight * ce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

def train():
    model.train()
    loss_all = 0.0
    for data in train_loader:
        data = data.cuda()
        optimizer.zero_grad()
        out = model(data, return_reg=(lambda_consistency > 0))
        if isinstance(out, tuple):
            logits, reg = out
            loss = crit(logits, data.y) + reg
        else:
            logits = out
            loss = crit(logits, data.y)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(dataset)

if __name__ == '__main__':
    # 安全清理
    for d in ['GNN_data/train/raw', 'GNN_data/train/processed']:
        shutil.rmtree(d, ignore_errors=True)

    epochs = 6000
    train_root = 'GNN_data/train'
    suffix_train = '0417_gen_1'
    train_file_name = f'GNN_data/train/train_{suffix_train}_new.csv'
    edge_file_name = f'GNN_data/train/train_{suffix_train}_edge_features.csv'
    train_model_name = f'GNN_data/model/GNN_{suffix_train}_1015_10_1.pkl' #08和10 _new
    train_loss_name  = f'GNN_data/model/train_loss_{suffix_train}_1015_10_1.pickle'

    dataset = MyBinaryDataset(root=train_root,
                              file_name=train_file_name,
                              oriented_pkl="GNN_data/grid/origin_topology_3_oriented.pickle",
                              edge_feat_csv=edge_file_name)
    train_loader = DataLoader(dataset, batch_size=batchsize, shuffle=True, worker_init_fn=seed_worker)

    model = NetA_NodeOnly(lambda_consistency=lambda_consistency).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=stepsize, gamma=gama)

    # 使用 logits 版 Focal Loss
    crit = FocalLossBinaryLogits(alpha=0.25, gamma=2.0, reduction="mean").cuda()

    loss_list = []
    for epoch in range(epochs):
        print('epoch:', epoch)
        loss = train()
        print(loss)
        loss_list.append(loss)
        scheduler.step()

        if (epoch + 1) % 1000 == 0:
            ckpt = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(ckpt, f'GNN_data/model/GNN_{suffix_train}_{epoch+1}.pkl')
            with open(f'GNN_data/model/train_loss_{suffix_train}_{epoch+1}.pickle', 'wb') as f:
                pickle.dump(loss_list, f, protocol=pickle.HIGHEST_PROTOCOL)

    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, train_model_name)
    with open(train_loss_name, 'wb') as f:
        pickle.dump(loss_list, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('完成训练')

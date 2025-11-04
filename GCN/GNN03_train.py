# main.py 关键改动
import pickle
import torch
import random
import numpy as np
from torch_geometric.loader import DataLoader
from GNN_args import *
from GNN01_apply_dataset import MyBinaryDataset
from GNN_network_06 import NetA_NodeOnly  # ← 使用无池化版本
import shutil, os

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

lambda_consistency = 0.0  # >0 启用一致性正则，例如 1e-3

# ========== 1) 设定随机种子（可复现） ==========
SEED = 12101  # ← 你可以改成任意固定整数

def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)   # 影响 Python 哈希/字典顺序
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # # cuDNN 确定性 & 关闭 benchmark（否则会选用非确定性算子/不同算法）
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # # 严格模式（如遇到不确定性算子直接报错）——可选
    # try:
    #     torch.use_deterministic_algorithms(True)
    # except Exception:
    #     pass

# DataLoader 多进程时的 worker 复现：给每个 worker 派生独立但可复现的种子
def seed_worker(worker_id: int):
    worker_seed = (SEED + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

set_seed(SEED)
# ============================================


def train():
    model.train()
    loss_all = 0.0
    for data in train_loader:
        data = data.cuda()
        optimizer.zero_grad()
        out = model(data, return_reg=(lambda_consistency > 0))
        if isinstance(out, tuple):
            pred, reg = out
            loss = crit(pred, data.y) + reg
        else:
            pred = out
            loss = crit(pred, data.y)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(dataset)

if __name__ == '__main__':
    # 安全清理
    for d in ['GNN_data/train/raw', 'GNN_data/train/processed']:
        shutil.rmtree(d, ignore_errors=True)

    epochs = 1000
    train_root = 'GNN_data/train'
    suffix_train = '0417_gen_0'
    train_file_name = f'GNN_data/train/train_{suffix_train}.csv'
    edge_file_name = f'GNN_data/train/train_{suffix_train}_edge_features.csv'
    train_model_name = f'GNN_data/model/GNN_{suffix_train}_1015_06_1.pkl'
    train_loss_name  = f'GNN_data/model/train_loss_{suffix_train}_1015_06_1.pickle'

    dataset = MyBinaryDataset(root=train_root,
                              file_name=train_file_name,
                              oriented_pkl="GNN_data/grid/origin_topology_3_oriented.pickle",
                              edge_feat_csv=edge_file_name)
    train_loader = DataLoader(dataset, batch_size=batchsize, shuffle=True)

    model = NetA_NodeOnly(lambda_consistency=lambda_consistency).cuda()

    # model = NetA_NodeOnly(
    #     lambda_consistency=0.0,  # 需要时设个小值如 1e-4
    #     hidden_dim=256,
    #     fuse_mode='feature',  # 或 'feature'
    #     dropout_p=0.5
    # ).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=stepsize, gamma=gama)
    crit = torch.nn.BCELoss().cuda()  # 如改用 BCEWithLogitsLoss，请同步移除网络里的 sigmoid

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

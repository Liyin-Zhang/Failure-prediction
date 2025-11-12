# main_focal.py — training with Focal Loss (logits), args-driven

import os
import pickle
import random
import shutil
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from GNN_args import get_args
from GNN_dataset import MyBinaryDataset
# Use the logits-version network (no internal sigmoid)
from GNN_networks import GNN_network_09 as net
from GNN_networks.GNN_network_09 import NetA_NodeOnly


# -------------------- Reproducibility --------------------
def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name: str) -> torch.device:
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


def seed_worker_builder(base_seed: int):
    def seed_worker(worker_id: int):
        worker_seed = (base_seed + worker_id) % (2 ** 32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    return seed_worker


# -------------------- Data utils --------------------
def build_dataloader(ARGS, dataset):
    g = torch.Generator()
    g.manual_seed(ARGS.seed)
    return DataLoader(
        dataset,
        batch_size=ARGS.batch_size,
        shuffle=True,
        num_workers=ARGS.num_workers,
        pin_memory=ARGS.pin_memory,
        worker_init_fn=seed_worker_builder(ARGS.seed),
        generator=g
    )


def maybe_clean_pyg_cache(pyg_root: str, enable: bool):
    if not enable:
        return
    for d in ['raw', 'processed']:
        path = os.path.join(pyg_root, d)
        shutil.rmtree(path, ignore_errors=True)


def build_dataset(ARGS):
    return MyBinaryDataset(
        root=ARGS.pyg_root,
        file_name=ARGS.train_csv,
        oriented_pkl=ARGS.oriented_pkl,
        edge_feat_csv=ARGS.edge_feat_csv,
        node_feature_cols=ARGS.node_feature_cols,
        edge_feature_cols=ARGS.edge_feature_cols,
        scale_cols_order=ARGS.scale_cols_order,
        apply_discretize=ARGS.apply_discretize
    )


# -------------------- Model --------------------
def build_model(ARGS, device: torch.device):
    # Inject three hyper-parameters into the network module's globals
    net.max_node_number = int(ARGS.max_node_number)
    net.embed_dim = int(ARGS.embed_dim)
    net.feature_num = int(ARGS.feature_num)

    # Set edge_dim according to the number of edge features
    edge_dim = len(ARGS.edge_feature_cols)

    model = NetA_NodeOnly(
        lambda_consistency=ARGS.lambda_consistency,
        edge_dim=edge_dim
    )
    return model.to(device)


# -------------------- Focal Loss (logits) --------------------
class FocalLossBinaryLogits(torch.nn.Module):
    """
    Per-sample: FL = - alpha_t * (1 - p_t)^gamma * log(p_t)

    where:
        p = sigmoid(logits)
        p_t = p   if target == 1
              1-p if target == 0
        alpha_t = alpha for target==1, (1-alpha) for target==0

    CE = -log(p_t) is computed via BCEWithLogits for numerical stability.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits, targets):
        logits = logits.view(-1)
        targets = targets.view(-1).float()

        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
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


# -------------------- Train one epoch --------------------
def train_one_epoch(model, loader, device, optimizer, criterion, lambda_consistency: float):
    model.train()
    loss_sum = 0.0
    sample_cnt = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        out = model(data, return_reg=(lambda_consistency > 0))
        if isinstance(out, tuple):
            logits, reg = out
            loss = criterion(logits, data.y) + reg
        else:
            logits = out
            loss = criterion(logits, data.y)

        loss.backward()
        optimizer.step()

        loss_sum += data.num_graphs * loss.item()
        sample_cnt += data.num_graphs

    return loss_sum / max(sample_cnt, 1)


# -------------------- Main --------------------
def main():
    ARGS = get_args()
    set_seed(ARGS.seed)
    device = get_device(ARGS.device)

    maybe_clean_pyg_cache(ARGS.pyg_root, ARGS.clean_pyg_cache)

    dataset = build_dataset(ARGS)
    train_loader = build_dataloader(ARGS, dataset)

    model = build_model(ARGS, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=ARGS.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=ARGS.step_size, gamma=ARGS.gamma
    )

    # Focal loss with logits (model must output raw scores)
    criterion = FocalLossBinaryLogits(
        alpha=getattr(ARGS, "focal_alpha", 0.25),
        gamma=getattr(ARGS, "focal_gamma", 2.0),
        reduction="mean"
    ).to(device)

    loss_history = []
    for epoch in range(ARGS.epochs):
        epoch_id = epoch + 1
        loss = train_one_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            criterion=criterion,
            lambda_consistency=ARGS.lambda_consistency
        )
        loss_history.append(loss)
        print(f"epoch {epoch_id} | loss {loss:.6f}")
        scheduler.step()

        if ARGS.checkpoint_every > 0 and (epoch_id % ARGS.checkpoint_every == 0):
            os.makedirs(os.path.dirname(ARGS.checkpoint_path), exist_ok=True)
            torch.save(
                {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch_id},
                ARGS.checkpoint_path
            )

    os.makedirs(os.path.dirname(ARGS.final_model_path), exist_ok=True)
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': ARGS.epochs},
               ARGS.final_model_path)

    os.makedirs(os.path.dirname(ARGS.train_loss_path), exist_ok=True)
    with open(ARGS.train_loss_path, 'wb') as f:
        pickle.dump(loss_history, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Training completed.")


if __name__ == '__main__':
    main()

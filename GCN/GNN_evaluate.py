# evaluate.py

import os
import shutil
import pickle
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from GNN_args import get_args
from GNN_dataset import MyBinaryDataset
from GNN_networks import GNN_network_06 as net
from GNN_networks.GNN_network_06 import NetA_NodeOnly


# -------------------- utils --------------------
def set_seed(seed: int):
    """Set all relevant random seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name: str) -> torch.device:
    """Resolve device by name ('auto' -> cuda if available)."""
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


def seed_worker_builder(base_seed: int):
    """Build worker_init_fn for DataLoader to ensure deterministic workers."""
    def seed_worker(worker_id: int):
        import random
        worker_seed = (base_seed + worker_id) % (2 ** 32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    return seed_worker


def build_dataloader(ARGS, dataset):
    """Create a deterministic DataLoader; keep order for feeder grouping."""
    g = torch.Generator()
    g.manual_seed(ARGS.seed)
    return DataLoader(
        dataset,
        batch_size=ARGS.batch_size,
        shuffle=False,                 # keep order for feeder grouping
        num_workers=ARGS.num_workers,
        pin_memory=ARGS.pin_memory,
        worker_init_fn=seed_worker_builder(ARGS.seed),
        generator=g
    )


def maybe_clean_pyg_cache(pyg_root: str, enable: bool):
    """Optionally clear PyG cache (raw/processed) before building dataset."""
    if not enable:
        return
    for d in ['raw', 'processed']:
        path = os.path.join(pyg_root, d)
        shutil.rmtree(path, ignore_errors=True)


def read_csv_any(path: str) -> pd.DataFrame:
    """Read CSV with common encodings tried in order."""
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception:
            pass
    return pd.read_csv(path, low_memory=False)


def plot_confusion_with_metrics(cm, title="Confusion Matrix (with metrics)"):
    """Pretty confusion matrix heatmap with key metrics annotated."""
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) * 100 if (tp + tn + fp + fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    mat = np.array([[tp, fp], [fn, tn]])
    labels = np.array([
        [f"{tp}\nTP", f"{fp}\nFP\n{precision:.2f}% precision"],
        [f"{fn}\nFN\n{recall:.2f}% recall", f"{tn}\nTN\n{specificity:.2f}% specificity"]
    ])

    plt.figure(figsize=(6.5, 5.2))
    ax = sns.heatmap(mat, annot=labels, fmt="", cmap="Blues", cbar=False, linewidths=2, linecolor='white')
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_title(title)
    plt.text(2.15, 0.5, f"F1  : {f1_score:.2f}%", ha="left", va="center", fontsize=11, color="darkblue")
    plt.text(2.15, 1.5, f"ACC : {accuracy:.2f}%", ha="left", va="center", fontsize=11, color="darkblue")
    plt.tight_layout()


# -------------------- pipeline builders --------------------
def build_dataset(ARGS):
    """Build evaluation dataset using evaluation-specific paths from ARGS."""
    ds = MyBinaryDataset(
        root=ARGS.evaluate_root,                 # use evaluate cache root
        file_name=ARGS.evaluate_csv,             # evaluation CSV
        oriented_pkl=ARGS.evaluate_oriented_pkl, # evaluation topology pkl
        edge_feat_csv=ARGS.evaluate_edge_feat_csv,  # evaluation edge features CSV
        node_feature_cols=ARGS.node_feature_cols,
        edge_feature_cols=ARGS.edge_feature_cols,
        scale_cols_order=ARGS.scale_cols_order,
        apply_discretize=ARGS.apply_discretize
    )
    return ds


def build_model(ARGS, device: torch.device):
    """Build model and inject network-level globals from ARGS (same as train).”
    """
    # set network globals from args (same as train)
    net.max_node_number = int(ARGS.max_node_number)
    net.embed_dim = int(ARGS.embed_dim)
    net.feature_num = int(ARGS.feature_num)

    edge_dim = len(ARGS.edge_feature_cols)
    model = NetA_NodeOnly(
        lambda_consistency=0.0,
        edge_dim=edge_dim
    )
    return model.to(device)


# -------------------- evaluation core --------------------
def evaluate(loader, model, device):
    """Forward all graphs to get probabilities and compute ROC/AUC."""
    model.eval()
    probs, labels = [], []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)  # expect probability [B]
            probs.append(out.detach().cpu().numpy())
            labels.append(data.y.detach().cpu().numpy())

    probs = np.concatenate(probs, axis=0)
    labels = np.concatenate(labels, axis=0)

    fpr, tpr, thresholds = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)
    return auc, fpr, tpr, thresholds, labels, probs


# -------------------- main --------------------
def main():
    ARGS = get_args()
    set_seed(ARGS.seed)
    device = get_device(ARGS.device)

    # Clear PyG cache under evaluation root if requested
    maybe_clean_pyg_cache(ARGS.evaluate_root, getattr(ARGS, "clean_pyg_cache", True))

    # Build dataset/dataloader with evaluation paths
    dataset = build_dataset(ARGS)
    loader = build_dataloader(ARGS, dataset)

    # Build model and load checkpoint
    model = build_model(ARGS, device)
    state = torch.load(ARGS.model_ckpt_path, map_location=device)
    model.load_state_dict(state['model'], strict=False)

    # Evaluate
    auc, fpr, tpr, thresholds, labels, probs = evaluate(loader, model, device)
    print(f"AUC: {auc:.6f}")

    # Probability histogram
    plt.figure(figsize=(6, 4))
    sns.histplot(probs, binwidth=0.1)
    plt.title("Prediction probability histogram")
    plt.xlabel("Probability")
    plt.tight_layout()

    # Binarize by threshold from args
    TH = float(ARGS.threshold)
    preds = (probs > TH).astype(int)

    # Feeder-level aggregation (preserve CSV grouping order)
    df_eval = read_csv_any(ARGS.evaluate_csv)
    feeder_names, feeder_labels = [], []
    for feeder_name, group in df_eval.groupby('feeder_name', sort=False):
        feeder_names.append(feeder_name)
        feeder_labels.append(group['label'].iloc[0])

    result = pd.DataFrame({
        'feeder_name': feeder_names,
        'label': feeder_labels,
        'risk': preds
    })
    os.makedirs(os.path.dirname(ARGS.output_csv), exist_ok=True)
    result.to_csv(ARGS.output_csv, index=False, encoding='gbk')
    print(f"[SAVE] feeder-level result -> {ARGS.output_csv}")

    # Confusion matrix & metrics
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    print(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}")

    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) * 100 if (tp + tn + fp + fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"Precision : {precision:.2f}%")
    print(f"Recall    : {recall:.2f}%")
    print(f"Specificity: {specificity:.2f}%")
    print(f"Accuracy  : {accuracy:.2f}%")
    print(f"F1_score  : {f1_score:.2f}%")

    plot_confusion_with_metrics(cm, title=f"Confusion Matrix (threshold={TH})")

    # ROC curve
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()

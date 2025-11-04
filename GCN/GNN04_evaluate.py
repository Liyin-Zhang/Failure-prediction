# -*- coding: utf-8 -*-
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
import os
import sys
sys.path.append(os.path.dirname(__file__))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from GNN_args import *                     # 使用你的超参/路径，如 evaluate_root / evaluate_file_name / evaluate_model_name / batchsize 等
from GNN01_apply_dataset import MyBinaryDataset
from GNN_network_06 import NetA_NodeOnly
 # network04-->09/10
evaluate_root = 'GNN_data/evaluate'
suffix_evaluate = '1101'  # '1013_1_filtered'  # '0415_3_filtered_3'#'0207_3''0310_2'
evaluate_file_name = 'GNN_data/evaluate/evaluate_' + suffix_evaluate + '.csv'
edge_file_name = 'GNN_data/evaluate/evaluate_' + suffix_evaluate + '_edge_features.csv'

suffix_model_used = '0417_gen_0_1015_06_1'  # 1015_07' 08_focal_loss
evaluate_model_name = 'GNN_data/model/GNN_' + suffix_model_used + '.pkl'
evaluate_loss_name = 'GNN_data/model/train_loss_' + suffix_model_used + '.pickle'

save_model_used = '0417_gen_0_1101_06_1'
evaluate_resule_file_name = 'GNN_data/evaluate/evaluate_result_' + save_model_used + '.csv'
out_path = f'GNN_data/output/result_feeder_{save_model_used}.csv'
# -------------------- 小工具 --------------------
def safe_rmtree(path: str):
    shutil.rmtree(path, ignore_errors=True)

def read_csv_any(path: str) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception:
            pass
    # 最后再尝试不指定编码
    return pd.read_csv(path, low_memory=False)

def plot_confusion_with_metrics(cm, title="Confusion Matrix (with metrics)"):
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0.0
    accuracy  = (tp + tn) / (tp + tn + fp + fn) * 100 if (tp + tn + fp + fn) > 0 else 0.0
    f1_score  = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    mat = np.array([[tp, fp],[fn, tn]])
    labels = np.array([
        [f"{tp}\nTP", f"{fp}\nFP\n{precision:.2f}% precision"],
        [f"{fn}\nFN\n{recall:.2f}% recall", f"{tn}\nTN\n{specificity:.2f}% specificity"]
    ])

    plt.figure(figsize=(6.5, 5.2))
    ax = sns.heatmap(mat, annot=labels, fmt="", cmap="Blues", cbar=False, linewidths=2, linecolor='white')
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_title(title)

    # 右侧补充整体指标
    plt.text(2.15, 0.5, f"F1  : {f1_score:.2f}%", ha="left", va="center", fontsize=11, color="darkblue")
    plt.text(2.15, 1.5, f"ACC : {accuracy:.2f}%", ha="left", va="center", fontsize=11, color="darkblue")
    plt.tight_layout()

# -------------------- 评估主函数 --------------------
def evaluate(loader, model, device):
    model.eval()
    probs, labels = [], []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)                 # NetA_NodeOnly 末端已有 sigmoid → 概率
            # data.y 形状 [B]；out 形状 [B]
            probs.append(out.detach().cpu().numpy())
            labels.append(data.y.detach().cpu().numpy())

    probs  = np.concatenate(probs, axis=0)    # [num_graphs]
    labels = np.concatenate(labels, axis=0)   # [num_graphs]

    fpr, tpr, thresholds = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)
    return auc, fpr, tpr, thresholds, labels, probs

# -------------------- 主入口 --------------------
if __name__ == '__main__':
    # 安全清理 evaluate 缓存
    safe_rmtree('GNN_data/evaluate/raw')
    safe_rmtree('GNN_data/evaluate/processed')



    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据集 & DataLoader（不要打乱，保证顺序与 CSV 分组一致）
    dataset = MyBinaryDataset(root=evaluate_root,
                              file_name=evaluate_file_name,
                              oriented_pkl="GNN_data/grid/origin_topology_3_oriented.pickle",
                              edge_feat_csv=edge_file_name)

    loader  = DataLoader(dataset, batch_size=batchsize, shuffle=False)

    model = NetA_NodeOnly(lambda_consistency=0).cuda()
    # # 模型
    # model = NetA_NodeOnly(
    #     lambda_consistency=0.0,  # 需要时设个小值如 1e-4
    #     hidden_dim=256,
    #     fuse_mode='feature',  # 或 'feature'
    #     dropout_p=0.5
    # ).cuda()
    state = torch.load(evaluate_model_name, map_location=device)
    model.load_state_dict(state['model'], strict=False)

    # 评估
    auc, fpr, tpr, thresholds, labels, probs = evaluate(loader, model, device)
    print(f"AUC: {auc:.6f}")

    # 概率直方图
    plt.figure(figsize=(6,4))
    sns.histplot(probs, binwidth=0.1)
    plt.title("Prediction probability histogram")
    plt.xlabel("Probability")
    plt.tight_layout()

    # 0/1 化（默认 0.5，可按需改）
    TH = 0.5
    preds = (probs > TH).astype(int)

    # 按 feeder 汇总导出 CSV（与原脚本一致）
    df_eval = read_csv_any(evaluate_file_name)
    feeder_names, feeder_labels = [], []
    for feeder_name, group in df_eval.groupby('feeder_name', sort=False):
        feeder_names.append(feeder_name)
        feeder_labels.append(group['label'].iloc[0])

    # 这里假设 DataLoader 遍历顺序与 CSV 分组顺序一致（和我们构建 dataset 的方式一致）
    result = pd.DataFrame({
        'feeder_name': feeder_names,
        'label': feeder_labels,
        'risk': preds
    })
    os.makedirs('GNN_data/output', exist_ok=True)

    # 若需要 GBK 保存可改编码；这里保持原脚本的 'gbk'
    result.to_csv(out_path, index=False, encoding='gbk')
    print(f"[SAVE] feeder-level result -> {out_path}")

    # 混淆矩阵与指标
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    print(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}")

    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0.0
    accuracy  = (tp + tn) / (tp + tn + fp + fn) * 100 if (tp + tn + fp + fn) > 0 else 0.0
    f1_score  = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"Precision : {precision:.2f}%")
    print(f"Recall    : {recall:.2f}%")
    print(f"Specificity: {specificity:.2f}%")
    print(f"Accuracy  : {accuracy:.2f}%")
    print(f"F1_score  : {f1_score:.2f}%")

    # 混淆矩阵可视化（带指标）
    plot_confusion_with_metrics(cm, title="Confusion Matrix (threshold=0.5)")

    # ROC
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
    plt.plot([0,1], [0,1], linestyle='--', color='r', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.tight_layout()
    # plt.show()

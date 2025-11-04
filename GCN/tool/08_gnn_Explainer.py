# -*- coding: utf-8 -*-
"""
单馈线因果贡献解释（GNNExplainer，边/节点重要性）
- 适配你的 NetA_NodeOnly（当前使用无向两层的 H_und 做图分类）
- 直接对“有向边（edge_index_dir, edge_attr）”做解释，得到 edge_mask（因果贡献）
- 将 edge_mask 聚合为节点重要性（入边贡献之和），并按经纬度可视化
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from torch_geometric.utils import to_undirected

sys.path.append(os.path.dirname(__file__))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.explain.config import ModelConfig
from GNN01_apply_dataset import MyBinaryDataset
from GNN_network_01 import NetA_NodeOnly
from GNN_args import *
import torch.nn.functional as F

# ========= 自行设置 =========
EVAL_CSV     = "GNN_data/evaluate/evaluate_0415_3_filtered_3.csv"
EDGE_CSV     = "GNN_data/evaluate/evaluate_0415_3_filtered_3_edge_features.csv"
ORIENTED_PKL = "GNN_data/grid/origin_topology_3_oriented.pickle"
MODEL_CKPT   = "GNN_data/model/GNN_0417_gen_0_1015_01.pkl"

FEEDER_TARGET = "东方F3棠吓线03364"#"东方F3棠吓线03364" "10kV石滩F8碧江线03447"
TOPK_EDGES = 30
TOPK_NODES = 20

EXPLAIN_EPOCHS = 300       # GNNExplainer 迭代步数
EXPLAIN_LR     = 0.01      # 学习率
TARGET_CLASS   = 1         # 二分类正类
# ==========================

plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False


def read_csv_any(path: str) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception:
            pass
    return pd.read_csv(path, low_memory=False)


def build_pos_from_csv(csv_path: str, feeder_raw: str):
    df = read_csv_any(csv_path)
    g = df[df["feeder_name"] == feeder_raw].copy()
    if g.empty:
        raise ValueError(f"CSV 中找不到 feeder_name={feeder_raw}")
    g["pole_id_str"] = g["pole_id"].astype(str)
    order = g["pole_id_str"].tolist()
    pos = {pid: (float(lon), float(lat)) for pid, lon, lat in zip(g["pole_id_str"], g["longitude"], g["latitude"])}
    label = int(pd.Series(g["label"]).mode().iloc[0])
    return order, pos, bool(label)


def pick_data_for_feeder(dataset: MyBinaryDataset, feeder_raw: str):
    src_df = read_csv_any(dataset.file_name)
    feeders_in_order = [name for name, _ in src_df.groupby("feeder_name", sort=False)]
    try:
        idx = feeders_in_order.index(feeder_raw)
    except ValueError:
        raise ValueError(f"Dataset 源 CSV 中找不到 feeder_name={feeder_raw}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    for i, data in enumerate(loader):
        if i == idx:
            return data.to(device), idx
    raise RuntimeError("遍历 DataLoader 时未找到目标样本（索引越界？）")


class WrappedModel(torch.nn.Module):
    def __init__(self, base_model: NetA_NodeOnly, proto_data: Data):
        super().__init__()
        self.base = base_model
        self.proto_x     = proto_data.x.detach().clone()      # 离散索引 (long)
        self.proto_batch = proto_data.batch.detach().clone()

    def forward(self, x, edge_index, **kwargs):
        edge_attr = kwargs.get('edge_attr', None)
        batch     = kwargs.get('batch', self.proto_batch)

        g = Data()
        # 1) 用 node mask 后的 x 作为“节点门控系数”，形状应为 [N, 1]
        #    如果 Explainer 传的是空占位，会是全 0/1；训练过程中 GNNExplainer 会把它乘上 node_mask.sigmoid()
        node_gate = x  # [N,1] float in [0,1]

        # 2) 取原始离散索引做 embedding，然后用 node_gate 缩放
        #    ⚠️ 这里直接复用你模型里的 embedding 层逻辑：item_embedding + reshape
        with torch.no_grad():
            # 我们不在封装里算 embedding，只是做个占位以匹配字段。
            pass

        # 把门控系数传到 base_model 里处理：我们在 data 里放两个字段，提示 base_model 做门控
        g.x = self.proto_x                     # 仍然喂离散索引给原模型
        g.node_gate = node_gate                # <- 新增：节点门控（N,1），float
        g.edge_index_dir = edge_index          # 你现在是“纯有向”NetA，就只给有向边
        g.edge_attr      = edge_attr
        g.batch = batch

        # 让 base_model 在前向里使用 node_gate 缩放 embedding（见下一段小补丁）
        out = self.base(g)
        return out



def build_explainer(wrapped_model: torch.nn.Module):
    """
    适配 PyG 新版 Explainer：需要 explanation_type
    - explanation_type='phenomenon'：解释某个具体预测现象（推荐）
    - mode='binary_classification'，task_level='graph'，return_type='probs'
    - 只学习 edge_mask（edge_mask_type='object'）
    """
    model_config = ModelConfig(
        mode='binary_classification',
        task_level='graph',
        return_type='probs',
    )

    explainer = Explainer(
        model=wrapped_model,
        algorithm=GNNExplainer(epochs=EXPLAIN_EPOCHS, lr=EXPLAIN_LR),
        explanation_type='phenomenon',   # ★ 必填：'phenomenon' 或 'model'
        model_config=model_config,
        edge_mask_type='object',         # 学习边子图
        node_mask_type='object',             # 不学习节点/特征 mask
    )
    return explainer

def run_gnnexplainer_on_graph(explainer, data_one, target_class: int = 1):
    device = data_one.x.device
    N = data_one.x.size(0)
    x_dummy = torch.zeros((N, 1), device=device, dtype=torch.float)

    # 与 WrappedModel 一致：用无向边
    if hasattr(data_one, 'edge_index_und') and hasattr(data_one, 'edge_attr_und') \
       and data_one.edge_index_und is not None:
        ei = data_one.edge_index_und
        ea = data_one.edge_attr_und
    else:
        ei, ea = to_undirected(data_one.edge_index_dir, edge_attr=data_one.edge_attr, reduce='mean')

    # 构造同形状 target Tensor
    with torch.no_grad():
        y_hat_probe = explainer.model(x_dummy, ei, edge_attr=ea, batch=data_one.batch)
        target_tensor = torch.full_like(y_hat_probe, float(target_class), dtype=y_hat_probe.dtype, device=y_hat_probe.device)

    explanation = explainer(
        x=x_dummy, edge_index=ei, edge_attr=ea, batch=data_one.batch, target=target_tensor
    )
    # 调试断言（可留可删）
    em = explanation.edge_mask
    print("E_mask =", int(em.numel()), " E_used =", int(ei.size(1)))
    assert em.numel() == ei.size(1), "edge_mask 与使用的 edge_index 边数不一致"

    return explanation, ei, ea

def edges_nodes_from_mask(ei, edge_mask, node_order, colname="score"):
    src = ei[0].detach().cpu().numpy()
    dst = ei[1].detach().cpu().numpy()
    a = edge_mask.detach().cpu().numpy().astype(float)

    edges = pd.DataFrame({"src_idx": src, "dst_idx": dst, colname: a})
    edges["u"] = [node_order[i] for i in edges["src_idx"]]
    edges["v"] = [node_order[i] for i in edges["dst_idx"]]

    # 节点重要性：入边掩码累加（也可改成 max/Top-k 入边等）
    node_scores = edges.groupby("dst_idx")[colname].sum().reindex(range(len(node_order)), fill_value=0.0)
    nodes = pd.DataFrame({
        "idx": range(len(node_order)),
        "pole_id": node_order,
        "score_in": node_scores.values
    })
    edges = edges.sort_values(colname, ascending=False)
    nodes = nodes.sort_values("score_in", ascending=False)
    return edges, nodes


def plot_edges_scores(pos, edges_df, title, score_col="score", topk_edges=30, save=None):
    segs, vals = [], []
    for _, r in edges_df.iterrows():
        u, v, w = r["u"], r["v"], float(r[score_col])
        if (u in pos) and (v in pos):
            segs.append([pos[u], pos[v]])
            vals.append(w)
    if not segs:
        print("[WARN] 该馈线没有可画的边段")
        return

    vals = np.asarray(vals, dtype=float)
    ql, qr = np.quantile(vals, [0.02, 0.98]) if len(vals) > 10 else (vals.min(), vals.max() + 1e-12)
    vmin, vmax = float(ql), float(qr) + 1e-12

    fig, ax = plt.subplots(figsize=(7, 7))
    widths = 0.5 + 2.5 * ((np.clip(vals, vmin, vmax) - vmin) / (vmax - vmin + 1e-12))
    lc = LineCollection(
        segs,
        array=np.clip(vals, vmin, vmax),
        linewidths=widths,
        cmap="viridis",
        norm=plt.Normalize(vmin=vmin, vmax=vmax),
    )
    ax.add_collection(lc)

    xs = [xy[0] for xy in pos.values()]
    ys = [xy[1] for xy in pos.values()]
    ax.scatter(xs, ys, s=8, alpha=0.6, zorder=3)

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(title)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.grid(True, linestyle="--", alpha=0.3)

    cbar = plt.colorbar(lc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"GNNExplainer edge importance ({score_col})")

    if topk_edges and topk_edges > 0:
        top_edges = edges_df.head(topk_edges)
        segs2 = []
        for _, r in top_edges.iterrows():
            u, v = r["u"], r["v"]
            if (u in pos) and (v in pos):
                segs2.append([pos[u], pos[v]])
        if segs2:
            lc2 = LineCollection(segs2, linewidths=3.0, colors="r", alpha=0.7)
            ax.add_collection(lc2)

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=220)
    plt.show()


def plot_top_nodes(pos, nodes_df, title, topk_nodes=20, save=None):
    fig, ax = plt.subplots(figsize=(7, 7))
    xs = [xy[0] for xy in pos.values()]
    ys = [xy[1] for xy in pos.values()]
    ax.scatter(xs, ys, s=8, alpha=0.4, zorder=1, label="All nodes")

    topN = nodes_df.head(topk_nodes)
    sizes = 30 + 170 * (np.abs(topN["score_in"].values) / (np.abs(topN["score_in"].values).max() + 1e-12))
    xs2, ys2 = [], []
    for pid in topN["pole_id"].values:
        if pid in pos:
            xs2.append(pos[pid][0]); ys2.append(pos[pid][1])
    ax.scatter(xs2, ys2, s=sizes, color="crimson", alpha=0.85, zorder=3, label=f"Top-{topk_nodes} nodes")

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(title)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=220)
    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === 数据 ===
    dataset = MyBinaryDataset(
        root="GNN_data/evaluate",
        file_name=EVAL_CSV,
        oriented_pkl=ORIENTED_PKL,
        edge_feat_csv=EDGE_CSV,
    )
    data_one, idx = pick_data_for_feeder(dataset, FEEDER_TARGET)

    # === 模型 ===
    model = NetA_NodeOnly(lambda_consistency=0.0, hidden_dim=256, edge_dim=6, dropout_p=0.5).to(device)
    state = torch.load(MODEL_CKPT, map_location=device)
    model.load_state_dict(state["model"], strict=False)
    model.eval()

    with torch.no_grad():
        base_prob = model(data_one).item()
    print(f"[INFO] feeder={FEEDER_TARGET} | prob={base_prob:.6f} -> {'POS' if base_prob>0.5 else 'NEG'}")

    # === 包装模型并构建 Explainer（仅学习 edge_mask） ===
    wrapped = WrappedModel(model, data_one).to(device)
    explainer = build_explainer(wrapped)

    # === 运行解释 ===
    explanation, ei_used, _ = run_gnnexplainer_on_graph(explainer, data_one, target_class=TARGET_CLASS)

    # 取出边重要性（长度 = 有向边数 E_dir）
    edge_mask = explanation.edge_mask
    assert edge_mask is not None, "edge_mask is None — 请检查 PyG 版本或 Explainer 配置"

    # === 排序/打印 Top-K ===
    node_order, pos_xy, feeder_label = build_pos_from_csv(EVAL_CSV, FEEDER_TARGET)
    edges_sorted, nodes_sorted = edges_nodes_from_mask(ei_used, edge_mask, node_order, colname="score")

    print("\n=== Top edges by GNNExplainer (directed) ===")
    print(edges_sorted.head(TOPK_EDGES).to_string(index=False))
    print("\n=== Top nodes by incoming edge importance ===")
    print(nodes_sorted.head(TOPK_NODES).to_string(index=False))

    # === 可视化 ===
    title_edge = f"Feeder: {FEEDER_TARGET} | label={feeder_label} | GNNExplainer edge importance"
    plot_edges_scores(pos_xy, edges_sorted, title_edge, score_col="score", topk_edges=TOPK_EDGES)

    title_node = f"Feeder: {FEEDER_TARGET} | Top-{TOPK_NODES} nodes (incoming importance)"
    plot_top_nodes(pos_xy, nodes_sorted, title_node, topk_nodes=TOPK_NODES)

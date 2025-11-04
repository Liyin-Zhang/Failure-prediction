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
from GNN_network_02 import NetA_NodeOnly
from GNN_args import *
import torch.nn.functional as F

# ========= 自行设置 =========
EVAL_CSV     = "GNN_data/evaluate/evaluate_0415_3_filtered_3.csv"
EDGE_CSV     = "GNN_data/evaluate/evaluate_0415_3_filtered_3_edge_features.csv"
ORIENTED_PKL = "GNN_data/grid/origin_topology_3_oriented.pickle"
MODEL_CKPT   = "GNN_data/model/GNN_0417_gen_0_1015_02.pkl"

FEEDER_TARGET = "东方F3棠吓线03364"#"东方F3棠吓线03364"
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
    """
    适配当前“纯有向”NetA：forward(x, edge_index, **kwargs)
    - kwargs: edge_attr, batch
    - 直接把 explainer 提供的 edge_index/edge_attr 映射到 edge_index_dir/edge_attr
    """
    def __init__(self, base_model: NetA_NodeOnly, proto_data: Data):
        super().__init__()
        self.base = base_model
        self.proto_x     = proto_data.x.detach().clone()
        self.proto_batch = proto_data.batch.detach().clone()

    def forward(self, x, edge_index, **kwargs):
        edge_attr = kwargs.get('edge_attr', None)
        batch     = kwargs.get('batch', self.proto_batch)

        g = Data()
        g.x = self.proto_x
        # ★ 关键：只用“有向”字段
        g.edge_index_dir = edge_index
        g.edge_attr      = edge_attr
        g.batch = batch

        out = self.base(g)  # [B] 概率
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
        node_mask_type=None,             # 不学习节点/特征 mask
    )
    return explainer
def run_gnnexplainer_on_graph(explainer, data_one, target_class: int = 1):
    device = data_one.x.device
    N = data_one.x.size(0)
    x_dummy = torch.zeros((N, 1), device=device, dtype=torch.float)

    # ★ 用训练/推理一致的“有向边”
    ei = data_one.edge_index_dir
    ea = data_one.edge_attr

    # 构造同形状 target Tensor
    with torch.no_grad():
        y_hat_probe = explainer.model(x_dummy, ei, edge_attr=ea, batch=data_one.batch)
        target_tensor = torch.full_like(y_hat_probe, float(target_class),
                                        dtype=y_hat_probe.dtype, device=y_hat_probe.device)

    explanation = explainer(
        x=x_dummy, edge_index=ei, edge_attr=ea, batch=data_one.batch, target=target_tensor
    )
    em = explanation.edge_mask
    print("E_mask =", int(em.numel()), " E_used =", int(ei.size(1)))
    assert em.numel() == ei.size(1), "edge_mask 与 edge_index 边数不一致"

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



# # -*- coding: utf-8 -*-
# """
# 单馈线可解释分析（有向图版，TransformerConv）：
# - 指定 feeder_name，基于 NetA_NodeOnly（有向阶段 TransformerConv）导出注意力/重要度
# - 若当前 PyG 版本不支持 return_attention_weights，将回退用 edge_attr 的 L2 范数作为“边重要度”近似
# - 计算重要边/节点（基于重要度），并按经纬度可视化
# """
#
# import os
# import sys
# import torch
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.collections import LineCollection
# import seaborn as sns
#
# sys.path.append(os.path.dirname(__file__))
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#
# from torch_geometric.loader import DataLoader
# from GNN01_apply_dataset import MyBinaryDataset
# from GNN02_network import NetA_NodeOnly   # ← 你的“单纯有向图”模型类（如文件名/类名不同请改）
# from GNN_args import *                    # 使用你的超参/路径
#
# # ========= 自行设置 =========
# EVAL_CSV   = "GNN_data/evaluate/evaluate_0415_3_filtered_3.csv"          # 节点表（含经纬度）
# EDGE_CSV   = "GNN_data/evaluate/evaluate_0415_3_filtered_3_edge_features.csv"  # 边特征（与 Dataset 保持一致）
# ORIENTED_PKL = "GNN_data/grid/origin_topology_3_oriented.pickle"         # 构图
# MODEL_CKPT = "GNN_data/model/GNN_0417_gen_0_1015_02.pkl"                 # 有向模型权重
#
# FEEDER_TARGET = "10kV石滩F8碧江线03447"   # ← 指定要分析的馈线（CSV 原始名称）
# TOPK_EDGES = 30
# TOPK_NODES = 20
# # ==========================
#
# plt.rcParams['font.sans-serif'] = ['SimSun']
# plt.rcParams['axes.unicode_minus'] = False
#
#
# def read_csv_any(path: str) -> pd.DataFrame:
#     for enc in ("utf-8", "utf-8-sig", "gbk"):
#         try:
#             return pd.read_csv(path, encoding=enc, low_memory=False)
#         except Exception:
#             pass
#     return pd.read_csv(path, low_memory=False)
#
#
# def build_pos_from_csv(csv_path: str, feeder_raw: str):
#     """从总表构造该馈线的 node 顺序与经纬度映射。顺序必须与 Dataset 中一致（按 group 原顺序）"""
#     df = read_csv_any(csv_path)
#     g = df[df["feeder_name"] == feeder_raw].copy()
#     if g.empty:
#         raise ValueError(f"CSV 中找不到 feeder_name={feeder_raw}")
#     g["pole_id_str"] = g["pole_id"].astype(str)
#     order = g["pole_id_str"].tolist()  # 节点顺序
#     pos = {pid: (float(lon), float(lat)) for pid, lon, lat in
#            zip(g["pole_id_str"], g["longitude"], g["latitude"])}
#     label = g["label"].iloc[0]
#     return order, pos, bool(label)
#
#
# def pick_data_for_feeder(dataset: MyBinaryDataset, feeder_raw: str):
#     """依据 CSV 的分组顺序，定位到该馈线的 Data 对象（假设 Dataset 是按 group 顺序生成 Data 的）"""
#     src_df = read_csv_any(dataset.file_name)
#     feeders_in_order = [name for name, _ in src_df.groupby("feeder_name", sort=False)]
#     try:
#         idx = feeders_in_order.index(feeder_raw)
#     except ValueError:
#         raise ValueError(f"Dataset 源 CSV 中找不到 feeder_name={feeder_raw}")
#     loader = DataLoader(dataset, batch_size=1, shuffle=False)
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     for i, data in enumerate(loader):
#         if i == idx:
#             return data.to(device), idx
#     raise RuntimeError("遍历 DataLoader 时未找到目标样本（索引越界？）")
#
#
# def attention_from_directed(model: NetA_NodeOnly, data):
#     """
#     手动执行“有向阶段”，取第二层 TransformerConv 的注意力权重/边重要度。
#     返回：
#       H_dir: [N, d_h]
#       (edge_index_dir, alpha_dir2): ([2,E_dir], [E_dir])  —— 若注意力不可得，则 alpha 来源于 edge_attr 的 L2 范数
#     """
#     with torch.no_grad():
#         # 有向边及特征
#         ei_dir = data.edge_index_dir
#         eattr  = data.edge_attr
#
#         # Embedding 展开（你的模型：dir_conv 直接吃 x）
#         x = model.item_embedding(data.x)                 # [N, feat_num, emb]
#         x = x.reshape(-1, embed_dim * feature_num)       # [N, d_in]
#
#         # dir_conv1
#         try:
#             h1, (ei_d1, alpha_d1) = model.dir_conv1(
#                 x, ei_dir, edge_attr=eattr, return_attention_weights=True
#             )
#         except TypeError:
#             # 某些旧版 PyG 的 TransformerConv 不支持 return_attention_weights
#             # 退化：不取第一层注意力，先做前向
#             h1 = model.dir_conv1(x, ei_dir, edge_attr=eattr)
#             ei_d1, alpha_d1 = ei_dir, None
#         h1 = torch.relu(h1)
#
#         # dir_conv2 —— 我们重点取这一层
#         try:
#             h2, (ei_d2, alpha_d2) = model.dir_conv2(
#                 h1, ei_dir, edge_attr=eattr, return_attention_weights=True
#             )
#             H_dir = torch.relu(h2)
#
#             # 多头情况：按边取均值
#             if alpha_d2 is not None and alpha_d2.dim() == 2:   # [E, heads]
#                 alpha = alpha_d2.mean(dim=1)
#             else:
#                 alpha = alpha_d2
#
#         except TypeError:
#             # 回退：无注意力输出，用 edge_attr 的 L2 范数作为边重要度近似
#             h2 = model.dir_conv2(h1, ei_dir, edge_attr=eattr)
#             H_dir = torch.relu(h2)
#             ei_d2 = ei_dir
#             if eattr is not None:
#                 alpha = torch.norm(eattr, p=2, dim=1).to(H_dir.device)  # [E_dir]
#                 print("[WARN] 当前 TransformerConv 不支持 return_attention_weights；"
#                       "已使用 edge_attr 的 L2 范数作为边重要度近似。")
#             else:
#                 alpha = torch.zeros(ei_dir.size(1), device=H_dir.device)
#                 print("[WARN] 未提供 edge_attr，也无法取注意力；重要度将为 0。")
#
#     return H_dir, (ei_d2, alpha)
#
#
# def topk_edges_nodes(ei, alpha, node_order):
#     """从边重要度得到 Top-K 边与节点分数；节点分数=入边重要度和"""
#     src = ei[0].detach().cpu().numpy()
#     dst = ei[1].detach().cpu().numpy()
#     a   = alpha.detach().cpu().numpy().astype(float)
#
#     edges = pd.DataFrame({"src_idx": src, "dst_idx": dst, "alpha": a})
#     edges["u"] = [node_order[i] for i in edges["src_idx"]]
#     edges["v"] = [node_order[i] for i in edges["dst_idx"]]
#
#     node_scores = edges.groupby("dst_idx")["alpha"].sum().reindex(range(len(node_order)), fill_value=0.0)
#     nodes = pd.DataFrame({
#         "idx": range(len(node_order)),
#         "pole_id": node_order,
#         "score_in": node_scores.values
#     })
#     return edges.sort_values("alpha", ascending=False), nodes.sort_values("score_in", ascending=False)
#
#
# def plot_attention_map(pos, edges_df, title, root_id=None, topk_edges=30, save=None):
#     """按经纬度画边重要度。粗线条/颜色深浅表示权重；节点散点；可标 root ⭐"""
#     segs, vals = [], []
#     for _, r in edges_df.iterrows():
#         u, v, w = r["u"], r["v"], float(r["alpha"])
#         if (u in pos) and (v in pos):
#             segs.append([pos[u], pos[v]])
#             vals.append(w)
#     if not segs:
#         print("[WARN] 该馈线没有可画的边段")
#         return
#
#     vals = np.asarray(vals, dtype=float)
#     vmin, vmax = vals.min(), vals.max() + 1e-9
#
#     fig, ax = plt.subplots(figsize=(7, 7))
#     widths = 0.5 + 2.5 * ((vals - vmin) / (vmax - vmin + 1e-12))
#     lc = LineCollection(segs, array=vals, linewidths=widths, cmap="viridis",
#                         norm=plt.Normalize(vmin=vmin, vmax=vmax))
#     ax.add_collection(lc)
#
#     xs = [xy[0] for xy in pos.values()]
#     ys = [xy[1] for xy in pos.values()]
#     ax.scatter(xs, ys, s=8, alpha=0.7, zorder=3)
#
#     if root_id is not None and root_id in pos:
#         rx, ry = pos[root_id]
#         ax.scatter([rx], [ry], s=110, marker="*", edgecolors="k", linewidths=0.9,
#                    zorder=5, label=f"Root: {root_id}")
#         ax.legend(loc="best", fontsize=9)
#
#     ax.set_aspect("equal", adjustable="datalim")
#     ax.set_title(title)
#     ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
#     ax.grid(True, linestyle="--", alpha=0.3)
#
#     cbar = plt.colorbar(lc, ax=ax, fraction=0.046, pad=0.04)
#     cbar.set_label("Importance (dir_conv2)")
#
#     if topk_edges and topk_edges > 0:
#         top_edges = edges_df.nlargest(topk_edges, "alpha")
#         segs2 = []
#         for _, r in top_edges.iterrows():
#             u, v = r["u"], r["v"]
#             if (u in pos) and (v in pos):
#                 segs2.append([pos[u], pos[v]])
#         if segs2:
#             lc2 = LineCollection(segs2, linewidths=3.0, colors="r", alpha=0.7)
#             ax.add_collection(lc2)
#
#     plt.tight_layout()
#     if save:
#         plt.savefig(save, dpi=220)
#     plt.show()
#
#
# def plot_top_nodes(pos, nodes_df, title, topk_nodes=20, save=None):
#     """按节点分数（入边重要度和）画 Top-K"""
#     fig, ax = plt.subplots(figsize=(7, 7))
#     xs = [xy[0] for xy in pos.values()]
#     ys = [xy[1] for xy in pos.values()]
#     ax.scatter(xs, ys, s=8, alpha=0.4, zorder=1, label="All nodes")
#
#     topN = nodes_df.nlargest(topk_nodes, "score_in")
#     sizes = 30 + 170 * (topN["score_in"].values / (topN["score_in"].max() + 1e-12))
#     xs2, ys2 = [], []
#     for pid in topN["pole_id"].values:
#         if pid in pos:
#             xs2.append(pos[pid][0]); ys2.append(pos[pid][1])
#     ax.scatter(xs2, ys2, s=sizes, color="crimson", alpha=0.8, zorder=3,
#                label=f"Top-{topk_nodes} nodes")
#
#     ax.set_aspect("equal", adjustable="datalim")
#     ax.set_title(title)
#     ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
#     ax.grid(True, linestyle="--", alpha=0.3)
#     ax.legend(loc="best", fontsize=9)
#     plt.tight_layout()
#     if save:
#         plt.savefig(save, dpi=220)
#     plt.show()
#
#
# if __name__ == "__main__":
#     # 设备
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     # Dataset（与训练/评估一致）
#     dataset = MyBinaryDataset(
#         root="GNN_data/evaluate",
#         file_name=EVAL_CSV,
#         oriented_pkl=ORIENTED_PKL,
#         edge_feat_csv=EDGE_CSV,   # 若你的 Dataset 没有这个参数，请去掉
#     )
#
#     # 取该馈线的 Data
#     data_one, idx = pick_data_for_feeder(dataset, FEEDER_TARGET)
#
#     # 模型
#     model = NetA_NodeOnly(lambda_consistency=0.0, hidden_dim=256, edge_dim=6, dropout_p=0.5).to(device)
#     state = torch.load(MODEL_CKPT, map_location=device)
#     model.load_state_dict(state["model"], strict=False)
#     model.eval()
#
#     # 1) 正常前向拿到概率
#     with torch.no_grad():
#         prob = model(data_one).item()
#     print(f"[INFO] feeder={FEEDER_TARGET} | prediction prob={prob:.6f} "
#           f"(threshold=0.5 -> {'POS' if prob>0.5 else 'NEG'})")
#
#     # 2) 导出“有向阶段”的注意力/重要度（第二层）
#     H_dir, (ei_dir, alpha_dir2) = attention_from_directed(model, data_one)
#
#     # 3) 恢复节点顺序 & 经纬
#     node_order, pos_xy, feeder_label = build_pos_from_csv(EVAL_CSV, FEEDER_TARGET)
#     print(f"[INFO] feeder label={feeder_label}")
#
#     # 4) Top-K 边/节点
#     edges_sorted, nodes_sorted = topk_edges_nodes(ei_dir, alpha_dir2, node_order)
#     print("\n=== Top directed edges by importance ===")
#     print(edges_sorted.head(TOPK_EDGES).to_string(index=False))
#     print("\n=== Top nodes by incoming directed importance ===")
#     print(nodes_sorted.head(TOPK_NODES).to_string(index=False))
#
#     # 5) 可视化
#     title_edge = f"Feeder: {FEEDER_TARGET} | label={feeder_label} | dir_conv2 importance"
#     plot_attention_map(pos_xy, edges_sorted, title_edge, root_id=None, topk_edges=TOPK_EDGES)
#
#     title_node = f"Feeder: {FEEDER_TARGET} | Top-{TOPK_NODES} nodes (incoming directed importance)"
#     plot_top_nodes(pos_xy, nodes_sorted, title_node, topk_nodes=TOPK_NODES)

# -*- coding: utf-8 -*-
"""
多馈线因果贡献解释（GNNExplainer，边/节点重要性，批量保存）
- 对每个 feeder_name 计算 edge_mask（边重要度），节点重要度=入边重要度求和
- 图内 L1 归一化后，分别合并保存：
    * 节点分数 -> 合并到 EVAL_CSV（按 feeder_name + pole_id）
    * 边分数   -> 合并到 EDGE_CSV（按 feeder_name + u + v）
- 生成新文件：*_with_scores.csv
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.explain.config import ModelConfig
from torch_geometric.utils import to_undirected
import torch.nn.functional as F

sys.path.append(os.path.dirname(__file__))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from GNN01_apply_dataset import MyBinaryDataset
from GNN_network_01 import NetA_NodeOnly
from GNN_args import *

# ========= 自行设置 =========
EVAL_CSV     = "GNN_data/evaluate/evaluate_0415_3_filtered_3.csv"
EDGE_CSV     = "GNN_data/evaluate/evaluate_1020_filtered_edge_features.csv"
ORIENTED_PKL = "GNN_data/grid/origin_topology_3_oriented.pickle"
MODEL_CKPT   = "GNN_data/model/GNN_0417_gen_0_1015_01.pkl"

EXPLAIN_EPOCHS = 300
EXPLAIN_LR     = 0.01
TARGET_CLASS   = 1
# ==========================

def read_csv_any(path: str) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception:
            pass
    return pd.read_csv(path, low_memory=False)

def feeders_in_source(csv_path: str):
    df = read_csv_any(csv_path)
    # 按原始顺序取分组（与 Dataset 的生成顺序一致）
    return [name for name, _ in df.groupby("feeder_name", sort=False)]

def build_node_order_and_label(csv_path: str, feeder_raw: str):
    df = read_csv_any(csv_path)
    g = df[df["feeder_name"] == feeder_raw].copy()
    if g.empty:
        raise ValueError(f"CSV 中找不到 feeder_name={feeder_raw}")
    pole_ids = g["pole_id"].astype(str).tolist()
    label = int(pd.Series(g["label"]).mode().iloc[0])
    return pole_ids, bool(label)

class WrappedModel(torch.nn.Module):
    """Explainer 期望 forward(x, edge_index, **kwargs)。我们把传入边映射给模型。"""
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
        # 这里使用“无向阶段”或“to_undirected” —— 与 run_explainer 保持一致
        g.edge_index_und = edge_index
        g.edge_attr_und  = edge_attr
        # 同时填充有向字段，避免模型内部分支触发 to_undirected
        g.edge_index_dir = edge_index
        g.edge_attr      = edge_attr
        g.batch = batch
        return self.base(g)  # [B] 概率

def build_explainer(wrapped_model: torch.nn.Module):
    model_config = ModelConfig(
        mode='binary_classification',
        task_level='graph',
        return_type='probs',
    )
    explainer = Explainer(
        model=wrapped_model,
        algorithm=GNNExplainer(epochs=EXPLAIN_EPOCHS, lr=EXPLAIN_LR),
        explanation_type='phenomenon',
        model_config=model_config,
        edge_mask_type='object',
        node_mask_type=None,
    )
    return explainer

@torch.no_grad()
def _probe_target_tensor(explainer, x_dummy, ei, ea, batch, target_class: int):
    y_hat = explainer.model(x_dummy, ei, edge_attr=ea, batch=batch)
    # 与 y_hat 同形状/类型的 target
    return torch.full_like(y_hat, float(target_class))

def run_explainer_on_one_graph(model, data_one: Data, target_class: int = 1):
    """对单个图运行 GNNExplainer，返回 edge_index_used, edge_mask"""
    device = data_one.x.device
    N = data_one.x.size(0)
    x_dummy = torch.zeros((N, 1), device=device, dtype=torch.float)

    # 选择与模型一致的一套边（若数据提供无向边优先用之，否则由有向边生成）
    if hasattr(data_one, 'edge_index_und') and hasattr(data_one, 'edge_attr_und') and data_one.edge_index_und is not None:
        ei = data_one.edge_index_und
        ea = data_one.edge_attr_und
    else:
        ei, ea = to_undirected(data_one.edge_index_dir, edge_attr=data_one.edge_attr, reduce='mean')

    wrapped = WrappedModel(model, data_one).to(device)
    explainer = build_explainer(wrapped)
    target_tensor = _probe_target_tensor(explainer, x_dummy, ei, ea, data_one.batch, target_class)

    explanation = explainer(x=x_dummy, edge_index=ei, edge_attr=ea, batch=data_one.batch, target=target_tensor)
    em = explanation.edge_mask
    assert em is not None and em.numel() == ei.size(1), "edge_mask 与使用的 edge_index 边数不一致"
    return ei, ea, em

def edges_nodes_from_mask(ei, edge_mask, node_order):
    """返回（边表, 节点表）；节点分数=入边分数和"""
    src = ei[0].detach().cpu().numpy()
    dst = ei[1].detach().cpu().numpy()
    a   = edge_mask.detach().cpu().numpy().astype(float)

    edges = pd.DataFrame({"src_idx": src, "dst_idx": dst, "edge_score": a})
    edges["u"] = [node_order[i] for i in edges["src_idx"]]
    edges["v"] = [node_order[i] for i in edges["dst_idx"]]

    node_scores = edges.groupby("dst_idx")["edge_score"].sum().reindex(range(len(node_order)), fill_value=0.0)
    nodes = pd.DataFrame({"idx": range(len(node_order)),
                          "pole_id": node_order,
                          "node_score": node_scores.values})
    return edges, nodes

def l1_normalize_in_graph(df_edges, df_nodes):
    eps = 1e-12
    # 边：和=1
    s_e = df_edges["edge_score"].to_numpy(dtype=float)
    s_e = s_e / (s_e.sum() + eps)
    df_edges["edge_score_norm"] = s_e
    # 节点：和=1
    s_n = df_nodes["node_score"].to_numpy(dtype=float)
    s_n = s_n / (s_n.sum() + eps)
    df_nodes["node_score_norm"] = s_n
    return df_edges, df_nodes

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === 加载原始总表（用于合并分数） ===
    eval_df_all = read_csv_any(EVAL_CSV).copy()
    eval_df_all["pole_id"] = eval_df_all["pole_id"].astype(str)

    edge_df_all = read_csv_any(EDGE_CSV).copy()
    # 兼容不同命名，尽量获得 u/v 两列（字符串）
    cand_u = [c for c in ["u", "src", "from", "pole_u", "pole_src"] if c in edge_df_all.columns]
    cand_v = [c for c in ["v", "dst", "to",   "pole_v", "pole_dst"] if c in edge_df_all.columns]
    if not cand_u or not cand_v:
        # 若没有键列，就先创建占位，后续仅输出新的得分文件
        edge_df_all["u"] = ""
        edge_df_all["v"] = ""
    else:
        if "u" not in edge_df_all.columns: edge_df_all["u"] = edge_df_all[cand_u[0]]
        if "v" not in edge_df_all.columns: edge_df_all["v"] = edge_df_all[cand_v[0]]
    edge_df_all["u"] = edge_df_all["u"].astype(str)
    edge_df_all["v"] = edge_df_all["v"].astype(str)

    # === Dataset & 模型 ===
    dataset = MyBinaryDataset(
        root="GNN_data/evaluate",
        file_name=EVAL_CSV,
        oriented_pkl=ORIENTED_PKL,
        edge_feat_csv=EDGE_CSV,
    )
    model = NetA_NodeOnly(lambda_consistency=0.0, hidden_dim=256, edge_dim=6, dropout_p=0.5).to(device)
    state = torch.load(MODEL_CKPT, map_location=device)
    model.load_state_dict(state["model"], strict=False)
    model.eval()

    # === 遍历所有馈线，收集分数 ===
    feeders = feeders_in_source(EVAL_CSV)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 结果临时表（跨馈线）
    edges_scores_pool = []  # 每行包含：feeder_name, u, v, edge_score_norm
    nodes_scores_pool = []  # 每行包含：feeder_name, pole_id, node_score_norm

    for idx, data_one in enumerate(loader):

        feeder_name = feeders[idx]
        if feeder_name == '七星F1803397':
            continue
        data_one = data_one.to(device)

        # 原模型预测（可选打印）
        with torch.no_grad():
            prob = model(data_one).item()
        print(f"[{idx+1}/{len(feeders)}] feeder={feeder_name} | prob={prob:.6f}")

        # 该馈线的节点顺序与标签
        node_order, feeder_label = build_node_order_and_label(EVAL_CSV, feeder_name)

        # 解释
        ei, ea, edge_mask = run_explainer_on_one_graph(model, data_one, target_class=TARGET_CLASS)

        # 计算边/节点分数并做 L1 归一化
        df_e, df_n = edges_nodes_from_mask(ei, edge_mask, node_order)
        df_e, df_n = l1_normalize_in_graph(df_e, df_n)

        df_e["feeder_name"] = feeder_name
        df_n["feeder_name"] = feeder_name

        # 仅保留需要合并的键与分数列
        edges_scores_pool.append(df_e[["feeder_name", "u", "v", "edge_score_norm","edge_score"]].copy())
        nodes_scores_pool.append(df_n[["feeder_name", "pole_id", "node_score_norm","node_score"]].copy())

    edges_scores_all = pd.concat(edges_scores_pool, ignore_index=True)
    nodes_scores_all = pd.concat(nodes_scores_pool, ignore_index=True)

    # === 合并并另存：节点到 EVAL_CSV ===
    eval_merge = eval_df_all.merge(
        nodes_scores_all, how="left", on=["feeder_name", "pole_id"]
    )
    eval_out = EVAL_CSV.replace(".csv", "_with_scores_und_1103.csv")
    eval_merge.to_csv(eval_out, index=False, encoding="utf-8-sig")
    print(f"[SAVE] 节点分数已保存: {eval_out}  (rows={len(eval_merge)})")

    # === 合并并另存：边到 EDGE_CSV ===
    # 注意：若原 EDGE_CSV 中没有 u/v 或键不匹配，将只保存得分表（不 merge）
    can_merge_edges = all(k in edge_df_all.columns for k in ["feeder_name", "u", "v"])
    if can_merge_edges:
        edge_merge = edge_df_all.merge(
            edges_scores_all, how="left", on=["feeder_name", "u", "v"]
        )
        edge_out = EDGE_CSV.replace(".csv", "_with_scores_und_1103.csv")
        edge_merge.to_csv(edge_out, index=False, encoding="utf-8-sig")
        print(f"[SAVE] 边分数已保存: {edge_out}  (rows={len(edge_merge)})")
    else:
        # 回退：仅输出分数列表，供后续手工 join
        edge_out = EDGE_CSV.replace(".csv", "_scores_only_1103.csv")
        edges_scores_all.to_csv(edge_out, index=False, encoding="utf-8-sig")
        print(f"[SAVE] 边分数键列不足，已输出独立分数表: {edge_out}  (rows={len(edges_scores_all)})")

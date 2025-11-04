# -*- coding: utf-8 -*-
"""
基于 *_with_scores.csv 的可视化（scatter）：
- 读取节点或边的 score（已做每图 L1 归一化）与指定特征列
- 支持四种重要度选点/锐化策略：topk | topq | cover_p | power
- 支持重要性采样（按 label 类别分开，带权抽样）
- True/False 两类不同颜色；可将 score 映射到点大小/透明度
"""

import os
import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ========== 配置参数（请按需修改） ==========
# 输入文件（刚保存的 *_with_scores.csv）
# INPUT_FILE = "GNN_data/evaluate/evaluate_0415_3_filtered_3_with_scores_dir.csv"  # 节点示例
INPUT_FILE = "GNN_data/evaluate/evaluate_0415_3_filtered_3_edge_features_with_scores_und.csv"  # 边示例

# 数据类型：'node' 或 'edge'
DATA_KIND = "edge"   # "edge"

# 画哪个特征列（必须存在于 INPUT_FILE 中）
FEATURE_COL = "PRE"

# 评分列名（节点/边）
NODE_SCORE_COL = "WIN"#"node_score_norm"
EDGE_SCORE_COL = "flow"#"edge_score_norm"

# 若 DATA_KIND='edge' 且文件内没有 label，需要从节点总表合并（按 feeder_name 多数票）
LABEL_SOURCE_CSV = "GNN_data/evaluate/evaluate_0415_3_filtered_3.csv"

# 可选：仅画指定馈线（为空则全部）
FILTER_FEEDERS = []  # 例如 ["东方F3棠吓线03364"]

# ---- 选点/锐化策略（四选一） ----
SELECT_MODE = "topk"     # "topk" | "topq" | "cover_p" | "power"
TOPK_K      = 10            # topk 时每图保留的节点/边数
TOPQ_Q      = 0.10          # topq 时每图保留比例（0~1）
COVER_P     = 0.20          # cover_p 时每图累积覆盖率阈值（0~1）
POWER_GAMMA = 2.0           # power 锐化的幂次（>1 越尖锐）
KEEP_REST   = False         # 仅 topk/topq/cover_p 模式有效；True=保留所有点并标注_selected

# ---- 重要性采样（可选） ----
ENABLE_IMPORTANCE_SAMPLING = False
SAMPLE_PER_CLASS = 8000     # 每个 label 类最多抽取数
SAMPLING_RANDOM_STATE = 42

# ---- 可视化外观 ----
LOG_X = False
LOG_Y = False
JITTER = 0.0                 # y 方向微小抖动，避免重叠（例如 0.001）
COLOR_TRUE  = "#d62728"
COLOR_FALSE = "#1f77b4"
BASE_POINT_SIZE = 18         # 基础点大小
POINT_ALPHA = 0.35           # 基础透明度

# 将 score 映射到点大小/透明度（二选零到多）
SIZE_BY_SCORE = True
SIZE_MIN, SIZE_MAX = 16, 200
ALPHA_BY_SCORE = False
ALPHA_MIN, ALPHA_MAX = 0.10, 0.80

# 输出图片路径（留空自动命名）
OUTPUT_PNG = ""
FIGSIZE = (7.5, 5.2)
DPI = 220
# ===========================================


# ----------------- 工具函数 -----------------
def read_csv_any(path: str) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception:
            pass
    return pd.read_csv(path, low_memory=False)

def ensure_label_for_edges(df_edge: pd.DataFrame, label_source_csv: str) -> pd.DataFrame:
    """
    边文件若无 label，从节点总表按 feeder_name 合并多数票 label。
    """
    if "label" in df_edge.columns:
        return df_edge

    if "feeder_name" not in df_edge.columns:
        raise ValueError("边文件中没有 feeder_name，无法补充 label。")

    eval_df = read_csv_any(label_source_csv).copy()
    if "label" not in eval_df.columns:
        raise ValueError("LABEL_SOURCE_CSV 中没有 label 列，无法补充。")

    lab_map = (eval_df.groupby("feeder_name")["label"]
                     .agg(lambda s: int(pd.Series(s).mode().iloc[0]))
                     .reset_index())
    out = df_edge.merge(lab_map, how="left", on="feeder_name")
    if out["label"].isna().any():
        print("[WARN] 部分边合并不到 label，可能 feeder_name 不一致。")
    return out

def label_to_bool(x):
    if isinstance(x, str):
        t = x.strip().lower()
        if t in {"true", "t", "yes", "y"}:  return True
        if t in {"false", "f", "no", "n"}:  return False
    try:
        return bool(int(x))
    except Exception:
        return bool(x)

def filter_or_sharpen_groups(
    df: pd.DataFrame,
    feeder_col: str,
    score_col: str,
    mode: str = "cover_p",   # "topk" | "topq" | "cover_p" | "power"
    k: int = 50,
    q: float = 0.10,
    p: float = 0.80,
    gamma: float = 2.0,
    keep_rest: bool = False
) -> pd.DataFrame:
    """
    选点/锐化：
      - topk/topq/cover_p: 返回筛选后的子集（keep_rest=True 时保留所有点并加 _selected 标志）
      - power: 对每图做 s' = s^gamma / sum(s^gamma)（保留所有点）
    """
    assert feeder_col in df.columns and score_col in df.columns, "缺少必要列"
    df = df.copy()

    def topk_group(g):  return g.nlargest(k, score_col)

    def topq_group(g):
        m = max(1, int(np.ceil(len(g) * q)))
        return g.nlargest(m, score_col)

    def cover_p_group(g):
        g2 = g.sort_values(score_col, ascending=False).copy()
        g2["cum"] = g2[score_col].cumsum()
        cut = g2[g2["cum"] <= p]
        if cut.empty:
            cut = g2.head(1)
        return cut.drop(columns=["cum"])

    def power_group(g):
        s = g[score_col].to_numpy(dtype=float)
        s_new = np.power(s, gamma)
        s_new /= (s_new.sum() + 1e-12)
        g2 = g.copy()
        g2[score_col] = s_new
        return g2

    if mode == "power":
        return df.groupby(feeder_col, group_keys=False).apply(power_group)

    if mode == "topk":
        sel = df.groupby(feeder_col, group_keys=False).apply(topk_group)
    elif mode == "topq":
        sel = df.groupby(feeder_col, group_keys=False).apply(topq_group)
    elif mode == "cover_p":
        sel = df.groupby(feeder_col, group_keys=False).apply(cover_p_group)
    else:
        raise ValueError("mode 必须为 'topk' | 'topq' | 'cover_p' | 'power'")

    if keep_rest:
        df["_selected"] = False
        df.loc[sel.index, "_selected"] = True
        return df
    else:
        return sel

def importance_sample_by_label(
    df: pd.DataFrame,
    score_col: str,
    label_col: str = "label_bool",
    per_class_n: int = 5000,
    random_state: int = 42
) -> pd.DataFrame:
    """
    按 label 分组做带权（score）不放回抽样，控制显示点数。
    """
    rng = np.random.default_rng(random_state)
    out = []
    for lbl, g in df.groupby(label_col):
        if len(g) <= per_class_n:
            out.append(g)
            continue
        w = g[score_col].to_numpy(dtype=float)
        w = w / (w.sum() + 1e-12)
        idx = rng.choice(len(g), size=per_class_n, replace=False, p=w)
        out.append(g.iloc[idx])
    return pd.concat(out, ignore_index=True)

def auto_output_name(input_file: str, data_kind: str, feature_col: str, score_col: str) -> str:
    base = os.path.splitext(os.path.basename(input_file))[0]
    return f"scatter_{data_kind}_{feature_col}_vs_{score_col}_{base}.png"


# ----------------- 主流程 -----------------
def main():
    if DATA_KIND not in {"node", "edge"}:
        raise ValueError("DATA_KIND 必须为 'node' 或 'edge'")

    df = read_csv_any(INPUT_FILE).copy()

    # 统一键列类型
    for col in ["pole_id", "u", "v"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    if "feeder_name" not in df.columns:
        raise ValueError("文件中缺少 feeder_name 列，后续分组/筛选需要它。")

    # 过滤馈线
    if FILTER_FEEDERS:
        df = df[df["feeder_name"].isin(FILTER_FEEDERS)].copy()

    # score 列
    score_col = NODE_SCORE_COL if DATA_KIND == "node" else EDGE_SCORE_COL
    if score_col not in df.columns:
        raise ValueError(f"文件中找不到评分列：{score_col}")

    # label：节点文件一般自带；边文件可能需要补充
    if "label" not in df.columns and DATA_KIND == "edge":
        df = ensure_label_for_edges(df, LABEL_SOURCE_CSV)
    if "label" not in df.columns:
        raise ValueError("无法找到 label 列。节点文件应自带；边文件请提供 LABEL_SOURCE_CSV 以合并。")

    # 特征列检查
    if FEATURE_COL not in df.columns:
        raise ValueError(f"文件中找不到特征列：{FEATURE_COL}")

    # 清理缺失
    df = df.dropna(subset=[score_col, FEATURE_COL, "label"]).copy()

    # label -> bool
    df["label_bool"] = df["label"].map(label_to_bool)

    # 选点/锐化（按 feeder_name 分组）
    df_sel = filter_or_sharpen_groups(
        df, feeder_col="feeder_name", score_col=score_col,
        mode=SELECT_MODE, k=TOPK_K, q=TOPQ_Q, p=COVER_P,
        gamma=POWER_GAMMA, keep_rest=KEEP_REST
    )

    # 重要性采样（可选）
    if ENABLE_IMPORTANCE_SAMPLING:
        df_sel = importance_sample_by_label(
            df_sel, score_col=score_col, label_col="label_bool",
            per_class_n=SAMPLE_PER_CLASS, random_state=SAMPLING_RANDOM_STATE
        )

    # 画散点
    plt.figure(figsize=FIGSIZE)

    def size_from_score(s):
        if not SIZE_BY_SCORE:
            return np.full_like(s, BASE_POINT_SIZE, dtype=float)
        s = np.asarray(s, dtype=float)
        if s.max() <= 0:
            return np.full_like(s, BASE_POINT_SIZE, dtype=float)
        s01 = s / (s.max() + 1e-12)
        return SIZE_MIN + (SIZE_MAX - SIZE_MIN) * s01

    def alpha_from_score(s):
        if not ALPHA_BY_SCORE:
            return POINT_ALPHA
        s = np.asarray(s, dtype=float)
        if s.max() <= 0:
            return POINT_ALPHA
        s01 = s / (s.max() + 1e-12)
        return ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * s01

    for lbl, sub in df_sel.groupby("label_bool"):
        x = sub[FEATURE_COL].to_numpy(dtype=float)
        y = sub[score_col].to_numpy(dtype=float)
        # 抖动（仅 y 方向）
        if JITTER and JITTER > 0:
            y = y + np.random.uniform(-JITTER, JITTER, size=len(y))
        sizes = size_from_score(y)
        alpha = alpha_from_score(y)
        col = COLOR_TRUE if lbl else COLOR_FALSE
        lab = "True" if lbl else "False"
        # 如果 alpha_by_score，matplotlib 只能接受标量或与点等长数组
        if isinstance(alpha, np.ndarray):
            for xi, yi, si, ai in zip(x, y, sizes, alpha):
                plt.scatter([xi], [yi], s=si, alpha=ai, c=col, label=None)
            # 添加图例句柄（用均值 alpha 画一个虚拟点）
            plt.scatter([np.nan], [np.nan], s=BASE_POINT_SIZE, alpha=np.mean(alpha), c=col, label=lab)
        else:
            plt.scatter(x, y, s=sizes, alpha=alpha, c=col, label=lab)

    if LOG_X: plt.xscale("log")
    if LOG_Y: plt.yscale("log")

    title_kind = "Node" if DATA_KIND == "node" else "Edge"
    plt.xlabel(FEATURE_COL)
    plt.ylabel(score_col)
    plt.title(f"{title_kind} scatter: {FEATURE_COL} vs {score_col}  [{SELECT_MODE}]")
    plt.grid(alpha=0.3, linestyle="--")
    plt.legend(loc="best")
    plt.tight_layout()

    out_png = OUTPUT_PNG or auto_output_name(INPUT_FILE, DATA_KIND, FEATURE_COL, score_col)
    plt.savefig(out_png, dpi=DPI)
    print(f"[SAVE] 图已保存: {out_png}  (n={len(df_sel)})")
    plt.show()


if __name__ == "__main__":
    main()

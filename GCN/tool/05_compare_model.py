# -*- coding: utf-8 -*-
"""
分析：有向模型(m2)预测对而无向模型(m1)预测错的样本，是否在边特征统计上有显著差异
输入：
  - M1_CSV: 无向模型预测结果（feeder_name, label, risk）
  - M2_CSV: 有向模型预测结果（feeder_name, label, risk）
  - EDGE_CSV: 边特征明细（由你之前脚本生成：feeder_name, u, v, WIN, PRE, TMP, PRS, SHU, flow）
输出（到 out_dir）：
  - per_feeder_features.csv   # 馈线级聚合特征明细
  - compare_main.csv          # 主分析（m2对&m1错 vs 两者皆对）的检验结果
  - compare_TN_only.csv       # 仅 TN 场景下的检验结果（可选）
  - figs/*.png                # 可选的箱线图/小提琴图
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import warnings
warnings.filterwarnings("ignore")

# ========== 配置（自行替换路径） ==========
M1_CSV   = "GNN_data/output/result_feeder_0417_gen_0_1020_01.csv"  # 无向+Transformer
M2_CSV   =  "GNN_data/output/result_feeder_0417_gen_0_1020_02.csv"  # 有向+Transformer
EDGE_CSV = 'GNN_data/evaluate/evaluate_0415_3_filtered_3_edge_features.csv'  # 之前导出的边特征
OUT_DIR  = "edge_weight_analysis"
PLOT     = True  # 画图可设为 False 关闭
# ========================================

os.makedirs(OUT_DIR, exist_ok=True)
if PLOT:
    import matplotlib.pyplot as plt
    import seaborn as sns
    os.makedirs(os.path.join(OUT_DIR, "figs"), exist_ok=True)

def read_any_csv(path):
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception:
            pass
    raise RuntimeError(f"无法读取 CSV: {path}")

def norm_pred(df, name):
    cols = {c.lower(): c for c in df.columns}
    need = {"feeder_name", "label", "risk"}
    lower = set(k.lower() for k in df.columns)
    miss = need - lower
    if miss:
        raise ValueError(f"{name} 缺少列: {miss}")
    g = df.rename(columns={
        cols["feeder_name"]: "feeder_name",
        cols["label"]: "label",
        cols["risk"]: "risk",
    }).copy()
    g = g.drop_duplicates("feeder_name", keep="first")
    def to_bool(x):
        if isinstance(x, str):
            t = x.strip().lower()
            if t in ("true","t","yes","y","1"): return True
            if t in ("false","f","no","n","0"): return False
        return bool(x)
    g["label"] = g["label"].map(to_bool)
    g["risk"]  = g["risk"].map(lambda v: int(bool(v)))
    g["model"] = name
    return g

def cliff_delta(x, y):
    """Cliff's delta: [-1,1], 绝对值越大效应越强"""
    x = np.asarray(x); y = np.asarray(y)
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return np.nan
    # 高效近似：排序后双指针；这里用简化实现（适中样本量够用）
    gt = 0; lt = 0
    for xi in x:
        gt += np.sum(xi > y)
        lt += np.sum(xi < y)
    return (gt - lt) / (nx * ny)

# 1) 读取并对齐预测
m1 = norm_pred(read_any_csv(M1_CSV), "m1")
m2 = norm_pred(read_any_csv(M2_CSV), "m2")

common = set(m1.feeder_name) & set(m2.feeder_name)
m1 = m1[m1.feeder_name.isin(common)].reset_index(drop=True)
m2 = m2[m2.feeder_name.isin(common)].reset_index(drop=True)

# 合并一个表，便于分组
both = m1[["feeder_name","label","risk"]].merge(
    m2[["feeder_name","risk"]].rename(columns={"risk":"risk_m2"}),
    on="feeder_name", how="inner"
).rename(columns={"risk":"risk_m1"})
# both: feeder_name, label(bool), risk_m1(0/1), risk_m2(0/1)

# 2) 读取边特征，并做馈线级聚合
edges = read_any_csv(EDGE_CSV)
# 只保留在 common 内的馈线
edges = edges[edges["feeder_name"].isin(common)].copy()

# 基础特征列（你可增减）
edge_feats = ["WIN","PRE","TMP","PRS","SHU","flow"]
# 聚合函数集合
def topk_mean(arr, k_ratio=0.1):
    if len(arr) == 0: return np.nan
    k = max(1, int(len(arr)*k_ratio))
    return np.mean(np.sort(arr)[-k:])

def flow_weighted_mean(feat, flow):
    if len(feat) == 0 or len(flow) == 0: return np.nan
    w = np.asarray(flow, dtype=float)
    x = np.asarray(feat, dtype=float)
    s = w.sum()
    if s <= 0: return np.nan
    return float(np.dot(x, w) / s)

agg_rows = []
for feeder, g in edges.groupby("feeder_name"):
    row = {"feeder_name": feeder}
    for f in edge_feats:
        vals = g[f].astype(float).values
        row[f+"_mean"]   = np.mean(vals)
        row[f+"_median"] = np.median(vals)
        row[f+"_std"]    = np.std(vals, ddof=1) if len(vals)>1 else 0.0
        row[f+"_max"]    = np.max(vals) if len(vals)>0 else np.nan
        row[f+"_p95"]    = np.percentile(vals, 95) if len(vals)>0 else np.nan
        row[f+"_top10"]  = topk_mean(vals, 0.1)
        # flow 加权均值（对每个特征，用 flow 作权）
        row[f+"_floww"]  = flow_weighted_mean(vals, g["flow"].values)
    agg_rows.append(row)

per_feeder = pd.DataFrame(agg_rows)
per_feeder.to_csv(os.path.join(OUT_DIR, "per_feeder_features.csv"), index=False, encoding="utf-8-sig")

# 3) 组定义
# 主分析：m2 正确 & m1 错误   vs   两者都正确
both["correct_m1"] = (both["risk_m1"] == both["label"].astype(int))
both["correct_m2"] = (both["risk_m2"] == both["label"].astype(int))

mask_main    =  both["correct_m2"] & (~both["correct_m1"])
mask_control =  both["correct_m2"] &   both["correct_m1"]

# 只看 TN 的可选分析：
mask_TN_case    = (~both["label"]) & (both["risk_m2"]==0) & (both["risk_m1"]!=0)
mask_TN_control = (~both["label"]) & (both["risk_m2"]==0) & (both["risk_m1"]==0)

# 4) 统计检验函数
def compare_groups(per_df, id_main, id_ctrl, tag):
    A = per_df[per_df["feeder_name"].isin(id_main)]
    B = per_df[per_df["feeder_name"].isin(id_ctrl)]
    feats = [c for c in per_df.columns if c!="feeder_name"]
    rows = []
    for f in feats:
        a = A[f].dropna().values
        b = B[f].dropna().values
        if len(a)==0 or len(b)==0:
            p = np.nan; d = np.nan
        else:
            # Mann–Whitney U（非参数，稳健）
            try:
                stat, p = mannwhitneyu(a, b, alternative="two-sided")
            except Exception:
                p = np.nan
            d = cliff_delta(a, b)
        rows.append({
            "feature": f,
            f"{tag}_mean(A)": np.mean(a) if len(a) else np.nan,
            f"{tag}_mean(B)": np.mean(b) if len(b) else np.nan,
            f"{tag}_median(A)": np.median(a) if len(a) else np.nan,
            f"{tag}_median(B)": np.median(b) if len(b) else np.nan,
            f"{tag}_std(A)": np.std(a, ddof=1) if len(a)>1 else 0.0,
            f"{tag}_std(B)": np.std(b, ddof=1) if len(b)>1 else 0.0,
            f"{tag}_pval_MWU": p,
            f"{tag}_cliffs_delta": d
        })
    out = pd.DataFrame(rows)
    return out.sort_values(f"{tag}_pval_MWU")

ids_main    = set(both.loc[mask_main, "feeder_name"])
ids_control = set(both.loc[mask_control, "feeder_name"])

cmp_main = compare_groups(per_feeder, ids_main, ids_control, tag="main")
cmp_main.to_csv(os.path.join(OUT_DIR, "compare_main.csv"), index=False, encoding="utf-8-sig")

print(f"[INFO] 主分析样本量：A(m2对&m1错)={len(ids_main)}, B(两者皆对)={len(ids_control)}")
print(cmp_main.head(12).to_string(index=False))

# —— 可选：只看 TN 的对比 —— #
ids_TN     = set(both.loc[mask_TN_case, "feeder_name"])
ids_TN_ctl = set(both.loc[mask_TN_control, "feeder_name"])
if len(ids_TN)>0 and len(ids_TN_ctl)>0:
    cmp_TN = compare_groups(per_feeder, ids_TN, ids_TN_ctl, tag="TNonly")
    cmp_TN.to_csv(os.path.join(OUT_DIR, "compare_TN_only.csv"), index=False, encoding="utf-8-sig")
    print(f"[INFO] TN-only 样本量：A(m2 TN&m1非TN)={len(ids_TN)}, B(两者同为TN)={len(ids_TN_ctl)}")
    print(cmp_TN.head(12).to_string(index=False))
else:
    print("[WARN] TN-only 对比样本不足，未输出 compare_TN_only.csv")

# 5) （可选）快速可视化几个显著特征
if PLOT:
    # 选取主分析中最显著的前 6 个特征画箱线图
    top_feats = cmp_main.nsmallest(6, "main_pval_MWU")["feature"].tolist()
    long = per_feeder.merge(
        both[["feeder_name","correct_m1","correct_m2"]],
        on="feeder_name", how="left"
    )
    long["group"] = np.select(
        [ ( long["correct_m2"] & (~long["correct_m1"]) ),
          ( long["correct_m2"] &   long["correct_m1"]  ) ],
        ["m2_correct_m1_wrong", "both_correct"],
        default="others"
    )

    for f in top_feats:
        plt.figure(figsize=(6,4))
        sns.boxplot(data=long[long["group"].isin(["m2_correct_m1_wrong","both_correct"])],
                    x="group", y=f)
        sns.stripplot(data=long[long["group"].isin(["m2_correct_m1_wrong","both_correct"])],
                      x="group", y=f, alpha=0.25, color="k")
        plt.title(f"[{f}] 主分析组间分布")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "figs", f"box_{f}.png"), dpi=200)
        plt.close()

    print(f"[DONE] 图已保存到 {os.path.join(OUT_DIR, 'figs')}")

# === 负类（label=False）且两模型预测不一致的馈线名称 ===
neg_mask   = ~both["label"]                    # 负类（实际为 False）
diff_mask  = both["risk_m1"] != both["risk_m2"]  # 两模型预测不一致

neg_diff = both.loc[neg_mask & diff_mask, ["feeder_name", "label", "risk_m1", "risk_m2"]].copy()

# 细分两种情况：
# A) m2 正确(TN=0) 且 m1 错误(FP=1) —— 负类上 m2 优于 m1
case_a = neg_diff[(neg_diff["risk_m2"] == 0) & (neg_diff["risk_m1"] == 1)].copy()
case_a["case"] = "m2_TN & m1_FP"

# B) m2 错误(FP=1) 且 m1 正确(TN=0) —— 负类上 m2 劣于 m1
case_b = neg_diff[(neg_diff["risk_m2"] == 1) & (neg_diff["risk_m1"] == 0)].copy()
case_b["case"] = "m2_FP & m1_TN"

# 合并总表
neg_diff_all = pd.concat([case_a, case_b], ignore_index=True)

# 输出文件
case_a_out = os.path.join(OUT_DIR, "neg_m2_TN_m1_FP.csv")
case_b_out = os.path.join(OUT_DIR, "neg_m2_FP_m1_TN.csv")
all_out    = os.path.join(OUT_DIR, "neg_diff_all.csv")

case_a.to_csv(case_a_out, index=False, encoding="utf-8-sig")
case_b.to_csv(case_b_out, index=False, encoding="utf-8-sig")
neg_diff_all.to_csv(all_out, index=False, encoding="utf-8-sig")

# 控制台打印摘要 & 名称列表
print("\n=== 负类且两模型预测不一致的馈线 ===")
print(f"A) m2_TN & m1_FP：{len(case_a)} 条")
if not case_a.empty:
    print(case_a["feeder_name"].to_list())
print(f"B) m2_FP & m1_TN：{len(case_b)} 条")
if not case_b.empty:
    print(case_b["feeder_name"].to_list())
print(f"[SAVE] 详细列表已保存：\n - {case_a_out}\n - {case_b_out}\n - {all_out}")

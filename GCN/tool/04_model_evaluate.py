# -*- coding: utf-8 -*-
"""
对比多个模型的 feeder 级分类结果 并新增极端样本分层评估与分区准确率汇总
统一进行标签翻转：把原测试集与评估结果中的少数类(原0)作为新的正类
实现位置：normalize_frame 中将 label 与 risk 同时翻转，其余逻辑保持不变

- 输入 N 个模型结果 CSV 至少含 feeder_name label risk
- 新增输入 TEST_FEATURE_CSV 至少含 feeder_name WIN PRE 可选 TRAIN_FEATURE_CSV 用于阈值计算
- 处理 按 feeder_name 将 WIN 与 PRE 聚合 可选均值或最大值 基于分位数分层
- 输出
  results/summary_metrics.csv                           传统整体指标 加入各分区准确率列
  results/overlap_<CAT>.csv                             各类别交集数量矩阵
  results/jaccard_<CAT>.csv                             各类别Jaccard矩阵
  results/deltas_<other>_vs_<baseline>.csv              相对基线的改进与回退
  results/extreme_accuracy_by_bin.csv                   极端分层准确率与指标明细

新增开关 ONLY_NEG_LABEL_FOR_BIN_METRICS
  若为 True 在计算分层指标时 仅使用标签为 False 的样本进行评估
  注意 分层划分仍基于整体样本 仅评估阶段筛选标签
"""

import os
import pandas as pd
import numpy as np

# ================= 配置区 =================
# CSV_PATHS = [
#     "GNN_data/output/result_feeder_0417_gen_0_1020_00.csv",
#     "GNN_data/output/result_feeder_0417_gen_0_1020_07.csv",
#     "GNN_data/output/result_feeder_0417_gen_0_1020_01.csv",
#     "GNN_data/output/result_feeder_0417_gen_0_1020_02.csv",
#     "GNN_data/output/result_feeder_0417_gen_0_1020_03.csv",
#     "GNN_data/output/result_feeder_0417_gen_0_1020_04.csv",
#     "GNN_data/output/result_feeder_0417_gen_0_1020_05.csv",
#     "GNN_data/output/result_feeder_0417_gen_0_1020_06.csv",
#     "GNN_data/output/result_feeder_0417_gen_0_1020_08.csv",
#     "GNN_data/output/result_feeder_0417_gen_0_1020_09.csv",
#     "GNN_data/output/result_feeder_0417_gen_0_1020_10.csv",
# ]
CSV_PATHS = [
    "GNN_data/output/result_feeder_0417_gen_0_1101_00.csv",
    "GNN_data/output/result_feeder_0417_gen_0_1101_07.csv",
    "GNN_data/output/result_feeder_0417_gen_0_1101_01.csv",
    "GNN_data/output/result_feeder_0417_gen_0_1101_02.csv",
    "GNN_data/output/result_feeder_0417_gen_0_1101_03.csv",
    "GNN_data/output/result_feeder_0417_gen_0_1101_04.csv",
    "GNN_data/output/result_feeder_0417_gen_0_1101_05.csv",
    "GNN_data/output/result_feeder_0417_gen_0_1101_06_1.csv",
    "GNN_data/output/result_feeder_0417_gen_0_1101_08.csv",
    "GNN_data/output/result_feeder_0417_gen_0_1101_09_1.csv",
    "GNN_data/output/result_feeder_0417_gen_0_1101_10.csv",
]

# CSV_PATHS = [
#     "GNN_data/output/result_feeder_0417_gen_0_1102_00.csv",
#     "GNN_data/output/result_feeder_0417_gen_0_1102_07.csv",
#     "GNN_data/output/result_feeder_0417_gen_0_1102_01.csv",
#     "GNN_data/output/result_feeder_0417_gen_0_1102_02.csv",
#     "GNN_data/output/result_feeder_0417_gen_0_1102_03.csv",
#     "GNN_data/output/result_feeder_0417_gen_0_1102_04.csv",
#     "GNN_data/output/result_feeder_0417_gen_0_1102_05.csv",
#     "GNN_data/output/result_feeder_0417_gen_0_1102_06.csv",
#     "GNN_data/output/result_feeder_0417_gen_0_1102_08.csv",
#     "GNN_data/output/result_feeder_0417_gen_0_1102_09.csv",
#     "GNN_data/output/result_feeder_0417_gen_0_1102_10.csv",
# ]

# CSV_PATHS = [
#     # "GNN_data/output/result_feeder_0417_gen_0_1102_00.csv",
#     # "GNN_data/output/result_feeder_0417_gen_0_1102_07.csv",
#     # "GNN_data/output/result_feeder_0417_gen_0_1102_01.csv",
#     # "GNN_data/output/result_feeder_0417_gen_0_1102_02.csv",
#     # "GNN_data/output/result_feeder_0417_gen_0_1102_03.csv",
#     "GNN_data/output/result_feeder_0417_gen_0_1103_04.csv",
#     # "GNN_data/output/result_feeder_0417_gen_0_1102_05.csv",
#     # "GNN_data/output/result_feeder_0417_gen_0_1102_06.csv",
#     "GNN_data/output/result_feeder_0417_gen_0_1103_08.csv",
#     "GNN_data/output/result_feeder_0417_gen_0_1103_09.csv",
#     "GNN_data/output/result_feeder_0417_gen_0_1103_10.csv",
# ]
# MODEL_NAMES = ['m06','m09','m10','m11']

MODEL_NAMES = ['m01','m02','m03','m04','m05','m06','m07','m08','m09','m10','m11']

OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

# ===== 极端评估的额外配置 =====
TEST_FEATURE_CSV   = "GNN_data/evaluate/evaluate_1102.csv"   # 必填：测试集特征
TRAIN_FEATURE_CSV  = None  # 可选：若提供则阈值来自训练集
AGG_METHOD         = "mean"                               # 可选 "mean" 或 "max"
Q_LIST             = [50, 70, 80]                         # 分层分位数
MODES_TO_EVAL      = ["win", "pre", "union"]              # 可选 "win" "pre" "union" "intersect"
THRESH_SOURCE      = "train_if_available_else_test"       # 阈值来源策略
ONLY_NEG_LABEL_FOR_BIN_METRICS = False                    # 若 True 分层评估时仅用 label=False 的样本

# ================= 工具函数 =================
def read_any_csv(path):
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception:
            pass
    raise RuntimeError(f"无法读取 CSV: {path}")

def normalize_frame(df, model_name):
    """
    标准化列并统一进行标签翻转
    输入列: feeder_name label risk
    翻转规则: label' = 1 - label, risk' = 1 - risk
    """
    cols = {c.lower(): c for c in df.columns}
    need = {"feeder_name", "label", "risk"}
    lower_cols = set(k.lower() for k in df.columns)
    miss = need - lower_cols
    if miss:
        raise ValueError(f"{model_name} 缺少必要列: {miss}")

    g = df.rename(columns={
        cols["feeder_name"]: "feeder_name",
        cols["label"]: "label",
        cols["risk"]: "risk"
    }).copy()

    g = g.drop_duplicates(subset=["feeder_name"], keep="first")

    def to_bool(x):
        if isinstance(x, str):
            t = x.strip().lower()
            if t in ("true", "t", "1", "yes", "y"):
                return True
            if t in ("false", "f", "0", "no", "n"):
                return False
        return bool(x)

    # 原始转为0或1
    label_raw = g["label"].map(to_bool).map(lambda b: 1 if b else 0).astype(int)
    risk_raw  = g["risk"].map(lambda v: 1 if int(bool(v)) == 1 else 0).astype(int)

    # 统一翻转
    g["label"] = (1 - label_raw).astype(int)
    g["risk"]  = (1 - risk_raw).astype(int)

    g["model"] = model_name
    return g[["feeder_name", "label", "risk", "model"]]

def confusion_sets(df_one):
    y = df_one["label"].values
    p = df_one["risk"].values
    name = df_one["feeder_name"].values
    tn = set(name[(y == 0) & (p == 0)])
    tp = set(name[(y == 1) & (p == 1)])
    fn = set(name[(y == 1) & (p == 0)])
    fp = set(name[(y == 0) & (p == 1)])
    return {"TN": tn, "TP": tp, "FN": fn, "FP": fp}

def metrics_from_sets(sets):
    tn, tp, fn, fp = map(len, (sets["TN"], sets["TP"], sets["FN"], sets["FP"]))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    acc  = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return dict(TN=tn, TP=tp, FN=fn, FP=fp,
                Precision=prec, Recall=rec, Specificity=spec,
                Accuracy=acc, F1=f1)

def overlap_matrix(models_sets, cat):
    names = list(models_sets.keys())
    n = len(names)
    mat = np.zeros((n, n), dtype=int)
    jac = np.zeros((n, n), dtype=float)
    for i in range(n):
        Si = models_sets[names[i]][cat]
        for j in range(n):
            Sj = models_sets[names[j]][cat]
            inter = len(Si & Sj)
            union = len(Si | Sj) if (len(Si) or len(Sj)) else 1
            mat[i, j] = inter
            jac[i, j] = inter / union
    mat_df = pd.DataFrame(mat, index=names, columns=names)
    jac_df = pd.DataFrame(jac, index=names, columns=names)
    return mat_df, jac_df

def deltas_against_baseline(base_sets, other_sets, other_name, baseline_name):
    fn2tp = base_sets["FN"] & other_sets["TP"]
    fp2tn = base_sets["FP"] & other_sets["TN"]
    tp2fn = base_sets["TP"] & other_sets["FN"]
    tn2fp = base_sets["TN"] & other_sets["FP"]

    def to_df(change, s):
        return pd.DataFrame({"feeder_name": sorted(list(s))}).assign(change=change)

    out = pd.concat([
        to_df("FN_to_TP", fn2tp),
        to_df("FP_to_TN", fp2tn),
        to_df("TP_to_FN", tp2fn),
        to_df("TN_to_FP", tn2fp),
    ], ignore_index=True)
    out = out.assign(against=other_name, baseline=baseline_name)
    return out, {"FN_to_TP": fn2tp, "FP_to_TN": fp2tn, "TP_to_FN": tp2fn, "TN_to_FP": tn2fp}

# ===== 特征读取与聚合 =====
def normalize_feature_df(df):
    cols = {c.lower(): c for c in df.columns}
    need = {"feeder_name", "win", "pre"}
    lower_cols = set(k.lower() for k in df.columns)
    miss = need - lower_cols
    if miss:
        raise ValueError(f"特征 CSV 缺少必要列: {miss}")
    return df.rename(columns={
        cols["feeder_name"]: "feeder_name",
        cols["win"]: "WIN",
        cols["pre"]: "PRE"
    }).copy()

def aggregate_by_feeder(feat_df, agg="mean"):
    if agg not in ("mean", "max"):
        raise ValueError("AGG_METHOD 需为 'mean' 或 'max'")
    if agg == "mean":
        g = feat_df.groupby("feeder_name", as_index=False).agg({"WIN":"mean", "PRE":"mean"})
    else:
        g = feat_df.groupby("feeder_name", as_index=False).agg({"WIN":"max", "PRE":"max"})
    g = g.rename(columns={"WIN": f"WIN_{agg}", "PRE": f"PRE_{agg}"})
    return g

def compute_thresholds(train_agg, agg="mean", q_list=(90,95,99)):
    w_col = f"WIN_{agg}"
    p_col = f"PRE_{agg}"
    w_thrs = {q: np.percentile(train_agg[w_col].values, q) for q in q_list}
    p_thrs = {q: np.percentile(train_agg[p_col].values, q) for q in q_list}
    return w_thrs, p_thrs

def assign_bins(test_agg, w_thrs, p_thrs, agg="mean", modes=("win","pre","union","intersect"), q_list=(90,95,99)):
    """
    生成每个 feeder 的分层标签
    返回长表 DataFrame 列含 feeder_name mode bin_label in_bin
    bin 三段 [Pq1,Pq2) [Pq2,Pq3) [>=Pq3]
    """
    w_col = f"WIN_{agg}"
    p_col = f"PRE_{agg}"
    df = test_agg.copy()

    q1, q2, q3 = q_list
    w1, w2, w3 = w_thrs[q1], w_thrs[q2], w_thrs[q3]
    p1, p2, p3 = p_thrs[q1], p_thrs[q2], p_thrs[q3]

    out_rows = []
    for _, r in df.iterrows():
        fname = r["feeder_name"]
        wv = r[w_col]; pv = r[p_col]

        masks = {}
        masks["win_P{0}_{1}".format(q1,q2)] = (wv >= w1) and (wv < w2)
        masks["win_P{0}_{1}".format(q2,q3)] = (wv >= w2) and (wv < w3)
        masks["win_P{0}plus".format(q3)]    = (wv >= w3)

        masks["pre_P{0}_{1}".format(q1,q2)] = (pv >= p1) and (pv < p2)
        masks["pre_P{0}_{1}".format(q2,q3)] = (pv >= p2) and (pv < p3)
        masks["pre_P{0}plus".format(q3)]    = (pv >= p3)

        union_masks = {
            "union_P{0}_{1}".format(q1,q2): masks["win_P{0}_{1}".format(q1,q2)] or masks["pre_P{0}_{1}".format(q1,q2)],
            "union_P{0}_{1}".format(q2,q3): masks["win_P{0}_{1}".format(q2,q3)] or masks["pre_P{0}_{1}".format(q2,q3)],
            "union_P{0}plus".format(q3):    masks["win_P{0}plus".format(q3)]    or masks["pre_P{0}plus".format(q3)],
        }
        intersect_masks = {
            "intersect_P{0}_{1}".format(q1,q2): masks["win_P{0}_{1}".format(q1,q2)] and masks["pre_P{0}_{1}".format(q1,q2)],
            "intersect_P{0}_{1}".format(q2,q3): masks["win_P{0}_{1}".format(q2,q3)] and masks["pre_P{0}_{1}".format(q2,q3)],
            "intersect_P{0}plus".format(q3):    masks["win_P{0}plus".format(q3)]    and masks["pre_P{0}plus".format(q3)],
        }

        pack = {}
        for k,v in masks.items():
            mode = "win" if k.startswith("win_") else "pre"
            if mode in modes:
                pack.setdefault(mode, {})[k.split("_",1)[1]] = v
        if "union" in modes:
            pack["union"] = {k.split("_",1)[1]: v for k,v in union_masks.items()}
        if "intersect" in modes:
            pack["intersect"] = {k.split("_",1)[1]: v for k,v in intersect_masks.items()}

        for mode, bins in pack.items():
            for bin_label, hit in bins.items():
                out_rows.append({"feeder_name": fname, "mode": mode, "bin_label": bin_label, "in_bin": bool(hit)})

    return pd.DataFrame(out_rows)

def metrics_on_subset(df_pred, mask_names, label_filter_value=None):
    """
    df_pred 包含 feeder_name label risk
    mask_names 参与评估的 feeder_name 集合
    label_filter_value 若为 True 或 False 则在评估前先按该标签筛选
    """
    if len(mask_names) == 0:
        return dict(Accuracy=np.nan, Precision=np.nan, Recall=np.nan, Specificity=np.nan, F1=np.nan, Support=0)

    sub = df_pred[df_pred["feeder_name"].isin(mask_names)].copy()
    if label_filter_value is not None:
        sub = sub[sub["label"] == bool(label_filter_value)].copy()

    if len(sub) == 0:
        return dict(Accuracy=np.nan, Precision=np.nan, Recall=np.nan, Specificity=np.nan, F1=np.nan, Support=0)

    sets = confusion_sets(sub)
    m = metrics_from_sets(sets)
    m["Support"] = len(sub)
    return m

def sanitize_col(s: str) -> str:
    return s.replace("+", "plus").replace("-", "_").replace("/", "_").replace(" ", "_")

# ================= 主流程 =================
if __name__ == "__main__":
    # 读取模型输出并标准化
    if MODEL_NAMES is None:
        MODEL_NAMES = [f"model{i+1}" for i in range(len(CSV_PATHS))]
    assert len(MODEL_NAMES) == len(CSV_PATHS), "MODEL_NAMES 与 CSV_PATHS 长度需一致"

    dfs = []
    for path, name in zip(CSV_PATHS, MODEL_NAMES):
        df = read_any_csv(path)
        dfs.append(normalize_frame(df, name))  # 此处已统一翻转 label 与 risk

    # 对齐 只保留所有模型共同覆盖的 feeder
    feeders_intersection = set(dfs[0]["feeder_name"])
    for df in dfs[1:]:
        feeders_intersection &= set(df["feeder_name"])
    if not feeders_intersection:
        raise RuntimeError("各模型的 feeder_name 没有交集 请检查数据")

    dfs = [df[df["feeder_name"].isin(feeders_intersection)].copy() for df in dfs]

    # 逐模型计算混淆集合与整体指标
    models_sets = {}
    metrics_rows = []
    for df in dfs:
        name = df["model"].iloc[0]
        s = confusion_sets(df)
        models_sets[name] = s
        m = metrics_from_sets(s)
        m["model"] = name
        metrics_rows.append(m)

    summary_df = pd.DataFrame(metrics_rows)[
        ["model", "TN", "TP", "FN", "FP", "Precision", "Recall", "Specificity", "Accuracy", "F1"]
    ].sort_values("model")

    # ============== 极端样本分层评估 ==============
    # 读取测试特征并聚合
    test_feat_raw = read_any_csv(TEST_FEATURE_CSV)
    test_feat = normalize_feature_df(test_feat_raw)
    test_agg = aggregate_by_feeder(test_feat, agg=AGG_METHOD)

    # 若阈值来自训练集则读取并聚合 否则使用测试集自身
    use_train = False
    if THRESH_SOURCE.lower().startswith("train") and TRAIN_FEATURE_CSV:
        try:
            train_feat_raw = read_any_csv(TRAIN_FEATURE_CSV)
            train_feat = normalize_feature_df(train_feat_raw)
            train_agg = aggregate_by_feeder(train_feat, agg=AGG_METHOD)
            use_train = True
        except Exception:
            use_train = False

    if not use_train:
        train_agg = test_agg.copy()

    # 仅保留交集 feeder
    test_agg = test_agg[test_agg["feeder_name"].isin(feeders_intersection)].reset_index(drop=True)
    train_agg = train_agg[train_agg["feeder_name"].isin(feeders_intersection)].reset_index(drop=True)

    # 计算阈值与分层
    w_thrs, p_thrs = compute_thresholds(train_agg, agg=AGG_METHOD, q_list=Q_LIST)
    bin_long = assign_bins(test_agg, w_thrs, p_thrs, agg=AGG_METHOD, modes=MODES_TO_EVAL, q_list=Q_LIST)

    # 生成分层子集
    subset_dict = {}
    for (mode, bin_label), g in bin_long.groupby(["mode", "bin_label"]):
        subset = set(g.loc[g["in_bin"] == True, "feeder_name"].astype(str))
        subset_dict[(mode, bin_label)] = subset

    # 分层评估明细表
    rows = []
    for df in dfs:
        model = df["model"].iloc[0]
        for key, subset in subset_dict.items():
            mode, bin_label = key
            label_filter = False if ONLY_NEG_LABEL_FOR_BIN_METRICS else None
            m = metrics_on_subset(df, subset, label_filter_value=label_filter)
            row = dict(model=model, mode=mode, bin_label=bin_label,
                       Accuracy=m["Accuracy"], Precision=m["Precision"], Recall=m["Recall"],
                       Specificity=m["Specificity"], F1=m["F1"], Support=m["Support"])
            row["WIN_q"] = ",".join(str(q) for q in Q_LIST)
            row["PRE_q"] = ",".join(str(q) for q in Q_LIST)
            row["Agg"]   = AGG_METHOD
            row["ThreshFrom"] = "train" if use_train else "test"
            row["OnlyNegForEval"] = bool(ONLY_NEG_LABEL_FOR_BIN_METRICS)
            rows.append(row)

    extreme_df = pd.DataFrame(rows).sort_values(by=["mode","bin_label","model"])
    extreme_df.to_csv(os.path.join(OUT_DIR, "extreme_accuracy_by_bin.csv"), index=False, encoding="utf-8-sig")

    # 将分区准确率并入 summary_metrics.csv 为每个分区生成一列 ACC_<mode>_<bin>
    acc_pivot = extreme_df.pivot_table(index="model", columns=["mode", "bin_label"], values="Accuracy")
    if acc_pivot is not None and acc_pivot.shape[1] > 0:
        acc_pivot.columns = [sanitize_col(f"ACC_{m}_{b}") for m, b in acc_pivot.columns]
        acc_pivot = acc_pivot.reset_index()
        summary_df = summary_df.merge(acc_pivot, on="model", how="left")

    # 输出最终 summary
    summary_df.to_csv(os.path.join(OUT_DIR, "summary_metrics.csv"), index=False, encoding="utf-8-sig")

    # 各类别的重合矩阵
    for cat in ("TN", "TP", "FN", "FP"):
        mat_df, jac_df = overlap_matrix(models_sets, cat)
        mat_df.to_csv(os.path.join(OUT_DIR, f"overlap_{cat}.csv"), encoding="utf-8-sig")
        jac_df.to_csv(os.path.join(OUT_DIR, f"jaccard_{cat}.csv"), encoding="utf-8-sig")

    # 以第一个模型为基线 生成改进与回退清单
    baseline_name = dfs[0]["model"].iloc[0]
    base_sets = models_sets[baseline_name]
    for k in range(1, len(dfs)):
        other_name = dfs[k]["model"].iloc[0]
        other_sets = models_sets[other_name]
        delta_df, _ = deltas_against_baseline(base_sets, other_sets, other_name, baseline_name)
        delta_df.to_csv(os.path.join(OUT_DIR, f"deltas_{other_name}_vs_{baseline_name}.csv"),
                        index=False, encoding="utf-8-sig")

    print(f"\n[Done] 结果已输出到文件夹: {OUT_DIR}")

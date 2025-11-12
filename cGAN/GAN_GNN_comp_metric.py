# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import entropy
from scipy.linalg import sqrtm
from sklearn.metrics.pairwise import rbf_kernel, pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.utils.validation import check_array
import matplotlib.pyplot as plt
import sys

# ========================= 参数设置 =========================
model_files = {
    'MLP': 'single_node_gen_0402.csv',
    'CNN': 'multi_node_gen_0402.csv',
    'Graph':'graph_gen_0402.csv',
    'sage': 'cgan_gen_0402.csv', #cgan_gen_0402.csv'
    'gat':    'cGAN_gen_1026_gat_1.csv',
    'gin': 'cGAN_gen_1026_gin_1.csv',
    'transformer': 'cGAN_gen_1026_transformer_1.csv',
    'concat_mlp':'cGAN_gen_1029_concat_mlp.csv',
    'gated':'cGAN_gen_1029_gated.csv',
    'hadamard':'cGAN_gen_1029_hadamard_concat.csv',
    'cross_stitch':'cGAN_gen_1029_cross_stitch.csv',
    'film':'cGAN_gen_1029_film.csv',
    'SMOET':'SMOTETomek_gen_extreme.csv',
    'Real sample':  'train_0310_2_gen_clustered.csv', #train_0310_2_gen_clustered.csv
}
data_dir        = 'GAN_data/gen_data/'
feature_columns = ['PRE_extreme', 'WIN_extreme', 'PRS_extreme', 'TMP_extreme', 'SHU_extreme']
output_csv      = 'GAN_data/generation_model_metrics_summary_GNN_comp.csv'

FEEDER_COL      = 'feeder_name'  # 按馈线分组所用列

# ========= 可选表征方式开关 =========
EMBED_MODE = "pca"   # "raw" | "pca" | "disc"
PCA_DIM    = 5
TSNE_PERP  = 30
TSNE_SEED  = 42
TSNE_FIG   = "GAN_data/tsne_models.png"

# ========================= 工具 & 校验 =========================
def assert_numeric_df(df: pd.DataFrame, feats, label: str):
    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise KeyError(f"[{label}] 缺少列: {missing}，现有列: {list(df.columns)}")
    offenders = [c for c in feats if not np.issubdtype(df[c].dtype, np.number)]
    if offenders:
        examples = {c: df[c].head(3).tolist() for c in offenders}
        print(f"[WARN] {label} 存在非数值列: {offenders}，示例值: {examples}")
        try:
            df[offenders] = df[offenders].apply(pd.to_numeric, errors='raise')
            print(f"[INFO] {label} 非数值列已成功强制转为数值。")
        except Exception as e:
            raise TypeError(f"[{label}] 特征列中存在非数值，且无法转为数值: {offenders}。错误: {e}")
    values = df[feats].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        bad = np.argwhere(~np.isfinite(values))
        sample_bad = values[bad[:5,0], :][:5]
        raise ValueError(f"[{label}] 特征中包含 NaN/Inf。坏行前5: {bad[:5,0].ravel().tolist()}，示例: {sample_bad}")
    return df

def coerce_array(name, arr):
    if hasattr(arr, "values"):
        arr = arr.values
    elif isinstance(arr, (list, tuple)):
        arr = arr[0] if len(arr) > 0 else np.empty((0, 0), dtype=np.float64)
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    try:
        arr = arr.astype(np.float64, copy=False)
    except Exception as e:
        raise TypeError(f"[{name}] 嵌入无法转为 float64；dtype={arr.dtype}，示例={arr.ravel()[:5]!r}；错误：{e}")
    mask = np.isfinite(arr).all(axis=1)
    if mask.sum() < arr.shape[0]:
        print(f"[WARN] {name} 中 {(~mask).sum()} 条样本含 NaN/Inf，已丢弃。")
        arr = arr[mask]
    if arr.size == 0:
        raise ValueError(f"[{name}] 嵌入在清洗后为空。")
    return arr

# ========================= 指标函数 =========================
def compute_mmd(x, y, gamma=10):
    K_XX = rbf_kernel(x, x, gamma=gamma)
    K_YY = rbf_kernel(y, y, gamma=gamma)
    K_XY = rbf_kernel(x, y, gamma=gamma)
    return K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()

def compute_kl_1d(p, q, bins=50, range_min=0, range_max=1):
    p_hist, _ = np.histogram(p, bins=bins, range=(range_min, range_max), density=True)
    q_hist, _ = np.histogram(q, bins=bins, range=(range_min, range_max), density=True)
    p_hist += 1e-8; q_hist += 1e-8
    return entropy(p_hist, q_hist)

def compute_wasserstein_by_feeder(real_df, fake_df, features, feeder_col='feeder_name'):
    real = real_df.copy(); fake = fake_df.copy()
    real['feeder_key'] = real[feeder_col].astype(str).str[:-5]
    fake['feeder_key'] = fake[feeder_col].astype(str).str[:-5]
    real_grouped = real.groupby('feeder_key')
    fake_grouped = fake.groupby('feeder_key')
    common_keys = set(real_grouped.groups.keys()).intersection(fake_grouped.groups.keys())
    distances = []
    for key in common_keys:
        real_group = real_grouped.get_group(key)[features].values
        fake_group = fake_grouped.get_group(key)[features].values
        if real_group.shape[0] == 0 or fake_group.shape[0] == 0:
            continue
        distances.append(np.linalg.norm(real_group.mean(axis=0) - fake_group.mean(axis=0)))
    return (np.mean(distances), len(common_keys)) if common_keys else (np.nan, 0)

def compute_coverage(real, fake, radius=0.1):
    dists = cdist(real, fake)
    return (dists < radius).any(axis=1).mean()

def compute_prd(real, fake, radius=0.1):
    dist_rf = cdist(real, fake); dist_fr = cdist(fake, real)
    return (dist_fr < radius).any(axis=1).mean(), (dist_rf < radius).any(axis=1).mean()

# ========================= 嵌入/表征 & FID =========================
def get_disc_embeddings(df: pd.DataFrame) -> np.ndarray:
    raise NotImplementedError("请实现 get_disc_embeddings(df)->np.ndarray")

def _mu_cov(x: np.ndarray):
    return x.mean(axis=0), np.cov(x, rowvar=False)

def compute_fid_features(x_real: np.ndarray, x_fake: np.ndarray, eps=1e-6) -> float:
    mu1, cov1 = _mu_cov(x_real); mu2, cov2 = _mu_cov(x_fake)
    c1 = cov1 + np.eye(cov1.shape[0]) * eps
    c2 = cov2 + np.eye(cov2.shape[0]) * eps
    covmean = sqrtm(c1.dot(c2))
    covmean = covmean.real if np.iscomplexobj(covmean) else covmean
    return float(np.sum((mu1 - mu2) ** 2) + np.trace(c1 + c2 - 2.0 * covmean))

def build_embedder(real_array: np.ndarray, mode: str = "pca"):
    if mode == "raw":
        scaler = StandardScaler().fit(real_array)
        return (lambda x: scaler.transform(x)), "raw(Standardized)"
    if mode == "pca":
        scaler = StandardScaler().fit(real_array)
        x_scaled = scaler.transform(real_array)
        pca = PCA(n_components=min(PCA_DIM, x_scaled.shape[1]), random_state=TSNE_SEED).fit(x_scaled)
        return (lambda x: pca.transform(scaler.transform(x))), f"PCA{pca.n_components_}"
    if mode == "disc":
        return (lambda df: get_disc_embeddings(df)), "DiscriminatorEmb"
    raise ValueError("EMBED_MODE must be one of: raw | pca | disc")

# ===== 新增：按馈线计算 FID 的辅助函数（最小改动接入） =====
def _add_feeder_key(df: pd.DataFrame, feeder_col: str) -> pd.DataFrame:
    if feeder_col not in df.columns:
        raise KeyError(f"未找到馈线列 {feeder_col}")
    out = df.copy()
    out['feeder_key'] = out[feeder_col].astype(str).str[:-5]
    return out

def compute_fid_per_feeder_mean(real_df: pd.DataFrame,
                                fake_df: pd.DataFrame,
                                embedder,
                                emb_mode: str,
                                feats: list,
                                feeder_col: str) -> tuple:
    real_k = _add_feeder_key(real_df, feeder_col)
    fake_k = _add_feeder_key(fake_df, feeder_col)
    gids_r = real_k.groupby('feeder_key')
    gids_f = fake_k.groupby('feeder_key')
    common = sorted(set(gids_r.groups.keys()).intersection(gids_f.groups.keys()))
    fids = []
    for key in common:
        sub_r = gids_r.get_group(key)
        sub_f = gids_f.get_group(key)
        # 至少需要2个样本以稳定协方差
        if len(sub_r) < 2 or len(sub_f) < 2:
            continue
        if emb_mode == "disc":
            emb_r = embedder(sub_r)
            emb_f = embedder(sub_f)
        else:
            emb_r = embedder(sub_r[feats].to_numpy(dtype=float))
            emb_f = embedder(sub_f[feats].to_numpy(dtype=float))
        emb_r = coerce_array("emb_real_feeder", emb_r)
        emb_f = coerce_array("emb_fake_feeder", emb_f)
        try:
            fid_val = compute_fid_features(emb_r, emb_f)
            if np.isfinite(fid_val):
                fids.append(fid_val)
        except Exception as e:
            print(f"[WARN] 按馈线FID失败: key={key}, err={e}")
            continue
    if len(fids) == 0:
        return np.nan, 0
    return float(np.mean(fids)), len(fids)

# ========================= 主流程 =========================
def main():
    print("📊 正在评估生成模型效果（含 KL/MMD/PRD/Coverage/按馈线W 距离 + FID + t-SNE）...")

    # 1) 读数据 + 数值校验
    df_dict, data_dict = {}, {}
    for label, file in model_files.items():
        path = os.path.join(data_dir, file)
        if not os.path.exists(path):
            raise FileNotFoundError(f"[{label}] 路径不存在: {path}")
        df = pd.read_csv(path, encoding='gbk')
        if df.empty:
            raise ValueError(f"[{label}] 读入 DataFrame 为空：{path}")
        df = df.dropna(subset=feature_columns).copy()
        df = assert_numeric_df(df, feature_columns, label=label)
        df_dict[label]   = df
        data_dict[label] = df[feature_columns].to_numpy(dtype=float)
        print(f"[LOAD] {label}: shape={df.shape}, features dtype={df[feature_columns].dtypes.to_dict()}")

    real_data = data_dict['Real sample']
    real_df   = df_dict['Real sample']

    # 2) 基本指标
    results = []
    for model, fake_data in data_dict.items():
        if model == 'Real sample':
            continue
        fake_df = df_dict[model]
        if real_data.shape[1] != fake_data.shape[1]:
            raise ValueError(f"[{model}] 特征维度不一致：real={real_data.shape[1]} vs fake={fake_data.shape[1]}")
        mmd = compute_mmd(real_data, fake_data, gamma=10)
        wasser, feeder_count = compute_wasserstein_by_feeder(real_df, fake_df, feature_columns)
        cov = compute_coverage(real_data, fake_data)
        precision, recall = compute_prd(real_data, fake_data)
        kl_dims, kl_values = {}, []
        for i, feat in enumerate(feature_columns):
            kl_i = compute_kl_1d(real_data[:, i], fake_data[:, i])
            kl_dims[f'KL_{feat}'] = kl_i; kl_values.append(kl_i)
        results.append({
            'Model': model,
            'MMD': mmd,
            'Wasserstein Distance': wasser,
            'Matched Feeders': feeder_count,
            # 'Coverage Score': cov,
            'PRD Precision': precision,
            'PRD Recall': recall,
            'KL_Mean': float(np.mean(kl_values)),
            **kl_dims
        })
        print(f"✅ 完成评估（基本指标）：{model}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False, encoding='gbk')
    print(f"✅ 基本指标已保存到 {output_csv}")
    print(results_df)

    # 3) 构建表征器 + FID
    embedder, emb_name = build_embedder(real_data, mode=EMBED_MODE)

    def to_embed(label):
        if EMBED_MODE == "disc":
            arr = embedder(df_dict[label])
        else:
            arr = embedder(data_dict[label])
        if isinstance(arr, (list, tuple)):
            arr = arr[0]
        arr = np.asarray(arr)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        try:
            arr = arr.astype(np.float64, copy=False)
        except Exception as e:
            raise TypeError(f"[{label}] 嵌入无法转 float64（{emb_name}）：dtype={arr.dtype}, e={e}")
        if not np.isfinite(arr).all():
            raise ValueError(f"[{label}] 嵌入包含 NaN/Inf（{emb_name}）。")
        return arr

    embeds_for_tsne = {'Real sample': to_embed('Real sample')}
    fid_rows = []
    for model in data_dict.keys():
        if model == 'Real sample':
            continue
        emb_real = embeds_for_tsne['Real sample']
        emb_fake = to_embed(model)
        embeds_for_tsne[model] = emb_fake
        try:
            fid_global = compute_fid_features(emb_real, emb_fake)
        except Exception as e:
            fid_global = np.nan
            print(f"[WARN] FID 计算失败（{model}, {emb_name}）：{e}")

        # ===== 新增：按馈线计算并求均值 =====
        fid_mean_by_feeder, feeder_used = compute_fid_per_feeder_mean(
            real_df=df_dict['Real sample'],
            fake_df=df_dict[model],
            embedder=embedder,
            emb_mode=EMBED_MODE,
            feats=feature_columns,
            feeder_col=FEEDER_COL
        )

        fid_rows.append({
            'Model': model,
            'Embedding': emb_name,
            'FID': fid_global,
            'FID_byFeeder_Mean': fid_mean_by_feeder,
            'FID_byFeeder_Count': feeder_used
        })
        print(f"✅ 完成评估：{model}（FID on {emb_name}；按馈线均值={fid_mean_by_feeder}，馈线数={feeder_used}）")

    out_df = results_df.merge(pd.DataFrame(fid_rows), on='Model', how='left')
    out_df.to_csv(output_csv, index=False, encoding='gbk')
    print(f"✅ 含 FID 的指标已回写 {output_csv}")
    print(out_df.to_string(index=False, max_rows=None, max_cols=None))


if __name__ == '__main__':
    main()

# GNN01_apply_dataset.py
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import to_undirected
from tqdm import tqdm
import pandas as pd
import pickle5 as pickle5
from GNN_args import *

class MyBinaryDataset(InMemoryDataset):
    def __init__(self, root, file_name, transform=None, pre_transform=None,
                 oriented_pkl="GNN_data/grid/origin_topology_3_oriented.pickle",
                 edge_feat_csv="GNN_data/train/edge_features_all_feeders.csv"):
        """
        参数
        ----
        root : 数据缓存根目录（PyG 规范）
        file_name : 训练用的节点 CSV（含 feeder_name、pole_id、节点特征、label）
        oriented_pkl : （可选）定向拓扑 pickle；本版不强依赖，仅用于存在性检查（可去掉）
        edge_feat_csv : 边特征 CSV（之前程序导出的 5 个气象均值 + flow），
                        必须包含列：feeder_name, u, v, WIN, PRE, TMP, PRS, SHU, flow
        """
        self.file_name = file_name
        self.oriented_pkl = oriented_pkl
        self.edge_feat_csv = edge_feat_csv
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['ZZ_distribution_system_failure.dataset']

    def download(self):
        pass

    def _read_csv_any(self, path: str) -> pd.DataFrame:
        for enc in ("utf-8", "utf-8-sig", "gbk"):
            try:
                return pd.read_csv(path, encoding=enc, low_memory=False)
            except Exception:
                pass
        return pd.read_csv(path, low_memory=False)

    def process(self):
        # 读取（可选）定向拓扑（仅用于存在性检查/兼容；本流程主要依赖边特征 CSV）
        try:
            with open(self.oriented_pkl, 'rb') as f:
                feeder_graphs = pickle5.load(f)
            graphs_available = isinstance(feeder_graphs, dict)
        except Exception:
            feeder_graphs = {}
            graphs_available = False

        # 节点 CSV（训练表）
        df = self._read_csv_any(self.file_name).fillna(0)

        # === 节点离散化逻辑（保持原有做法） ===
        header_order_for_scale = ['pole_material', 'SHU', 'TMP', 'PRS', 'WIN', 'PRE']
        t = 0
        for h in header_order_for_scale:
            t += 1
            vals = df[h].values
            vals[vals > 1] = 1
            df[h] = vals * 50 + t * 50

        # 输入网络时的列顺序（与你现网一致）
        feat_cols_for_tensor = ['pole_material', 'WIN', 'PRE', 'PRS', 'SHU', 'TMP']

        # 读取边特征 CSV（作为 TransformerConv 的 edge_attr）
        edge_df = self._read_csv_any(self.edge_feat_csv)
        need_edge_cols = ["feeder_name", "u", "v", "WIN", "PRE", "TMP", "PRS", "SHU", "flow"]
        miss = [c for c in need_edge_cols if c not in edge_df.columns]
        if miss:
            raise ValueError(f"edge_feat_csv 缺少必要列: {miss}")

        data_list = []
        missing_in_edgecsv = []   # 边特征缺失的馈线名
        empty_after_align = []    # 对齐节点后无任何有效边的馈线名
        missing_in_pkl = []       # （可选）在 pickle 不存在的馈线（仅提示）

        # 按 feeder_name（原始名称，不截断）处理
        for feeder_name, group in tqdm(df.groupby('feeder_name', sort=False)):
            group = group.copy()
            group['pole_id_str'] = group['pole_id'].astype(str)
            idx_of = {pid: i for i, pid in enumerate(group['pole_id_str'].tolist())}

            # 该馈线的边特征子表（注意：edge_df 里 feeder_name 是“原始名称”，不截断）
            ef_sub = edge_df[edge_df['feeder_name'] == feeder_name].copy()
            if ef_sub.empty:
                print(f"[SKIP] feeder has no edge features in CSV: {feeder_name!r}")
                missing_in_edgecsv.append(feeder_name)
                # （可选提示）在 oriented_pkl 里是否存在
                if graphs_available:
                    key = feeder_name[:-5] if len(feeder_name) > 5 else feeder_name
                    if key not in feeder_graphs:
                        missing_in_pkl.append(feeder_name)
                continue

            # 用“边特征 CSV”驱动构造 edge_index_dir 与 edge_attr（确保同序对齐）
            src_idx, dst_idx, edge_attr_rows = [], [], []
            for _, row in ef_sub.iterrows():
                u = str(row["u"])
                v = str(row["v"])
                if u == v:
                    continue  # 去自弧
                if (u in idx_of) and (v in idx_of):
                    src_idx.append(idx_of[u])
                    dst_idx.append(idx_of[v])
                    edge_attr_rows.append([
                        float(row["WIN"]),
                        float(row["PRE"]),
                        float(row["TMP"]),
                        float(row["PRS"]),
                        float(row["SHU"]),
                        float(row["flow"]),
                    ])
            if not src_idx:
                print(f"[SKIP] feeder has no valid edges after aligning nodes: {feeder_name!r}")
                empty_after_align.append(feeder_name)
                continue

            edge_index_dir = torch.tensor([src_idx, dst_idx], dtype=torch.long)
            edge_index_und = to_undirected(edge_index_dir, num_nodes=len(group))
            edge_attr = torch.tensor(edge_attr_rows, dtype=torch.float)  # [E, 6] → (WIN, PRE, TMP, PRS, SHU, flow)

            # 节点特征 & 标签
            x = torch.LongTensor(group[feat_cols_for_tensor].values)
            y = torch.FloatTensor([float(group.label.values[0])])

            data = Data(
                x=x,
                edge_index_dir=edge_index_dir,
                edge_index_und=edge_index_und,
                edge_attr=edge_attr,   # ← 给 TransformerConv 的 edge_attr
                y=y
            )
            data_list.append(data)

        # 汇总打印（可选）
        if missing_in_edgecsv:
            print(f"[INFO] feeders missing in edge CSV: {len(missing_in_edgecsv)}")
            for name in missing_in_edgecsv[:50]:
                print(" -", name)
            if len(missing_in_edgecsv) > 50:
                print(f"... and {len(missing_in_edgecsv) - 50} more")

        if empty_after_align:
            print(f"[INFO] feeders with zero aligned edges: {len(empty_after_align)}")
            for name in empty_after_align[:50]:
                print(" -", name)
            if len(empty_after_align) > 50:
                print(f"... and {len(empty_after_align) - 50} more")

        if missing_in_pkl:
            print(f"[INFO] feeders missing in oriented pickle (for reference): {len(missing_in_pkl)}")
            for name in missing_in_pkl[:50]:
                print(" -", name)
            if len(missing_in_pkl) > 50:
                print(f"... and {len(missing_in_pkl) - 50} more")

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

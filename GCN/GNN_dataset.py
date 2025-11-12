# GNN01_apply_dataset.py
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import to_undirected
from tqdm import tqdm
import pandas as pd
import pickle5 as pickle5


class MyBinaryDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        file_name,
        transform=None,
        pre_transform=None,
        oriented_pkl="GNN_data/grid/origin_topology_3_oriented.pickle",
        edge_feat_csv="GNN_data/train/edge_features_all_feeders.csv",
        # New configurable options. Defaults keep behavior identical to your current version.
        node_feature_cols=None,          # Node feature column order to build x. (Samples are shown here; please extend according to your actual data schema.)
        edge_feature_cols=None,          # Edge feature column order to build edge_attr. (Samples are shown here; please extend according to your actual data schema.)
        scale_cols_order=None,           # Column order for node feature discretization.
        apply_discretize=True            # Whether to apply discretization to node features.
    ):
        """
        Parameters
        ----------
        root : Root directory for PyG-style caching.
        file_name : Node CSV path (must contain feeder_name, pole_id, node features, and label).
        oriented_pkl : Optional; only used to check existence / provide hints.
        edge_feat_csv : Edge feature CSV; must contain feeder_name, u, v, and the configured edge feature columns.
        node_feature_cols : Node feature column order. Default ['pole_material','WIN','PRE','PRS','SHU','TMP'].
                            (Samples are shown here; please extend according to your actual data schema, 'NDVI', 'Soil', 'Height', 'W_Years')
        edge_feature_cols : Edge feature column order. Default ['WIN','PRE','TMP','PRS','SHU','flow'].
                            (Samples are shown here; please extend according to your actual data schema, 'length', 'W_Years')
        scale_cols_order : Discretization order for node features. Default ['pole_material','SHU','TMP','PRS','WIN','PRE'].
        apply_discretize : Whether to apply discretization. Default True.
        """
        self.file_name = file_name
        self.oriented_pkl = oriented_pkl
        self.edge_feat_csv = edge_feat_csv

        # Defaults aligned with the legacy implementation
        self.node_feature_cols = (
            node_feature_cols
            if node_feature_cols is not None
            else ['pole_material', 'WIN', 'PRE', 'PRS', 'SHU', 'TMP']  # Samples only; extend for your data.
        )
        self.edge_feature_cols = (
            edge_feature_cols
            if edge_feature_cols is not None
            else ['WIN', 'PRE', 'TMP', 'PRS', 'SHU', 'flow']           # Samples only; extend for your data.
        )
        self.scale_cols_order = (
            scale_cols_order
            if scale_cols_order is not None
            else ['pole_material', 'SHU', 'TMP', 'PRS', 'WIN', 'PRE']
        )
        self.apply_discretize = bool(apply_discretize)

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

    def _check_required_cols(self, df: pd.DataFrame, cols, name_hint: str):
        miss = [c for c in cols if c not in df.columns]
        if miss:
            raise ValueError(f"{name_hint} is missing required columns: {miss}")

    def process(self):
        # Optional: oriented topology is only used for existence hints
        try:
            with open(self.oriented_pkl, 'rb') as f:
                feeder_graphs = pickle5.load(f)
            graphs_available = isinstance(feeder_graphs, dict)
        except Exception:
            feeder_graphs = {}
            graphs_available = False

        # Read node CSV
        df = self._read_csv_any(self.file_name).fillna(0)

        # Basic column checks
        self._check_required_cols(
            df,
            ['feeder_name', 'pole_id', 'label'] + self.node_feature_cols,
            "Node CSV"
        )

        # Node discretization logic (kept identical to the original behavior)
        if self.apply_discretize:
            t = 0
            for h in self.scale_cols_order:
                if h not in df.columns:
                    continue
                t += 1
                vals = df[h].values.copy()
                vals[vals > 1] = 1
                df[h] = vals * 50 + t * 50

        # Read edge feature CSV
        edge_df = self._read_csv_any(self.edge_feat_csv)
        base_edge_cols = ["feeder_name", "u", "v"]
        self._check_required_cols(edge_df, base_edge_cols, "Edge feature CSV")
        self._check_required_cols(edge_df, self.edge_feature_cols, "Edge feature CSV")

        data_list = []
        missing_in_edgecsv = []
        empty_after_align = []
        missing_in_pkl = []

        # Group by feeder_name
        for feeder_name, group in tqdm(df.groupby('feeder_name', sort=False)):
            group = group.copy()
            group['pole_id_str'] = group['pole_id'].astype(str)
            idx_of = {pid: i for i, pid in enumerate(group['pole_id_str'].tolist())}

            ef_sub = edge_df[edge_df['feeder_name'] == feeder_name].copy()
            if ef_sub.empty:
                print(f"[SKIP] feeder has no edge features in CSV: {feeder_name!r}")
                missing_in_edgecsv.append(feeder_name)
                if graphs_available:
                    key = feeder_name[:-5] if len(feeder_name) > 5 else feeder_name
                    if key not in feeder_graphs:
                        missing_in_pkl.append(feeder_name)
                continue

            src_idx, dst_idx, edge_attr_rows = [], [], []
            for _, row in ef_sub.iterrows():
                u = str(row["u"])
                v = str(row["v"])
                if u == v:
                    continue
                if (u in idx_of) and (v in idx_of):
                    src_idx.append(idx_of[u])
                    dst_idx.append(idx_of[v])
                    edge_attr_rows.append([float(row[c]) for c in self.edge_feature_cols])

            if not src_idx:
                print(f"[SKIP] feeder has no valid edges after aligning nodes: {feeder_name!r}")
                empty_after_align.append(feeder_name)
                continue

            edge_index_dir = torch.tensor([src_idx, dst_idx], dtype=torch.long)
            edge_index_und = to_undirected(edge_index_dir, num_nodes=len(group))
            edge_attr = torch.tensor(edge_attr_rows, dtype=torch.float)

            # Node features and labels (kept as LongTensor to match your embedding pipeline)
            x = torch.LongTensor(group[self.node_feature_cols].values)
            y = torch.FloatTensor([float(group.label.values[0])])

            data = Data(
                x=x,
                edge_index_dir=edge_index_dir,
                edge_index_und=edge_index_und,
                edge_attr=edge_attr,
                y=y
            )
            data_list.append(data)

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

        if hasattr(locals(), "missing_in_pkl") and missing_in_pkl:
            print(f"[INFO] feeders missing in oriented pickle (for reference): {len(missing_in_pkl)}")
            for name in missing_in_pkl[:50]:
                print(" -", name)
            if len(missing_in_pkl) > 50:
                print(f"... and {len(missing_in_pkl) - 50} more")

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

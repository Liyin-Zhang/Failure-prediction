# Centralized configuration for training. Only modify values here.

from types import SimpleNamespace

def get_args():
    ARGS = SimpleNamespace(
        # Reproducibility
        seed=12101,
        device='auto',

        # Training settings
        epochs=10,
        batch_size=256,
        learning_rate=1e-4,
        step_size=200,
        gamma=0.8,

        # Consistency regularization
        lambda_consistency=0.0,

        # Data paths for training
        pyg_root="GNN_data/train",
        train_csv="GNN_data/train/train_node_features_subset_obf.csv",
        edge_feat_csv="GNN_data/train/train_edge_features_subset_obf.csv",
        oriented_pkl="GNN_data/grid/train_topology_oriented_subset_obf.pickle",

        # Evaluation paths
        evaluate_root="GNN_data/evaluate",
        evaluate_csv="GNN_data/evaluate/evaluate_node_features_subset_obf.csv",
        evaluate_edge_feat_csv="GNN_data/evaluate/evaluate_edge_features_subset_obf.csv",
        evaluate_oriented_pkl="GNN_data/grid/evaluate_topology_oriented_subset_obf.pickle",
        model_ckpt_path="GNN_data/model/GNN_06.pkl",
        output_csv="GNN_data/output/resul.csv",
        threshold=0.5,  # decision threshold for evaluation

        # Configurable node and edge feature column order
        # Example uses a subset of features. You can extend with fields like 'NDVI', 'Soil', 'Height', 'W_Years'
        node_feature_cols=['pole_material', 'WIN', 'PRE', 'PRS', 'SHU', 'TMP'],
        edge_feature_cols=['WIN', 'PRE', 'TMP', 'PRS', 'SHU', 'flow'],  # You can extend with fields like 'length', 'W_Years'

        # Node discretization settings
        # Example uses a subset of features. You can extend with fields like 'NDVI', 'Soil', 'Height', 'W_Years'
        scale_cols_order=['pole_material', 'SHU', 'TMP', 'PRS', 'WIN', 'PRE'],
        apply_discretize=True,

        # DataLoader settings
        num_workers=0,
        pin_memory=True,
        clean_pyg_cache=True,

        # Save and checkpoint
        checkpoint_every=200,
        checkpoint_path="GNN_data/model/checkpoint.pkl",
        final_model_path="GNN_data/model/GNN_train_new.pkl",
        train_loss_path="GNN_data/model/train_loss_new.pickle",

        # Three hyperparameters used by the network
        # max_node_number is used as num_embeddings for the node embedding
        # embed_dim is the embedding dimension for one discrete feature
        # feature_num equals the number of discrete features per node
        max_node_number=1000,
        embed_dim=24,
        feature_num=6,
    )
    return ARGS

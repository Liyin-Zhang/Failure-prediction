# How to Use This Project

This guide explains the minimal steps to configure, train, and evaluate the models in this repository. All configuration happens in one place so you can keep experiments consistent and reproducible.

============================================================

## 1) Configure parameters in GNN_args.py

Open GNN_args.py and edit the following fields

* Reproducibility
  seed
  device  set to "auto" to prefer CUDA when available

* Training setup
  epochs
  batch_size
  learning_rate
  step_size
  gamma

* Regularization
  lambda_consistency

* Training data paths
  pyg_root
  train_csv
  edge_feat_csv
  oriented_pkl

* Evaluation data paths
  evaluate_root
  evaluate_csv
  evaluate_edge_feat_csv
  evaluate_oriented_pkl
  model_ckpt_path
  output_csv
  threshold

* Feature definitions  examples only, extend to match your dataset
  node_feature_cols
  edge_feature_cols
  scale_cols_order
  apply_discretize

* DataLoader
  num_workers
  pin_memory
  clean_pyg_cache

* Saving
  checkpoint_every
  checkpoint_path
  final_model_path
  train_loss_path

* Network hyperparameters
  max_node_number  used by the embedding layer
  embed_dim        dimension for each discrete feature
  feature_num      number of discrete node feature columns

Note
These feature lists are samples. Replace or extend them to match your real columns.

============================================================

## 2) Train

Run the training script to start training. The model checkpoint and loss curve will be saved under GNN_data/model

Example
    python train.py

Outputs
* Final checkpoint saved to ARGS.final_model_path
* Periodic checkpoints saved every ARGS.checkpoint_every epochs
* Training loss history saved to ARGS.train_loss_path

Backbone reminder
Make sure the imported backbone in train.py matches the network you want to train. For example
    from GNN_networks import GNN_network_06 as net
    from GNN_networks.GNN_network_06 import NetA_NodeOnly

============================================================

## 3) Evaluate

Run the evaluation script to score a saved checkpoint. Feeder level results will be written to GNN_data/output

Example
    python evaluate.py

Before running evaluation ensure
* ARGS.model_ckpt_path points to the trained checkpoint
* ARGS.evaluate_csv and ARGS.evaluate_edge_feat_csv point to the evaluation CSVs
* ARGS.evaluate_oriented_pkl points to the oriented topology pickle
* ARGS.threshold is the decision threshold for 0 or 1

Backbone reminder
The backbone imported in evaluate.py must match the network that produced the checkpoint. For example
    from GNN_networks import GNN_network_06 as net
    from GNN_networks.GNN_network_06 import NetA_NodeOnly

Outputs
* Feeder level CSV written to ARGS.output_csv
* Console prints AUC precision recall specificity accuracy F1
* Confusion matrix and ROC figures are displayed or can be saved if you call plt.savefig

============================================================

## 4) Notes on model and backbone pairing

One checkpoint pairs with one backbone. If they do not match the state dict load will fail or outputs will be incorrect. When you switch ARGS.model_ckpt_path also switch the backbone import in evaluate.py to the corresponding GNN_network_xx module.

Project specific mapping
* model9 uses network09
* model11 uses network06

============================================================

## 5) Typical folder layout

GNN_data
  train
    train_node_features_subset_obf.csv
    train_edge_features_subset_obf.csv
  evaluate
    evaluate_node_features_subset_obf.csv
    evaluate_edge_features_subset_obf.csv
  grid
    train_topology_oriented_subset_obf.pickle
    evaluate_topology_oriented_subset_obf.pickle
  model
    checkpoints and final models
  output
    evaluation results

============================================================

## 6) Repro tips

* Set ARGS.seed for all runs
* Keep ARGS.node_feature_cols and ARGS.edge_feature_cols stable across train and evaluate
* Clean PyG cache by toggling ARGS.clean_pyg_cache when you change datasets or schemas

That is all you need to run the full pipeline


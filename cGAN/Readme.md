Graph based learning and conditional generation for feeder level fault risk under extreme weather
This repository contains a GNN conditioned cGAN that learns on feeder graphs and synthesizes rare extreme weather feature samples to improve prediction robustness

## Key components

1. GNN based predictor and feature encoders that respect feeder topology
2. Conditional GAN based generator that produces extreme weather node features from normal weather conditions
3. Unified args system for train and evaluation with reproducible seeds
4. Visual utilities for quick 2D previews during training

## Project structure

* GAN_dataset.py                # build PyG graph list from node CSV and topology pickle
* GAN_args.py                   # argument parsers for train and eval
* GAN_train_CGAN.py             # training entry with args
* GAN_evaluate_CGAN.py          # evaluation and sample generation with args
* GAN_networks/                 # model backbones
  * GAN_network_cgan_SAGE.py    # SAGE based G and D
  * ...                         # other GNN variants, for example GAT GIN TransformerConv
* GAN_data/                     # data and outputs
  * train/                      # training CSV and topology pickle
  * model/                      # saved weights and logs
  * gen_data/                   # generated samples
* visualizer.py                 # simple scatter visualizer

## Data inputs

* Node CSV with columns
  feeder_name, label, and paired features
  WIN_normal, PRE_normal, PRS_normal, SHU_normal, TMP_normal
  WIN_extreme, PRE_extreme, PRS_extreme, SHU_extreme, TMP_extreme

* Topology pickle
  a Python dict keyed by base feeder name
  each value is a list of tuples (src, dst) for directed edges

The function create_graph(csv_path, pickle_path) loads both files and returns a list of torch_geometric.data.Data objects

## Quick start

### 1. Install

pip install torch torch-geometric pandas numpy matplotlib seaborn
Make sure PyTorch Geometric matches your CUDA and PyTorch versions

## 2. Train

* **Command**
  * `python train_cgan.py`

* **Arguments**
  * `--data_csv`  `GAN_data/train/test_subset_obf.csv`
  * `--topo_pkl`  `GAN_data/train/topology_subset_obf.pickle`
  * `--model_dir` `GAN_data/model`
  * `--model_suffix` `sage`
  * `--epochs` `1000`
  * `--batch_size` `100`
  * `--in_dim` `5`
  * `--hidden_dim` `128`
  * `--lr_g` `5e-4`
  * `--lr_d` `5e-4`

* **Note**
  * During training, the visualizer shows a 2D scatter preview of real and generated projections for quick sanity checks.


## 3. Evaluate and generate samples

* **Command**
  * `python evaluate_cgan.py`

* **Arguments**
  * `--data_csv`   `GAN_data/train/train_0313_1_clustered.csv`
  * `--topo_pkl`   `GAN_data/train/topology_subset_obf.pickle`
  * `--model_path` `GAN_data/model/cGAN_generator_model_sage.pth`
  * `--output_csv` `GAN_data/gen_data/cGAN_gen_sage.csv`
  * `--batch_size` `1`
  * `--in_dim` `5`
  * `--hidden_dim` `128`
  * `--feature_cols` `WIN_extreme,PRE_extreme,PRS_extreme,SHU_extreme,TMP_extreme`

* **Output**
  * Writes a CSV where extreme feature columns are replaced by generated values.
  * Feeder names are duplicated with a “G” mark to indicate generated copies.

## Arguments

All arguments are defined in GAN_args.py

* Train parser get_train_args
  data_csv, topo_pkl, model_dir, model_suffix, vis_title, save_gif, seed, batch_size, in_dim, hidden_dim, lr_g, lr_d, betas, step_size, gamma, lambda_gp, n_critic, epochs, dump_args

* Eval parser get_eval_args
  data_csv, topo_pkl, model_path, output_csv, batch_size, seed, in_dim, hidden_dim, feature_cols, dump_args

You can export the effective configuration to JSON using --dump_args path.json

## Models

* Generator
  GNN backbone on concatenated [normal features, noise] followed by an MLP head to output extreme features at node level

* Discriminator
  Two branch design with a structural GNN branch and an MLP joint feature branch, fused then pooled to graph level for real or fake decision

Model variants are provided using SAGE, GAT, GIN, and TransformerConv backbones

## Reproducibility

Set seeds with --seed in args
CUDNN deterministic flags are enabled in both train and eval

## Data availability

Original utility data is confidential
Deidentified samples and minimal runnable examples are included for reproduction and validation


## License

This project is for academic use For other uses please contact the author


import argparse
import json
import os

# -------- shared helpers --------
def _finalize_args(args):
    if hasattr(args, "feature_cols") and isinstance(args.feature_cols, str):
        args.feature_cols = [c.strip() for c in args.feature_cols.split(",") if c.strip()]
    if getattr(args, "dump_args", ""):
        os.makedirs(os.path.dirname(args.dump_args) or ".", exist_ok=True)
        with open(args.dump_args, "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2, ensure_ascii=False)
    return args

# -------- train args --------
def get_train_args(argv=None):
    p = argparse.ArgumentParser(description="GRAPH cGAN training")

    # data and io
    p.add_argument("--data_csv",     type=str, default="GAN_data/train/test_subset_obf.csv")
    p.add_argument("--topo_pkl",     type=str, default="GAN_data/train/topology_subset_obf.pickle")
    p.add_argument("--model_dir",    type=str, default="GAN_data/model")
    p.add_argument("--model_suffix", type=str, default="sage")
    p.add_argument("--vis_title",    type=str, default="GAN Graph Visualization of WIN+PRE")
    p.add_argument("--save_gif",     action="store_true")

    # seed and loader
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--batch_size",   type=int, default=400)

    # model
    p.add_argument("--in_dim",       type=int, default=5)
    p.add_argument("--hidden_dim",   type=int, default=128)

    # optim
    p.add_argument("--lr_g",         type=float, default=5e-4)
    p.add_argument("--lr_d",         type=float, default=5e-4)
    p.add_argument("--betas",        type=float, nargs=2, default=(0.5, 0.9), metavar=("B1","B2"))

    # schedulers
    p.add_argument("--step_size",    type=int,   default=150)
    p.add_argument("--gamma",        type=float, default=0.8)

    # gan
    p.add_argument("--lambda_gp",    type=float, default=5.0)
    p.add_argument("--n_critic",     type=int,   default=3)
    p.add_argument("--epochs",       type=int,   default=1000)

    # optional dump
    p.add_argument("--dump_args",    type=str, default="")

    args = p.parse_args(argv)
    return _finalize_args(args)

# -------- eval args --------
def get_eval_args(argv=None):
    p = argparse.ArgumentParser(description="GRAPH cGAN evaluation and generation")

    # data and io for evaluation
    p.add_argument("--data_csv",     type=str, default="GAN_data/evaluate/test_subset_obf.csv")
    p.add_argument("--topo_pkl",     type=str, default="GAN_data/evaluate/topology_subset_obf.pickle")
    p.add_argument("--model_path",   type=str, default="GAN_data/model/cGAN_generator_model_sage.pth")
    p.add_argument("--output_csv",   type=str, default="GAN_data/gen_data/cGAN_gen_sage.csv")

    # loader and seed
    p.add_argument("--batch_size",   type=int, default=1)
    p.add_argument("--seed",         type=int, default=42)

    # model
    p.add_argument("--in_dim",       type=int, default=5)
    p.add_argument("--hidden_dim",   type=int, default=72)

    # columns to overwrite
    p.add_argument("--feature_cols", type=str,
                   default="WIN_extreme,PRE_extreme,PRS_extreme,SHU_extreme,TMP_extreme")

    # optional dump
    p.add_argument("--dump_args",    type=str, default="")

    args = p.parse_args(argv)
    return _finalize_args(args)

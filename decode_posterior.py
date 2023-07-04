import os
import argparse
import yaml
import pickle
from pathlib import Path

import numpy as np
import torch

import utils
from PracticalCoding.decode import decode_irec

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed", type=int, default=22,
        help="manual seed"
    )
    parser.add_argument(
        "--dataset", default="kodak", choices=("cifar", "kodak", ),
        help="dataset selection"
    )
    parser.add_argument(
        "--image_id", type=int, required=True,
        help="kodak image id"
    )
    parser.add_argument(
        "-c", "--config_path", type=Path, required=True,
        help="configuration file"
    )
    args = parser.parse_args()
    return args


def make_args(args, cfgs):
    for key, value in cfgs.items():
        setattr(args, key, value)
    args.total_gpus = torch.cuda.device_count()
    args.log_dir = "log_{}_num{}_emd{}_lat{}_beta{}".format(
        args.dataset, 
        args.num_layers, 
        args.dim_emb, 
        args.dim_hid,
        args.weight_kl
    )
    args.save_dir = "save_{}_num{}_emd{}_lat{}_beta{}".format(
        args.dataset, 
        args.num_layers, 
        args.dim_emb, 
        args.dim_hid,
        args.weight_kl
    )
    return args


def main():
    args = parse_args()
    cfgs = yaml.load(open(args.config_path, "r"), Loader=yaml.FullLoader)
    args = make_args(args, cfgs)
    print(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model_prior = utils.load_model_prior(args)

    # Load parameter groups. How to partition the parameters is synced for the encoder and decoder.
    groups_filename = os.path.join(args.save_dir, "groups.pkl")
    assert os.path.exists(groups_filename), "Parameters should be partitioned into groups first"
    with open(groups_filename, "rb") as f:
        groups = pickle.load(f)

    decode_irec(args, model_prior, groups)

if __name__ == "__main__":
    main()
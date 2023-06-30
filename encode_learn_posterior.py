import os
import argparse
import yaml
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp

import utils
from models import SirenPrior
from run_ddp import TesterDDP
from PracticalCoding.partition import group_from_trainset


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
        "-c", "--config_path", type=Path, required=True,
        help="configuration file"
    )
    parser.add_argument(
        "-p", "--port-offset", type=int, default=0, 
        help="port used for ddp training"
    )

    args = parser.parse_args()
    return args


def make_args(args, cfgs):
    for key, value in cfgs.items():
        setattr(args, key, value)
    args.total_gpus = torch.cuda.device_count()
    args.port = str(29600 + args.port_offset)
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


def main_worker(rank, args, model_prior, groups):
    tester = TesterDDP(rank, args)
    tester.run(model_prior, groups)


def main():
    args = parse_args()
    cfgs = yaml.load(open(args.config_path, "r"), Loader=yaml.FullLoader)
    args = make_args(args, cfgs)
    print(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model_prior = utils.load_model_prior(args)
    for param in model_prior.parameters():
        param.requires_grad = False

    # Partition the parameters into groups
    groups_filename = os.path.join(args.save_dir, "groups.pkl")
    if not os.path.exists(groups_filename):
        groups = group_from_trainset(args, model_prior, groups_filename)
    else:
        with open(groups_filename, "rb") as f:
            groups = pickle.load(f)

    mp.spawn(main_worker, args=(args, model_prior, groups), nprocs=args.total_gpus)


if __name__ == "__main__":
    main()

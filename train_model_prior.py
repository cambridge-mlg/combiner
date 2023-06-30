import os
import argparse
import yaml
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp

import utils
from models import SirenPrior
from run_ddp import TrainerDDP


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
    parser.add_argument(
        "--model_resume", type=str, default=None,
        help="checkpoint path"
    )
    parser.add_argument(
        "--epoch_start", type=int, default=1,
        help="start point for training"
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
    return args


def main_worker(rank, args, model_prior, epoch):
    print("Epoch {%3d}/{%3d}" % (epoch, args.num_epoch), flush=True)
    trainer = TrainerDDP(rank, args)
    trainer.run(model_prior, epoch)


def main():
    args = parse_args()
    cfgs = yaml.load(open(args.config_path, "r"), Loader=yaml.FullLoader)
    args = make_args(args, cfgs)
    print(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    model_prior = SirenPrior(
        dim_emb=args.dim_emb, 
        dim_hid=args.dim_hid, 
        dim_out=3, 
        num_layers=args.num_layers, 
        init_std_scale=args.init_std_scale, 
        w0=args.w0, 
        c=args.c
    )

    if args.model_resume is not None:
        model_prior.load_state_dict(torch.load(args.model_resume))

    for epoch in range(args.epoch_start, args.num_epoch + 1):
        # Expectation Step
        mp.spawn(main_worker, args=(args, model_prior, epoch), nprocs=args.total_gpus)

        # Maximization Step
        with torch.no_grad():
            model_prior = utils.update_prior(args, model_prior) 

        # Save the model prior
        if (epoch) % args.save_interval == 0:
            torch.save(model_prior.state_dict(), args.log_dir + "/model_prior_" + str(epoch) + ".pt")


if __name__ == "__main__":
    main()

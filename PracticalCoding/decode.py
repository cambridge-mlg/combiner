import os
import struct 

import torch

import utils
from models import SirenPosterior
from PracticalCoding.rec import prior_samples


def generate_prior_samples(args):
    n_samples = 2 ** args.kl2_budget
    samples_list = [prior_samples(n_samples, n_variable, args.seed_rec) for n_variable in range(1, args.max_group_size + 1)]
    return samples_list


def decode_rec(args, model_prior, groups):
    bin_file_name = os.path.join(args.save_dir, f"kodim{str(args.image_id).zfill(2)}_bit.bin")
    with open(bin_file_name, 'rb') as f:  
        num_byte = f.read(2)  
        height = struct.unpack('H', num_byte)[0]  
        width = struct.unpack('H', num_byte)[0] 
        print(height, width)

        coordinates = utils.make_coord_grid([height, width], (-1, 1)).view(-1, 2).cuda()
        model = SirenPosterior(
            dim_in=args.dim_in,
            dim_emb=args.dim_emb, 
            dim_hid=args.dim_hid, 
            dim_out=args.dim_out,
            num_layers=args.num_layers, 
            std_init=args.std_init, 
            c=args.c
        )

        model_prior = model_prior.cuda()
        p_mu_list = [model_prior.prior_mu[i] for i in range(args.num_layers)]
        p_std_list = [model_prior.prior_std[i] for i in range(args.num_layers)]
        mask_list = [torch.ones_like(model_prior.prior_mu[i]) for i in range(args.num_layers)]
        yhat_list = [torch.zeros_like(model_prior.prior_mu[i]) for i in range(args.num_layers)]

        samples_list = generate_prior_samples(args)

        for block_id in range(len(groups)):
            block_param_num = len(groups[block_id])
            num_byte = f.read(2)  
            num = struct.unpack('H', num_byte)[0]  

            received_samples = samples_list[block_param_num - 1][num]
            for j in range(block_param_num):
                layer_id, index, _ = groups[block_id][j]
                row, col = index // p_mu_list[layer_id].size(1), index % p_mu_list[layer_id].size(1)
                yhat_list[layer_id][row, col] = received_samples[j]

    x_hat, kld_list = model(coordinates, model_prior, mask_list=mask_list, yhat_list=yhat_list)
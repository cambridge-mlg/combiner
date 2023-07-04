import os
import math
import imageio
import struct
import pickle

import torch
import torch.nn as nn
import torchvision

import utils
from models import SmoothStd, SirenPosterior
from PracticalCoding.partition import adjust_beta_with_mask
from PracticalCoding.irec import iREC

def load_single_image(args):
    transform = torchvision.transforms.ToTensor()
    img = imageio.imread(os.path.join(args.test_dir, f"kodim{str(args.image_id).zfill(2)}.png"))
    img = transform(img).float()
    coordinates, features = utils.to_grid_coordinates_and_features(img)
    return coordinates.cuda(), features.cuda(), img.shape


def build_model(args):
    model = SirenPosterior(
        dim_in=args.dim_in,
        dim_emb=args.dim_emb, 
        dim_hid=args.dim_hid, 
        dim_out=args.dim_out,
        num_layers=args.num_layers, 
        std_init=args.std_init, 
        c=args.c
    )
    model_path = os.path.join(args.save_dir, f"model_test_{args.image_id}.pt")
    model.load_state_dict(torch.load(model_path))
    return model.cuda()


def set_num_tune(block_id):
    if block_id < 5:
        num_tune = 100
    elif block_id < 100:
        num_tune = 50
    elif block_id % 10 == 0:
        num_tune = 30
    else:
        num_tune = 2
    return num_tune


def progressive_encode(args, model_prior, groups):
    coordinates, features, img_shape = load_single_image(args)
    num_pixels = coordinates.shape[0]
    model = build_model(args)

    model_prior = model_prior.cuda()
    p_mu_list = [model_prior.prior_mu[i] for i in range(args.num_layers)]
    p_std_list = [model_prior.prior_std[i] for i in range(args.num_layers)]

    beta_path = os.path.join(args.save_dir, f"model_beta_{args.image_id}.pt")
    with open(beta_path, "rb") as f:
        beta_list = pickle.load(f)
    beta_list = [beta_layer.cuda() for beta_layer in beta_list]
    mask_list = [torch.zeros_like(model_prior.prior_mu[i]) for i in range(args.num_layers)]
    yhat_list = [torch.zeros_like(model_prior.prior_mu[i]) for i in range(args.num_layers)]
    
    bit_total = 0
    bin_file_name = os.path.join(args.save_dir, f"kodim{str(args.image_id).zfill(2)}_bit.bin")
    if os.path.exists(bin_file_name):
        print(f"Warning: remove old binary file in {bin_file_name}")
        os.remove(bin_file_name)
        
    with open(bin_file_name, "wb") as f:
        print(f"Image height: {img_shape[1]}, width: {img_shape[1]}")
        f.write(struct.pack('H', img_shape[1]))
        f.write(struct.pack('H', img_shape[2]))

        print(f"Total block number: {len(groups)}")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        for block_id in range(len(groups)):
            block_param_num = len(groups[block_id])
            with torch.no_grad():
                mask_list, yhat_list, received_index = encode_block(
                    args,
                    model, 
                    groups[block_id], 
                    p_mu_list, 
                    p_std_list,
                    mask_list, 
                    yhat_list, 
                    block_param_num
                )
            bit_total += args.kl2_budget
            f.write(struct.pack('H', received_index))

            num_tune = set_num_tune(block_id)
            for step in range(num_tune):
                optimizer.zero_grad()

                x, kld_list = model(coordinates, model_prior, mask_list=mask_list, yhat_list=yhat_list)
                loss_mse = nn.MSELoss()(x, features)
                loss_kl_beta = sum([(kld_layer * beta_list[i]).sum() for i, kld_layer in enumerate(kld_list)])
                loss_kl = sum([kld_layer.sum() for kld_layer in kld_list])

                loss = loss_mse + loss_kl_beta
                loss.backward()    
                nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)  # replace 1.0 with your desired clip_value

                optimizer.step()
                # if step % 15 == 0:
                #     beta_list = adjust_beta_with_mask(args, groups, beta_list, kld_list, mask_list)

                psnr = utils.get_clamped_psnr(x, features)

            # to ensure each block requires approximately 16 bits.
            if block_id % 10 == 0:
                beta_list = adjust_beta_with_mask(args, groups, beta_list, kld_list, mask_list)

            if block_id % 5 == 0 or block_id == len(groups) - 1:
                print('PSNR: %.3f, Bits_used: %.2f, Blocks: %d' % (psnr, bit_total, block_id + 1))

    print(f"Encoding completed, bits / bpp / PSNR: {bit_total} / {(bit_total /  num_pixels):.4f} / {psnr:.3f}")


def encode_block(args, model, group, p_mu_list, p_std_list, mask_list, yhat_list, block_param_num):
    mu_q_rec = torch.zeros([block_param_num])
    std_q_rec = torch.zeros([block_param_num])
    mu_p_rec = torch.zeros([block_param_num])
    std_p_rec = torch.zeros([block_param_num])

    for j in range(block_param_num):
        layer_id, index, _ = group[j]
        row, col = index // p_mu_list[layer_id].size(1), index % p_mu_list[layer_id].size(1)

        mask_list[layer_id][row, col] = 1
        mu_q_rec[j] = model.net[layer_id].mu.data[row, col]
        std_q_rec[j] = SmoothStd(model.net[layer_id].std.data)[row, col]
        mu_p_rec[j] = p_mu_list[layer_id][row, col]
        std_p_rec[j] = p_std_list[layer_id][row, col]

    received_samples, received_index = iREC(args, mu_q_rec, std_q_rec, mu_p_rec, std_p_rec)
    for j in range(block_param_num):
        layer_id, index, _ = group[j]
        row, col = index // p_mu_list[layer_id].size(1), index % p_mu_list[layer_id].size(1)
        yhat_list[layer_id][row, col] = received_samples[j]
    return mask_list, yhat_list, received_index

import os
import math
import numpy as np
import imageio
import pickle

import torch
import torchvision
from models import SirenPosterior
from utils import to_grid_coordinates_and_features


def group_from_trainset(args, model_prior, groups_filename):
    """ Use the model posteriors w.r.t. training to partition network parameters into groups
    """
    avg_kl2_list = param_average_kld(args, model_prior)
    groups = group_elements_shuffled(avg_kl2_list, kl2_budget=args.kl2_budget, max_group_size=args.max_group_size)
    with open(groups_filename, "wb") as f:
        pickle.dump(groups, f)
    torch.cuda.empty_cache()
    return groups

        
def param_average_kld(args, model_prior):
    transform = torchvision.transforms.ToTensor()
    num_data = args.num_training
    kld_sum = [0 for _ in range(args.num_layers)] 
    for image_id in range(num_data):   
        img = imageio.imread(os.path.join(args.train_dir, f"img{str(image_id)}.png"))
        img = transform(img).float()
        coordinates, features = to_grid_coordinates_and_features(img)

        model = SirenPosterior(
            dim_in=args.dim_in,
            dim_emb=args.dim_emb, 
            dim_hid=args.dim_hid, 
            dim_out=args.dim_out,
            num_layers=args.num_layers, 
            std_init=args.std_init, 
            c=args.c
        )
        model_path = os.path.join(args.log_dir, f"model_siren_{image_id}.pt")
        model.load_state_dict(torch.load(model_path))

        # forward
        predicted, kld_list = model.cuda()(coordinates.cuda(), model_prior.cuda())
        for ind in range(args.num_layers):
            kld_sum[ind] += kld_list[ind]
    
    return [kld_sum[i] / num_data / math.log(2) for i in range(args.num_layers)] 


def group_elements_shuffled(tensor_list, kl2_budget, max_group_size):
    # Flatten and concatenate all 2D tensors into a single 1D tensor
    flat_tensors = [tensor.view(-1) for tensor in tensor_list]
    all_elements = torch.cat(flat_tensors)
    
    all_indices = torch.cat(
        [torch.arange(len(tensor)) for tensor in flat_tensors]
    )
    all_layer_indices = torch.cat(
        [torch.full_like(tensor, layer_id) for layer_id, tensor in enumerate(flat_tensors)]
    )

    # Shuffle the 1D tensor and the corresponding indices
    shuffled_indices = torch.randperm(len(all_elements))
    shuffled_values = all_elements[shuffled_indices]
    shuffled_layer_indices = all_layer_indices[shuffled_indices]
    shuffled_original_indices = all_indices[shuffled_indices]

    groups = []
    current_group = []
    current_sum = 0
    current_count = 0

    for layer_id, idx, element in zip(shuffled_layer_indices, shuffled_original_indices, shuffled_values):
        if current_sum + element > kl2_budget or current_count >= max_group_size:
            groups.append(current_group)
            current_group = []
            current_sum = 0
            current_count = 0
        current_group.append((int(layer_id.item()), idx.item(), element.item()))
        current_sum += element.item()
        current_count += 1

    if current_group:  # Add the last group if it's not empty
        groups.append(current_group)

    return groups

def adjust_beta_with_mask(args, groups, beta_list, kld_list, mask_list=None):
    adjusted_beta_list = [beta.clone() for beta in beta_list]
    group_kl_sums = []
    for group in groups:
        group_kl_sum = sum([kld_list[layer_id].view(-1)[idx].item() / math.log(2) for layer_id, idx, _ in group])
        
        if group_kl_sum > (args.kl2_budget + args.buffer_range):
            for layer_id, idx, _ in group:
                if mask_list is None or mask_list[layer_id].view(-1)[idx] == 0:
                    adjusted_beta_list[layer_id].view(-1)[idx] *= args.adjust_beta_coef
        elif group_kl_sum < (args.kl2_budget - args.buffer_range):
            for layer_id, idx, _ in group:
                if mask_list is None or mask_list[layer_id].view(-1)[idx] == 0:
                    adjusted_beta_list[layer_id].view(-1)[idx] /= args.adjust_beta_coef

        group_kl_sums.append(group_kl_sum)
    return adjusted_beta_list
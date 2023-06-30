import os
import math
import numpy as np
import imageio

import torch
import torchvision
from models import SmoothStd, SirenPosterior, SirenPrior

def bpp(image, model):
    """Computes size in bits per pixel of model.

    Args:
        image (torch.Tensor): Image to be fitted by model.
        model (torch.nn.Module): Model used to fit image.
    """
    num_pixels = np.prod(image.shape) / 3  # Dividing by 3 because of RGB channels
    return model_size_in_bits(model=model) / num_pixels


def psnr(img1, img2):
    """Calculates PSNR between two images.

    Args:
        img1 (torch.Tensor):
        img2 (torch.Tensor):
    """
    return 20. * np.log10(1.) - 10. * (img1 - img2).detach().pow(2).mean().log10().to('cpu').item()


def clamp_image(img):
    """Clamp image values to like in [0, 1] and convert to unsigned int.

    Args:
        img (torch.Tensor):
    """
    # Values may lie outside [0, 1], so clamp input
    img_ = torch.clamp(img, 0., 1.)
    # Pixel values lie in {0, ..., 255}, so round float tensor
    return torch.round(img_ * 255) / 255.


def get_clamped_psnr(img, img_recon):
    """Get PSNR between true image and reconstructed image. As reconstructed
    image comes from output of neural net, ensure that values like in [0, 1] and
    are unsigned ints.

    Args:
        img (torch.Tensor): Ground truth image.
        img_recon (torch.Tensor): Image reconstructed by model.
    """
    return psnr(img, clamp_image(img_recon))


def make_coord_grid(shape, range, device=None):
    """
        Args:
            shape: tuple
            range: [minv, maxv] or [[minv_1, maxv_1], ..., [minv_d, maxv_d]] for each dim
        Returns:
            grid: shape (*shape, )
    """
    l_lst = []
    for i, s in enumerate(shape):
        l = (0.5 + torch.arange(s, device=device)) / s
        if isinstance(range[0], list) or isinstance(range[0], tuple):
            minv, maxv = range[i]
        else:
            minv, maxv = range
        l = minv + (maxv - minv) * l
        l_lst.append(l)
    grid = torch.meshgrid(*l_lst, indexing="ij")
    grid = torch.stack(grid, dim=-1)
    return grid
    

def to_grid_coordinates_and_features(img):
    """Converts an image to a set of coordinates and features.

    Args:
        img (torch.Tensor): Shape (channels, height, width).
    """
    # Coordinates are indices of all non zero locations of a tensor of ones of
    # same shape as spatial dimensions of image

    coordinates = make_coord_grid(img.shape[1:], (-1, 1), device=img.device).view(-1, 2)
    # Convert image to a tensor of features of shape (num_points, channels)
    features = img.reshape(img.shape[0], -1).T
    return coordinates, features


def evaluate_rd(args, model_q_list, model_prior):
    """Evaluate the RD performance from a list of model posteriors and the model prior.

    Args:
        model_q_list
        model_prior
    """
    num_data = args.num_training
    sum_kl, sum_bpp, sum_psnr = 0, 0, 0
    transform = torchvision.transforms.ToTensor()
    for image_id in range(num_data):   
        img = imageio.imread(os.path.join(args.train_dir, f"img{str(image_id)}.png"))
        img = transform(img).float()
        coordinates, features = to_grid_coordinates_and_features(img)

        # forward
        predicted, kld_list = model_q_list[image_id].cuda()(coordinates.cuda(), model_prior.cuda())
        kl = sum([kld_layer.sum() for kld_layer in kld_list])

        psnr = get_clamped_psnr(predicted.cuda(), features.cuda())
        sum_psnr += psnr.item()
        sum_kl += kl.item()
        sum_bpp += kl.item() / math.log(2) / np.prod(img.shape[1:])
    
    avg_kl, avg_bpp, avg_psnr = sum_kl / num_data, sum_bpp / num_data, sum_psnr / num_data
    return avg_kl, avg_bpp, avg_psnr


def analytical_update(args, model_q_list, model_prior):
    num_data = args.num_training
    for layer_id in range(args.num_layers):
        # update mu of model prior
        mu_q = 0
        for image_id in range(num_data):
            mu_q += model_q_list[image_id].net[layer_id].mu.detach()
        model_prior.prior_mu[layer_id].data = mu_q / num_data

        # update sigma of model prior
        mu_sqrdiff = 0
        sigma_q = 0
        for image_id in range(num_data):
            mu_sqrdiff += (model_q_list[image_id].net[layer_id].mu.detach() - model_prior.prior_mu[layer_id].data) ** 2
            sigma_q += SmoothStd(model_q_list[image_id].net[layer_id].std.detach()) ** 2
        model_prior.prior_std[layer_id].data = torch.sqrt((sigma_q + mu_sqrdiff) / num_data)
    return model_prior


def update_prior(args, model_prior):
    print('Updating model priors!', flush=True)
    num_data = args.num_training

    model_q_list = []
    kl_before = 0
    for image_id in range(num_data):            
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
        model_q_list.append(model)

    avg_kl, avg_bpp, avg_psnr = evaluate_rd(args, model_q_list, model_prior)
    torch.cuda.empty_cache()
    print(f"Before Prior Update, KL / BPP / PSNR - {avg_kl:.1f} / {avg_bpp:.5f} / {avg_psnr:.3f}", flush=True)

    analytical_update(args, model_q_list, model_prior)

    avg_kl, avg_bpp, avg_psnr = evaluate_rd(args, model_q_list, model_prior)
    torch.cuda.empty_cache()
    print(f"After Prior Update, KL / BPP / PSNR - {avg_kl:.1f} / {avg_bpp:.5f} / {avg_psnr:.3f}", flush=True)
    return model_prior.cpu()


def load_model_prior(args):
    model_prior = SirenPrior(
        dim_emb=args.dim_emb, 
        dim_hid=args.dim_hid, 
        dim_out=3, 
        num_layers=args.num_layers, 
        init_std_scale=args.init_std_scale, 
        w0=args.w0, 
        c=args.c
    )
    model_prior_path = os.path.join(args.log_dir, f"model_prior_{args.num_epoch}.pt")
    print("Load model prior from ", model_prior_path)
    model_prior.load_state_dict(torch.load(model_prior_path))
    return model_prior
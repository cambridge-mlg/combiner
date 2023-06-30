from math import sqrt, pi
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence


def SmoothStd(std):
    return F.softplus(std, beta=1, threshold=20)


def gaussian_analytical_kl(mu1, mu2, sigma1, sigma2, eps=1e-10):
    sigma1 = sigma1 + eps
    sigma2 = sigma2 + eps
    kl = - 0.5 + torch.log(sigma2) - torch.log(sigma1) + 0.5 * (sigma1 ** 2 + (mu1 - mu2) ** 2) / (sigma2 ** 2)
    return F.relu(kl)


class Sine(nn.Module):
    """Sine activation with scaling.
    Args:
        w0 (float): Omega_0 parameter from Siren paper.
    """
    def __init__(self, w0=1.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)
    

class VarSirenLayer(nn.Module):
    """ variational Bayesian Siren layer.
    Args:
        dim_in: the input dimension of this layer.
        dim_out: the output dimension of this layer.
        std_init: the initializations of std.
        is_first: we empirically adjust the initialization of variance.
        w0 and c are parameters from Siren paper.
    """
    def __init__(
        self, 
        dim_in, 
        dim_out, 
        std_init, 
        is_first,
        w0=30., 
        c=6., 
        activation=None
    ):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

        # initialization in original Siren model.
        w_std = (1 / dim_in) if is_first else (sqrt(c / dim_in) / w0)
        self.w_std = w_std

        # the variance may influence the training stability, so the initialization of mu is empirically adjusted.
        self.mu = nn.Parameter(torch.FloatTensor(dim_in + 1, dim_out).uniform_(- w_std / 12 * 11, w_std / 12 * 11))
        self.std = nn.Parameter(torch.FloatTensor(dim_in + 1, dim_out).fill_(std_init))

        # no activation for the last layer.
        self.activation = Sine(w0) if activation is None else activation

    def forward(self, x, mu_prior, std_prior, mask=None, yhat=None):
        if mask is None:
            mask, yhat = 0, 0
            
        mu = self.mu * (1 - mask) + yhat * mask
        std = SmoothStd(self.std)
        std = std * (1 - mask) + 1e-7 * mask

        mu_w_q, mu_b_q, std_w_q, std_b_q = mu[:-1], mu[-1], std[:-1], std[-1]

        # local reparameterization trick
        act_w_mu = torch.mm(x, mu_w_q)  
        act_w_std = torch.sqrt(torch.mm(x.pow(2), std_w_q.pow(2)) + 1e-14)

        kld_w = gaussian_analytical_kl(mu_w_q, mu_prior[:-1], std_w_q, std_prior[:-1])
        kld_b = gaussian_analytical_kl(mu_b_q, mu_prior[-1], std_b_q, std_prior[-1])        
        eps_w = torch.empty_like(act_w_mu).normal_(0., 1.).to(x.dtype)
        eps_b = torch.empty_like(std_b_q).normal_(0., 1.).to(x.dtype)
        act_w_out = act_w_mu + act_w_std * eps_w  # (batch_size, n_output)
        act_b_out = mu_b_q + std_b_q * eps_b
        out = act_w_out + act_b_out

        out = self.activation(out)
        kld_cat = torch.cat([kld_w, kld_b.unsqueeze(0)], dim=0) * (1 - mask)
        return out, kld_cat


class SirenPosterior(nn.Module):
    """ variational posterior of Siren model.
    Args:
        dim_in: the output dimension (2 on image dataset).
        dim_out: the output dimension (3 on image dataset).
        dim_emb: the input dimension of the siren model after Fourier embedding.
        dim_hid: the hidden unit dimension.
        num_layers: the number of linear layers.
        std_init: we empirically adjust the initialization of variance.
        w0 is a parameter from Siren paper.
    """
    def __init__(
        self, 
        dim_in,
        dim_emb, 
        dim_hid, 
        dim_out, 
        num_layers, 
        std_init, 
        w0=30.,
        c=6.
    ):
        super().__init__()
        self.dim_in = dim_in
        self.dim_emb = dim_emb
        layers = []
        for ind in range(num_layers):
            layers.append(VarSirenLayer(
                dim_in = dim_emb if ind == 0 else dim_hid, 
                dim_out = dim_out if ind == (num_layers - 1) else dim_hid, 
                std_init = std_init,
                is_first = True if ind == 0 else False, 
                w0 = w0, 
                c = c,
                activation = nn.Identity() if ind == num_layers - 1 else None, 
            ))
        self.net = nn.Sequential(*layers)

    def forward(self, x, model_prior, mask_list=None, yhat_list=None, training=True):
        x = self.convert_posenc(x)
        kld_list = []
        for ind, layer in enumerate(self.net):
            x, kld_cat = layer(
                x, 
                mu_prior = model_prior.prior_mu[ind], 
                std_prior = model_prior.prior_std[ind],
                mask = None if mask_list is None else mask_list[ind], 
                yhat = None if yhat_list is None else yhat_list[ind]
            )
            kld_list.append(kld_cat)
        return x, kld_list

    def convert_posenc(self, x):
        assert self.dim_emb % (2 * self.dim_in) == 0, "Embedding size must be the integer multiple of 2 * self.dim_in!"
        w = torch.exp(torch.linspace(0, np.log(1024), self.dim_emb // (2 * self.dim_in), device=x.device))
        x = torch.matmul(x.unsqueeze(-1), w.unsqueeze(0)).view(*x.shape[:-1], -1)
        x = torch.cat([torch.cos(pi * x), torch.sin(pi * x)], dim=-1)
        return x

class SirenPrior(nn.Module):
    """ Siren Model Prior.
    Args:
        dim_emb: the input dimension of the siren model after Fourier embedding.
        dim_hid: the hidden unit dimension.
        dim_out: the output dimension (3 on image dataset).
        num_layers: the number of linear layers.
        init_std_scale: we empirically adjust the initialization of variance.
        w0 and c are parameters from Siren paper.
    """
    def __init__(
        self, 
        dim_emb, 
        dim_hid, 
        dim_out, 
        num_layers, 
        init_std_scale=0.5, 
        w0=30., 
        c=6.
    ):
        super().__init__()
        self.dim_emb = dim_emb
        self.prior_mu = nn.ParameterList()
        self.prior_std = nn.ParameterList()

        for ind in range(num_layers):
            # Empirically set the inialization of model prior according to original SIREN.
            layer_dim_in = dim_emb if ind == 0 else dim_hid
            layer_dim_out = dim_out if ind == (num_layers - 1) else dim_hid
            w_std = (1 / dim_emb) if ind == 0 else (sqrt(c / layer_dim_in) / w0)

            std = w_std * init_std_scale 
            self.prior_mu.append(nn.Parameter(torch.zeros([layer_dim_in + 1, layer_dim_out])))
            self.prior_std.append(nn.Parameter(torch.ones(layer_dim_in + 1, layer_dim_out) * std))

import os
import time 
import imageio
import pickle

import torch
import torch.distributed as dist
import torch.optim.lr_scheduler as LS
import torchvision

import utils
from models import SirenPosterior
from PracticalCoding.partition import adjust_beta_with_mask


class TrainerDDP():
    def __init__(self, rank, args):
        self.rank = rank
        self.args = args
        self.is_master = (rank == 0)

        self.total_gpus = args.total_gpus
        self.distributed = (args.total_gpus > 1)
        # Setup distributed devices
        torch.cuda.set_device(rank)
        self.device = torch.device("cuda", torch.cuda.current_device())
        if self.distributed:
            dist_url = f"tcp://localhost:{args.port}"
            dist.init_process_group(backend="nccl", init_method=dist_url,
                                    world_size=self.total_gpus, rank=rank)
            print(f"Distributed training enabled. GPU: {str(self.device)}", flush=True)

        self.transform = torchvision.transforms.ToTensor()
        self.mse_func = torch.nn.MSELoss()

    def run(self, model_prior, epoch):
        args = self.args
        device = self.device
        min_id, max_id = self.assign_image(args)

        for image_id in range(min_id, max_id):
            s_time = time.time()

            # load the model
            model_path = os.path.join(args.log_dir, f"model_siren_{image_id}.pt")
            model = self.build_model(args, model_path, epoch)
            model = model.to(device)

            # variational optimization
            model, losses = self.trainer(
                args,
                model, 
                model_prior.to(device), 
                image_id, 
                epoch
            )

            e_time = time.time()
            print(f"({e_time - s_time:.2f} s) Image {image_id:3d}, PSNR / KL {losses['psnr']:.3f}/{losses['kl']:.1f}", flush=True)

            # save the INR model representing the current image
            torch.save(model.cpu().state_dict(), model_path)

    def trainer(self, args, model, model_prior, image_id, epoch):
        # load the image
        coordinates, features = self.load_image(args, image_id)
        coordinates, features = coordinates.to(self.device), features.to(self.device)

        # optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        num_iters = args.num_iters * 2 if epoch == 1 else args.num_iters

        # optimization
        for step in range(num_iters):
            optimizer.zero_grad()
            predicted, kld_list = model(coordinates, model_prior)
            loss_mse = self.mse_func(predicted, features)
            loss_kl = sum([kld_layer.sum() for kld_layer in kld_list])
            loss = loss_mse + loss_kl * args.weight_kl

            loss.backward()
            optimizer.step()
            
            psnr = utils.get_clamped_psnr(predicted, features)
            losses = {'loss': loss.item(), 'psnr': psnr, 'kl': loss_kl.item()}
        return model, losses

    # assign images to different gpus
    def assign_image(self, args):
        num_per_gpu = args.num_training // self.total_gpus
        min_id = num_per_gpu * self.rank
        max_id = num_per_gpu * (self.rank + 1) if self.rank + 1 != self.total_gpus else args.num_training
        return min_id, max_id

    # set up the model
    def build_model(self, args, model_path, epoch):
        model = SirenPosterior(
            dim_in=args.dim_in,
            dim_emb=args.dim_emb, 
            dim_hid=args.dim_hid, 
            dim_out=args.dim_out,
            num_layers=args.num_layers, 
            std_init=args.std_init, 
            c=args.c
        )
        if epoch != 1:
            model.load_state_dict(torch.load(model_path))
        return model
    
    # load the training image
    def load_image(self, args, image_id):
        img = imageio.imread(os.path.join(args.train_dir, f"img{str(image_id)}.png"))
        img = self.transform(img).float()
        coordinates, features = utils.to_grid_coordinates_and_features(img)
        if args.dataset == "kodak":
            n = features.shape[0]
            index = torch.randperm(n)[:n//4]
            coordinates = torch.index_select(coordinates, 0, index)
            features = torch.index_select(features, 0, index)
        return coordinates, features



class TesterDDP():
    def __init__(self, rank, args):

        self.rank = rank
        self.args = args
        self.is_master = (rank == 0)

        self.total_gpus = args.total_gpus
        self.distributed = (args.total_gpus > 1)
        # Setup distributed devices
        torch.cuda.set_device(rank)
        self.device = torch.device("cuda", torch.cuda.current_device())
        if self.distributed:
            dist_url = f"tcp://localhost:{args.port}"
            dist.init_process_group(backend="nccl", init_method=dist_url,
                                    world_size=self.total_gpus, rank=rank)
            print(f"Distributed training enabled. GPU: {str(self.device)}", flush=True)

        self.transform = torchvision.transforms.ToTensor()
        self.mse_func = torch.nn.MSELoss()

    def run(self, model_prior, groups):
        args = self.args
        device = self.device
        min_id, max_id = self.assign_image(args)

        for image_id in range(min_id, max_id):
            s_time = time.time()

            # load the model
            model_path = os.path.join(args.save_dir, f"model_test_{image_id}.pt")
            model = self.build_model(args, model_path)
            model = model.to(device)

            # variational optimization
            model, losses, beta_list = self.trainer(
                args,
                model, 
                model_prior.to(device), 
                image_id, 
                groups
            )

            e_time = time.time()
            print(f"({e_time - s_time:.2f} s) Image {image_id:3d}, PSNR / KL {losses['psnr']:.3f}/{losses['kl']:.1f}", flush=True)
            
            # save the INR model representing the current image
            torch.save(model.cpu().state_dict(), model_path)

            # save beta list
            beta_list = [beta_layer.cpu() for beta_layer in beta_list]
            beta_path = os.path.join(args.save_dir, f"model_beta_{image_id}.pt")
            with open(beta_path, "wb") as f:
                pickle.dump(beta_list, f)

    def trainer(self, args, model, model_prior, image_id, groups):
        # load the image
        coordinates, features = self.load_image(args, image_id)
        coordinates, features = coordinates.to(self.device), features.to(self.device)

        # optimizer
        num_iters = args.num_iters
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler_used = LS.MultiStepLR(optimizer , milestones=[int(num_iters * 0.8)], gamma=0.5)

        # optimization
        beta_list = [torch.ones_like(model_prior.prior_mu[i]) * args.weight_kl for i in range(args.num_layers)]
        for step in range(num_iters):
            optimizer.zero_grad()

            # optimization with all the pixels in the last stage.
            if step < int(num_iters * 0.95):
                if args.dataset == "kodak":
                    n = features.shape[0]
                    index = torch.randperm(n)[:n//4].to(self.device)
                    coordinates_input = torch.index_select(coordinates, 0, index)
                    features_input = torch.index_select(features, 0, index)
                else:
                    coordinates_input = coordinates
                    features_input = features_input

            predicted, kld_list = model(coordinates_input, model_prior)
            loss_mse = self.mse_func(predicted, features_input)
            loss_kl_beta = sum([(kld_layer * beta_list[i]).sum() for i, kld_layer in enumerate(kld_list)])
            loss_kl = sum([kld_layer.sum() for kld_layer in kld_list])

            loss = loss_mse + loss_kl_beta

            loss.backward()
            optimizer.step()
            scheduler_used.step()
            
            psnr = utils.get_clamped_psnr(predicted, features_input)
            losses = {'loss': loss.item(), 'psnr': psnr, 'kl': loss_kl.item()}

            # adjust_beta to ensure the KL budget is satisfied
            if step > num_iters // 10 and step % args.beta_adjust_interval == 0:
                beta_list = adjust_beta_with_mask(args, groups, beta_list, kld_list, mask_list=None)

        return model, losses, beta_list

    # assign images to different gpus
    def assign_image(self, args):
        num_per_gpu = args.num_test // self.total_gpus
        min_id = num_per_gpu * self.rank
        max_id = num_per_gpu * (self.rank + 1) if self.rank + 1 != self.total_gpus else args.num_test
        return min_id + 1, max_id + 1

    # set up the model
    def build_model(self, args, model_path):
        model = SirenPosterior(
            dim_in=args.dim_in,
            dim_emb=args.dim_emb, 
            dim_hid=args.dim_hid, 
            dim_out=args.dim_out,
            num_layers=args.num_layers, 
            std_init=args.std_init, 
            c=args.c
        )
        return model
    
    # load the test image
    def load_image(self, args, image_id):
        img = imageio.imread(os.path.join(args.test_dir, f"kodim{str(image_id).zfill(2)}.png"))
        img = self.transform(img).float()
        coordinates, features = utils.to_grid_coordinates_and_features(img)
        return coordinates, features
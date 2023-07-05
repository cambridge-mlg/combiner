# Combiner

Offical implementation of [Compression with Bayesian Implicit Neural Representations](https://arxiv.org/abs/2305.19185).

## Installation

We recommend using conda environment for installation.

```bash
conda create --name $ENV_NAME
conda activate $ENV_NAME
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip3 install -r requirements.txt
```

### Preparing training and test data.

Currently, we only provide the code for Kodak dataset. 

We recommend using cropped patches as training set to learn the model prior. The training datasets can be generated from CLIC dataset or DIV2K dataset. The folder structure of training set is like this:

	|-- trainset
		|-- img0.png
		|-- img1.png
		|-- img2.png
		...

To align with the resolution of Kodak, the training images should be randomly cropped into 512x768 or 768x512 patches.

## Running the code

Run the training script `train_model_prior.py` to learn the model prior by coordinate descent (parallel training):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u train_model_prior.py -c=./cfgs/model_prior_kodak.yaml 
```

Run the encoding scripts for a test image on Kodak dataset. 
First, learn the model posterior of all the 24 Kodak dataset (parallel optimization).

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u encode_learn_posterior.py -c=./cfgs/model_posterior_kodak.yaml 
```

Specifying a test image, finetune the model posterior progressively and encode the model parameters into binary file. 

```bash
CUDA_VISIBLE_DEVICES=0 python -u encode_tune_posterior.py -c=./cfgs/model_posterior_kodak.yaml --image_id=3
```

You can decode the binary file by running

```bash
CUDA_VISIBLE_DEVICES=0 python -u decode_posterior.py -c=./cfgs/model_posterior_kodak.yaml --image_id=3
```


## Configuration

You can adjust the hyper parameters of training and test settings by modifying the files in ./cfgs

## Citation

If you use this library for research purposes, please cite:

```
@article{combiner2023,
  title={Compression with Bayesian Implicit Neural Representations},
  author={Guo, Zongyu# and Flamich, Gergely# and He, Jiajun and Chen, Zhibo and Hern{\'a}ndez-Lobato, Jos{\'e} Miguel},
  journal={arXiv preprint arXiv:2305.19185},
  year={2023}
}
```


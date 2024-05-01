import argparse
from typing import cast
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader, random_split
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm.notebook import tqdm
import zipfile
import os
import numpy as np
import math
from torchvision.transforms import v2
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Config
import transformers
import random
import mingpt.gen as gen
from mingpt.gen import unnormalize, device, train
import mingpt.data as data
import shlex
import argparse

import vqvae
import vqvae.utils
import json
"""

@dataclass
class ExperimentConfig():
    work_dir : str
    image_width : int
    image_height : int
    image_channel : int 
    image_token_length : int # = image_width * image_height * image_channel for pixel images, otherwise it depends on latent space size

    total_length : int # = image_token_length + text_codebook_size + special_token_num
    special_token_num : int # equals to 2, for <bos> and <eos>
    bos_id : int # 256 for pixel image
    eos_id : int # 257 for pixel image

    dataset_type : str # mnist or fashion mnist or...

    # tunable hyperparameter for training
    batch_size : int
    learning_rate : float
    n_epoch : int
    n_warmup : int = 1000

    # in steps. If you have a powerful gpu, you can upscale the value
    log_iter : int = 300
    save_iter : int = 1000
    eval_iter : int = 300

    # hyperparameter for model (default is 3M)
    n_embed : int = 256
    n_head : int = 8

    use_vqvae : bool = False # whether use vqvae to compress the image first

    image_codebook_size : int = 256 # for image this is [0,255] pixel value

    text_codebook_size : int = 0 # unconditional generation
"""
# Create the parser
parser = argparse.ArgumentParser()

timestamp = vqvae.utils.readable_timestamp()
# Add arguments

parser.add_argument('--dataset_type', type=str, help='type of dataset, can only be "mnist", "fashion", "spirit" or "face"')
# work directory of the model
# the plot and model will be saved to this directory
# I recommend you to use different work directory for different hyper parameters
parser.add_argument('--work_dir', type=str)
parser.add_argument('--data_dir', type=str)
# logging interval, you can increase this value if there are too many logs
parser.add_argument('--log_iter', type=int, default=300)
# checkpoint interval
parser.add_argument('--save_iter', type=int, default=1000)
# evaluation interval
parser.add_argument('--eval_iter', type=int, default=300)
# embedding size of GPT2 model
parser.add_argument('--n_embed', type=int, default=256)
# number of attention head
parser.add_argument('--n_head', type=int, default=8)
# number of layer
parser.add_argument('--n_layer', type=int)

# hyper parameter
parser.add_argument('--batch_size', type=int)
parser.add_argument('--learning_rate', type=float)
parser.add_argument('--n_epoch', type=int)
parser.add_argument('--pdropout', type=float)

# I prefer 1000 for learning rate, it should be around 1% of the total training steps
parser.add_argument('--n_warmup', type=int)
# set to 2 for mnist and fashion dataset, if you want a quick experiment
# this will downscale the image to a smaller size
parser.add_argument('--downscale', type=int)

# parameter used only in evaluation
# if you want to perform evaluation with a model, then provide the fullpath to a model.pt file
parser.add_argument('--checkpoint', type=str)
# row = 5 will generate 5*5 images
parser.add_argument('--row', type=int, default=5)
# higher temperature will increase diversity of output, you can tune this parameter by yourself
parser.add_argument('--temperature', type=float, default=1.0)
# try 2 to 5, will slow down generation
parser.add_argument('--num_beams', type=int, default=2)

# vqvae parameter
parser.add_argument('--vqvae_checkpoint', type=str)

# important!!!
# you must set seed to 0 during training for reproducibility 
# for evaluation you can ignore this parameter, then the script will pick a seed for you
parser.add_argument('--seed', type=int, default=random.randrange(100000))

# whether we do unconditional generation
parser.add_argument('--unconditional', action='store_true')
example = """
--dataset_type vqvae-flower \
       --work_dir work/mingpt \
       --log_iter 300 \
       --save_iter 1000 \
       --eval_iter 3000 \
       --n_embed 256 \
       --n_head 8 \
       --n_layer 4 \
       --pdropout 0.01 \
       --batch_size 16 \
       --learning_rate 1e-4 \
       --n_epoch 256 \
       --n_warmup 3000 \
       --downscale 1 \
       --row 5 \
       --num_beams 2 \
       --temperature 1 \
       --data_dir=data \
       --vqvae_checkpoint=work/vqvae/mon_apr_15_11_00_48_2024/results/flower-vqvae.pt\
"""
def in_ipython():
    try:
        return __IPYTHON__
    except NameError:
        return False
    
if in_ipython():
    args = parser.parse_args(shlex.split(example))
else:
    args = parser.parse_args()
print(args)

args.timestamp = timestamp
if args.checkpoint is None:
    is_training = True
    for i in ["data", "plot", "runs", "model"]:
        target = os.path.join(args.work_dir, i)
        if not os.path.exists(target):
            os.mkdir(target)
        else:
            print("overwrite data in the folder!")
    with open(os.path.join(args.work_dir, "config.json"), 'w') as f:
        json.dump(args.__dict__, f, indent=4)
else:
    # otherwise we won't create the working directory
    is_training = False

random.seed(args.seed)
import numpy as np
np.random.seed(args.seed)
import torch
torch.manual_seed(args.seed)

total_data, custom_dataset_info = data.get_data(args.dataset_type, args.data_dir, downscale=args.downscale)
flatten_database = custom_dataset_info.flatten_images
if args.unconditional:
    text_codebook_size = 0
else:
    text_codebook_size = custom_dataset_info.label_num

from vqvae.model import VQVAE, CodeLayer
from vqvae.unet_model import UNetVQVAE
import vqvae
vqvae_model = None
if custom_dataset_info.vqvae:
    assert args.vqvae_checkpoint != None
    vqvae_model = vqvae.utils.load_module(args.vqvae_checkpoint).to(device)
    if isinstance(vqvae_model, VQVAE):
        image_codebook_size = 0
        for i in vqvae_model.codebooks:
            ci = cast(CodeLayer, i)
            image_codebook_size += ci.n_embed
    else:
        assert isinstance(vqvae_model, UNetVQVAE)
        image_codebook_size = vqvae_model.codebook.n_embed
    token_length = len(vqvae_model.get_latent_offset( custom_dataset_info.shape[:-1]))
else:
    image_codebook_size = 256
    token_length = math.prod(custom_dataset_info.shape)

total_vocab_size = image_codebook_size + 2 + text_codebook_size


"""
[0, image_codebook_size] image code book vocab
[image_codebook_size, image_codebook_size + text_codebook_size) text code book vocab
[image_codebook_size + text_codebook_size, image_codebook_size + text_codebook_size + 2) special token
"""

config = gen.ExperimentConfig(
    work_dir = args.work_dir,
    checkpoint = args.checkpoint,
    shape = custom_dataset_info.shape,
    image_token_length = token_length,
    # one class token, two special tokens and image tokens
    total_length = token_length + 2 + (not args.unconditional),
    special_token_num = 2,
    bos_id = image_codebook_size + text_codebook_size,
    eos_id = image_codebook_size + text_codebook_size + 1,
    row=args.row,
    temperature=args.temperature,
    num_beams=args.num_beams,

    dataset_type = args.dataset_type,
    downscale=args.downscale,

    # tunable hyperparameter for training
    batch_size = args.batch_size,
    learning_rate = args.learning_rate,
    n_epoch = args.n_epoch,
    n_warmup =  args.n_warmup,
    pdropout = args.pdropout,

    log_iter = args.log_iter,
    save_iter = args.save_iter,
    eval_iter = args.eval_iter,

    # hyperparameter for model (default is 3M)
    n_embed = args.n_embed,
    n_head = args.n_head,
    n_layer = args.n_layer,
    use_vqvae = custom_dataset_info.vqvae, # whether use vqvae to compress the image first
    vqvae_checkpoint = args.vqvae_checkpoint,
    total_vocab_size = total_vocab_size,
    image_codebook_size = image_codebook_size,
    text_codebook_size=text_codebook_size,
    unconditional=args.unconditional
)



training_data, eval_data = random_split(total_data, [0.9, 0.1])
total_data = None

pdrop = config.pdropout
model_config = GPT2Config(
    vocab_size=config.total_vocab_size,
    n_positions=config.total_length,
    bos_token_id=config.bos_id,
    eos_token_id=config.eos_id,
    n_embd = config.n_embed,
    n_layer = config.n_layer,
    n_head = config.n_head,
    resid_pdrop=pdrop,
    embd_pdrop=pdrop,      
    attn_pdrop=pdrop
)
model = GPT2LMHeadModel(model_config)

if is_training:
    optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9,0.95), lr=config.learning_rate, weight_decay=0.1)
    num_epochs = config.n_epoch
    bs = config.batch_size
    train_dl = DataLoader(training_data, batch_size=bs, shuffle=True, drop_last=True)
    eval_dl =  DataLoader(eval_data, batch_size=bs, shuffle=False, drop_last=True)
    device = "cuda"
    num_training_steps = num_epochs * len(train_dl)
    lr_scheduler = transformers.get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.n_warmup
        )

    train(config, 
            model,
            vqvae_model, 
            optimizer, 
            lr_scheduler, 
            train_dl, 
            eval_dl, 
            custom_dataset_info,
            n_iter=0)
else:
    cpt = cast(str, config.checkpoint)
    model.load_state_dict(torch.load(cpt))
    gen.generate_images(config,
                    model,
                    vqvae_model,
                    "eval",
                    custom_dataset_info,
                    row=config.row,
                    temperature=config.temperature,
                    num_beams=config.num_beams)
    
    gen.reconstruct_images(config,
                    model,
                    vqvae_model,
                    eval_data,
                    "reconstruct",
                    custom_dataset_info,
                    row=config.row,
                    temperature=config.temperature,
                    num_beams=config.num_beams,
                    ratio=0.5) 
from vqvae.model import VQVAE
from vqvae.unet_model import UNetVQVAE
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import matplotlib.pyplot as plt
import vqvae.utils as utils
from tqdm import tqdm
from vqvae.data import load_data
from vqvae.patchgan import *
from torch.utils.data import DataLoader
import shlex
import random
import os
import json

device = "cuda"

parser = argparse.ArgumentParser()
timestamp = utils.readable_timestamp()

# can be unet vqvae or vqvae-2
parser.add_argument("--vqvae_type", type=str, default="vqvae-2")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--n_updates", type=int, default=5000)
parser.add_argument("--n_hiddens", type=int, default=128)
parser.add_argument("--n_residual_hiddens", type=int, default=32)
parser.add_argument("--n_residual_layers", type=int, default=2)
parser.add_argument("--embedding_dim", type=int, default=16)
parser.add_argument("--n_embeddings", type=int, default=512)
parser.add_argument("--beta", type=float, default=0.25)
parser.add_argument("--learning_rate", type=float, default=3e-4)
parser.add_argument("--log_interval", type=int, default=50)
parser.add_argument("--dataset", type=str)
parser.add_argument("--work_dir", type=str)
parser.add_argument("--data_dir", type=str)
parser.add_argument("--scaling_rates", nargs="+", type=int)
parser.add_argument("--ch_mult", nargs="+", type=int)
parser.add_argument("--use_attention", action='store_true')

# whether or not to save model
parser.add_argument("-save", action="store_true")
example = """
-save \
  --vqvae_type unet \
  --log_interval=500 \
  --n_hiddens=32\
  --n_residual_layers=1\
  --n_updates=30000\
  --data_dir=data \
  --dataset=ffhq \
  --work_dir=work/vqvae \
  --n_embeddings=256 \
  --embedding_dim=4 \
  --ch_mult 1 2 4
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

args.timestamp = timestamp
for i in ["data", "plot", "results", "runs"]:
    target = os.path.join(args.work_dir, i)
    if not os.path.exists(target):
        os.mkdir(target)
    else:
        print("overwrite result in the folder!")
with open(os.path.join(args.work_dir, "config.json"), 'w') as f:
    json.dump(args.__dict__, f, indent=4)

print(args)
training_data, validation_data, training_loader, validation_loader, x_train_var, shape = load_data(args.dataset, args.data_dir, args.batch_size)

if args.vqvae_type == "vqvae-2":
    assert args.ch_mult is None
    model = VQVAE(
        nb_levels=len(args.scaling_rates),
        scaling_rates=args.scaling_rates,
        in_channels=shape[-1],
        hidden_channels=args.n_hiddens,
        res_channels=args.n_residual_hiddens,
        nb_res_layers=args.n_residual_layers,
        embed_dim=args.embedding_dim,
        nb_entries=args.n_embeddings,
    ).to(device)
elif args.vqvae_type == "unet":
    assert args.scaling_rates is None
    model = UNetVQVAE(in_channels=shape[-1], 
              ch_mult = args.ch_mult,
              hidden_channels=args.n_hiddens, 
              nb_res_layers=args.n_residual_layers, 
              embed_dim=args.embedding_dim,
              nb_entries=args.n_embeddings, 
              use_attention=args.use_attention).to(device)
else:
    print("Invalid vqvae type")
    assert False 

print(model)
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
print(f"number of parameters : {sum([np.prod(p.size()) for p in model_parameters])}")

optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)

def show_image(n_iter, id=8, clip=False, useRandom=False):
    f, axarr = plt.subplots(2, id, figsize=(id * 4, 8))
    model.eval()
    scale_img = lambda x: (x + 1) / 2
    for i in range(0, id):
        if useRandom:
            data = training_data[random.randrange(0, len(training_data) - 1)][0]
        else:
            data = training_data[i][0]
        axarr[0][i].axis("off")
        axarr[0][i].imshow(scale_img(data.permute((1, 2, 0)).cpu().detach().numpy()))
        with torch.no_grad():
            output, _, _, _, _ = model(data.unsqueeze(0).to(device))
        if clip:
            output = torch.clip(output, min=-1, max=1)
        output = scale_img(output)
        output = output.squeeze(0).permute((1, 2, 0)).cpu().detach().numpy()
        axarr[1][i].axis("off")
        axarr[1][i].imshow(output)
    f.suptitle("Face plot" + str(n_iter))
    import os
    plt.savefig(os.path.join(args.work_dir, "plot/" + str(n_iter) + ".png"))
    plt.close()

show_image("test", useRandom=True, clip=True)
patchgan_model = PatchGAN(d=16).to(device)

from torch.utils.tensorboard.writer import SummaryWriter

def train():
    import os
    writer = SummaryWriter(os.path.join(args.work_dir, "runs"))
    pbar = tqdm(range(args.n_updates), position=0)
    for i in pbar:
        model.train()
        (x, _) = next(iter(training_loader))
        x : torch.Tensor = x.to(device)
        optimizer.zero_grad(set_to_none=True)

        x_hat, diffs, _, _, _ = model(x)
        embedding_loss : torch.Tensor = sum(diffs)
        mse_loss = torch.mean((x_hat - x) ** 2)
        # firstly we train patch gan model
        recons_loss = mse_loss + args.beta * embedding_loss
        results = {}
        if i > len(training_loader):
            d_loss = train_patchgan(patchgan_model, x, x_hat.detach())
            g_loss = gan_loss(patchgan_model, x_hat)
            loss = recons_loss + 1e-3 * g_loss

            results["d_loss"] = d_loss.item()
            results["ratio"] = (recons_loss/ g_loss).item()
        else:
            loss = recons_loss
        # x_hat is fake data and x is real data
        # g_loss = gan_loss(patchgan_model, x_hat.detach())
        # g_loss.backward()
        # last_patch_gan_param = list(patchgan_model.parameters())[-1]
        # last_patch_gan_param_norm = last_patch_gan_param.grad.data.norm(2)

        # recons_loss = recon_loss + args.beta * embedding_loss
        # recons_loss.backward(retain_graph=True)

        # last_model_param = list(model.parameters())[-1]
        # last_model_param_norm = last_model_param.grad.data.norm(2)
        # ratio = last_model_param_norm / (last_patch_gan_param_norm + 1e-6)
        # g_loss = ratio * gan_loss(patchgan_model, x_hat.detach())
        # g_loss.backward()
        loss.backward()
        optimizer.step()

        results["recon_errors"] = mse_loss.item()
        results["embedding_loss"] = embedding_loss.item()
        results["g_loss"] = loss.item()
        results["loss_vals"] = loss.item()
        results["n_updates"] = i

        for k in results:
            writer.add_scalar(k, results[k], i)
        if i % args.log_interval == 0 or i == len(pbar) - 1:
            show_image(i, 10, clip=True, useRandom=True)
            """
            save model and print values
            """
            if args.save:
                pbar.set_postfix({"saving": i})
                hyperparameters = args.__dict__
                hyperparameters["channel_num"] = x.shape[1]
                utils.save_model_and_results(args.dataset,
                    args.work_dir, model, results, hyperparameters
                )

if __name__ == "__main__":
    train()
    model.eval()
    off = model.get_latent_offset(shape[:-1]).to("cuda")
    cs = []
    for input in DataLoader(training_data, batch_size=64, shuffle=False, drop_last=False):
        c = model.encode_latent(input[0].to(device)) + off
        cs.append(c)
    codes = torch.cat(cs, dim=0)
    import os
    np.save(os.path.join(args.work_dir, f"{args.dataset}_image_token.npy"), codes.detach().cpu().numpy())
    test_data = codes[0:64]
    outputs = model.decode_latent(test_data, shape[:-1]) # (b, c, w, h)
    outputs = ((outputs + 1) / 2) 
    images = outputs.permute((0, 2,3,1))
    images = images.detach().cpu().numpy()
    plt.imshow(images[1])
    plt.savefig(os.path.join(args.work_dir, "plot/reconstruct-test.png"))
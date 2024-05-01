import torch
from .model import VQVAE, CodeLayer
from typing import List
import os
import time
from torch.utils.data import DataLoader
from typing import cast, Tuple, List
def readable_timestamp():
    return time.ctime().replace('  ', ' ').replace(
        ' ', '_').replace(':', '_').lower()

def save_model_and_results(dataset, work_dir, model, results, hyperparameters):
    results_to_save = {
        'model': model.state_dict(),
        'results': results,
        'hyperparameters': hyperparameters
    }
    torch.save(results_to_save, os.path.join(work_dir, f"results/{dataset}-vqvae.pt"))


def load_module(checkpoint):
    d = torch.load(checkpoint)
    model_state_dict = d["model"]
    args = d["hyperparameters"]
    if args["vqvae_type"] == "vqvae-2":
        model = VQVAE(
            nb_levels=len(args["scaling_rates"]),
            scaling_rates=args["scaling_rates"],
            in_channels=args["channel_num"],
            hidden_channels=args["n_hiddens"],
            res_channels=args["n_residual_hiddens"],
            nb_res_layers=args["n_residual_layers"],
            embed_dim=args["embedding_dim"],
            nb_entries=args["n_embeddings"],
        )
    elif args["vqvae_type"] == "unet":
        from .unet_model import UNetVQVAE
        model = UNetVQVAE(
              in_channels=args["channel_num"], 
              ch_mult = args["ch_mult"],
              hidden_channels=args["n_hiddens"], 
              nb_res_layers=args["n_residual_layers"], 
              embed_dim=args["embedding_dim"],
              nb_entries=args["n_embeddings"], 
              use_attention=args["use_attention"],
        )
    else:
        assert False
    model.load_state_dict(model_state_dict)
    model.eval()
    return model

def get_parameter_count(net: torch.nn.Module) -> int:
    return sum(p.numel() for p in net.parameters() if p.requires_grad)
def get_device(cpu):
    if cpu or not torch.cuda.is_available(): return torch.device('cpu')
    return torch.device('cuda')
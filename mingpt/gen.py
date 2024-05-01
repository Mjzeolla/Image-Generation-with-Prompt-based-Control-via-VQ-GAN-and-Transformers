import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm import tqdm
import os
import numpy as np
from dataclasses import dataclass
from typing import Union, List, Tuple, cast
from . import data
from vqvae.model import VQVAE
from vqvae.unet_model import UNetVQVAE
import vqvae
import vqvae.utils

VQVAEType = UNetVQVAE |  VQVAE | None
device = "cuda"
unnormalize = lambda x : (x+1)/2

n_iter = 0

@dataclass
class ExperimentConfig():
    work_dir : str
    checkpoint : Union[str, None]
    shape : Tuple[int, int, int]
    image_token_length : int # = image_width * image_height * image_channel for pixel images, otherwise it depends on latent space size

    total_length : int #
    special_token_num : int # equals to 2, for <bos> and <eos>
    bos_id : int # 256 for pixel image
    eos_id : int # 257 for pixel image

    row : int
    temperature : float
    num_beams : int

    dataset_type : str # mnist or fashion mnist or...
    downscale : int
    # tunable hyperparameter for training
    batch_size : int
    learning_rate : float
    n_epoch : int
    pdropout : float
    n_warmup : int = 1000
    
    
    # in steps. If you have a powerful gpu, you can upscale the value
    log_iter : int = 300
    save_iter : int = 1000
    eval_iter : int = 300

    # hyperparameter for model (default is 3M)
    n_embed : int = 256
    n_head : int = 8
    n_layer : int = 4

    use_vqvae : bool = False # whether use vqvae to compress the image first
    vqvae_checkpoint : str = ""
    total_vocab_size : int = 258 # image_codebook_size + text_codebook_size + special_token_num
    image_codebook_size : int = 256 # for image this is [0,255] pixel value
    text_codebook_size : int = 0 # unconditional generation

    unconditional : bool = True

    
    
"""
    Generate (row, 2*row) image arrays from the model 
    flatten_database is the original database of shape (b, image_token_length)
    we use it to match input dataset, to ensure the model doesn't just memorize the inputs
"""
@torch.no_grad
def generate_images(config : ExperimentConfig,
                    model,
                    vqvae_model : VQVAEType,
                    n_iter,
                    custom_dataset : data.CustomDatasetInfo,
                    row=10,
                    temperature=1.0,
                    num_beams=2):
    bs = row * row
    model.eval()
    model.to(device)
    total_length = config.total_length
    start_token = torch.full((bs,1), config.bos_id).to(device)
    if config.unconditional:
        inputs = start_token
    else:
        label_token = torch.arange(0, bs).unsqueeze(-1) % config.text_codebook_size + config.image_codebook_size
        assert len(label_token.shape) == 2
        label_token= label_token.to(device)
        inputs = torch.cat([start_token, label_token], dim=-1)
    with torch.no_grad():
        if config.use_vqvae:
            result = model.generate(inputs=inputs, max_length=total_length, min_length=total_length, pad_token_id=config.eos_id, do_sample=True, top_k=100, temperature=temperature)
        else:
            # based on my own experiment
            # in pixel space we should use beam search
            result = model.generate(inputs=inputs, max_length=total_length, min_length=total_length, pad_token_id=config.eos_id, do_sample=True, num_beams=num_beams, temperature=temperature)
    decode_f = lambda x : x - config.image_codebook_size
    if config.unconditional:
        labels = None
        result = result[:,1:-1]
    else:
        labels = decode_f(result[:, 1])
        result = result[:,2:-1]
    image_height, image_width, image_channel = config.shape
    
    if not config.use_vqvae:
        images = result.reshape((bs, image_channel, image_height, image_width)).permute((0, 2,3,1))
        images = images / 255
        flatten_images = (result/255).to(device)
    else:
        vqvae_model = cast(VQVAE, vqvae_model).to(device)
        vqvae_model.eval()
        off = vqvae_model.get_latent_offset(config.shape[:-1]).to(device)
        result -= off
        outputs = vqvae_model.decode_latent(result, config.shape[:-1]) # (b, c, w, h)
        # outputs is in [-1, 1]
        outputs = ((outputs + 1) / 2) 
        images = outputs.permute((0, 2,3,1)) # (b, w, h, c)
        flatten_images = outputs.flatten(start_dim=1) # (b, c, w, h)
    images = np.maximum(0, np.minimum(1, images.detach().cpu().numpy()))
    
    image_height, image_width, image_channel = config.shape
    # for pixel space image, we clip the value into [0, 255], sometimes the model generates special tokens
    # search image in the database
    flatten_database = custom_dataset.flatten_images
    indices = torch.argmin(torch.cdist(flatten_images, flatten_database), dim = -1)
    match_images = flatten_database[indices].reshape((bs, image_channel, image_height, image_width)).permute((0, 2,3,1)).detach().cpu().numpy()
    ncol = row
    nrow = row
    f, axarr = plt.subplots(nrow,2*ncol, figsize=(ncol*2*2,nrow*2))
    for i in range(nrow):
        for j in range(ncol):
            d = i *ncol+j
            ax = axarr[i][j]
            data = images[d]
            

            if data.shape[-1] == 3:
                ax.imshow(data, interpolation='none')
            else:
                ax.imshow(data, interpolation='none', cmap="gray")
            if not config.unconditional:
                labels = cast(torch.Tensor, labels)
                label = custom_dataset.label_map[labels[d].item()]
                ax.title.set_text(label)
            ax.axis('off')
    match_labels = custom_dataset.labels
    if match_labels is not None:
        match_labels = match_labels[indices]
    for i in range(nrow):
        for j in range(ncol):
            d = i *ncol+j
            ax = axarr[i][ncol+j]
            data = match_images[d]
            if match_labels is not None:
                iii : int = cast(int, match_labels[d].item())
                label = custom_dataset.label_map[iii]
                ax.title.set_text(label)
            if data.shape[-1] == 3:
                ax.imshow(data, interpolation='none')
            else:
                ax.imshow(data, interpolation='none', cmap="gray")
            ax.axis('off')
    plt.savefig(os.path.join(config.work_dir, f"plot/{n_iter}.png"), dpi=90)
    plt.close(f)
        # Clear the current axes.
    plt.cla() 
    # Clear the current figure.
    plt.clf() 
    # Closes all the figure windows.
    plt.close('all')

@torch.no_grad
def reconstruct_images(config : ExperimentConfig,
                    model,
                    vqvae_model : VQVAEType,
                    dl,
                    n_iter,
                    custom_dataset : data.CustomDatasetInfo,
                    row=10,
                    temperature=1.0,
                    num_beams=2,
                    ratio=0.5):
    bs = row * row
    model.eval()
    model.to(device)
    total_length = config.total_length 
    start_token = torch.full((bs,1), config.bos_id).to(device)
    from torch.utils.data import RandomSampler, DataLoader
    sample_sampler = RandomSampler(dl)
    sample_dl = DataLoader(dl, sampler=sample_sampler, batch_size=bs)
    inputs, label_token = next(iter(sample_dl))
    inputs = inputs[:,:int(ratio*inputs.shape[-1])]
    print(inputs.shape)
    if config.unconditional:
        inputs = torch.cat([start_token, inputs], dim = -1)
    else:
        #label_token = torch.randint(config.image_codebook_size, 
        #                        config.image_codebook_size + config.text_codebook_size, (bs,1)).to(device)
        inputs = torch.cat([start_token, label_token.unsqueeze(-1) + config.image_codebook_size, inputs], dim = -1)
    with torch.no_grad():
        if config.use_vqvae:
            result = model.generate(inputs=inputs, max_length=total_length, min_length=total_length, pad_token_id=config.eos_id, do_sample=True, top_k=100, temperature=temperature)
        else:
            # based on my own experiment
            # in pixel space we should use beam search
            result = model.generate(inputs=inputs, max_length=total_length, min_length=total_length, pad_token_id=config.eos_id, do_sample=True, num_beams=num_beams, temperature=temperature)
    decode_f = lambda x : x - config.image_codebook_size
    if config.unconditional:
        labels = None
        result = result[:,1:-1]
    else:
        labels = decode_f(result[:, 1])
        result = result[:,2:-1]
    image_height, image_width, image_channel = config.shape
    
    if not config.use_vqvae:
        images = result.reshape((bs, image_channel, image_height, image_width)).permute((0, 2,3,1))
        images = images / 255
        flatten_images = (result/255).to(device)
    else:
        vqvae_model = cast(VQVAE, vqvae_model).to(device)
        vqvae_model.eval()
        off = vqvae_model.get_latent_offset(config.shape[:-1]).to(device)
        result -= off
        outputs = vqvae_model.decode_latent(result, config.shape[:-1]) # (b, c, w, h)
        # outputs is in [-1, 1]
        outputs = ((outputs + 1) / 2) 
        images = outputs.permute((0, 2,3,1)) # (b, w, h, c)
        flatten_images = outputs.flatten(start_dim=1) # (b, c, w, h)
    images = np.maximum(0, np.minimum(1, images.detach().cpu().numpy()))
    
    image_height, image_width, image_channel = config.shape
    # for pixel space image, we clip the value into [0, 255], sometimes the model generates special tokens
    # search image in the database
    flatten_database = custom_dataset.flatten_images
    indices = torch.argmin(torch.cdist(flatten_images, flatten_database), dim = -1)
    match_images = flatten_database[indices].reshape((bs, image_channel, image_height, image_width)).permute((0, 2,3,1)).detach().cpu().numpy()
    ncol = row
    nrow = row
    f, axarr = plt.subplots(nrow,2*ncol, figsize=(ncol*2*2,nrow*2))
    for i in range(nrow):
        for j in range(ncol):
            d = i *ncol+j
            ax = axarr[i][j]
            data = images[d]
            

            if data.shape[-1] == 3:
                ax.imshow(data, interpolation='none')
            else:
                ax.imshow(data, interpolation='none', cmap="gray")
            if not config.unconditional:
                labels = cast(torch.Tensor, labels)
                label = custom_dataset.label_map[labels[d].item()]
                ax.title.set_text(label)
            ax.axis('off')
    match_labels = custom_dataset.labels
    if match_labels is not None:
        match_labels = match_labels[indices]
    for i in range(nrow):
        for j in range(ncol):
            d = i *ncol+j
            ax = axarr[i][ncol+j]
            data = match_images[d]
            if match_labels is not None:
                iii : int = cast(int, match_labels[d].item())
                label = custom_dataset.label_map[iii]
                ax.title.set_text(label)
            if data.shape[-1] == 3:
                ax.imshow(data, interpolation='none')
            else:
                ax.imshow(data, interpolation='none', cmap="gray")
            ax.axis('off')
    plt.savefig(os.path.join(config.work_dir, f"plot/{n_iter}.png"), dpi=90)
    plt.close(f)
        # Clear the current axes.
    plt.cla() 
    # Clear the current figure.
    plt.clf() 
    # Closes all the figure windows.
    plt.close('all')

def train(config : ExperimentConfig,
          model, 
          vqvae_model : VQVAEType,
          optimizer, 
          lr_scheduler, 
          train_dl, 
          eval_dl, 
          custom_dataset : data.CustomDatasetInfo,
          n_iter = 0):
    from torch.utils.tensorboard.writer import SummaryWriter
    writer = SummaryWriter(log_dir=os.path.join(config.work_dir, 'runs'))
    bs = config.batch_size
    num_training_steps = len(train_dl) * config.n_epoch
    progress_bar = tqdm(range(num_training_steps))
    start_token = torch.full((bs,1), config.bos_id).to(device)
    eos_token = torch.full((bs,1), config.eos_id).to(device)
    model.to(device)
    print(config)
    # you can see the config in tensorboard's text section
    writer.add_text('experiment config', str(config))
    writer.add_text('model_size', str(sum([i.numel() for i in model.parameters()]))) 
    log_iter = config.log_iter
    eval_iter = config.eval_iter
    save_iter = config.save_iter
    offset_f = lambda x : x + config.image_codebook_size
    for epoch in range(config.n_epoch):
        for (batch, class_label) in train_dl:
            model.train()
            with torch.no_grad():
                batch = batch.to(device)
                class_label = offset_f(class_label)
                if config.unconditional:
                    batch = torch.cat([start_token, batch, eos_token], dim=-1)
                    target = batch.clone()
                else:
                    batch = torch.cat([start_token, class_label.unsqueeze(-1), batch, eos_token], dim=-1)
                    target = batch.clone()
                    # # we don't predict class, so we ignore the loss for it
                    # target[:,1] = -100
            assert batch.shape[-1] == config.total_length
            outputs = model(input_ids = batch, labels = target)
            loss = outputs.loss
            loss.backward()

            # gradient clip
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            
            writer.add_scalar('train_loss', loss.item() ,n_iter)
            writer.add_scalar('lr', lr_scheduler.get_last_lr()[-1] ,n_iter)
            if n_iter % log_iter == 0 or n_iter == num_training_steps - 1:
                generate_images(
                    config,
                    model,
                    vqvae_model,
                    n_iter,
                    custom_dataset,
                    row=config.row,
                    temperature=config.temperature,
                    num_beams=config.num_beams)
            if n_iter % save_iter == 0:
                torch.save(model.state_dict(), os.path.join(config.work_dir, f'model/{n_iter}.pt'))
            if n_iter % eval_iter == 0 or n_iter == num_training_steps - 1 :
                total_loss = 0
                for (batch, class_label) in eval_dl:
                    model.eval()
                    with torch.no_grad():
                        class_label = offset_f(class_label)
                        batch = batch.to(device)
                        if config.unconditional:
                            batch = torch.cat([start_token, batch, eos_token], dim=-1)
                            target = batch.clone()
                        else:
                            batch = torch.cat([start_token, class_label.unsqueeze(-1), batch, eos_token], dim=-1)
                            target = batch.clone()
                        outputs = model(input_ids = batch, labels = batch)
                        total_loss += outputs.loss
                writer.add_scalar('val_loss', total_loss / len(eval_dl) ,n_iter)
            n_iter += 1
    torch.save(model.state_dict(), os.path.join(config.work_dir, f'model/final.pt'))
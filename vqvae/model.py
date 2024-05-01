import torch
import torch.nn as nn
import torch.nn.functional as F

from math import log2
from typing import Any, Tuple, List, cast
class HelperModule(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.build(*args, **kwargs)


class ReZero(HelperModule):
    def build(self, in_channels: int, res_channels: int):
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, res_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(res_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(res_channels, in_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.layers(x) * self.alpha + x

class ResidualStack(HelperModule):
    def build(self, in_channels: int, res_channels: int, nb_layers: int):
        self.stack = nn.Sequential(*[ReZero(in_channels, res_channels) 
                        for _ in range(nb_layers)
                    ])

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.stack(x)

class Encoder(HelperModule):
    def build(self, 
            in_channels: int, hidden_channels: int, 
            res_channels: int, nb_res_layers: int,
            downscale_factor: int,
        ):
        assert log2(downscale_factor) % 1 == 0, "Downscale must be a power of 2"
        downscale_steps = int(log2(downscale_factor))
        layers = []
        c_channel, n_channel = in_channels, hidden_channels // 2
        for _ in range(downscale_steps):
            layers.append(nn.Sequential(
                nn.Conv2d(c_channel, n_channel, 4, stride=2, padding=1),
                nn.BatchNorm2d(n_channel),
                nn.ReLU(inplace=True),
            ))
            c_channel, n_channel = n_channel, hidden_channels
        layers.append(nn.Conv2d(c_channel, n_channel, 3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(n_channel))
        layers.append(ResidualStack(n_channel, res_channels, nb_res_layers))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.layers(x)

class Decoder(HelperModule):
    def build(self, 
            in_channels: int, hidden_channels: int, out_channels: int,
            res_channels: int, nb_res_layers: int,
            upscale_factor: int,
        ):
        assert log2(upscale_factor) % 1 == 0, "Downscale must be a power of 2"
        upscale_steps = int(log2(upscale_factor))
        layers : List[Any] = [nn.Conv2d(in_channels, hidden_channels, 3, stride=1, padding=1)]
        layers.append(ResidualStack(hidden_channels, res_channels, nb_res_layers))
        c_channel, n_channel = hidden_channels, hidden_channels // 2
        for _ in range(upscale_steps):
            layers.append(nn.Sequential(
                nn.ConvTranspose2d(c_channel, n_channel, 4, stride=2, padding=1),
                nn.BatchNorm2d(n_channel),
                nn.ReLU(inplace=True),
            ))
            c_channel, n_channel = n_channel, out_channels
        layers.append(nn.Conv2d(c_channel, n_channel, 3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(n_channel))
        # layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.layers(x)

"""
    Almost directly taken from https://github.com/rosinality/vq-vae-2-pytorch/blob/master/vqvae.py
    No reason to reinvent this rather complex mechanism.

    Essentially handles the "discrete" part of the network, and training through EMA rather than 
    third term in loss function.
"""
class CodeLayer(HelperModule):
    def build(self, in_channels: int, embed_dim: int, nb_entries: int):
        self.conv_in = nn.Conv2d(in_channels, embed_dim, 1)

        self.dim = embed_dim
        self.n_embed = nb_entries
        self.decay = 0.99
        self.eps = 1e-5

        embed = torch.randn(embed_dim, nb_entries, dtype=torch.float32)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(nb_entries, dtype=torch.float32))
        self.register_buffer("embed_avg", embed.clone())

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x: torch.FloatTensor) -> Tuple[torch.Tensor, torch.Tensor, torch.LongTensor]:
        x = self.conv_in(x.float()).permute(0,2,3,1)
        flatten = x.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*x.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            # TODO: Replace this? Or can we simply comment out?
            # dist_fn.all_reduce(embed_onehot_sum)
            # dist_fn.all_reduce(embed_sum)

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - x).pow(2).mean()
        quantize = x + (quantize - x).detach()

        return quantize.permute(0, 3, 1, 2), diff, embed_ind

    def embed_code(self, embed_id: torch.LongTensor) -> torch.Tensor:
        return F.embedding(embed_id, self.embed.transpose(0, 1))

class Upscaler(HelperModule):
    def build(self,
            embed_dim: int,
            scaling_rates: list[int],
        ):

        self.stages = nn.ModuleList()
        for sr in scaling_rates:
            upscale_steps = int(log2(sr))
            layers = []
            for _ in range(upscale_steps):
                layers.append(nn.ConvTranspose2d(embed_dim, embed_dim, 4, stride=2, padding=1))
                layers.append(nn.BatchNorm2d(embed_dim))
                layers.append(nn.ReLU(inplace=True))
            self.stages.append(nn.Sequential(*layers))

    def forward(self, x: torch.FloatTensor, stage: int) -> torch.FloatTensor:
        return self.stages[stage](x)

"""
    Main VQ-VAE-2 Module, capable of support arbitrary number of levels
    TODO: A lot of this class could do with a refactor. It works, but at what cost?
    TODO: Add disrete code decoding function
"""
class VQVAE(HelperModule):
    def build(self,
            in_channels: int                = 3,
            hidden_channels: int            = 128,
            res_channels: int               = 32,
            nb_res_layers: int              = 2,
            nb_levels: int                  = 3,
            embed_dim: int                  = 64,
            nb_entries: int                 = 512,
            scaling_rates: list[int]        = [8, 4, 2]
        ):
        self.nb_levels = nb_levels
        self.scaling_rates = scaling_rates
        assert len(scaling_rates) == nb_levels, "Number of scaling rates not equal to number of levels!"

        self.encoders = nn.ModuleList([Encoder(in_channels, hidden_channels, res_channels, nb_res_layers, scaling_rates[0])])
        for i, sr in enumerate(scaling_rates[1:]):
            self.encoders.append(Encoder(hidden_channels, hidden_channels, res_channels, nb_res_layers, sr))

        self.codebooks = nn.ModuleList()
        for i in range(nb_levels - 1):
            self.codebooks.append(CodeLayer(hidden_channels+embed_dim, embed_dim, nb_entries))
        self.codebooks.append(CodeLayer(hidden_channels, embed_dim, nb_entries))

        self.decoders = nn.ModuleList([Decoder(embed_dim*nb_levels, hidden_channels, in_channels, res_channels, nb_res_layers, scaling_rates[0])])
        for i, sr in enumerate(scaling_rates[1:]):
            self.decoders.append(Decoder(embed_dim*(nb_levels-1-i), hidden_channels, embed_dim, res_channels, nb_res_layers, sr))

        self.upscalers = nn.ModuleList()
        for i in range(nb_levels - 1):
            self.upscalers.append(Upscaler(embed_dim, scaling_rates[1:len(scaling_rates) - i][::-1]))

    def forward(self, x):
        encoder_outputs : List[torch.Tensor] = []
        code_outputs = []
        decoder_outputs : List[torch.Tensor]  = []
        upscale_counts = []
        id_outputs: List[torch.Tensor] = []
        diffs = []

        for enc in self.encoders:
            if len(encoder_outputs):
                encoder_outputs.append(enc(encoder_outputs[-1]))
            else:
                encoder_outputs.append(enc(x))

        for l in range(self.nb_levels-1, -1, -1):
            codebook = cast(CodeLayer, self.codebooks[l])
            decoder = cast(Decoder, self.decoders[l])

            if len(decoder_outputs): # if we have previous levels to condition on
                code_q, code_d, emb_id = codebook(torch.cat([encoder_outputs[l], decoder_outputs[-1]], dim=1))
            else:
                code_q, code_d, emb_id = codebook(encoder_outputs[l])
            diffs.append(code_d)
            id_outputs.append(emb_id)

            code_outputs = [self.upscalers[i](c, upscale_counts[i]) for i, c in enumerate(code_outputs)]
            upscale_counts = [u+1 for u in upscale_counts]
            decoder_outputs.append(decoder(torch.cat([code_q, *code_outputs], dim=1)))

            code_outputs.append(code_q)
            upscale_counts.append(0)

        return decoder_outputs[-1], diffs, encoder_outputs, decoder_outputs, id_outputs

    def decode_codes(self, *cs):
        decoder_outputs = []
        code_outputs = []
        upscale_counts = []

        for l in range(self.nb_levels - 1, -1, -1):
            codebook = cast(CodeLayer, self.codebooks[l])
            decoder = cast(Decoder, self.decoders[l])
            code_q = codebook.embed_code(cs[l]).permute(0, 3, 1, 2)
            code_outputs = [self.upscalers[i](c, upscale_counts[i]) for i, c in enumerate(code_outputs)]
            upscale_counts = [u+1 for u in upscale_counts]
            decoder_outputs.append(decoder(torch.cat([code_q, *code_outputs], dim=1)))

            code_outputs.append(code_q)
            upscale_counts.append(0)

        return decoder_outputs[-1]
    
    def encode_latent(self, data : torch.Tensor, device="cuda"):
        self.to(device)
        with torch.no_grad():
            ids : List[torch.Tensor]= self(data.to(device))[-1]
        for i in range(len(ids)):
            ids[i] = ids[i].flatten(start_dim=1)
        return torch.cat(ids, dim=-1)

    def get_latent_offset(self, shape : Tuple[int, int]):
        cs = []
        code_acc = 0
        image_height, image_width = shape
        vqvae_shapes = []
        for i in self.scaling_rates:
            image_height //= i
            image_width //= i
            vqvae_shapes.append((image_height, image_width))
        for l in range(self.nb_levels - 1, -1, -1):
            codebook = cast(CodeLayer, self.codebooks[l])
            h, w = vqvae_shapes[l]
            cc = torch.full((h*w,), code_acc)
            cs.append(cc)
            code_acc += codebook.n_embed
        return torch.cat(cs)

    def decode_latent(self, data : torch.Tensor, shape : Tuple[int, int]):
        acc = 0
        cs = []
        image_height, image_width = shape
        vqvae_shapes = []
        for i in self.scaling_rates:
            image_height //= i
            image_width //= i
            vqvae_shapes.append((image_height, image_width))
        total_dim = 0
        for i in vqvae_shapes:
            total_dim += i[0] * i[1]
        print(total_dim, data.shape)
        assert total_dim == data.shape[-1]
        for l in range(self.nb_levels - 1, -1, -1):
            codebook = cast(CodeLayer, self.codebooks[l])
            h, w = vqvae_shapes[l]
            cc = data[:,acc:(acc+w*h)].reshape((-1,h,w))
            # we need to clip the codes here!
            cc = torch.clip(cc, 0, codebook.n_embed-1)
            cs.append(cc)
            acc += w*h
        cs.reverse()
        return self.decode_codes(*cs)
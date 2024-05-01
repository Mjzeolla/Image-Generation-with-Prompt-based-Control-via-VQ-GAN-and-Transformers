#!/usr/bin/env python
# coding: utf-8
import zipfile
from io import BytesIO
from attr import dataclass
import numpy as np
import os
from tomli import TOMLDecodeError
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
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
from dataclasses import dataclass
from typing import Tuple, Dict, List
import functools

class PixelArtDataset(Dataset):
    def __init__(self, data_dir: str, device):
        # Create a ToTensor instance
        with torch.no_grad():
            # self.images = transform(torch.FloatTensor(np.load('data.npy').transpose((0,3,1, 2)) / 255))
            # we must permute here, otherwise the pixels are discontinued...
            labels = np.load(os.path.join(data_dir, "spirit/sprites_labels.npy"))
            raw = np.load(os.path.join(data_dir, "spirit/sprites.npy"))
            # self.data is of shape (b, c * h * w), channel dim is contingent
            self.data = (
                torch.IntTensor(
                    np.reshape(
                        raw.transpose((0, 3, 1, 2)).flatten(),
                        (raw.shape[0], -1),
                    )
                ).to(device),
                torch.IntTensor(np.argmax(labels, -1)).to(device),
            )

    def __getitem__(self, idx):
        return self.data[0][idx], self.data[1][idx]

    def __len__(self):
        return len(self.data[0])


class InGPUDataset(Dataset):
    def __init__(self, dataset, device):
        self.dataset = dataset
        with torch.no_grad():
            self.data = (
                torch.stack([i[0] for i in dataset]).flatten(start_dim=1).to(device),
                torch.IntTensor([i[1] for i in dataset]).to(device),
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return self.data[0][i], self.data[1][i]


class ImageNet(Dataset):
    @torch.no_grad
    def __init__(self, data_dir: str, device):
        images = np.load(os.path.join(data_dir, "tiny-imagenet/image.npy"))
        assert images.shape[1:] == (64, 64, 3)
        tokens = np.load(os.path.join(data_dir, "tiny-imagenet/image_token.npy"))
        labels = np.load(os.path.join(data_dir, "tiny-imagenet/label.npy"))
        # images is in pixel space
        # tokens is in latent space
        self.data = (
            torch.FloatTensor(images)
            .to(device)
            .permute((0, 3, 1, 2))
            .flatten(start_dim=1)
        )
        self.data /= 255
        self.tokens = torch.tensor(tokens).to(device)
        assert len(self.data.shape) == 2
        self.labels = torch.IntTensor(labels).to(device)

    def __len__(self):
        return self.tokens.shape[0]

    def __getitem__(self, i):
        return self.tokens[i], self.labels[i]


class VQVAE_MNIST(Dataset):
    @torch.no_grad
    def __init__(self, raw_dataset: InGPUDataset, data_dir: str, device):
        tokens = np.load(os.path.join(data_dir, "MNIST/image_token.npy"))
        self.labels = raw_dataset.data[1]
        self.tokens = torch.tensor(tokens).to(device)
        assert len(self.labels) == len(self.tokens)

    def __len__(self):
        return self.tokens.shape[0]

    def __getitem__(self, i):
        return self.tokens[i], self.labels[i]


class VQVAE_CIFAR(Dataset):
    @torch.no_grad
    def __init__(self, raw_dataset: InGPUDataset, data_dir: str, device):
        tokens = np.load(os.path.join(data_dir, "cifar-10-batches-py/image_token.npy"))
        self.labels = raw_dataset.data[1]
        self.tokens = torch.tensor(tokens).to(device)
        assert len(self.labels) == len(self.tokens)

    def __len__(self):
        return self.tokens.shape[0]

    def __getitem__(self, i):
        return self.tokens[i], self.labels[i]


class VQVAE_FACE(Dataset):
    @torch.no_grad
    def __init__(self, raw_dataset: InGPUDataset, data_dir: str, device):
        tokens = np.load(os.path.join(data_dir, "face/image_token.npy"))
        self.tokens = torch.tensor(tokens, device=device)

    def __len__(self):
        return self.tokens.shape[0]

    def __getitem__(self, i):
        return self.tokens[i], 1


class VQVAE_Flower(Dataset):
    @torch.no_grad
    def __init__(self, raw_dataset: InGPUDataset, data_dir: str, device):
        tokens = np.load(os.path.join(data_dir, "102 flower/image_token.npy"))
        self.labels = raw_dataset.data[1]
        self.tokens = torch.tensor(tokens).to(device)
        assert len(self.labels) == len(self.tokens)

    def __len__(self):
        return self.tokens.shape[0]

    def __getitem__(self, i):
        return self.tokens[i], self.labels[i]


@dataclass
class CustomDatasetInfo:
    flatten_images: torch.Tensor
    shape: Tuple[int, int, int]
    label_num: int
    label_map: Dict[int, str]
    # whether this is a latent space dataset
    vqvae: bool = False
    labels: torch.Tensor | None = None


import torch
import torchvision.transforms.functional as F


@torch.no_grad
def crop_and_resize_to_square(image, target_size):
    """
    Crops an image tensor to a square shape while maintaining the aspect ratio,
    and then resizes the cropped square image to the target size.
    """
    # Get the dimensions of the image
    height, width = image.shape[-2:]

    # Determine the shorter side
    min_dim = min(height, width)

    # Calculate the crop window
    crop_size = min_dim
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size

    # Crop the image to a square
    image = F.crop(image, top, left, crop_size, crop_size)

    # Resize the cropped square image to the target size
    image = F.resize(image, target_size)

    return image


@torch.no_grad
@functools.cache
def get_data(
    tt, data_dir, downscale=1, device="cuda"
) -> Tuple[Dataset, CustomDatasetInfo]:
    trans = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((28 // downscale, 28 // downscale)),
            v2.ToDtype(torch.int32),
        ]
    )
    if tt == "spirit":
        assert downscale == 1
        ds = PixelArtDataset(data_dir, device)
        m = labels_map = {i: str(i) for i in range(5)}
        return ds, CustomDatasetInfo(
            ds.data[0] / 255, shape=(16, 16, 3), label_num=5, label_map=m, vqvae=False
        )
    elif tt == "mnist":
        data = torchvision.datasets.MNIST(data_dir, download=True, transform=trans)
        ds = InGPUDataset(data, device)
        m = labels_map = {i: str(i) for i in range(10)}
        return ds, CustomDatasetInfo(
            ds.data[0] / 255,
            shape=(28 // downscale, 28 // downscale, 1),
            label_num=len(m),
            label_map=m,
            vqvae=False,
        )
    elif tt == "fashion":
        data = torchvision.datasets.FashionMNIST(
            data_dir, download=True, transform=trans
        )
        ds = InGPUDataset(data, device)
        m = labels_map = {
            0: "T-Shirt",
            1: "Trouser",
            2: "Pullover",
            3: "Dress",
            4: "Coat",
            5: "Sandal",
            6: "Shirt",
            7: "Sneaker",
            8: "Bag",
            9: "Ankle Boot",
        }
        return ds, CustomDatasetInfo(
            ds.data[0] / 255,
            shape=(28 // downscale, 28 // downscale, 1),
            label_num=len(m),
            label_map=m,
            vqvae=False,
        )
    elif tt == "tiny-imagenet":
        ds = ImageNet(data_dir, device)
        #import imagenet_label

        m = {i:str(i) for i in range(200)}
        # no division here
        return ds, CustomDatasetInfo(
            ds.data,
            shape=(64, 64, 3),
            label_num=len(m),
            label_map=m,
            labels=ds.labels,
            vqvae=True,
        )
    elif tt == "vqvae-mnist":
        assert downscale == 1
        data = torchvision.datasets.MNIST(data_dir, download=True, transform=trans)
        ds = InGPUDataset(data, device)
        m = {i: str(i) for i in range(10)}
        return VQVAE_MNIST(ds, data_dir, device), CustomDatasetInfo(
            ds.data[0] / 255,
            shape=(28 // downscale, 28 // downscale, 1),
            label_num=len(m),
            label_map=m,
            labels=ds.data[1],
            vqvae=True,
        )
    elif tt == "vqvae-face":
        assert downscale == 1
        image_size = (64, 64)
        # no normalizationn here!!!
        full_dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(data_dir, "face"),
            transform=transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                ]
            ),
        )
        ds = InGPUDataset(full_dataset, device)
        m = {0: "0"}
        return VQVAE_FACE(ds, data_dir, device), CustomDatasetInfo(
            ds.data[0],
            shape=(64, 64, 3),
            label_num=len(m),
            label_map=m,
            labels=None,
            vqvae=True,
        )
    elif tt == "vqvae-flower":
        assert downscale == 1
        image_size = (64, 64)
        # no normalizationn here!!!
        full_dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(data_dir, "102 flower/flowers/train"),
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Lambda(
                        lambda x: crop_and_resize_to_square(x, image_size)
                    ),
                ]
            ),
        )
        ds = InGPUDataset(full_dataset, device)
        m = [
            "pink primrose",
            "hard-leaved pocket orchid",
            "canterbury bells",
            "sweet pea",
            "english marigold",
            "tiger lily",
            "moon orchid",
            "bird of paradise",
            "monkshood",
            "globe thistle",
            "snapdragon",
            "colts foot",
            "king protea",
            "spear thistle",
            "yellow iris",
            "globe-flower",
            "purple coneflower",
            "peruvian lily",
            "balloon flower",
            "giant white arum lily",
            "fire lily",
            "pincushion flower",
            "fritillary",
            "red ginger",
            "grape hyacinth",
            "corn poppy",
            "prince of wales feathers",
            "stemless gentian",
            "artichoke",
            "sweet william",
            "carnation",
            "garden phlox",
            "love in the mist",
            "mexican aster",
            "alpine sea holly",
            "ruby-lipped cattleya",
            "cape flower",
            "great masterwort",
            "siam tulip",
            "lenten rose",
            "barbeton daisy",
            "daffodil",
            "sword lily",
            "poinsettia",
            "bolero deep blue",
            "wallflower",
            "marigold",
            "buttercup",
            "oxeye daisy",
            "common dandelion",
            "petunia",
            "wild pansy",
            "primula",
            "sunflower",
            "pelargonium",
            "bishop of llandaff",
            "gaura",
            "geranium",
            "orange dahlia",
            "pink-yellow dahlia?",
            "cautleya spicata",
            "japanese anemone",
            "black-eyed susan",
            "silverbush",
            "californian poppy",
            "osteospermum",
            "spring crocus",
            "bearded iris",
            "windflower",
            "tree poppy",
            "gazania",
            "azalea",
            "water lily",
            "rose",
            "thorn apple",
            "morning glory",
            "passion flower",
            "lotus",
            "toad lily",
            "anthurium",
            "frangipani",
            "clematis",
            "hibiscus",
            "columbine",
            "desert-rose",
            "tree mallow",
            "magnolia",
            "cyclamen ",
            "watercress",
            "canna lily",
            "hippeastrum ",
            "bee balm",
            "ball moss",
            "foxglove",
            "bougainvillea",
            "camellia",
            "mallow",
            "mexican petunia",
            "bromelia",
            "blanket flower",
            "trumpet creeper",
            "blackberry lily"
        ]
        # remap the label
        newm = ["" for i in m]
        idx_to_class = {}
        for m_id in full_dataset.class_to_idx:
            current_id = full_dataset.class_to_idx[m_id]
            idx_to_class[current_id] = m_id
            newm[current_id] = m[int(m_id)-1]
        m = newm
        m = {i:f"{idx_to_class[i]}-{m[i]}" for i in range(len(m))}
        assert len(m) == 102
        return VQVAE_Flower(ds, data_dir, device), CustomDatasetInfo(
            ds.data[0],
            shape=(64, 64, 3),
            label_num=len(m),
            label_map=m,
            labels=None,
            vqvae=True,
        )
    elif tt == "vqvae-cifar":
        trans = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.int32),
            ]
        )
        assert downscale == 1
        data = torchvision.datasets.CIFAR10(data_dir, download=True, transform=trans)
        ds = InGPUDataset(data, device)
        m = cifar10_label_map = {
            0: "airplane",
            1: "automobile",
            2: "bird",
            3: "cat",
            4: "deer",
            5: "dog",
            6: "frog",
            7: "horse",
            8: "ship",
            9: "truck",
        }
        return VQVAE_CIFAR(ds, data_dir, device), CustomDatasetInfo(
            ds.data[0] / 255,
            shape=(32, 32, 3),
            label_num=len(m),
            label_map=m,
            labels=ds.data[1],
            vqvae=True,
        )
    else:
        raise Exception("not a valida dataset")

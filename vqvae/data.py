import torch.utils
import numpy as np
import torch
import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset, random_split
import functools

class InGPUDataset(Dataset):
    def __init__(self, dataset, device="cuda"):
        self.dataset = dataset
        images = []
        labels = []
        for (img, label) in dataset:
            images.append(img)
            labels.append(label)
        self.data = (torch.stack(images).to(device), torch.tensor(labels).to(device))
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, i):
        return self.data[0][i], self.data[1][i]

def load_cifar(root):
    train = datasets.CIFAR10(root=root, train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize(
                                     (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ]))

    val = datasets.CIFAR10(root=root, train=False, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(
                                   (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           ]))
    return train, val

def load_mnist(root):
    train = datasets.MNIST(root=root, train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize(
                                     0.5,0.5)
                             ]))

    val = datasets.MNIST(root=root, train=False, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(
                                   0.5,0.5)
                           ]))
    return train, val

class ImageNet(Dataset):
    def __init__(self, dir, device):
        image_dir = os.path.join(dir, "tiny-imagenet/image.npy")
        label_dir = os.path.join(dir, "tiny-imagenet/label.npy")
        self.images = torch.tensor(np.load(image_dir)).to(device)
        with torch.no_grad():
            self.images = (2/255)*self.images.permute((0, 3,1,2)) - 1 
        torch.cuda.empty_cache()
        self.labels = torch.tensor(np.load(label_dir)).to(device)
    def __len__(self):
        return self.images.shape[0]
    def __getitem__(self, i):
        return self.images[i], self.labels[i]


def data_loaders(train_data, val_data, batch_size):

    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True)
    val_loader = DataLoader(val_data,
                            batch_size=batch_size,
                            shuffle=False)
    return train_loader, val_loader

def load_data_and_data_loaders_imagenet(dir, batch_size):
    full_dataset = ImageNet(dir, 
                            device="cuda")
    #train_size = int(0.95 * len(full_dataset))
    #test_size = len(full_dataset) - train_size
    training_data = full_dataset
    #training_data, validation_data = random_split(full_dataset, [train_size, test_size])
    x_train_var = 1
    #training_data = InGPUDataset(training_data)
    #validation_data = InGPUDataset(validation_data)
    validation_data = None
    validation_loader = None
    training_loader = DataLoader(training_data,
                              batch_size=batch_size,
                              shuffle=True)
    return training_data, validation_data, training_loader, validation_loader, x_train_var

@torch.no_grad
def load_face(dir, batch_size):
    # Create the dataset
    image_size = (64,64)
    import torchvision
    full_dataset = torchvision.datasets.ImageFolder(root=os.path.join(dir, "face"),
                            transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    train_size = int(0.95 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    training_data, validation_data = random_split(full_dataset, [train_size, test_size])
    x_train_var = 1
    training_data = InGPUDataset(training_data)
    validation_data = InGPUDataset(validation_data)
    training_loader, validation_loader = data_loaders(
        training_data, validation_data, batch_size)
    return training_data, validation_data, training_loader, validation_loader, x_train_var, (64,64,3)

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
def load_flower(dir, batch_size):
    # Create the dataset
    image_size = (64,64)
    import torchvision
    training_data = torchvision.datasets.ImageFolder(root=os.path.join(dir, "102 flower/flowers/train"),
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Lambda(lambda x : crop_and_resize_to_square(x, image_size)),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    validation_data = torchvision.datasets.ImageFolder(root=os.path.join(dir, "102 flower/flowers/test"),
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Lambda(lambda x : crop_and_resize_to_square(x, image_size)),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    x_train_var = 1
    training_data = InGPUDataset(training_data)
    validation_data = InGPUDataset(validation_data)
    training_loader, validation_loader = data_loaders(
        training_data, validation_data, batch_size)
    return training_data, validation_data, training_loader, validation_loader, x_train_var, (image_size[0],image_size[1],3)

class FFHQDataset(Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset['train'].with_format("torch")
    def __len__(self):
        return len(self.dataset)
    @torch.no_grad
    def __getitem__(self, index):
        item = self.dataset[index]
        return ((item["image"].permute((2,0,1)) /255)*2 - 1, item["label"])
    
@torch.no_grad
def load_ffhq_face(dir, batch_size):
    from datasets import load_dataset
    dataset = load_dataset("Dmini/FFHQ-64x64")
    training_data = FFHQDataset(dataset)
    training_loader = DataLoader(training_data,
                              batch_size=batch_size,
                              shuffle=True)
    validation_loader = None
    validation_data = None
    x_train_var = 1.0
    return training_data, validation_data, training_loader, validation_loader, x_train_var, (64, 64, 3)

@torch.no_grad
@functools.cache
def load_data(dt, root, batch_size):
    x_train_var = 1
    if dt == "mnist":
        train, val = load_mnist(root)
        training_data = InGPUDataset(train)
        validation_data = InGPUDataset(val)
        training_loader, validation_loader = data_loaders(
        training_data, validation_data, batch_size)
        return training_data, validation_data, training_loader, validation_loader, x_train_var, (28, 28, 1)
    elif dt == "imagenet":
        shape = (64,64,3)
        return *load_data_and_data_loaders_imagenet(root, batch_size), shape
    elif dt == "cifar":
        train, val = load_cifar(root)
        training_loader, validation_loader = data_loaders(train, val, batch_size)
        return train, val, training_loader, validation_loader, x_train_var, (32,32,3)
    elif dt == "flower":
        return load_flower(root, batch_size)
    elif dt == "face":
        return load_face(root, batch_size)
    elif dt == "ffhq":
        return load_ffhq_face(root, batch_size)
    else:
        assert False

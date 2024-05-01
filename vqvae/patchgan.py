import torch
import torch.nn as nn

class PatchGAN(nn.Module):
    # initializers
    def __init__(self, d=64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.LazyConv2d(d, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.LazyConv2d(d * 2, 4, 2, 1),
            nn.LazyBatchNorm2d(),
            nn.LeakyReLU(0.2),
            nn.LazyConv2d(d * 4, 4, 1, 1),
            nn.LazyBatchNorm2d(),
            nn.LeakyReLU(0.2),
            nn.LazyConv2d(1, 4, 1, 1),
            nn.Sigmoid(),
        )
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=2e-4, betas=(0.5, 0.999)
        )

    # forward method
    def forward(self, input):
        return self.layers(input)
    
def train_patchgan(model: PatchGAN, x_real: torch.Tensor, x_fake: torch.Tensor):
    model.train()
    model.optimizer.zero_grad(set_to_none=True)
    x_real = model(x_real)
    x_fake = model(x_fake)
    loss_func = torch.nn.BCELoss()
    bs = x_real.shape[0]
    num_patches = x_real.shape[2] * x_real.shape[3]
    x_real = x_real.flatten(start_dim=0)
    x_fake = x_fake.flatten(start_dim=0)
    ll = bs * num_patches
    real_target = torch.full((ll,), 1.0).to(x_real.device)
    fake_target = torch.full((ll,), 0.0).to(x_real.device)
    loss = loss_func(x_real, real_target) + loss_func(x_fake, fake_target)
    loss.backward()
    model.optimizer.step()
    return loss


def gan_loss(model: PatchGAN, x_fake: torch.Tensor):
    model.train()
    x_fake = model(x_fake)
    loss_func = torch.nn.BCELoss()
    bs = x_fake.shape[0]
    num_patches = x_fake.shape[2] * x_fake.shape[3]
    x_fake = x_fake.flatten(start_dim=0)
    ll = bs * num_patches
    real_target = torch.full((ll,), 1.0).to(x_fake.device)
    loss = loss_func(x_fake, real_target)
    # per patch loss
    return loss

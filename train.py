import os

import torch
from torch import nn, optim
from torch.utils import data
from torchvision import transforms
from tensorfn import load_arg_config, load_wandb
from tensorfn import distributed as dist
from tensorfn.optim import lr_scheduler
from tqdm import tqdm

from model import UNet
from diffusion import GaussianDiffusion, make_beta_schedule
from dataset import MultiResolutionDataset
from config import DiffusionConfig

'''
training process
'''

def sample_data(loader):

    loader_iter = iter(loader)
    epoch = 0

    # generator code
    while True:
        try:
            yield epoch, next(loader_iter)

        except StopIteration:
            epoch += 1
            loader_iter = iter(loader)

            yield epoch, next(loader_iter)


def accumulate(model1, model2, decay=0.9999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def train(conf, loader, model, ema, diffusion, optimizer, scheduler, device, wandb):
    # get training data
    loader = sample_data(loader)

    # training steps visualization
    pbar = range(conf.training.n_iter + 1)

    if dist.is_primary():
        pbar = tqdm(pbar, dynamic_ncols=True)

    # training loop
    for i in pbar:
        # get image
        epoch, img = next(loader)
        img = img.to(device)

        # random generate a step [t]
        time = torch.randint(
            0,
            conf.diffusion.beta_schedule["n_timestep"],
            (img.shape[0],),
            device=device,
        )

        # calculate loss with step [t]
        loss = diffusion.p_loss(model, img, time)
        
        # backward propagation
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        scheduler.step()
        optimizer.step()

        accumulate(
            ema, model.module, 0 if i < conf.training.scheduler.warmup else 0.9999
        )

        if dist.is_primary():
            # logger
            lr = optimizer.param_groups[0]["lr"]
            pbar.set_description(
                f"epoch: {epoch}; loss: {loss.item():.4f}; lr: {lr:.5f}"
            )

            if wandb is not None and i % conf.evaluate.log_every == 0:
                wandb.log({"epoch": epoch, "loss": loss.item(), "lr": lr}, step=i)

            if i % conf.evaluate.save_every == 0:
                if conf.distributed:
                    model_module = model.module

                else:
                    model_module = model
                # save model
                torch.save(
                    {
                        "model": model_module.state_dict(),
                        "ema": ema.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "conf": conf,
                    },
                    f"checkpoint/diffusion_{str(i).zfill(6)}.pt",
                )


def main(conf):
    # wandb configuration
    wandb = None
    if dist.is_primary() and conf.evaluate.wandb:
        wandb = load_wandb()
        wandb.init(project="denoising diffusion")

    # cuda device visualization, cuda only
    device = "cuda"
    beta_schedule = "linear"

    # check if multi-gpus training
    conf.distributed = dist.get_world_size() > 1

    # data preprocessing, including reinforement learning
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    # training dataset path
    train_set = MultiResolutionDataset(
        conf.dataset.path, transform, conf.dataset.resolution
    )
    # sampling dataset path
    train_sampler = dist.data_sampler(
        train_set, shuffle=True, distributed=conf.distributed
    )
    # define dataloader
    train_loader = conf.training.dataloader.make(train_set, sampler=train_sampler)

    # init model and model ema
    model = conf.model.make()
    model = model.to(device)
    ema = conf.model.make()
    ema = ema.to(device)

    # multi-gpus setting
    if conf.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.get_local_rank()],
            output_device=dist.get_local_rank(),
        )

    # training optimizer
    optimizer = conf.training.optimizer.make(model.parameters())
    scheduler = conf.training.scheduler.make(optimizer)

    # load former model if existed
    if conf.ckpt is not None:
        ckpt = torch.load(conf.ckpt, map_location=lambda storage, loc: storage)

        if conf.distributed:
            model.module.load_state_dict(ckpt["model"])

        else:
            model.load_state_dict(ckpt["model"])

        ema.load_state_dict(ckpt["ema"])

    # diffusion algorithm
    betas = conf.diffusion.beta_schedule.make()
    diffusion = GaussianDiffusion(betas).to(device)

    # training
    train(
        conf, train_loader, model, ema, diffusion, optimizer, scheduler, device, wandb
    )


if __name__ == "__main__":
    #data type definition and basic machine learning configuration
    conf = load_arg_config(DiffusionConfig)

    dist.launch(
        main, conf.n_gpu, conf.n_machine, conf.machine_rank, conf.dist_url, args=(conf,)
    )

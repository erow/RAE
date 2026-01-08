"""
Training script for β-Diffusion.

β-Diffusion learns high-level visual representations through a noise-conditioned 
VAE architecture with DiT-based modulation.
"""
import argparse
import logging
import math
import os
from collections import defaultdict, OrderedDict
from typing import Dict, Optional, Tuple

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
from pathlib import Path
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR
from omegaconf import OmegaConf, DictConfig

##### model imports
from stage1.beta_diffusion import BetaDiffusion, create_beta_diffusion

##### general utils
from utils import wandb_utils
from utils.model_utils import instantiate_from_config
from utils.train_utils import (
    center_crop_arr,
    update_ema,
    prepare_dataloader,
    get_autocast_scaler,
)
from utils.optim_utils import build_optimizer, build_scheduler
from utils.resume_utils import configure_experiment_dirs, find_resume_checkpoint, save_worktree
from utils.wandb_utils import create_logger
from utils.dist_utils import setup_distributed, cleanup_distributed


def parse_beta_diffusion_configs(config: DictConfig) -> Tuple[DictConfig, DictConfig, DictConfig]:
    """Load a config file and return component sections as DictConfigs."""
    model_config = config.get("model", None)
    training_config = config.get("training", None)
    data_config = config.get("data", None)
    return model_config, training_config, data_config


def save_checkpoint(
    path: str,
    step: int,
    epoch: int,
    model: DDP,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
) -> None:
    state = {
        "step": step,
        "epoch": epoch,
        "model": model.module.state_dict(),
        "ema": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(
    path: str,
    model: DDP,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
) -> Tuple[int, int]:
    checkpoint = torch.load(path, map_location="cpu")
    model.module.load_state_dict(checkpoint["model"])
    ema_model.load_state_dict(checkpoint["ema"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint.get("epoch", 0), checkpoint.get("step", 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train β-Diffusion model.")
    parser.add_argument("--config", type=str, required=True, help="YAML config file.")
    parser.add_argument("--data-path", type=Path, required=True, help="Directory with ImageFolder structure for training.")
    parser.add_argument("--results-dir", type=str, default="ckpts", help="Directory to store training outputs.")
    parser.add_argument("--image-size", type=int, default=256, help="Input image resolution.")
    parser.add_argument("--precision", type=str, choices=["fp32", "fp16", "bf16"], default="fp32", help="Compute precision for training.")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.", default=False)
    parser.add_argument("--compile", action="store_true", help="Use torch compile.", default=False)
    parser.add_argument("--ckpt", type=str, default=None, help="Optional checkpoint path to resume training.")
    parser.add_argument("--global-seed", type=int, default=None, help="Override training.global_seed from the config.")
    
    parser.add_argument("--local-rank", type=int, default=0, help="Local rank for distributed training.")
    args = parser.parse_args()
    return args


@torch.no_grad()
def visualize_reconstructions(
    model: torch.nn.Module,
    images: torch.Tensor,
    save_path: str,
    epoch: int,
    beta_values: list,
    logger: logging.Logger,
    use_wandb: bool = True,
):
    """Save visualization of reconstructions at different β values."""
    import torchvision.utils as vutils
    
    model.eval()
    device = next(model.parameters()).device
    images = images[:8].to(device)  # Take first 8 images
    
    # Get reconstructions at different β values
    recons = model.interpolate_beta(images, beta_values)  # (B, num_betas, C, H, W)
    
    # Create grid: original | recon@β=1 | recon@β=10 | ... 
    B, num_betas, C, H, W = recons.shape
    
    # Prepare grid
    grid_images = [images]
    for i in range(num_betas):
        grid_images.append(recons[:, i])
    
    grid = torch.cat(grid_images, dim=0)  # (B * (1 + num_betas), C, H, W)
    grid = vutils.make_grid(grid.clamp(0, 1), nrow=B, padding=2)
    
    # Save
    vis_path = os.path.join(save_path, f'recon_epoch{epoch:04d}.jpg')
    vutils.save_image(grid, vis_path)
    logger.info(f"Saved reconstruction visualization to {vis_path}")
    
    if use_wandb and wandb_utils.is_main_process():
        wandb_utils.log({
            'visualizations/recon_epoch': wandb_utils.wandb.Image(vis_path),
            'epoch': epoch,
        })

def evaluate_model(model: torch.nn.Module, val_loader: DataLoader, epoch: int,
                   vis_dir: str, logger: logging.Logger, beta_min: float, beta_max: float):
    model = model.module if isinstance(model, DDP) else model
    model.eval()
    
    logger.info("Generating EMA reconstructions...")
    sample_batch = next(iter(val_loader))
    beta_values = torch.linspace(beta_min, beta_max, 4).tolist()
    visualize_reconstructions(
        model, sample_batch[0], vis_dir, epoch, beta_values, logger, use_wandb=True
    )
    model.train()
    
def main():
    """Train β-Diffusion model using config-driven hyperparameters."""
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Training currently requires at least one GPU.")
    
    rank, world_size, device = setup_distributed()
    full_cfg = OmegaConf.load(args.config)
    model_config, training_config, data_config = parse_beta_diffusion_configs(full_cfg)
    
    def to_dict(cfg_section):
        if cfg_section is None:
            return {}
        return OmegaConf.to_container(cfg_section, resolve=True)
    
    model_cfg = to_dict(model_config)
    training_cfg = to_dict(training_config)
    data_cfg = to_dict(data_config)
    
    # Data config
    image_size = int(data_cfg.get("image_size", args.image_size))
    num_workers = int(data_cfg.get("num_workers", 4))
    
    # Training config
    grad_accum_steps = int(training_cfg.get("grad_accum_steps", 1))
    if grad_accum_steps < 1:
        raise ValueError("Gradient accumulation steps must be >= 1.")
    clip_grad_val = training_cfg.get("clip_grad", 1.0)
    clip_grad = float(clip_grad_val) if clip_grad_val is not None else None
    if clip_grad is not None and clip_grad <= 0:
        clip_grad = None
    ema_decay = float(training_cfg.get("ema_decay", 0.9999))
    num_epochs = int(training_cfg.get("epochs", 100))
    
    global_batch_size = training_cfg.get("global_batch_size", None)
    if global_batch_size is not None:
        global_batch_size = int(global_batch_size)
        assert global_batch_size % world_size == 0, "global_batch_size must be divisible by world_size"
    else:
        batch_size = int(training_cfg.get("batch_size", 64))
        global_batch_size = batch_size * world_size * grad_accum_steps
    
    log_interval = int(training_cfg.get("log_interval", 100))
    sample_every = int(training_cfg.get("sample_every", 500))
    checkpoint_interval = int(training_cfg.get("checkpoint_interval", 5))  # epoch based
    default_seed = int(training_cfg.get("global_seed", 0))
    
    # β-Diffusion specific
    beta_min = float(model_cfg.get("beta_min", 1.0))
    beta_max = float(model_cfg.get("beta_max", 100.0))
    
    global_seed = args.global_seed if args.global_seed is not None else default_seed
    seed = global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    micro_batch_size = global_batch_size // (world_size * grad_accum_steps)
    
    ### AMP init
    scaler, autocast_kwargs = get_autocast_scaler(args)
    
    experiment_dir, checkpoint_dir, logger = configure_experiment_dirs(args, rank)
    vis_dir = os.path.join(experiment_dir, "visualizations")
    if rank == 0:
        os.makedirs(vis_dir, exist_ok=True)
    
    #### Model init
    logger.info(f"Creating β-Diffusion model...")
    model: BetaDiffusion = create_beta_diffusion(**model_cfg).to(device)
    
    if args.compile:
        try:
            model.forward = torch.compile(model.forward)
            logger.info("Successfully compiled model.forward")
        except Exception as e:
            logger.warning(f"Model compile failed, falling back to no compile: {e}")
    
    ema_model = deepcopy(model).to(device)
    ema_model.requires_grad_(False)
    ema_model.eval()
    # model.requires_grad_(True)
    ddp_model = DDP(model, device_ids=[device.index], broadcast_buffers=False, find_unused_parameters=True)
    model = ddp_model.module
    ddp_model.train()
    
    model_param_count = sum(p.numel() for p in model.parameters())
    trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model Parameters: {model_param_count/1e6:.2f}M (Trainable: {trainable_param_count/1e6:.2f}M)")
    
    #### Optimizer and Scheduler init
    optimizer, optim_msg = build_optimizer([p for p in model.parameters() if p.requires_grad], training_cfg)
    
    ### Data init
    beta_diffusion_transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    loader, sampler = prepare_dataloader(
        args.data_path, micro_batch_size, num_workers, rank, world_size, transform=beta_diffusion_transform
    )
    
    val_transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
        transforms.ToTensor(),
    ])
    val_path = args.data_path.parent / "val" if (args.data_path.parent / "val").exists() else args.data_path
    val_dataset = ImageFolder(str(val_path), transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=micro_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    loader_batches = len(loader)
    if loader_batches % grad_accum_steps != 0:
        logger.warning("Number of loader batches not divisible by grad_accum_steps, some samples may be dropped.")
    steps_per_epoch = loader_batches // grad_accum_steps
    if steps_per_epoch <= 0:
        raise ValueError("Gradient accumulation configuration results in zero optimizer steps per epoch.")
    
    scheduler = None
    sched_msg = "No LR scheduler."
    if training_cfg.get("scheduler"):
        scheduler, sched_msg = build_scheduler(optimizer, steps_per_epoch, training_cfg)
    
    ### Resuming and checkpointing
    start_epoch = 0
    global_step = 0
    try:
        maybe_resume_ckpt_path = find_resume_checkpoint(experiment_dir)
        if maybe_resume_ckpt_path is not None:
            logger.info(f"Experiment resume checkpoint found at {maybe_resume_ckpt_path}, automatically resuming...")
            ckpt_path = Path(maybe_resume_ckpt_path)
            if ckpt_path.is_file():
                start_epoch, global_step = load_checkpoint(
                    str(ckpt_path),
                    ddp_model,
                    ema_model,
                    optimizer,
                    scheduler,
                )
                logger.info(f"[Rank {rank}] Resumed from {ckpt_path} (epoch={start_epoch}, step={global_step}).")
            else:
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        else:
            # starting from fresh, save worktree and configs
            if rank == 0:
                save_worktree(experiment_dir, full_cfg)
                logger.info(f"Saved training worktree and config to {experiment_dir}.")
    except ValueError:
        # No checkpoint directory found, starting fresh
        if rank == 0:
            save_worktree(experiment_dir, full_cfg)
            logger.info(f"Saved training worktree and config to {experiment_dir}.")
    
    ### Logging experiment details
    if rank == 0:
        logger.info(f"β-Diffusion Model: β_min={beta_min}, β_max={beta_max}")
        if clip_grad is not None:
            logger.info(f"Clipping gradients to max norm {clip_grad}.")
        else:
            logger.info("Not clipping gradients.")
        logger.info(optim_msg)
        logger.info(sched_msg)
        logger.info(f"Training for {num_epochs} epochs, batch size {micro_batch_size} per GPU.")
        logger.info(f"Dataset contains {len(loader.dataset)} samples, {steps_per_epoch} steps per epoch.")
        logger.info(f"Running with world size {world_size}, starting from epoch {start_epoch} to {num_epochs}.")
    
    log_steps = 0
    running_loss = 0.0
    running_recon_loss = 0.0
    running_kl_loss = 0.0
    start_time = time()
    
    dist.barrier()
    for epoch in range(start_epoch, num_epochs):
        ddp_model.train()
        sampler.set_epoch(epoch)
        epoch_metrics: Dict[str, torch.Tensor] = defaultdict(lambda: torch.zeros(1, device=device))
        num_batches = 0
        optimizer.zero_grad()
        
        if checkpoint_interval > 0 and epoch % checkpoint_interval == 0 and rank == 0:
            logger.info(f"Saving checkpoint at epoch {epoch}...")
            ckpt_path = f"{checkpoint_dir}/ep-{epoch:07d}.pt"
            save_checkpoint(
                ckpt_path,
                global_step,
                epoch,
                ddp_model,
                ema_model,
                optimizer,
                scheduler,
            )
        
        for step, (images, _) in enumerate(loader):
            images = images.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            with autocast(**autocast_kwargs):
                outputs = ddp_model(images)
                loss = outputs['loss']
            
            loss.float()
            if scaler:
                scaler.scale(loss / grad_accum_steps).backward()
            else:
                (loss / grad_accum_steps).backward()
            
            if clip_grad:
                if scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), clip_grad)
            
            if global_step % grad_accum_steps == 0:
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                update_ema(ema_model, ddp_model.module, decay=ema_decay)
            
            running_loss += loss.item()
            running_recon_loss += outputs['recon_loss'].item()
            running_kl_loss += outputs['kl_loss'].item()
            epoch_metrics['loss'] += loss.detach()
            epoch_metrics['recon_loss'] += outputs['recon_loss'].detach()
            epoch_metrics['kl_loss'] += outputs['kl_loss'].detach()
            
            if log_interval > 0 and global_step % log_interval == 0 and rank == 0:
                avg_loss = running_loss / log_interval
                avg_recon = running_recon_loss / log_interval
                avg_kl = running_kl_loss / log_interval
                avg_beta = outputs['beta'].mean().item()
                stats = {
                    "train/loss": avg_loss,
                    "train/recon_loss": avg_recon,
                    "train/kl_loss": avg_kl,
                    "train/beta_mean": avg_beta,
                    "train/lr": optimizer.param_groups[0]["lr"],
                }
                logger.info(
                    f"[Epoch {epoch} | Step {global_step}] "
                    + ", ".join(f"{k}: {v:.4f}" for k, v in stats.items())
                )
                if args.wandb:
                    wandb_utils.log(stats, step=global_step)
                running_loss = 0.0
                running_recon_loss = 0.0
                running_kl_loss = 0.0
            
            if sample_every > 0 and global_step % sample_every == 0 and rank == 0:
                logger.info("Generating reconstructions...")
                sample_batch = next(iter(val_loader))
                beta_values = torch.linspace(beta_min, beta_max, 4).tolist()
                visualize_reconstructions(
                    model, sample_batch[0], vis_dir, epoch, beta_values, logger, use_wandb=False
                )
                ddp_model.train()
            
            global_step += 1
            num_batches += 1
        
        if rank == 0 and num_batches > 0:
            avg_loss = epoch_metrics['loss'].item() / num_batches
            avg_recon = epoch_metrics['recon_loss'].item() / num_batches
            avg_kl = epoch_metrics['kl_loss'].item() / num_batches
            epoch_stats = {
                "epoch/loss": avg_loss,
                "epoch/recon_loss": avg_recon,
                "epoch/kl_loss": avg_kl,
            }
            logger.info(
                f"[Epoch {epoch}] "
                + ", ".join(f"{k}: {v:.4f}" for k, v in epoch_stats.items())
            )
            if args.wandb:
                wandb_utils.log(epoch_stats, step=global_step)
        evaluate_model(ema_model, val_loader, epoch, vis_dir, logger, beta_min, beta_max)
    # save the final ckpt
    if rank == 0:
        logger.info(f"Saving final checkpoint at epoch {num_epochs}...")
        ckpt_path = f"{checkpoint_dir}/ep-last.pt"
        save_checkpoint(
            ckpt_path,
            global_step,
            num_epochs,
            ddp_model,
            ema_model,
            optimizer,
            scheduler,
        )
    
    dist.barrier()
    logger.info("Done!")
    cleanup_distributed()


if __name__ == "__main__":
    main()

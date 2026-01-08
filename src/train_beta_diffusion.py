"""
Training script for β-Diffusion.

β-Diffusion learns high-level visual representations through a noise-conditioned 
VAE architecture with DiT-based modulation.
"""

import os
import sys
import argparse
import yaml
import math
import wandb
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import torchvision.transforms as T
from torchvision.datasets import ImageFolder

try:
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data.distributed import DistributedSampler
    HAS_DISTRIBUTED = True
except ImportError:
    HAS_DISTRIBUTED = False

from stage1.beta_diffusion import BetaDiffusion, create_beta_diffusion


def get_args():
    parser = argparse.ArgumentParser(description='Train β-Diffusion')
    
    # Data
    parser.add_argument('--data-path', type=str, required=True, help='Path to ImageNet or similar dataset')
    parser.add_argument('--image-size', type=int, default=224, help='Input image size')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size per GPU')
    parser.add_argument('--num-workers', type=int, default=8, help='DataLoader workers')
    
    # Model
    parser.add_argument('--encoder', type=str, default='dinov2-base', 
                        choices=['dinov2-base', 'dinov2-large', 'mae-base', 'mae-large'], help='Encoder backbone')
    parser.add_argument('--modulator-depth', type=int, default=4, help='Number of DiT blocks in modulator')
    parser.add_argument('--decoder-layers', type=int, default=4, help='Number of MLP layers in decoder')
    parser.add_argument('--decoder-hidden-dim', type=int, default=1024, help='Decoder hidden dimension')
    parser.add_argument('--decoder-patch-size', type=int, default=16, help='Decoder patch size')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.05, help='Weight decay')
    parser.add_argument('--warmup-epochs', type=int, default=5, help='Warmup epochs')
    parser.add_argument('--min-lr', type=float, default=1e-6, help='Minimum learning rate')
    
    # β-Diffusion specific
    parser.add_argument('--beta-min', type=float, default=1.0, help='Minimum β value')
    parser.add_argument('--beta-max', type=float, default=100.0, help='Maximum β value')
    parser.add_argument('--kl-weight', type=float, default=1.0, help='Additional KL weight multiplier')
    
    # Logging & Checkpointing
    parser.add_argument('--output-dir', type=str, default='./outputs/beta_diffusion', help='Output directory')
    parser.add_argument('--log-interval', type=int, default=100, help='Log every N steps')
    parser.add_argument('--save-interval', type=int, default=5, help='Save checkpoint every N epochs')
    parser.add_argument('--wandb', action='store_true', help='Use Weights & Biases logging', default=False)
    parser.add_argument('--wandb-project', type=str, default='beta-diffusion', help='W&B project name')
    parser.add_argument('--exp-name', type=str, default=None, help='Experiment name')
    
    # Resume
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    # Distributed
    parser.add_argument('--local-rank', type=int, default=0)
    
    # Config file (overrides command line args)
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file')
    
    args = parser.parse_args()
    
    # Load config file if provided
    if args.config is not None:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        for key, value in config.items():
            # Set all config values, including nested dicts like 'model'
            setattr(args, key, value)
    
    return args


def setup_distributed():
    """Initialize distributed training."""
    if not HAS_DISTRIBUTED:
        return 0, 1, False
    
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        
        return rank, world_size, True
    else:
        return 0, 1, False


def get_transforms(image_size: int, is_train: bool = True):
    """Get data transforms."""
    if is_train:
        return T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.8, 1.0), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ])
    else:
        return T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.ToTensor(),
        ])


def create_dataloader(args, is_train: bool = True, distributed: bool = False, rank: int = 0, world_size: int = 1):
    """Create training/validation dataloader."""
    transform = get_transforms(args.image_size, is_train)
    
    split = 'train' if is_train else 'val'
    data_path = os.path.join(args.data_path, split)
    
    if not os.path.exists(data_path):
        # Try without split subdirectory
        data_path = args.data_path
    
    dataset = ImageFolder(data_path, transform=transform)
    
    sampler = None
    if distributed and is_train:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None and is_train),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=is_train,
    )
    
    return loader, sampler


def get_lr_scheduler(optimizer, args, steps_per_epoch: int):
    """Create learning rate scheduler with warmup."""
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return args.min_lr / args.lr + (1 - args.min_lr / args.lr) * 0.5 * (1 + math.cos(math.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def visualize_reconstructions(model, images, save_path, epoch, beta_values=[1, 10, 50, 100]):
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
    vutils.save_image(grid, os.path.join(save_path, f'recon_epoch{epoch:04d}.png'))
    if wandb.is_initialized():
        wandb.log({
            'visualizations/recon_epoch': 
                wandb.Image(os.path.join(save_path, f'recon_epoch{epoch:04d}.png')),
            'epoch': epoch,
        })
    
    model.train()


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    args,
    rank: int = 0,
    global_step: int = 0,
):
    """Train for one epoch."""
    model.train()
    device = next(model.parameters()).device
    
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    
    for batch_idx, (images, _) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        
        # Forward pass
        outputs = model(images)
        
        # Apply additional KL weight if specified
        loss = outputs['recon_loss'] + args.kl_weight * outputs['beta'].mean() * outputs['kl_loss']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        # Accumulate losses
        total_loss += loss.item()
        total_recon_loss += outputs['recon_loss'].item()
        total_kl_loss += outputs['kl_loss'].item()
        
        global_step += 1
        
        # Logging
        if batch_idx % args.log_interval == 0 and rank == 0:
            lr = scheduler.get_last_lr()[0]
            avg_beta = outputs['beta'].mean().item()
            
            print(f'Epoch [{epoch}][{batch_idx}/{len(dataloader)}] '
                  f'Loss: {loss.item():.4f} '
                  f'Recon: {outputs["recon_loss"].item():.4f} '
                  f'KL: {outputs["kl_loss"].item():.4f} '
                  f'β: {avg_beta:.2f} '
                  f'LR: {lr:.6f}')
            
            if args.wandb:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/recon_loss': outputs['recon_loss'].item(),
                    'train/kl_loss': outputs['kl_loss'].item(),
                    'train/beta_mean': avg_beta,
                    'train/lr': lr,
                    'step': global_step,
                })
    
    n_batches = len(dataloader)
    return {
        'loss': total_loss / n_batches,
        'recon_loss': total_recon_loss / n_batches,
        'kl_loss': total_kl_loss / n_batches,
    }, global_step


def main():
    args = get_args()
    print(args)
    
    # Setup distributed training
    rank, world_size, distributed = setup_distributed()
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    
    # Create output directory
    if args.exp_name is None:
        model_args = args.model
        args.exp_name = f'{model_args.encoder}_mod{model_args.modulator_depth}_dec{model_args.decoder_layers}'
    
    output_dir = Path(args.output_dir) / (args.exp_name + f'_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    args.output_dir = str(output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / 'checkpoints').mkdir(exist_ok=True)
        (output_dir / 'visualizations').mkdir(exist_ok=True)
        
        # Save config
        with open(output_dir / 'config.yaml', 'w') as f:
            yaml.dump(vars(args), f)
    
    # Initialize wandb
    if args.wandb and rank == 0:
        wandb.init(
            project=args.wandb_project,
            name=args.exp_name,
            config=vars(args),
        )
    
    # Create model
    if rank == 0:
        print(f'Creating β-Diffusion model with {args.encoder} encoder...')
    
    model_confg = args.model
    model = create_beta_diffusion(**model_confg).to(device)
    
    # Count parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(f'Trainable params: {trainable_params / 1e6:.2f}M / Total: {total_params / 1e6:.2f}M')
        print(model)
    # Distributed model
    if distributed:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    # Create dataloaders
    train_loader, train_sampler = create_dataloader(
        args, is_train=True, distributed=distributed, rank=rank, world_size=world_size
    )
    val_loader, _ = create_dataloader(
        args, is_train=False, distributed=distributed, rank=rank, world_size=world_size
    )
    
    if rank == 0:
        print(f'Training samples: {len(train_loader.dataset)}')
        print(f'Validation samples: {len(val_loader.dataset)}')
    
    # Create optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    
    scheduler = get_lr_scheduler(optimizer, args, len(train_loader))
    
    # Resume from checkpoint
    start_epoch = 0
    global_step = 0
    if args.resume is not None:
        if rank == 0:
            print(f'Resuming from {args.resume}')
        checkpoint = torch.load(args.resume, map_location=device)
        model_state = checkpoint['model']
        if distributed:
            model.module.load_state_dict(model_state)
        else:
            model.load_state_dict(model_state)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint.get('global_step', 0)
    
    # Training loop
    if rank == 0:
        print(f'Starting training from epoch {start_epoch}...')
    
    best_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        if distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        # Train
        train_metrics, global_step = train_one_epoch(
            model, train_loader, optimizer, scheduler, epoch, args, rank, global_step
        )
        
        if rank == 0:
            print(f'Epoch {epoch} - Train Loss: {train_metrics["loss"]:.4f} '
                  f'Recon: {train_metrics["recon_loss"]:.4f} KL: {train_metrics["kl_loss"]:.4f}')
            
            if args.wandb:
                wandb.log({
                    'epoch/train_loss': train_metrics['loss'],
                    'epoch/train_recon_loss': train_metrics['recon_loss'],
                    'epoch/train_kl_loss': train_metrics['kl_loss'],
                    'epoch': epoch,
                })
            
            # Visualize reconstructions
            sample_batch = next(iter(val_loader))
            model_for_vis = model.module if distributed else model
            beta_values = torch.linspace(args.beta_min, args.beta_max, 4).tolist()
            visualize_reconstructions(
                model_for_vis, sample_batch[0], 
                output_dir / 'visualizations', epoch, beta_values
            )
            
            # Save checkpoint
            if (epoch + 1) % args.save_interval == 0:
                model_state = model.module.state_dict() if distributed else model.state_dict()
                checkpoint = {
                    'epoch': epoch,
                    'global_step': global_step,
                    'model': model_state,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'args': vars(args),
                    'train_metrics': train_metrics,
                }
                
                ckpt_path = output_dir / 'checkpoints' / f'epoch_{epoch:04d}.pt'
                torch.save(checkpoint, ckpt_path)
                print(f'Saved checkpoint to {ckpt_path}')
                
                # Save best model
                if train_metrics['loss'] < best_loss:
                    best_loss = train_metrics['loss']
                    torch.save(checkpoint, output_dir / 'checkpoints' / 'best.pt')
                    print(f'New best model saved!')
    
    # Save final model
    if rank == 0:
        model_state = model.module.state_dict() if distributed else model.state_dict()
        torch.save({
            'epoch': args.epochs - 1,
            'model': model_state,
            'args': vars(args),
        }, output_dir / 'checkpoints' / 'final.pt')
        print(f'Training complete! Final model saved.')
    
    if args.wandb and rank == 0:
        wandb.finish()


if __name__ == '__main__':
    main()


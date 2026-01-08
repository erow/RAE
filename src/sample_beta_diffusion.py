"""
Sampling script for β-Diffusion.

Demonstrates:
- Loading a trained β-Diffusion model
- Reconstructing images at different β values
- Visualizing the effect of the noise-aware bottleneck
"""

import argparse
import torch
import torchvision.transforms as T
from torchvision.utils import save_image, make_grid
from PIL import Image
import os

from stage1.beta_diffusion import BetaDiffusion, create_beta_diffusion


def get_args():
    parser = argparse.ArgumentParser(description='Sample from β-Diffusion')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--output-dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--image-size', type=int, default=224, help='Image size')
    parser.add_argument('--beta-values', type=float, nargs='+', default=[1, 5, 10, 25, 50, 75, 100],
                        help='β values to sample at')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    return parser.parse_args()


def load_image(path: str, size: int) -> torch.Tensor:
    """Load and preprocess an image."""
    transform = T.Compose([
        T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(size),
        T.ToTensor(),
    ])
    img = Image.open(path).convert('RGB')
    return transform(img).unsqueeze(0)


@torch.no_grad()
def sample_at_betas(model, image, beta_values, device):
    """Generate reconstructions at different β values."""
    model.eval()
    image = image.to(device)
    
    reconstructions = []
    latent_stats = []
    
    # Encode once
    h = model.encode(image)
    
    for beta in beta_values:
        beta_tensor = torch.full((1,), beta, device=device)
        
        # Get latent distribution
        mu, sigma = model.get_latent_distribution(h, beta_tensor)
        
        # Sample (deterministic - use mean)
        z = mu
        
        # Decode
        x_rec = model.decode(z, beta_tensor)
        
        reconstructions.append(x_rec)
        latent_stats.append({
            'beta': beta,
            'mu_mean': mu.mean().item(),
            'mu_std': mu.std().item(),
            'sigma_mean': sigma.mean().item(),
            'sigma_std': sigma.std().item(),
        })
    
    return reconstructions, latent_stats


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    print(f'Loading checkpoint from {args.checkpoint}...')
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Get model config from checkpoint
    ckpt_args = checkpoint.get('args', {})
    encoder_name = ckpt_args.get('encoder', 'dinov2-base')
    
    # Create model
    model = create_beta_diffusion(
        encoder_name=encoder_name,
        modulator_depth=ckpt_args.get('modulator_depth', 4),
        decoder_layers=ckpt_args.get('decoder_layers', 4),
        decoder_hidden_dim=ckpt_args.get('decoder_hidden_dim', 1024),
        decoder_patch_size=ckpt_args.get('decoder_patch_size', 16),
        encoder_input_size=args.image_size,
    ).to(device)
    
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    print(f'Loading image from {args.image}...')
    image = load_image(args.image, args.image_size).to(device)
    
    print(f'Generating reconstructions at β values: {args.beta_values}')
    reconstructions, latent_stats = sample_at_betas(model, image, args.beta_values, device)
    
    # Print latent statistics
    print('\nLatent Statistics:')
    print('-' * 60)
    print(f'{"β":>6} | {"μ mean":>10} | {"μ std":>10} | {"σ mean":>10} | {"σ std":>10}')
    print('-' * 60)
    for stats in latent_stats:
        print(f'{stats["beta"]:>6.1f} | {stats["mu_mean"]:>10.4f} | {stats["mu_std"]:>10.4f} | '
              f'{stats["sigma_mean"]:>10.4f} | {stats["sigma_std"]:>10.4f}')
    
    # Save individual reconstructions
    for i, (recon, beta) in enumerate(zip(reconstructions, args.beta_values)):
        save_path = os.path.join(args.output_dir, f'recon_beta_{beta:.0f}.png')
        save_image(recon.clamp(0, 1), save_path)
        print(f'Saved: {save_path}')
    
    # Save comparison grid
    # Row 1: Original repeated, Row 2: Reconstructions
    all_images = [image] + reconstructions
    grid = make_grid(torch.cat(all_images, dim=0).clamp(0, 1), nrow=len(all_images))
    
    grid_path = os.path.join(args.output_dir, 'comparison_grid.png')
    save_image(grid, grid_path)
    print(f'\nSaved comparison grid: {grid_path}')
    
    # Create annotated grid with β values
    print('\n' + '='*60)
    print('β-Diffusion Analysis Complete!')
    print('='*60)
    print(f'\nKey observations:')
    print('- Low β (≈1): High reconstruction fidelity, less regularization')
    print('- High β (≈100): More regularization, smoother latent space')
    print('\nThe model learns a noise-aware latent manifold that allows')
    print('flexible trade-offs between reconstruction and generative quality.')


if __name__ == '__main__':
    main()


"""
β-Diffusion: A generative framework that integrates controllable noise regimes 
of diffusion models into a VAE architecture.

The modulator dynamically adjusts the latent distribution parameters (μ, σ) 
as a function of a continuous noise scale β ∈ U(1, 100).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
from math import sqrt
import numpy as np

from .encoders import ARCHS
from .decoders import GeneralDecoder
from transformers import AutoConfig, AutoImageProcessor


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, cls_token: bool = False) -> np.ndarray:
    """Generate 2D sinusoidal positional embeddings."""
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    """Generate positional embeddings from a 2D grid."""
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """Generate 1D sinusoidal positional embeddings."""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)


class GaussianFourierEmbedding(nn.Module):
    """Gaussian Fourier embedding for continuous β values."""
    
    def __init__(self, hidden_size: int, embedding_size: int = 256, scale: float = 1.0):
        super().__init__()
        self.embedding_size = embedding_size
        self.scale = scale
        self.W = nn.Parameter(torch.normal(0, scale, (embedding_size,)), requires_grad=False)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size * 2, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
    
    def forward(self, beta: torch.Tensor) -> torch.Tensor:
        """
        Args:
            beta: (B,) continuous noise scale values
        Returns:
            (B, hidden_size) β embeddings
        """
        # Normalize β to [0, 1] range for stable embedding (β ∈ [1, 100])
        beta_norm = (beta - 1) / 99.0
        with torch.no_grad():
            W = self.W
        t = beta_norm[:, None] * W[None, :] * 2 * torch.pi
        t_embed = torch.cat([torch.sin(t), torch.cos(t)], dim=-1)
        return self.mlp(t_embed)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network."""
    
    def __init__(self, in_features: int, hidden_features: int, out_features: Optional[int] = None):
        super().__init__()
        out_features = out_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=True)
        self.w3 = nn.Linear(hidden_features, out_features, bias=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class ModulatorAttention(nn.Module):
    """Multi-head self-attention with optional RoPE for the modulator."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_norm: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply adaptive layer normalization modulation."""
    if shift is not None:
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    return x * (1 + scale.unsqueeze(1))


class DiTModulatorBlock(nn.Module):
    """
    Diffusion Transformer block for β-conditioned modulation.
    Uses AdaLN-Zero for conditioning on β.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_swiglu: bool = True,
    ):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)
        
        self.attn = ModulatorAttention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=True,
        )
        
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if use_swiglu:
            self.mlp = SwiGLUFFN(hidden_size, int(2/3 * mlp_hidden_dim))
        else:
            self.mlp = nn.Sequential(
                nn.Linear(hidden_size, mlp_hidden_dim),
                nn.GELU(approximate="tanh"),
                nn.Linear(mlp_hidden_dim, hidden_size),
            )
        
        # AdaLN modulation: produces scale and gate for attention and MLP
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 4 * hidden_size, bias=True)
        )
    
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) input features
            c: (B, D) conditioning (β embedding)
        """
        scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(4, dim=-1)
        
        # Self-attention with modulation
        x = x + gate_msa.unsqueeze(1) * self.attn(self.norm1(x) * (1 + scale_msa.unsqueeze(1)))
        # MLP with modulation  
        x = x + gate_mlp.unsqueeze(1) * self.mlp(self.norm2(x) * (1 + scale_mlp.unsqueeze(1)))
        
        return x


class BetaDiffusionModulator(nn.Module):
    """
    VAE Modulator powered by Diffusion Transformer (DiT) blocks.
    
    Takes encoder features h and β, outputs modulation scales s_μ(h;β) and s_log_var(h;β)
    that multiply the base μ and log_var projections.
    """
    
    def __init__(
        self,
        hidden_size: int,
        latent_dim: int,
        num_heads: int = 8,
        depth: int = 4,
        mlp_ratio: float = 4.0,
        num_patches: int = 196,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.num_patches = num_patches
        
        # β embedding
        self.beta_embedder = GaussianFourierEmbedding(hidden_size)
        
        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, hidden_size), requires_grad=False
        )
        
        # Input projection (if encoder dim differs from modulator dim)
        self.input_proj = nn.Linear(hidden_size, hidden_size) if hidden_size != latent_dim else nn.Identity()
        
        # DiT blocks for modulation
        self.blocks = nn.ModuleList([
            DiTModulatorBlock(hidden_size, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        
        # Final normalization
        self.norm = RMSNorm(hidden_size)
        
        # Output projections for modulation scales
        # These produce multiplicative scales for μ and log_var
        self.scale_mu = nn.Linear(hidden_size, latent_dim)
        self.scale_log_var = nn.Linear(hidden_size, latent_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        # Initialize positional embeddings
        pos_embed = get_2d_sincos_pos_embed(
            self.hidden_size, int(self.num_patches ** 0.5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        # Zero-out adaLN modulation layers for stable training
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        # Initialize output projections to produce near-identity scaling initially
        nn.init.constant_(self.scale_mu.weight, 0)
        nn.init.constant_(self.scale_mu.bias, 1)  # s_μ starts at 1 (identity)
        nn.init.constant_(self.scale_log_var.weight, 0)
        nn.init.constant_(self.scale_log_var.bias, 1)  # s_log_var starts at 1 (identity)
    
    def forward(
        self, 
        h: torch.Tensor, 
        beta: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: (B, N, D) encoder hidden states
            beta: (B,) noise scale values in [1, 100]
        
        Returns:
            s_mu: (B, N, latent_dim) multiplicative scale for μ
            s_log_var: (B, N, latent_dim) multiplicative scale for log_var  
        """
        # Project input if needed
        x = self.input_proj(h)
        
        # Add positional embedding
        if x.shape[1] == self.num_patches:
            x = x + self.pos_embed
        
        # Get β conditioning
        c = self.beta_embedder(beta)  # (B, D)
        
        # Apply DiT blocks
        for block in self.blocks:
            x = block(x, c)
        
        # Final norm and project to scales
        x = self.norm(x)
        s_mu = self.scale_mu(x).sigmoid()          # (B, N, latent_dim)
        s_log_var = self.scale_log_var(x).sigmoid()  # (B, N, latent_dim)
        
        return s_mu, s_log_var


class DecoderMLPBlock(nn.Module):
    """MLP block with AdaLN-Zero β conditioning."""
    
    def __init__(self, hidden_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.norm = RMSNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, hidden_dim),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim)
        )
        # Zero init for stable training
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
    
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) input features
            c: (B, D) conditioning (β embedding)
        """
        scale, gate = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = x + gate.unsqueeze(1) * self.mlp(self.norm(x) * (1 + scale.unsqueeze(1)))
        return x


class DecoderAttentionBlock(nn.Module):
    """Attention block with AdaLN-Zero β conditioning."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.norm1 = RMSNorm(hidden_dim)
        self.norm2 = RMSNorm(hidden_dim)
        
        self.attn = ModulatorAttention(
            hidden_dim,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=True,
        )
        
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_dim, int(2/3 * mlp_hidden))
        
        # AdaLN modulation: scale and gate for attention and MLP
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 4 * hidden_dim, bias=True)
        )
        # Zero init for stable training
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
    
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) input features
            c: (B, D) conditioning (β embedding)
        """
        scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(4, dim=-1)
        
        # Self-attention with modulation
        x = x + gate_msa.unsqueeze(1) * self.attn(self.norm1(x) * (1 + scale_msa.unsqueeze(1)))
        # MLP with modulation
        x = x + gate_mlp.unsqueeze(1) * self.mlp(self.norm2(x) * (1 + scale_mlp.unsqueeze(1)))
        
        return x


class BetaConditionedDecoder(nn.Module):
    """
    Decoder that receives z and β as inputs.
    Supports either MLP blocks or Attention blocks.
    Adapts reconstruction strategy based on the enforced bottleneck intensity.
    """
    
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        output_dim: int,  # patch_size^2 * 3
        num_patches: int,
        num_layers: int = 4,
        block_type: str = 'mlp',  # 'mlp' or 'attention'
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_patches = num_patches
        self.output_dim = output_dim
        self.block_type = block_type
        
        # β embedding for decoder
        self.beta_embedder = GaussianFourierEmbedding(hidden_dim)
        
        # Input projection
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        
        # Positional embedding (for attention blocks)
        if block_type == 'attention':
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, hidden_dim), requires_grad=False
            )
            pos_embed = get_2d_sincos_pos_embed(hidden_dim, int(num_patches ** 0.5))
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        else:
            self.pos_embed = None
        
        # Build decoder blocks
        if block_type == 'mlp':
            self.blocks = nn.ModuleList([
                DecoderMLPBlock(hidden_dim, mlp_ratio=mlp_ratio)
                for _ in range(num_layers)
            ])
        elif block_type == 'attention':
            self.blocks = nn.ModuleList([
                DecoderAttentionBlock(hidden_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
                for _ in range(num_layers)
            ])
        else:
            raise ValueError(f"Unknown block_type: {block_type}. Choose 'mlp' or 'attention'.")
        
        self.norm = RMSNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, z: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, N, latent_dim) or (B, C, H, W) latent samples
            beta: (B,) noise scale values
        
        Returns:
            (B, N, output_dim) reconstructed patches
        """
        # Handle 2D latent format
        if z.dim() == 4:
            B, C, H, W = z.shape
            z = z.view(B, C, H * W).transpose(1, 2)  # (B, N, C)
        
        # Get β conditioning
        c = self.beta_embedder(beta)  # (B, hidden_dim)
        
        # Project latent
        x = self.input_proj(z)  # (B, N, hidden_dim)
        
        # Add positional embedding for attention blocks
        if self.pos_embed is not None and x.shape[1] == self.num_patches:
            x = x + self.pos_embed
        
        # Apply blocks
        for block in self.blocks:
            x = block(x, c)
        
        # Final norm and output projection
        x = self.norm(x)
        out = self.output_proj(x)  # (B, N, output_dim)
        
        return out


class BetaDiffusion(nn.Module):
    """
    β-Diffusion: A VAE with controllable noise regimes.
    
    Architecture:
    - ViT Encoder (E): Extracts h = E(x)
    - VAE Modulator (M): DiT blocks that output β-dependent modulation
    - Latent sampling: z ~ N(μ, exp(log_var)) where:
        μ = f_μ(h) * s_μ(h;β)
        log_var = f_log_var(h) * s_log_var(h;β)
    - Decoder (D_ψ): Reconstructs x from z with β conditioning
    
    Training: β ~ U(1, 100)
    Loss: L_ELBO = ||x - D_ψ(z, β)||² + β · KL(q(z|x;β) || p(z))
    """
    
    def __init__(
        self,
        # Encoder configs
        encoder_cls: str = 'Dinov2withNorm',
        encoder_config_path: str = 'facebook/dinov2-base',
        encoder_input_size: int = 224,
        encoder_freeze: bool = True,
        encoder_params: dict = {},
        # Modulator configs
        modulator_depth: int = 4,
        modulator_heads: int = 8,
        modulator_mlp_ratio: float = 4.0,
        # Latent configs
        latent_dim: int = 64,
        # Decoder configs
        decoder_hidden_dim: int = 1024,
        decoder_num_layers: int = 4,
        decoder_patch_size: int = 16,
        decoder_block_type: str = 'mlp',  # 'mlp' or 'attention'
        decoder_num_heads: int = 8,
        decoder_mlp_ratio: float = 4.0,
        # Training configs
        beta_min: float = 1.0,
        beta_max: float = 100.0,
        # Normalization
        normalization_stat_path: Optional[str] = None,
        eps: float = 1e-5,
    ):
        super().__init__()
        
        # Store configs
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.latent_dim = latent_dim
        self.eps = eps
        
        # Initialize encoder
        encoder_cls_obj = ARCHS[encoder_cls]
        self.encoder = encoder_cls_obj(**encoder_params)
        self.encoder.requires_grad_(not encoder_freeze)  # Freeze encoder
        
        # Load encoder normalization
        proc = AutoImageProcessor.from_pretrained(encoder_config_path)
        self.register_buffer('encoder_mean', torch.tensor(proc.image_mean).view(1, 3, 1, 1))
        self.register_buffer('encoder_std', torch.tensor(proc.image_std).view(1, 3, 1, 1))
        
        self.encoder_input_size = encoder_input_size
        self.encoder_patch_size = self.encoder.patch_size
        encoder_hidden_size = self.encoder.hidden_size
        
        # Calculate number of patches
        self.num_patches = (encoder_input_size // self.encoder_patch_size) ** 2
        
        # Base latent projections: f_μ and f_log_var
        self.f_mu = nn.Linear(encoder_hidden_size, latent_dim)
        self.f_log_var = nn.Linear(encoder_hidden_size, latent_dim)
        
        # VAE Modulator (DiT)
        self.modulator = BetaDiffusionModulator(
            hidden_size=encoder_hidden_size,
            latent_dim=latent_dim,
            num_heads=modulator_heads,
            depth=modulator_depth,
            mlp_ratio=modulator_mlp_ratio,
            num_patches=self.num_patches,
        )
        
        # β-conditioned decoder
        output_dim = decoder_patch_size ** 2 * 3
        self.decoder = BetaConditionedDecoder(
            latent_dim=latent_dim,
            hidden_dim=decoder_hidden_dim,
            output_dim=output_dim,
            num_patches=self.num_patches,
            num_layers=decoder_num_layers,
            block_type=decoder_block_type,
            num_heads=decoder_num_heads,
            mlp_ratio=decoder_mlp_ratio,
        )
        
        self.decoder_patch_size = decoder_patch_size
        
        # Optional latent normalization
        if normalization_stat_path is not None:
            stats = torch.load(normalization_stat_path, map_location='cpu')
            self.register_buffer('latent_mean', stats.get('mean', torch.zeros(1)))
            self.register_buffer('latent_var', stats.get('var', torch.ones(1)))
            self.do_normalization = True
        else:
            self.do_normalization = False
        
        self._init_weights()
    
    def _init_weights(self):
        # Initialize f_μ and f_log_var
        nn.init.xavier_uniform_(self.f_mu.weight)
        nn.init.zeros_(self.f_mu.bias)
        nn.init.xavier_uniform_(self.f_log_var.weight)
        nn.init.zeros_(self.f_log_var.bias)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features using frozen encoder."""
        _, _, h, w = x.shape
        if h != self.encoder_input_size or w != self.encoder_input_size:
            x = F.interpolate(
                x, size=(self.encoder_input_size, self.encoder_input_size),
                mode='bicubic', align_corners=False
            )
        x = (x - self.encoder_mean) / self.encoder_std
        with torch.no_grad():
            h = self.encoder(x)  # (B, N, D)
        return h
    
    def get_latent_distribution(
        self, 
        h: torch.Tensor, 
        beta: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute latent distribution parameters modulated by β.
        
        μ = f_μ(h) * s_μ(h; β)
        log_var = f_log_var(h) * s_log_var(h; β)
        """
        # Base projections
        base_mu = self.f_mu(h)            # (B, N, latent_dim)
        base_log_var = self.f_log_var(h)  # (B, N, latent_dim)
        
        # Get β-conditioned modulation scales
        s_mu, s_log_var = self.modulator(h, beta)  # (B, N, latent_dim)
        
        # Apply modulation
        mu = base_mu * s_mu
        log_var = base_log_var * s_log_var
        
        return mu, log_var
    
    def reparameterize(
        self, 
        mu: torch.Tensor, 
        log_var: torch.Tensor
    ) -> torch.Tensor:
        """Sample z using reparameterization trick: z = μ + std * ε where std = exp(0.5 * log_var)"""
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(mu)
            z = mu + std * eps
        else:
            z = mu
        return z
    
    def decode(
        self, 
        z: torch.Tensor, 
        beta: torch.Tensor
    ) -> torch.Tensor:
        """Decode latent z with β conditioning."""
        # Decoder expects (B, N, latent_dim)
        if z.dim() == 4:
            B, C, H, W = z.shape
            z = z.view(B, C, H * W).transpose(1, 2)
        
        out = self.decoder(z, beta)  # (B, N, patch_size^2 * 3)
        
        # Unpatchify to image
        x_rec = self.unpatchify(out)
        
        return x_rec
    
    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert patch predictions to image.
        x: (B, N, patch_size^2 * 3)
        """
        p = self.decoder_patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(x.shape[0], h, w, p, p, 3)
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(x.shape[0], 3, h * p, w * p)
        
        # Denormalize
        imgs = imgs * self.encoder_std + self.encoder_mean
        
        return imgs
    
    def compute_kl_divergence(
        self, 
        mu: torch.Tensor, 
        log_var: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL divergence: KL(q(z|x;β) || p(z))
        where p(z) = N(0, I)
        
        KL = 0.5 * (exp(log_var) + μ² - 1 - log_var)
        """
        kl = 0.5 * (torch.exp(log_var) + mu.pow(2) - 1 - log_var)
        return kl.sum(dim=-1)
    
    def forward(
        self, 
        x: torch.Tensor,
        beta: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.
        
        Args:
            x: (B, 3, H, W) input images
            beta: (B,) noise scales. If None, sampled from U(beta_min, beta_max)
        
        Returns:
            Dict containing:
                - x_rec: reconstructed images
                - mu: latent means
                - log_var: latent log variances
                - z: sampled latents
                - beta: noise scales used
                - loss: total ELBO loss
                - recon_loss: reconstruction loss
                - kl_loss: KL divergence
        """
        B = x.shape[0]
        device = x.device
        
        # Sample β if not provided
        if beta is None:
            beta = torch.rand(B, device=device) * (self.beta_max - self.beta_min) + self.beta_min
        
        # Encode
        h = self.encode(x)  # (B, N, D)
        B, N, D = h.shape
        
        # Get modulated latent distribution
        mu, log_var = self.get_latent_distribution(h, beta)
        
        # Sample z
        z = self.reparameterize(mu, log_var)
        
        # Decode
        x_rec = self.decode(z, beta)
        
        # Compute losses
        # Reconstruction loss
        recon_loss = F.mse_loss(x_rec, x, reduction='none').sum(dim=(1, 2, 3)) / N
        
        # KL divergence (β-weighted)
        kl_loss = self.compute_kl_divergence(mu, log_var)
        
        # Total ELBO loss with β weighting on KL
        # L_ELBO = ||x - D_ψ(z, β)||² + β · KL(q(z|x;β) || p(z))
        loss = recon_loss + beta * kl_loss.mean(1)
        loss = loss.mean()
        
        return {
            'x_rec': x_rec,
            'mu': mu,
            'log_var': log_var,
            'z': z,
            'beta': beta,
            'loss': loss,
            'recon_loss': recon_loss.mean(),
            'kl_loss': kl_loss.mean(),
        }
    
    @torch.no_grad()
    def sample(
        self,
        x: torch.Tensor,
        beta: float = 1.0,
    ) -> torch.Tensor:
        """
        Sample reconstruction at specific β value.
        Lower β = higher fidelity, higher β = more regularization.
        """
        self.eval()
        B = x.shape[0]
        device = x.device
        
        beta_tensor = torch.full((B,), beta, device=device)
        
        h = self.encode(x)
        mu, log_var = self.get_latent_distribution(h, beta_tensor)
        z = mu  # Use mean for deterministic sampling
        x_rec = self.decode(z, beta_tensor)
        
        return x_rec
    
    @torch.no_grad()
    def interpolate_beta(
        self,
        x: torch.Tensor,
        beta_values: list = [1, 10, 50, 100],
    ) -> torch.Tensor:
        """
        Generate reconstructions at different β values for visualization.
        """
        self.eval()
        B = x.shape[0]
        device = x.device
        
        results = []
        h = self.encode(x)
        
        for beta in beta_values:
            beta_tensor = torch.full((B,), beta, device=device)
            mu, log_var = self.get_latent_distribution(h, beta_tensor)
            z = mu
            x_rec = self.decode(z, beta_tensor)
            results.append(x_rec)
        
        return torch.stack(results, dim=1)  # (B, num_betas, C, H, W)


# Factory function for creating models
def create_beta_diffusion(
    encoder: str = 'dinov2-base',
    modulator_depth: int = 4,
    decoder_layers: int = 4,
    **kwargs
) -> BetaDiffusion:
    """Create a β-Diffusion model with specified configuration."""
    
    config_map = {
        'dinov2-base': {
            'encoder_cls': 'Dinov2withNorm',
            'encoder_config_path': 'facebook/dinov2-base',
            'encoder_params': {'dinov2_path': 'facebook/dinov2-with-registers-base'},
            
            'modulator_heads': 12,
        },
        'dinov2-large': {
            'encoder_cls': 'Dinov2withNorm', 
            'encoder_config_path': 'facebook/dinov2-large',
            'encoder_params': {'dinov2_path': 'facebook/dinov2-with-registers-large'},
        },
        'mae-base': {
            'encoder_cls': 'MAEwNorm',
            'encoder_config_path': 'facebook/vit-mae-base',
            'encoder_params': {'model_name': 'facebook/vit-mae-base'},
        },
        'mae-large': {
            'encoder_cls': 'MAEwNorm',
            'encoder_config_path': 'facebook/vit-mae-large',
            'encoder_params': {'model_name': 'facebook/vit-mae-large'},
        },
    }
    
    if encoder not in config_map:
        raise ValueError(f"Unknown encoder: {encoder}. Available: {list(config_map.keys())}")
    
    config = config_map[encoder]
    config['modulator_depth'] = modulator_depth
    config['decoder_num_layers'] = decoder_layers
    config.update(kwargs)
    
    return BetaDiffusion(**config)

